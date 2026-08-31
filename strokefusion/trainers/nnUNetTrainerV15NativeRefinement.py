"""Lesion-presence-gated local/global Primus-M refinement."""

from __future__ import annotations

import torch
from dynamic_network_architectures.architectures.primus import Primus
from einops import rearrange
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerV14PrimusMAE import (
    nnUNetTrainerV14PrimusMAE,
)
from nnunetv2.training.nnUNetTrainer.primus.primus_trainers import (
    nnUNet_Primus_M_Trainer,
)
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)
from torch import nn
from torch.nn import functional as F
from torch import autocast


class PrimusMLocalStemFusion(Primus):
    """Fuse gated local 3D context into the global Transformer tokens."""

    def __init__(self, input_channels: int, num_classes: int, **kwargs) -> None:
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            **kwargs,
        )
        adapter_channels = 64
        self.local_stem_refinement = nn.Sequential(
            nn.Conv3d(self.embed_dim, adapter_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(
                adapter_channels,
                adapter_channels,
                kernel_size=3,
                padding=1,
                groups=adapter_channels,
            ),
            nn.GELU(),
            nn.Conv3d(adapter_channels, self.embed_dim, kernel_size=1),
        )
        nn.init.kaiming_normal_(
            self.local_stem_refinement[0].weight, nonlinearity="relu"
        )
        nn.init.zeros_(self.local_stem_refinement[0].bias)
        nn.init.kaiming_normal_(
            self.local_stem_refinement[2].weight, nonlinearity="relu"
        )
        nn.init.zeros_(self.local_stem_refinement[2].bias)
        nn.init.zeros_(self.local_stem_refinement[4].weight)
        nn.init.zeros_(self.local_stem_refinement[4].bias)

        # The auxiliary head predicts whether each 8^3 token contains lesion.
        # Zero initialization makes the initial gate exactly one, so the whole
        # network begins at the published Primus-M function.
        self.token_presence_head = nn.Conv3d(self.embed_dim, 1, kernel_size=1)
        nn.init.zeros_(self.token_presence_head.weight)
        nn.init.zeros_(self.token_presence_head.bias)
        self.local_gate_amplitude = 0.25

    def forward(self, image: torch.Tensor, ret_mask: bool = False):
        full_shape = image.shape[2:]
        local_stem = self.down_projection(image)
        batch_size, _, width, height, depth = local_stem.shape
        num_patches = width * height * depth

        local_tokens = rearrange(local_stem, "b c w h d -> b (w h d) c")
        tokens = local_tokens
        if self.register_tokens is not None:
            tokens = torch.cat(
                (self.register_tokens.expand(batch_size, -1, -1), tokens), dim=1
            )
        tokens, keep_indices = self.eva(tokens)
        if self.register_tokens is not None:
            tokens = tokens[:, self.register_tokens.shape[1] :]

        restored_tokens, restoration_mask = self.restore_full_sequence(
            tokens, keep_indices, num_patches
        )
        global_features = rearrange(
            restored_tokens,
            "b (w h d) c -> b c w h d",
            w=width,
            h=height,
            d=depth,
        )
        local_residual = self.local_stem_refinement(local_stem)
        token_presence_logits = self.token_presence_head(global_features)
        local_gate = 1.0 + self.local_gate_amplitude * torch.tanh(
            token_presence_logits
        )
        fused_features = global_features + local_gate * local_residual
        logits = self.up_projection(fused_features)

        if self.training and not ret_mask:
            return logits, token_presence_logits
        if not ret_mask:
            return logits
        if restoration_mask is None:
            return logits, None
        mask = rearrange(
            restoration_mask,
            "b (w h d) -> b w h d",
            w=width,
            h=height,
            d=depth,
        )
        full_mask = (
            mask.repeat_interleave(full_shape[0] // width, dim=1)
            .repeat_interleave(full_shape[1] // height, dim=2)
            .repeat_interleave(full_shape[2] // depth, dim=3)
        )
        return logits, full_mask[:, None]


class nnUNetTrainerV15NativeRefinement(nnUNetTrainerV14PrimusMAE):
    """Fresh Primus-M with gated local fusion and token-presence supervision."""

    research_version = "V15"
    token_presence_loss_weight = 0.05

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        # This is a native 400-epoch schedule from epoch 0. V14's continuation
        # scheduler is deliberately not part of V15.
        self.num_epochs = 400
        self.warmup_duration_whole_net = 15
        self.initial_lr = 1e-4
        self.save_every = 50
        self.scheduled_validation_interval = 50
        # Whole-case metrics, not patch pseudo Dice, select V15 checkpoints.
        # Suppress nnU-Net's large pseudo-best checkpoint to protect disk.
        self._best_ema = float("inf")

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        return PrimusMLocalStemFusion(
            input_channels=num_input_channels,
            embed_dim=864,
            patch_embed_size=(8, 8, 8),
            num_classes=num_output_channels,
            eva_depth=16,
            eva_numheads=12,
            input_shape=tuple(configuration_manager.patch_size),
            drop_path_rate=0.2,
            scale_attn_inner=True,
            init_values=0.1,
        )

    @staticmethod
    def _balanced_token_presence_loss(
        logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Balanced BCE on coarse lesion occupancy without component reweighting."""

        target = target.float()
        per_token = F.binary_cross_entropy_with_logits(
            logits.float(), target, reduction="none"
        )
        positive = target
        negative = 1.0 - target
        positive_count = positive.sum()
        negative_loss = (per_token * negative).sum() / negative.sum().clamp_min(1.0)
        if positive_count.item() == 0:
            return negative_loss
        positive_loss = (per_token * positive).sum() / positive_count
        return 0.5 * (positive_loss + negative_loss)

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [item.to(self.device, non_blocking=True) for item in target]
            full_resolution_target = target[0]
        else:
            target = target.to(self.device, non_blocking=True)
            full_resolution_target = target

        self.optimizer.zero_grad(set_to_none=True)
        context = (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        )
        with context:
            segmentation_logits, token_presence_logits = self.network(data)
            segmentation_loss = self.loss(segmentation_logits, target)
            token_presence_target = F.adaptive_max_pool3d(
                (full_resolution_target > 0).float(),
                token_presence_logits.shape[2:],
            )
            token_presence_loss = self._balanced_token_presence_loss(
                token_presence_logits, token_presence_target
            )
            loss = segmentation_loss + (
                self.token_presence_loss_weight * token_presence_loss
            )

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
            self.optimizer.step()

        return {
            "loss": loss.detach().cpu().numpy(),
            "segmentation_loss": segmentation_loss.detach().cpu().numpy(),
            "token_presence_loss": token_presence_loss.detach().cpu().numpy(),
        }

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        # Use the published warmup trainer's native resume path. This rebuilds
        # the correct epoch-0-to-400 scheduler and never installs V14's
        # epoch-100 continuation curve.
        nnUNet_Primus_M_Trainer.load_checkpoint(self, filename_or_checkpoint)
