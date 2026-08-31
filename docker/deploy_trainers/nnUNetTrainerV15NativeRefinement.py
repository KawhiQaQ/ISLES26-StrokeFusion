"""Inference-only architecture definition for the final V15 checkpoint."""

import torch
from dynamic_network_architectures.architectures.primus import Primus
from einops import rearrange
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)
from torch import nn


class PrimusMLocalStemFusion(Primus):
    def __init__(self, input_channels: int, num_classes: int, **kwargs) -> None:
        super().__init__(input_channels=input_channels, num_classes=num_classes, **kwargs)
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
        self.token_presence_head = nn.Conv3d(self.embed_dim, 1, kernel_size=1)
        self.local_gate_amplitude = 0.25

    def forward(self, image: torch.Tensor, ret_mask: bool = False):
        full_shape = image.shape[2:]
        local_stem = self.down_projection(image)
        batch_size, _, width, height, depth = local_stem.shape
        num_patches = width * height * depth
        tokens = rearrange(local_stem, "b c w h d -> b (w h d) c")
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
        logits = self.up_projection(global_features + local_gate * local_residual)
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


class nnUNetTrainerV15NativeRefinement(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = False,
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
