"""V9: OpenMind MAE initialized ResEnc-L with published sawtooth adaptation."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.lr_scheduler.warmup import (
    Lin_incr_LRScheduler,
    Lin_incr_offset_LRScheduler,
    PolyLRScheduler_offset,
)
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import (
    nnUNetTrainer_250epochs,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerV3UFL import (
    nnUNetTrainerV3UFL,
)
from nnunetv2.utilities.helpers import empty_cache


WORKSPACE = Path(os.environ.get("ISLES26_WORKSPACE", Path.cwd())).resolve()
PRETRAINED_CHECKPOINT = (
    WORKSPACE
    / "external_models"
    / "ResEncL-OpenMind-MAE"
    / "checkpoint_final.pth"
)
EXPECTED_PRETRAINED_SHA256 = (
    "7a847af785635335c00e711d16ff4d225d86ecd5992b14c059df2b520e3ee933"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class nnUNetTrainerV9OpenMindMAE(nnUNetTrainerV3UFL):
    """Fine-tune one ResEnc-L model from public OpenMind MAE weights.

    The encoder and stem are initialized from self-supervised OpenNeuro MRI.
    The segmentation decoder stays randomly initialized. The four-stage
    schedule follows the authors' published ResEnc-L sawtooth adaptation:
    decoder warm-up, decoder polynomial decay, whole-network warm-up, then
    whole-network polynomial decay.
    """

    research_version = "V9"
    retained_scheduled_checkpoints = frozenset((150, 200, 250, 300, 350, 400))

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 400
        self.save_every = 50
        self.scheduled_validation_interval = 50
        self.enable_deep_supervision = False
        self.initial_lr = 1e-3
        self.warmup_duration_decoder = 50
        self.warmup_duration_whole_net = 50
        self.training_stage: str | None = None
        self.pretrained_checkpoint = PRETRAINED_CHECKPOINT
        self.pretrained_sha256 = EXPECTED_PRETRAINED_SHA256
        self.pretrained_encoder_tensors = 0

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = False,
    ):
        return nnUNetTrainer_250epochs.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )

    def _network_module(self):
        module = self.network.module if isinstance(self.network, DDP) else self.network
        return module._orig_mod if isinstance(module, OptimizedModule) else module

    def _load_openmind_encoder(self) -> None:
        checkpoint_path = self.pretrained_checkpoint
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"missing pretrained checkpoint: {checkpoint_path}")
        actual_sha256 = _sha256(checkpoint_path)
        if actual_sha256 != self.pretrained_sha256:
            raise RuntimeError(
                "OpenMind checkpoint hash changed: "
                f"{actual_sha256} != {self.pretrained_sha256}"
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        pretrained = checkpoint["network_weights"]
        encoder_state = {
            key.removeprefix("encoder."): value
            for key, value in pretrained.items()
            if key.startswith("encoder.")
        }
        network = self._network_module()
        expected = network.encoder.state_dict()
        if set(encoder_state) != set(expected):
            missing = sorted(set(expected) - set(encoder_state))
            unexpected = sorted(set(encoder_state) - set(expected))
            raise RuntimeError(
                "OpenMind/current ResEnc-L encoder mismatch: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        wrong_shapes = [
            key
            for key in expected
            if tuple(expected[key].shape) != tuple(encoder_state[key].shape)
        ]
        if wrong_shapes:
            raise RuntimeError(f"OpenMind encoder shape mismatch: {wrong_shapes[:5]}")
        network.encoder.load_state_dict(encoder_state, strict=True)
        self.pretrained_encoder_tensors = len(encoder_state)
        self.print_to_log_file(
            "Loaded public OpenMind MAE encoder: "
            f"{len(encoder_state)} tensors, sha256={actual_sha256}; "
            "decoder remains randomly initialized.",
            also_print_to_console=True,
        )

    def initialize(self) -> None:
        super().initialize()
        self._load_openmind_encoder()

    def _stage_for_epoch(self, epoch: int) -> str:
        if epoch < self.warmup_duration_decoder // 2:
            return "warmup_decoder"
        if epoch < self.warmup_duration_decoder:
            return "train_decoder"
        if epoch < self.warmup_duration_decoder + self.warmup_duration_whole_net:
            return "warmup_all"
        return "train"

    def _set_trainable_stage(self, stage: str) -> None:
        network = self._network_module()
        train_whole_network = stage in ("warmup_all", "train")
        for parameter in network.parameters():
            parameter.requires_grad_(train_whole_network)
        if not train_whole_network:
            # ResidualEncoderUNet's decoder stores a registered reference back
            # to the encoder. Calling decoder.parameters() would therefore
            # silently unfreeze the encoder as well. Top-level named_parameters
            # de-duplicates that reference and cleanly identifies decoder-only
            # tensors by their canonical ``decoder.`` prefix.
            for name, parameter in network.named_parameters():
                if name.startswith("decoder."):
                    parameter.requires_grad_(True)

    def configure_optimizers(self, stage: str = "warmup_decoder"):
        valid_stages = {"warmup_decoder", "train_decoder", "warmup_all", "train"}
        if stage not in valid_stages:
            raise ValueError(f"unknown V9 training stage: {stage}")
        if self.training_stage == stage:
            return self.optimizer, self.lr_scheduler

        self._set_trainable_stage(stage)
        network = self._network_module()
        decoder_parameters = [
            parameter
            for name, parameter in network.named_parameters()
            if name.startswith("decoder.")
        ]
        all_parameters = list(network.parameters())

        if stage == "warmup_decoder":
            self.print_to_log_file("V9 stage: decoder linear warm-up")
            optimizer = torch.optim.SGD(
                decoder_parameters,
                self.initial_lr,
                weight_decay=self.weight_decay,
                momentum=0.99,
                nesterov=True,
            )
            lr_scheduler = Lin_incr_LRScheduler(
                optimizer,
                self.initial_lr,
                self.warmup_duration_decoder // 2,
            )
        elif stage == "train_decoder":
            self.print_to_log_file("V9 stage: decoder polynomial decay")
            if self.training_stage == "warmup_decoder":
                optimizer = self.optimizer
            else:
                optimizer = torch.optim.SGD(
                    decoder_parameters,
                    self.initial_lr,
                    weight_decay=self.weight_decay,
                    momentum=0.99,
                    nesterov=True,
                )
            lr_scheduler = PolyLRScheduler_offset(
                optimizer,
                self.initial_lr,
                self.warmup_duration_decoder,
                self.warmup_duration_decoder // 2,
            )
        elif stage == "warmup_all":
            self.print_to_log_file("V9 stage: whole-network linear warm-up")
            optimizer = torch.optim.SGD(
                all_parameters,
                self.initial_lr,
                weight_decay=self.weight_decay,
                momentum=0.99,
                nesterov=True,
            )
            lr_scheduler = Lin_incr_offset_LRScheduler(
                optimizer,
                self.initial_lr,
                # The pinned nnU-Net 2.8.1 scheduler divides by max_steps
                # directly. Passing 50 preserves the published 50-epoch
                # whole-network ramp from epoch 50 through epoch 99.
                self.warmup_duration_whole_net,
                self.warmup_duration_decoder,
            )
        else:
            self.print_to_log_file("V9 stage: whole-network polynomial decay")
            if self.training_stage == "warmup_all":
                optimizer = self.optimizer
            else:
                optimizer = torch.optim.SGD(
                    all_parameters,
                    self.initial_lr,
                    weight_decay=self.weight_decay,
                    momentum=0.99,
                    nesterov=True,
                )
            lr_scheduler = PolyLRScheduler_offset(
                optimizer,
                self.initial_lr,
                self.num_epochs,
                self.warmup_duration_decoder + self.warmup_duration_whole_net,
            )

        self.training_stage = stage
        empty_cache(self.device)
        return optimizer, lr_scheduler

    def on_train_epoch_start(self) -> None:
        desired_stage = self._stage_for_epoch(self.current_epoch)
        if self.training_stage != desired_stage:
            self.optimizer, self.lr_scheduler = self.configure_optimizers(desired_stage)

        self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file("")
        self.print_to_log_file(f"Epoch {self.current_epoch}")
        self.print_to_log_file(f"V9 training stage: {self.training_stage}")
        learning_rate = self.optimizer.param_groups[0]["lr"]
        self.print_to_log_file(f"Current learning rate: {np.round(learning_rate, 6)}")
        self.logger.log("lrs", learning_rate, self.current_epoch)

    def set_deep_supervision_enabled(self, enabled: bool) -> None:
        # V9 is deliberately built without deep-supervision heads, matching the
        # published OpenMind adaptation recipe. nnU-Net toggles this around
        # whole-case validation, so this must remain a no-op.
        return None

    def _build_loss(self):
        return nnUNetTrainer_250epochs._build_loss(self)

    def _run_scheduled_whole_case_validation(self, completed_epoch: int) -> None:
        super()._run_scheduled_whole_case_validation(completed_epoch)
        # ResEnc-L checkpoints are large. Keep every official metric artifact,
        # but discard the two very early warm-up weights after their successful
        # evaluation; later candidates remain reproducible and selectable.
        if (
            self.local_rank == 0
            and completed_epoch not in self.retained_scheduled_checkpoints
        ):
            checkpoint = Path(self.output_folder) / f"checkpoint_epoch{completed_epoch:03d}.pth"
            if checkpoint.exists():
                checkpoint.unlink()
                self.print_to_log_file(
                    "Removed evaluated warm-up checkpoint to conserve storage: "
                    f"{checkpoint.name}",
                    also_print_to_console=True,
                )

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        """Restore the optimizer with the same parameter set as the saved stage."""
        if not self.was_initialized:
            self.initialize()
        checkpoint = (
            torch.load(
                filename_or_checkpoint,
                map_location=self.device,
                weights_only=False,
            )
            if isinstance(filename_or_checkpoint, str)
            else filename_or_checkpoint
        )

        network = self._network_module()
        network_keys = network.state_dict().keys()
        state_dict = {}
        for key, value in checkpoint["network_weights"].items():
            normalized = key[7:] if key.startswith("module.") and key[7:] in network_keys else key
            state_dict[normalized] = value
        network.load_state_dict(state_dict)

        self.my_init_kwargs = checkpoint["init_args"]
        self.current_epoch = checkpoint["current_epoch"]
        self.logger.load_checkpoint(checkpoint["logging"])
        self._best_ema = checkpoint["_best_ema"]
        self.inference_allowed_mirroring_axes = checkpoint.get(
            "inference_allowed_mirroring_axes", self.inference_allowed_mirroring_axes
        )

        saved_epoch = max(0, self.current_epoch - 1)
        saved_stage = self._stage_for_epoch(saved_epoch)
        self.training_stage = None
        self.optimizer, self.lr_scheduler = self.configure_optimizers(saved_stage)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.grad_scaler is not None and checkpoint.get("grad_scaler_state") is not None:
            self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state"])
        self.print_to_log_file(
            f"Restored V9 checkpoint at epoch {self.current_epoch} in stage {saved_stage}",
            also_print_to_console=True,
        )
