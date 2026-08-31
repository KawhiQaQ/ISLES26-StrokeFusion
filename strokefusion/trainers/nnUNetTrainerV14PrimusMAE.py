"""OpenMind-MAE initialized Primus-M with cross-center RASS."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Tuple, Union

import numpy as np
import torch
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerV12RASS import (
    RandomAmplitudeSpectrumSynthesisTransform,
)
from nnunetv2.training.nnUNetTrainer.primus.primus_trainers import (
    nnUNet_Primus_M_Trainer,
)


EXPECTED_SPLIT_SHA256 = (
    "da10108f65fdff2954c7f68f9e69456a5d9bb8f78b28512e3714656ba2bd9885"
)
WORKSPACE = Path(os.environ.get("ISLES26_WORKSPACE", Path.cwd())).resolve()
PRETRAINED_CHECKPOINT = (
    WORKSPACE
    / "external_models"
    / "PrimusM-OpenMind-MAE"
    / "checkpoint_final.pth"
)
EXPECTED_PRETRAINED_SHA256 = (
    "b866ac5f61d7e90d3a6cbb00a759ffc9d73beb5e63baa6b3cd654671ebc9a552"
)


class _ContinuationPolyLRScheduler:
    """Polynomial decay anchored to the last LR stored in the epoch-100 checkpoint."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        anchor_lr: float,
        anchor_epoch: int,
        end_epoch: int,
        exponent: float = 0.9,
    ) -> None:
        self.optimizer = optimizer
        self.anchor_lr = anchor_lr
        self.anchor_epoch = anchor_epoch
        self.end_epoch = end_epoch
        self.exponent = exponent

    def lr_at(self, epoch: int) -> float:
        progress = (epoch - self.anchor_epoch) / (
            self.end_epoch - self.anchor_epoch
        )
        progress = min(max(progress, 0.0), 1.0)
        return self.anchor_lr * (1.0 - progress) ** self.exponent

    def step(self, current_step: int | None = None) -> None:
        if current_step is None:
            raise RuntimeError("V14 continuation scheduler requires an epoch")
        new_lr = self.lr_at(current_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class nnUNetTrainerV14PrimusMAE(nnUNet_Primus_M_Trainer):
    """Fine-tune the public Primus-M OpenMind MAE on frozen RAW fold 0.

    This is an architecture-level successor to V12. The segmentation decoder
    remains randomly initialized; all 371 Transformer encoder tensors and both
    input-projection tensors are loaded exactly from the public checkpoint.
    """

    research_version = "V14"
    rass_probability = 0.30
    rass_alpha = 3.0
    rass_beta = 0.25
    rass_gamma = 2.0
    retained_scheduled_checkpoints = frozenset(range(50, 401, 50))
    continuation_anchor_epoch = 99
    continuation_start_epoch = 100
    continuation_original_end_epoch = 150
    continuation_end_epoch = 400

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        # The first 100 completed epochs used the authors' 150-epoch adaptation
        # schedule. Continue from that fully resumable checkpoint to epoch 400
        # without changing the optimizer state or jumping the learning rate.
        self.num_epochs = self.continuation_end_epoch
        self.warmup_duration_whole_net = 15
        self.initial_lr = 1e-4
        self.weight_decay = 5e-2
        self.save_every = 50
        self.scheduled_validation_interval = 50
        self._inside_scheduled_validation = False
        self._skip_next_post_training_validation = False

        # Primus-M's learned positional embedding was pretrained on 160^3.
        # Reuse the RAW-derived 1 mm arrays, but request exact 160^3 crops and a
        # measured-safe batch size for the 24 GiB research GPU.
        self.configuration_manager.configuration["patch_size"] = [160, 160, 160]
        self.configuration_manager.configuration["batch_size"] = 2
        self.pretrained_checkpoint = PRETRAINED_CHECKPOINT
        self.pretrained_sha256 = EXPECTED_PRETRAINED_SHA256
        self.pretrained_encoder_tensors = 0
        self.pretrained_stem_tensors = 0

    def _continuation_anchor_lr(self) -> float:
        return self.initial_lr * (
            1
            - (
                self.continuation_anchor_epoch - self.warmup_duration_whole_net
            )
            / (
                self.continuation_original_end_epoch
                - self.warmup_duration_whole_net
            )
        ) ** 0.9

    def _continuation_scheduler(self) -> _ContinuationPolyLRScheduler:
        return _ContinuationPolyLRScheduler(
            self.optimizer,
            self._continuation_anchor_lr(),
            self.continuation_anchor_epoch,
            self.continuation_end_epoch,
        )

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        super().load_checkpoint(filename_or_checkpoint)
        if self.current_epoch < self.continuation_start_epoch:
            raise RuntimeError(
                "V14's 400-epoch continuation must resume from epoch 100 or later; "
                f"checkpoint current_epoch={self.current_epoch}"
            )

        scheduler = self._continuation_scheduler()
        expected_loaded_lr = scheduler.lr_at(self.current_epoch - 1)
        loaded_lrs = [group["lr"] for group in self.optimizer.param_groups]
        if not all(
            np.isclose(lr, expected_loaded_lr, rtol=1e-6, atol=1e-12)
            for lr in loaded_lrs
        ):
            raise RuntimeError(
                "V14 continuation checkpoint LR does not match the frozen curve: "
                f"epoch={self.current_epoch}, loaded={loaded_lrs}, "
                f"expected={expected_loaded_lr}"
            )
        self.lr_scheduler = scheduler
        next_lr = scheduler.lr_at(self.current_epoch)
        self.print_to_log_file(
            "Installed continuous V14 epoch-100-to-400 LR schedule: "
            f"checkpoint_epoch={self.current_epoch}, loaded_lr={loaded_lrs[0]:.12g}, "
            f"next_lr={next_lr:.12g}; AdamW state preserved.",
            also_print_to_console=True,
        )

    def _network_module(self):
        module = self.network.module if isinstance(self.network, DDP) else self.network
        return module._orig_mod if isinstance(module, OptimizedModule) else module

    def _load_openmind_primus(self) -> None:
        checkpoint_path = self.pretrained_checkpoint
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"missing pretrained checkpoint: {checkpoint_path}")
        actual_sha256 = _sha256(checkpoint_path)
        if actual_sha256 != self.pretrained_sha256:
            raise RuntimeError(
                "Primus-M checkpoint hash changed: "
                f"{actual_sha256} != {self.pretrained_sha256}"
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        pretrained = checkpoint["network_weights"]
        encoder_state = {
            key.removeprefix("eva."): value
            for key, value in pretrained.items()
            if key.startswith("eva.")
        }
        stem_state = {
            key.removeprefix("down_projection."): value
            for key, value in pretrained.items()
            if key.startswith("down_projection.")
        }
        network = self._network_module()
        expected_encoder = network.eva.state_dict()
        expected_stem = network.down_projection.state_dict()
        for label, loaded, expected in (
            ("encoder", encoder_state, expected_encoder),
            ("stem", stem_state, expected_stem),
        ):
            if set(loaded) != set(expected):
                missing = sorted(set(expected) - set(loaded))
                unexpected = sorted(set(loaded) - set(expected))
                raise RuntimeError(
                    f"Primus-M {label} key mismatch: "
                    f"missing={missing[:5]}, unexpected={unexpected[:5]}"
                )
            wrong_shapes = [
                key
                for key in expected
                if tuple(expected[key].shape) != tuple(loaded[key].shape)
            ]
            if wrong_shapes:
                raise RuntimeError(
                    f"Primus-M {label} shape mismatch: {wrong_shapes[:5]}"
                )
        network.eva.load_state_dict(encoder_state, strict=True)
        network.down_projection.load_state_dict(stem_state, strict=True)
        self.pretrained_encoder_tensors = len(encoder_state)
        self.pretrained_stem_tensors = len(stem_state)
        self.print_to_log_file(
            "Loaded public Primus-M OpenMind MAE: "
            f"encoder={len(encoder_state)} tensors, stem={len(stem_state)} tensors, "
            f"sha256={actual_sha256}; decoder remains randomly initialized.",
            also_print_to_console=True,
        )

    def initialize(self) -> None:
        super().initialize()
        self._load_openmind_primus()

    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA,
        deep_supervision_scales,
        mirror_axes,
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm=None,
        is_cascaded: bool = False,
        foreground_labels=None,
        regions=None,
        ignore_label=None,
    ):
        composed = nnUNet_Primus_M_Trainer.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=use_mask_for_norm,
            is_cascaded=is_cascaded,
            foreground_labels=foreground_labels,
            regions=regions,
            ignore_label=ignore_label,
        )
        insertion_index = next(
            index
            for index, transform in enumerate(composed.transforms)
            if isinstance(transform, RandomTransform)
            and isinstance(transform.transform, GaussianNoiseTransform)
        )
        composed.transforms.insert(
            insertion_index,
            RandomTransform(
                RandomAmplitudeSpectrumSynthesisTransform(
                    alpha=nnUNetTrainerV14PrimusMAE.rass_alpha,
                    beta=nnUNetTrainerV14PrimusMAE.rass_beta,
                    gamma=nnUNetTrainerV14PrimusMAE.rass_gamma,
                ),
                apply_probability=nnUNetTrainerV14PrimusMAE.rass_probability,
            ),
        )
        return composed

    def _run_scheduled_whole_case_validation(self, completed_epoch: int) -> None:
        if self.local_rank != 0:
            return
        workspace = WORKSPACE
        split_path = Path(self.preprocessed_dataset_folder_base) / "splits_final.json"
        split_hash = _sha256(split_path)
        if split_hash != EXPECTED_SPLIT_SHA256:
            raise RuntimeError(f"frozen split hash changed: {split_hash}")

        epoch_tag = f"epoch{completed_epoch:03d}"
        checkpoint = Path(self.output_folder) / f"checkpoint_{epoch_tag}.pth"
        # With fold=all every labeled case is a training case, so a scheduled
        # whole-case pass would only score the training set. Retain the exact
        # checkpoints needed for final-data SWA without performing that pass.
        if str(self.fold) == "all":
            self.print_to_log_file(
                f"FULL-DATA CHECKPOINT SAVE START: {epoch_tag}",
                also_print_to_console=True,
            )
            self.save_checkpoint(str(checkpoint))
            original_bytes = checkpoint.stat().st_size
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            payload["optimizer_state"] = {}
            payload["grad_scaler_state"] = None
            temporary = checkpoint.with_suffix(".compact.tmp")
            try:
                torch.save(payload, temporary)
                temporary.replace(checkpoint)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
            compact_bytes = checkpoint.stat().st_size
            self.print_to_log_file(
                f"FULL-DATA CHECKPOINT SAVE COMPLETE: {epoch_tag}; "
                f"{original_bytes / 1024**2:.1f} -> {compact_bytes / 1024**2:.1f} MiB",
                also_print_to_console=True,
            )
            return
        metrics_dir = (
            workspace
            / "outputs"
            / self.research_version
            / f"fold{self.fold}"
            / "checkpoint_metrics"
            / epoch_tag
        )
        prediction_root = metrics_dir / "_prediction_work"
        if prediction_root.exists():
            shutil.rmtree(prediction_root)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        self.print_to_log_file(
            f"SCHEDULED WHOLE-CASE VALIDATION START: {epoch_tag}",
            also_print_to_console=True,
        )
        self.save_checkpoint(str(checkpoint))
        original_output_folder = self.output_folder
        try:
            self.output_folder = str(prediction_root)
            self._inside_scheduled_validation = True
            super().perform_actual_validation(save_probabilities=True)
        finally:
            self._inside_scheduled_validation = False
            self.output_folder = original_output_folder

        prediction_dir = prediction_root / "validation"
        subprocess.run(
            [
                sys.executable,
                str(workspace / "scripts" / "evaluate_isles26.py"),
                str(
                    workspace
                    / "data"
                    / "derived"
                    / "center_grouped_folds"
                    / "manifest.csv"
                ),
                str(prediction_dir),
                str(metrics_dir),
                "--fold",
                str(self.fold),
                "--label-dir",
                str(workspace / "nnUNet_raw" / "Dataset026_ISLES26" / "labelsTr"),
                "--workers",
                "8",
            ],
            check=True,
        )
        summary = json.loads((metrics_dir / "summary.json").read_text(encoding="utf-8"))
        metadata = {
            "version": self.research_version,
            "epoch": completed_epoch,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "split_sha256": split_hash,
            "selection_metrics": summary["case_mean"],
            "note": "pseudo Dice is not used for checkpoint selection",
        }
        (metrics_dir / "checkpoint_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        shutil.rmtree(prediction_root)
        self.print_to_log_file(
            f"SCHEDULED WHOLE-CASE VALIDATION COMPLETE: {epoch_tag} "
            f"metrics={json.dumps(summary['case_mean'], sort_keys=True)}",
            also_print_to_console=True,
        )

        original_bytes = checkpoint.stat().st_size
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        payload["optimizer_state"] = {}
        payload["grad_scaler_state"] = None
        temporary = checkpoint.with_suffix(".compact.tmp")
        try:
            torch.save(payload, temporary)
            temporary.replace(checkpoint)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        compact_bytes = checkpoint.stat().st_size
        self.print_to_log_file(
            f"Compacted evaluated {self.research_version} checkpoint for inference-only retention: "
            f"{checkpoint.name} {original_bytes / 1024**2:.1f} -> "
            f"{compact_bytes / 1024**2:.1f} MiB; checkpoint_latest remains resumable.",
            also_print_to_console=True,
        )

    def on_epoch_end(self) -> None:
        completed_epoch = self.current_epoch + 1
        if completed_epoch % self.scheduled_validation_interval == 0:
            self._run_scheduled_whole_case_validation(completed_epoch)
        super().on_epoch_end()

    def on_train_end(self) -> None:
        super().on_train_end()
        self._skip_next_post_training_validation = True

    def perform_actual_validation(self, save_probabilities: bool = False):
        if self._skip_next_post_training_validation and not self._inside_scheduled_validation:
            self._skip_next_post_training_validation = False
            self.print_to_log_file(
                f"Skipping duplicate post-training validation for {self.research_version}; epoch400 official "
                "whole-case metrics already exist.",
                also_print_to_console=True,
            )
            return None
        return super().perform_actual_validation(save_probabilities)
