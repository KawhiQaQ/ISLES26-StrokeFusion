"""Base nnU-Net trainer with scheduled whole-case ISLES evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch
from torch import nn

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import (
    nnUNetTrainer_250epochs,
)


EXPECTED_SPLIT_SHA256 = (
    "da10108f65fdff2954c7f68f9e69456a5d9bb8f78b28512e3714656ba2bd9885"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AsymmetricUnifiedFocalLoss(nn.Module):
    """Binary 3D asymmetric UFL using the paper's published defaults."""

    def __init__(
        self,
        weight: float = 0.5,
        delta: float = 0.6,
        gamma: float = 0.5,
        smooth: float = 1e-6,
        ignore_label: int | None = None,
    ) -> None:
        super().__init__()
        self.weight = float(weight)
        self.delta = float(delta)
        self.gamma = float(gamma)
        self.smooth = float(smooth)
        self.ignore_label = ignore_label
        self.last_asymmetric_focal = None
        self.last_asymmetric_focal_tversky = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 5 or logits.shape[1] != 2:
            raise ValueError(f"V3 requires Bx2xDxHxW logits, found {logits.shape}")
        if target.ndim == logits.ndim and target.shape[1] == 1:
            target = target[:, 0]
        if target.shape != logits.shape[:1] + logits.shape[2:]:
            raise ValueError(f"target/logit mismatch: {target.shape} vs {logits.shape}")

        valid = torch.ones_like(target, dtype=torch.bool)
        if self.ignore_label is not None:
            valid = target != self.ignore_label
        foreground = ((target > 0) & valid).to(dtype=torch.float32)
        background = valid.to(dtype=torch.float32) - foreground
        one_hot = torch.stack((background, foreground), dim=1)

        # Computing probabilities and logarithms in fp32 avoids unstable focal
        # terms under AMP without changing the network's mixed-precision path.
        probability = torch.softmax(logits.float(), dim=1).clamp(1e-7, 1 - 1e-7)
        cross_entropy = -one_hot * probability.log()
        background_focal = (
            (1.0 - self.delta)
            * (1.0 - probability[:, 0]).pow(self.gamma)
            * cross_entropy[:, 0]
        )
        foreground_focal = self.delta * cross_entropy[:, 1]
        valid_count = valid.sum().clamp_min(1).to(dtype=torch.float32)
        asymmetric_focal = (
            (background_focal + foreground_focal) * valid
        ).sum() / valid_count

        spatial_axes = tuple(range(2, probability.ndim))
        true_positive = (one_hot * probability).sum(dim=spatial_axes)
        false_negative = (one_hot * (1.0 - probability)).sum(dim=spatial_axes)
        false_positive = (
            (1.0 - one_hot) * probability * valid[:, None]
        ).sum(dim=spatial_axes)
        tversky = (true_positive + self.smooth) / (
            true_positive
            + self.delta * false_negative
            + (1.0 - self.delta) * false_positive
            + self.smooth
        )
        background_tversky = 1.0 - tversky[:, 0]
        foreground_tversky = (1.0 - tversky[:, 1]).pow(1.0 - self.gamma)
        asymmetric_focal_tversky = torch.stack(
            (background_tversky, foreground_tversky), dim=1
        ).mean()

        self.last_asymmetric_focal = asymmetric_focal.detach()
        self.last_asymmetric_focal_tversky = asymmetric_focal_tversky.detach()
        return (
            self.weight * asymmetric_focal_tversky
            + (1.0 - self.weight) * asymmetric_focal
        )


class nnUNetTrainerV3UFL(nnUNetTrainer_250epochs):
    """PlainConv single model with UFL and in-process native whole-case checks."""

    research_version = "V3"

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.save_every = 25
        self.scheduled_validation_interval = 25
        self._inside_scheduled_validation = False
        self._skip_next_post_training_validation = False

    def _build_loss(self) -> nn.Module:
        loss: nn.Module = AsymmetricUnifiedFocalLoss(
            weight=0.5,
            delta=0.6,
            gamma=0.5,
            smooth=1e-6,
            ignore_label=self.label_manager.ignore_label,
        )
        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def _run_scheduled_whole_case_validation(self, completed_epoch: int) -> None:
        if self.local_rank != 0:
            return
        workspace = Path(os.environ.get("ISLES26_WORKSPACE", Path.cwd())).resolve()
        split_path = Path(self.preprocessed_dataset_folder_base) / "splits_final.json"
        split_hash = _sha256(split_path)
        if split_hash != EXPECTED_SPLIT_SHA256:
            raise RuntimeError(f"frozen split hash changed: {split_hash}")

        epoch_tag = f"epoch{completed_epoch:03d}"
        checkpoint = Path(self.output_folder) / f"checkpoint_{epoch_tag}.pth"
        # Full-data training has no independent validation set. Saving the
        # scheduled checkpoint is still required for SWA, but evaluating the
        # complete training set here would add hours and leak no useful model
        # selection signal. Subclasses such as V12 compact it after this call.
        if str(self.fold) == "all":
            self.print_to_log_file(
                f"FULL-DATA CHECKPOINT SAVE START: {epoch_tag}",
                also_print_to_console=True,
            )
            self.save_checkpoint(str(checkpoint))
            self.print_to_log_file(
                f"FULL-DATA CHECKPOINT SAVE COMPLETE: {epoch_tag}",
                also_print_to_console=True,
            )
            return
        metrics_dir = (
            workspace / "outputs" / self.research_version / f"fold{self.fold}"
            / "checkpoint_metrics" / epoch_tag
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
        summary_path = metrics_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
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

    def on_epoch_end(self) -> None:
        completed_epoch = self.current_epoch + 1
        if completed_epoch % self.scheduled_validation_interval == 0:
            self._run_scheduled_whole_case_validation(completed_epoch)
        super().on_epoch_end()

    def on_train_end(self) -> None:
        super().on_train_end()
        self._skip_next_post_training_validation = True

    def perform_actual_validation(self, save_probabilities: bool = False):
        # nnUNetv2_train normally repeats a final validation after run_training.
        # Epoch 250 is already evaluated by the scheduled hook, so skip only
        # that duplicate. Explicit --val on a loaded checkpoint still works.
        if self._skip_next_post_training_validation and not self._inside_scheduled_validation:
            self._skip_next_post_training_validation = False
            self.print_to_log_file(
                "Skipping duplicate post-training validation; epoch250 official "
                "whole-case metrics already exist.",
                also_print_to_console=True,
            )
            return None
        return super().perform_actual_validation(save_probabilities)
