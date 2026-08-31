"""ResEnc-L training with random amplitude spectrum synthesis."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from scipy import fft

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerV9OpenMindMAE import (
    nnUNetTrainerV9OpenMindMAE,
)


class RandomAmplitudeSpectrumSynthesisTransform(ImageOnlyTransform):
    """Perturb the amplitude spectrum while retaining image phase and support.

    The frequency-dependent standard deviation follows Qiao et al., MICCAI
    2024: sigma = 2 * alpha * r2**gamma + beta. FFT work remains on the CPU in
    the augmentation workers, so the inference graph and GPU footprint are
    unchanged.
    """

    def __init__(self, alpha: float = 3.0, beta: float = 0.25, gamma: float = 2.0):
        super().__init__()
        if alpha < 0 or beta < 0 or gamma <= 0:
            raise ValueError("RASS requires alpha>=0, beta>=0 and gamma>0")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self._sigma_cache: dict[tuple[int, ...], np.ndarray] = {}

    def _sigma(self, shape: tuple[int, ...]) -> np.ndarray:
        cached = self._sigma_cache.get(shape)
        if cached is not None:
            return cached
        frequency_axes = [np.fft.fftfreq(size) * size for size in shape[:-1]]
        frequency_axes.append(np.fft.rfftfreq(shape[-1]) * shape[-1])
        grids = np.meshgrid(*frequency_axes, indexing="ij", sparse=True)
        squared_radius = sum(
            grid.astype(np.float32, copy=False) ** 2 for grid in grids
        ) / float(sum(size**2 for size in shape))
        sigma = (
            2.0 * self.alpha * np.power(squared_radius, self.gamma) + self.beta
        ).astype(np.float32, copy=False)
        self._sigma_cache[shape] = sigma
        return sigma

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        if img.device.type != "cpu":
            raise RuntimeError("V12 RASS must run in CPU augmentation workers")
        if img.ndim not in (3, 4):
            raise ValueError(f"expected channel-first 2D/3D image, got {img.shape}")

        image = img.detach().numpy()
        output = np.empty_like(image, dtype=np.float32)
        spatial_shape = tuple(int(i) for i in image.shape[1:])
        sigma = self._sigma(spatial_shape)
        for channel_index, channel in enumerate(image):
            support = channel != 0
            spectrum = fft.rfftn(channel.astype(np.float32, copy=False), workers=1)
            standard_normal = np.random.standard_normal(sigma.shape).astype(
                np.float32, copy=False
            )
            multiplier = 1.0 + sigma * standard_normal
            synthesized = fft.irfftn(
                spectrum * multiplier,
                s=spatial_shape,
                workers=1,
            ).astype(np.float32, copy=False)
            synthesized[~support] = 0.0
            output[channel_index] = synthesized
        return torch.from_numpy(output).to(dtype=img.dtype)


class nnUNetTrainerV12RASS(nnUNetTrainerV9OpenMindMAE):
    """Preserve V9 and add one memory-neutral cross-center augmentation."""

    research_version = "V12"
    rass_probability = 0.30
    rass_alpha = 3.0
    rass_beta = 0.25
    rass_gamma = 2.0
    # Preserve each official candidate, but strip optimizer-only resume state
    # after its evaluation to fit the bounded data disk. checkpoint_latest
    # remains the full resumable state during training.
    retained_scheduled_checkpoints = frozenset(range(50, 401, 50))

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
        composed = nnUNetTrainerV9OpenMindMAE.get_training_transforms(
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
                    alpha=nnUNetTrainerV12RASS.rass_alpha,
                    beta=nnUNetTrainerV12RASS.rass_beta,
                    gamma=nnUNetTrainerV12RASS.rass_gamma,
                ),
                apply_probability=nnUNetTrainerV12RASS.rass_probability,
            ),
        )
        return composed

    def _run_scheduled_whole_case_validation(self, completed_epoch: int) -> None:
        super()._run_scheduled_whole_case_validation(completed_epoch)
        if self.local_rank != 0:
            return
        checkpoint = Path(self.output_folder) / f"checkpoint_epoch{completed_epoch:03d}.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"scheduled V12 checkpoint missing: {checkpoint}")
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
            "Compacted evaluated V12 checkpoint for inference-only retention: "
            f"{checkpoint.name} {original_bytes / 1024**2:.1f} -> "
            f"{compact_bytes / 1024**2:.1f} MiB; checkpoint_latest remains resumable.",
            also_print_to_console=True,
        )
