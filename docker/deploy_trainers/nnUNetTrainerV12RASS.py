"""Inference-only architecture definition for ResEnc-L RASS SWA."""

from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import (
    nnUNetTrainer_250epochs,
)


class nnUNetTrainerV12RASS(nnUNetTrainer_250epochs):
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
