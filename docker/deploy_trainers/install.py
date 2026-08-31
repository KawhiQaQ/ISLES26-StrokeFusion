"""Install the two inference-only trainer definitions into nnU-Net."""

from pathlib import Path
import shutil

import nnunetv2


source = Path(__file__).resolve().parent
destination = (
    Path(nnunetv2.__file__).resolve().parent / "training" / "nnUNetTrainer"
)
for name in ("nnUNetTrainerV12RASS.py", "nnUNetTrainerV15NativeRefinement.py"):
    shutil.copy2(source / name, destination / name)
print(f"Installed deployment trainers into {destination}")
