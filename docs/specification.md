# Method specification

## Task and data invariants

- Segment ischemic stroke lesions from one native-space, skull-stripped T1 MRI.
- Train only on the licensed ISLES'26/ATLAS R3.0 RAW release.
- Preserve native geometry in submitted masks and probability maps.
- Never introduce subject or center leakage across validation folds.
- Never store the dataset, its encryption key, or patient images in Git.

The frozen five-fold split groups by center and is deterministically generated
from the licensed RAW release at `versions/V1/splits_final.json`; its expected
SHA-256 is recorded in the configs. The generated split and manifest remain
local rather than being redistributed. Fold0 was the primary development fold;
the split was not changed during model development.

## Final ensemble

### E1-SWA

- nnU-Net v2 residual-encoder 3D U-Net, large configuration
- ResEnc-L OpenMind-MAE initialization
- Native-space RASS augmentation during training only
- 400-epoch schedule
- Uniform checkpoint average of epochs 300, 350, and 400

### V15-SWA

- Primus-M OpenMind-MAE transformer
- Low-rank local patch-stem refinement fused into the published decoder
- Native-space RASS augmentation during training only
- 400-epoch schedule
- Uniform checkpoint average of epochs 200, 250, and 300

### Inference

1. Run both members with mirror TTA.
2. Average their foreground probability maps uniformly.
3. Submit the unchanged mean as the lesion probability map.
4. Threshold at 0.5 for the binary mask.
5. Remove a 6-connected component only when it is smaller than 0.002 ml and
   its maximum foreground probability is below 0.65.

## Evaluation

Track Dice, lesion F1, PR-AUC, absolute lesion-count difference, and absolute
volume difference. Under the retained Rule-2 analysis:

1. Average each metric over all cases for each method.
2. Rank methods separately on each mean metric.
3. Rank Dice, lesion F1, and PR-AUC descending.
4. Rank count and volume differences ascending.
5. Average the five metric positions; lower is better.

Legal empty masks retain their normally defined metric values and remain in the
all-case means. No additional cross-metric penalty is introduced.

## Experiment history

`versions/`, `configs/`, and `docs/research-log.md` preserve the exploratory
path from V1 through V15. Those files document ablations and rejected methods;
they are not all required to run the frozen final ensemble.
