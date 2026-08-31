# Grand Challenge submission

## Frozen resources

Final algorithm: **E1-SWA + V15-SWA**, equal foreground-probability mean,
mirror TTA, threshold 0.5, followed by the conservative low-confidence
tiny-lesion cleanup. A 6-connected component is removed only when it is below
0.002 ml and its maximum foreground probability is below 0.65. The probability
map is unchanged.

| Grand Challenge resource | Local release asset | SHA-256 |
|---|---|---|
| Algorithm image | `submission_artifacts/isles26-e1-v15-swa-postprocess-fixed.tar.gz` | `3a87ff03f71e4d05eb4e57e7bb4bf43716cf599612f9b0599c465ab07fff6484` |
| Model | `submission_artifacts/model-postprocess.tar.gz` | `06c159dce3059f319f916d264c78b2b5e76e29456f91f907077c50520446ffd6` |

The image is `linux/amd64`, runs as a non-root user, listens on port 4743,
implements the `invoke` API, and requires an NVIDIA GPU. Weights live in the
separate Model resource and are verified by SHA-256 at container startup.

## Interfaces and resources

- Inputs: **T1 Brain MRI** and **Stroke metadata**
- Outputs: **Stroke Lesion Segmentation** and **Lesion Probability Map**
- Instance: **NVIDIA T4 Tensor Core GPU (16 GB VRAM)**
- Maximum main memory: **32 GB**
- Per-case time limit: **7 minutes**

Upload and activate the Algorithm image, then upload and activate the Model
resource for the same algorithm. Do not configure the frozen two-member
ensemble as CPU-only.

## Preliminary reference

The exact full-data Docker path was run on the two published Preliminary cases:

| Case | Dice | PR-AUC |
|---|---:|---:|
| `sub-r001s001` | 0.866565 | 0.956818 |
| `sub-soop0468` | 0.958098 | 0.994933 |
| Mean | **0.912331** | **0.975876** |

The complete five-metric record is retained in
`results/preliminary_metrics.json`. These two training cases are a packaging
check, not a model-selection validation set.

## Build and verification

```bash
bash scripts/restore_final_models.sh
bash scripts/package_final_model.sh
cd docker
bash do_build.sh
bash do_save.sh
```

Before upload, run `bash scripts/verify_final_release.sh` from the repository
root and compare the two release hashes above.
