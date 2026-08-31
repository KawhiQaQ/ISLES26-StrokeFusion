# Grand Challenge submission

## Frozen resources

Final algorithm: **ResEnc-L RASS SWA + Primus-M Local-Refinement SWA**, equal
foreground-probability mean, mirror test-time augmentation, threshold 0.5,
followed by conservative low-confidence tiny-lesion cleanup. A 6-connected
component is removed only when it is below 0.002 mL and its maximum foreground
probability is below 0.65. The probability map is unchanged.

| Grand Challenge resource | Local release asset |
|---|---|
| Algorithm image | `submission_artifacts/isles26-strokefusion-rebuilt.tar.gz` |
| Model | `submission_artifacts/model-rebuilt.tar.gz` |

The image is `linux/amd64`, runs as a non-root user, listens on port 4743,
implements the `invoke` API, and requires an NVIDIA GPU. Weights live in the
separate Model resource and are verified when the container starts.

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
root.
