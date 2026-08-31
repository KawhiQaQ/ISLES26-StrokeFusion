# Final solution reproducibility

The frozen StrokeFusion submission combines **ResEnc-L RASS SWA** and
**Primus-M Local-Refinement SWA**. Both consume the official skull-stripped
RAW/native-space T1 image. Their foreground probabilities are averaged
uniformly with mirror test-time augmentation. The unchanged mean is submitted
as the probability map. The binary mask uses threshold 0.5 and removes only
6-connected components smaller than 0.002 mL whose maximum foreground
probability is below 0.65.

## Immutable inputs

- encrypted licensed ISLES'26/ATLAS R3.0 RAW archive;
- deterministically generated center-grouped five-fold split;
- ResEnc-L OpenMind-MAE initialization;
- Primus-M OpenMind-MAE initialization.

Exact file identities are enforced by the supplied configuration and
verification scripts. The dataset and its encryption key are licensed and are
intentionally excluded from Git. Public pretrained checkpoints are also
excluded as large binaries. Store them at:

```text
external_models/ResEncL-OpenMind-MAE/checkpoint_final.pth
external_models/PrimusM-OpenMind-MAE/checkpoint_final.pth
```

## Frozen full-data weights

- **ResEnc-L RASS SWA:** uniform weight average of epochs 300, 350, and 400.
- **Primus-M Local-Refinement SWA:** uniform weight average of epochs 200, 250,
  and 300.

The SWA checkpoint metadata retains the source-checkpoint identities and the
verification script checks both released models before packaging or inference.

## Reproduce training

1. Obtain and extract the licensed RAW release at
   `data/raw/ATLAS3_Training_Raw`.
2. Put both public OpenMind checkpoints under `external_models/` at the paths
   above.
3. Create the environment with `bash scripts/bootstrap_env.sh isles26`.
4. Run `bash scripts/prepare_data.sh setup`,
   `bash scripts/prepare_data.sh plan`, and
   `bash scripts/prepare_data.sh preprocess`. Setup deterministically
   regenerates the local manifest and center-grouped split from licensed data.
5. Run `bash scripts/train_strokefusion.sh`. This trains both members on fold
   `all`, builds the two fixed SWA checkpoints, verifies source compatibility,
   and stores the final checkpoints in `outputs/final_full_models`.

The scripts derive the workspace from their own location. Machine-specific
locations remain configurable through `ISLES26_WORKSPACE`, `nnUNet_raw`,
`nnUNet_preprocessed`, `nnUNet_results`, and `ISLES26_FULL_RESULTS`.

The exact inference metadata extracted from the submitted model archive is
tracked in `reproducibility/model_metadata/`. Training configurations,
architecture plans, and custom trainer implementations are in `configs/`, `plans/`, and
`nnunet_extensions/`. Historical class and directory identifiers are retained
internally only for checkpoint compatibility.

## Release artifacts

- Grand Challenge algorithm image: generated as
  `submission_artifacts/isles26-strokefusion-rebuilt.tar.gz`.
- Grand Challenge Model resource: generated as
  `submission_artifacts/model-rebuilt.tar.gz`.

These large files belong in GitHub Releases or external model storage, not Git
history. Run `bash scripts/verify_final_release.sh` before publishing or
submitting them.

To restore the frozen final checkpoints from the retained Model resource, run
`bash scripts/restore_final_models.sh`. To rebuild that resource after
full-data training, run `bash scripts/package_final_model.sh`. Build and export
the algorithm image from `docker/` using `bash do_build.sh` and
`bash do_save.sh`.

The exact full-data Docker path produced the following two-case Preliminary
means after cleanup: Dice 0.9123314102, PR-AUC 0.9758757533, absolute volume
difference 3.8912479280 mL, lesion-count difference 1.0, and lesion F1 0.7.
