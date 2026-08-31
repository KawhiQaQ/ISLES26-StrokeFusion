# Final solution reproducibility

This repository's frozen final candidate is **E1-SWA + V15-SWA**. Both models
consume the official skull-stripped RAW/native-space T1 image. Their foreground
probabilities are averaged uniformly, mirror TTA is enabled, and the binary mask
uses threshold 0.5. The probability map is submitted unchanged. The binary mask
then removes only 6-connected components smaller than 0.002 ml whose maximum
foreground probability is below 0.65.

## Immutable inputs

| Item | SHA-256 |
|---|---|
| Encrypted licensed RAW archive | `62d55b2fe862961ec598ebe28ec8c9ae17711359d7e60475ef88e3785bcdeacc` |
| Frozen center-grouped split | `da10108f65fdff2954c7f68f9e69456a5d9bb8f78b28512e3714656ba2bd9885` |
| ResEnc-L OpenMind-MAE initialization | `7a847af785635335c00e711d16ff4d225d86ecd5992b14c059df2b520e3ee933` |
| Primus-M OpenMind-MAE initialization | `b866ac5f61d7e90d3a6cbb00a759ffc9d73beb5e63baa6b3cd654671ebc9a552` |

The dataset and its encryption key are licensed and are intentionally excluded
from Git. The public pretrained checkpoints are also ignored as large binaries.
Store them as `external_models/ResEncL-OpenMind-MAE/checkpoint_final.pth` and
`external_models/PrimusM-OpenMind-MAE/checkpoint_final.pth`.

## Frozen full-data weights

E1 is the uniform weight average of V12 epochs 300/350/400. V15-SWA is the
uniform weight average of V15 epochs 200/250/300.

| Checkpoint | SHA-256 |
|---|---|
| E1-SWA | `b974c29405011d4ddcb1850544c6d6fc054cc39544d3ada99b4b2808db88647b` |
| V15-SWA | `98429330ea0e800d0ce9d38b446494343fdaeba04c7e83caa6a50ff032dc05d2` |

The SWA checkpoints retain the source checkpoint hashes in their
`checkpoint_swa` metadata. E1 source hashes are
`08e425f7a7a413af1bd16eb94ef795eb92f4ff33c14bc74817d139746b50ad24`,
`28a68f18123383f4343aa529d5e3459717a596a1a477a93b973a4cd1ca5e04fd`,
and `32f410c42510c8adf4da81d222b771d7d1979f03da40d9860c9ec083cbb0c872`.
V15 source hashes are
`54cc658eaa34a45483e99d9bcf29494b8d0a97dc2b10e47390cf38b42262a3d8`,
`ccd33592fe10ca14ac74e0eed9184af89407b98a54f0e5e4c500bb7e7aaba9e3`,
and `4aca3f1e7f488e0810be67535cd804fa2492d7b6dbf5aaf0e586b1c0a4eb19ac`.

## Reproduce training

1. Obtain and extract the licensed RAW release at
   `data/raw/ATLAS3_Training_Raw`.
2. Put both public OpenMind checkpoints under `external_models/` as specified
   by the configs.
3. Create the environment with `bash scripts/bootstrap_env.sh isles26`.
4. Run `bash scripts/run_v1.sh setup`, `bash scripts/run_v1.sh plan`, and
   `bash scripts/run_v1.sh preprocess`. Setup deterministically regenerates the
   local manifest and center-grouped split from licensed data; verify the split
   against the SHA-256 above. These files and the nnU-Net preprocessing cache
   are not redistributed.
5. Run `bash scripts/run_full_final.sh`. This trains V12 and V15 on fold `all`,
   builds the two fixed SWAs, verifies source compatibility, and persists the
   final checkpoints in `outputs/final_full_models`.

The scripts derive the workspace from their own location. Machine-specific
locations remain configurable through `ISLES26_WORKSPACE`, `nnUNet_raw`,
`nnUNet_preprocessed`, `nnUNet_results`, and `ISLES26_FULL_RESULTS`.

The exact inference metadata extracted from the submitted model archive is
tracked in `reproducibility/model_metadata/`. Training configuration and custom
trainer implementations are in `configs/`, `versions/`, and
`nnunet_extensions/`.

## Frozen release artifacts

| Artifact | SHA-256 |
|---|---|
| `isles26-e1-v15-swa-postprocess-fixed.tar.gz` | `3a87ff03f71e4d05eb4e57e7bb4bf43716cf599612f9b0599c465ab07fff6484` |
| `model-postprocess.tar.gz` | `06c159dce3059f319f916d264c78b2b5e76e29456f91f907077c50520446ffd6` |

These large files belong in GitHub Releases or external model storage, not Git
history. Run `bash scripts/verify_final_release.sh` before publishing or
submitting them.

To restore the frozen final checkpoints from the retained Model resource, run
`bash scripts/restore_final_models.sh`. To rebuild that resource after
full-data training, run `bash scripts/package_final_model.sh`. Build the
algorithm image from `docker/` with `bash do_build.sh`, then export it with
`docker save isles26-e1-v15-swa-final | gzip -c > algorithm-image.tar.gz`.
Set `ISLES26_DOCKER_TEMPLATE` only if the template is stored elsewhere.
Container and gzip timestamps may change the outer archive hash; the packaging
script and runtime manifest verify the immutable checkpoint hashes instead.

The exact full-data Docker path produced the following two-case Preliminary
means after cleanup: Dice 0.9123314102, PR-AUC 0.9758757533, absolute volume
difference 3.8912479280 ml, lesion-count difference 1.0, and lesion F1 0.7.
