# V2 — nnU-Net ResEnc-L

V1 was a standard 3D PlainConvUNet. V1.1--V1.4 explored multi-size labels,
context attention, contralateral symmetry, and disease-stage FiLM. None made a
large, statistically credible improvement on frozen fold 0. V1.4 improved
missed-lesion counts but over-segmented, especially for center R032 and scans
31--179 days after stroke.

V2 changes the single-model architecture to nnU-Net's official 24 GB
Residual Encoder L preset. It uses a deeper residual encoder with blocks
`[1, 3, 4, 6, 6, 6]`, a larger `160 x 192 x 160` patch, and batch size 3.
The RTX 4090 matches the preset's documented 24 GB target. The nnU-Net authors
report that residual encoder presets substantially improve segmentation and
recommend ResEnc-L as the new default for 24 GB GPUs.

This is an architecture-level V2 baseline, not a tuning pass. RAW T1 remains
the only input. The original Dice+CE loss, augmentation, optimizer, schedule,
250-epoch budget, frozen center-grouped split, native export, and official
five-metric evaluation remain unchanged. No CENTER, days-post-stroke, or
validation statistic enters the model. If V2 is sound but still misses small
instances, a later single-model version (V3 or newer) can add connected-component/instance-wise
supervision motivated by Blob Loss and CC-DiceCE.

The independently generated plans file is preserved with the version and does
not overwrite V1's plans.

## Gate

Fold 0 is the only research fold. Fold 1 runs once only if official fold-0
Dice exceeds 0.60 with comparable improvement in the other four challenge
metrics. Fold 2 is not run.

References:

- https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md
- https://arxiv.org/abs/2404.09556
- https://arxiv.org/abs/2205.08209
- https://arxiv.org/abs/2511.17146
