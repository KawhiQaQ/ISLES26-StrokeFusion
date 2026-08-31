# StrokeFusion: Confidence-Guided Hybrid Segmentation

**English** | [简体中文](README_zh-CN.md)

<p align="center">
  <strong>Official implementation of our ISLES'26 submission</strong><br>
  Native-space ischemic stroke lesion segmentation on T1-weighted MRI
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11-3776AB.svg" alt="Python 3.11"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.13-EE4C2C.svg" alt="PyTorch 2.13"></a>
  <a href="https://github.com/MIC-DKFZ/nnUNet"><img src="https://img.shields.io/badge/nnU--Net-2.8.1-4B8BBE.svg" alt="nnU-Net 2.8.1"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-green.svg" alt="Apache-2.0"></a>
  <a href="https://grand-challenge.org/algorithms/strokefusion-confidence-guided-hybrid-segmentation/"><img src="https://img.shields.io/badge/Grand%20Challenge-StrokeFusion-5A4FCF.svg" alt="Grand Challenge algorithm"></a>
</p>

StrokeFusion is a confidence-guided ensemble of a residual-encoder 3D U-Net
and a locally refined Primus-M Transformer. Both members are initialized with
public OpenMind masked-autoencoder weights and trained on the official
ISLES'26 RAW/native-space release. Their foreground probabilities are averaged
uniformly. A conservative confidence-aware cleanup is applied only to the
binary mask; the submitted probability map remains unchanged.

## Method

| Component | Frozen specification |
|---|---|
| **ResEnc-L RASS SWA** | nnU-Net v2 ResEnc-L, `160 × 192 × 160` patch, OpenMind-MAE initialization, training-only RASS, weight average of epochs 300/350/400 |
| **Primus-M Local-Refinement SWA** | Primus-M, `160³` patch, OpenMind-MAE initialization, low-rank local patch-stem refinement, token-presence auxiliary loss, training-only RASS, weight average of epochs 200/250/300 |
| **Probability fusion** | Equal foreground-probability average with mirror test-time augmentation |
| **Confidence-guided output** | Submit the probability map unchanged; threshold at `0.5`, then remove a 6-connected component only when volume `< 0.002 mL` and maximum foreground probability `< 0.65` |

The Transformer refinement is zero-initialized and operates on the native
Primus token grid. It adds an `864 → 64 → 864` local residual path containing
a depthwise `3 × 3 × 3` convolution. A lesion-presence head gates this path and
adds a `0.05`-weighted balanced token-level binary cross-entropy objective to
the standard Dice and cross-entropy segmentation loss.

Historical trainer class and checkpoint-directory names are retained
internally for compatibility with the trained weights. They are implementation
identifiers, not model names.

## Repository layout

```text
configs/                 frozen model and training specifications
docker/                  Grand Challenge inference container
docs/                    method, evaluation, dataset, and submission details
external_models/         expected locations for public OpenMind checkpoints
nnunet_extensions/       custom nnU-Net trainers used by StrokeFusion
plans/                    frozen nnU-Net architecture plans
reproducibility/         frozen inference metadata
results/                 compact Preliminary Evaluation summary
scripts/                 preparation, training, evaluation, and packaging tools
```

## Reproduce the final method

### 1. System requirements

- Linux x86-64
- Conda or Miniforge
- Python 3.11
- CUDA-capable NVIDIA GPU; 24 GB VRAM is recommended for training
- Docker with NVIDIA Container Toolkit for container testing
- Enough storage for the licensed RAW release and nnU-Net preprocessing cache

The submitted inference container targets an NVIDIA T4 GPU with 16 GB VRAM.

### 2. Clone and create the environment

```bash
git clone git@github.com:KawhiQaQ/ISLES26-StrokeFusion.git
cd ISLES26-StrokeFusion

bash scripts/bootstrap_env.sh isles26
conda activate isles26
```

The concise dependency specification is in `requirements-core.txt`; the
resolved reference environment is retained in `environment.lock.txt`.

### 3. Prepare the licensed RAW data

Download and extract the official ISLES'26/ATLAS R3.0 **RAW** training release.
The standardized/preprocessed archive is not used. Place the extracted folder
at exactly:

```text
data/raw/ATLAS3_Training_Raw/
```

The expected archive identity and data inventory are documented in
[`docs/dataset.md`](docs/dataset.md). Do not store the dataset encryption key
in scripts, shell history, Docker layers, or Git.

### 4. Download the public self-supervised initialization

```bash
hf download MIC-DKFZ/ResEncL-OpenMind-MAE checkpoint_final.pth \
  --local-dir external_models/ResEncL-OpenMind-MAE

hf download MIC-DKFZ/PrimusM-OpenMind-MAE checkpoint_final.pth \
  --local-dir external_models/PrimusM-OpenMind-MAE
```

| Initialization | Local path |
|---|---|
| ResEnc-L OpenMind-MAE | `external_models/ResEncL-OpenMind-MAE/checkpoint_final.pth` |
| Primus-M OpenMind-MAE | `external_models/PrimusM-OpenMind-MAE/checkpoint_final.pth` |

These checkpoints were pretrained without labels on the OpenMind/OpenNeuro
brain-MRI collection. They are not derived from the ISLES validation or test
sets.

### 5. Generate the frozen split and preprocess

```bash
bash scripts/prepare_data.sh setup
bash scripts/prepare_data.sh plan
bash scripts/prepare_data.sh preprocess
```

`setup` deterministically rebuilds the center-grouped five-fold split from the
licensed release. Generated manifests and patient-level split files remain
local and are ignored by Git.

### 6. Train both full-data members

```bash
bash scripts/train_strokefusion.sh
```

The script performs the complete frozen procedure:

1. train ResEnc-L RASS for 400 epochs on fold `all`;
2. average epochs 300, 350, and 400 into ResEnc-L RASS SWA;
3. train Primus-M Local-Refinement for 400 epochs on fold `all`;
4. average epochs 200, 250, and 300 into Primus-M Local-Refinement SWA;
5. persist both verified models under `outputs/final_full_models/`.

Training resumes safely from each member's `checkpoint_latest.pth` when
available.

## Released weights

| Artifact | Download | Extraction code |
|---|---|---|
| Complete StrokeFusion model bundle (both SWA checkpoints, plans, metadata, and Grand Challenge Model resource) | [Baidu Netdisk](https://pan.baidu.com/s/1U1IR3pC7o73z7Xe1IdFDdw?pwd=z6hx) | `z6hx` |

Place the downloaded `model-postprocess.tar.gz` at
`submission_artifacts/model-postprocess.tar.gz` and restore the two checkpoint
files with:

```bash
bash scripts/restore_final_models.sh
```

After downloading all licensed and released artifacts, validate the frozen
release with:

```bash
bash scripts/verify_final_release.sh
```

## Grand Challenge inference container

The container follows the official ISLES'26 `invoke` interface. It accepts a
native-space T1 MRI and stroke metadata and returns both the binary lesion mask
and float probability map in the original geometry. Metadata is accepted for
interface compatibility but is not used by the model.

```bash
cd docker
bash do_build.sh
bash do_test_run.sh
bash do_save.sh
```

See [`docker/README.md`](docker/README.md) and
[`docs/submission.md`](docs/submission.md) for the exact input/output sockets,
resource limits, and packaging contract.

## Evaluation

The evaluation code implements Dice, lesion F1, PR-AUC, absolute volume
difference, and absolute lesion-count difference:

```bash
python scripts/evaluate_isles26.py --help
python scripts/evaluate_probability_ensemble.py --help
```

The two published Preliminary cases are recorded in
[`results/preliminary_metrics.json`](results/preliminary_metrics.json). They
are used only as a container sanity check, not for model selection. The local
model-comparison pool and case-level predictions are intentionally excluded
because they are derived from the licensed training release.

## Reproducibility notes

- Supervised training uses only the official RAW release.
- Center identity, days post stroke, and chronicity are not model inputs.
- RASS is training-only and does not modify the inference graph.
- The two ensemble weights, threshold, and cleanup criteria are fixed.
- The probability map is never altered by binary-mask postprocessing.
- Artifact verification and restoration instructions are in
  [`docs/reproducibility.md`](docs/reproducibility.md).

## References

- Wald et al., [An OpenMind for 3D Medical Vision Self-Supervised Learning](https://arxiv.org/abs/2412.17041), 2024.
- Wald et al., [Revisiting MAE Pre-training for 3D Medical Image Segmentation](https://arxiv.org/abs/2410.23132), 2024.
- Wald et al., [Primus: Enforcing Attention Usage for 3D Medical Image Segmentation](https://arxiv.org/abs/2503.01835), 2025.
- Isensee et al., [nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation](https://arxiv.org/abs/2404.09556), 2024.

Citation information for the ISLES'26 challenge manuscript will be added after
publication.

## License

This repository is released under the [Apache License 2.0](LICENSE). The
ISLES'26 dataset and pretrained or trained weights remain subject to their own
licenses and terms of use.
