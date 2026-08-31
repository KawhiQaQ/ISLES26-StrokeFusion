# V15 — Lesion-presence-gated local/global Primus-M

V15 is the final planned Transformer single-model experiment. It keeps V14's
public OpenMind-MAE initialized Primus-M encoder and V12's validated
training-only RASS augmentation, but trains from epoch 0 with one native
400-epoch schedule. It does not resume V14 or inherit V14's interrupted
150-to-400 continuation schedule.

## Failure analysis that determines the change

V14-e250 is genuinely complementary to E1: in the smallest lesion-volume
quartile it improves PR-AUC by 0.0172, Dice by 0.0074 and lesion F1 by 0.0180.
It also predicts fewer positive validation cases as empty than E1 (9 versus
12). Its lesion-count problem is therefore not simply low sensitivity.

The regression is predominantly fragmentation and coarse localization. Against
E1, V14 worsens absolute lesion-count error by 0.526 on single-lesion cases,
0.683 on 2–3-lesion cases and 0.262 on cases with at least four lesions. The
largest degradation is the second-smallest volume stratum (0.903 extra count
error), while the largest lesions lose 2.118 ml AVD and 0.080 lesion F1. The
early subacute 8–30 day group loses 1.088 count-error units and the ATLAS3
source loses 1.267. This is consistent with an 8-cubed-token Transformer whose
minimal transposed-convolution decoder lacks local multi-scale context.

Equal-probability E1+V14 improves the local official-style aggregate rank, but
increases predicted-empty positive cases to 21 because two differently
calibrated models must jointly push their average above 0.5. V15 must therefore
retain V14's tiny-lesion complementarity while producing more spatially
coherent and E1-compatible probabilities.

## Architecture change

The public RAW patch stem produces an 864-channel 20-cubed feature grid before
the Transformer. V15 retains this pretrained local representation and adds a
capacity-limited residual path at that same token resolution:

1. a 1x1x1 projection from 864 to 64 channels;
2. a 3x3x3 depthwise convolution that restores local 3D adjacency;
3. a 1x1x1 projection back to 864 channels;
4. addition to the final global Transformer feature immediately before the
   unchanged Primus decoder.

A one-channel token-presence head predicts whether each 8-cubed token contains
any lesion. Its sigmoid-derived gate can modulate the local residual by at most
25%, but cannot suppress or replace the global Transformer path. Training adds
a low-weight (0.05) balanced binary cross-entropy loss on max-pooled lesion
occupancy. This supplies coarse lesion-presence supervision to both positive
and negative tokens, so it can discourage isolated false-positive fragments
without the aggressive component weighting that damaged prior versions.

Both the residual output projection and token-presence head are zero
initialized. Consequently the initial segmentation function and local gate are
exactly the unmodified pretrained Primus-M function. The additional operators
run only on the 20-cubed grid; Docker inference cost remains close to V14 and
well below that of adding another full-resolution decoder.

## Deliberate exclusions

- V1.2's component-adaptive Tversky reduced missed lesions but worsened count
  error by 0.398 and lesion F1 by 0.0166.
- V4's component-local DiceCE increased count error to 5.327 and lesion F1
  collapsed to 0.214 at the first evaluation.
- V11's component-uniform sampler again worsened count error and lesion F1.

V15 therefore does not use connected-component reweighting, recall-heavy
sampling, threshold tuning, validation-derived post-processing, or metadata as
input. The token-presence auxiliary target is a deterministic max pooling of
the augmented training mask and never reads validation masks.

## Frozen protocol and deployment

- Official RAW/native, skull-stripped T1 only.
- Frozen center-grouped split SHA256
  `da10108f65fdff2954c7f68f9e69456a5d9bb8f78b28512e3714656ba2bd9885`.
- V15-specific plans permanently record the actual 160x160x160 patch and batch
  size 2. This prevents the V14 training/deployment positional-embedding
  mismatch.
- One single model, 400 epochs from epoch 0, official whole-case fold-0
  evaluation every 50 epochs; pseudo Dice is health-only.
- E1+V15 is evaluated only after a mature V15 checkpoint exists, with fixed
  equal probability weights and no validation-case weight search.

Primary design evidence:

- Primus: https://arxiv.org/abs/2503.01835
- HyLT local/global Transformer fusion:
  https://proceedings.mlr.press/v172/luo22a.html
- FuseUNet multi-scale feature integration (ICML 2025):
  https://proceedings.mlr.press/v267/he25p.html
- ICI lesion-center supervision on ATLAS v2.0:
  https://arxiv.org/abs/2304.06229
