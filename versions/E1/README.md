# E1 — V12 late-checkpoint SWA

E1 is a single-inference model obtained by uniformly averaging the network
weights of V12 epochs 300, 350, and 400. All three checkpoints lie on the same
OpenMind-pretrained ResEnc-L + RASS training trajectory and have identical
network keys and tensor shapes. Optimizer state is excluded.

This is a predeclared equal-weight average; no coefficient or checkpoint-subset
sweep is performed on fold 0. The hypothesis is that averaging the mature,
low-learning-rate portion of one trajectory reduces checkpoint variance while
retaining V12's cross-center robustness. Unlike a probability ensemble, E1
uses exactly one ResEnc-L forward pass and therefore does not increase Docker
inference time. A final all-data version would require only one 400-epoch V12
training run with the three late checkpoints retained.

E1 uses the same frozen fold, RAW/native-derived preprocessing, inputs,
architecture, and inference pipeline as V12. It is promoted only if the
official five-metric whole-case evaluation and case-level aggregate rank show a
coherent improvement over V12-e400 and the other retained single-model pool.

## Fold-0 result

The predeclared average completed all 284 frozen fold-0 cases. It reaches
PR-AUC 0.610256, Dice 0.534525, AVD 5.634948 ml, lesion-count error 1.816901,
and lesion F1 0.460115. All five case means improve over V12-e400: +0.001868
PR-AUC, +0.000787 Dice, -0.130154 ml AVD, -0.080986 count error, and +0.010727
lesion F1. In the direct official-style case-wise comparison, E1 has mean rank
1.447 versus V12-e400's 1.553 and wins the per-case aggregate rank on 56.7% of
cases.

The improvement is not uniform: E1 has 46 zero-Dice cases versus V12-e400's
41 and 12 positive cases predicted empty versus 10. The gain is mainly better
probability/volume stability and instance quality on detected lesions rather
than better tiny-lesion recall. Retain E1 as the strongest zero-extra-inference
candidate, while subsequent ensembles must explicitly track missed-case risk.
