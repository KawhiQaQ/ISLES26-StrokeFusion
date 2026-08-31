# V12 — frequency-domain cross-center robustness

V12 is one model, not an ensemble. It keeps V9's RAW/native-space fold, public
OpenMind-MAE encoder, ResEnc-L architecture, original DiceCE objective,
no-deep-supervision output, 400-epoch sawtooth schedule and 50-epoch official
whole-case validation cadence.

The only scientific change is training-only Random Amplitude Spectrum Synthesis
(RASS). The amplitude spectrum of 30% of patches is multiplied by independent
Gaussian perturbations whose standard deviation increases with spatial
frequency; the original phase, labels and skull-stripped support are retained.
The published MICCAI 2024 parameters (`alpha=3`, `beta=0.25`, `gamma=2`) are
used without a validation sweep. The transform is absent during validation and
inference, so V12 has exactly V9's inference graph and cost.

This targets a specific V9 failure mode: substantial performance variation
across acquisition centers and chronicity groups despite a strong pretrained
encoder. The hypothesis is that exposing the model to plausible scanner/style
variation in the Fourier amplitude will reduce reliance on center-specific
texture while retaining lesion geometry in the phase spectrum.

Before launch V12 must pass one bounded worst-case test: RAW-derived fold-0
data, the complete encoder unfrozen, batch size 3, and at least two optimizer
steps so the second backward runs with full SGD momentum state resident. Both
the train and validation augmentation-worker counts are explicitly bounded at
4 and 2. V12 is rejected at scheduled whole-case checks if its apparent gain is
only pseudo Dice or if Dice improves by trading away lesion F1/count behavior.

The data disk has 5.8 GiB free at launch. Every 50-epoch checkpoint is retained
for model selection but, after its official evaluation succeeds, its optimizer
and scaler resume state are removed atomically. The rolling `checkpoint_latest`
keeps the complete state needed for recovery. This prevents a predictable late
disk-full failure without changing any learned weights or predictions.

## Launch check

The bounded test passed on fold 0 with all 1169/284 frozen train/validation
cases, batch shape `3x1x160x192x160`, and two complete optimizer steps in the
fully unfrozen stage. Losses were 0.4768 and 0.4797; the optimizer held momentum
for 256 parameter entries. Peak CUDA allocation/reservation was 18.94/19.75
GiB on the 23.52-GiB RTX 4090. cuDNN rejected several oversized benchmark
workspaces during its first convolution search but selected feasible kernels;
the second step completed and the process exited normally. No augmentation
workers or GPU processes remained afterward.

## Fold-0 epoch 50

The first scheduled whole-case evaluation completed for all 284 validation
cases and removed its 2.7-GiB prediction workspace successfully. V12 records
PR-AUC 0.5421, Dice 0.4631, AVD 6.913 ml, lesion-count error 3.528 and lesion
F1 0.2977. Against the exactly matched V9 epoch-50 checkpoint, the case means
change by +0.0002 PR-AUC, +0.0026 Dice, -0.6486 ml AVD improvement, -0.0528
count-error improvement and +0.0004 lesion F1 (positive improvement is better).
The pairwise official-style mean rank favors V9, 1.471 versus V12's 1.529.

The intended domain-generalization effect is not established yet: center-macro
Dice changes by -0.0025 and AVD worsens with a center-bootstrap 95% interval of
[-1.203, -0.178] ml. V12 gains Dice mainly in the middle-volume strata, not in
tiny or largest lesions; ATLAS3 Dice also regresses. Epoch 50 is nevertheless a
decoder-only comparison: the pretrained encoder was frozen for all preceding
updates, so the augmentation could not yet teach frequency-invariant encoder
features. The first complete-network epoch finishes normally in 161 seconds.
Continue the frozen experiment to epoch 100 for one decisive whole-network
comparison, and reject V12 there if its paired rank and center behavior do not
improve over V9.

## Fold-0 epoch 100

After 50 epochs of full-network warm-up, V12 records PR-AUC 0.5582, Dice
0.4701, AVD 7.063 ml, lesion-count error 2.859 and lesion F1 0.3213. Against
the exactly matched V9-e100 checkpoint, PR-AUC and Dice improve by 0.0319 and
0.0517. Their center-macro improvements are +0.0482 (95% bootstrap interval
[0.0221, 0.0782]) and +0.0369 ([-0.0014, 0.0798]), respectively. The paired
official-style mean rank now favors V12 1.488 to V9 1.512, with a 51.8% case
rank win rate. Gains appear across ATLAS3, Testing and Training sources and are
not confined to one center.

The trade-off is material: lesion F1 falls 0.0394 and count error worsens by
0.785, both with clearly negative center-level evidence; AVD worsens 0.848 ml.
V12 reduces positive empty predictions from 49 to 20, zero-Dice cases from 71
to 53 and zero-F1 cases from 103 to 74, so it detects more lesions but produces
too many or poorly matched components. Large-lesion Dice/F1 and volume error
are the clearest residual regression. Because the overall paired rank and
cross-source overlap/PR-AUC have now improved, continue the frozen polynomial
stage to epoch 150. Reject V12 there if component quality does not recover
without losing these gains.

## Fold-0 epoch 150

V12 records PR-AUC 0.5859, Dice 0.5007, AVD 5.991 ml, lesion-count error
2.380 and lesion F1 0.3759. All five metrics improve from epoch 100. The
paired official-style mean rank strongly favors epoch 150 over epoch 100,
1.372 to 1.628, with a 68.0% case-rank win rate; lesion F1 and count-error
improvements have 99% bootstrap support. Relative to the matched V9-e150
checkpoint, V12 also wins the paired rank 1.460 to 1.540 and improves Dice,
PR-AUC, lesion F1 and AVD, with only a negligible 0.025 increase in count
error. The earlier component fragmentation is recovering, so continue the
unchanged run to epoch 200.

## Fold-0 epoch 200

V12 records PR-AUC 0.6117, Dice 0.5167, AVD 6.401 ml, lesion-count error
2.254 and lesion F1 0.4166. Relative to epoch 150, the paired official-style
mean rank improves to 1.425 versus 1.575 with a 58.8% case-rank win rate.
Dice improves 0.0161 and lesion F1 0.0407 with bootstrap intervals entirely
above zero; PR-AUC, count error and missed-case counts also improve, while AVD
regresses 0.410 ml. Improvements span ATLAS3, Testing and Training sources and
are strongest for tiny/small and large lesions; acute overlap and large-lesion
volume calibration remain residual weaknesses.

At equal training progress V12 beats V9-e200 by paired rank, 1.479 to 1.521,
but it does not yet beat the mature V9-e300 checkpoint: V12 has slightly higher
PR-AUC but trails Dice, lesion F1, AVD and lesion-count error. Since V12 is
still improving credibly and its intended cross-source effect is not isolated
to one center, continue the frozen run to epoch 250 rather than comparing an
immature checkpoint with V9's mature selection.

## Final fold-0 decision

The unchanged run completed all 400 epochs and all scheduled whole-case
evaluations without an exception. Epochs 250/300/350/400 respectively reached
Dice 0.5262/0.5345/0.5281/0.5337. Epoch 300 has the strongest absolute PR-AUC,
Dice and lesion F1, whereas epoch 350 has the lowest AVD and epoch 400 provides
the most balanced case-level distribution.

In an official-like nine-way comparison of every model with both complete
fold-0 and sanity results (V1, V2-e200, V7-e400, V8-e250, V9-e300/e400 and
V12-e300/e350/e400), ranks are computed per case and metric and then averaged.
V12-e400 ranks first at 4.665, followed by V9-e300 at 4.735 and V12-e350 at
4.749. V12-e400 also wins direct paired rank comparisons against the other
late checkpoints. Retain V12-e400 as the primary fold-0 checkpoint, V12-e350
as the balanced backup and V9-e300 as the conservative architecture backup.

The two published sanity cases produced valid binary outputs at V12 epochs
300/350/400. Their nine-way ranks are 4.85/4.20/5.00, respectively; these two
training cases are retained only as a submission-pipeline check and do not
override the 284-case frozen-fold decision. The target Dice of 0.60 remains
unmet, so fold 1 stays locked and no ensemble or next version is launched.
