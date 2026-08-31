# Local Rule 2 evaluation

The local comparison follows Rule 2: average each metric across all validation
cases, rank methods independently on the five metric means, then average the
five positions. Dice, lesion F1, and PR-AUC are ranked in descending order;
absolute volume and lesion-count differences are ranked in ascending order.
Metric ties use their average position, and a lower Mean Position is better.

This deliberately does not apply case-wise rank aggregation or a Dice-zero
worst-rank override. A legal empty prediction retains each metric's normally
defined value and remains in the all-case mean. Failed cases can be tracked as
a safety diagnostic but do not form a sixth ranking criterion.

The comparison pool, case-level predictions, and patient-derived summaries are
not distributed because they are derived from the licensed training release.
They can be regenerated locally with
`scripts/rank_isles26_model_pool_rule2.py` when the required per-case inputs are
available.
