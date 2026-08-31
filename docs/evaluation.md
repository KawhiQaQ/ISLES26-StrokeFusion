# Fold0 Rule 2 ranking

`results/fold0_rule2.csv` ranks the frozen 34-method fold0 pool by
Rule 2: average each metric across all 284 cases, rank methods independently on
the five metric means (Dice/F1/PR-AUC descending; absolute volume and lesion
count differences ascending), then average the five positions. Metric ties use
their average position.

This deliberately does not apply the older case-wise rank aggregation or its
Dice-zero worst-rank override. `failed_cases` is retained as a safety diagnostic
but is not a sixth ranking criterion.

The pool is the prior fixed 33-method postprocessing comparison, with
`PP-baseline` relabelled as `E1+V15-SWA`, plus the frozen final conservative
`E1+V15-SWA+final-clean0002` candidate. Recompute it with
`scripts/rank_isles26_model_pool_rule2.py` when the same per-case CSV inputs
are available.
