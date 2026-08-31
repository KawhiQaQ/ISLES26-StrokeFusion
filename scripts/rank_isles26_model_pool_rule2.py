#!/usr/bin/env python3
"""Rank ISLES methods after aggregating each metric across cases (Rule 2).

Rule 2 first computes one case mean per metric and method. It then ranks the
methods independently on those five means and averages the five positions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_DIRECTIONS = {
    "dice": False,
    "abs_volume_diff_ml": True,
    "abs_lesion_count_diff": True,
    "lesion_f1": False,
    "pr_auc": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        nargs=2,
        metavar=("NAME", "PER_CASE_CSV"),
        required=True,
        help="Repeat once per method in the frozen comparison pool.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    names = [name for name, _ in args.model]
    if len(names) < 2:
        raise ValueError("at least two methods are required")
    if len(names) != len(set(names)):
        raise ValueError("method names must be unique")

    rows: list[dict[str, float | int | str]] = []
    expected_cases: set[str] | None = None
    expected_empty_gt: pd.Series | None = None
    for name, csv_path in args.model:
        frame = pd.read_csv(csv_path)
        required = {"case_id", "gt_volume_ml", *METRIC_DIRECTIONS}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
        if frame.case_id.duplicated().any():
            raise ValueError(f"{name} has duplicate case IDs")
        frame = frame.set_index("case_id").sort_index()
        cases = set(frame.index)
        if expected_cases is None:
            expected_cases = cases
        elif cases != expected_cases:
            raise ValueError(
                f"{name} case mismatch: missing={sorted(expected_cases - cases)[:5]}, "
                f"extra={sorted(cases - expected_cases)[:5]}"
            )

        empty_gt = frame["gt_volume_ml"].le(0)
        if expected_empty_gt is None:
            expected_empty_gt = empty_gt
        elif not empty_gt.equals(expected_empty_gt):
            raise ValueError(f"{name} disagrees on empty-ground-truth cases")

        row: dict[str, float | int | str] = {
            "model": name,
            "cases": len(frame),
            "failed_cases": int(frame["dice"].eq(0).sum()),
        }
        for metric in METRIC_DIRECTIONS:
            values = frame[metric].astype(float)
            if metric == "pr_auc":
                # PR-AUC is undefined for an empty reference mask in ISLES.
                values = values.mask(empty_gt)
            if np.isinf(values.to_numpy()).any() or values.dropna().empty:
                raise ValueError(f"invalid values for {name}/{metric}")
            if metric != "pr_auc" and values.isna().any():
                raise ValueError(f"missing values for {name}/{metric}")
            row[metric] = float(values.mean())
        rows.append(row)

    summary = pd.DataFrame(rows)
    rank_columns: list[str] = []
    for metric, lower_is_better in METRIC_DIRECTIONS.items():
        rank_column = f"rank_{metric}"
        summary[rank_column] = summary[metric].rank(
            method="average",
            ascending=lower_is_better,
        )
        rank_columns.append(rank_column)
    summary["mean_position"] = summary[rank_columns].mean(axis=1)
    summary = summary.sort_values(
        ["mean_position", "rank_dice", "model"],
        ignore_index=True,
    )
    summary.insert(0, "rule2_position", np.arange(1, len(summary) + 1))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
