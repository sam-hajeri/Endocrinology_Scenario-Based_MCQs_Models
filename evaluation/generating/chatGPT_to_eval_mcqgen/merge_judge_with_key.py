#!/usr/bin/env python3
"""
Merge a judge JSONL (one JSON object per line) with a key CSV that maps item_id -> model_id/topic.

INPUTS
- judge_jsonl: e.g., chatgpt_judge_results.jsonl
  Each line must contain at least:
    item_id, judge, scores{...}, flags{...}, rationale, weighted_total_0_100

- key_csv: e.g., judge_key_strict_pass.csv
  Must contain:
    item_id, model_id, topic
  (Extra columns are kept.)

OUTPUTS
- out_items_csv: item-level rows with model_id/topic merged + flattened judge fields
- out_models_csv: model-level summary stats aggregated from item-level rows
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


REQUIRED_KEY_COLS = {"item_id", "model_id", "topic"}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i} in {path}: {e}") from e
    return rows


def flatten_judge_row(o: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten nested `scores` and `flags` dicts into top-level columns:
      score_answer_correctness, flag_multiple_correct_answers, etc.
    """
    out: Dict[str, Any] = {}
    out["item_id"] = o.get("item_id")
    out["judge"] = o.get("judge")
    out["rationale"] = o.get("rationale")
    out["weighted_total_0_100"] = o.get("weighted_total_0_100")

    scores = o.get("scores") or {}
    flags = o.get("flags") or {}

    if not isinstance(scores, dict):
        scores = {}
    if not isinstance(flags, dict):
        flags = {}

    for k, v in scores.items():
        out[f"score_{k}"] = v
    for k, v in flags.items():
        out[f"flag_{k}"] = v

    return out


def load_judge_df(judge_jsonl: Path) -> pd.DataFrame:
    raw = read_jsonl(judge_jsonl)
    flat = [flatten_judge_row(o) for o in raw]
    df = pd.DataFrame(flat)

    if "item_id" not in df.columns:
        raise ValueError("judge_jsonl must contain item_id on every line.")

    # Basic sanity
    df["item_id"] = df["item_id"].astype(str)

    # Check duplicates
    dup = df["item_id"].duplicated().sum()
    if dup:
        raise ValueError(
            f"judge_jsonl has duplicate item_id rows: dup_item_ids={dup}. "
            f"Fix by deduplicating before merging."
        )

    return df


def load_key_df(key_csv: Path) -> pd.DataFrame:
    key = pd.read_csv(key_csv)
    missing = REQUIRED_KEY_COLS - set(key.columns)
    if missing:
        raise ValueError(
            f"key_csv is missing required columns: {sorted(missing)}. "
            f"Found columns: {list(key.columns)}"
        )

    key["item_id"] = key["item_id"].astype(str)

    # Check duplicates in key
    dup = key["item_id"].duplicated().sum()
    if dup:
        raise ValueError(
            f"key_csv has duplicate item_id rows: dup_item_ids={dup}. "
            f"Fix by deduplicating the key file (item_id must be unique)."
        )

    return key


def model_summary(items: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-model metrics from item-level merged table.
    """
    # Ensure numeric
    items["weighted_total_0_100"] = pd.to_numeric(items["weighted_total_0_100"], errors="coerce")

    # Pass defined by answer_correctness==5 if present; else NaN
    pass_col = None
    if "score_answer_correctness" in items.columns:
        pass_col = "score_answer_correctness"
        items[pass_col] = pd.to_numeric(items[pass_col], errors="coerce")

    agg = {
        "item_id": "count",
        "topic": pd.Series.nunique,
        "weighted_total_0_100": ["mean", "median", "std", "min", "max"],
    }

    if pass_col:
        # pass-rate = proportion with score_answer_correctness == 5
        items["_pass"] = (items[pass_col] == 5).astype(float)
        agg["_pass"] = "mean"

    g = items.groupby("model_id", dropna=False).agg(agg)

    # Flatten multiindex columns
    g.columns = [
        "_".join([c for c in col if c]).rstrip("_") if isinstance(col, tuple) else str(col)
        for col in g.columns
    ]
    g = g.reset_index()

    # Rename a few
    rename_map = {
        "item_id_count": "n_items",
        "topic_nunique": "topic_n",
        "weighted_total_0_100_mean": "mean_total",
        "weighted_total_0_100_median": "median_total",
        "weighted_total_0_100_std": "std_total",
        "weighted_total_0_100_min": "min_total",
        "weighted_total_0_100_max": "max_total",
        "_pass_mean": "pass_rate_answer_correctness_eq_5",
    }
    g = g.rename(columns=rename_map)

    # Optional: mean per score dimension if present
    score_cols = [c for c in items.columns if c.startswith("score_")]
    if score_cols:
        s = items.groupby("model_id", dropna=False)[score_cols].mean(numeric_only=True).reset_index()
        g = g.merge(s, on="model_id", how="left")

    # Sort best to worst
    g = g.sort_values(["mean_total", "n_items"], ascending=[False, False]).reset_index(drop=True)
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge_jsonl", required=True, help="Path to judge results JSONL (e.g., chatgpt_judge_results.jsonl)")
    ap.add_argument("--key_csv", required=True, help="Path to key mapping CSV (e.g., judge_key_strict_pass.csv)")
    ap.add_argument("--out_items_csv", required=True, help="Output CSV path for item-level merged results")
    ap.add_argument("--out_models_csv", required=True, help="Output CSV path for model-level summary")
    args = ap.parse_args()

    judge_jsonl = Path(args.judge_jsonl).expanduser().resolve()
    key_csv = Path(args.key_csv).expanduser().resolve()
    out_items_csv = Path(args.out_items_csv).expanduser().resolve()
    out_models_csv = Path(args.out_models_csv).expanduser().resolve()

    if not judge_jsonl.exists():
        raise FileNotFoundError(f"judge_jsonl not found: {judge_jsonl}")
    if not key_csv.exists():
        raise FileNotFoundError(f"key_csv not found: {key_csv}")

    judge_df = load_judge_df(judge_jsonl)
    key_df = load_key_df(key_csv)

    merged = key_df.merge(judge_df, on="item_id", how="left", validate="one_to_one")

    missing_judge = merged["weighted_total_0_100"].isna().sum()
    if missing_judge:
        # Not fatal, but you likely want to know.
        print(f"[warn] items missing judge rows after merge: {missing_judge}", file=sys.stderr)

    # Write item-level
    out_items_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_items_csv, index=False)

    # Model summary
    models = model_summary(merged)
    out_models_csv.parent.mkdir(parents=True, exist_ok=True)
    models.to_csv(out_models_csv, index=False)

    print(f"Wrote: {out_items_csv}")
    print(f"Wrote: {out_models_csv}")


if __name__ == "__main__":
    main()


# To run:
# python3 merge_judge_with_key.py \
#   --judge_jsonl "/chatgpt_judge_results.jsonl" \
#   --key_csv     "/judge_key_strict_pass.csv" \
#   --out_items_csv  "/chatgpt_items_with_model.csv" \
#   --out_models_csv "/chatgpt_models_summary.csv"