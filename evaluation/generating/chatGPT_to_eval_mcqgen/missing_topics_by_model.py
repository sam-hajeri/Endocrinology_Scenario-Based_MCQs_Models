#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path

MISSING_MODEL_ID = "(missing_model_id)"

def read_items(items_csv: Path):
    topics_all = set()
    topics_by_model = defaultdict(set)
    n_by_model = defaultdict(int)
    empty_model_rows = 0

    with items_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header row.")

        # Expect at least these columns (your merged output has them)
        # model_id, topic
        for i, row in enumerate(reader, start=2):
            mid = (row.get("model_id") or "").strip()
            topic = (row.get("topic") or "").strip()

            if not mid:
                empty_model_rows += 1
                mid = MISSING_MODEL_ID

            if topic:
                topics_all.add(topic)
                topics_by_model[mid].add(topic)

            n_by_model[mid] += 1

    topics_all = sorted(topics_all)
    return topics_all, topics_by_model, n_by_model, empty_model_rows


def write_wide(out_csv: Path, topics_all, topics_by_model, n_by_model, empty_model_rows):
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_id",
            "n_items",
            "topic_n",
            "missing_topic_n",
            "missing_topics",          # semicolon-separated
            "missing_model_id_rows"    # nonzero only for (missing_model_id)
        ])

        for mid in sorted(n_by_model.keys(), key=lambda x: (-n_by_model[x], x)):
            seen = topics_by_model.get(mid, set())
            missing = [t for t in topics_all if t not in seen]
            missing_topics = "; ".join(missing)
            miss_rows = empty_model_rows if mid == MISSING_MODEL_ID else 0

            writer.writerow([
                mid,
                n_by_model[mid],
                len(seen),
                len(missing),
                missing_topics,
                miss_rows
            ])


def write_long(out_long_csv: Path, topics_all, topics_by_model, n_by_model):
    out_long_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_long_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_id", "n_items", "missing_topic"])

        for mid in sorted(n_by_model.keys(), key=lambda x: (-n_by_model[x], x)):
            seen = topics_by_model.get(mid, set())
            missing = [t for t in topics_all if t not in seen]
            for t in missing:
                writer.writerow([mid, n_by_model[mid], t])


def main():
    ap = argparse.ArgumentParser(
        description="Report missing topics per model from an item-level CSV (no pandas)."
    )
    ap.add_argument("--items_csv", required=True, help="Path to item-level CSV (e.g., chatgpt_items_with_model.csv)")
    ap.add_argument("--out_csv", required=True, help="Output wide CSV path")
    ap.add_argument("--out_long_csv", default="", help="Optional output long CSV path (one row per missing topic)")

    args = ap.parse_args()
    items_csv = Path(args.items_csv)
    out_csv = Path(args.out_csv)
    out_long_csv = Path(args.out_long_csv) if args.out_long_csv else None

    if not items_csv.exists():
        raise FileNotFoundError(f"items_csv not found: {items_csv}")

    topics_all, topics_by_model, n_by_model, empty_model_rows = read_items(items_csv)

    # Write outputs
    write_wide(out_csv, topics_all, topics_by_model, n_by_model, empty_model_rows)
    if out_long_csv is not None:
        write_long(out_long_csv, topics_all, topics_by_model, n_by_model)

    print(f"Wrote wide CSV : {out_csv}")
    if out_long_csv is not None:
        print(f"Wrote long CSV : {out_long_csv}")
    print(f"Total unique topics = {len(topics_all)}")
    print(f"Rows with missing model_id = {empty_model_rows}")


if __name__ == "__main__":
    main()


# To run:

# cd /mnt/c/Users/saawq/Desktop/eval_mcqgen/sccritps

# python3 missing_topics_by_model.py \
#   --items_csv "/chatgpt_items_with_model.csv" \
#   --out_csv   "/missing_topics_by_model_wide.csv" \
#   --out_long_csv "/missing_topics_by_model_long.csv"
