
# prepare_strict_pass_items.py
# input: mcqgen_all.csv
# output: judge_items_strict_pass.jsonl & judge_key_strict_pass.csv


import re, json, hashlib, argparse
import pandas as pd

EXPECTED_TOPICS_DEFAULT = 10

REQ_COLS = [
    "model_id", "topic", "question",
    "option_A", "option_B", "option_C", "option_D",
    "answer", "explanation",
]

LEAK_PATTERNS = [
    r"\bOutput must\b", r"\bGenerate exactly\b", r"\bRules:\b", r"\bScenario:\b",
]

PLACEHOLDER_RE = re.compile(
    r"^\s*(?:\.{3,}|<\s*option\s*>|<\s*text\s*>|\[\s*text\s*\]|\[\s*option\s*\]|tbd|n/?a)\s*$",
    flags=re.IGNORECASE,
)

def _norm(x) -> str:
    if pd.isna(x): return ""
    s = str(x).strip()
    return "" if s.lower() in {"nan","none","null"} else s

def _opt_key(s: str) -> str:
    return re.sub(r"\s+", " ", _norm(s).lower()).strip()

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    for c in REQ_COLS:
        if c not in df.columns: df[c] = ""
        df[c] = df[c].apply(_norm)
    df["answer"] = df["answer"].str.upper()
    df["answer"] = df["answer"].str.extract(r"([A-D])", expand=False).fillna("")
    return df

def has_placeholder(r) -> bool:
    for f in [r["question"], r["option_A"], r["option_B"], r["option_C"], r["option_D"], r["explanation"]]:
        s = _norm(f)
        if s and PLACEHOLDER_RE.match(s):
            return True
    return False

def dup_options(r) -> bool:
    opts = [_opt_key(r["option_A"]), _opt_key(r["option_B"]), _opt_key(r["option_C"]), _opt_key(r["option_D"])]
    if any(o == "" for o in opts):  # empties handled by validity
        return False
    return len(set(opts)) < 4

def compute_flags(df: pd.DataFrame) -> pd.DataFrame:
    leak_re = re.compile("|".join(LEAK_PATTERNS), flags=re.IGNORECASE)
    df["missing_any"] = df.apply(lambda r: any(_norm(r[c]) == "" for c in REQ_COLS), axis=1)
    df["has_placeholder"] = df.apply(has_placeholder, axis=1)
    df["dup_options"] = df.apply(dup_options, axis=1)
    df["answer_ok"] = df["answer"].isin(list("ABCD"))
    df["prompt_leak"] = (
        df["question"].astype(str).str.contains(leak_re, na=False)
        | df["explanation"].astype(str).str.contains(leak_re, na=False)
    )
    df["valid"] = (~df["missing_any"]) & (~df["has_placeholder"])
    df["strict"] = df["valid"] & df["answer_ok"] & (~df["dup_options"]) & (~df["prompt_leak"])
    return df

def item_id(model_id: str, topic: str) -> str:
    h = hashlib.sha1(f"{model_id}||{topic}".encode("utf-8")).hexdigest()
    return h[:12]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_jsonl", default="judge_items_strict_pass.jsonl")
    ap.add_argument("--out_key_csv", default="judge_key_strict_pass.csv")
    args, _ = ap.parse_known_args()

    df = pd.read_csv(args.in_csv)
    df = normalize_df(df)

    # fail-fast uniqueness check
    dups = df.duplicated(subset=["model_id","topic"]).sum()
    if dups:
        raise ValueError(f"Found {dups} duplicate (model_id, topic) rows. Fix first.")

    df = compute_flags(df)
    strict_df = df[df["strict"]].copy()

    # Write judge items WITHOUT model_id (blind)
    items = []
    for _, r in strict_df.iterrows():
        iid = item_id(r["model_id"], r["topic"])
        items.append({
            "item_id": iid,
            "topic": r["topic"],
            "question": r["question"],
            "options": {"A": r["option_A"], "B": r["option_B"], "C": r["option_C"], "D": r["option_D"]},
            "keyed_answer": r["answer"],
            "explanation": r["explanation"],
        })

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Key to map back to models
    key = strict_df[["model_id","topic"]].copy()
    key["item_id"] = key.apply(lambda r: item_id(r["model_id"], r["topic"]), axis=1)
    key.to_csv(args.out_key_csv, index=False)

    print(f"STRICT-pass items: {len(items)}")
    print("Wrote:", args.out_jsonl)
    print("Wrote:", args.out_key_csv)

if __name__ == "__main__":
    main()

#---------------------------------------------
# To run:
#python prepare_strict_pass_items.py \
#  --in_csv /mcqgen_all.csv \
#  --out_jsonl /judge_items_strict_pass.jsonl \
#  --out_key_csv /judge_key_strict_pass.csv