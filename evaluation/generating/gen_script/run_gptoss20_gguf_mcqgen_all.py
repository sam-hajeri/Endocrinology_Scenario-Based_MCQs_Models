#!/usr/bin/env python3
"""
run_gptoss20_gguf_mcqgen_all.py

Generic GGUF MCQGen runner for GPT-OSS-20B (llama-cpp-python), matching the SAME
structure/behavior as run_gemma_gguf_mcqgen_all.py:

- Reads GGUF model list from --models_file (JSON list), like the Gemma GGUF runner.
- Optionally reads topics from --topics_file (one topic per line).
- Resolves each GGUF either by:
    (A) direct gguf_path provided in models.json, OR
    (B) searching common Hugging Face cache roots (including optional --hf_home), OR
    (C) auto-downloading the repo's *.gguf files using huggingface_hub.snapshot_download (default ON).
- Generates 1 MCQ per topic (default 10 topics).
- Uses tolerant parser + retries.
- Writes 1 CSV per model.

Notes about "Harmony-style prompting":
- Some GPT-OSS checkpoints use special tokens like <|start|>, <|end|>.
- This script uses Harmony formatting by default, but still keeps parsing tolerant.
- You can disable Harmony and use plain prompt via --no_harmony if needed.

Install:
  pip install llama-cpp-python huggingface-hub

gptoss20_gguf_models.json file:
[
  { "model_id": "HAJERI/gpt-oss-20b-endocrinology-scenario-mcq-it-Q4_K_M-gguf", "kind": "gguf", "contains": "Q4_K_M" },
  { "model_id": "HAJERI/gpt-oss-20b-endocrinology-scenario-mcq-it-fp16-gguf",  "kind": "gguf", "contains": "fp16"   }
]

Run:
  python run_gptoss20_gguf_mcqgen_all.py \
    --out_dir outputs/mcqgen_gguf_gptoss20 \
    --models_file gptoss20_gguf_models.json \
    --resume --n_ctx 4096 --n_gpu_layers -1
"""

import os
import re
import gc
import csv
import json
import glob
import time
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

from llama_cpp import Llama

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None


# -----------------------------------------------------------------------------
# Default topics (10). Can be overridden with --topics_file.
# -----------------------------------------------------------------------------
DEFAULT_TOPICS = [
    "Diabetes mellitus & hypoglycemia",
    "Thyroid disorders",
    "Adrenal disorders",
    "Pituitary & hypothalamic disorders",
    "Calcium, parathyroid & metabolic bone disease",
    "Female reproductive endocrinology",
    "Male reproductive endocrinology",
    "Obesity, lipids & nutrition",
    "Endocrine neoplasia & hereditary syndromes",
    "Pediatric endocrinology & growth/puberty",
]


# -----------------------------------------------------------------------------
# Prompt rules (strict schema)
# -----------------------------------------------------------------------------
SYSTEM_RULES = (
    "You are an endocrinology educator.\n"
    "Create ONE high-quality clinical scenario–based MCQ about the given TOPIC.\n\n"
    "Hard requirements (MUST follow exactly):\n"
    "Question: <scenario stem + question line>\n"
    "A) <option>\n"
    "B) <option>\n"
    "C) <option>\n"
    "D) <option>\n"
    "Answer: <single letter A/B/C/D>\n"
    "Explanation: <short rationale>\n\n"
    "Do NOT add extra headings, JSON, code blocks, analysis, or commentary.\n"
    "Return ONLY the MCQ."
)

FORMAT_EXAMPLE = (
    "Example (format only):\n"
    "Question: A 55-year-old man presents with polyuria and polydipsia. What is the best next step?\n"
    "A) ...\n"
    "B) ...\n"
    "C) ...\n"
    "D) ...\n"
    "Answer: A\n"
    "Explanation: ...\n"
)


def build_user_text(topic: str) -> str:
    return (
        f"{SYSTEM_RULES}\n\n{FORMAT_EXAMPLE}\n\n"
        f"TOPIC: {topic}\n"
        "Now write ONE NEW MCQ for this topic."
    )


# -----------------------------------------------------------------------------
# Optional "Harmony" wrapper (commonly used with GPT-OSS tokenization)
# -----------------------------------------------------------------------------
def harmony_prompt(developer_text: str, user_text: str) -> str:
    """
    Harmony-like format. If your GGUF doesn't use these tokens, use --no_harmony.
    """
    return (
        "<|start|>developer<|message|>\n"
        f"{developer_text.strip()}\n"
        "<|end|>\n\n"
        "<|start|>user<|message|>\n"
        f"{user_text.strip()}\n"
        "<|end|>\n\n"
        "<|start|>assistant<|message|>\n"
    )


def extract_final_if_any(text: str) -> str:
    """
    Some Harmony-ish outputs may include channel wrappers. We strip them if present.
    """
    t = (text or "").strip()
    m = re.search(r"<\|channel\|>final<\|message\|>(.*)", t, flags=re.S)
    if m:
        t = m.group(1)
    for cut in ["<|end|>", "<|return|>", "<|start|>"]:
        if cut in t:
            t = t.split(cut)[0]
    return t.strip()


# -----------------------------------------------------------------------------
# Tolerant MCQ parser
# -----------------------------------------------------------------------------
_RE_OPT      = re.compile(r"(?mi)^\s*([ABCD])\s*[\)\.\:]\s+(.*)\s*$")
_RE_ANS_LINE = re.compile(r"(?mi)^\s*Answer\s*[:\-]\s*([ABCD])\s*$")
_RE_ANS_ANY  = re.compile(r"(?mi)\bAnswer\s*[:\-]\s*([ABCD])\b")
_RE_EXP_LINE = re.compile(r"(?mi)^\s*Explanation\s*[:\-]\s*(.*)$")


def normalize_text(t: str) -> str:
    t = (t or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    t = re.sub(r"(?mi)^\s*(system|developer|user|assistant)\s*$", "", t).strip()
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def parse_mcq(raw: str) -> Tuple[bool, Dict[str, str], str]:
    t = normalize_text(raw)
    lines = t.split("\n")

    opt_idx: Dict[str, int] = {}
    for i, line in enumerate(lines):
        m = _RE_OPT.match(line)
        if m:
            lab = m.group(1).upper()
            opt_idx.setdefault(lab, i)

    for lab in ["A", "B", "C", "D"]:
        if lab not in opt_idx:
            return False, {}, f"missing {lab})"

    q = "\n".join(lines[:opt_idx["A"]]).strip()
    q = re.sub(r"(?mi)^\s*Question\s*:\s*", "", q).strip()
    if len(q) < 20:
        return False, {}, "question too short"

    def slice_between(start_i: int, end_i: int) -> str:
        block = "\n".join(lines[start_i:end_i]).strip()
        block = re.sub(r"(?mi)^\s*[ABCD]\s*[\)\.\:]\s*", "", block).strip()
        block = re.sub(r"\s+", " ", block).strip()
        return block

    a = slice_between(opt_idx["A"], opt_idx["B"])
    b = slice_between(opt_idx["B"], opt_idx["C"])
    c = slice_between(opt_idx["C"], opt_idx["D"])

    ans_line_i: Optional[int] = None
    for i in range(opt_idx["D"], len(lines)):
        if _RE_ANS_LINE.match(lines[i].strip()):
            ans_line_i = i
            break

    if ans_line_i is None:
        m = _RE_ANS_ANY.search(t)
        if not m:
            return False, {}, "missing Answer"
        answer = m.group(1).upper()
        d = slice_between(opt_idx["D"], len(lines))
        exp = ""
        mexp = re.search(r"(?mi)\bExplanation\s*[:\-]\s*(.+)$", t, flags=re.S)
        if mexp:
            exp = normalize_text(mexp.group(1))
        return True, {
            "question": q,
            "option_A": a, "option_B": b, "option_C": c, "option_D": d,
            "answer": answer,
            "explanation": exp,
        }, ""

    d = slice_between(opt_idx["D"], ans_line_i)

    m_ans = _RE_ANS_LINE.match(lines[ans_line_i].strip())
    if not m_ans:
        return False, {}, "bad Answer line"
    answer = m_ans.group(1).upper()

    exp = ""
    for j in range(ans_line_i + 1, len(lines)):
        mexp = _RE_EXP_LINE.match(lines[j])
        if mexp:
            exp = mexp.group(1).strip()
            tail = "\n".join(lines[j + 1:]).strip()
            if tail:
                exp = (exp + "\n" + tail).strip()
            break
    exp = normalize_text(exp)

    return True, {
        "question": q,
        "option_A": a, "option_B": b, "option_C": c, "option_D": d,
        "answer": answer,
        "explanation": exp,
    }, ""


# -----------------------------------------------------------------------------
# Models file loading (same style as Gemma GGUF runner)
# -----------------------------------------------------------------------------
@dataclass
class GGUFSpec:
    model_id: str
    kind: str = "gguf"
    gguf_path: Optional[str] = None
    contains: Optional[str] = None   # prefer file basename containing substring (e.g., Q4_K_M / fp16)
    select: str = "largest"          # "largest" | "smallest" | "first"


def load_specs_from_json(path: str) -> List[GGUFSpec]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("models_file must be a JSON list of objects")

    specs: List[GGUFSpec] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"models_file item #{i} must be an object")

        model_id = item.get("model_id")
        if not model_id:
            raise ValueError(f"models_file item #{i} missing 'model_id'")

        kind = item.get("kind", "gguf")
        if kind != "gguf":
            raise ValueError(f"models_file item #{i} has kind={kind!r}; this runner expects kind='gguf'.")

        specs.append(GGUFSpec(
            model_id=model_id,
            kind=kind,
            gguf_path=item.get("gguf_path"),
            contains=item.get("contains"),
            select=item.get("select", "largest"),
        ))
    return specs


def load_topics(topics_file: str) -> List[str]:
    topics: List[str] = []
    with open(topics_file, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                topics.append(t)
    if not topics:
        raise ValueError("topics_file is empty")
    return topics


# -----------------------------------------------------------------------------
# GGUF discovery & selection (same style as Gemma GGUF runner)
# -----------------------------------------------------------------------------
def safe_filename(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")


def default_cache_roots() -> List[str]:
    """
    Search multiple possible HF cache roots because "manual downloads" often end up elsewhere.
    """
    roots: List[str] = []
    for k in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME"):
        v = os.environ.get(k)
        if v:
            roots.append(os.path.expanduser(v))
    roots.append(os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))

    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        roots.append(os.path.join(os.path.expanduser(xdg), "huggingface"))

    out: List[str] = []
    seen = set()
    for r in roots:
        r = os.path.abspath(r)
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def repo_dirs_for_root(root: str, model_id: str) -> List[str]:
    org, name = model_id.split("/", 1)
    repo_leaf = f"models--{org}--{name}"

    candidates: List[str] = []
    base = os.path.abspath(root)

    if os.path.basename(base) == "hub":
        candidates.append(os.path.join(base, repo_leaf))
        candidates.append(os.path.join(os.path.dirname(base), "hub", repo_leaf))
        candidates.append(os.path.join(os.path.dirname(base), repo_leaf))
    else:
        candidates.append(os.path.join(base, "hub", repo_leaf))
        candidates.append(os.path.join(base, repo_leaf))

    out: List[str] = []
    seen = set()
    for p in candidates:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def find_ggufs_under_repo_dir(repo_dir: str) -> List[str]:
    if not os.path.isdir(repo_dir):
        return []
    hits = sorted(glob.glob(os.path.join(repo_dir, "snapshots", "*", "*.gguf")))
    if not hits:
        hits = sorted(glob.glob(os.path.join(repo_dir, "snapshots", "*", "**", "*.gguf"), recursive=True))
    return hits


def pick_from_hits(hits: List[str], contains: Optional[str], select: str) -> str:
    if not hits:
        raise FileNotFoundError("No .gguf candidates found")

    if contains:
        c = contains.lower()
        filt = [p for p in hits if c in os.path.basename(p).lower()]
        if filt:
            hits = filt

    if select == "first":
        return hits[0]
    if select == "smallest":
        return min(hits, key=lambda p: os.path.getsize(p))
    return max(hits, key=lambda p: os.path.getsize(p))


def pick_gguf_from_directory(root_dir: str, contains: Optional[str], select: str) -> str:
    ggufs: List[str] = []
    for r, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(".gguf"):
                ggufs.append(os.path.join(r, fn))
    if not ggufs:
        raise FileNotFoundError(f"No .gguf found under: {root_dir}")
    return pick_from_hits(sorted(ggufs), contains=contains, select=select)


def resolve_gguf_path(spec: GGUFSpec,
                      extra_roots: Optional[List[str]],
                      download_missing: bool,
                      download_cache_dir: Optional[str],
                      local_files_only: bool) -> Tuple[str, List[str]]:
    """
    Returns (gguf_path, searched_repo_dirs_log)
    """
    searched: List[str] = []

    # 1) Direct path in models.json
    if spec.gguf_path:
        p = os.path.abspath(os.path.expanduser(spec.gguf_path))
        if os.path.isfile(p):
            return p, searched
        raise FileNotFoundError(f"gguf_path was provided but not found: {p}")

    # 2) model_id can be a direct .gguf path
    maybe_file = os.path.abspath(os.path.expanduser(spec.model_id))
    if maybe_file.lower().endswith(".gguf") and os.path.isfile(maybe_file):
        return maybe_file, searched

    # 3) search cache roots
    roots: List[str] = []
    if extra_roots:
        roots.extend([os.path.abspath(os.path.expanduser(r)) for r in extra_roots])
    roots.extend(default_cache_roots())

    uniq_roots: List[str] = []
    seen = set()
    for r in roots:
        if r not in seen:
            seen.add(r)
            uniq_roots.append(r)

    hits_all: List[str] = []
    for root in uniq_roots:
        for repo_dir in repo_dirs_for_root(root, spec.model_id):
            searched.append(repo_dir)
            hits_all.extend(find_ggufs_under_repo_dir(repo_dir))

    hits_all = sorted(list(dict.fromkeys(hits_all)))
    if hits_all:
        return pick_from_hits(hits_all, contains=spec.contains, select=spec.select), searched

    # 4) auto-download fallback
    if download_missing:
        if snapshot_download is None:
            raise RuntimeError(
                "Auto-download requested but huggingface_hub is not available. "
                "Install with: pip install huggingface-hub"
            )
        snap_dir = snapshot_download(
            repo_id=spec.model_id,
            allow_patterns=["*.gguf", "*.GGUF"],
            cache_dir=download_cache_dir,
            local_files_only=local_files_only,
            resume_download=True,
        )
        gguf = pick_gguf_from_directory(snap_dir, contains=spec.contains, select=spec.select)
        return gguf, searched

    raise FileNotFoundError(
        f"No .gguf found for {spec.model_id}.\n"
        f"Searched repo dirs (first 8 shown):\n  - " + "\n  - ".join(searched[:8]) +
        ("\n  ... (more)" if len(searched) > 8 else "") +
        "\nTip: provide gguf_path in models.json OR run without --no_download."
    )


# -----------------------------------------------------------------------------
# Generation (llama-cpp-python)
# -----------------------------------------------------------------------------
def generate_one(llm: Llama,
                 prompt: str,
                 max_tokens: int,
                 temperature: float,
                 stop: Optional[List[str]]) -> Tuple[str, float, int]:
    """
    Returns: (text, seconds, completion_tokens_if_available)
    """
    t0 = time.time()
    out = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1.0,
        stop=stop,
    )
    t1 = time.time()
    text = (out["choices"][0].get("text") or "").strip()
    ntoks = int(out.get("usage", {}).get("completion_tokens", 0) or 0)
    return text, (t1 - t0), ntoks


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--models_file", required=True)
    ap.add_argument("--topics_file", default=None)

    # Cache behavior:
    ap.add_argument("--hf_home", default=None,
                    help="Optional cache root to search FIRST (e.g., /path/to/hf_cache or ~/.cache/huggingface).")
    ap.add_argument("--no_download", action="store_true",
                    help="Disable auto-download fallback if GGUF isn't found locally.")
    ap.add_argument("--download_cache_dir", default=None,
                    help="Optional cache_dir for snapshot_download. If omitted, huggingface_hub uses its default.")
    ap.add_argument("--local_files_only", action="store_true",
                    help="If set, snapshot_download will not hit the network (offline mode).")

    # Run control:
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip_missing", action="store_true",
                    help="If a model GGUF can't be resolved, skip it instead of stopping the whole run.")

    # Generation:
    ap.add_argument("--max_tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--retries", type=int, default=2)

    # Prompt format:
    ap.add_argument("--no_harmony", action="store_true",
                    help="Disable Harmony wrapper and use plain prompt text.")
    ap.add_argument("--stop_tokens", default="<|end|>,<|return|>",
                    help="Comma-separated stop tokens used for Harmony (default: '<|end|>,<|return|>'). "
                         "Ignored if --no_harmony is set.")

    # llama.cpp runtime knobs:
    ap.add_argument("--n_ctx", type=int, default=4096)
    ap.add_argument("--n_gpu_layers", type=int, default=-1)
    ap.add_argument("--n_threads", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    specs = load_specs_from_json(args.models_file)
    topics = load_topics(args.topics_file) if args.topics_file else DEFAULT_TOPICS

    extra_roots = [args.hf_home] if args.hf_home else []
    download_missing = (not args.no_download)

    # Stop tokens are only used in Harmony mode
    stop_tokens = None
    if not args.no_harmony:
        stop_tokens = [s.strip() for s in args.stop_tokens.split(",") if s.strip()]

    for spec in specs:
        out_csv = os.path.join(args.out_dir, f"{safe_filename(spec.model_id)}.csv")

        if (not args.overwrite) and args.resume and os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
            print(f"[SKIP] exists: {out_csv}")
            continue

        try:
            gguf_path, _searched = resolve_gguf_path(
                spec=spec,
                extra_roots=extra_roots,
                download_missing=download_missing,
                download_cache_dir=args.download_cache_dir,
                local_files_only=args.local_files_only,
            )
        except Exception as e:
            print(f"[MISSING] {spec.model_id}: {e}")
            if args.skip_missing:
                continue
            raise

        print("\n" + "=" * 110)
        print(f"LOADING GGUF: {spec.model_id}")
        print(f"GGUF PATH : {gguf_path}")
        print("=" * 110)

        llm = Llama(
            model_path=gguf_path,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            n_threads=args.n_threads,
            seed=args.seed,
            verbose=False,
        )

        rows: List[Dict[str, str]] = []
        ok_count = 0

        for topic in topics:
            user_text = build_user_text(topic)

            # Build the actual prompt string
            if args.no_harmony:
                prompt = user_text
            else:
                # Use SYSTEM_RULES as "developer" content, and full user_text as user content.
                prompt = harmony_prompt(SYSTEM_RULES, user_text)

            raw = ""
            parsed_ok = False
            fields: Dict[str, str] = {}
            err = ""
            sec = 0.0
            ntoks = 0

            for attempt in range(args.retries + 1):
                if attempt > 0:
                    retry_user = (
                        "Your previous output did not match the required format.\n"
                        "Rewrite it to EXACTLY match the format.\n\n"
                        f"TOPIC: {topic}\n\n"
                        f"Previous output:\n{raw}\n\n"
                        "Now output ONLY the corrected MCQ."
                    )
                    if args.no_harmony:
                        prompt = f"{SYSTEM_RULES}\n\n{retry_user}"
                    else:
                        prompt = harmony_prompt(SYSTEM_RULES, retry_user)

                text, sec, ntoks = generate_one(
                    llm=llm,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature if attempt == 0 else 0.0,
                    stop=stop_tokens,
                )

                # If Harmony tokens leak, strip them
                raw = extract_final_if_any(text) if not args.no_harmony else text

                parsed_ok, fields, err = parse_mcq(raw)
                if parsed_ok:
                    ok_count += 1
                    break

            status = "OK" if parsed_ok else f"FAIL({err})"
            print(f"  - {topic}: {status}")

            rows.append({
                "model_id": spec.model_id,
                "gguf_path": gguf_path,
                "topic": topic,

                "parse_ok": str(int(parsed_ok)),
                "parse_error": err,

                "seconds": f"{sec:.3f}",
                "new_tokens": str(ntoks),

                "question": fields.get("question", ""),
                "option_A": fields.get("option_A", ""),
                "option_B": fields.get("option_B", ""),
                "option_C": fields.get("option_C", ""),
                "option_D": fields.get("option_D", ""),
                "answer": fields.get("answer", ""),
                "explanation": fields.get("explanation", ""),

                "raw_output": raw,
            })

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        rate = (ok_count / max(1, len(topics))) * 100.0
        print(f"[SAVED] {out_csv} | parse_ok_rate={rate:.2f}%")

        del llm
        gc.collect()

    print("\nDONE.")


if __name__ == "__main__":
    main()
