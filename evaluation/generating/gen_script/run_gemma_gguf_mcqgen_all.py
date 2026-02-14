#!/usr/bin/env python3
"""
run_gemma_gguf_mcqgen_all.py

Generic GGUF MCQ generator/evaluator (llama-cpp-python), designed for GitHub:
- No hard-coded personal paths.
- Models are provided via --models_file (JSON list).
- Each model can be resolved by:
    1) "gguf_path" (direct local file path), OR
    2) searching common Hugging Face cache roots, OR
    3) (optional) auto-downloading with huggingface_hub.snapshot_download.

Why you saw "No .gguf found ... in HF cache" even after "manual download":
- You likely downloaded into a different cache root (e.g., ~/.cache/huggingface) than the one the script searched,
  OR downloaded into a folder that is not the HF hub "models--.../snapshots/<sha>/" layout.

Install:
  pip install llama-cpp-python huggingface-hub

gemma_ggu_models.json. file:
[
  { "model_id": "HAJERI/gemma-3-270m-endocrinology-scenario-mcq-it-Q4_K_M-gguf", "contains": "Q4_K_M" },
  { "model_id": "HAJERI/gemma-3-270m-endocrinology-scenario-mcq-it-fp16-gguf",  "contains": "fp16"   },

  { "model_id": "HAJERI/gemma-3-1b-endocrinology-scenario-mcq-it-Q4_K_M-gguf",  "contains": "Q4_K_M" },
  { "model_id": "HAJERI/gemma-3-1b-endocrinology-scenario-mcq-it-fp16-gguf",   "contains": "fp16"   },

  { "model_id": "HAJERI/gemma-3-4b-endocrinology-scenario-mcq-it-Q4_K_M-gguf",  "contains": "Q4_K_M" },
  { "model_id": "HAJERI/gemma-3-4b-endocrinology-scenario-mcq-it-fp16-gguf",   "contains": "fp16"   },

  { "model_id": "HAJERI/gemma-3-12b-endocrinology-scenario-mcq-it-Q4_K_M-gguf", "contains": "Q4_K_M" },
  { "model_id": "HAJERI/gemma-3-12b-endocrinology-scenario-mcq-it-fp16-gguf",  "contains": "fp16"   },

  { "model_id": "HAJERI/gemma-3-27b-endocrinology-scenario-mcq-it-Q4_K_M-gguf", "contains": "Q4_K_M" },
  { "model_id": "HAJERI/gemma-3-27b-endocrinology-scenario-mcq-it-fp16-gguf",  "contains": "fp16"   }
]

Run:
  python run_gemma_gguf_mcqgen_all.py \
    --out_dir outputs/mcqgen_gguf \
    --models_file gemma_ggu_models.json \
    --resume --n_ctx 4096 --n_gpu_layers -1

Notes:
- If a model isn't found locally, the script can auto-download it (default on).
- If you want purely offline behavior, use --no_download (and optionally provide gguf_path).
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
from typing import Dict, Tuple, List, Optional, Iterable

from llama_cpp import Llama

try:
    # Used only if auto-download is enabled.
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None


# -----------------------------------------------------------------------------
# Default topics (10). You can replace/extend or add --topics_file (one per line).
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
# Prompt rules
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
    "Do NOT add extra headings, JSON, code blocks, or commentary.\n"
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

def build_prompt(topic: str) -> str:
    return (
        f"{SYSTEM_RULES}\n\n{FORMAT_EXAMPLE}\n\n"
        f"TOPIC: {topic}\n"
        "Now write ONE NEW MCQ for this topic.\n"
    )


# -----------------------------------------------------------------------------
# Tolerant MCQ parser
# -----------------------------------------------------------------------------
_RE_OPT      = re.compile(r"(?mi)^\s*([ABCD])\s*[\)\.\:]\s+(.*)\s*$")
_RE_ANS_LINE = re.compile(r"(?mi)^\s*Answer\s*[:\-]\s*([ABCD])\s*$")
_RE_ANS_ANY  = re.compile(r"(?mi)\bAnswer\s*[:\-]\s*([ABCD])\b")
_RE_EXP_LINE = re.compile(r"(?mi)^\s*Explanation\s*[:\-]\s*(.*)$")

def normalize_text(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n").strip()
    t = re.sub(r"(?mi)^\s*(system|user|assistant)\s*$", "", t).strip()
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
# Models file
# -----------------------------------------------------------------------------
@dataclass
class GGUFSpec:
    model_id: str
    gguf_path: Optional[str] = None   # direct file path (best if you manually downloaded somewhere)
    contains: Optional[str] = None    # substring filter (e.g., "Q4_K_M" or "fp16")
    select: str = "largest"           # "largest" | "smallest" | "first"


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

        specs.append(GGUFSpec(
            model_id=model_id,
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
# GGUF resolution helpers
# -----------------------------------------------------------------------------
def safe_filename(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")


def default_cache_roots() -> List[str]:
    """
    Return common HF cache roots (best-effort).
    We intentionally do not assume a single correct location.
    """
    roots: List[str] = []
    # If user exported these, they matter.
    for k in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME"):
        v = os.environ.get(k)
        if v:
            roots.append(os.path.expanduser(v))

    # Default HF home used by huggingface_hub if nothing is set:
    roots.append(os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))

    # Also consider XDG cache base if present
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        roots.append(os.path.join(os.path.expanduser(xdg), "huggingface"))

    # De-duplicate while preserving order
    out: List[str] = []
    seen = set()
    for r in roots:
        r = os.path.abspath(r)
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def repo_dirs_for_root(root: str, model_id: str) -> List[str]:
    """
    HF cache layout is usually:
      <root>/hub/models--ORG--REPO/...
    but sometimes user passes:
      <root>/hub  (already the hub directory)
    or (less commonly):
      <root>/models--ORG--REPO/...
    so we check all plausible variants.
    """
    org, name = model_id.split("/", 1)
    repo_leaf = f"models--{org}--{name}"

    candidates: List[str] = []

    base = os.path.abspath(root)
    if os.path.basename(base) == "hub":
        candidates.append(os.path.join(base, repo_leaf))
        # also consider parent as HF_HOME root
        candidates.append(os.path.join(os.path.dirname(base), "hub", repo_leaf))
        candidates.append(os.path.join(os.path.dirname(base), repo_leaf))
    else:
        candidates.append(os.path.join(base, "hub", repo_leaf))
        candidates.append(os.path.join(base, repo_leaf))

    # De-duplicate
    out: List[str] = []
    seen = set()
    for p in candidates:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def find_ggufs_under_repo_dir(repo_dir: str) -> List[str]:
    """
    Look under snapshots. We search shallow and deep.
    """
    if not os.path.isdir(repo_dir):
        return []

    hits = sorted(glob.glob(os.path.join(repo_dir, "snapshots", "*", "*.gguf")))
    if not hits:
        hits = sorted(glob.glob(os.path.join(repo_dir, "snapshots", "*", "**", "*.gguf"), recursive=True))
    return hits


def pick_from_hits(hits: List[str], contains: Optional[str], select: str) -> str:
    if not hits:
        raise FileNotFoundError("No .gguf candidates found")

    # Filter by substring if requested
    if contains:
        c = contains.lower()
        filt = [p for p in hits if c in os.path.basename(p).lower()]
        if filt:
            hits = filt

    if select == "first":
        return hits[0]
    if select == "smallest":
        return min(hits, key=lambda p: os.path.getsize(p))
    # default: largest
    return max(hits, key=lambda p: os.path.getsize(p))


def pick_gguf_from_directory(root_dir: str, contains: Optional[str], select: str) -> str:
    """
    snapshot_download returns a directory (often a snapshot folder).
    We recursively find *.gguf and pick one.
    """
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
    Returns (gguf_path, searched_locations_log)

    searched_locations_log is a list of repo dirs checked (useful for debugging).
    """
    searched: List[str] = []

    # 1) Direct gguf_path if provided (best for manual downloads)
    if spec.gguf_path:
        p = os.path.abspath(os.path.expanduser(spec.gguf_path))
        if os.path.isfile(p):
            return p, searched
        raise FileNotFoundError(f"gguf_path was provided but not found: {p}")

    # 2) If model_id itself is a local gguf file path
    maybe_file = os.path.abspath(os.path.expanduser(spec.model_id))
    if maybe_file.lower().endswith(".gguf") and os.path.isfile(maybe_file):
        return maybe_file, searched

    # 3) Search known cache roots (including user-provided --hf_home as *one* candidate)
    roots: List[str] = []
    if extra_roots:
        roots.extend([os.path.abspath(os.path.expanduser(r)) for r in extra_roots])
    roots.extend(default_cache_roots())

    # De-duplicate roots
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
            hits = find_ggufs_under_repo_dir(repo_dir)
            hits_all.extend(hits)

    # De-duplicate hit files
    hits_all = sorted(list(dict.fromkeys(hits_all)))

    if hits_all:
        return pick_from_hits(hits_all, contains=spec.contains, select=spec.select), searched

    # 4) Auto-download fallback (optional)
    if download_missing:
        if snapshot_download is None:
            raise RuntimeError(
                "Auto-download requested but huggingface_hub is not available. "
                "Install with: pip install huggingface-hub"
            )

        # cache_dir can be None -> use huggingface_hub defaults (often ~/.cache/huggingface/hub)
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
        "\nTip: provide gguf_path in models.json OR enable auto-download."
    )


# -----------------------------------------------------------------------------
# Generation (llama-cpp-python)
# -----------------------------------------------------------------------------
def generate_one(llm: Llama, topic: str, max_tokens: int, temperature: float) -> Tuple[str, float, int]:
    """
    Returns: (text, seconds, completion_tokens_if_available)
    """
    prompt = build_prompt(topic)
    t0 = time.time()
    out = llm.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1.0,
        stop=None,  # rely on tolerant parser; keep generic across different GGUF chat formats
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

    for spec in specs:
        out_csv = os.path.join(args.out_dir, f"{safe_filename(spec.model_id)}.csv")

        if (not args.overwrite) and args.resume and os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
            print(f"[SKIP] exists: {out_csv}")
            continue

        try:
            gguf_path, searched = resolve_gguf_path(
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
            raw = ""
            sec = 0.0
            ntoks = 0
            parsed_ok = False
            fields: Dict[str, str] = {}
            err = ""

            for attempt in range(args.retries + 1):
                if attempt == 0:
                    raw, sec, ntoks = generate_one(llm, topic, args.max_tokens, args.temperature)
                else:
                    retry_prompt = (
                        f"{SYSTEM_RULES}\n\n"
                        "Your previous output did not match the required format.\n"
                        "Rewrite it to EXACTLY match the format. Output ONLY the corrected MCQ.\n\n"
                        f"TOPIC: {topic}\n\n"
                        f"Previous output:\n{raw}\n"
                    )
                    t0 = time.time()
                    out = llm.create_completion(
                        prompt=retry_prompt,
                        max_tokens=args.max_tokens,
                        temperature=0.0,  # deterministic retry formatting
                        top_p=1.0,
                        stop=None,
                    )
                    t1 = time.time()
                    raw = (out["choices"][0].get("text") or "").strip()
                    sec = (t1 - t0)
                    ntoks = int(out.get("usage", {}).get("completion_tokens", 0) or 0)

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
