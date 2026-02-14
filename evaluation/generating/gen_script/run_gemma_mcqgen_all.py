#!/usr/bin/env python3
"""
run_gemma_mcqgen_all.py

Generate scenario-based endocrinology MCQs per model (Hugging Face inference).
Outputs: one CSV per model containing the raw generation + parsed MCQ fields.

Key design goals (generic for GitHub):
1) No hard-coded personal paths.
2) LoRA inference loads the *same base model used during LoRA training*.
   - We auto-detect the base using PEFT metadata (PeftConfig.base_model_name_or_path).
   - If auto-detection fails/missing, we fall back to a user-provided base_model in models.json.
   - If both are missing, we fail loudly (no guessing).
3) Robust prompt rendering:
   - Uses apply_chat_template only if a chat_template exists.
   - Otherwise uses a manual Gemma-style prompt format.
4) Robust parsing:
   - Tolerant parser validates A–D, Answer, Explanation.

Usage example:
  python run_gemma_mcqgen_all.py \
    --out_dir outputs/mcqgen \
    --models_file gemma_models.json \
    --dtype bf16 --load_in_4bit --resume

gemma_models.json file:
[
  {
    "model_id": "HAJERI/gemma-3-270m-endocrinology-scenario-mcq-it-lora",
    "kind": "lora"
  },
  {
    "model_id": "HAJERI/gemma-3-270m-endocrinology-scenario-mcq-it-merged-fp16",
    "kind": "merged"
  },

  {
    "model_id": "HAJERI/gemma-3-1b-endocrinology-scenario-mcq-it-lora",
    "kind": "lora"
  },
  {
    "model_id": "HAJERI/gemma-3-1b-endocrinology-scenario-mcq-it-merged-fp16",
    "kind": "merged"
  },

  {
    "model_id": "HAJERI/gemma-3-4b-endocrinology-scenario-mcq-it-lora",
    "kind": "lora"
  },
  {
    "model_id": "HAJERI/gemma-3-4b-endocrinology-scenario-mcq-it-merged-fp16",
    "kind": "merged"
  },

  {
    "model_id": "HAJERI/gemma-3-12b-endocrinology-scenario-mcq-it-lora",
    "kind": "lora"
  },
  {
    "model_id": "HAJERI/gemma-3-12b-endocrinology-scenario-mcq-it-merged-fp16",
    "kind": "merged"
  },

  {
    "model_id": "HAJERI/gemma-3-27b-endocrinology-scenario-mcq-it-lora",
    "kind": "lora"
  },
  {
    "model_id": "HAJERI/gemma-3-27b-endocrinology-scenario-mcq-it-merged-fp16",
    "kind": "merged"
  }
]

"""

import os
import re
import gc
import csv
import json
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import torch
from unsloth import FastLanguageModel
from peft import PeftModel, PeftConfig


# -----------------------------------------------------------------------------
# Performance / stability flags (generic; no user-specific paths)
# -----------------------------------------------------------------------------
# Keep these defaults, but allow users to override by setting env vars beforehand.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# -----------------------------------------------------------------------------
# Topics (default 10). Can be overridden via --topics_file.
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
# Model specs
# -----------------------------------------------------------------------------
@dataclass
class ModelSpec:
    model_id: str
    kind: str                      # "lora" | "merged"
    base_model: Optional[str] = None  # optional fallback for LoRA only


def load_model_specs_from_json(path: str) -> List[ModelSpec]:
    """
    Read a JSON list of model specs to avoid hard-coding.
    This makes the repo reusable for any HF namespace.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("models_file must be a JSON list of objects")

    specs: List[ModelSpec] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"models_file item #{i} must be an object")
        model_id = item.get("model_id")
        kind = item.get("kind")
        base_model = item.get("base_model")
        if not model_id or not kind:
            raise ValueError(f"models_file item #{i} missing required keys: model_id, kind")
        if kind not in ("lora", "merged"):
            raise ValueError(f"models_file item #{i} kind must be 'lora' or 'merged'")
        specs.append(ModelSpec(model_id=model_id, kind=kind, base_model=base_model))
    return specs


def load_topics(topics_file: str) -> List[str]:
    """
    One topic per line text file.
    """
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
# Prompting
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
        f"{SYSTEM_RULES}\n{FORMAT_EXAMPLE}\n"
        f"TOPIC: {topic}\n"
        "Now write ONE NEW MCQ for this topic."
    )


# -----------------------------------------------------------------------------
# Tokenizer / processor handling
# -----------------------------------------------------------------------------
def split_tokenizer_and_template_obj(tok_or_proc):
    """
    Some repos return a processor-like object (has .tokenizer).
    - tokenizer: used for encode/decode/ids
    - template_obj: used for apply_chat_template if available
    """
    tokenizer = getattr(tok_or_proc, "tokenizer", tok_or_proc)
    template_obj = tok_or_proc if hasattr(tok_or_proc, "apply_chat_template") else tokenizer
    return tokenizer, template_obj


def render_prompt(tokenizer, template_obj, user_text: str) -> str:
    """
    Prefer chat templates when available; otherwise fall back to a manual prompt.
    This avoids breaking on models without chat_template configured.
    """
    chat_template = getattr(template_obj, "chat_template", None) or getattr(tokenizer, "chat_template", None)
    if chat_template:
        msgs = [{"role": "user", "content": user_text}]
        return template_obj.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # Manual Gemma-like prompt format
    bos = tokenizer.bos_token or ""
    return f"{bos}<start_of_turn>user\n{user_text}\n<end_of_turn>\n<start_of_turn>model\n"


# -----------------------------------------------------------------------------
# Parsing (tolerant)
# -----------------------------------------------------------------------------
_RE_OPT = re.compile(r"(?mi)^\s*([ABCD])\s*[\)\.\:]\s+(.*)\s*$")
_RE_ANS = re.compile(r"(?mi)^\s*Answer\s*[:\-]\s*([ABCD])\s*$")
_RE_EXP = re.compile(r"(?mi)^\s*Explanation\s*[:\-]\s*(.*)$")


def normalize_text(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n").strip()
    t = re.sub(r"(?mi)^\s*(system|user|assistant)\s*$", "", t).strip()
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def parse_mcq(raw: str) -> Tuple[bool, Dict[str, str], str]:
    """
    Returns: (ok, fields, error_message).
    Enforces existence of A–D options and an Answer line with A/B/C/D.
    """
    t = normalize_text(raw)
    lines = t.split("\n")

    opt_idx: Dict[str, int] = {}
    for i, line in enumerate(lines):
        m = _RE_OPT.match(line)
        if m:
            lab = m.group(1).upper()
            if lab not in opt_idx:
                opt_idx[lab] = i

    for lab in ["A", "B", "C", "D"]:
        if lab not in opt_idx:
            return False, {}, f"missing {lab})"

    a_i = opt_idx["A"]
    q_block = "\n".join(lines[:a_i]).strip()
    q_block = re.sub(r"(?mi)^\s*Question\s*:\s*", "", q_block).strip()
    if len(q_block) < 20:
        return False, {}, "question too short"

    def slice_between(start_i: int, end_i: int) -> str:
        block = "\n".join(lines[start_i:end_i]).strip()
        block = re.sub(r"(?mi)^\s*[ABCD]\s*[\)\.\:]\s*", "", block).strip()
        return re.sub(r"\s+", " ", block).strip()

    a = slice_between(opt_idx["A"], opt_idx["B"])
    b = slice_between(opt_idx["B"], opt_idx["C"])
    c = slice_between(opt_idx["C"], opt_idx["D"])

    # Find "Answer:" line at/after D option
    ans_line_i = None
    for i in range(opt_idx["D"], len(lines)):
        if _RE_ANS.match(lines[i].strip()):
            ans_line_i = i
            break

    # Fallback: regex search if Answer line not neatly formatted
    if ans_line_i is None:
        m = re.search(r"(?mi)\bAnswer\s*[:\-]\s*([ABCD])\b", t)
        if not m:
            return False, {}, "missing Answer"
        answer = m.group(1).upper()
        d = slice_between(opt_idx["D"], len(lines))
        exp = ""
        mexp = re.search(r"(?mi)\bExplanation\s*[:\-]\s*(.+)$", t, flags=re.S)
        if mexp:
            exp = normalize_text(mexp.group(1))
        return True, {
            "question": q_block,
            "option_A": a, "option_B": b, "option_C": c, "option_D": d,
            "answer": answer,
            "explanation": exp,
        }, ""

    d = slice_between(opt_idx["D"], ans_line_i)

    m_ans = _RE_ANS.match(lines[ans_line_i].strip())
    if not m_ans:
        return False, {}, "bad Answer line"
    answer = m_ans.group(1).upper()

    # Explanation may be on same line or subsequent lines
    exp = ""
    for j in range(ans_line_i + 1, len(lines)):
        mexp = _RE_EXP.match(lines[j])
        if mexp:
            exp = mexp.group(1).strip()
            exp_tail = "\n".join(lines[j + 1:]).strip()
            if exp_tail:
                exp = (exp + "\n" + exp_tail).strip()
            break
    exp = normalize_text(exp)

    return True, {
        "question": q_block,
        "option_A": a, "option_B": b, "option_C": c, "option_D": d,
        "answer": answer,
        "explanation": exp,
    }, ""


# -----------------------------------------------------------------------------
# LoRA base auto-detection + model loading
# -----------------------------------------------------------------------------
def autodetect_lora_base_model(adapter_id: str) -> Optional[str]:
    """
    Try to detect the *exact* base model used for LoRA training from adapter metadata.

    Why this matters:
    - A LoRA adapter is trained as deltas on a specific base checkpoint.
    - Loading it on a different base (e.g., IT instead of PT) often degrades quality.
    """
    try:
        cfg = PeftConfig.from_pretrained(adapter_id)
        base_id = getattr(cfg, "base_model_name_or_path", None)
        if isinstance(base_id, str) and base_id.strip():
            return base_id.strip()
    except Exception:
        # Any error here means we cannot rely on metadata (offline, missing config, etc.)
        return None
    return None


def load_hf_model(spec: ModelSpec, max_seq_len: int, load_in_4bit: bool, dtype_str: str):
    """
    Returns: (model, tokenizer, template_obj)

    For merged models:
      - Load directly from spec.model_id.

    For LoRA:
      - Prefer base model from PEFT config (auto-detect).
      - If missing/unavailable, fallback to spec.base_model (user-provided).
      - If both missing, raise error (do not guess).
    """
    dtype = torch.bfloat16 if dtype_str.lower() == "bf16" else torch.float16

    if spec.kind == "merged":
        model, tok_or_proc = FastLanguageModel.from_pretrained(
            model_name=spec.model_id,
            max_seq_length=max_seq_len,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
        tokenizer, template_obj = split_tokenizer_and_template_obj(tok_or_proc)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer, template_obj

    if spec.kind == "lora":
        # 1) auto-detect base from adapter metadata
        base_id = autodetect_lora_base_model(spec.model_id)

        # 2) fallback to user-provided base_model if auto-detect failed
        if not base_id:
            base_id = spec.base_model

        # 3) if still missing, fail (no guessing)
        if not base_id:
            raise ValueError(
                f"LoRA adapter '{spec.model_id}' has no detectable base_model_name_or_path and "
                f"spec.base_model was not provided. Add 'base_model' in models.json."
            )

        base, tok_or_proc = FastLanguageModel.from_pretrained(
            model_name=base_id,
            max_seq_length=max_seq_len,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )

        # Attach LoRA to that base
        model = PeftModel.from_pretrained(base, spec.model_id, is_trainable=False)
        model.eval()
        FastLanguageModel.for_inference(model)

        tokenizer, template_obj = split_tokenizer_and_template_obj(tok_or_proc)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer, template_obj

    raise ValueError(f"Unknown kind={spec.kind}. Expected 'lora' or 'merged'.")


def unload(*objs):
    """
    Best-effort cleanup between models to avoid VRAM accumulation.
    """
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------
def safe_filename(model_id: str) -> str:
    """
    Produce a filesystem-safe filename from a HF model ID.
    """
    return model_id.replace("/", "__").replace(":", "_")


def generate_one(model, tokenizer, template_obj, user_text: str, max_new_tokens: int, temperature: float) -> str:
    """
    Single generation call.
    """
    prompt = render_prompt(tokenizer, template_obj, user_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    # EOS handling:
    # - always include eos_token_id
    # - also include <end_of_turn> if it exists in vocab (some Gemma-style templates)
    eos_ids = [tokenizer.eos_token_id]
    try:
        eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if isinstance(eot, int) and eot >= 0 and eot != tokenizer.unk_token_id:
            eos_ids.append(eot)
    except Exception:
        pass

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=1.0,
            eos_token_id=eos_ids,                 # list is accepted by HF generate
            pad_token_id=tokenizer.pad_token_id,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    gen_ids = out[0][input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # If <end_of_turn> appears in decoded text, cut it off for clean parsing.
    if "<end_of_turn>" in text:
        text = text.split("<end_of_turn>")[0].strip()

    return text


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Directory to write one CSV per model.")
    ap.add_argument("--models_file", required=True, help="JSON list of model specs (see header docstring).")
    ap.add_argument("--topics_file", default=None, help="Optional: one topic per line (overrides defaults).")

    ap.add_argument("--hf_home", default=None,
                    help="Optional HF cache directory. If not set, uses existing environment/default.")

    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    args = ap.parse_args()

    # Optional: set HF cache directory in a portable way (no personal paths).
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    os.makedirs(args.out_dir, exist_ok=True)

    model_specs = load_model_specs_from_json(args.models_file)
    topics = load_topics(args.topics_file) if args.topics_file else DEFAULT_TOPICS

    for spec in model_specs:
        out_csv = os.path.join(args.out_dir, f"{safe_filename(spec.model_id)}.csv")

        # Resume mode: skip if already created
        if args.resume and os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
            print(f"[SKIP] exists: {out_csv}")
            continue

        print("\n" + "=" * 110)
        print(f"LOADING: {spec.model_id} | kind={spec.kind}")
        if spec.kind == "lora":
            # Show what base will be used (auto-detect first, fallback next)
            auto_base = autodetect_lora_base_model(spec.model_id)
            chosen = auto_base or spec.base_model or "<MISSING>"
            print(f"  LoRA base auto-detect: {auto_base}")
            print(f"  LoRA base chosen     : {chosen}")
        print("=" * 110)

        model, tokenizer, template_obj = load_hf_model(
            spec=spec,
            max_seq_len=args.max_seq_len,
            load_in_4bit=args.load_in_4bit,
            dtype_str=args.dtype,
        )

        rows: List[Dict[str, str]] = []
        ok_count = 0

        for topic in topics:
            user_text = build_prompt(topic)

            raw = ""
            parsed_ok = False
            fields: Dict[str, str] = {}
            err = ""

            # Try initial generation + N retries that explicitly ask to fix formatting.
            for attempt in range(args.retries + 1):
                if attempt == 0:
                    raw = generate_one(model, tokenizer, template_obj, user_text, args.max_new_tokens, args.temperature)
                else:
                    retry_text = (
                        f"{SYSTEM_RULES}\n"
                        "Your previous output did not match the required format.\n"
                        "Rewrite it to EXACTLY match the format.\n\n"
                        f"TOPIC: {topic}\n\n"
                        f"Previous output:\n{raw}\n\n"
                        "Now output ONLY the corrected MCQ."
                    )
                    raw = generate_one(model, tokenizer, template_obj, retry_text, args.max_new_tokens, args.temperature)

                parsed_ok, fields, err = parse_mcq(raw)
                if parsed_ok:
                    ok_count += 1
                    break

            status = "OK" if parsed_ok else f"FAIL({err})"
            print(f"  - {topic}: {status}")

            rows.append({
                "model_id": spec.model_id,
                "kind": spec.kind,
                # Store the fallback base_model field only (explicit); chosen base is printed in logs above.
                # If you want to persist chosen base, add a column and set it in load_hf_model.
                "base_model_fallback": spec.base_model or "",
                "topic": topic,
                "raw_output": raw,
                "parse_ok": str(int(parsed_ok)),
                "parse_error": err,
                "question": fields.get("question", ""),
                "option_A": fields.get("option_A", ""),
                "option_B": fields.get("option_B", ""),
                "option_C": fields.get("option_C", ""),
                "option_D": fields.get("option_D", ""),
                "answer": fields.get("answer", ""),
                "explanation": fields.get("explanation", ""),
            })

        # Write one CSV per model
        fieldnames = list(rows[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

        rate = (ok_count / max(1, len(topics))) * 100.0
        print(f"[SAVED] {out_csv} | parse_ok_rate={rate:.2f}%")

        unload(model, tokenizer, template_obj)

    print("\nDONE.")


if __name__ == "__main__":
    main()

# =============================================================================
# NOTES / TROUBLESHOOTING (Unsloth + Gemma-3)
# =============================================================================
# 1) Log line you may see:
#    "Unsloth: Gemma3 does not support SDPA - switching to fast eager."
#    Meaning: Unsloth cannot use PyTorch SDPA (scaled_dot_product_attention) for Gemma-3
#    in this stack, so it falls back to Unsloth's "fast eager" attention path.
#    This is typically informational (not an error). Performance may differ vs SDPA.
#
# 2) Known runtime crash seen in some environments:
#      NameError: name 'VARIANT_KWARG_KEYS' is not defined
#    The traceback often points to:
#      ./unsloth_compiled_cache/Linear_peft_forward.py
#    This is usually caused by a stale/broken Unsloth compiled-cache module
#    (not by your MCQ parsing / prompting logic).
#
#    Recommended fixes (try in this order):
#
#    A) Remove the compiled cache directory (fastest unblock):
#       rm -rf ./unsloth_compiled_cache
#
#    B) Disable Unsloth auto-compiler (most robust for evaluation/inference scripts):
#       export UNSLOTH_COMPILE_DISABLE=1
#       # optional: for maximum compatibility (may be slower)
#       export UNSLOTH_DISABLE_FAST_GENERATION=1
#
#       IMPORTANT: If you set these inside Python, they must be set BEFORE:
#         from unsloth import FastLanguageModel
#
#    C) Upgrade Unsloth + Unsloth Zoo, then delete the cache again:
#       pip install -U --no-cache-dir unsloth unsloth_zoo
#       rm -rf ./unsloth_compiled_cache
#
#    D) If it still persists, a workaround reported by some users is pinning PEFT:
#       pip install -U --no-cache-dir "peft==0.17.1"
#       rm -rf ./unsloth_compiled_cache
#
#    After any of the above, rerun your command, e.g.:
#      python run_gemma_hf_mcqgen_all.py --out_dir outputs/mcqgen --models_file models.json ...
# =============================================================================
