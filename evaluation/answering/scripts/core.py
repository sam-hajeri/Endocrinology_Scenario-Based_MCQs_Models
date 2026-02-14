# core.py
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


# -----------------------------
# ModelSpec
# -----------------------------
@dataclass
class ModelSpec:
    repo_id: str
    kind: str  # "hf" or "gguf"
    family: Optional[str] = None  # "gemma" or "gptoss" (optional)
    gguf_filename: Optional[str] = None
    llama_bin: Optional[str] = None  # legacy; not used (llama-cpp-python is used)
    port: Optional[int] = None       # legacy; not used (llama-cpp-python is used)

    def safe_name(self) -> str:
        return self.repo_id.replace("/", "__")


# -----------------------------
# Repo list helpers
# -----------------------------
def read_repo_ids(models_list_path: str) -> List[str]:
    p = Path(models_list_path)
    if not p.exists():
        raise FileNotFoundError(f"models_list not found: {models_list_path}")
    repo_ids: List[str] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        repo_ids.append(line)
    return repo_ids


def _infer_kind(repo_id: str) -> str:
    return "gguf" if "gguf" in repo_id.lower() else "hf"


def _infer_family(repo_id: str) -> Optional[str]:
    r = repo_id.lower()
    if "gemma" in r:
        return "gemma"
    if "gpt-oss" in r or "gptoss" in r:
        return "gptoss"
    return None


def build_specs_from_list(
    repo_ids: Sequence[str],
    llama_bin: str,
    port_base: int,
) -> List[ModelSpec]:
    ids = list(repo_ids)

    specs: List[ModelSpec] = []
    gguf_port = int(port_base)

    for rid in ids:
        kind = _infer_kind(rid)
        fam = _infer_family(rid)
        if kind == "gguf":
            gguf_fn = os.environ.get("GGUF_FILENAME", None)
            specs.append(
                ModelSpec(
                    repo_id=rid,
                    kind="gguf",
                    family=fam,
                    gguf_filename=gguf_fn,
                    llama_bin=llama_bin,
                    port=gguf_port,
                )
            )
            gguf_port += 1
        else:
            specs.append(ModelSpec(repo_id=rid, kind="hf", family=fam))

    return specs


def filter_specs(
    specs: Sequence[ModelSpec],
    family: Optional[str] = None,
    kind: Optional[str] = None,
    allow_repo_prefixes: Optional[Sequence[str]] = None,
) -> List[ModelSpec]:
    out: List[ModelSpec] = []
    for s in specs:
        if family and (s.family != family):
            continue
        if kind and (s.kind != kind):
            continue
        if allow_repo_prefixes:
            ok = any(s.repo_id.startswith(pfx) for pfx in allow_repo_prefixes)
            if not ok:
                continue
        out.append(s)
    return out


# -----------------------------
# Dataset column normalization
# -----------------------------
def _first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"Missing columns. Tried: {list(candidates)}")


def normalize_harrison_df(df: pd.DataFrame) -> pd.DataFrame:
    qcol = _first_existing(df, ["question", "Question", "stem", "prompt"])
    acol = _first_existing(df, ["option_a", "OptionA", "option A", "A", "choice_a", "choiceA"])
    bcol = _first_existing(df, ["option_b", "OptionB", "option B", "B", "choice_b", "choiceB"])
    ccol = _first_existing(df, ["option_c", "OptionC", "option C", "C", "choice_c", "choiceC"])
    dcol = _first_existing(df, ["option_d", "OptionD", "option D", "D", "choice_d", "choiceD"])
    goldcol = _first_existing(df, ["answer", "Answer", "CorrectLetter", "correct", "label"])

    out = pd.DataFrame(
        {
            "question": df[qcol].astype(str),
            "A": df[acol].astype(str),
            "B": df[bcol].astype(str),
            "C": df[ccol].astype(str),
            "D": df[dcol].astype(str),
            "gold": df[goldcol].astype(str).str.strip(),
        }
    )
    out["gold"] = out["gold"].str.upper().str.extract(r"([ABCD])", expand=False).fillna(out["gold"])
    return out


def normalize_medmcqa_df(df: pd.DataFrame) -> pd.DataFrame:
    qcol = _first_existing(df, ["Question", "question", "stem", "prompt"])
    acol = _first_existing(df, ["OptionA", "option_a", "A", "choice_a", "choiceA"])
    bcol = _first_existing(df, ["OptionB", "option_b", "B", "choice_b", "choiceB"])
    ccol = _first_existing(df, ["OptionC", "option_c", "C", "choice_c", "choiceC"])
    dcol = _first_existing(df, ["OptionD", "option_d", "D", "choice_d", "choiceD"])
    goldcol = _first_existing(df, ["CorrectLetter", "answer", "Answer", "gold", "label"])

    out = pd.DataFrame(
        {
            "question": df[qcol].astype(str),
            "A": df[acol].astype(str),
            "B": df[bcol].astype(str),
            "C": df[ccol].astype(str),
            "D": df[dcol].astype(str),
            "gold": df[goldcol].astype(str).str.strip(),
        }
    )
    out["gold"] = out["gold"].str.upper().str.extract(r"([ABCD])", expand=False).fillna(out["gold"])
    return out


def normalize_df_by_benchmark(df: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    b = (benchmark or "").lower()
    if "harrison" in b:
        return normalize_harrison_df(df)
    if "medmcq" in b or "medmcqa" in b:
        return normalize_medmcqa_df(df)

    # heuristic fallback
    if "CorrectLetter" in df.columns and "OptionA" in df.columns:
        return normalize_medmcqa_df(df)
    if "option_a" in df.columns and "option_d" in df.columns:
        return normalize_harrison_df(df)

    raise ValueError(f"Unknown benchmark={benchmark!r}. Use 'harrison' or 'medmcqa'.")


# -----------------------------
# Prompting & extraction
# -----------------------------
_DEFAULT_SYSTEM_TEXT = (
    "You are a medical exam assistant. "
    "Answer multiple-choice questions by selecting the best option."
)

def build_mcq_prompt(
    question: str,
    A: str,
    B: str,
    C: str,
    D: str,
    system_text: Optional[str] = None,
) -> str:
    st = system_text or _DEFAULT_SYSTEM_TEXT
    return (
        f"{st}\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n"
        f"A. {A}\n"
        f"B. {B}\n"
        f"C. {C}\n"
        f"D. {D}\n\n"
        "Instruction: Respond with ONLY one letter: A, B, C, or D.\n"
        "Answer:"
    )


_RE_CHOICE = re.compile(
    r"(?:(?:final\s*)?answer\s*[:\-\s]*|correct\s*answer\s*[:\-\s]*|^\s*)([ABCD])\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)
_RE_ANY_LETTER = re.compile(r"\b([ABCD])\b", flags=re.IGNORECASE)


def extract_choice(text: str) -> Optional[str]:
    if text is None:
        return None
    t = str(text).strip()

    m = _RE_CHOICE.search(t)
    if m:
        return m.group(1).upper()

    # last line only
    lines = [x.strip() for x in t.splitlines() if x.strip()]
    if lines:
        tail = lines[-1].strip()
        if tail in ["A", "B", "C", "D"]:
            return tail

    # fallback: last standalone letter anywhere
    allm = list(_RE_ANY_LETTER.finditer(t))
    if allm:
        return allm[-1].group(1).upper()

    return None


# -----------------------------
# HF loading helpers (adapter-aware + auto fallback from 4bit)
# -----------------------------
def _is_adapter_only_repo(repo_id: str) -> bool:
    """Heuristic: LoRA adapter repos have adapter_config.json + adapter_model.safetensors."""
    try:
        from huggingface_hub import list_repo_files  # type: ignore
        files = set(list_repo_files(repo_id))
        return ("adapter_config.json" in files) and ("adapter_model.safetensors" in files or "adapter_model.bin" in files)
    except Exception:
        return False


def _get_base_model_from_adapter_config(repo_id: str) -> Optional[str]:
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
        p = hf_hub_download(repo_id, "adapter_config.json")
        cfg = json.loads(Path(p).read_text())
        return cfg.get("base_model_name_or_path") or cfg.get("base_model_name") or cfg.get("model_name_or_path")
    except Exception:
        return None


def _load_base_model_unsloth_or_tf(
    base_repo_id: str,
    load_in_4bit: bool,
    max_seq_length: int = 2048,
):
    """
    Load a base CausalLM model + tokenizer.
    Tries Unsloth first, then Transformers.
    """
    # Unsloth
    try:
        from unsloth import FastLanguageModel  # type: ignore

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_repo_id,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=bool(load_in_4bit),
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer, "unsloth"
    except Exception as e_unsloth:
        # Transformers
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

            tokenizer = AutoTokenizer.from_pretrained(base_repo_id, use_fast=True)
            kwargs: Dict[str, Any] = {}
            if load_in_4bit:
                kwargs["load_in_4bit"] = True

            model = AutoModelForCausalLM.from_pretrained(
                base_repo_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                **kwargs,
            )
            model.eval()
            return model, tokenizer, "transformers"
        except Exception as e_tf:
            raise RuntimeError(
                "Failed to load base HF model.\n"
                f"Unsloth error: {repr(e_unsloth)}\n"
                f"Transformers error: {repr(e_tf)}"
            )


def _attach_lora_adapter(model, adapter_repo_id: str):
    """Attach PEFT adapter to an already-loaded base model."""
    try:
        from peft import PeftModel  # type: ignore
        model = PeftModel.from_pretrained(model, adapter_repo_id, is_trainable=False)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to attach LoRA adapter {adapter_repo_id}: {e!r}")


def _load_hf_model_and_tokenizer_auto(
    repo_id: str,
    load_in_4bit_requested: bool,
    max_seq_length: int = 2048,
) -> Tuple[Any, Any, str]:
    """
    Automatic logic:
    - If repo is adapter-only: load base (from adapter_config) then attach adapter.
      (4bit requested is allowed)
    - Else (merged/full model): try load with requested 4bit; if it fails and 4bit was requested,
      retry once with load_in_4bit=False (BF16) automatically.
    """
    is_adapter = _is_adapter_only_repo(repo_id)
    if is_adapter:
        base_id = _get_base_model_from_adapter_config(repo_id)
        if not base_id:
            raise RuntimeError(f"Adapter repo detected but base_model_name_or_path not found in adapter_config.json: {repo_id}")
        model, tokenizer, backend = _load_base_model_unsloth_or_tf(
            base_id,
            load_in_4bit=bool(load_in_4bit_requested),
            max_seq_length=max_seq_length,
        )
        model = _attach_lora_adapter(model, repo_id)
        return model, tokenizer, backend + "+peft"

    # full/merged model
    try:
        model, tokenizer, backend = _load_base_model_unsloth_or_tf(
            repo_id,
            load_in_4bit=bool(load_in_4bit_requested),
            max_seq_length=max_seq_length,
        )
        return model, tokenizer, backend
    except Exception as e_first:
        if not load_in_4bit_requested:
            raise
        # retry BF16
        model, tokenizer, backend = _load_base_model_unsloth_or_tf(
            repo_id,
            load_in_4bit=False,
            max_seq_length=max_seq_length,
        )
        return model, tokenizer, backend + "+retry_bf16"


def _hf_generate_one(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    import torch  # type: ignore

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            pad_token_id=getattr(tokenizer, "eos_token_id", None),
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if text.startswith(prompt):
        return text[len(prompt):].strip()
    return text.strip()


# -----------------------------
# GGUF inference (llama-cpp-python)
# -----------------------------
def _resolve_gguf_file_from_hf_repo(repo_id: str) -> str:
    from huggingface_hub import snapshot_download  # type: ignore

    local_dir = snapshot_download(repo_id, local_files_only=False)
    p = Path(local_dir)

    ggufs = sorted([x for x in p.glob("*.gguf")])
    if not ggufs:
        ggufs = sorted([x for x in p.rglob("*.gguf")])
    if not ggufs:
        raise FileNotFoundError(f"No .gguf found in snapshot for {repo_id} at {local_dir}")

    ggufs = sorted(ggufs, key=lambda x: x.stat().st_size, reverse=True)
    return str(ggufs[0])


def _gguf_generate_one(llama, prompt: str, max_new_tokens: int) -> str:
    res = llama(
        prompt,
        max_tokens=int(max_new_tokens),
        temperature=0.0,
        top_p=1.0,
        stop=["\n\n", "\r\n\r\n"],
        echo=False,
    )
    try:
        return res["choices"][0]["text"].strip()
    except Exception:
        return str(res).strip()


# -----------------------------
# Evaluation runner
# -----------------------------
def _compute_accuracy_from_pred_file(pred_csv: Path) -> Optional[float]:
    try:
        df = pd.read_csv(pred_csv)
        if "correct" in df.columns:
            return float(df["correct"].mean())
        if "pred" in df.columns and "gold" in df.columns:
            gold = df["gold"].astype(str).str.strip().str.upper()
            pred = df["pred"].astype(str).str.strip().str.upper()
            return float((pred == gold).mean())
        return None
    except Exception:
        return None


def run_eval(
    df: pd.DataFrame,
    specs: Sequence[ModelSpec],
    out_dir: str,
    resume: bool = False,
    benchmark: str = "medmcqa",
    load_in_4bit: bool = False,
    max_new_tokens: Optional[int] = None,
    system_text: Optional[str] = None,   # <-- for compatibility with your hf scripts
    progress_every: int = 25,
    **_: Any,  # swallow any future/extra args so scripts don't crash
) -> None:
    out_dir_p = Path(out_dir)
    preds_dir = out_dir_p / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    norm = normalize_df_by_benchmark(df, benchmark=benchmark)
    n = len(norm)

    # We'll accumulate summary for models processed *this run*,
    # then at end we will ALSO scan preds_dir and include everything that exists.
    run_summaries: Dict[str, Dict[str, Any]] = {}

    for spec in specs:
        safe = spec.safe_name()
        out_csv = preds_dir / f"{safe}.csv"

        if resume and out_csv.exists():
            print(f"[SKIP] {safe} already done: {out_csv}")
            continue

        print(f"[START] model={spec.repo_id} kind={spec.kind} out={out_csv}")
        t0 = time.time()

        rows: List[Dict[str, Any]] = []
        correct = 0
        done = 0

        try:
            if spec.kind == "hf":
                # GPT-OSS sometimes prints reasoning; keep tokens enough to reach the final letter.
                mnt = int(max_new_tokens) if max_new_tokens is not None else 32

                model, tokenizer, backend = _load_hf_model_and_tokenizer_auto(
                    spec.repo_id,
                    load_in_4bit_requested=bool(load_in_4bit),
                    max_seq_length=2048,
                )

                for i in range(n):
                    q = norm.at[i, "question"]
                    A = norm.at[i, "A"]
                    B = norm.at[i, "B"]
                    C = norm.at[i, "C"]
                    D = norm.at[i, "D"]
                    gold = str(norm.at[i, "gold"]).strip().upper()

                    prompt = build_mcq_prompt(q, A, B, C, D, system_text=system_text)
                    completion = _hf_generate_one(model, tokenizer, prompt, max_new_tokens=mnt)

                    pred = extract_choice(completion) or "?"
                    is_ok = (pred == gold)

                    done += 1
                    if is_ok:
                        correct += 1

                    rows.append(
                        {
                            "model": safe,
                            "gold": gold,
                            "pred": pred,
                            "correct": int(is_ok),
                            "completion": completion,
                            "backend": backend,
                        }
                    )

                    if progress_every and done % progress_every == 0:
                        acc = correct / done if done else 0.0
                        print(f"[PROGRESS] model={safe} {done}/{n} acc={acc:.4f}")

                # cleanup
                try:
                    del model
                    del tokenizer
                except Exception:
                    pass
                gc.collect()
                try:
                    import torch  # type: ignore
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            elif spec.kind == "gguf":
                mnt = int(max_new_tokens) if max_new_tokens is not None else 24

                from llama_cpp import Llama  # type: ignore

                gguf_path = _resolve_gguf_file_from_hf_repo(spec.repo_id)
                llama = Llama(
                    model_path=gguf_path,
                    n_ctx=2048,
                    n_threads=max(1, os.cpu_count() or 8),
                    n_gpu_layers=-1,
                    verbose=False,
                )

                for i in range(n):
                    q = norm.at[i, "question"]
                    A = norm.at[i, "A"]
                    B = norm.at[i, "B"]
                    C = norm.at[i, "C"]
                    D = norm.at[i, "D"]
                    gold = str(norm.at[i, "gold"]).strip().upper()

                    prompt = build_mcq_prompt(q, A, B, C, D, system_text=system_text)
                    completion = _gguf_generate_one(llama, prompt, max_new_tokens=mnt)

                    pred = extract_choice(completion) or "?"
                    is_ok = (pred == gold)

                    done += 1
                    if is_ok:
                        correct += 1

                    rows.append(
                        {
                            "model": safe,
                            "gold": gold,
                            "pred": pred,
                            "correct": int(is_ok),
                            "completion": completion,
                        }
                    )

                    if progress_every and done % progress_every == 0:
                        acc = correct / done if done else 0.0
                        print(f"[PROGRESS] model={safe} {done}/{n} acc={acc:.4f}")

                try:
                    del llama
                except Exception:
                    pass
                gc.collect()

            else:
                raise ValueError(f"Unknown model kind: {spec.kind}")

            pd.DataFrame(rows).to_csv(out_csv, index=False)

            acc = correct / done if done else 0.0
            dt = time.time() - t0
            run_summaries[safe] = {"model": safe, "accuracy": acc, "eval_time": dt}
            print(f"[DONE] model={safe} acc={acc:.4f} time_sec={dt:.1f}")

        except Exception as e:
            print(f"[ERROR] model={safe} err={e!r}")
            if rows:
                pd.DataFrame(rows).to_csv(out_csv, index=False)

    # -----------------------------
    # Summary: include EVERYTHING in preds folder (not just this run)
    # -----------------------------
    summary_rows: List[Dict[str, Any]] = []
    pred_files = sorted(preds_dir.glob("*.csv"))

    for pf in pred_files:
        model_name = pf.stem
        acc = _compute_accuracy_from_pred_file(pf)
        if acc is None:
            continue
        if model_name in run_summaries:
            summary_rows.append(run_summaries[model_name])
        else:
            # older/previous runs: no runtime, keep blank
            summary_rows.append({"model": model_name, "accuracy": acc, "eval_time": ""})

    summary_csv = out_dir_p / "summary_overall.csv"
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.sort_values("accuracy", ascending=False)
        summary_df.to_csv(summary_csv, index=False)
        print(f"[SUMMARY] wrote {summary_csv}")
    else:
        pd.DataFrame(columns=["model", "accuracy", "eval_time"]).to_csv(summary_csv, index=False)
        print(f"[SUMMARY] wrote EMPTY summary (no valid preds) -> {summary_csv}")


# -----------------------------
# CLI helper (optional)
# -----------------------------
def _load_csv_any_encoding(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def main_cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to benchmark CSV")
    ap.add_argument("--models_list", required=True, help="Text file with repo_ids")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--benchmark", required=True, choices=["harrison", "medmcqa"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--system_text", type=str, default=None)
    args = ap.parse_args()

    df = _load_csv_any_encoding(args.csv)
    repo_ids = read_repo_ids(args.models_list)

    specs_all = build_specs_from_list(
        repo_ids,
        llama_bin="/dev/null",
        port_base=8080,
    )

    run_eval(
        df,
        specs_all,
        out_dir=args.out_dir,
        resume=args.resume,
        benchmark=args.benchmark,
        load_in_4bit=args.load_in_4bit,
        max_new_tokens=args.max_new_tokens,
        system_text=args.system_text,
    )


if __name__ == "__main__":
    main_cli()