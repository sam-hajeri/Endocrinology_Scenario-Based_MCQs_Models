
"""
Merge a 4-bit QLoRA finetuned gpt-oss-20b Unsloth model into a 16-bit merged
checkpoint ready for later GGUF conversion.
"""

import os
import torch
from unsloth import FastLanguageModel

# -------------------------------------------------------------------
# 0) FIXED PATHS (edit here if you change locations)
# -------------------------------------------------------------------
FINETUNED_DIR = "/home/saawq/LLM/gpt_oss_20b_medical_finetuned"
MERGED_DIR    = FINETUNED_DIR + "_merged_fp16"   # /home/saawq/LLM/gpt_oss_20b_medical_finetuned_merged_fp16

MAX_SEQ_LEN   = 2048
MAX_MEM_FRAC  = 0.5   # fraction of GPU VRAM used during merge (0.5 = 50%)

# -------------------------------------------------------------------
# Environment tweaks (optional but recommended)
# -------------------------------------------------------------------

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def print_gpu_info():
    if not torch.cuda.is_available():
        print("[WARN] CUDA is not available. This script is meant for GPUs.")
        return

    device_id = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device_id)
    total = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)
    print(f"[INFO] Using GPU {device_id}: {name} ({total:.2f} GiB)")


def merge_lora_to_fp16():
    print_gpu_info()
    print(f"[INFO] finetuned_dir     = {FINETUNED_DIR}")
    print(f"[INFO] merged_output_dir = {MERGED_DIR}")
    print(f"[INFO] max_seq_len       = {MAX_SEQ_LEN}")
    print(f"[INFO] max_mem_frac      = {MAX_MEM_FRAC}")

    # 1) Load finetuned QLoRA model in 4-bit (to avoid OOM)
    print("[INFO] Loading finetuned QLoRA model in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = FINETUNED_DIR,
        max_seq_length = MAX_SEQ_LEN,
        load_in_4bit   = True,   # QLoRA weights in 4-bit
        dtype          = None,   # let Unsloth pick best dtype
    )

    if torch.cuda.is_available():
        model = model.cuda()

    # 2) Merge into 16-bit HF checkpoint
    print(f"[INFO] Saving merged 16-bit model to: {MERGED_DIR}")
    os.makedirs(MERGED_DIR, exist_ok=True)

    model.save_pretrained_merged(
        MERGED_DIR,
        tokenizer,
        save_method          = "merged_16bit",
        maximum_memory_usage = MAX_MEM_FRAC,  # fraction of GPU VRAM
    )

    print("[INFO] 16-bit merged model saved successfully.")
    print("[DONE] You can now convert this merged dir to GGUF with llama.cpp.")


if __name__ == "__main__":
    merge_lora_to_fp16()
