import os

# -----------------------------------------------------------
# 0) Environment & stability settings
# -----------------------------------------------------------
# Disable compilers – unnecessary for merging and can cause extra overhead
os.environ["TORCH_COMPILE_DISABLE"]   = "1"
os.environ["TORCHDYNAMO_DISABLE"]     = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

# Make CUDA allocator more flexible for big models
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from unsloth import FastLanguageModel

# -----------------------------------------------------------
# 1) Paths and model names
# -----------------------------------------------------------
# Base model used for finetuning (for documentation / clarity)
BASE_MODEL_NAME = "unsloth/gemma-3-27b-pt"

# Your finetuned LoRA adapter directory (already on disk)
ADAPTER_DIR = "/gemma_3_27b_endo_finetuned" # make sure to have both lora adaptor with its tokenizer in the same dir.

# Where to save the fully merged fp16/bf16 model
MERGED_DIR  = "/gemma_3_27b_endo_finetuned_merged_fp16"

MAX_SEQ_LEN = 2048

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# -----------------------------------------------------------
# 2) Load the finetuned model (base + LoRA) via Unsloth
# -----------------------------------------------------------
# NOTE:
# - Unsloth will read the adapter config in ADAPTER_DIR.
# - That config points back to BASE_MODEL_NAME = "unsloth/gemma-3-27b-pt".
# - We load in 4-bit to avoid huge 27B 16-bit allocations on GPU.
print(f"Loading LoRA adapter from: {ADAPTER_DIR}")
print(f"Underlying base model (for reference): {BASE_MODEL_NAME}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = ADAPTER_DIR,   # directory with your finetuned LoRA
    max_seq_length = MAX_SEQ_LEN,
    dtype          = None,          # let Unsloth pick (bf16 on A100)
    load_in_4bit   = True,          # crucial for GPU memory
)

# Optional sanity check: print what Unsloth actually loaded
print("Model name_or_path:", getattr(model, "name_or_path", None))
print("Tokenizer name_or_path:", getattr(tokenizer, "name_or_path", None))

# -----------------------------------------------------------
# 3) Merge LoRA into base and save as a full 16-bit model
# -----------------------------------------------------------
print(f"Merging LoRA into base and saving 16-bit model to:\n  {MERGED_DIR}")

model.save_pretrained_merged(
    MERGED_DIR,
    tokenizer,
    save_method = "merged_16bit",  # outputs a standard fp16/bf16 HF model
)

print("✅ Merge complete – you now have a standalone Gemma-3-27B Endocrinology model.")
