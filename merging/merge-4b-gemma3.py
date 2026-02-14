

# ================================================================
# MERGE LoRA into Base Model
# ================================================================
import torch
from unsloth import FastLanguageModel
from peft import PeftModel

BASE_MODEL_NAME = "unsloth/gemma-3-4b-pt"
ADAPTER_DIR     = "/home/saawq/LLM/gemma_3_4b_endo_finetuned" # make sure to have both lora adaptor with its tokenizer in the same dir.
MERGED_DIR      = "/home/saawq/LLM/gemma_3_4b_endo_finetuned_merged_fp16"
MAX_SEQ_LEN     = 2048

# 1) Load base model in full precision (not 4-bit)
base_model, tok = FastLanguageModel.from_pretrained(
    model_name     = BASE_MODEL_NAME,
    dtype          = torch.bfloat16,   # or torch.float16 if no bf16 support
    max_seq_length = MAX_SEQ_LEN,
    load_in_4bit   = False,            # <-- must be False for merge
)

# 2) Attach LoRA adapter
lora_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

# 3) Merge LoRA weights into base
print("🔄 Merging LoRA into base model...")
merged_model = lora_model.merge_and_unload()

# 4) Save merged model
print(f"💾 Saving merged model to {MERGED_DIR}")
merged_model.save_pretrained(MERGED_DIR)
tok.save_pretrained(MERGED_DIR)

print("✅ Merge complete! You now have a standalone Gemma‑3‑4B Endo model.")
