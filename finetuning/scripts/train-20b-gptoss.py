import os
import time
import math

# -------------------------------------------------------------------
# 0) ENV + IMPORTS
# -------------------------------------------------------------------
os.environ["TORCH_COMPILE_DISABLE"]   = "1"
os.environ["TORCHDYNAMO_DISABLE"]     = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastLanguageModel
import torch
import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import TrainingArguments
from trl import SFTTrainer

# Small perf boost (DO NOT disable gradients for training)
# torch.set_grad_enabled(False)  # <-- remove / comment this
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("=" * 60)
print("Starting Training (gpt-oss-20b, LoRA r=32)")
print("=" * 60)

# -------------------------------------------------------------------
# 1) CONFIG
# -------------------------------------------------------------------
MODEL_NAME    = "unsloth/gpt-oss-20b"
MAX_SEQ_LEN   = 2048
OUTPUT_DIR    = "./gpt_oss_20b_endo_finetuned"
TOKENIZER_DIR = "./gpt_oss_20b_endo_finetuned_tokenizer"
TEST_SIZE     = 0.05
SEED          = 42

# -------------------------------------------------------------------
# 2) LOAD BASE MODEL (20B, 4-bit) + LORA r=32
# -------------------------------------------------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name       = MODEL_NAME,
    dtype            = None,          # let Unsloth pick (bf16/fp16)
    max_seq_length   = MAX_SEQ_LEN,
    load_in_4bit     = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32,                # higher-rank LoRA for more capacity
    lora_alpha = 64,       # ~2x r is a good rule of thumb
    lora_dropout = 0,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing = "unsloth",
    random_state = SEED,
)

print("Model with LoRA loaded on:", next(model.parameters()).device)

# -------------------------------------------------------------------
# 3) LOAD DATA (three Excel files)
# -------------------------------------------------------------------
df_mcq         = pd.read_excel("./datasets/Generate_Scenario_Endo.xlsx")
df_qa_scenario = pd.read_excel("./datasets/QA_Scenario_Endo.xlsx")
df_qa_general  = pd.read_excel("./datasets/QA_General_Endo.xlsx")

# -------------------------------------------------------------------
# 4) FORMAT FUNCTIONS (same pattern as before)
# -------------------------------------------------------------------
def format_mcq(example):
    return {
        "text": f"""<|start|>developer<|message|>
You are an expert endocrinologist creating clinical MCQs for medical education.
<|end|>

<|start|>user<|message|>
{example['instruction']}
<|end|>

<|start|>assistant<|return|><|message|>
{example['output']}
<|end|><|return|>
"""
    }

def format_qa_scenario(example):
    return {
        "text": f"""<|start|>developer<|message|>
You are a clinical endocrinology expert answering scenario-based medical questions.
<|end|>

<|start|>user<|message|>
{example['Question']}
<|end|>

<|start|>assistant<|return|><|message|>
{example['Answer']}
<|end|><|return|>
"""
    }

def format_qa_general(example):
    return {
        "text": f"""<|start|>developer<|message|>
You are a medical expert answering general endocrinology and clinical questions.
<|end|>

<|start|>user<|message|>
{example['Question']}
<|end|>

<|start|>assistant<|return|><|message|>
{example['Answer']}
<|end|><|return|>
"""
    }

# -------------------------------------------------------------------
# 5) HF DATASETS + TRAIN/VAL SPLIT
# -------------------------------------------------------------------
ds_mcq = Dataset.from_pandas(df_mcq).map(
    format_mcq,
    remove_columns = df_mcq.columns.tolist(),
)

ds_qa_scenario = Dataset.from_pandas(df_qa_scenario).map(
    format_qa_scenario,
    remove_columns = df_qa_scenario.columns.tolist(),
)

ds_qa_general = Dataset.from_pandas(df_qa_general).map(
    format_qa_general,
    remove_columns = df_qa_general.columns.tolist(),
)

dataset = concatenate_datasets([ds_mcq, ds_qa_scenario, ds_qa_general])
split_dataset = dataset.train_test_split(
    test_size = TEST_SIZE,
    shuffle   = True,
    seed      = SEED,
)

train_ds = split_dataset["train"]
val_ds   = split_dataset["test"]

print(f"Total: {len(dataset)}, Train: {len(train_ds)}, Val: {len(val_ds)}")

# -------------------------------------------------------------------
# 6) TRAINING ARGUMENTS
#     (no evaluation_strategy here to avoid your previous error)
# -------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir                  = OUTPUT_DIR,
    per_device_train_batch_size = 1,
    per_device_eval_batch_size  = 4,   # eval/test batch size (can tune)
    gradient_accumulation_steps = 16,
    warmup_steps                = 50,
    num_train_epochs            = 1,
    learning_rate               = 5e-4,
    lr_scheduler_type           = "cosine",
    logging_steps               = 10,   # training loss every 10 steps
    save_steps                  = 200,
    save_total_limit            = 2,
    optim                       = "adamw_8bit",
    weight_decay                = 0.01,
    seed                        = SEED,
    report_to                   = "none",
    bf16                        = True,
    gradient_checkpointing      = True,
    max_grad_norm               = 0.3,
    dataloader_num_workers      = 4,
)

# -------------------------------------------------------------------
# 7) TRAINER (SFTTrainer) – WITH eval_dataset
# -------------------------------------------------------------------
trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    args               = training_args,
    train_dataset      = train_ds,
    eval_dataset       = val_ds,       # used when we call trainer.evaluate()
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LEN,
)

# -------------------------------------------------------------------
# 8) TRAIN
# -------------------------------------------------------------------
print("Starting training...")
t0 = time.time()
trainer.train()
t1 = time.time()
print(f"Training finished in {(t1 - t0)/60:.2f} minutes.")

# -------------------------------------------------------------------
# 9) EVAL ON TEST/VAL SPLIT (loss + perplexity)
# -------------------------------------------------------------------
print("\nRunning evaluation on validation/test split...")
eval_metrics = trainer.evaluate(eval_dataset=val_ds)
print("Eval metrics:", eval_metrics)

eval_loss = eval_metrics.get("eval_loss", None)
if eval_loss is not None:
    try:
        ppl = math.exp(eval_loss)
        print(f"Validation loss: {eval_loss:.4f}")
        print(f"Validation perplexity: {ppl:.4f}")
    except OverflowError:
        print("Perplexity overflow; eval_loss too large or invalid.")

# -------------------------------------------------------------------
# 10) SAVE ADAPTER + TOKENIZER
# -------------------------------------------------------------------
print("\nSaving LoRA adapter + tokenizer...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(TOKENIZER_DIR)
print("✓ Saved LoRA adapter to:", OUTPUT_DIR)
print("✓ Saved tokenizer to:", TOKENIZER_DIR)
