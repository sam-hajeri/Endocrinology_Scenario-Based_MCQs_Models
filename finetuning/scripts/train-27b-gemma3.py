import unsloth
import os
import time
import math

# -------------------------------------------------------------------
# 1) CRITICAL IMPORT ORDER FIX
# Unsloth MUST be imported before trl, transformers, etc.
# -------------------------------------------------------------------
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

# ENV Config
os.environ["TORCH_COMPILE_DISABLE"]   = "1"
os.environ["TORCHDYNAMO_DISABLE"]     = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Now import the rest
import torch
import pandas as pd
from datasets import Dataset, concatenate_datasets
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments

print("=" * 60)
print("Starting Training (Gemma 3 27B Pt - Import Order Fixed)")
print("=" * 60)

# -------------------------------------------------------------------
# 2) CONFIG
# -------------------------------------------------------------------
MODEL_NAME    = "unsloth/gemma-3-27b-pt-bnb-4bit"
MAX_SEQ_LEN   = 2048
OUTPUT_DIR    = "./gemma_3_27b_pt_medical_finetuned"
TOKENIZER_DIR = "./gemma_3_27b_pt_medical_tokenizer"
TEST_SIZE     = 0.05
SEED          = 3407

# -------------------------------------------------------------------
# 3) LOAD MODEL
# -------------------------------------------------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = MODEL_NAME,
    max_seq_length  = MAX_SEQ_LEN,
    dtype           = None,
    load_in_4bit    = True,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma",
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = SEED,
    use_rslora = False,
    loftq_config = None,
)

print(f"Model loaded: {MODEL_NAME}")

# -------------------------------------------------------------------
# 4) LOAD DATA & FORMAT
# -------------------------------------------------------------------
try:
    df_mcq         = pd.read_excel("./datasets/Generate_Scenario_Endo.xlsx")
    df_qa_scenario = pd.read_excel("./datasets/QA_Scenario_Endo.xlsx")
    df_qa_general  = pd.read_excel("./datasets/QA_General_Endo.xlsx")
except FileNotFoundError:
    print("Warning: Data files not found. Creating dummy data.")
    df_mcq = pd.DataFrame([{"instruction": "test", "output": "test"}])
    df_qa_scenario = pd.DataFrame([{"Question": "test", "Answer": "test"}])
    df_qa_general = pd.DataFrame([{"Question": "test", "Answer": "test"}])

def format_mcq(example):
    system_text = "You are an expert endocrinologist creating clinical MCQs for medical education.\n\n"
    msg = [
        {"role": "user", "content": system_text + str(example['instruction'])},
        {"role": "assistant", "content": str(example['output'])},
    ]
    return {"text": tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)}

def format_qa_scenario(example):
    system_text = "You are a clinical endocrinology expert answering scenario-based medical questions.\n\n"
    msg = [
        {"role": "user", "content": system_text + str(example['Question'])},
        {"role": "assistant", "content": str(example['Answer'])},
    ]
    return {"text": tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)}

def format_qa_general(example):
    system_text = "You are a medical expert answering general endocrinology and clinical questions.\n\n"
    msg = [
        {"role": "user", "content": system_text + str(example['Question'])},
        {"role": "assistant", "content": str(example['Answer'])},
    ]
    return {"text": tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)}

ds_mcq = Dataset.from_pandas(df_mcq).map(format_mcq)
ds_qa_scenario = Dataset.from_pandas(df_qa_scenario).map(format_qa_scenario)
ds_qa_general = Dataset.from_pandas(df_qa_general).map(format_qa_general)

def clean_cols(ds):
    return ds.remove_columns([c for c in ds.column_names if c != "text"])

dataset = concatenate_datasets([clean_cols(ds_mcq), clean_cols(ds_qa_scenario), clean_cols(ds_qa_general)])
split_dataset = dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)
train_ds, val_ds = split_dataset["train"], split_dataset["test"]

# -------------------------------------------------------------------
# 5) TRAINING CONFIG
# With correct import order, SFTConfig will now accept these args
# -------------------------------------------------------------------
sft_config = SFTConfig(
    output_dir                  = OUTPUT_DIR,
    max_seq_length              = MAX_SEQ_LEN,   # <--- Correctly in SFTConfig
    dataset_text_field          = "text",        # <--- Correctly in SFTConfig
    packing                     = False,         # <--- Correctly in SFTConfig
    per_device_train_batch_size = 1,
    per_device_eval_batch_size  = 1,
    gradient_accumulation_steps = 16,
    warmup_steps                = 50,
    num_train_epochs            = 1,
    learning_rate               = 2e-4,
    lr_scheduler_type           = "cosine",
    logging_steps               = 10,
    save_steps                  = 200,
    save_total_limit            = 2,
    optim                       = "adamw_8bit",
    weight_decay                = 0.01,
    seed                        = SEED,
    report_to                   = "none",
    bf16                        = is_bfloat16_supported(),
    fp16                        = not is_bfloat16_supported(),
    gradient_checkpointing      = True,
    max_grad_norm               = 0.3,
    dataloader_num_workers      = 4,
)

# -------------------------------------------------------------------
# 6) TRAINER
# -------------------------------------------------------------------
trainer = SFTTrainer(
    model            = model,
    args             = sft_config,
    train_dataset    = train_ds,
    eval_dataset     = val_ds,
    processing_class = tokenizer,
)

print("\nStarting Training...")
trainer.train()

print("\nSaving...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(TOKENIZER_DIR)
print(f"✓ Done! Saved to {OUTPUT_DIR}")




# # -------------------------------------------------------------------

# import os
# import time
# import math

# # -------------------------------------------------------------------
# # 1) CRITICAL IMPORT ORDER
# # Unsloth MUST be imported before trl, transformers, etc.
# # -------------------------------------------------------------------
# from unsloth import FastLanguageModel, is_bfloat16_supported
# from unsloth.chat_templates import get_chat_template

# # ENV Config for A100 optimization
# os.environ["TORCH_COMPILE_DISABLE"]   = "1"
# os.environ["TORCHDYNAMO_DISABLE"]     = "1"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# # Now import the rest
# import torch
# import pandas as pd
# from datasets import Dataset, concatenate_datasets
# from trl import SFTTrainer, SFTConfig

# print("=" * 60)
# print("Starting Training (Gemma 3 27B Pt - A100 Optimized)")
# print("=" * 60)

# # -------------------------------------------------------------------
# # 2) CONFIG
# # -------------------------------------------------------------------
# MODEL_NAME    = "unsloth/gemma-3-27b-pt-bnb-4bit"
# MAX_SEQ_LEN   = 2048
# OUTPUT_DIR    = "./gemma_3_27b_pt_medical_finetuned"
# TOKENIZER_DIR = "./gemma_3_27b_pt_medical_tokenizer"
# TEST_SIZE     = 0.05
# SEED          = 3407

# # -------------------------------------------------------------------
# # 3) LOAD MODEL
# # -------------------------------------------------------------------
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name      = MODEL_NAME,
#     max_seq_length  = MAX_SEQ_LEN,
#     dtype           = None,          # Auto-detects bfloat16 for A100
#     load_in_4bit    = True,
# )

# # Teach the Pretrained model the Gemma chat format
# tokenizer = get_chat_template(
#     tokenizer,
#     chat_template = "gemma",
# )

# model = FastLanguageModel.get_peft_model(
#     model,
#     r = 16,
#     target_modules = [
#         "q_proj", "k_proj", "v_proj", "o_proj",
#         "gate_proj", "up_proj", "down_proj",
#     ],
#     lora_alpha = 16,
#     lora_dropout = 0,
#     bias = "none",
#     use_gradient_checkpointing = "unsloth",
#     random_state = SEED,
#     use_rslora = False,
#     loftq_config = None,
# )

# print(f"Model loaded: {MODEL_NAME}")

# # -------------------------------------------------------------------
# # 4) LOAD DATA & FORMAT
# # -------------------------------------------------------------------
# try:
#     df_mcq         = pd.read_excel("./datasets/Generate_Scenario_Endo.xlsx")
#     df_qa_scenario = pd.read_excel("./datasets/QA_Scenario_Endo.xlsx")
#     df_qa_general  = pd.read_excel("./datasets/QA_General_Endo.xlsx")
# except FileNotFoundError:
#     print("Warning: Data files not found. Creating dummy data for testing.")
#     df_mcq = pd.DataFrame([{"instruction": "test", "output": "test"}])
#     df_qa_scenario = pd.DataFrame([{"Question": "test", "Answer": "test"}])
#     df_qa_general = pd.DataFrame([{"Question": "test", "Answer": "test"}])

# # Formatting functions that apply the Gemma Chat Template
# def format_mcq(example):
#     system_text = "You are an expert endocrinologist creating clinical MCQs for medical education.\n\n"
#     msg = [
#         {"role": "user", "content": system_text + str(example['instruction'])},
#         {"role": "assistant", "content": str(example['output'])},
#     ]
#     return {"text": tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)}

# def format_qa_scenario(example):
#     system_text = "You are a clinical endocrinology expert answering scenario-based medical questions.\n\n"
#     msg = [
#         {"role": "user", "content": system_text + str(example['Question'])},
#         {"role": "assistant", "content": str(example['Answer'])},
#     ]
#     return {"text": tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)}

# def format_qa_general(example):
#     system_text = "You are a medical expert answering general endocrinology and clinical questions.\n\n"
#     msg = [
#         {"role": "user", "content": system_text + str(example['Question'])},
#         {"role": "assistant", "content": str(example['Answer'])},
#     ]
#     return {"text": tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)}

# # Map and Clean
# ds_mcq = Dataset.from_pandas(df_mcq).map(format_mcq)
# ds_qa_scenario = Dataset.from_pandas(df_qa_scenario).map(format_qa_scenario)
# ds_qa_general = Dataset.from_pandas(df_qa_general).map(format_qa_general)

# def clean_cols(ds):
#     return ds.remove_columns([c for c in ds.column_names if c != "text"])

# dataset = concatenate_datasets([clean_cols(ds_mcq), clean_cols(ds_qa_scenario), clean_cols(ds_qa_general)])
# split_dataset = dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)
# train_ds, val_ds = split_dataset["train"], split_dataset["test"]

# print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

# # -------------------------------------------------------------------
# # 5) TRAINING CONFIG (A100 Optimized)
# # -------------------------------------------------------------------
# sft_config = SFTConfig(
#     output_dir                  = OUTPUT_DIR,
#     max_seq_length              = MAX_SEQ_LEN,
#     dataset_text_field          = "text",
#     packing                     = False,
    
#     # --- A100 SPEED BOOST SETTINGS ---
#     # Increased batch size to saturate the A100 80GB
#     per_device_train_batch_size = 4,   
#     gradient_accumulation_steps = 4,   # 4 * 4 = 16 effective batch size
#     dataloader_num_workers      = 8,   # Feed data faster
#     bf16                        = True, # Native A100 precision
#     fp16                        = False,
#     # ---------------------------------

#     per_device_eval_batch_size  = 4,
#     warmup_steps                = 50,
#     num_train_epochs            = 1,
#     learning_rate               = 2e-4,
#     lr_scheduler_type           = "cosine",
#     logging_steps               = 5,
#     save_steps                  = 200,
#     save_total_limit            = 2,
#     optim                       = "adamw_8bit",
#     weight_decay                = 0.01,
#     seed                        = SEED,
#     report_to                   = "none",
#     gradient_checkpointing      = True,
#     max_grad_norm               = 0.3,
# )

# # -------------------------------------------------------------------
# # 6) TRAINER
# # -------------------------------------------------------------------
# trainer = SFTTrainer(
#     model            = model,
#     args             = sft_config,
#     train_dataset    = train_ds,
#     eval_dataset     = val_ds,
#     processing_class = tokenizer,
# )

# print("\nStarting Training...")
# t0 = time.time()
# trainer.train()
# t1 = time.time()
# print(f"Training finished in {(t1 - t0)/60:.2f} minutes.")

# print("\nSaving...")
# model.save_pretrained(OUTPUT_DIR)
# tokenizer.save_pretrained(TOKENIZER_DIR)
# print(f"✓ Done! Saved to {OUTPUT_DIR}")