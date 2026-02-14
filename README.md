# Generative AI Medical Pipeline (Endocrinology)

This project created **LLM models that generate and answer Endocrinology Scenario-Based Multiple Choice Questions (MCQs)**. These models are saved in the follwoing huggingface repo: **https://huggingface.co/HAJERI**

**Educational use only:** This repository and its outputs are intended **for education purposes only** and are **not** intended for clinical decision-making, diagnosis, treatment, or patient-specific medical advice.

This repository implements an end-to-end **Generative AI engineering pipeline** tailored for the **medical domain**, with a specific focus on **endocrinology**. It covers:

- **Fine-tuning** (parameter-efficient adaptation)
- **Merging** (integrating LoRA adapters back into base models)
- **Export / deployment formats** (including GGUF quantization)
- **Rigorous evaluation** across both **MCQ answering** and **MCQ generation**, including an **LLM-as-a-judge** workflow

The pipeline is designed around model families such as **Gemma** and **GPT-OSS**, with explicit support for multiple sizes and export targets.

---

## Repository Structure

### 1) Fine-Tuning Pipeline (`finetuning/`)
Core materials for adapting large language models to medical contexts using parameter-efficient methods.

- **Datasets** (`finetuning/datasets/`)
  - `Generate_Scenario_Endo.csv` — training data oriented toward generating clinical endocrinology scenarios
  - `QA_General_Endo.csv` — endocrinology Q&A (general)
  - `QA_Scenario_Endo.csv` — endocrinology Q&A (scenario-based)

- **Training Scripts** (`finetuning/scripts/`)
  - Gemma 3 training scripts across multiple parameter scales:
    - `train-270m-gemma3.py`
    - `train-1b-gemma3.py`
    - `train-4b-gemma3.py`
    - `train-12b-gemma3.py`
    - `train-27b-gemma3.py`
  - GPT-OSS training script:
    - `train-20b-gptoss.py`

> These scripts are intended for efficient training workflows (commonly LoRA-style adaptation), enabling iteration across multiple model sizes.

---

### 2) Model Merging & Export (`merging/` and `gguf/`)
After LoRA adapters are trained, they are fused and prepared for evaluation/deployment.

- **Merging Scripts** (`merging/`)
  - Corresponding merge scripts for the trained models (examples):
    - `merge-12b-gemma3.py`
    - `merge-20b-gptoss.py`
  - `tokenizer_notes.txt` — notes for tokenizer/special token handling aligned to medical prompting.

- **GGUF Export** (`gguf/`)
  - `README_GGUF.md` — guidance for quantized deployment workflows to run large models efficiently on consumer hardware.

---

### 3) Comprehensive Evaluation Framework (`evaluation/`)
Evaluation is split into two distinct capabilities:

#### A) Answering Evaluation (`evaluation/answering/`)
Benchmarks model performance on answering existing endocrinology MCQs.

- **Datasets & Results**
  - `Endo_Test.csv` — gold-standard benchmark dataset
  - `model_accuracy.csv` — tracked accuracy per model run
  - `Endo_Test_with_correct_AND_ChatGP_answers.xlsx - Sheet1.csv` — cross-referenced output comparisons (including ChatGPT outputs)

- **Evaluation Scripts**
  - `core.py` — backbone evaluation engine
  - Model-specific evaluation scripts (examples):
    - `eval_gemma_hf.py` — evaluation for Hugging Face format models
    - `eval_gptoss_gguf.py` — evaluation for GGUF/quantized variants

#### B) Generation Evaluation (`evaluation/generating/`)
Benchmarks model performance on generating new endocrinology MCQs.

- **Generation Scripts** (`evaluation/generating/gen_script/`)
  - Model configuration:
    - `gemma_models.json`
    - `gptoss20_gguf_models.json`
  - Execution scripts:
    - `run_gemma_mcqgen_all.py`
    - `run_gptoss20_mcqgen_lora.py`
  - `topics.csv` — topic coverage tracking for generation runs

- **LLM-as-a-Judge Pipeline** (`evaluation/generating/chatGPT_to_eval_mcqgen/`)
  Uses ChatGPT as an external judge to score the quality of locally generated MCQs.

  - Prompting:
    - `chatgpt_prompt.DOCX`
  - Parsing / post-processing:
    - `merge_judge_with_key.py`
    - `prepare_strict_pass_items.py`
  - Outputs:
    - `chatgpt_judge_results.jsonl`
    - `judge_items_strict_pass.jsonl`
  - Gap analysis:
    - `missing_topics_by_model.py`

---

### 4) Project Configuration (Root)
Repository-wide configuration and documentation.

- `pyproject.toml` — Python dependencies and environment configuration
- `models_hf.md` — documentation linking local workflows to hosted model artifacts under the HAJERI Hugging Face namespace (e.g., model cards, download references, deployment notes)

---

## Compute acknowledgement
Compute resources (**NVIDIA A100 80GB**) were supplied by the **National eLearning Center (NeLC), Saudi Arabia**, as part of the **AI Sandbox in Digital Learning** initiative.

---

## Authors and contact
- **Model author:** Professor Saeed Awad M. Alqahtani  
- **Affiliation:** Taibah University, Saudi Arabia  
- **Email:** samqahtani@taibahu.edu.sa / dr_alqahtani@hotmail.com
- **Linkedin:** https://www.linkedin.com/in/sam-alqahtani/

---

## Notes
- The repository is organized to support **repeatable experiments** across model sizes, formats (HF vs GGUF), and evaluation modes (answering vs generation).
- Evaluation is designed to produce **traceable outputs** (accuracy tables, judged JSONL runs, strict-pass subsets, and topic gap reports).

