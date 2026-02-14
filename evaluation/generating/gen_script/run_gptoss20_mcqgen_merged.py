import re
import torch
from dataclasses import dataclass
from typing import List

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)

# -----------------------------
# CONFIG
# -----------------------------
MODEL_ID = "HAJERI/gpt-oss-20b-endocrinology-scenario-mcq-it-merged-fp16"

DEVICE = "cuda"
DTYPE = torch.bfloat16  # or torch.float16


SYSTEM_PROMPT = (
    "You are an expert endocrinology tutor specialized in creating scenario-based "
    "single-best-answer MCQs for postgraduate exams.\n\n"
    "IMPORTANT INSTRUCTIONS:\n"
    "- Output EXACTLY ONE MCQ.\n"
    "- Use the EXACT field labels and order below.\n"
    "- Do NOT output any commentary, analysis, or extra text before or after.\n"
    "- Do NOT repeat the user request.\n"
    "- You MUST provide a separate explanation line for EACH option (A, B, C, D).\n"
    "- Do NOT use a generic 'Explanation:' block.\n\n"
    "Your output MUST be in this exact format (labels and order):\n\n"
    "Scenario: <clinical scenario paragraph>\n"
    "Question: <stem as a clear question>\n"
    "A) <option 1>\n"
    "B) <option 2>\n"
    "C) <option 3>\n"
    "D) <option 4>\n"
    "Correct option: <A/B/C/D>\n"
    "Explanation for A: <brief explanation>\n"
    "Explanation for B: <brief explanation>\n"
    "Explanation for C: <brief explanation>\n"
    "Explanation for D: <brief explanation>\n"
)

USER_TASK_TEMPLATE = (
    "Create EXACTLY ONE high-quality endocrinology scenario-based MCQ about: {topic}.\n"
    "You must NOT create more than one question.\n"
    "Do NOT propose additional topics or future questions.\n"
    "Only output the single MCQ in the required format.\n"
    "Requirements:\n"
    "- The scenario is realistic and includes age, sex, key symptoms, and key labs when relevant.\n"
    "- The question tests applied reasoning, not just recall.\n"
    "- Only ONE option is clearly best.\n"
    "- All options are plausible in an exam setting.\n"
)


@dataclass
class MCQ:
    scenario: str
    question: str
    options: List[str]
    correct_letter: str
    explanations: List[str]


# -----------------------------
# MODEL / PIPELINE LOADING
# -----------------------------

def load_mcq_pipeline():
    print(f"Loading merged model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map="auto",
    )

    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=DTYPE,
    )
    return gen_pipe, tokenizer


# -----------------------------
# PROMPT BUILDING
# -----------------------------

def build_plain_prompt(topic: str) -> str:
    user_part = USER_TASK_TEMPLATE.format(topic=topic)
    return (
        SYSTEM_PROMPT
        + "\n\nUser request:\n"
        + user_part
        + "\n\nAnswer (start directly with 'Scenario:' on the first line):\n"
    )


# -----------------------------
# POST-PROCESSING / TRUNCATION
# -----------------------------

def truncate_after_first_mcq(text: str) -> str:
    """
    Keep only the first MCQ and remove extra content if the model
    starts generating additional prompts or code fencing.
    """
    stop_markers = [
        "\nUser Request:",
        "\nUser request:",
        "```",
    ]
    end = len(text)
    for marker in stop_markers:
        idx = text.find(marker)
        if idx != -1:
            end = min(end, idx)
    return text[:end].strip()


def generate_mcq_raw(gen_pipe, topic: str, max_new_tokens: int = 512) -> str:
    prompt = build_plain_prompt(topic)
    out = gen_pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.05,
        pad_token_id=gen_pipe.tokenizer.eos_token_id,
    )
    # pipeline returns list[{"generated_text": "..."}]
    text = out[0]["generated_text"]
    answer = text[len(prompt):].strip()
    answer = truncate_after_first_mcq(answer)
    return answer


# -----------------------------
# PARSER
# -----------------------------

def parse_mcq_text(text: str) -> MCQ:
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    def find_prefix(prefix: str):
        for i, line in enumerate(lines):
            if line.lower().startswith(prefix.lower()):
                return i
        return -1

    idx_scenario = find_prefix("Scenario:")
    idx_question = find_prefix("Question:")
    idx_correct_option = find_prefix("Correct option:")
    idx_correct_answer = find_prefix("Correct answer:")  # fallback
    idx_exp_a = find_prefix("Explanation for A:")
    idx_exp_b = find_prefix("Explanation for B:")
    idx_exp_c = find_prefix("Explanation for C:")
    idx_exp_d = find_prefix("Explanation for D:")
    idx_explanation_block = find_prefix("Explanation:")  # fallback

    if idx_scenario == -1 or idx_question == -1:
        raise ValueError("Could not find Scenario/Question sections in the output.")

    scenario = lines[idx_scenario][len("Scenario:") :].strip()
    question = lines[idx_question][len("Question:") :].strip()

    # Options
    options = {}
    for line in lines:
        if re.match(r"^A\)", line):
            options["A"] = line[2:].strip()
        elif re.match(r"^B\)", line):
            options["B"] = line[2:].strip()
        elif re.match(r"^C\)", line):
            options["C"] = line[2:].strip()
        elif re.match(r"^D\)", line):
            options["D"] = line[2:].strip()

    options_list = [options.get(k, "") for k in ["A", "B", "C", "D"]]

    # Correct option / answer
    correct_letter = ""
    if idx_correct_option != -1:
        after = lines[idx_correct_option][len("Correct option:") :].strip()
        m = re.search(r"[A-D]", after.upper())
        if m:
            correct_letter = m.group(0)
    elif idx_correct_answer != -1:
        after = lines[idx_correct_answer][len("Correct answer:") :].strip()
        m = re.search(r"[A-D]", after.upper())
        if m:
            correct_letter = m.group(0)

    def extract_after(prefix: str, idx: int) -> str:
        if idx == -1:
            return ""
        return lines[idx][len(prefix) :].strip()

    explanations = [
        extract_after("Explanation for A:", idx_exp_a),
        extract_after("Explanation for B:", idx_exp_b),
        extract_after("Explanation for C:", idx_exp_c),
        extract_after("Explanation for D:", idx_exp_d),
    ]

    # Fallback: single "Explanation:" block with A:, B:, C:, D:
    if not any(explanations) and idx_explanation_block != -1:
        block_lines = lines[idx_explanation_block + 1 :]
        per_opt_expl = {"A": "", "B": "", "C": "", "D": ""}
        current_key = None
        for line in block_lines:
            m = re.match(r"^([ABCD]):\s*(.*)", line)
            if m:
                current_key = m.group(1)
                per_opt_expl[current_key] = m.group(2).strip()
            elif current_key is not None:
                per_opt_expl[current_key] += " " + line
        explanations = [
            per_opt_expl["A"].strip(),
            per_opt_expl["B"].strip(),
            per_opt_expl["C"].strip(),
            per_opt_expl["D"].strip(),
        ]

    return MCQ(
        scenario=scenario,
        question=question,
        options=options_list,
        correct_letter=correct_letter,
        explanations=explanations,
    )


# -----------------------------
# MAIN
# -----------------------------

if __name__ == "__main__":
    # huggingface-cli login should already be done
    gen_pipe, tokenizer = load_mcq_pipeline()

    topic = "Diabetes mellitus & hypoglycemia"
    raw_output = generate_mcq_raw(gen_pipe, topic)

    print("===== RAW MODEL OUTPUT =====")
    print(raw_output)

    if "Scenario:" in raw_output and "Question:" in raw_output:
        try:
            mcq = parse_mcq_text(raw_output)
            print("\n===== PARSED MCQ =====")
            print("Scenario:", mcq.scenario)
            print("Question:", mcq.question)
            print("Options:")
            for label, opt in zip(["A", "B", "C", "D"], mcq.options):
                print(f"  {label}) {opt}")
            print("Correct option:", mcq.correct_letter)
            print("Explanations:")
            for label, expl in zip(["A", "B", "C", "D"], mcq.explanations):
                print(f"  {label}: {expl}")
        except Exception as e:
            print("\nParsing failed:", e)
    else:
        print("\nOutput did not follow the required MCQ format.")


'''
[project]
name = "llm"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "accelerate>=1.0.0",
    "huggingface-hub>=0.26.0",
    "peft>=0.18.0",
    "torch==2.3.0",
    "transformers>=4.46.0",
]

'''

# To run:
# uv run python run_gptoss20_mcqgen_merged.py

