import os
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

# Paths to question files
TASKS = [
    ("definition_questions.json", "definition_questions_answers.csv", "definition"),
    ("classify_questions.csv", "classify_questions_answers.csv", "classify"),
    ("generate_questions.csv", "generate_questions_answers.csv", "generate"),
    ("edit_questions.csv", "edit_questions_answers.csv", "edit"),
]

def generate_text(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def _find_prompt_column(df):
    for col in ["Prompts", "prompt"]:
        if col in df.columns:
            return col
    return df.columns[0]

def run_csv_questions(model, tokenizer, csv_path, output_path, max_new_tokens=256, temperature=0.7, repeat=5):
    df = pd.read_csv(csv_path)
    prompt_col = _find_prompt_column(df)
    all_records = []
    for i in range(repeat):
        records = []
        for prompt in df[prompt_col].astype(str).tolist():
            response = generate_text(model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
            records.append({"prompt": prompt, "response": response, "run": i+1})
        all_records.extend(records)
    out_df = pd.DataFrame(all_records)
    out_df.to_csv(output_path, index=False)
    return out_df

def run_definition_questions(model, tokenizer, json_path, output_path, max_new_tokens=256, temperature=0.7, repeat=5):
    questions = json.load(open(json_path, "r"))
    all_records = []
    for i in range(repeat):
        records = []
        for item in questions:
            prompt = str(item.get("Articulate", ""))
            response = generate_text(model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
            records.append({
                "Concept": item.get("Concept", ""),
                "Domain": item.get("Domain", ""),
                "prompt": prompt,
                "response": response,
                "run": i+1
            })
        all_records.extend(records)
    out_df = pd.DataFrame(all_records)
    out_df.to_csv(output_path, index=False)
    return out_df

def batch_run(models: List[str], repeat: int = 5):
    for model_id in models:
        print(f"\nRunning for model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        for file_in, file_out, task_type in TASKS:
            out_path = f"{os.path.splitext(file_out)[0]}_{model_id.replace('/', '_')}.csv"
            print(f"  Task: {task_type} -> {out_path}")
            if task_type == "definition":
                run_definition_questions(model, tokenizer, file_in, out_path, repeat=repeat)
            else:
                run_csv_questions(model, tokenizer, file_in, out_path, repeat=repeat)

if __name__ == "__main__":
    # Example usage: pass a list of model ids
    models = [
        "Mohammedxo51/llama-3.3-70b-q4",
        # Add more model ids here
    ]
    batch_run(models, repeat=5)
