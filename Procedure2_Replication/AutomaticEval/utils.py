import os
import json
import csv
import random
import re
import time

from openai import OpenAI
from together import Together
from anthropic import Anthropic
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from private.models import api_keys
from prompts import contains_concept_prompt, subquestion_generation_prompt

# ---------------------------------------------------------------------------
# Local model cache — keyed by model name, holds all models simultaneously
# to avoid costly reloads when alternating between responder and judge.
# ---------------------------------------------------------------------------
_model_cache: dict = {}  # model_name -> {"model": ..., "tokenizer": ..., "processor": ...}

# Patterns that identify reasoning/chain-of-thought models requiring more output tokens.
_REASONING_MODEL_PATTERNS = [
    "deepseek-r1", "deepseek_r1", "r1-distill", "r1_distill",
    "-r1", "_r1", "o1-", "o3-", "qwq", "skywork-o1",
]

# Default token budgets.
_DEFAULT_MAX_NEW_TOKENS = 1024
_REASONING_MAX_NEW_TOKENS = 8192


def _is_reasoning_model(model_name: str) -> bool:
    """Return True if the model name matches a known reasoning/CoT model pattern."""
    lower = model_name.lower()
    return any(pat in lower for pat in _REASONING_MODEL_PATTERNS)


def _is_awq_model(model_name: str) -> bool:
    """Return True if the model should be loaded with AutoAWQ."""
    return "awq" in model_name.lower()


def _is_vl_config(model_name):
    """Return True if the model uses a vision-language config (e.g. Qwen2VL)."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config_class = type(config).__name__
    return "VL" in config_class or config_class.endswith("VLConfig")


def load_local_model(model_name):
    """Load a local model using transformers or AutoAWQ. Caches ALL loaded models by name
    so switching between responder and judge never triggers a reload."""
    global _model_cache

    if model_name in _model_cache:
        entry = _model_cache[model_name]
        return entry["model"], entry["tokenizer"]

    import torch

    print(f"Loading local model: {model_name}")

    if _is_awq_model(model_name):
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoAWQForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    elif _is_vl_config(model_name):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_name)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, dtype="auto", device_map="auto"
        )
        tokenizer = processor.tokenizer
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    _model_cache[model_name] = {"model": model, "tokenizer": tokenizer, "processor": processor}
    print(f"Model loaded successfully.")
    return model, tokenizer


def generate_local_inference(prompt, model_name, max_new_tokens=None):
    """Generate inference using a local HuggingFace model.

    If max_new_tokens is None, the budget is chosen automatically:
    reasoning models (DeepSeek-R1, QwQ, etc.) get _REASONING_MAX_NEW_TOKENS
    tokens; all other models get _DEFAULT_MAX_NEW_TOKENS.
    """
    import torch

    if max_new_tokens is None:
        max_new_tokens = (
            _REASONING_MAX_NEW_TOKENS if _is_reasoning_model(model_name)
            else _DEFAULT_MAX_NEW_TOKENS
        )

    model, tokenizer = load_local_model(model_name)
    processor = _model_cache[model_name]["processor"]

    # AutoAWQForCausalLM does not expose a .device attribute; use parameters() instead.
    device = model.device if hasattr(model, 'device') else next(model.parameters()).device

    if processor is not None:
        # VL model path (text-only input)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    else:
        # Standard causal LM path
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = prompt

        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    return response


# ---------------------------------------------------------------------------
# API model registry — uncomment entries to enable them
# ---------------------------------------------------------------------------
models_to_developer = {
    "gpt-4o": "openai",
    # "gpt-4.5-preview": "openai",
    # "o3-mini": "openai",
    # "o1-mini": "openai",
    # "gemini-2.0-flash-exp": "gemini",
    # "claude-3-5-sonnet-20241022": "claude",
    # "meta-llama/Llama-3.3-70B-Instruct-Turbo": "together",
    # "mistralai/Mistral-7B-Instruct-v0.2": "together",
    # "deepseek-ai/DeepSeek-V3": "together",
    # "deepseek-ai/DeepSeek-R1": "together",
    # "Qwen/Qwen2-VL-72B-Instruct": "together",
    # "Qwen/Qwen2-72B-Instruct": "together",
}

models = list(models_to_developer.keys())

BENCHMARK_PATHS = {
    'bbh': 'raw_data/benchmarks/bbh',
    'mmlu': 'raw_data/benchmarks/mmlu',
}
FINAL_TAG = "FINAL ANSWER:"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
_OPENAI_MODEL_PREFIXES = ("gpt-", "GPT-", "o1-", "o3-", "o4-")


def _is_openai_model(model_name: str) -> bool:
    """Return True if the model name looks like an OpenAI API model."""
    return any(model_name.startswith(p) for p in _OPENAI_MODEL_PREFIXES)


def generate_inference(prompt, model):
    """Route inference to the appropriate backend (local or API)."""
    if model.startswith("local:"):
        return generate_local_inference(prompt, model[6:])
    elif model not in models_to_developer:
        if _is_openai_model(model):
            # OpenAI API model not yet in the registry — call it directly
            client = OpenAI(api_key=api_keys["openai"])
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return completion.choices[0].message.content
        # Assume it's a local HuggingFace model path
        return generate_local_inference(prompt, model)

    provider = models_to_developer[model]

    if provider == "openai":
        client = OpenAI(api_key=api_keys["openai"])
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content

    elif provider == "together":
        client = Together(api_key=api_keys["together"])
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
        )
        return resp.choices[0].message.content

    elif provider == "gemini":
        genai.configure(api_key=api_keys["gemini"])
        gemini_model = genai.GenerativeModel(model)
        backoff = 1  # seconds; doubles on each retry up to 60 s
        while True:
            try:
                return gemini_model.generate_content(prompt).text
            except ResourceExhausted as e:
                delay = getattr(e, "retry_delay", None)
                delay = delay.seconds if delay else backoff
                print(f"[Gemini] rate-limited, sleeping {delay}s")
                time.sleep(delay)
                backoff = min(backoff * 2, 60)

    elif provider == "claude":
        client = Anthropic(api_key=api_keys["claude"])
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    else:
        raise ValueError(f"Unknown provider '{provider}' for model '{model}'.")


# ---------------------------------------------------------------------------
# Question answering helpers
# ---------------------------------------------------------------------------
def answer_benchmark_question(question: str, model: str) -> tuple[str, str]:
    cot_prompt = (
        "You are an expert tutor. You may think step-by-step to reach the solution, "
        f"but you MUST finish with a line that starts exactly with `{FINAL_TAG}` "
        "followed by your single best answer.\n\n"
        f"Question:\n{question}\n"
    )
    full_msg = generate_inference(cot_prompt, model)
    m = re.search(rf"{re.escape(FINAL_TAG)}\s*(.+)", full_msg, flags=re.I | re.S)
    extracted = m.group(1).strip() if m else None
    return full_msg, extracted


def answer_open_ended_question(question: str, model: str) -> str:
    prompt = (
        "You are an expert tutor. You may think step-by-step to reach the solution, "
        f"but you MUST finish with an answer that starts exactly with `{FINAL_TAG}` "
        "followed by your single best answer. If the question asks you to explain your "
        f"answer, you should also include the explanation after `{FINAL_TAG}`.\n\n"
        f"Question:\n{question}\n"
    )
    full_answer = generate_inference(prompt, model)
    return extract_final_answer(full_answer)


def answer_and_grade_benchmark_question(question: str,
                                        model: str,
                                        gold_answer: str,
                                        multiple_choice: bool) -> tuple[bool, str]:
    full_msg, extracted = answer_benchmark_question(question, model)
    return grade_benchmark(extracted, gold_answer, multiple_choice), full_msg


def edit_to_introduce_error(question: str, initial_answer: str, model: str) -> tuple[str, str]:
    prompt = (
        "Modify the following answer to introduce a subtle error. The error should be "
        "subtle but one such that a human who knows the concept would know the answer "
        "is incorrect. If the answer is already incorrect, you can leave it the same. "
        f"You can reason all you'd like, but end the response with `{FINAL_TAG}` followed "
        "by the full modified answer.\n\n"
        f"Question: {question}\n"
        f"Answer: {initial_answer}\n"
    )
    full_answer = generate_inference(prompt, model)
    modified_answer = extract_final_answer(full_answer)
    return full_answer, modified_answer


def edit_to_remove_error(question: str, initial_answer: str, model: str) -> tuple[str, str]:
    prompt = (
        "Modify the following answer to remove any error. If there are errors, "
        "they might be subtle but one such that a human who knows the concept would "
        "know the answer is incorrect. If the answer is already correct, you can leave "
        "it the same. "
        f"You can reason all you'd like, but end the response with `{FINAL_TAG}` followed "
        "by the full modified answer.\n\n"
        f"Question: {question}\n"
        f"Answer: {initial_answer}\n"
    )
    full_answer = generate_inference(prompt, model)
    modified_answer = extract_final_answer(full_answer)
    return full_answer, modified_answer


def extract_final_answer(response: str) -> str:
    match = re.search(rf'{re.escape(FINAL_TAG)}\s*(.*)', response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


# ---------------------------------------------------------------------------
# Grading helpers
# ---------------------------------------------------------------------------
def grade_benchmark(model_answer: str | None,
                    gold_answer: str,
                    multiple_choice: bool) -> bool:
    """
    Simple exact-match grader for BBH & MMLU:
      - MMLU-style MCQ  → compare first char (A/B/C/…)
      - BBH open answer → case-/space-insensitive string match
    """
    if model_answer is None:
        return False
    if multiple_choice:
        return model_answer[0].upper() == gold_answer.upper()
    else:
        return model_answer.strip().lower() == gold_answer.strip().lower()


def grade_open_ended_question(question: str, model_answer: str, model: str) -> tuple[str, str]:
    cot_prompt = (
        "You are an expert tutor. You will be given a question and a possible "
        "answer to the question. Your job is to determine if the answer is correct "
        "or incorrect. You should only grade it correct if the answer (including the reasoning) "
        "is completely correct. "
        f"You can reason all you'd like, but end the response with `{FINAL_TAG}` "
        "followed by either 'correct' or 'incorrect', and nothing should come after that.\n\n"
        f"Question: {question}\n"
        f"Answer: {model_answer}\n"
    )
    full_msg = generate_inference(cot_prompt, model)
    extracted_answer = extract_final_answer(full_msg).replace("*", "").strip()
    return extracted_answer, full_msg


# ---------------------------------------------------------------------------
# Subquestion helpers
# ---------------------------------------------------------------------------
def generate_subquestions(question, concept, model, num_subquestions):
    prelude = (
        f"The following is a question about the following concept: {concept}. "
        f"Here is the question: {question}.\n\n"
        f"Write {num_subquestions} other questions that test whether someone who understands "
        f"the concepts the question is testing truly understands them."
    )
    prompt = prelude + "\n\n" + subquestion_generation_prompt
    inference = generate_inference(prompt, model)
    return parse_questions(inference)


def parse_questions(inference):
    """Extract subquestions from model output with robust fallbacks.

    Primary path expects `<question>...</question>` tags. Many local models
    ignore formatting, so we additionally accept numbered or bullet lists and
    lines that end with a question mark. Duplicates are removed while
    preserving order.
    """

    candidates = []

    # Preferred: explicit tags
    tagged = re.findall(r'<question>(.*?)</question>', inference, re.DOTALL)
    candidates.extend([q.strip() for q in tagged if q.strip()])

    # Fallback: Q1:/Question 1: style blocks
    q_blocks = re.findall(r'(?:^|\n)(?:Q|Question)\s*\d+\s*[:.-]\s*(.+?)(?=\n(?:Q|Question)\s*\d+\s*[:.-]|\Z)',
                          inference, flags=re.IGNORECASE | re.DOTALL)
    candidates.extend([q.strip() for q in q_blocks if q.strip()])

    # Fallback: bullet/numbered lines ending with a question mark
    for line in re.split(r'\n+', inference):
        cleaned = re.sub(r'^\s*(?:[-*]|\d+[.)])\s*', '', line).strip()
        if cleaned.endswith('?') and len(cleaned) > 8:  # avoid stray short strings
            candidates.append(cleaned)

    # De-duplicate while preserving order
    seen = set()
    unique_candidates = []
    for q in candidates:
        if q not in seen:
            seen.add(q)
            unique_candidates.append(q)

    return unique_candidates


def relies_on_concept(question, model):
    """Return (bool, concept_name) indicating whether the question tests a specific concept."""
    prompt = contains_concept_prompt + "\n\n" + question + "\n"
    response = generate_inference(prompt, model)
    answer_match = re.search(r'ANSWER:\s*(Yes|No)', response, re.IGNORECASE)
    concept_match = re.search(r'CONCEPT:\s*(.*)', response, re.IGNORECASE)
    answer = answer_match.group(1) if answer_match else None
    concept = concept_match.group(1).strip() if concept_match else None
    try:
        classification = answer.lower() == "yes"
    except AttributeError:
        # answer is None — fall back based on whether a concept name was detected
        return (True, concept) if concept is not None else (False, None)
    return classification, concept


# ---------------------------------------------------------------------------
# Dataset sampling
# ---------------------------------------------------------------------------
def sample_question(benchmark, subject=None):
    base_path = BENCHMARK_PATHS[benchmark]
    if subject is None:
        files = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]
        subject = random.choice(files)
    file_path = os.path.join(base_path, subject)

    print("Opening file:", file_path)

    if benchmark == 'bbh':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            question_data = random.choice(data['examples'])
            question_str = question_data['input']
            answer = question_data['target']

    elif benchmark == 'mmlu':
        with open(file_path, 'r', encoding='utf-8') as f:
            rows = list(csv.reader(f))
            row = random.choice(rows)
            question_str = row[0]
            choices = row[1:-1]
            answer_letter = row[-1].strip()
            formatted_choices = "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)])
            question_str = f"{question_str}\n{formatted_choices}"
            answer = answer_letter

    else:
        raise ValueError(f"Unsupported benchmark: '{benchmark}'.")

    return question_str, answer, subject
