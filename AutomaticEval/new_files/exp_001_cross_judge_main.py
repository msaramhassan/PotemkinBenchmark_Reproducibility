"""
EXP-001: Cross-Model Judging Experiment
========================================
Tests whether self-judging inflates coherence scores by decoupling
the RESPONDER (answers + generates subquestions) from the JUDGE (grades answers).

Hypothesis: A model that consistently misapplies a concept will appear
coherent under self-judging but incoherent under cross-model evaluation.

Supports both local HuggingFace models and API models (OpenAI, Anthropic,
Together, Gemini). Local models are used by default; API models activate
when their keys are set in environment variables.

Usage (local models — zero API cost):
    python exp_001_cross_judge_main.py \
        --responder Qwen/Qwen2.5-7B-Instruct \
        --judge meta-llama/Meta-Llama-3.1-8B-Instruct \
        --benchmark mmlu \
        --num_trials 10

Usage (API models — requires keys):
    python exp_001_cross_judge_main.py \
        --responder gpt-4o \
        --judge meta-llama/Meta-Llama-3.1-8B-Instruct \
        --benchmark mmlu \
        --num_trials 10

Usage (self-judge baseline for comparison):
    python exp_001_cross_judge_main.py \
        --responder Qwen/Qwen2.5-7B-Instruct \
        --judge Qwen/Qwen2.5-7B-Instruct \
        --benchmark mmlu \
        --num_trials 10

Environment Variables for API models:
    OPENAI_API_KEY      — for gpt-4o, gpt-4.5-preview, o3-mini, etc.
    ANTHROPIC_API_KEY   — for claude-3-5-sonnet-20241022, etc.
    TOGETHER_API_KEY    — for Together AI hosted models
    GEMINI_API_KEY      — for gemini-2.0-flash-exp, etc.
"""

import argparse
import os
import sys
from datetime import datetime

from tqdm import tqdm
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allows importing from parent AutomaticEval directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

from utils import (
    answer_and_grade_benchmark_question,
    answer_open_ended_question,
    edit_to_introduce_error,
    generate_subquestions,
    grade_open_ended_question,
    relies_on_concept,
    sample_question,
    models_to_developer,
)
from experiment_logger import ExperimentLogger

# ---------------------------------------------------------------------------
# Ensure API models are registered if keys exist
# (local models don't need registration — utils.py auto-routes them)
# ---------------------------------------------------------------------------
_API_MODELS = {
    "gpt-4o": "openai",
    "gpt-4.5-preview": "openai",
    "o3-mini": "openai",
    "o1-mini": "openai",
    "gemini-2.0-flash-exp": "gemini",
    "claude-3-5-sonnet-20241022": "claude",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": "together",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": "together",
    "deepseek-ai/DeepSeek-V3": "together",
    "deepseek-ai/DeepSeek-R1": "together",
    "Qwen/Qwen2-VL-72B-Instruct": "together",
}
models_to_developer.update(_API_MODELS)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="EXP-001: Cross-Model Judging Matrix",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
)
parser.add_argument(
    "--responder", type=str, required=True,
    help="Model that answers questions and generates subquestions. "
         "Can be a HuggingFace path (local) or an API model name."
)
parser.add_argument(
    "--judge", type=str, required=True,
    help="Model that grades answer correctness. "
         "Can be a HuggingFace path (local) or an API model name."
)
parser.add_argument("--benchmark", type=str, default="mmlu", choices=["mmlu", "bbh"])
parser.add_argument("--num_subquestions", type=int, default=5,
                    help="Number of subquestions (m) per concept (paper default: 5)")
parser.add_argument("--num_trials", type=int, default=10,
                    help="Number of concept trials (paper default: 10)")
parser.add_argument("--seed", type=int, default=None,
                    help="Random seed for reproducibility")
args = parser.parse_args()

if args.seed is not None:
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
responder_tag = args.responder.replace("/", "_").replace(":", "_")
judge_tag = args.judge.replace("/", "_").replace(":", "_")
pair_tag = f"{responder_tag}__judged_by__{judge_tag}"
is_self_judge = (args.responder == args.judge)

index_to_category = {0: "initial", 1: "edited_with_error"}
overall_coherence = []
score_per_concept = []

log_dir = os.path.join(PARENT_DIR, "logs", "exp001", pair_tag, args.benchmark)
logger = ExperimentLogger(output_dir=log_dir, model=args.responder, benchmark=args.benchmark)

# Change working directory to AutomaticEval for benchmark data paths
os.chdir(PARENT_DIR)

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"\n{'=' * 70}")
print(f"  EXP-001: Cross-Model Judging Experiment")
print(f"  Timestamp  : {timestamp}")
print(f"  Responder  : {args.responder}")
print(f"  Judge      : {args.judge}")
print(f"  Self-judge : {'YES (baseline)' if is_self_judge else 'NO (cross-model)'}")
print(f"  Benchmark  : {args.benchmark}")
print(f"  Trials     : {args.num_trials}")
print(f"  Subquestions: {args.num_subquestions}")
print(f"  Seed       : {args.seed}")
print(f"{'=' * 70}\n")

# ---------------------------------------------------------------------------
# Main experiment loop — mirrors main.py but JUDGE != RESPONDER
# ---------------------------------------------------------------------------
bar = tqdm(range(args.num_trials), desc="Trials")
skipped_trials = 0

for trial_index in bar:
    print(f"\n--- Trial {trial_index + 1}/{args.num_trials} ---")

    # Step 1: Find a concept-bearing, correctly-answered question
    # (RESPONDER answers; RESPONDER detects concept)
    attempts = 0
    max_sampling_attempts = 50
    while attempts < max_sampling_attempts:
        attempts += 1
        question, gold_answer, subject = sample_question(args.benchmark)
        logger.log_sample_question(trial_index, question, subject, gold_answer)

        concept_classification, concept = relies_on_concept(question, args.responder)
        logger.log_concept_detection(trial_index, question, subject,
                                     concept_classification, concept)
        if concept_classification:
            correct, full_answer = answer_and_grade_benchmark_question(
                question, args.responder, gold_answer, args.benchmark == "mmlu"
            )
            logger.log_initial_answer(trial_index, question, subject,
                                      concept, full_answer, correct)
            if correct:
                break
    else:
        print(f"  WARNING: Could not find a valid question after {max_sampling_attempts} "
              f"attempts. Skipping trial {trial_index}.")
        skipped_trials += 1
        continue

    print(f"  Concept: {concept}")
    print(f"  Question: {question[:80]}...")

    # Step 2: Generate subquestions (RESPONDER generates)
    max_gen_attempts = 10
    subquestions = []
    while max_gen_attempts > 0:
        subquestions = generate_subquestions(
            question, concept, args.responder, args.num_subquestions
        )
        logger.log_subquestion_generation(
            trial_index, question, concept, subquestions, args.num_subquestions
        )
        if len(subquestions) >= args.num_subquestions:
            subquestions = np.random.choice(
                subquestions, size=args.num_subquestions, replace=False
            ).tolist()
            break
        max_gen_attempts -= 1

    if not subquestions:
        print(f"  WARNING: Could not generate subquestions. Skipping trial {trial_index}.")
        skipped_trials += 1
        continue

    # Step 3: For each subquestion — RESPONDER answers, JUDGE evaluates
    for sq_idx, subquestion in enumerate(subquestions):
        # RESPONDER answers the subquestion
        extracted_answer = answer_open_ended_question(subquestion, args.responder)
        logger.log_subquestion_answering(
            trial_index, question, concept, sq_idx, subquestion, extracted_answer
        )

        # RESPONDER introduces a subtle error
        _, answer_with_error = edit_to_introduce_error(
            subquestion, extracted_answer, args.responder
        )
        logger.log_answer_editing(
            trial_index, question, concept, sq_idx,
            subquestion, extracted_answer, answer_with_error
        )

        # JUDGE evaluates both versions (this is the key decoupling)
        expected_answers = ["correct", "incorrect"]
        candidate_answers = [extracted_answer, answer_with_error]

        for version_idx, (answer_to_grade, expected) in enumerate(
            zip(candidate_answers, expected_answers)
        ):
            # *** JUDGE model grades — NOT the responder ***
            judge_answer, full_judge_answer = grade_open_ended_question(
                subquestion, answer_to_grade, args.judge
            )

            judge_lower = judge_answer.strip().lower()
            valid_grading = (
                judge_lower[:7] == "correct" or judge_lower[:9] == "incorrect"
            )

            if valid_grading:
                coherent = 1 if judge_lower[:len(expected)] == expected else 0
                judge_label = "correct" if judge_lower[:7] == "correct" else "incorrect"
            else:
                coherent = None
                judge_label = None

            category = index_to_category[version_idx]

            logger.log_grading(
                trial_index=trial_index, question=question, concept=concept,
                subquestion_index=sq_idx, subquestion=subquestion,
                model_answer=answer_to_grade, judge_answer_raw=full_judge_answer,
                judge_label=judge_label, expected_label=expected,
                category=category, coherent=coherent, valid=valid_grading
            )

            if valid_grading:
                overall_coherence.append(coherent)
                logger.log_coherence_scoring(
                    trial_index=trial_index, question=question, concept=concept,
                    coherent=coherent, expected_label=expected,
                    judge_label=judge_label, category=category
                )

    # Potemkin rate: normalized incoherence — 0 is perfect, 1 is random.
    # Mirrors main.py: computed from ALL coherence accumulated so far (running).
    if not overall_coherence:
        print("  WARNING: No valid coherence judgments for this trial. Skipping rate update.")
        skipped_trials += 1
        continue
    print("\nCalculating Potemkin rate...")
    potemkin_rate = 2 * (1 - np.mean(overall_coherence))
    score_per_concept.append(potemkin_rate)
    stderr = np.std(score_per_concept) / np.sqrt(len(score_per_concept))
    bar.set_description(
        f"[{responder_tag[:15]}→{judge_tag[:15]}] "
        f"PR={np.mean(score_per_concept):.3f}±{stderr:.3f}"
    )
    logger.log_potemkin_rate(
        trial_index=trial_index,
        concept=concept,
        potemkin_rate=potemkin_rate,
        overall_coherence_mean=float(np.mean(overall_coherence)),
        running_scores=score_per_concept,
    )

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print(f"\n{'=' * 70}")
if score_per_concept:
    final_rate = np.mean(score_per_concept)
    final_err = np.std(score_per_concept) / np.sqrt(len(score_per_concept))
    print(f"  RESULT — Potemkin rate (lower bound): {final_rate:.4f} ± {final_err:.4f}")
else:
    print("  RESULT — No valid trials completed.")
    final_rate = None

print(f"  Responder  : {args.responder}")
print(f"  Judge      : {args.judge}")
print(f"  Self-judge : {'YES' if is_self_judge else 'NO'}")
print(f"  Benchmark  : {args.benchmark}")
print(f"  Completed  : {args.num_trials - skipped_trials}/{args.num_trials} trials")
print(f"  Skipped    : {skipped_trials}")
if final_rate is not None:
    print(f"  Coherence  : {np.mean(overall_coherence):.4f} ({len(overall_coherence)} judgments)")
print(f"{'=' * 70}\n")

# Save logs
log_file = logger.save()
log_jsonl = logger.save_jsonl()
print(f"Logs saved to: {log_file}")
print(f"JSONL saved to: {log_jsonl}")

# Save potemkin rate summary (mirrors what main.py tracks in score_per_concept)
import json as _json
summary = {
    "responder": args.responder,
    "judge": args.judge,
    "is_self_judge": is_self_judge,
    "benchmark": args.benchmark,
    "num_trials": args.num_trials,
    "completed_trials": args.num_trials - skipped_trials,
    "skipped_trials": skipped_trials,
    "seed": args.seed,
    "timestamp": timestamp,
    "score_per_concept": score_per_concept,
    "final_potemkin_rate": float(np.mean(score_per_concept)) if score_per_concept else None,
    "final_potemkin_rate_stderr": float(np.std(score_per_concept) / np.sqrt(len(score_per_concept))) if len(score_per_concept) > 1 else None,
    "overall_coherence_mean": float(np.mean(overall_coherence)) if overall_coherence else None,
    "num_coherence_judgments": len(overall_coherence),
}
summary_path = os.path.join(log_dir, f"potemkin_rates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
with open(summary_path, "w") as _f:
    _json.dump(summary, _f, indent=2)
print(f"Potemkin rate summary saved to: {summary_path}")
