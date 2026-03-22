"""
EXP-001: Cross-Model Judging Matrix (RQ: Self-Judge Circularity)
================================================================
Modified pipeline where RESPONDER and JUDGE are decoupled.

Hypothesis: A model applying a wrong interpretation consistently will
appear coherent under self-judging but incoherent under cross-model evaluation.

Usage:
    python exp_001_cross_judge_main.py \
        --responder gpt-4o \
        --judge claude-3-5-sonnet-20241022 \
        --benchmark mmlu \
        --num_trials 20
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from tqdm import tqdm

import numpy as np

# Ensure sibling imports work regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils
from utils import (
    answer_and_grade_benchmark_question,
    answer_open_ended_question,
    edit_to_introduce_error,
    generate_subquestions,
    grade_open_ended_question,
    relies_on_concept,
    sample_question,
)
from experiment_logger import ExperimentLogger

# ---------------------------------------------------------------------------
# Extend model routing to cover all three experiment models
# (does not modify utils.py on disk – only the in-process dict)
# ---------------------------------------------------------------------------
utils.models_to_developer.update({
    "gpt-4o": "openai",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": "together",
    "claude-3-5-sonnet-20241022": "claude",
})

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="EXP-001: Cross-Model Judging Matrix")
parser.add_argument("--responder", type=str, required=True,
                    help="Model that answers questions and generates subquestions")
parser.add_argument("--judge", type=str, required=True,
                    help="Model that evaluates answer correctness (may differ from responder)")
parser.add_argument("--benchmark", type=str, default="mmlu", choices=["mmlu", "bbh"])
parser.add_argument("--num_subquestions", type=int, default=5)
parser.add_argument("--num_trials", type=int, default=20,
                    help="Number of concept trials (paper uses 10; increase for stability)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
responder_tag = args.responder.replace("/", "_").replace(":", "_")
judge_tag     = args.judge.replace("/", "_").replace(":", "_")
pair_tag      = f"{responder_tag}__judged_by__{judge_tag}"

index_to_category   = {0: "initial", 1: "edited_with_error"}
category_to_coherence = defaultdict(list)
overall_coherence   = []
score_per_concept   = []

log_dir = os.path.join("logs", "exp001", pair_tag, args.benchmark)
logger  = ExperimentLogger(output_dir=log_dir, model=args.responder, benchmark=args.benchmark)

print(f"\n{'='*65}")
print(f"  EXP-001: Cross-Model Judging Matrix")
print(f"  Responder : {args.responder}")
print(f"  Judge     : {args.judge}")
print(f"  Benchmark : {args.benchmark}  |  Trials: {args.num_trials}")
print(f"{'='*65}\n")

bar = tqdm(range(args.num_trials), desc="Trials")

# ---------------------------------------------------------------------------
# Main loop – mirrors main.py but judge != responder
# ---------------------------------------------------------------------------
for trial_index in bar:
    print(f"\n--- Trial {trial_index + 1}/{args.num_trials} ---")

    # ---- Step 1: Find a concept-bearing, correctly-answered question ----
    while True:
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

    # ---- Step 2: Generate subquestions (RESPONDER generates) ----
    max_attempts = 10
    subquestions = []
    while max_attempts > 0:
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
        max_attempts -= 1

    if not subquestions:
        print(f"  WARNING: Could not generate subquestions for trial {trial_index}. Skipping.")
        continue

    # ---- Step 3: For each subquestion, RESPONDER answers, JUDGE evaluates ----
    for sq_idx, subquestion in enumerate(subquestions):
        # RESPONDER answers
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

        # JUDGE evaluates both versions (key decoupling point)
        expected_answers = ["correct", "incorrect"]
        all_answers      = [extracted_answer, answer_with_error]

        for version_idx, (answer_to_grade, expected) in enumerate(
            zip(all_answers, expected_answers)
        ):
            # *** JUDGE model grades – NOT the responder ***
            judge_answer, full_judge_answer = grade_open_ended_question(
                subquestion, answer_to_grade, args.judge
            )

            valid_grading = (
                judge_answer.strip().lower()[:7]  == "correct"  or
                judge_answer.strip().lower()[:9]  == "incorrect"
            )

            if valid_grading:
                coherent    = 1 if judge_answer.strip().lower()[:len(expected)] == expected else 0
                judge_label = "correct" if judge_answer.strip().lower()[:7] == "correct" else "incorrect"
            else:
                coherent    = None
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
                category_to_coherence[category].append(coherent)
                overall_coherence.append(coherent)
                logger.log_coherence_scoring(
                    trial_index=trial_index, question=question, concept=concept,
                    coherent=coherent, expected_label=expected,
                    judge_label=judge_label, category=category
                )

                # Persist per-result JSON
                result_record = {
                    "responder":        args.responder,
                    "judge":            args.judge,
                    "original_question": question,
                    "concept":          concept,
                    "subquestion":      subquestion,
                    "answer":           answer_to_grade,
                    "judge_answer":     full_judge_answer,
                    "category":         category,
                    "expected_answer":  expected,
                    "coherent":         coherent,
                    "trial_index":      trial_index,
                    "subquestion_index": sq_idx,
                    "benchmark":        args.benchmark,
                }
                save_dir = os.path.join(
                    "results", "exp-001-cross-judge",
                    pair_tag, args.benchmark,
                    str(trial_index), category
                )
                os.makedirs(save_dir, exist_ok=True)
                safe_concept = concept.replace("/", "_").split(" ")[0]
                fname = f"{safe_concept}_{sq_idx}_{version_idx}.json"
                with open(os.path.join(save_dir, fname), "w") as fh:
                    json.dump(result_record, fh, indent=2)

    # Running Potemkin rate
    if overall_coherence:
        potemkin_rate          = 2 * (1 - np.mean(overall_coherence))
        score_per_concept.append(potemkin_rate)
        stderr = np.std(score_per_concept) / np.sqrt(len(score_per_concept))
        bar.set_description(
            f"[{responder_tag[:12]}→{judge_tag[:12]}] "
            f"Potemkin={np.mean(score_per_concept):.3f}±{stderr:.3f}"
        )

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
if score_per_concept:
    final_rate = np.mean(score_per_concept)
    final_err  = np.std(score_per_concept) / np.sqrt(len(score_per_concept))
    print(f"\n{'='*65}")
    print(f"  Potemkin rate: {final_rate:.4f}  ±  {final_err:.4f}")
    print(f"  Responder: {args.responder}")
    print(f"  Judge    : {args.judge}")
    print(f"  N trials : {len(score_per_concept)}")
    print(f"{'='*65}\n")

logger.save()
logger.save_jsonl()
print("Logs saved.")
