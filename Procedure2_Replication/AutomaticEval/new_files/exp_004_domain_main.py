"""
EXP-004: Domain Stratification & Reliability Analysis
======================================================
Modified main.py that (a) restricts benchmark sampling to questions
identifiable with a specific domain and (b) records the domain of
every concept detected.

Domains:
  1. Game Theory  – formal, auto-gradable
  2. Literary     – author-annotated, moderate reliability
  3. Psychology   – expert-annotated via Upwork, high irreproducibility risk

Domain assignment:
  After `relies_on_concept` returns a concept name, the concept is matched
  against hard-coded concept lists (drawn from BenchmarkDataset/constants.py).
  Any concept not matching a known domain is labelled "other".

Usage:
    python exp_004_domain_main.py \
        --model gpt-4o \
        --benchmark mmlu \
        --num_trials 30 \
        --run_id 1
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from tqdm import tqdm

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
# Model routing
# ---------------------------------------------------------------------------
utils.models_to_developer.update({
    "gpt-4o":                                      "openai",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": "together",
    "claude-3-5-sonnet-20241022":                   "claude",
})

# ---------------------------------------------------------------------------
# Domain concept lists (from BenchmarkDataset/constants.py)
# ---------------------------------------------------------------------------
GAME_THEORY_CONCEPTS = {
    "strict dominance", "weak dominance", "iterated dominance",
    "iterated elimination", "pure nash equilibrium", "pure strategy nash",
    "mixed nash equilibrium", "mixed strategy nash", "pareto optimality",
    "pareto optimal", "best response", "zero-sum game", "symmetric game",
    "dominant strategy", "nash equilibrium", "subgame perfect",
    "backward induction", "minimax", "maximin",
}

LITERARY_CONCEPTS = {
    "haiku", "shakespearean sonnet", "analogy", "paradox", "anacoluthon",
    "asyndeton", "hyperbaton", "synesis", "accismus", "slant rhyme",
    "enthymeme", "anapest", "alliteration", "metaphor", "simile",
    "personification", "irony", "oxymoron", "antithesis", "chiasmus",
    "anaphora", "epistrophe", "zeugma", "synecdoche", "metonymy",
    "allegory", "euphemism", "litotes", "hyperbole", "understatement",
    "onomatopoeia", "iambic pentameter", "blank verse", "free verse",
    "sonnet", "villanelle", "sestina", "ode", "elegy",
}

PSYCHOLOGY_CONCEPTS = {
    "fundamental attribution error", "black and white thinking",
    "sunk cost fallacy", "ikea effect", "pseudocertainty effect",
    "endowment effect", "naive cynicism", "normalcy bias",
    "spotlight effect", "illusory superiority", "catastrophizing",
    "confirmation bias", "anchoring bias", "availability heuristic",
    "availability bias", "dunning-kruger effect", "cognitive dissonance",
    "social desirability bias", "hindsight bias", "framing effect",
    "gambler's fallacy", "base rate neglect", "representativeness heuristic",
    "planning fallacy", "optimism bias", "negativity bias",
    "in-group bias", "out-group homogeneity", "self-serving bias",
    "just-world hypothesis", "actor-observer bias", "recency bias",
    "peak-end rule", "denomination effect", "hot-hand fallacy",
    "blind spot bias", "ostrich effect", "curse of knowledge",
}


def classify_domain(concept: str) -> str:
    """Classify a concept into one of the three domains."""
    if concept is None:
        return "other"
    lower = concept.strip().lower()
    # Check substrings (allows partial matches like "Pure Nash Equilibrium")
    if any(kw in lower for kw in GAME_THEORY_CONCEPTS):
        return "game_theory"
    if any(kw in lower for kw in LITERARY_CONCEPTS):
        return "literary"
    if any(kw in lower for kw in PSYCHOLOGY_CONCEPTS):
        return "psychology"
    return "other"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="EXP-004: Domain Reliability")
parser.add_argument("--model",          type=str, default="gpt-4o")
parser.add_argument("--benchmark",      type=str, default="mmlu", choices=["mmlu", "bbh"])
parser.add_argument("--num_trials",     type=int, default=30,
                    help="Total trials to attempt (many may be 'other' domain)")
parser.add_argument("--num_subquestions", type=int, default=5)
parser.add_argument("--run_id",         type=int, default=1,
                    help="Run identifier for variance analysis (1-5)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
model_tag = args.model.replace("/", "_").replace(":", "_")

index_to_category      = {0: "initial", 1: "edited_with_error"}
domain_coherence: dict[str, list[int]]  = defaultdict(list)
domain_concepts:  dict[str, list[str]]  = defaultdict(list)
overall_coherence: list[int]            = []
score_per_concept: list[float]          = []

log_dir = os.path.join("logs", "exp004", model_tag, args.benchmark, f"run{args.run_id}")
logger  = ExperimentLogger(output_dir=log_dir, model=args.model, benchmark=args.benchmark)

save_base = os.path.join(
    "results", "exp-004-domain-reliability",
    model_tag, args.benchmark, f"run{args.run_id}"
)
os.makedirs(save_base, exist_ok=True)

print(f"\n{'='*65}")
print(f"  EXP-004: Domain Reliability  |  Run {args.run_id}")
print(f"  Model    : {args.model}")
print(f"  Benchmark: {args.benchmark}  |  Trials: {args.num_trials}")
print(f"{'='*65}\n")

bar = tqdm(range(args.num_trials), desc=f"EXP-004 run{args.run_id}")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for trial_index in bar:
    # Find concept-bearing, correctly-answered question
    while True:
        question, gold_answer, subject = sample_question(args.benchmark)
        logger.log_sample_question(trial_index, question, subject, gold_answer)

        concept_classification, concept = relies_on_concept(question, args.model)
        logger.log_concept_detection(trial_index, question, subject,
                                     concept_classification, concept)

        if concept_classification:
            correct, full_answer = answer_and_grade_benchmark_question(
                question, args.model, gold_answer, args.benchmark == "mmlu"
            )
            logger.log_initial_answer(trial_index, question, subject,
                                      concept, full_answer, correct)
            if correct:
                break

    # Classify concept into domain
    domain = classify_domain(concept)
    domain_concepts[domain].append(concept)

    # Generate subquestions
    max_attempts = 10
    subquestions = []
    while max_attempts > 0:
        subquestions = generate_subquestions(
            question, concept, args.model, args.num_subquestions
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
        continue

    trial_coherence = []
    for sq_idx, subquestion in enumerate(subquestions):
        extracted = answer_open_ended_question(subquestion, args.model)
        logger.log_subquestion_answering(
            trial_index, question, concept, sq_idx, subquestion, extracted
        )

        _, answer_with_error = edit_to_introduce_error(subquestion, extracted, args.model)
        logger.log_answer_editing(
            trial_index, question, concept, sq_idx,
            subquestion, extracted, answer_with_error
        )

        for v_idx, (answer, expected) in enumerate(
            zip([extracted, answer_with_error], ["correct", "incorrect"])
        ):
            judge_ans, full_judge = grade_open_ended_question(subquestion, answer, args.model)
            valid = (
                judge_ans.strip().lower()[:7]  == "correct" or
                judge_ans.strip().lower()[:9]  == "incorrect"
            )
            if valid:
                coherent    = 1 if judge_ans.strip().lower()[:len(expected)] == expected else 0
                judge_label = "correct" if judge_ans.strip().lower()[:7] == "correct" else "incorrect"

                category = index_to_category[v_idx]
                logger.log_grading(
                    trial_index=trial_index, question=question, concept=concept,
                    subquestion_index=sq_idx, subquestion=subquestion,
                    model_answer=answer, judge_answer_raw=full_judge,
                    judge_label=judge_label, expected_label=expected,
                    category=category, coherent=coherent, valid=True
                )
                logger.log_coherence_scoring(
                    trial_index=trial_index, question=question, concept=concept,
                    coherent=coherent, expected_label=expected,
                    judge_label=judge_label, category=category
                )

                trial_coherence.append(coherent)
                overall_coherence.append(coherent)
                domain_coherence[domain].append(coherent)

                # Save result JSON
                result_record = {
                    "model":          args.model,
                    "run_id":         args.run_id,
                    "domain":         domain,
                    "concept":        concept,
                    "original_question": question,
                    "subquestion":    subquestion,
                    "answer":         answer,
                    "judge_answer":   full_judge,
                    "category":       category,
                    "expected_answer": expected,
                    "coherent":       coherent,
                    "trial_index":    trial_index,
                    "benchmark":      args.benchmark,
                }
                cat_dir = os.path.join(save_base, domain, category)
                os.makedirs(cat_dir, exist_ok=True)
                safe_concept = concept.replace("/", "_").split(" ")[0]
                fname = f"{safe_concept}_{trial_index}_{sq_idx}_{v_idx}.json"
                with open(os.path.join(cat_dir, fname), "w") as fh:
                    json.dump(result_record, fh, indent=2)

    if trial_coherence:
        potemkin_rate = 2 * (1 - np.mean(trial_coherence))
        score_per_concept.append(potemkin_rate)
        stderr = np.std(score_per_concept) / np.sqrt(len(score_per_concept))
        bar.set_description(
            f"run{args.run_id} Potemkin={np.mean(score_per_concept):.3f}±{stderr:.3f}"
        )

# ---------------------------------------------------------------------------
# Save run summary with per-domain breakdown
# ---------------------------------------------------------------------------
domain_summary = {}
for domain in ["game_theory", "literary", "psychology", "other"]:
    coh = np.array(domain_coherence[domain], dtype=float)
    n   = len(coh)
    domain_summary[domain] = {
        "n_observations": n,
        "mean_coherence":  float(coh.mean()) if n > 0 else None,
        "potemkin_rate":   float(2 * (1 - coh.mean())) if n > 0 else None,
        "std":             float(coh.std(ddof=1)) if n > 1 else None,
        "sem":             float(coh.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
        "concepts_seen":   domain_concepts[domain],
    }

overall_rate = float(np.mean(score_per_concept)) if score_per_concept else None
run_summary  = {
    "model":          args.model,
    "benchmark":      args.benchmark,
    "run_id":         args.run_id,
    "num_trials":     args.num_trials,
    "overall_potemkin_rate": overall_rate,
    "domains":        domain_summary,
}

summary_path = os.path.join(save_base, "run_summary.json")
with open(summary_path, "w") as fh:
    json.dump(run_summary, fh, indent=2)

logger.save()
logger.save_jsonl()

print(f"\n{'='*65}")
print(f"  Run {args.run_id} complete  |  Overall Potemkin rate: {overall_rate:.4f}" if overall_rate else f"  Run {args.run_id} complete (no trials completed)")
print(f"  Domain breakdown:")
for domain, info in domain_summary.items():
    if info["n_observations"] > 0:
        print(f"    {domain:15s}: n={info['n_observations']:3d}  "
              f"potemkin={info['potemkin_rate']:.4f}")
print(f"{'='*65}")
print(f"\nResults saved → {save_base}/")
