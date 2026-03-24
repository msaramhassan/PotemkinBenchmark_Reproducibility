"""
EXP-001 Analysis: Cross-Model Judging Matrix
============================================
Reads all per-result JSON files from results/exp-001-cross-judge/
and produces:
    results/exp-001-cross-judge/
    ├── judge_matrix.csv           3×3 incoherence scores
    ├── disagreement_rates.csv     % where judge differs from self-judging
    ├── per_concept_analysis.json  Breakdown by concept
    └── statistical_summary.txt    Significance tests (p-values, 95% CI)

Run from inside AutomaticEval/:
    python analyze_exp_001.py
"""
import json
import os
import re
import csv
from collections import defaultdict
from itertools import product

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration – must match run_exp_001.py
# ---------------------------------------------------------------------------
MODELS = {
    "gpt-4o":                                          "GPT-4o",
    "meta-llama_Meta-Llama-3.1-8B-Instruct-Turbo":    "Llama-3.1-8B",
    "claude-3-5-sonnet-20241022":                      "Claude-3.5-Sonnet",
}

MODEL_TAGS = list(MODELS.keys())   # directory tag names
MODEL_LABELS = list(MODELS.values())

RESULTS_BASE = os.path.join("results", "exp-001-cross-judge")
OUTPUT_DIR   = RESULTS_BASE
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _tag(model_str: str) -> str:
    """Normalise a model string to the directory tag used when saving results."""
    return model_str.replace("/", "_").replace(":", "_")


def load_results(responder_tag: str, judge_tag: str, benchmark: str = "mmlu") -> list[dict]:
    """Load all JSON result files for a given (responder, judge) pair."""
    pair_dir = os.path.join(
        RESULTS_BASE,
        f"{responder_tag}__judged_by__{judge_tag}",
        benchmark
    )
    records = []
    if not os.path.isdir(pair_dir):
        return records

    for trial_dir in os.listdir(pair_dir):
        trial_path = os.path.join(pair_dir, trial_dir)
        if not os.path.isdir(trial_path):
            continue
        for cat_dir in os.listdir(trial_path):
            cat_path = os.path.join(trial_path, cat_dir)
            if not os.path.isdir(cat_path):
                continue
            for fname in os.listdir(cat_path):
                if not fname.endswith(".json"):
                    continue
                fpath = os.path.join(cat_path, fname)
                try:
                    with open(fpath) as fh:
                        records.append(json.load(fh))
                except Exception:
                    pass
    return records


def incoherence_score(records: list[dict]) -> tuple[float, float]:
    """Return (incoherence_score, 95% CI half-width) from a list of records."""
    coherence_vals = [r["coherent"] for r in records if r.get("coherent") is not None]
    if not coherence_vals:
        return float("nan"), float("nan")
    arr = np.array(coherence_vals, dtype=float)
    mean_coh = arr.mean()
    incoherence = 2 * (1 - mean_coh)   # same formula as main.py
    # 95% CI via normal approximation of mean coherence, then propagated
    se = arr.std(ddof=1) / np.sqrt(len(arr))
    ci_half = 1.96 * se * 2   # multiply by 2 because incoherence = 2*(1-mean)
    return incoherence, ci_half


def disagreement_rate(cross_records: list[dict], self_records: list[dict]) -> tuple[float, float]:
    """
    Fraction of (concept, subquestion) pairs where the cross-judge verdict
    differs from the self-judge verdict.  Returns (rate, 95% CI half-width).

    Matching is done by (concept, subquestion text, category).
    """
    def _key(r):
        return (r.get("concept", ""), r.get("subquestion", "")[:80], r.get("category", ""))

    self_lookup = {_key(r): r.get("coherent") for r in self_records}
    disagree, total = 0, 0
    for r in cross_records:
        k = _key(r)
        if k in self_lookup and r.get("coherent") is not None and self_lookup[k] is not None:
            if r["coherent"] != self_lookup[k]:
                disagree += 1
            total += 1

    if total == 0:
        return float("nan"), float("nan")
    rate = disagree / total
    ci = 1.96 * np.sqrt(rate * (1 - rate) / total)
    return rate, ci


# ---------------------------------------------------------------------------
# Build the 3×3 matrices
# ---------------------------------------------------------------------------
incoherence_matrix    = np.full((3, 3), np.nan)
ci_matrix             = np.full((3, 3), np.nan)
disagreement_matrix   = np.full((3, 3), np.nan)
disagreement_ci_mat   = np.full((3, 3), np.nan)
per_concept_records   = defaultdict(lambda: defaultdict(list))  # [pair][concept]

print("Loading results …")
for r_idx, r_tag in enumerate(MODEL_TAGS):
    for j_idx, j_tag in enumerate(MODEL_TAGS):
        records = load_results(r_tag, j_tag)
        incoherence_matrix[r_idx, j_idx], ci_matrix[r_idx, j_idx] = incoherence_score(records)

        # Collect per-concept
        for rec in records:
            concept = rec.get("concept", "Unknown")
            coh     = rec.get("coherent")
            if coh is not None:
                per_concept_records[(r_tag, j_tag)][concept].append(coh)

        n = sum(1 for r in records if r.get("coherent") is not None)
        label = MODELS[r_tag] if r_tag in MODELS else r_tag[:20]
        jlabel = MODELS[j_tag] if j_tag in MODELS else j_tag[:20]
        print(f"  {label[:18]:18s} → {jlabel[:18]:18s} : "
              f"incoherence={incoherence_matrix[r_idx, j_idx]:.4f}  n={n}")

print()

# ---------------------------------------------------------------------------
# Compute disagreement rates (cross vs self-judging)
# ---------------------------------------------------------------------------
for r_idx, r_tag in enumerate(MODEL_TAGS):
    self_records = load_results(r_tag, r_tag)
    for j_idx, j_tag in enumerate(MODEL_TAGS):
        if r_idx == j_idx:
            disagreement_matrix[r_idx, j_idx]  = 0.0
            disagreement_ci_mat[r_idx, j_idx]  = 0.0
        else:
            cross_records = load_results(r_tag, j_tag)
            dr, ci = disagreement_rate(cross_records, self_records)
            disagreement_matrix[r_idx, j_idx]  = dr
            disagreement_ci_mat[r_idx, j_idx]  = ci

# ---------------------------------------------------------------------------
# Save judge_matrix.csv
# ---------------------------------------------------------------------------
matrix_path = os.path.join(OUTPUT_DIR, "judge_matrix.csv")
with open(matrix_path, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["Responder \\ Judge"] + MODEL_LABELS)
    for r_idx, r_tag in enumerate(MODEL_TAGS):
        row_vals = []
        for j_idx in range(3):
            inc  = incoherence_matrix[r_idx, j_idx]
            ci   = ci_matrix[r_idx, j_idx]
            cell = f"{inc:.4f} ± {ci:.4f}" if not np.isnan(inc) else "N/A"
            row_vals.append(cell)
        writer.writerow([MODEL_LABELS[r_idx]] + row_vals)
print(f"Saved: {matrix_path}")

# ---------------------------------------------------------------------------
# Save disagreement_rates.csv
# ---------------------------------------------------------------------------
disagree_path = os.path.join(OUTPUT_DIR, "disagreement_rates.csv")
with open(disagree_path, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["Responder \\ Judge (disagreement with self-judge)"] + MODEL_LABELS)
    for r_idx, r_tag in enumerate(MODEL_TAGS):
        row_vals = []
        for j_idx in range(3):
            dr = disagreement_matrix[r_idx, j_idx]
            ci = disagreement_ci_mat[r_idx, j_idx]
            cell = f"{dr:.4f} ± {ci:.4f}" if not np.isnan(dr) else "N/A"
            row_vals.append(cell)
        writer.writerow([MODEL_LABELS[r_idx]] + row_vals)
print(f"Saved: {disagree_path}")

# ---------------------------------------------------------------------------
# Save per_concept_analysis.json
# ---------------------------------------------------------------------------
concept_output = {}
for (r_tag, j_tag), concept_dict in per_concept_records.items():
    pair_key = f"{MODELS.get(r_tag, r_tag)} → {MODELS.get(j_tag, j_tag)}"
    concept_output[pair_key] = {}
    for concept, coh_list in concept_dict.items():
        arr = np.array(coh_list, dtype=float)
        concept_output[pair_key][concept] = {
            "n":            len(arr),
            "mean_coherence": float(arr.mean()),
            "incoherence":  float(2 * (1 - arr.mean())),
            "std":          float(arr.std(ddof=1)) if len(arr) > 1 else None,
        }

concept_path = os.path.join(OUTPUT_DIR, "per_concept_analysis.json")
with open(concept_path, "w") as fh:
    json.dump(concept_output, fh, indent=2)
print(f"Saved: {concept_path}")

# ---------------------------------------------------------------------------
# Statistical summary: t-tests comparing self-judging vs cross-judging
# ---------------------------------------------------------------------------
stat_lines = [
    "EXP-001: Statistical Summary",
    "=" * 60,
    "",
    "Hypothesis: Cross-judge incoherence > Self-judge incoherence",
    "(One-sided paired t-tests at alpha=0.05)",
    "",
]

for r_idx, r_tag in enumerate(MODEL_TAGS):
    self_records = load_results(r_tag, r_tag)
    self_coh = [rec["coherent"] for rec in self_records if rec.get("coherent") is not None]

    stat_lines.append(f"\nRESPONDER: {MODELS.get(r_tag, r_tag)}")
    stat_lines.append("-" * 40)

    for j_idx, j_tag in enumerate(MODEL_TAGS):
        if r_idx == j_idx:
            continue
        cross_records = load_results(r_tag, j_tag)
        cross_coh = [rec["coherent"] for rec in cross_records if rec.get("coherent") is not None]

        if len(self_coh) < 2 or len(cross_coh) < 2:
            stat_lines.append(
                f"  Judge={MODELS.get(j_tag,j_tag)}: insufficient data (n_self={len(self_coh)}, n_cross={len(cross_coh)})"
            )
            continue

        # Independent-samples t-test: does cross-judging yield higher incoherence?
        # incoherence = 2*(1-coherence), so lower coherence = higher incoherence
        self_arr  = np.array(self_coh,  dtype=float)
        cross_arr = np.array(cross_coh, dtype=float)

        # One-sided: cross < self_coh → cross-judge is harder (more incoherent)
        t_stat, p_two = stats.ttest_ind(cross_arr, self_arr, equal_var=False)
        p_one = p_two / 2 if t_stat < 0 else 1 - p_two / 2

        self_inc  = 2 * (1 - self_arr.mean())
        cross_inc = 2 * (1 - cross_arr.mean())
        delta_inc = cross_inc - self_inc

        # 95% CI on delta
        pooled_se = np.sqrt(
            self_arr.std(ddof=1)**2 / len(self_arr) +
            cross_arr.std(ddof=1)**2 / len(cross_arr)
        ) * 2   # factor 2 from incoherence formula
        ci = 1.96 * pooled_se

        sig = "***" if p_one < 0.001 else ("**" if p_one < 0.01 else
              ("*" if p_one < 0.05 else "ns"))

        stat_lines.append(
            f"  Judge={MODELS.get(j_tag,j_tag)[:18]:18s}: "
            f"self_inc={self_inc:.4f}  cross_inc={cross_inc:.4f}  "
            f"Δ={delta_inc:+.4f} [{-ci:+.4f},{+ci:+.4f}]  "
            f"t={t_stat:.3f}  p(one-sided)={p_one:.4f} {sig}"
        )

stat_lines += [
    "",
    "Legend: *** p<.001  ** p<.01  * p<.05  ns = not significant",
    "",
    "INTERPRETATION GUIDE:",
    "  Δ > 0 and significant → cross-judging reveals MORE incoherence",
    "  → supports hypothesis that self-judging masks misunderstandings",
    "  Δ ≈ 0 → judge identity does not matter",
]

stat_path = os.path.join(OUTPUT_DIR, "statistical_summary.txt")
with open(stat_path, "w") as fh:
    fh.write("\n".join(stat_lines))
print(f"Saved: {stat_path}")

# ---------------------------------------------------------------------------
# Pretty-print the 3×3 matrix to console
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("  3×3 Incoherence Matrix (Responder → rows, Judge → cols)")
print("=" * 65)
header = f"{'':22s}" + "".join(f"{lbl[:18]:>20s}" for lbl in MODEL_LABELS)
print(header)
for r_idx, lbl in enumerate(MODEL_LABELS):
    row = f"{lbl[:20]:22s}"
    for j_idx in range(3):
        v = incoherence_matrix[r_idx, j_idx]
        row += f"{'N/A':>20s}" if np.isnan(v) else f"{v:>20.4f}"
    print(row)
print()
print("Diagonal = self-judging (baseline)")
print("Off-diagonal = cross-judging")
print(f"\nAll outputs → {OUTPUT_DIR}/")
