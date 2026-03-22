"""
EXP-002 Analysis: Keystone Sensitivity
=======================================
Reads per-concept JSON files from results/exp-002-keystone/
and produces:

    results/exp-002-keystone/
    ├── keystone_comparison.csv          Potemkin rates × 3 variants × models
    ├── model_performance_by_keystone.json  Per-model breakdowns
    ├── statistical_tests.txt            Paired t-tests A vs B, A vs C
    └── keystones_definition.txt         Exact prompts used for each keystone type

Run from inside AutomaticEval/:
    python analyze_exp_002.py
"""
import json
import os
import csv
from collections import defaultdict

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS = {
    "gpt-4o":                                          "GPT-4o",
    "meta-llama_Meta-Llama-3.1-8B-Instruct-Turbo":    "Llama-3.1-8B",
    "claude-3-5-sonnet-20241022":                      "Claude-3.5-Sonnet",
}
VARIANTS       = ["A", "B", "C"]
RESULTS_BASE   = os.path.join("results", "exp-002-keystone")
OUTPUT_DIR     = RESULTS_BASE
os.makedirs(OUTPUT_DIR, exist_ok=True)

VARIANT_LABELS = {
    "A": "Definition-as-Keystone",
    "B": "MCQ Recognition",
    "C": "Classification Keystone",
}


def load_variant_results(model_tag: str, variant: str) -> list[dict]:
    """Load all per-concept JSON files for a given model & variant."""
    base = os.path.join(RESULTS_BASE, model_tag, f"variant_{variant}")
    if not os.path.isdir(base):
        return []
    records = []
    for fname in os.listdir(base):
        if fname.endswith(".json") and fname != "summary.json":
            with open(os.path.join(base, fname)) as fh:
                records.append(json.load(fh))
    return records


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size between two 1-D arrays."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_std = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else 0.0


# ---------------------------------------------------------------------------
# Collect results per model × variant
# ---------------------------------------------------------------------------
# Structure: results[model_tag][variant] = {"rates": [...], "keystone_pass": [...]}
results = defaultdict(lambda: defaultdict(lambda: {"rates": [], "keystone_pass": []}))

for model_tag in MODELS:
    for variant in VARIANTS:
        records = load_variant_results(model_tag, variant)
        for rec in records:
            kp = rec.get("keystone_passed", False)
            pr = rec.get("potemkin_rate")
            results[model_tag][variant]["keystone_pass"].append(1 if kp else 0)
            if pr is not None:
                results[model_tag][variant]["rates"].append(pr)

# ---------------------------------------------------------------------------
# keystone_comparison.csv
# ---------------------------------------------------------------------------
csv_rows = [["Model", "Variant", "Variant Name",
             "Keystone Pass Rate", "Potemkin Rate Mean", "Potemkin Rate SEM",
             "95% CI Lower", "95% CI Upper", "N"]]

for model_tag, label in MODELS.items():
    for variant in VARIANTS:
        rates = np.array(results[model_tag][variant]["rates"], dtype=float)
        kp    = np.array(results[model_tag][variant]["keystone_pass"], dtype=float)

        kp_mean = float(kp.mean()) if len(kp) > 0 else float("nan")
        n       = len(rates)
        mean_   = float(rates.mean()) if n > 0 else float("nan")
        sem     = float(rates.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        ci_lo   = mean_ - 1.96 * sem if not np.isnan(sem) else float("nan")
        ci_hi   = mean_ + 1.96 * sem if not np.isnan(sem) else float("nan")

        csv_rows.append([
            label, variant, VARIANT_LABELS[variant],
            f"{kp_mean:.4f}", f"{mean_:.4f}", f"{sem:.4f}",
            f"{ci_lo:.4f}", f"{ci_hi:.4f}", str(n)
        ])

comp_path = os.path.join(OUTPUT_DIR, "keystone_comparison.csv")
with open(comp_path, "w", newline="") as fh:
    csv.writer(fh).writerows(csv_rows)
print(f"Saved: {comp_path}")

# ---------------------------------------------------------------------------
# model_performance_by_keystone.json
# ---------------------------------------------------------------------------
perf_output = {}
for model_tag, label in MODELS.items():
    perf_output[label] = {}
    for variant in VARIANTS:
        rates = np.array(results[model_tag][variant]["rates"], dtype=float)
        kp    = np.array(results[model_tag][variant]["keystone_pass"], dtype=float)
        n     = len(rates)
        perf_output[label][VARIANT_LABELS[variant]] = {
            "keystone_pass_rate": float(kp.mean()) if len(kp) > 0 else None,
            "potemkin_rate_mean": float(rates.mean()) if n > 0 else None,
            "potemkin_rate_sem":  float(rates.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
            "n_downstream":       n,
        }

perf_path = os.path.join(OUTPUT_DIR, "model_performance_by_keystone.json")
with open(perf_path, "w") as fh:
    json.dump(perf_output, fh, indent=2)
print(f"Saved: {perf_path}")

# ---------------------------------------------------------------------------
# statistical_tests.txt
# ---------------------------------------------------------------------------
stat_lines = [
    "EXP-002: Statistical Tests – Keystone Sensitivity",
    "=" * 60,
    "",
    "All tests: paired t-test on per-concept Potemkin rates across variants.",
    "Hypothesis: Variant A (definition) inflates Potemkin rates vs B and C.",
    "",
]

for model_tag, label in MODELS.items():
    stat_lines.append(f"\nMODEL: {label}")
    stat_lines.append("-" * 40)

    rates_A = np.array(results[model_tag]["A"]["rates"], dtype=float)
    rates_B = np.array(results[model_tag]["B"]["rates"], dtype=float)
    rates_C = np.array(results[model_tag]["C"]["rates"], dtype=float)

    for var_other, rates_other in [("B", rates_B), ("C", rates_C)]:
        # Match lengths for paired test
        n = min(len(rates_A), len(rates_other))
        if n < 2:
            stat_lines.append(f"  A vs {var_other}: insufficient data (n_A={len(rates_A)}, n_{var_other}={len(rates_other)})")
            continue

        a_arr = rates_A[:n]
        b_arr = rates_other[:n]
        t_stat, p_two = stats.ttest_rel(a_arr, b_arr)
        # one-sided: A > other?
        p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
        d     = cohen_d(a_arr, b_arr)
        delta = a_arr.mean() - b_arr.mean()
        sem   = np.sqrt((a_arr - b_arr).std(ddof=1) ** 2 / n)
        ci    = 1.96 * sem

        sig = "***" if p_one < 0.001 else ("**" if p_one < 0.01 else
              ("*" if p_one < 0.05 else "ns"))

        stat_lines.append(
            f"  A vs {var_other}: "
            f"mean_A={a_arr.mean():.4f}  mean_{var_other}={b_arr.mean():.4f}  "
            f"Δ={delta:+.4f} [95%CI ±{ci:.4f}]  "
            f"Cohen's d={d:.3f}  t={t_stat:.3f}  p(one-sided)={p_one:.4f} {sig}"
        )

stat_lines += [
    "",
    "Legend: *** p<.001  ** p<.01  * p<.05  ns = not significant",
    "",
    "INTERPRETATION:",
    "  A >> B and A >> C  → definition keystones inflate incoherence (memorisation hypothesis)",
    "  A ≈ B ≈ C          → keystone choice does not drive the effect",
    "  A >> B only        → MCQ recognition is easier than definition generation",
]

stat_path = os.path.join(OUTPUT_DIR, "statistical_tests.txt")
with open(stat_path, "w") as fh:
    fh.write("\n".join(stat_lines))
print(f"Saved: {stat_path}")

# ---------------------------------------------------------------------------
# keystones_definition.txt – document exact prompts used
# ---------------------------------------------------------------------------
prompts_doc = """EXP-002: Exact Keystone Prompts
================================

VARIANT A – Definition-as-Keystone
-----------------------------------
Prompt:
  "Define the game-theory concept '{concept}' in 1-2 sentences.
   You can reason, but end with `FINAL ANSWER:` followed by your definition."

Grading:
  Judge model evaluates whether the generated definition is correct by
  comparing against the gold definition for the concept.  The judge is the
  same as the responder (self-judging, matching the original paper's setup).

---

VARIANT B – Multiple-Choice Recognition
-----------------------------------------
Prompt:
  "Which of the following best defines '{concept}' in game theory?

   A. {option_A}
   B. {option_B}
   C. {option_C}
   D. {option_D}

   Think briefly, then end with `FINAL ANSWER:` followed by the single letter
   (A, B, C, or D) of the best answer."

Grading:
  Deterministic: model's choice letter must match the pre-specified correct letter.
  No LLM judge required for the keystone itself.

Note on distractor design:
  Each MCQ set contains one correct definition and three distractors that each
  capture a common misconception.  Distractors were hand-crafted to be plausible
  but factually wrong, to test recognition vs. rote memorisation.

---

VARIANT C – Classification Keystone
--------------------------------------
Prompt:
  "Consider the following scenario:

   {scenario}

   Does this scenario exhibit the concept of '{concept}'?
   Reason briefly, then end with `FINAL ANSWER:` followed by 'yes' or 'no'."

Grading:
  Model's yes/no classification is compared against a pre-determined ground truth
  label for each scenario (whether the scenario does or does not exhibit the concept).
  Some scenarios intentionally do NOT exhibit the concept (ground truth = no) to
  control for acquiescence bias.

---

Downstream Subquestion Pipeline (all variants, if keystone passed):
  Same as original paper Procedure 2:
  1. Generate 5 subquestions about the concept
  2. Answer each subquestion
  3. Introduce a subtle error into each answer
  4. Judge (same model) evaluates each (correct, incorrect) version
  5. Potemkin rate = 2 * (1 - mean(coherence))
"""

prompts_path = os.path.join(OUTPUT_DIR, "keystones_definition.txt")
with open(prompts_path, "w") as fh:
    fh.write(prompts_doc)
print(f"Saved: {prompts_path}")

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("  EXP-002 Summary: Potemkin rates by keystone variant")
print("=" * 65)
print(f"{'Model':22s}  {'Var A':>10s}  {'Var B':>10s}  {'Var C':>10s}")
print("-" * 60)
for model_tag, label in MODELS.items():
    row = f"{label[:22]:22s}"
    for variant in VARIANTS:
        rates = np.array(results[model_tag][variant]["rates"], dtype=float)
        v = f"{rates.mean():.4f}" if len(rates) > 0 else "N/A"
        row += f"  {v:>10s}"
    print(row)
print()
print(f"All outputs → {OUTPUT_DIR}/")
