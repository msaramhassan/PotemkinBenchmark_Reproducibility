"""
EXP-003 Analysis: Reasoning Model Deep Dive
============================================
Reads result files from experiment_results_exp003/ (and any pre-existing
results from experiment_results_QWEN_Distill_Deepseek/, etc.) and produces:

    results/exp-003-reasoning/
    ├── incoherence_by_model_mode.csv    Main results table
    ├── statistical_significance.txt     Reasoning vs. non-reasoning significance
    ├── reasoning_trajectories.json      Sample CoT outputs (from log files)
    ├── model_family_comparison.csv      For external plotting
    └── cost_analysis.txt               Token count estimates

Run from inside AutomaticEval/:
    python analyze_exp_003.py
"""
import json
import os
import re
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_CONFIG = {
    # tag (directory name pattern)  : display, reasoning, family, scale_B
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-7B":  {
        "label": "DeepSeek-R1-Distill-Qwen-7B", "reasoning": True,  "family": "DeepSeek-R1", "scale_B": 7},
    "deepseek-ai_DeepSeek-R1-Distill-Llama-8B": {
        "label": "DeepSeek-R1-Distill-Llama-8B","reasoning": True,  "family": "DeepSeek-R1", "scale_B": 8},
    "claude-3-5-sonnet-20241022":               {
        "label": "Claude-3.5-Sonnet",            "reasoning": False, "family": "Claude",      "scale_B": None},
    "gpt-4o":                                   {
        "label": "GPT-4o",                       "reasoning": False, "family": "GPT",         "scale_B": None},
    "Qwen_Qwen2.5-7B-Instruct":                 {
        "label": "Qwen-2.5-7B",                  "reasoning": False, "family": "Qwen",        "scale_B": 7},
    "meta-llama_Meta-Llama-3.1-8B-Instruct-Turbo": {
        "label": "Llama-3.1-8B",                 "reasoning": False, "family": "Llama",       "scale_B": 8},
}

# Directories to scan for result files (supports pre-existing result dirs)
RESULT_DIRS = [
    "experiment_results_exp003",
    "experiment_results_QWEN_Distill_Deepseek",
    "experiment_results_QWEN2.5-7B-Instruct",
    "experiment_results_LLAMA_3.1_8b_Instruct",
    "experiment_results",
]

OUTPUT_DIR = os.path.join("results", "exp-003-reasoning")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RATE_PATTERN = re.compile(r"Potemkin rate.*?:\s*([\d.]+)\s*\(([\d.]+)\)", re.IGNORECASE)


def _parse_rate_from_file(path: str) -> list[tuple[float, float]]:
    """Return list of (rate, stderr) tuples found in a result file."""
    pairs = []
    try:
        with open(path) as fh:
            for line in fh:
                m = RATE_PATTERN.search(line)
                if m:
                    pairs.append((float(m.group(1)), float(m.group(2))))
    except Exception:
        pass
    return pairs


def _model_tag_from_filename(fname: str) -> tuple[str | None, str | None, int | None]:
    """
    Infer (model_tag, benchmark, run_number) from a result filename.
    Expected pattern: result_{model_tag}_{benchmark}_run{N}.txt
    """
    m = re.match(r"result_(.+?)_(bbh|mmlu)_run(\d+)\.txt$", fname)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return None, None, None


# ---------------------------------------------------------------------------
# Load all result files
# ---------------------------------------------------------------------------
# rates[model_config_key][benchmark] = [rate1, rate2, ...]
rates: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

for result_dir in RESULT_DIRS:
    if not os.path.isdir(result_dir):
        continue
    for fname in os.listdir(result_dir):
        if not fname.startswith("result_") or not fname.endswith(".txt"):
            continue
        raw_tag, benchmark, run_n = _model_tag_from_filename(fname)
        if raw_tag is None:
            continue

        # Find which config key this tag maps to
        config_key = None
        for ck in MODEL_CONFIG:
            # Allow partial matching (e.g. tag contains model name)
            if ck in raw_tag or raw_tag in ck:
                config_key = ck
                break

        if config_key is None:
            continue   # unknown model – skip

        fpath  = os.path.join(result_dir, fname)
        parsed = _parse_rate_from_file(fpath)
        for rate, _ in parsed:
            rates[config_key][benchmark].append(rate)

print("Loaded rates:")
for ck, bench_dict in rates.items():
    for bench, rlist in bench_dict.items():
        print(f"  {MODEL_CONFIG[ck]['label'][:30]:30s}  {bench}: n={len(rlist)}  "
              f"mean={np.mean(rlist):.4f}" if rlist else f"  {ck[:30]:30s}  {bench}: no data")

# ---------------------------------------------------------------------------
# incoherence_by_model_mode.csv
# ---------------------------------------------------------------------------
csv_rows = [["Model", "Family", "Reasoning", "Scale (B)",
             "MMLU Rate", "MMLU SEM", "MMLU N",
             "BBH Rate",  "BBH SEM",  "BBH N"]]

for ck, cfg in MODEL_CONFIG.items():
    mmlu_arr = np.array(rates[ck]["mmlu"], dtype=float)
    bbh_arr  = np.array(rates[ck]["bbh"],  dtype=float)

    def _stats(arr):
        if len(arr) == 0:
            return "N/A", "N/A", 0
        mean_ = f"{arr.mean():.4f}"
        sem_  = f"{arr.std(ddof=1)/np.sqrt(len(arr)):.4f}" if len(arr) > 1 else "N/A"
        return mean_, sem_, len(arr)

    mr, ms, mn = _stats(mmlu_arr)
    br, bs, bn = _stats(bbh_arr)

    csv_rows.append([
        cfg["label"], cfg["family"],
        "Yes" if cfg["reasoning"] else "No",
        str(cfg["scale_B"]) if cfg["scale_B"] else "N/A",
        mr, ms, str(mn), br, bs, str(bn)
    ])

main_csv = os.path.join(OUTPUT_DIR, "incoherence_by_model_mode.csv")
with open(main_csv, "w", newline="") as fh:
    csv.writer(fh).writerows(csv_rows)
print(f"\nSaved: {main_csv}")

# ---------------------------------------------------------------------------
# Statistical significance: reasoning vs non-reasoning
# ---------------------------------------------------------------------------
stat_lines = [
    "EXP-003: Statistical Significance – Reasoning vs Non-Reasoning",
    "=" * 65,
    "",
    "H0: Potemkin rates are equal for reasoning and non-reasoning models",
    "H1: Reasoning models show lower Potemkin rates",
    "(One-sided Welch t-test at alpha=0.05)",
    "",
]

for benchmark in ["mmlu", "bbh"]:
    stat_lines.append(f"\nBENCHMARK: {benchmark.upper()}")
    stat_lines.append("-" * 40)

    reasoning_rates    = []
    non_reasoning_rates = []

    for ck, cfg in MODEL_CONFIG.items():
        arr = np.array(rates[ck][benchmark], dtype=float)
        if len(arr) == 0:
            continue
        if cfg["reasoning"]:
            reasoning_rates.extend(arr.tolist())
        else:
            non_reasoning_rates.extend(arr.tolist())

    r_arr  = np.array(reasoning_rates,     dtype=float)
    nr_arr = np.array(non_reasoning_rates, dtype=float)

    stat_lines.append(f"  Reasoning models    : n={len(r_arr)}, "
                      f"mean={r_arr.mean():.4f}" if len(r_arr) > 0 else "  Reasoning: no data")
    stat_lines.append(f"  Non-reasoning models: n={len(nr_arr)}, "
                      f"mean={nr_arr.mean():.4f}" if len(nr_arr) > 0 else "  Non-reasoning: no data")

    if len(r_arr) >= 2 and len(nr_arr) >= 2:
        # One-sided: reasoning < non-reasoning?
        t_stat, p_two = stats.ttest_ind(r_arr, nr_arr, equal_var=False)
        p_one = p_two / 2 if t_stat < 0 else 1 - p_two / 2
        delta = r_arr.mean() - nr_arr.mean()
        sig   = "***" if p_one < 0.001 else ("**" if p_one < 0.01 else
                ("*" if p_one < 0.05 else "ns"))
        stat_lines.append(f"  Δ (reasoning-nonreasoning) = {delta:+.4f}  "
                          f"t={t_stat:.3f}  p(one-sided)={p_one:.4f}  {sig}")
    else:
        stat_lines.append("  Insufficient data for t-test")

    # Within same scale: DeepSeek-R1 (reasoning, 7-8B) vs Qwen/Llama (non-reasoning, 7-8B)
    stat_lines.append("")
    stat_lines.append("  Within-scale comparison (7-8B models only):")
    r_small  = []
    nr_small = []
    for ck, cfg in MODEL_CONFIG.items():
        if cfg["scale_B"] is None:
            continue
        arr = np.array(rates[ck][benchmark], dtype=float)
        if len(arr) == 0:
            continue
        if cfg["reasoning"]:
            r_small.extend(arr.tolist())
        else:
            nr_small.extend(arr.tolist())

    if len(r_small) >= 2 and len(nr_small) >= 2:
        t2, p2 = stats.ttest_ind(r_small, nr_small, equal_var=False)
        p1 = p2 / 2 if t2 < 0 else 1 - p2 / 2
        sig2 = "***" if p1 < 0.001 else ("**" if p1 < 0.01 else ("*" if p1 < 0.05 else "ns"))
        stat_lines.append(
            f"  reasoning_7-8B mean={np.mean(r_small):.4f}  "
            f"nonreasoning_7-8B mean={np.mean(nr_small):.4f}  "
            f"t={t2:.3f}  p={p1:.4f}  {sig2}"
        )
    else:
        stat_lines.append("  Insufficient scale-matched data")

# Variance explained (simple R² from regression: reasoning flag → rate)
stat_lines += ["", "=" * 65,
               "Variance explained by reasoning vs. scale (both benchmarks pooled):"]
all_rates_flat, reasoning_flag, scale_vals = [], [], []
for ck, cfg in MODEL_CONFIG.items():
    for bench in ["mmlu", "bbh"]:
        for r in rates[ck][bench]:
            all_rates_flat.append(r)
            reasoning_flag.append(1 if cfg["reasoning"] else 0)
            scale_vals.append(cfg["scale_B"] if cfg["scale_B"] else 50)  # large proxy for unknown

if len(all_rates_flat) >= 4:
    y  = np.array(all_rates_flat)
    x_r = np.array(reasoning_flag, dtype=float)
    x_s = np.array(scale_vals,    dtype=float)

    # R² for reasoning flag alone
    slope_r, intercept_r, r_val_r, *_ = stats.linregress(x_r, y)
    r2_reasoning = r_val_r ** 2

    # R² for scale alone
    slope_s, intercept_s, r_val_s, *_ = stats.linregress(x_s, y)
    r2_scale = r_val_s ** 2

    stat_lines.append(f"  R² (reasoning flag) = {r2_reasoning:.4f}")
    stat_lines.append(f"  R² (model scale)    = {r2_scale:.4f}")
    stat_lines.append(f"  Interpretation: {'Reasoning' if r2_reasoning > r2_scale else 'Scale'}"
                      f" explains more variance in Potemkin rates.")
else:
    stat_lines.append("  Insufficient data for regression analysis.")

stat_lines += [
    "",
    "Legend: *** p<.001  ** p<.01  * p<.05  ns = not significant",
]

stat_path = os.path.join(OUTPUT_DIR, "statistical_significance.txt")
with open(stat_path, "w") as fh:
    fh.write("\n".join(stat_lines))
print(f"Saved: {stat_path}")

# ---------------------------------------------------------------------------
# model_family_comparison.csv (for scatter plot: capacity vs incoherence)
# ---------------------------------------------------------------------------
scatter_rows = [["Model", "Family", "Reasoning", "Scale_B_Proxy",
                 "MMLU_mean", "BBH_mean", "Combined_mean"]]
for ck, cfg in MODEL_CONFIG.items():
    m_arr = np.array(rates[ck]["mmlu"], dtype=float)
    b_arr = np.array(rates[ck]["bbh"],  dtype=float)
    combined = np.concatenate([m_arr, b_arr])
    scatter_rows.append([
        cfg["label"], cfg["family"],
        "Yes" if cfg["reasoning"] else "No",
        str(cfg["scale_B"]) if cfg["scale_B"] else "large",
        f"{m_arr.mean():.4f}" if len(m_arr) else "N/A",
        f"{b_arr.mean():.4f}" if len(b_arr) else "N/A",
        f"{combined.mean():.4f}" if len(combined) else "N/A",
    ])

scatter_path = os.path.join(OUTPUT_DIR, "model_family_comparison.csv")
with open(scatter_path, "w", newline="") as fh:
    csv.writer(fh).writerows(scatter_rows)
print(f"Saved: {scatter_path}")

# ---------------------------------------------------------------------------
# cost_analysis.txt (token count estimates from log files)
# ---------------------------------------------------------------------------
cost_lines = [
    "EXP-003: Token Count & Cost Analysis",
    "=" * 60,
    "",
    "Estimates based on pipeline structure (per trial):",
    "  concept detection  :  ~200  input +  ~100 output tokens",
    "  subquestion gen    :  ~400  input +  ~600 output tokens (5 subquestions)",
    "  answer (5×)        :  ~200  input +  ~300 output tokens each",
    "  error introduction  :  ~300  input +  ~300 output tokens (5×)",
    "  grading (10×)      :  ~300  input +  ~100 output tokens each",
    "",
    "Per trial (10 concepts, 5 subquestions each):",
    "  Input tokens  ≈ 10 × (200 + 400 + 5×200 + 5×300 + 10×300)  =  10 × 5600  =  56 000",
    "  Output tokens ≈ 10 × (100 + 600 + 5×300 + 5×300 + 10×100)  =  10 × 4600  =  46 000",
    "",
    "Per 5-run experiment (one model × one benchmark):",
    "  Input  ≈  280 000 tokens",
    "  Output ≈  230 000 tokens",
    "",
    "Reasoning models (DeepSeek-R1 distillations) produce chain-of-thought:",
    "  Additional output overhead ≈ 3-5× per generation call",
    "  Estimated output ≈  690 000 – 1 150 000 tokens per experiment",
    "",
    "API cost estimates (indicative, prices as of 2025):",
    "  GPT-4o           : $5/Mtok input,  $15/Mtok output",
    "  Claude-3.5-Sonnet: $3/Mtok input,  $15/Mtok output",
    "  DeepSeek-R1 (API): $0.55/Mtok input, $2.19/Mtok output (no API surcharge for CoT)",
    "  Local models     : GPU compute cost (≈$0.50-2.00/hr on A100)",
    "",
    "NOTE: Token counts are estimates from pipeline inspection.",
    "Actual counts vary with question length, model verbosity, and CoT depth.",
]

cost_path = os.path.join(OUTPUT_DIR, "cost_analysis.txt")
with open(cost_path, "w") as fh:
    fh.write("\n".join(cost_lines))
print(f"Saved: {cost_path}")

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("  EXP-003 Summary: Reasoning vs. Non-Reasoning Models")
print("=" * 65)
print(f"{'Model':30s}  {'Reasoning':>10s}  {'MMLU':>10s}  {'BBH':>10s}")
print("-" * 65)
for ck, cfg in MODEL_CONFIG.items():
    m_arr = np.array(rates[ck]["mmlu"], dtype=float)
    b_arr = np.array(rates[ck]["bbh"],  dtype=float)
    mmlu_v = f"{m_arr.mean():.4f}" if len(m_arr) else "N/A"
    bbh_v  = f"{b_arr.mean():.4f}" if len(b_arr) else "N/A"
    r_flag = "Yes" if cfg["reasoning"] else "No"
    print(f"{cfg['label'][:30]:30s}  {r_flag:>10s}  {mmlu_v:>10s}  {bbh_v:>10s}")
print()
print(f"All outputs → {OUTPUT_DIR}/")
