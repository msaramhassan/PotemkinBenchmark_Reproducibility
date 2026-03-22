"""
aggregate_results.py
--------------------
Parses all terminal output result files from Procedure 2 runs and produces:
  1. A per-run table (all raw rates)
  2. Table 3: mean ± std per model per benchmark (for the paper)
  3. A CSV export of both tables

Usage:
    python aggregate_results.py [--base_dir PATH] [--output_dir PATH]

Defaults to reading from:
    ./Procedure 2 results/Terminal Outputs/

Outputs written to:
    ./Procedure 2 results/aggregated_table3.csv
    ./Procedure 2 results/per_run_rates.csv
"""

import argparse
import os
import re
import numpy as np
import csv
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE = os.path.join(SCRIPT_DIR, "Procedure 2 results", "Terminal Outputs")
DEFAULT_OUT  = os.path.join(SCRIPT_DIR, "Procedure 2 results")

# Human-readable short names for model IDs found in filenames
MODEL_SHORT_NAMES = {
    "meta-llama_Llama-3.1-8B-Instruct":            "LLaMA-3.1-8B",
    "meta-llama_Llama-3.2-3B-Instruct":            "LLaMA-3.2-3B",
    "Qwen_Qwen2.5-7B-Instruct":                    "Qwen2.5-7B",
    "Qwen_Qwen2.5-0.5B-Instruct":                  "Qwen2.5-0.5B",
    "Qwen_Qwen2-VL-7B-Instruct":                   "Qwen2-VL-7B",
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-7B":     "DeepSeek-R1-Distill-7B",
    "Mohammedxo51_llama-3.3-70b-q4":               "LLaMA-3.3-70B-q4",
}

RATE_PATTERN = re.compile(
    r"Potemkin rate \(lower bound\):\s*([\d.]+)\s*\(([\d.]+)\)"
)
FILE_PATTERN = re.compile(r"result_(.+)_(mmlu|bbh)_run(\d+)\.txt")


def parse_all(base_dir: str):
    """Walk all result dirs and return list of record dicts."""
    records = []
    for dir_name in sorted(os.listdir(base_dir)):
        dir_path = os.path.join(base_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue
        for fname in sorted(os.listdir(dir_path)):
            m = FILE_PATTERN.match(fname)
            if not m:
                continue
            model_id, benchmark, run = m.group(1), m.group(2), int(m.group(3))
            fpath = os.path.join(dir_path, fname)
            with open(fpath) as f:
                content = f.read()
            matches = RATE_PATTERN.findall(content)
            if matches:
                rate = float(matches[-1][0])
                se   = float(matches[-1][1])
                records.append({
                    "model_id":  model_id,
                    "model":     MODEL_SHORT_NAMES.get(model_id, model_id),
                    "benchmark": benchmark,
                    "run":       run,
                    "rate":      rate,
                    "se":        se,
                    "status":    "complete",
                    "dir":       dir_name,
                })
            else:
                records.append({
                    "model_id":  model_id,
                    "model":     MODEL_SHORT_NAMES.get(model_id, model_id),
                    "benchmark": benchmark,
                    "run":       run,
                    "rate":      None,
                    "se":        None,
                    "status":    "incomplete",
                    "dir":       dir_name,
                })
    return records


def aggregate(records):
    """Aggregate complete runs: mean ± std per (model_id, benchmark)."""
    buckets = defaultdict(list)
    for r in records:
        if r["status"] == "complete":
            buckets[(r["model_id"], r["benchmark"])].append(r["rate"])

    rows = []
    for (model_id, benchmark), rates in sorted(buckets.items()):
        arr = np.array(rates)
        rows.append({
            "model_id":  model_id,
            "model":     MODEL_SHORT_NAMES.get(model_id, model_id),
            "benchmark": benchmark,
            "n_runs":    len(arr),
            "mean":      float(np.mean(arr)),
            "std":       float(np.std(arr)),
            "min":       float(np.min(arr)),
            "max":       float(np.max(arr)),
            "rates":     rates,
        })
    return rows


def print_table(agg_rows, records):
    """Print per-run table and aggregated Table 3 to stdout."""
    print("\n" + "=" * 80)
    print("TABLE: Per-run Potemkin Rates")
    print("=" * 80)
    header = f"{'Model':<30} {'Bench':<6} {'Run':<4} {'Rate':<8} {'SE':<8} {'Status'}"
    print(header)
    print("-" * 80)
    for r in records:
        rate_str = f"{r['rate']:.4f}" if r["rate"] is not None else "---"
        se_str   = f"{r['se']:.4f}"   if r["se"]   is not None else "---"
        print(f"{r['model']:<30} {r['benchmark']:<6} {r['run']:<4} {rate_str:<8} {se_str:<8} {r['status']}")

    print("\n" + "=" * 80)
    print("TABLE 3: Aggregated Potemkin Rates (mean ± std across runs)")
    print("=" * 80)
    print(f"{'Model':<30} {'Bench':<6} {'N':<4} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max'}")
    print("-" * 80)
    for r in agg_rows:
        print(
            f"{r['model']:<30} {r['benchmark']:<6} {r['n_runs']:<4} "
            f"{r['mean']:<8.4f} {r['std']:<8.4f} {r['min']:<8.4f} {r['max']:.4f}"
        )

    # Also print as LaTeX-style table for copy-paste into paper
    print("\n" + "=" * 80)
    print("TABLE 3 (LaTeX-friendly, grouped by model)")
    print("=" * 80)
    models_seen = {}
    for r in agg_rows:
        models_seen.setdefault(r["model"], {})[r["benchmark"]] = r
    for model, benches in models_seen.items():
        mmlu = benches.get("mmlu")
        bbh  = benches.get("bbh")
        mmlu_str = f"{mmlu['mean']:.2f} ± {mmlu['std']:.2f} (n={mmlu['n_runs']})" if mmlu else "—"
        bbh_str  = f"{bbh['mean']:.2f} ± {bbh['std']:.2f} (n={bbh['n_runs']})"  if bbh  else "—"
        print(f"  {model:<30}  MMLU: {mmlu_str:<30}  BBH: {bbh_str}")

    # Highlight notable findings
    print("\n" + "=" * 80)
    print("NOTABLE FINDINGS")
    print("=" * 80)
    for r in agg_rows:
        if r["mean"] > 1.0:
            print(f"  *** Potemkin rate > 1.0: {r['model']} on {r['benchmark'].upper()} → {r['mean']:.4f} ± {r['std']:.4f}")
        if r["std"] > 0.15:
            print(f"  *** High variance (std > 0.15): {r['model']} on {r['benchmark'].upper()} → std = {r['std']:.4f}")
    incomplete = [(r["model"], r["benchmark"]) for r in records if r["status"] == "incomplete"]
    if incomplete:
        print(f"\n  INCOMPLETE RUNS ({len(incomplete)} total):")
        for model, bench in sorted(set(incomplete)):
            count = sum(1 for x in incomplete if x[0] == model and x[1] == bench)
            print(f"    {model} / {bench.upper()}: {count} incomplete run(s)")


def save_csvs(agg_rows, records, out_dir):
    """Save per-run and aggregated tables to CSV."""
    os.makedirs(out_dir, exist_ok=True)

    # Per-run CSV
    per_run_path = os.path.join(out_dir, "per_run_rates.csv")
    with open(per_run_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "model_id", "benchmark", "run", "rate", "se", "status"])
        writer.writeheader()
        for r in records:
            writer.writerow({k: r[k] for k in ["model", "model_id", "benchmark", "run", "rate", "se", "status"]})
    print(f"\nPer-run CSV saved to: {per_run_path}")

    # Aggregated Table 3 CSV
    agg_path = os.path.join(out_dir, "aggregated_table3.csv")
    with open(agg_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "model_id", "benchmark", "n_runs", "mean", "std", "min", "max"])
        writer.writeheader()
        for r in agg_rows:
            writer.writerow({k: r[k] for k in ["model", "model_id", "benchmark", "n_runs", "mean", "std", "min", "max"]})
    print(f"Aggregated Table 3 CSV saved to: {agg_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate Procedure 2 Potemkin rate results.")
    parser.add_argument("--base_dir", default=DEFAULT_BASE,
                        help="Path to 'Terminal Outputs' directory")
    parser.add_argument("--output_dir", default=DEFAULT_OUT,
                        help="Directory to write CSV outputs")
    parser.add_argument("--no_csv", action="store_true",
                        help="Skip CSV export")
    args = parser.parse_args()

    if not os.path.isdir(args.base_dir):
        raise FileNotFoundError(f"Base directory not found: {args.base_dir}")

    records  = parse_all(args.base_dir)
    agg_rows = aggregate(records)

    print_table(agg_rows, records)

    if not args.no_csv:
        save_csvs(agg_rows, records, args.output_dir)


if __name__ == "__main__":
    main()
