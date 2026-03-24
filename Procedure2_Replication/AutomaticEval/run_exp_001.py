"""
run_exp_001.py — Convenience launcher for the cross-model judging experiment.
==============================================================================
Runs multiple (responder, judge) pairs to build a cross-model judging matrix.

Usage:
    python run_exp_001.py                    # Run all configured pairs
    python run_exp_001.py --local-only       # Only local model pairs (no API)
    python run_exp_001.py --api-only         # Only API model pairs
    python run_exp_001.py --dry-run          # Print commands without executing

To add or change model pairs, edit the EXPERIMENT_PAIRS list below.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration — Edit these to change which experiments run
# ---------------------------------------------------------------------------

# Local model pairs (zero API cost, run on GPU)
LOCAL_PAIRS = [
    # (responder, judge, benchmark, num_trials)
    # Self-judge baselines (for comparison)
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "mmlu", 10),
    ("meta-llama/Meta-Llama-3.1-8B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct", "mmlu", 10),
    # Cross-model pairs (the actual experiment)
    ("Qwen/Qwen2.5-7B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct", "mmlu", 10),
    ("meta-llama/Meta-Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "mmlu", 10),
    # BBH variants
    ("Qwen/Qwen2.5-7B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct", "bbh", 10),
    ("meta-llama/Meta-Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "bbh", 10),
]

# API model pairs (require API keys in environment)
API_PAIRS = [
    # Cross-family: API responder, local judge
    ("gpt-4o", "Qwen/Qwen2.5-7B-Instruct", "mmlu", 10),
    ("gpt-4o", "meta-llama/Meta-Llama-3.1-8B-Instruct", "mmlu", 10),
    # Cross-family: local responder, API judge
    ("Qwen/Qwen2.5-7B-Instruct", "gpt-4o", "mmlu", 10),
    ("meta-llama/Meta-Llama-3.1-8B-Instruct", "gpt-4o", "mmlu", 10),
    # Self-judge baseline for API model
    ("gpt-4o", "gpt-4o", "mmlu", 10),
]


def run_pair(responder, judge, benchmark, num_trials, dry_run=False, seed=42):
    """Launch a single exp_001 run."""
    script = os.path.join(os.path.dirname(__file__), "new_files", "exp_001_cross_judge_main.py")
    cmd = [
        sys.executable, script,
        "--responder", responder,
        "--judge", judge,
        "--benchmark", benchmark,
        "--num_trials", str(num_trials),
        "--seed", str(seed),
    ]

    responder_short = responder.split("/")[-1][:20]
    judge_short = judge.split("/")[-1][:20]
    label = f"{responder_short} → {judge_short} ({benchmark})"

    if dry_run:
        print(f"  [DRY RUN] {label}")
        print(f"    {' '.join(cmd)}")
        return None

    print(f"\n{'=' * 70}")
    print(f"  STARTING: {label}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")

    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run cross-model judging experiment matrix")
    parser.add_argument("--local-only", action="store_true", help="Only run local model pairs")
    parser.add_argument("--api-only", action="store_true", help="Only run API model pairs")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for all runs")
    args = parser.parse_args()

    pairs = []
    if not args.api_only:
        pairs.extend(LOCAL_PAIRS)
    if not args.local_only:
        pairs.extend(API_PAIRS)

    print(f"\nEXP-001 Cross-Model Judging Matrix")
    print(f"Total pairs to run: {len(pairs)}")
    print(f"{'[DRY RUN]' if args.dry_run else ''}")

    results = {}
    for i, (responder, judge, benchmark, trials) in enumerate(pairs):
        print(f"\n--- Pair {i + 1}/{len(pairs)} ---")
        rc = run_pair(responder, judge, benchmark, trials, args.dry_run, args.seed)
        key = f"{responder} → {judge} ({benchmark})"
        results[key] = rc

    # Summary
    print(f"\n{'=' * 70}")
    print("  EXPERIMENT SUMMARY")
    print(f"{'=' * 70}")
    for key, rc in results.items():
        status = "OK" if rc == 0 else ("DRY" if rc is None else f"FAIL({rc})")
        print(f"  [{status}] {key}")


if __name__ == "__main__":
    main()
