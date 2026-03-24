"""
EXP-001 Batch Runner: Cross-Model Judging Matrix
=================================================
Runs all 9 responder×judge combinations (3×3) plus self-judging baselines
for the three candidate models.

Output structure:
    experiment_results_exp001/
    ├── log_{responder}__judged_by__{judge}_{benchmark}_run{n}.txt
    └── result_{responder}__judged_by__{judge}_{benchmark}_run{n}.txt

Run from inside AutomaticEval/:
    python run_exp_001.py
"""
import os
import subprocess
from itertools import product
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESPONDERS = [
    "gpt-4o",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "claude-3-5-sonnet-20241022",
]

JUDGES = [
    "gpt-4o",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "claude-3-5-sonnet-20241022",
]

# MMLU subset of 200 questions across 3 domains as per spec.
# Each run processes num_trials=20 questions; 10 runs → ~200 total.
BENCHMARKS   = ["mmlu"]
NUM_RUNS     = 5       # 5 runs × 20 trials = 100 evaluations per pair
NUM_TRIALS   = 20      # per run

RESULTS_DIR = "experiment_results_exp001"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Build list of (responder, judge) pairs
# Self-judging pairs (same model) serve as baseline
# ---------------------------------------------------------------------------
all_pairs = list(product(RESPONDERS, JUDGES))
total_iterations = len(all_pairs) * len(BENCHMARKS) * NUM_RUNS

print(f"EXP-001: Total jobs = {total_iterations}  "
      f"({len(all_pairs)} pairs × {len(BENCHMARKS)} benchmarks × {NUM_RUNS} runs)")

bar = tqdm(total=total_iterations, desc="EXP-001 Jobs")

for responder, judge in all_pairs:
    for benchmark in BENCHMARKS:
        r_tag = responder.replace("/", "_").replace(":", "_")
        j_tag = judge.replace("/", "_").replace(":", "_")
        pair_tag = f"{r_tag}__judged_by__{j_tag}"
        judge_type = "self" if responder == judge else "cross"

        for run in range(1, NUM_RUNS + 1):
            log_file    = os.path.join(
                RESULTS_DIR,
                f"log_{pair_tag}_{benchmark}_run{run}.txt"
            )
            result_file = os.path.join(
                RESULTS_DIR,
                f"result_{pair_tag}_{benchmark}_run{run}.txt"
            )

            cmd = [
                "python", "exp_001_cross_judge_main.py",
                "--responder",       responder,
                "--judge",           judge,
                "--benchmark",       benchmark,
                "--num_trials",      str(NUM_TRIALS),
                "--num_subquestions", "5",
            ]

            print(f"\n[{judge_type.upper()}] run {run}/{NUM_RUNS}: "
                  f"{r_tag[:20]} → {j_tag[:20]}  [{benchmark}]")

            with open(log_file, "w") as log_fh, open(result_file, "w") as res_fh:
                proc = subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                log_fh.write(proc.stdout)
                log_fh.write(proc.stderr)

                for line in proc.stdout.split("\n"):
                    if "Potemkin rate" in line:
                        res_fh.write(line + "\n")
                        print(f"  {line.strip()}")

            bar.update(1)

bar.close()
print("\nEXP-001 batch runs complete.")
print(f"Logs:    {RESULTS_DIR}/")
print("Next step: python analyze_exp_001.py")
