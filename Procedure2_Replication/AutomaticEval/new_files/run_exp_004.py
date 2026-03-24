"""
EXP-004 Batch Runner: Domain Stratification & Reliability
=========================================================
Runs 5 independent runs per (model × benchmark) combination using
exp_004_domain_main.py, enabling variance analysis across runs.

Run from inside AutomaticEval/:
    python run_exp_004.py [--skip_existing]
"""
import argparse
import os
import subprocess
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--skip_existing", action="store_true")
cli_args = parser.parse_args()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS = [
    "gpt-4o",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "claude-3-5-sonnet-20241022",
]

BENCHMARKS   = ["mmlu", "bbh"]
NUM_RUNS     = 5
NUM_TRIALS   = 30    # per run; domain filter means fewer analysed, so go higher

RESULTS_DIR  = "experiment_results_exp004"
os.makedirs(RESULTS_DIR, exist_ok=True)

total = len(MODELS) * len(BENCHMARKS) * NUM_RUNS
bar   = tqdm(total=total, desc="EXP-004 Jobs")

for model in MODELS:
    m_tag = model.replace("/", "_").replace(":", "_")
    for benchmark in BENCHMARKS:
        for run_id in range(1, NUM_RUNS + 1):
            log_path = os.path.join(
                RESULTS_DIR, f"log_{m_tag}_{benchmark}_run{run_id}.txt"
            )
            res_path = os.path.join(
                RESULTS_DIR, f"result_{m_tag}_{benchmark}_run{run_id}.txt"
            )

            if cli_args.skip_existing and os.path.isfile(res_path) and os.path.getsize(res_path) > 0:
                print(f"  [SKIP] {m_tag} {benchmark} run{run_id}")
                bar.update(1)
                continue

            print(f"\n{m_tag[:30]}  {benchmark}  run {run_id}/{NUM_RUNS}")

            cmd = [
                "python", "exp_004_domain_main.py",
                "--model",          model,
                "--benchmark",      benchmark,
                "--num_trials",     str(NUM_TRIALS),
                "--run_id",         str(run_id),
                "--num_subquestions", "5",
            ]

            with open(log_path, "w") as log_fh, open(res_path, "w") as res_fh:
                proc = subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                log_fh.write(proc.stdout)
                log_fh.write(proc.stderr)

                for line in proc.stdout.split("\n"):
                    if "Potemkin rate" in line or "Domain breakdown" in line.lower():
                        res_fh.write(line + "\n")
                        print(f"  {line.strip()}")

            bar.update(1)

bar.close()
print(f"\nEXP-004 batch complete.  Logs → {RESULTS_DIR}/")
print("Next step: python analyze_exp_004.py")
