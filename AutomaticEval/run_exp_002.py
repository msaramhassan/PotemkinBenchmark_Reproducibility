"""
EXP-002 Batch Runner: Keystone Sensitivity
==========================================
Runs all three keystone variants (A, B, C) for each configured model.

Run from inside AutomaticEval/:
    python run_exp_002.py
"""
import os
import subprocess
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS = [
    "gpt-4o",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "claude-3-5-sonnet-20241022",
]

VARIANTS      = ["A", "B", "C"]
NUM_CONCEPTS  = 9   # all game theory concepts
RESULTS_DIR   = "experiment_results_exp002"
os.makedirs(RESULTS_DIR, exist_ok=True)

total = len(MODELS) * len(VARIANTS)
bar   = tqdm(total=total, desc="EXP-002 Jobs")

for model in MODELS:
    for variant in VARIANTS:
        m_tag    = model.replace("/", "_").replace(":", "_")
        log_path = os.path.join(RESULTS_DIR, f"log_{m_tag}_variant{variant}.txt")
        res_path = os.path.join(RESULTS_DIR, f"result_{m_tag}_variant{variant}.txt")

        cmd = [
            "python", "exp_002_keystone_main.py",
            "--model",        model,
            "--variant",      variant,
            "--num_concepts", str(NUM_CONCEPTS),
        ]

        print(f"\n[Variant {variant}]  Model: {m_tag[:30]}")

        with open(log_path, "w") as log_fh, open(res_path, "w") as res_fh:
            proc = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            log_fh.write(proc.stdout)
            log_fh.write(proc.stderr)

            for line in proc.stdout.split("\n"):
                if "Potemkin rate" in line or "Keystone pass" in line:
                    res_fh.write(line + "\n")
                    print(f"  {line.strip()}")

        bar.update(1)

bar.close()
print(f"\nEXP-002 batch complete.  Logs → {RESULTS_DIR}/")
print("Next step: python analyze_exp_002.py")
