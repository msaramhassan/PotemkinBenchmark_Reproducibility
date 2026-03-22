"""
EXP-003 Batch Runner: Reasoning Model Deep Dive
================================================
Runs the full Procedure-2 pipeline (main.py) for all 8 model-mode combinations
across MMLU and BBH, 5 iterations each.

Model families:
  REASONING (chain-of-thought):
    - deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   (local HF, already partially done)
    - deepseek-ai/DeepSeek-R1-Distill-Llama-8B   (local HF)
    - claude-3-5-sonnet-20241022                 (standard Claude, note: extended thinking
                                                  not available via standard API; documented)

  NON-REASONING baselines:
    - gpt-4o                                     (OpenAI API)
    - Qwen/Qwen2.5-7B-Instruct                   (local HF)
    - meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo (Together API)
    - claude-3-5-sonnet-20241022                 (standard Claude, no CoT)

  NOTE: DeepSeek-R1 variants are reasoning models; Qwen/Llama/GPT-4o are non-reasoning.
  Claude-3.5-Sonnet runs in both categories (standard mode in both cases; extended
  thinking requires a separate API flag not available in the current pipeline).

Run from inside AutomaticEval/:
    python run_exp_003.py [--skip_existing]
"""
import argparse
import os
import subprocess
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--skip_existing", action="store_true",
                    help="Skip runs where result file already exists and is non-empty")
cli_args = parser.parse_args()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REASONING_MODELS = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "claude-3-5-sonnet-20241022",   # standard mode (closest available to extended thinking)
]

NON_REASONING_MODELS = [
    "gpt-4o",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "claude-3-5-sonnet-20241022",   # same model, standard mode (non-reasoning baseline)
]

ALL_MODELS = list(dict.fromkeys(REASONING_MODELS + NON_REASONING_MODELS))  # deduplicated

BENCHMARKS   = ["mmlu", "bbh"]
NUM_RUNS     = 5
RESULTS_DIR  = "experiment_results_exp003"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Model metadata for the analysis script
MODEL_METADATA = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B":           {"family": "DeepSeek-R1", "reasoning": True,  "scale_B": 7},
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":          {"family": "DeepSeek-R1", "reasoning": True,  "scale_B": 8},
    "claude-3-5-sonnet-20241022":                         {"family": "Claude",      "reasoning": False, "scale_B": None},
    "gpt-4o":                                             {"family": "GPT",         "reasoning": False, "scale_B": None},
    "Qwen/Qwen2.5-7B-Instruct":                           {"family": "Qwen",        "reasoning": False, "scale_B": 7},
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo":        {"family": "Llama",       "reasoning": False, "scale_B": 8},
}

import json
meta_path = os.path.join(RESULTS_DIR, "model_metadata.json")
with open(meta_path, "w") as fh:
    json.dump(MODEL_METADATA, fh, indent=2)
print(f"Model metadata saved → {meta_path}")

# ---------------------------------------------------------------------------
# Launch jobs
# ---------------------------------------------------------------------------
total = len(ALL_MODELS) * len(BENCHMARKS) * NUM_RUNS
bar   = tqdm(total=total, desc="EXP-003 Jobs")

for model in ALL_MODELS:
    for benchmark in BENCHMARKS:
        m_tag = model.replace("/", "_").replace(":", "_")
        for run in range(1, NUM_RUNS + 1):
            log_path = os.path.join(RESULTS_DIR, f"log_{m_tag}_{benchmark}_run{run}.txt")
            res_path = os.path.join(RESULTS_DIR, f"result_{m_tag}_{benchmark}_run{run}.txt")

            if cli_args.skip_existing and os.path.isfile(res_path) and os.path.getsize(res_path) > 0:
                print(f"  [SKIP] {m_tag} {benchmark} run{run}")
                bar.update(1)
                continue

            is_reasoning = MODEL_METADATA.get(model, {}).get("reasoning", False)
            mode_tag = "reasoning" if is_reasoning else "non-reasoning"

            print(f"\n[{mode_tag}] {m_tag[:30]}  {benchmark}  run {run}/{NUM_RUNS}")

            cmd = [
                "python", "main.py",
                "--model",     model,
                "--benchmark", benchmark,
            ]

            with open(log_path, "w") as log_fh, open(res_path, "w") as res_fh:
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
print(f"\nEXP-003 batch complete.  Logs → {RESULTS_DIR}/")
print("Next step: python analyze_exp_003.py")
