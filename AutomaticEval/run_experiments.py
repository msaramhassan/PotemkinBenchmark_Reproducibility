import os
import subprocess
from tqdm import tqdm

# Define the arguments for the runs
models = ["Llama-3.1-8B-Instruct"]
benchmarks = ["bbh", "mmlu"]
num_runs = 5

# Create a directory to store results
results_dir = "experiment_results"
os.makedirs(results_dir, exist_ok=True)

# Total iterations for progress tracking
total_iterations = len(models) * len(benchmarks) * num_runs
progress_bar = tqdm(total=total_iterations, desc="Running experiments")

for model in models:
    for benchmark in benchmarks:
        for run in range(1, num_runs + 1):
            print(f"\n==============================")
            print(f"Starting run {run}/{num_runs} for model: {model}, benchmark: {benchmark}")
            print(f"==============================\n")

            log_file = os.path.join(results_dir, f"log_{model.replace('/', '_')}_{benchmark}_run{run}.txt")
            result_file = os.path.join(results_dir, f"result_{model.replace('/', '_')}_{benchmark}_run{run}.txt")

            command = [
                "python", "main.py",
                "--model", model,
                "--benchmark", benchmark
            ]

            with open(log_file, "w") as log, open(result_file, "w") as result:
                process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                log.write(process.stdout)
                log.write(process.stderr)

                for line in process.stdout.split("\n"):
                    if "Potemkin rate" in line:
                        result.write(line + "\n")
                        print(line)

            print(f"Run {run} completed. Logs saved to {log_file}, results saved to {result_file}.")

            progress_bar.update(1)

progress_bar.close()