"""
GPT-as-Judge Re-Evaluation for Procedure 1 Results
====================================================
Reads existing result CSVs from Procedure1_Replication_Results, uses GPT as
an impartial judge to re-evaluate whether model answers are correct, and
writes new CSVs with updated verdicts.

Methodology is identical to Procedure1_Replication_Playbook.ipynb:
- Same judge prompt templates for define, generate, and edit tasks
- Classify is auto-graded (model_label vs. true_label) — no judge needed
- FINAL ANSWER: tag extraction for both model outputs and judge verdicts
- Verdict: 1 (correct), -1 (incorrect), 0 (judge failed to parse)

Usage:
    export OPENAI_API_KEY=sk-...
    python gpt_judge_reeval.py [--model gpt-4o] [--results-dir PATH] [--out-dir PATH]

The script auto-discovers all define/generate/edit/classify CSVs under
--results-dir and writes re-judged outputs to --out-dir (default: a
`gpt_judged/` subdirectory next to each CSV).
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Prompt templates — copied verbatim from Procedure1_Replication_Playbook.ipynb
# ---------------------------------------------------------------------------
FINAL_TAG = "FINAL ANSWER:"


def extract_final_answer(response: str) -> str:
    """Extract text after the FINAL ANSWER: tag."""
    match = re.search(
        rf"{re.escape(FINAL_TAG)}\s*(.*)", response, re.IGNORECASE | re.DOTALL
    )
    return match.group(1).strip() if match else ""


# --- Define judge prompt ---
DEFINE_JUDGE_PROMPT_TEMPLATE = (
    "You are an impartial judge. Evaluate whether the following definition of a concept "
    "is factually correct.\n\n"
    "Concept: {concept}\n"
    "Definition: {definition}\n\n"
    "Rules:\n"
    "- Class 1 (CORRECT): The definition accurately captures the core meaning of the concept.\n"
    "- Class -1 (INCORRECT): The definition is wrong, misleading, or fundamentally misrepresents the concept.\n\n"
    f"End your response with `{FINAL_TAG}` followed by either 1 or -1.\n"
)

# --- Generate judge prompt ---
GENERATE_JUDGE_PROMPT_TEMPLATE = (
    "You are an impartial judge. Evaluate whether the generated example correctly "
    "demonstrates the given concept.\n\n"
    "Concept: {concept}\n"
    "Task prompt: {prompt}\n"
    "Model output: {output}\n\n"
    "Rules:\n"
    "- Class 1 (CORRECT): The output is a valid instance of the concept and follows the constraints.\n"
    "- Class -1 (INCORRECT): The output fails to correctly instantiate the concept or violates constraints.\n\n"
    f"End your response with `{FINAL_TAG}` followed by either 1 or -1.\n"
)

# --- Edit judge prompt ---
EDIT_JUDGE_PROMPT_TEMPLATE = (
    "You are an impartial judge. Evaluate whether the edited example correctly "
    "satisfies the editing task for the given concept.\n\n"
    "Concept: {concept}\n"
    "Edit task: {prompt}\n"
    "Model output: {output}\n\n"
    "Rules:\n"
    "- Class 1 (CORRECT): The edit correctly achieves the goal described in the task.\n"
    "- Class -1 (INCORRECT): The edit fails to achieve the goal or introduces errors.\n\n"
    f"End your response with `{FINAL_TAG}` followed by either 1 or -1.\n"
)

# ---------------------------------------------------------------------------
# GPT inference
# ---------------------------------------------------------------------------

def call_gpt(client: OpenAI, prompt: str, model: str, max_retries: int = 3) -> str:
    """Send a prompt to GPT and return the response text."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  GPT error (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(2 ** attempt)
    return ""


def parse_verdict(response: str) -> int:
    """
    Extract 1/-1 verdict from a judge response.
    Returns 1 (correct), -1 (incorrect), or 0 (parse failure).
    """
    answer = extract_final_answer(response)
    if not answer:
        answer = response  # fallback: scan full response
    clean = answer.strip()
    if "-1" in clean:
        return -1
    if "1" in clean:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Task-specific judging functions
# ---------------------------------------------------------------------------

def judge_define(client: OpenAI, model: str, df: pd.DataFrame) -> pd.DataFrame:
    """Re-judge define results using the exact DEFINE_JUDGE_PROMPT_TEMPLATE."""
    verdicts, corrects = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  judging define"):
        prompt = DEFINE_JUDGE_PROMPT_TEMPLATE.format(
            concept=row["concept"],
            definition=str(row["definition"]),
        )
        response = call_gpt(client, prompt, model)
        verdict = parse_verdict(response)
        verdicts.append(verdict)
        corrects.append(verdict == 1)

    out = df.copy()
    out["gpt_judge_verdict"] = verdicts
    out["gpt_judge_correct"] = corrects
    return out


def judge_generate(
    client: OpenAI,
    model: str,
    df: pd.DataFrame,
    task_prompts: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Re-judge generate results using GENERATE_JUDGE_PROMPT_TEMPLATE.

    task_prompts maps concept -> list of task prompts loaded from the
    BenchmarkDataset (generate/questions.csv).  When available, the first
    matching prompt for the concept is used.  If no BenchmarkDataset prompts
    are available a placeholder is used so the judge still evaluates the output.
    """
    verdicts, corrects = [], []
    prompt_iters: dict[str, int] = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  judging generate"):
        concept = str(row["concept"])

        # Cycle through available task prompts for this concept
        prompts_for_concept = task_prompts.get(concept, [])
        if prompts_for_concept:
            idx = prompt_iters.get(concept, 0)
            task_prompt = prompts_for_concept[idx % len(prompts_for_concept)]
            prompt_iters[concept] = idx + 1
        else:
            task_prompt = f"Generate an example that correctly demonstrates the concept: {concept}."

        judge_prompt = GENERATE_JUDGE_PROMPT_TEMPLATE.format(
            concept=concept,
            prompt=task_prompt,
            output=str(row["output"]),
        )
        response = call_gpt(client, judge_prompt, model)
        verdict = parse_verdict(response)
        verdicts.append(verdict)
        corrects.append(verdict == 1)

    out = df.copy()
    out["gpt_judge_verdict"] = verdicts
    out["gpt_judge_correct"] = corrects
    return out


def judge_edit(
    client: OpenAI,
    model: str,
    df: pd.DataFrame,
    task_prompts: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Re-judge edit results using EDIT_JUDGE_PROMPT_TEMPLATE.
    Same task_prompts logic as judge_generate (loaded from edit/questions.csv).
    """
    verdicts, corrects = [], []
    prompt_iters: dict[str, int] = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  judging edit"):
        concept = str(row["concept"])

        prompts_for_concept = task_prompts.get(concept, [])
        if prompts_for_concept:
            idx = prompt_iters.get(concept, 0)
            task_prompt = prompts_for_concept[idx % len(prompts_for_concept)]
            prompt_iters[concept] = idx + 1
        else:
            task_prompt = (
                f"Edit the following example so that it correctly demonstrates "
                f"the concept: {concept}."
            )

        judge_prompt = EDIT_JUDGE_PROMPT_TEMPLATE.format(
            concept=concept,
            prompt=task_prompt,
            output=str(row["output"]),
        )
        response = call_gpt(client, judge_prompt, model)
        verdict = parse_verdict(response)
        verdicts.append(verdict)
        corrects.append(verdict == 1)

    out = df.copy()
    out["gpt_judge_verdict"] = verdicts
    out["gpt_judge_correct"] = corrects
    return out


def judge_classify(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify is auto-graded (model_label vs. true_label) — no GPT judge needed.
    This function re-computes the `correct` column from the raw labels so the
    output is consistent with the other judged CSVs.
    """
    out = df.copy()
    out["gpt_judge_verdict"] = out.apply(
        lambda r: 1 if str(r["model_label"]).lower() == str(r["true_label"]).lower() else -1,
        axis=1,
    )
    out["gpt_judge_correct"] = out["gpt_judge_verdict"] == 1
    return out


# ---------------------------------------------------------------------------
# Potemkin rate computation (mirrors the playbook)
# ---------------------------------------------------------------------------

def potemkin_rate(df: pd.DataFrame, task: str) -> dict:
    """Compute potemkin rate for a task dataframe using gpt_judge_correct."""
    if task == "classify":
        valid = df[df["model_label"] != "unparseable"].copy()
        mult = 2  # scaled by 2 since chance = 50 %
    else:
        valid = df[df["gpt_judge_verdict"] != 0].copy()
        mult = 1

    if len(valid) == 0:
        return {"task": task, "n": 0, "accuracy": None, "potemkin_rate": None, "se": None}

    acc = valid["gpt_judge_correct"].mean()
    se = math.sqrt(acc * (1 - acc) / len(valid)) * mult * 100
    pr = (1 - acc) * mult * 100
    return {
        "task": task,
        "n": len(valid),
        "accuracy": round(acc * 100, 2),
        "potemkin_rate": round(pr, 2),
        "se": round(se, 2),
    }


# ---------------------------------------------------------------------------
# BenchmarkDataset loader (optional — enriches generate/edit judge prompts)
# ---------------------------------------------------------------------------

def load_task_prompts(benchmark_dir: str | None, task: str) -> dict[str, list[str]]:
    """
    Load task prompts from BenchmarkDataset/{task}/questions.csv.
    Returns {concept: [prompt, ...]} or empty dict if file not found.
    """
    if not benchmark_dir:
        return {}
    qfile = Path(benchmark_dir) / task / "questions.csv"
    if not qfile.exists():
        print(f"  [info] BenchmarkDataset {task}/questions.csv not found — using fallback prompts")
        return {}

    result: dict[str, list[str]] = {}
    with open(qfile) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if not row:
                continue
            prompt_text = row[0].strip()
            # Extract concept name from prompt text (same heuristic as playbook)
            m = re.search(r"concept\s+([A-Z][\w\s\-]+?)[\.\?]", prompt_text, re.I)
            if not m:
                m = re.search(r"concept\s+(.+?)[\.\s]", prompt_text, re.I)
            concept = m.group(1).strip() if m else "Unknown"
            result.setdefault(concept, []).append(prompt_text)
    print(f"  [info] Loaded {sum(len(v) for v in result.values())} {task} prompts "
          f"for {len(result)} concepts from BenchmarkDataset")
    return result


# ---------------------------------------------------------------------------
# CSV discovery
# ---------------------------------------------------------------------------

def find_result_csvs(results_dir: str) -> dict[str, list[Path]]:
    """
    Walk results_dir and return a dict mapping task type to list of CSV paths.
    Recognises filenames containing: define_results, classify_results,
    generate_results, edit_results.
    """
    found: dict[str, list[Path]] = {
        "define": [], "classify": [], "generate": [], "edit": []
    }
    for p in Path(results_dir).rglob("*.csv"):
        name = p.stem.lower()
        if "define_results" in name:
            found["define"].append(p)
        elif "classify_results" in name:
            found["classify"].append(p)
        elif "generate_results" in name:
            found["generate"].append(p)
        elif "edit_results" in name:
            found["edit"].append(p)
    return found


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model to use as judge (default: gpt-4o)",
    )
    parser.add_argument(
        "--results-dir",
        default=str(
            Path(__file__).parent / "Procedure1_Replication_Results"
        ),
        help="Root directory containing result CSVs (searched recursively)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Directory to write judged CSVs.  Defaults to "
            "<results-dir>/gpt_judged/"
        ),
    )
    parser.add_argument(
        "--benchmark-dir",
        default=None,
        help=(
            "Path to BenchmarkDataset directory.  When provided, the exact task "
            "prompts are used in generate/edit judge calls (matching the playbook). "
            "Auto-detected if a sibling BenchmarkDataset/ folder exists."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover CSVs and print a plan without calling GPT.",
    )
    args = parser.parse_args()

    # --- API key ---
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        sys.exit("ERROR: Set OPENAI_API_KEY environment variable before running.")
    client = OpenAI(api_key=api_key) if api_key else None

    # --- Output directory ---
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.results_dir) / "gpt_judged"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Auto-detect BenchmarkDataset ---
    benchmark_dir = args.benchmark_dir
    if not benchmark_dir:
        # Look for BenchmarkDataset as a sibling of the results dir's parent
        candidate = Path(args.results_dir).parent.parent / "BenchmarkDataset"
        if candidate.exists():
            benchmark_dir = str(candidate)
            print(f"[info] Auto-detected BenchmarkDataset at: {benchmark_dir}")
        else:
            print("[info] BenchmarkDataset not found — generate/edit will use fallback prompts")

    # --- Load BenchmarkDataset prompts ---
    gen_task_prompts = load_task_prompts(benchmark_dir, "generate")
    edit_task_prompts = load_task_prompts(benchmark_dir, "edit")

    # --- Discover CSVs ---
    csv_map = find_result_csvs(args.results_dir)
    total = sum(len(v) for v in csv_map.values())
    print(f"\nFound {total} result CSVs under: {args.results_dir}")
    for task, paths in csv_map.items():
        for p in paths:
            print(f"  [{task:8s}] {p.relative_to(args.results_dir)}")

    if args.dry_run:
        print("\nDry run — exiting without calling GPT.")
        return

    # --- Process each CSV ---
    all_summaries: list[dict] = []

    for task, paths in csv_map.items():
        for csv_path in paths:
            print(f"\n{'='*60}")
            print(f"  Task : {task.upper()}")
            print(f"  File : {csv_path.name}")
            print(f"{'='*60}")

            df = pd.read_csv(csv_path)
            model_name = df["model"].iloc[0] if "model" in df.columns else csv_path.stem

            if task == "define":
                df_judged = judge_define(client, args.model, df)

            elif task == "classify":
                df_judged = judge_classify(df)

            elif task == "generate":
                df_judged = judge_generate(client, args.model, df, gen_task_prompts)

            elif task == "edit":
                df_judged = judge_edit(client, args.model, df, edit_task_prompts)

            else:
                continue

            # Build output path mirroring the source subdirectory structure
            rel = csv_path.relative_to(args.results_dir)
            out_path = out_dir / rel.parent / f"gpt_judged_{csv_path.name}"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df_judged.to_csv(out_path, index=False)
            print(f"  Saved → {out_path.relative_to(out_dir.parent)}")

            # Compute and print potemkin rate
            stats = potemkin_rate(df_judged, task)
            stats["model"] = model_name
            stats["gpt_judge_model"] = args.model
            stats["source_csv"] = str(csv_path.name)
            all_summaries.append(stats)

            if stats["accuracy"] is not None:
                print(
                    f"  Accuracy: {stats['accuracy']:.1f}%  |  "
                    f"Potemkin rate: {stats['potemkin_rate']:.1f}% ± {stats['se']:.1f}%  |  "
                    f"N={stats['n']}"
                )

    # --- Save aggregate summary ---
    if all_summaries:
        summary_path = out_dir / "gpt_judge_summary.csv"
        pd.DataFrame(all_summaries).to_csv(summary_path, index=False)
        print(f"\nSummary written to: {summary_path}")

        print(f"\n{'='*60}")
        print("  AGGREGATE RESULTS (GPT-as-Judge)")
        print(f"{'='*60}")
        df_sum = pd.DataFrame(all_summaries)
        for _, row in df_sum.iterrows():
            pr = f"{row['potemkin_rate']:.1f}% ± {row['se']:.1f}%" if row["potemkin_rate"] is not None else "N/A"
            print(f"  {row.get('model','?'):30s}  {row['task']:10s}  PR={pr:>14s}  N={row['n']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
