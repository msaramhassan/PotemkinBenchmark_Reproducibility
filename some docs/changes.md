# Codebase Changes Log

This document records all code and file changes made during the reproducibility study.
It exists as a reference for report/paper writing and for audit purposes.

---

## Session 1 — File and Directory Cleanup

### Deleted: `experiment_results/` (entire directory)
**Reason:** All 20 files inside were either 0 bytes (empty) or contained only a hardcoded
path error string from failed March 2025 runs (`No such file or directory: 'results/...'`).
None contained valid experimental data. The directory will be recreated fresh on the next
successful run via `run_experiments.py`'s `results_dir = "experiment_results"` line.

### Deleted: `unsloth.ipynb`
**Reason:** 2-cell empty placeholder notebook. No code, no outputs, no purpose.

### Deleted: `potemkin_qwen2.5-7b_iter1.txt` (root level)
**Reason:** Single orphaned result line at the repository root, superseded by the organized
result files in `AutomaticEval/Procedure 2 results/Terminal Outputs/`.

### Deleted: All `__pycache__/` directories
**Directories removed:**
- `AutomaticEval/__pycache__/`
- `BenchmarkDataset/__pycache__/`
- `Incoherence/__pycache__/`
- `unsloth_compiled_cache/` (contained compiled Triton kernels, not source)

**Reason:** Compiled bytecode caches; auto-regenerated on next run. Not appropriate to
version-control.

### Deleted: `Task_Experiment/classify_questions.csv`
**Reason:** MD5-confirmed byte-for-byte identical to `BenchmarkDataset/classify/questions.csv`.
Verified with: `md5sum Task_Experiment/classify_questions.csv BenchmarkDataset/classify/questions.csv`

### Deleted: `Task_Experiment/edit_questions.csv`
**Reason:** MD5-confirmed byte-for-byte identical to `BenchmarkDataset/edit/questions.csv`.

### Deleted: `Task_Experiment/generate_questions.csv`
**Reason:** MD5-confirmed byte-for-byte identical to `BenchmarkDataset/generate/questions.csv`.

### Deleted: `Task_Experiment/definition_questions.json`
**Reason:** MD5-confirmed byte-for-byte identical to `BenchmarkDataset/define/definition_questions.json`.

### Kept: `Task_Experiment/definition_questions.csv`
**Reason:** No counterpart exists in `BenchmarkDataset/`. Unique file, retained.

---

## Session 2 — AutomaticEval Code Cleanup

### `AutomaticEval/main.py`

**Change 1 — Removed dead imports**
```python
# REMOVED:
import json
import os
from collections import defaultdict
import pandas as pd
```
**Reason:** All four were imported but never used after the per-question JSON save block
was removed. Removing them eliminates confusion about intent.

**Change 2 — Removed dead variable `category_to_coherence`**
```python
# REMOVED:
category_to_coherence = defaultdict(list)
```
**Reason:** Initialized but never populated or read anywhere in the script.

**Change 3 — Removed per-question JSON save block (lines 170–183)**
```python
# REMOVED block:
# log_dict = { "original_question": question, ... }
# save_dir = f"results/{args.model}/..."
# os.makedirs(save_dir, exist_ok=True)
# with open(os.path.join(save_dir, f"...json"), "w") as f:
#     json.dump(log_dict, f)
```
**Reason:** The `ExperimentLogger` (via `logger.save()` and `logger.save_jsonl()`) already
handles all per-question logging in a structured, versioned format. The manual JSON save
was redundant, used a different path convention, and would have recreated the deleted
`experiment_results/` directory with inconsistent content.

**Change 4 — Fixed variable shadowing in outer sampling loop**
```python
# BEFORE (shadowed outer 'answer' variable with inner loop variable):
for answer in candidate_answers:
    grade_open_ended_question(subquestion, answer, args.model)

# AFTER:
for i, candidate in enumerate(candidate_answers):
    grade_open_ended_question(subquestion, candidate, args.model)
```
**Reason:** `answer` was already used as a variable name in the `relies_on_concept` call
above. Shadowing it caused subtle readability issues and potential for bugs if the loop
order changed.

**Change 5 — Extracted repeated string expression**
```python
# BEFORE (repeated twice):
if judge_answer.strip().lower()[:7] == "correct" or ...

# AFTER:
judge_lower = judge_answer.strip().lower()
valid_grading = judge_lower[:7] == "correct" or judge_lower[:9] == "incorrect"
```
**Reason:** Avoids calling `.strip().lower()` twice on the same value; makes the validity
check explicit and reusable as a named boolean.

**Change 6 — Cleaned tqdm description string**
```python
# BEFORE:
bar.set_description(f"Potemkin rate: {np.mean(score_per_concept):.2f} ...")

# AFTER:
bar.set_description(
    f"Model: {args.model} | "
    f"Potemkin rate: {np.mean(score_per_concept):.2f} (±{score_per_concept_std_err:.2f})"
)
```
**Reason:** Including the model name in the progress bar makes multi-process runs
distinguishable at a glance.

---

### `AutomaticEval/utils.py`

**Change 1 — Fixed `bare except` in `relies_on_concept`**
```python
# BEFORE:
try:
    classification = answer.lower() == "yes"
except:
    return (True, concept) if concept is not None else (False, None)

# AFTER:
try:
    classification = answer.lower() == "yes"
except AttributeError:
    return (True, concept) if concept is not None else (False, None)
```
**Reason:** `bare except` catches `SystemExit`, `KeyboardInterrupt`, and `GeneratorExit`,
which should never be silently swallowed. The only realistic failure here is `AttributeError`
when `answer` is `None` (Gemini occasionally returns `None`). Changed to explicit catch.

**Change 2 — Fixed Gemini variable shadowing**
```python
# BEFORE (model variable overwritten):
model = genai.GenerativeModel(model)
while True:
    try:
        return model.generate_content(prompt).text

# AFTER:
gemini_model = genai.GenerativeModel(model)
while True:
    try:
        return gemini_model.generate_content(prompt).text
```
**Reason:** Overwriting the `model` parameter (a string) with a `GenerativeModel` object
made the parameter unusable downstream within the same function and is confusing to read.

**Change 3 — Added explicit error raise for unknown providers**
```python
# BEFORE (silently returned None for unknown provider):
else:
    pass  # or just fell through

# AFTER:
else:
    raise ValueError(f"Unknown provider '{provider}' for model '{model}'.")
```
**Reason:** Silent `None` return would cause a cryptic `AttributeError` or `TypeError`
downstream. Raising `ValueError` immediately makes the misconfiguration obvious.

**Change 4 — Cleaned `models_to_developer` dict formatting**
**Reason:** The dict had inconsistent indentation across entries. Normalized to 4-space
indentation throughout. No functional change.

---

### `AutomaticEval/experiment_logger.py`

**Change 1 — Removed unused typing imports**
```python
# BEFORE:
from typing import Optional, List, Any

# AFTER:
from typing import List
```
**Reason:** `Optional` and `Any` were imported but not used in any type annotation in the
file. Removed to reduce noise.

---

### `AutomaticEval/run_exp_003.py`

**Change 1 — Moved misplaced `import json` to top of file**
```python
# BEFORE: import json appeared at approximately line 70, mid-function
# AFTER: moved to top with all other imports
import argparse
import json   # ← moved here
import os
import subprocess
from tqdm import tqdm
```
**Reason:** PEP 8 requires all imports at the top of the module. A mid-file import is
confusing and can obscure the file's dependencies.

---

### `AutomaticEval/run_experiments.py`

**Change 1 — Removed stale commented model**
```python
# REMOVED:
# models = ["Mohammedxo51/llama-3.3-70b-q4"]
```
**Reason:** Orphaned commented-out line with no context. The active model list is defined
separately; this line added noise.

**Change 2 — Fixed stray space in list literal**
```python
# BEFORE:
benchmarks = [ "bbh"]

# AFTER:
benchmarks = ["bbh"]
```
**Reason:** Minor formatting fix.

---

### Syntax Verification
After all edits, ran `python3 -c "import ast; ast.parse(open('FILE').read())"` on all
9 modified Python files. Zero syntax errors confirmed.

---

## Session 2 — New Files Created

### `RESEARCH_PLAN.md` (created at project root)
**Contents:**
- Full artifact audit of `original_paper.pdf` and `my_draft_paper.pdf`
- Gap matrix: 24 paper claims mapped to reproduction status
  (REPRODUCED / PARTIAL / MISSING / CONTRADICTED)
- Prioritized experiment roadmap (P0 / P1 / P2 / SKIP tiers)
- Extracted and tabulated all raw Procedure 2 results from Terminal Output files
- 4-day day-by-day execution plan toward ACM REP submission
- Key findings framework for the paper (4 findings)
- Blockers requiring human decisions
- Paper presentation notes

---

## Session 3 — Procedure 1 Verification

### Procedure 1 Reproduction Confirmed
Ran Python comparison script computing absolute differences between `potemkin_rates.py`
output and original paper's Table 1 values for all 24 per-model-per-task entries.

**Result:** Every value is within 2× standard error. Maximum absolute deviation: 0.006.
**Status updated in gap matrix:** REPRODUCED (computational reproduction confirmed).

**Key finding added:** Procedure 1 is **computationally reproducible** (same pre-labeled data
→ same numbers) but **not independently replicable** (annotation required Upwork domain experts
with specialized knowledge; cannot be redone without equivalent human labor and domain expertise).
This distinction is itself a reproducibility finding worthy of discussion in the paper.

---

## Session 3 — Table 3 Aggregation

### `AutomaticEval/aggregate_results.py` (new file)
**Purpose:** Parses all terminal output result files from `Procedure 2 results/Terminal Outputs/`
and produces:
1. Per-run table (all raw rates with completion status)
2. Table 3: mean ± std per model per benchmark
3. CSV exports for both tables

**Outputs generated:**
- `AutomaticEval/Procedure 2 results/per_run_rates.csv`
- `AutomaticEval/Procedure 2 results/aggregated_table3.csv`

**Key findings from aggregation:**
- LLaMA-3.1-8B BBH: **1.024 ± 0.103** (5 runs) — Potemkin rate **exceeds 1.0**
- LLaMA-3.2-3B MMLU: **1.018 ± 0.072** (5 runs) — Potemkin rate **exceeds 1.0**
- Qwen2.5-0.5B MMLU: **1.123 ± 0.230** (3 runs) — exceeds 1.0 with very high variance
- Qwen2-VL-7B: MMLU 0.34 vs BBH 0.96 — anomalous gap of 0.62, likely multimodal model
  behaving differently on text-only tasks; needs more runs to confirm
- DeepSeek-R1-Distill-7B: 12/13 runs incomplete — model's verbose reasoning chains
  cause the pipeline to time out or exceed token limits before final grading completes
- Qwen2.5-0.5B BBH: all 5 runs incomplete — sub-capability-threshold failure

**Methodological note on Potemkin rate > 1.0:**
The formula `2 * (1 - mean_coherence)` has no practical upper bound when `mean_coherence < 0.5`.
This happens when the model is systematically wrong — i.e., it grades its own correct answers
as incorrect and its injected-error answers as correct. This is an unreported property of the
metric that the original paper does not acknowledge. Values > 1.0 do not indicate a bug;
they indicate a model that is anti-coherent (worse than random). This should be called out
explicitly in the paper.

---

## Session 3 — Cross-Model Judging Experiment (exp_001)

### `AutomaticEval/new_files/exp_001_cross_judge_main.py` (rewritten)
**Purpose:** Decoupled cross-model judging experiment. Responder and judge are separate
models, allowing us to test whether self-judging inflates coherence scores.

**Changes from previous version:**
- Removed redundant per-result JSON saves (logger handles this)
- Removed `category_to_coherence` defaultdict (unused beyond collection)
- Added `--seed` argument for reproducibility
- Added max sampling attempts guard (prevents infinite loop)
- Added proper working directory setup (cwd to AutomaticEval for data paths)
- Added timestamp and self-judge detection in output header
- Cleaned up tqdm description truncation
- Supports both local HuggingFace models and API models via `utils.py` routing

**Key design:** When `--responder` and `--judge` are the same model, this is a self-judge
baseline. When they differ, it's the cross-model experiment. Both must be run for comparison.

### `AutomaticEval/run_exp_001.py` (new file)
**Purpose:** Convenience launcher that runs all configured (responder, judge) pairs.
Supports `--local-only`, `--api-only`, and `--dry-run` flags.

**Default configuration (local models, zero API cost):**
- 2 self-judge baselines (Qwen2.5-7B, LLaMA-3.1-8B)
- 4 cross-model pairs (2 benchmarks × 2 directions)

**API configuration (when keys are available):**
- 5 additional pairs including GPT-4o as responder/judge

---

## Session 3 — Procedure 1 Independent Replication Notebook

### `procedure1_replication.ipynb` (new file, 13 cells)
**Purpose:** First-ever independent replication of Procedure 1 from the original paper.
Runs the exact same questions from BenchmarkDataset through local models and uses
LLM-as-judge instead of human annotators.

**Cells:**
0. Title/methodology markdown
1. Configuration (model paths, output directory, experiment parameters)
2. Imports and inference setup (reuses AutomaticEval/utils.py)
3. Prompt templates (aligned with original paper's methodology)
4. Helper functions (judge_output, safe_inference, subsample)
5. Load all questions from BenchmarkDataset CSV/JSON files
6. Task 1: Define (keystone test) — model defines concepts, judge evaluates
7. Task 2: Classify — auto-gradeable (Yes/No comparison to True Label)
8. Task 3: Generate — model generates instances, judge evaluates
9. Task 4: Edit — model edits examples, judge evaluates
10. Final Potemkin rate computation (Table 1 equivalent)
11. Comparison to original paper's Table 1
12. Save experiment metadata

**Methodology notes:**
- Uses same prompts as original paper (loaded from BenchmarkDataset CSVs)
- Keystone filtering: only concepts the model correctly defines are counted
- Classify is auto-graded; Generate/Edit use LLM-as-judge (documented deviation)
- All results saved to `Procedure1_Replication_Results/` with per-task CSVs
- Supports both local models (default) and API models (when keys available)

---

## Pending / In-Progress Experiments

| Experiment | Status | Notes |
|---|---|---|
| Procedure 1 reproduction | DONE | Verified within 2×SE for all 24 values |
| Procedure 2 Table 3 | DONE | Aggregated 5 models, partial data for 2 others |
| exp_001 cross-model judging | NOT STARTED | Code exists in `new_files/exp_001_cross_judge_main.py`; needs API decision |
| DeepSeek-R1 full runs | INCOMPLETE | Need 4 more MMLU, 5 BBH runs on local GPU |
| Qwen2-VL-7B domain effect | 1 run each | Need 2+ more runs to confirm MMLU/BBH gap |
| LLaMA-3.3-70B | 1 BBH run | MMLU incomplete; needs full run |
| Qwen2.5-0.5B BBH | 0 complete | All 5 runs failed; capability threshold issue |
