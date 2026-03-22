# Cleanup Changelog

## File & Directory Deletions

### `experiment_results/` (entire directory)
**Reason:** All 20 files in this directory were either empty (0 bytes) or contained a single error line — a hardcoded path that no longer existed on the machine. These were failed runs from March 2025 that never produced usable output. Safe to delete entirely.

### `unsloth.ipynb`
**Reason:** Contained only 2 cells with no meaningful content. Functioned as an empty placeholder with no purpose in the pipeline.

### `potemkin_qwen2.5-7b_iter1.txt`
**Reason:** A single-line orphaned result file at the root level. No associated code reads from or writes to this path. Output was already captured properly by the logging system.

### All `__pycache__/` directories
**Files affected:** `AutomaticEval/__pycache__/`, `BenchmarkDataset/__pycache__/`, `Incoherence/__pycache__/`, `unsloth_compiled_cache/__pycache__/`
**Reason:** Auto-generated Python bytecode caches. These are regenerated automatically by Python on next run and should not be tracked.

### `Task_Experiment/classify_questions.csv`
### `Task_Experiment/edit_questions.csv`
### `Task_Experiment/generate_questions.csv`
### `Task_Experiment/definition_questions.json`
**Reason:** Verified via MD5 checksum to be byte-for-byte identical copies of the canonical files in `BenchmarkDataset/classify/questions.csv`, `BenchmarkDataset/edit/questions.csv`, `BenchmarkDataset/generate/questions.csv`, and `BenchmarkDataset/define/definition_questions.json` respectively. Duplicates with no modification. `Task_Experiment/definition_questions.csv` was kept as it had no counterpart in `BenchmarkDataset/`.

---

## Code Changes

### `AutomaticEval/main.py`

**Removed unused imports:**
- `import json` — only used by the per-question JSON save block (which was removed)
- `import os` — only used by the per-question JSON save block (which was removed)
- `import pandas as pd` — imported but never referenced anywhere in the file
- `from collections import defaultdict` — only used by `category_to_coherence`, which was also removed

**Removed dead variable `category_to_coherence`:**
- Reason: It was a `defaultdict(list)` that was populated inside the grading loop but never read, used in any calculation, or printed. The final Potemkin rate is computed solely from `overall_coherence` and `score_per_concept`. Removing it also allowed the `defaultdict` import to be dropped.

**Removed per-question JSON save block (lines ~170–183):**
- Reason: User confirmed this is redundant — the `ExperimentLogger` already captures all the same data in a structured JSON/JSONL format. The save block was duplicating logging effort by writing individual JSON files per question into a `results/` subdirectory.

**Fixed variable shadowing bug:**
- The outer sampling loop used `question, answer, subject = sample_question(...)`, but the inner grading loop also used `for i, answer in enumerate(all_answers)`, silently overwriting the gold answer. Renamed outer variable to `gold_answer` and inner loop variable to `candidate` to make both scopes clear and safe.

**Minor cleanups:**
- Extracted repeated `judge_answer.strip().lower()` calls into a single local variable `judge_lower` to avoid redundant computation.
- Tightened `bar.set_description` formatting to use `±` symbol for standard error.
- Removed stale comment `# Set up save directory` which referred to the now-deleted save logic.

---

### `AutomaticEval/utils.py`

**Fixed bare `except:` in `relies_on_concept`:**
- Reason: `except:` with no exception type catches everything, including `KeyboardInterrupt`, `SystemExit`, and `GeneratorExit`. The actual failure being guarded against is `answer.lower()` failing when `answer is None` (i.e., the regex found no match), which is an `AttributeError`. Changed to `except AttributeError:`.

**Fixed variable shadowing in Gemini branch of `generate_inference`:**
- The parameter `model` (a string model name) was being overwritten by `model = genai.GenerativeModel(model)`, shadowing the original string. Although it happened to work because the parameter wasn't needed after that point, it's a latent bug risk. Renamed the local to `gemini_model`.

**Added explicit error raise for unknown providers:**
- Previously, if `models_to_developer[model]` held an unrecognized provider string, `generate_inference` would fall through all the `elif` branches and return `None` silently, causing confusing downstream errors. Added `raise ValueError(f"Unknown provider...")` as a final fallback.

**Cleaned up `models_to_developer` dict:**
- Inconsistent indentation (mix of 2-space and 6-space) and a commented entry placed before the active entry. Reformatted to consistent 4-space indentation with the active `gpt-4o` entry first and all commented options below, with a section comment explaining their purpose.

**Moved imports to top:**
- `import re` and `import time` were scattered mid-file. Moved to the top import block with the rest.

**Added section comments:**
- Divided the file into logical sections: local model cache, API model registry, inference, question answering helpers, grading helpers, subquestion helpers, dataset sampling. Improves navigability.

---

### `AutomaticEval/experiment_logger.py`

**Removed unused imports `Optional` and `Any` from `typing`:**
- Reason: Neither `Optional` nor `Any` appear anywhere as type annotations in the file. Only `List` is used (in `log_subquestion_generation`, `log_answer_editing`, etc.). Removing them keeps the import clean.

---

### `AutomaticEval/run_exp_003.py`

**Moved `import json` to the top of the file:**
- Reason: The import was placed at line ~70, inside the configuration section between the `MODEL_METADATA` dict definition and the code that writes it to disk. Imports should always appear at the top of a module. Moved it to join `import argparse`, `import os`, `import subprocess`, `from tqdm import tqdm`.

---

### `AutomaticEval/run_experiments.py`

**Removed stale commented model line:**
- `# models = ["Mohammedxo51/llama-3.3-70b-q4"]` — an old configuration line with no useful reference context. Removed to reduce noise.

**Fixed stray space in list literal:**
- `benchmarks = [ "bbh"]` → `benchmarks = ["bbh"]`
