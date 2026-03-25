# PotemkinBenchmark Reproducibility

An independent replication and extension of *"Potemkin Understanding in Large Language Models"* (Mancoridis et al., 2025). This repository contains all code, data, and results for evaluating whether LLMs genuinely understand concepts or merely exhibit surface-level pattern matching.

---

## What is the Potemkin Benchmark?

The benchmark tests **conceptual understanding** by having models perform tasks that require genuine knowledge of a concept — not just recall. A model that can define a concept but cannot classify, generate, or edit examples of it is said to exhibit *Potemkin Understanding*.

**Potemkin Rate (PR):** The key metric. `PR = 2 × (1 − accuracy)`.
- `PR = 0` → perfect understanding
- `PR = 1` → chance-level performance
- `PR > 1` → systematically worse than random (anti-correlated with understanding)

---

## Repository Structure

```
PotemkinBenchmark_Reproducibility/
├── BenchmarkDataset/                          # Core dataset and evaluation utilities
│   ├── classify/questions.csv                 # 1,887 classification questions
│   ├── generate/questions.csv                 # 681 generation prompts
│   ├── edit/questions.csv                     # 1,568 editing prompts
│   ├── define/define_labels.csv               # Human-labeled definitions
│   ├── labeled_instances.csv                  # Full benchmark (4,136 instances)
│   ├── iterators.py                           # Task-specific data iterators
│   ├── potemkin_rates.py                      # Potemkin rate computation
│   ├── constants.py                           # 32 concept definitions & model mappings
│   └── helpers.py                             # Utility functions
│
├── Procedure1_Replication/                    # Four-task benchmark replication
│   ├── Procedure1_Replication_Playbook.ipynb  # Main tutorial notebook
│   ├── Procedure1_Replication_for_reasoning_models.ipynb
│   ├── gpt_judge_reeval.py                    # Re-judge results with GPT
│   └── Procedure1_Replication_Results/
│       ├── Qwen/
│       │   ├── Qwen2-VL-7B-Instruct/          # Per-task CSVs + summary
│       │   └── Qwen2.5-7B-Instruct/
│       ├── meta-llama/Llama-3.1-8B-Instruct/
│       ├── mistralai_Mistral-7B-Instruct-v0.3/
│       └── deepseek-ai_DeepSeek-R1-Distill-Qwen-14B/
│
├── Procedure2_Replication/AutomaticEval/      # Automated evaluation on MMLU & BBH
│   ├── main.py                                # Core evaluation engine
│   ├── autoeval_runner.py                     # Batch runner (all models × benchmarks)
│   ├── run_exp_001.py                         # Single experiment runner
│   ├── utils.py                               # Model loading, inference, grading
│   ├── prompts.py                             # Concept detection & subquestion prompts
│   ├── experiment_logger.py                   # Results logging
│   ├── Procedure2_Automatic_Eval_Results/
│   │   ├── aggregated_table3.csv              # Summary: 8 models × 2 benchmarks
│   │   ├── per_run_rates.csv                  # Per-run breakdown with SE
│   │   └── JSON outputs/[model]/[benchmark]/  # Full trial logs
│   ├── Procedure2_AutomaticEval_Cross-Judge_Results/
│   │   └── Cross-Judge Results/
│   │       ├── cross-judge results.txt        # All cross-judge experiment outputs
│   │       └── [responder__judged_by__judge]/ # Pairwise result directories
│   └── new_files/                             # Extended experiments
│       ├── exp_001_cross_judge_main.py        # Cross-judge evaluation
│       ├── exp_002_keystone_main.py           # Keystone (define) tests
│       └── exp_004_domain_main.py             # Domain-specific analysis
│
├── Incoherence/                               # Incoherence rate analysis
│   ├── main.py
│   ├── incoherence_rates.py
│   └── inferences/coherence_results.csv
│
├── environment.yml                            # Conda environment specification
├── paper.pdf                                  # Original Potemkin paper
└── README.md
```

---

## Dataset: 32 Concepts Across 3 Domains

| Domain | Concepts |
|--------|----------|
| **Literature** | Haiku, Shakespearean Sonnet, Analogy, Paradox, Anacoluthon, Asyndeton, Hyperbaton, Synesis, Accismus, Slant Rhyme, Enthymeme, Anapest |
| **Psychology** | Fundamental Attribution Error, Demanding Bias, Black & White Thinking, Sunk Cost Fallacy, IKEA Effect, Pseudocertainty Effect, Endowment Effect, Naive Cynicism, Normalcy Bias, Spotlight Effect, Illusory Superiority, Catastrophizing |
| **Game Theory** | Strict Dominance, Iterated Dominance, Weak Dominance, Pure Nash Equilibrium, Mixed Strategy Nash Equilibrium, Pareto Optimality, Best Response, Zero-Sum Game, Symmetric Game |

Total benchmark instances: **4,136** (Classify: 1,887 · Generate: 681 · Edit: 1,568)

---

## Procedure 1: Four-Task Benchmark Replication

Tests understanding of the 32 concepts above across four tasks. Each task is independently sufficient to evaluate understanding — high PR on any one signals shallow knowledge.

### The Four Tasks

| Task | What the Model Does | Evaluation |
|------|---------------------|------------|
| **Define** | Write a definition for a concept (keystone test) | Human / GPT judge |
| **Classify** | Label examples as positive/negative instances of a concept | Auto (label match) |
| **Generate** | Create a new example of a concept under constraints | Human / GPT judge |
| **Edit** | Modify text to introduce or remove a concept | Human / GPT judge |

### How to Run

**Option A — Interactive notebook (recommended for first run):**
```bash
jupyter notebook Procedure1_Replication/Procedure1_Replication_Playbook.ipynb
```

**Option B — Reasoning models:**
```bash
jupyter notebook Procedure1_Replication/Procedure1_Replication_for_reasoning_models.ipynb
```

**Option C — Re-judge existing results with GPT:**
```bash
export OPENAI_API_KEY=sk-...
python Procedure1_Replication/gpt_judge_reeval.py \
  --model gpt-4o \
  --results-dir Procedure1_Replication/Procedure1_Replication_Results
```

### Result File Format

Each model produces four output files in `Procedure1_Replication_Results/[model]/`:

| File | Contents |
|------|----------|
| `classify_results_[model].csv` | Per-instance: model label, true label, correct/incorrect |
| `generate_results_[model].csv` | Per-instance: generated text, judge verdict |
| `edit_results_[model].csv` | Per-instance: edited text, judge verdict |
| `procedure1_summary_[model].csv` | Aggregated: Potemkin rate, SE, accuracy, n, judge |

### Results: Procedure 1

> **Potemkin Rate** = `2 × (1 − accuracy)`. Lower is better. PR > 100 means below-chance performance.

| Model | Task | Accuracy | Potemkin Rate | SE | n | Judge |
|-------|------|----------|---------------|----|---|-------|
| **Qwen2-VL-7B-Instruct** | Classify | 53.28% | 93.43 | ±6.03 | 274 | Self |
| | Generate | 97.50% | 2.50 | ±1.75 | 80 | Self |
| | Edit | 60.65% | 39.35 | ±3.92 | 155 | Self |
| **Qwen2.5-7B-Instruct** | Classify | 58.27% | 83.46 | ±6.19 | 254 | Self |
| | Generate | 67.53% | 32.47 | ±5.34 | 77 | Self |
| | Edit | 65.36% | 34.64 | ±3.85 | 153 | Self |
| **Llama-3.1-8B-Instruct** | Classify | 48.50% | 102.99 | ±7.73 | 167 | Self |
| | Generate | 62.50% | 37.50 | ±6.47 | 56 | Self |
| | Edit | 50.59% | 49.41 | ±5.42 | 85 | Self |
| **Mistral-7B-Instruct-v0.3** | Classify | 51.52% | 96.97 | ±6.15 | 264 | Self |
| | Generate | 98.70% | 1.30 | ±1.29 | 77 | Self |
| | Edit | 95.09% | 4.91 | ±1.69 | 163 | Self |

**Key observations:**
- All models score ~50% on Classify — near random chance — despite high accuracy on Generate and Edit
- Mistral-7B shows the largest task gap: near-perfect on Generate (PR=1.3) and Edit (PR=4.9), yet near-chance on Classify (PR=97.0)
- Llama-3.1-8B is the only model with PR > 100 on Classify, meaning it performs *below* chance

---

## Procedure 2: Automatic Evaluation on MMLU & BBH

Extends the benchmark to arbitrary knowledge benchmarks (MMLU, Big Bench Hard) using an automated pipeline that does not require pre-labeled concept data.

### Pipeline Steps

1. **Sample question** from MMLU or BBH
2. **Concept detection** — LLM determines if the question relies on a specific concept
3. **Initial answer** — model answers the question
4. **Subquestion generation** — model generates 5 subquestions testing different aspects of the concept
5. **Answer editing** — model produces a deliberately incorrect version of its answer
6. **Grading** — judge model scores both correct and incorrect answers
7. **Coherence scoring** — coherence = % of verdicts that are correct (`PR = 2 × (1 − coherence)`)

### How to Run

**Single experiment:**
```bash
python Procedure2_Replication/AutomaticEval/main.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --benchmark mmlu \
  --num_subquestions 5 \
  --num_trials 10
```

**Batch runner (all configured models × benchmarks):**
```bash
python Procedure2_Replication/AutomaticEval/autoeval_runner.py
# Results saved to: Procedure2_Replication/AutomaticEval/Procedure2_Automatic_Eval_Results/
```

**Extended experiments:**
```bash
# Cross-judge evaluation (different responder/judge pairs)
python Procedure2_Replication/AutomaticEval/new_files/exp_001_cross_judge_main.py

# Keystone definition tests
python Procedure2_Replication/AutomaticEval/new_files/exp_002_keystone_main.py

# Domain-specific analysis
python Procedure2_Replication/AutomaticEval/new_files/exp_004_domain_main.py
```

### Result File Format

| File | Contents |
|------|----------|
| `aggregated_table3.csv` | Model × benchmark summary: mean PR, std, min, max, n_runs |
| `per_run_rates.csv` | Per-run rates with SE and completion status |
| `JSON outputs/[model]/[benchmark]/experiment_log_*.json` | Full trial-level logs |

Each JSON trial log records: question, subject, concept detected, original answer, edited answer, subquestions, judge verdicts, and coherence label.

### Results: Procedure 2 — Aggregated Potemkin Rates

> Lower Potemkin Rate = better conceptual understanding. Benchmark: MMLU and BBH (Big Bench Hard).

| Model | MMLU (mean ± std) | MMLU runs | BBH (mean ± std) | BBH runs |
|-------|-------------------|-----------|------------------|----------|
| **Qwen2-VL-7B** | **0.34 ± —** | 1 | 0.96 ± — | 1 |
| **Qwen2.5-7B** | 0.86 ± 0.11 | 5 | **0.78 ± 0.05** | 5 |
| **Qwen2.5-0.5B** | 1.12 ± 0.23 | 3 | — (incomplete) | — |
| **DeepSeek-R1-Distill-7B** | 0.94 ± — | 1 | 0.89 ± — | 1 |
| **LLaMA-3.1-8B** | 0.95 ± 0.11 | 5 | 1.02 ± 0.10 | 5 |
| **LLaMA-3.2-3B** | 1.02 ± 0.07 | 5 | 0.97 ± 0.10 | 5 |
| **LLaMA-3.3-70B-q4** | — (incomplete) | — | 0.77 ± — | 1 |

**Key observations:**
- Qwen2-VL-7B achieves the best MMLU score (PR=0.34), far ahead of other 7B models
- Qwen2.5-7B is consistently the most reliable across runs (lowest std on BBH: ±0.05)
- LLaMA-3.2-3B and LLaMA-3.1-8B both average PR≈1.0 (chance-level understanding)
- Qwen2.5-0.5B exceeds PR=1.0 on MMLU (worse than random)
- BBH generally yields higher PRs than MMLU for the same model, suggesting harder concept discrimination

### Results: Procedure 2 — Per-Run Breakdown

| Model | Benchmark | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 |
|-------|-----------|-------|-------|-------|-------|-------|
| Qwen2.5-7B | MMLU | 0.79 | 1.05 | 0.74 | 0.81 | 0.92 |
| Qwen2.5-7B | BBH | 0.72 | 0.80 | 0.84 | 0.73 | 0.83 |
| LLaMA-3.1-8B | MMLU | 1.11 | 1.05 | 0.90 | 0.80 | 0.90 |
| LLaMA-3.1-8B | BBH | 1.09 | 1.14 | 0.90 | 1.09 | 0.90 |
| LLaMA-3.2-3B | MMLU | 1.08 | 1.04 | 1.10 | 0.91 | 0.96 |
| LLaMA-3.2-3B | BBH | 1.06 | 0.82 | 1.01 | 0.87 | 1.08 |
| Qwen2.5-0.5B | MMLU | — | — | 1.03 | 1.44 | 0.90 |

---

## Cross-Judge Evaluation

Tests whether the Potemkin rate depends on the judge model identity, using different responder/judge pairings on MMLU.

### How to Run

```bash
python Procedure2_Replication/AutomaticEval/new_files/exp_001_cross_judge_main.py
```

### Results: Cross-Judge Potemkin Rates (IVE CHECKED THIS (THESE ARE CORRECT))

| Responder | Judge | Self-Judge | PR | SE | Coherence | Trials |
|-----------|-------|------------|----|----|-----------|--------|
| Qwen2.5-1.5B | Qwen2.5-1.5B | Yes | 1.301 | ±0.046 | 0.400 | 10/10 |
| Qwen2.5-1.5B | LLaMA-3.1-8B | No | 0.627 | ±0.030 | 0.645 | 10/10 |
| Qwen2.5-7B | Qwen2.5-7B | Yes | 0.620 | ±0.025 | — | 10/10 |
| DeepSeek-R1-7B | DeepSeek-R1-7B | Yes | 0.946 | ±0.037 | 0.546 | 9/10 |
| DeepSeek-R1-7B | LLaMA-3.1-8B | No | 1.056 | ±0.020 | 0.488 | 9/10 |

**Key observations:**
- Judge identity significantly affects the Potemkin rate
- Qwen2.5-1.5B self-judges very poorly (PR=1.30) but appears much better when judged by LLaMA-3.1-8B (PR=0.63)
- DeepSeek-R1-7B shows the reverse: better self-judge (PR=0.95) than when judged by LLaMA-3.1-8B (PR=1.06)
- Lower coherence scores (≈0.40–0.65) indicate that judges themselves struggle with calibration

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate potemkin  # or the env name defined in environment.yml
```

**Key dependencies:**

| Package | Version |
|---------|---------|
| Python | 3.11 |
| PyTorch | 2.8.0 (CUDA 12.6) |
| transformers | 4.57.6 |
| accelerate | 0.34.2 |
| pandas | 2.3.2 |
| numpy | 1.26.4 |
| openai | latest |
| anthropic | latest |

The code supports:
- Local HuggingFace models (standard + AWQ quantized)
- Vision-language models (Qwen2-VL series)
- API models (OpenAI, Anthropic, Together, Google Generative AI)
- vLLM for optimized local inference

Set required API keys as environment variables before running:
```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export TOGETHER_API_KEY=...
```

---

## Citation

If you use this codebase, please cite the original paper:

```
Mancoridis et al. (2025). Potemkin Understanding in Large Language Models.
```

---

## Quick Start

| Goal | Starting point |
|------|---------------|
| Understand the benchmark | Read `paper.pdf`, then open `Procedure1_Replication_Playbook.ipynb` |
| Replicate Procedure 1 on a new model | Edit and run `Procedure1_Replication_Playbook.ipynb` |
| Run Procedure 2 on a new model | Add model to `autoeval_runner.py`, then run it |
| Run a cross-judge experiment | Edit and run `new_files/exp_001_cross_judge_main.py` |
| Inspect raw results | Browse `Procedure2_Automatic_Eval_Results/JSON outputs/` |
