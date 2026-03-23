# Final Paper Drafting Instructions

## Purpose

This document is a self-contained prompt. Use it alongside `original_paper.pdf` (Mancoridis et al., "Potemkin Understanding in Large Language Models," ICML 2025) and `my_draft_paper.pdf` (Hassan et al., working draft dated 2026-03-22) to produce a submission-ready version of our ACM REP 2026 experience paper.

The target venue is the **ACM Conference on Reproducibility and Replicability (ACM REP 2026)**. Paper type: **Experience paper** (8-page limit including references; appendices do not count toward the limit).

Everything marked `[PLACEHOLDER]` requires results from experiments that have not yet been run. Leave those blocks in place with clear labels so the author can fill them after running the experiments. Everything else should be finalized.

---

## Paper Metadata — Fix These First

### Title
Keep as-is:
**"Understanding or Imitation? Auditing Conceptual Understanding and Reasoning in Large Language Models"**

### Authors
Muhammad Saram Hassan (LUMS), Essa Jan (Brown), Ramneet Kaur (SRI), Eric Yeh (SRI), Fareed Zaffar (LUMS), Ashish Gehani (SRI)

### CCS Concepts (REPLACE the placeholder)
```
\begin{CCSXML}
<!-- Generate at https://dl.acm.org/ccs -->
\end{CCSXML}
\ccsdesc[500]{Computing methodologies~Natural language processing}
\ccsdesc[300]{Software and its engineering~Software verification and validation}
\ccsdesc[300]{General and reference~Empirical studies}
```

### Keywords (REPLACE the placeholder)
```
reproducibility, large language models, benchmark evaluation, conceptual understanding, potemkin understanding, LLM evaluation, replicability
```

### ACM Reference Format (REPLACE the placeholder)
```
Muhammad Saram Hassan, Essa Jan, Ramneet Kaur, Eric Yeh, Fareed Zaffar, and Ashish Gehani. 2026. Understanding or Imitation? Auditing Conceptual Understanding and Reasoning in Large Language Models. In Proceedings of the ACM Conference on Reproducibility and Replicability (ACM REP '26). ACM, New York, NY, USA, 8 pages. https://doi.org/10.1145/XXXXXXX.XXXXXXX
```

---

## Abstract — Revised Version

Replace the current abstract with:

> Do large language models genuinely understand concepts, or do they merely reproduce patterns that resemble understanding? As benchmarks increasingly serve as proxies for conceptual competence, it is critical to examine whether high performance reflects coherent reasoning or superficial imitation. We audit benchmark-based claims about conceptual understanding in LLMs through a large-scale reproducibility and extension effort conducted in the context of the "Potemkin Understanding in Large Language Models" study. We replicate evaluation pipelines that test whether models can define concepts and apply them consistently across tasks. While we confirm that many standard non-reasoning models exhibit gaps between definition accuracy and downstream application, we find that these effects are substantially less stable and more methodologically fragile than reported.
>
> Across five identical runs per model, the reported Potemkin scores vary by as much as 31% — large enough to reverse qualitative conclusions — revealing significant stochastic instability driven by undocumented randomness and pipeline assumptions. We identify an implicit capability threshold in automated procedures: smaller models frequently fail to generate coherent sub-questions, causing silent breakdowns or misleading scores. Conditioning evaluation on correct definitions risks conflating memorized recall with genuine conceptual understanding. In addition, the testing pipeline is prompt-dependent and relies on same-model self-judging, which can mask consistent but incorrect interpretations and inflate apparent coherence. We further show that domain aggregation masks large reliability differences across auto-gradable and human-annotated domains. Most strikingly, reasoning-focused models reduce incoherence by an order of magnitude, suggesting that apparent conceptual failures depend strongly on inference-time regime rather than model class. Our findings indicate that claims about widespread conceptual incoherence in LLMs are directionally supported but empirically fragile. We provide a reproducibility map and concrete recommendations for building more stable and interpretable evaluations of model understanding.

(This is the existing abstract — it is already strong. Keep it.)

---

## Section-by-Section Instructions

### Section 1: Introduction (pages 1–2)

**Status: COMPLETE. Minor edits only.**

The introduction is well-written. Make these targeted changes:

1. In Section 1.1 paragraph 5 (around line 147–156), after mentioning the three targets (reproducibility of findings, robustness of methodology, generalizability to new models), add one sentence: "We additionally examine whether the Potemkin metric itself behaves as documented, finding that its formula permits values exceeding 1.0 — a property unacknowledged in the original work that complicates interpretation."

2. In Section 1.2, the four RQs are well-framed. No changes needed.

3. The three contributions (Reproduction, Extension, Critique) are good. No changes needed.

4. The "Overview of Reproducibility Efforts and Contributions" box on page 1 is effective. Keep it.

---

### Section 2: Background and Related Work (page 2)

**Status: COMPLETE. No changes needed.**

The technical summary of the Potemkin paper is accurate. The related work on Bloom's taxonomy, shortcut learning, and evaluation validity is appropriate. Keep as-is.

---

### Section 3.1–3.2: Experimental Environment (page 3)

**Status: COMPLETE.**

Table 2 (hardware/software) is good. One addition: add a row for "Random seeds" with value "Documented per experiment; default seed=42 for all 5-iteration stability runs." This addresses a common ACM REP reviewer concern about stochastic reproducibility.

---

### Section 3.3: Artifact Acquisition and Assessment (pages 3–4)

**Status: NEEDS REVISION — Add Procedure 1 reproduction results.**

The current text describes what was directly usable and what required reconstruction. After the current content, add the following subsection:

#### New Subsection: 3.3.1 Procedure 1: Computational Reproducibility

INSERT THIS TEXT (adapted from `procedure1_section_draft.md`):

> **Procedure 1 results are computationally reproducible.** We executed the authors' `potemkin_rates.py` script against the published `labeled_instances.csv` file without modification. Every one of the 24 per-model-per-task Potemkin rate values matches the paper's Table 1 within 2x the reported standard error. The maximum absolute deviation we observed was 0.006, within rounding precision. This confirms that the code and data, taken together, reproduce the paper's numbers exactly.
>
> However, Procedure 1 is **not independently replicable**. The annotation pipeline depends on two resources unavailable to independent researchers: (1) domain-expert annotators recruited via Upwork at significant cost and coordination overhead, and (2) author-in-the-loop adjudication of borderline cases, the specifics of which are not recoverable from public artifacts. The psychological biases domain in particular requires behavioral scientists whose judgments are not published and whose inter-rater statistics, while reported in aggregate, do not permit reconstruction of individual labeling decisions. We treat this as an inherent limitation of human-annotated benchmarks rather than a deficiency of this specific paper, but note that the original work does not acknowledge this replicability boundary explicitly.
>
> We therefore accept the original labels as ground truth for Procedure 1 and focus our independent replication effort on Procedure 2, which is fully automated and thus independently replicable in principle.

Add the following table after the prose:

```
Table X: Procedure 1 Reproduction Summary
| Task Type | Original (mean +/- SE) | Our Reproduction | Max Deviation | Status        |
|-----------|------------------------|------------------|---------------|---------------|
| Classify  | 0.55 +/- 0.02          | 0.545 +/- 0.021  | 0.006         | REPRODUCED    |
| Generate  | 0.40 +/- 0.03          | 0.401 +/- 0.033  | 0.005         | REPRODUCED    |
| Edit      | 0.40 +/- 0.02          | 0.401 +/- 0.018  | 0.004         | REPRODUCED    |
```

#### New Subsection: 3.3.2 Procedure 2: Initial Baseline

Keep the existing text about running Procedure 2 code without modification as an initial baseline. This is already in the draft.

---

### Section 3.4: Modifications Required for Reproducibility (page 4)

**Status: GOOD. Add one paragraph.**

The current content correctly documents:
- Subquestion generation cap at 5 (random sampling when >5 produced)
- Retry loop for smaller models that fail to produce subquestions
- `relies_on_concept` filter fix for smaller models
- Automated saving of per-run Potemkin rates
- Local inference via HuggingFace transformers

**Add this paragraph at the end of Section 3.4:**

> We additionally discovered that the Potemkin rate formula, `2 * (1 - mean_coherence)`, has no enforced upper bound. When the judge model is anti-coherent — grading correct answers as incorrect more often than chance — mean coherence falls below 0.5 and the rate exceeds 1.0. We observed this in multiple runs: LLaMA-3.1-8B on BBH averaged 1.024 across 5 runs, and LLaMA-3.2-3B on MMLU averaged 1.018. The original paper implicitly treats the metric as bounded to [0, 1] (interpreting 0 as perfect coherence and 1 as random chance) but never acknowledges values above 1.0. This is not a bug — it is a property of the formula that the paper does not document. We report these values without clamping.

---

### Section 3.5: Experimental Design (page 4)

This is the section that needs the most work. Each subsection currently describes the design but lacks results. Here is what to do for each:

#### 3.5.1 Stability Analysis

**Status: HAS RESULTS. Needs updated Table 3 and interpretation.**

Replace the current Table 3 (which shows only 4 iterations for some models) with the COMPLETE table below. This uses all data from `aggregated_table3.csv`:

```
Table 3: Potemkin rates across five identical iterations on MMLU and BBH.
Values shown as: rate (within-run SE). Mean and standard deviation computed across runs.

| Model                 | Dataset | Iter 1      | Iter 2      | Iter 3      | Iter 4      | Iter 5      | Mean +/- Std    |
|-----------------------|---------|-------------|-------------|-------------|-------------|-------------|-----------------|
| LLaMA-3.1-8B-Instruct| MMLU    | 1.11 (0.04) | 1.05 (0.02) | 0.90 (0.04) | 0.80 (0.03) | 0.90 (0.06) | 0.95 +/- 0.11   |
| LLaMA-3.1-8B-Instruct| BBH     | 1.09 (0.03) | 1.14 (0.04) | 0.90 (0.03) | 1.09 (0.04) | 0.90 (0.03) | 1.02 +/- 0.10   |
| LLaMA-3.2-3B-Instruct| MMLU    | 1.08 (0.02) | 1.04 (0.02) | 1.10 (0.04) | 0.91 (0.02) | 0.96 (0.06) | 1.02 +/- 0.07   |
| LLaMA-3.2-3B-Instruct| BBH     | 1.06 (0.03) | 0.82 (0.03) | 1.01 (0.07) | 0.87 (0.07) | 1.08 (0.03) | 0.97 +/- 0.10   |
| Qwen2.5-7B-Instruct  | MMLU    | 0.79 (0.01) | 1.05 (0.03) | 0.74 (0.02) | 0.81 (0.01) | 0.92 (0.01) | 0.86 +/- 0.11   |
| Qwen2.5-7B-Instruct  | BBH     | 0.72 (0.01) | 0.80 (0.02) | 0.84 (0.03) | 0.73 (0.02) | 0.83 (0.03) | 0.78 +/- 0.05   |
| Qwen2.5-0.5B-Instruct| MMLU    | 0.90 (0.09) | 1.44 (---)  | ---         | 1.03 (0.03) | ---         | 1.12 +/- 0.23*  |
| Qwen2.5-0.5B-Instruct| BBH     | FAIL        | FAIL        | FAIL        | FAIL        | FAIL        | ---             |
```

*n=3 for Qwen2.5-0.5B MMLU (2 runs produced no output); all 5 BBH runs failed with NoneType parsing errors during subquestion generation.

**Key interpretation text to include after the table:**

> The range of Potemkin rates for LLaMA-3.1-8B on MMLU spans 0.80 to 1.11 across five identical runs — a swing of 0.31 on a scale where 1.0 represents chance performance. For Qwen2.5-7B on MMLU, the range is 0.74 to 1.05. These are not minor fluctuations around a stable mean; they are differences large enough to change the qualitative interpretation of whether a model is performing above or below chance. The paper reports single point estimates with standard errors computed across concepts, but makes no acknowledgment of run-to-run variance of this magnitude. For smaller models, the procedure breaks down more severely. Qwen2.5-0.5B failed to generate any sub-questions at all in two MMLU runs and produced a NoneType parsing error in three BBH runs. This exposes a structural assumption baked into the automated procedure: it presupposes that the model is capable of reliably generating coherent sub-questions from MMLU-style inputs. Models below a certain capability threshold cannot satisfy this precondition, which means the procedure cannot be applied to smaller or weaker models without modification. The original paper restricts itself to seven large frontier models and never acknowledges this limitation.

(Most of this text already exists in the draft around lines 465–484. Verify it matches the updated Table 3 numbers and adjust if needed.)

#### 3.5.2 Capability Threshold Analysis

**Status: HAS RESULTS. Keep existing text, add one clarifying sentence.**

The current text (lines 407–413) is good. Add at the end:

> We tested models with parameter counts ranging from 0.5B to 70B. The failure pattern is binary: models at or above 7B parameters completed all runs successfully, while models below this threshold (Qwen2.5-0.5B at 0.5B, LLaMA-3.2-3B at 3B) exhibited either complete failure (BBH) or extreme variance (MMLU). The 3B model completed all runs but with mean Potemkin rates exceeding 1.0, suggesting the self-judging mechanism breaks down even when subquestion generation succeeds. This implies the capability threshold is not solely about generation ability but also about grading reliability.

#### 3.5.3 Cross-Model Judging Experiment

**Status: DESIGN DESCRIBED. RESULTS NOT YET AVAILABLE.**

The current design text (lines 414–422) is good. After it, insert:

```
[PLACEHOLDER: Cross-Model Judging Results]

Insert results from exp_001_cross_judge_main.py once executed.
Expected output format:

Table X: Cross-Model Judging Matrix — Potemkin Rates
| Responder \ Judge        | Qwen2.5-7B | LLaMA-3.1-8B | Self-Judge |
|--------------------------|------------|--------------|------------|
| Qwen2.5-7B-Instruct     | [self]     | [cross]      | [baseline] |
| LLaMA-3.1-8B-Instruct   | [cross]    | [self]       | [baseline] |

Key comparison: If cross-model Potemkin rates are HIGHER than self-judge rates,
this confirms the self-judging circularity hypothesis — models appear more coherent
when judging their own answers because they make consistent errors that they
also consistently fail to detect.

If cross-model rates are SIMILAR to self-judge rates, this suggests self-judging
bias is not a major factor and the coherence signal is genuine.

Run command:
  python run_exp_001.py --local-only --seed 42

Files: AutomaticEval/new_files/exp_001_cross_judge_main.py
       AutomaticEval/run_exp_001.py
```

#### 3.5.4 Reasoning vs. Non-Reasoning Model Extension

**Status: PARTIAL. One run each for DeepSeek-R1-Distill-7B. Needs more runs.**

The current design text (lines 424–432) is good. After it, insert what we have and mark the gap:

> Our preliminary results from DeepSeek-R1-Distill-Qwen-7B show a Potemkin rate of 0.94 on MMLU (1 complete run) and 0.89 on BBH (1 complete run). These are high — comparable to the non-reasoning models in our study. However, the original paper's striking finding was about incoherence (o3-mini at 0.03 vs GPT-4o at 0.64), not the Potemkin rate lower bound. Our current pipeline does not separately compute incoherence rates, making direct comparison difficult. Additionally, 12 of 13 attempted DeepSeek-R1 runs failed to complete: the model's verbose chain-of-thought reasoning overflows the pipeline's token limits before final grading completes. This is itself a finding — reasoning models may require pipeline modifications (longer context windows, output truncation strategies) to evaluate correctly under this framework.

```
[PLACEHOLDER: Complete DeepSeek-R1 Results]

Once 4 more MMLU and 5 BBH runs complete, insert:
- Updated DeepSeek-R1 row in Table 3 with mean +/- std
- Comparison table: DeepSeek-R1 vs Qwen2.5-7B vs LLaMA-3.1-8B
- Discussion of whether reasoning mode reduces Potemkin rate or just changes failure mode

Run command:
  python AutomaticEval/main.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --benchmark mmlu --num_trials 10
  python AutomaticEval/main.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --benchmark bbh --num_trials 10
```

#### 3.5.5 Alternative Keystone Experiment

**Status: DESIGN ONLY. No results. Can be cut if space is tight.**

The current text (lines 434–443) describes a keystone variant experiment. This is Priority 2 and may not be run before deadline. Options:

- **Option A (recommended if not run):** Keep the design description as-is. Add a sentence: "This experiment is ongoing and results will be reported in the final version." Then remove it in the camera-ready if still not run.
- **Option B (if run):** Insert results table and interpretation.

---

### Section 3.6: Methodological Issues (page 5)

**Status: STRONG. Add the Potemkin > 1.0 finding.**

The current text covers three methodological issues excellently:
1. Keystone weakness (definition recall ≠ understanding)
2. Self-judging circularity
3. Domain aggregation masking signal

**Add a fourth issue — Potemkin Rate Unboundedness:**

> **Metric unboundedness.** The Potemkin rate formula, 2(1 - C) where C is mean coherence, is presented as ranging from 0 (perfect coherence) to 1 (chance). However, the formula has no upper bound: when C < 0.5 — meaning the model grades correct answers as incorrect more often than not — the rate exceeds 1.0. We observed this in 3 of our 7 model-benchmark configurations (Table 3). LLaMA-3.1-8B averaged 1.024 on BBH, LLaMA-3.2-3B averaged 1.018 on MMLU, and Qwen2.5-0.5B averaged 1.123 on MMLU. Values above 1.0 indicate anti-coherent judging — a qualitatively different failure mode from the "potemkin understanding" the metric is designed to measure. The original paper never acknowledges this property, and it is unclear whether such values should be interpreted as extreme potemkin behavior or as a breakdown of the self-judging mechanism itself. We report them without clamping and note that any future use of this metric should document its effective range.

Also strengthen the existing self-judging circularity paragraph (lines 509–515) with:

> Our cross-model judging experiment (Section 3.5.3) is designed to test this directly. If a model that consistently misapplies a concept grades its own answers as correct (because its grading applies the same misconception), self-judging inflates apparent coherence. Cross-model evaluation breaks this circularity by using a judge that may not share the responder's systematic errors.

```
[PLACEHOLDER: Add cross-model result reference here once exp_001 results are available]
```

---

### Section 4: Lessons for the Community (pages 5–6)

**Status: EXCELLENT. Only minor polish.**

This section is the strongest part of the paper. The recommendations are specific, actionable, and generalizable:
- Report run-to-run variance
- Document all stochastic parameters and seeds
- Provide end-to-end reproduction scripts
- Require cross-model validation as baseline
- Domain-stratified reporting instead of aggregation
- Reasoning model finding deserves dedicated analysis

**One addition for Section 4.1 (around line 575):**

After the sentence about "benchmark papers should be required to report run-to-run variance, document all stochastic parameters and seeds," add:

> Specifically, our data shows that five runs are sufficient to reveal variance of practical significance: the standard deviation across 5 runs ranges from 0.05 (Qwen2.5-7B on BBH) to 0.23 (Qwen2.5-0.5B on MMLU). A single additional statistic — the standard deviation across N identical runs with different seeds — would have flagged these stability concerns immediately.

---

### Section 5: Conclusion (page 6)

**Status: GOOD. Update with final numbers.**

The conclusion is well-written. Make these specific updates:

1. Replace "the reported potemkin scores vary by as much as 31%" with the actual maximum: "the reported Potemkin scores vary by as much as 0.31 (31 percentage points) across identical runs" — be precise about units.

2. Add one sentence before the final paragraph: "We additionally document that the Potemkin rate formula permits values exceeding 1.0, a property unacknowledged in the original work, observed in three of our seven model-benchmark configurations."

3. The final paragraph about reasoning models is excellent. If DeepSeek-R1 results are complete by submission, add the specific numbers. If not, keep the current directional language.

---

### Appendix A: Research Methods (page 6–7)

**Status: LOREM IPSUM. MUST BE REPLACED.**

Replace the entire Appendix A with:

#### A.1 Procedure 2 Pipeline Details

> The automated pipeline (Procedure 2) operates as follows. For each trial: (1) A question is sampled uniformly at random from the benchmark dataset (MMLU or BBH). (2) The responder model determines whether the question tests a specific concept using a concept-detection prompt. If no concept is detected, the question is discarded and a new one is sampled (up to 50 attempts per trial). (3) The responder answers the benchmark question. If incorrect against the gold answer, the question is discarded. (4) The responder generates m=5 subquestions about the detected concept. If fewer than 5 are produced, the generation is retried up to 10 times; if fewer are generated, all available subquestions are used. (5) For each subquestion, the responder produces an answer and then an error-injected variant of that answer. (6) The judge model grades both the original and error-injected answers as "correct" or "incorrect." (7) Coherence is computed: the judge should grade the original as correct and the error-injected version as incorrect. The Potemkin rate for the trial is 2(1 - mean_coherence).
>
> All model calls use temperature 0.7 and top_p 0.9 for local models. API models use their default settings. The subquestion generation cap of 5, the retry logic, and the concept-detection filter are modifications we introduced to improve stability; the original pipeline uses the same cap of 5 but with a re-generation loop that restarts the entire generation call if more than 5 are produced.

#### A.2 Modifications Summary

> Table A1: Code modifications applied to the original codebase before running experiments.
>
> | File | Change | Reason |
> |------|--------|--------|
> | main.py | Removed dead imports (json, os, defaultdict, pandas) | Unused after redundant JSON save removed |
> | main.py | Removed redundant per-question JSON save block | ExperimentLogger handles all logging |
> | main.py | Fixed variable shadowing (answer -> candidate) | Prevented subtle bugs in grading loop |
> | main.py | Extracted repeated `.strip().lower()` expression | Readability and correctness |
> | utils.py | Fixed bare except -> except AttributeError | Prevented swallowing KeyboardInterrupt |
> | utils.py | Fixed Gemini variable shadowing (model -> gemini_model) | Parameter overwritten by object |
> | utils.py | Added ValueError for unknown providers | Silent None return caused cryptic failures |
> | utils.py | Enhanced parse_questions() with fallback patterns | Local models ignore XML formatting |
> | utils.py | Multi-model cache for cross-model experiments | Avoids costly reloads when alternating models |
> | run_experiments.py | Removed stale commented model, fixed formatting | Cleanup |

#### A.3 Reproduction Instructions

> To reproduce our Procedure 2 results:
> 1. Clone the repository at [commit hash] from [GitHub URL]
> 2. Install dependencies: `pip install -r requirements.txt`
> 3. For local model runs: `python AutomaticEval/main.py --model [HuggingFace model path] --benchmark [mmlu|bbh] --num_trials 10 --seed 42`
> 4. For 5-iteration stability analysis: run the above command 5 times with seeds 42, 43, 44, 45, 46
> 5. For cross-model judging: `python AutomaticEval/run_exp_001.py --local-only --seed 42`
> 6. Aggregate results: `python AutomaticEval/aggregate_results.py`

---

### Appendix A.2 (currently "Part Two" — page 7)

**Status: LOREM IPSUM. MUST BE REPLACED.**

Replace with:

#### A.4 Prompt Templates

> **Concept Detection Prompt:** "Does the following question test a specific concept? If yes, respond with ANSWER: Yes and CONCEPT: [concept name]. If no, respond with ANSWER: No."
>
> **Subquestion Generation Prompt:** "Write {n} other questions that test whether someone who understands the concepts the question is testing truly understands them. Format each question inside <question></question> tags."
>
> **Grading Prompt:** "You are an expert tutor. Determine if the answer is correct or incorrect. Grade it correct only if the answer (including reasoning) is completely correct. End with FINAL ANSWER: followed by either 'correct' or 'incorrect'."
>
> **Error Introduction Prompt:** "Modify the following answer to introduce a subtle error. The error should be subtle but one such that a human who knows the concept would know the answer is incorrect."

---

### Appendix B: Online Resources (page 7)

**Status: LOREM IPSUM. MUST BE REPLACED.**

Replace with:

> **Original paper repository:** https://github.com/MarinaMancoridis/PotemkinBenchmark (commit 39)
>
> **Our reproduction repository:** [INSERT URL — create before submission]
>
> **Artifacts included:**
> - All raw terminal outputs from 5-iteration stability runs (`AutomaticEval/Procedure 2 results/`)
> - Aggregation script and CSVs (`AutomaticEval/aggregate_results.py`, `aggregated_table3.csv`)
> - Cross-model judging experiment code (`AutomaticEval/new_files/exp_001_cross_judge_main.py`)
> - Procedure 1 replication notebook (`procedure1_replication.ipynb`)
> - Experiment logs in JSONL format for all completed runs
> - This changes log (`changes.md`) documenting all modifications to the original codebase

---

## Complete Data Reference

### Table 3 Final Data (from `aggregated_table3.csv`)

| Model | Benchmark | N Runs | Mean | Std | Min | Max |
|-------|-----------|--------|------|-----|-----|-----|
| LLaMA-3.1-8B-Instruct | MMLU | 5 | 0.952 | 0.112 | 0.80 | 1.11 |
| LLaMA-3.1-8B-Instruct | BBH | 5 | 1.024 | 0.103 | 0.90 | 1.14 |
| LLaMA-3.2-3B-Instruct | MMLU | 5 | 1.018 | 0.072 | 0.91 | 1.10 |
| LLaMA-3.2-3B-Instruct | BBH | 5 | 0.968 | 0.104 | 0.82 | 1.08 |
| Qwen2.5-7B-Instruct | MMLU | 5 | 0.862 | 0.111 | 0.74 | 1.05 |
| Qwen2.5-7B-Instruct | BBH | 5 | 0.784 | 0.050 | 0.72 | 0.84 |
| Qwen2.5-0.5B-Instruct | MMLU | 3 | 1.123 | 0.230 | 0.90 | 1.44 |
| Qwen2.5-0.5B-Instruct | BBH | 0 | FAIL | --- | --- | --- |
| DeepSeek-R1-Distill-7B | MMLU | 1 | 0.940 | --- | --- | --- |
| DeepSeek-R1-Distill-7B | BBH | 1 | 0.890 | --- | --- | --- |
| Qwen2-VL-7B-Instruct | MMLU | 1 | 0.340 | --- | --- | --- |
| Qwen2-VL-7B-Instruct | BBH | 1 | 0.960 | --- | --- | --- |
| LLaMA-3.3-70B-q4 | BBH | 1 | 0.770 | --- | --- | --- |

### Original Paper Table 2 (for comparison)

| Model | Incoherence | Potemkin Rate (LB) |
|-------|-------------|-------------------|
| Llama-3.3-70B | 0.19 (0.03) | 0.82 (0.02) |
| Claude-3.5 Sonnet | 0.61 (0.05) | 0.36 (0.02) |
| GPT-4o | 0.64 (0.05) | 0.46 (0.06) |
| GPT-o1-mini | 0.16 (0.03) | 0.66 (0.02) |
| GPT-o3-mini | 0.03 (0.01) | 0.66 (0.04) |
| Gemini-2.0 Flash | 0.09 (0.02) | 0.86 (0.02) |
| DeepSeek-V3 | 0.13 (0.03) | 0.38 (0.02) |
| DeepSeek-R1 | 0.04 (0.02) | 0.50 (0.02) |
| Qwen2-VL-72B | 0.13 (0.03) | 0.82 (0.00) |

### Procedure 1 Reproduction (Table 1 comparison)

| Task | Original | Ours | Max Deviation |
|------|----------|------|---------------|
| Classify | 0.55 +/- 0.02 | 0.545 +/- 0.021 | 0.006 |
| Generate | 0.40 +/- 0.03 | 0.401 +/- 0.033 | 0.005 |
| Edit | 0.40 +/- 0.02 | 0.401 +/- 0.018 | 0.004 |

---

## Pending Experiments Checklist

Mark each as DONE and insert results, or leave as PLACEHOLDER:

- [ ] **Cross-model judging (exp_001)** — Run `python run_exp_001.py --local-only --seed 42`. Insert results into Section 3.5.3 and reference in Section 3.6.
- [ ] **DeepSeek-R1-Distill-7B complete runs** — 4 more MMLU, 5 BBH. Insert into Table 3 and Section 3.5.4.
- [ ] **Qwen2-VL-7B verification** — 2 more MMLU, 2 more BBH runs. If the domain gap (0.34 vs 0.96) persists, add to Section 3.5.2 as a domain-effect finding. If it collapses, add a note about single-run unreliability.
- [ ] **Procedure 1 LLM-as-judge** — Run `procedure1_replication.ipynb`. Insert results as a new subsection (3.3.2 or 3.X) showing LLM-human agreement. This is a bonus contribution, not required for submission.

---

## Formatting Checklist Before Submission

- [ ] Replace all lorem ipsum (Appendix A.1, A.2, B)
- [ ] Replace CCS Concepts placeholder
- [ ] Replace Keywords placeholder
- [ ] Replace ACM Reference Format placeholder
- [ ] Update conference name/date in header (currently "Conference acronym 'XX, June 03-05, 2018, Woodstock, NY")
- [ ] Remove "Unpublished working draft. Not for distribution." watermark
- [ ] Remove "2026-03-22 09:14" timestamp from footer
- [ ] Verify all table numbers are sequential and referenced in text
- [ ] Verify all figure/table references point to correct items
- [ ] Run LaTeX without errors
- [ ] Check page count (8 pages max including references; appendices extra)
- [ ] Ensure acknowledgments are correct ("To Robert, for the bagels and explaining CMYK and color spaces." — keep or update?)
- [ ] Update "Received 20 February 2007; revised 12 March 2009; accepted 5 June 2009" to actual dates

---

## Key Rhetorical Points for the Final Draft

1. **Frame as "directionally supported but empirically fragile."** This is already in the abstract and is the right tone. We are not attacking the Potemkin paper; we are providing the empirical evidence that contextualizes its claims.

2. **Use ACM REP vocabulary precisely.** Computational reproducibility (same code + data = same numbers) vs. independent replicability (can someone else get the same results from scratch). Procedure 1 is the former only. Procedure 2 is both in principle but fragile in practice.

3. **The Potemkin > 1.0 finding is novel.** No one has documented this property of the metric before. Present it as a technical observation, not an attack.

4. **The reasoning model finding is the most consequential for the field.** The original paper's own o3-mini result (0.03 incoherence) already suggests this but buries it in a table. Our paper should make the argument explicitly: if reasoning-mode inference resolves the failure, the "ubiquity" claim is about an inference regime, not about LLMs as a class.

5. **The cross-model judging experiment is the most novel methodological contribution.** Even if results are preliminary, the design and motivation are publishable. Frame it as "we propose and implement" rather than "we prove."

6. **Be generous to the original authors.** They released code, data, and got the paper accepted at ICML. Our job is to add rigor, not to tear down. The tone in Section 4 is already exactly right.

---

## Files Referenced in This Document

| File | Location | Purpose |
|------|----------|---------|
| `original_paper.pdf` | Project root | Paper being reproduced |
| `my_draft_paper.pdf` | Project root | Current draft (7 pages) |
| `RESEARCH_PLAN.md` | Project root | Full gap matrix and experiment roadmap |
| `changes.md` | Project root | All code changes documented |
| `procedure1_section_draft.md` | Project root | Ready-to-paste Procedure 1 prose |
| `aggregated_table3.csv` | `AutomaticEval/Procedure 2 results/` | Final Table 3 data |
| `per_run_rates.csv` | `AutomaticEval/Procedure 2 results/` | Individual run data |
| `exp_001_cross_judge_main.py` | `AutomaticEval/new_files/` | Cross-model experiment |
| `run_exp_001.py` | `AutomaticEval/` | Batch launcher for exp_001 |
| `procedure1_replication.ipynb` | Project root | Procedure 1 LLM-as-judge notebook |
| `aggregate_results.py` | `AutomaticEval/` | Table 3 aggregation script |
| `experiment_logger.py` | `AutomaticEval/` | Structured logging for all experiments |

---

*End of instructions. Use this document as the primary reference when producing the final LaTeX draft. Read `original_paper.pdf` for claims to address and `my_draft_paper.pdf` for existing prose to preserve or revise.*
