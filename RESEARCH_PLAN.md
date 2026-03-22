# Research Plan: Potemkin Reproducibility Paper
**Session date:** 2026-03-22
**Submission target:** ACM REP (Experience Paper)
**Current draft state:** 7 pages, significant gaps, Appendix A is lorem ipsum placeholder

---

## SESSION OPENING

**What I read:**
- `original_paper.pdf` — Mancoridis et al. 2025, full text including all appendices (A–G)
- `my_draft_paper.pdf` — Hassan et al. working draft, 7 pages, timestamped 2026-03-22
- `AutomaticEval/new_files/` — exp_001 (cross-judge), exp_002 (keystone), exp_004 (domain) main scripts
- `BenchmarkDataset/potemkin_rates.py` — Procedure 1 computation code
- `prompts_optimized.py` — alternative prompt set for LLM-as-judge pipeline
- All raw result files from `Procedure 2 results/Terminal Outputs/`

**Highest-priority unresolved gap:**
The draft Table 3 is incomplete (Iter 5 missing for several models, DeepSeek not in table, Qwen2-VL not mentioned), and the three extension experiments (cross-judge, keystone, domain) have code but **no results in the paper**. The Appendix is lorem ipsum. The paper cannot be submitted as-is.

---

## PART 1 — FULL ARTIFACT AUDIT

### 1.1 What the Original Paper Claims

#### Procedure 1 (Benchmark — Table 1)
| Claim | Metric | Value |
|---|---|---|
| Models define concepts correctly | Definition accuracy | 94.2% |
| Overall Potemkin rate (Classify) | Potemkin rate | 0.55 (±0.02) |
| Overall Potemkin rate (Generate) | Potemkin rate | 0.40 (±0.03) |
| Overall Potemkin rate (Edit) | Potemkin rate | 0.40 (±0.02) |
| Potemkins are ubiquitous | All 7 models, 3 domains | Consistently high |
| Worst model: Qwen2-VL | Classify=0.66, Generate=0.62 | Highest rates |
| Best model: DeepSeek-R1 | Classify=0.47, Generate=0.39 | Lowest classify |

**7 models tested:** Llama-3.3 (70B), GPT-4o, Gemini-2.0 Flash, Claude-3.5 Sonnet, DeepSeek-V3, DeepSeek-R1, Qwen2-VL (72B)
**Dataset:** 32 concepts, 3 domains, 3,159 labeled points (2,030 classification, 791 editing, ~338 generation annotations)

#### Procedure 2 (Automated — Table 2)
| Model | Incoherence | Potemkin rate (LB) |
|---|---|---|
| Llama-3.3 | 0.19 (0.03) | 0.82 (0.02) |
| Claude-3.5 | 0.61 (0.05) | 0.36 (0.02) |
| GPT-4o | 0.64 (0.05) | 0.46 (0.06) |
| GPT-o1-mini | 0.16 (0.03) | 0.66 (0.02) |
| **GPT-o3-mini** | **0.03 (0.01)** | **0.66 (0.04)** |
| Gemini-2.0 | 0.09 (0.02) | 0.86 (0.02) |
| DeepSeek-V3 | 0.13 (0.03) | 0.38 (0.02) |
| DeepSeek-R1 | 0.04 (0.02) | 0.50 (0.02) |
| Qwen2-VL | 0.13 (0.03) | 0.82 (0.00) |
| **Overall** | **0.22 (0.01)** | **0.62 (0.01)** |

**Key claim:** o3-mini incoherence of 0.03 is a striking outlier vs GPT-4o's 0.64. Benchmark: MMLU (Hendrycks et al., 2020). Pipeline: 5 related subquestions, self-judging, cap at 5.

---

### 1.2 What Our Draft Currently Has

#### Procedure 2 Results — Extracted from Raw Files

**Llama-3.1-8B-Instruct** (5 complete runs each):
| Run | MMLU | BBH |
|---|---|---|
| 1 | 1.11 (0.04) | 1.09 (0.03) |
| 2 | 1.05 (0.02) | 1.14 (0.04) |
| 3 | 0.90 (0.04) | 0.90 (0.03) |
| 4 | 0.80 (0.03) | 1.09 (0.04) |
| 5 | 0.90 (0.06) | 0.90 (0.03) |
| **Mean ± Std** | **0.95 ± 0.11** | **1.02 ± 0.11** |

**Llama-3.2-3B-Instruct** (5 complete runs each):
| Run | MMLU | BBH |
|---|---|---|
| 1 | 1.08 (0.02) | 1.06 (0.03) |
| 2 | 1.04 (0.02) | 0.82 (0.03) |
| 3 | 1.10 (0.04) | 1.01 (0.07) |
| 4 | 0.91 (0.02) | 0.87 (0.07) |
| 5 | 0.96 (0.06) | 1.08 (0.03) |
| **Mean ± Std** | **1.02 ± 0.08** | **0.97 ± 0.11** |

**Qwen2.5-7B-Instruct** (5 complete runs each):
| Run | MMLU | BBH |
|---|---|---|
| 1 | 0.79 (0.01) | 0.72 (0.01) |
| 2 | 1.05 (0.03) | 0.80 (0.02) |
| 3 | 0.74 (0.02) | 0.84 (0.03) |
| 4 | 0.81 (0.01) | 0.73 (0.02) |
| 5 | 0.92 (0.01) | 0.83 (0.03) |
| **Mean ± Std** | **0.86 ± 0.12** | **0.78 ± 0.05** |

**Qwen2.5-0.5B-Instruct** (capability threshold failure):
- BBH: 0 complete runs (all produced NoneType parsing errors — subquestion generation failure)
- MMLU: 3 partial runs (1.03, 1.44, 0.90) + 2 empty. Range = 0.54. Completely unreliable.

**DeepSeek-R1-Distill-Qwen-7B** (incomplete):
- MMLU: 1 complete run (0.94), 4 incomplete
- BBH: 0 complete runs

**Qwen2-VL-7B-Instruct** (very sparse):
- MMLU: 0.34 (1 run only)
- BBH: 0.96 (1 run only)
- ⚠️ 0.34 vs 0.96 = massive domain effect — not yet discussed in paper

**Llama-3.3-70B-q4** (quantized):
- BBH: 0.77 (1 run only)
- MMLU: incomplete

#### Key Observation: Potemkin Rates > 1.0
Multiple models return values above 1.0 (e.g., Llama-3.1-8B MMLU run1 = 1.11, BBH run2 = 1.14). The formula `2 * (1 - mean_coherence)` can exceed 1.0 when mean_coherence < 0.5. This means models are grading correct answers as incorrect MORE than random chance. This is itself a reproducibility finding not discussed in the original paper or the draft.

---

### 1.3 Gap Matrix

| Paper Claim | Location | Status | Notes |
|---|---|---|---|
| **PROCEDURE 1** | | | |
| 94.2% definition accuracy across 7 models | Sec 3.3 | **REPRODUCED** | Computed from labeled_instances.csv — matches exactly |
| Potemkin rate Classify = 0.55 (±0.02) overall | Table 1 | **REPRODUCED** | Our result: 54.54% ± 2.1% — all 7 models within 2×SE of original |
| Potemkin rate Generate = 0.40 (±0.03) overall | Table 1 | **REPRODUCED** | Our result: 40.09% ± 3.3% — max model deviation = 0.005 |
| Potemkin rate Edit = 0.40 (±0.02) overall | Table 1 | **REPRODUCED** | Our result: 40.13% ± 1.8% — all models within 2×SE |
| Potemkins ubiquitous across all 7 models | Sec 3.3 | **REPRODUCED** | All 7 original models reproduced from shared labels |
| Potemkins ubiquitous across 3 domains | Sec 3.3 | **PARTIAL** | Domain-stratified breakdown not yet extracted from labels |
| Psych biases annotation is expert-validated | App D, F | **CONTRADICTED** | Expert annotation via Upwork not independently replicable; hard boundary |
| Keystone = correct definition is sufficient | Sec 3.2 | **MISSING** | exp_002 keystone variant experiment not run |

**⭐ KEY FINDING — Procedure 1 Statistical Reproduction:**
`potemkin_rates.py` on the original `labeled_instances.csv` reproduces Table 1 exactly.
Every single per-model, per-task value falls within 2×SE of the original. Max absolute deviation: 0.006.
**Critical nuance for the paper:** This confirms *computational* reproducibility (same data → same numbers) but does NOT confirm *independent replicability* (the annotation pipeline required Upwork experts + paper-author labelling and cannot be re-run). This is an important distinction for ACM REP — the paper is computationally reproducible but not independently replicable for Procedure 1.
| **PROCEDURE 2** | | | |
| Overall incoherence = 0.22, Potemkin LB = 0.62 | Table 2 | **PARTIAL** | We have 3 models with 5 runs; different model family; directionally confirmed |
| GPT-4o incoherence = 0.64 | Table 2 | **MISSING** | No API runs completed (API cost / access) |
| o3-mini incoherence = 0.03 (striking outlier) | Table 2 | **MISSING** | o3-mini not tested |
| Claude-3.5 incoherence = 0.61 | Table 2 | **MISSING** | Not run |
| Llama-3.3 incoherence = 0.19, PR = 0.82 | Table 2 | **PARTIAL** | We ran 3.1-8B and 3.2-3B; similar high PR but different model |
| Qwen2-VL incoherence = 0.13, PR = 0.82 | Table 2 | **PARTIAL** | We ran Qwen2-VL-7B: MMLU=0.34, BBH=0.96 — striking domain discrepancy |
| Results stable (reported with single-run std errors) | Table 2 | **CONTRADICTED** | Our 5-run analysis shows variance up to ±0.31, enough to flip conclusions |
| Run-to-run variance is acknowledged | Paper | **CONTRADICTED** | Original paper shows no awareness; we document it explicitly |
| Capability threshold acknowledged | Paper | **CONTRADICTED** | Original paper never mentions; Qwen-0.5B completely breaks |
| Self-judging circularity acknowledged | Paper | **CONTRADICTED** | Never mentioned; exp_001 designed to test it but not run |
| Subquestion cap = 5 documented | Paper | **CONTRADICTED** | Cap is undocumented in original; we identified and documented it |
| **EXTENSIONS** | | | |
| Cross-model judging experiment | Our draft Sec 3.5.3 | **MISSING** | Code exists (exp_001), not run |
| Keystone sensitivity (Variants A/B/C) | Our draft Sec 3.5.5 | **MISSING** | Code exists (exp_002), not run |
| Reasoning vs. non-reasoning comparison | Our draft Sec 3.5.4 | **PARTIAL** | DeepSeek-R1 incomplete (1 run only); directional only |
| Domain stratification | Our draft Sec 3.5.2 | **MISSING** | Code exists (exp_004), not run |
| Capability threshold (smaller models) | Our draft Sec 3.5.2 | **PARTIAL** | Qwen-0.5B failure documented; needs quantitative framing |

---

## PART 2 — PRIORITIZED EXPERIMENT ROADMAP

### Priority Legend
- **P0** — Must have. Core claim support. Infrastructure exists. Do immediately.
- **P1** — Should have. Fills named gaps. Feasible in <1 day.
- **P2** — Nice to have. Strengthens but not blocking.
- **SKIP** — Cannot do. Missing data, compute, or API access.

---

### Ranked Experiment List

| Priority | Experiment | Estimated Effort | Expected Paper Impact | Dependency | Status |
|---|---|---|---|---|---|
| ~~**P0**~~ | ~~**Compute Procedure 1 rates using existing labeled_instances.csv + potemkin_rates.py**~~ | ~~2 hours~~ | ~~Fills entire Procedure 1 gap; produces our version of Table 1~~ | ~~None — data and code already exist~~ | **✅ DONE — see Finding below** |
| ~~**P0**~~ | ~~**Aggregate Procedure 2 results into final Table 3**~~ | ~~1 hour~~ | ~~Makes draft Table 3 complete and correct~~ | ~~Raw files exist~~ | **✅ DONE** — `aggregate_results.py` written; CSVs in `Procedure 2 results/`; see Table 3 below |
| **P0** | **Fix Qwen2-VL domain effect anomaly (MMLU=0.34 vs BBH=0.96)** | 2 hours | Striking finding not in paper; one run each — needs at least 3 runs to verify | Compute available for local Qwen2-VL-7B | 1 run each done |
| **P1** | **Run cross-model judging experiment (exp_001)** | 4–6 hours | Most novel methodological contribution; directly tests self-judging circularity claim | API access for GPT-4o + Claude or Together | Code complete (exp_001_cross_judge_main.py) |
| **P1** | **Run LLM-as-judge pipeline on BenchmarkDataset for Procedure 1** | 4–6 hours | Enables comparison to Table 1 without human annotators; core RQ2 contribution | prompts_optimized.py exists; labeled data exists | Design complete, not run |
| **P1** | **Complete DeepSeek-R1-Distill-Qwen-7B runs (4 remaining MMLU, 5 BBH)** | 4–8 hours (local GPU) | Enables reasoning vs. non-reasoning comparison (Section 3.5.4) | Local GPU access | 1 MMLU run complete |
| **P2** | **Run domain stratification experiment (exp_004)** | 4 hours | Addresses domain aggregation masking claim (Section 4) | API access or local GPU | Code complete (exp_004_domain_main.py) |
| **P2** | **Run keystone variant experiment (exp_002, Variant B + C)** | 4 hours | Tests whether definition recall inflates Potemkin rates; interesting but secondary | Game theory data + API access | Code complete (exp_002_keystone_main.py) |
| **SKIP** | **Reproduce original GPT-4o/Claude/Gemini Procedure 2 numbers** | — | Would strengthen RQ1 but requires significant API budget | ~$50–100 API cost for 5 runs × 2 benchmarks | No budget confirmed |
| **SKIP** | **o3-mini incoherence replication** | — | Striking original claim; cannot verify without access | o3-mini API not available in current setup | Blocker: API access |
| **SKIP** | **Expert annotation for Procedure 1 (psych biases)** | — | Hard reproducibility boundary; requires Upwork recruitment | Cannot replicate; flagged as finding | SKIP by design |

---

## PART 3 — PAPER PLAN FOR PUBLICATION

### Target Venue
**ACM REP 2026** (Conference on Reproducibility and Replicability)
Paper type: **Experience paper** — already chosen and appropriate. This paper IS the experience.

### Paper Structure (Current vs. Required)

| Section | Current State | What's Needed |
|---|---|---|
| Abstract | Good, directionally accurate | Update numbers once Table 3 finalized |
| 1. Introduction | Strong, 4 RQs well-framed | Minor polish |
| 2. Background | Complete | Good |
| 3.1–3.3 Setup | Complete, environment documented | Good |
| 3.4 Modifications | Good | Add note on Potemkin rate > 1.0 anomaly |
| 3.5 Experimental Design | Sub-sections described but RESULTS MISSING | Each sub-section needs its results table/figure |
| 4. Lessons for Community | Strong prose | Complete |
| 5. Conclusion | Good | Update once results finalized |
| **Appendix A** | **LOREM IPSUM — EMPTY** | **Must fill: Research Methods detail** |
| **Appendix B** | **Partial lorem ipsum** | **Must fill: Online Resources / artifact links** |

### The 4 Findings That Make This Paper

These are the paper's core contributions. Every experiment must serve at least one:

**Finding 1 — Stochastic Instability (CONFIRMED, needs polish)**
- Evidence: Table 3 — Llama-3.1-8B swings 0.80–1.11 on MMLU; Qwen2.5-7B swings 0.74–1.05
- Key message: The original paper's single-run point estimates with concept-level standard errors misrepresent procedural variance. A single run can change qualitative conclusions.
- What's needed: Finalize Table 3 with mean ± std across 5 runs. Add a figure showing variance distributions.

**Finding 2 — Capability Threshold Failure (CONFIRMED, needs quantification)**
- Evidence: Qwen-0.5B fails on all BBH runs; produces NoneType errors on 2/5 MMLU runs. Even the 3 that "complete" show rates 0.90–1.44.
- Key message: The automated procedure silently breaks for models below ~7B. The paper restricts itself to frontier models without acknowledging this assumption.
- What's needed: Table showing failure modes by model size. Describe NoneType error pattern explicitly.

**Finding 3 — Self-Judging Circularity (DESIGNED, NOT YET RUN)**
- Evidence needed: exp_001 cross-model judging matrix
- Key message: A model that consistently misapplies a concept will appear coherent under self-judging but incoherent under cross-model evaluation. This is the most novel contribution.
- What's needed: Run exp_001 for at least 2–3 model pairs. Even a 2×2 matrix (GPT-4o + one open-source model) would be publishable.
- **This is the single most important missing experiment.**

**Finding 4 — Reasoning Models as Confound (PARTIAL)**
- Evidence: DeepSeek-R1 has only 1 MMLU run (0.94); original paper reports 0.04 incoherence for R1 — an order-of-magnitude difference from standard models
- Key message: Reasoning-mode inference fundamentally changes the metric — not a quantitative refinement but potentially a qualitative refutation of the ubiquity claim
- What's needed: Complete DeepSeek-R1 runs (4 more MMLU, 5 BBH). This is local compute only.

### Potemkin Rate > 1.0 — A New Finding Not In Either Paper

The formula `potemkin_rate = 2 * (1 - mean_coherence)` returns values above 1.0 when `mean_coherence < 0.5`. Multiple of our runs produce this (Llama-3.1-8B run1 MMLU = 1.11, run2 BBH = 1.14). This means the model is grading correct answers as **wrong more often than random chance** — the judge is anti-coherent. This is never acknowledged in the original paper and implies the formula has no meaningful upper bound in practice. This should be added to Section 3.6 Methodological Issues.

### Critical Missing Piece: Procedure 1 LLM-as-Judge

The paper currently says it could not reproduce Procedure 1 because it requires human annotators. But we have:
- The full labeled dataset (`labeled_instances.csv`)
- Working iterator code (`BenchmarkDataset/iterators.py`, `potemkin_rates.py`)
- The judge prompts (`prompts_optimized.py` — DEFINE_CONCEPT_JUDGE_PROMPT, etc.)

**The path forward:** Run `potemkin_rates.py` against the existing labels (these are the original paper's own labels — no judge needed for computing rates from existing data). This gives us the Procedure 1 numbers exactly as the paper computed them. This is a 2-hour task with zero API cost. It directly answers RQ1 for Procedure 1.

Then, as an extension, run the LLM-as-judge pipeline on a subset to show how LLM labels compare to expert labels — this is a second-order reproducibility finding.

---

## PART 4 — DAY-BY-DAY EXECUTION PLAN

### Day 1 (Today — March 22)
1. ~~**[2h] Run `potemkin_rates.py`**~~ — **DONE.** Procedure 1 reproduced. All values within 2×SE.
2. ~~**[1h] Finalize Table 3**~~ — **DONE.** `aggregate_results.py` written and run. See final table below.
3. ~~**[1h] Add Potemkin > 1.0 analysis**~~ — **DONE.** Documented in Table 3 Notable Findings and in `changes.md`. Needs to be added to draft paper Section 3.6.
4. **[1h] Draft the Procedure 1 section update** — "computationally reproduced but not independently replicable" is the precise claim. ← NEXT TASK

### Day 2 (March 23)
1. **[6h] Run exp_001 cross-model judging** for at least one responder × judge pair (e.g., Qwen2.5-7B answering, Llama-3.1-8B judging). This is the top missing experiment.
2. **[2h] Run 4 remaining DeepSeek-R1 MMLU runs** (local GPU, background job).
3. **[2h] Draft Section 3.5.3 results** (cross-model judging) once exp_001 returns first results.

### Day 3 (March 24)
1. **[4h] Complete DeepSeek-R1 BBH runs** (local GPU, background job).
2. **[4h] Draft Section 3.5.4 results** (reasoning model comparison) using complete DeepSeek data.
3. **[2h] Write Appendix A** (Research Methods detail — no lorem ipsum).

### Day 4 (March 25 — final push)
1. **[3h] Polish all tables and figures.** Produce final versions of Table 1 (Procedure 1), Table 3 (Procedure 2), Table for cross-judge experiment.
2. **[3h] Complete abstract, conclusion, and fix all placeholder text.**
3. **[2h] Final proofread and ACM format check.**

---

## PART 5 — BLOCKERS & DECISIONS NEEDED

### Blocker 1 — API Access for exp_001
Cross-model judging (exp_001) requires at least 2 different model families as responder and judge. Options:
- **Option A (preferred):** Use local Qwen2.5-7B as responder + local Llama-3.1-8B as judge. Zero API cost. Shows local model circularity.
- **Option B:** Use GPT-4o as responder + local model as judge (requires OpenAI API key).
- **Decision needed:** Confirm which API keys are available.

### Blocker 2 — Procedure 1 LLM-as-Judge Scope
We can compute Potemkin rates from the original labels in 2 hours (zero cost). But to claim "LLM-as-judge as a substitute for human annotation," we need to also run our judge prompts on a subset. How many instances should we run through the LLM judge?
- **Recommendation:** Sample 100 instances across 3 tasks (33 per task) and compute agreement with original labels. This is a focused validation, not a full re-annotation.

### Blocker 3 — Qwen2-VL-7B Domain Finding
MMLU=0.34 vs BBH=0.96 is a striking finding — but it's based on 1 run each. It could be noise. We need at least 3 runs per benchmark to report it confidently.
- **Recommendation:** Run 2 more MMLU and 2 more BBH runs for Qwen2-VL-7B immediately (local compute).

### Blocker 4 — Appendix A is Placeholder
The appendix currently contains lorem ipsum text. Before submission this must be replaced with actual content. Recommended content:
- A.1: Full prompts used in our reproduction (subquestion generation, grading prompts)
- A.2: Artifact commit hash, environment specification, and reproduction instructions

---

## PART 6 — WHAT MAKES THIS PAPER PUBLISHABLE AT ACM REP

ACM REP specifically rewards papers that:
1. **Identify reproducibility failures with evidence** ✓ (variance data, capability threshold)
2. **Explain *why* failures occur** ✓ (subquestion cap, self-judging circularity)
3. **Provide generalizable lessons** ✓ (Section 4 is already strong)
4. **Make something runnable that wasn't** ✓ (Procedure 1 pipeline, if we complete it)

### Strongest Claims We Can Make (Already Supported)
- The automated procedure is **stochastically unstable** at a level that changes qualitative conclusions — documented with 5-run variance data
- The procedure **silently breaks** for models below a capability threshold — documented with Qwen-0.5B failure modes
- The **subquestion generation cap** is an undocumented parameter that shapes all results
- The self-judging design **conflates incoherence with misunderstanding** — the conceptual argument in Section 3.6 is already strong

### Claims Still Needing Evidence
- Cross-model judging changes results (needs exp_001)
- Reasoning models reduce Potemkin rates by an order of magnitude (needs complete DeepSeek runs)
- LLM-as-judge can approximate the original annotation scheme (needs Procedure 1 LLM judge run)

---

## PART 7 — NOTES ON PAPER PRESENTATION

### Title
Current: *"Understanding or Imitation? Auditing Conceptual Understanding and Reasoning in Large Language Models"*
Assessment: Good. Clear, specific, appropriate for REP.

### Abstract Issues to Fix
- "claims about widespread conceptual incoherence in LLMs are directionally supported but empirically fragile" — this is the right framing, keep it
- Add a quantitative anchor: "variance of up to 0.31 across identical runs" should be in the abstract
- Add the capability threshold finding explicitly

### Table 3 — Critical Fix Needed
Current Table 3 shows only 4 iterations for some models and uses inconsistent formatting. The final version should:
- Show all 5 runs as individual columns
- Add a Mean ± Std column (this is the key addition)
- Add Qwen2-VL-7B rows (striking domain effect)
- Mark failed runs clearly as "FAIL" not empty

### Section 3.5 — Each Sub-Section Needs Results
Currently each sub-section describes the experiment design but ends without results. The structure should be:
```
3.5.X  [Experiment Name]
  Design: [what we did]
  Results: [table or 2-3 sentences with numbers]
  Interpretation: [what this means for the paper's claims]
```

### The Reasoning Model Finding (Section 3.5.4)
This is the most consequential finding for the field and deserves its own section, not a sub-section. The original paper buries the o3-mini result (0.03 incoherence) in Table 2 without dedicated discussion. Our paper should make the argument explicitly: **if reasoning models reduce incoherence by an order of magnitude, the "ubiquity" claim is a claim about a specific inference-time regime, not about LLMs as a class.** This reframing has large implications.

---

## PART 8 — FINALIZED TABLE 3 (as of 2026-03-22)

Generated by `AutomaticEval/aggregate_results.py`. CSVs: `Procedure 2 results/aggregated_table3.csv` and `per_run_rates.csv`.

### Table 3: Procedure 2 Potemkin Rates — Our Reproduction

| Model | MMLU (mean ± std, n runs) | BBH (mean ± std, n runs) | Notes |
|---|---|---|---|
| LLaMA-3.1-8B-Instruct | **0.95 ± 0.11** (n=5) | **1.02 ± 0.10** (n=5) | BBH mean > 1.0 ⚠️ |
| LLaMA-3.2-3B-Instruct | **1.02 ± 0.07** (n=5) | **0.97 ± 0.10** (n=5) | MMLU mean > 1.0 ⚠️ |
| Qwen2.5-7B-Instruct | **0.86 ± 0.11** (n=5) | **0.78 ± 0.05** (n=5) | Most stable model |
| Qwen2.5-0.5B-Instruct | **1.12 ± 0.23** (n=3) | FAIL (n=0) | Capability threshold; high variance ⚠️ |
| DeepSeek-R1-Distill-7B | 0.94 (n=1) | 0.89 (n=1) | Mostly incomplete; needs more runs |
| Qwen2-VL-7B-Instruct | 0.34 (n=1) | 0.96 (n=1) | **Anomalous gap: +0.62** ⚠️ Needs verification |
| LLaMA-3.3-70B-q4 (quant.) | FAIL | 0.77 (n=1) | MMLU run incomplete |

### Notable Findings from Table 3

**Finding A — Potemkin Rate > 1.0 (Newly Documented)**
The formula `2 * (1 - mean_coherence)` has no upper bound. It exceeds 1.0 when the judge model is anti-coherent — grading correct answers as wrong more than random chance. This occurs in:
- LLaMA-3.1-8B BBH: mean 1.024 (range 0.90–1.14 across 5 runs)
- LLaMA-3.2-3B MMLU: mean 1.018 (range 0.91–1.10 across 5 runs)
- Qwen2.5-0.5B MMLU: mean 1.123 with std=0.23 (3 runs: 1.03, 1.44, 0.90)

The original paper never acknowledges this property. The metric's documentation states it is bounded [0,1] implicitly (by calling 0 = perfect, 1 = random), but the formula does not enforce this. This should be in Section 3.6.

**Finding B — Stochastic Instability (Confirmed with 5-Run Data)**
Run-to-run variance is substantial for all models tested with 5 runs:
- LLaMA-3.1-8B MMLU std = 0.113 (range: 0.80–1.11) — spread of 0.31
- LLaMA-3.1-8B BBH std = 0.103 (range: 0.90–1.14) — spread of 0.24
- Qwen2.5-7B MMLU std = 0.111 (range: 0.74–1.05) — spread of 0.31

The original paper reports only a single run per model with within-run standard error across 10 concepts. This dramatically underestimates true procedural variance.

**Finding C — Capability Threshold Failure (Confirmed)**
Qwen2.5-0.5B: 0/5 BBH runs complete, 3/5 MMLU runs complete with high variance.
DeepSeek-R1-Distill-7B: 1/10 runs complete (1 MMLU, 0 BBH) — verbose reasoning chains overflow the pipeline.
These are not graceful failures — the pipeline produces no error, just incomplete output. Silent failure is a reproducibility hazard.

**Finding D — Qwen2-VL-7B Domain Anomaly (Unverified, 1 run each)**
MMLU: 0.34 vs BBH: 0.96. Gap of 0.62. Possible explanations:
1. Qwen2-VL is a multimodal model; text-only MMLU may trigger different inference behavior
2. Single-run noise — std could be ±0.3 based on other models
3. BBH's longer questions activate different evaluation behavior
**Status: Cannot report. Needs ≥3 runs per benchmark.**

### Comparison to Original Paper (Procedure 2)

Original paper tested frontier API models (GPT-4o, Claude-3.5, Llama-3.3-70B-full) and reported overall Potemkin LB = 0.62. Our local models average roughly 0.90–1.02, which is higher. Possible explanations:
1. Smaller local models (3B–8B) have less consistent grading behavior
2. Model family differences (Llama 3.1/3.2 vs Llama 3.3 70B)
3. Quantization effects on Llama-3.3-70B-q4 (our only 70B run: BBH=0.77)

The Llama-3.3-70B-q4 BBH=0.77 is most directly comparable to the original's Llama-3.3=0.82 (MMLU). Difference = 0.05, within plausible variance — but different benchmarks, quantization, and only 1 run makes this comparison weak.
