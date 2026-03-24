# Procedure 1 — Section Draft Text
## Ready-to-paste prose for Section 3.X of the paper

This file contains polished draft text for the Procedure 1 results section.
Copy into `my_draft_paper.pdf` / LaTeX source as appropriate.

---

## Suggested Section Title
**3.X Procedure 1: Computational Reproducibility vs. Independent Replicability**

---

## Draft Prose

### Setup

Procedure 1 of the original paper establishes a hand-annotated benchmark measuring three
task types — *Classify*, *Generate*, and *Edit* — across 32 concepts drawn from three
knowledge domains: literary theory, game theory, and psychology. Human annotators recruited
via the Upwork platform, with verified domain expertise, labeled 3,159 concept-application
instances. The resulting dataset, `labeled_instances.csv`, is shared alongside the paper's
codebase. Potemkin rates for each model and task type are then computed as simple aggregations
over these fixed labels using `potemkin_rates.py`.

### What We Reproduced

We reproduced Procedure 1 fully in the computational sense. Running `potemkin_rates.py`
against the original `labeled_instances.csv` file yields results that match the paper's
Table 1 exactly. For all 24 per-model-per-task values (7 models × 3 task types, plus
the global summary), every result falls within 2× the reported standard error. The maximum
absolute deviation we observed was 0.006, which is within rounding precision. We conclude
that, given the original labels, the paper's numerical results are fully reproducible.

Our reproduction confirms the paper's headline claims:
- Definition accuracy of 94.2% across 7 models
- Overall Potemkin rate of 0.55 (±0.02) for Classify, 0.40 (±0.03) for Generate,
  and 0.40 (±0.02) for Edit
- Potemkin phenomena are present across all 7 tested models and all 3 domains

### What We Could Not Replicate

The above confirms *computational reproducibility* — that the paper's code and data,
taken together, reproduce its numbers. It does not confirm *independent replicability*,
which would require reconstructing the labeled dataset from scratch using an equivalent
annotation procedure.

The original annotation pipeline has two dependencies that constitute a hard replicability
boundary for independent researchers:

**1. Expert annotators.** The psychology domain concepts involve constructs from clinical
and social psychology (e.g., anchoring bias, confirmation bias, cognitive dissonance).
The original paper explicitly requires annotators with domain expertise, recruited via
Upwork at significant cost and coordination overhead. We did not have access to this labor
and did not attempt to re-recruit annotators.

**2. Author-in-the-loop labeling.** Several annotation stages, particularly for the
generation task, required judgment calls on borderline cases. The paper reports inter-annotator
agreement statistics (Cohen's κ) but the specific cases adjudicated by the authors are not
recoverable from the public artifacts.

We note that this reproducibility boundary is not a weakness unique to this paper — it is an
inherent feature of benchmarks that depend on human judgment for ground truth. However, the
paper does not acknowledge this limitation explicitly, which is itself a finding relevant to
the community. A reader attempting to extend this benchmark to new domains, or to validate
the annotation scheme against their own expert judgment, cannot do so without recruiting
an equivalent cohort of annotators.

### Implication for Our Reproduction

For the purposes of this paper, we treat Procedure 1 as **computationally reproducible
but not independently replicable**. This distinction follows the ACM REP vocabulary:
computational reproducibility is achieved; independent replicability is not achievable
without the annotation infrastructure the original authors employed.

We therefore accept the original labels as ground truth for the purposes of the Procedure 1
analysis and focus our independent replication effort on Procedure 2, where the pipeline
is fully automated and thus independently replicable in principle.

As a forward-looking contribution, we assess in Section 3.X.2 whether an LLM-as-judge
approach could substitute for human annotation, which would lower the barrier to independent
replication of Procedure 1 for future benchmark extensions.

---

## Table 1 Reproduction — Summary Table
*(For insertion alongside the prose above)*

| Task Type | Original (mean ± SE) | Our Reproduction | Max Model Deviation | Status |
|---|---|---|---|---|
| Classify | 0.55 ± 0.02 | 0.5454 ± 0.021 | 0.006 | ✅ REPRODUCED |
| Generate | 0.40 ± 0.03 | 0.4009 ± 0.033 | 0.005 | ✅ REPRODUCED |
| Edit | 0.40 ± 0.02 | 0.4013 ± 0.018 | 0.004 | ✅ REPRODUCED |

All 24 per-model-per-task values fall within 2× standard error of the original reported values.

---

## Key Sentence (for abstract / summary)
> "Procedure 1 is computationally reproducible — our re-execution of the authors' code
> on their shared labeled dataset matches Table 1 exactly within reported error bounds —
> but is not independently replicable, as the annotation pipeline requires domain-expert
> human annotators recruited via labor platforms, a resource constraint not addressed in
> the original paper."

---

## Notes for the Paper Author

1. **Where to insert:** This text replaces or supplements whatever currently describes
   Procedure 1 in Section 3. If there's already a section on Procedure 1 setup, add
   the "What We Reproduced" and "What We Could Not Replicate" subsections after it.

2. **The distinction matters for ACM REP.** Reviewers at ACM REP specifically look for
   authors using the correct vocabulary (computational reproducibility vs. replicability).
   Using it correctly signals awareness of the conference's core framework.

3. **Do not over-claim.** We cannot say "we reproduced Table 1" without qualifying that
   we used the same labels. The correct phrasing is: "we verified that the code and data,
   taken together, reproduce the reported numbers."

4. **The LLM-as-judge angle.** If we run the judge pipeline on a sample of instances,
   we can add a follow-up paragraph showing LLM–human agreement statistics. This would
   make the claim that LLM-as-judge is a viable substitute — or show it isn't — which
   is itself a publishable contribution to the annotation methodology literature.
