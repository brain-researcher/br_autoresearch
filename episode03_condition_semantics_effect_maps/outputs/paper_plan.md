# EP03 paper plan — How much contrast information survives beyond task family?

Design milestone: 2026-09-30. This plan changes the argument, not the frozen
operative search contract. No current-contract experiment or audit has run.

Amendment 2026-10-02: the [scope review](scope_novelty_review_20261002.md)
identified an ambiguity in complete-prediction polarity. The current Goal and
policy now specify canonical orientation for both the family comparator and
semantic residual, before any current-contract map computation. The loss,
model families, budgets, evidence roles, and audit decision are unchanged.

## The paper in one sentence

Determine whether a methods-only signed description of an experimental
comparison predicts normalized group-Z-map geometry in a different dataset
**beyond a task-family prototype**, or whether apparently specific predictions
are explained by family recognition, semantic neighbors, or dataset metadata.

## Why this is a scientific question

Task labels compress experimental meaning. "Working memory" includes load,
stimulus-category, and control-condition manipulations that need not produce
the same spatial contrast. Predicting a common family pattern is useful, but
it is different from recovering information about the signed comparison. A
map that looks plausible does not distinguish these possibilities.

The paper therefore asks what explanatory resolution the description supports.
The prototype is not a weak benchmark to defeat: it is a rival account of why
text-to-map prediction works. The signed residual must earn its contribution
on independent groups and survive the existing falsifiers. The conclusion
concerns predictive information, not neural causation or cognitive ontology.

## Closest work and the actual overlap

| Verified primary source | Already established | Consequence for EP03 |
| --- | --- | --- |
| [NeuroQuery, Dockès et al., eLife 2020](https://elifesciences.org/articles/53385) | Free-text mapping from literature and a contrast-description illustration | Free text, prediction, and descriptions of experiments are not novel. |
| [Text2Brain, Ngo et al., 2022](https://arxiv.org/abs/2208.00840) | Transformer text-to-map generation, IBC/HCP contrast evaluation, and reliability comparisons | Contrast prediction and reliability-sensitive assessment are not novel. |
| [NeuroConText, Meudec et al., MICCAI 2024](https://papers.miccai.org/miccai-2024/560-Paper3550.html) | Contrastive text/map association and IBC contrast reconstruction | Semantic/contrastive representation alone is not a contribution. |
| [NeuroConText, Ghayem et al., Imaging Neuroscience 2026](https://pubmed.ncbi.nlm.nih.gov/41852941/) | Joint retrieval/reconstruction, rich text and short-query augmentation, NeuroVault contrast evaluation | Neither retrieval nor testing short statistical-map labels is new. |
| [NeuroVLM, Hammonds et al., bioRxiv v3, 2026-07-01](https://www.biorxiv.org/content/10.64898/2026.02.06.704508v3) | Generative/contrastive neuroimage-language framework evaluated on statistical and coordinate-derived maps | Statistical-map text-to-image generation is not a first claim. This is the Hammonds framework, not the same-name AD diagnosis model. |

EP03's proposed distinction is the **incremental-information estimand**:
family prototype versus prototype plus signed semantics, independently
grouped and equally weighted, with semantic-cluster exclusion and
complete-procedure nulls. This is an inference from the reviewed sources,
not a verified global priority claim. Strong dataset grouping by itself is
good evaluation practice, not sufficient paper novelty.

Published systems have different target/training meanings and may have seen
related literature. A numerical comparison against their released weights
would not automatically isolate the scientific question. This milestone adds
no external model, training objective, or new architecture to the grammar.
Any future such benchmark requires a separate eligible-data/training-overlap
decision and an explicit contract amendment if outside the registered family.

## Existing design, told as an evidence sequence

1. **Establish the rival explanation.** Fit the train-fold global map and
   task-family prototype. Display family support and independent group counts;
   577 contrast/program rows are not the sample size.
2. **Measure incremental detail.** Evaluate the registered semantic readouts
   on independent outer groups with the unchanged normalized target, mask,
   loss, group weighting, and primary relative-MSE gain.
3. **Ask if the gain is genuinely signed and transferable.** Apply polarity
   reversal, semantic-cluster exclusion, duplicate-lineage exclusion,
   within-family full-procedure permutation, and the separate exchangeable
   negative-control contrast. A failed mandatory falsifier prevents promotion.
4. **Identify alternative explanations.** Report lexical, ontology, encoder,
   random-feature, metadata, nuisance, ensemble, influence, and reliability
   analyses already required by the grammar and falsifier contract.
5. **Separate discovery from confirmation.** The exposed 51 groups support
   adaptive development only. One selected, locked pipeline must later face a
   genuinely independent, previously sealed corpus. No such corpus exists now.

This is a presentation sequence for the existing experiments, not permission
to skip branch coverage, shorten the 30–72-trial search, remove the required
40% post-coverage falsifier fraction, or substitute a favorable diagnostic for
the primary decision rule.

## Target and leakage choices that carry the argument

- The target is a mean-centered, unit-L2, sign-preserving **group Z statistical
  contrast map**. "Effect-map geometry" does not mean an effect-size map.
- All family prototypes, semantic vocabularies/transforms, map bases, nuisance
  fits, and predictors are training-fold objects. Families and semantic
  clusters come from methods, not target maps or model scores.
- A fixed methods/ontology ordering defines canonical target `Z*=s Z` and
  canonical description `t*`. The baseline predicts `s p_family`; the model
  predicts `s[p_family+r(t*)]`. Reversal negates the complete predictions,
  without canceling the prototype. Both receive the same polarity convention;
  thus the comparator is family plus orientation, not a sign-blind average.
  This algebraic invariant is not a semantic validation result. The required
  transfer and complete-procedure null comparisons must establish the latter.
- Dataset/participant/derivative lineage defines exclusion. Paper identity,
  filenames, author names, result language, reported coordinates, anatomical
  labels, and target-derived taxonomies cannot be predictors.
- Semantic-neighbor exclusion tests interpolation dependence. Passing it is
  not evidence for universal transfer to unseen task families; the registered
  leave-one-family-out diagnostic has its own limitations.
- A frozen general-purpose encoder can carry prior textual knowledge.
  Redacting episode inputs does not prove its pretraining never encountered
  related publications. Report that limitation rather than claiming perfect
  removal of every form of prior knowledge.

## A null is useful only when it can discriminate

The old report of approximately 2.2% gain over a global mean is a feasibility
note, not a current measurement, meaningful margin, or success target. A family
prototype is a stronger and scientifically different comparator. Historical
repeatability gains do not quantify residual-map reliability.

Before candidate-discriminating search, the existing contract still needs
group support, an executable evaluator, scientifically meaningful thresholds,
and group-level uncertainty to be specified without using current outcomes to
choose favorable decisions. This plan supplies no new numerical margin. A
near-zero family loss can destabilize a relative gain; its behavior under the
authenticated data belongs to that evaluator/inference specification, not to
post-hoc trimming after a score is seen.

If the primary uncertainty is wide, the conclusion is **unresolved**, not
"text adds nothing". If a sufficiently precise comparison shows little useful
gain for the registered representations, the conclusion is a bounded
prototype-dominated result, not mathematical sufficiency of family labels.
This does not invent an equivalence test or change the terminal rule.

| Scientific result, if observed | Paper claim | Existing runtime boundary |
| --- | --- | --- |
| Valid gain and every required falsifier passes on fresh audit | Conditional semantic increment beyond family | Eligible for `candidate_ready`, subject to the frozen audit inference rule |
| Technically valid comparison, no eligible candidate or failed audit | No supported semantic candidate; prototype dominance only if precision supports it | `closed_no_candidate` |
| Development search completed, fresh audit absent | Adaptive evidence only; prospective claim unavailable | `search_exhausted_no_audit` |
| Source independence/sign/space invalid, or unresolved execution failure | No scientific estimate from this run | `technical_failure` |
| Leakage or forbidden post-audit adaptation | Invalid evidence | `policy_violation` |

## Figure and planned empirical displays

![EP03 conceptual question](ep03_conceptual_question.png)

**Concept figure, available now:** the rival prototype and semantic-residual
accounts, dataset-group transfer boundary, existing falsifiers, and missing
fresh audit. All visual patterns are abstract illustrations; check icons name
tests, not passing outcomes. The illustrative category contrast is within a
memory task. Both branches use the shared methods-defined orientation:
`s p_family` versus `s[p_family + r]`. Whole-prediction reversal is an
implementation invariant, not evidence that semantics adds information.

Empirical displays are planned, **not generated or required as new gates**:

1. Per-independent-group prototype/model losses and paired gains, with
   uncertainty and family support rather than only a pooled spatial score.
2. Polarity, neighbor-exclusion, and complete-procedure permutation comparisons
   for the same promoted pipeline, including failures.
3. Reliability/influence and representation ablations, clearly separated from
   the primary comparison.
4. A development versus prospective-audit evidence panel only if a fresh
   audit is actually acquired and opened under the lock.

## What would make this paper substantial

A sharper question and a better diagram do not make the study publication
ready. A substantive paper needs a supported estimate of incremental detail,
interpretable rival-model/null outcomes, independent group support, and a
defensible exposure account. If group coverage or precision is too limited,
the honest product may be a methodological pilot rather than a full research
paper. A leaderboard win from the exposed corpus alone is insufficient.

## Present state and next action

No current-contract analysis has run; no map payload was read for this
milestone. The 51-group handoff is not provisioned, the frozen encoder/cache
and evaluator are not qualified, and there is no prospective audit source.
Next in-scope scientific action is to bind/authenticate the read-only methods,
group-lineage, and map-eligibility handoff already required by `DATASETS.md`,
then complete the existing pre-search inference decisions. Do not add generic
SHA, schema, provenance, or preflight work as new scientific prerequisites.
