# What does a task description add beyond knowing its task family?

Two experiments can both be called "working memory" while asking different
questions: increasing memory load is not the same contrast as changing the
category of remembered images. A convincing-looking predicted brain map might
nevertheless recover only the common task-family pattern. EP03 asks whether
the **signed experimental comparison**, described without its results, supplies
reproducible spatial information beyond that pattern in another dataset.

The scientific distinction is **family recognition versus contrast-specific
information**. The paper is not another demonstration that language can be
mapped to a brain image. It tests how much detail that mapping actually earns,
and when apparently detailed predictions collapse to a task prototype.

![EP03 conceptual question](outputs/ep03_conceptual_question.png)

Both branches share the methods-defined orientation: predictions are
`s p_family` and `s[p_family + r]`. Reversal of the complete prediction is an
implementation invariant, not semantic evidence. Orange check icons label
planned comparisons, not passed tests; all patterns are schematic.

*Conceptual schematic, not empirical results. The illustrative working-memory
contrasts vary load or remembered image category; "faces minus places" here
means a category comparison within a memory task, not a claim that every
face/place experiment belongs to that family. The two model branches and
possible conclusions are hypotheses. Check icons name tests to perform, not
tests already passed. The prospective audit has not been acquired.*

## The comparison that would answer the question

For an independent dataset group left out of fitting, compare the same
sign-authenticated, unthresholded group Z maps with two predictions:

- a task-family prototype estimated from training groups only;
- that prototype plus the residual predicted from methods-only signed
  condition descriptions, using the bounded operator grammar below.

Both predictions use the same outcome-blind contrast orientation convention.
The prototype and residual are learned in canonical orientation, and the
requested polarity is applied to their **complete sum**, not to the residual
alone. The explicit rule below is a prospective 2026-10-02 design repair;
no current-contract computation or outcome opening has occurred.

Give each independent dataset group equal weight. Contrast counts and voxels
do not supply independent replications. The primary score remains the frozen
group-weighted relative MSE gain over the family prototype. Global-mean,
lexical-only, metadata-only, and capacity-matched random-feature controls
explain whether a gain actually requires condition semantics.

Three checks make the interpretation sharper than a plausible map: reversing
the comparison must reverse its predicted sign as an implementation invariant;
excluding semantic neighbors
must not erase the claimed direction of transfer; and permuting descriptions
within task family must not reproduce the gain through the complete search
procedure. These are existing required falsifiers, not new experiments.
Algebraically enforced sign reversal alone is not evidence that text recovers
contrast-specific neural information.

## What is and is not new

[NeuroQuery (2020)](https://elifesciences.org/articles/53385) already predicts
maps from free text and illustrates task-contrast descriptions.
[Text2Brain (2022)](https://arxiv.org/abs/2208.00840) already evaluates IBC and
HCP contrast descriptions and relates performance to map reliability.
[NeuroConText (2024)](https://papers.miccai.org/miccai-2024/560-Paper3550.html)
already reconstructs maps from IBC descriptions; its
[2026 extension](https://pubmed.ncbi.nlm.nih.gov/41852941/) combines text/map
retrieval and reconstruction and evaluates NeuroVault contrast descriptions.
[Hammonds et al.'s NeuroVLM (2026, preprint v3)](https://www.biorxiv.org/content/10.64898/2026.02.06.704508v3)
already supports text-to-neuroimage generation and evaluates statistical maps
as well as coordinate-derived images.

These verified overlaps rule out "first text-to-brain model", "first
contrast-description prediction", and "first evaluation on statistical
maps". EP03's proposed contribution is a **specific evaluation of incremental,
sign-sensitive information beyond task family across independent datasets**,
with semantic-neighbor, metadata, and complete-procedure null controls. This
is a proposed distinction, not an exhaustive priority claim or proof of
novelty. The frozen grammar contains lightweight predictors and frozen text
representations; importing or training a new generative architecture is not
authorized by this rewrite.

## Outcomes worth distinguishing

| Outcome | Defensible interpretation |
| --- | --- |
| Residual gain transfers and required falsifiers pass | Methods-only contrast semantics adds predictive geometry beyond family recognition, within supported tasks and this pipeline. |
| The family prototype remains competitive with adequately precise estimates | Little practically useful residual information is recovered by the registered representations and sample; report the bounded comparison, not a universal claim that task family is sufficient. |
| Gain disappears after neighbor exclusion, polarity, metadata, or permutation controls | The apparent transfer is not an eligible semantic contribution; identify the failed alternative explanation. |
| Estimates are imprecise or map reliability/family support is insufficient | Inconclusive, even if the mean gain is zero or negative. |

The prototype interpretation requires uncertainty narrow enough relative to a
scientifically meaningful gain and usable residual-map reliability. Those
quantities are not established by the historical feasibility reports and
must be decided under the existing pre-search inference contract. Failing to
promote a candidate is not an equivalence test. With no fresh audit, all these
comparisons remain adaptive development findings, not prospective confirmation.

See [the paper plan](outputs/paper_plan.md) for the scientific argument and
evidence sequence, and the [current review](outputs/scope_novelty_review_20261002.md)
for the prior-work boundary and sign-consistent amendment.

## One orientation rule for the baseline, model, and evaluator

Before target-map loading, the methods-only taxonomy assigns each comparison
one canonical ordering and a polarity `s` in `{+1,-1}`. The rule uses documented
experimental roles and a fixed ontology: for example task minus control,
higher minus lower load, and a declared order for category comparisons. It
must not choose the ordering from observed maps, fitted scores, filenames, or
which sign yields a stronger family prototype. Ambiguous comparisons are
reported as orientation-unresolved under the existing sign-eligibility rule.

Let `Z` be the normalized, source-sign-authenticated map. Fit `Z* = s Z` in
canonical orientation, and build each training-fold family prototype `p_f`
with equal dataset-group weight. The candidate learns the residual `Z* - p_f`
from the canonically ordered methods description `t*`. For the requested
comparison, evaluate

```text
family prediction = s p_f
semantic prediction = s [p_f + r(t*)].
```

Apply every candidate operator, nuisance adjustment, and ensemble combination
in canonical coordinates before this single final polarity restoration.
No unflipped intercept or correction may be added after multiplying by `s`.

Reversing only the contrast keeps its family, canonical description, support,
and fitted objects fixed and changes `s` to `-s`. Both full predictions then
negate exactly. Reversed copies remain the same comparison and dataset group;
they are not new training examples or independent observations. Baseline and
candidate both receive polarity as a coordinate convention, so the comparator
is precisely **task family plus the shared orientation convention**, not a
sign-blind prototype. No condition-specific fitted information enters that
baseline.

The global-map control uses the same canonical fit and sign restoration.
Within-family description permutations operate on the canonical descriptions
without changing the map's polarity or group identity. Report oriented-family
support, prototype loss, and residual reliability: cancellation or weak family
support cannot be advertised as a semantic discovery. No zero baseline or
antisymmetrization that silently cancels the prototype is substituted.

## Status, protocol, and exposure boundary

This is the current local episode contract. An explicit scientist instruction
naming EP03 may start bounded episode work. This file alone grants no authority
to access protected outcomes, submit a candidate, or change Brain Researcher
state. It follows
[`../ADAPTIVE_SEARCH_PROTOCOL.md`](../ADAPTIVE_SEARCH_PROTOCOL.md); this file
adds the EP03-specific scientific and search contract.

The 51 reported NeuroEffect dataset groups were already exposed during
question and method development and are development-only. Reorganizing the
episode does not make them fresh; an independent prospective audit source is
still required.

## Scientific question

Can methods-only, signed condition semantics predict the normalized geometry
of a completely held-out dataset's group fMRI effect map beyond a broad
task-family prototype, and which semantic representation and map-readout
mechanism generalize rather than memorize dataset or publication identity?

The unit of independence is an authenticated `independent dataset group`, not
a contrast, map, voxel, paper release, or processing program. The primary
outcome is a sign-preserving, mean-centered, unit-L2 unthresholded group Z-map
in a frozen space and mask. "Effect map" is shorthand for this statistical
contrast map: Z is not an effect-size estimate. The target is its normalized
spatial geometry, not effect amplitude, the magnitude of a neural mechanism,
or coordinate-reporting probability. This normalization does not make
between-study sampling or processing differences disappear.

## Evidence roles

- **Adaptive development and selection:** all 51 historically exposed dataset
  groups. They support grouped nested cross-validation, mechanism discovery,
  falsification, and model selection only. Every transform is fit inside its
  training fold. Out-of-fold scores on these groups are internal evidence.
- **Prospective audit:** a future, independently authenticated set of new
  dataset groups whose map outcomes, result text, and candidate-comparison
  scores were unavailable during question, grammar, code, and policy design.
  It must be sealed before ingestion and opened once after configuration lock.
- **No proxy audit:** holding back some of the 51 exposed groups, renaming an
  OpenNeuro release, or using another derivative of the same participants does
  not create fresh evidence.

The audit cannot run until the new corpus has enough independent groups and
task-family support to evaluate the locked model and primary comparator. The
exact minimum and support table must be frozen from a prospective power and
identifiability calculation, not invented in this contract.

## Bounded scientific operator grammar

One trial is a complete, declarative pipeline assembled only from the
following registered operators. Arbitrary code edits and unconstrained prompt
changes are not trials.

1. **Methods-only semantic input:** signed standardized contrast template;
   frozen lexical n-grams; frozen ontology indicators; one of a small,
   prespecified set of general-purpose text encoders named by stable model
   version; or a registered concatenation of lexical, ontology, and one
   encoder block.
2. **Semantic transforms:** no reduction, train-fold PCA, train-fold PLS, or a
   signed compositional transform that separately encodes positive condition,
   negative condition, and their difference.
3. **Map target:** family-prototype residual in either a train-fold voxel PCA
   basis or a registered atlas/network basis. Basis rank is selected from a
   finite schedule bounded by training groups and reliability.
4. **Predictor:** ridge, elastic net, partial least squares, kernel ridge with
   a registered linear/cosine/RBF kernel, or reduced-rank multi-output
   regression. Capacity is matched when comparing semantic blocks.
5. **Nuisance handling:** no adjustment, training-only residualization, or
   stratification for authenticated sample size, statistic type, smoothing,
   site/scanner, and map smoothness. Target-derived nuisance labels are
   forbidden.
6. **Weighting:** equal dataset-group weight is mandatory; within-group
   contrast weights may be uniform or reliability-capped under a registered
   cap learned without held-out outcomes.
7. **Ensemble:** at most two already evaluated, complementary pipelines with a
   convex blend weight selected inside grouped cross-validation. An ensemble
   cannot introduce a new representation after finalist freeze.

Dataset IDs, paper titles, authors, filenames, results/discussion language,
coordinates, ROI names, activation terms, target maps, and map-derived family
labels are forbidden predictors. Vocabulary, basis, normalization, nuisance
fit, and hyperparameters are always training-fold objects.

## Objective and constraints

For held-out group `d`, let `L_family,d` be contrast-averaged voxel MSE for the
frozen task-family prototype and `L_model,d` the corresponding model loss.
The primary score is the equally weighted group mean of
`G_d = 1 - L_model,d / L_family,d`.

Both predictions are evaluated on the same normalized target and held-out
groups. The family taxonomy is made without maps or scores, its prototype is
fit inside training folds, and only represented/eligible families contribute
to the primary comparison. Unsupported task families belong in the declared
leave-one-family-out diagnostic, not a silently redefined primary test.
The loss ratio's behavior and uncertainty under the actual group support must
be addressed by the pre-search evaluator/inference specification; this
writing milestone supplies neither a new tolerance nor a claim of precision.

A trial is feasible only if it also:

- preserves contrast-polarity equivariance;
- improves more than a frozen minimum fraction of development groups rather
  than relying on one group;
- remains directionally positive after canonical semantic-cluster exclusion;
- beats its complete-procedure within-family text-permutation distribution;
- is not explained by metadata-only or trivial lexical controls; and
- passes leakage, duplicate-lineage, finite-map, and reliability gates.

Spatial-correlation gain, leave-one-family-out performance, calibration, and
reliability-normalized gain are diagnostic objectives. They cannot rescue a
failed primary objective. Search reports a Pareto archive, but one executable
pipeline must be chosen before audit.

## Adaptive loop

1. **Readiness:** verify the 51 groups, participant and derivative lineages,
   map source/space compatibility, and condition/contrast target
   observability; freeze text redaction/taxonomy/polarity; and pass the
   synthetic leakage and sign fixtures. Repeat a check only after a concrete
   failure or a relevant input or logic change.
2. **Coverage trials:** run the family prototype, lexical, ontology, one frozen
   encoder, voxel-basis, and atlas-basis anchors so every major operator family
   is tested before exploitation.
3. **Adaptive search:** propose one falsifiable change at a time, evaluate it
   with identical grouped folds and seeds, append its hypothesis, parent,
   configuration, resource use, score, constraints, and failure reason to the
   trial ledger, and update the nonterminal incumbent/Pareto archive.
4. **Successive fidelity:** inexpensive inner-fold screening may eliminate
   clearly dominated trials; any promoted finalist must be rerun over all
   eligible development groups and required nulls. Low-fidelity scores never
   become terminal evidence.
5. **Finalist stress test:** rerun the incumbent and at most three challengers
   with cluster exclusion, duplicate exclusion, influence analysis, nuisance
   controls, reliability strata, and fixed seed replication.
6. **Configuration lock:** select exactly one pipeline using the prespecified
   primary objective and constraints, then write an immutable, write-once
   record of stable input, code, environment, and model version identifiers,
   the operator DAG, weights, thresholds, prediction fields, and deterministic
   tie rules.
7. **One-shot audit:** after verifying the prospective corpus remains sealed,
   execute the locked pipeline and comparator once. No retraining choice,
   threshold change, fallback model, or candidate swap is permitted from audit
   outcomes.

The incumbent is deliberately nonterminal: a good development score never
ends the episode before the minimum trial coverage, required falsifiers,
patience rule, finalist stress test, and audit gate are satisfied.
At least 40% of valid post-coverage trials must be falsifiers, ablations,
negative controls, influence guards, synthetic recovery, or direct
replications. Promotion also requires at least two outcome-adaptive successor
cycles and two recorded incumbent/challenger decisions.

## Mandatory falsifiers and ablations

- full-procedure within-family permutation of signed condition descriptions;
- polarity reversal with an expected prediction sign reversal;
- metadata-only and trivial lexical-only models;
- removal of all training examples in the held-out contrast's semantic
  cluster;
- duplicate-release/participant and near-duplicate-map exclusion;
- task-family-only, global-map, and capacity-matched random-feature baselines;
- leave-one-development-group influence and reliability-matched analysis;
- block ablations for ontology, lexical, encoder, nuisance, and ensemble
  contributions; and
- a negative-control contrast whose labels are exchangeable within family.

Failure of a mandatory falsifier invalidates promotion; it is not averaged
away by a better headline score.

## Budget and stopping

- minimum valid scientific trials: **30**;
- maximum valid scientific trials: **72**;
- improvement patience after the minimum: **12** consecutive valid trials
  without a material constrained-primary improvement;
- CPU ceiling: **1,500 core-hours**;
- GPU ceiling: **0 GPU-hours**;
- wall-clock ceiling: **120 hours** from first scientific trial;
- finalist ceiling: **4** pipelines including the incumbent;
- audit openings: **1**;
- maximum parallel CPU cores: **32**;
- per-trial memory ceiling: **128 GB**;
- scratch-storage ceiling: **1,000 GB**.

Readiness fixtures, deterministic reruns of an identical trial after a proven
infrastructure failure, and audit execution do not count as new hypotheses.
They still consume resource ceilings and remain in the append-only ledger.

## Terminal classes

- `candidate_ready`: the locked pipeline passes every development constraint
  and the one-shot prospective audit improves the family prototype under the
  frozen inference rule without a mandatory falsifier failure.
- `closed_no_candidate`: the complete search and audit are technically valid
  but no locked pipeline meets the scientific decision rule, or the locked
  pipeline fails audit.
- `search_exhausted_no_audit`: the bounded development search completes but no
  eligible prospective audit corpus exists; this is useful deep development,
  not confirmation.
- `technical_failure`: authenticated inputs, independence, map sign/space,
  legal use, or executable evaluation cannot be established after bounded
  recovery.
- `policy_violation`: audit leakage, post-audit adaptation, undeclared
  operators, or outcome-dependent relabeling invalidates the run.

For canonical outer status, `candidate_ready` maps to `candidate_ready`;
`closed_no_candidate` and `search_exhausted_no_audit` map to
`closed_no_candidate`; and `technical_failure` or `policy_violation` map to
`technical_failure`.

## Claim boundary

A positive audit would support cross-dataset prediction of normalized group-map
geometry from methods-only condition semantics beyond task family within the
represented tasks and pipeline. It would not show that text causes neural
activity, recover individual-brain representations, predict effect amplitude,
generalize to unseen task families, or establish population neuroscience from
voxels as independent samples. Development-only success on the exposed
51-group corpus must be labeled adaptive exploratory evidence.

The 2026-09-30 rewrite changes the scientific framing and paper narrative,
not the operative target, score, registered operators, mandatory falsifiers,
budgets, promotion requirements, or one-shot audit rule. No current-contract
experiment has run. Any future change to those decisions requires an explicit
scientific amendment before the affected outcomes are used; prose cannot
retroactively amend an exposed execution.
