# EP03 — adaptive condition semantics and effect-map geometry

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
in a frozen space and mask. The target is geometry, not effect amplitude.

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
   hash-pinned set of general-purpose text encoders; or a registered
   concatenation of lexical, ontology, and one encoder block.
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

1. **Preflight:** authenticate the 51-group manifest, group participant and
   derivative lineages, freeze text redaction/taxonomy/polarity, and pass
   synthetic leakage and sign fixtures.
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
   primary objective and constraints; hash code, environment, manifests,
   operator DAG, weights, thresholds, and prediction schema.
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

Preflight fixtures, deterministic reruns of an identical trial after a proven
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
