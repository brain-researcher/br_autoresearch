# Adaptive Search for Residual Projection-Morphology Organization

## Authority and history

This is the current local Episode 11 contract. It grants no data, compute,
audit, reward, or canonical state-transition authority. Every prior exposure
remains development evidence. This episode is
governed by [`../ADAPTIVE_SEARCH_PROTOCOL.md`](../ADAPTIVE_SEARCH_PROTOCOL.md)
and becomes executable only after a canonical binding and an explicit
launch decision.

Episodes 09, 10, and 11 use the same SEU-A1876 release. Their cell identity,
biological-group mapping, duplicates, exclusions, development/audit roles,
and first-outcome-access times must live in one shared ledger. Results from
these episodes are correlated sibling analyses, never independent
replications. If an audit group's relevant axonal outcomes are opened by one
sibling before this policy is locked, that group is no longer a fresh audit
group for this episode.

## Adaptive scientific question

After controlling for soma position, cortical depth, coarse anatomy,
reconstruction quality, and acquisition/batch variables available without
using the axonal outcome, which predictive organization transfers best to
held-out biological groups?

- a continuous Gaussian gradient (`G`);
- a flexible but numerically unimodal gradient (`U`);
- reusable residual components without within-component gradients (`D`); or
- a hybrid continuous gradient plus reusable residual components (`H`).

The adaptive search may improve representations and fair implementations of
these four explanations. It may not change the question into unconstrained
clustering, use the released axon-derived `Projection class`, or treat cells,
branches, parcels, or spatial blocks as independent biological replicates.

## Evidence roles

1. **Readiness only:** metadata, archive identities, group mapping, duplicate
   checks, and synthetic fixtures. These can stop the episode but cannot
   support the biological claim.
2. **Development:** verified biological groups assigned to development in the
   shared EP09–11 ledger. All representation learning, grammar search,
   thresholds, and model selection occur here with group-held-out inner
   evaluation. At least **12** complete biological groups are required.
3. **Audit:** complete verified biological groups assigned to the shared
   sealed role before real outcomes are inspected. They are opened once by a
   trusted evaluator after the winner and all decision rules are locked. At
   least **8** complete biological groups are required.

No random-neuron split may substitute for biological groups. If independent
group identity or enough support cannot be verified, the terminal is
`technical_failure`; search depth cannot repair missing replication.

## Bounded search grammar

Every trial is a declarative configuration consumed by a trusted evaluator.
The grammar contains only the following axes:

- **axon outcome vocabulary:** one authenticated CCFv3 occupancy vocabulary
  or a validated provider-arbor sensitivity, with frozen tract-versus-terminal,
  laterality, missingness, and eligibility rules;
- **outcome transform:** occupancy/count transform, topology/path summary
  block, centering/standardization, and development-fit PCA or non-negative
  low-rank dimension under a frozen rank ceiling;
- **anatomical design:** linear, spline, or low-rank smooth functions of
  eligible soma/anatomy covariates, with a bounded interaction order;
- **conditional family:** `G`; `U` as skew-normal, Student-t location-scale, or
  log-concave spline with 4 or 8 basis degrees of freedom; `D`; or `H`, with
  `K in {2, 3, 4}`;
- **covariance and regularization:** diagonal, shared low-rank-plus-diagonal,
  or shrinkage covariance, with bounded rank and penalty values; and
- **capacity-matched ensemble:** optional averaging of at most two already
  evaluated configurations from the same scientific family, using weights
  learned only from development folds.

Effective complexity is estimated by the frozen effective-degrees-of-freedom
routine; compared models must match within `max(2, 10%)`. Candidate code cannot
add a fifth scientific family, an outcome-derived predictor, a new atlas,
arbitrary neural network, or audit-dependent branch.

## Score, constraints, and incumbent

The primary development objective is group-equal, spatial-block-equal held-out
log predictive density. The key contrast is `H - C`, where `C` is the better
development-selected continuous-unimodal reference among `G` and `U`;
`D - C` and `D - H` adjudicate component-only structure. A feasible component
candidate must also pass:

- held-out posterior-predictive separation and minimum prevalence;
- cross-group component matching/reuse;
- approximately matched effective complexity across comparisons;
- likelihood calibration and synthetic recovery; and
- leakage, QC/batch, spatial-null, and influence constraints.

Search maintains a Pareto archive over predictive score, calibration,
separation/reuse, capacity, and stability. The incumbent is always a
nonterminal development object: an improved score is never
`candidate_ready`, and every incumbent may be replaced until the declared stop
rule fires.

## Stages

1. **Identity/readiness gate:** authenticate archives and CCF assets, verify
   cell matching and biological groups, freeze the shared EP09–11 exposure
   ledger, and determine whether a defensible sealed role exists.
2. **Synthetic qualification:** exercise every grammar family on Gaussian,
   skew/heavy-tailed unimodal, component-only, hybrid, unbalanced,
   spatially structured, and missing-data fixtures. Eliminate families that
   cannot recover their declared regime or control false components.
3. **Coverage stage:** run a balanced set spanning every retained model
   family, outcome-representation block, and covariance class.
4. **Adaptive stage:** propose successors from the append-only trial ledger;
   each successor must name its parent, hypothesis, changed fields, and
   falsifier. Failed and invalid trials remain in the ledger.
5. **Lock:** select one configuration and freeze code/environment,
   representation, thresholds, group/block weights, seeds, model capacities,
   nulls, exclusions, and decision table.
6. **Audit once:** the trusted evaluator opens all sealed audit groups once,
   runs the locked configuration and references, emits signed results, and
   prevents candidate access to labels or further tuning.

## Required falsifiers and ablations

- exactly **99** fixed-seed fitted-`C` parametric-null datasets, each starting
  with an empty ledger and rerunning the **entire** deterministic controller,
  to estimate search-induced false multimodality;
- conditional residual randomization and Moran/variogram-matched soma-graph
  surrogates;
- prediction of brain/group, batch, QC, truncation, cell count, and other
  technical fields from learned components;
- capacity-matched noise features and shuffled group/outcome controls;
- full-axon versus validated provider-arbor representation sensitivity;
- topology-only, occupancy-only, anatomy-only, and each feature-block-drop
  ablation;
- leave-one-development-group and leave-one-spatial-territory influence; and
- explicit `G`, `U`, `D`, and `H` comparisons on identical cells and folds.

Freeze controller code, proposal-model/version/prompt, sampling configuration,
and the 99-seed manifest before real outcomes. The Monte Carlo rule is
`p=(1 + #{null statistic >= observed}) / 100`, with `p <= 0.05` required. If
the proposal trajectory cannot replay deterministically, fewer than 12
development and 8 audit groups survive readiness, or profiling cannot fit all
null reruns inside the total CPU ceiling, stop before outcome search and revise
the contract rather than weakening the null.

## Numeric search contract

- minimum valid development trials: **48**;
- maximum valid development trials: **120**;
- patience: **18** consecutive valid trials without a preregistered
  meaningful Pareto improvement, evaluated only after trial 48;
- total compute ceiling: **1,500 CPU-core-hours**;
- per-trial ceiling: **32 CPU cores**, **128 GiB RAM**, and **12 wall-hours**;
- overall wall-clock ceiling: **120 hours**;
- GPU allocation: **0**;
- durable trial artifacts: configuration, parent, hypothesis, fold scores,
  constraints, resource use, and failure class for every attempted trial.

Reaching a budget or patience boundary selects the best feasible incumbent for
locking; it is not scientific success.

## Lock, one-shot audit, and terminals

Audit access is irrevocable for this episode. After it, no representation,
threshold, family, component number, exclusion, or claim may be changed. The
only valid terminals are:

- `candidate_ready_hybrid`: locked `H` exceeds the frozen meaningful
  `H - C` threshold, beats both `G` and `U`, retains a meaningful continuous
  gradient, and passes separation, reuse, null, technical, and influence
  gates on the sealed groups;
- `candidate_ready_component_only`: locked `D` exceeds the frozen `D - C`
  threshold and all component gates, while `H` adds no supported gradient;
- `closed_continuous_unimodal`: calibrated `C` is sufficient within the
  frozen meaningful margin and apparent components fail transfer or null
  gates;
- `closed_unresolved`: the sealed evidence does not distinguish the regimes;
  or
- `technical_failure`: grouping, observability, support, calibration,
  synthetic recovery, or audit integrity fails.

A positive claim is limited to **reusable residual projection-morphology
components** or **conditional residual multimodality** in the pinned release
and eligible groups. It is not evidence of molecular cell types, population
prevalence, causality, or independent confirmation of EP09/10. The two
positive conclusion classes map to canonical `candidate_ready`; both valid
negative/unresolved classes map to `closed_no_candidate`; integrity failure
maps to `technical_failure`.
