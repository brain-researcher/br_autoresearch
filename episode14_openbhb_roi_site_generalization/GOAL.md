# CPU-First Adaptive OpenBHB ROI Search Under Acquisition Shift

## Authority and history

This is the current local Episode 14 contract. It does not authorize use of
public validation labels, create a leaderboard submission, or bind a canonical
Brain Researcher program. Pinned data preparation and quick baselines are
exposed development history. The episode follows
[`../ADAPTIVE_SEARCH_PROTOCOL.md`](../ADAPTIVE_SEARCH_PROTOCOL.md) only after a
canonical registration and explicit launch decision.

## Adaptive scientific question

Within a CPU-only, precomputed-FreeSurfer setting, which fold-safe
neuroanatomical representation and shallow predictor best lowers age error on
unseen `siteXacq` domains while preventing acquisition setting from becoming
more decodable than it is under a fixed baseline?

This is a constrained prediction search, not a claim that scanner information
can be causally removed. The primary domain is `siteXacq`, because a new
acquisition protocol at a familiar physical site is still a domain shift.

## Development and audit roles

- **Development:** the 3,227 official training participants and three pinned
  RAMP robustness splits. Each split contains fit, internal-test (seen
  `siteXacq`), and external-test (unseen `siteXacq`) roles. The overlapping
  splits are robustness views, not independent replications.
- **Locked-probe selection:** at most 12 development-Pareto configurations are
  re-evaluated with the frozen five-fold multinomial logistic site probe. The
  quick probe uses three folds; both probes use fold-fit standardization,
  L2 multinomial logistic regression with `C=1`, balanced class weights,
  `lbfgs`, `max_iter=2000`, and hash-seeded folds. This remains development
  selection, not the final audit.
- **One-shot audit:** 757 public-validation participants—362 internal and 395
  external—whose age/site labels are available only to a trusted evaluator
  after one winner is locked. The audit runs with network egress disabled and
  without the upstream raw/source cache mounted.

Public validation labels are upstream-public but deliberately absent from the
candidate workspace. Looking them up, reconstructing participant identities,
or using them for selection invalidates the audit.

Before launch, an exposure ledger must name every human, process, cache, and
prior artifact that could have accessed public validation age, `siteXacq`, or
row identities. If a candidate/controller actor or its accessible context has
seen a row-level join or score, the 757 rows are development-exposed and cannot
serve as this audit; no-network execution cannot restore freshness.

## Declarative pipeline grammar

Candidates are data-only configuration records interpreted by a trusted
pipeline builder. Arbitrary edits to `candidate.py`, arbitrary imports, and
candidate-defined evaluation are forbidden. The finite grammar contains:

- **atlas input:** Desikan, Destrieux, or row-aligned concatenation;
- **neuro features:** allowed raw ROI values; atlas-derived bilateral means
  and differences; region/measurement-family aggregates and contrasts; and
  bounded interactions among those development-defined blocks;
- **scaling:** standard or robust scaling fit on the current fit role;
- **reduction:** none, PCA, supervised PLS, or random Fourier features with
  enumerated ranks/components and fit-only state;
- **site-free invariance:** no removal, or projection away from a bounded
  number of acquisition-predictive directions learned using **fit-role**
  `siteXacq` labels; `transform(X)` receives no site label and applies only the
  already learned map;
- **age model:** ridge, elastic net, Huber regression, or ridge on random
  Fourier features, with enumerated regularization; and
- **ensemble:** a nonnegative blend of at most three already evaluated locked
  development pipelines, with weights learned only from development
  out-of-fold predictions.

All imputers, scalers, feature constructors with learned state, supervised
reducers, invariance transforms, models, and ensemble weights fit only on the
current fit role. ComBat, site centering, target normalization, or any other
transform requiring an internal/external/audit participant's site label or
statistics is prohibited. Test-time adaptation and transductive fitting are
prohibited even without labels.

Identifiers, session, age, site, study, sex, scanner variables, global tissue
volumes, and QC variables are not model features. Raw MRI, pretrained
participant embeddings not in the pinned bundle, GPU models, and external
participant-level labels are outside the grammar.

## Metrics, feasibility constraints, and incumbent

For every development split, retain:

- external-test age MAE in years (primary predictive metric);
- internal-test age MAE;
- internal-test `siteXacq` balanced accuracy under the declared probe; and
- `external_MAE * site_BAcc ** 0.3`, minimized only alongside its components.

Because held-out rows overlap across the three views, repeated predictions for
one participant are averaged before any participant-level uncertainty summary.
The three split scores remain robustness views and are never treated as three
independent cohorts or multiplied into an effective sample size.

A configuration is feasible only if, against the frozen baseline, it:

1. improves mean external MAE by at least **2%**;
2. increases mean site balanced accuracy by no more than **0.01** absolute;
3. improves the combined value; and
4. improves external MAE in the same direction on at least **two of three**
   development splits.

Search uses a constrained Pareto archive over mean and worst-split external
MAE, site decodability, combined value, internal MAE, rank, and runtime. The
incumbent is explicitly nonterminal and replaceable. Quick-probe success is
never comparable with locked-probe success and never authorizes audit access.

## Stages

1. **Integrity and fixed baselines:** reproduce input hashes, row/header
   alignment, age-median, raw ridge, and PCA-ridge baselines; freeze evaluator,
   seeds, split roles, metric signs, and leakage tests.
2. **Stage 1 coverage — 24 configurations:** run a balanced design covering
   both atlases, raw versus neuro-derived blocks, no-reduction versus linear
   reduction, no-removal versus fit-only site-direction removal, and linear
   versus shallow nonlinear predictors. Coverage trials are retained even if
   poor.
3. **Adaptive quick-probe search:** use ledger history to propose one grammar
   change at a time or a declared interaction test. Continue to at least 48
   valid trials and at most 120, subject to patience and resource ceilings.
4. **Locked-probe tournament:** select at most 12 diverse feasible Pareto
   candidates without outcome peeking; rerun them with the frozen expensive
   five-fold multinomial logistic probe on development only.
5. **Winner lock:** choose exactly one pipeline by the prespecified ordering
   and freeze configuration, code/environment, feature headers, folds, seeds,
   preprocessing state rules, thresholds, probes, controls, and audit command.
6. **Audit once:** in a no-network clean process without raw cache, mount audit
   features to the locked candidate and labels only to the trusted evaluator;
   score all 757 rows once and emit a sealed result. No post-audit model
   selection, blending, threshold change, or rerun is permitted.

## Required falsifiers and ablations

- age-median, raw ridge, and PCA-ridge fixed references;
- shuffled-age and shuffled-site controls;
- a regression test that deliberately leaky preprocessing is detected and
  differs from the fold-fit implementation;
- Desikan-only, Destrieux-only, and concatenation-minus-each-atlas ablations;
- raw, bilateral, region, measurement-family, reduction, and invariance-block
  ablations;
- representation rank, finite-value, variance, and collapse checks;
- fit-site-label removal versus a site-label-free raw representation;
- per-siteXacq and age-range error summaries;
- leave-one-development-split influence and worst-split reporting;
- ensemble-member and blend-weight ablations; and
- confirmation that `transform` is invariant to removal of all test/audit site
  metadata because it never receives those labels.

## Numeric search and resource contract

- Stage 1 coverage: **24 valid configurations**;
- minimum total valid development trials: **48**;
- maximum total valid development trials: **120**;
- patience: **20** consecutive valid trials without a preregistered meaningful
  Pareto improvement, active only after trial 48;
- locked-probe tournament: **at most 12** configurations;
- total compute ceiling: **500 CPU-core-hours**;
- per-trial ceiling: **16 CPU cores**, **64 GiB RAM**, **4 wall-hours**;
- overall wall-clock ceiling: **120 hours**;
- GPU allocation: **0 GPU-hours**;
- failed, duplicate, infeasible, and invalid trials count in the append-only
  ledger but only valid unique configurations count toward 48–120.

Budget or patience exhaustion selects a feasible incumbent for the tournament;
it does not imply scientific success.

## Winner ordering, audit, and terminals

Among locked-probe-feasible candidates, select the winner lexicographically by
lower mean external MAE, lower worst-split external MAE, lower combined value,
lower site balanced accuracy, and then lower complexity/runtime. If no
candidate is feasible, lock the best fixed baseline and close without opening
the audit unless a separately registered null audit is explicitly authorized.

For the audit, refit the locked candidate and fixed baseline once on all 3,227
development rows. The trusted evaluator computes paired absolute-error
differences on the 395 external rows and frozen five-fold site-probe
predictions. It then makes **10,000** fixed-seed hierarchical bootstrap draws,
resampling `siteXacq` groups and then participants within sampled groups.
`candidate_ready` requires all development gates plus:

1. external relative MAE reduction of at least **2%**;
2. the one-sided 95% bootstrap upper bound for candidate-minus-baseline
   external MAE is below `0`;
3. site-probe balanced-accuracy increase is at most **0.01**; and
4. its one-sided 95% bootstrap upper bound is at most **0.01**.

If both point rules pass but either uncertainty rule fails, the result is
`closed_unresolved`. These thresholds, probe folds, bootstrap seed manifest,
and baseline predictions are locked before labels are mounted.

After the 757-row audit is opened, the valid terminals are:

- `candidate_ready`: the locked winner meets every exact development and audit
  margin/interval rule without leakage or collapse;
- `closed_no_generalization`: development success does not reproduce on the
  one-shot audit, or no candidate clears the development feasibility gates;
- `closed_unresolved`: the frozen audit decision margin is inconclusive; or
- `technical_failure`: data identity, evaluator integrity, leakage firewall,
  metric computation, or audit isolation fails.

Success supports only prediction from the pinned OpenBHB ROI derivatives under
the declared acquisition shifts. It does not establish a biological aging
mechanism, causal scanner removal, clinical utility, disease generalization,
private-leaderboard rank, or superiority on raw MRI. The positive class maps
to canonical `candidate_ready`; valid failure-to-generalize and unresolved
classes map to `closed_no_candidate`; integrity failure maps to
`technical_failure`.
