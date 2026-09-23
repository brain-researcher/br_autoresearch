# Adaptive Search for MaleCNS Type Sufficiency

## Authority, history, and claim scale

This is the current local Episode 12 contract. It does not authorize graph
access or compute and does not create a canonical Goal. It is governed by
[`../ADAPTIVE_SEARCH_PROTOCOL.md`](../ADAPTIVE_SEARCH_PROTOCOL.md). Existing
annotation-only inventories are exposed historical feasibility evidence; they
are not untouched confirmation data.

MaleCNS v1.0 contains one male fly. Whole-type and left-to-right/right-to-left
separation can test internal reproducibility within that specimen, but no
amount of adaptive search creates animal-level replication. Every terminal
claim is limited to the pinned release, eligible curated types, and one male.

## Adaptive scientific question

For individual neurons assigned the same curated provider `type`, is one
type-level wiring population predictively sufficient, or do residual outgoing
partner-allocation modes transfer across body sides after accounting for
connection strength, known anatomy, and technical status?

Every trial must fit a capacity-controlled triplet on the same profiles and
folds:

- `T`: one type template;
- `U`: one flexible but numerically unimodal population around that template;
  and
- `M`: two or three residual modes built from the same latent kernel as `U`.

The primary outcome is outgoing weighted partner-type composition. Incoming
composition is a separately locked, later scope analysis using its own
vocabulary and representation; it cannot rescue the outgoing primary result
and is not independent evidence because graph edges overlap.

## Development and final-type roles

Before any connectivity-derived values are opened, eligible complete provider
types are assigned deterministically from annotation-only fields to:

- **development types (80%)** for vocabulary, representation, support,
  nuisance, covariance, component, and regularization search; and
- **final types (20%)** whose source-owned outgoing profiles remain sealed
  until one-shot evaluation.

The split is by whole provider type, stratified only with prespecified
annotation-only scope/support variables, with one frozen seed and assignment
manifest. A neuron, edge, synapse, or side can never be split independently of
its focal type role. Within each development type, left-to-right and
right-to-left transfer supply selection scores; within each final type, both
directions are evaluated once without target-side refitting. The evaluator
reports every assigned final type. A frozen graph-support rule may mark a type
unscorable, but cannot silently drop it; too few scorable final types triggers
the prespecified unresolved or technical terminal rather than post-hoc
replacement.

Because partner identities can include final-type names, endpoint-to-type
lookup is allowed solely to construct the frozen partner vocabulary and
explicit unresolved bins. Final focal profiles, totals, supports, and outcome
summaries remain hidden.

## Bounded declarative grammar

Each trial is a validated configuration interpreted by trusted code; arbitrary
candidate programs are not allowed. The grammar searches:

- **partner vocabulary:** minimum development-type count in `{2,4,8}`, rare
  pooling at mass fractions `{0.01,0.025,0.05}`, explicit unresolved/untyped/
  proofreading bins, self-type handling, and hierarchy depth `{1,2}`, all
  learned from development-owned profiles only;
- **composition:** count-aware log-ratio or low-rank count/composition
  representation with pseudocount `{0.1,0.5,1.0}` and rank
  `{2,4,8,16}`;
- **nuisance:** prespecified strength/exposure, reconstruction/status, and
  known anatomical effects with bounded linear or smooth complexity;
- **observation/covariance:** shared overdispersed-multinomial-compatible
  kernels with diagonal, shrinkage, or low-rank covariance of rank `{2,4,8}`;
- **population family:** the mandatory `T/U/M` triplet, with `M` using
  `K in {2, 3}` and `U` passing a numerical one-mode check; and
- **regularization/hierarchical shrinkage:** penalties
  `{0.001,0.01,0.1,1,10,100}` shared fairly across `T/U/M`, fit only on
  development types. Prespecified anatomical nuisance splines use
  `{0,3,5}` degrees of freedom.

The grammar cannot use provider `group` or `instance` for eligibility,
initialization, fitting, selection, or partner categories; cannot choose
features from final outcomes; and cannot silently renormalize away untyped,
fragment, missing, or out-of-vocabulary mass.

## Objective, constraints, and incumbent

For type `t` and direction `d`, score held-out log predictive density per
eligible incident synapse, average neurons equally within `t,d`, then weight
types and the two transfer directions equally. The primary adaptive objective
is the development estimate of `M - C`, where `C` is the better fair
single-population reference among `T` and `U`.

A feasible incumbent must retain calibration and endpoint mass, show
meaningful improvement in **both** directions, maintain separated and
prevalent modes, match components across sides, pass synthetic recovery, and
avoid concentration in a few types or high-strength neurons. Search maintains
a Pareto archive over predictive gain, calibration, component transfer,
complexity, and influence.

An incumbent is never a scientific terminal. It remains replaceable until
the numeric stop rule fires; a high development score alone cannot authorize
opening final types or naming a biological cell type.

## Stages

1. **Annotation lock:** authenticate side derivation, known-structure
   exclusions, scope strata, eligibility formulas, split seed, and the 80/20
   whole-type assignment before connectivity access.
2. **Synthetic qualification:** test every retained triplet implementation on
   one-template, continuous-unimodal, true-mode, imbalanced-mode, sparse-count,
   missing-mass, and side-artifact fixtures.
3. **Development coverage:** evaluate configurations spanning every partner
   vocabulary family, representation family, covariance class, and `K`.
4. **Adaptive development:** propose ledger-linked successors; each changes a
   declared grammar field and states a hypothesis and falsifier. Every trial
   evaluates the complete fair `T/U/M` triplet.
5. **Outgoing lock:** select one triplet configuration and freeze cohort,
   vocabulary, rank, nuisance/covariance, support, capacities, seeds,
   thresholds, score, and controls.
6. **Audit once:** a trusted evaluator opens final-type source-owned profiles
   once and performs both reciprocal transfers. No final outcome reaches the
   controller.
7. **Incoming scope:** only after the outgoing result is immutable, run the
   separately prespecified incoming workflow. It is reported as correlated
   scope evidence and cannot change the outgoing terminal.

## Required falsifiers and ablations

- exactly **99** fixed-seed fitted-`T/U` parametric-null graphs at real
  development type sizes, strengths, missingness, and side structure, each
  starting from an empty ledger and rerunning the entire deterministic `T/U/M`
  adaptive search;
- strength- and margin-preserving graph randomization within frozen
  type-by-side strata;
- cross-side component-alignment permutation preserving component sizes;
- shuffled partner identities and side labels where structurally valid;
- binary-versus-weighted and rare-pooling/vocabulary/rank sensitivities;
- removal of each nuisance block and each partner-vocabulary block;
- T/U/M capacity-matched synthetic and noise-feature controls;
- leave-one-development-type and leave-one-partner-family influence;
- checks against status, endpoint coverage, side ambiguity, anatomy,
  neuromere/serial/optic position, and missingness; and
- a post-result novelty audit against forbidden provider `group`, `instance`,
  and other published assignments. Matches are rediscoveries, not novelty.

Freeze controller code, proposal-model/version/prompt, sampling configuration,
and the 99-seed manifest before connectivity outcomes. The Monte Carlo rule is
`p=(1 + #{null statistic >= observed}) / 100`, with `p <= 0.05` required. If
the proposal trajectory cannot replay deterministically or readiness profiling
cannot fit all null reruns within the total CPU ceiling, stop before graph
access and revise the contract rather than weakening the null.

## Numeric search contract

- minimum valid trials: **36**;
- maximum valid trials: **96**;
- patience: **16** consecutive valid trials without a preregistered
  meaningful Pareto improvement, active only after trial 36;
- total compute ceiling: **1,000 CPU-core-hours**;
- per-trial ceiling: **32 CPU cores**, **192 GiB RAM**, **12 wall-hours**;
- overall wall-clock ceiling: **120 hours**;
- GPU allocation: **0**;
- every trial must persist all three models, split scores, constraints,
  lineage, resource use, and any invalid/failure reason.

Budget or patience exhaustion triggers lock selection from the feasible Pareto
archive; it is not positive evidence.

## One-shot decision and claim boundary

After lock, the 20% final types are opened once. No vocabulary, support rule,
rank, model, threshold, exclusion, or component matching can change. Valid
terminals are:

- `candidate_ready_residual_modes`: locked `M` exceeds the frozen meaningful
  margin over both `T` and `U` in both transfer directions, modes are separated,
  prevalent, cross-side matched, and all null, provenance, technical, and
  influence gates pass;
- `closed_single_population_sufficient`: calibrated `T/U` is sufficient
  within the frozen margin, or mixture gains are unidirectional, continuous,
  unmatched, null-compatible, or concentrated;
- `closed_unresolved`: final types do not adjudicate the alternatives; or
- `technical_failure`: side, support, provenance, endpoint coverage,
  identifiability, calibration, synthetic recovery, or audit integrity fails.

The strongest permitted positive wording is **bilaterally reproducible
residual outgoing-wiring modes within eligible curated types in this MaleCNS
male** or **the released type aggregation was not predictively sufficient for
these profiles**. It does not establish new cell types, cross-animal or
cross-sex generalization, molecular identity, causality, or behavior. The
whole types are leakage/inference units for this within-male prediction only;
they may not be used to construct an animal-population confidence interval.
The positive class maps to canonical `candidate_ready`; the sufficient and
unresolved classes map to `closed_no_candidate`; integrity failure maps to
`technical_failure`.
