# Adopted audit of observed-map prediction

**Proposed October 16; adopted by the scientist October 19, 2026.**
The complete specification below is incorporated by
`scientist_amendment_20261016_observed_map_audit` in `SEARCH_POLICY.yaml`.

This tests whether the completed development predictors transport to previously
held-out people. It **does not distinguish stable biological organization from
stable person-specific measurement structure**: either can improve observed-map
prediction. Adoption changes the scientific scope, audit access and inferential claims;
executing it consumes the original audit for this narrower question.
The original reliable-signal/mechanism question remains unresolved, not impossible.

## Question and competing predictions

Does any of the existing 16 Task-A-calibrated predictors improve prediction of
observed Task-B maps over M0b in the original 12 audit participants, conditional
on the frozen development fit and 188-location support? A positive simultaneous
lower confidence bound would weaken the account that its apparent development
benefit does not transport to these held-out people. A negative simultaneous
upper bound would contradict positive mean benefit for that predictor under this
conditional model. An interval spanning zero is inconclusive; no equivalence or
absence claim follows. Neither outcome adjudicates biology versus stable
measurement effects. This is a prospective predictive evaluation, not an
independent replication or proof of a novel neural mechanism.

## Frozen data roles and complete panel

- Use all original audit people: sub-02, sub-03, sub-04, sub-06, sub-09, sub-10,
  sub-19, sub-22, sub-26, sub-28, sub-29 and sub-30. Keep all 18 B-only conditions,
  all four run partitions, all 16 canonical variants, M0a/M0b and every existing
  matched control and draw. No development winner or reduced panel is selected.
- Refit shared quantities using all 12 original development people only. Freeze
  the existing model recipes, weights, numerical tolerances, control seeds and
  analysis below before audit calibration. Audit anatomy and all 29 Task-A
  conditions may personalize a target; audit data never fit shared quantities.
- Reuse the development-frozen 188-location availability mask and its rebuilt
  spatial models. This defines a conditional observed-map target; it does not
  replace the original 208-location mechanism estimand. Any required missing or
  nonfinite audit value, or unresolved required numerical failure, stops the
  complete evaluation: no person, condition or location is dropped or imputed.
- Freeze every ordinary and control prediction for all 12 people before opening
  any audit Task-B values, including support checks on those values. Evaluate
  once. An engineering repair may resume the unchanged frozen procedure and must
  disclose previous access; it cannot retune predictions or create another
  independent test. The 14 shared Task-B conditions and private reliability
  remain closed.

## Endpoint, multiplicity and decision

Reuse the observed-beta loss: spatially center prediction and observation on the
188 locations; average squared error equally over locations and four partitions,
with equal weights for the seven domains and equal condition weights within each
domain. Compute one loss per person and variant, with no noise correction.
For each of the 16 variants, define `D[i,m] = L[i,M0b] - L[i,m]`.

The sole primary family is these 16 mean gains. Report simultaneous two-sided
95% Bonferroni participant-t intervals:
`mean(D[:,m]) +/- t(11, 1 - 0.05/(2*16)) * sd(D[:,m], ddof=1)/sqrt(12)`.
Declare **positive held-out observed-map prediction benefit** only for variants
whose lower bound exceeds zero. Report all 16 estimates and intervals together,
without ranking classes or selecting a winner. Zero is the adopted predictive
sign threshold; none of the original recovery, adequacy or increment margins is
imported. No threshold change follows from audit results.

These are model-based intervals conditional on the fixed development fit,
selected cohort and support. The original split jointly balanced demographics
and Task-A reliability across all 24 people; independent, exchangeable person
gains are a working assumption, not guaranteed by sealing B. Small-sample
coverage assumes the participant-t model, not an exact design-based guarantee.
The intervals omit training-cohort uncertainty and do not establish
general-population or cross-site transport. Conditions, runs, voxels and control
draws do not increase `n=12`.
Failure to show benefit is underidentified for an absence/equivalence claim;
there is no same-pipeline known-positive calibration at an adopted absence margin.

## Controls and limits

Keep condition permutation, identity derangement, matched random local bases
and matched random warps with their existing fixed draws and matching rules.
Construct the identity cycle from the canonical audit roster using the existing
fixed cycle algorithm/seed, before audit access. True and donor calibrations use
identical development-only sources; neither audit recipient nor audit donor is
in those sources. The audit donor cycle couples recipients, so report its losses
and paired contrasts **descriptively, without naive participant intervals**.
All control comparisons remain descriptive and outside the primary inferential
family; control outcomes cannot select variants, rescue a failure or establish
target-specific biological mechanisms. Average losses over draws, not predictions.

Report one figure showing all 16 participant-gain distributions and simultaneous
intervals, plus the complete numeric/control report and an exact three-line
summary: cohort/target, gain estimates and decisions, and claim/access limits.
Disclose same-account procedural separation. These audit outcomes could not later
serve as untouched confirmation of the original mechanism analysis.

## Prepared implementation and remaining execution

The adopted implementation provides a narrowly scoped audit reader/context,
all-development source fitting, audit-A calibration, prediction freezing and the
one-time B evaluator/report. Original development guards remain in force; the
new route requires the recorded adoption in the episode policy. Source review passed.
Fixture-only tests of access order, complete-panel requirements, loss aggregation
and interval reporting found and resolved an execution-record append bug. All13
passed in coordinated compute step47586225.0, without audit access. Empirical
execution follows the recorded adoption and selected finite job request.
No new dataset, simulation campaign or provider contact is required. Additional
compute alone cannot distinguish stable measurement from biology; that original
question still needs independently supported measurement-error information.

Basis: current `GOAL.md`, `DATASETS.md` (October 16 chronology assumption),
`SEARCH_POLICY.yaml`, `proposed_observed_beta_followup.md` and
`proposed_finite_support_followup.md`. The scientist explicitly adopted this complete scope and evaluation rule on
October 19; original proposal creation had not authorized access.
