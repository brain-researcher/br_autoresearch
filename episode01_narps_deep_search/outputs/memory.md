# Research memory

> Client-maintained workspace projection only. This is not canonical Brain
> Researcher memory and must not be promoted outside this workspace without a
> separate scientist-requested action.

## Active scientific lessons

- NARPS is scientifically distinctive because it combines a complete
  gain-by-loss grid, choice, RT, four response-strength categories, four runs,
  BOLD, and an ecological many-team result archive. It should not be treated as
  just another 108-participant gain/loss dataset.
- Conventional loss-aversion estimates can mix asymmetric valuation with a
  value-independent tendency to reject. Zhao et al. established this behaviorally;
  behavior-only repetition is not the EP01 contribution.
- A DDM starting point and a constant drift criterion can imitate one another.
  If exact-design simulation cannot distinguish them, the only defensible label
  is value-independent decision bias.
- The four-second deadline is data, not missingness to trim. No-response trials
  require a survival likelihood.
- Strong/weak accept/reject was one simultaneous four-button response. It may
  support held-out response-strength prediction, but it is not automatically an
  independent confidence report.
- Four runs permit honest within-session prediction and target-isolated
  behavior-BOLD tests. They do not establish trait test-retest reliability.
- A neural predictor for behavior in half B cannot use a difficulty regressor
  learned from half B. The optional incremental brain-behavior branch therefore
  uses predictor-half-only inner fitting, participant-out neural regression,
  one frozen task-version-matched LOPO whole-brain `S11` gain/loss expression
  vector, and an explicit DDM prior-mean mapping for recovered `eta` and
  `z`/`c`, or is omitted.
- fMRI cannot directly localize a temporally pre-valuation starting point in
  this task. There is no prestimulus manipulation, and the response categories
  are inseparable from four right-hand finger responses.
- The correct fMRI question is whether gain/loss maps survive decision duration,
  deadline-conditioned choice entropy, and response-phase controls under an
  otherwise fixed pipeline.
- The `S00/S10/S01/S11` cells form a 2x2 model factorial. Decision and response
  specification-displacement axes are marginal omission effects averaged over
  the other block; cellwise differences are simple-effect diagnostics, not
  order-invariant axes.
- `R11` keeps every `S11` trial: responses lock to response onset and misses to
  the four-second deadline. `R11-S11` is response-or-deadline timing
  sensitivity, not a pure commitment or temporal-order result.
- A signed DDM drift must not be added beside gain and loss when it is their
  linear combination. Nonlinear difficulty or entropy supplies the separable
  decision-state quantity.
- Public NARPS statistic maps must be returned to a common coefficient-
  direction sign. The original hypothesis-directed rectification makes H5 and
  H7 point in their predicted regional directions even though they correspond
  to opposite directions of the same loss-magnitude contrast. Restoring sign
  does not turn `t`/`z` maps into beta maps or comparable effect units.
- H1/H3, H2/H4, H5/H7, and H6/H8 can reuse one base map within a team. They are
  regional decisions, not independent cognitive maps, and must be deduplicated.
- The curated NARPS v2.0.1 archive already contains concatenated unthresholded
  team maps and rich first-level method descriptions. It supports an ecological
  estimand audit without reproducing 70 pipelines.
- Teams are analysis units, not biological samples. Method-coordinate relations
  are observational because teams chose their own pipelines.
- Team-map models must control recorded participant counts/exclusions/sample
  summaries and higher-level designs/estimators. If exact sample identity or
  group-model fields are missing, cognitive-method associations remain
  potentially confounded by unrecorded sample composition.
- Team maps and reference templates draw from the same NARPS participant pool,
  with possible team-specific exclusions. Their within-map-standardized
  projections are unitless shape coordinates, not cognitive mixture
  proportions or independent validation.

## Active paper architecture

The paper has three dependent evidence layers:

1. **Behavioral identification.** Recover the nested DDM family at the actual
   design and require held-out prediction of choice, RT, and deadline misses.
2. **Conditional neural estimands.** Compare `S00`, `S10`, `S01`, `S11`, and
   `R11` under one fixed technical pipeline; retain only conditional-slope and
   specification-displacement directions that reproduce across nonoverlapping
   run halves.
3. **Analysis ecology.** Verify numerical projection on known synthetic linear
   combinations, describe coefficient-direction-aligned and deduplicated team
   maps with same-data shape coordinates, and test blinded cognitive-method
   features beyond technical controls.

Later layers cannot rescue failed earlier identification. Behavior alone is
not paper-level. A successful fixed-pipeline layer with a failed team layer can
support a narrower controlled-estimand paper.

## Approaches not carried forward

- Behavior-only DDM decomposition.
- Generic subjective-value, decision-entropy, or brain-to-choice decoding.
- EI/ER range normalization, OFC RSA, or plastic-network adaptation.
- Simple time-on-task or recent-offer history as the headline.
- A new gain/loss activation map.
- Generic pipeline map similarity, clustering, consensus, or thresholded
  decision variability.
- Pipeline-induced participant re-ranking from the 70 public group maps.
- Causal interpretation of self-selected team methods.

## Retired EP01 methods lessons

The earlier smoothing and derivative-sufficiency audits remain useful but are
not active EP01 claims:

- Smoothing scale profiles and pre/post-GLM timing are heavily saturated.
- Fixed linear smoothing can commute with a fixed common temporal estimator;
  prewhitening and data-dependent operations can break pipeline replay.
- Derivative sufficiency is operation-specific. A point estimate may be
  recoverable while uncertainty after spatial mixing is not.
- `K Sigma K^T`, COPE/VARCOPE sharing, fixed-model summary sufficiency, and
  ordinary contrast-map group OLS are established prior art.
- An operation-indexed derivative envelope with a practical covariance repair
  remains a possible methods backlog project, not the formal EP01 question.

The chronological design history is preserved in `experiment_log.md`; the
[directions not pursued](paper_plan.md#directions-not-pursued) retain the
scientific reasons for the earlier pivots.

## Open implementation details

- Instantiate the behavioral priors, contaminant mixture, solver, and recovery
  replicate count without changing the model family or pass rules.
- Verify whether original task code identifies participant display-side and
  response-button mappings; otherwise omit laterality and finger-specific
  claims.
- Decode the many-team metadata into blinded cognitive features and quantify
  missingness before map coordinates are inspected.
- Freeze the common fMRI nuisance set, smoothing, prewhitening, and ROI masks
  before any `S00`--`R11` outcome is compared.
- Preserve participant-level uncertainty for biology and team-level uncertainty
  for analysis ecology; never use voxels as replicates.

## Promotion status

- Not requested.
