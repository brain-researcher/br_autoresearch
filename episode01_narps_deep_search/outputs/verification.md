# Verification projection

> Client-maintained record of scientific and mechanical checks. It does not
> itself establish an empirical result or broader campaign state.

## Current state

- Active contract: NARPS cognitive-estimand audit, policy v6
- Literature collision audit: complete at design level
- Participant-level asset observation: complete at design level
- Curated many-team archive observation: complete at design level
- Behavioral recovery and observed fits: not run
- fMRI design simulation and observed fits: not run
- Team-map projection: not run
- Current scientific result: none

## Design verification

| Check | Status | Scope or limitation |
| --- | --- | --- |
| Saturated directions excluded | pass | Range normalization/OFC, subjective value/entropy, generic decoding, simple time drift, smoothing, and derivative sufficiency are not current headlines |
| Paper-level object | pass at specification level | Recoverable behavior, conditional fMRI estimands, and ecological team-map shape coordinates form one dependent argument |
| Original event source | specified | `*_events_ORIGINAL.tsv` is required; transformed `events.tsv` is not the source |
| Complete participant BOLD | observed | 432 fMRIPrep 21.0.2 MNI 2-mm series, confounds, and masks are present |
| Behavioral deadline | specified | No-response trials use survival through four seconds and are neither dropped nor recoded as rejection |
| Response strength | specified | Strong/weak is secondary ordinal response strength, not assumed confidence |
| `z` versus `c` rivalry | specified, not run | Starting bias is named only if recovered; generic bias requires a multiplicity-aware bias-family ensemble to beat the no-bias ensemble |
| Behavioral independence | specified, not run | Primary `{1,4}` versus `{2,3}` split is fit and scored in both directions |
| Urgency/nonlinear-value rivals | specified, not run | Symmetric exponential boundary and anchored power-value forms are frozen; mechanism must survive both |
| Fixed fMRI factorial | specified, not run | `S00/S10/S01/S11/R11` share every non-cognitive pipeline step; decision/response axes are marginal factorial effects rather than order-dependent simple effects |
| `R11` trial support | specified, not run | Responses lock to observed response and misses to the four-second deadline on the identical `S11` trial/nuisance support; separate efficiency/VIF gates apply |
| BOLD temporal limit | pass at specification level | No before-versus-during valuation claim is permitted |
| Response/motor confound | pass at specification level | Four response-category regressors are labeled a choice/strength/finger mixture |
| fMRI design efficiency | open | Actual-onset/RT simulation must pass condition-number `30`, VIF `5`, and relative-efficiency `0.50` bounds |
| Neural cross-run evidence | open | Conditional gain/loss and specification-displacement maps must reproduce across complementary halves and EI/ER |
| Brain-behavior target isolation | specified, not run | Optional DDM extension uses only task-version-matched LOPO whole-brain `S11` gain/loss expression scores, maps them to recovered `eta` and `logit(z)`/`c` prior means, trains participant-out inside the predictor half, and leaves the target half untouched; ROI/alternative predictors cannot rescue it |
| Many-team maps | observed | Curated v2.0.1 contains concatenated unthresholded group maps for hypotheses 1--9 |
| Many-team method metadata | observed | Archive includes detailed first-level regressor, RT, response, preprocessing, software, statistic, and smoothing fields |
| Coefficient-direction sign restoration | specified, not run | Positive statistics are oriented to positive underlying magnitude coefficients; this does not create beta/effect units |
| Base-map deduplication | specified, not run | Region hypotheses reusing one base map are not counted independently |
| Projection recovery | open | Matched bases must pass collinearity, synthetic-coordinate error/coverage, and half/mask/smoothness stability bounds |
| Technical map controls | specified, not run | Sample size/exclusions/identity proxies, higher-level design/estimator, mask, smoothness, scale, statistic, software, and preprocessing remain explicit; missing sample/model fields limit attribution |
| Correct uncertainty units | specified | Participants are biological units; teams are analysis units; runs, hypotheses, maps, and voxels are repeated measurements |

## Exact checks required before interpretation

1. Candidate behavioral families and focal parameters meet the recovery rules
   in the exact four-run design.
2. The winning mechanism exceeds the fixed held-out joint choice/RT margin and
   calibrates deadline misses.
3. Any starting-point claim survives the `c`, urgency, and nonlinear-value
   rivals.
4. `S11` gain/loss effects remain estimable under the actual timing design;
   `R11-S11` is retained only if its separate efficiency/VIF gate passes on
   identical trial support.
5. Conditional gain/loss slopes, `S00-S11`, the marginal decision/response
   omission axes, and any retained interaction or timing sensitivity reproduce
   across nonoverlapping complementary halves before defining team-map
   templates.
6. Public statistic maps are restored to common coefficient-direction signs
   and deduplicated before projection.
7. Joint projection recovers synthetic shape coordinates and is stable to
   run-half, mask, smoothness, statistic, and spatial-holdout checks.
8. Method features either add held-out information beyond technical controls
   or their 90% interval lies inside both frozen equivalence margins; an
   imprecise null is reported as inconclusive.

Failure of an earlier check removes the dependent later claim. A favorable
team projection cannot rescue an unrecoverable behavioral mechanism or a
nonidentifiable fMRI design.

## Superseded checks

The derivative-bundle lattice, covariance-repair calibration, raw-rerun
external matrix, and smoothing-operator checks belonged to policy v5. None was
executed. They remain historical design material only and are not current
verification requirements.

## Scientific boundary

The design can support computational mechanism comparison, sensitivity of
gain/loss maps to declared cognitive controls, within-session cross-run
generalizability, and descriptive analysis-ecology associations. It cannot
establish temporal pre-valuation anatomy, a causal neural mechanism, a stable
trait, independent replication across teams, causal effects of self-selected
methods, or individual pipeline rankings from group maps.
