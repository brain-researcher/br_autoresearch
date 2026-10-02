# What Do Neural Loss-Aversion Maps Measure?

## The question in plain language

NARPS showed that many teams obtained different answers from the same fMRI
data. EP01 asks which part of that variation can be linked to the cognitive
quantity specified by their first-level models, rather than treating every
pipeline difference as a change in estimation alone.

EP01 tests a sharper possibility:

> When analysts report a neural effect of potential loss, does the conditional
> gain/loss estimand remain stable, or does the map shift systematically when
> valuation, decision-state, and response variables are operationalized
> differently?

The study first separates asymmetric valuation from value-independent rejection
bias in behavior. It then holds preprocessing and inference fixed while changing
only the cognitive first-level specification. Finally, it asks whether the 70
NARPS teams' unthresholded maps occupy the resulting estimand axes in ways that
are predicted by their declared model choices.

This is not another search for a gain or loss region, and it is not a claim
that one pipeline is the ground truth. It asks whether nominally identical
hypotheses became different cognitive estimands before statistical inference
began.

![EP01 cognitive-specification study design](outputs/ep01_conceptual_question-imagegen-v3.png)

The schematic connects choices/RT, the decision-state × response-control
factorial, and within-participant map shifts. Analyst-map ecology is a
dependent descriptive layer from the same participant pool. Brain outlines
and traces are illustrations, not measured maps; no process is isolated by
an icon. [Imagegen prompts](outputs/ep01_conceptual_question-imagegen-v3-prompt.md).

## At a glance

| Element | Fixed role |
| --- | --- |
| Dataset | Public NARPS OpenNeuro `ds001734`, 108 participants, four runs and 256 trials per participant |
| Behavioral question | Valuation asymmetry versus value-independent rejection bias |
| Primary behavioral evidence | Deadline-aware held-out prediction of choice and the full RT distribution |
| Primary fMRI question | Sensitivity of gain/loss maps to decision epoch, deadline-conditioned choice entropy, and response modeling |
| Many-team question | Whether team maps occupy different unitless shape coordinates on frozen specification-displacement axes and whether declared model choices predict those coordinates |
| Cross-validation | Nonoverlapping fit/score run halves, primary split `{1,4}` versus `{2,3}`, evaluated in both directions |
| Biological sampling unit | Participant |
| Analysis-ecology unit | Team/pipeline; teams are not biological replications |
| First kill gate | Behavioral model and parameter recovery plus held-out choice-and-RT prediction |
| Second kill gate | Actual-design fMRI efficiency and cross-run reproducibility of conditional slopes and specification-displacement maps |
| Claim | Cognitive operationalization can be tested as a source of map heterogeneity; no causal or temporal neural mechanism claim |

No empirical result is claimed in this document.

## The remaining candidate contribution

Several neighboring questions are already occupied:

- Zhao, Walasek, and Bhatia (2020) separated valuation asymmetry from a
  pre-evidence rejection bias using behavior and drift-diffusion modeling.
- Sheng et al. (2020) related those components to gaze and pupil signals.
- Wang et al. (2024) reported separable EEG/P3 correlates and explicitly named
  fMRI anatomy as a follow-up.
- Bobadilla-Suarez, Guest, and Love (2020) analyzed subjective value and
  decision entropy in NARPS and reported an in-sample behavioral-neural loss-
  aversion association.
- Brochard and Daunizeau (2024) already used NARPS to study gain-range
  adaptation, OFC geometry, and plastic neural-network models.
- Soch and Haynes already cross-validated generic brain-to-choice prediction
  across the four NARPS runs.
- The original NARPS paper quantified map and decision variability across
  teams, but related it mainly to coarse pipeline features rather than to a
  formal decomposition of the cognitive estimand.

Therefore behavior-only DDM, a generic gain/loss map, generic held-out choice
decoding, range normalization, or another map-similarity analysis is not the
contribution. The candidate advance is a specific connection between these
layers, not their conjunction alone:

1. show that the behavioral mechanisms are recoverable and predict new runs;
2. show how cognitive model specification moves gain/loss maps under an
   otherwise fixed pipeline; and
3. determine whether the ecological many-team maps express the same movement.

The first two layers establish recoverability and within-experiment stability.
The third describes maps from the same participant pool; it does not independently
validate their cognitive interpretation. Whether this connection supports a
paper depends on the eventual discriminating evidence and its uncertainty.

## Behavioral estimands

Accept is the upper boundary and reject the lower boundary. For participant
`s` and trial `t`, positive gain and loss magnitudes are rescaled to
`G*=G/10` and `L*=L/10` without within-version standardization. The primary
drift is

```text
v_st = exp(kappa_s - eta_s/2) G*_t
       - exp(kappa_s + eta_s/2) L*_t
       + c_s.
```

The focal parameters are:

- `eta_s = log(beta_loss/beta_gain)`, the valuation-asymmetry parameter;
- `z_s = logistic(rho_s)`, the starting point as a fraction of the
  accept-reject boundary, where `z < 0.5` favors rejection before evidence
  accumulation; and
- `c_s`, a stimulus-independent drift criterion after stimulus onset.

The distinction between `z` and `c` is scientifically essential. If the task
cannot recover them separately, EP01 will use the broader label
**value-independent decision bias** only if the prespecified bias-family
ensemble beats the no-bias ensemble on held-out data. It will not obtain that
label by relabeling one parameter, and will not call the effect pre-valuation,
pre-decisional, or a starting-point mechanism.

Boundary separation `a`, nondecision time `t0`, overall sensitivity `kappa`,
and a small contaminant mixture are nuisance parameters. Diffusion variance is
fixed to set scale. EI and ER receive partially pooled hyperparameters; they are
not pooled into an unqualified common loss-aversion coefficient.

## Behavioral candidate family

The fixed nested family is:

| Model | Valuation asymmetry `eta` | Starting bias `z` | Drift criterion `c` | Scientific role |
| --- | --- | --- | --- | --- |
| `M0` | fixed 0 | fixed 0.5 | fixed 0 | symmetric reference |
| `MV` | free | fixed 0.5 | fixed 0 | valuation-only account |
| `MZ` | fixed 0 | free | fixed 0 | starting-bias-only account |
| `MC` | fixed 0 | fixed 0.5 | free | post-onset criterion account |
| `MVZ` | free | free | fixed 0 | valuation plus starting bias |
| `MVC` | free | fixed 0.5 | free | valuation plus drift criterion |
| `MZC` / `MVZC` | as named | as named | as named | fitted only if simulation recovers `z` and `c` jointly |

A choice-only logistic model is the predictive benchmark, not a mechanistic
competitor. Fixed bounds are primary. A collapsing-bound/urgency model and a
monotone nonlinear-value model are mandatory robustness competitors because
either can mimic an apparent bias. A mechanism that reverses under those
competitors does not pass.

Those rivals are fixed rather than open-ended families. The urgency model uses
a symmetric exponential boundary `a(t)=a0 exp(-u t)`, `u>=0`, with the same
deadline and contaminant likelihood. The nonlinear-value model replaces each
linear magnitude with a power function,
`(G/10)^gamma_G` and `(L/10)^gamma_L`, with monotonic
`gamma_G,gamma_L in [0.5,1.5]` and scale anchored at `$10`; all other focal
bias terms are nested as above. Rival selection and recovery use the identical
training-only, two-SE, `0.01`-nat rule.

## Response strength, deadline, and missing responses

The primary DDM uses accept/reject sign and RT. Strong versus weak responses
are simultaneous four-button response categories, not an independent
confidence report. They are excluded from the core likelihood.

After fitting the binary model on training runs, an ordinal observation layer
maps model-implied decision certainty to weak/strong responses within each
choice and is scored on held-out runs. A four-category cumulative-link model
is a benchmark. Failure of this layer does not invalidate a binary
choice-and-RT result, but it prohibits a confidence or response-strength claim.

The four-second deadline is part of the likelihood. A response contributes a
first-passage density; a missed response contributes survival probability
through four seconds. Misses are never coded as rejection or silently
discarded. Implausibly fast responses enter the prespecified contaminant
mixture; 100-, 150-, and 200-ms exclusions are sensitivities only.

## Behavioral recovery and kill test

Before interpreting the observed mechanisms, simulate the exact offer
matrices, sample size, four-run structure, deadline, and missingness. Generate
from every candidate family at null, half, one, and twice the smallest effects
of interest:

- `eta = log(1.15)`;
- `|z-0.5| = 0.05`; and
- a participant-specific `c` calibrated at that simulated participant's drawn
  `eta`, `z`, `a`, `t0`, and `kappa`: at an offer with zero drift when `c=0`,
  solve numerically for a five-percentage-point change in four-second
  acceptance probability. The median absolute calibrated `c` defines the
  group-level SESOI.

Recovery uses the same held-out joint choice/RT/censoring score as the observed
comparison. A mechanism set is selected only when its best training-only
predictive model exceeds every nested model lacking that mechanism by both two
standard errors and `0.01` nat per trial; otherwise the simpler mechanism set
wins. Thus “correct family” below means recovery of the generating valuation/
bias mechanism set under this frozen rule, not whichever fitted model has the
largest in-sample likelihood.

The behavioral stage passes only if:

1. the generating mechanism set is selected in at least 90% of simulations at or
   above the smallest effect of interest;
2. false selection of valuation or bias under its null is at most 5%;
3. confusion of `c` with `z` is at most 10%;
4. group-effect relative bias is at most 10% and nominal 95% interval coverage
   is between 90% and 98%; and
5. any parameter used for an individual neural association has simulated
   rank recovery of at least `0.60` and RMSE below `0.75` between-person SD.

If group effects recover but individual ranks do not, group mechanism
comparison may continue while every individual phenotype-to-fMRI analysis is
removed.

If `z` and `c` confuse each other above the allowed rate, generic bias is not
obtained by relabeling the winning parameter. Instead, a training-only stacked
predictive ensemble of all admissible bias models (`MZ`, `MC`, `MVZ`, `MVC`,
plus joint models only if recoverable) is compared with the matched no-bias
ensemble (`M0`, `MV`) on held-out runs using the same two-SE and `0.01`-nat
margin. Only an ensemble-level win licenses **value-independent decision
bias**; otherwise EP01 makes no bias claim.

## Held-out behavioral prediction

Every subject parameter and population hyperparameter is learned from training
runs only. The primary nonoverlap split is `{1,4}` versus `{2,3}`, fitted and
scored in both directions so both halves span earlier and later task periods.
The other two complementary two-run partitions and four leave-one-run-out fits
are robustness analyses, not independent replications.

The primary score is joint held-out log predictive density for choice, RT, and
deadline censoring. Secondary scores are choice log loss, Brier score and
calibration, RT 10th/50th/90th quantiles by choice, nonresponse rate, and the
held-out four-category score.

A more complex mechanism wins only if it:

- beats every scientifically relevant simpler rival by more than two standard
  errors and by at least `0.01` nat per trial;
- has the same advantage sign in EI and ER and in at least three of four
  omitted runs; and
- has a positive median participant-level advantage with a participant-
  clustered 95% interval above zero.

The full two-process neural study proceeds only if valuation plus one
recoverable value-independent bias beats both single-process accounts. A
decisive single-process winner supports a narrower paper. If predictive gains
are smaller than the fixed margin, the mechanisms are not distinguishable and
BOLD analysis stops.

## Fixed fMRI estimand factorial

All arms use the same fMRIPrep series, masks, confounds, HRF, temporal filter,
prewhitening, spatial smoothing, and group estimator. Gain and loss retain
their physical scale and are mean-centered within task version for numerical
conditioning. Serial orthogonalization is disabled. Models are fit runwise
using the original task events.

| Arm | Added cognitive blocks | Interpretation |
| --- | --- | --- |
| `S00` | one unit-height offer-onset impulse with gain/loss modulators, plus separate no-response trials | legacy stimulus-value specification |
| `S10` | `S00` plus cross-fitted choice entropy and a unit-height offer-to-response decision epoch | joint decision-state-block specification |
| `S01` | `S00` plus response-onset regressors for all four observed response categories | response-controlled specification |
| `S11` | `S00` plus both decision and response blocks | full primary specification |
| `R11` | `S11` controls with gain/loss modulation moved to response onset, or the four-second deadline for misses | response-or-deadline timing sensitivity only |

The primary difficulty regressor is binary entropy of the cross-fitted choice
probability,
`H(p) = -p log(p) - (1-p) log(1-p)`, evaluated from the frozen behavioral
model. Here `p = P(accept by 4s) / [P(accept by 4s) + P(reject by 4s)]`, excluding
the contaminant channel, so it is explicitly conditional on a boundary response
by the deadline. Three-outcome accept/reject/no-response entropy is a
sensitivity, not an interchangeable primary definition. Entropy is a zero-
duration offer-onset modulator, mean-centered and scaled by its SD within run
using covariate values only. The accompanying unit-height decision epoch runs
from offer onset to response, or to four seconds for a miss. Negative absolute
drift is a sensitivity only. Because `S10/S11` add entropy and the decision
epoch together, `Delta_D` is a **joint decision-state-block displacement**;
EP01 does not attribute it separately to difficulty or duration. A signed DDM
drift regressor is not inserted alongside gain and loss because it is their
linear combination. The response categories absorb an inseparable mixture of
choice, strength, and finger movement. They are never called pure motor or pure
response-bias signals.

`R11` retains exactly the `S11` trial set, miss indicator, nuisance model, and
gain/loss scaling. Its gain/loss modulators are locked to the observed response
for responded trials and to the four-second deadline for misses. Thus
`R11-S11` changes timing but not trial support. It is labeled a response-or-
deadline timing sensitivity, not a pure commitment effect.

The estimand contrasts are fixed:

- `S11` gain and loss slopes are the fully adjusted conditional gain/loss maps;
- `S00 - S11` is total omitted decision/response sensitivity;
- `Delta_D = 0.5[(S00-S10) + (S01-S11)]` is the order-invariant marginal
  displacement caused by omitting the decision-state block;
- `Delta_R = 0.5[(S00-S01) + (S10-S11)]` is the order-invariant marginal
  displacement caused by omitting the response block;
- `I_DR = S00-S10-S01+S11` tests nonadditivity of those omissions; and
- `R11 - S11` is a response-or-deadline timing sensitivity, not evidence of
  commitment specificity or neural ordering.

The sign convention is omission-positive: positive `Delta_D` or `Delta_R`
means that omitting that block increases the reported gain or positive-loss-
magnitude slope. The four cellwise differences remain prespecified simple-
effect diagnostics. They are not called marginal displacement axes. Although
`Delta_D + Delta_R = S00 - S11`, the interaction determines whether each
simple effect depends on the other block's inclusion.

Every cross-model displacement is formed from matched run-level effect
estimates in common physical units before group aggregation. Group inference
uses participant-paired differences and participant-level resampling; group
`t`/`z` images are never subtracted as though they were effect estimates.

These are **model-specification displacements**, not identified neural process
maps. Adding a decision or response block changes what the gain/loss slope is
conditional on, but does not prove that the difference image is a pure map of
difficulty, response bias, choice, or motor activity. `S11` is correspondingly
called fully adjusted or conditional, never a pure valuation map.

Before BOLD interpretation, actual onsets and RTs are used in design-only
simulations to measure efficiency, variance inflation, and separability. If
gain/loss are not estimable in `S11`, or stimulus- and response-locked blocks
cannot be separated, the relevant specification-displacement claim is
nonidentifiable. The
legacy-versus-full sensitivity comparison may still be reported, but not as a
neural localization of response bias.

The fixed design thresholds are: task-column condition number at most `30`
after nuisance residualization and unit scaling; gain/loss VIF at most `5`;
and `S11` gain/loss efficiency at least `50%` of `S00`. A retained matched
template must have complementary-half robust spatial cosine at least `0.20`, a
participant-bootstrap 95% lower bound above zero, and agreement beyond the
95th percentile of a participant sign-flip null. If qualified regional masks
are used, their gain/loss vector additionally requires cosine at least `0.50`
and a bootstrap lower bound above zero. Failure removes that template; it is
not repaired by a full-data map.

The same condition-number and focal VIF bounds apply to `R11`, whose gain/loss
efficiency must also be at least `50%` of `S11`. Failure removes the timing
sensitivity while leaving an otherwise identifiable factorial intact.

Regional summaries, if used, are limited to the public NARPS vmPFC, ventral-
striatum, and amygdala hypothesis masks frozen before fitting; no region is
selected from the new maps. If those masks cannot be tied unambiguously to the
original hypotheses, the regional gate is omitted rather than replaced by a
post-hoc ROI.

## Cross-run neural falsification

Behavior fitted on one run half generates difficulty regressors for the other
half; the direction is then reversed. fMRI is fitted runwise and combined only
within the declared half.

The neural layer must satisfy all of the following:

1. `S11` continuous whole-brain gain/loss patterns, and regional effects if
   the public masks qualify, agree across the complementary halves;
2. the `S00 - S11` displacement has the same direction across halves and
   within EI and ER;
3. `Delta_D`, `Delta_R`, and any retained `I_DR` template agree across halves
   and within EI and ER;
4. specification-displacement templates used in the team audit are
   reproducible across halves, masks, and smoothness matching;
5. any retained `R11-S11` response-or-deadline timing sensitivity agrees across
   halves on identical trial support; and
6. any participant-level neural-behavior analysis predicts the untouched
   target-half behavior from predictor-half behavior alone versus predictor-
   half behavior plus the neural gain/loss vector, and improves held-out joint
   choice/RT score in both directions.

That optional individual analysis uses dedicated predictor-half neural fits.
For each predictor run, its difficulty regressor is generated from the other
run in the predictor half, with population quantities learned from predictor-
half data only. The target half never helps construct its neural predictors.
Behavioral-family choice, priors, regularization, score construction, and every
other tuning decision are repeated inside the predictor half by swapping its
two runs; they are not inherited from the four-run winner.

The branch runs only if one family containing `eta` and one identifiable bias
parameter wins and both individual parameters pass the rank-recovery gate; a
generic `z/c` ensemble is not mapped to the brain. For participant `i`, both
models use the same winning DDM and target likelihood: first-passage density
for accept/reject plus survival through four seconds for a miss, with the
frozen contaminant mixture. The baseline predicts the target half from task
version and `i`'s predictor-half behavior. The expanded model adds the
standardized predictor-half neural vector `(gain, loss)` as linear covariates
of the population prior means for `eta_i` and for either `logit(z_i)` or `c_i`;
no neural term enters `a`, `t0`, `kappa`, or the contaminant channel.

For each `i`, neural regression coefficients, scaling, priors, and
regularization are learned from other participants' predictor-half data only;
`i`'s predictor-half behavior updates their random effects equally in the two
models. The neural vector has one frozen construction. Within task version,
the two predictor-half runwise `S11` beta maps are equally averaged for gain
and positive loss magnitude. Each map is spatially demeaned inside the fixed,
outcome-blind whole-brain intersection mask and projected onto the matching
unit-norm `S11` group template made from the other participants' predictor-
half maps. The two projection values are standardized using those other
participants only and become `(gain, loss)`. Participant `i` never contributes
to their template or scaling.

ROI scores, displacement maps, `R11`, and alternative templates are not
candidate predictors in this branch and cannot rescue it. The expanded model
is retained only if its mean target log predictive density improves by at
least `0.01` nat per trial and the participant-cluster bootstrap 95% lower
bound exceeds zero in both half directions. Otherwise the entire optional
branch is removed. Gain and loss remain a two-dimensional vector, not an
unstable beta ratio, and a pooled in-sample correlation is never a fallback.

Four runs establish within-session generalizability, not trait reliability.
A failed individual association cannot be rescued by a pooled in-sample
correlation.

## The 70-team estimand audit

The audit uses only unthresholded group statistic maps. Signs are oriented so
a positive statistic corresponds to a positive underlying coefficient for
increasing gain or positive loss magnitude. This restores coefficient
direction; it does not turn `t`/`z` maps into beta maps or comparable effect
units. The hypothesis-rectified signs used in the original NARPS analysis are
not treated as distinct cognitive estimands. Maps are then converted to a
comparable statistic when degrees of freedom permit, placed on a common MNI
grid and intersection mask, and robustly standardized within map. Amplitude is
not compared across incompatible statistics.

The nine requested hypotheses reduce to five base estimands:

- gain-EI (`H1/H3`);
- gain-ER (`H2/H4`);
- loss-EI (`H5/H7`, after coefficient-direction sign restoration);
- loss-ER (`H6/H8`, after coefficient-direction sign restoration); and
- loss ER-minus-EI (`H9`).

Maps that reuse the same file or are numerically equal after coefficient-
direction sign restoration on the same finite support are deduplicated. High
spatial correlation alone is not a deduplication rule. Hypotheses within team
are repeated analyses, not independent observations.

For each base estimand, stable maps from `S00`, `S10`, `S01`, `S11`, and—only
if its design and cross-run gates pass—`R11` define a matched frozen basis: the
fully adjusted conditional slope, the order-invariant `Delta_D` and `Delta_R`
specification displacements, any recoverable `I_DR` nonadditivity, and any
retained response-or-deadline timing sensitivity. Gain-EI team maps use EI
gain templates, gain-ER uses ER gain templates, loss-EI uses EI loss templates,
and loss-ER uses ER loss templates. The H9 basis is formed explicitly from the
ER-minus-EI difference of each loss template; it is descriptive because task
version is between participants. A pooled or cross-estimand template is never
substituted. `S00-S11` is retained as a total legacy-to-full displacement, not
as an additional independent axis.

Numerical coordinate recovery is first tested separately for each matched
basis on synthetic linear combinations and held-out spatial parcels. Team-map
projections are then repeated across run-half templates, masks, statistic
harmonization choices, and smoothness-matched templates.

After unit-norm scaling, a retained projection basis must have condition number
at most `10` and no pairwise absolute spatial correlation above `0.85`.
Synthetic-coordinate recovery requires RMSE at most `0.20` coordinate SD,
absolute bias at most `0.10` SD, and 95% interval coverage between `90%` and
`98%`. Across the declared mask, smoothness, statistic, and half-template
sensitivities, team-coordinate stability requires Spearman correlation at
least `0.80`, sign agreement at least `90%`, and median absolute shift at most
`0.25` coordinate SD. These thresholds are frozen before public team
coordinates are viewed.

The templates and public team maps draw from the same NARPS participant pool;
team-specific exclusions may differ, and no team map is held out from template
construction on a deliberately disjoint sample. Half swapping therefore tests
stability, not independence. Because maps are robustly standardized within
map, the projected coefficients are unitless **shape coordinates**. Synthetic
linear-combination recovery establishes only that those numerical coordinates
can be estimated; it does not validate a cognitive mixture, mixture proportion,
or map content. Such interpretation or generalization requires a genuinely
independent dataset.

The primary metadata predictors are declared first-level quantities:

- event support and duration;
- gain/loss scaling, centering, and orthogonalization;
- RT or decision-duration modeling;
- choice/response modeling; and
- first-level regressor set.

Participant count, exclusion count/rule, available sample-identity summaries,
higher-level covariates/design, group estimator, software, preprocessing
source, mask/grid, statistic type, effective smoothness, motion handling, and
inference family are frozen technical controls. Free-text methods are coded
without access to team projection scores. If exact sample identity or a group-
model field is unavailable, missingness is explicit and the association is
described as conditional on recorded sample-composition proxies; it cannot
isolate cognitive specification from unrecorded sample differences.
Associations are also descriptive because teams self-selected their pipelines.

The primary method analysis is predictive, not a feature-by-feature
significance screen. A multivariate partially pooled ridge model first predicts
the coordinate vector from base-estimand identity and the frozen technical
controls. The
expanded model adds the prespecified cognitive-feature blocks. Teams, with all
their retained maps kept together, form the outer leave-one-team-out folds;
regularization is chosen using training teams only. Added value is the change
in held-out joint log predictive density and multivariate `R^2`, with team-
cluster bootstrap intervals and a team-bundle permutation test. No stepwise
selection is allowed after coordinates are visible. Metadata missingness is
encoded explicitly and checked with a complete-case sensitivity.

The many-team layer stops if specification-displacement templates are too
collinear, synthetic coordinates cannot be recovered, map assignment changes
materially across run halves or smoothness controls, or the cognitive-feature
increment is neither positive nor precise enough for equivalence. Teams are
resampled as analysis units; voxels are never treated as independent
replicates.

A robustness null is allowed only by equivalence, not by `p > .05`. For fMRI,
the 90% participant-bootstrap interval must lie wholly within paired regional
`|d_z| < 0.20`, symmetric whole-brain relative RMS displacement `< 0.10`, and
spatial cosine `> 0.90`. For the team model, the 90% team-bootstrap interval
must lie wholly within `+/-0.01` nat per standardized coordinate for added
leave-one-team-out log score and `+/-0.02` for added multivariate `R^2`.
Failure to exclude both a meaningful effect and these small-effect bounds is
inconclusive. Equivalence applies only to the tested specifications and coded
features.

## EI/ER and sequence structure

EI and ER are partially pooled strata and a transport stress test. They are
not independent replications and are not described causally unless assignment
is documented. Behavioral sensitivities include the common dollar range
`$10-$20` and cluster by participant and the eight counterbalanced sequence
templates. A mechanism or map displacement driven by one task version or one
or two sequences fails the generality claim.

## Strong falsifiers

EP01 is designed to fail or narrow to the lowest surviving tier under any of
the following:

- valuation and value-independent bias are not recoverable at the actual
  trial count;
- the more complex behavioral mechanism does not improve held-out joint
  choice-and-RT prediction by the fixed margin;
- a claimed starting bias is indistinguishable from a drift criterion,
  urgency, or nonlinear value;
- gain/loss slopes become nonestimable after decision and response blocks are
  included;
- conditional gain/loss maps or `S00 - S11` displacement do not replicate
  across run halves;
- the optional individual branch is removed if its neural predictor fails to
  improve untouched target-half behavior or target-half behavior entered
  neural predictor construction;
- EI and ER reverse without a prespecified interaction explanation;
- team-map projections fail synthetic recovery or are unstable to mask,
  smoothness, global-scale, or statistic controls; or
- team metadata are neither incrementally predictive beyond the frozen sample,
  group-model, and map-technical
  features nor precise enough for the frozen equivalence claim.

## Work program

1. Freeze the behavioral family, simulation regimes, original-event semantics,
   and team-map/method inventory.
2. Run behavioral model/parameter recovery and held-out choice-and-RT
   prediction. Stop BOLD work if the mechanism gate fails.
3. Run actual-design fMRI efficiency simulations for `S00`, `S10`, `S01`,
   `S11`, and `R11`.
4. Fit the runwise factorial, perform complementary-half falsification, and
   freeze only reproducible conditional-slope and specification-displacement
   templates.
5. Harmonize and deduplicate the public team maps, verify numerical coordinate
   recovery on synthetic linear combinations, and estimate team-level shape
   coordinates.
6. Test whether blinded cognitive-model metadata predict coordinates beyond
   technical controls, then write the integrated claim at the lowest surviving
   tier.

## Permitted conclusions

EP01 may conclude that, in this task and analysis set, behavioral loss aversion
is better predicted by valuation asymmetry, value-independent decision bias, or
both; that gain/loss maps are sensitive or insensitive to specified cognitive
controls; and that ecological team maps occupy stable, same-data shape
coordinates on specification-displacement axes in association with declared
model choices.

EP01 may not claim:

- that BOLD establishes whether bias occurs before versus during valuation;
- a causal neural mechanism or a uniquely correct brain map;
- pure motor-versus-choice separation from the fixed four-button mapping;
- causal EI/ER effects without documented random assignment;
- reward learning or prediction error in a task without trialwise outcomes;
- stable personality traits from four runs in one session;
- individual-rank conclusions from the 70 group maps;
- that teams or maps are independent biological replications; or
- generality beyond this task, dataset, model family, and public analysis
  ecology.

## Closest prior work

- Botvinik-Nezer et al. (2020), *Variability in the analysis of a single
  neuroimaging dataset by many teams*, doi:10.1038/s41586-020-2314-9.
- Zhao, Walasek, and Bhatia (2020), *Psychological mechanisms of loss
  aversion: A drift-diffusion decomposition*,
  doi:10.1016/j.cogpsych.2020.101331.
- Sheng et al. (2020), *Decomposing loss aversion from gaze allocation and
  pupil dilation*, doi:10.1038/s41598-020-66238-7.
- Bobadilla-Suarez, Guest, and Love (2020), *Subjective value and decision
  entropy are jointly encoded by aligned gradients across the human brain*,
  doi:10.1038/s42003-020-01315-3.
- Wang et al. (2024), *Decomposing loss aversion from a single neural signal*,
  doi:10.1016/j.isci.2024.110153.
- Brochard and Daunizeau (2024), *Efficient value synthesis in the
  orbitofrontal cortex explains how loss aversion adapts to the ranges of gain
  and loss prospects*, doi:10.7554/eLife.80979.
- Soch and Haynes, *Decoding behavioral responses from fMRI data in NARPS*,
  doi:10.1101/2022.03.24.485588.
