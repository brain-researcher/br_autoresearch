# EP01 paper plan: what do neural loss-aversion maps measure?

This is a prospective paper architecture. It reports no empirical result.

![EP01 cognitive-specification study design](ep01_conceptual_question-imagegen-v3.png)

Choices/RT motivate competing accounts; the fixed-pipeline factorial tests
conditional map shifts within participants. The analyst archive is a
same-participant descriptive layer, not independent confirmation. All drawing
elements are schematic. [Imagegen prompts](ep01_conceptual_question-imagegen-v3-prompt.md).

## Proposed title

**What do neural loss-aversion maps measure? Behavioral mechanisms and
analysis-dependent conditional estimands in NARPS**

More methods-facing alternative:

**Same hypothesis, different conditional estimands? A computational audit of
the NARPS analysis ecology**

The first title is preferred if the behavioral mechanisms and fMRI
specification-displacement maps both pass. The second is more honest if the
many-team layer is strong but participant-level neural alignment is weak.

## One-sentence contribution

The paper tests whether variation among NARPS gain/loss maps reflects only
technical pipeline variability or also a deeper ambiguity in what analysts
operationalized as valuation, decision difficulty, choice, and response.

## Why this paper matters

NARPS established that plausible teams can reach different maps and decisions
from the same data. EP01 focuses on one specific source of that variation:
first-level choices can change the target itself. An impulse
offer model, an RT-duration model, a model with no response phase, and a model
that absorbs decision difficulty do not necessarily estimate the same
cognitive quantity.

Loss aversion makes that ambiguity substantive. A high rejection rate may
reflect asymmetric gain/loss valuation, a value-independent tendency to reject,
or both. If these processes are conflated behaviorally and their neural
correlates are absorbed differently by first-level models, “pipeline
variability” partly becomes **estimand variability**.

The paper is credible only if it connects all three levels:

```text
recoverable behavioral mechanisms
              |
              v
fixed-pipeline cognitive model factorial
              |
              v
ecological projection of 70-team maps
```

Behavior alone repeats prior DDM work. The candidate contribution is a
controlled test of cognitive-specification effects, followed by a descriptive
test of whether those effects organize the public analyst maps. Behavioral
and map half-splits test within-experiment prediction and stability; the team
layer reuses the same participants and is not independent validation of the
cognitive interpretation. Combining the layers does not by itself establish
novelty; their specific connection must survive the stated tests.

## Introduction logic

1. Many-analyst studies show that analytic choices change conclusions.
2. A shared verbal hypothesis can still be operationalized as different
   conditional estimands; test that specific possibility rather than attributing
   an unexamined assumption to earlier studies.
3. In mixed gambles, observed loss aversion is itself mechanistically
   ambiguous.
4. NARPS combines dense gain-by-loss behavior, RT, four runs, BOLD,
   and an ecological archive of many analysts' maps and method descriptions.
5. Therefore NARPS can test whether analyst heterogeneity includes cognitive-
   estimand drift rather than only technical estimation noise.

## Prior-work boundary

The paper does not claim the following as new:

- behavioral separation of valuation and starting-point bias;
- gaze, pupil, or EEG correlates of that separation;
- subjective value or decision-entropy mapping in NARPS;
- OFC range adaptation in the EI/ER task versions;
- generic cross-run brain-to-choice decoding;
- generic map similarity, clustering, sign rectification, t-to-z conversion,
  smoothness estimation, or pipeline-choice association in NARPS; or
- the fact that fMRI conclusions vary across analysts.

The exact remaining gap is whether a recoverable computational decomposition
can define stable map directions, whether controlled cognitive specifications
move maps along those directions, and whether public analyst maps exhibit the
same structure.

The NARPS public result archive v2.0.1 already supplies team maps, decisions,
confidence, method reports, harmonization, smoothness, and generic similarity
outputs; those operations are not contributions of EP01. Lefort-Besnard et al.
(2025) address dependent multiverse maps and same-data consensus evidence, so
consensus aggregation is also outside the proposed gap. HCP Multi-Pipeline and
related reliability multiverses have subject-level multi-pipeline measures;
NARPS public group maps cannot support comparable individual-rank claims.

### Directions not pursued

The 2026-09-29 literature assessment retired smoothing as a headline.
[Mikl et al. (2008)](https://doi.org/10.1016/j.mri.2007.08.006) compared pre-GLM
and post-GLM contrast smoothing and discussed OLS versus ReML/prewhitening
noncommutation; [Hagler, Saygin, and Sereno (2006)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1785301/)
compared smoothing time courses with coefficient maps.
[Worsley et al. (1996)](https://pubmed.ncbi.nlm.nih.gov/20408187/) established
scale-space inference, while [Ball et al. (2012)](https://pubmed.ncbi.nlm.nih.gov/21404370/)
and [Sacchet and Knutson (2013)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3618861/)
showed scale-dependent results and reward-domain localization. A `{4,6,8,10}`
smoothing profile, pre/post timing comparison, or smoothing commutator is
therefore insufficient as the paper's contribution.

Derivative sufficiency was also retired as the headline. Fixed multilevel
GLM reproduction from parameter estimates and covariances (Beckmann,
Jenkinson, and Smith, 2003), transformation covariance `K Sigma K^T` (Rohde
et al., 2005), ordinary group OLS on contrast maps under stated conditions
(Mumford and Nichols, 2009), precision-weighted group analysis (Chen et al.,
2012), and rich statistical-result sharing through NIDM-Results (Maumet et al.,
2016) already provide the core precedents. Germani et al. (2025) identify
false-positive risks when reusing heterogeneously processed contrasts. An
operation-indexed derivative envelope with a useful covariance repair remained
technically plausible, but the scientist selected the cognitive-estimand
question above. These earlier directions remain inactive.

## Study 1: behavioral adjudication

### Question

Does a valuation-only account predict new NARPS runs, or is a recoverable
value-independent bias required?

### Analysis

Fit the fixed nested DDM family from `M0` through `MVZC`, with joint `z/c`
models permitted only after exact-design recovery. Include the four-second
deadline through a survival likelihood, retain no-response trials, and compare
against urgency and nonlinear-value rivals.

The urgency rival uses a symmetric exponential boundary
`a(t)=a0 exp(-u t)`, `u>=0`. The nonlinear-value rival replaces linear
magnitudes with `(G/10)^gamma_G` and `(L/10)^gamma_L`, with each monotone
exponent constrained to `[0.5,1.5]` and scale anchored at `$10`. Both use the
same deadline/contaminant likelihood and the same training-only two-SE,
`0.01`-nat selection rule as the primary family.

The `c` SESOI is calibrated per simulated participant at that participant's
drawn nuisance values and zero-drift offer to change four-second acceptance by
five percentage points. Mechanism recovery uses the frozen held-out two-SE and
`0.01`-nat rule, not in-sample likelihood. If `z` and `c` remain confusable, a
training-only stacked bias-family ensemble must beat the matched no-bias
ensemble by that same margin before the broader label “value-independent
decision bias” is allowed.

The primary nonoverlap split is `{1,4}` versus `{2,3}` in both directions.
All model and population parameters are fit without the held-out half. Choice,
RT distribution, deadline misses, and the secondary four-category observation
layer are scored prospectively on that half.

### Decisive result

The full paper advances only if a model containing valuation asymmetry plus one
identifiable value-independent bias beats both single-process accounts by the
fixed held-out margin and transports in direction across EI and ER.

A valuation-only or bias-only winner narrows the paper. Failure to distinguish
the mechanisms stops the BOLD decomposition.

## Study 2: controlled fMRI estimand factorial

### Question

When preprocessing and inference are fixed, how much do gain/loss maps change
when decision duration, deadline-conditioned choice entropy, and the response
phase are modeled?

### Analysis

Fit five runwise arms:

| Arm | Scientific contrast |
| --- | --- |
| `S00` | legacy offer-onset gain/loss model |
| `S10` | add deadline-conditioned binary choice entropy and offer-to-response epoch |
| `S01` | add four response-onset category regressors |
| `S11` | include both control blocks |
| `R11` | move gain/loss modulation to response onset, or the four-second deadline for misses, as a timing sensitivity |

The principal map is the fully adjusted conditional `S11` gain/loss estimate.
Map displacement from `S00` to `S11` is decomposed into order-invariant
marginal decision- and response-block omission effects plus their interaction;
the four cellwise changes remain simple-effect diagnostics. `R11` uses exactly
the `S11` trial set, no-response indicator, nuisance model, and scaling.
Responses lock to observed response onset and misses to the deadline, so
`R11-S11` changes timing rather than support. It is a response-or-deadline
timing sensitivity, not a pure commitment effect or evidence of temporal order.

The decision-state block is frozen as offer-onset binary entropy of the cross-
fitted choice probability plus a unit-height offer-to-response epoch (offer-to-
deadline for misses). Entropy is centered and scaled within run from covariate
values only. Specifically,
`H(p)=-p log(p)-(1-p)log(1-p)`, where
`p=P(accept by 4s)/[P(accept by 4s)+P(reject by 4s)]` after excluding the
contaminant channel; it is conditional on a boundary response by the deadline.
Three-outcome accept/reject/no-response entropy is a sensitivity only. Because
both columns enter together, `Delta_D` is their joint
specification displacement; it cannot separate difficulty from duration.

Actual-design simulations precede BOLD interpretation. They must show that
gain/loss remains estimable and that offer- and response-linked blocks are not
hopelessly collinear.

Before observed fitting, the design gate is fixed at task-column condition
number `<=30`, gain/loss VIF `<=5`, and `S11` gain/loss efficiency at least
half of `S00`. Retained templates require complementary-half spatial cosine
`>=0.20`, bootstrap lower bound above zero, and agreement beyond a participant
sign-flip null. Projection bases additionally require condition number `<=10`,
pairwise `|r|<=0.85`, prespecified synthetic coordinate recovery, and coordinate
stability (`rho>=0.80`, sign agreement `>=90%`, median shift `<=0.25` SD)
across declared sensitivities.

`R11` must separately meet condition number `<=30`, gain/loss VIF `<=5`, and
gain/loss efficiency at least half of `S11`; otherwise the timing sensitivity
is dropped without invalidating the estimable factorial cells.

Any regional summary uses only the public NARPS vmPFC, ventral-striatum, and
amygdala hypothesis masks frozen before observed fitting. Ambiguous mask
provenance removes the regional summary rather than licensing a post-hoc ROI.

### Cross-run evidence

Each half uses difficulty regressors generated from behavior fit to the other
half. Retained conditional-slope and specification-displacement templates must
reproduce in direction and continuous spatial pattern after half-swap, within
EI/ER, and under mask and smoothness sensitivities.

The optional person-level test predicts untouched target-half choice/RT from a
predictor-half behavioral baseline, then asks whether the predictor-half neural
gain/loss vector adds held-out log score. Its dedicated neural fits construct
each predictor run's difficulty regressor from the other predictor-half run;
the target half never enters predictor construction. Behavioral-family choice,
priors, regularization, neural templates, and all tuning are repeated inside
the predictor half rather than inherited from the four-run winner.

This branch exists only for one identified family containing `eta` plus one
recoverable `z` or `c`; it is omitted for the generic bias-family ensemble. The
target likelihood is the same first-passage choice/RT density, four-second
survival term, and contaminant mixture as the winning behavioral DDM. The
baseline uses task version and predictor-half behavior. The expanded model
adds the standardized two-dimensional neural vector to the population prior
means of `eta_i` and either `logit(z_i)` or `c_i`, with no neural terms on
`a`, `t0`, `kappa`, or contamination. For participant `i`, that coefficient
matrix, scaling, priors, and regularization are learned from other participants'
predictor-half data; `i`'s predictor behavior updates random effects equally in
both models.

The neural vector has one primary source: within task version, equally average
the two predictor-half runwise `S11` beta maps separately for gain and positive
loss magnitude; spatially demean each map in the frozen outcome-blind whole-
brain intersection mask; and project it onto the matching unit-norm `S11`
template formed from other participants' predictor-half maps. Standardize the
two scores using those other participants only. ROI, displacement, `R11`, and
alternative-template scores are excluded from this branch rather than treated
as selectable predictors. Both directions must improve by at least `0.01` nat
per target trial with participant-cluster bootstrap 95% lower bound above zero.
Otherwise the optional branch is removed.

### Interpretation

The fMRI result concerns robustness of gain/loss maps to cognitive
operationalization. It does not localize a DDM starting point. Because the four
response categories map to four right-hand fingers, response regressors absorb
an inseparable choice/strength/motor mixture.

`S11`, `Delta_D`, and `Delta_R` are conditional slope and model-specification
displacement maps. They are not identified pure maps of valuation, difficulty,
response bias, choice, or motor activity.

## Study 3: the many-team estimand audit

### Question

Do the public team maps vary along the same specification-displacement
directions, and are those unitless shape coordinates associated with what teams
said they modeled?

### Map construction

Use the public unthresholded group statistic maps only. Restore a common
underlying coefficient direction for increasing gain/loss magnitude rather
than reuse the original hypothesis-directed rectification; this does not make
`t`/`z` images beta maps or comparable effect units.
Deduplicate region-specific hypotheses only when they reuse the same submitted
file or are numerically equal after sign reconstruction on the same support;
high correlation alone is insufficient. Reduce the nine hypotheses to
gain-EI, gain-ER, loss-EI, loss-ER, and the loss ER-minus-EI comparison.

Map shape is analyzed after common-grid/mask harmonization and robust within-
map standardization. Recorded participant counts, exclusions, sample-identity
summaries, higher-level design/estimator, smoothness, statistic family, mask,
software, preprocessing, and global scale are explicit controls; unrecorded
sample/model details limit attribution.

### Projection

Construct a joint basis from cross-run-stable conditional and displacement
maps:

- fully adjusted conditional gain/loss slope (`S11`);
- decision-state omission, averaged over response-block status
  (`Delta_D = 0.5[(S00-S10) + (S01-S11)]`);
- response-block omission, averaged over decision-block status
  (`Delta_R = 0.5[(S00-S01) + (S10-S11)]`);
- decision-by-response nonadditivity
  (`I_DR = S00-S10-S01+S11`), only if recoverable across run halves; and
- response-or-deadline timing sensitivity on identical trial support
  (`R11-S11`), only if its design and cross-run gates pass.

The omission-positive convention is fixed before fitting. The four cellwise
differences are simple-effect diagnostics, and `S00-S11` is the total
legacy-to-full displacement; neither is substituted for an order-invariant
factorial displacement axis.

All displacements are computed from matched run-level effect estimates in
common units and aggregated with participant-paired inference. Group `t` or
`z` maps are not subtracted to manufacture displacement effects.

Every team base estimand uses its matched basis: EI-gain, ER-gain, EI-loss,
and ER-loss maps are projected only onto the corresponding version-by-value
templates. H9 uses an explicit ER-minus-EI difference of every loss template;
that basis is descriptive because task version is between participants. Pooled
or cross-estimand templates are prohibited.

First test numerical coordinate recovery on synthetic linear combinations and
held-out spatial parcels. Freeze the basis before viewing method-coordinate
associations. Then project each deduplicated team map and repeat the result
across half-specific templates, masks, and smoothness-matched references.

The reference templates and team maps draw from the same NARPS participant
pool, with possible team-specific exclusions; no public team map is reserved
from template construction on a deliberately disjoint sample. Half swapping is
a stability test, not independent validation. After within-map robust
standardization, projections are unitless shape coordinates—not cognitive
mixture proportions or map-content estimates. Synthetic recovery validates
only their numerical estimability. Generalization requires an independent
dataset.

### Method coding

Code first-level event support, duration, gain/loss scaling and centering,
orthogonalization, RT/duration modeling, response modeling, and regressor set
without seeing team coordinates. Frozen technical controls include participant
count, exclusions, available sample-identity summaries, higher-level
covariates/design, group estimator, mask/grid, smoothness, statistic type,
software, preprocessing source, motion handling, and inference family.
Unreported fields receive explicit missingness indicators. If exact sample
identity or group-model details are unavailable, the ecological association
cannot isolate cognitive specification from unrecorded sample differences.

The primary comparison is a multivariate partially pooled ridge model using
base-estimand identity plus frozen technical controls versus the same model
with the prespecified cognitive-feature blocks. All maps from one team remain
together in an outer leave-one-team-out fold, and regularization is selected
inside the training teams. The outcomes are held-out joint log predictive
density and multivariate `R^2`; uncertainty uses team-cluster bootstrap and a
team-bundle permutation test. No coordinate-informed or stepwise feature
selection is allowed. Missing method fields receive explicit indicators and a
complete-case sensitivity.

Teams are the resampling unit and hypotheses are repeated measurements. The
association is ecological and descriptive because teams selected their own
pipelines.

## Integrated questions

The final analysis answers four questions in order:

1. Can behavior distinguish valuation asymmetry from value-independent bias?
2. Do conditional gain/loss maps survive decision and response adjustment?
3. Are the resulting specification-displacement directions stable enough to
   describe the same-data public maps as unitless shape coordinates?
4. Do team maps and declared methods show the corresponding estimand
   structure beyond the frozen sample, higher-level model, and map-technical
   controls?

The paper reports the lowest tier that survives. A later positive stage cannot
repair an earlier failed identification gate.

## Planned figures

| Figure | Question | Required content |
| --- | --- | --- |
| Figure 1 | What is being separated? | behavioral mechanisms, controlled fMRI arms, and many-team projection with claim limits |
| Figure 2 | Are the mechanisms identifiable? | exact-design recovery, model confusion, held-out choice/RT/deadline prediction, EI/ER transport |
| Figure 3 | How does cognitive specification move maps? | `S00/S10/S01/S11/R11` map vectors, efficiency, and complementary-half replication |
| Figure 4 | Does predictor-half neural signal add held-out behavioral value? | baseline-versus-neural incremental score on an untouched target half, only if inner and individual recovery pass |
| Figure 5 | Where do the team maps fall? | same-data shape-coordinate space, deduplicated hypothesis structure, uncertainty by team |
| Figure 6 | Which choices predict estimand position? | blinded method features, technical controls, and falsifier sensitivities |

## Planned tables

| Table | Content |
| --- | --- |
| Table 1 | direct literature collisions and the exact surviving gap |
| Table 2 | behavioral model family, recovery, and held-out comparison |
| Table 3 | fixed fMRI arms, estimand, efficiency, and allowed interpretation |
| Table 4 | team-map inventory, deduplication, method-code completeness, and dependence structure |

## Strong result branches

### Branch A: full estimand-drift paper

Behavior requires both mechanisms; conditional slopes and specification
displacements reproduce; team shape coordinates are recoverable and associated with
cognitive model choices beyond technical controls.

Allowed conclusion:

> Analysts who nominally tested the same neural gain/loss hypothesis produced
> maps occupying systematically different conditional-estimand positions,
> partly predictable from how duration, difficulty, and response were
> operationalized.

### Branch B: robust conditional gain/loss paper

Behavior separates mechanisms, but `S11` gain/loss maps remain nearly
unchanged and team maps load primarily on the stable conditional gain/loss
direction.

Allowed conclusion:

> Despite behavioral ambiguity, the declared gain/loss neural estimand was
> equivalent within prespecified small-effect bounds across the tested
> cognitive controls; the coded cognitive features added at most a bounded
> amount of team-coordinate prediction beyond the technical controls.

This branch requires the 90% interval to lie inside regional `|d_z|<0.20`,
symmetric whole-brain relative RMS displacement `<0.10`, and spatial cosine
`>0.90`. The team-feature increment must likewise lie inside `+/-0.01` nat per
standardized coordinate and `+/-0.02` multivariate `R^2`. A nonsignificant but
imprecise result is inconclusive, not robust, and does not identify what caused
the remaining variation.

This null is scientifically useful and does not require rescuing the old
smoothing project.

### Branch C: controlled-model paper only

The behavioral and fixed-pipeline layers pass, but public team projections or
metadata associations fail. Report the controlled estimand sensitivity and
state that the ecological archive could not identify its source.

### Branch D: behavioral narrowing or stop

If only one behavioral mechanism wins, remove two-process neural language. If
the mechanism family is not recoverable or does not improve held-out
prediction, stop before BOLD and do not claim a paper from the behavioral
exercise alone.

## Disallowed result language

The manuscript will not say that:

- fMRI proves bias occurs before valuation;
- a response-onset contrast is a pure response-bias or motor map;
- one pipeline reveals the true map;
- EI/ER is a causal manipulation without assignment evidence;
- four within-session runs establish a stable trait;
- 70 team maps are 70 biological replications;
- observed method-coordinate associations are causal;
- thresholded significance defines the cognitive estimand; or
- individual rankings can be compared across 70 group-only pipelines.

## Paper viability rule

A full paper requires all of the following:

1. exact-design recovery and calibrated held-out choice/RT prediction;
2. an identifiable fixed fMRI factorial with reproducible half-split
   conditional-slope and specification-displacement directions;
3. a team-map projection that recovers known synthetic coordinates and survives
   mask/smoothness/half-template changes; and
4. either an informative cognitive-method association or a strong, well-
   bounded equivalence result under the fixed margins.

Behavior-only DDM, generic neural decoding, range adaptation, a conventional
gain/loss map, or another generic multiverse is not sufficient.

## Core references

- Botvinik-Nezer R et al. (2019). fMRI data of mixed gambles from NARPS.
  doi:10.1038/s41597-019-0113-7.
- Botvinik-Nezer R et al. (2020). Variability in the analysis of a single
  neuroimaging dataset by many teams. doi:10.1038/s41586-020-2314-9.
- Zhao WJ, Walasek L, Bhatia S (2020). Psychological mechanisms of loss
  aversion. doi:10.1016/j.cogpsych.2020.101331.
- Sheng F et al. (2020). Decomposing loss aversion from gaze allocation and
  pupil dilation. doi:10.1038/s41598-020-66238-7.
- Bobadilla-Suarez S, Guest O, Love BC (2020). Subjective value and decision
  entropy are jointly encoded by aligned gradients across the human brain.
  doi:10.1038/s42003-020-01315-3.
- Wang R et al. (2024). Decomposing loss aversion from a single neural signal.
  doi:10.1016/j.isci.2024.110153.
- Brochard J, Daunizeau J (2024). Efficient value synthesis in the
  orbitofrontal cortex. doi:10.7554/eLife.80979.
- Soch J, Haynes JD. Decoding behavioral responses from fMRI data in NARPS.
  doi:10.1101/2022.03.24.485588.
