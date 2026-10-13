# Utility in discarded site directions

## Adopted final counterfactual — exact fitting-covariance protection

Following the completed age-adjustment intervention, continue the scientist's
requested investigation with one direct counterfactual, specified here before
its fits or outcomes. Analysis `covariance_protected_projection` tests the
training-covariance property that residualization did not produce. It is the
final diagnostic in this continuation, within the same allocation and shared
two-core-hour maximum below; no budget reset or new search is introduced.

For each original atlas/outer fit, reuse the completed age-adjustment run's
original standardized features, original rank-four site-direction matrix V,
and raw/original predictions and probe correctness. Verify the cached scales,
row/fit roles and original projection against the exposed bundle. Use true
fitting ages only to compute c = Zᵀ(y−mean(y))/n and q = c/||c||. Form
U = V(I−qqᵀ), then extract an orthonormal rank-four row basis W by SVD.
Project ORIGINAL Z with P = I−WᵀW, without column rescaling. Require rank four,
orthonormality and Pc=c within numerical precision; a vanishing covariance
vector or deficient projected rank is a reported support failure, not an
opportunity to change rank. Evaluation transforms accept ROI features only.

The single matched control uses c computed from the existing seed14014
within-site shuffled fitting ages, with the same construction and true ages
for downstream ridge. Preserve the original unprotected and raw references.
Use every fixed alpha [0.1,1,10,100,1000], alpha1000 anchor, the same all-row,
common-support and matched centered-eligible populations, both evaluation
roles, all original views, ordinary unrescaled seen-site probes, and the same
10,000 paired domain/participant bootstrap and single-sham limitations as
the completed intervention. No nuisance-classifier direction refits, tuning,
additional shuffle family, candidate promotion or holdout access occur.

Report protected minus original, sham-protected minus original, protected
minus sham-protected, protected minus raw, and original minus raw. Record
true-age covariance alignment, removed feature energy, principal angles and
Frobenius projector displacement from the original removal subspace for both
new arms. Their rotation magnitudes can differ, limiting attribution of a
prediction benefit specifically to age alignment. Preserving fitting marginal
covariance does not guarantee preserved ridge predictions or held-out age
utility: features with zero marginal target covariance can still contribute
through their relationships with other features. This empirical pipeline
counterfactual is neither a new theorem nor a biological/scanner mechanism.

## Adopted continuation — projection heterogeneity and age-adjusted directions

The scientist explicitly requested continuing after the completed diagnostic.
The next two analyses below are adopted before their execution. They remain
exposed-development investigations, with no candidate promotion, original
search reopening or public-validation access. Neither is independent
replication of the already-exposed corpus.

### Cached projection heterogeneity

Analysis `projection_heterogeneity` reads only the completed diagnostic's
saved predictions and summaries. Preserve both atlases, all external ages,
all13 domains, all original views and all participant appearances. At the
existing alpha1000 anchor, compare raw, unrescaled retained, rescaled retained
and restored predictions. Report paired per-domain and macro absolute and
centered MAE/MSE, and every leave-one-domain-out macro contrast. Participants
have equal total weight within domains, divided across eligible appearances;
domains are equally weighted. Recompute centers for each scored population.

For raw-to-retained, retained-to-rescaled and raw-to-rescaled predictions,
decompose MSE change exactly as `2 E[e_base * correction] + E[correction²]`.
Use centered errors/corrections for the centered MSE identity. Fixed ten-year
bins cover [0,10), ...,[90,100], with age100 in the last bin. Preserve every
domain/bin, show coverage and mark empty cells; there is no new eligibility
threshold or favorable age-band selection.

Compare all rows and common support on identical domain sets. Separately
report absent-domain reweighting and the effect of rebalancing a participant's
remaining appearances after support restriction. Supported/complement additive
decompositions use original appearance weights and must reconstruct the
full-population loss; actual common-support metrics use their declared
eligible-appearance weights. Do not apply additive subgroup accounting to
separately recentered metrics. No new interval, significance test or causal
attribution is introduced by this descriptive analysis.

### One fixed linear age-adjustment intervention

Analysis `age_adjusted_projection` asks whether accounting for age while
estimating the removal directions changes their prediction cost. Use both
atlases and the three original outer splits, with four fixed representations:
raw standardized ROIs; original rank-four site removal; rank-four directions
estimated after linear age adjustment; and an identical adjustment using a
within-site shuffled-age control. All removal transforms act on the original
standardized ROI coordinates without subsequent scaling.

Within each outer fitting role only:

1. Fit the existing raw-ROI scaler to obtain Z. Estimate the original site
   directions using the unchanged balanced C1 logistic classifier and SVD.
2. Fit ordinary least squares of every Z column on an intercept and
   centered/scaled fitting age. Train the same site classifier on these
   residuals and extract four orthonormal coefficient directions in the
   original Z coordinate system. Do not standardize the residual columns.
3. For the sham, sort fitting domains and permute ages within each domain
   using `RandomState(14014)`, then repeat the same residualization and
   direction estimation. This preserves each site's age distribution while
   breaking individual within-site age/ROI pairing. Downstream ridge always
   uses true fitting ages, including in the sham arm.
4. Project ORIGINAL Z onto the complement of the selected directions.
   Transform evaluation rows using fitting scales/directions and their ROI
   features only. Evaluation ages never enter residualization, transformation
   or model fitting. Require four nonzero orthonormal directions for every
   removal arm; report a support failure rather than changing rank.
5. Fit ridge at every existing fixed alpha [0.1,1,10,100,1000] by one SVD per
   representation. Alpha1000 remains the anchor; the other values are a fixed
   sensitivity curve. There is no new tuning, inner search or outcome-based
   selection.

Compare age-adjusted minus original, sham minus original, age-adjusted minus
sham, age-adjusted minus raw and original minus raw. Use the previous all-row,
common-support and matched centered-eligible-domain age estimands unchanged,
with both evaluation roles and per-view summaries visible. Use10,000 paired
hierarchical bootstrap draws, seed14014, preserving participant appearance
bundles and recomputing centers in every draw. Coverage/eligibility are fixed
from observed support before resampling. Intervals remain conditional and
descriptive, with no success threshold or multiplicity-adjusted discovery.

Fit ordinary seen-site probes on the same UNRESCALED representations supplied
to ridge, using the original balanced C1 classifier and outer fitting sites.
Score all original internal evaluation rows, average each participant's
appearance-level correctness, then average participants within true site and
sites equally. Report every site/view and10,000 paired hierarchical bootstrap
intervals for pooled accuracy contrasts with the same seed and clustering.
There is no separate site-probe scaler, conditional-site probe or noninferiority
decision. This measures unconditional site predictability. Age preservation
can itself preserve site prediction; increased accuracy does not isolate
scanner-specific nuisance.

Record, on original Z and TRUE fitting ages, the squared covariance-vector
alignment fraction `||V c||² / ||c||²`, where `c=Cov_fit(Z,age)` and rows of V
are orthonormal. Report undefined support when the covariance vector vanishes.
Also report removed feature energy. Neither quantity is a fraction of age
information or explained age variance. Residualizing classifier inputs does
NOT guarantee that the resulting feature directions are orthogonal to age
covariance when applied to original Z.

A true-age benefit beyond the within-site sham would support sensitivity to
individual age/ROI pairing in direction construction. Similar true/sham
benefits remain compatible with shared adjustment or between-site composition.
A single fixed shuffle is an active control, not a permutation null
distribution. The evaluation bootstrap does not capture shuffle or training
uncertainty. The within-site shuffle preserves the between-site age/ROI
covariance component; this contrast does not directly remove or identify
between-site confounding.
A null result would limit this particular linear intervention. None identifies
scanner causality, new biological age signal or fresh confirmation.

### Shared execution bound

One attended allocation covers both analyses, focused implementation checks,
a representative outer fit, full execution and figures: four CPUs,16GiB,
at most30minutes (two allocated core-hours), one worker, at most four numeric
threads, no GPU or automatic renewal. Preserve completed outer checkpoints;
do not reset cost on an engineering repair. Root coordinates submission and
tests. Original inputs and completed attempts remain unchanged.

## Adopted development diagnostic — October 4, 2026

The scientist's instruction to continue after the proposed discarded-information
test authorizes the narrow fitted diagnostic below. It is recorded before new
diagnostic fits or scores. The original R0–R3 result remains
`development_no_candidate`, and neither the broad representation search nor a
public-validation endpoint is activated. The older proposals below remain
context; only this section is executable. Analysis ID:
`discarded_direction_diagnostics`.

**Question.** Do the removed directions carry transferable age information,
and can the existing prediction harm instead be explained by changed feature
scaling or ridge penalties? These accounts may coexist. The exposed data can
support a descriptive diagnosis, not causal biological or scanner attribution.

### Fixed fits and comparisons

Use both existing atlases and all three original development views, with
exactly the existing exposed participant roles. Reuse the original scaler,
balanced L2 site classifier, seed 14014, ridge solver and alpha grid
`[0.1, 1, 10, 100, 1000]`. No public-validation feature or label is read.

For each inner and outer fit, fit a raw-ROI scaler on those fitting rows.
Construct four discarded scores from each of three direction sets: the
true-site classifier; seeded rank-matched random QR directions; and the
classifier fitted with seed-14014 permuted fit-site labels. True and permuted
classifiers must each provide rank four; otherwise report a rank-support
limitation rather than silently changing the comparison. Standardize the four
scores using the same fitting rows. Each auxiliary ridge selects alpha using
leave-one-fit-site-out, equal-domain absolute MAE on all inner validation rows;
ties choose larger alpha. Every direction and scale is refit in each inner
fold. Refit the selected procedure on the full outer fit and preserve predictions
for all original internal and external evaluation rows. The absolute-age
reference is the outer-fit median age.

For a separate geometry comparison, use true-site directions and each outer
fit's standardized raw matrix z. Let d=zV and r=z-dV-transpose, where columns
of V are orthonormal. Compare raw z, retained r with no further scaling,
retained r with the original final column standardization, and the restored
concatenation [r,d] with no final scaling. Apply identical ridge penalties to
these four representations. Alpha 1000 is the prespecified anchor because
every original primary outer arm selected it; all five original grid values
are reported as a fixed sensitivity curve, never selected using evaluation
scores. The restored representation has the same row Gram matrix as z and
must reproduce raw predictions at every alpha within numerical tolerance.
This control separates loss of available directions from changed ridge
geometry. Discarded-only prediction cannot establish that those directions
contain unique information unavailable in retained features.

### Evaluation support and estimands

Report all-row results first, preserving every original domain and view.
Also report a predeclared overlap population: partition ages into five-year
bins [0,5), ...,[95,100], with age 100 in the last bin. A bin qualifies only
when at least five distinct outer-fitting domains each supply at least five
fitting participants in that bin. Only fitting ages define the qualifying
bins. Evaluation ages determine membership inside the evaluator; they never
change features, fits, alpha or predictions. Out-of-range ages and all
excluded rows/domains are counted explicitly. This is multi-domain overlap,
not an assertion of overlap among all training domains.

Common-support absolute MAE includes domains with any supported rows. The
common-support centered diagnostic additionally requires at least ten unique
supported participants and age SD at least 2.5 years in the evaluated domain.
Otherwise mark that domain insufficient, while retaining its coverage and
all-row results. These domain rules are fixed from observed support before
resampling. No alternative age band or support threshold is tried when
overlap is inadequate.

Also report absolute MAE on exactly the centered-eligible domains, using the
same supported rows and participant weights. Joint absolute/centered utility
interpretations use this matched population; the broader absolute summary
remains separately visible. This reporting refinement was adopted in static
review before diagnostic fitting or evaluation.

For each scored domain, give every participant total weight one divided across
its eligible appearances, then give domains equal weight. Do not average
predictions across views. Absolute-age MAE is compared with the fitted-age
median; centered-age MAE compares centered predictions and centered ages with
the zero predictor for centered age. Centering is evaluator-only descriptive
accounting, not a deployable correction. Compare true-site discarded scores
with both random and permuted controls on identical support. Report every
view separately as a robustness description; pooled views are not independent
replications. Geometry uses the same scoring populations and summaries.

Use 10,000 paired hierarchical bootstrap draws, seed 14014, for pooled
contrasts: resample domains and participants, retaining all of a participant's
eligible appearances together; recompute weighted domain centers within each
draw. The same draws pair all compared models. Intervals are descriptive and
conditional on the included observed domains; no interval is emitted with
fewer than two contributing domains. Per-view summaries need no additional
bootstrap. Zero is the reference for directional utility; there is no new
minimum-effect promotion rule, equivalence margin, omnibus success label or
near-zero claim. Inadequate support or wide intervals leaves interpretation
unresolved rather than authorizing another search.

### Execution and decision boundary

One bounded CPU allocation covers focused implementation checks, the complete
two-atlas diagnostic and reporting: at most four CPUs, 16 GiB, 90 minutes
(six allocated core-hours), no GPU. Use at most four numerical threads and
one worker. Preserve per-fold checkpoints in a fresh attempt; resumption may
reuse only completed same-specification work and never reset consumed cost.
The checks target nested fitting boundaries, geometry equivalence, support,
participant grouping and score reconstruction. No new generic qualification
program or holdout check is part of this analysis.

The original practical thresholds and no-candidate decision stay unchanged.
The optional conditional-site probe, expanded representation search, new
calibration model and every holdout endpoint remain unactivated. A result here
can prioritize a later development question; it cannot confirm a new biological
mechanism or promote a candidate. Nearest prior work and novelty limits remain
those cited below; the scanner/age tradeoff itself is already established.

## Historical optional design

Retained from the earlier 2026-09-30 site-centered framing. This is an
unactivated diagnostic, not EP14's main question, a candidate, an exploration
prerequisite or a newly authorized holdout endpoint. The main direction is now
[open neuroanatomical representation search](representation_research_design.md).

## Paired age-utility views

For each outer fit role, reuse R3's fitted four directions `V` and fit-only
standardized ROI matrix `z`; retain scores `d=zV`. Fit an auxiliary ridge age
predictor with the same inner site-held-out alpha criterion. Repeat for the
rank-matched random and fit-site-label-permuted direction controls. These are
diagnostics, not candidates.

Evaluate on the same fit-defined common age support in two required views:

1. Unseen-domain absolute MAE against the outer-fit age-median predictor,
   domain by domain. This can still exploit between-domain age composition.
2. Within-domain utility: subtract each evaluated domain's mean age from age
   and mean prediction from prediction, comparing centered error with the
   centered zero predictor. Centering is evaluator-only, never a feature,
   tuning input or deployable MAE endpoint.

Both views, adequate support/precision and specificity against the declared
random/permuted controls are needed before claiming useful removed information
beyond a domain-average shortcut. Discarded scores may be redundant with
retained ROIs; projection also changes ridge's effective penalty. This tests
pipeline-relative predictive utility, not irreversible information loss or a
unique biological age signal.

## Supporting conditional-site probe

Fit matched age-only and age-plus-representation site classifiers inside each
outer fit role, with a prespecified nonlinear age basis and fitting rule.
Compare class-balanced log loss on held-out seen-domain rows. This measures
model-relative conditional site prediction, not conditional independence,
scanner purity or causal acquisition isolation. Age is used only by the
declared supervised diagnostic/evaluator, never the representation transform
or age predictor. The operative unconditional BAcc guardrail is unchanged.

## Support and confirmation limits retained

Before running this diagnostic, set the common age-support rule, per-domain
coverage and age variation, nonlinear age basis, probe regularization, utility/
near-zero margins, domain-level uncertainty and joint utility criterion using
exposed development only. Apply the outer-fit support rule unchanged to
evaluation rows; report excluded ages/domains and the overlap population.
Do not trim unfavorable domains, select age bands or redefine the criterion
after outcomes.

Every learned scale, direction, auxiliary model, age basis, weight or
residualization is fit-role-only and refit inside tuning folds where applicable.
No full-dataset residualization, held-out fitted state, target-dependent feature
transform or test-time adaptation is allowed. Held-out ages enter only the
locked evaluator's support membership, within-domain scoring and diagnostics.

Report each domain and cluster uncertainty by `siteXacq`. RAMP views overlap;
too few domains, insufficient overlap, little within-domain variation or broad
intervals imply non-identification/inconclusiveness. More participant resampling
cannot repair these limitations.

The 757-row one-shot scope currently covers only the frozen age/probe
comparison. A new endpoint requires a prospective amendment, eligible exposure
review and compatible evaluator before outcome access; otherwise it is
development/retrospective evidence needing genuinely new confirmation material.
No diagnostic threshold, candidate rule or new ASTRA analysis is activated.

## Interpretations, not new terminal rules

- A lower site score, worse external age error and both removed-score utility
  views beyond controls support an operational age-utility-loss diagnosis.
- Reduced conditional site gain, retained external age utility and a sufficiently
  precise bound on removed-score utility support only limited selective reduction.
- Similar random/permuted behavior, inadequate probes, failed support or unresolved
  utility margins leave generic compression or an unresolved diagnosis.

None establishes scanner invariance or biological mechanism. The scanner/age
tradeoff is already known, including [Dinsdale et al. (2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7903160/).

## 2026-10-04 — proposed follow-up after the fixed result

The October 3 reference completed with no eligible candidate. Removing four
site-predictive directions raised Desikan external domain-macro MAE from
5.83111 to 6.20696 years (+6.45%); the Destrieux change was only +0.225%.
Random and permuted-site removals remained near the Desikan baseline. See the
[completed result](experiment_log.md#2026-10-03--fixed-development-completed-no-eligible-candidate).
This motivates explanation, not a reopened selection or a general claim that
site removal is harmful. The fitted diagnostic below remains **proposed and
unactivated**; the completed reference and closed holdout are unchanged.

| Rival account | Discriminating comparison | Evidence against that account as the sole explanation |
| --- | --- | --- |
| Age/site composition and calibration offsets | Compare absolute and evaluator-centered within-domain errors, then discarded-score utility on training-defined shared age support | Harm and matched-control-specific utility persist within domains on adequate common support |
| Discarded directions contain transferable age variation | Fit age predictors on the four discarded scores, with random/permuted directions of the same rank; test incremental utility when restored beside retained features | No within-domain or incremental utility beyond controls at useful precision |
| Scaling/shrinkage geometry or atlas redundancy | Compare raw, projected, restandardized projected, and restored full representations in the same metric; repeat both atlases | Harm persists under matched geometry and fitting allowance in both atlases |

These accounts may coexist. Predictive age variation does not identify its
biological cause. Destrieux is a sensitivity on the same participants, not an
independent replication. Every primary arm selected alpha 1000, the original
grid's upper endpoint; prospective penalty sensitivity must be matched across
comparisons and must not change the finished reference.

### Selected description using existing predictions

An exploratory secondary analysis may first summarize the already-exposed
R0/R3 outer predictions for both atlases, without fitting or opening inputs.
Retain every cached domain, age, role and view. Use residual = prediction minus
age. In pooled-domain summaries, give each participant equal total weight and
split that weight equally among its appearances; do not average predictions.
Center using that same domain weighting. Also report separately centered
view-by-domain summaries, which answer a distinct descriptive question.

With these weights, **MSE = squared mean residual + centered residual
variance** exactly. Average these components equally across domains. Centered
MAE is reported separately: MAE has no additive offset/variance decomposition.
Centering uses evaluator targets and is neither a deployable correction nor a
new primary endpoint. Error components cannot establish a causal composition
mechanism. Reproduce the original absolute MAEs before interpreting diagnostics.

Source: `runs/fixed_development_comparison/attempt001/outer/`, only the twelve
Desikan/Destrieux R0/R3 files; the existing aggregate reports provide numerical
cross-checks. Runner: `tools/analyze_cached_site_projection.py`. Intended result:
`runs/cached_projection_diagnostics/attempt001/`. This description needs no
model or support amendment, and does not activate the fitted proposal.

### Smallest subsequent fitted test — still proposed

Reuse the original splits, fit-only scalers, site classifier, ridge solver and
appearance-level aggregation. Reconstruct the four discarded scores inside
each inner/outer training fold and fit their auxiliary age predictor alongside
rank-matched random and permuted-site controls. Require both external absolute
utility and evaluator-centered within-domain utility on the same prespecified
training-defined support; show coverage and age variation for every domain.
An additional retained-versus-restored comparison tests incremental utility,
because useful discarded scores can be redundant with retained measurements.

For the geometry control, use one standardized raw matrix `z` and orthonormal
complement/discarded bases `U,V`. A full `[zU,zV]` representation with one common
isotropic ridge penalty must reproduce raw `z` predictions. Contrast retained
`zU` and the current restandardized projection at matched penalties first.
Any added nested penalty tuning must give all comparisons the same allowance.
Repeat key contrasts in Destrieux; do not choose an atlas from favorable results.

Before this fitted stage, record a prospective development-only amendment for
age support, models/penalties, utility margins, uncertainty and resources, as
required above. The current caches hold metrics and predictions, not fitted
direction matrices; directions need fold-local refitting. No new holdout
endpoint, candidate, selection threshold or ASTRA stage is activated here.

The cached description should take a few minutes on one CPU with 4 GiB in the
coordinator's Slurm session. A focused two-atlas refit is tentatively 30–90
minutes on four CPUs/16 GiB, without a GPU; this is a planning estimate, not a
submission. The completed larger panel took 22m47s for 1,650 inner and 33 outer
fits. Keep new attempts distinct and preserve the completed outputs.

### Closest literature and contribution boundary

[Dinsdale et al. (2021), section 2.4.3](https://rcastoragev2.blob.core.windows.net/31ed25e0a433e90c689486cfed27dba6/PMC7903160.pdf)
already contrasted unlearning across whole age distributions versus their
overlap, including 5-, 10- and 15-year overlap conditions. Harmful removal of
task-related information under age/site imbalance is therefore established.
[Jirsaraie et al. (2023)](https://bpb-us-e2.wpmucdn.com/sites.wustl.edu/dist/1/1008/files/2023/01/2023JirsaraieHBM.pdf)
separately manipulated age-range and acquisition similarity and documented
generalization biases. The potential contribution here is distinguishing
composition, transferable within-domain utility and ridge geometry for the
observed ROI projection, not rediscovering that harmonization can hurt.

## 2026-10-04 literature update — novelty boundary

Published facts:

- [Dinsdale et al. (2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7903160/)
  found that restricting scanner unlearning to overlapping age ranges improved
  age prediction. Their overlap experiment favored both unlearning variants
  over ordinary training; it does not establish universal harm from unlearning.
- [Jirsaraie et al. (2023)](https://pubmed.ncbi.nlm.nih.gov/36346213/)
  documented brain-age generalization differences associated with scanner
  variation, age distributions and prediction bias.
- [Ravfogel et al. (2020), *Null It Out*](https://aclanthology.org/2020.acl-main.647/)
  introduced iterative nuisance-classifier nullspace projection. Removing
  classifier directions is therefore an established method family.
- [Holstege, Ravfogel and Wouters (2025)](https://proceedings.neurips.cc/paper_files/paper/2025/file/26dbc8565974cffe8a44731aa09b2aa8-Paper-Conference.pdf)
  address preserving task information during linear concept removal. Their
  Theorem 2 proves equivalent predictions after unregularized linear refitting
  for projections sharing a kernel, under uniqueness assumptions. Section 3.3
  explicitly explains that regularization can break this equivalence.

Interpretation for EP14 (inference, before the fitted diagnostic result): the
general risk and the distinction between removal and regularization geometry
are already known. A contribution could be a controlled empirical diagnosis
for this unseen-site ROI pipeline. Discarded-score age prediction shows useful
signal, but retained features may encode it redundantly. A raw-versus-retained
deficit without rescaling supports utility for this ridge pipeline, not
information-theoretic loss. Raw/restored equality is an algebraic control;
rescaled/unrescaled differences diagnose metric effects. Neither demonstrates
scanner causality or new biological information. Development exposure and
overlapping atlas participants limit confirmation; broader novelty and
independent replication remain unresolved. This update changes no adopted
experimental specification.
