# What Makes a Neuroanatomical Age Representation Transfer?

[Scope and novelty review, 2026-10-02](outputs/scope_novelty_review_20261002.md).

## The question in plain language

The same FreeSurfer measurements can be represented as individual ROIs,
bilateral shared and difference components, or anatomically grouped summaries.
These choices change what a predictor pools, compresses and regularizes. A
representation that predicts age well in familiar data need not retain that
advantage at an unseen acquisition domain.

EP14 returns to its original open question:

> Which neuroanatomical representations and shallow predictors make age
> prediction transfer, and what representation property explains an advantage
> beyond generic compression, regularization or model capacity?

The target is chronological-age prediction from the existing ROI derivatives.
Site/acquisition defines the generalization environment and supplies a
supporting diagnostic; removing site information is not the scientific goal.
This is an open predictive representation search, not a claim about biological
aging mechanisms. The question is broader than selecting one of four methods.

The [representation design](outputs/representation_research_design.md)
provides starting hypotheses and discriminating comparisons, not a closed
candidate list. The [paper plan](outputs/paper_plan.md) follows that question.
![EP14: anatomical representations and unseen-domain age prediction](outputs/ep14_conceptual_question-v4.png)

Conceptual design: the same ROI measurements support an open search over
representations and shallow predictors. Bilateral structure and anatomical
grouping are starting hypotheses, compared with matched generic compression
and regularization. A complete bilateral re-encoding adds no information;
different scaling or shrinkage changes the prior. Scanner icons indicate
training and test environments, not a site-removal objective. Illustrative
predictor icons are not measured age results. [Image-gen prompt](outputs/ep14_conceptual_question-v4-prompt.md).

The earlier [site-centered schematic](outputs/ep14_conceptual_question-v3.png)
and [four-arm schematic](outputs/ep14_conceptual_question-v2.png) remain
historical design assets, not illustrations of the revised main story.

## What is already known, and what this episode could add

[OpenBHB (2022)](https://doi.org/10.1016/j.neuroimage.2022.119637) already
benchmarks age prediction under acquisition differences. Simply improving
MAE or applying a new feature encoding is not a new scientific principle.

[Zhao et al. (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6819257/) already
separate shared and morphometric-family-specific variation and test age
prediction in another cohort. [Korbmacher et al. (2024)](https://www.nature.com/articles/s41467-024-45282-3)
already study hemispheric brain-age representations and asymmetry.
[Leiberg et al. (2025)](https://academic.oup.com/cercor/article/35/6/bhaf154/8171921)
study complementary information at physical cortical length scales; averaging
existing ROI summaries is not their surface coarse-graining operation.

The contribution must therefore emerge from a discriminating transfer result:
which anatomical organization is useful, under which observed conditions, and
whether matched generic alternatives explain it. A leaderboard winner alone
is a methods result. No starting branch is designated a novel discovery or
biologically privileged representation.

## Starting hypotheses, with room to search

- **Bilateral structure as a prior:** do shared and asymmetric components
  benefit from different shrinkage? A full mean/difference basis adds no
  information; use a metric-matched equivalence reference and randomized
  pairing controls to separate representation from regularization geometry.
- **Anatomical organization as compression:** do ROI/measurement-family
  groupings transfer better than generic compression at comparable dimension,
  predictor and fitting allowance? Global PCA and shuffled grouping can defeat
  the anatomical-specific interpretation.

Measurement-family combinations, atlas granularity, shared/contrast features,
nonlinear interactions and shallow predictor families remain open development
axes in the proposed design. These hypotheses are entry points, not a cap on
exploration. All features must derive from the allowed ROI columns and declared
anatomical mappings; no connectivity or physical surface scales are inferred.

## Research design and the operative reference

The existing executable contract remains the frozen R0--R3 reference below:
Desikan, the common ridge procedure, external domain-macro MAE, the +0.01 linear-site
guardrail, the -2% age margin, all controls, and the existing holdout rules are
unchanged. It can select one candidate and support only its original narrow
predictive claim; it does not exhaust the revised scientific question.

This revision authorizes writing the open representation design, not launching
it. Its candidate ranges, resource allowance, selection and confirmation scope
must be amended prospectively before execution. The current exclusions apply
to the fixed reference, not to what may be considered during design. No new
analysis, holdout endpoint or ASTRA experiment is activated by this text.

## Optional site-information diagnostic

The earlier [discarded-direction plan](outputs/site_information_diagnostic_prior.md)
is retained as an optional, unactivated branch. It can diagnose a projection
if that becomes useful; it is neither the headline nor a prerequisite for
representation exploration. Its support, fitting and confirmation limits
remain intact.

## Current access stage

The 3,227 training participants and their three official RAMP robustness views
are exposed development material. The 757 public-validation feature rows are
available without targets, but their age, `siteXacq`, and role labels remain
outside the candidate workspace.

No current experiment has started. No public-validation outcome has been
opened. A public label is not automatically a fresh audit label: the 757 rows
can be treated as a one-shot procedural holdout only if a dated exposure review
establishes that no designer, controller, accessible cache, or prior artifact
has seen a row-level join, candidate-discriminating score, or partial outcome.

If freshness is uncertain or has failed, those rows become retrospective
validation data. A genuinely new source is then required for confirmation.
Disabling network access later cannot restore freshness.

## Fixed representation comparison

The primary feature base is the 476 allowed Desikan ROI measurements. Desikan
is fixed before new evaluation for parsimony and transparent bilateral pairing;
it is not selected from held-out performance. Every arm uses the same rows,
ridge family, inner tuning rule, outer splits, and final fit-only
standardization. The fitted ridge alpha may differ by arm.

| Arm | Representation | Scientific role |
| --- | --- | --- |
| R0 | standardized raw ROIs | reference |
| R1 | deterministic bilateral mean/difference basis | anatomical representation |
| R2 | fit-only PCA with 64 retained components | generic linear compression control |
| R3 | raw ROIs after fit-only removal of four linear `siteXacq` directions | acquisition-targeted representation |

R1 pairs left/right homologues using the frozen atlas mapping and replaces each
pair by its mean and signed left-minus-right difference. Unpaired allowed ROIs
remain unchanged. It introduces no learned state.

For R3, standardize on the fit role, fit one fixed L2 multinomial logistic
model from ROI values to fit-role `siteXacq`, center its coefficient rows, and
take the leading four right-singular directions. Project those directions out
before fitting age. If fewer than four nonzero directions exist, use the full
available rank and report it. The learned transform accepts features only;
held-out site labels or statistics are never passed to `transform`.

Within this operative reference, Destrieux repeats the same four-arm comparison
as a labeled sensitivity analysis. It cannot change the primary Desikan winner or rescue a
failed primary comparison. Concatenated atlases, PLS, random Fourier features,
elastic net, Huber regression, interactions, and ensembles are outside this
fixed comparison; the open research design can consider them prospectively.

## Frozen pipeline order

Within every outer fit role and every inner tuning fold, execute the complete
arm-specific pipeline from scratch:

1. R0 passes raw Desikan columns through unchanged. R1 constructs the frozen
   bilateral basis on raw columns. R2 standardizes raw columns, fits PCA by
   deterministic full SVD without whitening, and keeps 64 scores. R3
   standardizes raw columns, fits the declared site classifier, and removes
   its four leading coefficient directions.
2. Fit a final standardizer to the resulting representation, discard only
   columns with exactly zero fit-role variance, and apply that same map to
   held-out rows.
3. Send this final standardized representation to ridge regression. The fixed
   site probe receives this same matrix, before ridge, with no arm-specific
   alternate preprocessing.

For R3, a coefficient singular value is nonzero when it exceeds
`max(number_of_site_classes, number_of_features) * machine_epsilon *
largest_singular_value`. The effective removed rank is reported. Randomized
controls, the site classifier, the probe, and all resampling use seed `14014`.

## Common age model and tuning

Each arm fits ridge regression with

\[
\alpha\in\{0.1,1,10,100,1000\}.
\]

Within each outer split, every learned step is refit using only that split's
fit role. Alpha is chosen by deterministic leave-one-`siteXacq`-out validation
inside the fit role. For each alpha, compute MAE within each inner held-out
domain and average domains equally. Choose the lowest domain-macro MAE; an
exact tie selects the larger alpha. Neither the internal-test nor external-test
rows may choose alpha, a representation, a rank, a feature block, or a stopping
point.

The fixed linear site probe is the same for every arm: fit-role
standardization followed by L2 multinomial logistic regression with `C=1`,
balanced class weights, `lbfgs`, `max_iter=2000`, and one frozen seed. It is
trained on each outer fit role and scored on the corresponding internal-test
seen-domain role. Probe accuracy is a diagnostic guardrail; it is not evidence
that scanner effects have been removed.

## Development estimands

The primary age estimand is the paired change from R0 in domain-macro external
MAE, measured in years. Compute absolute error separately for every held-out
appearance. For a participant appearing in more than one external-test view,
average that participant's appearance-level absolute errors. Then average
participant losses within each held-out `siteXacq` domain and average domains
equally. This avoids creating an ensemble only for repeated participants.

Define

\[
\Delta_{age}=MAE_{arm}-MAE_{R0},\qquad
r_{age}=\frac{MAE_{arm}-MAE_{R0}}{MAE_{R0}}.
\]

Negative values favor the alternative representation. Participant-pooled MAE,
each of the three split results, worst-domain MAE, and age-quartile errors are
secondary descriptions. The three overlapping RAMP splits are robustness
views, not independent cohorts or three inferential replicates.

For the guardrail, score the probe separately for every internal-test
appearance. Average each repeated participant's 0/1 correctness values, then
compute recall within each true `siteXacq` class and average classes. Define

\[
\Delta_{site}=BAcc_{arm}-BAcc_{R0}
\]

on the internal-test roles. The declared noninferiority margin is +0.01
absolute balanced accuracy. Passing it supports only the statement that this
fixed linear probe did not detect a material increase in seen-domain site
decodability.

Domain-clustered paired intervals resample `siteXacq` and then participants
within domain. Report every domain and a leave-one-domain-out sensitivity in
addition to the interval. All inference is conditional on the observed
domains; participant resampling cannot manufacture new domains or support a
general population-of-sites claim.

## Development selection rule

An alternative arm is development-eligible only if:

1. its point estimate has `r_age <= -0.02`;
2. its point estimate has `Delta_site <= 0.01`;
3. all preprocessing is fit-only and held-out transforms receive features
   only; and
4. the representation is finite, has the declared or reported deficient rank,
   and has not collapsed.

The direction and magnitude in each RAMP view are reported, but the overlapping
views do not vote on eligibility.

Among eligible arms, select the lowest domain-macro external MAE, then the
lowest worst-domain MAE, then lower linear site balanced accuracy, then the
simpler representation. Lock exactly one candidate. If no arm is eligible,
close the development comparison without opening the public holdout.

The 2% rule is a development selection margin, not yet a confirmed 2% effect.
Development intervals are descriptive after selection.

## Required validity checks, controls, and falsifiers

The leakage regression, fit-only access check, finite-value check, effective-
rank check, and collapse check are hard validity gates. A concrete failure
invalidates the affected arm and requires repair before selection.

The remaining controls are descriptive falsifiers:

- age-median and shuffled-age controls;
- a rank-matched seeded random-direction removal control;
- a fit-site-label-permuted direction-removal control;
- per-`siteXacq`, worst-domain, and age-quartile errors; and
- the prespecified Destrieux sensitivity analysis.

The random and permuted projections are falsifiers, not candidate arms. They
ask whether any apparent R3 benefit is specific to acquisition-predictive
directions rather than generic rank reduction or chance. Their numerical
results do not silently create a new eligibility threshold; they may narrow
the mechanistic interpretation. Only evidence of leakage or implementation
invalidity blocks a candidate lock.

## Conditional one-shot public holdout

The public holdout may open only after one candidate, R0, all code, feature
headers, mappings, seeds, thresholds, and commands are locked and the exposure
review establishes freshness.

For both the locked candidate and R0, select alpha separately on all 3,227
development participants using the same leave-one-`siteXacq`-out,
domain-macro criterion and larger-alpha tie break. Then refit each full pipeline
once on all development rows. The trusted evaluator uses:

- the 395 external rows for the paired age-error comparison; and
- the 362 internal rows for the seen-domain linear site-probe guardrail.

For the guardrail, train one probe on each final standardized representation
of all 3,227 development rows and score the corresponding 362 internal rows.
The probe never trains on public-validation rows.

Use paired one-sided 95% hierarchical bootstrap bounds, resampling
`siteXacq` and then participants within domain. A confirmed positive result
requires:

1. the upper bound for `r_age` to be below `-0.02`; and
2. the upper bound for `Delta_site` to be at most `0.01`.

Requiring the age interval—not only the point estimate—to clear -2% is what
supports a confirmed effect of at least 2%. If the age bound is below zero but
not below -2%, or either decision boundary is otherwise unresolved, report an
inconclusive holdout result rather than a confirmed 2% improvement.

Classify a valid holdout in this order:

1. `holdout_guardrail_failure` if the point `Delta_site` exceeds +0.01;
2. `holdout_no_improvement` if the point `r_age` is zero or positive;
3. `confirmed_representation_improvement` if both interval rules pass; and
4. `holdout_inconclusive` for every other valid result.

A technical failure before any candidate-discriminating outcome is visible
may be repaired. Once any score, partial metric, prediction-target join, or
other candidate-discriminating outcome is visible, the public holdout is
consumed and cannot be rerun for selection.

## Work program

1. Verify the prepared rows, atlas headers, bilateral mapping, split roles, and
   absence of cross-role leakage without opening public-validation targets.
2. Implement the four frozen representations, common ridge procedure, probe,
   and focused controls.
3. Run the complete development comparison once across all three robustness
   views and report participant- and domain-level summaries.
4. Run the falsifiers and Destrieux sensitivity without changing the primary
   selection rule.
5. Lock one eligible candidate and complete the dated exposure review.
6. Open the public holdout once only if freshness and evaluator isolation are
   established.

## Claim boundary

A positive development result supports selection of one locked representation
for external evaluation. It is not confirmation.

A valid positive holdout result supports only this narrow statement: for the
prespecified OpenBHB ROI pipeline and observed acquisition domains, the locked
representation improved paired external-domain age MAE over raw Desikan ridge
by at least 2% while seen-domain decodability under the fixed linear site probe
stayed within its +0.01 noninferiority margin.

The episode does not establish independent-cohort replication, general scanner
invariance, causal removal of acquisition effects, biological aging mechanism,
clinical utility, disease generalization, private-leaderboard performance, or
superiority for raw MRI.

This episode remains local and non-canonical. No Society decision, reward,
confirmation, or Landscape transition is implied.
