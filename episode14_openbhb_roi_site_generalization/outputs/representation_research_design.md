# EP14 — finding an age representation that transfers

## October 6 adopted bounded bilateral comparison

This selected comparison is **complete**, as recorded in the
[October 6 closeout](experiment_log.md#2026-10-06-0029-pacific--bilateral_regularization-complete-shared-allocation-closed).
Homologous pairing reproduced retuned raw ridge, while the expanded ridge
grid improved domain-macro accuracy but worsened participant-average,
worst-domain and oldest-quartile accuracy. The
[full interpretation](runs/bilateral_regularization/attempt001/interpretation.md)
retains all arms and uncertainty limits. The prospective specification below
is the record of that completed comparison, not authority to run it again.

The scientist requested further work on the available episodes while external
dependencies wait. Select one exposed-development comparison of the bilateral
regularization hypothesis already described below. The original R0–R3 result
remains closed with no eligible candidate. This follow-up cannot promote a
candidate or open public-validation features/labels. Earlier observed development
outcomes motivated the question; these are not fresh confirmatory data.

Use all 3,227 existing development participants, Desikan's 476 columns and all
three existing overlapping RAMP views. Fit raw column means/scales within each
inner training fold, then form orthonormal shared/asymmetric blocks. Do not
re-standardize those blocks. Fit ridge with shared penalty alpha in
{0.1,1,10,100,1000,10000,100000} and asymmetric/shared ratio rho in
{0.01,0.1,1,10,100}. Dividing asymmetric features by sqrt(rho) implements the
relative penalty. The expanded alpha range tests whether the original grid's
upper-bound selections explain the prior small bilateral improvement.

Tune strictly within each outer training role by mean held-domain MAE using
leave-one-siteXacq-out splits. Exact ties prefer rho closest to one on the log
scale, then larger alpha, then smaller rho. Refit once on that view's training
rows and report its original external/internal roles. Compare with raw ridge
using the same seven alphas and require the metric-matched rho=1 prediction
identity. Three fixed shams (seeds14015,14016,14017) permute right-hemisphere
partners within each complete provider-defined measurement family and receive
the same 35-setting tuning allowance. Report all shams; no winning-sham selection.
Authenticate families against `reference/freesurfer_channels.txt`, matching
complete names, not the final underscore suffix or physical unit alone.

Average repeated appearance-level losses within participants before equal-domain
aggregation. Report all views/domains, leave-one-domain-out sensitivity, worst
domain and age quartiles, plus the existing fixed linear site probe. Reuse the
existing 10,000-replicate paired domain/participant bootstrap only as descriptive
uncertainty on exposed development data. The old -2% age and +0.01 site margins
are reference values, not new promotion or confirmation gates. Homologous pairing
must be assessed against every same-capacity sham before discussing anatomical
specificity. Prediction improvement alone is not evidence of an aging mechanism.

Resource cap: one4CPU/16GiB/60-minute attended Slurm allocation, at most three
parallel workers with one BLAS thread each; worker cap3300seconds and actual
remaining-allocation closure reserve. This parent also aggregates the separately
scoped short EP01 saved-score and EP05 saved-component analyses, all charged once
to EP14. Tests and current ASTRA validation run inside it. No automatic retry,
extension or successor. Partial coverage is not a complete scientific comparison.

Bounded BR advisory retrieval returned `insufficient_evidence`, with a warning
that brain-age and asymmetry evidence came from separate publication chains.
Those generic matches do not support bilateral superiority or novelty. Earlier
primary-source context below is reused; independent implementation review and
the actual matched-control comparison determine what this experiment establishes.

Design revision, 2026-09-30. Restore the original open representation search:
which organization of the allowed FreeSurfer ROI measurements, coupled to
which shallow predictor, retains age-prediction utility at unseen acquisition
domains? Site is the environment of the test, not the object to be removed.
No candidate or favored representation is selected from data in this revision.

## Two starting hypotheses, not two allowed experiments

| Working hypothesis | Informative comparison | Alternative that can defeat the interpretation |
| --- | --- | --- |
| Bilateral organization is a useful regularization prior | Shared and asymmetric blocks with separate shrinkage versus raw ridge, a metric-matched full bilateral basis and randomized pairings | Rescaling, extra tuning or generic blockwise penalties explain the gain |
| Anatomical grouping supports transferable compression | Provider-declared ROI/measurement-family summaries versus global PCA and shuffled grouping at comparable retained dimension, predictor and fit allowance | Generic rank reduction or smoothing explains the gain |

For bilateral pairs, start from one fit-standardized raw feature matrix and
use `s=(x_L+x_R)/sqrt(2)`, `a=(x_L-x_R)/sqrt(2)`. Keeping both is invertible.
With a common metric and isotropic ridge penalty, the orthonormal change of
basis leaves predictions unchanged. Re-standardizing the transformed columns
or using different block penalties changes the metric/prior without adding
information. Truncating a block can discard information and changes capacity;
it is not an invertible re-encoding. The existing R1 re-standardizes after its
transform, so any R1 advantage is not evidence of newly added information.

The proposed shared/asymmetric comparison asks whether that changed prior is
useful for transfer. Randomized pairings should stay within the same declared
measurement family and preserve block sizes, tuning allowance and capacity.
Do not mix units or invent homologues to make a stronger-looking control.

Grouped summaries use only identities and families declared by the provider's
atlas/reference material. If the required mapping is absent, that branch stays
uninstantiated; do not infer connectivity from ROI covariance. Compare at
shared output dimensions, show the compression curve, and include group-
shuffled and generic compression alternatives. Report effective complexity
where estimable rather than equating equal dimension with equal flexibility.
Calling ROI aggregation a physical cortical length scale would be incorrect.

## Keep the exploration open

Allowed-ROI measurement families, shared/residual components, bilateral
contrasts, anatomical pooling, atlas granularity, dimensional reduction,
shallow nonlinear interactions and small ensembles remain design options.
Site-direction removal can be one optional branch, not a required stage.
The two working hypotheses seed comparisons; they do not fix a winning recipe,
exclude another scientifically motivated branch or set a two-branch quota.

Separate representation comparisons from predictor comparisons. First compare
representations within the same learner and fit allowance; then ask whether
the representation's advantage survives another learner. A better ensemble
can be a predictive result without explaining the representation. All scaling,
group weights, supervised reducers, learned projections and predictors must
be fit inside the assigned training roles, including inner tuning folds.

No additional parameter grid or trial count is frozen for the broader design
below. The narrow completed bilateral amendment above fixes its own ranges;
other candidate ranges, search resources and selection rules require their
own prospective amendment. The current CPU-only ROI scope
does not acquire raw MRI, surface meshes, connectivity, excluded tissue-volume
or QC fields, or external participant labels.

## What would count as a scientific result?

An unseen-domain MAE winner answers a prediction question. A representation
account needs more: a declared paired contrast survives its generic/shuffled
alternative, is not merely extra fitting capacity, and remains interpretable
across the observed domains and age support. Show individual domains, worst-
domain behavior and age-range errors beside domain-macro utility. Do not hide
a benefit confined to a narrow age population in a pooled average.

Evaluate utility within domain as well as across domains when a composition
shortcut is a plausible alternative. Any centered-age diagnostic belongs to
the evaluator, not to test-time features or target-dependent normalization.
Its exact population and uncertainty need prospective specification; it does
not replace deployable absolute-age MAE. Do not select age bands or domains
after seeing favorable results.

The overlapping RAMP views are development robustness views, not independent
replications. A representation-specific claim can fail while useful predictive
improvement remains. Conversely, a familiar anatomical description or stable
coefficient plot without incremental transfer utility is not a success.

## Relevant work and the actual novelty boundary

[Zhao et al. (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6819257/) use shared
and measurement-specific morphometric components for age prediction and test
transfer to another cohort. [Korbmacher et al. (2024)](https://www.nature.com/articles/s41467-024-45282-3)
study lateralization and hemisphere-specific brain-age predictions.
[Leiberg et al. (2025)](https://academic.oup.com/cercor/article/35/6/bhaf154/8171921)
study physical surface coarse-graining, which this ROI-only design cannot
reproduce. [Jirsaraie et al. (2023)](https://onlinelibrary.wiley.com/doi/10.1002/hbm.26144)
show why brain-age generalizability must be assessed directly under sampling
and acquisition differences.

These sources motivate options and competing explanations, not a claim that
EP14 invented decomposition, asymmetry or multiscale age prediction. A paper-
sized contribution remains conditional on the exact transfer pattern and its
discriminating controls; broadening the search alone does not establish it.

## Relation to the current execution contract

R0--R3, Desikan, the common ridge recipe, -2%/+0.01 margins, 3,227/757 roles,
six ASTRA analyses and closed holdout remain the existing operative reference.
Its four-arm eligibility and site guardrail are unchanged. They govern that
reference, not a universal claim that more site-decodable representations
cannot transfer. The revised open search is design-approved but not an
executable or confirmatory amendment. No new branch is inserted into ASTRA,
no trial is launched and no new endpoint gains access to the public holdout.

The prior site-information diagnostic and its v3 schematic remain optional
history. They are not the main scientific argument or an exploration gate.
The next design choice is the initial representation/predictor coverage and
paired comparisons, followed by the corresponding prospective amendment.
