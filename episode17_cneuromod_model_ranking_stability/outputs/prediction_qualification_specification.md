# Prospective prediction and calibration qualification

**Numerical conventions and margins adopted by the coordinator on October 4,
23:24 UTC after independent review; no neural/feature access or scoring activation.**
The reviewed linking, calibration, uncertainty, margin and structural-bin
procedures below are frozen for implementation. The completed structural
computation fixes **N=256 and q=32 per bin** for every participant/ROI before
calibration; it retains every nominated source-label component. The minimum
of the 64 structural bin counts is 104. References below to proposed choices describe
their scientific rationale, not a menu for later outcome-driven revision.
The private feature/fitting cost pilot remains unselected and its resource
request is provisional until the required inputs and reader handoffs exist.
Prepared from the existing GOAL, DATASETS, SEARCH_POLICY and adopted anatomical,
model and stimulus specifications. No source arrays, stimulus pixels, model
features, calibration values or candidate outcomes were inspected. Structural
feasibility results and the selected archive step remain coordinator-owned.

The structural runner has since completed. The coordinator's targeted check
found the manual atlas decode exactly equal to NiBabel and its affine equal;
the broad label footprint comes from components already present in the source
segmentation, not a demonstrated array-order error. Retain **all** nominated
source-label components, including small disconnected components. Do not prune
to a largest component, replace the atlas or treat native T1w coordinates as
MNI coordinates. The completed structural calculation froze N=256 and q=32
under the adopted bin rule; source-label validity and atlas registration
limits remain disclosed.

## Authority and decisions that are actually open

The operative contract requires prospective specification, not a separate
human-approval ceremony. SEARCH_POLICY says
`grids_layers_and_tie_rule_fixed_before_scores: true`,
`each_fixed_before_candidate_scores: true` for practical margins, and
`estimator_floor_thresholds_fixed_before_scores: true` for ceilings.
DATASETS says the ceiling choices “cannot be filled in after candidate scores
are visible.” None of GOAL, DATASETS or SEARCH_POLICY says that the scientist
must personally choose these values or approve ordinary implementation.
The model registry's earlier phrase “the scientist's practical-effect,
forecast-error and audit-adequacy margins remain unset” predates the October 4
23:24 adoption. The coordinator adopted the explicit numerical choices below
under existing delegated pre-score authority. They are frozen prospective
decisions, not empirical estimates or new requests for personal ratification.
Their scientific rationale is retained here; no later outcome-driven revision
is implied.

Changing the four participants, 720-concept universe, completed concept
roles, three required promotion pairs, five initial edges, 32–72 valid-trial
search, claim restrictions or one-shot audit would change the authorized
design. Such a change needs separate scientific resolution; an implementation
must not silently make it to rescue feasibility. Opening an audit, exposing
mixed source to a search worker, or replacing physical reader isolation with
same-UID directory conventions is not an ordinary numerical choice.

| Decision | Recorded disposition | Remaining work |
| --- | --- | --- |
| Backbones, preprocessing, four pooled layers and random seed | Already adopted in model registry | Reuse unchanged; no new layer menu or initialization |
| Inner folds, PCA/ridge grid and numerical rules | Exact shared linking method adopted below | Full role-safe feature/fitting execution |
| Repeat reliability and half assignment | Pearson correlation and numerical boundary handling adopted below | Trusted builder computes private calibration maps after successful role copying; worker release remains separate |
| Ceiling estimator, floor and quality threshold | Repeat-aware estimator and 0.10 floor adopted below | Compute actual qualified ceilings |
| Structural ROI and voxel-column mapping | DK1007/2007 and original union C-order mapping adopted | Consume completed structural evidence; never use unionNonNaN |
| N, spatial quotas and anatomical bank | Structural result fixes N=256 and q=32; bank rule adopted | Calibration must satisfy those unchanged quotas |
| Practical/forecast/adequacy margins | Scale-specific decision table adopted October 4 | Apply unchanged; they do not establish power |
| Reader handoffs and EP17/EP18 exposure | Exact overlap complete; designs retained as correlated and exposed | Physical readers remain unresolved |

### October 9 clarification: builder calibration precedes worker release

GOAL's trusted-builder permission and SEARCH_POLICY's
`support_calibration.raw_neural_reader: trusted_builder_only` already permit
calibration computation. The former table wording “after reader release”
incorrectly coupled that private stage to the later search-worker handoff.
After successful full role-copy attempt002, a bounded trusted builder may read
only the materialized `support_calibration` B/C/D responses and their fixed
role, eligibility, chronology and original-column geometry records. It may
produce the already specified reliability, ceiling and support artifacts,
keeping all responses and derived arrays builder-private. Public progress
reports contain operational dimensions, timing, resource use and completion
state, without response values or calibration statistics.

This sequencing clarification changes no image roster, numerical convention,
support quota, margin, candidate family or audit rule. It does not qualify
physical isolation or release a search/evaluation reader. The feature/fitting
cost pilot below continues to require the documented reader boundary, and
neither calibration completion nor a private directory substitutes for it.
The implementation and its bounded execution must be recorded separately;
this clarification alone does not assert completed copying or calibration.

The coordinator has adopted the delivered-response convention: B/C/D are
GLMsingle outputs from runwise voxel-standardized BOLD, with provider scaling
preserved. Physiological percent-BOLD units remain unverified. Reuse that
convention without renormalizing response voxels. R²/Pearson invariance to a
common positive scalar does not make absolute variance thresholds or
cross-voxel CKA insensitive to differing voxel scales.

### Builder-private calibration adapter and fixed cost pilot

The selected adapter is `outputs/tools/calibrate_private_roles.py`. It requires
successful full role-copy attempt002 and checks all selected ordinary private
calibration HDF5 headers and physical stripe bounds before reading responses.
It consumes the exact copied row map, frozen eligible image identities and
retained-acquisition ordering, original-column ROI/bin table and unchanged
calibration/support cores. It reads no development, audit or mixed-source
response array. Every eligible calibration image is retained; no finite-value
image deletion is allowed. B/C/D response eligibility defines A_all, while
D-stage valid reliability selects R_N with the existing quotas and tie rule.
Negative finite ceilings are preserved with the already frozen 0.10 floor.

The prospective cost pilot uses the first128 ascending original-column IDs in
sub-01's left anatomical ROI, all605 eligible calibration images and their
retained repeats at B/C/D. Each copied calibration product has2129 rows; the
adapter preserves the metadata-defined eligibility rather than treating every
copied row as a qualified image. Select one CPU,2GiB,300seconds internally,
360seconds externally plus10seconds termination grace and64MiB derived-output
cap. The actual touched512-column physical stripes and padded row-chunk bytes
are frozen from metadata before responses. Do not extrapolate solely from the
128 selected columns: sparse columns can touch many storage chunks.

Run the six focused adapter checks in coordinated compute before that pilot.
Pilot response-derived arrays and the exact read plan remain builder-private;
only operational counts, elapsed time, RSS, completion and fixed error codes
are public. The pilot is not candidate fitting and cannot select supports.
Full calibration requires its successful matching pilot and a separately
selected measured-cost duration, at most7200internal seconds with the same
2GiB limit and256MiB combined derived-output cap including preserved pilot
artifacts. No full duration or new job is selected here.

The full adapter's `calibration_support_complete` covers only saved B/C/D
reliability/ceiling vectors and A_all/A_N/R_N/V_NC artifacts. Gain-operator
applicability, worker release, candidate fitting and evaluator qualification
remain separate. The execution log records the completed pilot/full routes
and their actual compute evidence.

### Builder-private calibration-derived operators

`tools/derive_private_operators.py` reads only the completed private calibration
read plan, maps and supports. It joins original column IDs and keeps each
participant/ROI's complete frozen A_all axis. B→C and C→D Fisher-z gain maps
use the existing spatial-bin design, residual variance floor1e-8 and relative
SVD tolerance1e-12. Undefined gain or a failed frozen design condition makes
that beta explanation and its corresponding all-group H component inapplicable;
it does not remove voxels or invalidate otherwise eligible beta endpoints.
Original-column tie rules define the high/low gain strata.

Preserve the 32 anatomical banks and q=32 quotas. Save A_all, mean A_N and R_N
weights, their registered differences, and fixed V_NC raw/normalized endpoint
weights. The canonical positive ceiling map `g=1/max(NC_D,0.10)` is a separate
artifact from the edge-difference weights `(g-1)/len(V_NC)`; neither is
renormalized to sum one. Negative finite ceilings remain eligible under the
fixed floor. All maps, designs, strata and applicability decisions stay
builder-private; public output exposes only operational counts/status/resources.

Use one CPU,2GiB,120internal/180external seconds plus10second kill grace and
32MiB new derived-output cap. Four adapter checks accompany the implementation;
reuse the already passing numerical cores. Run this stage within a substantive
allocation. It performs no raw response/image access, model fitting, forecast,
scoring, worker release or evaluator qualification.

## Adopted common linking method

Use the six already frozen 80-concept outer folds. For each outer fold, the
other five folds are the five inner validation folds: each inner fit uses
320 concepts and validates on80; outer refit uses all400 training concepts.
Every eligible image of a concept stays together. Unique images have equal
training and scoring weight; repetition averaging is the registered equal
mean, not an additional weight. Reuse identical partitions for all four
models, participants, ROIs and B/C/D stages.

Use the adopted ranks **32,64,128** and penalties
**0.0001,0.001,0.01,0.1,1,10,100**, with the four adopted layers L1–L4.
This is84 settings per model per selection problem. No zero-penalty fit,
layer concatenation, spatial flattening, seed search or grid expansion is
included. Before any neural selection, the source-free implementation and
training-feature feasibility check must establish that the full grid is
available to every model on every required training partition; otherwise
stop and resolve the common prospective grid before scores, rather than give
one model a smaller opportunity set or silently lower its rank.

For each model/layer/training partition, convert stored features to float64,
subtract the training-image feature mean, and compute a deterministic full
SVD. Do not variance-standardize individual feature channels or whiten PCA.
Retain singular values with `s > 1e-12 * s_max`; an all-zero centered matrix
has rank zero. A requested rank must be available, and a cutoff through an
exactly degenerate singular-value group is a numerical ambiguity to report
rather than resolve from neural scores. Fix the sign of each retained loading
by making its largest-absolute coordinate positive, taking the first feature
index on an exact tie. Apply the stored training mean/loadings to validation
or audit features; no held-out transform is learned.

For rank q, scale the resulting training score matrix Z by
`sqrt(n_train / sum(Z**2))` and apply that same scalar to held-out scores.
This single global scale gives the shared penalty grid a defined meaning
without whitening or changing relative principal-component scales. Fit the
training response mean as an unpenalized intercept separately for each voxel
and stage. With centered targets Y, solve

`min_B mean_images ||Y - Z B||² + lambda ||B||²`,

equivalently `(Z.T Z + n_train * lambda * I) B = Z.T Y`, in float64.
Predict the voxel mean plus ZB. Do not center responses with a held-out mean.

Select one layer/rank/penalty **per model**, shared across all B/C/D stages,
participants and nominated ROIs for that outer fit. Construct the inner
out-of-fold SSE and baseline SST for each voxel over all400 outer-training
concepts, with each image's baseline supplied by its inner-training response
mean. Compute one pooled inner-validation R² per voxel, then average voxels,
the two ROIs, four participants and three stages in that order. This fixes
the otherwise unspecified aggregation across inner folds without giving
individual images extra weight. Exact objective ties use the registered
shallower layer, smaller rank, larger penalty order; do not round objective
values or add a data-chosen tolerance.

Refit endpoint-specific B/C/D voxel coefficients using the chosen setting
on the400 outer-training concepts. The D predictions are the **same bank**
for all anatomical, reliable-support and ceiling endpoints; those masks and
weights act only after prediction. A missing/nonfinite prediction invalidates
the paired trial. Retain all negative R² and all normalized values above one.
For the final audit predictor, select the same common setting using the six
development folds as the all-development validation scheme, then fit on all
480 development concepts. Freeze transforms, means and B/C/D coefficients
before audit. No audit refit is permitted.

For the required secondary linear-CKA falsifier, use the adopted common rank64 at
each model's frozen encoding-selected layer, using its training-only transform.
Use centered linear CKA on the same held-out images and response support as
the compared encoding endpoint, retaining the delivered response column
scales without per-voxel whitening. For development, compute it separately
in each of the six held-out outer folds with that fold's frozen model layer
and rank64 transform, then average using each fold's unique-image count.
For audit use the all-development frozen transform and compute it once on
the audit images. Declare this exact development-to-audit counterpart; no
single shared PCA basis is falsely claimed across development folds. For an
anatomical-bank endpoint average the32 per-subset CKA values, as with the
encoding endpoint's bank, and retain equal ROI/participant averaging. Use
`||Xc.T Yc||_F² / (||Xc.T Xc||_F * ||Yc.T Yc||_F)` in float64; zero/nonfinite
denominators invalidate this secondary comparison. Centering over the scored
image set is the definition of this symmetric descriptive statistic, not a
new fitted feature transform. CKA has its own margin and never replaces R²;
its response scaling dependence remains part of the declared linking test.

### Development secondary resampling composition — October 9

The following pre-score implementation convention applies the existing
participant-by-concept-block bootstrap to the adopted secondary estimands.
Use the separately frozen24-block metadata order. An image's integer weight
is its concept block's bootstrap multiplicity; this is exactly whole-block
row replication, with equal weight for every unique image in each copy.
It does not introduce repetitions as independent observations or refit a
model/feature transform. Participant multiplicities act only after the same
within-ROI and equal-ROI calculations.

A participant absent from a bootstrap draw has zero mass and need not have
that draw's statistic evaluated; this matches literal participant replication.
Every positive-multiplicity participant must have both valid ROI statistics.
The original unit-weight estimate requires all four participants. This rule
does not permit deleting a selected participant or repairing a failed draw.

For held-out correlation, pool each voxel's complete OOF image predictions
and observed image means, compute centered weighted Pearson correlation, then
apply the existing fixed Fisher transform before support/ROI/person averaging.
Use the existing relative variance safeguard for each scored prediction and
response vector; retain the existing1e-12 Pearson roundoff tolerance and1e-7
Fisher boundary distance. A nonfinite or degenerate required voxel invalidates
the comparison rather than changing its frozen support.

For development CKA, retain each outer fold's already frozen selected layer
and training-only rank64 transform. Within that fold, weighted centering and
weighted cross/Gram products must equal computing CKA on its replicated image
rows. Weight each fold's CKA by its **resampled image count**, normalized over
all folds. A fold containing no resampled images has zero weight and requires
no numerical CKA; this absence follows solely from resampling metadata. Any
positive-weight fold with an undefined statistic invalidates the whole draw
and comparison; do not redraw, drop its weight or shrink the support/family.
With unit block weights this reduces to the already adopted original unique-
image-weighted fold mean. Audit remains its separately specified single
all-development transform, not a development fold mixture.

For A_N, always average the32 actual frozen subset CKAs in each fold before
the fold/ROI/person aggregation. Mean inclusion weights are sufficient only
for the linear average of voxelwise Fisher-z scores; they are not a CKA
substitute. Bind supplied bank membership to original columns, spatial quotas
and frozen inclusion counts. Each scale retains its complete eligible six-
support family and separate max-t calculation using the existing10000 draws,
seed and failure rules. No new margin, model, support or audit authority is
introduced. Root and independent method review agreed before empirical scores;
these conventions and their implementation remain untested on compute.

## Calibration and finite-response rules

Preserve the completed acquisition-order/alternating-half mapping. For each
participant, stage and voxel, correlate the two half means over all eligible
calibration images. Use float64 centered sums and the same image roster for
both halves. Require finite values and nonzero numerical variance in both
halves. Values outside[-1,1] by more than `1e-12` are invalid; only roundoff
within that tolerance may be brought to the boundary. For Fisher-z maps use
`atanh(clip(r, -1+1e-7, 1-1e-7))` and report the count affected by this fixed
boundary convention. Ranking R_N uses the unclipped finite Pearson r, with
original model-column index ascending as its final tie-break.

Avoid treating unequal-sized retained halves as equal-sized observations in
a naive Spearman–Brown correction. Use the adopted repeat-aware calibration ceiling:
for image i with n_i retained presentations, calculate its equal mean m_i and
unbiased within-image sample variance v_i. Across the M calibration images,
let V be the unbiased sample variance of m_i, and E be `mean_i(v_i/n_i)`.
Define `NC = 1 - E/V` separately for B,C,D. Require all fixed calibration
images with n_i≥2, M≥120, finite inputs and positive qualified V; no selective
image deletion. Keep negative estimates as estimates, with the normalization
denominator `max(NC,0.10)` for every stage. The ceiling-quality rule is finite
NC from this complete estimator, not an additional positive-NC selection.

This estimates the variance reliability of the calibration image-mean
response under independent, possibly unequal-variance repeat noise. Correlated
noise across repeats or different development/audit repetition distributions
can make it an imperfect normalization for those roles. It is not a certified
attainable R² ceiling for a score centered at a different development mean.
Keep that limitation explicit; do not call the normalized score percent of
biological truth. The adopted estimator and 0.10 floor are prospective choices,
not data-supported optimal settings. The public mixed-role D ceiling remains
closed to this step.

For numerical response eligibility, y is the vector of registered equal-
repetition **image means across all eligible calibration images**, separately
for each participant, voxel and B/C/D stage; it is not the pooled vector of
trialwise betas. Require centered variance
`V_population > 1e-12 * mean(y**2)` and `V_population > 0`, computed in float64
with centered accumulation. Both sides rescale together if response units are
multiplied by a constant. This is a numerical nondegeneracy rule, not a minimum
scientific response amplitude; it does not establish physiological percent-
BOLD units. Apply the same relative safeguard separately to each half's
image-mean vector before a Pearson calculation. Require finite values at
B,C,D. Use calibration only
when defining support; development and audit values must never rank, repair
or shrink a selected support. Before candidate predictions, check the frozen
support against development endpoint and fold/block denominator validity,
using the appropriate image-mean vector in each training and validation
partition. **Adopted October12:** only for primary R², do not use individual
uncertainty-block centered variance or its relative threshold as an independent
veto. Retain finite inputs and finite computed moments in every such block,
and strictly positive finite SST for every required column in each individual
block around its corresponding frozen training mean. There is no pooled-only
exception, denominator floor, support deletion or changed baseline. Within-set
variance and training-baseline SST are not interchangeable. See the
[adopted bounded amendment](proposed_score_specific_validity_followup.md).
All training/validation and calibration variance requirements are unchanged.
Pearson/Fisher-z and CKA retain their own actual-vector validity rules, and
all bootstrap/undefined-draw and registered-family requirements remain.
A failure invalidates that configuration, with no voxel deletion. Audit does
the homologous check only inside the one-shot evaluator, on the frozen support:
retain whole-audit response variance, finite input/moment checks and positive
finite training-baseline SST in each of its12 blocks; only the individual
primary-R² block centered-variance veto is removed. It cannot pre-screen audit
responses or rescue invalid columns.

For the beta operator, standardize the calibration Fisher-z gain within each
participant/ROI. Use the already fixed spatial-bin-intercept OLS, with no
tuned penalty. Use the adopted residual gain variance floor of `1e-8` after that
unit-variance scaling and projection onto the spatial-bin intercepts; use
relative SVD rank tolerance `1e-12`. An absent gain dimension is inapplicable.
Sort by gain and then original column ID for the median split; the first
floor(n/2) form the low group and the remainder the high group, which resolves
median ties without deleting voxels. Retain the registered twelve/twelve
development-block cross-fit and separate pooled-image primary effect versus
equal-block-mean forecast target.

## Frozen structural bins and quota

Keep left/right DK fusiform as two separate ROIs and give them equal weight.
Each is one nominated anatomical parcel; no extra anatomical subparcel is
invented. Retain all source-label components and use their actual authenticated
native T1w geometry; disconnected fragments are not silently removed or moved
to another binning universe. Before calibration, define eight geometry-only bins per
participant/ROI: sort original structural columns by world y (posterior to
anterior in the authenticated RAS frame), then model-column ID; split into
four consecutive groups whose sizes differ by at most one, assigning remainder
to the earlier group. Within each group sort by world z then original column
ID and split into lower/upper consecutive halves, extra element to the lower
half. These are spatial bins, not eight biological replicates or subregions
chosen from responses. Their assignments never change after filtering.

The adopted rule is `q = min(32, floor(min_all_64_structural_bin_counts / 2))`, with
`N=8*q` and q≥8. Thus every participant/ROI and bin has the same quota, N is
at most256, and each bin initially leaves at least twice the requested count
available for a nontrivial support comparison. The completed
[structural report](runs/structural_support/attempt001/structural_support_report.json)
records all 8,558 nominated rows, minimum bin count104, q32 and N256, with
all original columns retained in their recorded bins. No neural-response
filter was used to choose the quota. A structurally infeasible nomination is unresolved, not
permission to replace fusiform, merge hemispheres or drop a person.

Once the actual N and bin quotas are frozen, later calibration eligibility
must supply at least q A_all voxels and at least q finite-D-reliability A_all
voxels per bin. Otherwise refuse the affected contract; do not lower q or
move a voxel across bins. Build32 anatomical subsets by selecting q A_all
voxels per bin under deterministic seed17 rankings, independent of reliability
magnitude or whether that voxel's D-reliability is finite. Rank by SHA-256 of
UTF-8 `ep17-anatomical-bank-v1|17|SUBJECT|ROI|BIN|BANK|COLUMN`, then original
column index on a digest tie. Use ROI tokens `left`/`right`, BIN01–08 in
posterior-to-anterior group order with lower then upper half, BANK01–32, and
un-padded decimal nonnegative COLUMN. This is a reproducible sampling rule,
not a source-identity receipt. No reroll for a favorable bank is allowed.
R_N takes the top q finite D-reliability A_all voxels in each identical bin. A_N is
the arithmetic mean of all32 subset means, not the union of their voxels.
Report overlap and unique-bank counts descriptively; bank members do not
increase trial count or inferential sample size.

## Adopted margins and their meaning

These values are adopted decision thresholds, **not established field
standards, power estimates or thresholds estimated from this dataset**.
The October 4 23:24 adoption fixes them before candidate effects. On October 6
the corresponding stale null policy fields were synchronized to that record;
this did not choose new values. Keep exact decimals and separate scales.

| Effect scale | Adopted practical margin | Maximum full simultaneous audit interval width |
| --- | ---: | ---: |
| Raw held-out R² advantage |0.01|0.02|
| Ceiling-normalized R² advantage |0.05|0.10|
| Fisher-z held-out correlation advantage |0.02|0.04|
| Linear-CKA advantage |0.02|0.04|
| Beta high-minus-low gain effect, raw-R² units |0.01|0.02|

The raw margin requires one percentage point of held-out variance fraction;
the normalized margin is five percentage points on its distinct
ceiling-weighted scale. Neither is an equivalent-size conversion of the other.
Full-width thresholds of twice the corresponding practical margin prevent
calling a very imprecise failure adequately precise. They do not guarantee
adequate precision with four participants.

Use absolute forecast-error limits 0.02 raw-R² units for beta block-mean
and support aggregate forecasts, and0.05 for the registered raw-to-normalized
aggregate edge-change quantity. The latter remains the contract's particular
edge contrast, not a pooled test of raw and normalized endpoint effects.
Freeze all limits before any candidate score; final relation selection only
chooses the already defined relevant limit. A wrong sign or excessive error
fails explanation even when the reversal itself passes.

## Remaining implementation details and useful next computation

Implement exactly10,000 crossed bootstrap draws: resample four participants
with replacement and independently24 development or12 audit blocks with
replacement. Preserve whole concepts, recompute pooled-image SSE/SST endpoints
with their frozen training means, and keep the beta forecast's block-mean
companion separate. Use the adopted NumPy PCG64 seed1729. In each draw generate the
four participant indices first, then the block indices; use common draws for
all paired components. Use componentwise bootstrap standard deviations with
denominator9999, and the95% max-absolute-standardized-deviation critical value
over the frozen family: center draw b at the original estimate,
`theta_b,j - theta_hat_j` (not the mean of bootstrap estimates), and divide by its
across-draw standard deviation, take the maximum absolute value over
components, and select sorted order statistic9501 (one-based), corresponding
to `ceil(.95*(10000+1))`. Report `theta_j +/- critical * SE_j`.
Reject degenerate zero/nonfinite standard errors or undefined resampled
statistics without a tuned variance floor, skipped draws or redraws.
Raw/normalized components share a max-t family but retain their own effects,
margins and units. This document does not claim that the implementation or
evaluator qualification is complete.

The calibration, support, measurement and complete crossed-uncertainty cores
subsequently passed all 33 focused checks on October 4; the
[execution record](experiment_log.md#2026-10-04-2350-utc--numerical-cores-pass-exact-endpoint-conventions-adopted)
preserves their evidence. Reuse these passing checks. They do not establish
empirical calibration eligibility, reader isolation or a completed evaluator,
and are not a reason to expand a synthetic benchmark.

After exact role-filtered handoffs and the documented reader boundary actually
exist, run an outcome-blind cost pilot inside the trusted builder, before an
exhaustive all-partition SVD feasibility pass: hold outer-fold01 completely
aside, use folds03–06 as the320-concept training set and fold02 as the80-concept
validation set. Select the first16 original-column IDs
per participant/ROI on the frozen support, all three trained models and the
single random model,
all B/C/D targets, and every registered layer/rank/penalty. Do not compute
validation scores, selector objectives or a chosen hyperparameter. Keep all
fits and validation predictions private and discard pilot coefficient/prediction arrays;
release only dimensions, operation counts, finite/success flags, timings and
peak RSS. This pilot would expose private candidate fits to the builder and
therefore needs the pre-score design fully frozen and the access boundary
resolved, even though no controller score is returned. Propose initially
one CPU,6GiB,600 seconds internally/660 seconds externally plus10 seconds,
with resource selection reviewed after source feature and structural sizes.
No pilot is selected by this document; full six-fold work must use measured
cost, the original global budget and a separately selected bounded job.

The five-edge endpoint/audit-homolog table, numerical degeneracy rules,
frozen margins and EP17/EP18 exposure decision are complete. Actual calibrated
support, full matched readout execution, isolated source/feature handoffs and
the outcome-free evaluator dry run remain required before scoring. This is the existing
contract's scoring boundary, not an extra approval gate. Unknown checkpoint
exposure stays unknown under the adopted measurement-comparison amendment;
do not invent a requirement to prove clean pretraining before writing code.
