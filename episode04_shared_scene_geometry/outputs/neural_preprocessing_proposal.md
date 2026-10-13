# EP04 adopted neural preprocessing recipe

Prepared and adopted prospectively by the coordinator on October 3, 2026,
with the identical-training-image context requirement below. **No neural
preprocessing or scoring has executed.** The fixed first-session selected-row
handoff completed, and the coordinator then selected the same source-only
route for the remaining149 sessions.
No beta, ROI, noise-ceiling or raw-image value was opened to prepare this
document. The separate frozen development-embedding handoff subsequently
completed in job46433944.3; its code is unchanged by this proposal.

## Scope and decision authority

These are concrete pre-score implementation choices inside the current
contract: `GOAL.md` grammar item6 specifies session-standardized,
repetition-averaged TYPED patterns, minimum two valid presentations, and
training-fold preprocessing/reliability; item8 fixes the three LAION sectors
and retinotopic EVC. `DATASETS.md`, “Required frozen development inputs,”
requires the standardization, finite-value, repetition and ROI rules to be
recorded before scoring. Neither provision imposes another scientist-approval
step for choosing those rules. The coordinator adopted these choices before
numeric execution under the existing episode authorization.

The explicit separate approval remains the raw-image/new-detector extension:
`DATASETS.md` says, “If a proposed extension requires raw images or a new
detector, stop until the separate DUA and model/version are approved and freeze
a revised policy before neural scoring.” This proposal uses neither. The
multiple-embedding scientist-choice clause is also unchanged; the documented
default four embeddings have already been bound. No new ROI, panel row,
primary distance, practical margin or audit opening is proposed.

## Frozen inputs and source interpretation

Use only the existing `metadata_folds04` development-image/trial exports,
`array_metadata02` header/sidecar report and adopted `config/roi_mapping.json`.
They bind sub-01/03/05/06/07, sessions01–30, 23,560 unique images, 94,216
presentations and the same five globally grouped, source-balanced folds.
The supplied intended count is four repeats; observed metadata have 23,536
four-repeat and24 three-repeat images. Neural validity remains unmeasured.

The [provider beta documentation](https://laion-fmri.hebartlab.com/glmsingle_betas.html)
identifies TYPED as the released representation, with voxelwise shrinkage and
possible NaNs. The original beta scale is percent signal change, subject to
that shrinkage. The [pinned loader](https://github.com/ViCCo-Group/LAION-fMRI/blob/cf82765ced029de81c3bec7254b8cbd6a65e3686/laion_fmri/subject.py)
distinguishes a failed-fit NaN from a zero estimate. Preserve that distinction;
zero is a valid beta, and unavailable estimates are never zero-filled.

The [provider ROI description](https://laion-fmri.hebartlab.com/rois.html)
states that the three high-level sectors subdivide `laion-general`, whose
visual selectivity uses main-experiment GLMsingle R². These fixed release ROIs
are required by EP04; they are not anatomy-only masks or masks proved
independent of every evaluated stimulus. Released TYPED preprocessing is also
upstream of EP04's folds. The claim is training-only **downstream EP04**
preprocessing, not refitting the original GLM or guaranteeing that every
released preprocessing decision excluded evaluation images. Report this
lineage in the exposure/interpretation record; do not replace the prescribed
ROI family or invent a new readiness test from it.

## 1. One development-only numerical handoff

The trusted reader uses exact metadata image roles and the original
`session_beta_axis_index`. The run-local `beta_index` remains an identity
field; it must never be substituted for the session's fourth-axis index.
No filename prefix heuristic, provider random split or historic response
cache supplies the role join.

Read the15 fixed high-level masks and30 retinotopic components on their
already-qualified native array indices. Retinotopic EVC is the union of
V1d/V1v/V2d/V2v/V3d/V3v. A finite nonzero mask value is in-mask; NaN is
out-of-mask, as in the provider's Boolean convention. Infinite mask values
are a malformed-input failure, not an inclusion decision. Preserve the source
voxel ordering, same-index mapping and measured affine discrepancy. No
interpolation, reorientation, added R² mask, released NC threshold or ROI
boundary change is introduced.

For the selected beta NIfTIs, stream one original volume at a time and decode
numeric beta elements only for the exact development trial rows. Nonselected
volumes may have to pass through the gzip decoder as opaque bytes; disclose
this codec boundary. They must not become numerical arrays, summaries or
exported responses. Retain selected voxel indices and the union of the four
registered masks, with a separate membership column for each ROI. A full-volume
or provider `get_betas(..., stimuli="unique")` call is unsuitable: the pinned
loader applies its stimulus filter after numerically loading all trials.

The role-filtered candidate artifact contains only original development
float32 betas, original image/trial IDs, mask membership and validity reasons.
No shared-regular/OOD beta array, R²/NC map or source path that requires opening
a mixed array is a candidate input. Float64 is used for later statistics.
No additional anatomical brain-mask product is needed: the fixed ROI membership
and the training-only finite/variance rules below define the working support.
The same-UID steward/source layout is not claimed to be an enforced OOD seal;
the audit still requires its actual separate custodian boundary.

## 2. Fixed primary voxel rule; no reliability-ranked subset

For each participant and outer fold `f`, training images are the other four
frozen folds. All repetitions of an image inherit that membership. Within
each registered ROI, retain every voxel that:

1. is in the fixed source mask;
2. has finite betas for **every training presentation**, over all30 sessions;
3. has a finite strictly positive training standard deviation in every session.

No beta magnitude, RSA value, image embedding, repeat correlation, NC cutoff,
top-k ranking or percentile selects a voxel. The resulting mask may differ
across outer folds but is identical for the four model rows, neural-distance
sensitivities and encoding within that fold. It is frozen before evaluation
betas are used. Evaluation NaNs never remove or reweight voxels.

Require at least three retained voxels per ROI for a nondegenerate
correlation-pattern analysis. Record input/retained voxel counts and each
training-only exclusion reason. An empty or structurally unusable required
ROI becomes an explicit not-evaluable row; it is not replaced with a different
ROI, participant, mask threshold or earlier result.

This deliberately retains noisy but computable voxels. Reliability is reported
as context rather than used for a tunable voxel-selection rule.

## 3. Train-image-only session standardization

For participant `p`, session `s`, fold `f`, retained voxel `v`, let `T_psf`
contain every development presentation in that session whose image is in the
four training folds. At least two such presentations are required. Define

```text
mu_psfv    = mean_{t in T_psf} beta_t,v
sigma²_psfv = mean_{t in T_psf} (beta_t,v - mu_psfv)²  (ddof=0)
z_t,v       = (beta_t,v - mu_psfv) / sigma_psfv.
```

The population-variance convention is fixed, not selected from a score. It
also makes the training presentation variance one within each session for the
provider-formula noise-ceiling calculation below. Apply these same parameters
to training and evaluation presentations in that session. Do not estimate
location/scale from all images in a session, from the evaluation fold, from
regular-shared/OOD presentations, or from a selected model's predictions.

A session with fewer than two training presentations cannot borrow parameters
from another session or use evaluation observations. Nonpositive/nonfinite
voxel scales cause the training-only voxel exclusion above; if usable ROI
support is lost, report not evaluable. No epsilon scale, winsorization,
additional smoothing, response PCA, whitening, spatial reliability weighting
or model-fitted nuisance regression enters the primary neural geometry.

## 4. Valid repeats and common image support

A development presentation is primary-valid only when its standardized vector
is finite on every retained voxel in each of the **three primary sectors**.
This common presentation rule makes the primary images and repeat counts
identical across sectors and all four model rows. It uses only the frozen
finite-value rule, never model agreement, beta amplitude or an RSA contribution.

For each image, average **all** its valid standardized presentations with equal
presentation weight. Include the image in its original evaluation fold only
when at least two valid presentations remain. Never replace an invalid repeat,
choose the best two, weight by session reliability or impute a response.
All excluded image IDs/counts/reasons remain in the accounting. Folds and
near-neighbour components are not reassigned or rebalanced after missingness.

The mandatory complete-repeat sensitivity uses the same frozen preprocessing
and voxel masks but restricts evaluation images to at least four valid
presentations. The24 metadata three-repeat images can enter the minimum-two
primary but cannot enter this sensitivity. Retain original intended, observed
and valid counts as three distinct fields. Unexpected over-repetition is
reported; it is not silently truncated to four.

The EVC diagnostic uses its own training-qualified EVC mask and valid repeats
for the **same retained primary evaluation image IDs**. It cannot remove an
image from the primary because EVC is noisy. If fewer than two EVC repeats
remain for a required diagnostic image, retain that missingness explicitly;
do not claim a complete same-image comparison from a favorable intersection.
This is diagnostic completeness, not an EVC-versus-high-level score threshold.

After these prespecified finite/repeat exclusions, at least three images must
remain in each required fold and source-stratum analysis. A missing cell or
constant/undefined RDM is not evaluable under the existing missingness rule;
no row or stratum is silently dropped. Identical candidate/comparator coverage
is mandatory. Feature coverage and the pending low-level control remain
separate requirements; neural eligibility cannot make a missing model feature
acceptable.

## 5. Reliability and provider-formula NC, fitted on training repeats

Reliability and NC use only training images and are computed separately for
each participant, fold and registered ROI on that ROI's frozen working mask.
Use one common training image-ID set across the ROI comparison, retaining all
valid training repeats. Do not read the released pooled `Noiseceiling4rep`,
`12rep`, `Allrep`, per-session R² or NC values for voxel selection or parameters.

For repeat consistency, order presentations by session, run and original
session-axis index, then assign alternating repeats to two groups. For an image
with three repeats this is2 versus1, and for four repeats2 versus2. Average
within each group and correlate the two image-response vectors across training
images separately for each voxel. Report the ROI median, distribution and
undefined-count of these signed Pearson correlations. Do not clip negative
correlations, apply Spearman-Brown as though unequal partitions were equal,
or select a favorable partition. Chronological alternation is a descriptive
repeat-consistency definition, not proof of independent errors.

The [provider's NC formula](https://laion-fmri.hebartlab.com/glmsingle_betas.html#noise-ceilings)
uses within-image residual variance and an unequal-repeat correction. Here is
the proposed re-estimation on the allowed training set `I`, **after the fixed
within-session training z-scores above**:

```text
n_i = number of valid training presentations of image i
m_i,v = mean_j z_i,j,v
s²_i,v = sum_j (z_i,j,v - m_i,v)² / (n_i - 1)
eta_v = (1/|I|) sum_i s²_i,v
signal_v = max(0, 1 - eta_v)
h_I = (1/|I|) sum_i (1/n_i)
NC_v(I) = 100 * signal_v / (signal_v + eta_v * h_I).
```

Each image contributes one residual variance with its actual3/4 (or otherwise
valid) repeat count; a four-repeat image is not implicitly weighted more than
a three-repeat image. Session standardization precedes grouping repeats across
sessions. No session-average NC is substituted for this cross-session formula.
Record every image count, repeat histogram, session contribution and whether
repeats span sessions; source metadata presence alone does not establish
independent measurement errors. Pooling within-image residuals across sessions
also captures session-to-session inconsistency, which is part of this context.

The mean residual variance must be finite and nonnegative. `eta=0` gives the
formula's limiting NC100 (zero estimated repeat noise), recorded as such
without an epsilon or an infinite stored SNR. `eta>=1` gives NC0 through the
provider's nonnegative-signal convention; it does not exclude that voxel.
Nonfinite inputs, fewer than two valid repetitions, or no eligible image set
make that estimate undefined, not zero. Report ROI equal-voxel median and
10th/90th percentiles plus undefined counts; do not use NC to rank/select
voxels or to normalize primary RSA. This is an estimated encoding-variance
context, not a strict ceiling for Spearman RSA.

The exact policy requirement is
`frozen_roi_reliability_and_provider_noise_ceiling_values_on_the_same_image_set`.
Released pooled NC maps cannot certify a fold-specific unique-image set.
**Adopted implementation interpretation:** use the provider-defined formula
re-estimated on the common permitted training image set, and identify its
source, exact IDs, repeats, session standardization and voxel support. It is
not a claim that the released map is a training-only estimate or that this
diagnostic is already fulfilled. The ordinary evaluation-fold high-level/EVC
RSA comparison must clearly identify its evaluation IDs separately from those
training-context IDs. A separately labeled **training-context** table of raw
high-level/EVC RSA, repeat reliability and provider-formula NC on identical
training image IDs `I` and the same frozen per-ROI voxel/repeat support is
required. It cannot select the primary candidate or be called held-out
performance. The separately reported held-fold raw high-level/EVC RSA must
never be described as having a noise ceiling estimated on those held images.
Completion requires actual matched identity/support evidence; no existing
diagnostic requirement is waived by this prospective interpretation.

## 6. Folds, neural geometry and the permitted encoding diagnostic

Reuse all five folds unchanged; the same source component never crosses folds,
including across participants. Never split RDM edges. For each model/participant/
ROI/fold, both endpoints of every scored primary edge are eligible evaluation
images in that fold. Compute correlation distance on the standardized,
repeat-averaged response vectors and the fixed native cosine model RDM.
The cosine-neural sensitivity uses the same vectors/image IDs/voxels without
the per-image spatial centering intrinsic to correlation distance.

The adopted identical-fold encoding diagnostic is ordinary multivariate ridge
on row-L2-normalized native embeddings, with an unpenalized intercept and one
fixed normalized-loss penalty `lambda=0.001` for every model, participant,
sector and fold:

```text
min_W,b ||Y_train - 1 b^T - X_train W||_F² / N_train
        + 0.001 ||W||_F².
```

Center X and Y using training-image means to fit the intercept; do not whiten,
change layers, search ranks or select lambda using evaluation outcomes. The
fixed lambda is a concrete diagnostic capacity choice, not a claimed optimal
value or a new primary candidate. Fit only on training images with at least
two valid repeats and use the same outer-fold neural preprocessing and voxel
support. Report held-out squared errors and training-mean-baseline `R²_SSE`
over the full image-by-voxel matrix, preserving negative values:

```text
SSE = sum_{i in evaluation, v} (Y_i,v - prediction_i,v)²
SSE_baseline = sum_{i in evaluation, v} (Y_i,v - mean_train(Y_,v))²
R²_SSE = 1 - SSE / SSE_baseline.
```

This explicitly uses the training-mean predictor as the baseline; it is not
test-mean-centered variance explained. A zero/nonfinite baseline or nonfinite
prediction makes this diagnostic undefined without an epsilon. The exact
evaluation image IDs match primary RSA; any required diagnostic missingness
is explicit. Encoding disagreements are reported as readout dependence and
cannot alter native-RSA selection.

No additional session nuisance fit is proposed. The already-registered
within-evaluation-fold partial Spearman low-level control is retained exactly
as its specified association statistic; it is not replaced by a new learned
neural preprocessing transform. Its required low-level features and separate
raw-route approval remain pending. Optional crossnobis stays inactive unless
valid independent repeat partitions and a permitted noise estimator are
established; alternate repeat labels alone do not establish independence.

## Concrete next implementation and bounded cost pilot

The initial implementation is `code/prepare_development_neural_pilot.py`, a
development-row NIfTI streamer and fixed-mask handoff only. It reuses existing
header/trial joins. Five new fixture methods cover NIfTI F-storage/C-index
correspondence, session row versus run-local index, exclusion before numerical
decoding, preservation of NaN versus zero, mismatched headers and truncation.
These five tests passed in root's46433944.7 compute step. Later preprocessing fixtures
should check that arbitrary
evaluation-data changes cannot change session parameters or voxel masks, that
3/4 repeats retain their fixed weights, and that candidate/comparator rows share
the same eligible images. No generic qualification ladder is added.

The qualified spatial shape gives11,696,828 uncompressed bytes per float32
volume, about12.2–12.5GB per session and up to roughly1.87TB across150 sessions.
Gzip may require decoding the byte stream for every volume even when only
development trials are materialized. Therefore do not promise a fast whole-
corpus read from the modest selected output size. Start with the fixed first
metadata-listed development session, sub-01/ses-01, as an I/O/cost pilot, not
a scientific subset. Proposed envelope:1CPU/4GiB/10minutes with a540second
internal bound, at most16GiB transient scratch and no automatic retry. The
pilot needs only sub-01's nine fixed mask components; other participants are
not read by it. Stop before allocating an output that exceeds the scratch
cap. Record elapsed time, peak memory,
selected/excluded trial counts, actual ROI voxel counts, output bytes and
source-byte handling. No RDM, model fit or score is part of that pilot.

Keep transient selected beta matrices under the episode's nonlegacy scratch
path and compact metadata/reports under `outputs/`. A selected-session array
needs `4 * N_development_session * N_union_voxels` bytes; working memory is one
volume plus the selected-row buffer, with sequential NPY output. Full
handoff storage is `4 * sum_session(N_development_session * N_union_voxels)`
before headers, not yet numerically known because no mask has been opened.
Use this first bounded pilot to size the subsequent all150-session extraction
within the existing1500GB scratch and48-core ceilings. It cannot drop expensive
sessions, relax missingness or choose a better neural subset.

The first-session handoff completed in26.020652s with94,360KiB peak RSS,
575 development trials and12,833 union voxels. Based on that measured cost,
the coordinator selected all150 source handoffs in five participant groups,
reusing this completed session. The selected wrapper, resource limits and
remaining149 fresh output paths are recorded in the execution log. This
scaling does not launch a candidate score, open OOD numeric values, establish
complete diagnostics, or approve the pending raw-image extension.

## Implemented preprocessing continuation

`code/preprocess_development_neural.py` implements the adopted session
standardization, support, validity and repetition averaging above. It requires
all30 completed handoffs for its selected participant before any beta
computation. Six focused fixtures live in `tests/test_neural_preprocessing.py`;
the execution log records independent review, proposed bounds and exact command.
The first complete participant is sub-05. No full-grid completion or score is
inferred from that source readiness.

Each fold exports four float64 ROI pattern arrays, original C-flat voxel
identities, training parameters and exact per-presentation/per-image validity.
The common4,712-ID row order remains intact; ineligible mean-pattern rows are
NaN with explicit counts/reasons. Separate ROI arrays preserve distinct EVC
repeat support even where masks overlap. If training-session counts are
insufficient, parameters and support are marked unassessed/unestimable rather
than called observed nonfinite data. All original source-handoff payloads are
read-only. Reliability/NC, matched training-context RSA and model diagnostics
are not part of this implementation stage.
