# Constant-coordinate brain comparator — conditionally adopted October12

The scientist adopted this exact rule **conditionally** on October12 at16:16
Pacific: "yes, adopt the sub-01 fix conditionally"; see the
[decision](experiment_log.md#2026-10-12-1616-pacific--interindividual_converter_training--scientist-adopts-the-constant-coordinate-comparator-conditionally).
The [separate regenerated discriminator-only lock](protocol_lock_brain_constant_discriminator_2026-10-12.json)
inherits the original bindings and admits the bounded check below. Attempt002
now establishes the first-fold guard (one exact constant with exact mean).
The separate [adapted-training lock](brain_constant_panel_authorization.json)
selects all20 fresh fits after affected checks; every later fold retains its
own guard and held-out readout remains closed. This proposal itself is not a fit/readout grant,
replacement result or new numbered episode. The original code, lock,
models, attempts, source-inferred failures and running jobs remain unchanged.
The question stays whether the fixed content-loss converter outperforms the
shared-stimulus brain comparator in the five LAION participants. The proposal
changes how that comparator treats one narrowly defined numerical case; it
does not create a discovery or causal claim.

## Evidence and unresolved cause

The [existing diagnostic](runs/brain_pair_prefix_diagnostic/attempt002/report.json)
observed a sub-01 fold0 required-scale failure before any fit. Its categorical
reason combines nonfinite mean, nonfinite SD and nonpositive SD. It does not
identify an offending coordinate or prove that the case is a constant voxel.
The [source-based disposition](experiment_log.md#2026-10-11-0449-pacific--interindividual_converter_training--shared-sub01-prerequisite-establishes-eight-brain-pair-dispositions)
applies that same necessary prerequisite to all eight original brain pairs
involving sub-01; seven were inferred, not executed.

The [adopted protocol](protocol_decisions_2026-09-30.yaml) makes a required
zero scale unavailable. Tagged V1.0.0 `NCC_brain_loss/ncc_train.py` uses the
full ROI, float32 mean/SD with ddof1 and unmodified division; it supplies no
zero-scale recovery. A different rule must be labeled an adapted comparator.

## Conditionally adopted rule

Retain all 897 tau-train images, five common image folds, the ten fixed alphas,
source NumPy solver, full native axes, penalized bias, float32 arithmetic,
validation-image weighting, exact alpha tie rule and final refit. For each
source and target coordinate, independently within each fold's fitting rows:

1. Compute the original float32 mean and ddof1 SD, in the original order.
2. Keep every finite positive SD unchanged, including tiny positive values.
3. Use an **effective normalization scale of one** only if all three hold:
   the computed SD is exactly zero; all finite fitting values exactly equal
   the first stored float32 value; and the computed float32 mean exactly
   equals that value. This ensures the original centering is exactly zero.
4. Nonfinite means/SDs/values, negative SDs, zero-SD varying columns and
   constants whose rounded mean fails the equality remain not evaluable.
5. Recompute this rule on all and only the 897 training images for the final
   refit. Never pool a validity mask across folds or use validation/test rows
   to determine a fitting transform.

There is no epsilon, near-zero threshold, precision change, mean substitution,
coordinate deletion or response imputation. The stricter equality guard may
leave mathematically constant columns unavailable because of rounding; it is
not a universal constant-column repair and does not promise to rescue sub-01.
Apply the same adopted rule uniformly to all original pairs, not selectively
to favorable participants or outcomes.

The unit-scale convention has a familiar precedent: scikit-learn documents a
scale of one for zero-variance features. That is supporting context, not the
proposed implementation: its documented ddof0 and NaN handling differ from
this retained ddof1/fatal-nonfinite protocol. No library replacement is
proposed. [Official StandardScaler documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
(accessed October 11, 2026).

## What this changes in the comparison

Preserve the full target-coordinate denominator in each image's validation
MSE. For admitted constant targets, residuals use explicitly declared original
response units because effective scale is one; they cannot be described as
unit-variance standardized residuals. Exactly centered constant targets have
no training variation to fit. Their validation residuals add an
alpha-independent term in exact arithmetic. This is an explanation of the
rule, not a guarantee of bitwise finite-precision equivalence.

The alternative of dropping constant targets and averaging MSE over only
remaining coordinates changes support and relative fold weighting when their
counts differ. It also needs explicit axis expansion before decoder use.
Retaining the full layout is the smaller proposed adaptation. Neither option
would turn a modified success into a success under the original protocol.

The amended artifact must identify its distinct normalization policy and
separate raw observed SDs from effective scales, preserving replacement
masks/counts. The existing original validator must continue rejecting such an
artifact as an original-protocol model. Existing original artifacts are not
automatically accepted or relabeled as amended results either. Any future reuse
requires an explicit prospective equivalence and provenance decision; otherwise
use separate amended fitting/publication. This proposal orders no refits.
Predictions are restored to physical target coordinates before the fixed target decoder's separate normalization.
No tau-derived replacement mask changes decoder eligibility. The original224
tau-test image support, feature metrics, undefined-correlation handling, all20
pair slots, biological units and missing-pair rule remain fixed. There is no
new comparability margin or participant-population claim.

## Selected next empirical action — bounded discriminator only

A single bounded first-fold discriminator can determine whether this rule is
relevant before any amended fit. Reuse the admitted original sub-03/sub-01
binder and full native axes, but report only the failed sub-01 fitting slice:
counts of nonfinite means, nonfinite SDs, negative SDs, finite zero SDs, and
among finite zero-SD columns, exact constants with/without exact mean equality
versus varying columns. Include the total invalid-column union. No coordinate
IDs, neural values, extrema, validation variation, losses, scores or models
need to be exposed. This classifies the known failure; it does not establish
all-fold or full-panel eligibility.

Selected limit: one numerical thread, at most300seconds,6.5GiB processRSS
within1CPU/7GiB, and1MiB durable categorical output. Bundle it within a
separately selected substantive compute parent with at least330seconds
remaining and the same parent deadline/accounting. No checking-only job,
automatic retry or fit follows. The prior diagnostic attempt stays preserved.
The separate discriminator-only lock provides this narrow authority; the
coordinator must select the substantive parent and its verified end time.

If the case meets the adopted guard, implement new isolated
normalization/artifact/entrypoint code,
with focused constant/nonconstant-underflow/nonfinite/fold-boundary checks
inside the next substantive allocation. Reuse unaffected scientific evidence;
do not add a simulation sweep. Select finite fitting resources only then.

## Scientific decision and exposure boundary

The scientist selected the explicitly labeled comparator adaptation only if
the discriminator establishes its relevance; otherwise retain the original
not-evaluable finding. The operative
[lock](protocol_lock.yaml) requires an explicit scientist amendment and a
regenerated lock for a pre-score scientific change; a consequential scientific
change after any neural-score/outcome exposure requires a new episode instead. This proposal
is now recorded by the explicit conditional decision and the separate scoped
lock. These records do not modify running fits or authorize amended fitting.

Predefined internal tau-train CV is model fitting, not held-out evaluation.
Its existence does not establish that CV magnitudes were inspected to choose
this proposal. Current permitted evidence here consists of categorical
prerequisite failures and source analysis. Do not inspect model/CV magnitudes
to choose the change or declare an unstated exemption from the exposure rule.
The cancelled first readout processed tau-test responses, truth and predictions;
no final collection or completion report was published, and no final-score
exposure was reported. Source/file-identity arguments support the recorded
pre-score disposition under their stated assumptions; they are not a claim
that test data were unread or that file length alone proves every arithmetic
stage. Contrary evidence of earlier score/outcome exposure must invoke the
original new-episode requirement, not an exemption created by this amendment. The
B/C not-evaluable locks, OOD closure and separate scoring authority remain.

Two independent source reviews support the full-axis recommendation and
interpretation, with root review adding the exact-mean guard. Applicable BR
review criteria already retrieved for EP21 are reused: training-only
transforms, complete support, explicit method changes and claim limits. BR
has not read these local artifacts or issued a tailored acceptance or launch.
