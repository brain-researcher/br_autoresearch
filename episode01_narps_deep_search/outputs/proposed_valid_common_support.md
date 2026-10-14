# Approved amendment: one numerically valid common spatial support

**Status: approved and adopted on October 13, 2026.** The scientist explicitly
approved this written common-valid-voxels plan. The operative [GOAL](../GOAL.md),
[DATASETS](../DATASETS.md), and version 9 [search policy](../SEARCH_POLICY.yaml)
adopt the exact rule below. This approval authorizes implementation; it does
not assert that the amended mask, remaining fits, or full-cohort summaries are
already complete. Preserve the original mask, prepared designs and results.

## Decision and reason

The approved decision is one fixed common mask containing the original common-mask voxels whose
BOLD time series are finite at every acquired time point and have a strictly
positive run mean in **every one of the 432 runs**. Keep all 108 participants,
all four runs, all five first-level arms, all model settings, and all six
specification contrasts.

The original [GOAL](../GOAL.md#authorized-exploratory-real-fmri-continuation--2026-10-13)
freeze stated: “No common-mask voxel may be removed: invalid data there stops
that run.” Version 8 of the policy implemented this stop. The scientist's
explicit approval now permits the single support amendment below. The original
frozen-support complete-cohort analysis remains unavailable and is preserved
as such; authorization does not turn missing fits into completed results.

## Observed input problem

The coordinator's compute-node input-only diagnosis on October 13, 2026 found
identically zero BOLD time series inside the original 37,615-voxel common mask:

| Participant/run | Invalid original-common-mask voxels |
|---|---:|
| sub-021, run 2 | 1 |
| sub-037, run 2 | 5 |
| sub-054, run 4 | 1 |
| sub-096, run 1 | 1 |
| Distinct union currently verified | **8** |

These four failures were not caused by NaNs, negative run means, or a detected
coordinate-order mismatch. A zero time series has no defined percent-mean
signal normalization. Its inclusion in an anatomical brain mask does not
provide the missing measurement.

At the proposal decision, eight verified coordinates met the exclusion rule,
and 59 then-unattempted runs still required checking. The final exclusion count
must come from the complete 432-run validity inventory. If there are no
additional invalid coordinates, the
new fixed mask would contain 37,607 voxels. Any additional exclusions must
satisfy exactly the same numerical-validity rule below.

## Exact approved rule

Let `M0` be the preserved original intersection of all 432 source run masks.
For original preprocessed BOLD sample `Y[s,r,t,v]`, define

```text
valid[s,r,v] = all_t isfinite(Y[s,r,t,v])
               and mean_t(Y[s,r,t,v]) > 0

M_valid[v] = M0[v] and all_(s,r) valid[s,r,v]
```

Use all acquired frames for this check, including frames represented by spike
regressors. Apply it to the original preprocessed BOLD before percent-mean
normalization or model fitting. Record the excluded grid indices, their
participant/run causes, and the final count. Construct and freeze `M_valid`
once from the complete 432-run validity inventory before recomputing group
results.

This criterion uses only finite-valuedness and whether percent-mean scaling is
defined. It does not use gain/loss effects, significance, response associations,
map similarity, anatomical-region selection, model preference, or the direction
of a result. No participant or run is excluded. Every group map, participant
spatial metric, run-half comparison, and offer-sequence sensitivity uses this
same fixed spatial support; voxelwise participant counts cannot vary.

## Authorized reuse and recomputation

1. Preserve the original common mask, prepared metadata, design matrices,
   completed fits, failed attempts, and original unavailability record. Save
   the approved support as a separately named artifact. Runner and collector
   changes must identify the selected support explicitly rather than changing
   the meaning of an existing prepared path.
2. Reuse existing valid voxel coefficients with their original designs,
   runwise signal scaling, and S00-derived AR(1) coefficients. These fits are
   voxelwise: changing the spatial reporting support does not alter a retained
   voxel's model or its whitening. Reuse does not require refitting successful
   runs.
3. Finish the previously stopped runs with the unchanged five-arm model and
   original participant-mask ordering, retaining explicit NaNs at unusable
   positions outside the newly approved common mask. Any invalid position
   inside that new fixed mask remains an error; do not make another silent
   support change.
4. Recompute **all** group maps and participant spatial summaries on the one
   final `M_valid` support, including S11, total displacement, the decision and
   response omission effects, their interaction, and R11−S11 for both gain
   and positive loss. Preserve equal two-run half averages, paired participant
   uncertainty, EI/ER strata, and the original offer-sequence sensitivities.
5. Require all 108 participants and 432 run results for a complete-cohort
   summary. Retain the original handling of nonestimable coefficients; a
   narrower valid spatial support cannot turn an unidentified model coefficient
   into an available estimate.

This work stays within the existing branch's resource limits and preserves all
previous charges. It does not resume simulations or open team-map analyses.

## Resulting claim

Any eventual finding concerns exploratory conditional gain/loss estimates and
their specification sensitivity on the explicitly amended, numerically valid
common support. It cannot establish a whole-brain result outside that support,
an untouched preregistered spatial analysis, a psychological mechanism, or
independent replication. Disclose that this spatial-support amendment followed
observed input-validity failures; retain the original frozen-support analysis
as unavailable.

**Decision recorded:** the scientist approved this deterministic common-support
amendment, keeping all 108 participants, all 432 runs, and the unchanged
five-arm analysis. Execution and final observed support counts belong in the
episode's experiment log and result artifacts.
