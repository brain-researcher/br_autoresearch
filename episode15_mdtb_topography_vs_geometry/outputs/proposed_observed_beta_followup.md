# Observed-beta development follow-up — adopted; implementation pending

The scientist adopted this specification at **2026-10-12 22:29 UTC (15:29
Pacific)**. Its operative authority is
`SEARCH_POLICY.yaml: scientist_amendment_20261012_observed_beta`.
Implementation is pending; this record is not a data-access release, job
selection or completed result. A qualified complete-panel fit is authorized
once the coordinator records release and remaining source/resource admission.
EP15's original question—what explains stable individual differences in
cerebellar task maps—and original reliable-variation route remain intact.
No new numbered episode or canonical campaign binding is created. This
conditional development analysis does not validate the original estimands
or complete their formal qualification.

## Evidence and the adopted exception

The completed [Task-A diagnostic](runs/task_a_error_support/attempt002/diagnostic.json)
uses all twelve development participants, sixteen runs, twenty-nine Task-A
conditions and the fixed 208 locations. S is positive in 25 conditions and
remains positive under every single-person deletion in 22. N has both signs.
The deletion ranges are descriptive, not confidence intervals. This reveals
structured quarter differences; physiological session changes, shared nuisance
and sampling variation remain inseparable. It neither proves correlated
measurement errors nor establishes independent errors. Task A cannot
authenticate Task B's error structure, and a common effect identical across
sessions can be invisible to these contrasts. No Task-B or audit values were
read by this diagnostic.

[The earlier GOAL amendment](../GOAL.md#adopted-procedural-evaluation-separation--october-12-2026)
and `SEARCH_POLICY.yaml`'s `scientist_amendment_20261012.empirical_fitting_still_requires`
retain functional-validity/error-independence support as a fitting prerequisite.
The October 6 exploratory branch also retains it in
`scientist_amendment_20261006.exploratory_real_data.before_scoring`.
[DATASETS.md](../DATASETS.md#map-construction-to-freeze-before-scoring) states
that favorable Task-B scores cannot waive shared nuisance invalidating the
cross-products. The Task-A result does not clear those conditions. The later
[adopted amendment](../GOAL.md#adopted-observed-beta-descriptive-route--october-12-2026)
is the narrow exception below; diagnostic completion alone did not authorize
it. Adoption authorizes the qualified complete fit subject to the remaining
release and resource requirements, rather than launching it automatically.

The adopted change is explicit: permit the fixed development panel to describe
prediction of the supplied beta maps while cross-partition error support is
unresolved. That waives the support prerequisite **only for this conditional
descriptive route**, not for the original estimands, audit or class decisions.
All other access, source, numerical, control and resource requirements remain.

## Adopted scope

- Participants, unchanged and equally weighted: `sub-08`, `sub-12`, `sub-14`,
  `sub-15`, `sub-17`, `sub-18`, `sub-20`, `sub-21`, `sub-24`, `sub-25`,
  `sub-27`, `sub-31`. Only their explicitly released development containers
  may be read. The prepared
  [point-panel manifest](runs/development_release/attempt001/view/point_panel_manifest.json)
  is an input binding, not an existing Task-B read grant.
- All 18 Task-B-only conditions remain, on every target: `CPRO`, `Prediction`,
  `PredictViol`, `PredictScram`, `SpatialMapEasy`, `SpatialMapMed`,
  `SpatialMedDiff`, `NatureMovie`, `RomanceMovie`, `LandscapeMovie`,
  `MentalRotEasy`, `MentalRotMed`, `MentalRotDiff`, `BodyMotionIntact`,
  `BodyMotionScram`, `RespAltEasy`, `RespAltMed`, `RespAltDiff`.
  Each of seven domains retains weight 1/7; conditions share their domain's
  weight equally. The fourteen shared Task-B conditions stay outside this route.
- Preserve all 208 frozen common-support locations, equal voxel weighting,
  spatial centering, arithmetic run/regressor averaging, SUIT pull direction,
  interpolation, coordinates and voxel order. No new smoothing, whitening,
  nuisance regression, condition selection or Task-B-derived scaling.
- Preserve the sixteen runs, map halves 1–8 and 9–16, geometry partitions
  1–4, 5–8, 9–12 and 13–16, and primary geometry pairings 1×2 and 3×4.
  No replacement pairing or selected sensitivity result may rescue a result.
- Preserve ordinary leave-one-participant-out fitting: every shared source,
  basis, prior, profile, scale and normalization uses the other eleven people.
  Target personalization uses anatomy and all 29 Task-A conditions only,
  with the existing common positive gain rule. Target Task B is supplied only
  to evaluation. Keep the identity control's paired reduced source excluding
  both recipient and donor, rather than substituting the ordinary source.

Run all sixteen existing variants in their current canonical order:

| Family | Four existing variants |
| --- | --- |
| M1 | reference probabilistic; hard assignment; coarse granularity; fine granularity |
| M2 | reference diffeomorphism; tight regularity; loose regularity; displacement stress |
| M3 | reference local isometry; tight locality; broad locality; active-subspace stress |
| M4 | anisotropic stretch; localized low-rank stretch; localized low-rank stretch and shear; spectral-bound stress |

The exact identifiers and model implementations remain those in
[development_lopo.py](tools/development_lopo.py) and
[development_point_panel.py](tools/development_point_panel.py). This amendment
adds no candidate, hyperparameter sweep, model replacement or outcome-adaptive
search. Report the complete panel, including unfavorable values, without
declaring a winning class or selecting a positive subset.

Retain M0a and M0b, the existing Task-A condition permutation, participant
identity derangement, eight random-local-basis draws and eight matched-warp
draws for each of four M2 variants. Keep seeds, donor assignments, shared-source
exclusions and all numerical/locality/identifiability certificates unchanged.
Compute errors separately for every fixed draw before averaging them; do not
replace controls by an averaged-prediction ensemble. Retain the existing M2
alignment summary as descriptive. Every implemented control in the complete
panel runs; no control is dropped because its result is inconvenient.
Unfinished native-signature and formal uncertainty obligations remain
unfinished, not silently satisfied by this follow-up.

## Quantities and permitted interpretation

Add one unadjusted observed-map loss, fixed before opening development Task B.
Let `X[i,q,c]` be the existing 208-location beta map for participant `i`,
four-run partition `q` and condition `c`; let `Yhat[i,m,c]` be that target's
unchanged prediction from model or control `m`. `C` removes a map's spatial
mean over all 208 locations, and `w[c]` is the existing seven-domain weight.

```text
L[i,m] = (1/4) sum_q sum_c w[c] *
         mean_voxel((C X[i,q,c] - C Yhat[i,m,c])^2)
L[m]   = (1/12) sum_i L[i,m]
```

This is squared prediction error for observed, spatially centered beta maps,
including their measurement error. It performs no noise subtraction,
reliability normalization or participant demeaning beyond the stated spatial
centering. Retain per-participant, per-condition and per-partition contributions
so every condition and sign of a comparison can be reported. Conditions,
partitions, voxels, model variants and control draws do not increase biological
`n` above twelve. Overlapping LOPO training sets also prevent treating the
twelve fitted folds as independent experiments.

Report all model/control losses and signed descriptive differences to the
existing matched references: M0a/M0b for the ordinary panel; true versus
permuted Task A within the ordinary source; true versus donor Task A within
the paired reduced source; and their existing matched random controls. No
best-draw or best-variant selection, significance test, threshold decision,
confidence interval or nominal superiority claim follows from these values.
The existing participant-deletion fitting expansion is not added.

The existing signed `V_map`, `V_geometry`, `R_map`, `R_geometry`, source scales
and comparator/falsifier summaries may accompany the observed losses, with
their existing arithmetic and signs preserved. Present them explicitly as
**cross-partition statistics conditional on unresolved error assumptions**.
Their legacy field names do not establish reliable signal or unbiased residual
variation. Do not subtract the Task-A S or N statistic as a Task-B noise
correction. Keep the M3 projected-geometry control's algebraic zero identified
as an identity, not empirical support.

The strongest permitted conclusion is descriptive: under the fixed source and
personalization rules, a stated predictor had a stated error on these twelve
participants' observed Task-B beta maps. Prediction could reflect shared
measurement structure as well as biology. It does not identify parcel changes,
relocation, isometry or geometry change as the generating mechanism.

Prohibit reliable-signal claims, recovered/recoverable variance fractions,
`1-R/V` or other recovery ratios, adequacy, equivalence, statistical superiority,
model-class support and mechanistic or causal conclusions. The adopted 0.20
remaining-variation and 0.05 increment standards remain historical prospective
standards of the original route; they are not pass rules or recovery benchmarks
for observed loss. Do not apply the five unset margins. No audit calibration,
audit Task B, private reliability values, published fitted cache or positive-
condition selection is authorized. Same-account separation remains procedural,
not technically enforced or blinded evaluation.

## Required implementation and resource admission

Reuse the repaired [released reader](tools/development_handoff_reader.py),
[complete panel](tools/development_point_panel.py) and
[entrypoint](tools/run_development_point_panel.py). They already retain all
twelve targets and sixteen variants, the genuine-A/control branches and the
authenticated live forward header. The panel contains 24 source fits and 48
complete calibration panels; it excludes the 132 participant-deletion refits.

Only a small reporting extension is required for this adopted route:

1. Add the stated observed-map loss contraction over the complete existing
   ordinary predictions and the existing source-matched control panels. These
   predictions and observed partitions already exist in the returned fit
   records; no new model fitting or data reader is needed.
2. Carry the complete losses and contributions into the final output, label
   the legacy signed summaries as assumption-conditional, and record the
   expressly adopted analysis scope. Preserve exclusive new-attempt ownership,
   resource guards, categorical failures and publication only after the full
   panel and all selected controls finish. No partial cohort/model summary.
3. Check the new loss arithmetic and paired-source/control joins with focused
   source fixtures in the coordinator's compute allocation. Reuse unaffected
   passing reader/model/panel checks; no new recovery simulation program.

No real complete-panel time or peak-memory measurement is available in the
reviewed record. Task-A diagnostic attempt002 took 94.708 seconds and 165432
KiB peak RSS; that establishes a Task-A mapping cost, **not** a fitting-cost
envelope. The missing measurement is the complete twelve-target route with
all 24 source fits, 48 calibration panels, sixteen variants, fixed controls,
source-map construction, predictions and complete summary persistence. The fixed
scratch limit also needs an explicit coordinator selection; existing mapping
uses streaming rather than extracted NIfTI files.

Accordingly no numerical job request or launch is selected here. Before an
empirical invocation, the coordinator must specify how the required full-panel
cost is measured under a bounded first execution of this same complete route,
with scientific outputs withheld during cost admission. Do not invent a smaller
cohort/model pilot, infer full-panel cost from fixture timings, or promise that
a selected cap guarantees completion. Any change to the existing cost-admission
requirement must be stated explicitly rather than silently waived.

Retain the existing limits: 1200 CPU-core hours, 120 wall-clock hours,
at most 32 CPU cores and 128 GiB per trial, no GPU, and an explicitly fixed
scratch allowance. All historical charges and the applicable clock remain;
this amendment does not reset them. The 120-hour limit belongs to the search;
source-preparation dates do not establish its start. The coordinator must
record the applicable search start and establish the remaining
allowance and concrete request before a job. A budget or solver failure is an
incomplete execution, not a reduced panel, scientific failure or automatic retry.

## Decision record

The scientist selected option 1 below at the adoption time above. Option 2 is
retained as the considered alternative, not a blocker on the adopted route.

1. **Adopt the conditional observed-map route — selected.** Authorize the narrow
   independence-support exception and reporting limits above, then complete
   the reporting extension and remaining source/firewall/resource admission.
   Record the amendment in the operative documents before any Task-B access.
   This amendment alone does not select a job or open an audit.
2. **Wait for authentic estimation-covariance/support information — not selected.** Preserve
   the completed Task-A finding and the original prerequisites. Useful evidence
   would tie the actual derivative to run/session mapping and the estimation
   scope of temporal covariance, nuisance terms and beta scaling, or otherwise
   substantiate the cross-partition assumptions. A document specifically named
   GLM7 and a mathematical proof of zero covariance are not universal demands.
   Generic provider code, small contrasts or a favorable future model score
   cannot substitute for the missing support. No further simulation expansion
   or Task-B comparison is selected while waiting.

Neither option retroactively makes the observed Task-A diagnostic a prospective
validation pass. The original EP15 question and formal route remain available,
with their unresolved assumptions and obligations recorded honestly.
