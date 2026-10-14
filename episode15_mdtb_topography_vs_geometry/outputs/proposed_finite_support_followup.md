# Proposed finite common development support — ADOPTED October 12, 2026

**Status:** at about 16:18 Pacific on October 12 the scientist replied "adopt
the EP15 finite-support proposal". The machine-readable authority is
`scientist_amendment_20261012_finite_support` in `SEARCH_POLICY.yaml`. The text
below is kept unchanged as the adopted specification; its statements that it
adopts nothing describe the draft as prepared. The one9024-map intersection
has now frozen188 original locations. The subsequent complete panel stopped
under the unchanged M4 numerical-certificate requirement; see the
[execution log](experiment_log.md). No further retry is selected.

Prepared October 12, 2026. This is a concrete review proposal, not an adopted
amendment, mask-construction grant, data release or permission to retry fitting.
The current 208-location requirement remains active. The original anatomical
support, failed `cost001`, and both mapping diagnostics remain unchanged.
Adoption would explicitly amend
`SEARCH_POLICY.yaml: scientist_amendment_20261012_observed_beta.fixed_panel.common_support_locations`
and its equal-208-voxel loss rule for this route, plus the
[original support specification](anatomical_support_specification.md)'s
prohibition on later Task-A/B support revision. The preceding observed-beta
adoption did not authorize this additional change.

## Observed problem

The coordinator reports that `cost001_mapping_diagnostic002` completed all
9,024 selected development beta maps. All Task-A maps were valid. There were
1,728 invalid Task-B maps and 6,336 map-location invalidity incidents, involving
six participants. Each of their 288 Task-B maps had the following invalid
location count: sub-12: 3; sub-14: 2; sub-15: 1; sub-17: 1; sub-18: 9; sub-21: 6.
The cause was nonfinite raw beta values at positive-weight interpolation
corners. There were no outside-grid, nonfinite-coordinate or finite-corner
accumulation-overflow cases.

These are quality counts, not prediction results. The census did not retain
location masks. Equal counts across maps do **not** establish identical missing
locations; neither the eventual retained count nor its geometry is known.
No model fitting or candidate score has completed. This proposal uses the
coordinator's aggregate census, not an additional payload read.

## Exact proposed support rule

If explicitly adopted, make one new development support `M`, a subset of the
original anatomical support `M0`. For each original location, require a valid,
finite result from the **unchanged strict sampler in every one of the 9,024
already released selected maps**, across all twelve development participants,
all sixteen runs, all 29 Task-A conditions and all 18 Task-B-only conditions.
Use the existing complete metadata plan and original within-run regressor
roster; do not evaluate a condition mean in place of its constituent maps.

```text
M[v] = M0[v] AND all_selected_maps(
           strict_positive_weight_stencil_is_valid[v]
           AND interpolated_value_is_finite[v])
```

The same authenticated headers, millimetre affines, forward transform,
intensity scaling, coordinate ordering and positive-weight corner rule apply.
Do not include instructions, the fourteen shared Task-B conditions, audit
people, audit values or private reliability. Do not use beta magnitude, sign,
variance, effect size, condition identity preference, model residual or score
to choose locations. No interpolation repair, zero filling, weight
renormalization, smoothing, nearest-neighbour replacement or participant,
condition, run or regressor removal is allowed.

Construct the full intersection once, completing every selected member even
if it becomes empty. Write a separate derived mask with original integer
location IDs and their existing coordinate order; retain `M0` byte-for-byte.
Bind all later reader/model/control/reporting views to that one frozen derived
axis. The map is common to every fold, model, condition, partition and control
draw. No target-specific, condition-specific or model-specific support and no
second mask selected after fitting. Empty support, insufficient geometry or
failed numerical qualification stops this proposed route; it does not permit
a smaller cohort, altered capacities, weaker cutoff or another support search.

## Scope and interpretation

Keep all twelve named development participants, eighteen Task-B-only
conditions, sixteen fixed model variants, all controls/draws/seeds, original
run partitions, seven-domain weights and source exclusions. Preserve ordinary
LOPO sources trained on the other eleven people and the identity-control
sources excluding both recipient and donor. All fitted source maps, scales,
profiles and target personalization continue to use their existing allowed
training/Task-A inputs on `M`.

Spatial means and equal-voxel losses are now taken over `M`, with weight
`1 / |M|`. This changes the spatial estimand. Retain the adopted observed-beta
descriptive route and its complete per-person/condition/partition/control
contributions; retain any legacy signed summaries only with their existing
unresolved-error-assumption labels. Do not compare the new spatial loss to a
purported completed 208-location loss, because that panel did not complete.

The mask uses **target Task-B availability**, even though it uses no finite
response magnitudes or scores. It is therefore a data-dependent common
availability mask, selected after seeing a quality failure, not the original
anatomy-only prospective support. LOPO fitting remains conditional on this
shared availability information; target Task B cannot be described as wholly
unseen. Missingness may reflect acquisition or preprocessing and may correlate
with anatomy or responses. The retained region need not represent the original
208 points, the whole cerebellum, future participants or unavailable locations.

No reliability/recovery fraction, adequacy/equivalence, formal superiority,
winning class, mechanism or causal claim becomes available. No audit release,
audit-driven mask repair or conversion of this exploratory result into
independent confirmation is authorized. This does not complete EP15's original
reliable-variation question or validate its error assumptions.

## Can the existing full panel operate on this support?

**In principle, with the same model equations, declared ranks and parameter
counts; not with the current executable unchanged.** Shape guards, array allocations and several
geometry checks encode exactly 208. Moreover, dropping rows of an existing
basis/operator would generally destroy its zero-mean and orthonormality
certificates. No model fit or certificate can be reused by simple slicing.

The proposed deterministic adaptation is to rerun the existing coordinate/ID
median-split and graph-basis construction on `M`, preserving its split depths,
mode counts, axis/ID ties, tolerances and all model bounds. Rebuild the common
cardinal representation from those coordinates using the same bandwidth
formula. Recompute the source and every control under the same exclusion
rules. Geometry and parameter values will change because their domain changes;
the model families, objective formulas, declared ranks and parameter counts
will not be replaced. Effective statistical complexity can change with the
spatial domain and rebuilt bandwidths; it is not claimed to remain identical.

Two current assertions require explicit adaptation beyond shape substitution:
M1 requires exactly equal 4/8/16-cell sizes, and M2 requires eight cells of
26 points. On a derived count not divisible by sixteen, preserve the existing
recursive median split's deterministic floor/ceiling cell sizes, still cover
every retained point exactly once, and retain arithmetic within-cell profiles,
equal parcel priors and eight field components. Recompute each cell's centre,
radius and certified bounds by the existing formulas. Do not discard extra
finite points to force divisibility. Adoption must include this geometric
consequence rather than calling it an unnoticed implementation fix.

Existing admission conditions remain substantive:

- Each anatomy cell needs more locations than retained nonconstant modes.
  The M3 recipes retain `(depth, modes) = (3,2), (4,1), (2,4), (3,3)`.
  Thus their respective leaves require at least 3, 2, 5 and 4 locations.
  The complete panel therefore has a necessary cardinality floor of 32
  implied by the existing capacities; this is **not** a new chosen minimum
  support threshold or a sufficient admission rule.
- The random-basis control needs at least four locations in every reference
  cell and a nonempty, eligible complementary local eigenspace after its
  existing orthogonality constraints. A count alone cannot establish this.
- Preserve full graph rank and all centering/orthonormality/locality checks.
  Reference Task-A active design rank is 16; variant ranks remain 16/16/16/24,
  with the fixed `1e-8` relative rank rule and global-scale local cross-product
  cutoff. The 29-condition axis remains intact. Neither sufficient point
  count nor finite maps guarantees these empirical ranks.
- Preserve positive finite source scales, common gain, parcel-profile
  separation, cardinal-system rank/amplification, M2 radius/Jacobian/inverse
  and optimizer certificates, M4 tangent-rank/curvature/solution certificates,
  and every actual random-control certificate. Failure keeps the full panel
  incomplete; it cannot be repaired by lowering dimension or dropping a
  control after observing the failure.

These requirements come from the existing implementation, particularly
[local_anatomical_basis.py](tools/local_anatomical_basis.py),
[local_functional_source.py](tools/local_functional_source.py),
[local_isometry_variants.py](tools/local_isometry_variants.py),
[matched_random_basis.py](tools/matched_random_basis.py),
[local_parcel_prediction.py](tools/local_parcel_prediction.py),
[smooth_relocation.py](tools/smooth_relocation.py),
[common_cardinal_source.py](tools/common_cardinal_source.py) and
[paired_stretch_variants.py](tools/paired_stretch_variants.py).

## Smallest implementation and next action after adoption

1. Build the single complete derived availability mask through the already
   qualified released reader and exact sampler. Unlike the count-only census,
   retain its 208-element Boolean intersection. Publish it only after full
   coverage, alongside its exact original-axis binding and retained count;
   do not reconstruct membership from the census's counts.
2. Parameterize the explicit support size/identity through the reader,
   point-panel joins, LOPO/source/model/calibration/control wrappers and
   observed-loss/alignment summaries. Retain a distinct original 208-point
   route. Replace the specific equal-cell assertions as described above;
   preserve all numerical certificates. Core spatial operator algebra and
   endpoint contractions already support variable voxel dimensions.
3. Check changed axis joins, derived-mask identity, deterministic uneven-cell
   partitioning and unchanged operator/control algebra in the coordinator's
   allocation. Reuse unaffected tests; no new simulation/recovery campaign.
   Admit the same complete twelve-target/sixteen-variant cost route, with
   scientific outputs withheld until complete resource admission. A changed
   support does not justify a reduced panel or automatic fit retry.

No new numerical allocation is selected here. Mask construction must have a
recorded bounded request; the complete census's measured cost can inform that
request, but does not measure fitting cost. Keep the existing 1,200 CPU-hour,
120 search-wall-hour, per-trial CPU/memory and scratch limits, historical
charges and applicable search clock; there is no reset.

The consequential decision is whether to adopt this explicitly
availability-conditioned descriptive route, including its deterministic
geometry adaptation, or retain the 208-location route while seeking authentic
finite inputs/corrections. This draft adopts neither. No mask is constructed
and no operative GOAL, DATASETS or SEARCH_POLICY requirement is changed here.
