# Prospective concept roles, folds and uncertainty blocks

**Coordinator adopted for implementation, focused checks and the bounded
metadata-only preparation/timing pilot at22:10UTC on2026-10-04.** Root and
independent scientific review passed. Full role assignment remains unselected
until measured resource feasibility supports the unchanged finite program. These numerical conventions replace the still-prospective
balance and open-ended best-swap recommendation in
`source_roles_proposal.md`; they do not change adopted image eligibility,
chronology, taxonomy identities or the operative episode contracts.

## Fixed inputs and boundary

Use the completed `source_qualification/attempt002` and
`source_chronology/attempt001` metadata artifacts, with their fixed
attempt003/retained001 source identities. Their completed reports establish
720 eligible concepts, 3,634 eligible images, all 49,980 retained presentation
coordinates, and qualified acquisition order and eligible-image repeat halves
for all four participants. The original common-image counts are five for 480
concepts and six for 240 concepts. The qualified-count histogram is
3:1, 4:95, 5:493, 6:131. These are input descriptions, not assignment quotas
or opportunities to change eligibility.

Require completed prerequisite reports and an exact ID join of the concept,
image, taxonomy and chronology projections. Preserve source concept IDs,
names, image IDs, every retained row coordinate, schedule exception and repeat
mapping. Use only eligible-image rows to calculate concept balance features;
an ineligible image remains documented but does not contribute to the primary
response or its balance features. Every image, including an ineligible one,
inherits its concept's eventual role. No exclusion is added at this stage.

No neural, behavioral-response, pixel, model-feature, candidate-score, private
timestamp or new upstream source access is needed. The metadata builder still
has no authority to release a mixed neural source or an audit feature array.
The role map is metadata; role-filtered neural handoffs and their access checks
are later stages. EP17/EP18 overlap disclosure remains required before either
sealed outcome is opened.

`DATASETS.md` requires: “Freeze the taxonomy release, concept crosswalk,
unmapped and multilabel handling, group-merging rule, minimum cell size,
balance objective, tolerances, and tie rule before assignment.” This proposal
supplies those conventions. The contracts require exact whole-concept sizes,
not a globally optimal split or a new scientist approval for each compatible
numerical implementation choice. A different universe, participant roster,
role size or outcome-informed assignment would require an explicit amendment.

## Taxonomy and units

Reuse the verified 53 binary membership columns in the pinned stimulus
commit `f8f5162f79fb2e9fb0d8cbb8bac610db11a54180`, already projected to
`source_qualification/attempt002/taxonomy_crosswalk.csv`. The source bridge
preserves `things_category_nr`, requires category53 row CID-minus-one to equal
the exact source concept/folder name, and joins `things_concepts` by exact
uniqueID. Its eight irrelevant table-order permutations do not change any of
the 720 selected anchors. There is no refetch, name normalization, dominant
label, normative-rating input or new crosswalk.

Keep all 53 labels and add the indicator `taxonomy_unmapped`, true precisely
when all 53 are zero. All 128 unmapped concepts stay eligible. Multilabel
concepts contribute one to every applicable membership. **No group merging;
minimum represented cell size one.** There is no hard minimum per role, fold
or block: a rare category cannot be required to occur in every group. This
does not qualify any optional semantic-stratum analysis.

Each concept is one assignment unit. For participant-specific features, first
give each eligible image equal weight and then each of that image's retained
presentations equal weight. Specifically, for concept c, participant s, image
i and its row t, use weight `a_csit = 1 / (m_c * n_csi)`, where m is the
qualified image count and n its retained repeat count. The weights sum to one
for each concept and participant. The four participants always receive equal
objective weight; extra rows in one participant do not increase their weight.

## Frozen feature definitions

Feature definitions and field order are fixed below. Retain constant fields
with zero standardized value; do not change a family's denominator when a
field is constant. Raw features are finite metadata summaries, not model
features. Taxonomy names retain the authenticated table's column order;
participants retain `sub-01, sub-02, sub-03, sub-06` order.

1. **Taxonomy:** the 53 indicators plus `taxonomy_unmapped` (54 fields).
2. **Exemplars:** indicator of original six-exemplar status, and qualified
   image count (two fields). Original status must be exactly five or six.
3. **Acquisition, separately for each participant:** one weighted presentation
   fraction for every session in that participant's fixed retained roster,
   followed by weighted mean and population variance of normalized retained
   acquisition rank. Session labels are categorical identities; sorting them
   for column order makes no claim of chronological order. Set
   `u = retained_acquisition_rank / (N_s - 1)`, using all N_s retained rows in
   that participant for the denominator. Then mean is `sum(a*u)` and variance
   is `sum(a*(u-mean)^2)`. Do not renormalize time separately within a concept,
   role or subset, and do not substitute GLMsingle row order or run numbers.
4. **Repetition lag, separately for each participant:** four mutually exclusive
   weighted fractions, then conditional log-lag mean and population variance
   for each of two known delay types (eight fields total). Classify each
   eligible-image presentation as follows, using the source cumulative
   repetition ordinal, not its retained rank:

   | Class | Exact condition |
   | --- | --- |
   | Expected first exposure | `source_repetition == 1` and both source delay fields are declared unknown/empty |
   | Known within-session delay | ordinal greater than one, finite `delay_seconds > 0`, and finite `delay_days == 0` |
   | Known between-session delay | ordinal greater than one, finite `delay_days > 0`, and finite `delay_seconds == 0` |
   | Unresolved source delay | every remaining nonnegative/unknown combination |

   The two zero sentinels are not physical zero-duration observations. In
   particular, both-zero, both-positive, one-missing, nonfirst-both-missing and
   an apparent delay on a source first exposure remain unresolved, with their
   reasons counted. Negative or nonfinite numeric values are a changed or
   malformed input, not imputation candidates. A retained first row with
   source ordinal greater than one may have a valid delay to an omitted
   predecessor; retain that meaning. Neither a source-ordinal gap nor the
   reconstructed trial-index gap is elapsed time.

   For the within-session type use `v = ln(1 + delay_seconds / 1 second)`;
   for the between-session type use `v = ln(1 + delay_days / 1 day)`. Never
   convert or pool the two units. For a known type with total weighted coverage
   p greater than zero, mean is `sum(a*v)/p` and population variance is
   `sum(a*(v-mean)^2)/p`, summing only rows of that type. If p is zero, that
   concept/participant/type's moments are unobserved, not zero delays.

For each assignment's parent concept set, impute an unobserved lag moment by
the equal-concept mean of that moment among parent concepts where the type is
observed. Mean and variance are imputed separately. If no parent concept has
that type, set both numerical columns to constant zero and explicitly mark
them wholly unobserved. Preserve observed/imputed masks and the coverage
fractions. Imputation is neutral input to balancing; it never becomes an
observed delay in a report or source table. No other feature needs imputation.

Schedule exceptions remain attached to every row. They are not exclusions or
additional post hoc balance variables. Report exception counts by resulting
group as descriptive metadata, without optimizing or rejecting on them.

## Scaling, precision and objective

Use the fixed participant order `sub-01, sub-02, sub-03, sub-06`; within each
participant order concepts by numeric `things_category_nr`, then exact
uniqueID UTF-8 bytes; within each concept order exact provider image paths by
UTF-8 bytes, then presentations by numeric
`glmsingle_trial_index_zero_based`. Parent-concept summaries and imputation
means use that same numeric-CID/uniqueID concept order. Session columns use
numeric session-label order. This pins every summation order; input CSV row
order and dictionary iteration cannot change it.

Keep rational-only session fractions, rank means/variances and lag-class
fractions as exact Fractions through their complete moment calculation, then
convert each resulting field once to Decimal. Compute logarithms, nonconstant
log-lag moments and subsequent nonconstant scaling using Python Decimal with
precision 80 and `ROUND_HALF_EVEN`. Parse the existing projected decimal strings directly;
do not round-trip them through a new binary float. This cannot restore
precision discarded by the already-frozen event projection. Integer counts
and rational weights remain exact until their explicitly defined Decimal
operations. Use Decimal `ln` and `sqrt`; reject invalid/nonfinite results.

Before averaging known log-lag values for a concept/participant/type, check
whether all its valid transformed values are identical. If so, return that
common value as the conditional mean and **exact zero** as the conditional
variance, regardless of the number or weights of repetitions. Likewise, if
all observed parent values used for an imputation mean are identical, return
their common value directly. These are exact equality branches, not a
variance floor, tolerance or rounding-based declaration of constancy.

For each assignment separately, let P be its parent concept set and N its
size. After the stated imputation, first test whether all parent values of a
field are exactly identical, including imputed constants. If so, set its
population mean to that common value, variance to exact zero, and all z and q
values to zero without a rounded sum. Otherwise compute its population mean
and variance across P with equal concept weight. Compute nonconstant variance
as a sum of weighted squared centered deviations, not a subtraction of raw
second moments. An unexpected zero or nonfinite variance for a nonconstant
column stops numerical preparation rather than adding a variance floor.
Otherwise, `z_cj = (x_cj - mean_j) / sqrt(variance_j)`. Set
`q_cj = round_half_even(10000 * z_cj)` as an integer. There is no clipping,
winsorization, ranking, learned binning or precision chosen from balance
results. The 0.0001 standardized-unit quantum defines the objective: exact
optimization of q is not a claim to optimize the unquantized features exactly.
Report residuals on both scales without choosing between assignments by the
unquantized residuals.

The four feature families each receive total weight 1/4:

| Field | Objective weight w_j |
| --- | ---: |
| Each of 54 taxonomy fields | 1/216 |
| Each of two exemplar fields | 1/8 |
| Each participant's session field, with S_s session levels | 1/(32*S_s) |
| Each participant's acquisition mean or variance | 1/64 |
| Each participant's eight lag fields | 1/128 |

Thus acquisition and lag first average participants equally. Within
acquisition, session composition and the two rank moments each receive half
the participant's weight. Within lag, the four class fractions receive half,
and the within/between moment pairs receive one quarter each. Constants keep
their listed weights but contribute zero. The complete field count is
`96 + sum_s S_s`; this is determined by the fixed run roster, not by outcomes.

For groups g of hard sizes n_g, minimize the quantized weighted
between-group mean loss:

```
L = sum_j w_j * sum_g (n_g / N)
      * (mean_g(q_j)/10000 - mean_P(q_j)/10000)^2.
```

Use exact integers/rationals for every comparison. Let D be the least common
multiple of all w_j denominators, `W_j = D*w_j`, C the least common multiple
of group sizes, and `S_gj = sum_{c in g} q_cj`. The comparable integer is

```
J = sum_g (C/n_g) * sum_j W_j * S_gj^2.
```

The omitted parent-mean term is constant within an assignment; no rounded
display of L is used to accept a move. Hard-count tolerance is zero, numerical
tie means exactly equal J, and a move requires strictly smaller J. Covariate
balance is soft: there is no post-assignment pass threshold, category rescue,
outcome-driven rerun or search for a more attractive split.

## Deterministic finite assignment

Perform four assignments in this fixed order:

| Assignment | Parent | Groups in order | Namespace |
| --- | --- | --- | --- |
| Roles | all 720 concepts | development 480, support_calibration 120, sealed_audit 120 | `ep17-source-roles-v1` |
| Outer folds | the 480 development concepts | fold01 through fold06, each 80 | `ep17-source-roles-v1/outer-fold` |
| Development blocks | the same 480 development concepts | block01 through block24, each 20 | `ep17-source-roles-v1/development-block` |
| Audit blocks | the 120 sealed-audit concepts | block01 through block12, each 10 | `ep17-source-roles-v1/audit-block` |

Each child assignment recalculates imputation and standardization using only
its parent IDs and metadata. Development-block construction does not receive
outer-fold labels, balance fold intersections, or subdivide folds. Its input
API should make that exclusion explicit. Distinct namespaces and no fold
input establish the required construction boundary; they do not imply
probabilistic independence between deterministic metadata-balanced partitions.

Define canonical concept order by numeric `things_category_nr`, then the
unmodified uniqueID UTF-8 bytes. Define identity bytes as the ASCII decimal
CID with no leading zero, a tab, then the exact uniqueID encoded UTF-8. Reject
an embedded tab, newline or NUL in an identifier instead of escaping or
normalizing it. SHA-256 is used only to specify deterministic permutations.

For a given namespace, initialization key is SHA-256 of
`namespace UTF-8 + NUL + "init" + NUL + identity_bytes`. Sort lexicographically
by its 32 digest bytes, then canonical concept order, and allocate consecutive
segments of the required sizes to the named groups. There is exactly one
initialization per assignment and no seed sweep.

Enumerate every unordered pair of different parent concepts, with the lower
canonical ID first. Its scan key is SHA-256 of
`namespace UTF-8 + NUL + "swap" + NUL + identity_a + NUL + identity_b`.
Sort pairs by digest bytes, then their canonical IDs. This one fixed pair
order is reused for every sweep. When a pair is visited, skip it if the
concepts currently share a group; otherwise calculate the exact swap delta
and accept immediately if it is strictly negative. Equal-loss moves are
rejected. Group capacities are preserved by every move.

Run at most **eight complete sweeps**. Stop earlier only after a full sweep
accepts zero swaps. The eight-sweep bound is an algorithmic end point fixed
before assignment; its result is a completed bounded-search assignment even
if the eighth sweep still accepts moves. Report that condition and do not
claim a pairwise local optimum. A zero-swap sweep permits a pairwise-local
statement for this particular quantized objective only. Neither case supports
a global-optimum claim. A wall-time, memory, validation or arithmetic failure
before the prescribed end point leaves an incomplete attempt and publishes
no complete role/fold/block map.

For feasibility, compute the exact weighted Gram matrix
`G_ab = sum_j W_j*q_aj*q_bj` and cache
`T_ag = sum_{c in g} G_ac`. For a candidate exchanging a in A with b in B,
let `h = G_aa + G_bb - 2*G_ab`. Its exact comparable delta is

```
delta_J = (C/n_A) * (2*(T_bA - T_aA) + h)
        + (C/n_B) * (-2*(T_bB - T_aB) + h).
```

After acceptance, update every parent's cache entries by
`T_iA += G_ib - G_ia` and `T_iB -= G_ib - G_ia`. This changes evaluation from
a full feature loop per pair to constant-time exact arithmetic; accepted
updates cost O(N). Require Python arbitrary-precision integers, or a proven
bound for every intermediate before using any fixed-width integer path.
Do not rely on NumPy overflow wrapping, float tolerances or a BLAS result
for accept/reject decisions. Recompute J and all cached sums directly at each
completed sweep; disagreement stops publication, not the data universe.

The maximum pair visits, including same-group skips, are eight times
`choose(720,2) + 2*choose(480,2) + choose(120,2)` = **3,967,200**. This finite
bound is independent of measured balance. Accepted swaps, completed sweeps,
initial/final J, per-family loss, every raw/quantized residual, imputation
coverage and all exact capacities must be reported. The fixed namespace and
initial J permit reconstruction of the initialization; publish only one
selected final map, not a menu for later outcome-based selection.

## Pilot, resources and publication

The next proposed implementation stage is code and meaningful source-free
fixtures, then a **metadata preparation/timing pilot** in a coordinator-owned
allocation: one CPU, provisionally 8 GiB, no GPU, 300 seconds internal and 360 seconds
external with 10 seconds kill grace. Bound each existing CSV to 32 MiB,
use only the fixed completed metadata paths, and make no network request.
The memory request is not selected: the coordinator must first review a
concrete bound for parsed metadata, Decimal/Fraction features, Python-integer
Gram matrices, pair ordering and caches. Do not assume that the input byte
cap bounds Python object memory.
The pilot constructs the 720-concept feature matrix and role Gram/cache,
checks 4,096 initially cross-group pairs in frozen scan order against an
independent direct feature-sum delta calculation, and measures preparation,
pair-check and temporary cache-update costs. It accepts no source-based swaps,
publishes no roles, and does not try multiple normalizations or objectives.
Any temporary initialization stays internal; pilot output is feature
definitions/masks, numerical checks, operation counts, resource measurements
and timing. Preserve the reusable exact feature artifact if preparation
completes, with `roles_assigned: false`.

Fixtures must establish the weighted equal-image/equal-participant summaries,
separate delay sentinels and omitted-history semantics, all-parent-missing
moments, half-even quantization/ties, exact Gram/cache deltas and capacity
preservation, bounded sweep completion versus timeout, and absence of fold
labels from development-block inputs. They should test the stated scientific
invariants without source values or manufactured empirical results.

After the pilot, select one full assignment attempt only if measured costs
support the fixed four-assignment/eight-sweep program. A provisional maximum
is one CPU, up to 8 GiB, 1,800 seconds internal and 1,860 seconds external plus
10 seconds kill grace; the coordinator chooses an allocation that contains
the complete envelope. Use measured Gram preparation and cache-update costs
to report an upper resource estimate before selecting that attempt. If costs
do not fit, report the cost and prepare a reviewed implementation improvement
or prospective resource change. Do not silently lower sweep counts, change
precision, add restarts or publish an interrupted incumbent. No full run or
retry is automatically selected by a successful pilot.

Only after all four assignments and exact joins pass may one terminal report
declare the role/fold/block assignment complete and expose the complete
metadata map. Write-once fresh attempts preserve failures. Diagnostic feature
tables or partial maps without that completed report are ineligible for
neural handoff. Record exact role sizes, every fold/block capacity, whole-
concept inheritance, all preserved row coordinates, and the explicit
development-block input boundary. Materializing or opening any neural
response remains outside this proposal.

Resource selection at adoption: independent structural bounds estimate less
than2GiB peak with fixed metadata parsed once and intermediates released;
720-squared arbitrary-integer Gram and pair keys are tens of MiB, and Decimal
feature matrices about100MiB each at the generous929-column bound. Select
6GiB for the pilot; this is an estimate to be measured, not observed RSS. Full
cost must include all four parent Gram preparations,3,967,200 visits and up
to2,380,665,600 per-concept cache updates; do not assume few accepted swaps.
