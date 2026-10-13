# EP15 scientific margins — adopted standards and remaining proposals

Prepared October 3, revised October 6, 2026. No functional
value, candidate score, participant role or audit outcome informed the original
proposal. The subsequent scientist agreement adopts two intended standards;
the other five margin fields in `SEARCH_POLICY.yaml` remain `null`.
The completed anatomical intersection is fixed at 208 locations; it is not
changed to make these margins achievable.

The seven numbers below define a practical resolution for this internal MDTB
question. They are not literature-established biological constants, power
calculations, observed effect sizes, or values selected to pass simulations.
Approval would choose what counts as meaningful signal, adequate explanation
and a material native signature. It would not establish that this design has
enough precision to decide at that resolution.

## Adopted scope — October 6, 2026

The scientist's “i agree with your judgement” accepts exploratory real-data
development and the highlighted intended standards: `tau_remaining = 0.20`
(at least 80% recovery) and `tau_increment = 0.05` (five additional percentage
points of recovery against each required comparator). The policy records these
two values. Their adoption is not a seven-threshold endorsement or evidence of
valid coverage, power, adequacy, equivalence or unique-class support.

`tau_signal`, `tau_geometry` and all three native-signature margins below remain
proposals. The nominal-0.05 Bonferroni interval, seven-fixture replacement and
weaker procedural access proposals also remain unadopted. The current
exploratory branch can report real development point summaries once its source,
model and enforced reader requirements pass, without the original 700-run
qualification. That program and finished inference remain deferred obligations
for the original formal class-decision route, not completed gates. No audit
outcome, simulation job or resource extension is opened by the agreement.

The remaining historical proposal documents definitions and alternatives; its
seven-value adoption request and pre-score restrictions do not override the
accepted two-value/exploratory scope above. A future formal rule needs a
scientific rationale for the five remaining values and defensible uncertainty;
the user should not be asked to approve them merely because a grid contains them.

## Earlier approval request under the original formal contract

`GOAL.md`, “Margins and uncertainty,” requires: **“Four primary scientific
margins are recorded with scientist approval before any candidate-discriminating
development Task-B score.”** It separately requires the three branch-signature
margins to have fixed directions and simultaneous bounds. `SEARCH_POLICY.yaml`
states `scientist_approval_before_candidate_scores: required`, prohibits
synthetic qualification from choosing or widening a margin, and requires
`values_directions_and_simultaneous_bounds_fixed_before_candidate_scores: true`
for the three native margins. The pre-score qualification list requires
`scientist_approved_margins_and_exact_inference_implementation`.

The proposed decision is to approve all seven values together with the units,
training-only scale definitions and contrast directions below. Their numerical
evaluation happens later in the appropriate role-separated training runtime.
Nothing here authorizes a Task-B opening, policy edit or candidate score.

**Earlier proposal, superseded on adoption status by the section above:** retain
all seven values below, including
**tau_remaining = 0.20**, under the already implemented metric corrected here.
That asks for at least 80% recovery at the required uncertainty bound. With
12 audit participants, every model may remain unresolved; this is a legitimate
consequence of the intended sufficiency claim. Changing this value to 0.50
would define a different, majority-account question, not a numerical repair
or an equally strong sufficiency claim. It is not recommended as a way to
obtain a passing result.

| Policy field | Proposed value | Actual threshold and meaning |
| --- | ---: | --- |
| `tau_signal` | 0.05 | `0.05 S_map`: between-person map-variance energy must exceed 5% of positive crossvalidated source-map energy |
| `tau_remaining` | 0.20 — adopted intended standard | `0.20 V`: at most 20% of reliable between-person variation may remain, separately for each applicable map/geometry adequacy contrast; calibrated formal adequacy is not established |
| `tau_increment` | 0.05 — adopted intended standard | `0.05 V`: five percentage points of reliable-variation recovery beyond each required comparator; calibrated superiority is not established |
| `tau_geometry` | 0.05 | `0.05 S_geom`: individual geometry-variance energy compared with 5% of the actual source-geometry energy |
| `M1_boundary_enrichment` | 0.10 | Boundary minus matched-interior recovery, each normalized by its own positive training-only signal opportunity, must exceed 0.10 |
| `M2_gradient_alignment_advantage` | `0.01 S_map` | Observed residual alignment advantage over the matched random-warp control must exceed 1% of source-map energy |
| `M4_mode_ablation_observed_geometry_gain` | `0.01 S_geom` | Ablating each declared mode must increase its observed projected-geometry squared error by more than 1% of source-geometry energy in **each** independent replicate |

The four `tau` values are dimensionless coefficients. The last two native
thresholds are scale-relative expressions, not bare dimensionless `0.01`
values in an otherwise dimensional comparison. If the eventual policy format
stores a numeric coefficient, its scale and units must be recorded alongside
it; this proposal does not prescribe a schema change.

## Common units, weights and source scales

Let `U` denote the released beta-amplitude unit, retained through the frozen
arithmetic map construction. Let `p = 208` be the fixed common voxel count.
For a spatial vector, `C x = x - mean(x) 1`; define
`<x,y>_v = x^T y / p`. Thus a map cross-product has units `U²`.
For condition `c` in domain `g`, use `w_c = 1/(7 n_g)`; the seven domain
condition counts are `(1,3,3,3,3,2,3)`. Conditions are never averaged before
forming cross-products.

For full symmetric 18-by-18 matrices, use the **implemented October 5/6
domain-pair convention** from [the selected metric](paper_plan.md#common-map-and-full-signed-geometry-metrics)
and [endpoint code](tools/registered_endpoint_metrics.py). Each of the 28
unordered domain pairs receives weight 1/28. Within a same-domain block,
distribute it equally over its n_g(n_g+1)/2 unique condition pairs, including
the diagonal; within a cross-domain block, over its n_g n_h pairs.
The equivalent full-matrix inner product is:

```text
<A,B>_Ω = sum_c sum_d Ω_cd A_cd B_cd

Ω_cc = 1/[14 n_g(n_g+1)]                  if c belongs to domain g
Ω_cd = 1/[28 n_g(n_g+1)]                  if c != d share domain g
       1/(56 n_g n_h)                    if g != h.
```

Weights sum to one and are positive for every entry. Both off-diagonal copies
receive half their unique-entry weight; they are not separate observations.
This replaces the October 3 proposal's same-domain 1/(28 n_g²) expression,
which was not the subsequently implemented unique-entry metric. Every S_geom,
gain-removal projector, geometry variance, mode projector and ablation margin
below uses this corrected Ω. Map weights and all seven proposed numerical
values remain unchanged. No already fitted empirical result is rescaled.

For a permitted training set `T`, `G_c(T)` means the **actual** development-only
group-functional source supplied to M3 and the paired panel. It is not replaced
by M0a's raw group mean or a released MDTB-fitted map. Construct `G_c^(1)(T)`
and `G_c^(2)(T)` by the same frozen source-map recipe from training participants'
Task-B runs 1–8 and 9–16 respectively. Then define

```text
S_map(T) = sum_c w_c <C G_c^(1)(T), C G_c^(2)(T)>_v.       [U²]

X0,q(T)[c,:] = C G_c,q(T), q = 1,2,3,4
K0,a(T) = (X0,2a-1 X0,2a^T + X0,2a X0,2a-1^T)/(2p), a = 1,2
K0(T)   = (K0,1(T) + K0,2(T))/2                         [U²]
S_geom(T) = <K0(T), K0(T)>_Ω.                            [U⁴]
```

`G_c,q` denotes the four actual source partitions (runs 1–4, 5–8,
9–12 and 13–16), after the same fixed training-only source transform.
`K0` averages their two symmetric crossvalidated geometries; it is not the
self-Gram of their mean map. This corrects an older proposal equation to the
later implemented source recipe in `local_functional_source.py` and the
October 7 source/LOPO record; no new empirical arithmetic is adopted here.
Any additional frozen, permitted source steps belong in those actual partitions;
one cannot substitute another convenient Gram matrix.
The geometry scale is the unprojected `K0` energy. Projecting `K0` off itself
would give zero and is not a valid scale definition. Neither scale uses a
target's matrix trace, target Task-B variance, target gain, audit outcome, or
the best-performing candidate.

The half-specific source construction must preserve the already-required
measurement-error independence. Any response-dependent shared fitting that
couples the two halves needs a supported crossfitted implementation before
these scales are evaluated; a positive number by itself cannot certify error
independence. Once the training-only source recipe is fixed, evaluate the
formula as written. Nonfinite or nonpositive `S_map`, or nonfinite/zero
`S_geom`, leaves the corresponding scale unresolved. No absolute value,
epsilon floor, trace replacement, alternate source, or threshold adjustment
is permitted to obtain a positive denominator.

For an audit, `T` is the 12 development participants, and these two constants
are frozen once before audit calibration. For development pseudo-target `i`,
`T_i` excludes that participant's Task B; its source, `K0`, projector and scales
are frozen before its score is computed. The pooled development signal
contrast uses the arithmetic mean of those crossfitted scale constants, not
a source/normalizer fitted on all 12 and reassigned to each pseudo-target.
No participant-specific recovery ratio is introduced. Development remains
selection evidence; the audit uses one common source and one pair of scales.

## Exact map and geometry contrasts

For audit participant `i`, condition `c` and map half `h`, write
`r_i,c,h = C(Y_i,c,h - G_c)`. The fixed model's individual residual prediction
is `d_i,c,m = C(prediction_i,c,m - G_c)`. It is the same prediction in both
observed halves. Any optional positive gain is estimated from Task A only and
applied identically to all voxels/conditions under the existing contract.

For `n = 12`, define `Pair(a,b)` as the mean over the `n(n-1)/2` unordered
participant pairs of `0.5 <a_i-a_j, b_i-b_j>`. The factor one-half puts the
statistic on participant-variance, rather than pair-difference, scale:

```text
V_map    = sum_c w_c Pair(r_c,1, r_c,2)                    [U²]
R_map(m) = sum_c w_c Pair(r_c,1-d_c,m, r_c,2-d_c,m).       [U²]
```

The equal-voxel inner product above applies within `Pair`. No reliable
variance or remaining variance is clipped to zero. After a positive signal
gate, `1 - R/V` may be displayed; decisions use the linear contrasts below,
never confidence intervals for a ratio or averages of participant ratios.

Let `X_i,q` be the 18-by-p spatially centered observed matrix for four-run
partition `q`. The two **primary** geometry replicates are

```text
K_i,1 = (X_i,1 X_i,2^T + X_i,2 X_i,1^T)/(2p)
K_i,2 = (X_i,3 X_i,4^T + X_i,4 X_i,3^T)/(2p).
```

They retain the full signed crossvalidated matrix, including possibly negative
eigenvalues. Define the fixed, weighted linear gain-removal operator

```text
P0(H)   = H - K0 <H,K0>_Ω / S_geom
z_i,a   = P0(K_i,a - K0)
ẑ_i,m  = P0(K̂_i,m - K0)
V_geom  = Pair_Ω(z_1, z_2)                               [U⁴]
R_geom(m) = Pair_Ω(z_1-ẑ_m, z_2-ẑ_m).                   [U⁴]
```

`K̂_i,m` averages the two symmetric crossvalidated geometries of the four
source partitions after the same frozen personalized transform, including the
one common Task-A gain, under the same centering and `1/p` convention. It is
not the self-Gram of the mean prediction. This matches the later implemented
four-partition prediction recipe. `Pair_Ω` replaces the spatial inner product
with `<.,.>_Ω`. M3's gain-only geometry direction is removed; one cannot
normalize each observed matrix to equal trace. Alternative partition pairings
remain nonselective sensitivity analyses and cannot rescue a primary failure.

Let `L` and `U` denote the contract-required one-sided simultaneous bounds from
the fixed participant inference procedure. Every inequality below is strict;
equality, an interval crossing zero, or a needed unresolved scale is unresolved.

| Decision | Oriented contrast | Required bound |
| --- | --- | --- |
| Map signal present | `V_map - 0.05 S_map` | `L > 0` |
| Map signal below practical resolution | same contrast | `U < 0` |
| Geometry signal present | `V_geom - 0.05 S_geom` | `L > 0` |
| Geometry equivalent at this resolution | same contrast | `U < 0` |
| Map adequacy of class `m` | `R_map(m) - 0.20 V_map` | `U < 0`, after map signal is present |
| Geometry adequacy of class `m` | `R_geom(m) - 0.20 V_geom` | `U < 0`, when geometry signal is present |
| Material increment of `m` over `r` | `R_r - R_m - 0.05 V` | `L > 0`, separately for map and geometry where required |
| No material increment of `m` over `r` | same increment contrast | `U < 0` |

Thus adequacy means at least 80% recovery at the required bound, and material
increment means more than five additional percentage points of recovery.
The 5% signal/geometry thresholds are **energy** fractions, not 5% amplitude
or correlation differences; under a simple equal-scale norm interpretation
their amplitude ratio is approximately `sqrt(0.05)`, not `0.05`.

M4 must pass separate increments over **each** M1, M2 and its paired M3,
for both maps and geometry. No minimum, best comparator or average across
comparators replaces those six contrasts. M3 additionally requires geometry
equivalence and **both** of these no-material-paired-M4-increment bounds:

```text
U[R_map(M3)  - R_map(paired M4)  - 0.05 V_map ] < 0
U[R_geom(M3) - R_geom(paired M4) - 0.05 V_geom] < 0.
```

The second remains required when geometry is within the equivalence margin.
It is a linear contrast, not a geometry-recovery ratio. Small or imprecisely
estimated geometry variation may leave it unresolved; that is an allowed
outcome and does not justify dropping the component. Geometry improvement
cannot compensate for a failed map comparison. M1 and M2 require geometry
adequacy whenever the geometry gate is present, and a resolved geometry gate
otherwise. The four nonnested classes are not selected by a scalar ranking.

## Three native signatures and their replicate rules

### M1: 0.10 boundary-enrichment difference

Freeze development-defined boundary neighborhoods `B0` and parcel interiors
`I0` before candidate scores. A fixed matching rule produces equal-voxel-weight
boundary and interior subsets for each target, using only allowed Task-A
measurement reliability, development-only Task-B signal opportunity and
development source gradients. No target Task B chooses masks or matching.

For target `i` and matched mask `H` (boundary or interior), let
`O_i,H(T)` be positive development-only between-person residual signal
opportunity on that exact mask, computed with the same weighted cross-half
pair statistic as `V_map`, using training people only. It has units `U²`.
Mask inner products average equally across the selected voxels; residuals
retain the frozen full-support spatial centering. Define

```text
A_i,H = sum_c w_c [ <r_i,c,1, r_i,c,2>_H
                  - <r_i,c,1-d_i,c,M1, r_i,c,2-d_i,c,M1>_H ]   [U²]
D_i,M1 = A_i,B / O_i,B(T) - A_i,I / O_i,I(T).                  [unitless]
θ_M1   = mean_i D_i,M1 - 0.10.
```

Require `L(θ_M1) > 0`. The two independent map halves enter the cross-product;
there is no selection of a favorable half. These denominators are fixed
training signal opportunities, not ratios to each target's observed signal.
Neither an empty matched set nor a nonpositive opportunity is replaced or
floored. Such a configuration is not evaluable under this signature.

Parcel interiors must separately be adequate under the same 0.20 remaining
fraction, using the common development-frozen `I0` support and corresponding
pairwise map statistic. A good enrichment difference cannot excuse inadequate
interiors. The numerical matching, boundary-width and parcel-capacity choices
remain separate pre-score specifications; this margin does not choose them.

### M2: 0.01 source-map-energy alignment advantage

Let `p_i,c = C(a_i [W_i G_c - G_c])`, using the frozen Task-A-derived field
and the shared optional Task-A gain `a_i`. This is the finite source-map
change under the field, without an arbitrary gradient-vector normalization.
Let `p_i,c,rand` be the fixed mean over the predeclared matched-random-warp
control ensemble, using the same source, gain, capacity, displacement and
smoothness rules. Its count/seeds are fixed before candidate scores, and
no random realization is selected using target B.

```text
D_i,M2 = sum_c w_c <p_i,c - p_i,c,rand,
                         (r_i,c,1 + r_i,c,2)/2>_v         [U²]
θ_M2   = mean_i D_i,M2 - 0.01 S_map.                      [U²]
```

Require `L(θ_M2) > 0`. The policy defines alignment averaged over the two
map halves; this proposal does not replace that average with the better
half or impose an unregistered both-half requirement. The positive direction
is greater agreement with observed B-only residuals than the matched random
warp. It is not a cosine or a claim of 1% map-variance recovery. Field stability,
displacement/Jacobian/inverse and interpolation certificates remain independently
required; this energy margin does not replace them.

### M4: 0.01 source-geometry-energy ablation gain in both replicates

Before audit, fix the finite mode list, the no-refit mode-ablation operation,
and mode-geometry projectors from development. A projector `Π_k` is orthogonal
under `<.,.>_Ω` on the symmetric-matrix space after gain removal. For a single
normalized pattern `H_k`, it is
`Π_k(Z) = H_k <H_k,Z>_Ω / <H_k,H_k>_Ω`.
The mode directions and normalizers cannot be fitted from audit B.

For each declared mode `k`, keep the paired `Q`, remaining mode coefficients,
global gain and every other parameter fixed when forming `ẑ_i,-k`. A valid
ablation must follow the frozen trace-free/operator construction; no refit,
alternate mode, changed bound or reselected projector is allowed.

```text
D_i,k,a = || Π_k(z_i,a - ẑ_i,-k) ||_Ω²
          - || Π_k(z_i,a - ẑ_i,M4) ||_Ω²                 [U⁴]
θ_k,a   = mean_i D_i,k,a - 0.01 S_geom,   a in {1,2}.     [U⁴]
```

Require `L(θ_k,1) > 0` **and** `L(θ_k,2) > 0` for **every** predeclared mode.
Do not average the replicates, drop a failing mode, replace its geometry
projection or use only the change in the model's own prediction. The two
observed replicates are the independent partition1×2 and partition3×4
geometries above; they are not additional participants. Map/geometry adequacy,
the six material-increment contrasts, Task-A-half stability, matched-smoothness
control and identifiability certificates remain separate required components.

## Inference and interpretation boundaries

The inferential unit remains the participant. For a fixed audit contrast
`θ`, its delete-one participant pseudovalue is
`q_i = n θ - (n-1) θ_(-i)`. Removing `i` removes every pair containing that
participant; shared development source/scales and the already-written target
predictions stay frozen. No audit-based refit occurs. With
`q̄ = mean(q_i)`, the standard error is
`sqrt(sum_i (q_i-q̄)² / [n(n-1)])`. For a participant-mean native signature,
this reduces to the usual participant-mean standard error. No condition,
voxel, pair, half, mode or run partition increases `n` above 12.

That frozen-source delete-one formula is the **audit** calculation conditional
on the completed development fit. Development LOPO sources and scales depend
on other development participants. The same fixed-prediction calculation is
not automatically calibrated development inference: its dependence and any
specified refitting must be included in the already-required complete-selection
simulation/inference definition. This is not another margin or source-access
gate, and no development confidence claim is made by this proposal.

The operative contract still requires intersection-union class claims and
Holm-inverted one-sided bounds over M1–M4; its exact implementation is
unresolved. The following is a **proposed replacement**, not an implementation
or a reinterpretation of that requirement. Both M4 replicates and every
declared mode remain required. Nonsignificance is never adequacy, equivalence,
or affirmative refutation.

### October 6 draft: one joint region and an explicit family error level

Propose a **nominal family-wise error level of 0.05 for the one locked audit
panel**, with one two-sided Bonferroni rectangle for the full scalar contrast
vector. Before any candidate-discriminating Task-B score, freeze its dimension
d and complete list: the implemented 18+2Q primitives for Q fixed M4 modes,
plus every still-to-be-specified inferential stability or matched-control
contrast used in any decision. Deterministic validity certificates remain
requirements but are not invented statistical observations.

For each scalar statistic T_j, the concrete approximate implementation would
use the paired participant jackknife standard error and

    [T_j - t_(11,1-0.05/(2d)) SE_J(T_j),
     T_j + t_(11,1-0.05/(2d)) SE_J(T_j)].

Center on the declared statistic T_j, not the pseudovalue mean. All lower and
upper decisions come from this same region, covering positive signal,
adequacy/equivalence, native signatures, increments and affirmative negative
decisions together. Evaluate the class Boolean regions without adding a
second data-dependent correction: support a class only if the entire joint
region satisfies its allowed paths; refute it only if the region cannot
intersect any allowed path; otherwise remain unresolved. M1/M2's alternative
geometry paths are a union inside that class region, not a selected new test.
A unique-class conclusion needs one supported class and every rival refuted.
Boundary touches are unresolved. A zero or nonfinite estimated standard error
does not produce an infinite z-score or automatic pass; that inferential
component is unresolved.

This replaces four-class Holm inversion rather than claiming to implement it.
Bonferroni's multiplicity control is valid if the marginal intervals have
their stated coverage; dependence among contrasts is allowed. **Jackknife
t calibration at n=12 is approximate here.** Calling the rectangle conservative
does not establish those marginal coverages or exact 5% error control.
Until a defensible calibration is supplied, adoption could authorize
exploratory interval reporting with this conditional-coverage limitation,
not an unqualified confirmatory sufficiency or unique-mechanism conclusion.
Development intervals additionally require the specified nested refits;
fixed-prediction deletion alone is not calibrated developmental inference.

### October 6 draft: bounded implementation checks in place of recovery expansion

The original formal route still requires seven worlds times 100 seeds (700
complete selection/decision runs); the adopted exploratory branch defers that
program. The following replacement remains unadopted: **seven fixed-seed known-answer
integration fixtures**, one per existing named world, using supplied model
predictions to check signed estimands, source/role isolation and decision logic.
They would run once inside an otherwise authorized substantive real-data job,
with a proposed aggregate 120 CPU-second limit and no retries, added seeds,
power search or full model-recovery sweep. A failed/incomplete check stops the
affected operation without enlarging the check program. This draft grants no
job or simulation authority.

These seven cases cannot measure power, coverage or family-wise error, and
do not replace scientific operating-characteristic evidence. A real-data
permutation diagnostic is optional only under a predeclared defensible
exchangeability null: preserve whole participants and both Task-B replicate
bundles, retain the fixed weights and redo every affected nuisance/selection
step when required. It cannot validate adequacy or geometry equivalence under
all alternatives. No permutation scheme, refitting null job or allowance is
selected here; without a justified null and bounded cost, omit that diagnostic
rather than call arbitrary shuffling a calibration.

Adopting this smaller checking program would also require explicitly revising
the original qualification requirement and limiting subsequent inference as
above. Seven passing fixtures or an attractive permutation result cannot
certify the original four-class scientific claim. The 0.20 sufficiency estimand
remains the recommended question; unresolved precision is retained rather
than solved by changing it to 0.50.

### Current anatomical support and the separate access proposal

Retain the already authenticated **208-location common support** for all
24 participants, with equal voxel weights and no Task-B-driven addition,
replacement or resampling. This is the existing support, not a new ratification
gate. It limits claims to that measured spatial resolution, not whole-cerebellum
sufficiency. If another resolution is needed for a different scientific claim,
that requires a prospective redesign before Task-B scoring, not a support
change after seeing results.

The cross-episode suggestion to replace enforced evaluator isolation by a
disclosed procedural boundary is a separate **unadopted** access amendment.
It would require corresponding claim language and explicit role handling;
this margin/inference proposal neither adopts it nor releases any handoff.

## What full formal approval would enable, and the remaining custodian dependency

Beyond the two adopted intended standards, approval of a justified complete
formal rule would resolve the remaining scientific choices, with the scale
and weighting conventions above. The actual `S_map`, `S_geom`, signal
opportunities and normalizers cannot yet be filled numerically: the Task-B-blind
12/12 roles are fixed, but no training Task-B handoff has been released to a
fitting worker and no actual M0b/M3 source recipe has been evaluated.
Training-only positivity and error
independence must subsequently be checked without consulting audit results.
The 208-location support may limit precision and signature feasibility; that
is a consequence to report, not a reason to adjust the requested resolution.

Task-A-only mapping/finite-support checks, all24 private Task-A reliability
summaries, the exact Task-B-blind 12/12 assignment and opaque-byte materialization
of all36,576 members into36 containers have completed. These facts are recorded
in [the execution log](experiment_log.md); they do not establish reader
isolation. The remaining handoff task is enforcement and release of distinct
development, audit-A-calibration and audit-B-evaluation views under the adopted
access contract.
The development worker may then receive development anatomy/A/B. Audit A is
available only to the locked calibrator after panel lock; audit B only to
the trusted evaluator once predictions are locked. The custodian returns
roles and balance summaries, not audit participants' private reliability values.

The current shared Unix identity and agent task scopes alone cannot enforce
this separation when that identity can read the mixed original archives.
The handoff needs a distinct authorized execution principal or equivalently
enforced runtime access boundary that actually withholds the mixed audit
source and exposed fitted caches from search. This is a remaining runtime
dependency, not a request to change source ownership, permissions or data roles
in this proposal. Root will coordinate the concrete handoff; workers do not
open functional source values or change the already fixed roles here.

Exploratory scoring still needs the map/source recipe, measurement-error
checks, fixed branch capacities and matching/control recipes, applicable
numerical certificates and bounded scratch/resources. Exact calibrated
inference, all remaining scientific margins, whole-procedure qualification
and final audit prediction locking belong to the later formal route. The
October 6 adoption defers those formal decisions without approving the proposed
inference/validation replacement or procedural access boundary. No scientific
job is selected in this document.
