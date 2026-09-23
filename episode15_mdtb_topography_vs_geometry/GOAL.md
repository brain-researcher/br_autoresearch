# Topography or Response Geometry?

## Task-A Prediction of Individual Cerebellar Task-B Maps

## Authority and scope

This is the current local Episode 15 design. It creates no canonical loop,
authorizes no compute, and opens no audit Task-B outcome. Launch requires a
canonical registration, the role-separated handoffs in `DATASETS.md`, and a
separate explicit decision. Until then, `SEARCH_POLICY.yaml` is a design
contract rather than an executable instruction.

The primary scientific scope is the **cerebellum**. Cortex is outside this
episode. MDTB is a heavily analysed public release, so the participant holdout
is a procedure-sealed internal audit, not a pristine replication. The maximum
claim is a model-class distinction within this release.

## The question in plain language

Previous MDTB studies have already shown that functional data from one task
set can improve prediction of an individual's organization in another task
set. This episode does not test that result again. It asks what kind of stable
individual organization is required to explain the reliable part of unseen
Task-B-only cerebellar maps.

Using target anatomy and Task A only, can stable between-person differences in
Task-B-only maps be adequately explained by:

1. individualized parcel membership and boundaries with shared parcel
   response profiles;
2. one smooth spatial relocation field applied to every condition;
3. one certified isometric functional correspondence that preserves centered
   condition geometry; or
4. a bounded non-isometric correspondence that permits stable,
   participant-specific changes in response geometry?

The target is an **adequate constrained explanation**, not the model with the
largest raw score. More flexible structure earns scientific credit only when
it predicts reliable individual variation and a branch-specific signature
that the plausible simpler explanations miss.

If every Task-A-derived class is inadequate, the valid result is that reliable
Task-B-only individual variation remains nontransportable under this
contract. Because Task A and Task B were acquired in different sessions, that
result alone cannot distinguish task dependence from session instability.

## Episode at a glance

| Item | Frozen design |
| --- | --- |
| Target input | Anatomy plus Task A only |
| Sealed outcome | 18 Task-B-only maps from 12 audit participants |
| Primary evidence | Reliable between-person map variation and individual condition-geometry variation |
| Scientific comparison | M1 parcels, M2 smooth relocation, M3 isometry, M4 bounded non-isometry |
| Search output | One composite panel containing one certified representative of every branch |
| Positive result | One uniquely supported adequate class, not merely the highest score |
| Main boundary | Internal MDTB mechanism discrimination; no task-dependence or external-replication claim |

## Why this is not another alignment benchmark

The four scientific classes form a **predeclared explanatory panel**, not a
nested ladder. Parcel personalization, a diffeomorphic warp, and an isometry
make different assumptions and can each alter map prediction in ways the
others cannot. Only the paired M3/M4 construction is nested.

Condition geometry is also not itself a new object: prior MDTB work has used
crossvalidated condition distances and task-by-task Gram matrices. The new
test here is narrower: can a transform learned from Task A predict a
participant's reproducible deviation from the development-derived Task-B-only
condition geometry, and does that deviation falsify an isometric account?

## Prediction contract

Let `D_dev` be the development participants and let \(G_c\) be the frozen
development-only Task-B source representation for condition \(c\). Each class
first fits its shared parameters from development A and B,

\[
\widehat\phi^{(m)}=h_m(D_{\mathrm{dev},A},D_{\mathrm{dev},B}),
\]

then estimates a target participant's parameters from anatomy and Task A only,

\[
\widehat\theta_i^{(m)}
=g_m(A_i,Y_{i,A};\widehat\phi^{(m)}),
\]

and predicts every Task-B-only condition with the same target-specific rule,

\[
\widehat Y_{i,B,c}^{(m)}
=f_m(G_c,\widehat\phi^{(m)},\widehat\theta_i^{(m)}).
\]

For a development pseudo-target, all shared priors, bases, response profiles,
source maps, thresholds, and normalizations are fitted without that target's
Task B. For the final audit, they are refitted once using development
participants only and locked before any audit participant is calibrated.

No released atlas, basis, map library, prior, checkpoint, or fitted parameter
that may contain an audit participant's Task-B information may enter a model,
initializer, synthetic generator, or selection rule. Published algorithms may
be reimplemented and refitted on development data.

## Primary endpoint

The endpoint contains all 18 Task-B-only conditions. The three movie
conditions are collapsed into one weighting domain, producing seven
equal-weight domains:

| Domain | Conditions |
| --- | --- |
| `CPRO` | `CPRO` |
| `prediction` | `Prediction`, `PredictViol`, `PredictScram` |
| `spatialMap` | `SpatialMapEasy`, `SpatialMapMed`, `SpatialMedDiff` |
| `movie` | `NatureMovie`, `RomanceMovie`, `LandscapeMovie` |
| `mentalRotation` | `MentalRotEasy`, `MentalRotMed`, `MentalRotDiff` |
| `emotionProcess` | `BodyMotionIntact`, `BodyMotionScram` |
| `respAlt` | `RespAltEasy`, `RespAltMed`, `RespAltDiff` |

Each condition has weight

\[
\alpha_c=\frac{1}{7|C_{d(c)}|}.
\]

Thus each domain has weight \(1/7\), while the 18 separate conditions remain
available to the condition-geometry endpoint. Cross-products are calculated
condition by condition and then weighted; maps are not averaged within a
domain, because that would allow condition effects to cancel.

The inferential unit is the participant. Voxels, conditions, domains, run
partitions, and condition pairs are repeated measurements, not independent
samples. The 14 shared A/B conditions, including `rest`, are excluded from the
primary endpoint and opened only as a locked post-decision diagnostic.

## Explanatory panel

All classes use the same support, interpolation, Task-B source information,
target Task-A information, and scoring code. The only global gain is one
positive participant scalar estimated from Task A and applied uniformly to
every voxel and every Task-B condition. The same gain rule is available to
every class; condition-, domain-, or voxel-specific target gains are
prohibited.

| Class | Frozen role | Scientific explanation |
| --- | --- | --- |
| **M0a** | Anatomy/group control | No target-specific functional organization |
| **M0b** | Group-functional control and residual origin | Development functional denoising is sufficient |
| **M1** | Mandatory hierarchical-parcel incumbent | Shared parcel response profiles plus personalized membership/boundaries are sufficient |
| **M2** | One Task-A-estimated smooth spatial warp | Stable continuous spatial relocation is sufficient |
| **M3** | Certified isometric correspondence | Spatial expression varies, but centered condition geometry is shared |
| **M4** | Bounded non-isometric challenger paired to M3 | Stable geometry-changing individual variation is required |

Pseudo-calibration, participant derangement, random local bases, and matched
random warps are negative controls, not scientific explanations.

M0a is the raw conditionwise development mean under the frozen anatomical/SUIT
resampling, with no functional basis or target Task A. M0b is a development-
fit group-functional prediction with no target-specific input; its condition
map is the \(G_c\) used below. Their difference measures group-level functional
denoising, not individualization.

### M1 — hierarchical parcel personalization

M1 is refitted on development participants. The target's Task A changes only
its parcel-membership probabilities or boundaries; parcel response profiles
remain shared. Required variants test probabilistic versus hard membership
and coarse, reference, and fine granularity.

Its native signature is boundary-localized recovery. The analysis must report
recovery in frozen boundary neighborhoods and parcel interiors. A gain that
occurs only in parcel interiors contradicts the proposed boundary mechanism.
Adequacy supports the sufficiency of this constrained model, not the literal
existence or uniqueness of discrete neural parcels.

### M2 — smooth spatial relocation

M2 learns one bounded diffeomorphic field from Task A and applies it unchanged
to every Task-B condition. It must pass frozen displacement, Jacobian,
inverse-consistency, interpolation, and Task-A-half stability checks. A
capacity- and smoothness-matched random warp is mandatory.

A usual warp need not preserve voxel-weighted condition geometry; therefore a
condition-geometry change alone does not distinguish M2 from M4.

### M3 — certified isometric correspondence

Write centered condition maps as rows \(X=YH\), with

\[
H=I-\frac{1}{V}\mathbf 1\mathbf 1^\top.
\]

M3 predicts \(XQ_i\), where the final operator on the registered support must
satisfy, to a frozen numerical tolerance,

\[
Q_i^\top Q_i=Q_iQ_i^\top=I,
\qquad Q_i\mathbf 1=\mathbf 1.
\]

Consequently,

\[
(XQ_i)(XQ_i)^\top=XX^\top.
\]

The identifiable implementation is

\[
Q_i=I+B(O_i-I)B^\top,\qquad O_i\in O(k),
\]

where the development-fixed basis \(B\) is orthonormal, centered, bounded in
dimension and effective degrees of freedom, and has a canonical identity
extension. The identity extension prevents arbitrary changes outside \(B\),
but does not by itself identify \(O_i\) within \(B\). Every valid instance must
also pass a frozen Task-A design-rank and unique-solution certificate under a
deterministic sign/order/tie convention; unidentified active directions are
removed or fixed to identity.

Weighted averages of orthogonal operators and overlapping local Procrustes
fits followed by averaging are prohibited unless the final operator is
reprojected and recertified. A composition is allowed only after the final
operator again passes every certificate.

### M4 — bounded stable non-isometry

M4 is the controlled falsifier of a paired M3 instance:

\[
L_i=Q_i\exp(S_i),
\qquad
S_i=BC_iB^\top,
\]

with

\[
C_i=C_i^\top,
\quad \operatorname{tr}(C_i)=0,
\quad \lVert C_i\rVert_2\le s_{\max},
\quad \operatorname{rank}(C_i)\le r.
\]

The basis is centered, so \(S_i\mathbf1=0\). The paired M3 uses the same
\(Q_i\), support, basis, preprocessing, and fitting rule; \(S_i=0\) must
reproduce it byte for byte. The singular values are bounded by
\([e^{-s_{\max}},e^{s_{\max}}]\) and the condition number by
\(e^{2s_{\max}}\). Rank, locality, effective degrees of freedom, smoothness,
and all bounds are frozen before outcome-guided search.

The two primary subfamilies are centered-basis anisotropic stretch and
localized low-rank stretch/shear. A voxel-diagonal stretch with
\(S\mathbf1=0\) would be identically zero and is not an allowed branch.
Sparsity of \(S\) is also not called locality of \(\exp(S)\); locality is
certified on the final operator.

M4 earns scientific credit only if it is adequate, is stable across Task-A
halves, and materially improves both map residual recovery and prediction of
individual condition-geometry deviations beyond M1, M2, and the paired M3.
A map-similarity gain alone is insufficient.

## Primary estimand 1: reliable between-person map variation

Task B is divided into two frozen eight-run map halves. Let
\(y_{i,c,h}\in\mathbb R^V\) be the map for participant \(i\), condition \(c\),
and half \(h\in\{1,2\}\). Define the centered residual from M0b and the model's
predicted residual as

\[
r_{i,c,h}=H(y_{i,c,h}-G_c),
\qquad
p_{i,c}^{(m)}=H(\widehat y_{i,c}^{(m)}-G_c),
\]

and \(e_{i,c,h}^{(m)}=r_{i,c,h}-p_{i,c}^{(m)}\). Spatial inner products are
divided by the fixed number of support voxels.

The reliable between-person signal is the condition-weighted pairwise
cross-half U-statistic

\[
V_B=
\sum_c\frac{\alpha_c}{n(n-1)}
\sum_{i<j}
\left\langle
r_{i,c,1}-r_{j,c,1},
r_{i,c,2}-r_{j,c,2}
\right\rangle_V.
\]

The reliable variation remaining after model \(m\) is

\[
R_B^{(m)}=
\sum_c\frac{\alpha_c}{n(n-1)}
\sum_{i<j}
\left\langle
e_{i,c,1}^{(m)}-e_{j,c,1}^{(m)},
e_{i,c,2}^{(m)}-e_{j,c,2}^{(m)}
\right\rangle_V.
\]

Pairwise differencing removes any shift common to the audit cohort and the
fixed development template. The interpretable aggregate effect is

\[
F_B^{(m)}=1-\frac{R_B^{(m)}}{V_B},
\]

but only after the registered signal gate resolves meaningful positive signal.
Let \(S_B>0\) be the frozen reliable-map scale. Signal is present only when the
simultaneous lower bound for
\(V_B-\tau_{\mathrm{signal}}S_B\) is above zero; it is negligible only when
the simultaneous upper bound is below zero; otherwise it is unresolved.
Participant-specific ratios are prohibited. Values are not clipped. An
observed value above one
indicates finite-sample negative residual cross-covariance or a failed
independence assumption, not superior-to-perfect recovery; bad overprediction
normally reduces \(F_B\), potentially below zero.

The factor \(1/[n(n-1)]\) is intentional: it is one half of the usual average
over unordered pairs, so a pairwise squared difference estimates one
participant-level variance rather than twice that variance.

The estimand requires independent half-specific measurement error and one
latent map per condition. Shared run nuisance can bias it and is tested in the
frozen split diagnostics.

## Primary estimand 2: individual condition geometry

Two independent estimates of reliable geometry require four independent
Task-B run partitions. The primary partition is runs `1--4`, `5--8`, `9--12`,
and `13--16`; the first two form one geometry estimate and the last two the
other. Alternative disjoint pairings are prespecified robustness checks and
cannot be chosen from their results.

Let \(X_{i,a}=Y_{i,a}H\in\mathbb R^{18\times V}\) contain the centered
condition maps for partition \(a\). For disjoint partitions \(a,b\), define

\[
K_i^{ab}
=\frac{X_{i,a}X_{i,b}^\top+X_{i,b}X_{i,a}^\top}{2V}.
\]

This crossvalidated matrix is symmetric but need not be positive semidefinite;
its trace can be nonpositive. It is therefore never normalized by its random
trace.

Let \(K_0\) be the actual development-only source geometry used by M3, with
its construction and positive scale frozen before search. It is not silently
replaced by the Gram matrix of raw group-mean maps, which can be attenuated by
heterogeneous rotations. Define domain-balanced geometry inner product and a
linear global-gain projection by

\[
\langle A,B\rangle_\Omega=\operatorname{tr}(\Omega A\Omega B),
\qquad
\Omega_{cc}=\alpha_c,
\]

\[
P_0(A)=A-
\frac{\langle A,K_0\rangle_\Omega}
     {\langle K_0,K_0\rangle_\Omega}K_0.
\]

For \(z_i^{ab}=P_0(K_i^{ab})\) and model prediction
\(\widehat z_i^{(m)}=P_0(\widehat K_i^{(m)})\), define

\[
V_K=\frac{1}{n(n-1)}\sum_{i<j}
\left\langle z_i^{12}-z_j^{12},z_i^{34}-z_j^{34}\right\rangle_\Omega,
\]

and

\[
R_K^{(m)}=\frac{1}{n(n-1)}\sum_{i<j}
\left\langle
(z_i^{12}-\widehat z_i^{(m)})-(z_j^{12}-\widehat z_j^{(m)}),
(z_i^{34}-\widehat z_i^{(m)})-(z_j^{34}-\widehat z_j^{(m)})
\right\rangle_\Omega.
\]

The geometry-recovery fraction \(F_K^{(m)}=1-R_K^{(m)}/V_K\) is reported only
after a geometry-signal gate. Let
\(S_K=\langle K_0,K_0\rangle_\Omega>0\). Individual geometry is materially
present only if the simultaneous lower bound for
\(V_K-\tau_{\mathrm{geometry}}S_K\) exceeds zero; it is equivalent to the
shared geometry only if the simultaneous upper bound is below zero; otherwise
the geometry status is unresolved. M3 predicts no participant-specific change
in \(K_0\) beyond the one uniform Task-A-estimated scalar, which \(P_0\)
removes. \(K\) is only a necessary isometric invariant on the
18-condition row span; equivalence does not prove a global isometry. M1 and M2
can also change voxel-weighted \(K\), so M4 must outperform them on this
endpoint rather than merely predict a nonzero deviation.

## Scientific margins and inference

Four scientific margins are frozen by human sign-off before synthetic
decision qualification and before the first
candidate-discriminating development Task-B score:

| Margin | Meaning |
| --- | --- |
| `tau_signal` | Minimum between-person signal relative to a frozen positive reliable-signal scale |
| `tau_remaining` | Maximum remaining fraction for a class to count as adequate |
| `tau_increment` | Minimum material recovery gain on the relevant dimensionless map or geometry scale |
| `tau_geometry` | Maximum individual condition-geometry variation compatible with the M3 equivalence claim |

Margin-free algebra and software checks may run first. Synthetic decision
qualification then validates operating characteristics using the already
signed margins; it cannot choose or widen them. The
positive map scale is the development-only crossvalidated source energy,

\[
S_B=\sum_c\alpha_c
\langle HG^{\mathrm{dev}}_{c,1},HG^{\mathrm{dev}}_{c,2}\rangle_V>0,
\]

frozen at panel lock. The geometry scale is
\(S_K=\langle K_0,K_0\rangle_\Omega>0\). The margin values, source-split
construction, and numerical tolerances are launch blockers, not quantities to
be selected from development wins.

Inference is made on linear contrasts, not unstable ratio confidence
intervals. Examples are

\[
C_{\mathrm{adequate}}^{(m)}
=R_B^{(m)}-\tau_{\mathrm{remaining}}V_B,
\]

\[
C_{\mathrm{increment}}^{(m:b)}
=R_B^{(b)}-R_B^{(m)}-\tau_{\mathrm{increment}}V_B.
\]

Adequacy requires a simultaneous one-sided upper confidence bound below zero.
A material increment requires a simultaneous lower bound above zero; evidence
of no material increment requires the upper bound below zero. Failure to
reject is neither adequacy nor equivalence.

Participant delete-one jackknife pseudovalues are used for the pairwise
estimands. Each model-class claim is an intersection-union of all required
components; Holm-inverted one-sided bounds control the four alternative class
claims. The exact contrast list, direction, family, and implementation are
frozen and simulation-tested before search. Any interval crossing a decision
boundary yields an unresolved component. With 12 audit participants, these
bounds may be wide; conditions and voxels never inflate \(n\).

## Branch signatures and adaptive depth

One valid trial is one completely frozen scientific model or falsifier
evaluated across every development participant-held-out fold, all 18
conditions, seven domains, both map halves, both geometry replicates, all
required endpoints, and all branch certificates. A fold, rank, radius,
regularization value, seed, metric replay, or scheduler job is not a trial.

The first 20 valid trials implement an exactly frozen coverage manifest: four
representatives from each of M1--M4 and four cross-panel controls. After
coverage, successors must cite scored parents, make a directional prediction,
change at most one scientific operator, and include a branch-native falsifier.

- **M1 successors:** granularity, hard versus probabilistic assignment,
  boundary width, and boundary-versus-interior recovery.
- **M2 successors:** regularity, displacement bound, anatomy coupling,
  split-half field stability, inverse consistency, and Jacobian behavior.
- **M3 successors:** locality, active-subspace dimension, canonical
  composition, split-half inverse consistency, and the exact operator
  certificate.
- **M4 successors:** stretch versus shear, rank and singular-value bounds,
  Task-A-half stability, geometry recovery, smoothness-matched nulls, and
  mode-by-mode ablation.

No branch is retired because one configuration loses or another branch has a
higher raw score. Retirement requires its coverage rows plus at least two
valid native post-coverage falsifiers. A mandatory branch that cannot produce
a certified representative blocks audit; it is not evidence against the
hypothesis.

## Synthetic qualification and controls

Before any candidate-discriminating development score, the complete decision
code is run on frozen synthetic worlds:

- group-only;
- parcel-only;
- smooth-warp-only;
- exact isometry;
- stable non-isometry;
- A/B-specific geometry; and
- reliability-only null.

The output is a model-class confusion matrix, not a candidate score. Across
100 frozen seeds per world, each stable identifiable world must receive its
correct class in at least 90 seeds, the null may promote at most five times,
and A/B-specific geometry may be mislabeled stable non-isometry at most five
times. Ambiguous synthetic regimes must abstain as prespecified; they cannot
be used to tune scientific margins.

Other mandatory controls are:

- Task-A split-half stability for every target-specific model;
- reliability-stratified Task-A condition-correspondence permutation;
- target-participant identity derangement;
- capacity- and output-smoothness-matched random local bases and warps;
- equal-Task-A-quantity and reliability subsampling;
- branch-specific component and mode ablations;
- participant, condition-domain, and development-source influence analyses;
- proof that audit Task B affects no mask, support, scaling, QC decision,
  threshold, prior, search proposal, or retry; and
- byte-identical audit predictions when every audit Task-B file is absent.

## Stages and lock

1. **Readiness and margin lock:** authenticate the release, coordinate route,
   four B partitions, condition table, exposure history, participant roles,
   physically separated handoffs, and human-signed scientific margins.
2. **Synthetic qualification:** after margin-free algebra/software checks,
   validate the full decision tree, certificates, error assumptions, and
   confusion matrix using the already locked margins.
3. **Branch coverage:** complete the 20-row frozen panel manifest.
4. **Adaptive development:** run ledger-linked successors and spend at least
   40% of post-coverage valid trials on falsification or ablation.
5. **Panel lock:** select exactly one certified M1, M2, M3, and M4 paired to
   that M3, plus M0a, M0b, and the negative controls. This composite panel is
   the single locked configuration; no cross-branch scalar winner is chosen.
6. **Audit calibration:** fit target-specific parameters from each audit
   participant's anatomy and Task A, with Task B unmounted; hash every
   prediction, certificate, and decision input.
7. **Audit once:** the trusted evaluator mounts Task B once, computes all
   branch statuses, seals the primary decision, and only then emits diagnostic
   tables.

## One-shot outcomes

The evaluator first resolves integrity and the three-way reliable-signal gate,
then computes every branch's adequacy and native signature before assigning
one label. A positive subtype requires a unique supported non-nested
explanation. M4 is supported only through its material paired increments over
M3; an M3 label requires an upper bound showing that increment is below the
materiality margin.

| Outcome | Required interpretation |
| --- | --- |
| `candidate_ready_parcel_sufficiency` | M1 is adequate and passes its boundary signature; neither non-nested M2 nor M3 is supported, and M4 lacks its required material increments |
| `candidate_ready_smooth_relocation` | M2 is adequate and passes warp stability, inverse, and Jacobian signatures; neither non-nested M1 nor M3 is supported, and M4 lacks its required material increments |
| `candidate_ready_shared_isometry` | M3 is adequate; individual condition geometry is equivalent within `tau_geometry`; M1 and M2 are unsupported; M4's incremental benefit is bounded below materiality |
| `candidate_ready_stable_nonisometry` | M4 is the unique supported class: it is adequate and stable, has material map and geometry increments over M1, M2, and paired M3, and no non-nested class remains supported |
| `closed_multiple_adequate_explanations` | Two or more non-nested explanations remain adequate and the signatures do not distinguish them |
| `closed_reliable_nontransportable` | Reliable individual variation exists, but no Task-A-derived class is adequate |
| `closed_no_resolvable_individual_signal` | The upper bound establishes that between-person Task-B signal is below `tau_signal` |
| `closed_unresolved` | Signal, adequacy, equivalence, multiplicity, or power remains indeterminate |

The first four map to canonical `candidate_ready`; the four `closed_*` labels
map to `closed_no_candidate`. Integrity, exposure, or policy violations map to
`technical_failure`. A non-isometric model that improves but remains
inadequate is not candidate-ready. If multiple non-nested classes are
adequate, a generic simplicity rule may not manufacture a unique mechanism.
If any competing branch needed for uniqueness is indeterminate, the outcome is
`closed_unresolved`, not a positive subtype.

## Post-decision diagnosis

Only after the primary decision is cryptographically sealed, the evaluator
reports the 13 non-rest shared Task-B conditions, rest separately, and a
non-rescuing within-session oracle. For each locked M1--M4 recipe, the oracle
replaces Task-A calibration with the target's Task-B runs `1--8`, refits only
the recipe's permitted target-specific parameters, and predicts the same
conditions in runs `9--16`; shared development parameters remain locked.

The rows below are representative patterns, not an exhaustive second decision
tree:

| A to B-only | A to shared B | B-half to B-half | Diagnostic reading |
| --- | --- | --- | --- |
| fail | fail | pass | More consistent with calibration/session shift |
| fail | pass | pass | Compatible with task-set-specific residual, but session interaction remains possible |
| fail | fail | fail | B reliability or model support is insufficient |
| pass | pass | pass | Stable cross-task organization is supported |

Every unlisted or discordant pattern is reported as diagnostically
indeterminate.

These diagnostics cannot select a class, change a margin, rescue the primary
outcome, or authorize a rerun.

## Claim boundary

An M1 result supports only that shared parcel response profiles plus
Task-A-personalized membership are sufficient at the registered resolution;
it does not prove a literal or unique parcel ontology. An M2 result supports
only that one stable smooth relocation field is sufficient under the frozen
Jacobian and stability bounds; it does not identify a developmental or
anatomical cause for that relocation.

An M3 result supports only:

> Within this public single-site MDTB release, a Task-A-estimated certified
> isometric correspondence recovered a prespecified fraction of reliable
> between-person variation in Task-B-only cerebellar maps; individual
> condition geometry was within the frozen equivalence margin, and the paired
> bounded non-isometric challenger supplied no material increment.

An M4 result supports only:

> Within this release, Task A contained stable participant-specific
> information that predicted reproducible changes in both Task-B-only maps and
> condition-level geometry beyond the tested parcel, smooth-warp, and
> isometric explanations.

Neither result establishes a universal cognitive coordinate system, literal
parcel ontology, task dependence, causal or behavioral relevance, cortical
generalization, cross-site generalization, or external replication. In
particular, non-isometric means stable geometry-changing variation under this
model, not task-specific variation. A population claim over possible training
samples would require resampling and refitting the development cohort, and a
flagship confirmation requires a new, prospectively sealed compatible
dataset.
