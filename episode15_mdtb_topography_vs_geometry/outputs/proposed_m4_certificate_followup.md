# ADOPTED October 13, 2026 — model-specific M4 uniqueness certificate

The scientist adopted this complete proposal with "OK agree with your next
steps" after the explicit certificate/retry recommendation. GOAL.md and
`scientist_amendment_20261013_m4_certificate` record the authority. The draft
wording below is preserved as the prospectively reviewed specification; its
inactive labels describe its pre-adoption state. Numerical limits are unchanged.

This is a source-derived proposal with an isolated candidate implementation
whose seven numerical and three integration checks have passed in the actual
EP15 runtime; it is not an adopted amendment,
empirical diagnosis or authorization to resume. The October 12 numerical stop
remains operative. Independent review found no bug in the existing sufficient
curvature certificate. Its failure establishes **unverified uniqueness**, not
multiple optima or a negative scientific result. The public stop does not
identify the failing variant, block or bound; no such identity is assumed here.

## Exact change proposed

Keep the existing certificate and permit a prospectively fixed, alternative
**full-domain lower bound for the actual restricted M4 modes**. Acceptance
still requires a strictly positive certified curvature margin in every one of
the eight blocks, exceeding the unchanged `1e-10` times that block's weighted
source-plus-target energy. A failed or unresolved bound still stops the whole
panel. A local Hessian, a sampled grid, optimizer success or a small gradient
cannot substitute for this requirement.

Keep the original evidence status: these are qualified floating-point numerical
acceptance certificates, not interval-arithmetic proofs or a declaration that
the returned optimizer is exactly global. Reuse existing qualified source,
moment, mode and optimizer calculations; no universal input re-certification
is proposed.

Change only the mathematical certificate, not the objective, optimizer or
model. Retain the sole frozen 188-location support; all twelve participants,
29 Task-A and eighteen Task-B-only conditions; all sixteen variants and every
control/draw; original partitions, weights and source exclusions; paired M3
rotation and common gain; all modes, ranks, parameters and spectral bounds.
Keep zero-start SLSQP, its `1e-12` tolerance and 500-iteration limit, numerical
and gap tolerances `1e-10`, tangent-rank tolerance `1e-8`, and the actual
operator/locality checks. No support search, mode reduction, alternate
initialization, Task-B calibration or partial scientific panel is proposed.

## Why a different sufficient certificate is possible

For one block, use exactly the existing weighted Task-A matrices `A,Y`, gain
`g`, and definitions

\[
 P=g^2A^TA,\qquad D=g(A^TY+Y^TA)/2.
\]

Its nonconstant loss is

\[
 f(C)=\operatorname{tr}(P e^{2C})-2\operatorname{tr}(D e^C).
\]

The complement loss and target squared norm are constant. This follows from
the existing objective; it neither changes spatial weighting nor refits any
quantity. Block-diagonal `C` makes the eight losses additive even when the
full source cross-product has off-block entries.

**Anisotropic diagonal variant.** With `C=diag(c)` and the unchanged global
trace-zero Helmert coordinates `c=H theta`, the exact loss is
`sum_i(P_ii exp(2 c_i) - 2 D_ii exp(c_i))`. Off-diagonal entries of `P,D` do
not enter this loss, although they contribute to the existing general bound.
The exact coordinate Hessian is

\[
 h_i(c_i)=4P_{ii}e^{2c_i}-2D_{ii}e^{c_i}.
\]

For `|c_i| <= rho`, its minimum `m_i` is the minimum of the quadratic
`4 P_ii z^2 - 2 D_ii z` on `[exp(-rho),exp(rho)]`, attained at the clipped
vertex `D_ii/(4 P_ii)`. Positive `P_ii` follows from the unchanged full-rank
source admission; an implementation must refuse a violated precondition.
`H.T diag(m_i) H` is a global Hessian lower bound on the actual coefficients.
The conservative proposed acceptance rule is still blockwise:
`min(m_2b,m_2b+1) > 1e-10 E_b`, with the **same** existing block energy `E_b`.
The global trace constraint only restricts directions. This proposal does not
use cross-block compensation to rescue a block failing that margin.

**Local diagonal variant.** For one Frobenius-normalized mode,
`C=diag(t,-t)`, `t=theta/sqrt(2)` and `|t| <= rho`. Its exact coefficient
Hessian is

\[
 h_\theta(t)=2P_{11}e^{2t}+2P_{22}e^{-2t}
               -D_{11}e^t-D_{22}e^{-t}.
\]

Its global minimum is attained at an interval endpoint or a stationary point.
Writing `z=exp(t)`, stationary points are the positive roots, within the same
interval, of

\[
 4P_{11}z^4-D_{11}z^3+D_{22}z-4P_{22}=0.
\]

A qualified calculation must cover **all** such roots (including repeated
roots) and endpoints, or use an equivalent complete interval bound. An ordinary
polynomial-root list without completeness and numerical-error handling is
insufficient. Compare the resulting lower bound with the original block
margin; there is no new fit or optimizer.

**Both local shear variants.** With the fixed two normalized modes,

\[
 C=\frac1{\sqrt2}\begin{pmatrix}\theta_1&\theta_2\\
                         \theta_2&-\theta_1\end{pmatrix},\quad
 r=\|\theta\|_2/\sqrt2\leq\rho,\quad
 e^C=\cosh(r)I+\frac{\sinh(r)}r C.
\]

The expression is analytic at zero; use its continuous power-series limit,
not division by zero or a changed neighborhood. It yields the actual 2-by-2
coefficient Hessian. For orthonormal symmetric coefficient modes and
`||C||op <= rho`, a conservative Hessian Lipschitz constant is

\[
 L=8e^{2\rho}\operatorname{tr}(P)+2e^\rho\|D\|_*.
\]

Indeed, the ordered third-Fréchet derivative formula for the exponential
bounds its operator norm by `exp(rho)` times the product of direction norms:
the permutation count cancels the simplex volume. Scaling `C` by two gives
the factor eight. Trace/nuclear-norm duality and `P` positive semidefinite give
the displayed bound. Frobenius-orthonormal modes bound each matrix direction's
operator norm by its coefficient norm.

For each cell of a finite cover of the **entire** coefficient disk, a
center-Hessian lower bound `ell` therefore gives `ell - L delta` throughout
that cell, where `delta` bounds every covered point's distance to its center.
Every retained cell must exceed the unchanged block margin. Centers and
connecting segments must respect the spectral domain used for `L`. One simple
safe construction covers the enclosing square and uses the enlarged radius
`sqrt(2)*rho` in `L`; only cells proved disjoint from the disk may be discarded.
This enlarges the bounding calculation, not the fitted parameter domain.

## Completed numerical qualification and fixed finite limits

The affected mathematical routines have passed the seven numerical checks below;
the candidate is **not adopted or empirically validated**.
The prospective [array-only module](tools/proposed_m4_certificate.py) and
[seven focused checks](tools/test_proposed_m4_certificate.py) are now prepared.
Independent source review found no remaining concrete defect. All seven
methods passed without skips in1.808seconds on compute step47584692.0;
the [operational evidence](runs/observed_beta_development/certificate_qualification001/step_stdout.txt)
records total3.734687seconds and57806848bytes peakRSS. No empirical certificate
success is known. The existing
production callback does not import or invoke this candidate.

The [unused full-panel adapter](tools/proposed_development_point_panel.py)
and [three affected integration checks](tools/test_proposed_development_point_panel.py)
are now complete and independently source-reviewed. All ten checks passed
without skips on step47584692.1 under the original EP15 encoding_model
Python3.11.15: unittest1.184seconds, total Python9.579483seconds,
peakRSS66854912bytes. The seven mathematical methods were repeated only
because the earlier step used EP18's different runtime. The
[actual-runtime evidence](runs/observed_beta_development/certificate_qualification002/step_stdout.txt)
closes that gap. These checks use analytical or wiring fixtures, not participant
data; no empirical certificate success is established.

The candidate preserves every original passing block bound. Only a failed
block invokes an alternative. Local diagonal bounds cover complete intervals;
shear bounds use analytic center Hessians with series tails and full-domain
variation bounds over a deterministic dyadic cover. Stored mode normalization
is retained. Prospective limits are16383 visited cells per block, depth12 and
30seconds total per certificate call, plus the caller's resource/deadline
check. An unresolved cell or reached cap refuses. The depth-first queue is
bounded; these limits are not measured runtime or a new compute grant.
Nonpositive anisotropic curvature refuses before Gram-lower scaling, avoiding
an invalid negative-bound multiplication. Scalar rounding expansions retain
the original numerical assurance level and make no libm interval-proof claim.

The checks cover the new bound, complete root/domain coverage and floating-point
roundoff handling at the existing numerical tolerances using fixed analytical
fixtures and an independent70-digit matrix-power derivative reference. Reuse already
qualified source moments and stored modes. New coordinate or direction-norm
calculations must handle their stored normalization consistently; an idealized
normalization must not silently change the actual model. This does not require
interval re-certification of every existing input. If a future implementation
claims an interval-arithmetic proof, it must separately establish the enclosures
needed for that stronger claim; this proposal does not make it. A finite grid
without a valid variation bound and qualified numerical handling is insufficient.

The global curvature lower bound supplied to the existing solution certificate
must be valid in its unchanged coefficient coordinates. Its existing exact
support-function gap and `sqrt(2 gap / mu)` distance bound can then be retained:
the domains are convex, and global positive curvature gives a unique minimizer.
Strict convexity in `exp(C)` alone does **not** suffice because the determinant
constraint makes the transformed feasible set nonconvex. Neither this argument
nor any proposed certificate makes the floating-point optimizer exactly global.

The completed numerical methods check agreement with the existing objective/Fréchet
derivatives, normalized coordinates, complete domain/endpoints including `r=0`,
new-bound roundoff and refusal on unresolved or nonpositive bounds. The isolated
integration checks verify unchanged optimizer/check/helper routing, identical
188-location support through ordinary and all calibration panels, and no
partial scoring or fallback after numerical refusal. Reuse all ten passing
methods without repeating them. The candidate's
subdivision/time caps above are fixed prospectively; coordinator compute and
memory admission remain required. Hitting a cap
means unverified and stopped, never permission to relax the criterion. No
expanded recovery campaign is proposed, and no empirical timing or successful
certificate is currently known. No additional compute grant is estimated here.

## Exact activation and fresh cost attempt, conditional on adoption

After an explicit scientific adoption, replace only the local panel import in
`run_development_point_panel.run_released_development` with the qualified
adapter; retain the existing reader, CLI, resource guards and summary path.
Update the candidate's inactive authority labels to reference the adopted
amendment without changing its numerical calculations. No live hook has changed.

Use the existing released `development_release/attempt001/view/point_panel_manifest.json`
and sole `observed_beta_development/finite_support001/selection.npz`.
Do not rebuild the mask, replay unrelated checks, overwrite cost002, or reopen
private scores. A fresh `cost003` retains the full twelve-person,
eighteen-Task-B-condition, sixteen-model point panel,24 source fits,
48 calibration panels and every original control/draw. Its publication mode
remains `withheld_cost_measurement` and scope `conditional_observed_beta`.

The previously authorized ceiling is1CPU/7GiB/24hours, with6GiB processRSS,
32MiB durable output,256MiB scratch, one numerical thread and no GPU. Whether
the full panel can finish within that ceiling is unknown. Preserve original
search anchor1791845107 and deadline1792277107; a run's deadline is the minimum
of actual start+selected wall limit−60seconds, actual end−60seconds, and that
original global deadline. Preserve all consumed charges. No automatic retry,
partial panel, new participant exposure or deadline reset is authorized here.

## Authority and consequences

The operative [GOAL](../GOAL.md) says a failed numerical certificate stops this
route; `SEARCH_POLICY.yaml` retains that explicit stop and requires unchanged
numerical certificates for full-panel fitting. Replacing or supplementing the
selected certificate after that stop therefore requires an **explicit adopted
certificate-follow-up amendment**, followed by implementation qualification and
coordinator resource admission. This document selects none of those actions.
Standing permission for more compute does not itself adopt a new certificate.

Any later execution must preserve `cost002`, the existing mask and charges,
use a fresh attempt, and keep the original failed certificate available in its
history. Every required M4 fit/control must pass the complete adopted rule;
success for one family does not complete the sixteen-variant panel. If the
alternative cannot certify the required cases, retain the stop. All existing
observed-map-only claim limits, procedural separation and closed audit remain.

Source basis: `outputs/tools/paired_stretch_variants.py` lines 26–76,
79–117, 154–194; `outputs/tools/fit_m4_task_a.py` lines 84–133 and 163–216.
Stop record: `outputs/experiment_log.md`, October 12, 17:04 and 17:06 Pacific.
The mathematical derivation used only these sources/contracts and public stop
statements. Subsequent qualification used the explicitly recorded compute
steps above. No empirical matrices, private fits or outcomes were inspected.
