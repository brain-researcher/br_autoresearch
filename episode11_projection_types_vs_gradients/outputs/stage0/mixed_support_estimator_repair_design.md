# EP11 next synthetic diagnostic: estimator attribution and full-family calibration

This is a synthetic-only design note. It does not authorize construction or
inspection of a real regional projection outcome, change the fixed 12/8/16
role-count contract, or qualify outcome-bearing search.

## Why the next run should target the estimator

The 32-replicate nested diagnostic does not support another role-count search.
For the strong fixture, moving all 15 available reserve units into development
changed fitted `H - C` from -0.0367 (D12/A8) to -0.0434 (D27/A8) and group
accuracy from 0.584 to 0.581. Replacing discovered development labels with the
true synthetic labels raised `H - C` only to 0.0389 and 0.0441, respectively,
while the projected-oracle proxy was about 0.117. Thus discovery is not the
only failure. Moving the reserve into audit reduced the mean one-sided
half-width from 0.139 to 0.076, but still missed 0.05. The geometry frontier
permits no more than 20 audit units under both depth conventions, so
reallocation alone cannot repair the observed precision.

The smallest useful next run should therefore separate four questions:

1. Does the fitted procedure discover the planted groups?
2. Given the correct development labels, does it estimate transportable
   projection signatures?
3. Given the correct signatures, does a development-fitted smooth gate recover
   the position-dependent prevalence used by the generator?
4. What is the planted effect after projecting onto the complete implemented
   `C` density family, rather than onto the current four-parameter,
   generator-direction proxy?

## Prerequisite: make the synthetic data-generating law subset-invariant

The generator used for the completed 32-replicate result standardized position
after applying `indices` in both `continuous_truth_features` and
`synthetic_groups`. Consequently, the same cell had a different continuous
feature and group probability when calibration was generated on the assigned
20 units than when the design diagnostic was generated on all 36 units. The
reported calibration and recovery therefore used different data-generating
laws. The working implementation now contains the minimal full-layout repair
described below, but no completed result should be reinterpreted retroactively;
the repaired diagnostic must be rerun.

The minimal repair is to compute and freeze the generator covariates once on
the complete 36-unit metadata layout, then subset those fixed arrays. In
particular:

- freeze the standardized continuous generator coordinate for every cell;
- freeze the standardized gating coordinate and therefore
  `p(group=1 | position)` for every cell; and
- for fixtures with unit-random synthetic covariates, key the draw to fixture,
  replicate, and unit rather than to the order or size of `indices`.

An executable self-test must choose several nontrivial 20-unit subsets and
assert, cell for cell, that continuous features and group probabilities equal
the corresponding rows from a full-36 construction. The test should cover
`reusable_groups`, `groups_plus_continuum`, and the unit-random continuous
fixture. Exact equality is expected for probabilities and deterministic
features; floating-point equality at `atol=1e-14, rtol=0` is sufficient for
derived arrays. Calibration and recovery must stop if this test fails.

## Common run design

- Use only synthetic outcomes and the current outcome-blind metadata layout.
- Use the same fixed provisional D12/A8 core in every arm. It remains a
  metadata-only diagnostic allocation, not a geometry-qualified role lock.
- Generate one full 36-unit synthetic dataset per fixture and replicate, then
  subset it for every arm. This preserves common random numbers.
- Run the zero-group fixture and the strong reusable-group fixture. Do not add
  more biological fixtures until this attribution run resolves.
- Average cell scores within unit and then weight units equally.
- Never condition an audit score on the audit truth label. Every audit group
  label is integrated under its development-frozen gate. Audit truth is used
  only to calculate diagnostic accuracy after scoring.
- Run batches of 32 replicates, stopping at the first batch boundary where the
  Monte Carlo standard error of every paired arm contrast below is at most
  0.005 nats/neuron, with a cap of 256 replicates. Failure to reach that
  precision is an unresolved diagnostic, not a pass.

## Panel A: population calibration on the full implemented families

For each candidate group amplitude, fit `C_pop` by maximizing expected
equal-unit log predictive density on repeated synthetic draws over the fixed
provisional A8 metadata support and evaluate it on independent repeated draws
over those same A8 metadata rows. This is a population-family ceiling
conditional on the fixed A8 metadata distribution, not a D12-to-A8 transport
test; finite-D12 estimation and transport remain Panel C's responsibility.
`C_pop` must be the complete currently implemented `C` density family: all
adjustment and position coefficients, the connected latent support/value map,
and noise scale are optimized. The current smoke grammar's seven registered
standard-normal quadrature nodes and weights remain fixed; free latent weights
would silently define a larger family. The fit may not hold the generator base
fixed, restrict itself to the known planted support/value directions, or use a
four-parameter surrogate. Multiple starts and an independent verification pool
are required.

Compare `C_pop` with `H_truth`, the exact label-integrated generator density
using the cell's true position-dependent group probabilities. Interpolate the
amplitude for

`E_equal-unit[log H_truth - log C_pop] = 0.100 nats/neuron`.

Also fit the complete implemented `H_pop` family, with smooth gating, to the
same conditional-A8 population objective. This separates failure of the
represented `H` family or its optimizer from finite-D12 estimation.

Panel-A arms are therefore:

| Arm | Density | Purpose |
| --- | --- | --- |
| `CAL_C_FULL` | population-optimal complete implemented `C` | full continuous projection |
| `CAL_H_FULL` | population-optimal complete implemented `H` with smooth gating | family/optimizer ceiling |
| `CAL_H_TRUTH` | exact generator `H`, labels integrated | calibration reference |

The working Panel-A implementation now uses the equal-unit weighted SVD of the
fixed A8 `P` design and a fixed Helmert basis for the value coordinates.  The
raw 39-term A8 design has rank 26, and the compositional value likelihood is
invariant to one all-coordinate shift in every value block.  Removing those
exact null directions changes the fitted C/H parameter counts from 4,593/4,709
to 3,109/3,224 and makes the weighted estimable design condition number one.
Executable tests require the raw and identified C scores and responsibilities
to agree on A8, preserve exact `C`-inside-`H` nesting, and preserve the latent-
sign and group-label density orbits.

The coefficient boxes are now explicitly numerical bounds on these identified
coordinates.  They are not claimed to be the exact quotient of the former raw-
coefficient box; a fitted solution that uses a coefficient or sigma bound fails
the numerical-boundary check.  The likelihood no longer hard-clips support or
gate predictors: stable conditional-nonempty and logistic formulas are used,
and the largest absolute fitted predictors are reported.  Reintroducing hard
predictor clipping would invalidate this calibration.  This identified A8
parameterization does not solve Panel C transport: D12 has a different
estimable subspace, and some D12-null directions are nonzero on A8. Panel C
must freeze an outcome-blind extension or regularized transport rule before it
is implemented.

### Finite-SAA diagnostic and population limit

The completed 32-draw fit is an optimizer benchmark, not a population
calibration. It contains 2,432 A8 synthetic observations for 3,109 identified
`C` parameters and overfits its fitting pool: despite KKT convergence, its
fresh-draw `H_truth - C` gap is about 1.13 nats/neuron at zero amplitude.
Because `C` exactly contains the zero-amplitude truth, this is finite-pool
estimation error rather than evidence that the continuous family misses the
truth.

Exact parameter anchors reproduce `C` truth at zero amplitude and `H` truth at
every tested amplitude. These validate the likelihood, parameterization, and
scorer; they do not establish recovery of a fitted population optimum. The
positive-amplitude `C_pop` projection must instead be approximated with
deterministic, group-stratified nested SAA sizes `R = 128, 512, 2048, 8192`
and untouched independent verification streams. A terminal result requires:
exact-anchor error at most `1e-10`; no optimizer cap, nonfinite value, or active
coefficient/sigma bound; best-start verification difference below `0.002`;
terminal-start difference below `0.005`; fresh-score change below `0.005` after
the final SAA doubling; no polishing improvement above `0.005`; and final
verification MCSE at most `0.005`. If these conditions are not reached, Panel
A remains unresolved rather than calibrated.

The first `R=128→512` run at diagnostic amplitude `1.25` converged at every
scheduled start without an active numerical bound, but its independent
`R=128` to `R=512` score change was `0.09333 ± 0.00458` nats/neuron, above the
`0.005` stability rule. A four-rung run is therefore required to determine the
positive-amplitude `C_pop` limit. Passing that run would not yet select the
amplitude that realizes the `0.100`-nat fixture and would not substitute for
fitted `H_pop` or fitted zero-amplitude recovery.

Panel A passes only if all of the following hold on independent verification:

- at zero amplitude, both `H_truth - C_pop` and `H_pop - C_pop` have absolute
  mean at most 0.005 nats/neuron;
- at the selected strong amplitude, the mean `H_truth - C_pop` is within 0.01
  of 0.100 and its Monte Carlo SE is at most 0.005;
- `H_truth - H_pop` has a one-sided 95% upper bound below 0.01; and
- the selected solution is stable across starts: the best two converged
  solutions differ by less than 0.002 nats/neuron on the independent pool.

This is a calibration of the complete **implemented smoke family**. If the
registered production `C/H` grammar is later enlarged, calibration must be
repeated for that family before qualification.

## Panel B: conditional oracle-rescue ladder for the group module

Use the true group-free conditional support logits, value mean, and noise scale
as a common backbone in this panel. This deliberately removes fitted-`C`
quality from the group-module diagnostic. All estimated quantities use
development cells only, and every audit score integrates the group.

| Arm | Development labels | Signatures | Audit gate | Isolates |
| --- | --- | --- | --- | --- |
| `MOD_DISC` | residual-PCA/two-means | current conditional estimator | equal-unit intercept only | current discovery module |
| `MOD_LABEL` | true synthetic labels | current conditional estimator | same intercept-only gate | discovery rescue |
| `MOD_SIGNATURE` | true labels | true planted support and value signatures | same intercept-only gate | signature-estimation rescue |
| `MOD_GATE_FIT` | true labels | true planted signatures | smooth gate fitted on development labels and metadata | gating estimator |
| `MOD_GATE_TRUTH` | true labels | true planted signatures | true generator probability for each audit cell | exact module ceiling |

The intercept-only and smooth fitted gates must use equal-unit contributions.
The smooth gate uses only the registered soma-position/depth basis and
development labels; no audit outcome or audit prevalence is available. The
true-label and true-signature arms are diagnostic ceilings and can never be
promoted as scientific models.

For each arm define oracle regret

`R_arm = E_equal-unit[log H_truth - log H_arm]`.

The following paired differences provide an order-explicit attribution:

- discovery loss: `R_MOD_DISC - R_MOD_LABEL`;
- signature-estimation loss, conditional on correct labels:
  `R_MOD_LABEL - R_MOD_SIGNATURE`;
- need for position gating: `R_MOD_SIGNATURE - R_MOD_GATE_TRUTH`; and
- smooth-gate estimation loss: `R_MOD_GATE_FIT - R_MOD_GATE_TRUTH`.

The attribution is intentionally conditional in this order; it should not be
reported as an order-invariant variance decomposition. A component is a
material bottleneck if the paired one-sided 95% lower bound for its rescue
gain exceeds 0.005 nats/neuron. It is provisionally adequate only if the
one-sided 95% upper bound for its remaining oracle regret is below 0.005.

`MOD_GATE_TRUTH` must reproduce `CAL_H_TRUTH` cellwise at tolerance `1e-10`
when given the same truth backbone, signatures, probabilities, and noise.
Failure is a scoring or state-ordering defect and stops the run.

## Panel C: repaired finite-D12 integration

Only after Panels A and B pass their structural checks, refit the complete
finite-D12 comparison with:

- equal-unit fitting objectives;
- a smooth development-only gate evaluated from audit metadata;
- marginal-likelihood fitting of the group signatures rather than the current
  point-latent residual regression; and
- joint optimization (or a converged alternating optimization) of each
  family's own parameters. `C` and `H` receive identical continuous capacity,
  hyperparameter choices, folds, starts, and optimizer budget, but `H` must not
  be forced to retain coefficient estimates obtained under `C` after `C` has
  absorbed group signal.

Compare only two finite-sample arms: the unchanged current estimator and the
repaired estimator. The repaired estimator advances to the broader registered
synthetic suite only when, on the independently calibrated 0.100-nat fixture:

- the across-replicate mean fitted `H - C` is at least 0.075 nats/neuron and
  its one-sided 95% Monte Carlo lower bound exceeds 0.05;
- the zero-group mean fitted `H - C` is not positive by more than 0.005 and no
  zero-group replicate satisfies the positive predictive-gain rule; and
- optimization failures, nonfinite scores, and C/H budget mismatches are all
  zero.

These are estimator-diagnostic gates, not registered recovery qualification.
The existing full null/stress-fixture, calibration, trajectory-adequacy,
random-effect, missingness, reuse, and deterministic-replay requirements still
apply afterward. In particular, this run must not use the A8 audit interval
precision gate as an estimator pass: the completed design sensitivity already
shows that the current unit-level variance cannot meet it.

### Minimal Panel-C implementation contract

Panel C should live in a new synthetic-only runner; it must not rewrite the
historical smoke runner.  For each training cell in unit (a), use weight
`1 / (number of development units * cells in a)` and optimize the penalized
marginal mixed-support likelihood.  Report unpenalized development-validation
and audit scores.  The implementation keeps the fixed seven-node connected
quadrature and two-group grammar, but fits these two families independently:

- `C` integrates its connected latent states after jointly optimizing its own
  adjustment/position coefficients, continuous support and value maps, and
  noise scale; and
- `H` integrates the Cartesian product of group and continuous states while
  jointly optimizing its own copies of those shared-capacity parameters plus
  a group support signature, group value signature, and development-only
  smooth gate.

Thus `H` contains an exact embedding of any fitted `C` at zero group
signature, but no coefficient estimate from the fitted `C` is frozen inside
`H`.  Both families receive the same whole-unit folds, shared-parameter
penalties and bounds, deterministic start slots, convergence tolerance, and
optimizer budget.  Each selects its own best converged likelihood; `H - C`
must never be used to select a start.  A nonzero-signature start is required
because the nested zero-signature solution may be stationary.

The smallest acceptable optimizer is joint analytic-gradient L-BFGS-B.  An
alternating implementation is acceptable only if every cycle increases the
penalized marginal likelihood and the shared coefficients are re-estimated
inside `H`.  Hard development labels and point estimates of the continuous
latent coordinate are not permitted in the repaired likelihood.

Before a one-replicate benchmark, executable tests must establish:

- setting the group signature to zero and copying a `C` parameter vector into
  the nested `H` embedding gives cellwise-identical scores for any gate and
  zero gate gradient;
- the best converged `H` training objective is no worse than that embedded
  `C` solution;
- latent responsibilities sum to one and the vectorized score matches an
  explicit two-by-seven state enumeration;
- analytic gradients for every block agree with finite differences on a tiny
  fixture;
- unit-weight sums are equal, and cell, unit, and label permutations preserve
  the objective after canonical relabeling;
- audit outcomes cannot affect fitted parameters or audit gate probabilities;
  and
- non-convergence, a hit optimizer cap, nonfinite values, noise-boundary
  collapse, or unmatched scheduled budgets invalidate the pair.

The raw model has roughly 4,600 coordinates, many of which are exactly
unidentified.  Panel A has 3,109 C and 3,224 H identified coordinates on A8;
the exact Panel-C count depends on its prospectively frozen D12-to-A8 extension
rule.  Strong frozen regularization, an identified transport parameterization,
and analytic gradients are therefore prerequisites.  After the exact tests,
run a single deterministic Slurm benchmark before attempting the 32-replicate
Panel-C batch.

## Concrete defects or mismatches in the current smoke implementation

1. The completed result came from versions of `synthetic_groups` and
   `continuous_truth_features` that standardized after subsetting, so its
   calibration and recovery did not share one data-generating law. The working
   code now constructs these quantities on the full layout before subsetting;
   its executable invariance test must pass and the diagnostic must be rerun
   before the repair can support a result.
2. The grouped generator uses position-dependent probabilities, but `D/H`
   store one constant pooled-cell `group_weights` vector. `score_model` has no
   row-specific gating path.
3. Group discovery, signature regression, and prevalence are pooled-cell
   weighted; units with more cells receive more influence than the intended
   equal-unit estimand.
4. The continuous latent coordinate is point-estimated from training outcomes
   before signature regression, whereas prediction integrates that coordinate.
   The fitted and scored objectives are therefore different.
5. `C` is fitted first and frozen into `H`. A fitted `C` can absorb group
   structure, after which even true-label signature regression cannot
   reallocate the shared backbone. Nested family fairness requires matched
   capacity and optimizer effort, not identical coefficient estimates from the
   smaller family.
6. The current effect calibration holds the true generator base fixed and
   optimizes four coefficients along known planted directions. It is not a
   projection onto the complete implemented `C` family, and its independent
   verification SE (0.016) is too large for a 0.005 target.
7. The value-signature regression subtracts an uncentered coordinate mean from
   a face-centered CLR response. For changing active sets, the offset should be
   projected/centered on each observed face or the signatures should be fit
   directly by the common face likelihood.
8. Support parameters are fit as unconditioned coordinatewise Bernoulli
   regressions, while generation and scoring condition that the active set is
   nonempty. The production estimator must optimize the same conditional
   support likelihood that it scores, even if the numerical effect is small in
   this fixture.

Panel B is complete but unresolved at its Monte Carlo cap. The immediate next
action is the Panel-A positive-amplitude `C` population-SAA ladder and fitted-H
recovery. Panel C and the registered recovery suite remain downstream; there
is no scientific value in another allocation, outer-CV, or large recovery run
before those steps.
