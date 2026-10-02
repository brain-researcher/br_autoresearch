# EP20 components — when must geometry and timing be designed together?

Design refinement, 2026-09-30. The question is whether a field's structure and
a device's measured response can predict which legal acquisition/reconstruction
recipe will recover cortical voltage best. The spatial/temporal four-quadrant
rule is a hypothesis to test, not an assumption built into the simulator.

This note specifies the component interfaces and already-required challenge
comparisons. It keeps the current hardware scope, C0–C4 comparisons, D0–D3
family definitions, primary endpoint, margins and resource limits. The later
[starting catalogue](parameter_candidate_catalogue.md) selects field controls,
proposed pad sizes and timing recipes; it does not resolve physical eligibility
or source-use decisions. No candidate or independent final model was run or
opened.

## 1. One target, three components

```text
source or field generator
    -> common cortical-surface voltage truth
    -> contact sensitivity and timestamped multiplexed measurement
    -> acquired values, masks, timestamps and available operator information
    -> causal reconstruction on the same cortical grid
```

| Component | Scientific variation | Held common for a paired comparison |
| --- | --- | --- |
| Field and source-to-surface model | Spectrum, orientation, phase, focality, source dependence and volume conduction | The same realization and ideal surface truth for every design |
| Measurement operator | Legal pad geometry and scan actions under uncertain coupled electronics | The same device/process/nuisance world and resource budget |
| Reconstruction | Conventional or one registered optimized family, fitted for its actual operator | The same training allowance, evaluation grid, reference, mask and reporting deadline |

The field generator describes the world; it must not encode the design that
should win. The design rule predicts a winner from its registered absolute
inputs; it does not give the scheduler or inverse access to hidden truth.

## 2. Cortical-field and source-to-surface components

### Surface truth is not a contact measurement

For a source-mediated generator, write

`u(r,t) = G_surface(theta) z(t)`

for the common ideal cortical-surface voltage, and

`u_contact(r,t) = G_contact(theta, contact_state) z(t)`

for voltage at actual contact positions/heights. Position is in micrometres,
time in seconds, and voltage in microvolts. The target stays `u` on the common
dense cortical grid, not whichever voltage a candidate's contacts happen to
observe. A contact gap therefore changes the observation, not the target.

Both maps use the same declared reference convention: `u` and `u_contact`
are already referenced voltages. Reference drift enters once as a measurement
nuisance. A reference sensitivity changes the convention consistently for
truth and observations in every arm, not only the candidate or its score.

Keep the required homogeneous/layered and heterogeneous/anisotropic forward
models. Conductivity, source distance and cortical curvature belong to the
shared world. Assign contact-gap effects, reference and pad sensitivity one
owner in the forward chain so none is applied twice. If an extended contact
model already produces electrode voltages, do not add another pad average.

A directly constructed surface field is also an implementation option for
development, but identify it as such. It is not a recovered latent-source model.
Define any nonlinear source interaction in the generator; a linear volume-
conduction operator does not itself create nonlinear collisions.

### Concrete implementations of the existing field families

| Existing family | Component construction and variations |
| --- | --- |
| Broad correlated field | Spatial spectral/covariance basis with temporal dynamics; vary spectrum, actual correlation structure, anisotropy, amplitude and common mode |
| Focal transient | Spatial envelope times an event waveform; vary size, location, duration, rate, amplitude and spectral envelope |
| Translating/expanding wave | Phase or arrival-time field plus envelope; vary direction, speed, wavelength, curvature, onset and boundary behavior |
| Multiscale mixture | Broad and local components with controlled energy fractions, overlap, dependence and nonstationarity |
| Interacting sources | Explicit component locations, separation, synchrony and phase; state any collision rule |
| Silent-region surprise | A quiet pre-event location followed by an event not privileged by the acquisition schedule |
| Nuisances | Separate line/reference/motion-like contributions from gaps, dropout, drift and saturation; state where each enters the chain |

Starting ranges now have empirical context or explicit conservative-design
labels in the catalogue and the compact source-to-surface choices below.
Absolute contact/electronics constants still need support. Do not select ranges from candidate scores,
remove difficult draws, or regenerate worlds until a preferred design succeeds.
Existing signal families and the common random-world pairing remain unchanged.

### Source-to-surface starting ranges — compact selection

| Quantity | Starting values | Interpretation |
| --- | --- | --- |
| Cortical conductivity | 0.2, 0.4, 0.6 S/m; CSF 1.79 S/m where represented | Selected sensitivity bracket around a literature-model setting |
| Source depth below the ideal surface | 0.1, 0.5, 1.5 mm | Shallow/intermediate/deeper source scenarios |
| Contact gap above the surface | 0, 50, 200 µm | Contact/lift-off scenarios; changes observations, not ideal surface truth |
| Conductivity tensor max/min eigenvalue ratio | 1, 2; 4 as a strong stress case | Fix geometric-mean conductivity; tensor orientation is declared, not inferred from field anisotropy |

These are selected exploration values, **not NeuroCam measurements or a
physiological distribution**. Rogers et al. use cortical conductivity 0.4 S/m
and CSF 1.79 S/m and demonstrate source-depth/separation effects on sensitivity.
[Rogers et al. (2020), source-to-surface context](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00763/full).
Start with `(0.4 S/m, 0.5 mm, 0 µm, ratio 1)` and one-factor perturbations:
nine world configurations, not an 81-combination sweep or extra hardware trials.
Recompute actual surface lambda/tau; all arms share each world. Existing
curvature/reference and heterogeneous/layered challenges remain open.

### What lambda and tau actually summarize

Keep the exact registered definitions:

```text
lambda_k = 1 / (2 k90)       [micrometres]
tau_k    = 1 / (2 f90)       [seconds]
```

`k90` is the smallest radial spatial frequency in cycles per micrometre
enclosing at least 90% of the common-mask 1–100 Hz field power; `f90` is the
smallest temporal frequency in Hz enclosing at least 90% of demeaned temporal
power. Unit conversions must not substitute angular frequency for cyclic
frequency. Use the same estimator, mask, demeaning
and boundary conventions in development and independent implementations.

These are **90%-power spatial and temporal scales**, not conventional
autocorrelation decay lengths or times. For a narrowband sinusoid they describe
half a wavelength and half a period. Actual correlation structure remains a
separate generator characteristic. Retain common-mode variation; do not
silently spatially demean it away to obtain a more favorable spatial cutoff.
Zero cutoff implies an infinite scale; zero eligible power is undefined, not
a reason to insert an epsilon. Register these cases with the estimators.

The two inputs come from truth or the already-permitted separate known-input
pilot, not candidate reconstructions. A pilot is an alternative measurement
source for this rule, not a newly required pilot experiment.

## 3. Contact, electronics and scan operator

### Spatial sensitivity, once

A useful leading approximation for pad `i` of legal geometry `g` is

`v_gi(t) = integral w_gi(r) u_contact(r,t) dr`,

with `integral w_gi(r) dr = 1`; no second reference subtraction is applied.
Uniform area averaging is an explicit model
approximation, not a measured transfer function for every candidate. Under that
approximation, a rectangular pad's response to a spatial Fourier component is
the product of its two directional sinc factors, so radial frequency alone
need not determine attenuation.

Electrode/contact properties can affect volume-conduction predictions; see
[Vermaas et al., 2020](https://doi.org/10.1088/1741-2552/abb11d).
If such effects are represented in an accepted coupled-contact variant, its
operator replaces the corresponding approximation rather than filtering the
field twice. Ordinary ECoG already reflects its own contacts and reference:
use it for the specified plausibility checks, not as dense unfiltered truth
for ranking unbuilt geometries.

### Causal analog state followed by actual conversions

The observation chain is

```text
contact voltages
    -> interface/loading + switching/settling + coupling/noise/analog saturation
    -> raw ADC sampling at actual acquisition times
    -> ADC quantization/clipping
    -> registered within-dwell aggregation and publication
```

The analog operator may use a state-space, causal impulse-response or another
accepted parameterization. Its state persists across row switches. Resetting
it independently at every row would remove the history-dependent settling
being tested. Analog clipping and its recovery belong to that persistent
operator; ADC clipping is a separate readout effect. Put each modeled noise
source at its physical stage rather than adding the same noise both before and
after aggregation. Register initialization
and gap/reset behavior once for all arms.

An acquisition action specifies the legal gate row/group, active source bank,
dwell, raw sample times, retained aggregation and publication time. Keep the
registered dwell multipliers `[1, 2, 4]` of the 62.5-microsecond base; no shorter
dwell, arbitrary pixel addressing, extra ADC family or on-array memory is added.
The published multiplexing architecture and row-clock description are in
[Xie et al., 2025](https://doi.org/10.1016/j.scib.2025.11.030).

A rolling scan is not a simultaneous image. Each inverse receives actual
sample timing. An adaptive scheduler can use only observations already
published when it acts, not the unobserved current field, future samples or
the condition's truth-derived lambda/tau. Fixed family representatives do not
switch hardware or operating recipes after seeing a condition label.

Keep the primary target at 1–100 Hz, but allow relevant out-of-band field and
nuisance content through the forward chain to expose aliasing and saturation.
Primary-band evaluation is not permission to pre-remove those effects before
measurement. Evaluation filtering, warm-up, edge masks and reporting latency
must be common across arms and must not supply future data to the inverse.

### Resource accounting is attached to the event sequence

For an interval of duration `T`, distinguish

```text
raw_conversion_rate = all actual ADC conversions / T
effective_output_rate = retained pixel observations / T
```

Discarded settling samples still count as conversions. Marker/reference work
also counts. Output bitrate includes the permitted sample precision and
required timing/index information. Higher precision from averaging spends
conversions/time; it is not a free variable-resolution ADC.

The source-supported raw-sample/aggregation timing still requires resolution;
do not round the nominal oversampling ratio into a fictitious integer count
per dwell. Compute latency, scan skew, switching, active readout channels and
modeled resource use from actual events. Preserve all existing power, thermal,
routing, area and yield dimensions; missing costs remain unknown, not zero.
The catalogue also distinguishes the existing source-stream work proxies from
physical internal ADC conversions, which the requested 100-kS/s rate alone
does not determine.

### What remains uncertain in device characterization

Keep noise, impedance, settling and crosstalk as independent probe outputs for
each frozen design, device condition and accepted reference model. They must
precede reconstruction outcomes, with common waveforms, durations and units.
Input-referred RMS is not output RMS without a declared transfer calibration;
crosstalk is an amplitude ratio, not a power ratio or an unidentified dB cost.
A response not settling within a probe supplies a censored bound, not zero.
The [electrical mechanisms/probes](electrical_model_and_probes.md) now select
finite waveforms, common-panel aggregation, a same-victim input-equivalent
crosstalk convention and conditional-mean step histories. These panel inputs
are not full-array maxima and do not reduce the primary scoring mask.

The article's impedance anchor is in **1–100 kHz**, while the rule needs
**1–100 Hz**. It cannot fill that table by direct substitution.
[NeuroCam article, main-text characterization](https://doi.org/10.1016/j.scib.2025.11.030).
Aggregate noise, gain and crosstalk also do not identify pad-area laws, noise
correlations or
switched-scan memory. Keep source-supported alternative laws rather than
inventing resistance, capacitance, loading or thermal constants. The accepted
reference set remains predeclared and scored separately, never averaged into
one apparently certain optimum.

## 4. Reconstruction and the two comparison namespaces

### Common inverse interface; open implementations

Every inverse receives acquired values available by the reporting deadline,
their masks/timestamps, known legal geometry and scan actions, and independently
calibrated or frozen operator information. It does not receive hidden per-world
noise realizations, latent sources, evaluation truth or an oracle field prior.
Knowledge of a nominal forward model is not knowledge of every world draw.

| Registered family | Concrete implementation space |
| --- | --- |
| Conventional | Training-fitted fixed-form Gaussian filtering/kriging with a geometry- and irregular-timing-compatible nominal measurement adapter; no backward smoothing |
| Physics-informed linear | Causal-window quadratic inverse using the calibrated linear operator or frozen linearization; searchable spatial/temporal regularizers, training-fitted noise weighting and propagated uncertainty |
| Structured state-space | `u = B z + r`: broad low-rank state plus local residual dynamics; predict over actual elapsed time and update from observed innovations |
| Compact nonlinear | Capacity-capped causal temporal/recurrent estimator using the same values, masks, timing and permitted operator descriptors; uncertainty heads count toward capacity |

An acquisition record contains value, observed-validity flag, site, contributing
aperture interval, assigned timestamp and completed availability time. Zero-filled
missing entries remain absent. The nominal operator bundle contains legal
geometry/actions and frozen calibration, not realized hidden device parameters,
analog state or truth-derived rule inputs.

Split whole independent field×device×nuisance worlds before making windows;
share split identities across arms. Fit normalization, priors and parameters on
training worlds only. C1 validation selects family/capacity/history on the whole
published frontier; held-out conditions stay separate. Freeze those choices,
then train each operator separately under the matched allowance.

Each returns common-grid voltage predictions and uncertainty. Training losses
must relate to the primary field error; uncertainty/likelihood weighting and
all tuning use development data and are frozen before independent outcomes.
Truth mean, evaluation mask and primary R² normalization stay common. Training
normalization does not use held-out data.

Keep rank, locality, regularization, history representation and compact
architecture choices open within these families. Use these small starting
allowances, not an exhaustive tuning grid:

| Choice | Starting allowance |
| --- | --- |
| Capacity | Initially ≤1M learned parameters; retain the 5M planning ceiling for matched follow-ups; count state/computation separately, with no artificial parameter padding |
| History | Initial C1 configurations: 0.5 s and 2 s; 0.1 s is a matched follow-up/ablation, not a third initial tuning configuration |
| Fitting | Three shared seeds/world replications; gradient-fitted models ≤10,000 updates per operator/seed; other families keep an appropriate fixed solver and matched data exposure/stopping rule |
| Selection | At most two initial joint configurations per optimized family; choose on the complete C1 frontier only, then freeze capacity/history and fit each operator separately |

For target time `t` and common publication deadline `d`, freeze their relation
before scoring. Strictly past-only inputs have complete contributing apertures
inside `[t-H,t]` and are published by `d`; no post-target sample is admitted
merely because reporting is delayed. Initially implement the bound by replaying
that window from a fixed training-derived prior at `t-H`, advancing through
gaps without backward smoothing. Recurrent/filter state, covariance and caches
cannot carry pre-window observations or adaptive-policy state. The **physical
analog state is not reset** by this inverse-history rule. Charge replay and
state storage to actual computation; incremental implementations remain open
if they preserve the same bound. Numerical latency is not resolved here.
Within the selected C1/C3/C4 family, match capacity, solver/optimizer, data,
actual fitting allowance and seeds. Outcome-blind runtime calibration can
reduce a common allowance before scoring; it cannot give candidates extra
fitting. All follow-ups stay inside existing trial/resources. These are design
choices, not evidence that the budgets suffice or the inverses converge.

Complete the C0/C1 published-setting frontier before candidate-hardware results
choose anything. Select the optimized family from C1 only, and train each
hardware operator separately. Within the selected optimized family, C1/C3/C4
have matched capacity, optimizer, data, steps, seeds and selection attempts;
actual training/inference cost also counts.
C0/C2 receive genuinely compatible conventional inverses, not deliberately
weak interpolation that ignores scan skew. No favorable C3 weights are copied
to another hardware arm.

### C comparisons and D contrasts answer different questions

| Comparison | Meaning |
| --- | --- |
| C3 versus the full C0/C1 frontier | Does the joint candidate beat all eligible published-hardware alternatives? |
| C3 versus matched C2 | Does optimized reconstruction help on the exact same candidate geometry **and scan**? |
| C3 versus C4 | Does the search add value beyond simple resource-matched acquisition rules? |
| D2 minus D0; D3 minus D1 | Geometry benefit under the baseline and time-focused recipes respectively |
| D1 minus D0; D3 minus D2 | Benefit of the bundled time-focused scan/software recipe under each geometry |

D2 retains the D0 scan/conventional recipe. It is therefore generally **not**
D3's matched C2: that control retains D3's joint geometry and scan while using
conventional reconstruction. Both comparisons are needed.

D1 is sourced from C1, which currently contains published operating settings.
Nominate its time-focused representative from that frontier; do not quietly
insert a novel schedule into C1. A broader D1 source would require a separate
amendment. A time-focused reduced mode can change coverage as well as revisit
rate; the common dense-grid mask must not shrink to its observed pixels.

For accepted model `m` and condition `k`, let `U_mdk` remain mean primary R².
The descriptive factorial interaction is

`I_mk = U_m,D3,k - U_m,D2,k - U_m,D1,k + U_m,D0,k`.

It summarizes geometry × bundled acquisition/reconstruction non-additivity.
It is not pure hardware/software causation, and it neither creates a promotion
threshold nor replaces the C-arm margins.

## 5. S/T separability: a measurable hypothesis and its rival

The current map stays `00 -> D0`, `01 -> D1`, `10 -> D2`, `11 -> D3`.
It assumes the registered spectral scales and device summaries retain enough
information about relative utility, and that noise/impedance versus
settling/crosstalk costs can be assigned locally to S versus T.

The max-of-two pad-cost and scan-cost definitions already inspect both
factorial contexts. That is useful, but does not prove locality: changing pads
can change loading/settling, and changing the scan can change noise. Scalar
RMS noise and maximum crosstalk can also omit correlation and history structure.

### A concrete mechanism not summarized by marginal scales

As a mathematical illustration, take a proposed travelling field

`u(x,y,t) = A cos[2 pi (k_x x + k_y y - f t)]`.

On a local row scan with observation time `t = t0 + alpha y`, the measured
phase has row coefficient `k_y - f alpha`. Waves travelling along different
directions can therefore interact differently with the **same** scan despite
equal radial `sqrt(k_x^2 + k_y^2)` and temporal `f`, and hence equal lambda/tau
in the ideal narrowband construction. This is an algebraic illustration, not
an observed NeuroCam result or a complete periodic scan model.

Use the already-required direction, curvature, mixture, coupled-electronics
and pad-assignment challenges to test this rival. Useful matched conditions
hold radial spatial/temporal power, amplitude and device draw fixed while
varying wave orientation, space–time phase dependence or anisotropy. Check
matching from truth before scoring and report finite-aperture discrepancies;
do not tune the worlds to obtain a rank reversal.

For the same frozen designs/device condition, such pairs also retain the
registered characterization table. If their four-arm utilities prefer
different families, the current summaries may be insufficient. Direction and
phase are **challenge labels**, not new inputs secretly added to the rule.
These conditions use the shared challenge budget, not an extra search budget.

### What would actually change the scientific conclusion?

- Different material unique winners across accepted reference models invoke
  the existing rank-reversal rule: identify the discriminating device response
  before nominating a design.
- An S/T-mapped family more than 0.005 below the frozen predicted-utility
  maximum invokes the existing pre-outcome inconsistency/abstention rule.
- Independent regret must retain its simultaneous upper bound of at most
  0.005 in every accepted model/S/T cell, with existing cell support and
  field-family requirements. A violation leaves the transferable rule
  unsupported, even if one device wins on average.
- A large descriptive interaction alone is not a failure threshold. A
  non-additive but accurately predictive S/T rule can still pass.

Adding orientation, covariance, phase or a more flexible predictor could be a
future development amendment. It is not a post-final-outcome repair of this
rule. Keep candidate success, transferable-rule success and model uncertainty
as separate reportable outcomes.

## 6. Next component choices, not an execution claim

The starting catalogue supplies concrete field controls, proposed discrete pad
maps and scan sequences. The electrical note now specifies six starting
contact/persistent-state templates and common probes, not accepted models or
identified constants. Compact source-to-surface ranges and inverse allowances
are now selected above; inverse inputs, world splits and bounded-state replay
are operationally defined. [Anchor roles](../DATASETS.md) and the electrical
note's minimal shared-parameter fit are also specified. Next are usable
measurement contexts and supported numerical circuit combinations,
physical pad/via exclusions, actual ADC/resource accounting and numerical
instantiation of the models. The matched direction/phase controls instantiate the
existing challenge tests; the catalogue does not provide guessed device
constants or certify the proposed geometry.

The 64 × 64 lattice, pitch/area/topology/line limits, dwell choices, 1–100 Hz
endpoint, C/D namespaces, 16-arm coverage, 32–64 valid-trial budget, promotion
and regret margins, and single independent final comparison remain unchanged.
This is a component-design milestone, not source preparation, simulation,
reference qualification, fabrication or an empirical result.
