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

#### October 9: direct-surface spatial contact bridge, implemented and checked

[surface_to_contact.py](tools/surface_to_contact.py) now supplies the missing
spatial bridge from the selected constructed surface factors to 4,096
once-averaged contact factors. The five registered mathematical pad maps use
150-µm lattice centres `x=(column+0.5)*150`, `y=(row+0.5)*150` in the 9.6-mm
square. The caller must explicitly provide whether x or y is the short pad
axis: the available catalogue says to retain published orientation but does
not establish its registration to these simulation axes. Returned metadata
marks both that registration and physical eligibility as unverified. Centre
origin and axis naming are prospective mathematical conventions, not recovered
NeuroCam layout or Rat5 facts.

Focal, surprise and interaction Gaussian modes use exact normalized rectangular
integrals from the stored centres/FWHM; wave quadratures use their stored
phase, origin and cyclic wavevector with the two directional sinc factors.
Analytic parameters must reproduce the stored unscaled spatial samples within
fixed float64 roundoff tolerances (`rtol=2e-13`, `atol=2e-14`); missing or
inconsistent parameters fail without fitting them from the raster. Spatial
mode peaks are not renormalized: all existing event/whole-world amplitude
scales remain in the unchanged raw temporal drivers.

For grid-defined broad and mixture modes, select bilinear interpolation of the
25-µm target samples, extended constantly across each outer half-cell to the
aperture edge. Sparse separable weights integrate that declared interpolant
exactly over each rectangle. This is a numerical surface model, not an
additional recovered transfer function. No nearest-sample pad approximation,
radial-only attenuation, reference subtraction or second pad average is added.
All methods preserve spatial common mode and the original ideal target.

`build_contact_factors` returns a 4,096-by-rank matrix and an owned unchanged
copy of the raw drivers. `raw_voltage_block(start,stop,...)` reconstructs only
the requested existing driver sample indices in a bounded block and converts
microvolts to volts once. It selects no temporal interpolation or hold law,
does not apply observation nuisances, contact-gap changes, contact impedance,
electronics or DAQ, and does not qualify a physical geometry or reference.
The bridge currently supports zero-gap direct-surface development worlds only.

Six [focused checks](tools/test_surface_to_contact.py) cover catalogue geometry,
explicit orientation, common mode and edge integration, independent Gaussian/
wave quadrature, mixed spatial modes, raw-amplitude/unit preservation and
refusal of missing metadata or exceeded bounds. All six passed in **0.032
seconds** in allocation47086577; the closed execution log records the evidence.

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

### October 9: conditional active buffer/filter integration

The selected [conditional R3 model](electrical_model_and_probes.md#october-9-conditional-active-conditioning-and-loaded-recovery)
adds explicit input loading and 128 persistent states for all 64 source lines.
[analog_conditioning.py](tools/analog_conditioning.py) binds the buffer/filter
parameters; [conditioned_array_solver.py](tools/conditioned_array_solver.py)
advances them jointly with every pad, source, contact and ADC hold. The optional
path in [cpe_array.py](tools/cpe_array.py) routes neural mux inputs to filter
nodes, preserving all physical ADC identities and external marker paths.
Original source observations remain diagnostics. No row, aperture or storage
boundary resets the added state, and input loading is applied once.

The controlled drive clips an internal target, not a physical capacitor
voltage. Loaded nodes may exceed those rails while recovering. This is a
prospective effective mechanism with explicit constants, not a measured
amplifier, hard output clamp or identified NI circuit. Selecting it does not
accept an additional reference model. The conditional physical-noise path
below supplies declared stochastic forcing; a complete noise inventory and
calibrated parameter values remain separate requirements. Saved N1–N4
observation overlays do not provide them.

The nonlinear path uses bounded implicit midpoint, Newton/GMRES and step
doubling. Full-state endpoints, aperture means and work quadratures must meet
caller tolerances before commit. These are consistency estimates, not
certified stiff-mode or continuous-time error bounds. Explicit callbacks run
inside the solver; finite iteration, operator and numeric workspace limits
do not replace process memory or walltime limits. The passive exponential
path remains selected when conditioning is absent.

Eager and streaming runners carry controlled voltage-source terminal work
and its drive-branch heat separately from field/external-source work and
stored energy. Terminal work may be negative during backdrive and is not
total supply consumption. The solver's partially restored Norton dissipation
does not include every physical voltage-port heat correction. Complete device
power remains unspecified. Source-review and execution evidence belongs in
the [implementation log](experiment_log.md#2026-10-09--model_specification-conditional-active-stage-implemented-checks-unrun).

### October 9: physical noise through the retained R2/R3 array

[spectral_noise.py](tools/spectral_noise.py) represents a caller's finite-band
cross spectrum using sparse complex factors and a named, frozen Gaussian
quadrature basis. Paired designs reuse the same frequency grid and draws.
[physical_noise_ports.py](tools/physical_noise_ports.py) binds current/voltage
amplitudes to actual retained-node and contact-star ports; it preserves the
induced correlations and field-source work correction from exact contact
junction elimination. Amplifier input noise enters before gain/target clipping.
No physical parameter, temperature, bandwidth or source independence is inferred.

[circuit_noise_provider.py](tools/circuit_noise_provider.py) connects that
trajectory to the assembled array and explicit per-port A/V tolerances. Both
midpoint planners refine against field and noise bounds. Eager and streaming
executors prevalidate original physical amplitudes, replay the same held
sample through aperture/storage seams, and retain only one live sample. A
noise layout fixed on an array requires an explicit sample on every advance;
omitting it cannot silently remove noise. The optional simulation factory
binds the trajectory before hold planning, with source evaluation/work counts
reported separately from solver products. The existing noise-free path remains
available as an explicit conditional mechanism.

The representation is a finite Gaussian spectral expansion, not ideal white
noise or proof of ergodicity. Analytic forcing bounds exclude spectral
discretization, waveform arithmetic and circuit-output error. A zero entry
means zero only in the declared conditional model. The physical layout owns
star branches/access noise; separately retained charge-transfer edges require
their own current port or an explicit omitted-component reason. Named external
DAQ current ports now own actual `(ADC, external input)` switch branches.
They add current at the hold node, signed noise work across its true terminals,
and the corresponding external ideal-source work correction. Their immutable
inventory includes track/off conductances; acquisition preflight refuses a
nonzero resistor-noise source on a later disconnected path before advancing
any state. Existing noise coordinates keep their order; external currents are
appended. This does not supply a switching-dependent thermal spectrum.

The separate [R1 reduction](tools/r1_noise.py) specifies original held current
ports, their conjugate one-state coefficients and voltage offsets. The scalar
[source-line pole](tools/switched_line.py) propagates the resulting constant
current exactly, retaining original, equivalent nodal and offset work before
committing. No RNG, temperature or spectral defaults enter this component.
It is not yet connected to a full R1 acquisition engine: 64 persistent poles
and their separate contact/neighbour/DAQ response remain to be integrated.
See the [external and R1 port equations](electrical_model_and_probes.md#october-9-external-daq-noise-and-the-reduced-r1-port).

Complete source inventory, spectra and numerical settings still
need selection; none of this establishes a calibrated reference or candidate
advantage. See the [equations and limits](electrical_model_and_probes.md#october-9-replayable-finite-band-physical-noise)
and [source implementation record](experiment_log.md#2026-10-09--model_specification-physical-noise-connected-source-review-complete).

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

## October 9: continuous raw time and bounded acquisition storage

[raw_world_time.py](tools/raw_world_time.py) implements the analytic extension
in the [world recipe](development_world_recipe.md#october-9-continuous-raw-time-extension-for-acquisition).
`RawWorldTime` checks the saved native raw record in bounded 512-sample blocks
against its fixed parameters before accepting the extension. The numerical
replay allowance is `5e-12 * (max(1, mode_native_peak) + abs(native_value))`
in microvolts; this is a floating-point consistency tolerance, not calibration
acceptance. Queries stay in `[0,16)`, preserve event silence/overlap and exact
M4 switches, and keep N1/N2 line signals and N3/N4 availability separate.
The breakpoint API exposes support/taper and switch boundaries; time16 is
boundary-only. No hold convention, real DAQ clock or electrical parameters are
inferred. `bind_contact_factors` validates raw identity once and owns a read-only
spatial snapshot; its reusable `volts(times)` avoids a full native-record scan
on each physical interval and performs one microvolt-to-volt conversion.

[array_acquisition.py](tools/array_acquisition.py) adds `ProviderInterval`
with an immutable field key and sparse boundary-charge entries. A caller-frozen
deterministic provider supplies each 64-by-64 voltage field. Complete schedule,
binding, aperture and field validation precedes any array advancement. Fields
are validated in one bounded-memory traversal and regenerated one physical
interval at a time. Packed 32-byte digests detect numeric changes between the
two traversals; these replace the eager route's field snapshots, not a generic
provenance check. A drift or provider failure permanently closes that attempt
before the current interval advances. Aperture splits reuse one field;
boundary charge is applied once and all electrical/readout state persists.

For 256000 physical intervals, digest payload is 8,192,000 bytes instead of
8,388,608,000 bytes of retained float64 fields. Sparse charges and the shared
zero vector replace per-interval dense charges. This is not a peak-RSS estimate:
descriptors, conversion/output plans, provider factors and solver workspaces
still need their explicit bounds. Existing eager callers keep their behavior.

Independent source reviews are complete, including correction of integer
provider keys larger than `2**53`. All six raw-time and seven acquisition
checks **passed in allocation47090250**, together with the six new midpoint
checks and changed ASTRA validation; see the
[compute evidence](experiment_log.md#2026-10-09-0142-pacific--model_specification-all-nineteen-raw-input-checks-and-astra-passed).
All84 completed worlds remain unchanged. These implementations do not establish
a calibrated reference, completed electrical integration or candidate result.

### October 9: saved-world simulation orchestration and incremental publication

[development_simulation.py](tools/development_simulation.py) exposes three
bounded Python interfaces. `load_saved_development_world` selects one completed
condition/seed group from the existing generator HDF5 container, checks numeric
shapes/dtypes/bytes before loading, and preserves its development identity and
seed group. Its scalar variable-length metadata byte cap is checked after that
single string is read. `run_development_simulation` accepts the existing world,
explicit pad orientation, full circuit/initial-state inputs, physical actions,
raw conversions, returned samples, observations, and caller numerical/resource
limits. `save_simulation_result` writes one exclusive HDF5 record with bounded
serialized payload. It does not overwrite an attempt or regenerate its world.

The runner constructs assembler-owned contact access, then preserves every one
of the 4096 pad and 64 source states, all contact memory and supplied DAQ hold
states across row, field and aperture changes. Raw support/M4 boundaries and
every declared aperture endpoint enter the same midpoint plan. Original action
charges remain once-only. Existing circuit results supply streaming physical
stored-energy changes and source/charge work; readout retains actual declared
conversion, return, pixel and bit counts. Complete device power remains unknown.
The remaining global propagation-product cap can tighten a per-interval work
cap without loosening numerical tolerances. Elapsed limits are checked between
operations; neither work counts nor retained numeric-byte caps certify peak
RSS, factorization cost or hard wall time. Failed solves contribute elapsed
time; their unreturned work count stays unknown with its explicit upper bound.
An incomplete attempt retains terminal state and diagnostics without replay.

`AcquisitionReadout.new_observations_available_by` now consumes a monotone
cursor over publication plans sorted once by availability and identity. Each
successful fresh batch is delivered once; a failing batch leaves the cursor
unchanged. `ArrayAcquisition.advance_next` uses this path. Existing cumulative
and history queries do not consume it, and finalization still reports the
complete declared outputs. This removes repeated construction of all prior
observations at every interval without changing visibility, counts or physics.

The new [four orchestration checks](tools/test_development_simulation.py) and
the changed readout/array suites are **unrun**. The latter contain five readout
checks, including two new incremental-publication fixtures, and seven array
checks. Passing raw-time/hold evidence remains valid; prior array/readout passes
do not cover the current edits. See the
[current handoff](experiment_log.md#2026-10-09-0202-pacific--model_specification-saved-world-orchestration-and-incremental-publication-implemented).

This earlier integration retains eager conversion, returned-sample and
observation inventories; the streaming extension below addresses that design.
The paper's 6.4-million/s
digitized-stream proxy would give 102.4 million conversion records over sixteen
seconds, but it does not establish actual physical ADC work. Full-scale support
requires compact ordered chunks carrying analog state, unfinished apertures,
digital averaging/publication state and cumulative costs; rerunning independent
chunks with fresh arrays is prohibited. Physical parameter/initialization and
DAQ recipes are explicit caller dependencies, not defaults selected here.
A prospective development recipe suffices for conditional simulation;
authenticated measurement context is separately needed for calibration claims.
This earlier eager route still refuses N1–N4; their selected observation-side
placement is now implemented in the streaming route below.
The 84 field inputs remain complete; no complete 84-world electrical panel,
accepted reference, calibrated co-design comparison or candidate score exists.

### October 9: persistent streaming forward chain

[streaming_development_simulation.py](tools/streaming_development_simulation.py)
accepts the retained `DevelopmentWorld` loaded by the bounded loader above,
explicit geometry/orientation, circuit and initial states, physical actions,
input bindings, a replayable `ReadoutWindow` factory, acquisition/publication
horizons, solver/readout/array limits, midpoint tolerance/caps, output limits and
a settings reference. It supplies no physical or DAQ defaults. It compiles the
bounded midpoint hold once without collecting a full ADC aperture plan, then
[streaming_input_plan.py](tools/streaming_input_plan.py) clips its immutable
intervals across windows without resampling held fields or repeating charge.
Later aperture splits retain the same held input and analytic input-error bound.
The compact full hold tuple remains in memory.

[streaming_array_acquisition.py](tools/streaming_array_acquisition.py) uses one
`NeurocamCPEArray` throughout. Complete bounded metadata/provider preflight
checks the entire schedule before analog mutation; packed window digests bind
replay, and current-window field digests detect numeric provider drift before
use. Shared quantizer tables are hashed once by numeric value per window instead
of expanded into every record. Windows partition all physical time, including
unsampled gaps. Continuation fragments preserve event identity, row, held field,
reference and DAQ selection, and carry no repeated charge. Unique events and
every solver split/work count remain separate. All 4096 pads, source/contact
states and supplied DAQ memory persist. Explicit caps cover windows, live
fragments, numeric/digest storage, updates and cumulative propagation products;
these are not a measured peak-RSS or hard solve-time guarantee.

[streaming_acquisition_readout.py](tools/streaming_acquisition_readout.py) keeps
only declared live conversions, digital dependencies and pending publications.
Canonical increasing IDs and explicit last-use windows support bounded eviction
without full identity histories. Cross-window apertures and delayed observations
survive until their dependencies are complete. Final publication can extend
beyond acquisition without advancing physical state. Quantization, invalid
contributors and signed averaging retain existing readout semantics. Metadata
preflight exercises the same coverage/dependency/eviction rules without analog
values. Earlier eager/cumulative APIs retain their behavior.

[streaming_simulation_output.py](tools/streaming_simulation_output.py) streams
raw, observation/dependency and interval evidence into one exclusive HDF5 file
with bounded batches and payload/record/identifier caps. It keeps no complete
output tuple. Final reports preserve initial/terminal states and cumulative
energy/source-work and conversion accounting, distinguishing computation,
returned callbacks and persisted records. Failure is terminal without replay;
best-effort interrupt closeout preserves completed work before rethrowing the
interrupt. Unknown interrupted solver work remains explicit. Output payload
caps exclude HDF5 overhead and do not replace external storage limits.

[observation_nuisance.py](tools/observation_nuisance.py) implements the selected
N1–N4 post-readout overlay from the same saved raw-time snapshot. Each pixel
needs a physical-site index and explicit acquisition point inside the union of
its contributing apertures and no later than publication. Preflight checks
metadata only. Replay digests bind this mapping; delayed outputs retain it
until publication. Line values convert microvolts to volts once, while missing
pixels remain records with absent values. Raw records, physical work, bit costs,
clipping, truth and non-pixel outputs are unchanged. Paired mask lookup reuses
the cached 500-Hz edges without Cartesian expansion. The settings reference
must retain the binding-generating convention; output records retain their
original site ID/support/times rather than extra binding columns. This does
not model analog pickup, ADC saturation from that pickup or electrical faults.

Independent source review closed for the streaming modules and nuisance
adapter. **17 streaming checks** (readout 5, executor 4, input plan 2, writer 3,
orchestration 3), **3 adapter**, **6 changed raw-time** and the previous **16
affected checks** remain **unrun: 42 total**. Changed ASTRA validation is pending.
The source path from saved world to streamed forward result is implemented,
but full-panel runtime and memory are unmeasured. All 84 saved inputs remain
intact. An explicit prospective electrical/initial-state/DAQ recipe still must
be selected for conditional engineering simulation; authentic calibration is a
separate claim. N1–N4 require complete explicit observation bindings. Future
truth/splits remain simulator-private, and no calibrated reference, full-panel
electrical result, acquisition-policy release or candidate score is implied.

## 6. Next component choices, not an execution claim

### October 9: selected numerical midpoint input approximation

The continuous raw waveform will connect to the constant-input electrical
executor through a numerical midpoint hold. This is an integration choice,
not a physical DAQ hold, clock or calibration assumption. The caller must
provide `epsilon_input_volts`, a maximum interval count and a maximum work
count before execution. No tolerance is selected from candidate outcomes.

For contact coefficients `A`, raw temporal drivers `d_r`, interval `[a,b)`
and midpoint `m`, a derivative bound `L_r` gives the input approximation bound
`1e-6 * max(m-a,b-m) * max_i sum_r abs(A_ir) * L_r` volts. Use outward-rounded
nonnegative arithmetic. A sinusoid has bound `2*pi*f*abs(scale)`; M4 within
one fixed coefficient interval uses `abs(scale)*(2*pi*f*1.25 + 0.25*pi)`.
The selected finite biphasic event and taper admit `abs(scale)/sigma` on
their support, zero outside; overlapping events add their bounds.

Start from raw support/taper/switch boundaries and caller-supplied physical
row/DAQ/aperture boundaries. Accept an interval only if its bound meets the
input tolerance; otherwise bisect deterministically. Stop unresolved at an
explicit work or interval cap without loosening tolerance. Preserve M4 jumps
as exact boundaries. Only the first numerical subdivision receives the
original physical boundary charge; all subdivisions share persistent
electrical/readout state. Materialize midpoint fields through the compact
provider interface without retaining a dense field plan.

Future truth, event boundaries and numerical subdivisions belong only to the
forward simulator. They must not become information supplied to an adaptive
acquisition policy. Observation nuisances remain separate. This certificate
covers the analytic input-hold approximation, not floating-point waveform
evaluation, complete circuit-output error, or physical model correctness.
The implementation in `tools/raw_input_hold.py` and supporting derivative
enclosures passed independent code/mathematical review and six focused compute
checks. No production electrical integration or candidate comparison has run.

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
