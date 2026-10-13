# Direct-surface development worlds: the retained B1–N4 panel

Prospective selection, October 9, 2026, before generating these worlds or
examining any candidate reconstruction. This selects the remaining numerical
definitions of the existing 28-condition catalogue under the scientist's
adopted simulation scope and delegated implementation judgment. It does not
select another study, supply historical measurement facts, qualify a reference
model, or implement the separately required source-to-surface/final models.

This recipe is a **development-only constructed surface-field model**. Every
hardware arm receives the same world identity and factors. The 28 conditions
remain within-trial worlds, not 28 new hardware trials. Existing comparison,
budget, claim, calibration and final-access requirements remain in force.

## Common grid, time and representation

The common target is a 9.6-by-9.6-mm rectangle. Its 384-by-384 sample centres
are `(index + 0.5) * 25` micrometres on each axis. The full rectangle is the
evaluation mask; no design or missing observation can shrink it. Coordinates
are unreferenced ideal surface voltage under one fixed mathematical zero
potential convention; retain spatial common mode and perform no spatial
demeaning. This convention is not an authenticated physical electrode reference.

Time samples are `n / 500` seconds, `n=0,...,7999`: a 16-second finite record.
The smallest selected 150-micrometre event has six samples across nominal
spatial FWHM; the 20-ms envelope has ten time samples across nominal FWHM.
The low-rate 0.5-Hz conditions contain several complete events, and the
two-second surprise prelude leaves time for the complete long envelope and
subsequent events. The field raster does **not** represent pad integration:
future contact sensitivity must integrate the declared spatial field separately,
with its own convergence/error support. No nearest-grid-point pad average is
implicitly selected. Temporal discretization and finite aperture remain
declared approximations, not recovered NeuroCam bandwidth or geometry facts.

Keep the exact factorization

`raw_truth[t,y,x] = sum_r spatial_modes[r,y,x] * raw_drivers[r,t]`.

Store float64 factors, coordinates and per-world reports, not a dense
time-by-grid cube. A caller can reconstruct bounded spatial/time blocks using
the existing field-combination primitive. This factorization changes no target
sample: all component interference and event overlap remain in the sum.

The evaluation transform is the finite-record orthonormal temporal DFT,
retaining inclusive 1–100-Hz bins and zeroing all others, with no taper or
padding. It acts on each temporal driver, so primary-band truth has the same
spatial factors. DFT periodic extension is an **evaluation convention**; raw
events retain their finite support, including exact pre-event zeros. Evaluation
ringing is retained and cannot become scheduler input. Use the existing
90%-power estimator conventions: lambda uses primary-band raw-truth power;
tau uses **all positive frequencies of demeaned raw truth**, not the filtered
drivers. A factorized implementation must retain all Gram cross terms and
agree with the existing dense estimator on the same finite field.

Initial development replicate seeds are **2001, 2002 and 2003**. Stable named
streams use NumPy `SeedSequence` with explicit integer stream identifiers and
PCG64; Python's salted hash and global RNG are not used. Replicate identity,
condition identity and streams are stored. Shared innovation draws are paired
where stated below, and all designs reuse the resulting world unchanged.
These initial worlds are development preparation; no held-out condition,
independent-final role, training/validation release or candidate fitting is
created by generation. Later role assignment must be fixed before its relevant
outcomes and split whole worlds, as the existing contract requires. In addition,
all 28 condition realizations with the same replicate seed carry the same
`independence_group=development_replicate_SEED` and must stay in one role.
Paired innovation/phase/event streams make these condition realizations
correlated. The initial panel has three independent seed groups, **not 84
independent replicates**; neither condition count nor grid samples may inflate
the independent sample size. More training/evaluation support, if required,
must be selected prospectively within the retained episode design and budget.

## B1–B4: finite-grid correlated fields

Retain the registered `(ell micrometres, carrier Hz, RMS microvolts)` tuples:
B1 `(4800,3,50)`, B2 `(1200,11,50)`, B3 `(300,37,50)`, and B4 `(600,80,10)`.
B1–B3 are isotropic; B4 has major/minor axis ratio 4 with major axis along y.

Before amplitude normalization, independent quadrature maps A and B have the
selected separable, nonperiodic Gaussian covariance on the finite grid. For
each coordinate axis, form its 384-by-384 covariance matrix and symmetric
positive-semidefinite square root. Clip only negative eigenvalues whose
magnitude is at most `128 * eps(float64) * max_eigenvalue`; a more negative
eigenvalue is a numerical failure. Add no diagonal jitter or extra noise.
With independent standard-normal matrices Z, maps are `Ly @ Z @ Lx.T`.
Record the clipped eigenvalue count, maximum negative magnitude and maximum
absolute covariance reconstruction error. The computed covariance is the
declared roundoff-limited PSD approximation to K, not an exact-arithmetic claim.
Reuse the same two Z draws across the four B conditions within each replicate;
this pairs worlds without making different covariance operators identical.

The raw field is `A*cos(2*pi*f*t) + B*sin(2*pi*f*t)`. These are deliberately
narrowband initial dynamics, not a claim about physiological spectra. Scale
the complete realization to its registered primary-band RMS over the common
space-time mask. Covariance K describes the **pre-normalization** Gaussian
draw; fixing each realization's RMS conditions its distribution, so do not
claim the normalized ensemble is exactly Gaussian with covariance K. Report
the applied scale, achieved RMS and measured lambda/tau rather than relabeling
ell or carrier as those scales.

## F1–F4: repeated finite-support biphasic events

Use exactly the catalogue's spatial FWHM, temporal-envelope FWHM, event
peak-to-peak amplitude and repetition rate. Each event has a positive isotropic
Gaussian spatial envelope. Its temporal envelope has sigma equal to nominal
FWHM divided by `2*sqrt(2*log(2))`, centre at support start plus `3*sigma`,
and standardized time z. Select biphasic driver **−z** multiplied by the
Gaussian envelope. The outer taper is one for `abs(z)<=2.5`, zero for
`abs(z)>=3`, and `1-10*u^3+15*u^4-6*u^5`, with `u=2*(abs(z)-2.5)`, between
those limits. The raw event is exactly zero outside this finite support.

For repetition rate r, draw a phase offset uniformly in `[0,1/r)`, independent
of locations and acquisition history. Support starts are `offset + k/r`;
retain all starts whose **complete six-sigma support lies in the finite
record**, determined before field values or candidate results. Record the
actual count and support span; nominal rate is not claimed to equal count/16
exactly. There is no rejection of overlaps or regeneration of difficult draws.
Draw each event centre independently and uniformly over the full rectangle.
Use paired uniform draws across focal conditions where their event indices
exist, without conditioning them on pad maps or scan histories.

The selected event amplitude is the **global sampled space-time peak-to-peak
range of that isolated event's primary-band field**, before summing events.
For a positive spatial envelope this equals its sampled spatial maximum times
the range of its transformed temporal driver, permitting exact factorized
normalization. Apply the same scale to raw and transformed drivers. Report
each isolated-event scale and the sum's achieved primary-band RMS separately;
the latter is not a claim about the sum's peak-to-peak range. Overlapping events are
not rescaled to force the sum back to the single-event amplitude.

Report achieved raw and primary-band temporal width at the sampled spatial
maximum as the span from first to last sample with absolute amplitude at least
half its maximum absolute amplitude. This is a declared biphasic half-height
span, not a claim that the waveform itself has the envelope's nominal FWHM.

## W1–W4: matched planar waves

Use the catalogue's initial four waves only: W1 at 7 Hz, speed 400 mm/s,
direction 0 degrees; W2 at 11 Hz, speed 100 mm/s, direction 90 degrees;
W3/W4 at 37 Hz and 100 mm/s, directions 0/90 degrees. Compute wavelength
from speed/frequency without using the rounded printed wavelengths.

Use the existing planar phase equation, origin at the rectangle centre,
reference time zero, and a replicate-specific uniform phase in `[0,2*pi)`.
W3 and W4 share this phase. Spatial cosine/sine factors and the matching
temporal quadratures give the exact sampled plane wave, with no amplitude
envelope or onset truncation. Normalize to 50 microvolts primary-band RMS.
The finite aperture is not spatially wrapped in the generator. Expanding,
reverse-direction, W-fast and sub-Nyquist controls remain the already-declared
follow-ups; they are not silently inserted into this initial four-condition set.

## M1–M4: mixtures and exact time-varying component power

M1–M3 use independent broad `(ell=2400 micrometres, f=3 Hz)` and local
`(ell=300 micrometres, f=37 Hz)` quadrature fields. Normalize component powers
to `2500*(1-q)` and `2500*q` square microvolts for q=0.1,0.5,0.9, respectively,
on the common primary-band mask. Reuse the two unscaled fields across those
three conditions within a replicate. Sum linearly without further amplitude
normalization; report component powers, uncentred cross moment, centred
covariance and summed power, following the existing mixture primitive.

M4 uses the same two underlying Gaussian innovation matrices for the broad
and local covariance operators, plus a common positive envelope
`1 + 0.25*cos(2*pi*0.5*t + psi)` with a uniform replicate-specific psi.
The two carriers share a replicate-specific phase. Equivalently, the common
input quadratures are `e(t)*(cos(2*pi*3*t+phi)+cos(2*pi*37*t+phi))` and
`e(t)*(sin(2*pi*3*t+phi)+sin(2*pi*37*t+phi))`. Apply fixed inclusive 2–4-Hz
and 36–38-Hz Fourier selectors to these **same input quadratures** before
switching. Both components thus retain a common envelope and shared innovations;
they are not independent carriers merely described as a shared input. No
candidate score sets a band.

The required fractions alternate q=0.1 and q=0.9 in successive one-second
half-open intervals, starting at 0.1. **Fractions refer to actual primary-band
component powers within each interval**, before cross terms; multiplying
filtered components by step coefficients alone is insufficient. Select one
positive raw amplitude coefficient per interval for each component. Filter
the resulting complete raw record once with the common 1–100-Hz operator.
Targets are `P_b,j=2500*(1-q_j)` and `P_l,j=2500*q_j` square microvolts.
Solve the 16 interval power constraints for each component using SciPy
`least_squares(method="trf")` on log power residuals
`log(achieved_power_j / target_power_j)`, with exact quadratic-form derivatives.
Initialize each amplitude at `sqrt(target/unswitched_interval_power)` and
optimize its log multiplier with fixed bounds `[log(1e-6),log(1e6)]`.
Select `ftol=xtol=gtol=1e-12`, `x_scale="jac"`, linear loss and maximum 200
function evaluations. These bounds limit numerical normalization, not an
inferred physical amplitude range. Require solver success and every achieved
interval power to be within relative
`1e-6` of its target. Failure is unresolved and retains the exact condition,
replicate seed, named streams and recipe for deterministic reconstruction of
the same input draw; unnormalized factor arrays are not published as completed
worlds. There is no replacement draw or relaxed target. Report coefficients, achieved
fractions, cross terms and solver termination. Switch-induced raw high-frequency
power remains in tau; the truth is not silently band-limited before estimation.

## I1–I4: linear interacting-component controls

Each spatial component is an isotropic Gaussian with 600-micrometre FWHM and
an 11-Hz temporal driver. For I1–I3, draw a shared centroid uniformly in the
central 4.8-by-4.8-mm square and an orientation uniformly in `[0,2*pi)`.
Place two components symmetrically at separations 4800, 300 and 1200
micrometres, respectively; all centres remain inside the aperture. I1 has
independent uniform temporal phases, I2 identical phases, and I3 phases
separated by pi. I4 has four components at the corners of a rotated square
with 300-micrometre side length, a centroid drawn independently in that same
central square, and independent phases. The minimum centre separation is
exactly the registered 300 micrometres. These are selected spatial challenges,
not an empirical point process. Normalize the full component sum to 50
microvolts primary-band RMS and retain interference; do not renormalize each
component independently or call linear superposition nonlinear interaction.

## S1–S4: silent-region surprise

S1 uses F2 after a 0.5-second local-event prelude with uniform event locations;
S2 uses F2 after two seconds with locations uniform in the **outer
600-micrometre-wide border** of the rectangle; S3 uses F3 after 0.5 seconds
with uniform locations. Construct a uniform border draw by area-weighted
selection of four nonoverlapping border rectangles, not by outcome-dependent
rejection. Add an independent uniform `[0,1/r)` onset offset after the stated
prelude. Reuse the focal waveform/amplitude/count rules thereafter.

S4 uses four independent uniform locations drawn once per world. At each
location, run the F4 waveform/rate after a two-second prelude and its own
independent uniform `[0,1/3)` offset. Each isolated event retains F4's amplitude;
their sum and any overlap remain visible. The four locations and all onset
offsets are independent of acquisition history. No broad background is added
to these initial surprise worlds. Raw local truth is exactly zero before the
first finite-support event; transformed prelude ringing is retained separately.

## N1–N4: observation-only nuisances

All four reuse the same default broad truth `(ell=1200 micrometres,11 Hz,
50 microvolts RMS)` within a replicate. N1/N2 supply a spatially common,
independently phased 60/50-Hz observation-side line signal with RMS 5/50
microvolts, respectively. They never enter ideal truth or its lambda/tau.

N3 draws independent 1%-probability missingness on a **declared nuisance clock
of 500 Hz** for each of the 64-by-64 physical array sites. The availability
state is held on half-open two-millisecond intervals; future acquisition
samples query that state at their actual sample times. This is an explicit
adversarial dropout process, not a claim about NeuroCam's dropout duration or
DAQ rate. Draw and bit-pack the mask in bounded time blocks. N4 selects one
uniformly drawn physical row as dead for the entire world. Neither missingness
process removes or zeros target-grid truth or changes the evaluation mask.
The world record identifies these as observation-side inputs; actual contact,
DAQ and readout application remains a separate unqualified stage.

### October 9: decoded-observation nuisance placement

Prospectively select the following application for this existing development
panel, under the delegated simulation judgment and before candidate outcomes.
Apply N1/N2 additive line values to decoded, site-labelled pixel observations
after the declared digital averaging/filtering and quantization. Use the saved
analytic phase, frequency and amplitude, converting microvolts to volts once.
N3/N4 mark those observations unavailable; an already invalid observation stays
absent. Apply this before valid-pixel counters and output callbacks. Preserve
the raw conversion record, all physical states and costs, output bit accounting,
hardware clipping flags, original timestamps and publication deadlines. Missing
outputs remain records with absent values, never dropped or filled with zero.

Each affected pixel observation must carry a replay-validated physical site
index and an explicit acquisition query point within its contributing raw
aperture support and no later than publication. Use the same declared binding
convention across hardware/software arms. Do not substitute a publication time,
infer a site for a multi-site mixture, or treat an arbitrary assigned timestamp
as an acquisition point. Non-pixel marker/reference outputs remain unchanged.
The nuisance is visible only at the observation's original publication time.
N3 queries its existing right-continuous half-open two-millisecond clock; N4
uses the saved dead row. No new nuisance draws, seeds, worlds or time grids.

This selects an adversarial post-readout observation model consistent with the
existing additive/missingness primitive. N1/N2 do not test analog suppression,
ADC saturation or digital-filter rejection of physical pickup; N4 denotes an
unavailable output row, not altered gate conductivity/loading. Those physical
mechanisms require their separate models. Ideal neural truth, its spectral
scales and the common evaluation mask remain unchanged. The independent final
model, calibration requirements and candidate-scoring boundary are unchanged.

## Execution bounds and evidence meaning

### October 9: continuous raw-time extension for acquisition

Evaluate the existing raw temporal factors analytically from the saved recipe
parameters at caller-supplied times in the half-open record `[0,16)` seconds.
Reuse saved carrier frequencies/phases, amplitude scales, finite event starts
and tapers, and M4's separately saved interval coefficients. This extension
does not interpolate the 500-Hz record, regenerate fields, refit amplitudes or
use the primary-band evaluation factors as acquisition input. Preserve raw
microvolt scaling; contact-voltage evaluation performs the single conversion
to volts. Nuisance line signals remain separate from ideal truth.

For M4, the common 0.5-Hz envelope and 3/37-Hz carriers place their sidebands
exactly on the finite 16-second Fourier bins retained by the fixed component
selectors. Their analytic envelope/carrier expression is therefore the same
pre-switch construction, up to floating-point arithmetic. At an integer-second
switch use the right-hand half-open interval's stored amplitude coefficient;
never smooth the jump. Require numerical replay against the original native-
time drivers for all 28 conditions before integrating this implementation.

Expose event support/taper boundaries and M4 switch times as interval
breakpoints. The final time 16 may be an interval boundary but is not permission
to extrapolate or periodically wrap values. The electrical executor still needs
a separately specified constant-input approximation and error criterion,
including supplied row, DAQ and aperture boundaries. A future waveform-error
bound cannot be presented as a bound on complete electrical-output error.

One caller-owned CPU, numerical threads fixed at one, no GPU, at most 7 GiB,
and a caller-enforced deadline no longer than two hours for an initial bounded
generation tranche. These are prospective execution limits, not a launch.
At most one world and its factorized spectral workspaces are live. No
147456-by-147456 covariance or 8000-by-384-by-384 voltage cube is allocated.
Each core array is bounded by an explicit `max_elements`; masks are chunked
and bit-packed. The caller's resource/deadline callback runs before expensive
linear algebra and between finite blocks. An incomplete tranche preserves
completed worlds and reports uncompleted identities; it does not schedule a
successor or change the episode budget.

The first runtime tranche may cover B1–B4 to measure actual generation cost,
but all 28 conditions remain implemented and required. Saved factors, scales,
event locations/support, nominal and achieved amplitudes, mixture diagnostics
and nuisance draws make the generated world reviewable. They establish
**constructed development inputs only**, not physiological discovery,
calibrated device behavior, hardware superiority, accepted-model support,
S/T-cell sufficiency, reconstruction performance or independent-final readiness.

## Selected implementation and artifact interface

`outputs/tools/development_worlds.py` exposes
`build_development_world(condition_id, replicate_seed, check=...,
max_elements=20000000)`. The public builder fixes the grid above and admits
only the 28 selected conditions and initial three seeds. It returns spatial,
raw temporal and primary-band temporal factors, observation-side nuisance
arrays and diagnostics including exact factorized raw-truth power scales.
The private smaller grids in focused tests are algebra fixtures and cannot be
selected through the public builder or CLI as generated production worlds.

Run its CLI with `--output-root NEW_DIRECTORY --max-seconds 6900`; the default
tranche is all 84 identities in seed-major order. An explicit `--world-ids`
comma-separated list such as `B1:2001,B2:2001` selects a bounded tranche from
that same retained panel, including a later manually authorized continuation.
The output directory must not already exist. One `development_worlds.h5`
container stores common coordinates and a group `CONDITION_seed_SEED` per
constructed world, with `complete=true` only after factor and metadata writes
have flushed. No compression or dense world cube is required. Each world
stores `spatial_modes`, `raw_drivers`, `primary_band_drivers`, `metadata_json`
and `observation_side_nuisance_inputs`. Strict JSON encodes infinite scale as
the string `+Infinity` and undefined scale as null, without changing estimator
semantics. `summary.json` records exact requested, complete and remaining
identities, including interrupted or unresolved identities and elapsed time.
Exit codes are 0 for complete, 75 for a bounded deadline with partial output,
and 1 for an unresolved generation error. Earlier completed worlds survive a
later failure; any partially written HDF5 group retains `complete=false`.
Neither interruption nor failure changes a draw, relaxes a target, schedules
a successor or assigns an evaluation role.

The initial implementation has eight focused generator checks, five focal/
surprise checks and six factorized-estimator checks authored for root's
coordinated compute allocation. They are **unrun as of source publication**;
only a subsequent actual compute receipt can supersede that status. These
checks do not qualify a forward electronics model or a reconstruction method.
