# EP20 electrical mechanisms and common characterization probes

Design selection, 2026-09-30. Use one coupled circuit to generate noise,
impedance, settling and crosstalk. A smaller pad or a different scan does not
receive four independent fitted penalties. This note completes previously
unspecified probe conventions; it contains no fitted constants or measurements.
The [component design](component_details.md) owns the forward chain and the
[starting catalogue](parameter_candidate_catalogue.md) owns pad maps and scans.

## 1. What the published anchors constrain

NeuroCam reports approximately 100 kΩ in **1–100 kHz**; its supplement describes
TFT impedance in PBS and Fig. S2 specifies `Vgs = 6 V`. Treat this as a device/path
anchor under that measurement condition, not a bare-Au contact resistance or
a measurement of 1–100-Hz contact impedance. Published transfer and the in-air
crosstalk test constrain different paths. Fig. S5 uses one driven pixel,
5 mVpp at 12 Hz; its exact scan/DAQ configuration remains unresolved.
[NeuroCam article and supplement, Fig. S2/S5 and methods](https://shengxingstars.github.io/www/images/publications/2025_High-resolution%20spatial%20mapping%20of%20electrocorticographic%20activities%20with%20NeuroCam%20a%204096-channel,%20multiplexed%20flexible%20thin-film%20transistor%20array.pdf).

Keep three electrical objects separate: passive contact impedance, biased
TFT/line circuitry, and DAQ conditioning/conversion. A measurement connects
particular terminals with a particular return, gate bias and load. Its circuit
prediction must recreate that context before it is used for fitting or checking.
The existing calibration-versus-held-aside assignment and tolerances still apply.

## 2. Two contact-scaling alternatives

For pad area `A`, let `q = A/A_L`, `A_L = 9450 µm²`. Use a passive
spreading-resistance plus parallel charge-transfer/constant-phase-element
(CPE) model:

```text
Z_e(s,q) = R_access,L q^(-1/2) + 1/[q^gamma (G_ct,L + Q_L s^alpha)]
R_access,L >= 0; G_ct,L >= 0; Q_L > 0; 0 < alpha <= 1
Q_L units: siemens × seconds^alpha; G_ct,L = 1/R_ct,L
```

This circuit structure is supported by equilibrium Au measurements, not by
transferring their material constants to NeuroCam. `alpha = 1` is the ideal
capacitor limit; `G_ct,L = 0` is the blocking-contact limit. Finite charge transfer
and fractional `alpha` remain fit alternatives. [Rocha et al. (2016), equivalent
circuit](https://pure.tue.nl/ws/files/40502864/srep34843.pdf).

| Contact template | Selected exponent | Meaning |
| --- | ---: | --- |
| E-area | `gamma = 1` | Uniform effective interface area |
| E-sublinear | `gamma = 0.7` | Deliberate sensitivity to weaker effective-area scaling, not a NeuroCam estimate |

Bare-Au disk measurements found radius exponents about −1.41 at 1 Hz and −1.86
at 1 kHz, with no small-electrode Pt-like perimeter transition. These motivate
testing non-exact area scaling; they do not identify our CPE exponent, the
mechanism causing those slopes, or rectangle-specific parameters. The rounded
`gamma = 0.7` is a selected challenge law. [Fan, Wolfrum and Robinson (2021),
Au scaling](https://pmc.ncbi.nlm.nih.gov/articles/PMC8819721/).

| Derived multiplier, relative to L | M: 50×95 µm | S: 35×70 µm |
| --- | ---: | ---: |
| Area `q` | 0.50265 | 0.25926 |
| Interface impedance, E-area `q^-1` | 1.989 | 3.857 |
| Interface impedance, E-sublinear `q^-0.7` | 1.619 | 2.573 |
| Access resistance `q^-1/2` | 1.410 | 1.964 |

These multiply components, not necessarily total impedance. The access law is
an inverse-linear-size approximation for approximately similar rectangles in
a homogeneous medium, not an Au perimeter-controlled interface law. If the
volume-conduction/contact solver already includes spreading resistance, this
term is represented there **once**, not added again. A heterogeneous contact
solver may replace that approximation consistently across every arm.

Absolute `Q_L`, `G_ct,L`, `R_access,L`, `alpha`, parasitic leakage/capacitance,
input loading and their device-condition distributions remain unidentified.
No numerical ranges are manufactured from the high-frequency 100-kΩ anchor.
The CPE needs a causal passive realization over the forward-model bandwidth;
an outcome-blind positive-RC approximation must preserve its response and tail,
not apply a future-looking FFT filter to each row independently.

## 3. Noise comes from that circuit

For an illustrative connected scalar path, let `Z_s` be the complete source
impedance and `Z_in` its electronics load. With independent noise sources:

```text
H_load = Z_in/(Z_s + Z_in); Z_eq = Z_s || Z_in
S_node = |H_load|^2 4 k_B T Re[Z_e] + S_e + |Z_eq|^2 S_i
         + separately propagated TFT/readout/reference contributions
```

The passive equilibrium contact contributes voltage PSD `4 k_B T Re[Z_e]`,
not `4 k_B T |Z_e|`. For an isolated CPE,
`Re[Z_CPE] = cos(pi alpha/2)/(Q omega^alpha)`; a large nearly capacitive
impedance need not imply correspondingly large thermal noise.
[Rocha et al. (2016), noise equations](https://pure.tue.nl/ws/files/40502864/srep34843.pdf).

Biased Ln-IZO noise is not inferred from that equilibrium expression. Its
white/low-frequency terms, amplifier voltage/current noise and reference noise
remain separately characterized contributions. If sources correlate, retain
their cross-spectral terms in a positive-semidefinite covariance model; shared
reference noise is not independent noise drawn for each contact. Resistance
noise inside the same represented network must not be counted twice.

Allow voltage-floor-, passive-thermal- and amplifier-current-dominated regimes
within both contact templates. These are parameter regimes, not three freely
assigned pad-RMS curves. In ideal unchanged-loading limits, a voltage floor
can be area independent; interface-dominated thermal RMS scales as
`q^(-gamma/2)`, and current-noise-induced RMS as `q^-gamma`. Filtering, loading,
aliasing and correlations can invalidate those limiting ratios.

Carry out-of-band signal/noise through the analog chain and actual apertures.
Apply the primary band for characterization/scoring, not as an oracle forward
prefilter. Published mode-specific RMS constrains the **whole corresponding
measurement**, not an additional RMS source summed onto these components.
More dwell does not guarantee a square-root-of-sample-count noise reduction.

## 4. Three persistent-state alternatives

Each contact template is paired with each row below: **six initial structural
templates**, all subject to the existing fitting/held-aside checks. Noise regimes
and parameter uncertainty remain within them. This is not six accepted models,
six extra candidate trials, or permission to discard an inconvenient mechanism.

| State template | Forward realization | Question it exposes |
| --- | --- | --- |
| R1: minimal source-line dynamics | One persistent equivalent pole per source line, finite on/off coupling, separately represented contact and configured DAQ response | Is a reduced switched model adequate? |
| R2: coupled array network | Contact, shared-line and output states with mutual capacitance, finite on/off TFT paths and switch charge | Do retained charge and neighbouring off-gate pads change the trade-off? |
| R3: R2 plus DAQ history | R2 plus persistent conditioning state driven by each ADC group's input-selection clock | Is apparent array settling partly DAQ multiplexing/ghosting? |

For R1, a source-line state can obey
`dx_s/dt = (a_g v_eff,g - x_s)/tau_g`, with physical neighbouring inputs in
`v_eff,g`. R2 uses a stable physically interpretable network
`C dx/dt = -G_g x + B_g v`, `y = L_g x + D_g v`.
Contact CPE states persist alongside these states. Do not reset any of them
between rows. A switch may add supported charge `q_switch`; for a fixed
capacitance/state basis, `C(x_plus - x_minus) = q_switch`. A changing basis
requires explicit charge conservation, not an arbitrary voltage reset.

Parasitic switch capacitance, load-dependent injection and retained off-state
charge are established generic mechanisms; manufacturer CMOS values are not
Ln-IZO constants. [Analog Devices, switch charge injection](https://www.analog.com/en/resources/analog-dialogue/articles/ask-the-applications-engineer-26.html).
Off-gate or skipped physical pads remain in the circuit; skipping is not grounding.

Only for a complete unchanged first-order path does
`t_1% = ln(100)/(2 pi f_c)`. A static cutoff is not a row-switch settling
measurement. R3 separates row-gate and DAQ-input clocks. No template may ignore
documented configured apertures or invent favourable ADC behavior; an omitted
memory mechanism is an approximation that must survive the existing checks.
Parameters are shared across maps/scans: geometry modifies registered physical
terms, and the scan modifies actions/history, not candidate-specific penalties.

### Minimal reference fit — shared combinations

For each of the six structural templates, predict each published observable
with its own fixture, scan, aggregation and statistic, using one shared physical
parameter set. Fit the smallest supported effective combinations, not a full
transistor model:

| Block | Calibration constrains | Does not separately identify |
| --- | --- | --- |
| Transfer | PBS gain centre/dispersion and the single-gate response/cutoff bound | Contact versus input loading, individual R/C values, switched settling |
| Noise | Circuit-propagated amplitudes and the complete static/reduced-mode summaries | Contact/TFT/amplifier/reference contributions or PSD/covariance shapes from RMS alone |
| Coupling | Nearest source-/gate-path magnitudes in the original air test | Phase, frequency dependence, distant decay or pad/scan scaling |
| Heterogeneity | Shared site/device variation, summarized on each verified readout mask | A unique spatial law or distribution from the reported centres and spreads |

Both 24-source settings imply nominally 16,000 row dwells/s at 62.5 µs and
the same source-stream rate. Their noise difference constrains the **whole
scan/history/aggregation prediction**, not an identified switching-rate term.
Do not subtract squared RMS-summary centres as if they were component variances.

Use calibration residuals in original units with the existing frozen tolerances;
unresolved `±` values are not likelihood weights. Predeclare spectral/covariance
and spatial-variation alternatives, then fit their shared amplitudes without
adding per-mode offsets. Unidentified contact/line/DAQ combinations remain
evidence-bounded nuisance cases; absent bounds stay unresolved, not zero.
Freeze fitted variants before predicting full-mode noise and distant coupling.
Retain every compatible variant and its parameter uncertainty, rather than only
the lowest-residual fit. This specifies `reference_fit`; no fitting has run.

## 5. ADC timing and resource accounting

The PXIe-4309 has eight 18-bit SAR ADCs, not delta-sigma ADCs. Its fixed AI
groups are `adc j: ai[j, j+8, j+16, j+24]`. Hardware averaging can span the
sample period despite a sample-clock timestamp; external-clock acquisition uses
single converts and does not support averaging/auto-zero/chopping. These facts
constrain an implementation but do not identify NeuroCam's actual wiring or mode.
[NI PXIe-4309 manual, pp. 2-8–2-12 and external clock](https://download.ni.com/support/manuals/377024a.pdf).

Keep four times: analog integration aperture, conversion times, assigned
timestamp, and publication/availability time. A causal inverse or policy may
use a sample only after its contributing conversions/window and publication
complete; an early timestamp does not make a future average available.

Keep three counts: internal conversions, hardware-returned source samples,
and retained pixel outputs. Charge settling/discards, marker/reference,
auto-zero/chop and startup work where configured. Source count × 100 kS/s is
only the catalogue's returned-stream proxy; neither instantaneous sampling nor
a universal 20× internal conversion factor follows from it. Actual board/AI
mapping, range, clock synchronization, filtering and marker occupancy remain
unknown. No ADC-work equality or power/thermal parity is certified here.

## 6. Common probe panel and finite protocols

### Spatial convention: one panel, not a survivor average

Before any C1 reconstruction score, freeze `P` as the maximal intersection of
physical neural-site centres read by **all eligible published hardware settings**.
Exclude marker/reference slots by role, not by observed quality. Actual source/
row mapping must identify this intersection; no coordinates are guessed here.
The four D representatives retain their prescribed published scans, so each
must read this same panel. Never shrink `P` after failures or recompute it to
favour the selected D1 recipe. Empty/unverified `P` leaves the rule uncharacterized.

These scalar inputs characterize this common panel, not full-array maxima.
Retain per-site/transition tables, all-site coverage and unavailable locations
separately; the primary R² mask remains the full registered common grid. The
panel does not exclude exploratory C2/C3/C4 designs that use other scans.
Off-panel degradation can invalidate the rule and remains an existing challenge,
not a new predictor. Dead/zero-gain panel sites are unresolved or unbounded,
never removed or assigned zero noise. Unknown values cannot pass a cost gate.

### Waveforms, times and scalar summaries

All amplitudes/durations below are **selected characterization proposals**, not
paper-reported conditions. Use identical protocols per accepted model `m`,
device condition `k` and frozen family `d`, before reconstruction outcomes.

| Probe | Selected waveform and record | Scalar input |
| --- | --- | --- |
| Independent direct gain | At each input `i`, `sum_(n=1..100) 10 µV cos(2 pi n t + pi n(n-1)/100)`; 10-s prehistory, 60-s record; repeat at half amplitude and with all phases shifted by pi/2 | No new rule input: complex direct transfer `H_ii` calibrates other probes |
| Zero-input noise | Zero external voltage inputs; retain contact/TFT/readout/reference noise and clocks; 10-s prehistory, 100-s record | `sqrt(mean_(i in P) sigma_i^2)` in input-referred µV, equal physical-site weights, not pooled visit weights |
| Contact impedance | 100-µV-peak single sine at each integer Hz from 1 through 100; discard five cycles, record 20 cycles per tone; defined terminals/return and load de-embedding | At each `i in P`, 90th ordered magnitude of the 100 complex `Z_i` measurements; then maximum of these site Q90 values, in Ω |
| Settling | ±100-µV steps at the registered pre-interface input; registered production-history panel, then transition and held victim row; 10-ms diagnostic horizon | Worst `t_1%` across panel sites, histories, polarities and clock phases, in µs; censor unresolved tails |
| Adjacent-line crosstalk | Same multisine/prehistory/record as gain, one physical input at a time, including off-gate aggressors; matched zero-input clock trace | Largest `abs(H_ij/H_ii)` across `i in P`, all physical nearest source-/gate-line neighbours `j`, registered tones and phases |

The multisine is 70.7 µV RMS and bounded by 1 mV peak by construction. Half
amplitude checks characterize linearity using outcome-blind calibrated
tolerances, not a new promotion margin. Quadrature replays help expose phase/
sampling degeneracies but do not guarantee identifiability. Use actual aperture
and availability metadata; unresolved transfer coefficients remain bounds.

The impedance statistic uses **equal linear-Hz weights**, not equally weighted
points from an arbitrary logarithmic grid. It is the registered discrete Q90,
not knowledge of between-tone resonances. `Z_i` is the passive contact port,
with its return/reference contribution explicitly represented or de-embedded;
do not label a contact-plus-biased-TFT path measurement as contact impedance.
Include the same access-resistance ownership used by the forward model.

For stationary identified scalar channels,
`sigma_i^2 = integral_(1..100) S_out,i(f)/abs(H_ii(f))^2 df`.
Switched/irregular observations require the independently calibrated
timestamp-aware transfer/covariance operator and its identifiable band subspace,
not interpolation plus an ordinary FFT or the trained field inverse. Apply this
definition per wall-time site, retaining aliasing and quantization. Never divide
all outputs by the aggregate published gain 0.78. A 100-s record supplies an
estimate, not an exact noise law; uncertainty stays with the table.

Ten-second prehistories define a common initial/history treatment, not a claim
that every contact equilibrates in ten seconds. Equilibrium stochastic contact
states and periodic switched states may be initialized from the registered
model law. If transient tails require longer conditioning, freeze a common
longer protocol from outcome-blind characterization before R², charge its work,
and apply it to every arm; do not silently reset slow states per row.

### Step meaning and history

The known-input step is normalized to a unit response, not a literal 1-V input.
Observe the victim's conditioned analog path immediately before quantization,
with the configured DAQ clocks continuing. Precondition with the nominal scan;
cover every distinct legal predecessor/selected-row transition, predecessor
input levels `-100, 0, +100 µV`, both step polarities, and distinct ADC phases.
Keep other inputs at zero except that declared predecessor. Finite symmetries
may reduce this panel only when the model establishes their equivalence.

For each conditional mean response, normalize error by the nonzero independently
calibrated steady step change. Compare with the phase-aligned periodic asymptote
when clocks continue, not necessarily a constant. `t_1%` is the first time after
which normalized error remains at most 0.01. A model needs a stable-tail bound
to certify the unobserved tail; a finite sampled trace gives interval/right-
censored bounds, not proof of remaining in-band forever. Zero DC gain or an
unknown normalization leaves this step metric undefined; do not use epsilon.

The held-row diagnostic is not a new candidate dwell: actual acquisition keeps
62.5/125/250-µs legal dwells. Separately retain end-of-dwell production residuals
under the real scan, without a held-row substitution. Frame-rate retained data
cannot resolve microsecond settling, and 10-µs source samples only bound shorter
events. Noise fluctuations do not define mean-response first crossings.

### Crosstalk meaning

`H_ij` is victim `i`'s response to known pre-interface input `j` under the same
scan/action history; `H_ii` is its independently characterized direct response.
The selected ratio is therefore **same-victim input-equivalent leakage**. It
can exceed one; supplementary dB is `20 log10(ratio)`. Do not clip it or replace
an unobserved/dead victim with zero. Include all existing physical nearest
neighbours, boundaries and pad-class pairings; skipped aggressors retain their
states. Excitation is electrical, not a spatial PBS field whose true voltage
spread is counted as circuit leakage.

The paper's reported crosstalk observable is a separate calibration target;
predict its verified original normalization in its own context. The inspected
caption does not settle its denominator, so that uncertainty remains open.
Its dB anchor is not directly the new ratio. Subtract a matched zero-input clock
response once, not signal-dependent switching leakage. For optional adaptive
scans, freeze a probe-only action replay and use the identical replay for direct
and cross responses; separately report the live-policy probe response. Missing
coefficients or rank-deficient probes leave bounds/unknowns, not a favourable
maximum over only identifiable pairs.

## 7. What to resolve next

The most useful discriminators are matched L/M low-frequency complex EIS;
contact versus gated-device/fixture paths; zero-input PSD with known source
impedances and an input short; and static/scanned raw gain, switching and timing
traces with actual DAQ configuration. These are proposed measurements, not
newly scheduled experiments. Layout exclusions and absolute power/thermal
costs still need their own evidence.

Retain every variant that passes the existing assigned calibration and held-
aside checks, including its identified parameter uncertainty. Existing sentinel
and finalist rank-reversal rules decide whether a further discriminator is
necessary; no preferred noise regime or simplified state model is selected
because it improves a candidate. The six templates do not relax those checks.

The [compact component choices](component_details.md) now supply starting
source-to-surface ranges and inverse allowances. [Anchor roles](../DATASETS.md)
are now selected; next resolve their measurement contexts and instantiate shared
parameters using the minimal fit above. Probe tables, numeric circuit
constants, fabricated-pad eligibility and internal ADC costs are not complete.
No model fit, candidate experiment, independent-final access or Git mutation
occurred. Endpoints, margins, C/D families and shared trial/resources are unchanged.
