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

### Causal passive realization — implemented, component checks passed

[causal_cpe.py](tools/causal_cpe.py) implements the selected interface with
explicit physical parameters, pad area, pole bounds and branch count. The
caller also supplies the forward-frequency band, an increasing check grid
containing both band endpoints, and distinct grid-error/truncation tolerances.
There are no fitted device values, automatically chosen poles, source-derived
parameter ranges or candidate scores.

For `0 < alpha < 1`, the diffusive representation gives

```text
s^alpha = sin(pi alpha)/pi * integral_0^infinity [s/(s+p)] p^(alpha-1) dp.
```

This follows by applying the fractional-integrator representation to
`1-alpha` and multiplying by `s`; see
[Baranowski, Bauer and Mularczyk (2021), diffusive realization and logarithmic quadrature](https://www.mdpi.com/2079-9292/10/15/1767).
With `x=log(p)`, positive midpoint quadrature on caller-set `[log(p_low),
log(p_high)]` uses `g_k = q^gamma Q_L sin(pi alpha)/pi * delta_x * p_k^alpha`.
Each admittance term `g_k s/(s+p_k)` is exactly a series resistor `1/g_k` and
capacitor `C_k=g_k/p_k`. The parallel `q^gamma G_ct,L` branch is retained.
At `alpha=1`, the interface uses an exact mutual capacitor `q^gamma Q_L` with
no approximation or internal RC states. Access resistance and equilibrium/
TFT/readout noise remain externally owned contributions, represented once.

The implementation reports two different approximation quantities. Bounding
the omitted integral using `|s/(s+p)| <= 1` below `p_low` and
`|s/(s+p)| <= omega/p` above `p_high` gives the following relative truncation
bound, uniform over positive `[omega_min, omega_max]`:

```text
sin(pi alpha)/pi * [(p_low/omega_min)^alpha / alpha
                    + (omega_max/p_high)^(1-alpha) / (1-alpha)].
```

That bound excludes quadrature error. The separately reported maximum
relative complex-admittance error is checked only on the supplied finite grid
against `q^gamma Q_L (j omega)^alpha`, without allowing a large charge-transfer
conductance to mask the error. It includes both truncation and discretization.
Passing these checks does not certify total error between grid frequencies,
the ideal singular response at zero time, or the fractional tail at arbitrarily
long times. The zero-history voltage-step response is available at explicitly
positive times as `G_ct + sum_k g_k exp(-p_k t)` per unit step; the exact
`alpha=1` capacitor impulse at `t=0` is outside that interface. A calibrated
forward model still needs justified bandwidth, order and temporal-tail control.

The stamp connects two explicitly named-by-index **dynamic terminals** in a
generic `SparsePassiveRCNetwork`: resistor terminal-a to internal-k, capacitor
internal-k to terminal-b. Initial voltages for all added states are caller
inputs; those same states and contact branches must survive every TFT/row
switch. Internal nodes receive zero reference capacitance. The sparse kernel
now allows this precisely when every connected component of positive mutual
capacitors includes a positive reference capacitor. Its energy quadratic form
is then positive definite; floating components are refused, without an epsilon
ground capacitor or altered physical parameter.

The original `NeurocamPassiveArray` wrapper still holds 4,160 states and does
not append CPE nodes automatically. The new explicit-voltage assembler below
provides that separate integration route. With `K` fractional branches on each
of 4,096 contacts, those contacts add `4096*K` internal states to the base array.
The earlier 4,160-state timing and memory observations do not establish
resources for this expanded CPE array.

Six [focused CPE checks](tools/test_causal_cpe.py) passed in **0.024 s** in
allocation **46864089**. All six [sparse kernel checks](tools/test_sparse_array_rc.py),
including the added grounded-storage case, passed in **0.098 s**. They cover branch positivity,
fractional response/area scaling at declared finite points, Schur-reduced
two-terminal admittance, state persistence across switches, the exact capacitor
limit, separate numerical refusal paths and grounded-versus-floating storage.
They are bounded arithmetic fixtures, not a new synthetic study or a device
validation result. The [execution record](experiment_log.md#2026-10-06-2258-pacific--model_specification-forward-chain-checks-and-astra-pass)
links the direct test and current ASTRA validation evidence.

### Full-array contact-voltage assembly — implemented, component checks passed

[cpe_array.py](tools/cpe_array.py) supplies `NeurocamCPEArray` for the existing
forward interface: already referenced and once-averaged contact voltage in
**volts** -> owned series access resistance -> passive contact interface ->
retained pad -> selected on/off TFT path -> retained shared source and load.
Reference ID, all physical parameters and initial states are explicit. The
caller supplies constant-voltage intervals at a justified waveform resolution;
this does not choose actual DAQ timing, filter bandwidth or a cortical-field
generator. Neither a field terminal nor an off-gate pad is silently grounded.

For a fractional contact, orient each series branch as interface junction
`a` -> resistor `1/g_k` -> internal state -> capacitor `g_k/p_k` -> pad. The
charge-transfer branch connects `a` to pad. Let `h=(G_ct,g_1,...,g_K)`,
`H=sum(h)` and let `R` be the explicitly supplied access resistance at that
pad's area. Eliminating only the **massless algebraic junction** yields

```text
G_contact = diag(h) - R/(1+R H) h h^T
b_contact = h/(1+R H),  nodal drive = b_contact U.
```

Every capacitor state remains. `R=0` is the exact prescribed-voltage limit.
`PassiveStarConductance` evaluates this positive resistor network through a
centered matrix-vector product, retaining its exact diagonal for the existing
trace preconditioner. It does not materialize all branch pairs: storage and
products scale with the sum of contact branch counts, rather than their
squares. The kernel accepts this validated physical-star object, not an
arbitrary claimed-passive matrix-free operator.

For `alpha=1, R>0`, one additional dynamic interface terminal carries the exact
mutual capacitor to pad, its parallel `G_ct` and the resistor to the supplied
field voltage. For `alpha=1, R=0`, the interface is an ideal capacitor directly
between pad and prescribed voltage `U`: the effective kernel diagonal gains
`C` and a voltage discontinuity contributes nodal charge `C delta(U)`.
Previous `U` persists across every row/bank interval. The actual stored energy
is the sum over physical capacitors; for this last case it is
`0.5 C (V_pad-U)^2`, not the kernel's surrogate `0.5 C V_pad^2`.

The wrapper returns physical energy before and after the boundary and at the
interval end, separately from the kernel's surrogate energy and Norton-current
work. For an instantaneous common linear ramp of the supplied voltage and
external charge, field-source work is
`0.5 (U_old+U_new) delta[q_U]`, where `q_U=C(U-V_pad)`. Supplied-charge work is
`0.5 (V_before+V_after)^T Q_supplied`; it does not identify device switch losses.
During a constant-`U` interval, direct-capacitor source work is
`-sum U C delta(V_pad)`. The physical resistor-port work is
`duration * sum_i U b_i (U-mean(V_i))`, which is distinct from nodal Norton
drive work. All accounting must be finite before a preview is committed; a
failed interval/accounting step leaves both voltages and previous `U` unchanged.
These quantities do not establish complete NeuroCam power or thermal parity.

Access ownership is explicit. The assembler-owned route receives the actual
area-scaled series resistance. In an upstream-owned route it adds zero series
resistance and requires an already solved upstream **interface** voltage;
an unloaded field voltage cannot be relabelled as that solution. Reported port
work excludes circuitry retained upstream. No access or contact noise is added
implicitly, and all further leakage, capacitance, coupling and source loads
remain explicit caller inputs.

Without the optional DAQ frontend, the state count is exactly
`4160 + sum(K_fractional) + number(alpha=1,R>0)`. The joint mux/hold component
below adds one state per explicitly configured ADC group.
Every pad, shared source and added contact state persists when an observation
bank changes. Assembly limits check states, capacitor edges and star terms
before construction, and retained numeric storage including the sparse LU
after construction. This is **not peak process memory**: transient SciPy/LU
workspace, temporary arrays and Python objects require the enclosing Slurm
memory allowance. The existing per-interval operator-product limit still
applies; fixed solves and factorization remain outside that work counter.

Seven [integration checks](tools/test_cpe_array.py) passed in **1.208 s** in
allocation **46864089**. They cover the structured Schur
operator and physical port work, exact driven-capacitor boundary/interval
energy balance, simultaneous explicit charge, an independently reduced dense
ideal-contact response, a full mixed array retaining every state across row
changes, accounting-refusal atomicity, and reference/ownership/resource refusal.
The full mixed-array fixture ran successfully; its test-suite timing does not
establish calibrated-reference throughput, reference qualification or a
candidate score. The [execution record](experiment_log.md#2026-10-06-2258-pacific--model_specification-forward-chain-checks-and-astra-pass)
links all seven cases and current ASTRA validation.

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

The [scalar loaded-node implementation](tools/loaded_node.py) connects the
existing contact component to the explicit loading/noise equation above at one
frequency. It returns signal transfer, parallel impedance and separate contact,
voltage, current and named additional node-noise contributions. The caller
supplies every impedance and noise spectrum; the owned passive contact noise is
propagated once, without adding that contact twice to the complete source path.
This implements only the explicitly independent-source branch. It does not
choose a noise regime, justify independent noise across physical sites, replace
required cross-spectra, generate a stochastic trace or select an actual DAQ
aperture. The six [analytic checks](tools/test_loaded_node.py) passed in job
46839671; the [execution record](experiment_log.md#2026-10-07--model_specification-loaded-node-checks-and-astra-pass)
links test and ASTRA evidence. Missing circuit constants and covariance/DAQ
context remain unresolved.

### October 9: replayable finite-band physical noise

Select a sparse spectral-factor representation for explicitly conditional
physical noise. For positive frequencies `f_k`, positive quadrature widths
`w_k` in Hz, supplied complex factors `L_k` and independent real standard
normal quadratures `a_k,b_k`, use

```text
S_k = L_k L_k*                             # one-sided cross spectrum
x(t) = sum_k sqrt(w_k) Re[L_k(a_k+i b_k) exp(i 2 pi f_k t)]
E[x(t+tau) x(t)^T] = sum_k w_k Re[S_k exp(i 2 pi f_k tau)]
```

The covariance expression follows directly from the independent quadratures;
there is no extra square-root-of-two factor with this convention. Each row
has its own declared physical unit (A or V), so factor rows have that unit
per square-root Hz, including mixed current/voltage cross spectra. Sparse
factors support independent branches and shared sources without a dense
port-by-port covariance matrix. The general use of a cross-spectral matrix
for multivariate simulation is described by [Deodatis (1996)](https://ascelibrary.org/doi/10.1061/%28ASCE%290733-9399%281996%29122%3A8%28778%29);
the equation above specifies this selected Gaussian-quadrature variant, not
a claim to reproduce that paper's ergodic random-phase algorithm.

Draw the complete named latent basis once, retain it, and reuse it with the
same frequency grid across paired designs. No random draws occur at solver
substeps, row switches, apertures or storage seams. Evaluation at an existing
midpoint key always gives the same original physical amplitudes. Finite
spectral atoms are not continuous white noise or an ergodicity guarantee.
No per-record RMS renormalization, mode-specific noise offsets, hidden DC
component, automatic bandwidth, thermal temperature or correlation law is
selected. Omitted sparse entries specify zeros in a caller's conditional
model; they do not prove absent physical contributions. The chosen forward
band must carry relevant out-of-band noise through the analog/readout chain,
and spectral discretization/truncation still needs its own numerical check.

Physical ports retain their actual circuit locations. Retained-node current
sources have explicit positive/negative terminals. A fractional contact's
massless access junction requires its original branch current sources and
series access voltage source; independent noise on the reduced nodes would
generally lose their induced correlations. For branch currents `eta_j`
directed from junction `a` to retained node `v_j`, access noise voltage `e`
added to external field port `U`, branch conductances `h_j`, and series
resistance `R`, the exact elimination is

```text
H = sum_j h_j; D = 1 + R H
a = (U + e + R sum_j h_j v_j - R sum_j eta_j) / D
n_j = eta_j + h_j (e - R sum_l eta_l) / D
I_access = (H (U+e) - sum_j h_j v_j + sum_j eta_j) / D
```

`n` adds to the existing full-array forcing; it creates no extra dynamic
states or ADCs. Independent equilibrium branch-current spectra `4 k_B T h_j`
and access-voltage spectrum `4 k_B T R` give reduced spectrum
`4 k_B T [diag(h)-R h h^T/D]`. This algebraic identity preserves the passive
network's correlation structure. It does not assign equilibrium noise to
biased TFTs or characterize amplifier/reference noise. The passive-contact
noise relation remains grounded in [Rocha et al. (2016)](https://pure.tue.nl/ws/files/40502864/srep34843.pdf).
For `R=0`, the access-resistance noise voltage must be zero; the original
branch current sources still have physical port work.

With amplitudes held constant within one numerical interval, correct field
work by `dt*U*(H e+sum eta)/D`. Separately retain access-noise work
`dt*e*mean(I_access)` and current-noise work
`dt*sum eta_j*(mean(v_j)-mean(a))`, plus explicit retained-node current-source
work. Reduced Norton work alone is not physical port accounting. Amplifier
input voltage noise shifts the sensed source voltage before gain and target
clipping; its effect on the controlled source remains in that source's
terminal work, not an invented input-power term.

The physical layout owns all represented star branches/access sources.
For ideal contacts with positive access resistance, the assembler represents
the access resistor as a star branch and the charge-transfer resistor as a
retained-node edge. That edge needs an explicit matching noise port or a
recorded conditional exclusion; it is not covered merely by supplying star
noise. Other omitted device contributions remain unresolved. This interface
does not itself certify a complete thermal/device-noise inventory.

The existing midpoint planners must satisfy both the field-voltage tolerance
and explicit per-port A/V tolerances. Realized spectral amplitudes provide
analytic derivative envelopes; the same midpoint sample persists through
all later aperture/storage splits. A bound covers the stored analytic forcing
only, excluding factor-contraction/trigonometric arithmetic, spectral
quadrature error and circuit-output error. All original physical amplitudes,
including ones that cancel in the nodal projection, join the existing replay
check. Only one live sample is retained, with separate finite source-work,
storage, evaluation and planning limits. These are numerical implementation
choices and do not open calibration, candidate scores or final data.

The retained R2/R3 array and separate R1 component use the additional port
rules below. A complete conditional source inventory, spectra, numerical
recipe and calibrated reference remain separate requirements. Conditional
omissions are not zero measured noise.

### October 9: external DAQ noise and the reduced R1 port

A current-noise source on a mux branch from a named external voltage `U` to
hold node `h` is distinct from a current source to the zero reference. Give
the actual `(ADC ID, external input ID)` branch its own immutable identity,
held current `eta`, and the frontend's existing track/off conductances. With
positive current directed from the external terminal into the hold node,

```text
C dh/dt includes g (U-h) + eta
W_noise = dt eta (mean(h)-U)
W_external = dt U [g (U-mean(h)) + eta]
```

The `dt U eta` correction belongs to the external ideal voltage source;
`dt eta mean(h)` alone is reduced nodal work. Their sum with the physical
noise-port term recovers that nodal injection without counting either source
twice. No new capacitor, boundary charge, ADC or reset is introduced. The
original noise amplitude participates in replay even if currents cancel at
the retained nodes. A declared resistor-noise port requires zero current in
a phase where its physical branch conductance is zero. An independent
nonresistive source would require its own explicit topology and ownership.

Validate port ownership against the actual assembled external branch inventory,
including off paths. Validate each noise sample against the interval's mux
selection during the complete acquisition preflight and again before use.
This avoids accepting an early part of a record whose later disconnected
branch carries a nonzero declared resistor-noise source. The rule does not
derive a thermal PSD from a biased switch or specify how its noise varies
between track and off phases. A complete recipe must supply that law using
the same retained latent draws; fixed spectral coefficients alone do not
establish a switching-dependent physical noise model.

For the scalar R1 pole, retain the stated equation and add an explicitly
reduced held current. If original current ports `i_j` have a declared
conjugate one-state reduction with dimensionless coefficients `b_j` and held
voltage offsets `d_j`, then

```text
I = sum_j b_j i_j
dx/dt = (gain v_eff-x)/tau + I/C
target = gain v_eff + tau I/C
W_original = dt sum_j i_j (b_j mean(x)+d_j)
W_equivalent_nodal = dt I mean(x)
W_offset = dt sum_j i_j d_j
```

Use the original exact exponential interval endpoint and aperture mean with
this changed target. A fixed sample survives interval subdivisions; the
component never draws noise. Require explicit state/reference identity and
component ownership, and finish all finite work calculations before state
commit. Keep original-port, nodal and offset work distinct. An arbitrary
offset is not a derivation of an eliminated network's missing supply work;
the caller must justify the reduction and account for other source ports.
Neither `C` nor `tau` supplies temperature, bandwidth or a measured noise PSD.

This R1 addition is a scalar component interface. There is currently no
production R1 acquisition caller: the eager/streaming full-array executors
instantiate the retained CPE array. A complete R1 alternative still needs
all 64 persistent source poles, its separately represented contact/neighbour
and configured DAQ response, and an acquisition adapter that preserves their
joint history. The scalar extension does not substitute for that route or
establish a complete six-template reference panel.

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

### Joint source and acquisition-state integration

A dynamic acquisition stage cannot in general be driven by an array source's
interval mean as though that voltage were constant. A downstream filter's
endpoint and aperture integral depend on the within-interval source waveform.
The selected passive mux/hold mechanism instead retains one physical voltage
state `z_a` for each explicitly configured ADC group in the same array solve:

```text
C_a dz_a/dt = sum_s g_as(t) (v_s - z_a) - g_leak,a z_a.
```

Each branch contributes the reciprocal current to source `s`. Selecting an
input changes its explicitly supplied track/off conductance, while all other
declared paths and capacitor states persist. An explicit all-off interval holds
the charge subject to the supplied off paths and leakage; it never resets the
state. Row-gate changes and ADC input changes are separate boundaries. The
existing source-load shunts describe additional loads and must exclude this
represented mux/hold network, avoiding double counting.

The implementation must propagate array, contact and hold states jointly and
return hold-node means with their physical ADC IDs. Only then may a caller
map them to the readout's explicitly declared input IDs and complete uniform
apertures. ADC-group identities are not source-channel identities or authentic
Rat5 geometry. All capacitances, conductances, initial voltages and source
mapping remain caller inputs; none is inferred from the manufacturer's ADC
count or the article's source-stream rate.

This is a conditional passive mechanism within R3, not an identification of
the PXIe-4309's internal circuitry. The manufacturer literature distinguishes
driver buffering, RC filtering and the ADC's switched-capacitor load, with
settling contributions from more than one stage. Active conditioning, gain,
saturation/recovery and noise therefore remain separate requirements; example
component values for another ADC are not NeuroCam parameters.
[Analog Devices: front-end amplifier and RC filter design](https://www.analog.com/en/resources/analog-dialogue/articles/front-end-amp-and-rc-filter-design.html).

[`daq_mux_hold.py`](tools/daq_mux_hold.py) implements this finite mechanism
through `PassiveMuxHoldGroup` and `PassiveDAQFrontend`. The optional
`daq_frontend` argument to `NeurocamCPEArray` appends its hold states after all
contact states. `daq_selected_sources` supplies one source or explicit `None`
per group for each interval. `DrivenArrayInterval.daq_hold_observations`
returns `HoldNodeObservation` records with `adc_id`, `state_id`, `mean_volts`
and `end_volts`, separately from unchanged source diagnostics. These analog
values are not quantized, time-published samples or inputs automatically
released to an inverse.

Hold capacitances contribute to physical stored energy and explicit injected
hold charge contributes to the same boundary-work accounting. Resource
metadata adds hold-state and physical-switch-path counts, including declared
off paths whose present conductance is zero. The numeric-array/LU storage
limit still excludes Python configuration objects and transient workspace;
it is not a total-process memory bound. All physical states remain subject to
the existing numerical/work limit and refuse before commit on invalid mapping
or failed propagation/accounting. The outer caller owns readout integration:
if readout refuses after an analog interval has committed, abort that attempt
rather than replaying the aperture from the already advanced state.

Three [focused checks](tools/test_daq_mux_hold.py) compare the complete array
with an independently reduced five-state circuit, feed joint hold-node means
through a complete readout aperture, verify charge/storage and bidirectional
loading, and exercise invalid selections and the unchanged array-only path.
The row and mux changes use separate boundaries. All three passed in **0.631 s**
in allocation **46864089**; [direct evidence](runs/component_checks_job_46864089/test_daq_mux_hold.txt)
records each case. Actual DAQ configuration and reference-model acceptance
remain unestablished.

### October 9: conditional active conditioning and loaded recovery

Select the following **prospective R3 macromodel** to implement the previously
missing active stage. It is a conditional model with caller-supplied parameters,
not an identified NeuroCam amplifier or a newly accepted reference variant.
The existing passive frontend remains an explicit alternative. No example
component values from another ADC are adopted. The manufacturer reference
above motivates representing amplifier, filter and switched-capacitor settling
together; it does not identify this internal topology or its constants.

Each of the64 physical source lines has an explicitly declared input
conductance/capacitance and two retained downstream states: buffer voltage
`b_s` and output-filter voltage `f_s`. The existing fixed source shunts exclude
these separately represented loads. With common-reference voltages:

```text
u_s = clip(A_s*v_source,s + offset_s, lower_rail_s, upper_rail_s)
C_b,s db_s/dt = g_b,s*(u_s-b_s) - g_out,s*(b_s-f_s)
C_f,s df_s/dt = g_out,s*(b_s-f_s) - g_filter,s*f_s
                - sum_a g_as(t)*(f_s-h_a)
```

The original source, contact, pad, buffer, filter and ADC-hold states propagate
jointly. Mux paths now load `f_s` bidirectionally; their physical ADC identities,
external marker paths, off conductances and hold histories are preserved.
The input stage is an explicitly unilateral controlled source, with finite
declared input loading; it is not a resistor connecting the source and buffer.
Do not drive these dynamics with source interval means or reset them at a row,
ADC, storage-window or numerical-substep boundary.

Every gain, offset, rail, conductance, capacitance and initial state is required
from the caller. Rails limit the **internal drive `u_s`**, not the loaded
buffer/filter voltage: passive backdrive may temporarily move those nodes
outside the rails. Persistent stored state creates load-dependent recovery.
This model asserts neither a hard output clamp nor measured overdrive delay,
slew-rate limiting, amplifier stability margins or authentic NI internals.

Keep physical accounting explicit. The ideal controlled voltage source's
terminal work is `integral sum_s g_b,s*u_s*(u_s-b_s) dt`. Its series-branch heat
is `integral sum_s g_b,s*(u_s-b_s)^2 dt`. Neither is the Norton nodal drive work
`integral sum_s g_b,s*u_s*b_s dt`. The first quantity is named
`controlled_voltage_source_interval_work_joule`; it is not total supply
consumption. Quiescent current, efficiency and transistor losses remain
unidentified. Buffer/filter capacitances enter stored energy. The joint solver
retains passive Norton dissipation and restores the controlled branch's heat
separately; the assembler restores contact and external voltage-port work.
The partially restored Norton quantity is not total physical heat. No
complete-power or thermal parity claim follows from these circuit quantities.

The active branch uses a separate joint nonlinear preview solver. A bounded
implicit-midpoint update and interval step doubling retain the full sparse
capacitance and passive-star structure, while the clipped controlled source
enters as a directed forcing term. Caller-supplied error, nonlinear/linear
iteration, operator-work and workspace limits must hold before the complete
interval can commit. Compare both voltage endpoints/means and integrated
work/heat across partitions. Partition disagreement remains a numerical
consistency estimate, not a certified continuous-time error bound; implicit
midpoint is not L-stable and its stiff-mode accuracy still needs verification
for the selected parameters. The existing
passive exponential path is unchanged when no active stage is supplied.

Physical stochastic noise uses the separately specified finite-band port
interfaces above; a complete spectral/covariance recipe and source inventory
still need selection. The saved N1–N4 post-readout overlays and ADC clipping
do not substitute for it. This selection
opens no calibration anchors, candidate scores, Rat5 payloads or final data,
and selects no job or numeric full-record recipe.

The [R1 source-line implementation](tools/switched_line.py) now propagates
this selected first-order equation exactly for piecewise-constant effective
inputs. A caller supplies initial voltage, fixed capacitance, gain, time
constant, switch charge and every interval duration. State persists across
row changes; discarded settling intervals still advance it. Exact endpoint
and interval-average outputs support point sampling or a specified aperture,
without selecting NeuroCam's unknown ADC behavior. Physical neighbour inputs,
contact CPE, clipping and DAQ history remain separate required model
components. Explicit reduced held-current noise and its original-port work
are now supported as specified above; no full R1 acquisition route follows.
The original seven [focused analytic checks](tools/test_switched_line.py)
passed in 0.001 s in coordinated job 46833971 for that earlier source.
Three new methods and the affected prior checks remain unrun after the
noise extension. Their constants are arithmetic fixtures, not device
estimates. The [earlier execution record](experiment_log.md#2026-10-07--model_specification-r1-checks-pass-scalar-loading-implemented)
retains the direct test and ASTRA evidence.

The [small coupled RC implementation](tools/coupled_rc.py) now supplies the
passive linear part of the selected R2 equation. Every C/G/B/L/D matrix, input,
initial voltage and interval is caller-specified; no DAQ or device constant is
inferred. C is symmetric positive definite and G is symmetric positive
semidefinite with no computed negative eigenvalue allowed. A mathematically
semidefinite G can be refused due to negative roundoff; it is not projected or
clipped. This avoids a small capacitance amplifying an admitted negative mode.
A constant-input matrix exponential propagates endpoint voltages and exact
interval means. Changing the observation leaves every node's persistent state
intact. For capacitance changes in the **same named state basis**, the kernel
solves `C_new x_after = C_old x_before + q_injected`; it records nodal charges
and stored energy `0.5 x^T C x` before and after. That energy difference is
not a complete switching-work or power estimate. Interval input work records
only `integral(x^T B v dt)` into these modeled states.

This dense kernel supports at most 64 electrical states as an implementation
limit, not a reduction of the frozen 4096-pad lattice. It remains a small-network
oracle for the sparse integration below. Causal contact-CPE realization, spatial
noise/cross-spectra, active or nonlinear dynamics, clipping and actual DAQ
history remain unfinished. The
eight [analytic checks](tools/test_coupled_rc.py) cover the known scalar limit,
coupled eigenmodes, isolated charge redistribution, interval partitioning,
capacitance switches, ideal storage, persistent unread state and invalid
parameters. All eight passed in 0.007 s in job 46842156; their fixtures are not
NeuroCam parameter estimates or candidate simulations. The changed ASTRA
program passed schema and semantic validation in job 46843116 after correcting
the missing-`yaml` environment. The
[test entry](experiment_log.md#2026-10-07--model_specification-coupled-rc-checks-pass-astra-environment-incomplete)
and [validation entry](experiment_log.md#2026-10-07--model_specification-astra-environment-repair-and-validation-pass)
link the direct evidence and preserve the original environment-failure history.

### Sparse integration of the full retained array — numerical checks passed

[sparse_array_rc.py](tools/sparse_array_rc.py) adds sparse passive-branch assembly
and a fixed **4096 pad + 64 shared-source node** row driver. Pad `(row, column)`
connects through one explicit TFT conductance to source `column`; activating a
row selects its supplied on values while every other pad retains its supplied
positive off conductance. Selecting observed sources changes readout only.
This is a virtual circuit ordering, not an inferred Rat5 acquisition map.
Reference and mutual capacitances, other resistor edges, on/off values, input
paths, loads, initial states, injected charge and every interval are explicit.
The supplied resistive drive path is not a substitute for the required contact
CPE; choosing an empty edge list or zero drive/load is an explicit caller
assumption, not a default estimate for NeuroCam.

Nonnegative resistor/capacitor branches establish passivity by construction.
Positive reference capacitance at every node gives a nonsingular storage matrix;
the newly implemented, not yet compute-tested extension allows zero reference
capacitance only in positively grounded mutual-capacitor components, as above.
Sparse LU solves apply `C^-1` without materializing its inverse. A transposable
matrix-free augmented operator supplies endpoint and interval-average voltages
using [SciPy 1.12 `expm_multiply`](https://docs.scipy.org/doc/scipy-1.12.0/reference/generated/scipy.sparse.linalg.expm_multiply.html).
The diagonal trace approximation is only exponential preconditioning, which
the documented API permits; actual propagation retains the coupled C/G.
No dense 4160-by-4160 matrix or exponential is formed, although sparse
factorization can still incur graph-dependent fill-in.

Caller-set absolute/relative tolerances compare a whole interval against finer
partitions for every retained state and observed endpoint/mean. The returned
partition disagreement is a numerical consistency estimate, **not a certified
error bound**. An explicit operator-product limit includes transpose and norm-
estimation work (fixed boundary/drive solves and factorization are excluded);
failure leaves persistent state and boundary charge unapplied.
Capacitance changes preserve all nodal charges with a reported solve residual
and stored-energy accounting. No switching-power estimate or actual ADC
conversion/aperture convention follows from this integrator.

The original five [focused checks](tools/test_sparse_array_rc.py) passed in 0.093 seconds
in job 46845646:
agreement with the validated dense kernel, variable-C charge/energy identities,
work-limit atomicity and invalid branches, the zero-conductance analytic limit,
and one full-lattice fixture spanning two row/bank observations and a capacitance
change. These are arithmetic fixtures, not published-device constants, an
authenticated reference scan or a candidate sweep. Source-supported parameter
fitting, noise/clipping, actual DAQ events and resource accounting remain
necessary. The later six-case sparse suite and expanded CPE/DAQ/executor suites
passed in 46864089, as recorded in the
[current execution evidence](experiment_log.md#2026-10-06-2258-pacific--model_specification-forward-chain-checks-and-astra-pass).
The ASTRA revision present in 46845646 passed schema and semantic validation;
the current CPE/DAQ/executor program also passed both checks in 46864089. The
full-lattice fixture reports 4,160 states, 12,224 capacitance nonzeros and 104
operator products; its speed does not estimate a calibrated reference's
throughput with unresolved stiffness/CPE/DAQ behavior. See the
[compute record](experiment_log.md#2026-10-06-1856-pacific--model_specification-sparse-full-array-checks-pass).

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

### October 7: calibration observables and partial transfer constraints

The adopted simulation scope and shared-fit design permit prospective choices
of effective parameterization, spectral/covariance and spatial-variation
alternatives, and numerical fitting method. Those choices are work to complete;
they are not unanswered requests for the scientist to approve simulation again.
They do not identify historical acquisition settings. The table below binds
the assigned calibration anchors to the actual quantities a fit must predict,
using the [existing source record](../DATASETS.md#compact-anchor-assignment--selected-design).
This is a **partial, unaccepted reference specification**, with no fitted
variant or candidate score.

| Assigned calibration anchor | Observation operator | Permitted use now and remaining limit |
| --- | --- | --- |
| Fig. 1f single-gate frequency response | Whole-path voltage transfer under the single-gate fixture, with its 3-dB frequency defined relative to the appropriate passband reference | Retain the documented approximate lower-bound bandwidth statement as a transfer constraint; the conditional first-order interpretation below is available now. The 1-/10-mVpp ambiguity does not change a linear normalized transfer, but source linearity, clipping and the actual fixture/DAQ response are not thereby verified. |
| Fig. 2a PBS array gain | `Vout/Vin` for a 12-Hz sine over all 4,096 channels in PBS; caption states amplitude 1 mV | Frequency, waveform and nominal channel coverage are documented. The effective extraction/summary and stimulus/acquisition operator remain unresolved; amplitude convention alone cancels under the conditional linear same-functional ratio below. The reported centre is 0.78; do not call it DC gain or use 0.12 as an SD/likelihood weight. |
| Fig. S3b static 23-signal-channel noise | RMS in the documented one-second windows, with the original channel-summary statistic and grounded platinum/PBS fixture | Preserve the reported static observable separately. Its filter/bandwidth binding and the meaning of the quoted dispersion still need evidence; one-second static windows do not establish the dynamic windows. No PSD component or physical noise source is separately identified by this summary. |
| Fig. S4 two reduced dynamic modes | Original per-site/window RMS followed by the original mode-summary operator, using each mode's scan history and acquisition/aggregation | Fit shared physical amplitudes only after those operators are specified. The common 24-source stream and 62.5-µs dwell are fixed inputs; the two summaries neither identify a switching-rate noise term nor justify independent mode offsets. |
| Fig. S5 nearest source/gate neighbours | Original victim response divided by the original measurement reference, under the air, single-driven-pixel, 5-mVpp/12-Hz fixture | Preserve the two directional observations. The denominator and relevant scan/DAQ configuration remain unresolved, so the independently defined same-victim probe ratio cannot substitute for this calibration operator. |

Fig. 2a's corrected fields come from the [final article caption, printed
p. 4136 / PDF page 5](https://shengxingstars.github.io/www/images/publications/2025_High-resolution%20spatial%20mapping%20of%20electrocorticographic%20activities%20with%20NeuroCam%20a%204096-channel,%20multiplexed%20flexible%20thin-film%20transistor%20array.pdf#page=5).
The [dataset correction](../DATASETS.md#fig-2a-transfer-context-correction--october-7-2026)
retains the conflicting supplement amplitude description and exact remaining
measurement dependency. This supersedes the earlier statement that this
anchor's frequency, waveform and nominal channel coverage were unknown.
Its array gain must not be silently combined with the separate single-gate
bandwidth observation as though they shared one transfer operator.

#### Fig. 2a minimum operator information

The following are conditional modeling deductions, not recovered descriptions
of the experiment. For linear, undistorted input/output sinusoids, using the
same amplitude functional on both cancels the peak/peak-to-peak/RMS conversion
factor. This does not prove actual linearity, absence of clipping, or equivalence
of different extractors for noisy, distorted or switched waveforms.

Let `H(12)` denote a conditional linear map from contact-input phasors to
the measured 12-Hz output phasors, including the acquisition path, and let `b`
be the contact forcing per unit external input. The channel response is
`abs([H(12) b]_i)`. If the bath drives every contact equally, `b = 1` gives
a row-sum response; it is generally not the isolated direct gain `abs(H_ii)`.
Neither uniform forcing nor its voltage-reference arrangement follows from
the statement that all channels were characterized in PBS.

A permutation of channel names does not change an equally weighted,
permutation-invariant aggregate. Actual ADC grouping, row history and filtering
can change the constituent gains. Thus the minimum information for a partial
shared-parameter fit is the amplitude extractor and channel-summary operator,
plus enough stimulus/reference and static-or-scanned acquisition information
(or source-supported bounds) to predict that operator. The dispersion cannot
be fitted as an SD until its definition is established. Equivalent evidence
for the effective operator suffices; complete software/wiring provenance is
not a new requirement. Rat5 mapping is a separate source dependency.

Without that binding, a free effective centre fitted to 0.78 only restates the
published scalar: contact loading, coupling, intrinsic gain, settling and DAQ
attenuation remain confounded. It neither identifies the six structural
templates nor merits an optimizer job. The next action is to bind the usable
observation operator and shared parameter combinations from the minimum
evidence above, then use the existing partial-fit route and calibration
dependency below. Do not choose a convenient historical statistic, assume
unity DAQ response, or replace required reference checks with a scalar fit.

Fixed architecture, counts and nominal dwell/rates constrain these operators;
they are not additional fitted observations. The held-aside full-mode noise and
distant-coupling observations retain their roles and are not used to complete
an unidentified calibration operator or select a preferred model.

The source record states a representative **3-dB frequency above approximately
1 kHz**. This is an approximate reported lower bound, not an exact measured
cutoff or a complete response curve. Separately, the existing policy's planned
acceptance threshold is **at least 800 Hz**. The latter is a check tolerance;
it does not replace the source statement with an 800-Hz measurement.

For the explicitly conditional, monotone first-order **whole-path** response

```text
H(f) = G / (1 + j 2 pi f tau_eff), with G != 0 and tau_eff > 0,
r(f) = abs(H(f)) / abs(H(0)),
f_3dB = 1 / (2 pi tau_eff).
```

a nominal reading `f_3dB >= 1000 Hz` gives
`tau_eff <= 1/(2 pi 1000) s`, approximately **159 µs**; equivalently,
`r(1000 Hz) >= 1/sqrt(2)`. These are conditional transformations of the
reported approximate lower bound, not exact confidence limits. Applying the
separate planned 800-Hz acceptance threshold to the same reduced response
would instead give approximately **199 µs**. Neither number identifies an
individual resistance, capacitance, CPE parameter or the R1 internal source-line
time constant. R1 retains separate contact and DAQ responses, so its complete
transfer need not have this single-pole form. A blocking contact, multiple poles,
zeros or nonmonotone response requires the full normalized transfer and its
correct passband/rolloff definition; the single-frequency inequality is not a
general replacement for that response.

This constraint leaves an allowed set, not a uniquely fitted time constant,
gain or noise model. Do not pick the boundary, its midpoint or another arbitrary
point as a calibrated device. A partial numerical transfer fit can next evaluate
the predeclared whole-path response against an explicitly bound, usable subset
of these calibration operators and retain the compatible parameter set. It may
report unresolved components and profile bounds; it may not emit an accepted
reference or silently omit required checks. The single bandwidth inequality
alone needs no optimizer job or additional simulation sweep.

Before that numerical fit, instantiate the actual operator(s), shared effective
parameters and any evidence-supported nuisance alternatives in
`model_specification`; satisfy the existing `reference_fit` dependency on the
outcome-blind calibration report for the quantities being fitted. No new generic
qualification program is introduced here. Acceptance and candidate scoring
still require every assigned calibration/held-aside check. Treating an unknown
historical statistic as a selected convenient convention would change that
evidence requirement and must be recorded explicitly as an amendment, rather
than presented as a recovered acquisition fact.

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

### Explicit interval-to-readout component

[acquisition_readout.py](tools/acquisition_readout.py) now supplies a bounded
`AcquisitionReadout` with three explicit plans: `RawConversion`,
`ReturnedSample` and `Observation`. The caller provides aperture endpoints,
conversion completion, assigned timestamps and availability times separately;
there is no inferred clock, channel map, oversampling factor or legal NI recipe.
The `RawConversion` aperture is explicitly **uniform and rectangular** in this
component. A nonuniform physical aperture or decimation filter requires an
explicit upstream model or separately specified readout; the rectangular rule
is not asserted to describe NeuroCam's unknown configuration.

For a finite open-loop recording, use one common timeline for piecewise-
constant field/row/mux actions and every declared aperture endpoint. Split a
physical interval at those exact supplied endpoints, apply its explicit switch
charge only on the first resulting segment, and advance the complete circuit
through gaps and discarded samples. A split is a numerical/acquisition
boundary, not another physical switching event. Source or hold-node means can
be assigned to readout inputs only through an explicit input/ADC binding.
An ADC with an assembled hold state cannot instead read a source directly and
silently bypass its modeled memory.

Validate the finite action plan, acquisition coverage, input bindings and
resource limits before mutating analog state. During execution, expose only
observations published by the reached physical time. A later final publication
horizon permits reporting already acquired values but does not advance the
physical circuit or reset it. Keep those times separate in terminal metadata.
If a propagation or readout failure occurs during execution, preserve the
failure and completed-prefix evidence and refuse further advancement of that
attempt. Do not retry a committed physical segment. This execution interface
does not implement an adaptive scan policy or certify actual hardware timing.

[`ArrayAcquisition`](tools/array_acquisition.py) implements this finite interface
with explicit `PhysicalInterval`, `ArrayInputBinding` and `AcquisitionLimits`
records. It owns advancement of the supplied array, copies the validated
field/charge/switch plan, and creates its own readout. `advance_next()` advances
one exact segment and returns newly published observations; `run()` executes
the declared finite plan, then `finalize()` publishes already acquired values
at the explicit final horizon. Execution failures, including interrupts, close
the attempt permanently. An interrupted array call can have unknown internal
work or commit state; only returned diagnostics enter its operator-work total.
The maximum local solver consistency estimate is not a global error bound.
Plan counters and retained numeric bytes are bounded, without storing a full
analog state vector per segment or claiming peak process memory.

Three [focused executor checks](tools/test_array_acquisition.py) passed in
**0.792 s** in allocation **46864089**, following independent static review.
[Direct evidence](runs/component_checks_job_46864089/test_array_acquisition.txt)
records all three cases. Independent analytic references
cover hold-capacitor exponential decay through unsampled gaps and a symmetric
two-node source mean. Other cases exercise exact aperture splits, once-only
boundary charge, input copying and prevalidation, source/hold binding, delayed
publication and permanent failure after physical advancement. These fixtures
do not establish calibrated DAQ behavior or production-record throughput.

The caller first advances its persistent analog/conditioning model on the
union of row, input and aperture boundaries, then supplies exact physical
interval means at the ADC inputs to `add_interval`. The accumulator sums
`mean × duration` across each whole aperture and divides by its duration.
It rejects gaps, overlaps, partial-interval means and nonfinite physical values
before committing any part of a shared update. It never prorates an interval
mean over an unobserved subinterval. Advancing unsampled states and applying
R3 conditioning or sample-and-hold loading remain the caller's responsibility;
passing an unconditioned array source mean does not complete R3.

An explicit `QuantizerTransfer` supplies every threshold, decoded code voltage
and input range. Each raw aperture mean is quantized and clipping is recorded
**before** the supplied returned-sample weights/divisor are applied. Optional
returned-word requantization is explicit. Observation arithmetic follows the
returned-sample stage. Every declared contributor supplies timing and validity
support, including zero-weight entries; there is no silent dropping or
renormalization. If any contributor is invalid, the published value is absent
(`None`) while validity, clipping, timing and resource counts remain available.
Raw codes remain terminal diagnostics rather than inputs to the causal query.

`observations_available_by` exposes only observations whose contributing
conversions and returned samples finish before declared publication.
`observations_for_history` additionally selects plans whose **entire**
contributing apertures lie in `[t-H,t]`, before evaluating any values. Thus a
delayed deadline cannot admit post-target input or require an out-of-window
conversion to be integrated. Nominal timestamps alone never grant access.
This is an observation access rule, not an inverse implementation or isolation
of the analog world's hidden parameters.

`finalize` counts every declared raw conversion exactly once, including unused
settling/startup, marker/reference and configured auto-zero/chop work. It
separately reports hardware-returned samples, retained pixel observations and
valid retained pixels, with rates over the explicit acquisition window. Known
observation payload/overhead bits describe only the declared output stream;
missing terms keep its total unknown. Hardware transport bits, conversion
energy and power remain unknown. This component does not certify resource
parity, ADC behavior, front-end noise or a calibrated reference.

Explicit limits bound conversion, returned-sample, observation and interval-
update counts. These are finite-plan limits, not a full-duration throughput or
peak-memory qualification; underlying analog state must persist across any
bounded plans. Three [focused checks](tools/test_acquisition_readout.py) cover
quantize-before-average, missingness/clipping and all three work counts, atomic
aperture refusal, and publication/history access without reading future values.
They passed in **0.001 s** in allocation **46864089**;
[direct evidence](runs/component_checks_job_46864089/test_acquisition_readout.txt)
records all three cases. No device parameter sweep was introduced.

### October 9: external marker/reference ports in the same ADC groups

The full-array source-stream recipe also needs the published G1 marker. The
earlier passive mux accepted only array source indices, so it could not represent
that signal's electrical path. The explicit extension in
[daq_mux_hold.py](tools/daq_mux_hold.py) keeps neural sources at 0–63 and adds
named external voltage inputs to existing physical ADC groups. It adds no ADC
or hold capacitor merely because a marker is present. Every external switch
path counts in the assembled branch inventory.

`ExternalVoltageInput` declares the track and off conductance of one passive
path. An immutable `DAQGroupDrive` supplies the selected neural/external input
or all-off state, every declared external voltage (including off inputs), and
the array's fixed voltage-reference identity. Reusing an external input name
across groups denotes the same physical source and requires the same voltage.
All voltages and changes in their waveform remain explicit conditional inputs;
neither historical marker levels nor actual NI circuitry are inferred.

For one constant interval of duration `T`, a port at voltage `u` through
conductance `g` into the existing hold node `h` adds `g` to its conductance
diagonal and `g*u` to its current drive. Neural mux paths retain their off
conductance while an external input is selected, and external off paths remain
present while neural inputs or all-off are selected. Hold voltage and the full
array/contact state persist. A finite-resistance voltage change creates no
instantaneous capacitor charge jump; any separately modeled switch injection
still uses the existing once-only charge input.

The physical external-source work is
`T * sum(g*u*(u - mean_hold_voltage))`, including signed energy returned to a
source. The array reports `daq_external_source_interval_work_joule` separately
from cortical-field work and the kernel's Norton-current work. Complete
represented-circuit work minus stored-energy change can give total dissipation;
the interval mean alone does not identify each branch's quadratic heat integral.
These quantities do not establish total device power or thermal cost.

Eager and streaming acquisition validate every external value and reference
before the first physical advance. Existing immutable physical events carry
the drive through midpoint holds, aperture splits, storage seams and replay.
Marker conversions use the existing hold binding, actual ADC identity and
`marker` role, so acquisition counts and publication rules remain in force.
Both simulation accounting paths retain the separate external-source work.

The new frontend, full-array analytic and acquisition-integration checks are
prepared in [frontend tests](tools/test_daq_external_inputs.py),
[array tests](tools/test_cpe_external_inputs.py) and
[acquisition tests](tools/test_external_input_acquisition.py). Their execution
status belongs in the [experiment log](experiment_log.md). This extension does
not select a full-record parameter recipe, calibrate the reference, implement
active conditioning/noise/saturation or open candidate scoring.

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
