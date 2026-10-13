# One conditional full-array R3 cost scenario

This is the prospectively selected **conditional_R3_B2_2001_G0_001** engineering
case for a bounded execution-cost attempt. It reuses the saved development
world **B2, seed 2001**, the complete G0 4,096-pad/64-source geometry and the
entire **16-second** record. It is not fitted to NeuroCam anchors, an authentic
NI acquisition configuration, a qualified reference, a candidate score, or a
promise that the full record finishes within the cost allocation. No new
field/cohort draw is introduced. All electrical/noise values below are explicit
conditional assumptions; none comes from the numerical test fixtures.

The only constructor is the existing production `NeurocamCPEArray` inside the
streaming wrapper. [conditional_forward_recipe.py](tools/conditional_forward_recipe.py)
returns its complete parameter mapping and expected state order; it does not
instantiate a disposable duplicate. The geometric short axis is explicitly x,
using the already selected virtual G0 centres and one pad average. Physical
registration remains unverified. All voltages share
`conditional_R3_zero_reference`.

## Electrical parameters and ownership

| Quantity | Selected conditional value |
| --- | --- |
| Contact template | E-area, reference pad area 9,450 µm², gamma = 1 |
| Every contact | fractional alpha = 0.5; Q = 2×10⁻⁸ S·s^0.5; Gct = 5×10⁻⁹ S |
| Every access resistance | 2,000 Ω, owned once by the assembler |
| Positive CPE realization | 16 RC branches per pad; poles between 2π×10⁻⁴ and 2π×10⁸ rad/s |
| CPE declared forward band | 0.125–32,768 Hz; 257 logarithmic check frequencies; separate 5% finite-grid and analytic truncation refusal thresholds |
| Pad-to-reference capacitance | 2 pF per pad |
| Fixed source capacitance | 30 pF per source, excluding the additional conditioning input capacitance |
| Mutual capacitance | 0.02 pF between every horizontal/vertical nearest pad neighbour; 0.2 pF between adjacent source lines |
| TFT paths | on 2 µS; off 20 pS; all off-gate pads remain connected |
| Fixed leaks/loads | pad 1 pS; source 1 nS, excluding conditioning and DAQ loads |
| Every conditioning input | additional 5 pF and 1 nS |
| Every buffer | Cb = 100 pF; gb = 20 µS |
| Every output/filter | gout = 200 µS; Cf = 2.2 nF; filter leak = 1 nS |
| Active target | gain 8, offset 0 V, internal target rails ±50 mV |
| Every ADC hold | 20 pF; track path 1 mS; off path 100 pS; leak 100 pS |
| Initial state | every pad, source, contact, buffer, filter and hold at 0 V; prior field at 0 V |

There are **69,848 retained states**: 4,096 pads, 64 source lines, 65,536
fractional-contact states, 64 buffers, 64 filters and 24 physical ADC holds.
The capacitance matrix includes 8,127 declared mutual parasitics plus 65,536
contact capacitor edges. A common contact parameter object does not merge any
physical state or noise source. Contact states, analog recovery and the holds
persist through every row, mux, aperture and storage boundary. The finite CPE
realization can refuse its declared approximation checks; there is no automatic
branch-count/tolerance adjustment or contact-order fallback.

Buffer/filter dynamics are the existing jointly propagated conditional active
macromodel. Rails constrain its internal drive, not a hard output clamp.
Controlled-source terminal work and branch heat are recorded, with no claim
about supply efficiency, quiescent power or complete device energy. Contact
and noise-port physical work uses the existing eliminated-junction correction;
it is not replaced by reduced nodal work.

## Noise recipe and retained draw

The represented passive resistors are assigned **300 K**. The one-sided
conditional input spectra are 4 kB T G for each contact-star resistor and each
fixed pad leak, source load, conditioning input/output/filter resistor and hold
leak; access resistors use 4 kB T R as series voltage noise. These factors use
the actual assembled contact branches and their explicit ownership. No extra
contact RMS floor or mode-specific measured RMS is added.

Each amplifier has independent input voltage ASD **8 nV/√Hz** and independent
input current ASD **0.3 pA/√Hz**. One additional **2 nV/√Hz** common reference
latent enters all 64 amplifier input-voltage coordinates before gain/clipping.
This is a common input-referred voltage source, not 64 independent reference
draws or a second subtraction from surface truth. Original current-port work
and access-voltage work remain separate; amplifier input-voltage forcing
changes the already represented controlled-source work.

The finite spectral grid has bin edges
`[0.125, 0.5, 2, 8, 32, 128, 512, 2048, 8192, 32768] Hz`, with geometric
centres `[0.25, 1, 4, 16, 64, 256, 1024, 4096, 16384] Hz` and explicit
linear bin widths. This is **nine Gaussian spectral atoms**, including
out-of-band forcing above 100 Hz. It is not continuous white noise, an
accuracy-qualified spectral quadrature, or a demonstration of stationary
equilibrium from the selected zero initial state. Neither the field nor this
noise is first restricted to the scoring band.

There are 78,232 original physical coordinates and 78,233 named latent sources,
including the one common reference. The full recipe retains 704,664 sparse
factor entries and 704,097 pairs of standard normal quadratures. The latter
are drawn once using PCG64 seed **202010092001** in the existing canonical
frequency/name order. Every later field-hold, aperture, solver and storage
split replays the same draw. No split redraw or change in selected ports is
permitted. Future paired arms must reuse the saved basis and quadratures;
equal seeds on a changed basis would not establish pairing.

`build_conditional_noise` writes one exclusive, bounded NPZ before returning
the provider. It retains exact recipe constants, frequencies, widths, original
port names/units, latent names, COO indices and complex factors, all named draw
indices/a/b values, time window and chosen per-port A/V hold tolerances. The
file cap is 256 MiB; a failed partial write is preserved and is not a runnable
noise artifact. A seed or metadata summary alone is insufficient. The caller
supplies the durable attempt path and numeric execution bounds.

Explicit conditional omissions are biased-TFT noise, switching-dependent
neural/external mux noise, extra controlled-drive noise beyond the specified
amplifier input sources, supply noise/efficiency/quiescent power, mechanical
motion, electrochemical drift and device-population variation. The biased
switches are not silently assigned equilibrium spectra from their conductance.
These omissions mean this case is a cost scenario, **not a complete noise
inventory or accepted physical reference**. All deterministic on/off paths,
external-voltage loads and switch charges remain present.

## Full 24-ADC acquisition

[conditional_acquisition_recipe.py](tools/conditional_acquisition_recipe.py)
maps source s to ADC `s % 24`, slot `s // 24`: the first 16 groups have three
neural inputs and the last eight have two. ADC0's fourth external input is
G1; it shares the same retained hold and conversion work. G1 is +5 mV during
row 0 and 0 V otherwise, in the common reference. This is an explicit
conditional mapping, not recovered hardware wiring.

The record contains 4,000 four-millisecond frames. All 64 rows have 62.5-µs
dwells. The 10-µs source cycle contains four 2.5-µs slots. A conversion aperture
runs from slot+0.5 to slot+2 µs, is timestamped at its midpoint and completes
at slot+2.5 µs. Each returned neural word is the corresponding raw conversion
without digital modification. Every source has 100,000 words/s. The shared
quantizer uses 18-bit uniform bin centres over ±10 mV.

The selected transport payload is 32 bits: a pixel uses the uint32 sum of its
three 18-bit raw codes with known divisor three and bin-centre decoding; a
marker carries its 18-bit raw code in uint32. This introduces no float32
rounding or extra output quantizer. Stored HDF5 values remain float64;
packet/validity overhead is unselected and is not claimed to be zero.

The third through fifth words in each pixel dwell are averaged and published
at row end, timestamped at the fourth word's midpoint. Every G1 word is a
separately published marker. No further filtering, averaging, chopping or
auto-zero was selected; this is not a historical absence claim. Every new
row injects source charge `(-1)^s × 3.5×10⁻¹⁷ C`; mux holds receive
`(-1)^slot × 2×10⁻¹⁷ C` only when selection changes, including transitions
to None. Charges are not repeated at numerical/storage splits.

The complete record has **104,000,000 raw conversions and returned words**
(102,400,000 neural + 1,600,000 marker), 16,384,000 pixel observations,
1,600,000 marker observations and 50,752,000 aperture-support entries. It has
6,400,000 physical actions and at least 19,200,000 physical/aperture splits.
Streaming retains physical and digital dependencies rather than materializing
all these Python records together.

## Finite cost attempt and limits

The selected attempt uses one CPU, 7 GiB and two hours, with BLAS/OpenMP threads
fixed to one. Existing 88 checks precede the one substantive full-record
attempt; no new simulated validation panel is selected. The attempt is
designed to retain incomplete operational evidence if its time, memory,
solver, input-hold, evaluation or output cap is reached. It does not shorten
the registered record, drop contacts, omit failed ports, retry with weaker
tolerances or silently reduce conversion counts.

The circuit constructor caps retained numeric storage at 512 MiB (70,000
states, 75,000 capacitance edges and 70,000 contact-star terms). Physical-noise
layout numeric storage is capped at 16 MiB/80,000 ports; spectral numeric
storage at 128 MiB/705,000 factors/705,000 draws. These are retained numeric
payload limits, not Python object or peak RSS estimates. Several temporary
copies, sparse factorization/workspace and named-draw objects remain subject
to the allocation's actual memory bound. The likely memory scale is GiB,
but no full-size RSS or runtime has been measured.

With 64-byte identifier fields, the selected complete schedule's tables already need
approximately **18,126,400,000 bytes**, or **18,646,400,000 bytes** at a
40-million interval cap, before complete states, JSON and HDF5 overhead.
The selected driver caps table payload at 24 GiB and 212,736,000 records,
with a 32-GiB total storage reservation including the separate 256-MiB-capped
noise artifact. Final work/operator totals must reflect
actual propagation, with uncompleted portions remaining uncompleted.

Whole-plan preflight, spectral construction/evaluation and input-hold
compilation can dominate the cost before any R3 state advance. The original
Python spectral loop would require roughly 4.5×10¹² term operations for one
evaluation at each physical action, before replay passes. Exact bounded
implementation optimizations may remove repeated name searches and identical
bound scans; they do not establish full-record throughput. A completed
preflight is not an analog result. Numerical A/V hold tolerances, solver
convergence controls and global work/deadline caps are explicit run inputs;
this recipe does not infer them from noise RMS or claim a circuit-output
error bound from an input-hold bound.

### Selected cost-driver controls

The [cost driver](tools/run_conditional_forward_cost.py) uses a 6,000-second
inclusive setup/preflight/propagation budget, a 6.5-GiB observed peak-RSS cap
checked at most twice per second, and in-process phase/elapsed/RSS messages
at most once per minute. The enclosing shell allows 6,100 seconds plus
termination grace for closeout. It opens only the saved `B2_seed_2001` group
and retains `settings.json` before world loading.

Selected input tolerances are 10-nV field/voltage-noise holds and 1-pA
current-noise holds. Joint solver settings are relative tolerance 10⁻⁵,
voltage absolute tolerance 10⁻⁸ V, charge tolerance 10⁻²⁰ C, at most eight
partition refinements and 200,000 operator products per interval. Linear
settings are relative tolerance 10⁻⁹, charge tolerance 10⁻²² C, 12 Newton
iterations per substep, 10,000 shared linear iterations per interval,
GMRES restart 40 and 512-MiB numeric workspace; the energy tolerance is
10⁻¹⁸ J. These are numerical-consistency controls, not empirical accuracy
criteria. Nanosecond-scale high CPE poles may cause an unresolved convergence
stop with the finite refinement allowance; no fallback changes the model.

Global limits are 40 million holds/split intervals, 100 million hold-planning
work units, one billion operator products, 200 million spectral evaluations
and 2×10¹⁴ spectral term evaluations. There are 4,000 windows; live records
are bounded to 52,000 raw, 52,000 returned and 9,000 observations, with
128-MiB window and 64-MiB readout numeric-plan limits. Writer buffering is
8,192 records, with at most three raw dependencies per observation.

Noise-binding time, its maximum normalized derivative and a conservative
midpoint-radius limit are recorded to explain an early hold-planning refusal.
The actual compiler still applies its outward-rounded radius calculation;
an ideal symmetric length alone is not an extra approximation certificate.
`cost_report.json` distinguishes setup failure, whole-plan preflight and
confirmed state advancement. A hard-killed incomplete container is never a
completed result.

## Measured constant-hold obstruction and proposed cubic forcing

The October9 full-size attempt retained the complete selected noise realization
but did not reach electrical propagation. Its global normalized first-derivative
bound was L1=168497465.6238511 per second. The present constant-hold predicate
requires outward(radius*L1)<=1, so16seconds necessarily needs at least
ceil(8*L1)=1,347,979,725holds. The selected cap is40million. See the
[actual attempt and cancellation](experiment_log.md#2026-10-09-1026-pacific--model_specification-full-size-noise-bound-proves-selected-hold-budget-infeasible).
This is a limitation of that numerical representation, not a physical device
finding or a lower bound on other integration methods.

A source-reviewed next method is a cubic Taylor representation of the SAME
frozen nine-tone trajectory on each original2.5microsecond physical slot.
For midpoint m, preserve P(t)=sum(j=0..3) x^(j)(m)*(t-m)^j/j!, where each
coefficient is derived from the retained complex spectral amplitudes and
frequencies. Per-port analytic remainder is at most M4,p*r^4/24; with the
existing derivative envelope, a common normalized bound is
L1*omega_max^3*r^4/24. At nominal radius1.25microseconds this is approximately
0.0187, below one. Actual admission must use outward arithmetic and actual
represented endpoints. This preserves the selected1pA/10nV analytic forcing
accuracy, with the same existing exclusions for coefficient/evaluation
floating-point error. It does not establish circuit-output accuracy.

The cubic path is now implemented, with compute checks pending. It evaluates
the polynomial at each internal solver midpoint,
including input-referred voltage noise before amplifier clipping, and integrate
all nodal, original noise-port, access-voltage and external/controlled-source
work at the same time/state points. Products of separate interval means are
not an acceptable substitute. Endpoints, aperture means and all physical work
must participate in refinement comparison before state commit. The same
slot-origin polynomial must survive aperture/storage splits; its four original
physical-port coefficient rows are bound during replay. It retains bounded
O(4P+4N) live numeric payload, not a dense port covariance. The existing raw
field/mux/reference rules and original noise archive remain unchanged.

The selected repair attempt is `conditional_forward_cost_attempt003`, using
the same 1-CPU, 7-GiB, two-hour envelope and full record. The driver explicitly
selects `--noise-representation cubic` and loads attempt002's saved NPZ with
no RNG call. Archive and member sizes, array dimensions, original port/latent
basis, factors, source recipe and tolerances are checked before use. The
original archive remains the retained realization; no second draw is created.
Each cell uses exactly the repeating acquisition schedule's endpoint
arithmetic. All four coefficient rows, origin and support enter the existing
provider replay comparison. Acquisition checks support for the whole interval
before advancing, and supplies the absolute start of every aperture split.

Affected checks run within the substantive allocation, followed by the full
case only if they pass. This is one measurement with no automatic retry;
the process timer and bounded outer timeout preserve a normal failed report
where possible. The earlier constant-hold guard remains active only when that
representation is selected. Its infeasibility bound is reported as context,
not applied incorrectly to the cubic method.

Removing this proven constant-hold obstruction would still leave field-hold
refinement, at least19.2million physical/aperture intervals, stiff integration,
whole-plan preflight and actual output cost to measure. No full-record timing,
reference calibration or candidate result follows from this proposal.
