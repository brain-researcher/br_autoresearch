# Sources used in Episode 20

EP20 combines four kinds of evidence that answer different questions:
published NeuroCam measurements, simulated cortical fields, device and
electronics models, and a physical-signal or phantom plausibility check. They
must remain separate. A public paper is not automatically a calibration source,
and ordinary ECoG is not dense ground truth for an unbuilt electrode geometry.

No source has yet been prepared as an EP20 input, and no candidate has been
scored.

## Source roles

| Source | Role | Current state |
| --- | --- | --- |
| NeuroCam article and supplement | Define published architecture and characterization anchors | Identified; EP20 use and rights still need approval |
| Calibration subset of published anchors | Fit a paper-derived NeuroCam model | Design roles assigned below; measurement context/use still unresolved |
| Held-aside published anchors | Check that the fitted reference reproduces measurements not used for fitting | Design roles assigned below; no check has run |
| All reference models that pass both checks | Test whether candidate rankings are identified by the published measurements | Starting templates/shared fit specified; acceptance set not constructed |
| Development cortical-field and device models | Develop designs and run challenge tests | Not implemented |
| Independent cortical-field and electronics models | Rerun frozen D0–D3 representatives and test the rule under every accepted reference mechanism | Not implemented |
| Development physical-signal or phantom source | Stress spectra, amplitudes, missingness, and software stability | Not selected |
| Separate final physical-signal or phantom source | Reject catastrophic implausibility after design selection | Not selected |
| Design and resource rules | Decide which designs are legal and resource matched | Starting pad/scan catalogue selected; physical eligibility and full accounting remain unresolved |
| Fabricated candidate measurements | Future physical validation | Outside EP20 |

## Documentary NeuroCam reference

| Item | Value |
| --- | --- |
| Device | NeuroCam flexible multiplexed micro-ECoG array |
| Final article | Xie et al., “High-resolution spatial mapping of electrocorticographic activities with NeuroCam: a 4096-channel, multiplexed flexible thin-film transistor array” |
| Journal | Science Bulletin 70 (2025), 4133–4137 |
| Final article DOI | `10.1016/j.scib.2025.11.030` |
| Publisher identifier | `S2095927325011582` |
| PubMed | `PMID:41309324` |
| Preprint DOI | `10.1101/2024.08.18.608446` |
| Supplement status | Observed with an author-hosted combined article-and-supplement file; the publisher's separate supplement identity still needs confirmation |

The final article is the scientific citation. The preprint can clarify version
history but cannot silently replace the final methods. Before any values are
used, EP20 must record the article and supplement versions, terms of use, page
or figure source, original units, and whether a number was stated directly or
read from a plot.

## Published architecture and measurements

| Property | Published value |
| --- | --- |
| Lattice | 64 × 64 pixels; 4,096 total |
| Pitch and active area | 150 micrometres; approximately 9.6 × 9.6 mm |
| Gold sensing pad | Approximately 70 × 135 micrometres |
| Substrate and device | 25-micrometre polyimide; Ln-IZO TFT |
| Addressing | 64 gate and 64 source lines |
| Representative gain | 0.78 ± 0.12 |
| Representative channel bandwidth | 3-dB frequency above approximately 1 kHz |
| Illustrated row dwell | Approximately 62.5 microseconds |
| Array thickness | Less than 30 micrometres |
| Representative transistor | Mobility 15.13 cm²/V/s; threshold 2.7 V; on/off ratio approximately 6 × 10⁶ |
| Gate drive | Approximately +6 V on and −2 V off |
| Device impedance | Approximately 100 kilo-ohms over 1–100 kHz; Fig. S2 specifies Vgs = 6 V, not a bare-contact 1–100-Hz measurement |
| Within-dwell aggregation | Stable third through fifth samples from six or seven raw samples, as described in the supplement |
| Static high-precision setting | 23 signal source lines plus one ADC marker channel; lowest stated noise 2.3 ± 0.3 microvolts RMS at 100,000 samples/s |
| First source-line neighbour crosstalk | Approximately −12.6 dB |
| First gate-line neighbour crosstalk | Approximately −37.2 dB |
| Second neighbour and farther | Approximately −45 dB |

### Published dynamic settings

| Setting | Pixels | Effective rate per pixel | Reported RMS noise | Approximate digitized-source samples/s |
| --- | ---: | ---: | ---: | ---: |
| 24 source × 16 gate | 384 | 1,000 samples/s | 8.8 ± 4.2 microvolts | 2.4 million |
| 24 source × 64 gate | 1,536 | 250 samples/s | 12.6 ± 8.4 microvolts | 2.4 million |
| 64 source × 64 gate | 4,096 | 250 samples/s | 63.6 ± 24.1 microvolts | 6.4 million |

The corresponding effective field outputs are approximately 384,000, 384,000,
and 1,024,000 pixel samples/s. Reference, ground, guard, power, and external DAQ
costs are counted separately from the 128 array lines.

The estimates previously labelled raw conversions are `source count × 100 kS/s`
digitized-source-stream proxies, not identified internal ADC clocks. NI's
[official PXIe-4309 allocation explanation](https://knowledge.ni.com/KnowledgeArticleDetails?id=kA0VU0000009lMH0AY&l=en-US)
distinguishes requested sample rate from internal conversions and channel-
sharing overhead. EP20 still needs the actual configuration, marker work and
conversion accounting before asserting physical-resource parity.

## What the paper does not determine

The article and supplement do not provide a complete generative electronics
model. Important uncertainties include:

- the exact meaning of every reported `±` value;
- a discrepancy between 10-mVpp and 1-mVpp frequency-response tests;
- whether the stated 6.25 oversampling ratio describes within-dwell sampling or
  the ADC itself;
- exact channel identities and marker meanings in reduced modes;
- a conflicting DAQ input/output description involving three PXIe-4309 cards
  and one PXIe-6739 card;
- how crosstalk changes outside the reported air, single-pixel, 5-mVpp,
  12-Hz test;
- the mismatch between a reported 1–199 Hz filter and the approximately
  125-Hz Nyquist limit of a 250-sample/s full-array setting; and
- absent raw I-V/C-V traces, compact device models, PDK, netlist, exact layout,
  DAQ code, power traces, and independently held-out devices.

Unknown quantities must become explicit uncertainty ranges or blocks. They may
not be guessed or treated as zero. Noise values measured with different
bandwidths, filters, channel counts, or within-dwell summaries cannot be pooled
without a source-supported conversion.

## Building and checking the paper-derived reference

Published values are divided before fitting into two roles:

- **calibration:** values used to fit the reference model; and
- **held-aside check:** at least one operating setting or response-curve region
  not used for fitting, used only to test whether the fitted model reproduces
  the paper within pre-set tolerances.

### Compact anchor assignment — selected design

| Published observable | Assigned role | Purpose |
| --- | --- | --- |
| Architecture, channel counts, dwell and nominal rates | Fixed inputs/implementation checks | Recreate the operating settings, not evidence of independent prediction |
| Fig. 2a PBS array gain and Fig. 1f single-gate frequency response | Calibration | Constrain transfer; keep their measurement contexts distinct |
| Fig. S3b static 23-signal-channel noise; Fig. S4 dynamic 24×16 and 24×64 noise | Calibration | Constrain complete static/scanned predictions with their original histories and aggregation |
| Fig. S5 nearest source-/gate-line crosstalk | Calibration | Constrain directional coupling under the original air/12-Hz test |
| Fig. S4 entire dynamic 64×64 noise summary, 63.6 ± 24.1 µV | Held-aside from fitting | Predict full-array noise centre and dispersion without a full-mode residual fit |
| Fig. S5 second-neighbour/farther crosstalk, approximately −45 dB | Held-aside from fitting | Check spatial decay without fitting a distant-coupling floor |
| Fig. S2 gated high-frequency impedance and other unassigned curves | Context only | No bare-contact low-frequency substitution or additional fitting targets |

Source: [NeuroCam final article/supplement, Fig. 1f/2a/S2–S5 and noise methods](https://shengxingstars.github.io/www/images/publications/2025_High-resolution%20spatial%20mapping%20of%20electrocorticographic%20activities%20with%20NeuroCam%20a%204096-channel,%20multiplexed%20flexible%20thin-film%20transistor%20array.pdf).
These public values have already been inspected: this is a **paper-consistency
check held aside from fitting**, not blind or independent-device validation.
Freeze shared physical laws before fitting; no normalization or source-count
penalty may be tuned to held-aside summaries. Calibration covers primarily the
24-source regime and does not identify an arbitrary 64-source noise law.
The bandwidth, `±` meanings, transfer-amplitude discrepancy and crosstalk
denominator remain unresolved; reproduce each original statistic/context,
without pooling RMS or treating `±` as a standard deviation by assumption.
Unusable required observables leave reference acceptance unresolved, not passed.

The [minimal shared fit](outputs/electrical_model_and_probes.md) separates
effective transfer/noise/coupling constraints from unidentified circuit
parameters. The two 24-source modes have the same nominal row-dwell rate:
their noise difference does not isolate a switching-rate coefficient.

Candidate design results cannot be used to choose this division or repair a
failed check. Planned tolerances are:

| Check | Tolerance |
| --- | ---: |
| Pixel, line, and setting counts | Exact |
| Sample rate and throughput | 2% relative error |
| Representative gain centre | 0.05 absolute error |
| Gain dispersion | 0.03 absolute error |
| Representative cutoff | At least 800 Hz |
| Noise centre | 20% relative error |
| Noise dispersion | 30% relative error |
| Nearest-neighbour crosstalk | 3 dB absolute error |
| Distant crosstalk | No greater than −42 dB |
| Mode trade-off ordering | Exact |

The check also covers scan timing and skew, history-dependent settling,
saturation, quantization, missing pixels, conditional noise, and rejection of
illegal designs. The numeric tolerances require outcome-blind synthetic
calibration and scientist approval.

If aggregate published values cannot distinguish plausible from clearly wrong
scaling laws, EP20 needs raw traces, additional devices, or a wider explicitly
uncertain model family. It must not describe the result as an exact NeuroCam
digital twin.

### Can equally credible reference models reverse the answer?

A successful held-aside check is necessary but not sufficient. Every
predeclared noise, impedance, pad-averaging, settling, and crosstalk scaling law
that passes calibration and the held-aside tolerances remains in an accepted
reference-model set. Before full optimization, the same small sentinel panel of
uniform, two-scale, and revisit candidates is evaluated under every accepted
model using the same fields and resource budgets. After development, one
representative or deterministic selection rule for each D0–D3 family is frozen
across the full accepted-model set and the rank-stability check is repeated.

A **material rank reversal** occurs when two accepted models prefer different
unique designs and, within each model, the simultaneous one-sided lower bound
for the winner versus its runner-up exceeds 0.005. If such a reversal occurs,
published aggregate measurements do
not identify a candidate. Report which unmeasured response—such as pad-area-
dependent noise, impedance, history-dependent settling, or crosstalk under
multiplexed scanning—would most reduce the disagreement. Candidate search stops
until an added measurement resolves the reversal. Reporting one optimum per
model is a sensitivity analysis, not an EP20 design nomination.

## Development cortical fields

Development must cover these field families:

| Family | Required variation |
| --- | --- |
| Broad correlated field | Temporal and spatial spectrum, correlation length, anisotropy, amplitude, common mode |
| Focal field transient | Size, duration, amplitude, event rate, location, spectral envelope |
| Translating or expanding wave | Direction, speed, wavelength, curvature, onset, boundaries |
| Multiscale mixture | Broad/local energy, dependence, overlap, nonstationarity |
| Interacting sources | Number, separation, synchrony, phase, collision rule |
| Silent-region surprise | Pre-event silence, unexpected location, amplitude, duration |
| Nuisance family | Line noise, reference drift, motion-like transients, gaps, dropout, gain drift, saturation |

Each range needs an empirical source, a physical bound, or an explicitly
conservative design interval. All candidates see the same field and device
draws; unfavourable worlds are not regenerated.

The [starting catalogue](outputs/parameter_candidate_catalogue.md) selects
finite field controls informed by NeuroCam, Rogers et al. (2019), Zhang et al.
(2018), and González-Ramírez et al. (2015), with direct primary citations and
recording contexts. It keeps reported observations distinct from deliberate
challenge values and measures lambda/tau from generated truth. The 28 starting
conditions are within-trial worlds, not an additional design-trial allowance.

Each condition must expose the following **absolute inputs before any
reconstruction result**:

| Input | Required definition and unit |
| --- | --- |
| Spatial 90%-power scale `lambda_k` | `1/(2*k90)`, where `k90` is the smallest radial spatial frequency containing 90% of common-mask 1–100 Hz truth or known-input pilot power; report micrometres and common-grid pixels |
| Temporal 90%-power scale `tau_k` | `1/(2*f90)`, where `f90` is the smallest temporal frequency containing 90% of demeaned truth or known-input pilot power; report seconds |
| Input-referred noise `noise_mdk` | Equal-site pooled RMS microvolts over common panel P during the zero-input 1–100-Hz probe, for every accepted model *m*, condition *k* and frozen D0–D3 representative *d* |
| Interface impedance `impedance_mdk` | Maximum over P of each site's discrete Q90 contact magnitude in ohms at equally weighted integer 1–100-Hz tones, for every *m,d,k* |
| Settling `settling_mdk` | Worst P/history microseconds for the normalized conditional-mean unit-step response to enter and remain within 1%, for every *m,d,k*; unresolved tails are censored |
| Crosstalk `crosstalk_mdk` | Largest same-victim input-equivalent adjacent-line ratio `abs(H_ij/H_ii)` over P victims, physical neighbours and registered 1–100-Hz tones/phases for every *m,d,k*; dB supplementary |

The first two quantities describe the field, not a candidate, and must come
from simulated truth or a separate known-input pilot. The last four are
candidate-indexed characterization tables measured independently or emitted by
the frozen device model before reconstruction. They are not differences from a
post hoc "proposed design," and none may use reconstructed R². Development and
final implementations must use the same definitions, probe waveforms, duration,
mask, and units. The [electrical model/probe note](outputs/electrical_model_and_probes.md)
now selects those finite protocols and scalar aggregations. Freeze P before C1
scores as the verified maximal intersection of published-setting neural readout
masks; do not drop failed sites or exclude off-gate aggressors. These are panel
characterizations, not full-array maxima. The actual mapping and absolute probe
tables remain unavailable; unknown/zero-gain responses cannot pass cost gates.
Development and
independent condition grids must provide at least five pre-outcome conditions
in each S/T cell under each accepted model; the counts may be increased after
outcome-blind interval-width calibration. Without these inputs and cell support,
EP20 can rank devices in simulations but cannot test a transferable design rule.

The primary target is voltage on a common cortical-surface grid. Latent-source
recovery is descriptive only.

## Source-to-surface, device, and electronics models

The [component design](outputs/component_details.md) separates common ideal
cortical-surface truth from contact voltage, one pad-sensitivity operation,
causal electronics state, raw conversions and retained timestamped samples.
Contact gaps affect observations, not the design-independent surface target.
Published high-frequency impedance does not populate the required 1–100 Hz
characterization input by extrapolation. These are model-interface and
uncertainty clarifications, not newly available measurements.

The [electrical note](outputs/electrical_model_and_probes.md) selects passive
Randles/CPE area and sublinear-area contact laws, each paired with minimal,
coupled-array and coupled-array-plus-DAQ persistent-state alternatives. All six
templates share physical parameters across candidates and derive their noise
from the same circuit. They are proposed mechanisms; fitting and held-aside
acceptance remain unperformed. It also separates internal converts, returned
source samples, retained pixels, integration apertures and publication times.

Development includes both an analytic homogeneous or layered source-to-surface
model and a heterogeneous or anisotropic alternative, with distance, curvature,
reference, contact-gap, and conductivity variation.

The [compact starting ranges](outputs/component_details.md) select cortical
conductivity 0.2/0.4/0.6 S/m, source depth 0.1/0.5/1.5 mm and gap 0/50/200 µm,
with conductivity-anisotropy sensitivities. They are selected world scenarios,
not measured NeuroCam values; initial one-factor variations avoid a full grid.

The device models vary together where physics couples them: pad averaging,
interface impedance, source loading, line RC, settling, switching transients,
crosstalk, noise, ADC sampling, aggregation, quantization, drift, saturation,
missing pixels, contact gaps, and voltage reference. Each rule must be marked as
measured, literature-derived, physics-derived, or deliberately adversarial.

The independent test reruns one frozen representative or deterministic
selection rule for each D0–D3 family, plus the required C4 controls, under every
accepted reference-model mechanism on separately implemented cortical-field,
source-to-surface, and electronics models. The four absolute characterization
values are recomputed before reconstruction. Merely changing random draws in
the development implementation is not an independent structural test.

## Physical-signal or phantom checks

One legally reusable source type must be selected before design scoring:

1. dense ECoG with known geometry, timestamps, reference, hardware, sample
   rate, preprocessing, and raw-source history; or
2. a known-input saline or phantom experiment with spatial input map, raw array
   outputs, scan markers, operating setting, and separate final traces.

Ordinary ECoG has already been spatially filtered by its contacts and reference.
It can reveal implausible spectra, amplitudes, missingness, or software
behaviour, but it cannot rank unbuilt electrode geometries. A reference-device
phantom can test more of the electronics chain but still cannot establish
candidate physical superiority.

Development and final plausibility checks use different sources or whole
subject, device, session, or waveform partitions. No overlapping window or
normalization statistic crosses between them. The catastrophic-failure
threshold is chosen before candidate scoring.

## Design and resource rules

Before candidate scoring, the study needs finite rules for pad dimensions and
clearances, transistor and via exclusions, trace loading, dwell and scan order,
active source banks, front-end and ADC allocation, conversions, bitrate,
latency, skew, modeled power and thermal burden, routing, area, and yield or
dead-pixel effects.

Selected virtual pad classes are 70 × 135, 50 × 95 and 35 × 70 micrometres.
The catalogue defines their balanced maps and scan-event sequences, including
coverage costs. Five-micrometre rounding and pitch-derived metal gaps are
design arithmetic, not a published PDK. Alternative-pad via landing/keepout
rules, coupled electrical response and physical resource costs remain unknown.

If no NeuroCam PDK or netlist becomes available, a scientist must approve
conservative substitute rules supported by published anchors or explicit safety
factors. Modeled power and thermal quantities remain proxies, not measured
equivalence.

The same rules determine which C0, C1, and C4 comparisons are eligible at each
resource budget. A shared process name or a short thermal image is not enough
to assign an unknown candidate the same physical cost as the reference.

## Data separation during the study

Design development may see documentary facts and aggregate results from the
development cortical-field and device models. It cannot see the independent
final models, their parameters, field traces, previews, ranks, or candidate-
specific quality summaries before one design is selected.

The paper-reference fitter uses only calibration values. The held-aside check
receives the fitted reference and both assigned subsets, and returns pass/fail
plus the pre-agreed residual summary. The final independent comparison returns
only the complete pre-agreed result after every arm finishes.

## Current readiness

Identified:

- the final article, preprint, and aggregate published anchors; and
- the intended scientific roles for each source type.

Still required before candidate scoring:

- approval of article and supplement use and rights;
- measurement contexts and source-use decisions for the assigned anchor roles;
- a fitted paper-derived reference and successful held-aside check;
- the complete accepted reference-model set and a candidate rank-stability
  result;
- a finite legal alternative-pad catalogue and conservative resource rules;
- executable development cortical-field, source-to-surface, and electronics
  models;
- independently implemented final models;
- separate development and final physical-signal or phantom data;
- the catastrophic-plausibility threshold and scientist approval;
- registered estimators for `lambda_k` and `tau_k`, plus independently
  characterized noise, impedance, settling, and crosstalk for each frozen
  D0–D3 representative and device condition; and
- the fixed family-specific utility functions and two-by-two decision rule.

Missing raw traces, a PDK, compact device model, netlist, layout, DAQ code,
exact reduced-mode maps, marker meanings, and per-setting power data narrow the
claim. They are not permission to invent parameters or describe the reference
as a physical digital twin.
