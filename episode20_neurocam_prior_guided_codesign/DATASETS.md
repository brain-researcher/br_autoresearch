# Sources used in Episode 20

EP20 combines four kinds of evidence that answer different questions:
published NeuroCam measurements, simulated cortical fields, device and
electronics models, and a physical-signal or phantom plausibility check. They
must remain separate. A public paper is not automatically a calibration source,
and ordinary ECoG is not dense ground truth for an unbuilt electrode geometry.

The scientist's October 6, 2026 exception explicitly permits simulations for
EP20. The absent measurement is fabricated-candidate performance, not all real
data: published NeuroCam anchors and the acquired Rat5 segment remain real
inputs with the roles below. Their unresolved qualification does not prevent
implementing selected model equations with explicit caller-supplied parameters.

No source has yet been qualified as an EP20 scientific input, and no candidate
has been scored. Source acquisition does not establish scientific readiness.

**October 13 conditional S5 route:** the scientist adopted the
[written prediction plan](outputs/proposed_conditional_reference_prediction.md).
Article use is authorized for this work. The partial route chooses an explicit
dry-air/probe/scan interpretation before fitting the assigned nearest anchors;
assigned distant observations remain out of fitting. These are prospective
assumptions, not recovered historical wiring or complete reference qualification.

## Public online resources — checked October 6, 2026

These primary sources supply usable documentation now. They narrow the
requests for experiment-specific evidence; availability is distinct from
qualification as a calibration or physical-signal input.

| Public source | Information available online | Use and remaining limit in EP20 |
| --- | --- | --- |
| [Author-hosted final article and supplement](https://shengxingstars.github.io/www/images/publications/2025_High-resolution%20spatial%20mapping%20of%20electrocorticographic%20activities%20with%20NeuroCam%20a%204096-channel,%20multiplexed%20flexible%20thin-film%20transistor%20array.pdf), supplement Data Acquisition System, Noise Characterization and Data Processing | Three PXIe-4309 cards and one PXIe-6739; a G1 control signal used as a marker; 100 kS/s per source line. Static recording assigns 23 signal lines plus one marker; 64-source operation distributes signals two or three per ADC. Processing describes row extraction, averaging samples 3–5, 49–51 Hz rejection, 1–199 Hz filtering and Savitzky–Golay smoothing. | Supplies published architecture and processing context. Exact source-to-ADC assignment, configured ranges/clocks/filter parameters and calibration uncertainty are still unverified; the stated filter versus full-array Nyquist inconsistency remains. |
| [NI PXIe-4309 overview](https://www.ni.com/docs/en-US/bundle/pxie-4309/page/overview.html) | Voltage-input hardware: 32 differential inputs, eight simultaneously sampling ADCs with channel multiplexing, selectable ranges and onboard averaging/filtering. | Resolves manufacturer-defined input function and possible configuration choices, not the settings used for NeuroCam. |
| [NI 6738/6739 manual](https://docs-be.ni.com/bundle/ni-6738-6739-features/raw/resource/enus/375140b.pdf) | Analog-output hardware, including the 6739's AO0–63 channels and timing/trigger routing documentation. | Resolves output function and exposes the article's inconsistent device description; it does not authenticate experimental wiring. |
| [DANDI 000932 published metadata, release 0.240317.0101](https://raw.githubusercontent.com/dandisets/000932/0.240317.0101/dandiset.yaml) | Open-access CC BY 4.0 release with 238 files, 199,844,377,549 bytes and ten subjects across pig and rat; NWB electrophysiology objects are identified. The release's protocol and related-resource lists are empty. | Identifies the selected release and its public inventory. Ten released subjects are not ten eligible EP20 animals. These metadata do not supply the acquired Rat5 file's missing authentic geometry or acquisition mapping. |

The DANDI version is also identified by its [release DOI](https://doi.org/10.48324/dandi.000932/0.240317.0101).
This lookup read source documentation and release metadata, not additional
animal recordings. The narrower remaining gaps are detailed below.

### Focused calibration-context check — October 6, 2026

The [combined article/supplement](https://shengxingstars.github.io/www/images/publications/2025_High-resolution%20spatial%20mapping%20of%20electrocorticographic%20activities%20with%20NeuroCam%20a%204096-channel,%20multiplexed%20flexible%20thin-film%20transistor%20array.pdf),
Fig. S3 caption (PDF page 16), specifies **one-second static-noise RMS windows**
and channel histograms with Gaussian fits. Its noise methods specify a grounded
platinum mesh in PBS. The caption does not identify `±` as SD, SEM or fitted
Gaussian width; Fig. S4 does not extend the one-second duration to dynamic
noise. The general processing paragraph does not bind each filter/smoothing
setting to each noise anchor. Those fields remain unresolved.

The final article carries a 2025 Science China Press copyright notice reserving
rights, including text/data mining; no open reuse license was observed there.
This records documentary terms; the scientist has authorized article use in
this episode. No additional reuse license is asserted, and source-use approval
is not an outstanding scientist decision.

The next calibration dependency is the original transfer/noise/crosstalk
observation recipe or source-supported bounds, as required by the
[minimal shared fit](outputs/electrical_model_and_probes.md#minimal-reference-fit--shared-combinations).
Further generic circuit code cannot supply that missing evidence. Existing
implementation checks remain useful but cannot qualify these anchors.

### Fig. 2a transfer context correction — October 7, 2026

The [final article, Fig. 2a caption, printed p. 4136 / PDF page 5](https://shengxingstars.github.io/www/images/publications/2025_High-resolution%20spatial%20mapping%20of%20electrocorticographic%20activities%20with%20NeuroCam%20a%204096-channel,%20multiplexed%20flexible%20thin-film%20transistor%20array.pdf#page=5)
identifies `Vout/Vin`, a **12-Hz sine**, stated amplitude **1 mV**, and
**all 4,096 channels in PBS**. Frequency, waveform and nominal channel coverage
are therefore documented, not missing. The caption does not distinguish peak
from peak-to-peak amplitude; the supplement's electrical-characterization
paragraph (PDF page 9) instead mentions a 500-µVpp, 12-Hz PBS test without
explicitly identifying it as Fig. 2a. Do not silently reconcile the two.

The remaining Fig. 2a dependency is enough information to predict its actual
gain statistic: input/output amplitude extraction, across-channel centre and
dispersion definitions, stimulus/reference arrangement, and the relevant
scan/DAQ/processing response or source-supported bounds. Peak versus
peak-to-peak convention alone is not blocking **conditional on linear,
undistorted sinusoids and the same amplitude functional for input and output**;
the convention's scale factor then cancels. This inference neither reconciles
the two reported amplitudes nor establishes linearity or the original extractor.

Channel names alone do not affect an equally weighted, permutation-invariant
all-channel summary; electrical grouping and acquisition history can affect
the gains being summarized. A conditional common-input PBS response also need
not equal an isolated channel's direct transfer; the actual bath forcing is
not documented by channel coverage. The
[operator clarification](outputs/electrical_model_and_probes.md#fig-2a-minimum-operator-information)
distinguishes these dependencies. Equivalent evidence defining the effective
operator is sufficient; perfect recovery of the entire acquisition history is
not an added requirement. Rat5 geometry/mapping belongs to its separate
physical-signal qualification, not this transfer-fit dependency. This narrows
the [partial transfer specification](outputs/electrical_model_and_probes.md#october-7-calibration-observables-and-partial-transfer-constraints);
it does not supply a fitted model, source-use disposition or reference
acceptance. Noise and crosstalk observation dependencies retain their roles.

## Source roles

| Source | Role | Current state |
| --- | --- | --- |
| NeuroCam article and supplement | Define published architecture and characterization anchors | Identified; EP20 use and rights still need approval |
| Calibration subset of published anchors | Fit a paper-derived NeuroCam model | Design roles assigned below; measurement context/use still unresolved |
| Held-aside published anchors | Check that the fitted reference reproduces measurements not used for fitting | Design roles assigned below; no check has run |
| All reference models that pass both checks | Test whether candidate rankings are identified by the published measurements | Starting templates/shared fit specified; acceptance set not constructed |
| Development cortical-field and device models | Develop designs and run challenge tests | Not implemented |
| Independent cortical-field and electronics models | Rerun frozen D0–D3 representatives and test the rule under every accepted reference mechanism | Not implemented |
| Development physical-signal or phantom source | Stress spectra, amplitudes, missingness, and software stability | DANDI 000932 Rat5 selected for source inspection; admissibility remains unresolved |
| Separate final physical-signal or phantom source | Reject catastrophic implausibility after design selection | Rat3 reserved as an unopened whole-animal candidate; final qualification remains unresolved |
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

### October 6 public documentation: DAQ module functions resolved

Manufacturer documentation checked on October 6, 2026 (Pacific) establishes
that the [PXIe-4309](https://www.ni.com/docs/en-US/bundle/pxie-4309/page/overview.html)
is an **analog-input** module with 32 differential inputs multiplexed across
eight ADCs. The [NI 6738/6739 manual](https://docs-be.ni.com/bundle/ni-6738-6739-features/raw/resource/enus/375140b.pdf)
identifies the PXIe-6739 as an **analog-output** device. The module functions
therefore need not be guessed or supplied as a user choice. They flag the
article/supplement's inconsistent input/output description; they do not
authenticate a corrected reconstruction of the experiment's wiring.

Actual channel masks, marker allocation, ranges, clock/trigger routing,
averaging/filter settings and calibration/noise-processing context still
require original experiment evidence. Manufacturer capabilities alone cannot
select those settings or establish conversion-work parity. This documentary
correction supplies no fitted reference parameters or new calibration anchors.

## What the paper does not determine

The article and supplement do not provide a complete generative electronics
model. Important uncertainties include:

- the exact meaning of every reported `±` value;
- a discrepancy between 10-mVpp and 1-mVpp frequency-response tests;
- whether the stated 6.25 oversampling ratio describes within-dwell sampling or
  the ADC itself;
- exact channel identities and marker meanings in reduced modes;
- the actual wiring/configuration behind a conflicting DAQ input/output
  description involving three PXIe-4309 cards and one PXIe-6739 card; the
  manufacturer-defined input/output functions are resolved above;
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

### Selected acquisition candidate: DANDI 000932

Use the rat branch of [Tchoe et al.'s iEEG-microdisplay dataset](https://dandiarchive.org/dandiset/000932/0.240317.0101),
release `0.240317.0101`, DOI `10.48324/dandi.000932/0.240317.0101`.
The [version metadata](https://api.dandiarchive.org/api/dandisets/000932/versions/0.240317.0101/)
specifies CC-BY-4.0 and animal recordings. This source is a physiological
plausibility candidate, not a NeuroCam phantom, an electrical calibration, or
dense ground truth for comparing unbuilt pads.

The [primary paper and supplement](https://iebl.ucsd.edu/sites/default/files/iebl/2024-04/An%20electroencephalogram%20microdisplay%20to%20visualize%20neuronal%20activity%20on%20the%20brain%20surface.pdf)
describe rat arrays with 1,024 contacts, 150-micrometre pitch, approximately
5 x 5 mm coverage and 30-micrometre PtNR contact diameter; ORB1024/Intan
acquisition at 20 ksample/s; and a scalp-needle reference beside the craniotomy.
Rat5 and Rat3 are whisker-mapping animals without neurotoxin. Rat6's induced
pathology is not a state-matched substitute. LED interference near 98.63 Hz
falls inside EP20's primary band and must be treated explicitly. Paper-level
acquisition settings do not establish the deposited file's preprocessing.

Assign roles by whole animal before inspecting signals:

| Role | Selected source | Current access |
| --- | --- | --- |
| Development source inspection | Rat5, earliest acquisition file `sub-Rat5_ses-20221107T110047_ecephys.nwb`; asset `c7f977e1-5c00-4b9d-a309-538051c7d3b2`; 627,332,180 bytes | One file only, under `outputs/cache/physical_signal_development/dandi000932/`; see the execution log for observed acquisition/header checks |
| Reserved final candidate | All Rat3 recordings | Public inventory metadata only; no download, waveform, preview or signal statistic |
| Other animals/files | Unassigned | No implied access or download |

The selected Rat5 file is an acquisition segment, not the complete session.
All Rat5 recordings remain on the development side if later provisioned; none
may enter final evaluation. Chronological selection does not select on signal
quality. No overlapping windows or normalization statistics may cross animal
roles. The full 199.8-GB collection is not needed for this source inspection.
Exact stimulus/block identity, deposited channel geometry and voltage units,
timing, filters, reference, removed channels and raw-source history must be
resolved from development metadata before qualification. If absent, document
the limitation; do not silently regard a processed display as raw ECoG.

The downloaded Rat5 file's observed header is summarized in the
[acquisition entry](outputs/experiment_log.md#2026-10-03---model_specification---physical-source-acquisition-attempt-001):
1,024 float32 voltage channels with 300,032 samples, declared volts, and
uniform explicit timestamps at 5 kHz over approximately 60 seconds. Every
electrode x/y/z coordinate is zero. The series says `Wideband data`, while
electrode filtering/location say `none`; session description and stimulus
tables do not identify the block or LED state. These are incomplete metadata,
not proof of zero filtering or a usable spatial map. The file is staged for
development, but spatial/source-history qualification remains open. No voltage
samples or final-animal data were inspected.

The authors' [Rat4 analysis code](https://zenodo.org/api/records/10825275/files/UCSD_Rat_10312022_Epileptiform_32x32.m/content)
documents some original 5-kHz acquisitions, so do not assume Rat5 was decimated
from 20 kHz. It depends on `elecCoords_ratHD.mat`, absent from the individually
listed files in the [inspected code release](https://zenodo.org/records/10825275).
The available pig map cannot replace it; uninspected archive contents remain
unknown. The remaining request is the Rat5 channel-to-grid mapping, original
header/conversion and filter/reference history, and block-specific stimulus/
LED state. These facts are needed before calling this an admissible spatial
plausibility source, not before retaining the already acquired file.

An additional public-metadata lookup on October 6, 2026 (Pacific) inspected
the DANDI-maintained [draft metadata](https://raw.githubusercontent.com/dandisets/000932/draft/dandiset.yaml).
Its `protocol` and `relatedResource` lists were empty, and its description
referred readers to the article for filtering instructions. This route
supplied no Rat5 conversion-code or channel-map link. The draft was inspected
only as a documentary lead; it does not replace the fixed `0.240317.0101`
input version. The bounded exact-name/provider search also found no new
authenticated Rat5 map or original NeuroCam DAQ program. This is a search
limit, not proof that those records do not exist. No pig geometry, generic
NWB conversion procedure or hardware default fills the missing correspondence.

Other inspected sources are fallbacks, not additional downloads:

- [Rogers et al., Figshare 7633418 v4](https://doi.org/10.6084/m9.figshare.7633418.v4)
  provides mouse-only files but already downsamples and windows the recordings;
  original interval/missingness history is not established by its README.
- [Neural Matrix, Zenodo 3686318](https://zenodo.org/records/3686318)
  provides a 4.23-GB rat archive under CC-BY-4.0; archive contents and independent
  recording support have not been inspected.
- [OpenNeuro ds004784 v1.0.4](https://github.com/OpenNeuroDatasets/ds004784/tree/1.0.4)
  is a real CC0 scalp-EEG phantom, but does not supply the required dense
  cortical input map and row-scan markers. Its artifact conditions are not
  automatically independent waveform partitions. It is not an EP20 replacement.

No qualifying public NeuroCam raw phantom source was found in the inspected
article/supplement. Dense-ECoG qualification can satisfy the existing alternative
plausibility route; an unrelated phantom download would not resolve unknown
NeuroCam electronics or prove candidate physical superiority.

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

- the final article, preprint, aggregate anchors and calibration/held-aside roles;
- the starting pad/scan catalogue and selected two passive-contact laws;
- the manufacturer-defined DAQ functions: PXIe-4309 analog input and
  PXIe-6739 analog output, distinct from the still-unknown experiment wiring;
- the implemented contact component, whose eleven analytic checks passed
  without fitting any NeuroCam parameter; and
- the acquired Rat5 segment's structural metadata and the preserved Rat3 final
  candidate role. The Rat5 channel map/acquisition history remains unqualified.

The [contact implementation record](outputs/experiment_log.md#2026-10-05-0450-utc--model_specification-contact-implementation-checks-pass)
is preparation, not a fitted reference or a candidate result. The existing
outer and two nested ZIP directory inspections, and the later rat-named script
review, supplied no authentic Rat5 mapping; they do not establish universal
absence and do not justify substituting a pig map.

Still required before candidate scoring:

- approval of article and supplement use and rights;
- measurement contexts and source-use decisions for the assigned anchor roles;
- a fitted paper-derived reference and successful held-aside check;
- the complete accepted reference-model set and a candidate rank-stability
  result;
- physical eligibility of the selected pad/scan catalogue and complete
  conservative resource accounting;
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

The [October 6 scientist decision](outputs/paper_plan.md#october-6-simulation-scope-exception--adopted)
retains simulation-based EP20 and supersedes the earlier deferral recommendation.
It does not identify unknown DAQ settings or Rat5 geometry, change the anchor
split, or increase resource allowances. The R1 persistent source-line component
is implemented with explicit parameters and clock intervals; all seven analytic
checks passed in job 46833971. Scalar loading and independent-source noise
propagation are also implemented, with six focused checks passed. A small
passive coupled RC component now retains node states and conserves charge
across explicit capacitance changes; its eight analytic checks passed.
All physical matrices and timing remain caller inputs; full-array assembly and
actual circuit/DAQ parameter support are unresolved. These components do not
depend on Rat5, fit the paper anchors or score a candidate. Calibration/plausibility
inputs still cannot turn simulated candidate outcomes into measured hardware
performance; Rat3 and independent final outcomes remain closed.
