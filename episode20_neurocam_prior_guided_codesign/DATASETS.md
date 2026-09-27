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
| Calibration subset of published anchors | Fit a paper-derived NeuroCam model | Not assigned |
| Held-aside published anchors | Check that the fitted reference reproduces measurements not used for fitting | Not assigned |
| Development cortical-field and device models | Develop designs and run challenge tests | Not implemented |
| Independent cortical-field and electronics models | Test one selected design under a different implementation | Not implemented |
| Development physical-signal or phantom source | Stress spectra, amplitudes, missingness, and software stability | Not selected |
| Separate final physical-signal or phantom source | Reject catastrophic implausibility after design selection | Not selected |
| Design and resource rules | Decide which designs are legal and resource matched | Not prepared |
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
| Device impedance | Approximately 100 kilo-ohms over the reported 1–100 kHz range |
| Within-dwell aggregation | Stable third through fifth samples from six or seven raw samples, as described in the supplement |
| Static high-precision setting | 23 signal source lines plus one ADC marker channel; lowest stated noise 2.3 ± 0.3 microvolts RMS at 100,000 samples/s |
| First source-line neighbour crosstalk | Approximately −12.6 dB |
| First gate-line neighbour crosstalk | Approximately −37.2 dB |
| Second neighbour and farther | Approximately −45 dB |

### Published dynamic settings

| Setting | Pixels | Effective rate per pixel | Reported RMS noise | Approximate raw conversions/s |
| --- | ---: | ---: | ---: | ---: |
| 24 source × 16 gate | 384 | 1,000 samples/s | 8.8 ± 4.2 microvolts | 2.4 million |
| 24 source × 64 gate | 1,536 | 250 samples/s | 12.6 ± 8.4 microvolts | 2.4 million |
| 64 source × 64 gate | 4,096 | 250 samples/s | 63.6 ± 24.1 microvolts | 6.4 million |

The corresponding effective field outputs are approximately 384,000, 384,000,
and 1,024,000 pixel samples/s. Reference, ground, guard, power, and external DAQ
costs are counted separately from the 128 array lines.

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

The primary target is voltage on a common cortical-surface grid. Latent-source
recovery is descriptive only.

## Source-to-surface, device, and electronics models

Development includes both an analytic homogeneous or layered source-to-surface
model and a heterogeneous or anisotropic alternative, with distance, curvature,
reference, contact-gap, and conductivity variation.

The device models vary together where physics couples them: pad averaging,
interface impedance, source loading, line RC, settling, switching transients,
crosstalk, noise, ADC sampling, aggregation, quantization, drift, saturation,
missing pixels, contact gaps, and voltage reference. Each rule must be marked as
measured, literature-derived, physics-derived, or deliberately adversarial.

The selected final design is tested under separately implemented cortical-field,
source-to-surface, and electronics models. Merely changing random draws in the
development implementation is not an independent structural test.

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

If no NeuroCam PDK or netlist becomes available, a scientist must approve
conservative substitute rules supported by published anchors or explicit safety
factors. Modeled power and thermal quantities remain proxies, not measured
equivalence.

The same rules determine which D0, D1, and D4 comparisons are eligible at each
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
- calibration and held-aside assignments for published anchors;
- a fitted paper-derived reference and successful held-aside check;
- a finite legal alternative-pad catalogue and conservative resource rules;
- executable development cortical-field, source-to-surface, and electronics
  models;
- independently implemented final models;
- separate development and final physical-signal or phantom data; and
- the catastrophic-plausibility threshold and scientist approval.

Missing raw traces, a PDK, compact device model, netlist, layout, DAQ code,
exact reduced-mode maps, marker meanings, and per-setting power data narrow the
claim. They are not permission to invent parameters or describe the reference
as a physical digital twin.
