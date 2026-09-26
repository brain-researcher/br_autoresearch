# Dataset and source contract — Episode 20

EP20 combines documentary evidence, simulated cortical fields, electronics
models, and one empirical or known-input replay. These sources answer
different questions and must never be blended into one undifferentiated
“dataset.”

This document records provenance, access, and scientific role. It does not
authorize a download, open protected audit material, run a simulator, or
promote a file into the experiment. No candidate-discriminating source is
currently provisioned under the episode’s read-only inputs directory.

## At a glance

| Source | Role in EP20 | Candidate-visible? | Current state |
| --- | --- | --- | --- |
| NeuroCam article and supplement | Define the published architecture and characterization anchors | Yes, before search | PDF exists in the private steward store; EP20 use and rights are not yet approved |
| Calibration anchors | Fit a paper-derived NeuroCam reference-model ensemble | Yes | Not provisioned |
| Qualification anchors | Test the fitted reference ensemble without reusing them for fitting | Values are public, but their role is locked | Not provisioned; gate not run |
| Development signal and device models | Develop candidates and run registered falsifiers | Only through the frozen development evaluator | Not implemented |
| Sealed virtual audit | Test a locked candidate under independent signal and electronics implementations | No pre-lock outcome access | Steward, engine, and payload absent |
| Development empirical replay | Stress spectra, amplitudes, missingness, and software stability | Through the development evaluator | Source and partition absent |
| Sealed empirical or phantom replay | Post-lock catastrophic-plausibility check | No pre-lock outcome access | Source, partition, and threshold absent |
| Hardware rules and resource model | Decide what designs are legal and resource matched | Static contracts are visible | Not provisioned |
| Raw traces, PDK, netlist, layout, and DAQ code | Potential future qualification evidence | Role must be assigned before opening | No authenticated handoff identified |
| Fabricated candidate data | Future physical validation, outside EP20 | No EP20 role | Absent and not authorized |

A public paper or dataset is not automatically a calibration source, an audit
source, or permission to use it for candidate selection. Before use, record
where it came from, whether EP20 may use it, and whether it is development or
sealed-audit material.

## Current physical location

The documentary PDF is stored in the private steward directory:

    /oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/steward_acquisition/neurocam_documentary_20260922

It has not yet been provisioned for EP20. Before use, record its citation and
rights and assign it a documentary, calibration, or qualification role.

Two small legacy JSON examples exist outside the episode inputs. They may be
used only as documentary examples if a scientist explicitly assigns that role.

## Documentary reference

| Item | Fixed identity |
| --- | --- |
| Device | NeuroCam flexible multiplexed micro-ECoG array |
| Final article | Xie et al., “High-resolution spatial mapping of electrocorticographic activities with NeuroCam: a 4096-channel, multiplexed flexible thin-film transistor array” |
| Journal | Science Bulletin 70 (2025), 4133–4137 |
| DOI | [10.1016/j.scib.2025.11.030](https://doi.org/10.1016/j.scib.2025.11.030) |
| Publisher identifier | S2095927325011582 |
| PubMed | [PMID 41309324](https://pubmed.ncbi.nlm.nih.gov/41309324/) |
| Preprint | [10.1101/2024.08.18.608446](https://doi.org/10.1101/2024.08.18.608446) |
| Supplement status | Seen inside an author-hosted combined PDF; separate publisher supplement identity remains unconfirmed |
| Draft-time observation | Combined article-plus-supplement PDF observed 2026-09-21 |
| Observed URL | [Author-hosted combined PDF](https://shengxingstars.github.io/www/images/publications/2025_High-resolution%20spatial%20mapping%20of%20electrocorticographic%20activities%20with%20NeuroCam%20a%204096-channel,%20multiplexed%20flexible%20thin-film%20transistor%20array.pdf) |
| Rights rule | Record terms separately for the article, supplement, preprint, figures, and any author-supplied file before copying or redistribution |
| Episode-local payload | Absent |

Before reference fitting, confirm which article and supplement version will be
used and record its rights status. The combined file is not proof that it is
the official article/supplement split.

The final article is the scientific citation. The preprint may document
version history but cannot silently replace the final methods or supplement.

## What the paper anchors

When extracting anchors, record original units, source page, figure or table,
and whether the value was stated directly or digitized.

### Array and operating modes

| Anchor | Published value |
| --- | --- |
| Lattice | 64 × 64 pixels |
| Pitch and active area | 150 micrometres; approximately 9.6 × 9.6 mm |
| Gold sensing pad | Approximately 70 × 135 micrometres |
| Substrate and active device | 25-micrometre polyimide; Ln-IZO TFT |
| Addressing | 64 gate and 64 source lines |
| Representative gain | 0.78 ± 0.12 |
| Representative channel bandwidth | 3-dB frequency above approximately 1 kHz |
| Illustrated full-scan row dwell | Approximately 62.5 microseconds |
| Array thickness | Less than 30 micrometres |
| Representative TFT | Mobility 15.13 cm²/V/s; threshold 2.7 V; on/off ratio approximately 6 × 10⁶ |
| Gate drive | Approximately +6 V on and −2 V off |
| Device impedance | Approximately 100 kilo-ohms over the reported 1–100 kHz range; retain the curve |
| Within-dwell samples | Stable third through fifth points from six or seven raw samples, as described in the supplement |
| 24S × 16G | 384 pixels; 1000 S/s; 8.8 ± 4.2 microvolts RMS |
| 24S × 64G | 1536 pixels; 250 S/s; 12.6 ± 8.4 microvolts RMS |
| 64S × 64G | 4096 pixels; 250 S/s; 63.6 ± 24.1 microvolts RMS |
| Static high-precision mode | 23 signal source lines plus one ADC marker channel; lowest stated noise 2.3 ± 0.3 microvolts RMS at 100 kS/s |
| First source-line neighbour crosstalk | Approximately −12.6 dB |
| First gate-line neighbour crosstalk | Approximately −37.2 dB |
| Second neighbour and farther | Approximately −45 dB |

The stated modes also imply distinct accounting quantities. With the reported
100-kS/s raw sampling per active source line, the two 24-source modes imply
about 2.4 million raw conversions/s and the 64-source mode about 6.4 million.
Their effective field outputs are approximately 384,000, 384,000, and
1,024,000 pixel samples/s. These are documentary checks, not a complete energy
or ADC-cost model.

Reference, ground, guard, power, external DAQ wiring, and metadata costs are
counted separately from the 128 array gate-plus-source lines.

### Documentary limits that must remain visible

The paper and supplement do not identify a complete generative electronics
model. The source contract must preserve these unresolved points:

- The meaning of reported ± values is not established in every case. Store
  them verbatim with dispersion semantics marked unspecified unless the exact
  source proves SD, SEM, a fitted width, or another statistic.
- The main figure caption and supplement report 10-mVpp versus 1-mVpp
  frequency-response test amplitudes.
- The stated 6.25 oversampling ratio may refer to samples within a
  62.5-microsecond dwell rather than ADC-internal oversampling.
- The exact spatial subsets, channel identities, and marker semantics of the
  reduced modes are not reported.
- The DAQ description names three PXIe-4309 cards and one PXIe-6739 card, but
  its first input/output-role sentence conflicts with the later description
  of PXIe-4309 ADCs.
- Reported crosstalk comes from a controlled air, single-pixel,
  approximately 5-mVpp/12-Hz test and does not define a universal in-vivo
  frequency-dependent kernel.
- A reported 1–199 Hz filter does not override the approximately 125-Hz
  Nyquist limit of the 250-S/s full-array mode.
- No public source currently supplies raw I-V/C-V traces, a compact model,
  PDK, transistor netlist, exact layout, full DAQ schematic, per-mode power
  traces, or independently held-out devices.

Unknown cost is not zero, and an aggregate noise value is not an independent
white-noise distribution. These gaps must become explicit uncertainty ranges
or hard blocks, never guessed constants.

Noise summaries from different bandwidths, filters, channel counts, or
within-dwell aggregation rules may not be pooled without a frozen,
source-justified conversion.

## Legacy examples

If the two legacy JSON examples are ever provisioned, their only acceptable
role is nonbinding documentary context and non-candidate-discriminating
simulation sanity checks:

- **neurocam_paper_direct_v1.json** transcribes paper-direct summaries.
- **neurocam_figure_derived_v1.json** contains assumption-bound algebraic or
  manually read figure estimates and must remain marked estimation-only.

Neither file may fit or qualify the reference, initialize model parameters,
define the search space, score a candidate, or substitute for raw traces,
compact models, a PDK, netlists, layout, or fabrication data.

## Calibration and qualification

Paper-derived values are public, so qualification is a locked fit holdout,
not a secret external validation set. Their roles must be frozen before any
model fitting.

| Role | Candidate contents |
| --- | --- |
| Calibration-visible | Geometry and topology; gate timing; representative gain; static high-precision noise; 24S × 16G and 24S × 64G rate/noise anchors; impedance and representative response curves |
| Fit-held-out qualification | 64S × 64G rate/noise; source-, gate-, and distant-crosstalk anchors; held-out curve points; independently obtained settling or switching traces; held-out device, wafer, bend, or soak measurements if available |

The implementation that fits the reference may not consume qualification
values. A trusted qualifier receives the fitted ensemble and role-separated
anchors, runs an outcome-blind gate, and returns only pass/fail plus the
permitted residual summary. Candidate selection may not tune from
qualification residuals.

An aggregate-only split is acceptable only if recovery simulations show in
advance that its held-out anchors distinguish wrong scaling laws. Otherwise
the study needs raw traces, additional devices, or an independently justified
model range. Failure of a properly run frozen gate is
**technical_reference_model_unqualified**, not a negative result about
co-design.

If raw assets remain absent, EP20 may continue only with scientist acceptance
of the narrower object “paper-derived NeuroCam-class reference-model
ensemble.” It must not claim an exact digital twin, transistor-accurate
reproduction, fabrication-ready layout, matched physical power or lifetime,
or superiority over fabricated NeuroCam.

## Development sources

The development evaluator requires frozen, executable sources in three layers.

### Cortical fields

The bank must cover broad correlated fields, focal field transients,
translating and expanding waves, multiscale mixtures, interacting sources,
silent-region surprises, and nuisance families. It must freeze their parameter
ranges, mixture weights, code revisions, seeds, and empirical or physical
justification.

| Family | Factors that must be frozen |
| --- | --- |
| Broad correlated field | Temporal and spatial PSD, correlation length, anisotropy, amplitude, and common-mode fraction |
| Focal field transient | Aperture, duration, amplitude, event rate, location distribution, and spectral envelope |
| Translating or expanding wave | Direction, speed, wavelength, curvature, onset, and boundary interaction |
| Multiscale mixture | Broad/local energy ratio, cross-component dependence, overlap, and nonstationarity |
| Interacting sources | Number, separation, synchrony, phase relation, and collision rule |
| Silent-region surprise | Pre-event silence, unexpected location, event amplitude, and duration |
| Nuisance family | Line noise, reference drift, motion-like transients, gaps, dropout, gain drift, and saturation |

A qualitative label such as “smooth” or “focal” is not a source contract.
Each range needs an empirical source, physical bound, or explicit conservative
design interval.

Each instance emits a dense cortical-surface field, or a latent source plus a
source-to-surface operator, along with the labels and masks needed for the
registered secondary endpoints. New seeds may not be introduced because one
candidate performs poorly.

Use one predeclared set of generator and device draws across candidates. Do not
regenerate it in response to results.

### Source-to-surface models

Development must include an analytic homogeneous or layered operator, a
heterogeneous or anisotropic operator, distance and curvature variation, an
explicit reference convention, and contact-gap and conductivity
perturbations. Any FEM, BEM, reciprocity solver, or surrogate must record mesh
or anatomy source, units, conductivity provenance, boundary conditions,
software version, convergence rule, and surrogate error.

Latent-source recovery is diagnostic. The scored physical target remains
cortical-surface voltage.

### Device and electronics models

The device ensemble is conditional on compiled hardware. Pad area, aperture
averaging, interface impedance, source loading, line RC, settling, switching
transients, crosstalk, noise, raw ADC sampling, aggregation, quantization,
drift, saturation, missing pixels, contact gaps, and reference convention must
change together where physics couples them.

Every scaling law is labelled measured, literature-derived, physics-derived,
or deliberately adversarial. The optimizer cannot choose a favourable model
member or discard an unfavourable one.

## Sealed audit sources

The sealed virtual audit is more than a new random seed. Before search, an
independent audit steward defines:

- a separately implemented structural signal generator;
- a source-to-surface operator absent from development;
- an independently implemented electronics scaling or coupling law;
- a moving or nonstationary source family and altered spatial spectrum;
- unseen combinations of noise, crosstalk, settling, drift, gaps, saturation,
  and dead pixels;
- audit-only seeds, device draws, masks, aggregation units, and evaluator.

Sharing the development mesh, code path, numerical kernel, or fitted surrogate
with only new parameters or seeds is a parameter holdout, not an independent
structural audit.

Operating-system permissions must prevent the adaptive runtime from reading
audit parameters, traces, previews, scores, ranks, or candidate-specific QC
before the candidate is locked. It may receive only the agreed final report.
The audit remains a virtual model audit even when independently authored.

## Empirical or known-input replay

One source type must be selected and frozen before search launch:

1. a legally reusable dense ECoG recording with geometry, timestamps,
   reference, hardware, sample rate, preprocessing, and raw-file provenance;
   or
2. a known-input PBS or phantom replay with a spatial input map, raw array
   outputs, scan markers, operating mode, and independent holdout traces.

Ordinary ECoG is already filtered by its original contacts and reference. It
is not dense ground truth for arbitrary unbuilt pad shapes. Development replay
is evaluated only on D0 and D1 for spectral, amplitude, missingness, and
software-stability stress. It is never run as an alternative-hardware endpoint
for D2, D3, or D4 and cannot rank, retire, or route their successors.

A reference-device phantom can test more of the electronics chain, but it
still does not validate an unbuilt candidate geometry. Physical superiority
requires a fabricated candidate and matched physical comparator.

Development and post-lock replay use separate sources or partitions split by
whole subject, device, session, or independently replayed waveform. Overlapping
windows are not independent. No item, normalization statistic, fitted
threshold, or derived summary may cross from the sealed partition into
development. The catastrophic-failure threshold is set and signed off before
search launch.

For each source, record its citation or owner, terms of use, preparation,
geometry, channel count, sample rate, reference, filters, raw or derived
status, assigned role, split unit, and why the two roles do not overlap.

## Who may see what

| Actor | Permitted access |
| --- | --- |
| Candidate and adaptive controller | Documentary facts, frozen public contracts, and permitted aggregate development-evaluator outputs |
| Trusted reference fitter | Calibration role only; no qualification-role consumption |
| Trusted qualifier | Calibration and qualification assignments plus the fitted ensemble; returns pass/fail and the permitted residual summary |
| Audit steward and evaluator | Sealed virtual and plausibility payloads after candidate lock |
| Adaptive runtime after audit | Only the predeclared audit report |

A shared folder containing calibration, qualification, development, and audit
content is not a firewall. Public availability does not make audit access
acceptable. The adaptive runtime must be unable to read or redownload sealed
content.

## Hardware and resource sources

Before candidate scoring, the compiler needs fixed rules for pad shapes,
dimensions, clearances, within-pixel exclusions, transistor and via placement,
trace loading, legal dwell and scan order, active source-bank configurations,
AFE/ADC allocation, conversions, bitrates, latency and skew, power and thermal
proxies, routing and area, reference and supply costs, and yield or dead-pixel
proxies.

A primitive absent from the source package is prohibited until its timing,
area, noise, power, calibration, and manufacturing rules are frozen. If no
NeuroCam PDK or netlist is obtained, the substitute must be a
scientist-approved conservative rule deck with cited anchors or declared
adversarial safety factors. The power and thermal quantities remain proxies,
not measured equivalence.

The same resource plan also fixes each resource-envelope ID and its
eligible or matched D0, D1, and D4 comparator map. It distinguishes physical
source lines, raw ADC conversions and oversampling, effective field samples,
output word size and metadata overhead, bitrate, latency, skew, and
reliability proxies.

For power and thermal proxies, set lower and upper component bounds,
activity factors, conversion cost, and uncertainty ranges. A shared process
name or a short thermal image does not make unknown candidate cost equal to
the incumbent.

The paper’s two-week PBS soak and bend summaries may motivate sensitivity
checks. They do not establish chronic in-vivo safety, biocompatibility,
reliability, or lifetime.

## Minimum source records before search

Keep a short, human-readable record of:

- the documentary source, version, rights, and extracted anchors;
- which anchors are used for calibration and which for qualification;
- hardware, resource, software, endpoint, and randomization rules fixed before
  outcomes;
- the development signal and device definitions and their predeclared draws;
- the separate development and sealed empirical sources or partitions; and
- the separately permissioned audit owner and location, opening time, and
  allowed final report.

The format is deliberately simple. The records may be ordinary text files.

## Inputs directory rule

The inputs directory is read-only after provisioning. Generated candidates,
logs, caches, weights, scores, and audit reports belong under outputs. Large
sources may remain in a governed external store; inputs needs only clear
pointers and scientific roles.

No link or mount may expose a mixed tree containing both development and audit
content. The repository should not contain uncontrolled copies of papers,
traces, or large simulations.

## Current provisioning state — 2026-09-26

Identified:

- the final article, DOI, preprint DOI, and aggregate documentary anchors;
- the private steward location of the documentary PDF; and
- the intended scientific roles.

Still blocking launch:

- an approved paper/supplement source and rights record;
- calibration and qualification assignments for paper-derived anchors;
- a finite legal alternative-pad catalogue and conservative compiler inputs;
- a fitted reference ensemble, trusted qualifier, and qualification result;
- development signal, forward, and electronics implementations;
- an independent audit steward, firewall, engine, and sealed payload;
- a selected empirical or phantom source with disjoint partitions;
- a frozen catastrophic-plausibility threshold and scientist signoff;
- a resource compiler or qualification evaluator; and
- any candidate-discriminating outcome.

Still missing, but not an automatic block to the narrower paper-derived study:

- raw NeuroCam traces, a PDK, compact model, netlist, design rules, layout,
  DAQ code, exact reduced-mode maps, marker semantics, and per-mode power data;
  and
- approval of the two legacy JSON examples as documentary context.

These missing device assets narrow the permitted claim and enlarge model
uncertainty. They are not permission to guess parameters, weaken the
comparison, or silently describe the reference as a physical digital twin.
