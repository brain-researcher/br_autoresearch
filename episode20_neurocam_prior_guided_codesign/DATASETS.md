# Dataset and Source Contract — Episode 20

EP20 is a virtual sensor co-design study. Its evidence sources are therefore a
mixture of immutable publications, device-characterization anchors, executable
field generators, source-to-surface forward models, electronics models, and an
empirical replay or known-input physical diagnostic. These roles must remain
separate.

This document is a local source and access contract. It does not authorize a
download, contact an author, expose an audit engine, run a simulator, or open a
device measurement. No candidate-discriminating source is currently
provisioned under `inputs/`.

## Source-role summary

| Source class | Fixed role | Candidate may see outcomes? | Current state |
| --- | --- | --- | --- |
| NeuroCam final article and supplement | Documentary architecture and characterization anchors | Yes, before search | Documentary PDF is stored under `private_steward_acquisition` as `asset_ep20_neurocam_documentary`; no episode-local handoff or rights record |
| Episode-local reference example bundles | Nonbinding documentary context and non-candidate-discriminating simulation sanity checks only | Yes, before search | Absent from this canonical checkout; `legacy_ep20_reference_bundles` is not a provisioned EP20 input |
| Reference calibration-anchor manifest | Fit the paper-derived reference-model ensemble | Yes | Absent; fit remains input-gated |
| Reference qualification-anchor manifest | Public documentary anchors excluded from fitting by a frozen role | Published values are visible; the fit may not use them and candidates may not tune from qualification residuals | Absent; qualification remains input-gated and unrun |
| Development surface-field and biophysical generators | Adaptive candidate development | Yes, through frozen aggregate evaluator | Not implemented |
| Development electronics/device ensemble | Adaptive candidate development | Yes, through frozen aggregate evaluator | Not implemented |
| Sealed structural audit engine | One-shot virtual audit after lock | No pre-lock access | Source and steward absent |
| Development empirical replay partition | Spectral, amplitude, missingness, and software-stability stress only | Yes, through frozen development evaluator | Exact source/partition absent |
| Sealed empirical or known-input plausibility partition | Post-lock catastrophic-failure diagnostic; never dense ground truth by assumption | No pre-lock outcome access | Exact source/partition absent |
| PDK, netlist, mask layout, raw NeuroCam traces, wafer/device data | Potential future qualification evidence | Role must be assigned before access | No public accession or local handoff identified |
| Fabricated candidate tile/full array/in-vivo comparison | Future physical episode only | No EP20 role | Absent and not authorized |

## Physical-location record

The canonical checkout has no repository-local `.steward_acquisition` tree.
The root
[`DATA_LOCATION_MANIFEST.json`](../DATA_LOCATION_MANIFEST.json) identifies the
canonical private steward root as `private_steward_acquisition` at
`/oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/steward_acquisition`
and groups its NeuroCam asset as `asset_ep20_neurocam_documentary`. The
steward tree was relocated by same-filesystem rename on 2026-09-24.
The two legacy JSON examples are separately inventoried as
`legacy_ep20_reference_bundles` with no approved destination.

This location record does not provision the PDF, metadata contracts, example
bundles, raw traces, or any role-filtered handoff into EP20. It carries no
effect on source qualification or protected-outcome access.

## Immutable documentary reference

| Item | Fixed identity |
| --- | --- |
| Device | NeuroCam flexible multiplexed micro-ECoG array |
| Final article | Xie et al., *High-resolution spatial mapping of electrocorticographic activities with NeuroCam: a 4096-channel, multiplexed flexible thin-film transistor array* |
| Journal | Science Bulletin 70 (2025), 4133-4137 |
| DOI | `10.1016/j.scib.2025.11.030` |
| Publisher item identifier | `S2095927325011582` |
| PubMed | `PMID:41309324` |
| Preprint | `10.1101/2024.08.18.608446` |
| Supplementary-material status | Observed inside the author-hosted combined PDF; publisher supplement-file identity and separate byte inventory remain unconfirmed |
| Draft-time remote observation | Author-hosted combined article-plus-supplement PDF, observed 2026-09-21; 2,947,500 bytes; SHA-256 `d6e4f4da2001421849247d6f7eeda7958f97309636de077c90370c2eb1a4464e` |
| Observed remote URL | <https://shengxingstars.github.io/www/images/publications/2025_High-resolution%20spatial%20mapping%20of%20electrocorticographic%20activities%20with%20NeuroCam%20a%204096-channel,%20multiplexed%20flexible%20thin-film%20transistor%20array.pdf> |
| Rights policy | Verify and record separately for final article, supplement, preprint, figures, and any author-supplied files before copying or redistribution |
| Current local payload | Absent from this episode |

Before reference fitting or candidate scoring, freeze the final article,
supplement, preprint version used for cross-checking, landing-page metadata,
acquisition timestamps, byte sizes, and SHA-256 hashes. The final article is
the scientific citation. The preprint can help document version history but
cannot silently replace the final methods or supplement.

The draft-time remote hash is a reproducibility observation, not a provisioned
input or proof that the combined author-hosted file is the official immutable
article/supplement split. Reference fitting still requires a rights record,
acquisition timestamp, local read-only handoff, complete inventory, and fresh
byte hashes.

The published report and supplement provide aggregate documentary anchors.
They are not raw device data, a process design kit, a transistor netlist, a
mask set, a complete DAQ schematic, or a validated digital twin.

## Episode-local reference examples

No JSON reference bundle is provisioned under `inputs/reference_bundles/` in
this canonical checkout. Two small examples exist only in the legacy
worktree. They are inventory candidates, not EP20 inputs, and remain absent
unless a future trusted provisioning step copies, hashes, and role-labels them.
If provisioned, their intended roles would be:

- `neurocam_paper_direct_v1.json` transcribes paper-direct summary anchors and
  is marked `reference_only`, `example_only`, and not estimation-only.
- `neurocam_figure_derived_v1.json` contains assumption-bound algebraic and
  manual visual estimates and is marked `reference_only`, `example_only`, and
  `estimation_only`.

Such files would be nonbinding examples. They are not authoritative for the
reference-model fit or qualification split, are not candidate-score inputs,
cannot initialize model parameters or the search space, and do not qualify
reference fitting or candidate scoring. They cannot substitute for raw I-V/C-V,
compact-model,
PDK, netlist, mask/layout, or fabrication-outcome assets. The planned
`NEUROCAM_ANCHORS.yaml` and `REFERENCE_ANCHOR_SPLIT.yaml` would be role
contracts for future paper-derived calibration and qualification work; neither
file is currently present here.

## Paper-supported architecture anchors

The immutable anchor manifest must preserve the original units, figure/table
locations, extraction method, and whether a value is stated in text, read from
a table, or digitized from a plot.

### Array and process

| Anchor | Published value |
| --- | --- |
| Pixel lattice | `64 x 64` |
| Pixel count | 4096 |
| Pixel pitch | `150 micrometres x 150 micrometres` |
| Active area | approximately `9.6 mm x 9.6 mm` |
| Gold sensing pad | approximately `70 micrometres x 135 micrometres` |
| Array thickness | less than 30 micrometres |
| Flexible substrate | 25-micrometre polyimide |
| Active device | lanthanide-doped InZnO TFT |
| Addressing | 64 gate lines and 64 source lines |
| Array fan-in/fan-out | 128 gate-plus-source lines |
| Representative TFT mobility | 15.13 square centimetres per volt-second |
| Representative threshold voltage | 2.7 V |
| Gate on/off values | approximately +6 V / -2 V |
| On/off ratio | approximately `6 x 10^6` |

The EP20 accounting convention, not a paper-stated device anchor, is that
reference, ground, guard, power, and external DAQ wiring or resources are
counted separately from those 128 array gate-plus-source lines.

### Electrical and dynamic-mode anchors

| Anchor | Published value or role |
| --- | --- |
| Representative channel gain | `0.78 +/- 0.12` |
| Representative 3-dB frequency | above approximately 1 kHz |
| Device impedance | approximately 100 kilo-ohms over the reported 1-100 kHz range; exact curve retained rather than reduced to one number |
| Row dwell | approximately 62.5 microseconds in the illustrated full scan |
| Samples used per dwell | stable third through fifth points from six or seven raw points, according to the supplement |
| `24S x 16G` | 384 pixels, 1000 S/s effective rate, `8.8 +/- 4.2` microvolts RMS |
| `24S x 64G` | 1536 pixels, 250 S/s effective rate, `12.6 +/- 8.4` microvolts RMS |
| `64S x 64G` | 4096 pixels, 250 S/s effective rate, `63.6 +/- 24.1` microvolts RMS |
| High-precision static mode | 23 signal source lines plus one ADC marker channel; lowest stated noise `2.3 +/- 0.3` microvolts RMS at 100 kS/s |
| First neighbour, source-line direction | approximately -12.6 dB crosstalk |
| First neighbour, gate-line direction | approximately -37.2 dB crosstalk |
| Second-neighbour and farther | approximately -45 dB crosstalk |

The published noise values are characterized under stated modes and
conditions. They cannot be treated as a universal in-vivo noise distribution,
and values from different bandwidths, filtering rules, channel counts, or
sample aggregation cannot be pooled without a frozen conversion.

Every published `+/-` value is stored verbatim as `reported_plus_minus`, with
`dispersion_semantics: unspecified` unless the frozen source extraction proves
whether it is SD, SEM, a fitted distribution width, or another statistic. A
cross-channel RMS histogram must not be converted silently into independent
white temporal noise.

The reported crosstalk anchors come from a controlled air, single-pixel,
approximately 5-mVpp/12-Hz test. They do not identify a general in-vivo or
frequency-dependent crosstalk kernel. Likewise, a reported 1--199 Hz filter
setting does not override the approximately 125-Hz Nyquist limit of the
250-S/s full-array mode.

### Resource accounting derived from the stated modes

The resource manifest must distinguish at least four quantities:

1. number of physical source lines acquired in parallel;
2. raw ADC conversions per second before within-dwell aggregation;
3. effective per-pixel output samples per second; and
4. output bits per second after the frozen representation and metadata.

Using the reported 100-kS/s raw sampling per active source line, the three
dynamic anchors imply approximately 2.4 million raw conversions/s for either
24-source mode and 6.4 million raw conversions/s for the 64-source mode. Their
effective field-sample outputs are 384,000, 384,000, and 1,024,000 pixel
samples/s, respectively. These calculations are documentary checks, not a
complete energy or ADC-cost model.

The external DAQ description uses three PXIe-4309 cards and one PXIe-6739
card. The supplement's first sentence labels the modules' input/output roles
in a way that conflicts with its later description of PXIe-4309 ADCs. Preserve
this as an unresolved source inconsistency. Resolve it from the exact hardware
manuals, acquisition code, schematic, or author clarification before freezing
the resource compiler; do not silently correct the source text.

The anchor manifest must also preserve three other unresolved documentary
limits: the main figure caption and supplement give 10-mVpp versus 1-mVpp
frequency-response test amplitudes; the stated 6.25 oversampling ratio may
describe samples within a 62.5-microsecond dwell rather than ADC-internal
oversampling; and the exact spatial subsets and channel identities used by the
two reduced modes are not reported. None may be guessed from the aggregate
plots.

## Reference-model calibration and qualification roles

The final anchor split must be frozen before any fitting. A proposed
aggregate-only split is:

### Calibration-visible anchors

- physical dimensions, addressing topology, and stated gate timing;
- representative gain distribution;
- static high-precision noise anchor;
- `24S x 16G` dynamic rate/noise anchor;
- `24S x 64G` dynamic rate/noise anchor; and
- published impedance and representative frequency-response curves.

### Fit-held-out qualification anchors

- `64S x 64G` effective rate and noise distribution;
- source-direction, gate-direction, and distant crosstalk anchors;
- withheld points digitized from the mode-dependent noise curves;
- settling or switch-transient traces if independently obtained; and
- held-out device, wafer, or soak/bend measurements if raw data become
  available under a frozen role.

This proposed split is insufficient if only the aggregate text values are
available. Before reference fitting or candidate scoring, outcome-blind
recovery simulations must show that
the held-out anchors discriminate incorrect scaling laws. If they do not, raw
traces, additional devices, or an independently justified model range are
required. Inability to provision a discriminating gate leaves the reference
model unqualified. If a frozen gate is actually run and fails, the terminal
is `technical_reference_model_unqualified`, not a scientific negative.

These paper values are public, so this is a fit holdout, not an access-hidden
external validation set. The anchor roles are frozen before fitting; the fit
implementation is prohibited from consuming qualification values. The gate
returns a receipt, pass/fail, predeclared residual summaries, and the frozen
ensemble. Candidate hardware/software selection may not tune from those
residuals. Any independently acquired raw qualification trace receives a
separate access role and is not made public merely by joining this manifest.

## Missing NeuroCam assets and honest fallback

The current contract has no authenticated handoff for:

- raw in-vitro or in-vivo voltage traces;
- raw noise and crosstalk recordings;
- LabVIEW acquisition code;
- exact scan-control waveforms;
- exact reduced-mode spatial subsets, channel identities, and marker channel;
- PCB/DAQ schematic and channel map;
- transistor netlist or compact model;
- process design kit and exact lithographic design rules;
- mask layout or parasitic extraction;
- wafer-, array-, and channel-level yield tables;
- current/power/thermal traces by operating mode; or
- independently withheld devices.

Before an author-supplied asset is opened or candidate scoring begins, search
and correspondence exposure must be recorded even when no asset is obtained.
An author-supplied asset receives an immutable source record, terms-of-use
record, checksum, and role assignment before it is opened.

If these assets remain absent, EP20 may proceed only if the scientist accepts
the narrower object `paper-derived NeuroCam-class reference-model ensemble`, the
aggregate-only qualification is shown to be discriminating, and every claim
retains that limitation. The following phrases remain prohibited:

- exact NeuroCam digital twin;
- transistor-accurate reproduction;
- foundry-ready or fabrication-ready layout;
- matched physical power, reliability, or chronic lifetime; and
- superiority over the fabricated NeuroCam system.

## Development signal environments

Development generators are outcome-visible to the adaptive controller only
through the frozen evaluator. Their code, parameter ranges, mixture weights,
seeds, and empirical calibration sources are frozen before the first candidate
score.

Each generator instance emits either a dense surface potential `phi` directly
or a latent source `S` plus source-to-surface operator `L`. It also emits
mechanism labels and exact masks required for the registered secondary
endpoints.

The development bank must contain all of the following executable families:

| Family | Required frozen factors |
| --- | --- |
| Broad correlated field | temporal PSD, spatial PSD, correlation length, anisotropy, amplitude, common-mode fraction |
| Focal field transient | aperture, duration, amplitude, event rate, location distribution, spectral envelope |
| Translating/expanding wave | direction, speed, wavelength, curvature, onset, boundary interaction |
| Multiscale mixture | broad/local energy ratio, cross-component dependence, overlap, nonstationarity |
| Interacting sources | number, separation, synchrony, phase relation, collision rule |
| Silent-region surprise | pre-event silence duration, unexpected location, event amplitude and duration |
| Nuisance family | line noise, reference drift, motion-like transient, contact gap, dropout, gain drift, saturation |

Parameter ranges must cite either an empirical source, a physical bound, or an
explicitly conservative design interval. A verbal label such as `smooth` or
`focal` is not a source contract.

Development seeds are common across all designs. A new seed cannot be
introduced because one candidate performed poorly. Generator and device draws
are stored as identifiers and hashes rather than regenerated from an
unversioned library.

## Source-to-surface forward models

Primary reconstruction is scored on dense cortical-surface potential. Latent
source recovery is diagnostic only.

The development forward-model panel must include at least:

- one analytic homogeneous or layered-medium operator;
- one anisotropic or spatially heterogeneous operator;
- pad-to-surface distance and curvature variation;
- a fixed external-reference convention; and
- contact-gap and conductivity perturbations.

Before search, choose whether the higher-fidelity model is FEM, BEM, a
reciprocity-based solver, or an independently validated surrogate. Record
mesh/anatomy source, geometry units, conductivity provenance, boundary
conditions, solver/version, convergence tolerance, and surrogate error.

The sealed audit must include an implementation lineage that is independent of
the primary development operator. Sharing the same mesh, code path, numerical
kernel, or fitted surrogate with only new random seeds is a parameter holdout,
not a structural audit.

## Development electronics and process ensemble

The development ensemble is conditional on the compiled design. Changing pad
area, row dwell, source loading, or routing changes impedance, aperture
averaging, settling, crosstalk, and noise jointly; these quantities cannot be
independent sliders.

At minimum the ensemble models:

- pad spatial integration and interface impedance;
- TFT gain and threshold variation;
- source/gate line resistance and capacitance;
- row-switch transient and history-dependent settling;
- anisotropic crosstalk;
- raw ADC sampling, within-dwell aggregation, and quantization;
- common-mode and device noise conditional on operating mode;
- drift, saturation, dead pixels, and contact gaps; and
- reference and grounding convention.

Every structural scaling law must be labelled as measured, literature-derived,
physics-derived, or deliberately adversarial. The optimizer cannot choose the
model member or discard an unfavourable member.

## Sealed virtual audit source

The audit is not `development generator plus new seed`. Before development,
an audit steward freezes and hashes:

- an independently implemented structural signal generator;
- at least one source-to-surface operator absent from development;
- an independently implemented electronics scaling or coupling law absent
  from development;
- a nonstationary or moving-source family absent from development;
- an altered spatial-spectrum family;
- an unseen combination of noise, crosstalk, settling, drift, contact gap, and
  dead-pixel state;
- audit-only seeds and device draws;
- scoring masks and aggregation units; and
- the exact evaluator command and allowed report fields.

The controller receives no audit parameter, trace, preview, score, rank, or
candidate-discriminating QC before configuration lock. Static facts required
to compile the frozen interface may be exposed only if they are identical for
all candidates and listed in the exposure ledger.

The audit bank remains a virtual model audit. It does not become an independent
biological or physical replication merely because its code is separately
authored.

The audit commitment includes a dependency report demonstrating that the
independent signal, forward, and electronics implementations do not import the
development simulator. Empirical replay is an additional plausibility gate;
it cannot replace the independent electronics shift for an unbuilt geometry.

## Empirical replay or known-input phantom

An exact source has not been selected. Before empirical replay or audit
evaluation, freeze one of the following source types:

1. a legally reusable, sufficiently dense raw ECoG recording with channel
   geometry, timestamps, reference, hardware, sample rate, and preprocessing
   provenance; or
2. a known-input PBS/phantom waveform-replay dataset with spatial input map,
   raw array outputs, scan markers, operating mode, and independent holdout
   traces.

Ordinary real ECoG is already filtered by its recording contacts and reference.
It cannot be treated as true continuous cortical-surface voltage for comparing
arbitrary pad shapes. Its EP20 role is limited to spectral/amplitude coverage,
missingness behaviour, software stability, and prespecified plausibility
failures.

Accordingly, development replay is restricted to `D0`/`D1` software and
simulator-envelope stress. Its outcomes are excluded from alternative-hardware
objectives and may not score, rank, retire, or route a `D2`/`D3`/`D4`
successor. The sealed partition is a global catastrophic-plausibility gate,
not evidence that an unbuilt geometry produced the replayed waveform.

A known-input phantom can test more of the electronics chain, but measurements
from a NeuroCam reference device still do not validate an unbuilt candidate's
new geometry. Only a fabricated candidate and matched physical comparator can
support physical superiority.

Development stress and post-lock plausibility roles must use either separate
sources or content-addressed, disjoint partitions frozen before any outcome is
opened. The sealed plausibility partition must be separated at the whole
subject, device, session, or independently replayed waveform level; splitting
overlapping windows from one trace is not sufficient. No item, derived summary,
normalization statistic, or fitted threshold crosses from the sealed partition
into development. The catastrophic-failure threshold is calibrated without
opening the sealed partition.

Each replay/phantom manifest records DOI or owner, version, license or terms,
species/preparation where relevant, geometry, channel count, sample rate,
reference, filters, raw-versus-derived status, file inventory, checksums,
acquisition date, exact development or sealed-plausibility role, partition
unit, and non-overlap proof.

## Resource and design-rule sources

The hardware constraint compiler requires immutable inputs for:

- allowed pad shapes, dimensions, clearances, and within-pixel exclusion
  zones;
- transistor and via placement;
- gate/source trace widths, lengths, loading, and line-count accounting;
- legal row dwell and scan order;
- active source-bank configurations;
- resource-envelope IDs and frozen eligible/matched `D0`/`D1`/`D4`
  comparator maps;
- AFE and ADC core/channel allocation;
- raw conversions/s and oversampling;
- output word size, metadata overhead, and bit-rate;
- scan latency and skew;
- power and thermal proxy;
- routing/array area proxy;
- reference, ground, guard, and supply costs; and
- yield/dead-pixel and reliability proxies.

An unknown cost is not zero. If a candidate uses a primitive absent from the
reference source package, the primitive is prohibited until its area, noise,
power, timing, calibration, and manufacturing rules are frozen.

If no NeuroCam PDK or netlist is obtained, candidate compilation requires a
versioned, scientist-approved conservative surrogate rule deck. Every clearance,
aperture, routing, loading, and rounding bound must cite a measured anchor,
manufacturer/datasheet limit, fabrication literature source, or a declared
adversarial safety factor. Likewise, the power/thermal quantity remains a
proxy: its lower and upper component bounds, activity factors, conversion
cost, and uncertainty range must be frozen from defensible sources. Neither
same process-class naming nor a short thermal image makes an unknown
manufacturing or power cost equal to the incumbent.

The published two-week PBS soak and bending summaries may inform sensitivity
diagnostics but do not establish chronic in-vivo reliability. Chronic safety,
biocompatibility, and lifetime remain outside EP20.

## Required role-filtered handoffs

Before role-specific access, a steward creates content-addressed handoffs for:

1. `documentary_reference` — final article, supplement, source metadata, and
   anchor manifest;
2. `reference_calibration` — only anchors and traces permitted for fitting the
   reference ensemble;
3. `reference_qualification` — public role-locked documentary anchors plus any
   separately governed evaluator-only raw traces;
4. `development_signal` — development generators, parameter manifests, and
   permitted empirical calibration summaries;
5. `development_device` — development electronics/process ensemble;
6. `audit_virtual` — permission-separated structural audit engine and payload;
7. `empirical_replay_development` — development-visible stress partition;
8. `empirical_replay_plausibility` — disjoint post-lock plausibility partition;
   and
9. `hardware_rules` — design-rule, resource, and compiler-test manifests.

A shared folder containing calibration and qualification traces is not a
qualification seal. Public availability of a paper, preprint, code repository,
or dataset does not make audit access acceptable. The adaptive runtime must be
unable to read or redownload audit content.

## Exact manifests required before outcome access

The following names are reserved under read-only `inputs/`:

```text
SOURCE_MANIFEST.json
NEUROCAM_ANCHORS.yaml
REFERENCE_ANCHOR_SPLIT.yaml
REFERENCE_MODEL_CONTRACT.yaml
REFERENCE_QUALIFICATION.yaml
FROZEN_HARDWARE_GRAMMAR.yaml
DESIGN_RULES.yaml
RESOURCE_MODEL.yaml
SOFTWARE_BUDGET.yaml
DEVELOPMENT_ENVIRONMENTS.yaml
DEVELOPMENT_DEVICE_ENSEMBLE.yaml
DEVELOPMENT_DRAW_MANIFEST.json
AUDIT_COMMITMENT.json
EMPIRICAL_DEVELOPMENT_REPLAY_MANIFEST.json
EMPIRICAL_PLAUSIBILITY_REPLAY_MANIFEST.json
EXPOSURE_LEDGER.json
EVALUATOR_CONTRACT.yaml
```

Every manifest has a schema version, creator, creation timestamp, source
revision, complete file hashes, role, permitted reader, and parent-manifest
hash. The episode lock binds all of them.

`FROZEN_HARDWARE_GRAMMAR.yaml` contains a finite pad catalogue and exact legal
scan/resource choices. `SOFTWARE_BUDGET.yaml` freezes the eligible `s1`
subclasses, capacity ceiling, training steps, data/seed allocation, optimizer,
and selection-call credit. `AUDIT_COMMITMENT.json` exposes identity, stewardship,
interface hash, sealed hidden-payload Merkle root, dependency-report hash, and
permitted report schema to development; it does not expose hidden parameters
or outcomes. The later audit receipt must match that payload root exactly.

`REFERENCE_MODEL_CONTRACT.yaml` freezes the model grammar, fitting code hash,
allowed calibration inputs, output schema, and deterministic seed derivation;
it is not a fitted model. `REFERENCE_QUALIFICATION.yaml` is the frozen pre-run
gate and anchor-role specification. After an authorized gate run, the trusted
qualifier creates immutable runtime artifacts
`outputs/reference_model_ensemble_manifest.json` and
`outputs/reference_qualification_receipt.json`. They contain the fitted
payload hash and provenance, contract hash, pass/fail, allowed residual
summary, execution environment, and timestamp. Both enter the later
configuration lock and are never written under `inputs/`.

## Inputs directory contract

`inputs/` is read-only after provisioning. Do not place generated candidates,
logs, caches, model weights, scores, or audit reports there. Large immutable
payloads may remain in a separately governed data store, but `inputs/` must
contain content-addressed role-filtered manifests or read-only handoffs that
bind the exact bytes.

No mixed link to a source tree containing both development and audit content
is permitted. The episode directory stores contracts and small manifests, not
an uncontrolled duplicate of papers, device traces, or large simulations.

## Current source and implementation state

At drafting time:

- the final paper, DOI, preprint DOI, and aggregate anchors are identified;
- the documentary PDF is stored under `private_steward_acquisition` as
  `asset_ep20_neurocam_documentary` after the same-filesystem relocation on
  2026-09-24, but is not an episode-local handoff and does not qualify reference
  fitting or candidate scoring;
- the five planned metadata-only paper-derived proxy contracts are absent;
- the two legacy JSON examples are not provisioned in this canonical episode;
- no absent contract or legacy example is configuration-lock eligible or
  authorizes reference fitting or candidate scoring;
- no immutable paper/supplement copy or hash is present in this episode;
- no raw NeuroCam calibration handoff, PDK, netlist, DRC, layout, or DAQ code is
  provisioned;
- no exact reduced-mode channel/subset map or marker semantics is provisioned;
- no finite legal alternative-pad catalogue is frozen;
- no development or audit generator is implemented;
- no independent audit steward or access boundary is established;
- no empirical replay or phantom source is selected;
- no resource compiler or qualification evaluator exists; and
- no candidate-discriminating signal or device outcome has been opened here.

This episode documentation specifies planned paper-derived proxy definitions
and example roles only. It adds no provisioned reference bundle, real I-V/C-V
payload, compact model, PDK, raw NeuroCam trace, fabrication outcome, execution
authority, or physical-device validation.

These gaps block reference fitting, candidate scoring, and audit access; they
are not invitations to substitute guessed parameters or weaken the claim
silently.
