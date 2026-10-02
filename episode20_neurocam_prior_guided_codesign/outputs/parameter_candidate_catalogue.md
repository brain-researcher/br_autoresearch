# EP20 — starting field ranges and pad/scan candidates

Design selection, 2026-09-30. Start with half-area two-scale pads, a small
set of explicit timing recipes, and fields spanning broad, local, slow and
rapid activity. Quarter-area pads and coverage-preserving adaptive revisits
remain follow-ups. This is a starting catalogue, not the entire search space:
legal proposals can still be added during development, before held-out access,
within the existing 32–64 trials and challenge allowance.

The entries below distinguish reported observations, mathematical consequences,
and deliberately selected challenge values. None is a candidate result or an
approval of manufacturing feasibility. The unresolved via/keepout rules and
physical-resource costs still determine eligibility before scoring.

## 1. Evidence that sets the scale

| Primary source | Reported context and anchor | Use in EP20 |
| --- | --- | --- |
| [Xie et al., NeuroCam final article and supplement](https://shengxingstars.github.io/www/images/publications/2025_High-resolution%20spatial%20mapping%20of%20electrocorticographic%20activities%20with%20NeuroCam%20a%204096-channel,%20multiplexed%20flexible%20thin-film%20transistor%20array.pdf), main text p. 3 and Fig. S6 | Penicillin-induced rabbit epileptiform bursts: approximately 3 Hz, 150 ms and 2 mV peak-to-peak; PBS test frequencies: 3, 7, 11, 19, 37 Hz | Pathological event context and known-input frequency anchors, not normal physiological distributions |
| [Rogers et al., 2019](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006769), Results: distance-averaged correlation | Human contact pitch 0.4 mm; mouse 0.2/0.25 mm. Frequency-dependent submillimetre structure; mouse correlation below 0.5 within 1.5 mm, human highest-frequency correlation above 0.5 to about 2 mm | Motivates a submillimetre-to-millimetre bracket; correlation distance is not `lambda90`, source width or unaveraged surface truth |
| [Zhang et al., 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6534129/), Summary and Fig. 3 discussion | Human memory-task ECoG waves: 2–15 Hz, approximately 0.25–0.75 m/s; median wavelength 11.7 cm | Broad-wave control: a 9.6-mm array observes only part of many such waves |
| [González-Ramírez et al., 2015](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004065), Results near Table 1 | Human intracortical seizure LFP, filtered 2–50 Hz; patient 1 speed about 0.08–0.38 m/s, pulse width 1.9–5.6 mm, reverberation 30–150 ms | Lower-speed pathological-wave context only: intracortical LFP is not surface ECoG, and pulse width is not wavelength |

These recordings already contain their contacts, reference and forward
conduction. Use them to constrain plausibility, not as dense unfiltered truth
on which alternative pads can be retrospectively averaged.

## 2. Selected development controls

Except for the identified anchors, the following are deliberate finite design
values, not estimated physiological percentiles. Generate a field with its
declared controls, then compute the registered `lambda_k` and `tau_k` from
common-mask truth. Do not rename a covariance width, envelope duration or
published correlation distance as either input.

| Control | Starting values | Definition / rationale |
| --- | --- | --- |
| Broad spatial width `ell` | 0.3, 0.6, 1.2, 2.4, 4.8 mm | Gaussian surface covariance `K(h)=exp(-norm(h)^2/(2*ell^2))`; logarithmic bracket from twice the pitch to half the aperture. Source-mediated fields require their actual surface spectrum to be measured separately |
| Anisotropy | Axis ratio 1, 2, 4; orientation 0°, 45°, 90°, 135° | Minor-axis width is ell, major-axis width is ratio times ell; orientation is the major-axis direction. Explicit challenge, not an empirical frequency of anisotropy |
| Temporal carrier | 3, 7, 11, 19, 37, 80 Hz | First five are instrument-input anchors above; 80 Hz stresses the upper primary band. A broadband envelope does not imply `tau=1/(2*carrier)` |
| Broad-field amplitude | 10, 50, 200 µV primary-band RMS | Deliberate weak/moderate/large signal bracket on the common target; not converted from reported peak-to-peak amplitudes |
| Focal spatial envelope | Gaussian FWHM 0.15, 0.6, 2.4 mm | One-pitch, four-pitch and broader events; one-pitch events are resolution challenges, not established physiological prevalence |
| Focal temporal envelope / repetition | Gaussian envelope FWHM 20, 50, 150, 300 ms; 0.5 or 3 events/s | Deliberate duration bracket. The paper's approximate 150-ms duration was not reported as Gaussian FWHM |
| Focal amplitude | 0.2, 1, 2 mV peak-to-peak in the primary-band truth event | Only the approximately 2-mV pathological context is a reported in-vivo anchor; envelope/biphasic waveform and post-filter normalization are common to all designs |
| Broad/local mixtures | Component-power fraction 0.1, 0.5, 0.9 | Define `P_local/(P_broad+P_local)` on the common target/mask before cross terms. Report covariance and summed power separately; this is not necessarily the local fraction of total summed power |
| Interacting sources | 2 or 4 components; separation 0.3, 1.2, 4.8 mm; independent, shared/in-phase or opposed-phase drivers | Linear superposition is the starting collision rule; no nonlinear interaction is attributed to a linear conduction operator |
| Silent-region surprise | Quiet prelude 0.5 or 2 s; event position independent of scheduler history | Never select locations to favor a pad map or active row set |
| Nuisance overlays | 50/60-Hz line signal with RMS 0.1 or 1 times target RMS; 0, 1, 5% random dropout versus one complete dead row | Deliberate robustness levels, not measured NeuroCam failure rates. Truth remains nonzero; missing observations do not shrink its mask |

Plane-wave speed, carrier and wavelength are coupled by `v=f*L`. Start with
these tuples rather than a Cartesian product:

| ID | Carrier Hz | Speed mm/s | Wavelength mm | Interpretation |
| --- | ---: | ---: | ---: | --- |
| W-wide | 7 | 400 | 57.143 | Broad human-ECoG-context control; parameters lie within the cited frequency/speed intervals, not a replicated observation |
| W-aperture | 11 | 100 | 9.091 | Deliberate lower-speed, aperture-scale control informed by the seizure context |
| W-local | 37 | 100 | 2.703 | Deliberate millimetre-wave challenge, not an observed surface-wave tuple |
| W-fast | 80 | 100 | 1.250 | Upper-band extrapolation beyond the cited 2–50-Hz seizure analysis |

Use directions 0°, 90°, 180°, 270°, and planar versus expanding wavefronts
with radius 1.2 or 4.8 mm. Phase-offset pairs 0 and pi/2 are mathematical
scan-coupling challenges. Confirm finite-aperture spectral matching from truth
before calling two conditions equal in `lambda/tau`; direction and phase are
not new inputs to the S/T rule.

Separately retain ideal narrowband wavelengths 0.30 and 0.15 mm at 37 Hz:
their ideal half-wavelength scales are 150 and 75 µm, at and below the lattice
Nyquist scale. Their corresponding speeds are 11.1 and 5.55 mm/s. These are
explicit aliasing challenges, not literature-supported physiological waves.
Do not make them the dominant evidence for smaller pads.

### A small condition panel, not a full grid

Begin with four conditions per existing family, 28 condition classes inside
the design trials. Unless specified otherwise, broad truth uses `ell=1.2 mm`,
11-Hz dynamics, 50-µV RMS and isotropy. Events use independently drawn
locations; every design receives identical worlds. The event-envelope choice
is common: a biphasic waveform within a Gaussian envelope truncated at three
standard deviations on each side, with a smooth taper over the outer half
standard deviation. FWHM is the nominal pre-truncation envelope control;
report the achieved waveform width too. Normalize amplitude on primary-band
truth. Overlapping repeated events are retained, not rejected.

A silent prelude means zero local event contribution before the finite-support
event starts in the full forward truth; broad activity elsewhere can remain.
Primary-band evaluation filtering may introduce pre-onset ringing, which is
retained and reported rather than forced to zero or made available to the
scheduler. This specifies contrasts, not an executed generator.

| Family / condition IDs | Four starting conditions |
| --- | --- |
| Broad B1–B4 | `(ell mm, carrier Hz, RMS µV)`: (4.8,3,50), (1.2,11,50), (0.3,37,50), (0.6,80,10); B4 axis ratio 4 at 90° |
| Focal F1–F4 | `(spatial FWHM mm, envelope FWHM ms, mVpp, events/s)`: (2.4,300,0.2,0.5), (0.6,150,2,3), (0.15,20,0.2,0.5), (0.6,50,1,3) |
| Waves W1–W4 | W-wide at 0°; W-aperture at 90°; W-local at 0° and 90° on matched draws. W-fast and reverse directions are follow-up challenges |
| Mixtures M1–M4 | Broad `ell=2.4 mm`, 3 Hz plus local `ell=0.3 mm`, 37 Hz: component-power fractions 0.1, 0.5, 0.9 with independent drivers; M4 common envelope/innovation filtered into the two declared bands, with fraction switching 0.1→0.9 every 1 s |
| Interacting I1–I4 | Two 0.6-mm-FWHM components: separation 4.8 mm/independent, 0.3 mm/in-phase, 1.2 mm/opposed-phase; four components with minimum separation 0.3 mm/independent. Component sum normalized to 50-µV target RMS |
| Surprise S1–S4 | F2 after 0.5 s silence/uniform location; F2 after 2 s/edge-region location; F3 after 0.5 s/uniform location; F4 after 2 s/four new locations. Event positions and onset offsets are independent of acquisition history |
| Nuisance N1–N4 | Default broad truth plus 60-Hz/0.1× line RMS; 50-Hz/1×; 1% random dropout; one dead row. Five-percent/clustered dropout, gaps, drift and saturation are existing follow-ups, with device constants still to be specified |

Condition classes are not additional hardware trials. This panel neither
guarantees five conditions per S/T cell nor guarantees any winning family.
Use the registered absolute inputs and frozen gates to establish cell support;
fill missing support without inspecting reconstruction outcomes. Conductivity,
source distance, curvature, contact-gap and electronics-law ranges remain
separate specification work; this table does not invent their device values.
The already-required common-mode challenge adds a spatially constant component,
with component-power fractions 0, 0.1 or 0.5; also retain a pure constant-spatial
control. Do not spatially demean that control to manufacture a finite lambda.

## 3. Three pad classes, two classes per starting geometry

| Class | Dimensions µm | Area µm² | Area relative to published pad | Role |
| --- | ---: | ---: | ---: | --- |
| L | 70 × 135 | 9,450 | 1.000 | Published baseline |
| M | 50 × 95 | 4,750 | 0.503 | Selected approximately half-area footprint |
| S | 35 × 70 | 2,450 | 0.259 | Selected approximately quarter-area follow-up |

M/S are deliberate approximately self-similar subfootprints of L, not measured
NeuroCam alternatives. Five-micrometre rounding is a catalogue discretization,
not a foundry resolution claim. Retain the published orientation, material,
pitch, lattice, TFT topology and routing family. No new passivation-aperture
process or rotated pad is silently substituted.

For zero-based `r,c=0,...,63`, place the smaller class `q` where

`(floor(r/b) + floor(c/b)) mod 2 = 1`,

and L elsewhere. Each map has exactly 2,048 L and 2,048 q pads.

| Geometry | q | b | Role |
| --- | --- | ---: | --- |
| G0 | — | — | Published uniform L |
| G-I2-M | M | 2 | Initial interleaved 2×2-block proposal; block width 0.3 mm |
| G-T8-M | M | 8 | Initial tiled 8×8-block proposal; block width 1.2 mm |
| G-I2-S | S | 2 | Smaller-pad follow-up |
| G-T8-S | S | 8 | Smaller-pad follow-up |

Tiles describe pad assignment only: they do not add tile aggregation, memory,
compute or wires. For each candidate's exact class pair/counts, C4 uses a
pixel checkerboard (`b=1`), coarse tiles (`b=16`, 2.4-mm blocks), and one
balanced random placement fixed before scoring. Complementing a binary map
implements the already-required reverse-assignment challenge. These controls
are intentionally distinct from the starting b=2/b=8 maps. A later literal
duplicate cannot earn the required positive gain over itself.

Under a common pad-centre convention, nearest-neighbour top-metal gaps are:

| Adjacent classes | Short-axis gap µm | Long-axis gap µm |
| --- | ---: | ---: |
| L/L | 80 | 15 |
| L/M | 90 | 35 |
| L/S | 97.5 | 47.5 |

These are pitch-minus-half-width calculations, not routing/via design rules.
The article describes a stepwise via to the drain but does not resolve numeric
landing/keepout dimensions in the reviewed methods. Even a smaller rectangle
could remove its via landing. Exact placement, electrical contact and
clearances therefore remain unresolved; none of these alternatives is yet
certified physically legal.

For the existing uniform-area-averaging approximation,
`H(kx,ky)=sinc(a*kx)*sinc(b*ky)`, using normalized sinc and cyclic frequency.
At a 300-µm wavelength along the long pad axis, derived amplitude transmission
is L 0.699, M 0.843, S 0.913. This is a mathematical averaging contrast,
not an observed response. Broad fields can show almost no such gain. Do not
attach an assumed noise, impedance, RC or settling improvement to smaller area.

## 4. Explicit scan recipes and their coverage cost

The published timing anchor is `d=62.5 µs`; allowed dwells remain `d,2d,4d`.
The supplement describes digitized samples 3–5 from six/seven within a base
row dwell. Use actual clock phase; do not round 6.25 to an integer sample count.
Longer-dwell aggregation still needs a common registered rule and device model.

Keep the complete three-setting C0/C1 frontier: 64-source/64-row at 250 Hz,
24-source/64-row at 250 Hz, and 24-source/16-row at 1,000 Hz. Their exact
reduced-mode maps/marker occupancy remain unresolved; the proposed masks below
must not be represented as published maps.

For proposed recipes, use zero-based rows:

`P16={0,4,...,60}`, `B16={2,6,...,62}`, `E32={1,3,...,63}`, `P8={0,8,...,56}`.

Within a listed set, visit rows in ascending order. All 64 source streams stay
active for the initial novel row recipes; this avoids claiming a new
bank-to-ADC precision map.

| Recipe | Repeating event sequence | Duration / actions | Maximum revisit gap / coverage | Retained outputs/s, before marker overhead |
| --- | --- | --- | --- | ---: |
| P-full | All 64 rows, dwell d | 4 ms / 64 | 4 ms; all rows | 1,024,000 |
| N-R32 | P16 → P16 → B16 → P16, dwell d | 4 ms / 64 | P: 2 ms; B: 4 ms; 32 rows never acquired | 1,024,000 |
| N-Q48 | Ascending P16 union E32; P dwell 2d, E dwell d | 4 ms / 48; 64 base-dwell units | 4 ms on 48 rows; 16 rows never acquired | 768,000 |
| N-Q40 | Ascending P8 union E32; P dwell 4d, E dwell d | 4 ms / 40; 64 base-dwell units | 4 ms on 40 rows; 24 rows never acquired | 640,000 |

N-R32 gives P rows 750 visits/s on average, but its maximum gap is 2 ms:
the observations are irregular, not uniform 750-Hz data. N-Q48/Q40 trade
coverage and publication count for more dwell on selected rows, not guaranteed
precision. All unobserved rows remain in the common dense-grid score. No
candidate receives a smaller mask or zero-cost missingness.

These recipes have the same nominal digitized-source work proxy,
`64*100,000=6.4 million samples/s`, but not automatically equal physical
conversion work, power or latency. Gate transitions also differ: N-Q48/Q40
use 12,000/10,000 row activations/s versus P-full's 16,000; each has longer
individual dwells. Count the actual transition sequence and persistent
electronics state, not only the macrocycle duration.

### Coverage-preserving follow-up: one fixed scan and one adaptive rule

Use these only at a **common, predeclared budget permitting 5-ms gaps and the
same reporting deadline for all arms**. They are not eligible by silently
relaxing a 4-ms limit.

Let Q=P16 and B be its 48-row complement. Each 40-action half contains eight
blocks `[Q[2j],Q[2j+1],B[3j],B[3j+1],B[3j+2]]`, `j=0,...,7`.
Use B[0:24] in the first half, B[24:48] in the second; every dwell is d.

- **N-R80 fixed:** repeat Q in both halves. Eighty actions/5 ms; Q gaps
  2.5 ms, every other row 5 ms. Complete spatial coverage is preserved.
- **N-A80, the single bounded adaptive rule:** keep first-half Q and every
  B slot compulsory at the same timestamps. Replace only the 16 second-half
  Q slots by the row with largest current predicted voltage uncertainty,
  aggregated across that row; tie-break by row index. Use only observations
  published by the action start and the frozen inverse's permitted information.
  Every row still has a compulsory visit every 5 ms. If a timely uncertainty
  update is unavailable, use that slot's fixed Q row, not a delayed action.

Policy updates, switches and settling are charged. The truth-derived scale
inputs/condition label never select bonus rows. Compulsory coverage makes
silent-region surprise a real challenge rather than an omitted region.

Fixed source-bank precision allocation remains in the open legal family, but
is not assigned a new initial recipe here: the marker slot, converter map and
bank-switch timing are not identified. No extra independent ADC is assumed.

### Three different resource quantities

Retained pixel outputs, pre-aggregation digitized source samples and physical
internal ADC conversions are different. NI's
[official PXIe-4309 allocation note](https://knowledge.ni.com/KnowledgeArticleDetails?id=kA0VU0000009lMH0AY&l=en-US)
explains that a requested 100 kS/s can involve 2 MS/s internal operation with
one input per ADC, with different oversampling/settling when inputs share an
ADC. This does not identify NeuroCam's actual internal configuration.

Thus the existing 2.4/6.4-million "raw conversion" estimates are source-stream
proxies, not established physical converter-clock counts. Preserve their
reported-setting interpretation while resolving internal work, marker/reference
overhead, discarded samples, precision, bitrate and power. Equal proxy rates
alone cannot establish the required physical-resource parity.

## 5. How this fills, rather than enlarges, the existing study

The four initial candidate hardware slots become:

| Initial proposal | Geometry / scan | C2/C3 pairing |
| --- | --- | --- |
| K-I | G-I2-M / P-full | Exact same hardware, conventional versus selected optimized inverse |
| K-T | G-T8-M / P-full | Same pairing |
| K-Q | G0 / N-Q48 | Same pairing; heterogeneous-dwell precision hypothesis |
| K-J | G-I2-M / N-R32 | Same pairing; joint geometry/revisit hypothesis |

Keep one complete-frontier C0 arm, three complete-frontier C1 software arms,
four C2, four matched C3 and four C4 arms: exactly 16 initial arms. C4 retains
the checkerboard, coarse-tile, balanced-random and uniform-geometry/best-legal-
schedule slots, with its finite schedule choice frozen before C3 scores and
charged the same tuning allowance. Candidate-specific class matching may
require evaluating control instances within their existing slots; it does
not make them independent extra search allocations.

The sentinel panel remains the published uniform design, interleaved proposal,
tiled proposal, fixed revisit and joint proposal. Its accepted-model rank check
still precedes candidate optimization; catalogue selection is not permission
to skip that requirement. Clearances and unknown costs can reject any entry.

D0–D3 are a separate factorial panel. D0 uses published-like acquisition;
D1 must come from the published-setting C1 frontier. D2 combines the selected
geometry with D0's scan/conventional inverse. D3 combines that **same geometry**
with D1's published acquisition recipe and separately trained optimized inverse.
K-J's novel scan is therefore an exploratory C3 proposal, not automatically a
D3 representative. Its geometry can be tested with the published D1 recipe in
the existing follow-ups; a novel-scan D1/D3 would require a separate amendment.

The starting choices do not guarantee feasibility, positive gain, a nominated
device or success of the separable rule. They make those hypotheses concrete
without changing endpoints, numerical margins, final access or total resources.

## Next scientific choice

The [electrical mechanisms/probes](electrical_model_and_probes.md) now select
six initial contact/state templates and common finite protocols for noise,
1–100-Hz impedance, settling and crosstalk. They do not provide accepted models
or identified constants. The [compact component choices](component_details.md)
now add source-to-surface sensitivities and inverse starting allowances. Next
resolve measurement contexts for the [selected anchor roles](../DATASETS.md)
and instantiate the electrical note's minimal shared fit. Inverse interfaces
now specify whole-world splits and bounded-state, target-relative causal history.
Obtain a layout rule or an explicitly approved
substitute for the pad/via exclusions, and resolve ADC internal-work accounting. These are
the substantive unknowns that could change which entries are usable; no
additional provenance, receipt or schema work is proposed.
