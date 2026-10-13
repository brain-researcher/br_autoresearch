# EP19 source and timing handoff

## October 9: downstream wrist-position unit clue

The provider's pinned
[WEEG_FindEvents.m, lines 440–450](https://github.com/luciw/way-eeg-gal-utilities/blob/c81320ab5da20de83b09ec62167200abdcf8c88c/Utilities/WEEG_FindEvents.m#L440-L450)
differentiates supplied wrist XYZ, forms speed and describes its threshold of
5 as “5 cm/s.” The
[derivative helper, lines 500–505](https://github.com/luciw/way-eeg-gal-utilities/blob/c81320ab5da20de83b09ec62167200abdcf8c88c/Utilities/WEEG_FindEvents.m#L500-L505)
uses a 0.002-second interval. **Inference:** this downstream WS wrist utility
expects position values in centimetres. This is more specific than saying
there is no unit evidence; it does not establish the HS export scale,
other sensors' conversions, calibration, or force units. No 0.01 conversion
or retrospective onset algorithm is adopted.

The concrete acquisition question remains how HS_P1_S1 and WS_P1_S1 were
exported: the synchronization/trim counts, the assignment of trial_start_time
relative to the first retained HS sample, and the recorded position/force
scale and calibration. Reusing the same 34 LED events cannot distinguish
the existing origin hypotheses. This documentary clue does not resolve the
two-second discrepancy or the complete onset uncertainty bound.

## Development acquisition settings omitted by the earlier header report

The [completed header reader](code/inspect_development_headers.py) retained Common, Channel and Binary Infos but did
not return the `[Comment]` amplifier setup and software-filter tables. Bounded
text inspection of two already materialized development headers identifies a
useful remaining source: sub-02/session-01/run-0001 records DC/69-Hz cutoffs at
250 Hz, whereas sub-15/session-01/run-0001 records DC/280-Hz cutoffs at 1,000 Hz.
Both inspected amplifier tables bind TRIG/X/Y/Z to physical channels
65/66/67/68, with 0.298023 microvolt resolution. These two examples do not
establish uniform settings across the development recordings.

The [focused reader](code/inspect_development_acquisition.py) extracts these
previously omitted settings from exactly the original 155 development `.vhdr`
paths, using the [completed header report](runs/source_views_and_timing/job_46425841_headers/development_headers.json)
and [materialized-view report](runs/source_views_and_timing/job_46425841_development_view/development_view.json). It
preserves all fifteen development people, leaves the prior report unchanged,
and neither opens archives nor follows DataFile/MarkerFile references. Channel
tables must match their Channel Infos identities and scales; missing or
unsupported acquisition fields remain explicitly unresolved. The four
[text/path fixtures](tests/test_development_acquisition.py) and the extraction
are prepared for the next shared substantive allocation, not yet executed.

The output records stated configuration, not a filter transfer function,
group delay, calibrated acceleration or a certified onset uncertainty bound.
The vendor's [manual index](https://www.brainproducts.com/downloads/manuals/)
requires an account, and the [actiCHamp manual landing page](https://www.brainproducts.com/downloads/manuals/actichamp-actichamp-plus-operating-instructions/)
did not expose version-specific manual text in the coordinator's October 7 UTC
check. Neither a 69/280-Hz cutoff nor current software documentation supplies
the study's missing transfer/delay evidence. No conversion or delay is inferred.

## October 6, 15:49 Pacific: published event generator and remaining exporter gap

Primary source code and repository metadata checked on October 6, 2026
(Pacific) resolve several documentary details. In the provider's pinned
[`WEEG_FindEvents.m`](https://github.com/luciw/way-eeg-gal-utilities/blob/c81320ab5da20de83b09ec62167200abdcf8c88c/Utilities/WEEG_FindEvents.m#L109),
lines 109–114 copy `ws.win(L).trial_start_time` directly into `StartTime` and
copy the stored `LEDon`/`LEDoff` fields. Line 149 constructs the window sensor
time axis as `0.002*(1:numel(GF_Index))`, whose first value is **2 ms**, not
zero. These are downstream event-generation conventions; they do not show
how the earlier raw-to-HS/WS exporter assigned the stored window start after
synchronizing and trimming the continuous streams. The 2-ms sample convention
does not explain or justify correcting the observed **2-second** discrepancy.

The same code, lines 116–148, authenticates the named channel conventions:
grip force uses the negative of each plate's stored FZ channel, load force
uses positive FX, sensor 1 supplies object position, and sensor 4 supplies
wrist position. This identifies code-level signs and channels, not an
authenticated conversion of stored positions/forces to metres/newtons or a
new past-only onset detector. The original [article's sensor description and
Figure 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC4365902/) independently name
the object/index/thumb/wrist sensors and force axes. Its reported calibration
and positional accuracy are general acquisition evidence, not the missing
export scale or calibration coefficients.

The provider repository's [branch listing](https://api.github.com/repos/luciw/way-eeg-gal-utilities/branches?per_page=30)
returned only `master` at the pinned 2014 commit, and its [all-status issue
listing](https://api.github.com/repos/luciw/way-eeg-gal-utilities/issues?state=all&per_page=20)
returned no issues. Those inspected routes supplied no correction or alternate
exporter; this is not proof that no other original acquisition record exists.
The concrete missing evidence remains the raw-to-HS/WS start-field/trim
assignment, authenticated physical-unit conversion, and the complete onset
uncertainty chain. No timestamp shift, calibration value, threshold, empirical
access or forecast is adopted by this documentary update.

## October 6, 11:15 Pacific: official importer and series-origin construction

The additional source investigation examined the provider's downstream
EEGLAB importer and the article's exact timestamp definitions, without reading
any HS/WS array or event record. The public utility repository currently points
to commit `c81320ab5da20de83b09ec62167200abdcf8c88c` (October 1, 2014).

The article's **Data processing** section says that SC/ZOOM and BCI2000 streams
were aligned by sync-channel cross-correlation and that extra samples were
removed at the beginning and end of each series. Its **Data Records** section
defines `trial_start_time` as the absolute start corresponding to `StartTime`;
Table 1 gives `StartTime` relative to the series start, while other event times
are relative to the window and `LEDOn` is always two seconds. It does not give
the per-series trim counts or the export assignments that reconcile these
origins. See the [provider article XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4365902/fullTextXML),
sections `sec9`, `sec14`, `sec15` and Table `t1`.

This makes an export/trim-origin difference a concrete alternative to the
previous padding hypothesis; neither cause is established. The missing
construction is specifically how the exporter assigns `trial_start_time` and
the HS first sample after synchronization/trimming. Repeating the same P1
residual comparison cannot identify that assignment or distinguish both
causes. The documented `eeg_t`/`emg_t` fields give per-row times but their field
descriptions alone do not authenticate their origin relative to the trimmed
HS stream. No access to those vectors is selected here.

In the official [EEGLAB importer](https://github.com/luciw/way-eeg-gal-utilities/blob/c81320ab5da20de83b09ec62167200abdcf8c88c/Utilities/WEEG_MakeEEGLABDataset.m),
lines 51–53 call `WEEG_GetEventsInHS`; the event loop passes its returned times
to `pop_importevent` without a two-second adjustment. The provider's
[conversion function, lines 94–103](https://github.com/luciw/way-eeg-gal-utilities/blob/c81320ab5da20de83b09ec62167200abdcf8c88c/Utilities/WEEG_GetEventsInHS.m#L94-L103)
still adds `StartTime` to each relative event. A commented subtraction on the
liftoff line is inactive and is not a correction instruction. The inspected
[Utilities directory listing](https://api.github.com/repos/luciw/way-eeg-gal-utilities/contents/Utilities?ref=c81320ab5da20de83b09ec62167200abdcf8c88c)
contains event derivation, plotting and EEGLAB consumers, but no separately
named raw-to-HS/WS export constructor. This is a limit of this repository
inspection, not proof that the constructor is unavailable elsewhere.

The importer also scales EEG by 0.1 before import; that code operation does
not define physical position/force conversions or their calibration. No new
source resolves those units, an empirical timestamp correction or the full
strictly-below-300-ms onset bound. The next useful documentary evidence is the
actual raw-to-HS/WS export assignment or an authenticated statement connecting
the trimmed series zero, window zero and stored start field. Forecasting,
sensor fitting, clock shifts and further source-array checks remain unselected.

## October 6: timing, units and decision review — recommendations unadopted

The supplied review identifies useful distinctions, but it does not establish
that timing is no longer a blocker. The cue/risk state and parameterized
[completed-return predicate](component_details.md#11-way-completed-return-predicate-parameterized-development-definition)
already exist. Their required physical conversions, cue availability,
return tolerances and complete onset uncertainty remain unresolved. The
primary **300–600-ms horizon remains unchanged**.

### Clock interpretation and proposed next evidence

The completed P1 comparison found all 34 LED upcrossings and all 34
downcrossings exactly two seconds earlier, within floating-point precision,
than the attempted `trial_start_time + LEDon/LEDoff` convention. Matching
interval durations support higher=illuminated for this file. A common
clock-origin or padding interpretation is consistent with that pattern;
double-counting of pre-cue padding is a hypothesis, not an established cause.
The [preserved comparison](experiment_log.md#2026-10-03-2135-utc--source_views_and_timing-cue-comparison-completed-absolute-origin-unresolved)
did not authenticate absolute clocks or apply a correction.

Recommend preparing one bounded development-only clock test **after** an
explicit candidate origin binding identifies what zero means in each clock,
which source construction supports it, and which independent cue/sync
evidence can distinguish it from alternatives. Declare the sign and exact
transformation before reading further values; preserve original timestamps
and report disagreement rather than fit a shift to make residuals vanish.
A candidate cannot be adopted merely because subtracting two seconds aligns
the same 34 events used to suggest it. An empirical comparison could resolve
the origin without a provider reply if adequate existing source evidence
establishes the binding; it cannot replace missing acquisition facts.

Even a correct constant-origin transformation does not bound clock drift,
sensor acquisition and synchronization, transmission, causal filtering,
resampling, sample discretization or threshold-crossing uncertainty. The
existing worst-case bound must still be strictly below 300 ms, with
confirmation delay recorded separately. Do not shorten the forecast horizon,
move a detected movement backward, or open held-out data to repair this chain.
This recommendation specifies the prerequisite for a later test, not a new
test, source read or timing correction selected here.

### Units and conditional support

A documented conversion from authenticated stored units to metres/newtons is
ordinary source integration. It needs the experiment's unit, calibration,
axis/sign, bias/gain and export information where applicable; plausible signal
magnitudes or manufacturer defaults cannot supply those facts. The existing
15-cm clearance predicate cannot be evaluated in unknown raw units. An
alternative normalized-sensor predicate would be a concrete methodological
amendment requiring its own definition and claim limits; the review's
unspecified "small units amendment" does not activate one.

The completed original S1 panel left P1, P3, P6, P9 and P12 unresolved, with
stationary counts 13/34, 8/34, 16/34, 11/34 and 10/34 respectively; P3 also had
only three fitting-parity windows. Seven participants met the unchanged
conditional support rule. These are S1 support dispositions, not permanent
whole-person exclusions or a dataset-wide scientific failure. Keep all twelve
participants in the record. No favorable subset, replacement session,
threshold relaxation or production detector fit is selected by this review.

### The 19 unset decision fields and the main scientific judgment

The current policy's `margins_required_before_candidate_EEG_scores` block has
**19 null fields**, plus the already fixed probability-sum tolerance of
`1e-6`. Calling all nineteen "effect margins" conflates scientific effect
criteria with numerical, timing, latency and participant rules:

| Kind | Exact unset fields |
| --- | --- |
| Six EEG increments | `real_minus_non_neural_primary_minimum`, `real_minus_zero_EEG_primary_minimum`, `real_minus_surrogate_primary_minimum`, `real_minus_non_neural_near_minimum`, `real_minus_zero_EEG_near_minimum`, `real_minus_surrogate_near_minimum` |
| Noninferiority and readiness | `zero_EEG_noninferiority_margin`, `near_event_readiness_minimum`, `longer_horizon_non_degradation_margin` |
| Model ordering and new-model gain | `model_pair_margin_near`, `model_pair_margin_primary`, `development_new_model_improvement_margin`, `final_new_model_improvement_margin` |
| Factorial equivalence | `factorized_full_reference_equivalence_margin`, `factorial_interaction_equivalence_margin` |
| Numerical tolerance | `block_mutation_numerical_tolerance` |
| Participant rule | `participant_robustness_rule` |
| Timing and deployment | `synchronization_uncertainty_max_ms`, `p95_inference_latency_ms_maximum` |

Additional unresolved settings, such as the non-neural probability floor and
alarm false-positive ceiling, lie outside that nineteen-field block. The list
is not a claim that only nineteen settings remain in the empirical pipeline.

Recommend prioritizing one user-facing scientific judgment: the smallest
meaningful **real-EEG minus non-neural** gain on the primary natural-prevalence
binary log score. Any concrete proposed value should state the logarithm
convention, decision-point prevalence, equal-participant aggregation and
practical interpretation before selection. No value is proposed as proven by
the current support counts. The coordinator can prepare the remaining
contract-preserving choices before scores under existing delegated authority;
other effect/equivalence margins still need scientific justification, whereas
floating-point tolerances need numerical justification. A numerical fixture
cannot select or relax an effect threshold, and a clock tolerance cannot
stand in for missing source timing evidence.

No threshold, role, model recipe, resource limit or source access changes here.
The implemented backends and corrected participant routing remain preparation;
raw feature extraction, final refits and the rest of the empirical pipeline
are unfinished. Source admission and forecasting authorization stay false.

## Preserved source investigation

October 3, 2026, `source_views_and_timing`. Acquisition already exists; no source
was downloaded again. Current preparation has read central-directory metadata,
development BrainVision text headers and three source documents. Raw
development extraction completed in job `46425841`; it read development
payload bytes for stream-copying without displaying signal values. One fitting
WAY file was subsequently deserialized for names/rates/shapes only. No EEG
forecast, held-out signal, derived event value or candidate score was opened.

## Actual development structure

The inventory contains 1,265 archive members. Of the WAY members, 144 belong to
series 1–6, 24 to series 7, 48 to series 8–9 and 24 are unresolved/mixed files. The
first three counts each include HS and WS, so they are not counts of continuous
recordings. Self-paced has 650 development, 359 held-out and 16 dataset-level
members. The raw development subset is 155 BrainVision triplets.

All 155 development headers cover the 15 exact policy participants and include
the complete 29-channel scientific intersection. There are 144 headers at
250 Hz with 39 channels, and 11 at 1,000 Hz with 67 channels. The extra numbered channels do not
change the scientific montage. The recorded format is multiplexed float32;
TRIG and wrist X/Y/Z are explicitly named. Auxiliary units are amplifier
microvolts, so physical acceleration units and sensitivity cannot be assumed.
All this is header evidence, not a measurement of signal clocks, sample values,
gaps or valid movement coverage.

Actual records:

- `runs/source_views_and_timing/job_46425841_inventory/source_inventory.json`
- `runs/source_views_and_timing/job_46425841_headers/development_headers.json`
- `runs/source_views_and_timing/job_46425841_headers/documents/`

## Source semantics that constrain the next step

The official WAY article identifies HS as continuous recordings; WS contains
windows ending after the future LED-off cue. Only HS is eligible for the
prospective raw view. HS_ST lacks the peripheral modalities; AllLifts combines
derived event information and remains excluded. The public HS structure has
`eeg`, `emg`, `kin`, `env`, and `misc`, each with `names`, `samplingrate`, and
`sig`. EEG acquisition used hardware/software anti-alias filtering; EMG had a
global mean removed. Consequently, literal use of released EMG magnitude can
retain an offline offset. A future stationary-baseline-centered path should
demonstrate invariance to that constant rather than assuming a raw baseline.
Published synchronization error alone does not bound the full EP19 chain.
[WAY source methods](https://www.nature.com/articles/sdata201447)

Self-paced documentation identifies TRIG as recorded audio routed through the
amplifier. Its event dictionary describes post-hoc onset extraction and
ultimately selected cup labels. Neither completed target identity nor
future-corrected event timing may become a current predictor. The task script
specifies 12-second start-to-end periods followed by 7-second breaks; its nominal
schedule does not prove recorded cue availability or clock accuracy. Event
TSVs and derivative MAT files are therefore excluded from the raw view.

## Completed development view

`development_member_plan.json` lists exactly 84 WAY continuous HS members
(72 for fitting and 12 for series 7 feedback) plus 465 self-paced raw members,
549 in total and 9,038,196,400 uncompressed bytes. The builder verifies roles,
archive identities and all selected member sizes before any decompression.
It keeps WAY feedback separate, copies only the 15 development people and
leaves every held-out member and source archive unchanged. A fresh scratch
projection is regenerable; the durable acquisition remains its source.
Read-only mode is not represented as a separate evaluator identity.

The selected extraction completed on October 3 at 19:17:32 UTC in job
`46425841`, in 538.5855 seconds, with all 549 declared members and exactly
9,038,196,400 bytes. Its report and selected member declaration are preserved
under `runs/source_views_and_timing/job_46425841_development_view/`. The raw
view remains in episode scratch and can be regenerated from the unchanged
durable archives. No held-out, provider-windowed or derived member was copied.
The separately selected structural inspection completed on `HS_P1_S1.mat`
only, at 19:23:26 UTC in the same job. The actual nested identity keys are
`participant` and `series`; the first failed lookup was preserved. This file
has 119,496 EEG samples by 32 channels at 500 Hz and 955,969 EMG samples by
5 channels at 4,000 Hz. Kinematic, environment and miscellaneous streams are
at 500 Hz with 36, 2 and 6 channels. All 29 scientific EEG channels are
present. Its report is
`runs/source_views_and_timing/job_46425841_way_structure/way_development_structure.json`.
This single structural example does not qualify every development recording,
show clock alignment, or bound acquisition/filter/onset uncertainty.

The next timing stage can inspect development-only stream dimensions,
sampling rates, named sensor channels, sample counts, acquisition gaps and raw
cue availability. It must resolve physical accelerometer interpretation,
WAY clock reconciliation/acquisition response and source-specific valid-run
eligibility before freezing detector thresholds. No current report certifies
the worst-case timing bound strictly below 300 ms.

## Implemented pieces and remaining science

`forecast_components.py` implements finite past-only FIR state with explicit
warm-up/reset, the exact complementary/recent/older-slow reconstruction,
19-category censoring and the existing provisional four-parameter calibration.
Eight focused tests pass under the existing Python 3.12/NumPy 1.26.3/
sklearn1.5.1 module stack. They verify equations and slot mutations; they do
not qualify any reference model, actual 32-event packet or onset detector.

Numeric margins, detector thresholds/coefficient choices, valid-run rules,
the finite model grammar, reference recipes and participant folds must still
be fixed through the declared development-only process. Do not activate a
margin merely because a synthetic fixture passes. No new scientific threshold
or alternate source is introduced here. The primary claim remains prospective
information before the registered sensors detect movement, with the weaker
wrist-accelerometry boundary in the second task.

## October 5, 2026: source mapping clarification

The published utility chain matches the local P1 comparison: [WEEG_FindEvents.m,
lines 109–114](https://github.com/luciw/way-eeg-gal-utilities/blob/master/Utilities/WEEG_FindEvents.m#L109-L114)
copies `trial_start_time` to `StartTime` and preserves `LEDon`/`LEDoff`;
[WEEG_GetEventsInHS.m, lines 94–102](https://github.com/luciw/way-eeg-gal-utilities/blob/master/Utilities/WEEG_GetEventsInHS.m#L94-L102)
then computes `HS_event_time = StartTime + LED_relative_time`. The local scalar
reader preserves those values, and `check_way_cue_polarity.py` uses the same sum.
The saved [P1 cue report](runs/source_views_and_timing/20261003-cue-polarity01/cue_polarity_report.json)
records all 34 onsets and 34 offsets with
`observed_crossing_time - published_reference_time = -2 s`, within floating-point
roundoff. Because every relative `LEDon = 2 s`, the saved summary implies
`raw_onset = trial_start_time` and
`raw_offset = trial_start_time + LEDoff - 2 s`. This is a P1 source/reference
inconsistency, not a demonstrated local arithmetic bug. No automatic two-second
correction is adopted; the export-specific origin remains unresolved. The
provider's plotting example starts the first HS sample at 0.002 s, versus zero
locally; that separate 2 ms convention cannot explain the two-second discrepancy.

[WEEG_FindEvents.m, lines 218–223](https://github.com/luciw/way-eeg-gal-utilities/blob/master/Utilities/WEEG_FindEvents.m#L218-L223)
negates the raw plate FZ channels before averaging grip force:
`GF = -(FZ1 + FZ2) / 2`. This establishes the utility's raw-channel sign
convention, not physical scales or offset calibration. Self-paced raw auxiliary
channels remain in amplifier microvolts; accelerometer sensitivity, physical
units, axis signs and acquisition/filter delay remain unresolved.

The WAY methods' reported synchronization error of at most 2 ms is a
document-level inter-stream alignment statement. It does not bound acquisition
and filtering delays, detector uncertainty, or the full prospective onset chain,
and does not certify EP19's required worst-case bound below 300 ms.

## October 6, 2026: publication supplement page 1 reviewed

The [official article XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4365902/fullTextXML)
identifies the 193,612-byte publication supplement
[`sdata201447-s1.pdf`](https://pmc-oa-opendata.s3.amazonaws.com/PMC4365902.1/sdata201447-s1.pdf).
The completed page-1 extraction and PDF metadata are preserved at
[`source_documents/way_information/20261005-page01/`](source_documents/way_information/20261005-page01/).
The verified source filename is `sdata201447-s1.pdf`; the local preserved name
`Information.pdf` does not establish identity with a separately named
data-release document. The earlier handoff's "Information.pdf supplement"
wording referred to this publication supplement, not a separately verified file.

The PDF has seven pages. Its [first page](source_documents/way_information/20261005-page01/page01.txt)
contains only a contents list: consent begins on page 2, equipment supplier
links on page 4, participant series schedules on page 5, participant information
on page 6, and an instruction checklist on page 7. Only page 1 was inspected
at this stage; the later bounded follow-up is recorded below. Page 1 provides
no sensor conversion, calibration, timestamp-origin, filter-delay or cue-polarity
specification; it does not establish that these
are absent from all provider documentation.

Consequently the P1 source/reference discrepancy remains unresolved, with no
automatic two-second correction. WAY physical position/force scales and
offsets, export-specific clock origins and acquisition/filter delay remain
unverified. The established utility FZ sign convention is not a calibration.
Self-paced accelerometer sensitivity, physical units/axis signs and timing
response likewise remain unresolved. Completed-return tolerances, other-file
cue polarity and the full strictly-below-300-ms onset bound are not established
by this contents page. No detector setting, threshold or data role changes.

## October 6, 2026: equipment and instruction documentation

The coordinator subsequently inspected only pages 4 and 7 of the same
[publication supplement](https://pmc-oa-opendata.s3.amazonaws.com/PMC4365902.1/sdata201447-s1.pdf).
Page 4 identifies Polhemus FASTRAK, an ATI six-axis F/T DAQ manual, actiCAP,
BrainAmp Standard and BrainVision Recorder. Page 7 instructs participants to
reach on red-light onset, lift about 5 cm using thumb and index finger, hold
until the light goes off, then replace the object and rest the hand on the blue
surface. This resolves the documented instruction direction, not stored LED
edge polarity, cue latency or completed-return detection. Supplement pages
2, 3, 5 and 6 remain unread.

The [FASTRAK manual, Rev. G, June 2012](https://polhemus.com/_assets/img/FASTRAK_User_Manual_OPM00PI002-G.pdf)
documents selectable inches/centimeters (glossary C-7; `u` selects centimeters,
printed p. 73), configurable adaptive position filtering (`x`, p. 81), and
filter-dependent response lag (Appendix E-3). Appendix E-1 defines tracker
latency to stop when a solution is ready for output, excluding transmission
and host delays; its externally synchronized example gives 5.5 ms
sync-to-output, while asynchronous operation lacks that precise timing
definition. These are device definitions, not the experiment's settings or an
onset bound. The provider-indexed manual sections were readable; opening the
old PDF URL redirected to the manufacturer homepage.

The supplement's [ATI manual compilation](http://www.ati-ia.com/app_content/documents/9610-05-1017%20DAQ.pdf)
identifies component manual
[`9620-05-DAQ.pdf`](https://www.ati-ia.com/app_content/documents/9620-05-DAQ.pdf).
Provider-indexed revision `9620-05-DAQ-25`, not authenticated as the study-era
revision, states that gauge voltages require bias subtraction and a
sensor-serial-specific 6-by-6 calibration matrix; individual gauges are not
force/torque axes (sections 3.3.1–3.3.3, p. 19). Additional amplifier gain needs
matrix correction (3.3.4, p. 20), calibrated voltage range depends on the
calibration (4.3.2.1, p. 31), and electronics bandwidth varies with gain
(5–10 kHz; 5.2–5.3, p. 37). Bandwidth is not a measured end-to-end delay.
Direct PDF retrieval exceeded the web tool's size limit; no local download or
processing of these equipment manuals was performed.

The concrete source inputs still needed are the experiment's FASTRAK unit,
coordinate-frame, sensor-count, filter, sync/interface and export/resampling
settings; ATI sensor/calibration identity and force units, bias/gain settings,
stored-channel conversion and DAQ/filter timing; and the clock-origin mapping
needed to reconcile P1. Manufacturer defaults do not supply these facts.
Self-paced accelerometer calibration and the complete onset uncertainty bound
remain unresolved. No offset correction, detector threshold, data-role change
or forecasting step is selected.

## October 6, 2026: all seven publication-supplement pages inspected

At 00:37:50 UTC, the bounded documentary review is complete: pages 2, 3, 5 and
6 of the existing [seven-page PDF](source_documents/way_information/20261005-page01/Information.pdf)
were inspected with exact-page `pdftotext` calls to stdout, supplementing the
previous reads of pages 1, 4 and 7. No pages of this particular publication
supplement remain uninspected. Its authenticated source filename remains
`sdata201447-s1.pdf`; storing it locally as `Information.pdf` does not authenticate
its identity with any separately named dataset-release `Information.pdf`.
No additional download, rendering, signal access or scientific computation was
performed.

Page 2 is participant information/consent. It describes lifting about 5 cm with
the right thumb and index finger, position sensors on the hand, index finger
and thumb, surface EMG and an EEG cap. These are placement/task descriptions,
not a complete tracker sensor-count, unit, coordinate-frame or calibration
record. Page 3 is a blank consent form and supplies no acquisition settings.

Page 5 explicitly separates experimental chronology from representation in
the released data structures. Series S1–S7 retain the experimental ordering.
For the remaining conditions, the mapping is:

| Participants | Experimental series 8 / 9 / 10 | Released S8 / S9 / ST (S10) |
| --- | --- | --- |
| P1, P3, P6, P9, P12 | Weight 4 / Weight 5 / Weight 6 | Weight 4 / Weight 5 / Weight 6 |
| P2, P5, P8, P11 | Weight 6 / Weight 4 / Weight 5 | Weight 4 / Weight 5 / Weight 6 |
| P4, P7, P10 | Weight 5 / Weight 6 / Weight 4 | Weight 5 / Weight 4 / Weight 6 |

Only EEG is available on Figshare for Weight 6, represented by
`HS_P1_ST.mat` through `HS_P12_ST.mat`. Thus ST identifies that condition even
when it occurred before the final experimental series. This documentary
file/condition mapping does not establish clock origins, LED edge polarity or
cue latency. It changes neither the frozen S1–S7 development/S8–S9 held-out
roles nor the exclusion of EEG-only ST files from the multimodal raw view.

Page 6 records study-specific quality observations:

- P3: weak EMG-3 (Flexor Digitorum).
- P5: weak EMG-3 and EMG-5 (First Dorsal Interosseus).
- P6: early EEG setup problems and an hour's delay involving electrodes,
  batteries and gel; Polhemus noise on EMG-5.
- P8: unexpected breaks in EMG-1 (Anterior Deltoid) and EMG-5.
- P9: unexpected breaks in EMG-2 (Brachioradialis), with occasional left-hand
  and foot movements also noted.
- P10: rare large EMG-4 (Common Extensor Digitorum) spikes and short EMG-1
  flatlines.
- P12: rare large EMG-4 spikes and brief EMG-3 noise.

P1 is described as not naive to the purpose, P2 as reacting late to light events
and moving slowly, and P4 as receiving more EEG gel before the second series.
The sheet specifies no affected time intervals, numeric validity thresholds or
delay bounds. In particular, behavioral reaction time and the setup delay are
not acquisition-latency calibrations. These notes do not select exclusions,
alter eligibility or certify usable peripheral coverage.

None of these four pages resolves physical position/force units and offsets,
FASTRAK frame/filter/sync/export settings, ATI calibration/bias/gain and DAQ
timing, stored cue-edge semantics, P1's two-second source/reference discrepancy,
completed-return tolerances, self-paced accelerometer calibration, or the full
strictly-below-300-ms onset bound. Those blockers and the existing no-automatic-
correction/no-new-threshold boundary remain unchanged.

## October 6: finite modeling and fold preparation

The coordinator separately authorized completion of the internal model/fold
definition. `component_details.md`, section 5, now specifies the finite
reference and non-neural recipes, the shared model interface, one-operator
secondary menu, and exact whole-participant/whole-series folds. The selected
prospective registry is `model_recipe_registry.json`; `code/model_recipes.py`
enumerates it without opening source files, fitting a backend or forecasting.
All 15 self-paced development people and all 12 WAY people remain represented;
the unresolved support participants are not excluded or replaced. No unchanged
source-support check is repeated.

Each reference family has exactly 12 configurations and three seeds. Final
primary/factorial enumeration requires an explicit development-frozen selection
of one recipe per family; it does not launch another configuration search.
All eight final surrogates and all four factorial conditions remain required.
All3 [focused enumeration/fold checks](runs/model_recipe_checks001/tests.txt)
passed in0.003s, following independent static review. Official ASTRA schema and
semantic validation also passed in46721202. The parent completed0:0 and is
charged once to EP06; see the single execution-log entry for terminal
accounting. These are implementation checks, not scientific simulation work.

This closes the *finite recipe and fold specification*, not the executable
forecasting pipeline. Numerical family backends/adapters, causal filter and
resampling coefficients, surrogate construction and scientific margins remain
internal work. Actual source timing, units/calibration, P1 clock reconciliation,
return/onset qualification and valid-run information remain explicit required
inputs, with no guessed defaults or new access permission. The seven-page
documentary review and its unresolved acquisition facts above are unchanged.

## October 7: interpreting the observed actiCHamp acquisition settings

The completed [155-header extraction](runs/source_views_and_timing/job_46877919_acquisition_metadata/acquisition_metadata.json)
supplies cohort-wide recording settings. Brain Products documents automatic
anti-aliasing before storage, with cutoff information in the header; optional
Recorder filters are a separate stage. Thus the observed `Software Filters:
Disabled` is not evidence of an unfiltered acquisition. The published
second-order Butterworth description must not be assigned to the automatic
anti-alias chain without its specific implementation binding.
[Manufacturer signal-path explanation](https://pressrelease.brainproducts.com/amplifier-signal-pipeline/).

The [current amplifier specification](https://www.brainproducts.com/solutions/actichamp/)
lists approximately 0.596 microvolts/bit for AUX revision 02, whereas these
headers record 0.298023 microvolts. That difference is a reason to retain the
actual header scale, not replace it with the current product value. Neither
number converts voltage into acceleration without the sensor calibration.

No sensor conversion, filter coefficient or delay bound is adopted. The
[vendor operating-manual link](https://www.brainproducts.com/download/actichamp-actichamp-plus-operating-instructions/)
redirected to login; no authenticated retrieval was attempted. Exact historical
acquisition/filter and sensor calibration bindings remain the required inputs.

## October 11: development stream dimensions and unresolved file references

The [completed stream-layout report](runs/source_views_and_timing/job_47355564_stream_layout_retry02/development_stream_layout.json)
retains all155 development runs. Binary sizes support nominal sample counts
for152 runs; three header-reference conflicts remain unresolved without opening
their binary or marker files. Those152 marker files each declare one starting
NewSegment at sample1 with a timezone-unspecified timestamp. Six marker
`DataFile` bindings are not verified, so their segment metadata is not accepted
as aligned to the corresponding EEG. The [execution record](experiment_log.md#2026-10-11-0136-pacific--source_views_and_timing--stream-metadata-complete-with-nine-unresolved-bindings)
contains checks and accounting; no signal array or stimulus label was read.

| Development participant | Session | Run(s) | Mapping still needed |
| --- | --- | --- | --- |
| sub-03 | 02 | 0009,0010,0011 | Authenticate marker `DataFile` binding to the released EEG |
| sub-09 | 02 | 0009 | Authenticate the old marker filename referenced by the header |
| sub-15 | 02 | 0016 | Authenticate marker `DataFile` binding to the released EEG |
| sub-18 | 02 | 0006 | Authenticate the old EEG filename referenced by the header |
| sub-20 | 01 | 0003 | Authenticate marker `DataFile` binding to the released EEG |
| sub-21 | 01 | 0003 | Authenticate the old marker filename referenced by the header |
| sub-21 | 02 | 0009 | Authenticate marker `DataFile` binding to the released EEG |

For the six marker bindings, the present report establishes failure of the
exact reference check; it does not yet distinguish a missing, duplicate or
differently named entry. A same-stem filename alone is not an adopted repair.
These nine runs remain in the declared cohort and are not scientific
exclusions. This issue is separate from the provider's238-valid-versus240-raw
run discrepancy. At this earlier stage, a corrected provider mapping or
authenticated export rule was still needed before using affected aligned signals.

**October11 documentary mapping decision:** the publisher's
[Data Records section](https://www.nature.com/articles/s41597-025-06039-9)
explicitly associates EEG, header and marker files by participant/session/run
stem. With the exact unique frozen release triplets, this supplies a source
rule for metadata association. Select
`publisher_same_entity_filename_contract` explicitly, keep every original
reference and verification flag, and retain any known competing reference to
another extant identified run as unresolved. A fresh bounded metadata run is
selected but has not yet executed. This supersedes the earlier requirement to
obtain provider clarification solely for legacy names; it does not establish
physiological trace identity, sample synchronization or calibration.

**Completed mapping pass:** the [new report](runs/source_views_and_timing/column_stripe_job_47357234_metadata/development_stream_layout.json)
inspected all155 triplets and accepted152 documentary file associations.
The three earlier header-reference skips and three legacy marker names have
a usable documentary mapping. Sub-03/session02/runs0009,0010,0011 each name
the existing run0008 EEG in their marker DataFile field; these three conflicting
associations remain unresolved. All original nine discrepancies remain in
the report, distinct from documentary acceptance. This pass used no signal
values or trial labels and does not qualify physiological alignment.

The four already captured run0008–0011 records have different binary frame
counts and start timestamps. This excludes identical whole EEG files or
identical retained NewSegment records, but does not establish how the repeated
run0008 DataFile text arose. Retain same-file layout facts and withhold the
three absolute marker-clock associations. Seek the targeted source mapping
correction if those starts are needed; no repeat extraction or blanket metadata
stop follows. Physical calibration and full timing qualification remain open.

The152 nominal stored durations total87,058.74seconds, about24.18hours. This
does not measure gap-free acquisition or valid forecast exposure. Physical
accelerometer calibration/axis signs, historical acquisition/filter response,
full onset uncertainty and WAY clock binding remain unresolved. A single
recorded starting marker cannot bound hidden gaps or establish the required
timing uncertainty strictly below300ms.
