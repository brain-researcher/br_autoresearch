# Data used in Episode 16

The October 6 scientist-adopted [local closeout](outputs/experiment_log.md#2026-10-06--structural_support--scientist-adopted-support-limited-closeout)
records **support_or_uncertainty_prevents_an_answer** for the original
three-held-out-participant claim. Existing structural evidence and exposure
boundaries remain; no neural analysis, role replacement or case-series
successor is adopted.

EP16 uses one published BrainGate release to compare how an offline neural
mapping transfers between sessions from the same participant. This file states
what is available, how participants are separated between development and the
final held-out test, and which limitations determine the scientific claim.

## Public online resources — checked October 6, 2026

| Public source | Available information and EP16 use | Remaining limit |
| --- | --- | --- |
| [Dryad release and embedded README](https://datadryad.org/dataset/doi:10.5061/dryad.x0k6djj1h) | Download links for the nine decoding archives and yield archives, plus the field codebook: 10-ms bins, task-coordinate x/y positions, zero-based starts/exclusive ends, electrode/array identifiers, impedance in kΩ, `sbp` in µV² and threshold crossings per bin. Use these definitions to implement source readers without guessing units or indexing. | Task families are named, but the documented fields do not supply per-session target size, controller versions or intended-transition rules. Yield arrays need not be synchronized. |
| [Pinned provider participant/array metadata](https://raw.githubusercontent.com/nptl-stanford/array-paper/93beb76b4329f0e3d5b3fd5ffba9be1c7cfa2834/array_paper/metadata.py) | Explicit decoding-participant order, array names and provider display conventions. Use for identity interpretation and cross-checking array labels. | It is not a session-level task/size map or a record of analyst exposure. |
| [Pinned provider decoding-method implementation](https://raw.githubusercontent.com/nptl-stanford/array-paper/93beb76b4329f0e3d5b3fd5ffba9be1c7cfa2834/array_paper/metrics/decoding_snr.py) | Public code for the provider's movement-direction/decoding calculation. It is a methods reference for interpreting the published features and task intervals. | The provider's analysis is not EP16's cross-session endpoint; no published scores or data summaries were imported, and its method cannot qualify EP16's missing schedule evidence. |

The online codebook resolves source semantics directly. Authentic session
annotations and local/publication exposure history remain the targeted gaps;
their absence from these inspected documents is not proof that no original
record exists. The fixed release and structural-only access remain unchanged.

## Source

| Item | Value |
| --- | --- |
| Dataset | Performance of intracortical microelectrode arrays in people with implanted brain-computer interfaces over a 20-year period |
| Provider | Dryad |
| DOI | `10.5061/dryad.x0k6djj1h` |
| Published release | Version 6, published 2026-09-09 |
| Provider version ID | `462702` |
| Official release size | 24 files; 84,729,840,092 bytes |
| License | CC0 |
| Related article | `10.1038/s41591-026-04530-3` |
| Associated code | `https://github.com/nptl-stanford/array-paper` |

The release has been acquired in shared read-only research storage. It is not
copied into this episode. No neural result from a held-out participant has been
opened for EP16.

## Two release components

| Component | Participants | Arrays | Sessions | EP16 role |
| --- | ---: | ---: | ---: | --- |
| Yield | 14 | 20 | 2,289 | Optional label-blind descriptions of recording state |
| Decoding | 9 | 14 | 729 | Cross-session prediction, recalibration, and later-session benchmarks |

Yield-only participants cannot be treated as decoding participants. The title's
20-year span describes the collection, not one participant followed for 20
years. Time is reported as deidentified days since implantation.

## Decoding participants

| Participant | Sessions | Implant-relative days | Primary array representation |
| --- | ---: | ---: | --- |
| T2 | 42 | 46–768 | Single array |
| T3 | 8 | 36–406 | Single array |
| T5 | 92 | 40–2667 | Combined lateral and medial arrays |
| T6 | 124 | 88–1211 | Single array |
| T7 | 35 | 50–545 | Combined lateral and medial arrays |
| T8 | 58 | 38–1033 | Combined lateral and medial arrays |
| T9 | 117 | 28–758 | Combined lateral and medial arrays |
| T10 | 57 | 49–363 | dPCG array |
| T11 | 196 | 41–1710 | Combined lateral and medial arrays |

Separate-array analyses in a participant with two arrays are repeated
measurements, not additional people.

## Available measurements

Each decoding session is a MATLAB file containing:

- participant and implant-relative day;
- neural features in 10-ms bins;
- electrode identity, array label, Utah-grid coordinates, and sometimes
  impedance;
- cursor and target positions;
- block and whole-trial boundaries; and
- spike-band power (`sbp`) plus threshold-crossing counts at `tx_3`, `tx_3_5`,
  `tx_4`, `tx_4_5`, `tx_5`, and `tx_5_5`.

Yield files contain electrode and array identities, threshold-based firing-rate
summaries, waveform summaries, threshold crossings, and impedance where
available. Yield signals from two arrays can have different time lengths and
must not be assumed synchronized.

The release does **not** provide raw continuous voltage, the historical online
decoder weights or versions, the complete online preprocessing state, reliable
task names, target size, muscle activity, raw video, or a direct intention
label. EP16 must not imply that these missing measurements were reconstructed.

The [verified provider-field note](outputs/schedule_definition.md#october-6-public-source-update--documented-fields-and-unresolved-session-annotations)
records the documented 10-ms bins, task-coordinate positions, zero-based
starts/exclusive ends, and feature/impedance units. These source definitions
do not identify each session's size dependence, intended transitions or
controller history. Published task-family names alone cannot supply that
session-level evidence.

## Matching sessions fairly

Primary pairs must share:

- the same participant;
- the same implanted physical array set;
- the same observed target schedule;
- a 1–30 day gap or a gap of at least 180 days; and
- enough complete trials and blocks for the fixed evaluation packet.

The primary direction is chronological: the source day precedes the target
day. The reciprocal direction is retained as a required sensitivity to
transport asymmetry, but it does not enter the adjusted long-gap or local-
signal estimands.

Days 31–179 are descriptive only. A 14-day versus 365-day sensitivity may be
reported but cannot replace the primary comparison.

Because task names and target size are incomplete, EP16 matches observed target
schedules rather than claiming verified task identity. Target coordinates are
centered and scaled before matching. Rotation and reflection are not used to
force agreement. Sessions with multiple layouts, unstable target coordinates,
conflicting transition patterns, inadequate target coverage, or a
size-dependent schedule without target size are excluded from the primary
comparison.

A preliminary support check may use participant, day, array/electrode identity,
block and trial counts, and one target coordinate per trial. It does not use
neural values, cursor trajectories, trial success, yield, impedance, or model
scores.

Each eligible session-schedule combination must contain at least 128 declared
whole trials that can be divided by whole block into two sides of at least 64
trials. Each held-out participant must have at least six near-gap and six
long-gap eligible pairs in a common schedule group. The final scoring step may
remove a pair that lacks finite neural, cursor, target, or electrode fields; it
may not add a new pair or replace a participant after outcomes are seen.

Those pair counts do not by themselves make the long-gap effect estimable. For
each connected participant-by-schedule component, the actual source-to-target
pairing pattern must allow the long-gap indicator to be separated from source-
and target-session effects: after those session terms are removed, the long-gap
indicator must retain non-zero variation. Before neural scores are interpreted,
the constrained design must have full rank and pass pre-set limits on condition
number, residual long-gap information, and maximum session leverage. Failure is
reported as insufficient support rather than handled by dropping awkward pairs
or choosing a replacement participant.

For a chronological pair `s -> t`, let `B_t`, `L_t`, and `F_st` denote the
null, target-local, and transported scores on the same target trials. Average
each score equally across the two reversed outer folds before defining
`R_st = (L_t - F_st) / (L_t - B_t)`. The denominator is the target's locally
recoverable above-null performance. If it is nonfinite or does not exceed the
development-fixed readiness minimum, the pair and its registered connected
component are unresolved rather than being dropped. The ratio is never clipped
and no small constant is added to its denominator.

The robust check takes a median over eligible chronological sources within
each target and lag class, then a median over targets, then long minus near
within each connected component. Schedule groups and components retain equal
weight, matching the primary aggregation hierarchy.

The separate local-signal comparison uses the same connected components for
its fractional and raw-score contrasts. Within a lag class, each target session
receives equal total weight and that weight is divided equally across its
eligible source sessions. Components without both gap classes cannot contribute
to either contrast. This prevents a target with many eligible sources from
silently determining the result.

## Development and held-out participants

All sessions and arrays from one participant stay on the same side. Six people
will be used for development and three for the final held-out evaluation.

Three participants begin on the development side because their relevant data
have already been discussed in prior work or design review:

- **T5:** PRI-T and chronic recalibration analyses;
- **T11:** NoMAD and MINDFUL-related analyses; and
- **T10:** public session-level decoding-signal and angular-error summaries
  were inspected during design work.

The initial held-out candidate pool is **T2, T3, T6, T7, T8, and T9**. Before
roles are assigned, publications, public notebooks, prior outputs, and analyst
notes are checked for participant-specific exposure to the EP16 cross-session
outcomes. A person with such exposure moves to development.

Among the remaining eligible people, the three-person held-out set is chosen
without neural scores by, in order:

1. maximizing the weakest near- and long-gap pair support;
2. maximizing target-schedule coverage and balance;
3. covering both single- and dual-array settings;
4. maximizing the shortest implant-day span and then session count; and
5. participant ID as the final tie-break.

If three eligible people cannot be found, the final evaluation cannot support
the planned claim. A participant is never replaced because their outcome is
unfavourable or uncertain.

## Analysis packet

Each session contributes one direction-balanced 128-trial packet. Whole blocks
form two 64-trial sides, and their fitting and evaluation roles are reversed.
Within each fitting side, the first 16, 32, and 64 eligible trials form nested
recalibration packets.

The primary feature is mean `sbp` in one 300-ms onset window. `tx_4_5` over the
same trials and window is a required sensitivity analysis. The label is the
cursor-to-target unit vector at the beginning of the window. Trial inclusion
uses only information available at that time.

Channels are placed on a common axis defined by physical array and electrode
identity. Missing channels remain represented as missing; changing column order
cannot change channel identity. A survivor-electrode analysis is reported
separately.

## Recording-state comparison

The [component design](outputs/component_details.md) distinguishes observed
channel loss/invalid samples from assumption-dependent attenuation and noise.
Marginal activity variance alone does not identify signal-to-noise ratio or
establish that a later recording state is worse. Emulator fits use fitting-
side channel summaries, not direction labels or decoding deficits. Its
evaluation uses transformed source trials; the observed comparison uses real
target trials. Both retain their own matched local/transport/null benchmarks.

The label-blind recording-state analysis may use electrode availability,
finite-value rates, and pre-specified marginal quality summaries. Same-day
yield and impedance may supplement those features when present, but their
missingness cannot determine the primary cohort.

The primary emulation applies only to chronological transitions where the
later recording is observably no better than the earlier recording in both
outer folds. Each participant needs at least six such near-gap and six such
long-gap transitions. Improvement-like, incomparable, or fold-discordant cases
remain required controls rather than being forced into the main average.

## Data separation during the study

Development analyses receive data only from the six development participants.
The preliminary support check for held-out participants exposes only the
limited structural fields listed above. Full held-out data are used only for
the final pre-specified evaluation after the development panel and numeric
thresholds have been set.

All three held-out participants are evaluated together. No partial result is
returned to development, and no result can trigger a new model, threshold,
participant, or second held-out evaluation.

## Current readiness

The structural inventory is complete for all 729 sessions, 328,548 trials and
5,150 blocks. The fixed conditional application evaluated 32 numerical designs
and passed only T11 (already development) and T9 (held-out candidate), pending
session annotations and exposure qualification. Thus the present panel does
not supply the required four supported development and three supported
held-out participants within the unchanged six/three roles. It is not a proof
that authentic missing metadata cannot change the supported design. See
[the adopted support disposition](outputs/paper_plan.md#october-6-review--current-support-and-adopted-closure).

The [prospective analysis definition](outputs/schedule_definition.md#october-6-prospective-analysis-and-exposure-definition)
also fixes raw score units, octant coverage, 30-bin feature handling,
source-only scaling, direct recording-quality order and the recorded normalized
effect anchors. Its three focused arithmetic checks passed; no neural
comparison or held-out neural outcome has been opened.

The following remain uncompleted requirements of the preserved original
design, not a selected continuation after this closeout:

- authentic session size/layout/transition evidence and exposure history,
  followed by qualification of any materially changed support panel;
- the final six/three participant assignment;
- the 128-trial block-aware packet builder;
- remaining emulation/model rules and unresolved
  readiness, raw-effect and inference settings;
- synthetic checks for the score, recalibration, missing-channel emulation,
  uncertainty, and leakage controls; and
- separate development and held-out data views.

The original support-limited closure is adopted. The optional single-case
draft remains unadopted. No unchanged support rerun or neural successor is
selected; authentic new metadata or an adopted redesign must first change
the support basis before any separately selected reopening.
