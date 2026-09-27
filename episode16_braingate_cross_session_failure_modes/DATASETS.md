# Data used in Episode 16

EP16 uses one published BrainGate release to compare how an offline neural
mapping transfers between sessions from the same participant. This file states
what is available, how participants are separated between development and the
final held-out test, and which limitations determine the scientific claim.

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

## Matching sessions fairly

Primary pairs must share:

- the same participant;
- the same implanted physical array set;
- the same observed target schedule;
- a 1–30 day gap or a gap of at least 180 days; and
- enough complete trials and blocks for the fixed evaluation packet.

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

The source release and required fields are known, but EP16 is not ready for
neural scoring. The following still need to be completed:

- target-schedule and near/long support assessment;
- the final six/three participant assignment;
- the 128-trial block-aware packet builder;
- a single onset offset and channel-quality rules;
- numeric readiness, effect, recovery, and equivalence thresholds;
- synthetic checks for the score, recalibration, missing-channel emulation,
  uncertainty, and leakage controls; and
- separate development and held-out data views.

Until those steps are complete, the source is identified but no scientific
comparison has been run.
