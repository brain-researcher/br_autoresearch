# Data used in Episode 16

The October 6 scientist-adopted [local closeout](outputs/experiment_log.md#2026-10-06--structural_support--scientist-adopted-support-limited-closeout)
records **support_or_uncertainty_prevents_an_answer** for the original
three-held-out-participant claim. Existing structural evidence and exposure
boundaries remain; no neural analysis, role replacement or case-series
successor is adopted.

**October 16 source-fact reassessment.** Under the scientist's 2026-10-16 standing authority (missing source facts are stated assumptions, not stopping points), the open size, layout, transition and precision items were resolved or assumed before application; see the [item table](outputs/schedule_definition.md#october-16-source-facts-and-adopted-assumptions). Support is unchanged under the primary rules and nine sensitivities: only T9 is held-out eligible, and T9 and T11 are supported. The 4+3 design remains unsupported; claims are conditional on those assumptions. The scientist then adopted the T11-to-T9 case series ([roles and T9 exposure review](#case-series-roles-and-t9-exposure-review)).

EP16 uses one published BrainGate release to compare how an offline neural
mapping transfers between sessions from the same participant. This file states
what is available, how participants are separated between development and the
final held-out test, and which limitations determine the scientific claim.

## Public online resources — checked October 6, 2026

| Public source | Available information and EP16 use | Remaining limit |
| --- | --- | --- |
| [Dryad release and embedded README](https://datadryad.org/dataset/doi:10.5061/dryad.x0k6djj1h) | Download links for the nine decoding archives and yield archives, plus the field codebook: 10-ms bins, task-coordinate x/y positions, zero-based starts/exclusive ends, electrode/array identifiers, impedance in kΩ, `sbp` in µV² and threshold crossings per bin. Use these definitions to implement source readers without guessing units or indexing. | Task families are named, but the documented fields do not supply per-session target size, controller versions or intended-transition rules. Yield arrays need not be synchronized. |
| [Pinned provider participant/array metadata](https://raw.githubusercontent.com/nptl-stanford/array-paper/93beb76b4329f0e3d5b3fd5ffba9be1c7cfa2834/array_paper/metadata.py) | Explicit decoding-participant order, array names and provider display conventions. Use for identity interpretation and cross-checking array labels. | It is not a session-level task/size map or a record of analyst exposure. |
| [Preprint supplement](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12236888/supplementaryFiles) (Supplementary Methods 1.5, 2.4) and methods sections of earlier BrainGate task papers | Each block is one task. Blocks were chosen as target-directed tasks (radial-8, Fitts, grid). Radial-8, four-target center-out-back and n×n grids use fixed-size targets; Fitts and random-target tasks vary size. Used for the October 16 layout-only rules. | No per-session target size, controller version or task label. The read was limited to methods; exposure is disclosed in the log. |
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

## Case-series roles and T9 exposure review

**Roles under the October 16 case-series amendment (adopted).** These override the six/three allocation above for the new claim.
- T11 is the primary development case.
- T9 is the single held-out evaluation case and is opened once.
- Proposed: T3, T5 and T10 (development under the original contract) supply component-level checks only.
- T2, T6, T7 and T8 stay unopened, with no role.

T9 was chosen from structural support alone: it is the only candidate that passes the support gates under every source-fact variant. The exposure listed below played no part in that choice.

**T9 exposure review — final, October 16.** It records source types and who saw them, never values.

| Source checked | What was accessed | Bearing on T9's EP16 outcomes |
| --- | --- | --- |
| EP16 structural work, Oct 3–16 | Session days, block and trial indices, one target point per trial, pair support and design metrics | Outcome-free; not exposure |
| Oct 3 reviewer disclosure ([log](outputs/experiment_log.md#2026-10-03--structural_support-public-method-exposure-disclosure)) | Reviewer `/root/inventory_ep08_14` saw a preprint excerpt with T9 yield and within-session decoding summaries | Within-session only; no cross-session transport result. That reviewer stays out of EP16 development and evaluation |
| Oct 16 source-fact reads | Abstract-level pooled results, including that tuning was stable over about one month across 11 arrays (T9's arrays included but not separable). Supplement 2.6 implies T9's arrays consistently reached dSNR ≥ 1 | The tuning statement is pooled, not participant-specific; the dSNR fact is qualitative and within-session |
| Provider repository @93beb76 | Code read. `data/` summaries, including `tuning_stability_*.csv`, and the notebooks were seen by name only | These unopened files hold T9's per-array tuning-stability results, the closest cross-session outcome; keep them unopened |
| Article and preprint figures | Not viewed: the article page required login, and the preprint was read through methods-only extraction | Figure 5 (tuning stability by array) probably shows T9; keep it unviewed |
| Other repository episodes | A filename-only search found no other episode contract that uses this release | No cross-episode exposure found |
| Other T9 publications (title inventory) | Titles only, none read: tablet-computer control 2018, intention-estimation calibration 2018, discriminative Kalman filter 2020, replay during rest 2020, retrospective click calibration 2016 | No title is a cross-session decoder-transfer study; contents unreviewed |
| Scientist | Confirmed on October 16: no detailed prior knowledge of T9's cross-session decoding or tuning-stability results | No exposure |
| Agent sessions outside the repository | Unknown beyond the disclosures above | Covered by the protections below |

**Final disposition (October 16).** There is no participant-specific EP16 cross-session outcome exposure for T9, and the scientist confirmed having no detailed prior knowledge of T9's cross-session results. T9 is therefore confirmed as the single held-out evaluation case under the existing rule. Its known yield and within-session exposure stays disclosed and does not reassign it.

Until the T9 evaluation finishes:
- EP16 workers do not open the article's results figures, the provider `data/` files or notebooks, or the results of any T9 publication;
- only the sealed evaluation step reads T9 neural and cursor fields.

**Role seal (October 16, implemented).** The development reader ([ep16_roles.py](outputs/code/ep16_roles.py), [mat5_neural.py](outputs/code/mat5_neural.py)) decodes neural and cursor fields for T11 only (T3/T5/T10 are allowed for component checks but not used). It refuses T2/T6/T7/T8 always. It refuses T9 unless given a token, and only [evaluate_t9_sealed.py](outputs/code/evaluate_t9_sealed.py) issues one, after it has:
1. verified the freeze manifest's SHA-256 digests;
2. confirmed that no earlier T9 evaluation directory exists;
3. created the write-once output directory and its `t9_opened.json`.

The role is checked twice, once from the archive name and once from the in-file participant ID, before any value is decoded. T9 arrays stay in memory; no T9 feature cache is written. This is a code-path seal, not operating-system isolation.

The report discloses T9's yield and within-session exposure, and that the release is publication-exposed. If anyone working on EP16 sees T9's tuning-stability results before the evaluation, record the exposure. The evaluation would then be reported as exposed, because there is no replacement case.

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

On October 16 the missing size, layout, transition and precision facts were resolved or replaced by stated assumptions fixed before application ([rules](outputs/config/source_fact_rules.json); [result](outputs/runs/structural_support/20261016-source_fact_support01/participant_support_by_variant.csv)). The re-run kept every session.
- T9 is held-out eligible with 153 near and 909 long pairs; T11 is supported.
- T2, T6 and T10 are rank-deficient; T5, T7 and T8 fail the adopted information or leverage gates.
- No variant gives three held-out-eligible candidates.

Missing metadata is therefore no longer the reason the design cannot be supported.

The [prospective analysis definition](outputs/schedule_definition.md#october-6-prospective-analysis-and-exposure-definition)
also fixes raw score units, octant coverage, 30-bin feature handling,
source-only scaling, direct recording-quality order and the recorded normalized
effect anchors. Its three focused arithmetic checks passed; no neural
comparison or held-out neural outcome has been opened.

**October 16–17: pre-evaluation pieces built** ([analysis specification](outputs/config/analysis_specification.json); [log](outputs/experiment_log.md)).

The role seal, packet builder, numbers, emulator and adaptation rules, positive control, checks (49 passing, including 20 seal and freeze checks), search rules and T9 command are all in place, revised after an independent review. T11 development features are extracted for its 64 component sessions. Three facts about the data emerged:
- *Channel identity.* `electrode_id` is a column number: T11 day 1549 lists the medial array first. Channels are therefore keyed by array and Utah-grid position.
- *Data quality.* Two T11 sessions have one dead 96-channel array. sbp window means contain artifacts up to 10⁷ µV², against a median near 40. A window above 10× the median of the channel's non-artifact windows (a fixed-point fence) is treated as a missing feature, never as a deleted trial.
- *Block shifts.* sbp baselines shift between blocks. Under the frozen source-only scaling T11's median local J is 0.03; block-wise normalization (the provider's convention) gives 0.41. Block normalization is the proposed primary, pending the scientist.

The T11 known-positive control fails under the frozen readiness rule: some T11 sessions have no local signal, so the component is unresolved.

Remaining before the single T9 evaluation:
- the scientist's decisions on block normalization and the readiness rule;
- a T11 control that passes under those decisions;
- the T11 development search;
- the freeze manifest;
- the one sealed run.

The original three-held-out claim keeps its support-limited disposition.
