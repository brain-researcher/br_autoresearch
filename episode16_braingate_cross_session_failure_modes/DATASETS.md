# Dataset Contract — Episode 16

This contract pins the public BrainGate longitudinal array-performance release
for a planned, unregistered episode. It provisions no new data, opens no audit
neural outcome, and authorizes no compute. Shared adaptive-search rules are in
[`../ADAPTIVE_SEARCH_PROTOCOL.md`](../ADAPTIVE_SEARCH_PROTOCOL.md).

## Fixed release

| Item | Fixed value |
| --- | --- |
| Dataset | Performance of intracortical microelectrode arrays in people with implanted brain-computer interfaces over a 20-year period |
| Provider | Dryad |
| DOI | `10.5061/dryad.x0k6djj1h` |
| Published release | version 6 |
| Dryad API version ID | `462702` |
| Publication date | 2026-09-09 |
| Official files | 24 |
| Official bytes | 84,729,840,092 |
| License | CC0 |
| Related article | `10.1038/s41591-026-04530-3` |
| Associated code | `https://github.com/nptl-stanford/array-paper` |
| Read-only local source | `/oak/stanford/groups/russpold/data/br_autoresearch_data/braingate_long_term_array_performance/dryad-x0k6djj1h-v6` |

The local source was acquired on 2026-09-16. Every official file matched the
pinned Dryad manifest by name, byte size, and SHA-256, and all 23 participant
archives passed non-extracting `tar -tzf` checks. The controlling records are:

- `SOURCE.md`;
- `source_metadata/dataset.json`;
- `source_metadata/version.json`;
- `source_metadata/files.json`;
- `source_metadata/verification.json`; and
- `source_metadata/archive_verification.json`.

Those transfer and archive checks establish package identity, not scientific
readiness. The shared source remains outside this episode and is never copied,
committed, or directly mounted into a candidate worker.

## Released cohorts must remain distinct

| Release component | Participants | Arrays | Sessions | Permitted role here |
| --- | ---: | ---: | ---: | --- |
| Yield | 14 | 20 | 2,289 | supplemental label-blind measurement descriptors only |
| Decoding | 9 | 14 | 729 | source/target transport, calibration, and session-local benchmarks |

The pooled title does not mean that one participant was followed for 20 years.
Time is deidentified implant-relative day. Yield-only participants cannot be
promoted to decoding participants and yield sessions contain no behavioral
target data.

## Decoding participant inventory

The official summary and archive identities give the following structural
inventory. Session counts, day ranges, and array regimes are metadata, not
candidate outcomes.

| Participant | Sessions | Implant-relative days | Primary array representation |
| --- | ---: | ---: | --- |
| T2 | 42 | 46–768 | single array |
| T3 | 8 | 36–406 | single array |
| T5 | 92 | 40–2667 | combined lateral + medial arrays |
| T6 | 124 | 88–1211 | single array |
| T7 | 35 | 50–545 | combined lateral + medial arrays |
| T8 | 58 | 38–1033 | combined lateral + medial arrays |
| T9 | 117 | 28–758 | combined lateral + medial arrays |
| T10 | 57 | 49–363 | dPCG array |
| T11 | 196 | 41–1710 | combined lateral + medial arrays |

Individual-array analyses for dual-array participants are repeated,
within-participant diagnostics. They are not additional participants.

## Available variables

Every decoding file is a session-level MATLAB file with:

- deidentified participant and post-implant day;
- a 10-ms bin size;
- electrode ID, array label, Utah-grid coordinates, and sometimes impedance;
- cursor and target (x,y) positions;
- block start/end indices;
- whole-trial start/end indices and block membership; and
- `sbp` plus threshold-crossing counts at `tx_3`, `tx_3_5`, `tx_4`,
  `tx_4_5`, `tx_5`, and `tx_5_5`.

Yield files contain electrode/array identities, threshold-specific firing-rate
summaries, processed waveform summaries, threshold crossings, and impedance
when available. Dual-array yield signals may have different time lengths and
must not be assumed synchronized.

The release does **not** include continuous raw voltage, historical online
decoder weights or versions, the complete online preprocessing state, explicit
task names, target size, raw video, muscle activity, or a direct intention
label. No analysis may reconstruct or imply those missing fields.

## Metadata-only task-support inventory

The release describes Radial-8, grid, Fitts, and related cursor tasks but does
not provide a trustworthy task-name column. Before any neural-feature access,
a trusted structural extractor must create one content-addressed support row
per participant and **observed target-schedule** stratum. This does not verify
task or instruction identity.

The extractor may read only:

- archive member names;
- participant and implant-relative day;
- array and electrode identities;
- block IDs and boundaries;
- trial indices and counts; and
- one target coordinate per trial, sufficient to derive target centers and
  transitions.

It may not retain or expose:

- `neural.*` values;
- cursor trajectories;
- trial duration or success/performance summaries;
- dSNR, tuning, angular-error, yield, or impedance outcomes; or
- any model prediction or score.

Because permitted and forbidden fields coexist inside each `.mat` file, this
scan belongs to a trusted role-assignment process, not the controller. Its
temporary extraction directory is destroyed after it emits the signed support
matrix and source hashes.

Before matching, target coordinates are centered on the target-set centroid and
divided by the median nonzero inter-target distance. The classifier is
translation- and isotropic-scale-invariant but orientation preserving: it may
not rotate or reflect a session into agreement. Its signature contains
normalized centers, axes/eccentricities, and the presence—not frequency—of
allowed transitions. One relative coordinate tolerance is frozen from
target-only precision diagnostics before role assignment. Multiple layouts,
inadequate target coverage, unstable quantization, conflicting transition
graphs, or detected material instruction differences are `ambiguous` and
excluded from primary pairing. A missing task-name field alone does not exclude
a coordinate-identifiable fixed-center schedule; its claim remains schedule-
matched only. Fitts-like and other size-dependent schedules are excluded when
target size is absent. This accommodates released coordinate-scale regimes
without treating coordinate similarity as verified task equivalence.

## Two-stage source–target eligibility

Role assignment has two strictly separated gates.

**Structural preassignment** may count declared whole-trial epochs and read one
target coordinate per declared trial, but it may not evaluate cursor-dependent
near-target exclusions, finite neural features, or a decoding score. A
structurally supported session–stratum cell has at least 128 declared whole-
trial epochs and permits whole blocks to be partitioned into two sides with at
least 64 declared trials each. A structural candidate pair shares participant,
implanted physical array set, frozen observed target schedule, and a permitted lag. Audit-role selection
requires at least six structural near and six structural long undirected pairs
per participant **within the same observed-schedule graph component**. Counts
cannot be pooled across strata to pass this gate. These counts do not claim
that a pair is scientifically scorable.

**Post-role scorability** starts only after the exposure ledger, participant
roles, structural pair manifest, and eligibility algorithm are hashed.
Development masks use development handoffs only; audit masks are produced only
by the one-shot evaluator after configuration lock. A retained pair must
satisfy every rule in `GOAL.md`:

- same participant, same physical array set, and same observed target-schedule
  stratum;
- near lag of 1–30 days or long lag of at least 180 days;
- one scorable 128-trial packet per included session–stratum cell, splittable
  into two block-disjoint 64-trial pools;
- valid blocks, trials, target and cursor coordinates, electrode IDs, and the
  primary plus falsifier neural feature; and
- both directions retained when eligible.

Days 31–179 are excluded from the primary long-versus-near contrast. A
14-day/365-day contrast may be a locked sensitivity only; it cannot replace the
primary contrast after results are known.

The evaluator may retain only pairs in the structural manifest. It cannot add
pairs, replace a participant, or relax a threshold. Every audit participant
must still have at least six scorable undirected near and six scorable
undirected long pairs in at least one common component and pass the full-rank
connected-component gates. Counts cannot be pooled across strata. Failure is
emitted atomically as `closed_task_support_incommensurate`; it authorizes
neither role replacement nor a second opening.

## Provisional 6/3 whole-participant roles

All sessions and arrays from one participant stay on one side. No session,
array, or electrode split can create independence.

The initial forced-development set is:

- **T5:** directly exposed by PRI-T and chronic recalibration work;
- **T11:** directly exposed by NoMAD and MINDFUL-related analyses; and
- **T10:** public session-level dSNR/interval/angular-error rows were displayed
  during the 2026-09-21 design audit.

No local raw T10 `.mat` file was opened, and no cross-session transport score
was computed. The conservative development assignment prevents that public
within-session exposure from being mistaken for a sealed local-signal audit.

`{T2,T3,T6,T7,T8,T9}` is only the initial audit-candidate pool. Before role
freeze, regenerate the exposure screen from publications, public notebooks and
outputs, caches, prior agent contexts, and analyst records. Additional
participant-specific exposure to cross-session transport, adaptation, or an
EP16 terminal-driving quantity moves that participant to development.
Dataset-wide within-session publication exposure instead downgrades the
evidence tier for all nine and does not by itself force all nine to development.

Recompute the audit pool as the complement of the regenerated forced set, then
enumerate every three-person audit set. Reject any set failing the **structural**
pair/trial rules. Among eligible sets, select deterministically by this
lexicographic ordering:

1. maximize the minimum participant-level near/long pair support;
2. maximize balance and coverage across target-schedule strata;
3. maximize coverage of single- and dual-array regimes;
4. maximize the minimum implant-day span, then session count; and
5. break an exact tie by SHA-256 with salt
   `ep16_braingate_role_v1`.

The selected three are audit; all six others are development. The exposure
ledger, role manifest, structural candidate-pair manifest, frozen scorable-mask
algorithm, support matrix, and rejected-set reasons are hashed before neural
data are opened. After this hash, participants cannot move sides. If fewer than
three unexposed structurally eligible candidates remain, the episode closes as
`closed_task_support_incommensurate`.

At least four of the six development participants must independently pass the
same scorable packet and rank gates before adaptive development can count a
complete panel. Exposed but unsupported participants remain on the development
side and in the exposure ledger; they are not counted as completed evidence.
Fewer than four keeps the episode launch-blocked rather than silently reducing
development diversity.

This split is not yet instantiated. The current names and 6/3 wording are a
provisional design, not a claim that three scientifically scorable audit
participants exist.

## Analysis-level data contract

The primary neural representation is combined-array `sbp` for dual-array
participants and the available single/dPCG array otherwise. Each session file
is first aligned to a frozen physical-electrode axis keyed by
`(implanted array identity, electrode ID)`. A missing/invalid target channel is
set to the source-training center after source-fitted scaling rather than
deleting the column. Target-local models use the same axis with target-fold-fit
scaling. Pairwise survivor channels are a control, not the primary axis.

Each whole trial contributes exactly one predictor and one proxy label. For a
globally shared onset offset `δ`, mean `sbp` is computed over one 300-ms
onset-relative window; the label is cursor-to-target direction at the start of
that window. `tx_4_5` is recomputed over the identical neural window. Nonfinite/near-target
rules are fixed in `GOAL.md`. Ten-millisecond bins and overlapping windows are
never separate examples. One numeric `δ` is human-signed from task timing and
physiological prior before any real neural score; it is not tuned on
development performance and cannot vary by participant, session, feature,
pair, or audit outcome.

For every session–stratum cell, a hash-seeded direction-balanced procedure
first assigns whole blocks to two disjoint sides, each with at least 64
scorable trials, then selects exactly 64 trials within each side. These form two
reciprocal outer folds: in each fold, `C64` is one side and `E64` the other,
with roles swapped in the other fold.
`C16 ⊂ C32 ⊂ C64` is nested and content-addressed. `B` and `L` train on target
`C64`; `F(s→t)` trains on source `C64` and evaluates target `E64`; `A(m,b)` may
use only target `Cb`. “Full labels” means `C64`, not the full session. Ridge
selection is nested within fitting trials or frozen from development. Blocks
never cross fitting/evaluation roles, and folds are aggregated before ratios.

The frozen-source baseline receives zero target samples. A fixed-packet
zero-label method receives exactly `X(Cb)` and never `Y(Cb)`; a supervised
method receives the same `X(Cb),Y(Cb)`. The primary packet is `C32`; 16 and 64
are sensitivities. No method may use `E64` features to estimate normalization,
covariance, alignment, rank, stopping, or a transform. A rolling-origin
zero-label branch, if enabled, must give every competitor the same past-only
stream and log cumulative `X` exposure. Whole-session transductive
normalization is a full-`X` diagnostic upper bound and cannot earn low-cost,
zero-label, or deployment credit.

The published per-session nested dSNR result is not `F(s→t)`: it
uses target-session labels for reaction-time/output calibration and cannot
represent frozen cross-session transport.

## Observable measurement state

Read-only preflight found exact-day yield coverage for 685 of 729 decoding
sessions and exact-day impedance for 430 of 729. Impedance coverage is highly
nonuniform, including only 8 of 196 T11 decoding sessions.

An exact participant/day join establishes only a shared implant-relative day.
It does not establish the same recording, block, simultaneity, or time alignment
with decoding data; dual-array yield matrices may also have different lengths.
Released waveform rows are structurally selected: waveforms occur only for
electrodes with a robust `-4.5 × RSD` threshold-crossing rate of at least 2 Hz,
and the release summarizes a median of 10-second-window means rather than the
paper's standard mean. Waveform absence is therefore missing-not-at-random.

Accordingly:

- primary `M_s` and `M_t` use electrode/array identity and prespecified
  label-blind marginal channel summaries computed from their decoding-session
  fitting packets;
- exact-day yield, waveform, and impedance variables are supplemental
  diagnostics with explicit missingness indicators;
- a complete-case impedance cohort cannot replace the primary cohort; and
- no supplemental field may define, filter, impute, or replace the primary
  cohort, and no missing measurement may be imputed from performance or labels.

Target marginal neural summaries can mix recording quality, neural state, and
behavioral context even when labels are hidden. They are observable
measurement-state descriptors, not pure hardware measures.

In each outer fold, primary neural marginals for `M_s` and `M_t` use the
respective source and target `X(C64)` fitting packets only, with labels hidden.
Evaluation-pool features cannot define or tune the primary emulator. A whole-
session marginal is a separately named transductive diagnostic and cannot
replace the primary profile.

The primary counterfactual is pairwise `M_s→M_t`, not an unconditional target
profile pasted onto every source. A human-frozen label-blind partial order must
classify chronological transitions as degradation-admissible, improvement-like,
or incomparable. The emulator may remove/attenuate/add noise but never create
information or restore a source-missing channel. Participant-level measurement
sufficiency requires at least six near and six long transitions classified as
degradation-admissible from both fold-specific `C64` profiles. Each consensus
pair counts once; a discordant fold call is a falsifier, not partial support.
The frozen weighted aggregate over the same admissible components must contain
a positive long-lag excess mismatch, and the emulator must reproduce every
contributing component's excess within the frozen maximum-error margin;
opposite reproduction errors cannot cancel. Matching marginal quality while
missing the terminal-driving deficit is insufficient. Reciprocal reverse,
improvement-like, incomparable, and fold-discordant transitions remain
falsifiers only.
Failure of this minimum in an audit participant is support failure and closes
atomically as `closed_task_support_incommensurate`; it is not a negative
measurement result.

The identity-aware emulator, survivor-electrode analysis, and matched-random
degradation bank must share one frozen electrode universe and channel-QC rule.
The random bank size, seeds, aggregation, and identity-versus-random separation
margin are part of the pre-score lock. This control participates in the main
measurement tri-state; all five random-emulator errors receive the same
simultaneous uncertainty treatment as their identity-aware counterparts. It is
not a post hoc subtype label.
Stable electrode ID controls explicit channel availability only; it does not
establish stable neuron identity.

## Required physical handoffs

The mixed participant archives cannot be mounted directly into either a
candidate worker or a nominal audit process. Before launch an operator must
create three immutable, content-addressed handoffs:

1. **Development full:** all permitted decoding/yield material for the six
   development participants.
2. **Audit structural:** only the frozen support/geometry manifest and opaque
   participant roles; no audit neural features, cursor trajectories, local
   scores, or measurement outcomes.
3. **Audit evaluator:** complete required files for the three audit
   participants, mounted only after configuration lock to a trusted evaluator
   with no network egress.

The candidate/controller sees only development full and audit structural. The
evaluator applies the locked procedure to the full audit handoff, emits the
complete terminal panel atomically, and never exposes pair-, session-, or
participant-level partial results during execution. After the one permitted
opening, no tuning, participant replacement, threshold change, or rescue rerun
is allowed.

`inputs/` remains read-only and contains no symlink to the shared 84.73-GB
source. Transient extraction belongs under
`$SCRATCH/autoresearch/episode16_braingate_cross_session_failure_modes/` only
after authorization.

## Exposure and novelty ledger

The dataset paper, provider README, and associated code are public. All nine
decoding participants are publication-exposed for within-session yield/dSNR/
tuning/electrode-count analyses. This episode therefore offers at most a
procedure-sealed internal replication of a new cross-session diagnostic panel,
not pristine external confirmation.

Specific prior exposure includes:

- T5 and T11 in MINDFUL-related chronic-instability work;
- T5 and 73 sessions in PRI-T recalibration comparisons;
- T11 in the human NoMAD alignment analysis; and
- T10 session-level rows in the official
  `data/decoding_summary.csv` viewed during this design audit at Git commit
  `93beb76b4329f0e3d5b3fd5ffba9be1c7cfa2834`.

The design audit did not extract or open local raw `.mat` neural values and did
not use public dSNR to select models, task strata, or effect margins. The T10
exposure is nevertheless recorded and conservatively forces T10 to
development.

Before role freeze, the exposure ledger must also search publications,
notebooks, caches, prior agent contexts, and analyst records for participant-
level cross-session transport or adaptation outcomes. Any additionally exposed
participant is forced to development. If that leaves fewer than three eligible
audit participants, the episode closes for task/exposure incompatibility rather
than weakening the firewall.

## Human-participant and reuse boundary

The release is public, deidentified clinical-trial data under CC0, but it
remains human-participant research data. Do not attempt reidentification,
combine deidentified timing with external personal information, or redistribute
large derived participant-level neural matrices through Git. Cite the dataset
and associated article in every scientific output.

## Launch blockers

The episode cannot launch until all of the following exist and are reviewed:

- target-only observed-schedule/support matrix, frozen coordinate tolerance,
  ambiguity rules, regenerated exposure ledger, and exact 6/3 role manifest;
- immutable structural near/long candidate-pair manifest plus frozen post-role
  scorable-mask algorithm;
- role-filtered development, audit-structural, and evaluator handoffs;
- frozen 128-trial/two-fold builder, single onset offset, calibration
  packets, and primary physical-electrode axis;
- frozen channel-QC rule, normalization constants, adaptation rank/df/
  regularization/stability bounds, finite candidate-recipe set, final `m*`
  selection/no-refit rule, and exact degradation-emulator operator;
- frozen measurement-state partial order, no-information-creation gate, and
  two-fold consensus admissibility/support rule, admissible-excess maximum-
  component-error model, matched-random bank, and identity-separation rule;
- full-rank connected-component rules and numeric condition, residualized-long-
  column, leverage, and heterogeneity-partition gates;
- at least four support-qualified development participants;
- human-signed denominator, effect, recovery, and equivalence margins;
- synthetic qualification and leakage tests plus the exact 20-row coverage
  allocation;
- profiled trial-summary cache and verified CPU/wall ceilings;
- permission-separated one-shot evaluator and audit firewall;
- registered adaptive controller/program binding; and
- a fresh canonical state read plus explicit scientist launch approval.

Source availability, local file verification, or this dataset contract alone
does not authorize extraction, search, audit access, or launch.
