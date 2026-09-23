# Dataset Contract — Episode 19

This contract pins the sources, roles, timing provenance, and access boundaries
for a planned, unregistered episode. It does not provision data, open held-out
signals, authorize compute, or establish that the sources are scientifically
ready. Shared adaptive-search rules are in
[`../ADAPTIVE_SEARCH_PROTOCOL.md`](../ADAPTIVE_SEARCH_PROTOCOL.md).

## Source and role summary

| Source | Fixed role | Current local state |
| --- | --- | --- |
| WAY-EEG-GAL Figshare collection v2 | open development, no-feedback series-8 lock, and public series-9 additional campaign-sealed evaluation | acquisition-verified, unextracted steward quarantine; role-filtered handoffs absent |
| Kaggle Grasp-and-Lift EEG Detection | historical task/split/metric specification only | not a scientific source handoff |
| Self-paced/free-choice EEG reaching Figshare v1 | 15-participant development set plus 8-participant whole-person replication set | acquisition-verified, unextracted steward quarantine; role-filtered handoffs absent |
| AJILE12 DANDI published version | deferred exploratory ECoG extension only | Acquisition complete for 55 archives in legacy-worktree quarantine; no role-filtered handoff |

No candidate may substitute a preprocessed mirror, notebook cache, or moving
dataset draft for a pinned source below.

### Physical-location record

The canonical checkout has no repository-local `.steward_acquisition` tree.
Resolve these sources through
[`DATA_LOCATION_MANIFEST.json`](../DATA_LOCATION_MANIFEST.json):

- `legacy_steward_acquisition` is the current legacy-worktree quarantine;
- `asset_ep19_public_sources`, `asset_ep19_safe_commit`, and
  `asset_ep19_ajile12` identify the EP19 acquisitions within that quarantine;
  and
- `planned_steward_acquisition` is their planned durable external root and
  remains `absent_planned_target`.

The EP19 assets remain `staged_not_moved`. These identifiers describe storage
inventory only: they are not extracted, role-filtered episode handoffs and do
not change the launch-blocked state.

## WAY-EEG-GAL

### Immutable source

| Item | Fixed value |
| --- | --- |
| Dataset | WAY-EEG-GAL: multimodal grasp-and-lift recordings |
| Figshare collection | `10.6084/m9.figshare.c.988376.v2` |
| Participant records | 12 child records, each currently version 1 |
| Participant ZIP bytes | 10,339,259,047 total |
| Descriptor | `10.1038/sdata.2014.47` |
| Official utilities | `https://github.com/luciw/way-eeg-gal-utilities` |
| Conservative license policy | CC BY 4.0 |
| Read-only local source | Logical asset `asset_ep19_public_sources` in the root location manifest; currently quarantined under `legacy_steward_acquisition`, not an episode handoff |

The descriptor reports a CC BY 4.0 release, while the current Figshare API
reports CC0 for the child records. EP19 uses the more conservative CC BY 4.0
attribution and redistribution policy and records the discrepancy in the
future `SOURCE.md`.

The collection DOI alone is not an integrity manifest. The steward acquisition
has retained all 12 child API records and frozen their article IDs, file IDs,
file names, byte sizes, MD5 values, API JSON, and acquisition timestamp. Before
use, a trusted builder must carry those records, the legacy child DOI strings,
and later collection modification timestamps into the signed episode source
manifest and verify the extracted inventory.

Official references:

- <https://doi.org/10.6084/m9.figshare.c.988376.v2>
- <https://doi.org/10.1038/sdata.2014.47>
- <https://github.com/luciw/way-eeg-gal-utilities>

The utilities repository is citation-only at drafting time. No moving branch
may enter preprocessing; any future code use requires an exact commit, license,
environment, and semantic check in the source manifest.

### Expected raw inventory

Use only the continuous participant archives:

- `HS_P{1..12}_S{1..9}.mat`: 108 participant-by-series files; and
- `P{1..12}_AllLifts.mat`: 12 event/metadata files.

The source provides 32-channel EEG at 500 Hz, five EMG channels at 4 kHz,
three-dimensional hand/wrist/object kinematics and force-related signals at
500 Hz, cue state, and event timing. Exact per-file shapes, clocks, units,
missing channels, and alignment are readiness facts to verify from the pinned
source; they are not inferred from the paper alone.

Third-party filtered arrays, extracted competition tables, and previously
normalized tensors are prohibited for the strict forecasting track.

Track A historical compatibility uses the native 32-channel montage expected
by the archived implementation. Tracks B and C use the frozen 29-channel
intersection listed below. A result cannot silently move between these channel
contracts.

### Series roles

Every participant P1–P12 follows the same time-ordered contract:

| Series | Role |
| --- | --- |
| 1–6 | fit, inner selection, and local qualification folds |
| 7 | rate-limited aggregate-feedback development board |
| 8 | campaign-sealed no-feedback lock set |
| 9 | public but campaign-sealed additional evaluation, pooled with series 8 |

After configuration lock, the frozen recipe may refit on series 1–7 only. The
trusted evaluator opens every participant's series 8 and 9 together with the
second-task participant audit. Series 8 and 9 are public and prior-exposed; they are campaign
seals, not globally private or independent confirmation. No series may be
subdivided into random windows that cross train, feedback, or evaluation roles.

### Historical Kaggle boundary

The Kaggle competition defines the legacy compatibility task:

- participant series 1–8 as training;
- series 9–10 as test;
- six framewise event labels expanded approximately plus/minus 150 ms;
- no use of future samples in a prediction; and
- mean column-wise event AUROC.

Official page: <https://www.kaggle.com/competitions/grasp-and-lift-eeg-detection/data>.

The archival winning implementation is pinned to repository commit
`36fe555d523c3ca3f201e765b1b1004dc5383dd2` under its BSD-3-Clause license:
<https://github.com/alexandrebarachant/Grasp-and-lift-EEG-challenge/tree/36fe555d523c3ca3f201e765b1b1004dc5383dd2>.
`Safe1` is the preferred positive-control stack. Its exact commit, source
archive, commit API response, repository metadata, BSD-3-Clause license, and
local SHA-256 values are now acquisition-verified in steward quarantine. It is
not yet executable benchmark infrastructure: the episode manifest must record
whether execution uses native Python 2.7/Theano or a separately validated,
explicitly labeled semantic port.

The public multimodal Figshare release contains only series 1–9. Kaggle series
10 is an unpublished EEG-only private-leaderboard series without public raw
EMG, kinematics, force, or complete outcome lineage. Competition artifacts
remain subject to Kaggle rules and are not redistributed as the immutable
scientific source.

Consequently, EP19 can reproduce a pre-audit legacy-compatible bridge on
series 1–6 to 7 and compare a frozen reference panel. The exact 1–8-to-9 public
compatibility run occurs only after the joint scientific evaluation and cannot
promote or rescue a model. EP19 cannot claim to have rerun the private
leaderboard or reordered historical entrants unless their exact code, features,
weights, and predictions become independently executable.

## WAY onset and causal history

Provider `HandStart` is a secondary agreement check, not the primary forecast
target. The primary peripheral onset is reconstructed from continuous raw
signals as

\[
\tau_{WAY}=\min(\tau_{EMG,1},\ldots,\tau_{EMG,5},\tau_{wrist}).
\]

Here, `wrist` is the P4 marker velocity computed with backward differences.
Object motion, grip force, and load force are frozen consistency checks rather
than target-defining channels.

The detector must satisfy all of the following:

1. Resting location, scale, and thresholds are fit only on WAY series 1–7
   pre-cue samples; series 8–9 can never tune them.
2. Filtering, envelope construction, derivatives, normalization, resampling,
   and state updates are one-sided and replayable sample by sample.
3. A candidate crossing must satisfy a frozen dwell/confirmation rule. The
   first provisional crossing immediately puts the stream in `pending`, during
   which no anchors are emitted. Future samples may confirm the first-crossing
   label or prospectively re-arm after a failed crossing, but intervening
   anchors are never reinstated.
4. The at-risk/stillness decision at anchor \(t\) uses only signals at or before
   \(t\).
5. Gaps and series boundaries reset every filter and model state.
6. Development signal injection estimates clock error and detector latency;
   the frozen safety bound must remain below the 300 ms primary-horizon edge.
7. Ambiguous/missing onsets are removed by a rule frozen before candidate EEG
   scores. Manual movement of an onset to improve results is prohibited.

All EEG, EMG, peripheral, and cue clocks must be reconciled in a signed timing
certificate. A future-to-past impulse response above numerical tolerance makes
the affected pipeline ineligible.

For both sources, the 19-category target uses 18 left-closed, right-open 50-ms
intervals, \([50(k-1),50k)\) ms, plus `>900 ms`. When acquisition ends at an
outcome-independent censoring time inside a bin, the censor time is floored to
the last complete 50-ms boundary. Training then sums only masses whose interval
starts at or after that boundary, plus `>900`; the straddling partial bin is
discarded. Binary endpoint scoring still requires observation through the
endpoint's full upper edge. Outcome-dependent truncation is ineligible.

## Self-paced/free-choice EEG reaching audit

### Immutable source

| Item | Fixed value |
| --- | --- |
| Dataset DOI | `10.6084/m9.figshare.28632599.v1` |
| Descriptor | `10.1038/s41597-025-06039-9` |
| File | `Freewill_EEG_Reaching_Grasping.zip` |
| Figshare file ID | `57518986` |
| Bytes | 13,591,548,048 |
| MD5 | `3b7c3039c5c9fb6abf1429a830301711` |
| License | CC BY 4.0 |
| Read-only local source | Logical asset `asset_ep19_public_sources` in the root location manifest; currently quarantined under `legacy_steward_acquisition`, not an episode handoff |

The release contains 23 people, 49 sessions, and 6,808 trials, with raw
continuous BrainVision EEG, four EOG channels, audio/TRIG, and three-axis wrist
accelerometry. Twenty-one participants were recorded at 250 Hz; `sub-13` and
`sub-15` were recorded at 1,000 Hz.

Already exposed header-only verification found the same first 31 EEG channels in all 49
sessions: 43 sessions use 250 Hz and the six sessions of `sub-13`/`sub-15` use
1,000 Hz. Those two participants have additional numbered empty physical
channels in the header; they do not add EEG signals. The archive contains 240
raw `.eeg` runs while the paper reports 238 valid runs, so a frozen event/valid-
run manifest—not filename count—must determine eligibility. This structural
inspection did not open signal samples or event-level outcomes and belongs in
the exposure ledger with its source and inventory hash. This narrow structural
exposure is whitelisted; signal-derived or event-derived audit QC remains
prohibited.

The API currently reports version 1, while the landing record has a 2026-09-17
modification date. Acquisition has pinned the versioned DOI, file ID, byte
size, MD5, version-1 API JSON, and an independent local SHA-256. During
controlled extraction, the trusted builder must additionally retain and verify
the archive's internal `CHANGES` record. The unversioned DOI alone is not
sufficient.

Official references:

- <https://doi.org/10.6084/m9.figshare.28632599.v1>
- <https://doi.org/10.1038/s41597-025-06039-9>

### Whole-participant roles

Roles are fixed from public metadata before signal or event-level access:

- **audit:** `sub-01`, `sub-05`, `sub-07`, `sub-10`, `sub-13`, `sub-16`,
  `sub-19`, `sub-23`;
- **development:** `sub-02`, `sub-03`, `sub-04`, `sub-06`, `sub-08`, `sub-09`,
  `sub-11`, `sub-12`, `sub-14`, `sub-15`, `sub-17`, `sub-18`, `sub-20`,
  `sub-21`, `sub-22`.

The eight-person audit set has both sexes represented equally, at least two
sessions per participant, and one 1,000-Hz participant. `sub-15` in development
exposes the training pipeline to the second sampling lineage without exposing
the held-out 1,000-Hz participant.

Before any self-paced EEG outcome is read, the 15 development participants are
assigned to one frozen five-fold whole-participant partition. Each model-search
trial returns only the fold-complete aggregate; participant or fold feedback is
withheld. The final audit-facing group encoder is fitted on all 15 only after
the recipe is locked.

All sessions, runs, EEG, accelerometry, events, and QC for an audit participant
stay evaluator-only. A participant cannot be replaced after any audit signal,
event, QC statistic, or model score is opened. Participant is the inferential
unit; session and run are dependence/block units.

### Common channels and replication regimes

The prespecified intersection has 29 EEG channels:

`Fp1 Fp2 F7 F3 Fz F4 F8 FC5 FC1 FC2 FC6 T7 C3 C4 T8 TP9 CP5 CP1 CP2 CP6 TP10 P7 P3 Pz P4 P8 O1 Oz O2`

Channel order, instantaneous common-average reference, missing-channel mask,
and resampling state are frozen before audit. No interpolation using future
samples is allowed.

The audit regimes are distinct:

Canonical audit-facing group encoders are fitted only on the 15 designated
self-paced development participants under the locked recipe. No signal from an
audit participant may enter pre-lock or C0/C32 encoder fitting, self-supervised
pretraining, model selection, or normalization-state initialization. The only
exceptions are the frozen four-parameter C32 calibration and the isolated,
non-promoting Cfull diagnostic copies defined below; neither mutates a canonical
checkpoint. Direct WAY-to-self-paced weight transfer is a separate
non-promoting diagnostic, not the primary C32 estimand.

- **C32, primary:** independently for each audit participant, the evaluator
  selects the earliest 32 complete eligible reach trials in chronological
  acquisition order under the frozen valid-run/QC rule. The participant-
  specific packet contains the full causal input history, every eligible risk
  anchor and censoring target in those trials, and their onset targets—not just
  32 timestamps. All models receive that same packet. Supervised updates are
  limited to one global temperature plus three free coarse-horizon logit
  offsets (the fourth group is the reference); the encoder, spatial projection,
  and 19 free category biases cannot update. Unlabeled channel affine state may
  update only sample by sample. The map is fitted separately to isolated copies
  of standalone \(B_d^\star\), real, zero, and every surrogate/reference/
  challenger model using the identical packet and frozen algorithm; parameters
  are model-specific, while all canonical checkpoints remain unchanged;
- **C0, stringent secondary:** no audit labels and no audit EEG weight update;
  only development-initialized causal normalization state may update from past
  unlabeled samples, never from full-session or future statistics; and
- **Cfull, diagnostic:** for a participant with at least four complete valid
  runs, use the earliest half of runs (chronological, rounded down) as a local-
  refit prefix and score only the later half. Within the prefix, complete runs
  are divided chronologically as close as possible to 80/20, with at least one
  training and one validation run. The inner split selects only the epoch
  count. Every fit starts from an isolated copy of the locked checkpoint and
  uses the frozen seed schedule for minibatch order and stochastic operators.
  The evaluator then resets a diagnostic copy to that checkpoint and fine-tunes
  the unchanged recipe on the full prefix for exactly the selected count.
  Architecture, preprocessing graph, loss, optimizer, hyperparameters, twin
  construction, and seed schedule stay fixed; every otherwise trainable weight
  in the diagnostic copies of the source-specific baseline and real, zero, and
  surrogate twins may update from the prefix only. Canonical \(B_d^\star\), C0,
  C32, and their stored predictions are never mutated. Cfull estimates an upper
  bound and cannot promote or rescue a claim. Fewer than
  four valid runs is prespecified as insufficient Cfull support, not silently
  repaired by a trial-level split.

C32 and Cfull are never described as zero-shot. For C32, the entire stream
prefix through the 32nd event plus its complete filter/history guard is excluded
from scoring; no anchor between calibration events is reused. C0 and C32 are
compared on the identical post-C32 suffix. Cfull has a different, explicitly
labeled score support and is never compared as if it shared that suffix. No
Cfull suffix label, suffix-derived normalization statistic, or suffix early-
stopping result may enter the local refit.
Calibrated parameters persist within participant, filter/recurrent state resets
at each session, and C0/C32/Cfull run in isolated evaluator copies. Audit
participants are opened together only after the model, regimes, evaluator, and
outputs are locked.

### Self-paced onset and claim limit

The provider's `AccStartIndex` used a zero-phase 10-Hz low-pass filter,
derivative thresholding, and manual adjustment. It is therefore a secondary
agreement diagnostic only.

The primary onset is recomputed from raw XYZ accelerometry with the frozen
one-sided detector family, thresholds fitted only on the 15 development
participants, gap resets, and timing certificate. C32 cannot alter detector
parameters. The same provisional-crossing `pending` rule used for WAY applies.
The non-neural baseline may use past accelerometry, EOG,
cue/start state, time since cue, fixed spatial setup, and stillness duration,
but never future samples. `TgtID` or the ultimately chosen cup is known from
the completed movement and is prohibited at forecast time.

This source has no EMG. It can replicate prediction before
accelerometer-detectable wrist motion beyond measured past non-neural history;
it cannot confirm prediction before all peripheral muscle activation. The
published aggregate ERP/classification results and use of 15 development
participants mean this is a second-task whole-participant campaign seal, not a
globally pristine or unseen-task source.

## Required permission-separated handoffs

Large payloads remain outside Git. `inputs/` receives only content-addressed,
role-filtered manifests or read-only mounts created by an authorized operator.
The required handoffs are:

1. **WAY structural pack:** immutable source manifest, clocks, shapes, channel
   identities, cue/trial/series boundaries, and hashes without signal-derived
   candidate scores.
2. **WAY development pack:** P1–P12 series 1–7 and required `AllLifts` rows.
3. **WAY joint-final-evaluation pack:** P1–P12 series 8 and 9, held only by the
   trusted final evaluator.
4. **Self-paced metadata pack:** participant/session/run/channel/sampling
   inventory and the frozen role manifest without audit event or QC outcomes.
5. **Self-paced development pack:** complete data for the 15 development
   participants.
6. **Self-paced audit pack:** complete packets for all eight audit participants,
   mounted only inside the no-egress trusted evaluator after lock.
7. **Legacy artifact pack:** any permissible Kaggle labels, metric code,
   historical submissions, or entrant implementations, including the pinned
   `Safe1` code, each with license, provenance, and hash. Missing artifacts
   narrow the compatibility claim.

No mixed source directory may be symlinked into `inputs/`. Candidate workers
must never receive an audit packet, audit path, audit file list beyond the
frozen structural manifest, or partial evaluator output.

## Audit firewall

Before any audit outcome is opened, hash and freeze:

- participant and series/session roles;
- complete human/agent/cache/paper exposure ledger;
- raw-source manifests and timing certificates;
- onset/stillness implementation, horizons, exclusions, and natural-risk
  weights;
- common channels, resampling, normalization, calibration events, and model
  states permitted to update;
- reference models, challenger, seeds, practical margins, uncertainty rule,
  and terminal decision table; and
- the evaluator image, no-egress policy, exact command, and output schema.

The evaluator opens WAY series 8, WAY series 9, and all eight self-paced audit
participants in one joint operation. It returns only the complete
participant-level sufficient
statistics and terminal inputs after both components finish. No partial score,
fold result, QC failure tied to outcome, or model ranking is returned early.
Audit data never update the search. A failed audit packet is a technical
failure; it is not replaced.

## AJILE12 is deferred

| Item | Fixed reference |
| --- | --- |
| DANDI version | `0.220127.0436` |
| DOI | `10.48324/dandi.000055/0.220127.0436` |
| Descriptor | `10.1038/s41597-022-01280-y` |
| Official repository | `https://github.com/BruntonUWBio/ajile12-nwb-data` |
| Assets | 55 day-level NWB files |
| Bytes | 845,869,698,341 |
| Participants | 12 |
| License | CC BY 4.0 |

Only the published version may be referenced; the moving DANDI `draft` is
prohibited. The public `ElectricalSeries` was processed from 1 kHz to 500 Hz
with DC/median removal, two-sided-looking discontinuity handling, band/notch
filtering, downsampling, and common-median reference. Public pose-derived
events additionally use future movement confirmation, speed selection, and
manual review; raw video is not public.

AJILE12 cannot support the primary strict-causal claim unless an authorized
future handoff supplies prefilter 1-kHz ECoG, exact filter/padding/resampling
and discontinuity code, a certified zero future-to-past impulse response, and
reconstructable continuous pose onset lineage. Until then it is restricted to
exploratory offline association and is not an audit fallback.

Official references:

- <https://doi.org/10.48324/dandi.000055/0.220127.0436>
- <https://doi.org/10.1038/s41597-022-01280-y>
- <https://github.com/BruntonUWBio/ajile12-nwb-data>

## Acquisition and readiness gates

Training cannot start until all applicable checks pass:

- every official source object matches its pinned manifest;
- all 108 WAY continuous series and 12 `AllLifts` files are present;
- EEG, EMG, kinematic, force, cue, and event clocks reconcile within a frozen
  bound;
- all 49 self-paced raw BrainVision sessions, triggers, accelerometry, and BIDS
  identities are complete;
- the one-sided onset detector passes injected-signal timing recovery and
  future-to-past impulse tests;
- the common 29-channel contract and both sampling lineages are executable;
- public/header-only structure is compatible with the frozen evaluator; exact
  C32 packets and endpoint support are constructed only inside the one-shot
  evaluator after lock, and insufficient support is a nonreplaceable technical
  failure rather than a pre-lock outcome-bearing readiness query;
- every audit and campaign-sealed evaluation packet is evaluator-only; and
- the exposure ledger confirms that no held-out event-level signal, QC, or
  score entered model design.

## Current local inventory and readiness

The legacy-worktree quarantine represented by `asset_ep19_public_sources` now
contains 12 official WAY participant archives and the official Freewill
archive: 13 archives totaling 23,930,807,095 bytes. All
provider byte sizes and MD5 values passed, and all 13 local SHA-256 values were
independently rechecked. The exact `Safe1` commit source and license are also
acquisition-verified. These archives remain unextracted, outside episode
`inputs/`, and unavailable to candidate workers.

All 55 AJILE12 archives (845,869,698,341 expected bytes) remain in their
separate legacy deferred-extension quarantine as `asset_ep19_ajile12`. Its move
is `staged_not_moved`, while the planned destination is
`absent_planned_target`. The acquisition has an `ACQUISITION_COMPLETE_UTC`
marker, an asset manifest, and a verified count/byte receipt for all 55 assets
and 845,869,698,341 payload bytes. Acquisition completion is not a role-filtered
EP19 handoff or readiness receipt, and AJILE12 remains ineligible for the
primary claim regardless.

No signal or event-level audit outcome was read during acquisition. Public
metadata, source code, archive inventory, and aggregate paper information are
design exposure and must be entered in the future ledger. The seven
permission-separated handoffs above remain absent.

The current readiness label is
`source_acquired_in_quarantine_role_filtered_handoffs_absent_launch_blocked`.
