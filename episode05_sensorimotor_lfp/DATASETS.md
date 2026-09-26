# EP05 adaptive data contract

## Status and source identity

This contract does not claim new data acquisition.
The intended source is the public Dryad release for Gallego-Carracedo et al.
(2022), DOI `10.5061/dryad.xd2547dkt`, dataset ID 93309, metadata version 7 /
internal ID 194056, with version-5 file payload / internal ID 192175.
Historical inventories report 24 MATLAB session files plus a 1,020-byte README
(25 objects; 9,571,571,204 bytes). Confirm the provider release identifiers and
this object inventory, then verify guide semantics and session identities from
the files actually used.

The publication analysis reference is BeNeuroLab commit
`cbda8e2e6106f5eb5ff98e18a689c595179ac5db`; the best-recoverable TrialData
compatibility tree reported during prior compatibility work is
`d5e5eeb1af592cf88c03599df7433878f9d11bbe`. Neither identifier proves a
turnkey environment or author-confirmed dependency pin.

## Synthetic prelaunch status

The 2026-09-26 prelaunch used generated state-machine fixtures only. It did not
open or list any object in the Dryad release, inspect LFP or kinematic values,
run publication code, or identify a prospective audit source. Its durable
artifacts are under `outputs/executor_qualification/`.

This synthetic no-access record does not erase the older exposure. The complete
reused Dryad release remains conservatively classified as outcome-exposed.
Real development still requires a read-only source location, frozen group and
session folds, qualified scientific evaluator, reproduction gate, cost profile,
and sealed future audit described below. Exact unresolved items are tracked in
`outputs/executor/REAL_DATA_GATES.md`.

## Known recording inventory

The original contracts report:

| Block | Sessions | Regions | Adaptive role |
| --- | ---: | --- | --- |
| Mihili | 6 | simultaneous M1 + PMd | eligible M1 development/selection after QC |
| Chewie-L | 6 | simultaneous M1 + PMd | eligible M1 development/selection after QC |
| Chewie-R | 4 | M1, separate implant | development/sensitivity; nested within Chewie |
| Han | 5 | area 2 | outside primary M1 search |
| Lando | 3 | area 2 | outside primary M1 search |

These counts are planning facts, not a final eligibility table. Chewie-L
and Chewie-R are one animal. Sessions, trials, bins, directions, channels, and
folds are repeated observations, not additional animals.

The release contains post-processed LFP feature matrices rather than raw
voltage. Raw phase, new filtering, PAC, coherence, traveling-wave, and
re-referencing analyses are unavailable and out of scope.

## Exposure classification

The historical EP05 work accessed candidate-discriminating neural outcomes and
produced partial benchmark/scientific artifacts before local technical
failure. Therefore the **entire reused Dryad release is conservatively treated
as exposed** for adaptive. New folds, code, or a clean directory do not make
its sessions fresh.

| Role | Source | Permitted use | Freshness |
| --- | --- | --- | --- |
| infrastructure reproduction | eligible release sessions plus pinned code | fixed fidelity gate; never search or trial count | exposed technical/scientific prior |
| adaptive development/selection | all structurally eligible M1 sessions | nested whole-session policy search and falsification | exposed exploratory evidence |
| sensitivity only | PMd, area 2, Chewie implant contrasts where prespecified | bounded specificity/context checks, not primary rescue | exposed |
| prospective audit | future compatible whole sessions sealed before outcome access | one locked run after configuration lock | absent |

## Required lean input qualification

Use a read-only copy of the identified provider release; do not read mutable
historical output trees. Before real analysis, record only what is needed to
establish the scientific source and grouping:

1. provider release identifiers, the local read-only location, and the
   expected 25-object count and total size;
2. animal, implant, session, region, array, trial, direction, event, channel,
   unit, and band-guide mapping with stable source-to-derived trial IDs;
3. confirmed `bin_size`, `tgtDir`, trial-result status, `idx_tgtOnTime`,
   `idx_goCueTime`, `idx_movement_on`, `idx_endTime`, velocity, spike, LFP, and
   guide fields;
4. a direct check that simultaneous arrays share trials, events, time bases, and
   behavior where such comparisons are used;
5. the publication code revision, licenses, known compatibility limitations,
   targeted synthetic fixtures, and expected reproduction outputs; and
6. a list of any historical artifacts consulted, with their prior-outcome
   exposure recorded before search.

Historical source-provisioning records may help locate the release, but
their presence is not permission to modify or silently import old results.

## Development and selection roles

After behavior/structure-only eligibility checks, freeze animal/implant/session
IDs and nested whole-session folds. A candidate's global policy is chosen only
from training sessions. Within every evaluated session:

- keep complete trials intact;
- use leave-one-direction-out as the primary split;
- fit scaling, channel transforms, temporal/direction baseline, feature
  transforms, regularization, and coefficients on permitted training trials;
- use identical evaluation trials for candidate and comparator; and
- collapse bins, X/Y, trials, directions, and repeated seeds to one session
  estimate before animal-balanced aggregation.

For clarity, leave-one-direction-out means that no trial from the evaluation
direction can be used to build its behavioral comparator. The comparator is a
cyclic direction-by-elapsed-time function fit on the seven permitted
directions and evaluated at the held-out angle. The circular basis,
interpolation/extrapolation behavior, support rule, and code identity must be
frozen before search; it is not an empirical mean of withheld trials.

The exposed corpus supplies internal development/selection only. A frozen
outer-session score is useful for honest generalization estimation but not
fresh confirmation.

## Prospective audit firewall

No audit source is currently identified or acquired. If compatible future
sessions become available:

- freeze their animal/task/hardware relationship and claim boundary before
  seeing neural outcomes;
- record provider/source version, stable animal/session IDs, and structural
  eligibility while neural values/comparison scores remain inaccessible to
  the search worker;
- prohibit audit-session metadata from changing features, windows, ranks,
  models, exclusions, thresholds, patience, or the winner;
- release the sealed sessions only after the dated, write-once configuration
  lock is recorded;
  and
- run the locked policy once, withholding partial session scores until the
  complete audit succeeds or fails.

An exact retry is allowed only for a proven infrastructure failure with no
released score and the same locked source sessions, policy, and code revision.
Once any score is revealed, no alternate pipeline may be audited.

## Data needed to explain a positive kinematic result

The [paper plan](outputs/paper_plan.md) proposes a later explanatory round. It
does not expand access during adaptive search and does not turn the exposed
Dryad sessions into fresh evidence.

| Source | Proposed role | Condition before use |
| --- | --- | --- |
| Exposed M1 development sessions | Test whether the locked LFP representation predicts training-derived spike-population latents and the same held-out velocity residuals | Start only after the primary development result is fixed; freeze the latent construction, residual readouts, models, margins, candidate, and comparison anchors first |
| Simultaneous spike and LFP fields from the qualified read-only source | Separate low-frequency/LMP signal from high-frequency or same-electrode spike bleed-through | Preserve physical-electrode and unit joins; fit every latent and transform inside training data; retain the prespecified same-electrode/unit-intersection sensitivity |
| Future compatible whole sessions | Test the already locked LFP-to-kinematic prediction once | Keep outcomes sealed until the primary configuration lock; do not use follow-up outcomes to select the primary policy |
| Future compatible sessions from a new animal | Test whether the explanatory relationship extends beyond Mihili and Chewie | Freeze a new source and claim boundary before neural access; do not count additional Chewie implants or sessions as another animal |

The primary residual target remains X/Y velocity after the training-only
cyclic direction-by-time baseline described above. The explanatory readouts decompose that same
residual into along-path speed and perpendicular/curvature components using a
formula frozen from training behavior; they are not alternate outcomes chosen
after seeing which looks strongest. Speed and acceleration remain diagnostic
under the primary contract.

To distinguish trial-paired information from a systematic correction to the
seven-direction interpolation, preserve complete evaluation-trial predictions
and stable trial/direction IDs. With all fits fixed, permute whole predicted
trajectories only among trials of the same held-out direction. Report, per
session, observed `delta_R2` minus the frozen center of that null distribution.
Its permutation count, center, meaningful margin, aggregation, and invalid-cell
rule must be fixed before candidate-discriminating use. This readout gates only
the stronger “trial-specific” interpretation; it does not replace the primary
endpoint or terminal mapping.

Spike-population latents are simultaneous measurements from the same trials,
not independent biological replication. They may explain a candidate only if
their rank, smoothing, temporal support, normalization, and alignment are fit
without evaluation-trial information. An LFP feature may not be called
low-frequency evidence if its construction or channel selection depends on
high-frequency power, unit quality, or test outcomes.

The future-session audit stays the sole one-shot audit in the current primary
round. Any later new-animal explanatory test needs a separate frozen contract,
exposure record, budget, and opening rule. Reusing a current-release
holdout, changing folds, or hiding an already exposed score does not create
independent confirmation.

## Missing assets and scientific gates

- No adaptive read-only source location or frozen group/session fold table is
  recorded here.
- Provider release identity, guide semantics, event conventions,
  complete-trial joins, session eligibility, and licensing must be verified.
- The publication dependency boundary and reproduction tolerances require a
  frozen, outcome-independent audit.
- Historical outcome exposure must be stated in the episode record.
- No fresh compatible whole-session audit dataset has been acquired or sealed.
- The adaptive evaluator has not been qualified against the frozen policy;
  this blocks scored search and audit opening, not explicit task startup.

## Storage and compute boundary

Large source data stay outside Git and are exposed read-only. Durable code,
group/split tables, trial records, predictions, and reports belong under the
eventual episode workspace. Transient arrays belong in a dedicated
`$SCRATCH/br_autoresearch/episode05_sensorimotor_lfp/` directory. Expensive
computation uses Slurm; prior run artifacts are not writable episode storage.
