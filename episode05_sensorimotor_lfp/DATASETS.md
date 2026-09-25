# EP05 adaptive data contract

## Status and source identity

This contract does not claim new data acquisition.
The intended source is the public Dryad release for Gallego-Carracedo et al.
(2022), DOI `10.5061/dryad.xd2547dkt`, dataset ID 93309, metadata version 7 /
internal ID 194056, with version-5 file payload / internal ID 192175.
Historical inventories report 24 MATLAB session files plus a 1,020-byte README
(25 objects; 9,571,571,204 bytes). Every byte, guide, and session identity must
be reverified from a provider-bound manifest.

The publication analysis reference is BeNeuroLab commit
`cbda8e2e6106f5eb5ff98e18a689c595179ac5db`; the best-recoverable TrialData
compatibility tree reported during prior compatibility work is
`d5e5eeb1af592cf88c03599df7433878f9d11bbe`. Neither identifier proves a
turnkey environment or author-confirmed dependency pin.

## Known recording inventory

The original contracts report:

| Block | Sessions | Regions | Adaptive role |
| --- | ---: | --- | --- |
| Mihili | 6 | simultaneous M1 + PMd | eligible M1 development/selection after QC |
| Chewie-L | 6 | simultaneous M1 + PMd | eligible M1 development/selection after QC |
| Chewie-R | 4 | M1, separate implant | development/sensitivity; nested within Chewie |
| Han | 5 | area 2 | outside primary M1 search |
| Lando | 3 | area 2 | outside primary M1 search |

These counts are planning facts, not a final eligibility manifest. Chewie-L
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

## Required immutable input pack

Create a new read-only pack from independently verified source bytes; do not
read mutable historical output trees. It must include:

1. provider and local path/size/SHA-256 manifest for all 25 release objects;
2. animal, implant, session, region, array, trial, direction, event, channel,
   unit, and band-guide inventory with source-to-derived trial IDs;
3. authenticated `bin_size`, `tgtDir`, trial-result status, `idx_tgtOnTime`,
   `idx_goCueTime`, `idx_movement_on`, `idx_endTime`, velocity, spike, LFP, and
   guide fields;
4. proof that simultaneous arrays share trials, events, time bases, and
   behavior where such comparisons are used;
5. publication code/dependency identities, licenses, clean compatibility
   implementation, synthetic fixtures, and expected outputs; and
6. an explicit capsule manifest for any historical artifact consulted, with
   its prior-outcome exposure recorded before search.

The historical source-provisioning receipts may help reconstruct the pack, but
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
- bind hashes and structural eligibility while neural values/comparison scores
  remain inaccessible to the search worker;
- prohibit audit-session metadata from changing features, windows, ranks,
  models, exclusions, thresholds, patience, or the winner;
- release the sealed sessions only after a signed configuration-lock receipt;
  and
- run the locked policy once, withholding partial session scores until the
  complete audit succeeds or fails.

An exact retry is allowed only for a proven infrastructure failure with no
released score and unchanged hashes. Once any score is revealed, no alternate
pipeline may be audited.

## Data needed to explain a positive kinematic result

The [paper plan](outputs/paper_plan.md) proposes a later explanatory round. It
does not expand access during adaptive search and does not turn the exposed
Dryad sessions into fresh evidence.

| Source | Proposed role | Condition before use |
| --- | --- | --- |
| Exposed M1 development sessions | Test whether the locked LFP representation predicts training-derived spike-population latents and the same held-out velocity residuals | Start only after the primary development result is fixed; freeze the latent construction, residual readouts, models, margins, candidate, and comparison anchors first |
| Simultaneous spike and LFP fields from the immutable pack | Separate low-frequency/LMP signal from high-frequency or same-electrode spike bleed-through | Preserve physical-electrode and unit joins; fit every latent and transform inside training data; retain the prespecified same-electrode/unit-intersection sensitivity |
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

- No adaptive read-only source pack or session-fold manifest is bound here.
- Provider/local hashes, guide semantics, event conventions, complete-trial
  joins, session eligibility, and licensing must be reverified.
- The publication dependency boundary and reproduction tolerances require a
  frozen, outcome-independent audit.
- Historical outcome exposure must be captured in the episode provenance.
- No fresh compatible whole-session audit dataset has been acquired or sealed.
- The adaptive evaluator has not been qualified against the frozen policy;
  this blocks scored search and audit opening, not explicit task startup.

## Storage and compute boundary

Large source data stay outside Git and are exposed read-only. Durable code,
manifests, ledgers, predictions, receipts, and reports belong under the eventual
episode workspace. Transient arrays belong in a dedicated
`$SCRATCH/br_autoresearch/episode05_sensorimotor_lfp/` directory. Expensive
computation uses Slurm; neither
prior run artifacts are writable episode storage.
