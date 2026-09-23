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
compatibility tree reported during prior readiness work is
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

The exposed corpus supplies internal development/selection only. A frozen
outer-session score is useful for honest generalization estimation but not
fresh confirmation.

## Prospective audit firewall

No audit source is currently identified or acquired. If compatible future
sessions become available:

- register their animal/task/hardware relationship and claim boundary before
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

## Missing assets and launch blockers

- No adaptive read-only source pack or session-fold manifest is bound here.
- Provider/local hashes, guide semantics, event conventions, complete-trial
  joins, session eligibility, and licensing must be reverified.
- The publication dependency boundary and reproduction tolerances require a
  frozen, outcome-independent audit.
- Historical outcome exposure must be captured in the episode provenance.
- No fresh compatible whole-session audit dataset has been acquired or sealed.
- The adaptive evaluator and policy are unregistered; canonical launch is
  blocked.

## Storage and compute boundary

Large source data stay outside Git and are exposed read-only. Durable code,
manifests, ledgers, predictions, receipts, and reports belong under the eventual
episode workspace. Transient arrays belong in a dedicated
`$SCRATCH/autoresearch/` directory. Expensive computation uses Slurm; neither
prior run artifacts are writable episode storage.
