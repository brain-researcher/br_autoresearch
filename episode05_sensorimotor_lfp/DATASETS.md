# EP05 data plan

## Source and current status

The analyzed source is the previously exposed public Dryad release for
Gallego-Carracedo et al. (2022), DOI `10.5061/dryad.xd2547dkt`, read directly at
`$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source`.
The completed [source-support record](outputs/runs/source_support/attempt002/source_support.json)
observed the expected 24 MATLAB sessions plus README (9,571,571,204 bytes),
stable trial identities and complete support for sixteen M1 sessions. This
is existing-data qualification, not new acquisition.

As reconciled October 6, the original 45-comparison development search and the
separately adopted calibration/decomposition follow-ups are complete. No
candidate met both practical margins. The [completed evidence](outputs/paper_plan.md#completed-evidence--october-6-2026)
and [primary closeout](outputs/primary_development_report.md) supersede older
not-yet-analyzed statements; future-session confirmation and the conditional
spike-population follow-up remain unopened.

The reference implementation is the public BeNeuroLab analysis code associated
with the paper. Its published examples are a scientific reproduction target,
not proof that the software will run unchanged in the current environment.

## Known recording inventory

### Public online resources — checked October 6, 2026

- [Official Dryad data record](https://datadryad.org/dataset/doi:10.5061/dryad.xd2547dkt): public session MAT files and README, displayed as a 9.57-GB August 16, 2022 file release with an August 29 update. Usage notes describe eight-target delayed reaching, manipulandum hand-position measurements and extracellular recordings; they link the source paper and author code. These are the existing exposed recordings already bound locally, not new sessions or animals.
- [Author code README](https://raw.githubusercontent.com/BeNeuroLab/Relationship_between_latent_dynamics_and_LFP/master/README.md): identifies the required TrialData repository. This makes the external implementation dependency explicit; EP05 retains its already pinned analysis/compatibility revisions rather than installing current heads.
- [Author `fCCA.m`](https://raw.githubusercontent.com/BeNeuroLab/Relationship_between_latent_dynamics_and_LFP/master/MainAnalysis/fCCA.m): publicly specifies guide-based shared spike/LFP electrode selection, spike square-root transformation, 0.05-width smoothing, PCA and per-band CCA. It is a reference for the original population/LFP analysis, not EP05's held-out velocity-residual estimator or evidence that the conditional population follow-up has run.

The inspected release shows no authenticated corrected March 6 count export
or fresh-session package. That bounded observation is not proof that neither
exists elsewhere. Its known count defect does not negate EP05's qualified
velocity endpoint, and downstream author transformations do not authorize an
inverse repair. No MAT bytes or numerical results were opened, no dependency
was installed, and the completed development stop remains unchanged.

The original contracts report:

| Block | Sessions | Regions | Planned use |
| --- | ---: | --- | --- |
| Mihili | 6 | simultaneous M1 + PMd | eligible M1 development/selection after QC |
| Chewie-L | 6 | simultaneous M1 + PMd | eligible M1 development/selection after QC |
| Chewie-R | 4 | M1, separate implant | development/sensitivity; nested within Chewie |
| Han | 5 | area 2 | outside primary M1 search |
| Lando | 3 | area 2 | outside primary M1 search |

The primary analysis retained all sixteen M1 sessions: six Mihili and ten
Chewie. Chewie-L and Chewie-R are one animal. The
[qualified session table](outputs/runs/source_support/attempt002/session_table.csv)
and [trial identities](outputs/runs/source_support/attempt002/trial_identity.csv)
record source support. Sessions, trials, bins, directions, channels and folds
remain repeated observations, not additional animals. The source-support
checks distinguish the paper −4:+14 and example 0:+15 native-tick windows;
neither is the primary fifteen-bin 0–420-ms comparison window.

Mihili 2014-03-06 remains supported for EP05's LFP/velocity endpoint. Its
noninteger spike representation matters for the separate spike benchmark or
conditional population follow-up, but EP07's native-count requirement does
not remove this session from EP05. No rounding or inverse transformation was
adopted.

The release contains post-processed LFP feature matrices rather than raw
voltage. Raw phase, new filtering, PAC, coherence, traveling-wave, and
re-referencing analyses are unavailable and out of scope.

## Exposure classification

The historical EP05 work accessed neural outcomes used to compare models and
produced partial benchmark/scientific artifacts before local technical
failure. Therefore the **entire reused Dryad release is conservatively treated
as exposed** for adaptive. New folds, code, or a clean directory do not make
its sessions fresh.

| Role | Source | Permitted use | Freshness |
| --- | --- | --- | --- |
| infrastructure reproduction | eligible release sessions plus pinned code | fixed fidelity gate; never search or trial count | exposed technical/scientific prior |
| adaptive development/selection | all structurally eligible M1 sessions | nested whole-session policy search and falsification | exposed exploratory evidence |
| sensitivity only | PMd, area 2, Chewie implant contrasts where prespecified | bounded specificity/context checks, not primary rescue | exposed |
| future-session test | future compatible whole sessions kept unseen during development | one evaluation after the analysis is final | absent |

## What must be checked before analysis

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
   small known-answer tests, and expected reproduction outputs; and
6. a list of any historical artifacts consulted, with their prior-outcome
   exposure recorded before search.

Earlier source records may help locate the release, but
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
interpolation/extrapolation behavior, support rule, and analysis definition
must be chosen before search; this is not an empirical mean of withheld trials.

The exposed corpus supplies internal development/selection only. A fixed
outer-session score is useful for honest generalization estimation but not
fresh confirmation.

## Future-session test

No future-test source is currently identified or acquired. If compatible future
sessions become available:

- freeze their animal/task/hardware relationship and claim boundary before
  seeing neural outcomes;
- record provider/source version, stable animal/session IDs, and structural
  eligibility while neural values/comparison scores remain inaccessible to
  the search worker;
- prohibit future-session metadata from changing features, windows, ranks,
  models, exclusions, thresholds, patience, or the winner;
- release the sealed sessions only after the dated final analysis plan is
  recorded;
  and
- run the final analysis once, withholding partial session scores until the
  complete evaluation succeeds or fails.

An exact retry is allowed only for a proven infrastructure failure with no
released score and the same source sessions, final analysis, and code revision.
Once any score is revealed, no alternate pipeline may be tried on those
sessions.

## Data needed to explain a positive kinematic result

The [paper plan](outputs/paper_plan.md) proposes a later explanatory round. It
does not expand access during adaptive search and does not turn the exposed
Dryad sessions into fresh evidence.

| Source | Proposed role | Condition before use |
| --- | --- | --- |
| Exposed M1 development sessions | Test whether the final LFP representation predicts training-derived spike-population latents and the same held-out velocity residuals | Start only after the primary development result is fixed; choose the latent construction, residual readouts, models, margins, candidate, and comparison anchors first |
| Simultaneous spike and LFP fields from the verified read-only source | Separate low-frequency/LMP signal from high-frequency or same-electrode spike bleed-through | Preserve physical-electrode and unit joins; fit every latent and transform inside training data; retain the prespecified same-electrode/unit-intersection sensitivity |
| Future compatible whole sessions | Test the already chosen LFP-to-kinematic prediction once | Keep outcomes sealed until the primary analysis is final; do not use follow-up outcomes to select the primary model |
| Future compatible sessions from a new animal | Test whether the explanatory relationship extends beyond Mihili and Chewie | Freeze a new source and claim boundary before neural access; do not count additional Chewie implants or sessions as another animal |

The primary residual target remains X/Y velocity after the training-only
cyclic direction-by-time baseline described above. The explanatory readouts decompose that same
residual into along-path speed and perpendicular/curvature components using a
formula chosen from training behavior; they are not alternate outcomes selected
after seeing which looks strongest. Speed and acceleration remain diagnostic
under the primary contract.

To distinguish trial-paired information from a systematic correction to the
seven-direction interpolation, preserve complete evaluation-trial predictions
and stable trial/direction IDs. With all fits fixed, permute whole predicted
trajectories only among trials of the same held-out direction. Report, per
session, observed `delta_R2` minus the prespecified center of that null distribution.
Its permutation count, center, meaningful margin, aggregation, and invalid-cell
rule must be fixed before it is used to compare models. This readout is needed
only for the stronger “trial-specific” interpretation; it does not replace the
primary result.

Spike-population latents are simultaneous measurements from the same trials,
not independent biological replication. They may explain a candidate only if
their rank, smoothing, temporal support, normalization, and alignment are fit
without evaluation-trial information. An LFP feature may not be called
low-frequency evidence if its construction or channel selection depends on
high-frequency power, unit quality, or test outcomes.

The future-session evaluation is the only held-out evaluation in the current
primary round. Any later new-animal explanatory test needs its own analysis
plan, exposure record, budget, and access rule. Reusing a current-release
holdout, changing folds, or hiding an already exposed score does not create
independent confirmation.

## Missing assets and scientific gates

Source/indexing/leakage checks, the scoped publication benchmark and the
primary analysis were completed before the recorded search closeout. The
[complete scientific report](outputs/primary_development_report.md) preserves
the sixteen-session support, training/evaluation rules, original count and
patience stop. These are no longer missing implementation or source gates.

No fresh compatible whole-session dataset has been acquired or sealed.
The primary practical-margin failure leaves no eligible candidate for that
test or the conditional spike-population explanation. The later fixed-HW
calibration remains exposed development and also fails both margins; it does
not reopen the original search or adopt the independent-penalty proposal.

## Storage and compute boundary

Large source data stay outside Git and are exposed read-only. Durable code,
group/split tables, trial records, predictions, and reports belong under the
eventual episode workspace. Transient arrays belong in a dedicated
`$SCRATCH/br_autoresearch/episode05_sensorimotor_lfp/` directory. Expensive
computation uses Slurm; prior run artifacts are not writable episode storage.
