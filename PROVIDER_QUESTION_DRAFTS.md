# Provider question drafts — October 16, 2026

**October20 navigation note:** this is a historical draft collection. Several
source questions below have since been resolved; their original wording is not
a current blocker or an active request. Use each episode's dated
`outputs/README.md` and execution log for current facts before reusing a draft.

Drafts for the scientist to review, edit and send. **Nothing has been sent.**
Agents do not contact providers ([AGENTS.md](AGENTS.md)); the scientist decides
whether and when to send each request.

**Optional fallback.** Under the October 16 rule in AGENTS.md, agents resolve
these facts themselves from papers, code and permitted data, or adopt stated
assumptions with sensitivity analyses. Send a request only where that route
leaves a fact that matters unresolved.

Each section was drafted from the episode's `GOAL.md`, `DATASETS.md` and
`outputs/README.md`, plus the source documents they link. No contact address
is recorded for any provider, so fill in each recipient from the cited article
or record page. The existing EP02 and EP03 drafts were expanded, not replaced.

**For every request**

- Ask only for outcome-free source facts: codebooks, timing, units, labels and
  provenance. Never request or cite outcome values, scores, or summaries
  derived from them.
- Do not describe internal data roles (development, held-out, audit,
  reserved), participant splits or episode results.
- When a reply arrives, screen it for outcome-derived values, then record it
  in the episode's log before anyone acts on it. A reply alone does not reopen
  an episode or change its question, data roles or claims.

| Episode | Recipient (from the records) | Asks for | Drafts |
| --- | --- | --- | --- |
| [EP02](#ep02) | Data authors of Coddington, Lindo & Dudman (2023) | Trial codes, timing axis, trigger and cap logic, randomization roster | 1 |
| [EP03](#ep03) | Producer of the OpenNeuro FitLins analyses; holder of the NeuroEffect collection | Fitted inputs per map; group provenance | 2 |
| [EP09](#ep09) | SEU-A1876 providers | Layer and class annotation provenance | 1 |
| [EP11](#ep11) | Gao et al. whole-cortex projectome providers | Sample-to-mouse identity; reconstruction metadata | 1 |
| [EP16](#ep16) | Authors of the BrainGate array-performance release | Target size, layout, transitions, coordinate export | 1 |
| [EP19](#ep19) | WAY-EEG-GAL authors; self-paced reaching dataset authors | Clock origin and calibration; run bindings and filters | 2 |

---

## EP02

`episode02_dopamine_learning_rate_causality`: no empirical fit can start until
the trial, session, lick-state and timing definitions are authenticated.

**Recipient.** Corresponding author of Coddington, Lindo & Dudman (2023),
*Nature*, doi:10.1038/s41586-022-05614-z. Dataset: Janelia Figshare 21816054
v1 (doi:10.25378/janelia.21816054.v1), file `seshMerge.mat`. Code:
github.com/dudmanj/RNN_learnDA and github.com/DudLab/TONIC. No contact is
recorded. This expands the draft in
[source_semantics_design.md](episode02_dopamine_learning_rate_causality/outputs/source_semantics_design.md:337).

> **Subject:** Trial codes and timing in seshMerge.mat (Janelia Figshare 21816054)
>
> Dear [name],
>
> I am reanalysing the dataset released with "Mesolimbic dopamine adapts the
> rate of learning from action" (Nature, 2023; Janelia Figshare 21816054 v1,
> `seshMerge.mat`). The release has no codebook, so I would like to confirm
> some definitions before using it. Methods text or code excerpts that
> identify these fields would be enough; I am not requesting any additional
> neural or behavioural data.
>
> 1. `trialID`: what does each value mean? Which codes mark cued rewarded
>    trials, uncued rewards, omissions and aborted or invalid trials, and which
>    trials count toward a session's total?
> 2. `seshID`: how do the values map to training days 1–8, including
>    pretraining, restarted or repeated sessions, and lost blocks? Is there a
>    list of lost blocks for each released record?
> 3. `lickState`: is this exactly the online binary lick state in the 750-ms
>    window before reward? What are its coding, its boundary convention
>    relative to reward time, and its value when lick acquisition was missing?
> 4. The 701-sample streams (`lick`, `nac`, `body` and the other 100-Hz
>    signals): what are the sample times (or start, step and end), which event
>    is time zero, where do cue onset and reward delivery fall, and were the
>    streams resampled?
> 5. Closed-loop stimulation: what was the exact online logic of the 750-ms
>    lick trigger in the `stimLick-` and `stimLick+` protocols? How was missing
>    acquisition handled, and how did the 50% session cap in `stimLick+` work
>    (its counter, denominator, reset and ordering)?
> 6. Design: for each released record (index 1–24 of `seshMerge`), which
>    randomization cohort or block did the mouse belong to, which arm was it
>    assigned within that cohort, and with what probabilities? For the four
>    mice removed after collection, what were their arms, cohorts and removal
>    timing?
> 7. Optional: are the original activation/learning callbacks and
>    initialization file for `dlRNN_engine.m` and `dlRNN_train_learnDA.m`
>    available, and what license applies to the TONIC repository?
>
> Thank you for making these data available.
>
> Best regards,
> [name]

**What each answer changes**

1. Fixes the eligible cued-reward trials and the per-epoch trial counts that
   the primary endpoint requires.
2. Defines the session-1 and session-8 epochs and the missing-block rule (no
   interpolation).
3. Authenticates the raw source of the primary binary endpoint.
4. Unlocks the continuous-lick replication, which is a terminal gate, and the
   NAc post-cue and post-reward windows.
5. Supplies the online trigger that every mechanism model shares; until then
   the cap is only a simulation scenario.
6. Allows a block-conditioned intention-to-treat estimand. Without it, only the
   noncausal 462-label sensitivity remains.
7. Affects only the report-only pinned-code sensitivity.

**Not for the provider:** the four raw margins are scientist decisions, not
missing data. **Leave out:** the record-to-role map, realized stimulation
counts, per-record diagnostics, neural or per-animal values, and the
462-label calculation. Refer to records only by their one-based index.

---

## EP03

`episode03_condition_semantics_effect_maps`: the actual fitted contributors
and the remaining contrast directions block any map evaluation. Configured N
is not fitted N.

### EP03a: FitLins group maps

**Recipient.** Producer of `$OAK/data/OpenNeuro_analyses/openneuro_fitlins`,
built with github.com/demidenm/openneuro_glmfitlins. The records name no
person; the path is in the group's own Oak space, so the producer is probably
reachable internally. Source data: OpenNeuro ds000001 v1.0.0
(doi:10.18112/openneuro.ds000001.v1.0.0). This expands the draft in
[paper_plan.md](episode03_condition_semantics_effect_maps/outputs/paper_plan.md:53).

> **Subject:** Fitted inputs for the ds000001 FitLins group maps
>
> Dear [name],
>
> I am using the FitLins group maps for OpenNeuro ds000001 (balloon analog
> risk task) in `$OAK/data/OpenNeuro_analyses/openneuro_fitlins`, built with
> openneuro_glmfitlins. The model configuration lists subjects 01–16, but I
> could not find a record of which inputs actually entered each Dataset-level
> fit. Records of what was executed are enough. I do not need a bitwise rerun,
> the exact historical container, or any map values.
>
> 1. For each of the 13 Dataset-node maps
>    (`node-dataLevel/contrast-<name>_stat-z_statmap.nii.gz` for allpumps,
>    cashpara, controlpara, explodepara, explodevcash, explodevcontrol,
>    parapumpsvcontrol, pumps, pumpscashvcontrol, pumpsexplodevcontrol,
>    pumpspara, pumpsvcontrol and rt), which subject IDs and subject-level
>    effect maps entered the fit? Were any dropped under `drop_missing=true`,
>    and is there an exclusion or missingness record?
> 2. Which checkout and model spec produced these maps? Do they match the
>    current public `main` files, or the local copy at commit
>    `e624fec2765a7869a4f482da701d85b82bc2f4b7`? Which FitLins build ran: the
>    declared `0.11.0.post0.dev94`, or the `dev42` build pinned in that
>    checkout's `uv.lock`?
> 3. How were the event columns `pumps_demean`, `control_pumps_demean`,
>    `cash_demean` and `explode_demean` constructed (centering scope and
>    scaling)?
>
> Thank you,
> [name]

**What each answer changes**

1. Fills each map's actual sample size and lineage. If contributors cannot be
   recovered, the analysis ends as a technical failure, not a negative result.
2. Binds the contrast weights and the five resolved directions to what actually
   ran.
3. Lets the parametric maps (`pumpspara`, `controlpara`, `cashpara`,
   `explodepara`, and the scale of `parapumpsvcontrol`) receive a canonical
   direction.

**Not for the provider:** canonical orientation for `explodevcash`, `pumps`,
`allpumps` and `rt` follows EP03's own rule.

### EP03b: NeuroEffect collection

**Recipient.** Not recorded. The records give no DOI, version or holder, and the
BR resolver returned `dataset_not_found`. The staged copy sits under your own
steward-acquisition directory, so this may be an internal question for
whoever built the collection.

> **Subject:** Provenance of the NeuroEffect spatial collection (51 dataset groups)
>
> Dear [name],
>
> I am using the NeuroEffect spatial collection staged on 2026-09-17 (producer
> tag `neuroeffect.data.preparation.source_binding_repair/v1`). To treat each
> dataset group as an independent unit, I need its provenance. No map values
> or results are needed.
>
> 1. What is the authoritative list of the 51 dataset groups, with OpenNeuro
>    accession and snapshot, participant and derivative lineage, and license
>    for each?
> 2. How do these 51 groups correspond to the 31 accessions and 46
>    dataset/task routes present in the staged copy?
> 3. Is there a list of the 577 statistical programs, with map paths,
>    statistic type, threshold, sample size and template space?
> 4. How was the repeat subset of 36 datasets and 341 contrasts selected?
> 5. Which NeuroEffect and FitLins commits or containers produced each group?
>
> Thank you,
> [name]

**What these answers change:** readiness step 1 needs the authenticated group
list. The independent dataset group is EP03's unit of analysis, and candidate
search needs authenticated group roles.

**Leave out (both EP03 requests):** map values, PNGs and scores; the fact that
the 51 groups are development-only; the historical summary percentages in
DATASETS.md; upstream `confirmatory_headline_eligible` flags.

---

## EP09

`episode09_local_global_projection`: authentic layer and class labels and
donor support are inadequate for the design. Animal identity is already
confirmed (38 distinct Animal IDs), so it is not asked.

**Recipient.** Providers of SEU-A1876 (Zenodo 13944322). EP09's records cite
the Nature Communications article doi:10.1038/s41467-024-54745-6 and its
Supplementary Data 1. EP11's records attribute the same Zenodo record to a
2025 Jiang et al. paper, so take the contact from the record page. Related
provider pages are at braintell.org/projects/fullmorpho. No contact is
recorded.

> **Subject:** Layer and class annotations in SEU-A1876 (Zenodo 13944322)
>
> Dear [name],
>
> I am using the SEU-A1876 single-neuron reconstructions (Zenodo 13944322,
> `Full_morphometry.xlsx`) together with Supplementary Data 1 of
> doi:10.1038/s41467-024-54745-6. Before analysing cortical IT neurons from
> the Cux2, Plxnd1 and Tlx3 lines, I would like to confirm how some
> annotations were made. I need only metadata and annotation methods. Please
> do not send axon-derived results such as projection classes or target
> summaries.
>
> 1. How was column L, `Cortical Lamination of soma`, assigned: by manual
>    annotation or by CCFv3 soma registration? What does a blank entry mean?
>    Could authentic layer assignments that are not derived from the axon be
>    provided for blank cells in these brains, especially `fMOST Brain ID`
>    18864 and 191812?
> 2. For the Cux2, Plxnd1 and Tlx3 brains (Image IDs 18461, 18452, 18453,
>    18864, 191801, 191803, 17300, 17786, 17787, 17788, 191812 and 17543),
>    were `Transgenic Line (driver;reporter)`, `Main neuronal types targeted`
>    and `Reference of target neurons` set from the labelling design or prior
>    literature, independently of the reconstructed axons? Do you regard every
>    reconstructed cell in these brains as cortical IT, including cells whose
>    recorded layer lies outside the targeted layers?
> 3. Is there any per-cell class label that is not derived from axon
>    reconstructions (genetic, molecular or experimental)? If so, where is it
>    released?
> 4. Which 140 `Morphology Name` entries are newly annotated, and which 1,736
>    refine earlier released neurons (with their earlier IDs)? Does this split
>    match `isManuallyChecked`?
> 5. What does `isManuallyChecked` certify: soma, dendrite, axon or the whole
>    neuron? Is there a per-cell dendrite or soma quality field, or a
>    reconstruction batch, annotator or version field?
> 6. Optional: brain 236174 has 43 cells in SEU-A1876 but 45 in the BrainTell
>    gallery. Which neurons differ? Does each `Animal ID` in Supplementary
>    Data 1 correspond to one mouse and one imaged brain? What voxel size or
>    transform links `Full_morphology_RAW.zip` coordinates to
>    `Full_morphology_CCFv3.zip`?
>
> Thank you,
> [name]

**What each answer changes**

1. Authentic layers could change the source × layer donor pools, which are
   currently at most 2 animals and 72 cells. Any change needs a fresh
   feasibility check. Blank cells may not be assigned layers just to meet the
   donor floor.
2. Decides the gate that requires an outcome-independent cortical IT
   definition.
3. A second, non-axon label source is required for further primary-cohort
   work. Without one, the episode stops or downgrades.
4. Documents release lineage and enables a check restricted to newly
   annotated cells.
5. Defines the allowed quality and batch covariates and the donor quality
   caliper. Axon-derived completeness cannot be used.
6. Closes residual lineage and provenance items. The transform may also be
   checkable internally.

**Leave out:** any axon-derived value (projection class, target detections,
prevalence); the internal design (development/audit split, the shared
EP09–EP11 ledger, the unadopted nine-animal route); anything taken from EP11.

---

## EP11

`episode11_projection_types_vs_gradients`: the 36 brain/sample units cannot
be treated as independent animals without a provider assertion or crosswalk.
The records also show that the original design needs about 51 audit units,
and at most 33 are assignable, so identity alone does not make that design
feasible.

**Recipient.** Providers of the Gao et al. whole-cortex projectome: Zenodo
16909126; *Neuron* doi:10.1016/j.neuron.2025.10.019; Brain Science Data Center
doi:10.12412/BSDC.1747279998.20001; portal mouse.digital-brain.cn. The article
names a resource-request route. No contact is recorded, and the source-lab
names in the inventory are not designated contacts. The sample IDs below come
from EP11's metadata inventory
([whole_cortex_ssptr_inventory.csv](episode11_projection_types_vs_gradients/outputs/stage0/whole_cortex_ssptr_inventory.csv)).

> **Subject:** Animal identity and reconstruction metadata for the whole-cortex projectome (Zenodo 16909126)
>
> Dear [name],
>
> I am analysing the 214 SSp-tr neurons from the whole-cortex single-neuron
> projectome (Neuron, doi:10.1016/j.neuron.2025.10.019; Zenodo 16909126;
> mouse.digital-brain.cn). They come from 36 fMOST samples. I have a few
> questions about sample identity and reconstruction metadata. Please do not
> include projection results (regional projection strengths, axon lengths,
> terminal counts or projection classes) in a reply; I need only identity and
> metadata.
>
> 1. Does each of the 36 samples listed below come from a distinct mouse?
>    Could you provide the mouse identifier for each sample, or state that no
>    two of these samples share a mouse?
> 2. What do `Axon_and_dendrite` and `Axon_only` mean for axon completeness?
>    Are there per-neuron records of completeness or truncation (image-volume
>    edge, damaged or missing sections, signal loss), or of review/QC status?
> 3. Which CCFv3 version, resolution, axis convention, units and registration
>    pipeline apply to the SWC files? Is per-sample registration QC available,
>    and how is each neuron's hemisphere assigned?
> 4. Do any of these samples, mice or reconstructions also appear in SEU-A1876
>    (Zenodo 13944322), the Peng et al. (2021) Nature release, or the F1877
>    resource?
> 5. Are the SWCs served under `neuron_download` on the portal the same
>    version as those in `data_v3.zip` on Zenodo?
> 6. Optional: is a per-sample imaging or registration batch ID available, and
>    what does sex code 0 in the sample metadata mean?
>
> The 36 `fMOST_id` values: 220191, 220192, 220193, 220239, 221066, 221067,
> 221068, 221089, 221224, 221227, 221299, 221365, 221366, 221367, 221416,
> 221428, 221467, 221468, 221471, 221475, 221478, 221479, 221481, 221512,
> 221513, 221535, 221540, 221644, 221726, 221728, 221750, 221752, 221753,
> 230011, 230065, 233794.
>
> Thank you,
> [name]

**What each answer changes**

1. Only an explicit assertion or crosswalk lets the units count as independent
   animals. Shared mice would merge units and require a new support census.
2. Supplies reconstruction evidence for EP11's outcome-blind eligibility rule:
   cell denominators, and truncated coordinates that are marginalized, not set
   to zero.
3. Authenticates the CCF binding and hemisphere assignment before any outcome
   is built.
4. Shared material would make the Gao and SEU-A1876 sources correlated. They
   are never pooled.
5. Fills the required resource and reconstruction-version fields.
6. Batch is an adjustment variable, not a replicate unit; sex must be
   recorded.

**Leave out:** projection classes and any regional projection, axon-length or
terminal-count summaries (a reply containing them would expose outcomes);
the role design; EP09's crosswalk. Do not frame completeness as a required
certificate.

---

## EP16

`episode16_braingate_cross_session_failure_modes`: the remaining question is
size dependence for otherwise contributing schedules, plus layout and
transition semantics. Full controller provenance is not required.

**Recipient.** Authors of the release "Performance of intracortical
microelectrode arrays in people with implanted brain-computer interfaces over
a 20-year period": Dryad doi:10.5061/dryad.x0k6djj1h (version 6), article
doi:10.1038/s41591-026-04530-3, code github.com/nptl-stanford/array-paper. No
contact is recorded. The release is deidentified CC0 data, and the request
needs no identifying information. The day lists come from EP16's
[conditional support report](episode16_braingate_cross_session_failure_modes/outputs/runs/structural_support/20261003-conditional_support01/conditional_support_report.json)
and [log](episode16_braingate_cross_session_failure_modes/outputs/experiment_log.md:779).

> **Subject:** Target size and layout metadata for the BrainGate array-performance release (Dryad x0k6djj1h)
>
> Dear [name],
>
> I am using the decoding sessions in "Performance of intracortical
> microelectrode arrays in people with implanted brain-computer interfaces
> over a 20-year period" (Dryad doi:10.5061/dryad.x0k6djj1h, version 6; code
> github.com/nptl-stanford/array-paper). To compare cursor-task sessions
> fairly, I need some task metadata that the release does not include. I am
> asking only for task annotations keyed by `participant_id`,
> `post_implant_day` and `blocks.block_id`. No identifying information and no
> outcome or performance values are needed.
>
> 1. Target size: for each session and block, was the task schedule
>    size-dependent (for example, a Fitts-type task) or size-independent?
>    Where size mattered or varied, what target size or sizes applied per
>    block or trial, in which units relative to the `target_position` task
>    coordinates, and what was the acquisition rule? Ideally this would cover
>    all 729 sessions; the most important are the T11 and T9 days listed below.
> 2. Layout: did all blocks within a session share one intended set of target
>    centres, so that a block missing some targets reflects finite sampling?
>    Or did layouts change between blocks? Can target locations change within
>    a block or within a trial? This matters most for the T11 days below and
>    T9 days 195, 273, 329, 336, 406, 413, 426, 681, 686, 689, 693 and 707.
> 3. Transitions: which target-to-target transitions were allowed? Does an
>    unobserved transition mean it was not permitted, or only that it was not
>    sampled? This matters especially for T11 days 367, 576, 632, 905 and 1017.
> 4. Coordinates: how were target coordinates generated and exported for each
>    task (units, normalization, quantization or rounding)? On T10 days 67, 112
>    and 130, some stored targets are distinct but differ only at about 1e-16
>    after normalization. Are these intended distinct targets or rounding
>    artifacts?
> 5. Supplementary: which task or controller protocol and version (or a
>    provider-defined schedule class), and which online decoder, applied to
>    each session or block?
>
> Post-implant days referred to above:
>
> - T11: 219, 226, 233, 364, 372, 385, 401, 406, 420, 427, 441, 444, 454, 456,
>   461, 468, 471, 492, 496, 504, 505, 527, 555, 560, 604, 610, 617, 639, 643,
>   646, 658, 672, 675, 689, 692, 706, 709, 715, 744, 749, 751, 758, 842, 868,
>   888, 952, 1001, 1022, 1025, 1057, 1059, 1303, 1549.
> - T9: 71, 77, 83, 105, 106, 132, 133, 170, 188, 190, 195, 219, 273, 279, 282,
>   307, 308, 314, 329, 336, 338, 349, 356, 384, 394, 406, 408, 413, 416, 422,
>   426, 435, 441, 447, 448, 457, 476, 490, 513, 518, 519, 531, 538, 554, 582,
>   595, 597, 603, 631, 650, 651, 658, 672, 681, 686, 689, 693, 707, 720.
>
> Thank you,
> [name]

**What each answer changes**

1. Documented size-dependent schedules without sizes are excluded, and size
   independence needs specific evidence. Every matched graph is then rebuilt
   and the fixed gates are re-applied, which can move support either way.
2. One intended layout resolves the coverage ambiguity; multiple or unstable
   layouts mean exclusion. Blocks are never dropped to restore coverage.
3. Declared permitted transitions turn observed differences into sampling; a
   known difference means exclusion. This is lower priority, because these
   five days lie outside the current design.
4. A provider-grounded precision rule may replace the strict roundoff rule. It
   must not be chosen to recover participants.
5. Supplementary only; this is not an eligibility gate.

**Not for the provider:** raw score units, octant support, the 30-bin window,
source-only scaling and margins, which were fixed internally on October 6.
**Leave out:** participant role allocation, geometry-group IDs, pair counts
and gates, the closeout disposition, and reviewer exposure notes. Never ask
for outcome values, or for the provider's `data/` summaries or notebooks.

---

## EP19

`episode19_causal_motor_forecasting_benchmark`: source timing and calibration
must be authenticated before any EEG fit, and a 2-second clock-origin
discrepancy is unexplained.

### EP19a: WAY-EEG-GAL

**Recipient.** Authors of WAY-EEG-GAL: *Scientific Data*
doi:10.1038/sdata.2014.47; Figshare collection
doi:10.6084/m9.figshare.c.988376.v2; utilities
github.com/luciw/way-eeg-gal-utilities. No contact is recorded.

> **Subject:** Clock origin and calibration in WAY-EEG-GAL (Figshare collection 988376)
>
> Dear [name],
>
> I am using the WAY-EEG-GAL grasp-and-lift recordings (Scientific Data 2014,
> doi:10.1038/sdata.2014.47) with the official utilities. For a forecasting
> analysis that may use only information available before each movement, I
> need to pin down a few timing and calibration details. These questions
> apply to all participants and series.
>
> 1. Clock origin: in `WS_P<n>_S<s>.mat`, what is `ws.win(L).trial_start_time`
>    measured from, relative to the first sample retained in
>    `HS_P<n>_S<s>.mat` after alignment and trimming? Does it mark the window
>    start (2 s before LED onset) or LED onset itself? In `HS_P1_S1`, every
>    `ParticipantLED` rising and falling edge is exactly 2 s earlier than
>    `trial_start_time + LEDon/LEDoff`. Which reading is correct, and is the
>    convention the same for every participant and series? The per-series trim
>    counts and the raw-to-HS/WS export code would settle this if available.
> 2. Position: what are the stored units, scale and coordinate frame of the HS
>    `kin` position channels (`P{axis}{sensor}`, sensors 1–4)? Which FASTRAK
>    settings were used (units, frame, number of sensors, adaptive filter,
>    sync/interface mode), and how were the data resampled to 500 Hz? Is the
>    scale the centimetres implied by the 5 cm/s threshold in
>    `WEEG_FindEvents.m`?
> 3. Force: what are the stored units of the `F{AXIS}{plate}` channels
>    (plates 1–2)? Which ATI sensors and calibration matrices were used, and
>    how were bias subtraction, gain, voltage-to-force conversion and DAQ
>    filtering handled?
> 4. LED cue: what generates the `ParticipantLED` signal and its intermediate
>    values? Does a higher value always mean illuminated, and what is the
>    latency from LED switching to the recorded sample?
> 5. Delays: what are the group delays of the EEG chain (5 kHz down to
>    500 Hz), the EMG chain (4 kHz) and the 500-Hz peripheral channels? Is
>    there residual clock drift after alignment beyond the reported ≤2 ms?
>
> Thank you,
> [name]

**What each answer changes**

1. An authenticated clock binding allows one bounded, development-only clock
   check; without it, no 2-s correction may be applied. It does not by itself
   bound drift or filter delay.
2. Enables the clearance and object checks. Without units, the movement-risk
   state cannot be defined.
3. Enables force-consistency checks in newtons.
4. Cue information enters the model only through the raw LED detector, and
   polarity is confirmed for P1 only.
5. Fills the synchronization-uncertainty bound that the comparison requires.
   A failed timing check invalidates the comparison.

### EP19b: self-paced reaching

**Recipient.** Authors of the self-paced reaching and grasping EEG dataset:
*Scientific Data* doi:10.1038/s41597-025-06039-9; Figshare
doi:10.6084/m9.figshare.28632599.v1 (`Freewill_EEG_Reaching_Grasping.zip`,
file 57518986). No contact is recorded.

> **Subject:** Run files and amplifier settings in the self-paced reaching EEG dataset (Figshare 28632599)
>
> Dear [name],
>
> I am using the self-paced reaching and grasping EEG dataset (Scientific
> Data 2025, doi:10.1038/s41597-025-06039-9; Figshare
> doi:10.6084/m9.figshare.28632599.v1). I have a few questions about file
> bindings and acquisition settings.
>
> 1. In `sub-03/ses-02/eeg/`, the `.vmrk` files for runs 0009, 0010 and 0011
>    each declare `DataFile=sub-03_ses-02_task-reachingandgrasping_run-0008_eeg.eeg`,
>    while their `.vhdr` headers point to their own files, and the four runs
>    have different sample counts and segment start times. Is the `DataFile`
>    line a stale copy, and do these marker files belong to runs 0009–0011?
> 2. The release reports 238 valid runs among 240 raw runs. Which two runs are
>    excluded and why, and where is run validity recorded?
> 3. For the accelerometer channels X, Y and Z (physical channels 66–68,
>    resolution 0.298023 µV), which sensor model and AUX revision were used,
>    and what are the sensitivity, zero-g offset and axis orientation?
> 4. What are the transfer function and group delay of the actiCHamp
>    anti-aliasing filter in the 250-Hz configuration (69-Hz cutoff) and the
>    1,000-Hz configuration (280-Hz cutoff), for EEG and AUX channels?
> 5. What does the TRIG channel (channel 65) carry relative to the sounds
>    played by `PlayStartEnd.m`, and with what latency?
> 6. Can a run contain acquisition interruptions that are not marked by a
>    NewSegment marker? Which time zone do the NewSegment timestamps use, and
>    which EEG reference was used?
>
> Thank you,
> [name]

**What each answer changes**

1. If confirmed, these three runs' marker clocks can be used; otherwise their
   marker starts are withheld. This is not a blanket stop.
2. Eligibility follows the release's valid-run information, which shapes the
   chronological movement order used downstream.
3. Physical units are needed before the detector thresholds are frozen.
4. The filter binding is a required input to the timing bound.
5. Cue availability must be rebuilt causally from the raw TRIG channel.
6. Model state resets at gaps, and false-alarm rates need gap-free stillness.
   The reference question is optional.

**Not for the provider (both EP19 requests):** onset labels (`HandStart` and
`AccStartIndex` serve only as agreement checks), calibration prefixes, the
matched EEG input conditions, null margins and return tolerances. **Leave
out:** held-out series and participants, the calibration-prefix design,
support dispositions, margins, review IDs and exposure notes. Ask about all
participants and series generically. If a reply is specific to held-out
participants or series, route it to the scientist before any episode uses it.
