# Experiment log

## 2026-10-06 Pacific — public model and event-definition sources added

At the scientist's request, incorporated the preceding bounded online review
into the [existing methods handoff](paper_plan.md#october-6-public-methods-sources).
The public producer model/details provide 13 contrast specifications and
16 configured subjects; their actual historical fitted participation remains
unknown. Public `main` is not bound to the local execution, and the five
resolved canonical directions remain unchanged. This records documentary
evidence and narrows the producer request; no payload, score, test, download,
new job, contract amendment or external contact occurred in this edit.

## 2026-10-06 Pacific — producer request and retention proposal clarified

Updated DATASETS and the existing paper plan from the completed first-route
and methods records: repair 46660232 succeeded; actual fitted IDs/N remain
unknown for all 13 maps; five canonical directions are resolved. Added the
precise unsent producer request and an unadopted proposal to retain two named
Scratch metadata files on Oak. The review's December 16 estimate is not a
verified purge date. No source/payload read, transfer, policy change, test,
scheduler action or job occurred; existing failed histories are preserved.

## 2026-10-05 20:32 UTC — BART methods pointer resolves task/control ordering

The bounded methods follow-up read only the BART fields of
`$OAK/data/OpenNeuro_analyses/openneuro_fitlins/analyses/ds000001/code/openneuro_glmfitlins/statsmodel_specs/ds000001/ds000001_basic-details.json`
and the Run-node transformations/model/contrasts in the already identified
`ds000001-balloonanalogrisktask_specs.json` beside it. The task's sole
`cite_links` entry is [Schonberg et al. (2012), DOI 10.3389/fnins.2012.00080](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2012.00080/full).
Its Task and fMRI Model sections identify reward-balloon pumps, cash-outs and
explosions; gray control balloons yield no money and do not explode. Blank
intertrial periods form an implicit baseline. The paper describes pump-number
parametric regressors, with within-trial centering for pumps and across-trial
centering for cash-outs/explosions. These are methods facts, not neural effects.

**Exposure disclosure:** the methods-specific web search returned surrounding
article text including neural results. That text was incidentally exposed;
no result, regional finding, figure or reported activation sign was used for
the ordering below. No raw map, image, event row or neural array was opened.
This remains exposed development evidence, not a fresh audit.

Applying the existing task-minus-control rule in GOAL lines 104–110 resolves
the following five source-to-canonical directions as `s=+1`; the last remains
a difference of signed parametric coefficients rather than an event contrast.
Weights and variable identities come from the local source specification,
not contrast filenames or the paper's outcomes.

| Selected comparison | Exact source direction and methods-only status |
| --- | --- |
| `pumpsvcontrol` | Reward pump event minus gray control event; task/control order resolved, `s=+1`. |
| `explodevcontrol` | Explosion event minus gray control event; task/control order resolved, `s=+1`. |
| `pumpscashvcontrol` | `0.5*pumps + 0.5*cash - control`; task/control order resolved, `s=+1`. |
| `pumpsexplodevcontrol` | `0.5*pumps + 0.5*explode - control`; task/control order resolved, `s=+1`. |
| `parapumpsvcontrol` | `pumps_demean - control_pumps_demean`; task modulation minus control modulation, `s=+1`. This does not establish an isolated risk effect or authenticate the preprocessing scale. |
| `explodevcash` | Explosion minus cash-out is source-signed. Neither event is the gray control; no frozen loss-versus-cash-out category ordering was identified. Canonical polarity remains unresolved. |
| `pumps` | Positive pump-event coefficient, with an explicit model intercept and no subtracted condition. It is not `pumpsvcontrol`; implicit-reference semantics are distinct from gray control. No new single-coefficient canonical rule is selected here. |
| `allpumps` | `0.33*pumps + 0.33*cash + 0.33*explode`, with no subtracted condition. Preserve literal weights; it is not all pump events or an exact one-third average. Same implicit-reference distinction as `pumps`. |
| `pumpspara` | Positive `pumps_demean` coefficient. Paper methods supply the intended pump-number interpretation; the metadata declares preprocessed events but does not show how that supplied column was constructed. |
| `controlpara` | Positive `control_pumps_demean` coefficient; retain signed control-modulation identity, without relabeling it as task-versus-control. |
| `cashpara` | Positive `cash_demean` coefficient; source sign known, not a cash-out-minus-other-event contrast. |
| `explodepara` | Positive `explode_demean` coefficient; source sign known, not an explosion-minus-other-event contrast. |
| `rt` | Positive `rt_reg.rt` coefficient. The local model explicitly assigns `response_time` to its **duration**, then convolves it; it is not a high-minus-low RT amplitude contrast. |

The latter eight source forms are now described more precisely, but this entry
does not silently add a category, single-coefficient, duration or parametric
ordering rule to the frozen ontology. Where needed, bind the existing
taxonomy's applicable rule and the actual event-column construction; otherwise
retain orientation-unresolved/sign-ineligible status. The cited paper used a
different analysis implementation and cannot certify FitLins execution or
actual group contributors. The five resolved role orderings therefore advance
the required semantics independently of the still-missing fitted Dataset-node
contributor IDs/N. No all-46 join, map loading, candidate score or new compute
was selected, and the completed route report remains unchanged.

## 2026-10-05 20:20 UTC — first-route contributor-support join completed

Analysis `source_handoff_qualification`. The repaired first-route step in
shared parent **46660232** completed with exit 0 on `sh03-06n06.int`, from
20:19:00.472386 to 20:19:45.117428 UTC. The preserved
[result](runs/source_handoff_qualification/20261005-first-route-repair-46660232/first_route_contributor_support.json)
records **44.668843090 seconds**, all 285 observed operations returned, no
active operations, no operation failures and no errors. The coordinator owns
the still-unsettled shared parent accounting under EP04; this episode adds no
separate parent or step charge.

All **13 ds000001/balloonanalogrisktask candidate maps** have matching signed
run-level t-contrast definitions and a simple positive-mean Subject-to-Dataset
chain, supporting `provider_declared_signed_group_z`. For every contrast,
all **16 configured subjects** have exactly one available matching effect-map
path: no configured-subject gap, extra subject, duplicate matching path, or
unavailable effect path was recorded. These are source-output availability
facts, not authenticated actual group-fit membership or sample size.

The provider metadata identifies FitLins `0.11.0.post0.dev94`, nilearn,
`MNI152NLin2009cAsym`, `5:run:iso` smoothing, `drop_missing=true`, CC0, and
OpenNeuro `ds000001.v1.0.0`. The report preserves source declarations; it does
not claim the copied environment lock is the executed version, decode NIfTI
values/headers, establish biological independence, or assign canonical semantic
polarity. Actual group contributor IDs and **actual group N remain null for
all 13 maps**. No candidate handoff, full 577-program equivalence, candidate
score, scientific trial, or audit has been qualified.

The smallest useful next handoff is an authentic fitted Dataset-node input
record, keyed to these 13 exact map/contrast/session rows, giving the subject
IDs and derivative paths actually included (or equivalent executed-input
evidence). The already documented source statistic chain can be retained;
bitwise replay or a perfect historical container is not being added as a
requirement. Methods/events excerpts sufficient to fix ambiguous canonical
comparisons are also needed under [GOAL.md](../GOAL.md#one-orientation-rule-for-the-baseline-model-and-evaluator).
These supply the map-keyed lineage/sample-size and methods fields required by
[DATASETS.md](../DATASETS.md#read-only-development-handoff). Unknown source
evidence is not recorded as absence. **No all-46-route expansion is selected**:
the same availability-only join cannot manufacture actual inclusion, polarity,
independence or eligibility. Consume a concrete contributor/methods handoff
next; preserve this completed route and both interrupted histories.

## 2026-10-05 20:04 UTC — same first-route repair selected

Analysis `source_handoff_qualification`. Root and independent static review
accept the name-first enumeration and non-`OSError` deadline repair described
below. Select exactly the same ds000001/BART 13-map route, with unchanged
90-second internal / 100+10-second external limits, one CPU and 2 GiB. The
EP04 remaining-RSA shared parent runs this action after its four workers finish;
the source output is fresh at
`runs/source_handoff_qualification/20261005-first-route-repair-JOBID/`.
No full 46-route retry, candidate score, map values or new source depth is
authorized by this repair. Any unresolved target operation remains visible;
unknown actual group inclusion is not treated as absence. The shared parent
cost is charged once under EP04. Launch ID and actual result follow separately.

Shared submission returned **46660232** at **20:05 UTC**. The EP03 result
destination is `runs/source_handoff_qualification/20261005-first-route-repair-46660232/`.
This is a submitted step, not a completed join.

## 2026-10-05 20:00 UTC — first-route stop localized; adapter repair prepared

The exact first-route step in shared parent `46658777` exited 2 after the
coordinator observed approximately 94 seconds. Its preserved
`runs/source_handoff_qualification/20261005-first-route-46658777/first_route_contributor_support.json`
records **92.805697373 seconds**, `route_complete=false`, and no `task_result`.
The shared EP04 parent is charged once there; no additional charge is entered
here. No candidate score, contributor qualification or map eligibility was
produced, and no scientific trial clock was started.

The shared parent's `ep03-first-route.out` shows both exact JSON reads and the
shallow Subject-node glob returning by 1.863 seconds. The second glob,
`node-subjectLevel/sub-*/ses-*/*_stat-effect_statmap.nii.gz`, started at 1.865
seconds and returned at 92.779 seconds. The one-shot traceback at 85 seconds
locates the wait at Python 3.12 `pathlib.py:212`, `entry.is_dir()`, called from
the unchanged contributor runner's second glob at line 111. Static reading of
that installed application helper shows it checks directory type **before**
matching `ses-*`, potentially statting nonmatching effect-file entries. The
specific entry and a shared-filesystem fault are not established.

The old adapter's `TimeoutError` inherits `OSError`, which the glob catches
while checking directories. Thus its ten recorded returned operations do not
prove an uninterrupted enumeration. Its explicit post-return deadline did
correctly reject the whole join. Historical reports and the original
`qualify_group_map_contributors.py` are preserved unchanged.

Selected preparation changes only `code/join_first_group_map_route.py`: for
the same two literal glob patterns, names are filtered before matched-directory
checks, including the same symlink-following behavior and final matching paths.
The original contributor still performs every final effect-file availability
check. Directory enumeration retains the original handling of `OSError`; a
dedicated non-`OSError` `RouteDeadline` now propagates interruptions and preserves
the failed/active operation instead of reporting interrupted work as returned.
Source JSON allowlists, first 13-map route, map-value prohibition, unchanged
actual-N uncertainty, and the **90-second internal / 100+10-second external**
bound remain fixed. No other route, deeper scan, cap increase, test chain,
automatic retry or source-system workaround is added.

The ASTRA `source_handoff_qualification` analysis now exposes this bounded
`first_route_contributor_support` output separately from the preserved full-route
recipe. Changed-object validation is reserved for the shared compute allocation;
no validation or source execution occurred during preparation.

The repaired command is prepared for coordinator and independent static review;
execution selection remains with the coordinator. Run only as a one-CPU/2-GiB
step in its suitable shared allocation, with six thread limits set to one:

```bash
timeout --signal=TERM --kill-after=10s 100s \
  python3.12 -u -B outputs/code/join_first_group_map_route.py \
  --output-dir outputs/runs/source_handoff_qualification/20261005-first-route-repair-${SLURM_JOB_ID}
```

## 2026-10-05 19:52 UTC — shared setup stopped before EP03; corrected parent selected

EP04 parent 46657948 stopped at its CPU guard before any EP03 step. The
8-GiB memory request exceeded the observed 8000-MiB per-CPU scheduler limit
and received two CPUs. Root selected a 7-GiB corrected parent; the exact
90-second first-route scope and source boundary remain unchanged. No source
read or join attempt occurred. The failed parent's 0.0144444444 core-hours
are charged once under EP04; this episode adds no charge.

Corrected shared parent **46658777** was submitted at 19:52 UTC. The first-route
output is `runs/source_handoff_qualification/20261005-first-route-46658777/`.

## 2026-10-05 19:44 UTC — fixed first-route contributor join selected

Under the user's priority for actual-data work, the coordinator selected one
bounded completion of the original first metadata route:
`ds000001/balloonanalogrisktask`, all 13 candidate maps from the unchanged
`20261003-group-map-metadata02/group_map_metadata_report.json`. This does not
restart the full 46-route join. The historical attempt and original runner
remain unchanged.

Static diagnosis found that `qualify_group_map_contributors.py` accumulates all
46 route results before creating its output directory or publishing a report.
The stopped attempt therefore provides no evidence identifying which JSON
read, directory enumeration, or file-availability check blocked. The recorded
source-I/O wait alone does not establish a site fault, absent contributors, or
map ineligibility. No infrastructure probe or workaround is selected.

The new `code/join_first_group_map_route.py` calls the original `process_task`
once. A process-local path wrapper forwards its existing operations unchanged
while publishing the active operation, exact path, returned-operation counts,
and failures. Allowed source access is still the two exact model/producer JSON
files and the original two declared Subject-node filename depths, with file
availability checks only. Map values, headers, PNGs, outcomes and scientific
scores stay closed. The returned 13-map join is retained in
`first_route_contributor_support.json`; configured/source-supported IDs remain
distinct from actual fitted contributors, and actual group N remains unknown.

Selected envelope: one CPU, 2 GiB, **90 seconds internal**, one traceback at
85 seconds if still active, **100 seconds external plus 10 seconds kill grace**.
This small step will share the next substantive EP04 allocation; its job ID is
pending. The parent is accounted once by the coordinator, not charged again
as an EP03 step. A timeout retains operation evidence and stops; there is no
automatic retry, bound increase, remaining-route dispatch, or new test chain.
The three existing contributor tests passed in the preserved prior attempt;
the new adapter passed coordinator and independent static review, without
execution, imports or further source reads.

The coordinator's 19:39:54 UTC accounting confirms `46429138` cancelled after
2,077 seconds on one CPU and `46425841` completed after 3,346 seconds on four
CPUs. Conservative whole historical parent bounds total **4.2947222 core-hours**.
Even including the entire proposed one-CPU/40-minute shared parent gives less
than five core-hours, below the registered 1,500-core-hour ceiling. No current
contract scientific trial has begun, so its 120-hour trial clock is unstarted;
this metadata join does not start it or qualify a scientific handoff.

Coordinator command from this episode directory, with all six numerical
thread limits set to one:

```bash
timeout --signal=TERM --kill-after=10s 100s \
  python3.12 -u outputs/code/join_first_group_map_route.py \
  --output-dir outputs/runs/source_handoff_qualification/20261005-first-route-${SLURM_JOB_ID}
```

Shared EP04 parent **46657948** was submitted at 19:49 UTC. The selected
first-route step remains bounded as above; its fresh directory uses that job
ID. Submission is not evidence of a completed source read or join. The parent
cost is charged once under EP04.

## 2026-10-03 21:05 UTC — contributor join runtime observation

The three focused contributor-boundary tests passed in 0.005 seconds in
allocation `46429138` on `sh03-06n50.int`. The selected metadata-only join
started at approximately 20:56:11 UTC in the coordinator's one-CPU / 2-GiB
allocation. It has exceeded the under-five-minute estimate without a final
report. A single bounded administrative process inspection at approximately
21:03 UTC found one second of process CPU, 15,508 KiB RSS, and source-I/O wait;
the earlier Slurm-step peak RSS was historical, not this process's memory.
Static review found no numerical-array access or concrete memory-growth defect.

The coordinator selects a ten-minute limit for this attempt: if no report
has completed by that point, interrupt its Slurm step, retain the traceback
and incomplete status, and diagnose the exact blocked application read before
any retry. No missing contributors, map ineligibility or system fault can be
inferred from this wait. No further filesystem scans or duplicate job are
selected.

At 21:06:20 UTC the coordinator sent Slurm INT to step `46429138.0`; it did
not return a report or traceback. Slurm cancellation of allocation `46429138`
followed at 21:06:52 UTC, and its PTY terminated with exit137. Accounting
records CANCELLED after 2,077 seconds on one CPU for the combined allocation.
No contributor-support report was produced. The exact blocked source read is
unknown, so this source route is stopped without a retry or system/configuration
probe. This is an incomplete metadata attempt, not evidence of absent source
contributors. Previously completed work in other episode paths is unaffected.

## 2026-10-03 — group-map lead and physical-identity repair

The coordinator executed the dataset-node census in job 46425841. It found
291 candidate Z-map filenames across 30 accessions and 45 dataset/task routes,
all with exact source models and declared space metadata. No image header,
voxel value or prior score was opened. The files remain candidates pending
sign, space, group-independence, license and source-method qualification.

The selected second identifier diagnosis showed that all unresolved keys are
ds004656 session labels `am/pm` versus `AM/PM`; their original artifact suffixes
and declared physical BOLD stems agree. Selected a narrow repair that accepts
this alias only if those untouched physical identifiers agree uniquely and
all other key fields match. It records each alias and does not rewrite source
metadata, globally case-fold identities or exclude acquisitions. Rerun the
metadata reconciliation to `20261003-design-support03` and the dataset-node
census to `20261003-group-map-metadata02`, one CPU/2 GiB/<5 minutes per step.
This restores source support if the checks pass; it does not qualify maps for
candidate scoring or instantiate an audit.

The physical-identity repair passed for all 60 aliases: all 3,667 source
acquisitions now match uniquely, with no missing or ambiguous key and 278
consistent source contrast definitions. The refreshed dataset-node census
found 297 candidate Z-map filenames across 31 accessions/46 dataset-task
routes, each with its exact source model and declared space metadata. Counts
do not establish 46 independent biological groups or 297 eligible effects.

Selected next: `code/inspect_group_map_headers.py`, limited to the first 352
uncompressed NIfTI header bytes of each explicitly censused candidate, in
job 46425841, one CPU/2 GiB/<5 minutes. Report dimensions, data/statistic codes,
qform/sform and voxel size; decode no voxel values. This resolves an explicit
space/format prerequisite without choosing canonical orientations, a common
mask, a taxonomy, group eligibility or a scientific pipeline. Methods-only
orientation and all other pre-search rules still precede target-value loading.
Output: `runs/source_handoff_qualification/20261003-group-map-headers01/`.

Header census completed: all 297 candidates are readable NIfTI-1, each with
97×115×97 spatial dimensions; no header errors. All carry intent code 0, so
the NIfTI header itself does not authenticate a Z statistic. Source generation
provenance remains required for that claim. No voxel values were decoded,
and equal dimensions alone do not establish equal affine/mask, independent
groups, sign eligibility or a scientifically qualified map handoff.

No candidate-scoring experiment has started under the current `GOAL.md`,
`DATASETS.md`, and `SEARCH_POLICY.yaml` contract. The authorized 2026-10-03
source-metadata work is recorded below.

## 2026-09-28 — Mandatory-falsifier policy repaired

- Added the Goal-required task-family-only and fold-local global-mean-map
  baselines to the policy's required falsifiers.
- Added the nuisance-handling ablation and the exchangeable-within-family
  negative-control contrast that were also mandatory in the Goal but absent
  from the policy list.
- Required every listed falsifier to pass before promotion and made the global
  baseline training-fold-only so it cannot use held-out target information.

This repair occurred before any current-contract target-map computation. No
scientific trial was run, and the prospective audit remains unopened.

## 2026-09-30 — Scientific-question and paper-design rewrite

- Reframed the paper as incremental **signed contrast information beyond a
  task-family prototype**, rather than another text-to-brain demonstration.
- Verified overlapping claims in NeuroQuery (2020), Text2Brain (2022),
  NeuroConText (2024/2026), and Hammonds et al.'s NeuroVLM (2026 preprint v3).
  Links and the proposed distinction are in `paper_plan.md` and `../GOAL.md`.
- Clarified normalized group Z-map geometry versus effect size, independent
  group weighting versus contrast count, and precise prototype-dominated
  outcomes versus inconclusive nulls.
- Added the question-first `paper_plan.md`, episode README, and built-in
  image-generation schematic `ep03_conceptual_question.png`; the prompt and
  visual inspection are in `ep03_conceptual_question.prompt.md`.
- Kept the existing operative grammar, primary score, required falsifiers
  (including the 2026-09-28 repairs), budgets, promotion and audit decisions.
  The search policy gained narrative annotations only.

Execution state remains **not started**; attempts and active Slurm/process
IDs: none. No map input, protected outcome, candidate score, or audit result
was read. Existing runtime history above is preserved. This is a design
milestone, not a scored trial, acceptance, or paper-readiness finding.

Actual blocker: the read-only 51-group handoff is not provisioned here; the
evaluator and pre-search support/inference decisions remain unresolved, and
there is no fresh audit source. Next action is the existing source handoff and
group/map eligibility work in `../DATASETS.md`, not a new generic validation
sequence. The episode ASTRA program has not yet been authored; that local
projection gap does not authorize execution or delay otherwise authorized
source work.

## 2026-09-30 — Concept-figure wording corrected

Root review identified an overstatement in the initial schematic: "Family
prototype is sufficient" could imply equivalence beyond the bounded null
interpretation in the Goal/paper plan. A targeted built-in imagegen edit
replaced it with **"No supported residual gain"**, retaining layout and other
content. The edited PNG was visually inspected and saved under the existing
`ep03_conceptual_question.png` filename; the initial version remains
`ep03_conceptual_question-v1-superseded.png`, historical only. The exact edit
prompt is in `ep03_conceptual_question.prompt.md`.

This is an illustration correction, not a scientific trial or runtime-state
change. No design/policy decision, outcome access, execution, job, canonical
action, or Git mutation occurred.

## 2026-10-02 local scope review

The user-authorized scope review found that a fixed prototype plus an odd residual does not make the full prediction odd. Prospectively amended GOAL, DATASETS, policy and paper plan: fit canonical targets and descriptions, restore the same polarity to both complete predictions, preserve the nonzero family comparator, and label reversal an implementation invariant. Primary literature confirms text-to-map overlap; see [review (consolidated)](paper_plan.md). No current-contract map, score, or audit was opened. The existing un-authored ASTRA projection remains unchanged and nonblocking.

## 2026-10-02 figure review

Personally viewed the current 1942 × 809 imagegen PNG and retained it. GOAL/paper-plan captions now explain shared canonical orientation, sign restoration of the entire prediction, and check icons as planned comparisons rather than passing results. Existing exact generation/edit prompt record retained. No image, scientific policy, data, or runtime change.

## 2026-10-03 — source_handoff_qualification

The scientist authorized resuming feasible episodes and completing their jobs.
This separately scoped EP03 work follows the active development/audit roles;
it does not open audit outcomes or narrow the registered search.

The coordinator supplied the steward source location from `../DATA_LOCATIONS.md`:
`$OAK/users/zijiao/br_autoresearch_data/steward_acquisition/neuroeffect_spatial_sources_20260917T200750Z`.
Only its quarantine and explicitly named provenance metadata were inspected.
The quarantine states that the preserved 17 pinned fMRIPrep repositories and
3,654 selected BOLD/transform files (42,372,096,471 bytes) are historical,
development-exposed spatial lineage. They are **not** the approximately
577 signed group-map handoff and do not resolve the reported 51-versus-46
dataset/task-group discrepancy. The lineage inventory records 31 accessions
and 3,667 acquisitions. None of these counts establishes independent biological
groups. No source map, BOLD array, transform payload, result text or audit
outcome was opened. No source file was changed.

The earlier BR read-only dataset resolver returned `dataset_not_found` for
`NeuroEffect`; the local steward metadata above supplies the concrete source
lead. This tool response did not establish source absence.

Selected substantive next step: `outputs/code/qualify_source_metadata.py`, a
bounded census of four named JSON/JSONL metadata files. It reports accession,
task and participant-label support, task-pair overlap within accessions,
design-key agreement, pinned public repositories, and existence of the three
declared historical metadata pointers. It never follows those pointers into
payloads and cannot qualify a candidate corpus by itself.

Attempt: `outputs/runs/source_handoff_qualification/20261003-metadata01`.
Resources: one CPU, up to 1 GiB, at most five minutes, bundled in the
coordinator's substantive Slurm allocation. No worker-side job submission or
testing. Job ID and observed result are pending coordinator execution.

The missing signed-map/methods handoff, authenticated independence, and
pre-search support/inference decisions remain candidate-scoring dependencies.
A fresh prospective audit is a later confirmation dependency; its absence
alone does not bar authorized development on a qualified exposed corpus.

Authored `outputs/astra/v0.0.14/astra.yaml` with stable source qualification,
grouped evaluator, adaptive development, finalist, configuration lock and
prospective audit analyses. The registered grammar, 30–72 scientific trials,
falsifiers, resource ceilings and evidence roles are unchanged. Only source
metadata qualification currently has an executable recipe. Official schema
and semantic validation is pending coordinator compute; this projection is
not candidate readiness or canonical BR state.

### Observed metadata census

The coordinator completed `20261003-metadata01` in Slurm allocation
`46425841`. The report confirms **31 OpenNeuro accessions, 46 dataset/task
pairs, and 3,667 acquisition rows**; the design mapping and source inventory
have identical acquisition-key multisets. These are source support counts,
not 51 authenticated independent biological groups. The output remains
`metadata_census_completed_signed_map_handoff_unresolved`.
Evidence: [source metadata report](runs/source_handoff_qualification/20261003-metadata01/source_metadata_report.json)
and [dataset support](runs/source_handoff_qualification/20261003-metadata01/dataset_support.tsv).

All three exact historical metadata pointers still exist in
`$SCRATCH/neuroeffect-full51-spatial-build/20260917T201500Z.2230efb/release-stage-v1/`.
Bounded inspection of their metadata, without any referenced payload, found:

- `inputs/full51_stage2_spatial_support_universe.json` contains 7,954
  acquisition-pair samples covering the 31 accessions / 46 dataset-task pairs.
  These paired BOLD acquisitions are not group statistical maps.
- `inputs/design_sidecars.runtime.json` is a 47,619,389-byte design/operator
  metadata receipt. Its stored summary reports 37 selected datasets, 343
  authorized conditions and 4,634 validated acquisitions. The larger design
  support must be reconciled with the spatial subset before any group claim.
- The design receipt includes signed contrast conditions/weights and exact
  FitLins definition-source paths. These are useful provenance leads but do
  not by themselves authenticate a group Z map, an EP03 task family or the
  methods-only canonical polarity.
- Its producer is `neuroeffect.data.preparation.source_binding_repair/v1`;
  relative bindings name the historical canonical-effect specification and
  training-scope receipt. Historical `confirmatory_headline_eligible` flags
  belong to that upstream project and grant no EP03 audit eligibility.

The next bounded steward step is a metadata-only join of the design receipt
with spatial support, followed by exact producer/contrast-source provenance.
It must not import upstream confirmatory roles, treat acquisition pairs as
independent groups, or replace EP03's signed group Z target with BOLD tensors
or design operators. No map, BOLD tensor, operator array or historical scored
result has been read.

Selected metadata reconciliation attempt:
`outputs/runs/source_handoff_qualification/20261003-design-support01` using
`outputs/code/reconcile_design_support.py`. Request one CPU, up to 2 GiB and
five minutes within coordinator compute. It reads the two explicitly permitted
metadata files, joins exact acquisition keys, retains signed condition/weight
definitions, reports missing/ambiguous keys and distinct upstream support, and
checks existence of exact source-definition paths. A second existence pointer
uses the documented shared FitLins source layout; existence is explicitly not
content authentication. It reads no referenced map, tensor, operator or score.

The exact old `ds000001` contrast-definition path no longer exists. The
documented executed `ds000117` statistics-model specification under the shared
OpenNeuro analysis root still exists and contains signed weights and model
definitions. This establishes a concrete metadata recovery lead, not eligible
group-map availability. The coordinator's suggested
`$OAK/users/zijiao/neuroeffect` path lacks source/config/README files at its top
level; no unrelated contents or results were enumerated.

The surviving `ds000117` source model declares a Dataset-level `data_level`
node. Bounded filename inspection found its `node-dataLevel/ses-mri` directory
and six corresponding `*_stat-z_statmap.nii.gz` paths. No image/header or
generated report was opened. This is a concrete candidate group-map source
lead, not authenticated sign, space, eligibility or independent-group evidence.

The coordinator requested a metadata/filename-only census across the exact
matched dataset/task routes. Implemented `outputs/code/census_group_map_metadata.py`
and selected `outputs/runs/source_handoff_qualification/20261003-group-map-metadata01`,
one CPU, up to 2 GiB, at most five minutes after the reconciliation completes.
It reads explicit statistics-model/dataset-description JSON, requires an
actual declared Dataset node, excludes nodes retaining subject/run grouping,
and inspects only that node's immediate or single-session Z-map filename
patterns. It compares signed source contrast definitions and reports declared
space metadata without inferring a template from filenames or a grid. Missing
targets remain unknown. It opens no images, headers, scores or generated
reports and does not declare the handoff qualified.

### Observed design-support reconciliation

The coordinator completed `20261003-design-support01` in allocation `46425841`.
Actual design rows contain **36 distinct accessions, 57 dataset/task pairs and
4,634 acquisition records**. The receipt's `37 selected datasets` is a stored
historical summary, not the observed count. The spatial subset has 3,667 keys
and 46 dataset/task pairs; **3,607 acquisitions join uniquely**, with 60
unmatched keys for `ds004656/foodstimhilo` and no ambiguous keys. The matched
scope supplies 275 contrast names / 275 definitions, with no conflicting
condition/weight definitions. Eleven additional design task pairs are outside
the declared spatial subset. None is automatically added as an EP03 input.

Evidence: [design support report](runs/source_handoff_qualification/20261003-design-support01/design_support_report.json).
The missing `ds004656` acquisition keys do not prove absent data: its task pair
exists in the design receipt, so exact identifier/physical-source provenance
must be examined before any exclusion. It is not silently removed from the
episode's scientific scope.

The official pinned ASTRA validator passed schema and semantic validation;
evidence is `outputs/astra/v0.0.14/validation_20261003_job46425841.json`.

Prepared a targeted diagnostic retry of the metadata reconciliation, selected
as `outputs/runs/source_handoff_qualification/20261003-design-support02`
(same one-CPU / 2-GiB / five-minute bound). This adds only the unmatched
source artifact identifiers and exact same-task design identifiers/physical
stems to diagnose the observed 60-key discrepancy. It does not change matching,
case-fold labels, substitute participants or exclude a dataset. Coordinator
execution is pending; this is metadata repair, not a new scientific trial.

## 2026-10-03 — source_handoff_qualification contributor support selected

Selected at **20:49:10 UTC**, before execution, under the existing metadata-only
source qualification authority. The completed `20261003-group-map-metadata02`
manifest contains 297 candidate map paths in 46 task routes / 31 accessions;
all 278 source contrast-definition comparisons match the design receipt.
Session-specific maps remain distinct rows, not independent biological groups.
These counts do not replace or authenticate the intended 577-program/51-group
handoff.

Bounded inspection of the directly referenced shared producer checkout found:

- `ds000001/task-balloonanalogrisktask/dataset_description.json` declares
  FitLins `0.11.0.post0.dev94`; `ds000117/task-facerecognition` instead declares
  `0.11.0.post0.dev16`. Both declare nilearn, `5:run:iso` smoothing,
  `MNI152NLin2009cAsym`, and `drop_missing=true`.
- Their copied `code/openneuro_glmfitlins` checkout is
  `e624fec2765a7869a4f482da701d85b82bc2f4b7`; its `uv.lock` names
  `jmumford/fitlins` `paddedint` commit
  `1bed1a2ba54e948d28b47562dc05b1809b3b1eb9`, version `dev42`, and nilearn
  `0.11.1`. The copied lock is not asserted to be the executed environment.
  Exact bitwise replay is not imposed as a new eligibility condition.
- `scripts/prep_report_py/groupmap_report.py:350–381` applies a threshold only
  when producing PNG plots; it does not write thresholded candidate NIfTI maps.
  Its line 94 reports `len(spec_results['subjects'])`, the configured subject
  count. With missing inputs allowed, that is not authenticated actual N.
- The inspected `ds000117` source model has explicit signed run-level t
  contrasts and positive intercept-only Subject/Dataset t nodes. Its source
  sign is recoverable from condition/weight definitions, independently of
  map values or contrast-name wording. Canonical semantic polarity remains
  unassigned. Immediate group-node metadata contains map/PNG paths, not a
  direct fitted group-input manifest; no generated report or PNG was opened.

Prepared `outputs/code/qualify_group_map_contributors.py`. It opens only the
fixed manifest, its exact source model JSON, and each task's dataset-description
JSON. It enumerates effect-map filenames only at declared Subject-node depths
`sub-*/*_stat-effect_statmap.nii.gz` and
`sub-*/ses-*/*_stat-effect_statmap.nii.gz`, and checks file availability without
reading images. It reports configured IDs/N, source-output-supported IDs/count,
missing configured support, extra identities, and repeated matching paths
separately. Within an accession only numeric zero padding is normalized; raw
paths and configured IDs are retained. Session-stratified group maps are joined
within their session; unstratified group maps retain all matching sessions and
report repeated subject paths rather than counting them as more people.

Provider-declared signed-Z status requires matching source t-contrast weights,
a simple positive intercept-only Subject/Dataset chain, the selected Z path,
and FitLins/nilearn producer metadata. Other source configurations remain
explicitly unresolved. This is provider-declared provenance, not NIfTI-intent or
voxel verification, canonical orientation, group independence, or full primary
eligibility. **Actual contributor IDs and actual group N remain null** without
direct fitted Dataset-node inclusion evidence. Source-output availability must
not certify a minimum actual N.

Selected command, to be run only by the coordinator in its verified allocation:

```bash
python3.12 -m unittest discover -s outputs/code -p test_group_map_contributors.py
python3.12 outputs/code/qualify_group_map_contributors.py \
  --group-map-manifest outputs/runs/source_handoff_qualification/20261003-group-map-metadata02/group_map_metadata_report.json \
  --analysis-root "$OAK/data/OpenNeuro_analyses/openneuro_fitlins/analyses" \
  --output-dir outputs/runs/source_handoff_qualification/20261003-contributor-support01
```

The three focused synthetic checks cover session pooling, actual-N uncertainty,
unavailable/duplicate source outputs and positive sign propagation. No tests or
source operations were executed by this worker. Request **one CPU, 1 GiB and a
five-minute bound**, bundled with already substantive coordinator work; expected
runtime is seconds to a few minutes of metadata I/O. No image/header/PNG values,
candidate scores, audit data, or sibling scientific outputs are accessed.
Execution result and job ID are pending. The pending question about the original
producer checkout has not been repeated or treated as answered.

Independent code-only review identified and corrected two sign-route conditions
before execution: both averaging nodes must preserve contrast identity, the
Subject node must preserve subject identity, and map session identity must agree
with Dataset grouping. A session-grouped map lacking a session is unresolved,
not silently pooled. Focused synthetic checks now cover these cases. No data or
test was opened to choose the correction; previous attempts remain unchanged.
