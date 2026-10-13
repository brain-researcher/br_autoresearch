# Experiment log

## 2026-10-06 01:21 UTC — candidate_decision — closed_no_candidate; reporting complete

The write-once [complete-panel decision](runs/panel_decision/registered_panel_decision.json)
finished at01:17:27.784300UTC with `complete_admission: true`, one candidate
decision, `closed_no_candidate` and no selected model. Both full collectors
and the screen exited0. Scheduler accounting at01:19:10UTC records parent
**46690071 COMPLETED0:0**,91 seconds on one CPU. The figure action also exited0
and root visually inspected its complete, readable three-panel output.

| Frozen model | Native VLM-minus-DINOv2 gain, Fisher z | Participants with positive gain |
| --- | ---: | ---: |
| CLIP | -0.0347765675 | 0/5 |
| PEcore | +0.0010264769 | 3/5 |
| SigLIP2 | -0.0392206391 | 0/5 |

No model reaches the registered0.02 gain or the required participant-direction
and leave-one-participant/ROI-out conditions. All three have positive
unsubtracted alignment exceeding both familywise null thresholds, which is
distinct from superiority over DINOv2. All fail the mandatory low-level-partial
and near-neighbour gates; PEcore's near-neighbour gain is-0.0107363188. The
full saved table retains every sensitivity, model and participant.

Independent completed-report review confirms complete original25-cell null
coverage,10000 permutations and1000 geometries per VLM,400 evaluable encoding
rows and no collector errors. Independent decision review additionally confirms
the matched400-row training-RSA/reliability table, complete descriptive EVC
context and all frozen gate outcomes. Encoding/RSA disagreements are retained
as diagnostics and cannot rescue a primary candidate. The conclusion is limited
to these five participants and frozen native geometries in selection-exposed
development. It establishes no equivalence, population confirmation, universal
DINOv2 superiority or causal effect of language pretraining.

This is the protocol's valid scientific terminal class. GOAL steps5–6 close
without a candidate before any selected-row rerun. No rerun allocation,
configuration lock or OOD audit is selected; all previously prepared rerun
source/tests remain unexecuted and its prospective25.5-hour allocation is never
reserved. No replacement model, endpoint or threshold change follows this result.
See the [figure](figures/registered_panel_decision/20261005-full01/registered_panel_decision.png)
and its [caption](figures/registered_panel_decision/20261005-full01/caption.md).

Charge **0.0252777778 CPU-hours once** for the entire collection/reporting
parent, replacing its0.6666666667-hour reservation. Explicit completed usage
is **24.8327777778 CPU-hours**; conservative exposure retaining historical
shared-parent bounds is **32.3619444444 CPU-hours**, leaving
**2467.6380555556 CPU-hours**. No EP04 reservation or live job remains.

The optional ASTRA validation passed schema checking but exited1 because its
semantic template parser interpreted the shell variable's braces as an ASTRA
placeholder. The immutable validation output is retained. Its conditional
rerun recipe was corrected to an explicit nonempty check and quoted shell
variable without braces; that documentation correction has not been revalidated
on compute. It did not block or alter the three successful scientific stages,
and no check-only allocation is introduced after scientific closure.

BR `request_scientific_review` returned advisory directive
`ext_review_dir_904fbb9462db` with a generic125-rule KG registry and no tailored
checks, based only on the supplied aggregate summary. It did not read local
files or issue a result verdict. Applicable unit/selection/claim-limit cautions
were considered by the independent local reviewer. No formal review session,
verdict submission or canonical campaign transition was opened.

## 2026-10-06 01:16 UTC — controls_diagnostics — all25 bundles complete; collection running

Scheduler accounting at01:16:10UTC records every array46688490 task0–24
COMPLETED0:0. The25 one-CPU parent-task durations sum to **71035 allocated
CPU-seconds =19.7319444444 CPU-hours**. Charge that sum once, replacing the
37.5-hour reservation; no array-master, step, batch or extern duplication.
Explicit completed EP04 usage becomes **24.8075000000 CPU-hours**. Conservative
exposure, retaining prior shared-parent bounds and the0.6666666667-hour
collection reservation, is **33.0033333333 CPU-hours**, leaving
**2466.9966666667 CPU-hours** under the2500-hour cap.

The last cell, sub07/fold1, records all10000 permutations and1000 geometries
for each VLM, `completed_selected_null_ranges`, no errors,3976.591935s process
time and1123484KiB peak RSS. Earlier retained prefixes remain unchanged.
All400 encoding diagnostic rows were already recorded complete/evaluable.
Full numerical collection/admission remains the responsibility of the
pre-submitted successor rather than these unit-level observations.

Dependency resolution started **46690071**, observed RUNNING with one CPU at
the same check. It collects both full panels, applies the one registered
candidate decision and reports the saved result. No completed collection or
candidate decision is yet observed. Conditional rerun preparation remains
unexecuted and unreserved.

## 2026-10-06 00:55 UTC — development_rerun implementation and conditional dispatch prepared

The [new unit/collect driver](code/run_selected_development_rerun.py),
[six focused checks](tests/test_selected_development_rerun.py) and
[bounded wrapper](code/selected_development_rerun.sbatch) are concrete.
Root and separate numerical/scope and admission/publication reviewers passed
static review. The wrapper also passed independent static review. Tests and
shell/runtime checks remain **unrun**; they are bundled before payload access
inside each future substantive allocation. Current producers and submitted
collection46690071 were not changed.

If the saved decision selects a candidate, the prospective dispatch is24
parallel1CPU/4GiB cells (slots1–24, explicit one-hour cap), then one dependent
1CPU/4GiB90-minute job performing slot0 and full collection. These25 disjoint
cells cover every required participant/fold once. This bundles collection with
substantive computation. Test and timeout ceilings total3410s in an array cell
and4030s in the final allocation, leaving startup room. The prospective upper
reservation is25.5CPU-hours, **not yet reserved or submitted**; actual selection
and remaining-budget accounting must precede launch. No standalone collector,
waiting process, retry, lock or OOD stage is included.

The existing ASTRA development_rerun node now binds the real sources and this
conditional dispatch. Its updated validation remains unrun and can be checked
by the already-submitted collection job's optional ASTRA step. There is still
no observed candidate decision or empirical rerun. A closed-no-candidate
decision makes this prepared branch unnecessary.

## 2026-10-06 00:48 UTC — all encoding unit reports complete; first eight bundles finished

Bounded inspection of the exact25 reports in the selected
[encoding manifest](code/remaining_controls_selected/20261005-full01/encoding_reports.json)
finds `encoding_diagnostic_complete`, all16 rows recorded, all16 evaluable
and `errors: []` in every cell:400 recorded diagnostic rows in total. Per-cell
process time is24.676–58.160 seconds and recorded peak RSS906044–961932KiB.
This establishes completed unit reports; the full collection/admission and
scientific candidate decision remain pending. No diagnostic score or sign
was used to alter the panel.

At00:47:28UTC, scheduler parent-task records show slots0–7 of46688490
COMPLETED0:0 and the remaining17 RUNNING. Each of the eight completed null
reports records its entire selected range and no errors. Slot0 preserves the
earlier256-permutation/eight-geometry prefix and contributes only the remaining
9744/992 draws; other completed cells contribute10000/1000. The dependent
46690071 remains pending. Retain the full37.5CPU-hour array reservation until
whole-array accounting replaces it; do not add the partial parent usage on
top of that reservation.

## 2026-10-06 00:43 UTC — prepare conditional selected-row rerun

The registered full-development rerun has no executable in the existing ASTRA
node. Preparation is now assigned to a separate new unit/collect driver;
running producers and the pending collection job remain unchanged. Execution
requires the one saved decision to be
`selected_candidate_pending_exact_full_rerun`. A `closed_no_candidate` result
ends this branch without a rerun, replacement, lock or OOD access. Admission
failure is diagnosed from its retained stage record and is not a no-candidate
decision.

Independent contract review confirms the minimal replay scope: freshly compute
the selected VLM and DINO native/partial RSA, all four mandatory sensitivities,
all 10000 selected/permuted-DINO contributions and 1000 selected random
geometries per cell, preserving the original VLM seed index. Retain the
authenticated complete other-two-VLM null components and rebuild the full
three-VLM familywise winners, signed statistics and higher95th thresholds for
comparison with the immutable originals. Every gain uses freshly computed DINO;
no chosen-only null, scalar-threshold shortcut or new candidate decision is
introduced. Completed encoding, EVC and training-context diagnostics remain
preserved.

The planned 25 disjoint participant/fold units and one collection retain one
rerun identity, the original120-hour clock, complete support and fresh output
paths. Any mismatch is a technical failure. This is implementation preparation
only: no score inspection, scientific execution, new reservation or submission
has occurred, and no candidate has yet been observed selected.

## 2026-10-06 00:13 UTC — select dependent complete-panel aggregation and reporting

**Submitted once at00:13:31UTC:46690071, afterok:46688490, with invalid-
dependency cancellation.** Root and independent static review passed both
the batch and the reporting source. No collection, decision or render is yet
observed. Submission is a bounded authorized stage, not scientific completion.
At00:13:45UTC all25 array tasks were RUNNING; no failed/canceled/timed-out/
out-of-memory task was reported. The successor remained PENDING(Dependency).

Select one [bounded batch](code/collect_and_screen_full01.sbatch) with1 CPU,
4GiB,40 minutes, no GPU, `russpold,normal`, after successful completion of
the whole46688490 array. The already-completed first parent supplies all
sensitivities and the retained pilots. Group both full collectors, the one
fixed screen and its descriptive figure in this allocation, rather than
submitting one short job per stage. Existing EP21 export uses all four
registered CPU threads and has no reserved memory for another task; it will
not be altered or oversubscribed for EP04 reporting.

All three scientific stages have1860s combined external ceilings, optional
ASTRA validation40s and plotting160s, totaling2060s plus startup under2400s.
The [plotter](code/plot_registered_panel_decision.py) reads only the one
fully admitted saved decision. It shows all15 participant gains and three
saved means, the separate unsubtracted-alignment/null comparison and all36
gate outcomes. It does not recompute scores, screen again, select subsets,
infer population uncertainty or open OOD. Its status is separate from the
scientific exit; a reporting failure cannot erase or change a valid decision.

Reserve **0.6666666667 CPU-hours** before submission. Conservative exposure
becomes **50.7713888889 CPU-hours**, leaving **2449.2286111111 CPU-hours**.
Replace this reservation with actual whole-job usage or zero if canceled
without allocation. The40-minute bound is an upper limit, not a runtime
measurement; no time-padding or waiting process is introduced. Dependencies,
fresh outputs, scientific non-evaluability and the original panel deadline
remain enforced. Runtime/rendering checks will occur only inside the job.

## 2026-10-06 00:09 UTC — complete collection and screen entrypoint prepared

The [bounded entrypoint](code/collect_and_screen_full01.sh) passed root and
independent static review against all three existing parsers. It consumes the
selected25 encoding/26 null report paths, collects nulls then encoding, and
performs the fixed screen only after successful complete admission. Each stage
uses the existing600s internal cap and610s external timeout plus10s kill grace;
all three fit1860s plus setup. Its fresh no-clobber operational status ledger
records started/exited actions. Any nonzero exit stops dependent stages and
preserves reports. A complete but scientifically not-evaluable encoding result
is retained as such and cannot be coerced into admission or a no-candidate
decision. No lock, OOD access or automatic successor is introduced.

This is lightweight job preparation only. The22 passing producer/collector/
screen checks are reused; no test, scientific execution, additional allocation,
budget reservation or collection has occurred. At00:07:36UTC the existing
array had13 running and12 pending slots. Root must select a suitable bounded
compute allocation after the complete inputs exist, respecting aggregation
and the original panel clock.

## 2026-10-06 00:02 UTC — first controls complete; remainder running

Parent46680856 completed with exit0 in519 seconds on2 CPUs. All scientific,
EP12 figure and EP19 page-extraction actions exited0. Independent report
review confirms all1600 sensitivity rows and100 fold records complete and
evaluable, no errors, exact model/ROI/training-context bindings, and all source
strata retained. The [sensitivity report](runs/native_sensitivities/20261005-all01/native_sensitivities.json)
finished at2026-10-06T00:00:50.889577+00:00 with493.839799s process elapsed
and1103776KiB process peak RSS. Prespecified near-neighbour and complete-repeat
subsets retain14285 and23536 images respectively; equal-source and neural-cosine
analyses retain all23560. There is no outcome-based pruning or candidate decision.

Charge **0.2883333333 CPU-hours once** for the whole parent, replacing its
1.6666666667-hour reservation. No step, batch/extern, EP12 rendering or EP19
document charge is added. Explicit completed EP04 allocation usage now totals
**5.0755555555 CPU-hours**. Conservative exposure, including the new37.5-hour
array reservation and prior conservative shared-parent bounds, is
**50.1047222222 CPU-hours**, leaving **2449.8952777778 CPU-hours**.

At00:01:25UTC, array46688490 had9 running slots and16 pending(Resources).
Slot0's actual compute shell onsh04-04n21 passed the single-CPU resource and
partition guards and started only its registered residual null ranges.
Do not resubmit pending slots. Complete collection and screening remain pending.

## 2026-10-05 23:58 UTC — measured controls pass; select 25 bundled remainder tasks

**Submitted once,23:59:34UTC: array46688490, slots0–24, concurrency25.**
The selected [manifest](code/remaining_controls_selected/20261005-full01/manifest.json)
and explicit encoding/null report lists retain every completed prefix.
Root and independent review passed the partition-membership correction and
the distinction between a fully recorded scientific not-evaluable encoding
result (continue independent null work, retain the result) and a technical
failure (stop). Scientific runners remain unchanged. Metadata preparation
performed only lightweight job preparation; no login-node scientific execution.

The scientific/check worker `46680856.1` completed with exit0 in108 seconds.
Its retained encoding report records all16 rows evaluable, no errors,
55.889489s elapsed and949272KiB process peak RSS. The retained null report
records all256 permutations and8 geometry draws for each of three VLMs,
no errors and zero observed-replay discrepancy:43.345494s total,
21.133539s setup and1123956KiB peak RSS. All22 focused checks passed
(4 encoding,4 null,4 null collector,3 encoding collector,7 panel screen).
Root and independent review agree that these completed artifacts qualify
the remaining scientific work; the still-running independent sensitivity,
EP12 figure and EP19 document worker is not a dependency of that work.

The recorded draw timings project about41 minutes per fresh participant/fold
cell and17.083 CPU-hours for the remaining work at pilot rates; neither is
a guarantee or a charge. Select exactly25 participant/fold bundles, each
with its encoding action followed by its full residual null action, except
the already-completed sub01/fold0 encoding. Preserve the completed null
prefix and use only permutations256:10000/geometry8:1000 for that cell;
the other24 cells retain the complete registered10000/1000 ranges.
Bundling avoids24 short encoding-only Slurm jobs. The existing complete
collectors remain authoritative; no incomplete candidate screening occurs.

Selected allocation per bundle:1 CPU,4096MiB,90 minutes, no GPU,
`russpold,normal`, array concurrency25. Encoding internal limit300s and
null limit4800s allow5190s including50s setup and all termination graces,
within5400s. The remaining210s and larger-than-pilot process allowance
cover startup and node-speed variation. Six BLAS/OpenMP limits remain1.
Including the current2-CPU parent, at most27 EP04 CPUs are requested
concurrently, below48; aggregate requested memory108GiB is below256GiB.
Reserve **37.5 CPU-hours** before submission. Conservative total exposure
becomes **51.4830555556 CPU-hours**, leaving **2448.5169444444 CPU-hours**
under the2500-hour cap. Replace reservations by whole-allocation actual
usage after completion, never adding step timings again. The original
120-hour deadline remains2026-10-10T23:29:34.346457+00:00.

Root will prepare the fresh `outputs/code/remaining_controls_selected/20261005-full01/`
metadata family and submit it once after bounded source review. The dispatch
does not retry failed actions, change scientific settings or open shared/OOD
data. The selected-partition check accepts only membership in the explicitly
requested list, since Slurm reports the single partition actually allocated.
No scientific code or tests were run on a login node.

## 2026-10-05 23:52 UTC — pending control job eligible for public normal partition

**Observed outcome,23:53UTC:** the fresh scheduler query found46680856 already
RUNNING in`russpold`, so **no partition update was performed**. Its actual
shell is`sh04-04n24.int`, job46680856,2CPUs. Updated EP04 ASTRA validation
passed; all four focused encoding checks passed in0.344s. The two scientific
workers started, and the sensitivity log had recorded160rows at63.6s. This
is progress, not completed controls or authority for full remaining dispatch.

The23:50UTC`sh_part`discovery confirms public`normal`and existing`russpold`
both support the selected2CPU/8GiB/50-minute job. The last observation of
**46680856** was PENDING(Resources) in russpold. Select expansion of this
same pending job's partition eligibility to`russpold,normal`, after verifying
it remains pending. Resources, job identity, scientific workers, dependency
and budget reservation stay fixed. This is normal scheduler eligibility,
not a duplicate job or preemptible owners execution. Record the update result
below; if it has already started, leave its running allocation in place.

## 2026-10-05 23:46 UTC — fixed_panel — all400 observed comparisons complete

Observed-panel parent **46675158** completed with exit0 in **945seconds** on
one CPU. The [saved report](runs/observed_native_panel/20261005-observed01/observed_native_panel.json)
finished at`2026-10-05T23:44:53.637265+00:00`after926.321s process time.
Root and independent review confirm the exact400 Cartesian rows across the
five frozen participants,25 folds,four models andfour ROIs:300 primary and
100EVC rows. All native and low-level partial rows are evaluable, with no errors,
missing EVC images or unexpected exclusions. Each participant's folds cover
all4712 images; row/fold image support, voxel/training-context bindings and
model declarations match, as does the complete approved low-level full01 input.
No candidate screen, null threshold or lock was performed. Required controls
remain incomplete, so this is not a selection or eligibility conclusion.

The first primary evaluation started at
**2026-10-05T23:29:34.346457+00:00**. Its original120-hour deadline is therefore
**2026-10-10T23:29:34.346457+00:00**; later stages do not reset it. Process peak
RSS is1263676KiB; Slurm batch MaxRSS is4193440K. These are separate reported
measurements, not evidence of a diagnosed memory failure.

Charge **0.2625 allocated CPU-hours once**, replacing the previous0.75-hour
reservation. Explicit completed EP04 charges now total **4.7872222222CPU-hours**.
With the deliberately conservative prior shared-parent bound and unchanged
46680856 reservation, total conservative exposure is **13.9830555556CPU-hours**,
leaving at least **2486.0169444444CPU-hours** before further reservations.
Do not add batch/extern timings again. At23:46:45UTC **46680856** was
PENDING(None) with its dependency satisfied; its two scientific workers and
separately scoped EP12/EP19 reporting actions remain unstarted.

## 2026-10-05 23:42 UTC — remaining controls dispatch prepared without resource selection

The [explicit residual plan](code/remaining_controls_plan.json),
[metadata/task dispatcher](code/remaining_controls_tasks.py) and
[array wrapper](code/remaining_controls_array.sbatch) passed root and independent
static review. The plan uses the exact frozen sub01/03/05/06/07 roster and all
25 folds, preserves the first encoding cell and the null prefix, and tiles only
permutations256:10000/geometry8:1000 for sub01/fold0 plus the complete ranges
for the other24cells. A sequential-ID assumption was caught and corrected in
preparation before any execution. Complete-result collectors remain authoritative.

Root must supply measured chunk lengths, process/walltime caps, memory,
concurrency, a verified partition and remaining CPU-hour authority after the
first-pass result. Preparation creates fresh explicit task/report lists and
prints a submission command; it never submits. Each array slot runs one existing
scientific runner once, with no overlap with retained draws, duplicate output,
automatic retry or successor. One CPU per slot and at most48 EP04 concurrent
cores remain the ceiling. No settings, additional CPU reservation or submission
have been selected, and no code/tests were executed during this preparation.

## 2026-10-05 23:39 UTC — existing control allocation also reads one EP19 documentation page

At23:37UTC **46680856** remained PENDING(Dependency), while46675158 was
RUNNING at8:08. Root adds a separately scoped EP19 public-document action to
the pending job's externally read sensitivity/reporting worker, after the
existing EP12 figure. The exact193612-byte WAY supplementary PDF and only its
first page are selected in EP19's own prospective log. This uses a90-second
external cap plus10-second kill grace, with separate status and output. It
reads no scientific arrays or EP04 outputs and does not change the EP04
sensitivity limit. Worst-case sensitivity/reporting/document time is2740s
plus parent startup and optional40s ASTRA validation, within the existing
3000s allocation. The other worker remains bounded at2715s. No new job or
CPU reservation is introduced; charge the eventual whole parent once here.

## 2026-10-05 23:31 UTC — full low-level coverage complete; observed panel running

Job **46674207** completed with exit0 in **5700 seconds**, using one CPU.
The [full preparation report](runs/source_readiness/20261005-low_level_full01/low_level_preparation_report.json)
records complete coverage of all **23,560** approved development images,
with **3150 float64 features per image**. Root and independent review verified
the exact approved source/metadata paths, finite checks before counting a row
complete, and no excluded shared/OOD image reads, neural reads or candidate
scores. The report finished at23:27:18 UTC. Process elapsed time was5690.97s;
Slurm reports984348KiB peak RSS. Those component times are not extra charges.

Charge **1.5833333333 allocated CPU-hours once** for the whole parent,
replacing its earlier3-hour reservation. Explicit completed EP04 charges now
total **4.5247222222 CPU-hours**. With the earlier deliberately conservative
shared-parent bound and the unchanged observed/control reservations, total
conservative exposure becomes **14.4705555556 CPU-hours**, leaving at least
**2485.5294444444 CPU-hours** under the2500-hour ceiling. Do not charge the
batch/extern records again.

At the23:31 UTC scheduler observation, dependent **46675158** was RUNNING
on`sh04-04n22` at1:45 elapsed; **46680856** remained dependency-held. Primary
panel completion and candidate decisions are not yet observed. The first
evaluation time recorded by the observed runner owns the unchanged120-hour
clock; this completion does not reset it.

## 2026-10-05 23:19 UTC — resource review before remaining controls

Root and an independent reviewer reconciled the existing parent-only accounting
before selecting any full-null or remaining-encoding reservation. Explicit
completed EP04 charges total **2.9413888889 allocated CPU-hours**: source tasks
46436056 (1.3972222222), preprocessing46438525 (0.23), context46654794/46655460
(0.1066666667), RSA parents46657948/46658777/46660232 (1.15), and representative
46673515 (0.0575). This is a summary of existing charges, not another charge.

For conservative dispatch authority, retain the earlier whole-shared-parent
bound **9.1563888889 hours**, including its deliberate over-attribution, then
add the later context/RSA/representative total **1.3141666667 hours**. Existing
reservations remain **3 hours** for46674207, **0.75 hours** for46675158 and
**1.6666666667 hours** for46680856. Total conservative exposure is therefore
**15.8872222223 hours**, leaving at least **2484.1127777777 hours** of the2500
CPU-hour ceiling before further reservations. Replace each live reservation
with its actual whole-parent charge after termination; never add overlapping
child/extern timings or shared EP01-owned diagnostics again. No new work is
reserved by this review. Remaining controls still require the selected first
pass's measured cost, and the original120-hour clock begins only at the first
primary panel evaluation.

## 2026-10-05 23:14 UTC — candidate_decision — complete-panel consumer prepared

The [screen consumer](code/screen_registered_panel.py) and seven focused checks
passed root and independent static review. It admits the complete observed,
null, four-sensitivity and encoding artifacts and the existing five matched
training-RSA/context reports. It derives each frozen pass/fail numerically,
including every partial-RSA participant leave-one-out effect, and publishes
all participant/ROI effects and concrete diagnostic tables. The completed-null
score arrays supply the unchanged unconditional selection and signed higher95%
thresholds; no outcome-selected subset or encoding rescue is allowed.

The consumer publishes at the single fixed
`runs/panel_decision/registered_panel_decision.json` only after complete
admission. It serializes and fsyncs an owned temporary file, then creates the
final name exclusively without overwriting any existing decision. Input-path
guards precede reads. Admission failure is not a candidate decision. A selected
row remains pending the registered exact full rerun; lock and OOD access remain
unauthorized. The consumer itself is prepared, **not executed or selected for
launch**, while the required scientific calculations remain incomplete.

Its seven small equation/coverage/publication checks are bundled after the
other diagnostic actions in pending **46680856**, with a separate45-second cap
and status. The worst-case `encoding_null` worker totals2715 seconds including
all external kill grace, within the existing50-minute parent envelope after
the optional40-second ASTRA validation bound. Sensitivity-plus-EP12 reporting
totals2640 seconds plus the same parent startup. No new allocation, enlarged
science budget, full-null dispatch or decision has been created. Runtime
checks remain pending; there is no automatic retry or successor.

## 2026-10-05 23:09 UTC — controls_diagnostics — encoding collector prepared

The [encoding collector](code/collect_identical_fold_encoding.py) and three
focused checks passed independent static review. It requires all 25 explicit
unit reports and 400 model/ROI rows, matching the frozen training complement,
evaluation IDs, voxel counts, preprocessing/context and original panel clock.
It preserves negative or undefined R2 and uses equal fold/primary-sector/person
weighting. All three VLM-versus-DINO RSA/encoding direction comparisons and
100 descriptive high-level/EVC cells retain the existing reliability context;
encoding cannot choose or rescue a native RSA candidate.

The three focused collector checks are appended to the already-selected
`encoding_null` worker in **46680856**, with a separate 45-second cap and exit
status. Existing null and encoding scientific work, 50-minute parent resources
and single accounting are unchanged. Full collection is prepared, not launched;
its 25 completed inputs still require the measured remaining computation.
Changed EP04 ASTRA validation remains inside that existing compute allocation.

## 2026-10-05 23:04 UTC — controls_diagnostics — complete-null collector prepared

The [registered-null collector](code/collect_registered_nulls.py) now accepts
explicit completed unit report paths and requires disjoint complete coverage
of all 25 participant/fold cells, 10,000 image permutations and 1,000 random
geometries per VLM. It checks the saved scores against their declared coverage,
exact observed support, model identities and original 120-hour clock before
pooling the Fisher-z components. It reuses the frozen unconditional three-VLM
gain argmax, signed unsubtracted selected alignment, higher 95th percentile
and strict comparison rule. It does not screen or select an observed candidate.

Independent static review passed. Four focused coverage/aggregation checks are
selected at the end of the existing `encoding_null` worker in pending job
**46680856**, with a 45-second process cap and separate exit status. The
externally read worker script was updated while the job was dependency-held;
the submitted parent resource envelope remains 2 CPUs / 8 GiB / 50 minutes.
This fits its reserved time alongside the two existing 1200-second scientific
units. No new allocation, full-null dispatch or collector execution is selected.
Runtime checks remain unexecuted until that compute job starts.

The sensitivity worker also bundles a separately scoped EP12 reporting action
after its EP04 computation: render the completed observed-description CSVs,
with a 120-second internal / 160-second external cap and 10-second grace.
Each action has a separate status in `sensitivity_reporting_exit_status.tsv`.
The scientific sensitivity cap remains 2400 seconds; the combined worst-case
worker bound remains below the existing 50-minute parent envelope. Neither
episode consumes the other's scientific outputs. Charge the shared allocation
once here after termination; no separate plot job or duplicated CPU charge.

## 2026-10-05 — controls_diagnostics — substantive first pass selected

Submitted as **46680856** with `afterok:46675158` and invalid-dependency
cancellation. This schedules the reviewed first pass; no control result is
yet observed. Root remains the single monitor for the dependency chain.

Select [controls_first_pass01.sbatch](code/controls_first_pass01.sbatch) after
successful observed-panel job **46675158**, with `afterok` and invalid-dependency
cancellation. Request **2 CPUs / 8 GiB / 50 minutes**, zero GPUs, at most
**1.6666666667 allocated CPU-hours**. Two exclusive one-CPU/4-GiB steps each
use six numerical thread caps of one. This is within the existing shared
budget; it does not reserve or launch the remaining nulls or encoding cells.

The first step computes the complete four registered real-data sensitivities,
**1600 rows**, with a 2400-second internal cap. The second runs the four focused
ridge checks (45-second cap), then **sub-01/fold0 encoding, 16 rows**, with a
1200-second cap; it next checks cached null arithmetic (45-second cap) and runs
the retained **256 permutation / 8 random-geometry draws per VLM** at the same
fold with a 1200-second cap. The checks address the new numerical code only;
passed original RSA/low-level checks are not rerun. Scientific outputs use
fresh paths `native_sensitivities/20261005-all01`,
`identical_fold_encoding/20261005-pilot01` and
`registered_nulls/20261005-prefix01`, all under `outputs/runs/`.

Final null/combined-wrapper static review passed before submission. The null
runner's missing `--low-level-dir` CLI argument was corrected to bind the
exact approved full01 path, and its four focused cached-arithmetic checks are
now present. Root reviewed the cached mean/SVD and rank-permutation expressions;
their actual execution and parity checks remain inside the selected allocation.

Sensitivity and encoding implementations passed independent static review.
Null implementation and final combined-wrapper review must finish before
submission. Site `module spider` confirms `py-scipy/1.12.0_py312` under `math`
for Cholesky factor/solve; load it alongside the existing Python3.12.1,
NumPy1.26.3 and Pillow10.2.0 modules. No installation is needed. Source syntax,
new focused arithmetic checks and changed ASTRA validation occur on compute
inside this substantive allocation, not in a separate checking-only job.

All actions preserve independent exit statuses and incomplete outputs. The
same original panel clock applies, including the120h limit. There is no
automatic retry or partial-result promotion. Successful encoding and null
prefixes are retained for disjoint remaining work sized from measured cost.

## 2026-10-05 22:00 UTC — controls_diagnostics — fixed null scope and parallel preparation

Before any primary score, explicitly bind the registered random-geometry
input to each exact eligible participant/evaluation-fold raw embedding matrix,
as recorded in [the source handoff](source_handoff.md) and policy. This matches
the covariance of the geometry actually scored. Preserve all fixed counts,
seeds, the existing centered SVD/Haar generator, global unconditional null
argmax, `method="higher"` percentile and strict exceedance. No synthetic-cohort
or recovery program is selected.

Three agents independently prepare the registered identical-fold ridge
diagnostic, four deterministic real-data sensitivities, and a retained prefix
of the required null draws. Root will select a finite combined allocation
after static review. Null prefix proposal: sub-01/fold0, permutation draws
0–255 and geometry draws 0–7 per VLM, preserved as part of the final fixed
counts. Full null reservations wait for measured cost; no count reduction,
approximation, expanded draw campaign or automatic remainder is selected.

BR `pipeline_plan_review` was consulted with the public methodological plan
only, `use_kg=false`. Its basic rule checker returned `approve`, low risk and
no findings, with no enforced rules and no KG rules consulted. This limited
advisory check does not validate the mathematics, authorize work or launch any
job; independent code review remains the relevant implementation evidence.
The first call rejected a `stages` rather than `steps` input; the corrected
four-step plan returned the result above. Execution remains standalone Slurm,
not a canonical BR campaign or autoresearch-harness launch.

## 2026-10-05 21:57 UTC — fixed_panel — complete observed panel selected

Select one complete observed native/partial-RSA panel after successful full
low-level extraction **46674207**, with `afterok` and invalid-dependency
cancellation. The driver independently requires all 23,560 finite feature
rows and all 25 frozen metadata joins before its first neural evaluation.
This is the already authorized fixed-panel analysis, not a new method choice.
Independent static review passed after exact low-level source bindings and
native-only EVC diagnostic completeness were corrected. Existing numerical
checks are reused; source AST and wrapper syntax checks run inside the job.

The [driver](code/run_observed_native_panel.py) reads saved training-fitted
evaluation patterns, four native model embeddings and the approved nuisance
features. It retains all **400 fixed participant/fold/ROI/model rows**, native
and partial correlations, and equal-weight effects without screening or winner
selection. Nulls, remaining sensitivities and identical-fold encoding remain
required before the single candidate decision. The report's
`first_panel_evaluation_utc` starts the existing 120-hour clock.

Select **1 CPU / 4 GiB / 45 minutes**, no GPU, internal 2400s and external
2460s plus 10s grace. The cached implementation is estimated at about 633M
ranked entries versus approximately 1.42B in the completed training table;
each evaluation fold has about 942 images, with forecast peak memory below
2 GiB. This envelope uses at most 0.75 allocated CPU-hours of the existing
2500-hour budget. Six numerical thread caps are one. No automatic retry or
partial-panel promotion. The [fixed wrapper](code/observed_native_panel01.sbatch)
writes `runs/observed_native_panel/20261005-observed01`. ASTRA validation is
included in the substantive allocation and does not reopen prior math checks.

Submitted as **46675158**, with `afterok:46674207` and
`--kill-on-invalid-dep=yes`. Final wrapper review identified the existing
low-level module's top-level Pillow import; the wrapper now loads the already
qualified Pillow 10.2.0 module. No new environment or package was installed.
Submission is not an observed score or clock start.

## 2026-10-05 21:53 UTC — full low-level extraction running

Slurm accepted the selected full extraction as **46674207**. At 21:53 UTC it
was RUNNING on `sh04-04n22`, with the selected 1 CPU / 2 GiB / 3h limits.
The unchanged full-pass wrapper writes
`runs/source_readiness/20261005-low_level_full01`; stdout/stderr are
`runs/source_readiness/20261005-low_level_full01-46674207.{out,err}`.
No observed panel is yet submitted; complete finite coverage is required first.

Representative allocation **46673515 COMPLETED 0:0**, 207 seconds with one
allocated CPU: charge **0.0575 allocated CPU-hours once to EP04**. The attended
PTY exited and is closed. Its step peak RSS was 284856 KiB. This parent also
contained the small, separately scoped EP21 ASTRA validation; do not charge
that step again to EP21 or add the extern-step duration.

## 2026-10-05 21:48 UTC — representative complete; measured full extraction selected

The approved representative completed all **60/60 images** in **21.9763s**
in allocation **46673515**, on verified `sh04-04n22.int`, with exit 0.
The [report](runs/source_readiness/20261005-low_level_representative01/low_level_preparation_report.json)
records 1.6671s source reads and 16.6345s decoding/transformation, 3150 float64
features per image, NumPy 1.26.3, Pillow 10.2.0 and h5py 3.10.0. Exact complete
development metadata joined before these selected image reads. No excluded
image element, neural value or candidate score was opened. Changed reader and
wrapper syntax checks and EP04 ASTRA validation passed in this allocation.

Scaling measured variable time predicts 7186s (about 2h); scaling total elapsed
predicts 8629s (about 2.4h). Neither fits the initial 90-minute allocation.
Finish that attended shell after its selected checks and submit one complete
pass with **1 CPU, 2 GiB, no GPU, 3h walltime**, internal 10680s and the existing
external 10s/10s grace, using [the fixed wrapper](code/low_level_full01.sbatch).
This is the originally approved full extraction sized from observed cost,
within the existing EP04 budget; no scientific recipe or sample size changes.
The fresh output is `runs/source_readiness/20261005-low_level_full01`.
There is no automatic retry, incomplete-coverage scoring or padded wait.
This full pass must finish all 23560 rows before the primary panel can score.

## 2026-10-05 — approved low-level control; representative and full extraction selected

The user's “agree， please go ahead” approves the two linked pending proposals,
including this episode's [fixed RGB block-moment control](low_level_control_proposal.md).
Reuse the accepted September 22 LAION source/DUA and the frozen metadata-fold
attempt04. Select only the 60-image representative, followed by all 23,560
unique-development images using the unchanged 5/10/20 grids and float64
3150-feature recipe. Existing analytic checks are reused; no new simulation
program, pixel display, regular-shared/OOD read or neural scoring is selected.

Root will request one bounded attended Slurm allocation with one CPU, 2 GiB,
no GPU and 90 minutes, allowing the representative and substantive extraction
to share an allocation. The proposal estimates tens of minutes for the full
pass. The representative retains its 300-second internal bound; its observed
cost will determine the full-pass bound before execution. If that pass cannot
fit the remaining allocation, do not begin an inadequately bounded full pass.
All six thread limits are one. Output attempts are
`runs/source_readiness/20261005-low_level_representative01` and
`runs/source_readiness/20261005-low_level_full01`; partial outputs are retained.
Source execution waits for the approved policy binding, not another user
confirmation. Root owns this allocation, commands, monitoring and accounting.

Slurm accepted allocation **46673515** and reported it queued for resources.
The root-owned attended PTY is open; no compute shell or source read is yet
confirmed. No duplicate submission or unrelated allocation is selected.

## 2026-10-05 20:20 UTC — all five participants' matched training table complete

Analyses `training_neural_context` and `training_context_display`. All four
remaining workers in **46660232** returned zero. Their complete reports and
the preserved sub-01 pilot yield **400 unique/evaluable finite rows, 25 folds,
100 participant-fold-ROI units**, with no errors or missing participants.
Every model shares the same saved training image/repeat/voxel support within
each unit. Independent review confirms the exact fixed row coverage, source
bindings and evidence limits. New process timings / peak RSS:

| Participant | Seconds | KiB |
| --- | ---: | ---: |
| sub-03 | 669.600444524 | 1,224,604 |
| sub-05 | 657.163274212 | 1,139,832 |
| sub-06 | 654.171461842 | 1,185,696 |
| sub-07 | 648.879663292 | 1,127,904 |

Reports remain under
`runs/training_context_rsa/20261005-rsa-remaining-46660232/<participant>/training_context_rsa.json`.
Do not add process timings as separate allocation charges. The parent still
runs its source actions; once-only whole-parent accounting follows termination.

Root reused one available CPU in that same live allocation as attended step
**46660232.5**, verified `sh03-06n06.int`, one CPU and the numerical thread
limits. The display finished in **3.709551446 seconds**, with **400 CSV rows,
80 plotted participant summaries and no errors**. Root inspected the PNG.
Artifacts: [CSV](figures/training_context_rsa/20261005-display-46660232/training_context_rsa.csv),
[PNG](figures/training_context_rsa/20261005-display-46660232/training_context_rsa.png),
[SVG](figures/training_context_rsa/20261005-display-46660232/training_context_rsa.svg),
and [display record](figures/training_context_rsa/20261005-display-46660232/display.json).
ASTRA validation passed for EP03/04/19 at parent setup and again for EP04's
new display object in this step. The attended step exited; no test session or
additional allocation remains for display.

This completes the required descriptive matched training-context table. The
figure shows the saved correlations by participant, ROI and model; overlapping
training folds and different participant image pools do not supply independent
replications. No model was selected, no held-out primary score/null was
computed, and the primary 120-hour clock remains unstarted. The next primary
stage still needs the pending separately approved raw-image low-level control.
No further training-context run or automatic successor is selected.

At **20:22:55 UTC**, Slurm confirms shared parent **46660232 COMPLETED 0:0**,
**868 seconds / four allocated CPUs**. All four RSA children, EP03 repair,
display step and EP19 remaining support step completed successfully. Charge
the entire parent **0.9644444444 allocated core-hours once to EP04**, with
no EP03/EP19/display duplicates. The unused three-core-hour reservation is
released. Across this RSA sequence, setup failure 46657948, pilot 46658777
and remainder 46660232 consumed exactly **1.15 allocated core-hours** in total.
No active compute session or successor remains from this sequence.

## 2026-10-05 20:17 UTC — completed-table display prepared inside the same allocation

Analysis `training_context_display`. To make the new real-MRI comparison
inspectable, prepare `code/summarize_training_context_rsa.py` against only the
exact sub-01 pilot and four remaining reports. It requires all **400 complete,
evaluable, finite rows**; any missing participant stops display. The CSV copies
the saved per-fold RSA and support/reliability/NC context. A fixed-order 2×2
ROI figure shows five participant dots per model, using `tanh(mean(saved z))`
across the five overlapping training folds. It makes no winner, contrast,
uncertainty, independent-image-pool or held-out-performance claim.

Root reviewed the small display implementation. Select one attended
one-CPU/2-GiB step in **46660232** only after all four RSA workers finish and
the parent still has time/resources while its EP03/EP19 steps run. The display
has a 90-second internal / 110+10-second external bound; the step is at most
two minutes. No standalone allocation, new data read, scientific fit or rerun
is selected. Fresh outputs will be
`outputs/figures/training_context_rsa/20261005-display-46660232/`.
The changed ASTRA display object can be validated in that same compute step;
validation is not a gate on the already completed science.

## 2026-10-05 20:04 UTC — sub-01 RSA complete; remaining four selected

Analysis `training_neural_context`, mode `matched_training_context_RSA`.
The [sub-01 report](runs/training_context_rsa/20261005-rsa-pilot-46658777/sub-01/training_context_rsa.json)
completed all **80 evaluable rows / five folds**, with no errors, finite RSA
values and identical saved image/repeat/voxel support across models. Runtime
was **490.139738456 seconds**, process peak RSS **1,143,280 KiB**. Root and an
independent worker reviewed the complete report and its evidence limits.
This is descriptive training context, not held-out performance, candidate
selection or five-participant completion.

The shared parent **46658777 FAILED 1:0 after 616 seconds / one CPU** because
EP03's first-route step returned 2; EP19 P2 and EP04 RSA both returned 0.
Preserve both completed scientific results and the interrupted EP03 trace.
Charge this whole parent **0.1711111111 allocated core-hours once to EP04**;
EP03 and EP19 do not charge it again. Slurm's batch MaxRSS was 3,414,024 KiB,
also within the 7-GiB request. The earlier setup failure's charge remains.

Select the reviewed [remaining-participant pack](code/training_context_rsa_remaining_participants.sbatch):
exactly **sub-03, sub-05, sub-06 and sub-07**, each all five folds/four ROIs/four
models, **320 new rows**. Preserve sub-01 without a rerun. Four exclusive
one-CPU/7-GiB workers use the unchanged driver and scientific functions, six
thread limits of one, and the same 1500-second internal / 1560+10-second
external caps. The measured pilot supports this envelope. The finite parent
requests **4 CPUs / 28 GiB / 45 minutes**, maximum **3 allocated core-hours**.
Even conservatively counting historical whole-parent allocations and this
reservation gives 12.4486111111 core-hours, below the 2500-core-hour ceiling;
that eligibility bound does not add duplicate charges to other episodes.

After all four RSA workers settle, separately selected EP03 first-route repair
and EP19 remaining P3–P12 S1 support steps run sequentially, with their own
episode-local scientific destinations. Three new entrypoints receive syntax
checks and the changed ASTRA objects receive bounded non-gating validation
inside this allocation. The combined capped sections total 2430 seconds,
leaving 270 seconds in the parent for setup/closure. Every substantive action
keeps its exit status; a failure does not erase other completed actions.
The parent is charged once here at termination. No retry, simulation, new raw
image access, held-out primary score or automatic successor is selected.
Raw-image control approval requested at 19:38 UTC remains pending.

Submission returned **46660232** at **20:05 UTC**. Operational outputs are
`runs/training_context_rsa/20261005-rsa-remaining-46660232/`; each participant
has its own `training_context_rsa.json`. Slurm logs remain in the shared
`runs/training_context_rsa/20261005-submission01/slurm-46660232.{out,err}`.
Submission is not a completion claim.

## 2026-10-05 19:52 UTC — memory-unit request corrected before any source processing

Parent **46657948 FAILED, exit 1:0**, after 26 seconds with two allocated CPUs.
Its sole stdout line records `CPUS=2`; the exact-one-CPU guard stopped before
module loading, syntax checks, source steps, output-root creation or RSA.
No scientific attempt or new source access occurred. Charge its
**0.0144444444 allocated core-hours once** and preserve both Slurm logs.

The observed job request was `cpu=1,mem=8G` but the allocation was `cpu=2`.
Live partition metadata for both eligible partitions states
`MaxMemPerCPU=8000`; `8G` requests 8192 MiB, exceeding that per-CPU limit.
This identifies a resource-unit request error, not a cluster fault or numerical
failure. Reduce the single-CPU parent request to **7 GiB (7168 MiB)**, still
well above the estimated under-2-GiB working payload. Retain the 40-minute
parent, all three exact child scopes, thread limits and internal/external caps.
Select one corrected setup submission with a fresh job-specific destination.
It does not repeat a scientific fit or source read. The prepared, unselected
four-participant wrapper is likewise corrected to 28 GiB total / 7 GiB per
single-CPU child, with no remainder dispatch selected.

Corrected submission returned **46658777** at 19:52 UTC. Scientific and
operational outputs use this new job ID; failed setup 46657948 stays preserved.

## 2026-10-05 19:47 UTC — matched training-context RSA pilot selected

Analysis `training_neural_context`, mode `matched_training_context_RSA`.
The adopted [preprocessing recipe](neural_preprocessing_proposal.md#5-reliability-and-provider-formula-nc-fitted-on-training-repeats)
requires a separately labeled table of raw model/ROI RSA, repeat reliability
and provider-formula NC on identical training image IDs and per-ROI voxel/repeat
support. Root selects the first **sub-01, all five folds × four ROIs × four
released models = 80 rows**. This is descriptive training context, not held-out
performance or primary-panel evaluation, and cannot select a candidate.
All five participants' completed reliability/NC artifacts remain unchanged.

The new [driver](code/run_training_context_rsa.py) reuses saved equal-repeat
mean patterns, saved validity/voxel support and the qualified development
embedding handoff. It joins exact image IDs, participant/fold/source-component
identities, original pattern rows and embedding rows; checks the common
training intersection, actual session/row/image/beta-axis repeat sets and
per-ROI original voxel IDs against the completed context; then reads only
those training pattern/embedding rows. It does not reopen beta arrays, raw
images, shared/OOD outcomes or released pooled noise-ceiling maps. Original
`native_rsa.py` cosine/correlation RDM, upper-triangle extraction, exact tied
ranks, Pearson-on-ranks and Fisher transform perform all arithmetic. Four model
rank vectors are cached per fold and one neural rank vector is retained at a
time. Existing NC/reliability summaries accompany each row without refitting,
normalization, voxel selection or an independent-error/strict-RSA-ceiling claim.
Undefined geometry remains explicitly not evaluable; incomplete prefixes do
not complete the participant or the five-person table.

For at most 3,770 training images, each RDM is approximately 114 MB and each
7,104,565-edge rank vector approximately 57 MB in float64. The fixed participant
requires 40 RDM/rank constructions and 80 pair correlations. Eight GiB leaves
room for the selected ROI matrix, ranking work arrays and four cached model
ranks. Ranking cost remains unmeasured at this size, so root selects a finite
**1 CPU / 8 GiB / 40-minute parent**, with six numerical thread limits equal
to one and no GPU. The RSA child has **1,500 seconds internal and 1,560 seconds
external plus 10 seconds termination grace**. Selected EP03 first-route and
EP19 P2 support steps run sequentially beforehand in this same parent, with
their separate 100+10-second and 190+10-second external envelopes. The combined
1,880-second child envelopes leave 520 seconds for setup and closure. This
is one shared allocation, charged once to EP04; no EP03/EP19 duplicate charge.

The [shared wrapper](code/training_context_rsa_pilot.sbatch) will create
`runs/training_context_rsa/20261005-rsa-pilot-JOBID/`, with the fresh scientific
child `sub-01/training_context_rsa.json`, five training-ID lists and shared
operational logs. Command, from EP04:

```bash
timeout --signal=TERM --kill-after=10s 1560s python3.12 -u -B \
  outputs/code/run_training_context_rsa.py --participant sub-01 \
  --output-dir outputs/runs/training_context_rsa/20261005-rsa-pilot-JOBID/sub-01 \
  --max-seconds 1500
```

The existing Python 3.12.1 / NumPy 1.26.3 module stack suffices for RSA; the
shared wrapper also loads the already available scikit-learn module for its
other selected step. Reuse passing native-RSA mathematical tests. Root and
independent source review cover forwarding/matching assertions; root will
syntax-parse the new drivers inside Slurm. There is no additional fixture
program, benchmark, null simulation, retry, automatic continuation or change
to scientific thresholds. Remaining four-participant resources and dispatch
stay unselected until the pilot's actual cost and complete support are known.

The conservative whole-parent historical upper bound is now 9.2630555556
core-hours after replacing earlier reservations with actual sub-01/remaining
costs. Adding this **0.6666666667 core-hour reservation** gives
**9.9297222223**, below the 2,500-core-hour budget. This is an upper-bound
authority check, not another historical charge. No primary panel evaluation
has begun the 120-hour clock. Root requested raw-image extension approval at
19:38 UTC; it remains pending and the prepared low-level-control proposal is
not activated. The low-level control, held-fold primary panel and conditional
OOD audit remain separate. This entry records prospective selection, not
submission, completed RSA or any model advantage.

Submission returned **46657948** at 19:49 UTC, `normal,russpold`, with the
selected 1 CPU / 8 GiB / 40-minute envelope. Root and independent static
reviews passed for the three entrypoints and corrected launcher. Slurm logs:
`runs/training_context_rsa/20261005-submission01/slurm-46657948.{out,err}`.
Submission alone does not establish execution or completion.

## 2026-10-05 19:35 UTC — all five participants' training context complete

Analysis `training_neural_context`. Root's 19:34:59 UTC scheduler/accounting
observation establishes **46655460 COMPLETED, exit 0:0**, with all three
one-CPU steps also COMPLETED. The parent allocated three CPUs for 90 seconds:
charge **0.075 allocated core-hours once**, replacing the one-core-hour
reservation. Do not add child-process or step durations to this charge.
Together with sub-01 parent 46654794, the two new EP04 parents consumed
**0.1066666667 allocated core-hours**. Earlier shared and dedicated allocations
retain their existing charges.

The [participant exits](runs/training_neural_context/20261005-remaining-context-46655460/participant_exit_status.tsv)
are all zero. Root read the three saved reports: every participant has five
complete folds, four support-evaluable ROIs per fold, no errors and no primary
training/evaluation images missing EVC support. No raw arrays were reopened
or statistics recomputed during this review.

| Participant | Report | Process seconds | Common training images by held fold 0–4 |
| --- | --- | ---: | --- |
| sub-03 | [context](runs/training_neural_context/20261005-remaining-context-46655460/sub-03/training_context.json) | 74.428950 | 3770, 3769, 3769, 3770, 3770 |
| sub-06 | [context](runs/training_neural_context/20261005-remaining-context-46655460/sub-06/training_context.json) | 70.374335 | 3769, 3770, 3769, 3770, 3770 |
| sub-07 | [context](runs/training_neural_context/20261005-remaining-context-46655460/sub-07/training_context.json) | 74.208792 | 3770, 3770, 3769, 3769, 3770 |

Together with the completed [sub-01](runs/training_neural_context/20261005-sub01-context-46654794/training_context.json)
and preserved [sub-05](runs/training_neural_context/20261003-sub05-folds01/training_context.json),
this completes the training-only repeat reliability/provider-formula noise
ceiling components for all five participants: 25 folds and 100 ROI units.
Original formulas, training means/scales, repeat assignments, support, missingness
and biological units remain fixed. Passing mathematical checks were reused;
root and independent static reviews covered the two operational wrappers.

The ASTRA program now retains the original sub-05 output ID and adds separate
sub-01 and remaining-participant artifacts, with all three branches feeding
the diagnostic dependency. This navigation edit was statically reviewed; no
new ASTRA validator run is claimed or separate validation allocation selected.

These components do not complete the matched training-context RSA table,
fixed model panel, low-level control or OOD audit. They establish no model
advantage or strict Spearman-RSA ceiling. No panel evaluation has begun, and
raw-image extension approval remains outstanding. All selected context jobs
are now terminal; no retry or new simulation is selected.

## 2026-10-05 19:28 UTC — sub-01 complete; finite remaining-participant pack selected

Analysis `training_neural_context`. Root's19:27:50UTC scheduler/accounting
observation records **46654794 COMPLETED/0:0**, one allocated CPU for114seconds:
**0.0316666667 core-hours**, charged once and replacing its0.75-hour reservation.
Root checked the [completed sub-01 report](runs/training_neural_context/20261005-sub01-context-46654794/training_context.json):
**100.218019712s**, all260 chunks, all five folds and all20 ROI units complete
and support-evaluable, no errors and no EVC support gaps. Common training-image
counts for folds0–4 are3,770/3,770/3,769/3,770/3,769. The original formulas,
saved parameters, source rows and fixed support are unchanged. Actual full-run
cost is much lower than the earlier single-chunk projection; that projection
was a conservative forecast, not evidence about the historical timeout cause.
No RDM/RSA, model comparison or OOD audit is completed by this context result.

With sub-01 checked and sub-05 already complete, root now prospectively selects
one [remaining-participant pack](code/training_context_remaining_participants.sbatch)
for **sub-03, sub-06 and sub-07**. Each uses the **original unchanged**
`training_neural_context.py`, its exact completed preprocessing report under
`runs/neural_preprocessing/20261003-remaining-folds01/<participant>/preprocessing.json`,
and `runs/source_readiness/20261003-metadata_folds04`. Each computes all five
folds/four ROIs, with original256-column chunks and840-second internal cap.
No participant, voxel, fold, image or scientific-setting selection is added;
the completed sub-01/sub-05 artifacts and all historical partials are preserved.

The parent requests **3CPU/12GiB/20minutes** (maximum **1 core-hour**), with three
concurrent, exclusive **1CPU/4GiB** steps. Every numeric thread limit is one;
there is no GPU, requeue or retry. Each child has a **900-second external bound
plus10seconds termination grace**. The longest child envelope is910seconds,
leaving290seconds for module setup, launch and closure within the1200-second
parent. The single pack waits for all three actual child statuses, preserves
each stdout/stderr and `participant_exit_status.tsv`, and exits nonzero if any
child fails. It launches no second attempt or subsequent work.

Fresh outputs will be
`runs/training_neural_context/20261005-remaining-context-JOBID/<participant>/`.
The module stack is the already qualified
`devel python/3.12.1 math py-numpy/1.26.3`; original passing mathematical tests
are reused. Root owns static wrapper review, submission, monitoring and any
compute-side shell check. No new scientific code, fixture, test execution,
source read or scheduler action was performed by the wrapper author.
This entry records selection, **not submission or completion**.

The conservative whole-parent resource check is now
9.1563888889 +0.0316666667 actual sub-01 +1.0 new reservation =
**10.1880555556 core-hours**, below2,500. As previously stated, this is an
upper-bound authority check, not a duplicate attribution of shared parents.
No primary panel evaluation has started the120-hour clock. This pack completes
required training context only; matched training-context RSA, fixed-panel
evaluation, raw-image authority and the conditional OOD audit remain separate.

Root and independent wrapper review passed. Submission returned **46655460**
at 19:30 UTC, with the selected 3 CPU / 12 GiB / 20-minute bounds on
`normal,russpold`. Result root:
`runs/training_neural_context/20261005-remaining-context-46655460/`.
No completed sub-01 or sub-05 work is repeated; completion of the three new
participants remains unobserved at submission.

## 2026-10-05 19:23 UTC — full sub-01 training context selected within existing budget

Analysis `training_neural_context`. Root selects completion of the existing
sub-01 scientific unit: **all five frozen folds and all four registered ROIs**,
with the same saved training parameters, validity/repeat assignments, original
256-column order and original moment functions. No further isolated chunk,
new scientific unit, parameter choice, participant selection, model scoring,
audit or automatic continuation is introduced.

The completed [one-chunk diagnostic](runs/training_neural_context/20261005-chunk-diagnostic-46651952/chunk_diagnostic.json)
in shared EP01 closeout46651952 returned in **13.682355882s wall/5.589395486s
process CPU**, peak394,936KiB, without errors. It read30 selected training
slices/3,859,456 values for15,076 presentations,3,770 training images and256
ventral columns, then returned one unchanged moment call. No scientific
values were exported. Source loading took6.652679s; the chunk's beta slicing
and standardization took about4.612s and moment arithmetic0.402974s.
The1CPU/37-second shared closeout cost **0.0102777778 core-hours** is already
included in EP01 once, and is not charged again here.

The existing sub-01 preprocessing report records, in every fold,3,767 ventral,
3,281 lateral,2,170 dorsal and3,621 EVC retained columns:15+13+9+15=52 chunks
per fold, **260 chunks total**. A same-cost extrapolation from the measured
5.018153-second chunk is about1,305 seconds plus preparation/output, roughly
22 minutes. This is a resource forecast, not a runtime guarantee: cache and
ROI layout can change later costs. It does not explain the historical missing
first checkpoint, which still cannot identify an infrastructure fault or a
particular stalled operation.

**Authority and budget checked before implementation.** GOAL.md:349–355 and
SEARCH_POLICY.yaml:147–152 permit2,500 core-hours,0GPU-hours,48 concurrent CPU
cores,256GB per evaluation and1,500GB scratch. GOAL.md:351 starts the120-hour
clock at the first panel evaluation. Existing records contain **zero panel
evaluations**; source preparation and training-context calculations have not
started that clock. Required context is specified in GOAL.md:331–334 and
SEARCH_POLICY.yaml:229–232; DATASETS.md:128–142 retains the training-only and
excluded-outcome boundaries. The old180/190-second attempt and the original
runner's840-second maximum were operational selections, not episode limits.

Root's19:20:05UTC parent-only scheduler accounting supplies these conservative
historical shared-parent resources:

| Parent | Allocated CPUs | Elapsed seconds | Core-seconds |
| --- | ---: | ---: | ---: |
| 46425841 | 4 | 3346 | 13384 |
| 46430982 | 1 | 2432 | 2432 |
| 46433944 | 2 | 2368 | 4736 |
| 46436630 | 1 | 2076 | 2076 |
| 46438625 | 1 | 2044 | 2044 |

These whole shared parents total6.8533333333 core-hours. The five dedicated
source tasks46436056 used5,030 core-seconds (1.3972222222 hours); dedicated
two-CPU preprocessing46438525 used828 core-seconds (0.23 hours). Adding even
the **entire** current EP01 observed-behavior branch's0.6758333333 hours gives
an upper-bound resource check of **9.1563888889 core-hours**, or
**9.9063888889 including the new0.75-hour reservation**, below2,500.
This deliberately conservative eligibility calculation is **not an exact
episode attribution or a new/double ledger charge**. Shared costs remain
charged once in their coordinating ledger. No other current EP04 reservation
is recorded.

The new [operational continuation](code/run_training_context_sub01.py) preserves
the old runner and every prior output. It uses the existing helper functions
for source loading, saved validity, chronological repeats, `roi_context` and
summaries. Each call to original `roi_context` receives exactly the next
original256-column chunk; returned vectors are copied into their original
positions without additional arithmetic. Final ROI files retain the original
four NPZ keys and filenames. After each returned chunk, an atomic
`.inprogress.npz` retains the full axis with only its reported leading
`completed_voxels` available; trailing NaNs are uncomputed, not measured
undefined values. Completed ROI/fold flags and the overall completion flag
remain distinct. A terminal deadline or error preserves these partials and
does not release overall completion.

Selected resources: **1CPU/4GiB/45minutes**, all six numerical thread limits
one, no GPU or requeue. The fixed whole-process internal cap is **2400s**;
the external cap is **2460s plus10s termination grace**. The0.75core-hour parent
reservation is counted once. There is no automatic retry, resume or launch of
another participant. From EP04, using the already qualified
`devel python/3.12.1 math py-numpy/1.26.3` modules:

```bash
timeout --signal=TERM --kill-after=10s 2460s \
  python3.12 -B -u outputs/code/run_training_context_sub01.py \
  --output-dir outputs/runs/training_neural_context/20261005-sub01-context-JOBID
```

Root owns the [batch wrapper](code/training_context_sub01.sbatch), compute-side
syntax check, submission and actual accounting. Existing mathematical checks
are reused; no new scientific qualification or separate testing allocation
is selected. The new runner and these records were prepared on login only;
its author performed no import, test, source-array read or scheduler action.
No full-run execution is yet claimed.

Root and independent driver/wrapper review passed. Submission returned
**46654794** at 19:25 UTC, `normal,russpold`, 1 CPU, 4 GiB, 45 minutes,
no requeue. Logs are under
`runs/training_neural_context/20261005-submission01/slurm-46654794.{out,err}`;
the fresh result directory is `20261005-sub01-context-46654794`.
Submission establishes neither running nor completion; root owns monitoring.

## 2026-10-05 19:03 UTC — completed initialization; one fixed training chunk selected

Analysis `training_neural_context`. The selected header/metadata-only prefix
returned successfully in shared EP01 parent **46650095** on `sh02-06n08.int`:
**7.970102414 seconds**, 4,712 images, 30 sessions and 12,833 union voxels.
The [phase record](runs/training_neural_context/20261005-initialization-diagnostic-46650095/initialization.jsonl)
records zero numerical beta rows read, no saved fold parameters read, and no
training-context computation. This establishes the current loader prefix,
not the cause of the historical timeout. The shared parent cost is already
charged once to EP01 (**0.1133333333 core-hours**); EP04 adds no duplicate charge.

Correction to the earlier localization: the original runner's first `save()`
is after its first entire ROI calculation. No fold/report emitted therefore
does **not** establish a pre-fold stall. The unrecorded interval also includes
saved-parameter/repeat preparation, beta slicing and first-ROI moment
arithmetic. The historical attempt remains incomplete; no cluster fault or
completed reliability/NC computation follows from its missing report.

Under the user's real-data priority, root now selects a single finite chunk
through the unchanged scientific functions, using new
[diagnose_training_context_chunk.py](code/diagnose_training_context_chunk.py).
The fixed input is the same completed sub-01 preprocessing report and original
handoff, with `metadata_folds04`. Only **fold 0**, **the first registered ROI
(`laion-ventral`)**, and **the first 256 retained columns in original order**
are used. The existing saved training means/scales/support, repeat support,
presentation validity and chronological assignments are reused unchanged.
The selected beta slices contain only that fold's training presentations.
No refitting, mask choice, alternate chunk, evaluation/shared/OOD response,
raw image or embedding access is selected.

The new runner calls original `roi_context` and original `repeat_moments`.
Timing-only forwarding wrappers record phases before/after each metadata
stage, beta slice and moment call, and restore their process-local binding on
all exits. Returned numerical arrays are discarded without exporting NC,
reliability, residual moments or other scientific values. Operational counts,
wall/CPU timings, attempted/returned accesses and partial errors are retained
in `chunk_diagnostic.json` and stdout JSON lines. This is a localization/cost
diagnostic, not a completed context analysis, qualification claim or permission
to continue automatically.

The selected bound is **one actual CPU, 4 GiB, 180 seconds internal**, with
**190 seconds external plus 10 seconds termination grace**, all six numerical
thread limits one. Root will aggregate it in the shared ten-minute EP01
closeout after the fixed-fit packs; no standalone allocation or retry is
selected. From EP04, with a fresh output directory:

```bash
timeout --signal=TERM --kill-after=10s 190s \
  python3.12 -u outputs/code/diagnose_training_context_chunk.py \
  --output-dir outputs/runs/training_neural_context/CHOSEN_FRESH_ATTEMPT
```

Independent static review of the forwarding, frozen chunk scope and whole
deadline passed. No new import, test, beta access, scientific calculation or
scheduler action has been executed by its author or reviewer.
Root owns submission, observed outcome and shared-allocation accounting.

## 2026-10-05 18:49 UTC — shared parent setup failed before EP04; corrected parent submitted

Analysis `training_neural_context`. EP01 parent46649499 failed after23seconds
at a test-fixture syntax error, before any EP04 step or source read. Its cost
is charged once to EP01. Root corrected that one unmatched parenthesis and
submitted otherwise identical40-minute parent46650095 at18:49UTC. The selected
EP04 metadata-only prefix and its120-second/1CPU/4GiB bounds are unchanged;
its fresh destination uses46650095. No diagnostic execution is yet established.

## 2026-10-05 18:43 UTC — `training_neural_context`: initialization-only diagnosis selected

Under the user's instruction to prioritize bounded real-data work and stop
expanding simulated-data validation, the coordinator selected one task-local
diagnosis of the recorded sub-01 initialization stop. This is not a retry of
the reliability/NC analysis or an increase to its prior190-second bound.

Static source review identifies missing phase evidence: the original runner
creates its output directory before initialization but first publishes a report
only after an entire ROI calculation. The prior empty directory/no-fold record
does not distinguish the completed-handoff loader from later first-fold
initialization or output I/O. It establishes neither an expensive NC kernel
nor an infrastructure fault. The original attempt and all completed source,
preprocessing and sub-05 context artifacts remain preserved.

The new [diagnostic](code/diagnose_training_context_initialization.py) accepts
no source, participant or time overrides. From EP04 it reads the fixed
`runs/neural_preprocessing/20261003-remaining-folds01/sub-01/preprocessing.json`,
checks its completed five-fold metadata, then calls the existing
`preprocess_development_neural.load_completed_inputs` with that report's
original participant handoff and `runs/source_readiness/20261003-metadata_folds04`.
Only declared source/ROI identity metadata and beta memmap headers/shape/dtype
are inspected. No beta values are indexed, no saved fold parameters are read,
and no reliability, NC, RDM, RSA or model score is computed.

The selected execution is a sequential **one-CPU/4GiB step** inside the
coordinator-owned **40-minute EP01 real-data pilot allocation46649499**,
submitted18:44UTC and still pending at18:45:50UTC. It runs before EP01's
four-CPU fit. Fresh logs will be under
`runs/training_neural_context/20261005-initialization-diagnostic-46649499/`.
Independent static review passed; no diagnostic execution is yet observed.
All six numerical thread limits are one. JSON lines retain host/job, exact
phase, wall/CPU elapsed time and returned identity counts; stderr retains the
one-shot90-second faulthandler traceback, which exits if reached. The command
has a120-second external bound plus10seconds termination grace:

```bash
timeout --signal=TERM --kill-after=10s 120s \
  python3.12 -u outputs/code/diagnose_training_context_initialization.py
```

The separate [10-minute batch wrapper](code/training_context_initialization_diagnostic.sbatch)
is prepared only, not selected for submission. No padded runtime, repeated
diagnosis or automatic continuation is authorized. A successful return clears
only this initialization prefix; it does not resolve the unexamined later
first-fold path or establish completed training context. Independent static
review of the diagnostic and actual shared wrapper passed. No new test, import,
source-payload read or scheduler action was performed by the diagnostic author.
The shared parent allocation will be
accounted once in EP01, with no separate EP04 allocation recharge.

Existing source roles, accepted DUA, frozen image folds and scientific settings
are unchanged. Raw-image control extraction, neural-analysis resumption and
OOD access remain unselected by this diagnosis.

## 2026-10-03 23:40 UTC — `training_neural_context`: remaining sequence stopped at first participant

The finite sequence returned external timeout exit124 in allocation46438625
on sh03-06n50. Only sub-01's output directory was created; no fold or final
training_context.json was emitted. The internal180s deadline did not produce
a diagnostic report before the external190s limit. The shell stopped on this
first failure, so sub-03/sub-06/sub-07 were not launched. This is not evidence
of completed reliability calculations or a scientific failure.

The observation does not locate the stalled operation or establish a cluster
outage. As clarified on October5, the first saved report follows the first
ROI computation: missing output cannot distinguish metadata preparation,
beta access or moment arithmetic, and does not establish a pre-fold stall.
Stop this affected route without retry, relocation or
filesystem/configuration workaround. Preserve the incomplete directory and
all completed source, preprocessing and sub-05 diagnostic artifacts. Root
reported the timeout to the user; resumption requires a diagnosed cause.

Optional support-report facts for the user: October3 around23:34–23:38UTC,
job46438625, node sh03-06n50, training_neural_context.py on sub-01's completed
preprocessing handoff; external190s timeout, empty report directory at
outputs/runs/training_neural_context/20261003-remaining-folds01/sub-01.
No support contact was made. Matched RSA and low-level approval remain open.

## 2026-10-03 23:34 UTC — `training_neural_context`: pilot complete; remaining four selected

The sub-05 all-five-fold diagnostic completed23:32:12.786072UTC in56.560033s,
with1,115,456KiB peak RSS in allocation46438625. All five focused fixtures
passed in0.014s; ASTRA schema/semantic validation passed without warnings
(`astra/v0.0.14/validation_20261003_training_neural_context.json`). The report
`runs/training_neural_context/20261003-sub05-folds01/training_context.json`
has complete components, five complete folds and no errors. Each fold uses
3,769 or3,770 common training images, no missing EVC support, and unchanged
primary coverage. All retained voxels have defined reliability and NC.
Across folds, ROI-median signed repeat correlations range approximately
0.141–0.146 ventral,0.236–0.239 lateral,0.147–0.154 dorsal and0.241–0.245 EVC;
NC medians are approximately24%,37–38%,25–26% and38–39%, respectively.
Negative individual-voxel correlations are retained. These are training-context
descriptions, not model performance or strict RSA ceilings; matched raw RSA
and the separate low-level-control requirement remain incomplete.

Select the same components for sub-01/sub-03/sub-06/sub-07, with the unchanged
scientific code and completed participant preprocessing reports. Use four
sequential single-thread calls in existing1CPU/4GiB allocation46438625,
180s internal/190s external per participant, and an840s aggregate outer bound.
The measured56.6s pilot supports this threefold per-participant allowance;
there is no outcome-based participant or voxel selection. Stop the finite
sequence on any failure, retain partial outputs, and never retry implicitly.
Outputs are `runs/training_neural_context/20261003-remaining-folds01/<participant>`.
Only the deadline message and report now identify the selected numeric bound;
no scientific equation, source handling or fixture changed. No RDM/RSA,
embedding or raw-image operation is included.

## 2026-10-03 23:32 UTC — `neural_preprocessing`: all five participants complete

The exact remaining-bundle report declares all four selected participants
complete, errors=[], and400.609669s elapsed. Slurm46438525 independently
reports COMPLETED, exit0:0, elapsed6:54, TotalCPU3:38.507 andMaxRSS8,387,952KiB.
Each child report passed the wrapper's exact five-fold completion check.
Together with the earlier sub-05 report this establishes all25 participant-
fold preprocessing units. These are preprocessing artifacts, not model scores.
Evidence: `runs/neural_preprocessing/20261003-remaining-folds01/bundle.json`
and its four named participant reports. Source extraction is also complete
as recorded below; no source or preprocessing retry is needed.

## 2026-10-03 — `training_neural_context`: adopted training-only diagnostic selected

Root reviewed section5 of the adopted neural recipe and selected implementation
of repeat reliability and provider-formula NC, followed by one bounded sub-05
all-five-fold pilot after static review and focused checks. The new
[diagnostic module](code/training_neural_context.py) reuses completed
preprocessing means/scales, masks, image order and saved presentation validity;
it does not refit parameters, rank voxels or alter primary image coverage.
The common training image-ID intersection across primary/EVC is diagnostic
only. Both primary training/evaluation images missing EVC support remain
explicit, and every training image's original repeat counts and diagnostic
inclusion/exclusion are retained. The three primary sectors share their saved
primary repeats; EVC retains all of its own valid repeats on the common IDs.

All retained repetitions are ordered by session, numeric run and original
session-axis index, then alternated into groups0/1. Three repeats form2:1,
four form2:2, with no selected subset or Spearman-Brown adjustment. Float64
256-voxel chunks compute equal-image ddof1 within-image residual variances,
`eta=mean_i(s²_i)`, `h=mean_i(1/n_i)`, and
`NC=100*max(0,1-eta)/(max(0,1-eta)+eta*h)` after saved session scaling.
Signed voxelwise split-repeat Pearson correlations retain negative values;
undefined correlations/NC remain NaN in arrays and explicit undefined counts
in summaries. Zero repeat-noise variance gives NC100; eta>=1 gives NC0 without
excluding that voxel. This estimates provider-formula encoding-variance context,
not a strict Spearman RSA ceiling or independent repeat errors.

Exact training-image/voxel identities, chronological repeat assignments,
per-ROI repeat histograms/session contributions/spanning-session counts and
voxelwise eta/NC/reliability are emitted with median/p10/p90/undefined summaries.
Only the existing development handoffs and saved preprocessing artifacts enter
the calculation. No mixed source, released NC/R² map, raw pixel, embedding,
RDM or RSA is opened/computed. The later matched training-context RSA table
remains incomplete; these components cannot select or rescue a candidate.

Five [focused fixture methods](tests/test_training_neural_context.py) address
specific scientific failure modes: unequal3/4 repeat counts must not weight
four-repeat images more; zero-noise/eta>=1/negative/constant/nonfinite cases
must remain distinct; chronology must use numeric runs and original indices;
EVC missingness must not alter primary coverage or discard valid primary
repeats; and held-fold perturbations must leave the saved-scaled training
statistics unchanged. Analytic expected values include eta1/2, h7/24,
NC2400/31 and signed correlations±1. No worker ran these tests or diagnostics.
Independent static review of the module and all five fixtures against adopted
recipe section5 **PASS**: equal-image eta/h, signed/undefined Pearson,
chronological common training IDs, per-ROI repeat support and saved scaling
before grouping agree; selected-row numerical reads exclude evaluation values.
No actionable defect was found. Root retains compute checks and execution.

Selected root-owned compute checks and first pilot:

```bash
python3.12 -m unittest discover -s outputs/tests -p test_training_neural_context.py -v
timeout 900s python3.12 outputs/code/training_neural_context.py \
  --preprocessing-report outputs/runs/neural_preprocessing/20261003-sub05-folds01/preprocessing.json \
  --metadata-dir outputs/runs/source_readiness/20261003-metadata_folds04 \
  --output-dir outputs/runs/training_neural_context/20261003-sub05-folds01 \
  --max-seconds 840
```

Use one numeric CPU/4GiB/15minutes,840s internal/900s external, in root's next
substantive attended allocation only after review/checks. Expected work is one
chunked training-repeat pass per ROI/fold, avoiding the original compressed
source and RDM construction; actual runtime remains unmeasured. Compact context
arrays/identity records are durable episode outputs, with no new beta cache.
This is one pilot with no automatic retry or other-participant scaling yet.
The new stable ASTRA analysis is `training_neural_context`, feeding the required
controls/diagnostics; root owns changed-program validation before recording
its result. The active preprocessing bundle/scientific preprocessor remain
unchanged. Separate raw-image extension approval is still pending.

## 2026-10-03 23:20 UTC — `neural_preprocessing`: remaining bundle running

Root's scheduler observation found preprocessing bundle
46438525 RUNNING at elapsed1:21 on`sh03-02n16`. This establishes start, not
participant or bundle completion; the active bundle/scientific code stays frozen.

## 2026-10-03 23:17 UTC — `source_readiness`: all150 source handoffs complete

Root's scheduler observation confirms array46436056 all five tasks COMPLETED.
The final task4/sub-07 completed with exit0:0, elapsed17:04, TotalCPU10:57.877
and Slurm MaxRSS2,096,296KiB. Its exact
[participant report](runs/source_readiness/20261003-development_neural_full01/sub-07/participant_handoff.json)
records30 complete sessions, errors=[],18,842 selected presentations,
885,290,480 reserved payload bytes,1015.755891s and
finished23:17:04.031007UTC. Together with the four previously completed reports,
this establishes all five30-session handoffs and94,216 selected development
presentations, including the reused first-session pilot. This aggregate claim
uses the five reports; each participant controller correctly makes no standalone
all150-session claim.

The dependent preprocessing bundle46438525 was PENDING(None) in the same
scheduler snapshot, with its source dependency now satisfied. Root owns the
next observation; source completion does not imply bundle start or completion.
No source numeric values were opened for this closeout, and active code remains
unchanged. Full source extraction is complete, while cohort preprocessing and
the primary scientific comparison remain separate unfinished stages.

## 2026-10-03 23:15 UTC — `neural_preprocessing`: remaining bundle submitted once

Root's static review passed, and Python AST parsing plus `bash -n` passed in
attended compute allocation46436630. The existing six numerical fixtures and
completed sub-05 pilot are reused; the scientific preprocessor is unchanged.
Root submitted the selected bundle exactly once with
`sbatch --parsable --dependency=afterok:46436056 outputs/code/neural_preprocessing_remaining01.sbatch`,
which returned job **46438525**. Resources remain2CPUs/8GiB/35minutes and two
single-thread children, with no requeue or retry. The dependency waits for the
whole source array to succeed. Submission establishes neither start nor
completion; root owns the next scheduler observation and both jobs' monitoring.
The23:10UTC source snapshot below remains the latest recorded source state.
Wrapper and scientific code are frozen while the bundle is active.

## 2026-10-03 — `neural_preprocessing`: remaining four participants selected

After the completed sub-05 pilot, root selected sub-01, sub-03, sub-06 and
sub-07 independently of any model score. The new finite
[bundle runner](code/preprocess_development_neural_remaining01.py) invokes the
unchanged scientific preprocessor in two fixed waves: sub-01/sub-03, then
sub-06/sub-07. Every child remains single-threaded with840s internal and900s
external bounds. The existing six passing fixtures and full five-fold pilot
are reused; root owns static review and compute-side syntax checks for the
new orchestration, with no scientific-code change or new numerical fixture.

The selected [Slurm script](code/neural_preprocessing_remaining01.sbatch)
requests2CPUs/8GiB/35minutes on the previously discovered normal partition and
`CPU_SKU:7502`, with no GPU, requeue or retry. A30-minute limit would leave no
setup margin around two full900s waves;35minutes accommodates those bounds.
The controller's2,040s launch envelope requires at least900s remaining before
every child launch. It first requires all four exact30-session completion
reports and fresh participant destinations, and waits for both launched wave
members before any next wave. A failure prevents another wave; all completed
and partial files remain. Each unchanged child retains its16GiB output cap,
at most64GiB additional. The requested allocation is at most1.167 core-hours,
well inside the episode ceiling. The sub-05 measured extrapolation is about
10.91minutes for two concurrent waves, assuming equal costs and no shared-I/O
slowdown; the other participants and concurrent cost are unmeasured.

Selected submission, to be performed once by root after its review/checks:

```bash
mkdir -p outputs/runs/neural_preprocessing/20261003-remaining-folds01/logs
sbatch --dependency=afterok:46436056 outputs/code/neural_preprocessing_remaining01.sbatch
```

The dependency requires successful completion of the whole existing source
array, including sub-07; it is not a second source extraction. Root's current
1CPU/4GiB attended allocation cannot host the selected concurrent bundle.
Reports will be
`runs/neural_preprocessing/20261003-remaining-folds01/{bundle.json,sub-*/preprocessing.json}`,
with per-child logs beside the bundle report and payloads under
`$SCRATCH/br_autoresearch/episode04_shared_scene_geometry/neural_preprocessing01/sub-*`.
Only root may submit/monitor; no new job ID exists at this selection entry.
This selected two-worker plan supersedes the earlier one-worker60-minute
recommendation below. Raw-image approval, scores, required diagnostics and OOD
access remain outside this preprocessing action.

## 2026-10-03 23:10 UTC — `source_readiness`: sub-06 complete; final participant still running

Root's scheduler observation confirms array46436056 task3/sub-06, raw job
46436153, COMPLETED with exit0:0, elapsed14:58, TotalCPU10:54.774 and Slurm
MaxRSS2,096,292KiB. The exact
[participant report](runs/source_readiness/20261003-development_neural_full01/sub-06/participant_handoff.json)
independently declares all30 sessions complete, every session exit0 and
errors=[]. It records18,844 selected presentations,950,816,584 reserved payload
bytes,890.662183s and finished23:05:16.948217UTC. Slurm accounting and the
reader/controller timing have different scopes and are retained separately.

Four participant handoffs (sub-01/03/05/06) are now complete. At the supplied
23:10UTC scheduler observation, task4/sub-07 was RUNNING with elapsed10:11;
full150-session completion is not yet established. Root owns further scheduler
checks and the active source reader remains unchanged. This closeout inspected
only the named report, with no beta/ROI payload or new numerical execution.

## 2026-10-03 23:06:11 UTC — `neural_preprocessing`: sub-05 five-fold pilot completed

The selected command completed in root's attended allocation46436630 on
`sh03-06n50.int`, with1 allocated CPU/4GiB. All six focused preprocessing
fixtures passed in0.020s, and the
[changed-program validation](astra/v0.0.14/validation_20261003_neural_preprocessing01.json)
passed ASTRA schema and semantics before execution. The
[preprocessing report](runs/neural_preprocessing/20261003-sub05-folds01/preprocessing.json)
records start23:00:44.415402UTC, finish23:06:11.742112UTC, elapsed327.326741s
and process peak RSS1,126,208KiB. All five folds are complete with errors=[].
The input comprises4,712 images,30 sessions and11,372 union voxels; the
prospective complete-output reservation was2,213,761,280 bytes (about2.06GiB),
not a measured filesystem-size total. Float64 outputs remain under the
episode's `neural_preprocessing01/sub-05` scratch path.

Training statistics were estimable in every session/fold. All folds retained
all3,127 ventral,3,185 lateral,1,332 dorsal and3,734 EVC source voxels, with
zero training nonfinite, scale or session-count exclusions. ROI counts overlap
and are not the union-voxel count. Every primary evaluation image has EVC
minimum-two support. Frozen and minimum-two counts agree in every source cell;
the only complete-four reductions are in the LAION stratum:

| Held fold | Frozen / primary minimum-two images | Complete-four images | LAION / THINGS / THINGSplus primary images |
| --- | ---: | ---: | --- |
| 0 | 943 | 943 | 850 / 29 / 64 |
| 1 | 942 | 941 | 849 / 29 / 64 |
| 2 | 943 | 941 | 849 / 29 / 65 |
| 3 | 942 | 941 | 849 / 28 / 65 |
| 4 | 942 | 941 | 849 / 29 / 64 |
| Total | 4,712 | 4,707 | 4,246 / 144 / 322 |

All primary and complete-four source cells retain at least three images. This
establishes sub-05's preprocessing support under the prospectively adopted
rules; it does not establish nondegenerate neural RDMs, useful reliability,
model alignment, diagnostic completeness or eligibility in other participants.
The original handoffs are preserved, mixed-role source arrays were not reopened,
and no embedding/RDM/reliability/NC/model score was computed. The primary panel
clock, candidate decision and conditional OOD audit have not started. Required
matched training-image RSA/reliability/provider-formula NC context remains later
work; the separate low-level raw-image proposal still awaits approval.

**Proposed continuation, not a submission:** after root verifies all remaining
complete30-session handoffs, process sub-01, sub-03, sub-06 and sub-07 once each,
sequentially in one bounded Slurm job. Reuse the passing checks and unchanged
preprocessing entrypoint, frozen folds, float64/chunk256 settings and selected
development-only inputs. The measured equal-participant extrapolation is
4×327.326741s =1,309.306964s (21.82minutes); this is a cost estimate, not a
guarantee for participants with different voxel counts or I/O conditions.
Recommend1CPU/4GiB/60minutes, one numeric thread, no GPU or retry, retaining the
existing840s per-participant internal ceiling (56minutes combined plus4minutes
setup/teardown). Stop the bundle on failure and preserve completed/partial
attempts. One bundle aggregates the short participant stages and requests at
most1 core-hour. Keep fresh participant-specific report and payload paths;
retain the existing16GiB output cap per participant, at most64GiB additional
or80GiB across all five. The coordinator must record its selected command and
actual allocation before launch. This continuation performs preprocessing only
and does not depend on authorizing the separate raw-image control.

## 2026-10-03 22:57:05 UTC — `source_readiness`: sub-01 and sub-03 complete

Root's scheduler observation confirms array46436056 tasks0/1, raw jobs
46436057/46436058, COMPLETED with exit0:0. Their respective elapsed times were
20:02/20:01, TotalCPU10:45.800/11:16.720 and Slurm MaxRSS
2,096,308/2,096,296KiB. These are Slurm accounting values, not the individual
reader process's peak RSS; no cause for the difference is inferred.

The exact saved participant reports now declare complete30-session handoffs
and errors=[]:

- [sub-01](runs/source_readiness/20261003-development_neural_full01/sub-01/participant_handoff.json):
  18,843 selected presentations,1,003,326,036 reserved bytes including the
  reused sub-01/ses-01 pilot,1191.927072s, finished22:56:06.540737UTC.
- [sub-03](runs/source_readiness/20261003-development_neural_full01/sub-03/participant_handoff.json):
  18,844 selected presentations,1,001,332,496 reserved bytes,1191.278861s,
  finished22:56:06.546799UTC.

Together with sub-05, three participant handoffs are complete. At the supplied
scheduler snapshot task3 was running6:46 and task4 awaited the array limit;
root owns their continued monitoring. No further source values were opened
for this closeout, no active reader code changed, and no duplicate job was
submitted. The previously selected sub-05 preprocessing run remains the next
numeric stage after root's EP08 step; these completions do not expand its scope.

## 2026-10-03 22:51:30 UTC — `source_readiness`: sub-05 accounting confirms completion

Root's scheduler/accounting observation confirms array46436056 task2, raw job
46436146, **COMPLETED**, exit0:0, elapsed11:45, TotalCPU11:07.348 and Slurm
MaxRSS2,096,308KiB. The completed participant report and all30 handoffs remain
the source of exact identity/count evidence. The last session's reader report
separately records22.448686s and process `getrusage` peak RSS96,464KiB. These
different process/accounting scopes must not be conflated; the reason for
their memory difference is not diagnosed by these reports. No OOM was reported.

Task3/raw46436153 was running1:09, tasks0/raw46436057 and1/raw46436058 were
still running15:22/15:21, and task4 awaited the array limit. Root continues
monitoring; no reader code changed and no duplicate was submitted. The
sub-05 preprocessing pilot remains selected, awaiting root's available CPU
after EP08; source completion does not itself constitute preprocessing.

## 2026-10-03 22:49 UTC — `neural_preprocessing`: adopted implementation ready for first complete participant

Root authorized implementation of the adopted foldwise preprocessing only.
`code/preprocess_development_neural.py` accepts only a completed exact
participant handoff, checks all30 sessions and frozen image/trial identities
before beta computation, and reads only the development-only candidate
artifacts. It never reopens the mixed-role source. All first-stage files and
active source readers are preserved.

For each frozen fold, float64 training-only per-session means and ddof0 scales
define the across30-session finite/positive-scale voxel support. The same
three-primary presentation mask supplies equal-weight all-valid-repeat means,
minimum-two eligibility and complete-four flags. EVC uses its own valid repeats
but remains restricted to primary-eligible image IDs. Four separate ROI arrays
are necessary because overlapping primary/EVC voxels can have different valid
repeat means. All4,712 image IDs remain explicit, and ineligible pattern rows
are NaN with counts/reasons, not zero-filled. Missing training support is marked
unestimable rather than mislabeled observed nonfiniteness. Fewer-than-three
retained ROI voxels yields explicit not-evaluable support without replacement.

Six focused fixtures cover held-fold perturbation invariance, across-session
zero/nonfinite exclusion, common-primary versus independent EVC validity,
unequal3/4 repeat weighting and overlapping-voxel means, insufficient session
training counts, and the all30-complete access boundary. C's independent
static review found no scientific/access-boundary defect. A small final
reporting clarification explicitly marks the masks unassessed when training
counts are insufficient; the eligibility rule is unchanged. Root reviews the
final source and owns all tests and numerical execution.

The exact sub-05 first-stage report is complete: job46436146/task2, all30
sessions,18,843 selected presentations, errors=[], elapsed695.898337s,
892,681,584 reserved bytes, finished22:47:51.515809UTC. This supplies the first
available complete participant for a bounded implementation/cost run, chosen
by handoff readiness rather than any neural score. Other participants' partial
sessions cannot substitute for a complete30-session input. This is not an
all150-session completion claim.

Selected test command for root's coordinated compute:

```bash
python3.12 -m unittest discover -s outputs/tests -p test_neural_preprocessing.py -v
```

Root's final source/fixture review found no numerical-boundary blocker and
selected the following first numerical command after the focused checks and
ASTRA validation. The report's analysis ID was mechanically corrected to the
matching stable `neural_preprocessing` ID before freeze. Root will launch only
if a full15-minute envelope remains in its attended allocation.

```bash
timeout 900s python3.12 outputs/code/preprocess_development_neural.py \
  --participant sub-05 \
  --participant-handoff outputs/runs/source_readiness/20261003-development_neural_full01/sub-05/participant_handoff.json \
  --metadata-dir outputs/runs/source_readiness/20261003-metadata_folds04 \
  --payload-dir "$SCRATCH/br_autoresearch/episode04_shared_scene_geometry/neural_preprocessing01/sub-05" \
  --output-dir outputs/runs/neural_preprocessing/20261003-sub05-folds01 \
  --max-seconds 840
```

The selected initial envelope is1CPU/4GiB/15minutes,840s internal alarm,16GiB output
scratch cap, one numeric thread and no automatic retry. Process one fold and
256-voxel chunk at a time; output one ROI file at a time. Read-only input page
residency is OS managed, so measured run RSS remains the resource evidence.
The source ROI membership gives a conservative complete-five-fold output bound
before writing numeric output. No standardization runtime has yet been measured.
No worker ran tests, scientific code or source arrays on the login node.

This stage computes no embeddings, RDM, reliability/NC, encoding fit or score.
The required matched training-image context table and full fixed panel remain
later work, and the raw low-level route still awaits its explicit approval.
The changed ASTRA program is ready for root's one compute-side validation.

## 2026-10-03 22:41 UTC — `source_readiness`: bounded array progress and actual CPU allocation

Root's22:40 scheduler observation found array46436056 tasks0/1/2 running on
`sh03-06n50`, each with **AllocCPUS1 and2GiB**; tasks3/4 were pending only on
JobArrayTaskLimit. The anticipated CPU rounding did not occur in this batch.
Thus the active ceiling is three allocated CPUs/three numeric threads/6GiB,
and the five30-minute tasks request at most2.5 allocated core-hours. Preserve
the earlier rounding estimate as a conservative pre-launch estimate, not an
observed allocation fact for this array.

A bounded read of only the exact full01 logs/reports found no reported errors:
sub-01 had sessions01–03 complete and04 streaming; sub-03 had01–05 complete
and06 streaming; sub-05 had01–12 complete and13 started. The explicitly named
sub-01/ses-04 report then showed completion,650 selected volumes and no
excluded numerical decoding. Its earlier progress showed a substantial delay
before the first64-volume checkpoint followed by regular forward progress;
the cause is unknown and no source/infrastructure diagnosis or retry occurred.
Stderr contained only locale and sticky-module notices in this bounded slice.

These are progress snapshots, not participant or full-grid completion. Source
code remains frozen. Root retains scheduler ownership and future polling;
this worker opened no beta/ROI array and ran no computation or scheduler call.

## 2026-10-03 22:34 UTC — `source_readiness`: full neural handoff array submitted once

Root submitted the frozen `code/development_neural_full01.sbatch` once as
Slurm job `46436056`, array0–4%3. Before submission, all eight focused source
and participant-controller tests passed in0.021s in allocation46433944;
`bash -n` passed, and the
[final changed-program validation](astra/v0.0.14/validation_20261003_neural_full01.json)
passed schema and semantics. A, C and root's independent static reviews are
recorded in the selected-stage entry below.

This launches the remaining149 original-value development session handoffs
while referencing the completed sub-01/ses-01 pilot. Resources remain one
requested CPU/2GiB/30minutes per task, at most three tasks and one numeric
thread per task, with the exact `CPU_SKU:7502` constraint. Up to six allocated
CPUs are anticipated from the observed site rounding; actual batch AllocTRES
will be recorded from root's scheduler observation. No scientific preprocessing
or scoring is launched. Code and inputs are frozen while active.

Per-participant reports will appear under
`runs/source_readiness/20261003-development_neural_full01/sub-*/participant_handoff.json`;
their named session reports preserve partial/completed attempts. Root alone
monitors this array. Submission is not completion, and a missing or pending
task is not authority to submit a duplicate.

## 2026-10-03 22:32 UTC — `source_readiness`: full frozen neural handoff selected

After the completed pilot below, the coordinator selected engineering scaling
of the same original-value, development-only reader across the frozen five
participants and30 sessions. `code/prepare_development_neural_participant.py`
processes one participant sequentially, and
`code/development_neural_full01.sbatch` fixes array0–4%3,1CPU/2GiB/30minutes
per task, normal partition, measured `CPU_SKU:7502`, no requeue/retry, and
single-thread numeric libraries. This is source handoff only: no downstream
standardization, repeat/voxel selection, reliability, RSA, nuisance control or
model fit is launched. Raw-image extension approval remains pending.

The existing reader now accepts only the frozen participant/session choices;
its original sub-01/ses-01 CLI defaults and completed attempt remain intact.
The participant wrapper references the completed sub-01/ses-01 pilot after
identity, role-count, completion and output-presence checks. It loads none of
that pilot's beta/ROI arrays and rereads none of its source files. Remaining149
sessions use fresh per-session report/payload paths. Other selected sessions
read their exact nine fixed mask components through the same proven path.

Every selected session remains in the participant report. Any session failure
stops that participant without retry or a complete/selected-subset claim;
completed predecessors and the failed session's artifacts are retained.
Cumulative reserved payload storage, including the reused pilot, is capped at
16GiB per participant,80GiB total. Each child receives only the remaining cap
and checks its output reservation before writing numeric payloads. An internal
1740-second participant envelope bounds new session launches; each child keeps
the existing540-second maximum or the smaller remaining allowance. Slurm's
30-minute limit remains the hard outer bound.

The observed26.02s/session suggests roughly13minutes per participant and about
1.1 numerical-worker CPU-hours for the150-session grid before variability;
provision roughly2 worker-hours. Requested CPUs are1/task, but the coordinator
observed that this hardware rounds ReqTRES cpu1 to AllocTRES cpu2. Concurrency3
therefore means at most3 numerical threads,3 requested CPUs, up to6 allocated
CPUs and6GiB. Five30-minute requests total at most2.5 requested core-hours or
5 allocated core-hours under that observed rounding. Record actual batch
AllocTRES at launch/accounting; all bounds remain within EP04's48-core ceiling.
The full stream remains roughly1.87TB uncompressed. This does not silently
enlarge other episode allocations.
The observed pilot is the scaling evidence, not an assumption that every
participant will share its timing or voxel count.

Focused checks selected for root's current compute allocation:

```bash
python3.12 -m unittest discover -s outputs/tests -p 'test_development_neural*.py' -v
```

This reruns the five source-boundary methods after parameterization and adds
three checks for the exact150-cell/one-reused-pilot routing, incomplete/wrong
identity rejection, and the cumulative storage cap. A and C independently
reviewed the participant controller and batch/parameterization and returned
PASS; root's review also passed. No worker ran
tests, source payloads or scheduler commands. Root corrected the prepared
constraint spelling to the exact already-verified `CPU_SKU:7502` before launch.

After successful review/checks, root alone may create
`outputs/runs/source_readiness/20261003-development_neural_full01/logs` and
submit `sbatch outputs/code/development_neural_full01.sbatch` from this episode.
The report root is that matching `development_neural_full01` directory;
payloads go to `$SCRATCH/br_autoresearch/episode04_shared_scene_geometry/neural_full01`.
Root must reuse any submitted/running array and not submit a duplicate.
No full-array job ID exists in this preparation record yet.

## 2026-10-03 22:27 UTC — `source_readiness`: first-session neural handoff completed

Root's allocation46433944 step7 on `sh03-02n16.int` completed the fixed
sub-01/ses-01 pilot at22:27:31.024831UTC,26.020652s elapsed and94,360KiB peak
RSS. All five focused tests passed in0.014s. The
[changed-program validation](astra/v0.0.14/validation_20261003_neural_pilot01.json)
passed schema and semantics. Root independently reviewed both the initial
source boundary and final sequential-writer refinement; A and C also returned
read-only PASS without executing tests or reading source arrays.

The [pilot report](runs/source_readiness/20261003-development_neural_pilot01/neural_handoff.json)
records all1,044 source volumes and12,211,488,432 uncompressed beta bytes
consumed,575 selected volumes numerically decoded and469 discarded as opaque
bytes. Excluded numerical decoding is zero. The exported selected beta payload
is29,515,900 bytes; all selected ROI values are finite, with575 zero values
retained rather than treated as missing. These are source handling facts, not
foldwise neural eligibility or scientific performance.

The fixed masks contain3,767 ventral,3,281 lateral,2,170 dorsal and3,622
retinotopic-EVC voxels; their union contains12,833. The nine component masks
had zero NaNs. The scratch report paths identify the complete original-value
handoff. No session standardization, reliability/NC, RDM, model fit, raw image
or protected numerical outcome was opened. This establishes the first-session
I/O route and measured scaling cost, not full source qualification.

## 2026-10-03 22:24 UTC — `source_readiness`: neural recipe adopted; selected first-session handoff

Before any beta or mask values, the coordinator adopted the
[neural preprocessing recipe](neural_preprocessing_proposal.md), including the
mandatory separately labeled training-image raw RSA/reliability/provider-formula
NC table on identical IDs and per-ROI support. Held-fold raw high-level/EVC RSA
remains separate. These context statistics cannot select the primary candidate.
No scientist-only decision was inferred for these delegated preprocessing
choices. The explicit raw-image extension approval remains pending.

The next effect is only the fixed `sub-01/ses-01` I/O handoff, implemented in
`code/prepare_development_neural_pilot.py`. It uses the nine already-qualified
mask components, direct native same-index mapping, and575 development rows
from the original1,044-volume session. The other469 volumes pass through gzip
as opaque bytes and are never numerically decoded or summarized. Exact frozen
headers, float32/identity scaling and development-image/trial joins must match.
Retain original beta zeros and nonfinite values; no voxel reliability/finite
selection, session standardization, repeat eligibility, RDM or fit runs here.

Five focused fixture methods cover F-storage/C-index correspondence, repeated
run-local indices versus original session indices, numeric decoding only after
role selection, NaN/zero mask/beta semantics, changed headers and truncation.
Independent static review found no concrete boundary defect. The final writer
uses sequential NPY rows instead of a writable memmap, keeping working memory
independent of total output length; the existing spatial/boundary fixture also
checks that generated NPY output loads with unchanged selected rows.

Selected coordinated command (not executed by the worker):

```bash
python3.12 -m unittest discover -s outputs/tests -p test_development_neural_pilot.py -v &&
timeout 600s python3.12 outputs/code/prepare_development_neural_pilot.py \
  --source-root "$OAK/users/zijiao/br_autoresearch_data/steward_acquisition/laion_fmri_mixed_role_20260922" \
  --metadata-dir outputs/runs/source_readiness/20261003-metadata_folds04 \
  --array-metadata outputs/runs/source_readiness/20261003-array_metadata02/array_metadata.json \
  --roi-mapping outputs/config/roi_mapping.json \
  --payload-dir "$SCRATCH/br_autoresearch/episode04_shared_scene_geometry/neural_pilot01" \
  --output-dir outputs/runs/source_readiness/20261003-development_neural_pilot01 \
  --max-seconds 540
```

Envelope:1CPU,4GiB,10minutes with540s internal alarm,16GiB scratch maximum,
one numeric thread and no retry/requeue. Root owns the allocation and may
launch only with sufficient remaining time. The one-session source stream is
about12.21GB uncompressed; all150 would total roughly1.87TB, so the full corpus
is not authorized from this estimate. Actual ROI union count and elapsed/RSS
from this fixed pilot will size any next extraction. Progress is saved every64
source volumes; incomplete outputs remain explicitly partial. Scratch payloads
are reproducible from the retained source and are not an OS audit seal.

The changed ASTRA program now contains this bounded pilot; root will validate
it on compute. No source payload, test, scheduler or numerical code was run
on the login node during preparation.

## 2026-10-03 22:19 UTC — development embedding handoff completed

The selected handoff completed in attended allocation `46433944`, step3,
on `sh03-02n16.int`, from22:19:26.062613 to22:19:33.259284UTC. The focused
fixture passed (one test,0.011s); the source handoff took7.196672s and exited0.
The [report](runs/source_readiness/20261003-development_embeddings01/embedding_handoff.json)
records all23,560 development rows for CLIP, DINOv2 and PEcore at1024 features
and SigLIP2 at1152 features, with zero errors. Original float16 values and
common identity order are preserved; the logical selected feature payload is
199,034,880 bytes. No normalization, RDM or model fit was performed.

The selected-row reader returned no excluded feature/validity elements to
Python. Complete source identity arrays were used only for joining; OOD
categories were not parsed/exported. The disclosed HDF5 storage-chunk boundary
and absence of an OS audit seal remain unchanged. No neural, ROI or raw-pixel
values were read. This completes the development embedding handoff, not full
source qualification or any empirical panel/audit.

The [neural preprocessing proposal](neural_preprocessing_proposal.md) is now
prepared for coordinator review. It specifies training-only session scaling,
computable voxel support, repeat validity, training-repeat reliability and
provider-formula NC, unchanged folds and a single fixed ridge diagnostic.
It is unadopted; no beta/ROI values or new source probes were used to prepare
it. The explicit raw-image extension approval remains separate and pending.

## 2026-10-03 22:19 UTC — selected development embedding handoff

ASTRA `development_embeddings`: execute the previously adopted, independently
reviewed handoff in attended allocation `46433944`, one CPU, 2 GiB, five-minute
bound. Run its focused test first, then `prepare_development_embeddings.py`
with the completed metadata/fold and array-header reports and output
`runs/source_readiness/20261003-development_embeddings01`. This reads only
the 23,560 qualified development rows of the four fixed embeddings; no neural,
ROI, raw-pixel or excluded-image feature values are selected. Partial outputs
are not published as complete. The internal limit remains 270 seconds.

No empirical panel or audit has started under the current `GOAL.md`,
`DATASETS.md`, and `SEARCH_POLICY.yaml` contract. Metadata preparation and
synthetic component checks are recorded below.

## 2026-10-03T21:57:27Z — `source_readiness`: development embedding handoff selected

The coordinator adopted the next existing-source step before any value access:
all23,560 frozen development image IDs across the four default released models,
with the original float16 values and identical output row order. Prepared
`code/prepare_development_embeddings.py` first joins each complete HDF5 ID array
to the frozen metadata_folds04 development list. All four joins must succeed
before any validity or feature elements are read. It then reads only matched
development rows, in ascending source-index chunks of512, and restores the
common output order. Missing/duplicate development IDs, false validity,
nonfinite values or zero vectors stop the handoff without shrinking folds.
No normalization, RDM, model fitting, neural/ROI values or raw pixels are read
or computed. Downstream conversion to float64 and fixed L2/cosine geometry
remain explicit later operations; native provider normalization is preserved.

HDF5 can internally read/decompress a chunk containing excluded rows. The
boundary is the selected dataset elements returned to Python and exported to
the development handoff, not physical-byte or codec-level exclusion. Source
image IDs are inspected only to join; excluded IDs/categories are not exported.
The candidate artifact is development-only, but same-UID access to the original
source is not an OS audit seal. Raw-image extension approval remains pending.

The generated fixture checks scrambled source ordering, exact float16 byte
preservation, selected-only validity/feature accesses and missing/duplicate
development identity rejection. Root will run it immediately before the
substantive handoff in the next coordinated allocation; no test/source code was
executed on the login node. Selected command from the episode directory:

```bash
python3.12 -m unittest discover -s outputs/tests -p test_development_embeddings.py -v &&
timeout 300s python3.12 outputs/code/prepare_development_embeddings.py \
  --source-root "$OAK/users/zijiao/br_autoresearch_data/steward_acquisition/laion_fmri_mixed_role_20260922" \
  --metadata-dir outputs/runs/source_readiness/20261003-metadata_folds04 \
  --array-metadata outputs/runs/source_readiness/20261003-array_metadata02/array_metadata.json \
  --output-dir outputs/runs/source_readiness/20261003-development_embeddings01 \
  --max-seconds 270
```

Resource plan:1CPU,2GiB, numeric threads1,5-minute step within the next
substantive allocation; Python3.12 with the already available NumPy/h5py stack.
The selected feature payload is199,034,880 bytes plus small identity/report
metadata and NumPy headers. A512-row feature block is at most1,179,648 bytes;
four disk-backed output arrays avoid a full in-memory copy. Expected elapsed
time is1–3minutes, unmeasured and filesystem dependent, with a270-second
application deadline and300-second outer bound. This is not a new panel row or
a neural-score launch. Attempt01 is selected but not yet executed.

Independent static review by agentA found no concrete launch blocker: it
checked all-four identity-first joins, ascending selected source chunks and
destination-order restoration, original float16 preservation, failure without
fold shrinkage, incomplete partial artifacts and the disclosed HDF5 codec
boundary. The reviewer ran no tests and opened no source payloads.

The preceding allocation46430982 ended at its wall limit before the planned
final ASTRA validation. The coordinator cleared the program freeze for this
additional adopted step and will validate the final changed program once in
the next necessary allocation.

## 2026-10-03T21:52:06Z — `source_readiness`: observed grid discrepancy and adopted index route

The [active-sform diagnostic](runs/source_readiness/20261003-roi_grid_offsets01/roi_grid_offsets.json)
completed in root's46430982 compute at21:51:20.792370UTC,0.012663s, all45 ROIs,
zero source-array reads. Maximum absolute coordinate displacement is
2.8875447071641247e-05 beta voxels; maximum Euclidean displacement is
2.8884325467587986e-05 beta voxels. These are observed geometric discrepancies,
not a threshold test. The diagnostic artifact remains unchanged with no
mapping-acceptance or resampling claim.

Combining this result with the pinned provider's direct flattened-index
masking implementation, the coordinator selects the same native array-index
mapping for these fixed same-shape exports. The adopted
[mapping recipe](config/roi_mapping.json) preserves original NIfTI voxel order
and applies each ROI at the corresponding beta index, without reorientation,
interpolation or resampling. The small active-affine discrepancy remains
explicit; it is not silently rounded away. This decision is scoped to the
qualified fixed export panel, not a universal tolerance for arbitrary files.
Inactive qform differences are irrelevant to this active-sform evidence.

All30 retinotopic components are acquired. Retinotopic EVC will use their
six-component union per participant, never laion-EVC. This index-route decision
does not inspect masks, qualify neural/ROI finite values or coverage, select
voxel reliability thresholds, open audit data or bypass pending raw approval.
Future numeric source preparation must apply the original session trial-row
ordinal as well as these spatial indices.

## 2026-10-03T21:48:23Z — `source_readiness`: no-threshold active-sform diagnostic selected

Prepared `code/describe_roi_grid_offsets.py` on root's explicit metadata-only
scope. It reads only the completed array_metadata02 JSON, checks all150
sessions for a common active beta sform within each participant, and computes
`inverse(A_beta) @ A_ROI` from coded sforms. Inactive qforms are ignored.
For each of the45 present masks it transforms the eight index-domain corners
(0 or dimension−1) and reports all displacements, maximum absolute coordinate
displacement by axis/overall, and maximum Euclidean displacement in beta voxel
index units. Matching encoded spatial units are required; no physical-unit
claim is made from the source's unspecified unit code.

The diagnostic applies no tolerance/pass threshold, adopts no nearest-center
identity rule and selects no resampling. Corner extrema summarize affine
coordinate displacement over the entire rectangular domain, not ROI voxel
membership, biological coverage or numerical masks. No source array is opened.
All30 retinotopic components are present; earlier unconfirmed availability is
resolved by the completed report, not a remaining acquisition blocker.

Selected bounded calculation, expected well below one second of CPU plus small
JSON I/O, on root's existing1CPU/2GiB allocation:

```bash
timeout 60s python3.12 outputs/code/describe_roi_grid_offsets.py \
  --metadata-report outputs/runs/source_readiness/20261003-array_metadata02/array_metadata.json \
  --output-dir outputs/runs/source_readiness/20261003-roi_grid_offsets01
```

This adds no source read or scientific score, and does not depend on pending
raw-image approval. Root reviews/runs it; no worker-side execution occurs.

## 2026-10-03T21:45:56Z — `source_readiness`: structural continuation completed

Root's46430982 step14 passed all three focused fixtures (0.115s), then completed
array_metadata02 at21:45:56.680141UTC in6.214576s with zero errors. The
[report](runs/source_readiness/20261003-array_metadata02/array_metadata.json)
reuses all150 checked beta header/sidecar records after the exact identity/axis
checks; those unchanged files were not reread.

All four released embedding arrays have25,052 rows, float16 storage,25,052
image-ID entries and25,052 boolean validity entries. Dimensions are1024 for
CLIP/DINOv2/PEcore and1152 for SigLIP2. Direct file attributes declare:

| Label | Released model_id | normalize |
| --- | --- | --- |
| CLIP | hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K | true |
| DINOv2 | vit_large_patch14_dinov2 | false |
| PEcore | PE-Core-L14-336 | true |
| SigLIP2 | google/siglip2-so400m-patch14-384 | true |

PEcore also declares image_size336. These are source declarations and header
shapes, not numerical normalization tests, checksum identities for pretrained
weights, image-ID coverage checks or valid-mask observations. No HDF5 dataset
element was read. The original May12 source timestamps remain recorded.

All45 requested ROI files are present in both the preserved provider index
and the completed acquisition:15 primary-sector masks plus all30 retinotopic
V1d/V1v/V2d/V2v/V3d/V3v components. All45 headers were read; there is no missing
retinotopic component in this requested panel and no laion-EVC substitution.
However, all45 literal spatial-grid-field comparisons with beta headers are
false. Representative primary masks have the same143³ dimensions, inactive
qform0 and active sform2, with small floating-point differences in pixel sizes,
sform elements and some translations. Literal inequality is not demonstrated
source incompatibility, and header presence is not a voxel mapping certificate.
Any numerical array handoff still needs a declared geometry-aware index or
resampling rule and metadata-only support for that rule.

The structural step is complete; full source qualification remains false.
Feature ID/validity joins, neural finite-repeat validity, ROI unions/coverage,
training-only preprocessing, encoding and role-filtered numeric handoffs
remain. The raw low-level proposal still awaits separate approval; pixels,
beta/ROI/embedding values and audit outcomes remain closed. Attempt01 is
preserved with its complete beta evidence and attribute API failure.

## 2026-10-03T21:40:33Z — `source_readiness`: header completion and narrow attribute repair

Structural attempt01 in46430982 ran from21:38:56.127512 to21:39:25.551300UTC
(29.423791s). All150 beta headers matched their frozen full-session TSV
lengths; the recorded spatial grid fields agreed across sessions within each
participant. All150 sidecars declared TYPED/betasmd. Those header/sidecar
records remain durable in
`runs/source_readiness/20261003-array_metadata01/array_metadata.json`.
Grid-field agreement is descriptive and does not qualify voxel mapping.

The first HDF5 attribute then raised `TypeError: 'h5py.h5a.AttrID' object does
not support the context manager protocol`. No completed embedding or ROI
record was published. This is an API handling defect, not source insufficiency.
Replace the unsupported context manager with explicit `try/finally: close()`.
Add one actual-h5py synthetic attribute fixture for the literal whitelist,
including a model string and input-size pair; no real source values are tested.

Selected fresh attempt02 reuses the prior150 header/sidecar records only after
checking the exact participant/session/path identities, all frozen row counts,
fourth-axis lengths, header-only flags and TYPED/betasmd declarations. No
unchanged beta header or sidecar is reread. The continuation opens only the
four embedding metadata handles and the previously selected ROI headers.
The failed attempt remains unchanged. Existing1CPU/2GiB allocation, bounded
300s, expected seconds to a minute for the remaining metadata:

```bash
python3.12 -m unittest discover -s outputs/tests -p test_array_metadata.py -v &&
timeout 300s python3.12 outputs/code/inspect_array_metadata.py \
  --source-root "$OAK/users/zijiao/br_autoresearch_data/steward_acquisition/laion_fmri_mixed_role_20260922" \
  --metadata-dir outputs/runs/source_readiness/20261003-metadata_folds04 \
  --prior-beta-report outputs/runs/source_readiness/20261003-array_metadata01/array_metadata.json \
  --output-dir outputs/runs/source_readiness/20261003-array_metadata02 \
  --max-seconds 300
```

No beta/feature/ROI values, feature IDs/valid-mask elements or raw pixels are
opened. Raw-source extension approval remains separately pending.

## 2026-10-03T21:37:02Z — `source_readiness`: structural array/ROI step selected

The coordinator adopted a bounded metadata-only continuation, independent of
the still-pending raw-image extension. Prepared `code/inspect_array_metadata.py`
and two focused synthetic fixtures in `tests/test_array_metadata.py`.
The exact inputs are successful metadata_folds04, the completed acquisition's
selected-object receipt, its preserved provider ROI index, and the source
files named below. No recursive source discovery or additional download occurs.

Read all150 exact T1w beta NIfTI headers and matching small JSON sidecars.
Compare every fourth-axis length with that session's original full TSV row
count, using the already frozen row-ordinal recipe. The sub01/ses01 sidecar
explicitly declares TYPED/betasmd and acquisition-ordered TSV-row alignment;
every sidecar must independently declare TYPED/betasmd. Header grid fields
are compared with the first session per participant and reported literally;
header metadata does not prove finite/valid beta values or voxel mapping.

The gzip reader emits only the first four decompressed bytes, identifies
NIfTI1/2 and then emits exactly348/540 header bytes total. It never flushes
the decoder, constructs an image array or decodes voxels. Bounded opaque
compressed input buffers may span later bytes; those bytes are not inflated
past the header. Focused fixtures cover both formats/endians and full-session
TSV-axis matching rather than the87-trial run-local example.

For the four preserved default embedding HDF5 files, inspect only the three
named datasets' shapes/dtypes and a literal model/preprocessing attribute
whitelist at the file and embedding-dataset levels. Limit attribute arrays
to16 elements and small stored/serialized metadata to8192 bytes. No dataset
elements are read: image IDs, valid masks and embedding values remain closed.
This binds structural dimensions and available model declarations, not
numerical normalization or unique-ID coverage. The source receipt dates these
preserved files to May12, before the OOD-only September28 correction.

Crosswalk the three registered primary sectors to exact `laion` T1w mask
paths for all five participants. Retinotopic EVC is the union of V1d/V1v,
V2d/V2v and V3d/V3v, as specified by the [provider](https://laion-fmri.hebartlab.com/rois.html).
The similarly named `laion-EVC` is response-restricted and is not substituted.
For these30 retinotopy components, report preserved-provider-index presence,
acquisition-receipt presence and headers only when selected locally. Absence
from the acquisition is not provider-wide unavailability. No masks are read,
unions formed, voxel counts calculated or ROI/scientific thresholds changed.

Selected command in root's existing1CPU/2GiB allocation, one numeric thread,
300s internal plus external deadline; expected seconds to a few minutes for
150 beta/sidecar pairs, four HDF5 metadata opens and at most45 ROI headers:

```bash
python3.12 -m unittest discover -s outputs/tests -p test_array_metadata.py -v &&
timeout 300s python3.12 outputs/code/inspect_array_metadata.py \
  --source-root "$OAK/users/zijiao/br_autoresearch_data/steward_acquisition/laion_fmri_mixed_role_20260922" \
  --metadata-dir outputs/runs/source_readiness/20261003-metadata_folds04 \
  --output-dir outputs/runs/source_readiness/20261003-array_metadata01 \
  --max-seconds 300
```

The existing official [ASTRA validation](astra/v0.0.14/validation_20261003_metadata_folds04.json)
passed schema and semantics in46430982 after the fold closeout. It predates
this structural recipe; the updated program awaits root's next validation.
No tests or source operations were run by this worker on the login node.

## 2026-10-03T21:31:11Z — `source_readiness`: metadata/folds completed

Root ran all seven focused metadata checks in allocation46430982 step8:
PASS,0.023s. Fresh attempt04 completed at21:31:11.701127UTC after16.784190s.
The [durable report](runs/source_readiness/20261003-metadata_folds04/source_metadata_report.json)
records all150 trial TSVs joined,23,560 unique-development images,94,216
development presentations and14,285 retained grouping units. Every one of
the15 participant/source cells differs by at most one image across its five
folds. All2,131 provider components,297 duplicate sets and66 seed42 correction
entries were validated; provider random folds were not used as EP04 folds.
The source revision remains `cf82765ced029de81c3bec7254b8cbd6a65e3686`.

All23,560 unique CSV repeat fields remain explicitly blank/null. The separately
documented intended repeat count is four. The actual trial metadata have
23,536 images with four presentations and24 with three; none has fewer than
two metadata presentations. This does not qualify finite/usable neural repeats.
Each file's run counts and run-local index ranges are recorded, with the
separate original session-row coordinate retained before role filtering.

The successful directory publishes `development_images.jsonl`,
`development_trials.jsonl`, `development_groups.jsonl` and `adopted_config.json`.
Only unique-development identity rows are exported; excluded/conditional
roles have aggregate counts. Attempts01–03, their source metadata and all
failure reports remain unchanged. No beta, embedding, ROI or raw-pixel values
were opened; full source qualification and audit authority remain false.

The inactive low-level proposal now references successful attempt04. Its three
analytic generated-pixel checks already passed; actual raw-source execution
still awaits the contract's separate source/settings approval. Remaining
same-contract work is the low-level geometry, role-filtered beta/feature
handoff, ROI crosswalk, training-only preprocessing and encoding diagnostic.
Metadata completion alone does not authorize neural scoring or establish an
OS-level audit boundary. The ASTRA program/navigation record this completed
step without marking the full source handoff complete.

## 2026-10-03T21:30:06Z — `source_readiness`: run-local trial indices and attempt04

The six focused metadata tests passed in46430982.5 (0.023s). Attempt03 then
stopped at21:26:13.933982UTC after0.904131s on the false assumption that
`beta_index` is unique across a session. Its report and root's bounded safe-field
[trial-index diagnosis](runs/source_readiness/20261003-metadata_folds03/trial_index_schema.json)
are preserved. The exact sub01/ses01 TSV has twelve runs01–12, each with87
distinct ordered indices0–86. No trial labels or neural values were emitted.

The [provider beta documentation](https://laion-fmri.hebartlab.com/glmsingle_betas.html#relation-to-stimuli)
states that TSV row order corresponds to the session NIfTI volume / HDF5
betasmd column. Its separate description of `beta_index` as session-wide
conflicts with the observed source. The [pinned loader](https://github.com/ViCCo-Group/LAION-fMRI/blob/cf82765ced029de81c3bec7254b8cbd6a65e3686/laion_fmri/subject.py)
preserves TSV order in `get_trial_info` and directly applies its trial mask
to session betas in `get_betas`, supporting row-order addressing.

The repair retains `(participant,session,run,beta_index)` as identity and
adds `session_beta_axis_index`, the original zero-based TSV row position
before role filtering. Every run must have ordered, unique, contiguous
indices from zero; run blocks must be ordered within the provider01–12
range. No fixed87 count or minimum number of runs is assumed. The all150
file pass will emit each run's count/range and distinguish this metadata
axis contract from later array shape/finite-value qualification. One focused
fixture verifies reset indices map to distinct session positions and rejects
duplicates, gaps and reordered run blocks. No rows are dropped or relabeled.

Selected fresh attempt04 in root's existing1CPU/2GiB allocation, estimated
seconds to a few minutes for150 small files, externally bounded300s:

```bash
python3.12 -m unittest discover -s outputs/tests -p test_source_metadata.py -v &&
timeout 300s python3.12 outputs/code/prepare_source_metadata.py \
  --source-root "$OAK/users/zijiao/br_autoresearch_data/steward_acquisition/laion_fmri_mixed_role_20260922" \
  --output-dir outputs/runs/source_readiness/20261003-metadata_folds04
```

This remains metadata-only. The raw-control source extension is still
inactive pending the separately required approval; no pixel reader ran.

## 2026-10-03T21:24:43Z — `source_readiness`: complete repeat-schema diagnosis and attempt03

Attempt02 in allocation46430982 failed after0.185061s at21:23:41.874799UTC,
before trial joins or fold output, because unique image CSV row1360 has an
empty `n_reps` field. Attempt02 and its report remain preserved. Root then
performed one complete aggregate of the four permitted nonidentity fields
in compute:25,052 rows,19 combinations. All23,560 unique rows have blank
`n_reps`: each participant has4246LAION,144THINGS and322THINGSplus images.
Shared rows are641LAION12rep,240THINGSplus12rep,240NSD4rep and371OOD4rep,
with the same five participant IDs. No image names/categories were emitted.
This complete schema observation replaces further parser-assumption retries.

The official [experimental design](https://laion-fmri.hebartlab.com/experimental_design.html#main-experiment)
explicitly defines four presentations for each participant-unique image.
Its separate incomplete-repetition section distinguishes occasional actual
three-presentation images from this planned four. The exact source quote,
URL and observation time are now in `config/development_metadata.json`.
For blank unique CSV rows the adapter preserves `planned_repeats=null` and
`n_reps_csv=""`, separately records `intended_repeats=4` and the provider
design URL, and derives the modal intended count from those declared plans.
The actual150-TSV occurrence histogram/mode and unknown valid-neural count
remain separate. Nothing infers intended repetitions from observed outcomes
or silently requires four valid repeats in the primary minimum-two rule.

The raw-control proposal reader now compares the exact preserved CSV repeat
field, including blank, rather than assuming a populated `4rep` string.
A focused missing-repeat test distinguishes documented unique blanks from
unsupported shared blanks and conflicting unique repeat claims. Selected
attempt03 command,1CPU/2GiB with the same pinned source and300s bound:

```bash
python3.12 -m unittest discover -s outputs/tests -p test_source_metadata.py -v &&
timeout 300s python3.12 outputs/code/prepare_source_metadata.py \
  --source-root "$OAK/users/zijiao/br_autoresearch_data/steward_acquisition/laion_fmri_mixed_role_20260922" \
  --output-dir outputs/runs/source_readiness/20261003-metadata_folds03
```

Root separately ran the three generated-pixel analytic tests in46430982:
PASS,1.125s. These tests read no source images. Required raw-extension approval
is still pending; no raw reader or feature extraction has run.

## 2026-10-03T21:22:57Z — `source_readiness`: attempt01 schema failure and selected repair

Root's allocation46430982 step2 ran the five focused metadata tests: PASS,
0.023s. The metadata adapter then stopped closed after1.273559s, at
21:22:02.641574UTC, before publishing any development folds/views.
`outputs/runs/source_readiness/20261003-metadata_folds01/source_metadata_report.json`
records `Unknown shared source disposition at line643`. Its five small
provider files were successfully pinned to
`cf82765ced029de81c3bec7254b8cbd6a65e3686`; they and the failed attempt remain.

Strictly bounded inspection of that CSV row's four nonidentity fields found
`dataset=THINGSplus`, the five declared participant IDs, `shared`, and
`12rep`. The adapter had recognized the documentation/filename spelling
`THINGS+` but omitted the literal source-column spelling. Add the exact
`THINGSplus`→`thingsplus` alias only; no filename inference, role relaxation,
grouping/threshold change or category interpretation. Bind the already
observed provider commit in the active config, so a retry cannot advance to
a different moving-main revision. The same exact five small files are fetched
at that commit; no broad source refresh or historical source mutation occurs.

Selected fresh attempt02 command (same1CPU/2GiB/300s bound):

```bash
timeout 300s python3.12 outputs/code/prepare_source_metadata.py \
  --source-root "$OAK/users/zijiao/br_autoresearch_data/steward_acquisition/laion_fmri_mixed_role_20260922" \
  --output-dir outputs/runs/source_readiness/20261003-metadata_folds02
```

The existing passing group/balance checks are unchanged. No beta, embedding,
ROI, pixel or neural score was opened; this failure does not change scientific
eligibility. Low-level raw-source proposal remains inactive.

## 2026-10-03T21:03:19Z — `source_readiness`: metadata joins and grouped folds selected

Before any neural score, the coordinator adopted five global component-preserving
folds, one deterministic participant/source balancing pass, and empirical95th
percentiles using NumPy `method="higher"` with strict exceedance. The precise
rules are frozen in `outputs/config/development_metadata.json`. They do not
change the native model panel, practical margin, null draws/seeds or evidence
roles. This is implementation authority, not an empirical/audit opening.

Selected effects: in compute, pin one current official ViCCo-Group/LAION-fMRI
commit; preserve only its four named grouping/correction JSONs and split-method
README; join the original completed acquisition's stimulus CSV and exactly150
trial TSVs; validate the corrected duplicate assignments; emit unique-only
image/trial/group tables and excluded-role aggregate counts. Original source
files remain untouched. A duplicate/correction inconsistency stops the fold
freeze without new groups, thresholds or random restarts. Image/feature/neural
payloads, OOD category extraction/interpretation, captions and sibling outputs
are outside this step.

The original embedding acquisition predates September28, but the provider's
release history explicitly says the24,936 noncorrected rows are unchanged;
only116 OOD rows changed. Consequently regular-unique development does not
require a wholesale embedding refresh. Existing September22 raw-stimulus
acquisition control metadata records official DUA version2026-05-12 acceptance;
no credentials or packed pixels were read. This does not approve a new raw-
derived control. The needed low-level source/settings will be a concrete
separate proposal under GOAL's raw-extension clause, not a duplicate DUA request.

The selected root-owned command, from the episode directory, is:

```bash
python3.12 -m unittest discover -s outputs/tests -p test_source_metadata.py -v &&
timeout 300s python3.12 outputs/code/prepare_source_metadata.py \
  --source-root "$OAK/users/zijiao/br_autoresearch_data/steward_acquisition/laion_fmri_mixed_role_20260922" \
  --output-dir outputs/runs/source_readiness/20261003-metadata_folds01
```

Resources:1 CPU,2GiB, bounded5min for acquisition/join/folds, bundled into an
existing substantive allocation. Five focused tests cover actual15-cell
bookkeeping, deterministic group preservation, shared-member removal, duplicate
inconsistency and excluded/unknown IDs. No test has been run by this worker.
The expected metadata volume is small (25,052 image rows,150 TSVs and grouping
JSONs); filesystem/network delay may dominate. Per-participant progress is
flushed. A timeout preserves its attempt and cannot publish a completed freeze.
No primary evaluation clock, candidate decision or audit opening has started.
The original CSV has exactly the five required identity/role/repetition columns.
Excluded image identity strings are necessarily read to join and remove their
trial rows; some identity strings encode category text. No OOD category is
parsed, interpreted or emitted as an analysis field.

The separate inactive `outputs/low_level_control_proposal.md` and pure
`outputs/code/low_level_pixels_proposal.py` now make the required control
decision concrete: fixed5/10/20-grid RGB means/SDs from the accepted official
source, unique-development IDs only, Euclidean nuisance geometry, no trained
model or neural fitting. No source reader/CLI is enabled and no pixels or
tests were run. The proposal cites GOAL152–154 and DATASETS120–124 for the
specific separate raw-extension approval, retains existing DUA acceptance,
and gives the60-image representative and full-coverage cost plan for review.

Before any proposed raw-source effect, root requested completion of the
reviewable trusted reader and three analytic generated-pixel checks. These are
now `outputs/code/prepare_low_level_proposal.py` and
`outputs/tests/test_low_level_pixels_proposal.py`. The selected compute action
is only `python3.12 -m unittest discover -s outputs/tests -p
test_low_level_pixels_proposal.py -v` in root's ordinary preparation allocation,
using1CPU/2GiB; expected seconds, no source path argument. The tests check
closed-form constant-color norms/RDMs, a7/30 squared-distance spatial-contrast
example, and five alpha rounding boundaries. No tests were run by this worker.
Root supplied exact raw `payload/` paths from bounded file-stat metadata.
The source-selecting reader/60-image/full recipes remain inactive proposals;
no approval token, source opening, role change or policy amendment was added.

## 2026-10-03 — `source_readiness` and `implementation_qualification`: selected preparation

The scientist authorized resuming feasible work across episodes. Current EP04
contracts, input documentation and location pointers were read before outputs.
No role-filtered current input pack, executable evaluator or ASTRA program
exists here. The mixed-role private steward source, legacy scratch neural
copy, previously exposed regular-shared outcomes and conditional OOD outcomes
remain unopened. No panel score or candidate decision is selected.

Selected next effects are implementation of the determinate native-RSA,
partial-RSA, whole-image permutation, covariance-matched random geometry,
participant aggregation and fixed candidate-decision equations, plus a
documented source handoff and tests on independently specified mathematical
invariants. Source session standardization, fold count/assignment, model-array
binding, low-level geometry and near-neighbour definition remain explicit
pre-score choices; code will not invent their values. The bounded qualifier
will accept no empirical inputs and run in the coordinator-owned Slurm
allocation using one numeric CPU,2GiB and at most10 minutes. This is component
implementation qualification, not a panel evaluation or audit. It starts no
120-hour panel clock and makes no scientific acceptance claim.

The implementation is `outputs/code/native_rsa.py`; the selected qualifier is
`outputs/code/qualify_native_rsa.py --output-root <fresh-scratch-directory>`.
Its ten tests use mathematical identities and counterexamples, including
preserved covariance when feature dimension exceeds image count, equal
participant weighting, unconditional null argmax, and mandatory-control
failure of the highest-score row. These are developer-authored checks, not an
independent evaluator. Empirical and full-panel execution remain incomplete.

Current provider documentation identifies default arrays for all four model
families, publicly CC0 derivative access, exact beta-trial join columns and
shared-only object masks. The September28 release correction changed116 OOD
embedding rows and also fixed duplicate grouping in subject random folds.
No corrected payload was fetched, and the previously relocated steward source
was not inspected. `outputs/source_handoff.md` records source links and the
remaining concrete bindings. The new ASTRA program preserves the contract's
fixed-panel stages without adding adaptive-search quotas or a campaign.

A bounded read-only Brain Researcher dataset lookup for `LAION-fMRI`
returned three unrelated OpenNeuro datasets rather than a matching source.
No returned row was adopted as EP04 evidence or an alternative audit. The
official provider documentation above remains the documentary basis. This
lookup does not establish source absence and made no canonical mutation.

Actual `implementation_qualification` execution on compute node
`sh03-06n50.int`, allocation `46425841`, at2026-10-03T18:59:57.328927Z:
all ten component tests **PASS** (test runner0.109s; total qualifier0.670s).
Report and test output are preserved at
`outputs/runs/implementation_qualification/job_46425841/`. Official ASTRA
schema and semantic validation also **PASS**, saved at
`outputs/astra/v0.0.14/validation_20261003.json`.
There were zero empirical panel evaluations, candidate decisions or audit
openings. Full production qualification and the source handoff remain pending.

## 2026-09-28 — Mandatory-falsifier policy repaired

- Added the Goal-required identical-fold comparison of RSA versus encoding
  conclusions to the policy's required falsifiers.
- Required exact participant-level finalist and locked-candidate results.
- Expanded the finalist rerun scope to all eligible subject-unique images,
  registered ROIs, frozen seeds, and required falsifiers.
- Required every listed falsifier to pass under its pre-search frozen rule
  before promotion.

This repair occurred before any current-contract neural outcome computation.
No scientific trial was run, and the one-shot audit remains unopened.

## 2026-09-28 — Primary question refocused on VLM–brain alignment

- Replaced the prediction-led adaptive shared-representation target with direct
  RSA between native frozen VLM image-embedding RDMs and image-disjoint fMRI
  response RDMs; no reproduction analysis is part of the new question.
- Made the participant-balanced VLM-minus-DINOv2 RSA contrast the primary
  endpoint, equally aggregating ventral, lateral, and dorsal high-level
  sectors. Low-level geometry is a required nuisance control; caption and
  object geometries are optional secondary anchors.
- Reclassified identical-fold encoding as a required secondary diagnostic. It
  cannot select or rescue the primary RSA result, and disagreement is reported
  as readout dependence rather than treated as a failed concordance test.
- Excluded the previously opened 1,121 regular-shared responses from the new
  analysis; no reproduction branch is planned.
- Fixed the biological evidence at five equally weighted participant effects.
  Image or near-neighbour-cluster resampling measures stimulus uncertainty;
  RDM edges are not independent observations.
- Made development a complete fixed comparison of three native VLM geometries
  and one fixed DINOv2 geometry; the single winner decision is
  selection-exposed, and final evidence is reserved for one locked OOD audit.
  The audit uses equal-weight within-category RSA, a 0.02 Fisher-z
  VLM-over-DINOv2 margin, four-of-five participant direction, leave-one-out
  stability, and a locked permutation null.
- Made PCA/whitening geometry, encoding, EVC comparison, and crossnobis
  secondary diagnostics rather than alternate primary endpoints or rescue
  routes.
- Removed the inherited adaptive-trial, patience, successor, Pareto, and
  finalist quotas because the primary panel contains no adaptive operators.
- Made the development and OOD image-label/random-geometry nulls explicitly
  selection-aware, deterministic, and one-sided for positive unsubtracted RSA.
- Required complete OOD VLM, DINOv2, low-level, category, and image coverage
  before neural opening, plus a positive OOD low-level partial-RSA check.

This design revision supersedes the earlier RSA-versus-encoding falsifier
classification above. It occurred before any current-contract neural outcome
computation; no scientific result or audit result was inspected.

## 2026-09-30 — conceptual figure added

- Generated `ep04_conceptual_question.png` with built-in imagegen and inspected
  the actual image. A targeted connector edit makes all four illustrative
  image identities feed every VLM, DINOv2 and fMRI branch.
- Added the figure and a scientific-schematic caption to GOAL and the outputs
  README; saved the exact generation/edit prompts alongside it.
- The figure preserves same-image native-geometry RSA, the fixed panel, five
  equally weighted participant effects, excluded exposed regular-shared
  images, and the conditional same-participant OOD stimulus-shift audit.
- The figure shows no observed alignment, winning model or causal language
  effect. No payload/outcome was read or supplied to imagegen; no analysis,
  acquisition, job, Git action or canonical transition occurred.
- Scientific thresholds, data roles, SEARCH_POLICY and ASTRA are unchanged.

## 2026-10-02 local scope review

Literature review identifies established VLM–brain RSA and CLIP/DINO/SigLIP encoder comparisons. Reframed the Goal around the existing within-category OOD native-geometry advantage, rather than a new-checkpoint ranking. See [review (consolidated)](../GOAL.md). Model panel, RSA definitions, 0.02 margin, roles and audit stay unchanged. No payload, score, or experiment opened; ASTRA path is absent and was not made a new gate.

## 2026-10-02 figure review

Viewed the former 1536 × 1024 PNG and regenerated a sparse 1672 × 941 imagegen figure emphasizing direct native RSA and within-category OOD comparison. Accepted after correcting independent VLM/DINOv2-to-fMRI connectors, neural-RDM symmetry and abstract category exemplars; personally inspected and independently reviewed. Exact prompts saved, earlier assets retained, GOAL/output-index links updated. No source data, outcome, policy, runtime or Git change.
