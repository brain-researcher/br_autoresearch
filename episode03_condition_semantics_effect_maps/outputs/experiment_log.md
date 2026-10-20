# Experiment log

## 2026-10-18 18:22 PDT — grouped_evaluator_qualification — mask_text002 complete; all 628 headers compatible

Step **47586225.9** passed the affected scaling regression (0.75 s, 40,620 KiB), built [mask_text002](runs/grouped_evaluator_qualification/mask_text002/preparation.json) (0.82 s, 71,928 KiB), and passed changed ASTRA validation (3.75 s, 73,488 KiB); the attended shell exited successfully.
The anatomical mask preserves **235,840 foreground voxels**. Step 47586225.8's independent [headers001](runs/grouped_evaluator_qualification/headers001/headers.json) confirms **628/628** compatible headers; no target voxel array was read.
Structured text is available for **574/628** maps. Among the **500** maps without current metadata exclusions, **477** have structured text (325 distinct descriptions), and **23** retain unresolved meanings without being silently excluded.
All identities, exclusions and roles remain; the failed mask_text001 and its encoding diagnosis are preserved. The six input checks and eight earlier evaluator checks pass, with only the affected mask test repeated after repair.
Next: resolve the 23 condition meanings from permitted methods metadata, then finish source eligibility and predictor composition; inference thresholds remain unadopted and baseline/scientific access stays false.

## 2026-10-18 18:20 PDT — grouped_evaluator_qualification — reference encoding diagnosed; mask_text002 selected

Step **47586225.8** found exact uint8 labels 0/255; float32 slope `0.003921568859368563` maps foreground to `1.0000000591389835`, so the exact-one check rejected legitimate encoding roundoff.
[Reference diagnosis](runs/grouped_evaluator_qualification/mask_text001/reference_diagnosis.json) confirms 235,840 foreground locations, finite two-level support and the selected grid.
The independent [header check](runs/grouped_evaluator_qualification/headers001/headers.json) passed **628/628** exact maps (34.73 s, 54,172 KiB); no target voxel arrays were read.
Implemented and independently reviewed a two-level scaling repair preserving exact support, with no probability threshold; 0/0.5 and multilevel references remain invalid.
Selected the single affected mask test, corrected `mask_text002` build and changed ASTRA validation in the same bounded attended step; the other passing checks and headers are reused.
The first regression invocation never ran: the coordinator replaced the module-provided `PYTHONPATH`, hiding NumPy. Step 47586225.8 exited 1; selected a new 1-CPU/512-MiB/5-minute companion using direct test-script invocation with the existing module path preserved.

## 2026-10-18 18:17 PDT — grouped_evaluator_qualification — mask_text001 reference check stopped; diagnosis selected

Step **47586225.7** passed all six focused tests (0.82 s, 40,152 KiB) but the real reference build stopped at `reference mask is not finite binary support` (0.77 s, 70,832 KiB).
The failed `mask_text001` attempt is preserved; no target array, scientific score or threshold was opened/changed. Header and ASTRA phases had not run when the attended step exited 1.
Selected a new bounded companion step in task-owned 47586225: 1 CPU/512 MiB/15 minutes, inspecting only the selected reference's datatype, scaling and value counts and finishing the independent 628-header check.
Diagnose before changing the reader or support rule; preserve the original anatomical support and continue a corrected build only if justified by those observations.

## 2026-10-18 18:16 PDT — grouped_evaluator_qualification — mask_text001 and headers001 execution selected

Selected six focused checks, the reference-mask/signed-text build, exact 628-map header-only compatibility and changed ASTRA validation after independent source review.
Use a coordinator-owned attended companion step in substantive allocation 47586225: 1 CPU, 512 MiB, 15-minute step; header I/O has a separate 10-minute command limit.
The active EP21 worker remains one numerical thread with a 15 GiB process RSS limit; combined work stays within the existing 3 CPUs/16 GiB, with EP03 outputs only and no EP21 payload access.
Preserve all map identities/exclusions and exact coefficient/polarity bindings; read only the selected anatomical reference array and enumerated target headers, with baseline/scientific access false.
Next: record actual step, text/support counts, compatibility outcomes and resource measurements; diagnose failures without adopting inference thresholds or pruning data.

## 2026-10-18 18:14 PDT — grouped_evaluator_qualification — anatomical mask and signed-text preparation selected

Selected the existing TemplateFlow MNI152NLin2009cAsym res02 brain mask and exact structured-role text under standing methods authority; full scientific comparison remains family prototype versus prototype plus signed semantics.
Prepared `code/prepare_mask_and_signed_text.py`, six focused checks and separate reference/text and 628-map header-only phases; all original identities, exclusions, orientations and data roles are retained.
Exact paths and model coefficients reuse the existing route census; raw titles/methods prose and source IDs are excluded from predictors, and unresolved meanings stay explicit.
No new code/test/job has run in this preparation; no target array or scientific score has been opened. The earlier eight checks and methods001 execution remain valid.
Coordinator estimate: 1 CPU/512 MiB; <1 minute reference/text and up to 10 minutes for header I/O, measured in task-owned compute before scaling.
Next: independent source review, focused compute checks and the two new attempts; diagnose any source/text mismatch without changing eligibility, while retaining the unadopted inference decision packet.

## 2026-10-18 18:03 PDT — grouped_evaluator_qualification — methods001 verified and built

Task-owned companion step **47586225.6** ran on `sh04-04n23.int` and exited successfully; no baseline maps or scientific scores were accessed.
All **8 focused tests passed**; [tests](runs/grouped_evaluator_qualification/methods001/tests.txt) took 1.19 s and 39,872 KiB peak RSS; the metadata build took 0.23 s and 17,824 KiB.
[Preparation](runs/grouped_evaluator_qualification/methods001/preparation.json) accounts for **103 routes, 49 source accessions and 628 maps**; 500 have no current metadata exclusion, and 38 accession groups support some outer comparison.
Those are provisional metadata counts, not authenticated independent groups or final map eligibility; baseline-target and scientific-scoring flags remain false.
Changed ASTRA schema/semantic validation passed (3.93 s, 73,340 KiB); no substantive scientific threshold was adopted.
Next: bind common anatomical space/mask and remaining source/text inputs; retain the unadopted inference decision packet before scientific baseline access.

## 2026-10-18 18:02 PDT — grouped_evaluator_qualification — coordinated methods001 verification selected

Selected the eight focused tests, 628-row metadata-only build and changed ASTRA validation; no target arrays, scientific scoring or threshold adoption.
Use an attended companion step in task-owned substantive allocation 47586225: 1 CPU, 512 MiB, 5 minutes; its existing EP21 worker remains limited to one numerical thread and 15 GiB process RSS within the allocation's 3 CPUs/16 GiB.
The coordinator owns this step; outputs stay in EP03's `outputs/runs/grouped_evaluator_qualification/methods001/`, with no reads or writes to EP21 result payloads.
Independent source review found no implementation defect; its positive-sensitivity wording clarification is incorporated in the unadopted specification.
Next: record actual step, test/build outcome and measured resources, then use the metadata support to prepare the remaining common-space/text inputs.

## 2026-10-18 17:55 PDT — grouped_evaluator_qualification — methods preparation selected

Selected under standing authority: the registered family-prototype versus prototype-plus-signed-semantics comparison, with equal independent-group weighting and every required control retained.
Prepared [methods/configuration](grouped_evaluator_specification.md), the metadata-only taxonomy/grouped-fold builder and exact normalization/prototype/loss primitives; no existing evaluator was found to reuse.
The frozen orientation/screens/control identity and October 17 fitted-N history are preserved; accession independence remains an explicit assumption, with evidence-based overlap merging and no new audit role.
Eight focused fixtures and the actual 628-row metadata build are unrun; no job submitted, target image opened, model fitted or scientific score read in this preparation.
The linked specification contains one concrete, unadopted inference/threshold packet; taxonomy and ordinary engineering choices do not require a new generic approval.
Next: coordinator-owned compute verification and metadata build, fixed common-space/mask and remaining source qualification, then the registered comparison only when the actual threshold/access prerequisites are satisfied.

## 2026-10-17 06:07 PDT — ds003545 compL1 contributors identified (job 48245679); fitted N resolved for all 628 maps

Job **48245679** (52 s) ran the leave-k-out search. In all 6 ds003545 compL1 maps, removing **sub-13 and sub-14** reproduces the Dataset effect (max relative Δ ≈ 5e-16) and t (r = 1.0000000000), out of 231 pairs per map. These maps were fitted on N=20, while 22 subject maps exist. The flag "conditions absent in 8/88 designs" matches 2 subjects × 2 sessions × 2 runs, but that link is not verified.
Fitted N is now resolved for all 628 maps: 622 confirmed as assumed, and 6 corrected to N=20 under the pre-registered rule. Downstream sample-size terms must use these values. The census JSON keeps its original counts; the corrections are in the 48245679 report.
Selected: ASTRA validation, job **48245776** (prebuilt validator, 1 CPU, 2 GB, 10 min), with `fitted_n_lineage` registered. Next scientific step: the family-taxonomy and inference-specification freezes, which come before the family-prototype baseline.
ASTRA validated in job **48245776**. EP03 compute on October 17: 10,045 core-seconds (2.8 core-hours) across five jobs; the lineage checks were allocated 2 CPUs.

## 2026-10-17 06:02 PDT — corpus N lineage: 622/628 confirmed (job 48221634); ds003545 compL1 diagnosis selected

Job **48221634** (81 min) applied the pre-registered rule to all 628 maps, and 622 are confirmed: the best df is n−1 with zero t→z error, the Dataset effect equals the mean of the n inputs within 1.2e-15, t correlates at r ≥ 0.9999999999999997, and the inputs match the census subjects. Confirmed N ranges from 2 to 108, so the census N assumption is resolved for these maps.
The 6 ds003545 compL1 maps (both sessions) are not confirmed. Their t→z conversion fits df = 19 exactly, so FitLins used 20 inputs, yet 22 subject maps exist and all were written by the same commit. Dropping any single subject does not match (the closest is sub-14).
Selected: a leave-k-out search with k = n−1−df = 2 (231 pairs per map), added to `code/check_group_df_recompute.py`. It reruns on ds003545/task-compL1 only in job **48245679** (1 CPU, 8 GB, 30 min). An exact pair match corrects N to 20 and identifies the dropped contributors. Otherwise the maps are excluded; they are already held out of the primary set for timing.

## 2026-10-17 00:25 PDT — BART N=16 confirmed (job 48221009); corpus lineage check selected

Job **48221009** (83 s) confirmed all 13 BART maps under the pre-registered rule:
- the best df is 15, with median |z_pred−z| = 0 (df 16 is off by 4e-3 to 1.2e-2);
- the Dataset effect equals the mean of the 16 subject effect maps (max relative Δ ≤ 1e-15);
- the recomputed t is identical (r = 1.0000000, max |Δt| ≤ 9e-15);
- the inputs are sub-01…16.
The BART N=16 assumption is resolved. ASTRA validated in job **48221008**.
The gate is passed. Selected job **48221634**: the same check on all routes (`run_fitted_n_lineage.sbatch corpus`, 1 CPU, 8 GB, 3 h, default 20,000 df voxels), reading only Dataset t/z/effect/variance images and the feeding nodes' effect images.

## 2026-10-17 00:21 PDT — BART fitted-N lineage check selected (first target load)

Decision rule, fixed before the run, applied per Dataset map with census input count n (the thresholds are constants in the script):
1. The best integer df fitted to paired t/z voxels (1<|t|<7) equals n−1, with median |z_pred−z| ≤ 1e-3 at n−1.
2. The Dataset effect equals the mean of the n inputs within the nonzero-t mask: max |Δ| ≤ 1e-4 × max |effect|.
3. The recomputed t (sd with ddof=1) correlates ≥ 0.9999 with the Dataset t, with median |Δt| ≤ 1e-3.
4. The inputs found equal the census subjects.
A map that fails gets a leave-one-out scan for a dropped contributor; N is then corrected, or the map excluded if contributors stay unknown. All routes run only if BART's 13 maps are confirmed or every failure is explained.
Selected job **48221009**: `code/run_fitted_n_lineage.sbatch bart ds000001/task-balloonanalogrisktask` (1 CPU, 8 GB, 30 min). It reads only the Dataset t/z/effect/variance images and the 208 subject effect images.

## 2026-10-17 00:21 PDT — orientation frozen; N lineage check next

On the scientist's instruction (2026-10-16, Pacific), all four proposal items were frozen as written: rules R1–R7 with their precedence and the role dictionary, the P/I readings, the eligibility screens, and the negative-control identity (trial split, with replicate order as fallback) and its route-selection rule.
Frozen references: the [proposal](orientation_freeze_proposal.md), the build-48189996 artifacts (now read-only) and `code/build_route_manifest.py`. SEARCH_POLICY, DATASETS and the ASTRA decision record the freeze. 21 maps remain unresolved.
Codex (gpt-6-astra, max, text-only) drafted these records and `code/check_group_df_recompute.py`. I reviewed both before applying them and made three fixes: corrected the review attribution, made report writing NaN-safe, and added the subject-match criterion. A parallel Codex start hit the shared login-node thread limit and was rerun alone.
The family taxonomy, the negative-control pass threshold and normalisation remain separate freezes before the family-prototype baseline. Selected: ASTRA validation with the prebuilt validator, job **48221008** (1 CPU, 2 GB, 10 min).

## 2026-10-16 18:37 PDT — proposal critique; amended rebuild selected

ASTRA validation passed in job **48189598**, with the census and proposal outputs registered. A read-only Codex critique of the proposal (text only, no data; review kept on Scratch) prompted these amendments, all before any outcome:
- R2 is stated in both directions, with an added check that reversed weights give −s.
- ds004056 go/no-go is unresolved because its roles are undocumented.
- self/friend and old/new recognition responses become category conventions.
- A lower-level decision is marked assumed when a higher factor is defined on one side only.
- R1 and R6 are described as coordinate conventions; their signs are unchanged.
- The trial-split negative control is preferred to replicate-order.
Selected: rerun of the build (same job file, 1 CPU, 2 GB, 15 min). The onset check repeats unchanged.
Job **48189996** finished. Orientation now stands at 208 resolved (all 9 frozen reproduced), 51 assumed, 345 proposed, 21 unresolved and 3 ineligible; included accessions: 106/33/163/19/1. All 628 maps pass the reversal check. Its outputs replace those of job 48188582 in the same directory; the earlier counts survive only in the 18:32 entry. The [proposal](orientation_freeze_proposal.md) is rewritten to match. Compute today for this task: 1,738 core-seconds (0.48 core-hours) across six jobs; the two census jobs were allocated 2 CPUs each. Next: freeze the rules, then the BART df/recompute load.

## 2026-10-16 18:32 PDT — route manifest and orientation proposal; job 48188582

Job **48188582** wrote the [route manifest](runs/source_handoff_qualification/20261016-orientation-proposal/route_manifest.tsv) and the [per-map orientation](runs/source_handoff_qualification/20261016-orientation-proposal/orientation_by_map.tsv) for 103 routes and 628 maps.
- Builds: resolved for all 96 routes with maps; 49 were read from the later in-git copy of the annexed `dataset_description.json`.
- Specs: resolved in 95 routes. ds000009 BART is assumed because its report predates its maps.
- N: assumed for all 628 maps; df check pending. 57 maps have N=2.
- Events: 26 accessions resolved, 2 assumed (ds000009, ds002872), 21 unedited.
- Orientation under the proposed R1–R7: 210 resolved, reproducing all 9 frozen BART decisions; 58 assumed, 339 proposed-rule, 18 unresolved, 3 ineligible. Included accessions: 108/40/157/16/1 of 322.
- ds003545 onset check inconclusive.
The rules and the replicate-order negative control are in the [proposal](orientation_freeze_proposal.md). Selected: ASTRA validation (1 CPU, 2 GB, 10 min) after adding `route_census`, `orientation_proposal` and `orientation_freeze`. Next: freeze the rules, then the BART df/recompute load.

## 2026-10-16 18:24 PDT — rerun 48185964 complete; manifest build selected

Rerun **48185964** (280 s) covered all 103 routes and 628 maps with no errors. Builds: dev16 on 58 routes, dev42 on 37, dev94 on 1, all nilearn `5:run:iso`. Embedded spec vs local: 86 exact, 2 pybids defaults, 2 input filter, 6 changed contrasts. Every map's input effect maps were rewritten by its own commit, one per subject. 7,578 design headers were read.
Definitions were checked against the run-level designs. Two kinds of defect surfaced: ds000009 discounting's pm regressors and ds004556's `rating` were never convolved, and five names do not match their definitions. ds000009's event edits came from code committed only after its run.
Job **48188518**: the onset check ran, but the builder failed while writing its summary (tuple JSON keys); fixed and resubmitted with the same inputs. Selected job: 1 CPU, 2 GB, 15 min. `code/build_route_manifest.py` applies the hand-authored `condition_roles.json` and `event_edit_review.json` to EP03 records only. `code/check_dummy_onsets.py` reads ds003545 events and BOLD JSON to test onsets against the 4 trimmed volumes. Output: `runs/source_handoff_qualification/20261016-orientation-proposal/`.

## 2026-10-16 18:00 PDT — census 48184490 complete; corrected rerun selected

Job **48184490** (1 CPU, 548 s) covered 103 routes and 628 Dataset Z maps with no errors. It reproduces BART: `d723670` rewrote 13/13 Dataset maps and 208/208 subject effect maps; the build is dev94; the spec equals `4dc69d7`; the only edit is n/a→0. Corpus-wide, the commit that wrote each Dataset map also wrote every one of its input effect maps, and each subject contributes one input.
Three defects need a rerun:
- 49 routes' `dataset_description.json` was still annexed at their map commit, so no version was read;
- ds003441 has two map commits;
- ds003858's basic-details key is `MID`.
Selected rerun: same shape, plus an unannex fallback, per-commit report models, spec-difference classes, prefix task keys, function history and run-level design-matrix headers (column names only). Output: `20261016-route-census-<job>/`. The first run is kept.

## 2026-10-16 17:45 PDT — per-route execution-record census selected

This extends the BART treatment to all 103 routes of the 49 NeuroEffect accessions with FitLins analyses (628 Dataset Z filenames). For each route, `code/census_route_provenance.py` reads the following, using git log/show only:
- the `analyses/<ds>` git history;
- `dataset_description.json` and the FitLins report as committed with each Dataset Z map;
- the history of the producer spec and of `modify_events.py`;
- events TSVs where the producer's input copy survives, plus the task sidecars.
It opens no image, header, PNG, group report or score. Its validity check: it must reproduce the BART facts (`d723670` rewrote 208/208 subject effect maps; dev94; spec match).
Selected job: 1 CPU, 8 GB, 60 min. Output goes to `runs/source_handoff_qualification/20261016-route-census-<job>/`.

## 2026-10-16 17:24 PDT — ASTRA program updated and validated

`source_handoff_qualification` gains the `source_fact_resolution` output and a `source_facts` decision. Schema and semantic validation passed in job **48182822**, using astra-spec 0.0.14 via the prebuilt `brain-researcher-astra` venv ([record](astra/v0.0.14/validation_20261016_job48182822.json)). Two earlier attempts did not validate. Job 48181001, via uvx, was cancelled after 11 min while it was still compiling `pyoxigraph`. Job 48182658 failed because the libpython module was not loaded. Today's EP03 compute totals 729 core-seconds (0.20 core-hours).

## 2026-10-16 17:06 PDT — source facts resolved; job 48169324

Job **48169324** (1 CPU; submitted 15:28:53 PDT, ran 15:30:40–15:30:52, 0.18 GB) read events TSVs, metadata and filenames only. In all 48 runs the producer's only edit was n/a→0. `pumps_demean` and `control_pumps_demean` rise +1 per pump within a balloon: 3,070/3,070 reward steps; 1,984 control steps, the rest resets. Reward pumps are centered within each balloon.
`cash_demean` and `explode_demean` equal the pump count minus the run mean over that outcome (slope 1, residual <2e-15 in 48 and 47 runs). Under the 15:28 rule, `pumpspara`, `controlpara`, `cashpara` and `explodepara` take `s=+1`, so 9/13 directions are now resolved. The remaining four need ontology rules, not more source facts.
Also resolved: the spec and FitLins dev94 build that ran. N=16 is assumed, with its check pending. The "51" are NeuroEffect candidate accessions (31 included / 20 excluded), and 628 Dataset Z filenames exist, not 577; see the [fact table](../DATASETS.md#source-facts-resolved-2026-10-16). One `git status` call created and removed a lock in the shared `input/ds000001/.git`; no content changed.
Next: resolve the other candidate routes the same way, then freeze the corpus orientation table and the negative-control identity before the first target load.

## 2026-10-16 15:28 PDT — source-fact rules fixed before the events check

These were fixed before any map access, and none occurs here. **Assumed:** each of the 13 BART Dataset maps has N=16, contributors sub-01–16; the evidence is in [DATASETS](../DATASETS.md#source-facts-resolved-2026-10-16). The numerical check (df from the t/z maps, recompute from the 16 subject maps) loads target maps, so it waits for the pre-load freezes: the GOAL orientation rule and the SEARCH_POLICY negative-control `identity_freeze`.
**Rule for the `*_demean` columns:** the +1 coefficient is "higher minus lower pump count", `s=+1`, if the source events rise by +1 per pump within a balloon (≥95% of steps) or, for cash/explode, equal the pump count minus a constant (|slope−1| and residual ≤0.01). A −1 relation gives `s=−1`; any other relation stays unresolved.
Selected job: 1 CPU, 4 GB, 20 min. It runs `code/check_bart_event_construction.py` plus a filename-only census of the 51 NeuroEffect candidates (`code/reconstruct_neuroeffect_candidates.py`), with output in `runs/source_handoff_qualification/20261016-source-facts-<job>/`.

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
