# EP24 attempt002 candidate packets

Built from each episode log at `960ab41`; picks drafted by Codex (gpt-6-astra, max) and verified mechanically (heading present, at most 300 words, every quote verbatim). Proposers see the packet only; raters also see [key.md](key.md).

## EP01

~~~~text
EP01 — What Do Neural Loss-Aversion Maps Measure?
Current log entry (2026-10-13):
2026-10-13 12:47 Pacific — exploratory_choice_score_report — saved real-choice report complete
The selected reporting-only command completed with exit 0 in attended
47679500, consuming the fixed prepared_20261013_002 behavioral metadata.
No model was refit and no BOLD was read. All 108 participants and both
opposite-half predictions are retained, with correctly pooled responded-trial
denominators and equal participant weighting. Across 27,454 responded trials
(194 misses excluded), mean intercept/model log loss is 0.617660/0.288500
nats per responded trial. Paired improvement is 0.329159, participant SE
0.008465, descriptive t95 interval [0.312378, 0.345941]; 107/108 improve.
EI improves by 0.301562 and ER by 0.356757. This documents the predictive
control used by the fMRI branch, not a mechanism, novelty or fresh confirmation.
Choice-only scores cannot be compared numerically with earlier joint RT scores.

Evidence: report,
participant CSV and JSON alongside it; execution evidence is
`continuation_20261013_47679500/choice_report.log` and `choice_report_exit.txt`.
Optional figure was unavailable because the science environment has no
matplotlib; numerical reporting succeeded. No package installation selected.
All parent cost will be charged once when 47679500 closes.

The optional figure was subsequently rendered from the saved CSV/JSON in the
same compute allocation using the existing `py-matplotlib/3.8.3_py312` module.
The report module's scientific imports were made lazy so its unchanged plotting
function can consume saved numbers without the fitting runtime. PNG/PDF and
`figure_generation.json` now sit alongside the report; the original report JSON
preserves its initial optional-figure-unavailable record. No scores or intervals
were recomputed. Visual inspection confirms all 108 paired values and the
all/EI/ER mean intervals are shown with appropriate labels.
~~~~

## EP04

~~~~text
EP04 — Does native VLM–brain alignment survive image-distribution shift?
Current log entry (2026-10-03):
2026-10-03 23:34 UTC — `training_neural_context`: pilot complete; remaining four selected
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
~~~~

## EP08

~~~~text
EP08 — What makes the next electrode or calibration trial useful?
Current log entry (2026-10-03):
2026-10-03 23:27:17 UTC — `development_search`: mapping heuristic loses across all orders
The complete mapping report
in root allocation46438625 records elapsed271.200377s, maximum
RSS273,288KiB and observed23:27:17.410798UTC. All eight development sessions,
four orders and20resource cells completed:2560inner systems,1792actual refits,
32,768proxy updates,36,864fitting-trial reads and320evaluation reads.
Logical neural bytes were387,075,840, with30,094,080unique logical bytes;
these are element-accounting quantities, not physical filesystem traffic.
Endpoint score replay is exactly zero. No held session was accessed. Charge
one valid mapping-policy comparison, bringing the shared total to 20.

Mapping Q is0.195818152276, versus balance0.202165946372 and mean-value
0.202601645117. Mapping-minus-balance is −0.006347794096 and
mapping-minus-mean-value is −0.006783492841. Against balance, the Chewie
and Mihili means are−0.006540705218 and−0.006154882974; all eight session means
are negative. The four original/0/1/2 order contrasts are−0.005073768727,
−0.006709435353,−0.007327984923 and−0.006279987380. All fixed-prediction
session-omission means remain negative. The intermediate-checkpoint losses
are−0.010901175363 in Chewie and−0.010258138290 in Mihili. Initial/final
checkpoints remain exactly replayed and all resource cells remain reported.

Thus this particular acquired-feature design heuristic underperforms count
balance and the small target-only mean-value gain robustly across the recorded
orders. It does not establish that mapping information or uncertainty-based
acquisition is generally useless. Fixed initializer coordinates and penalty,
full-linear unit-noise design support, small directional empirical pools and
the refitted reduced-rank decoder differ in ways that this comparison cannot
separate. The selected grouping ablation addresses one of those alternatives;
it cannot rescue the completed result. Twenty valid evaluations do not complete
the40minimum, required diverse-family coverage or alternative checks.
~~~~

## EP09

~~~~text
EP09 — At what biological resolution does dendrite–projectome coupling live?
Current log entry (2026-10-06):
2026-10-06 Pacific — metadata_feasibility: provider alias confirmed
The selected bounded metadata check exited 0 in coordinator-owned job
46842156 on `sh04-04n22.int`. Its
direct observation
returns exactly one matching row from the retained public supplement, row 81:
Animal ID `236174`, Image ID `16124`, fMOST,
`Gnb4-IRES2-CreERT2;Ai139`, provenance `BIL meta (20231223)`, and
`Claustral and cortical Car3+ IT`, referencing Peng et al. 2021. The provider
locator is `https://download.brainimagelibrary.org/7f/53/7f537a62e521a26a/mouseID_236174/`.
The animal ID does not already occur in the retained 38-animal join. Its
original unmatched record remains brain `236174` / 43 cells.

This resolves the factual alias/genotype gap: the original implementation
searched the Image ID column, while this source group uses the animal ID.
The observation agrees with the independently checked BrainTell string and
TeraVR Methods. No EP11 crosswalk, source-cell workbook, morphology or axonal
outcome was read. The shared parent allocation is charged once to EP18;
no separate EP09 allocation or additional charge is introduced.

Updated current GOAL, DATASETS, paper plan and README so they no longer ask
for this resolved identity/genotype. Preserved `brain_genotype_join.json`,
`genotype_support_summary.json` and all historical entries. Reuse the direct
observation as correction evidence; any future integrated join must use a
separate output and retain original brain, animal and image IDs explicitly.
No blanket cross-namespace fallback, primary IT eligibility, layer recovery,
donor support or role assignment follows. The 12-driver-animal and
two-potential-donor-animal feasibility limits, original 12/8 design and
unadopted nine-animal proposal remain unchanged. Remaining source questions
concern authentic layer/class and matched-donor support, not this alias.
~~~~

## EP10

~~~~text
EP10 — Does co-target context predict terminal organization inside a shared target?
Current log entry (2026-09-26):
2026-09-26 — Lean Gao MOp qualification
- Official sources: Gao et al., *Neuron*, DOI
  `10.1016/j.neuron.2025.10.019`; the provider whole-cortex portal and public
  neuron/sample JSON endpoints; Zenodo record `16909126` for the full archive.
- Action: downloaded only the two public metadata JSON files to transient
  scratch and selected exact provider soma acronyms beginning with `MOp`.
  Retained output fields were identity, source/layer, hemisphere,
  reconstruction scope, fMOST filename prefix, sample join, project, and
  transgenic line. The JSON transport contains an outcome-derived `class` key,
  but its values were not queried, emitted, retained in an artifact, or used.
- Support: 837 MOp cells in 76 unique fMOST filename-prefix groups; all 76 join
  once to distinct nonempty sample IDs. Layers are MOp1 27, MOp2/3 322, MOp5
  330, MOp6a 151, and MOp6b 7; hemispheres are 226 left and 611 right.
  Provider reconstruction scope is 833 `Axon_and_dendrite` and 4 `Axon_only`.
- Identity limit: public records expose no animal, mouse, or donor key. The 76
  groups are therefore conservative released brain/specimen units, not 76
  certified independent animals.
- Cross-episode check: 15 fMOST units overlap EP11's outcome-blind SSp-tr
  inventory. EP11 has no frozen role assignment, but a shared Gao exposure
  ledger is required before either episode opens outcomes.
- Decision: Gao MOp passes the released brain/specimen-count screen, but
  independent-animal identity, exact position support, reconstruction QC,
  complete-target observation, and roles remain unresolved. Outcome search
  remains closed.
- Transfer boundary: an exact 837-filename targeted downloader was prepared in
  transient scratch. The required DTN rejected noninteractive authentication,
  so no Gao SWC was downloaded from the login node. No modeling or morphology,
  target, branching, terminal, voxel, or final outcome access occurred.
- Evidence: `outputs/stage0/gao_mop_qualification.md` and
  `outputs/stage0/gao_source_group_support.csv`.
~~~~

## EP11

~~~~text
EP11 — Do nearby somatosensory neurons use reusable output patterns?
Current log entry (2026-09-30):
2026-09-30 — `population_h_optimizer_recovery`: requested babysitting resumed
- User requested “ok! continue the babysit”. Follow the same job46143347 through completion or its existing cutoff; no new experiment or extension.
- Observed at 2026-10-01T03:06:08.983021+00:00: RUNNING onsh03-06n12, startedSeptember30 18:21:20 PDT,72h endOctober3 18:21:23 PDT, and no requeue. Renewed episode clock deadline remainsOctober5 18:12:23 PDT.
- Completed logged start1/3 `embedded_fitted_C`: objective4.57373073603, KKT_converged=True.
- Completed logged start2/3 `truth_based_nonzero_signature`: objective4.44457198869, KKT_converged=True.
- Only the revised empirical start remains fitting. Completed baseline objectives match the previous attempt; no fresh verification or terminal adjudication exists. No unchanged baseline figure was replotted. Real outcomes remain closed.
~~~~

## EP12

~~~~text
EP12 — When does a cell-type average distort a circuit's wiring?
Current log entry (2026-10-02):
2026-10-02 11:07 PDT — focused BR-guided oracle-gate critique
At the scientist's request to continue and explain the blocker, the current
Slurm observation at 11:02 PDT confirmed job `45736217` RUNNING with 31/35
look-7 shards and job `45736218` PENDING Dependency. No look-7 aggregate exists.
The actual scientific blocker is unresolved oracle precision, followed by
required generated-method qualification before corrected development replay.

BR external scientific-review directive `ext_review_dir_c87e4d3047c6` supplied
four-axis criteria; local reviewer `/root/interpret_gate` examined completed
look 6, the frozen specifications and relevant wrappers. Root cross-checked
the report and descriptive width scaling. BR accepted the submitted external
verdict `ext_review_verdict_084bd26019de`: correctness pass within this limited
scope, judgment sound, completeness incomplete, overall `explore_more`.
This supports the unchanged existing bounded refinement; no diagnosed method
failure or new execution authority follows. The calculation suggests possible
unresolved precision at 12,500 rows but is neither an observed outcome nor an
early-stop rule. No matching KG violation was established; inapplicable MRI
and subject-manifest rules were not introduced as gates.

Actual directive, verdict, submission receipt and evidence-backed critique:
`scientific_qualification/reviews/oracle_gate_20261002/`. Reused and inspected
the existing completed-look-6 QC figure; no new synthetic experiment or real
connectivity access. This external review is not Formal Society or canonical
acceptance. ASTRA program/export state remains as previously recorded.
~~~~

## EP14

~~~~text
EP14 — What Makes a Neuroanatomical Age Representation Transfer?
Current log entry (2026-10-06):
2026-10-06 00:29 Pacific — `bilateral_regularization`: complete; shared allocation closed
All three views/five arms completed in 221.206665 seconds, workerexit0;
all five focused tests passed0.009seconds, ASTRA exited0, plotting exited0 and
the PNG passed visual review. Independent scientific/source and result reviews
passed. No incomplete views, substituted controls or opened public-validation
rows. Results and full interpretation: `runs/bilateral_regularization/attempt001/`.

Homologous pairing selectedrho1/alpha10000 in every view and exactly reproduces
retuned raw-ridge predictions. Homologous versus each of three fixed shams has
a descriptive age interval spanning0; no anatomical advantage is established.
All shams selectalpha100000/rho0.1, an upper-grid limitation without authority
for expansion. Original no-candidate result and no-promotion boundary remain.

The saved old/new raw comparison verifies equal domain counts and age-quartile
boundaries. Widening the alpha grid reduces domain-macro MAE 5.831111→5.468507
years (−6.22%), but participant-pooled MAE rises 6.949438→7.520867 (+8.22%),
worst-domain 9.603874→11.976468, and oldest-quartile 13.568802→17.700589.
The original primary-objective improvement must be reported with these harms.
No paired uncertainty was computed for that cross-run contrast; it is an
exposed-development description, not confirmation or a biological mechanism.

Parent 46735127 COMPLETED0:0, October6 00:22:22–00:29:30 Pacific,
428elapsedseconds/4CPUs/16GiB. The attended PTY54614 exited0 and is closed.
Slurm stepMaxRSS619804KiB. Charge 1712 allocated CPU-seconds =
0.4755555556CPU-hours once to EP14, including setup, all checks/ASTRA,
EP01 saved-score analysis, EP05 saved-component figure, EP14 fit/uncertainty/
plotting and closure. Do not add420-second step or430-second extern times.
Replace4CPU-hour reservation with actual cost; reservation now0. Preserve all
prior consumption. The aggregated work finished faster than estimated and
the allocation was closed promptly without padding or further search.
~~~~

## EP15

~~~~text
EP15 — Why do people have different cerebellar task maps?
Current log entry (2026-10-13):
2026-10-13 22:20 Pacific — observed_beta_development — cost008 reached fit 65; local vertex proof needed
47734809 failed 1:0 after 9m36s; cost008/operational.json records 562.97 s and 210,712 KiB peak RSS.
The repaired optimizer route progressed to fit 65, whose diagonal optimum candidate has eight lower and eight upper bounds.
The scalar-dual witness is insufficient and the conditional profile switches branches; nonuniqueness is not established.
Evidence: failed_global_diagnostic019.json and its protected Task-A-only objective cache, without Task-B summary access.
Select a coupled local-vertex growth certificate followed by exclusion of the rest of the original domain.
Retain all original growth, gap, rank, actual-operator checks and the complete panel; qualify on the saved objective before retry.
~~~~

## EP16

~~~~text
EP16 — Why does a human iBCI mapping work in one session but not another?
Current log entry (2026-10-06):
2026-10-06 — critique revision: support interpretation and unadopted closure
The existing paper plan now distinguishes the six/three role allocation from
the four-supported-development/three-supported-held-out requirement. The
conditional T11/T9 passes are not full eligibility. Rejected an unsupported
monotonicity claim: authentic session/edge exclusions can change conditioning,
information, leverage and connected components, so current failure is not
universal source infeasibility. Proposed the registered support-limited
disposition with explicit reopening conditions and an optional T11-development,
T9-evaluation case-series draft. Both recommendations remain unadopted.
No source, neural outcome, scheduler state or test was read/run; no policy,
role, margin, resource allowance or access boundary changed.
~~~~

## EP17

~~~~text
EP17 — Which visual-model advantages survive changes in the measured cortex?
Current log entry (2026-10-13):
2026-10-13 13:47 Pacific — fixed_panel_adjudication — complete released-panel and source review passed; reporting selected
Independent EP17-scoped review read the exact summary released by the
coordinator above, all four pair records/all20 edges, the adopted amendment,
the reporter and the existing `uncertainty_core.relation_decision` convention.
No other private artifact, checkpoint, calibration input, source response or
audit path was opened. The strict opposite-margin crossing and inclusive
equivalence bounds agree. All P1/P2/P3 complete-family intervals lie inside
their practical margins; F1 remains underidentified and nonpromotable; all20
registered edges lack qualifying reversals. There is no source-review blocker
or scientific change to `report_saved_fixed_panel.py`.

The missing H_BC/H_CD components arise from the frozen undefined calibration
gain in sub-02/right. They make eight beta explanatory forecasts inapplicable,
not their primary comparisons invalid. Four size-control forecasts are absent
by design; eight support/normalization forecasts remain applicable but have
not been tested on audit concepts. No false trial credit is supplied by these
forecasts, separate scales, participants, ROIs or interval families. Twenty
valid primary atoms complete only the newly adopted fixed panel, not the
original adaptive search. No candidate or audit is selected.

Select the reviewed reporter once as root's separately scoped≤120-second
saved-result companion after substantive EP06 work. It writes the complete
20-row CSV, JSON adjudication, development interpretation and all-endpoint
PNG/PDF from existing intervals without refitting/resampling. ASTRA now declares
`fixed_panel_adjudication` with the complete-panel and exact released-summary
dependencies. The main episode report points to this development closeout;
conditional audit stages remain preserved. GOAL/DATASETS and the living paper
plan distinguish the current adoption from historical pending language.

No tests, scheduler calls, imports or scientific computation ran on this
login shell (`sh04-ln04`, no SLURM_JOB_ID). Root owns execution and measured
resource accounting. Existing5.07+0.253055556CPUh and earlier charges remain;
the companion will not be double charged against a shared parent. Generated
output and visual review are still pending at this entry.
~~~~

## EP18

~~~~text
EP18 — Do different pictures of the same concept share an EEG pattern?
Current log entry (2026-10-13):
2026-10-13 22:16 Pacific — false_partition_preparation — selected finite search exhausted
47692682 ended with exit 2 after 7h03m27s; its public search/report.json records exhausted_unavailable.
The 64 starts used 6,400,000 legal proposals and yielded no accepted member; no candidate reached exact distribution checks.
This is exhaustion of the selected search, not a crash or proof that the required matched bank cannot exist.
The preserved checkpoint is terminal and the unchanged resume path rejects it; do not blindly resubmit.
Evidence: runs/false_partition_preparation/bounded_continuation_operations/segment_47692682/search/report.json.
All eight matching requirements remain. Prioritize the independently authorized complete-cohort G/T estimate;
at 22:15 Pacific, 24 of 30 core jobs had completed and the remaining six were running, with audit closed.
~~~~

## EP19

~~~~text
EP19 — Can EEG forecast movement before peripheral sensors detect its onset?
Current log entry (2026-10-11):
2026-10-11 01:52 Pacific — source_views_and_timing — 155 metadata rows complete; three competing bindings
Parent47357234's independent EP19 task completed successfully in its explicit
one-CPU step on sh03-06n05. Four focused reader methods passed in0.189seconds,
the metadata process completed in0.507209seconds, and ASTRA schema/semantic
validation passed. Evidence is in `runs/source_views_and_timing/column_stripe_job_47357234/`
and `column_stripe_job_47357234_metadata/development_stream_layout.json`.
All155 frozen development triplets were inspected using binary stat-only and
NewSegment-only metadata; no signal arrays, labels, held-out or WAY data opened.

The declared publisher filename mapping supports 152 file associations.
It resolves the three earlier header-reference skips and three legacy marker
references for metadata purposes while preserving all nine original defects.
The remaining sub-03/session02/runs0009,0010,0011 each name the extant
run0008 EEG in their marker DataFile declaration. These are competing
authentic references, not merely absent legacy names; their marker association
remains unresolved. No affected run is excluded and no trace, synchronization,
calibration, valid-run or forecast claim follows from accepted filename mapping.

The EP21 parent subsequently closed with its own pre-measurement fixture-file
cap failure, which does not invalidate this completed independent metadata task.
The whole parent26seconds×3CPUs is charged once to EP21; no EP19 duplicate cost.
No EP19 rerun is selected. Further work needs resolution of those three
competing associations plus the existing physical calibration/filter-delay,
onset uncertainty, WAY clock and valid-run questions.

Focused review of the already captured sub-03/session02/run0008–0011 records
finds distinct nominal frame counts144580/144645/144555/144515 and distinct
NewSegment starts11:23:28.518207,11:35:26.187849,11:51:51.095835 and12:04:02.445661
on2023-09-29, timezone unspecified. Thus whole EEG files and the retained
NewSegment records are not identical; copied/overlapping signal content and
editing history remain untested. Repeated legacy DataFile text is plausible,
not established. Keep same-file frame/layout metadata while withholding the
three absolute marker-clock associations. A targeted source correction is
needed if those absolute starts are required; the conflict does not invalidate
unrelated layout records or justify another extraction. No new payload was read.
~~~~

## EP21

~~~~text
EP21 — Can neural codes be converted across people and imaging sites without shared stimuli?
Current log entry (2026-10-12):
2026-10-12 17:09 Pacific — interindividual_converter_training — exact-constant guard passed
The selectedattempt002 completed in179.832276seconds inEP17parent47577068,
with601,714,688bytes peakprocessRSS. The full first-fold sub01 target axis
contains exactly one invalid scale: finite zeroSD, exactly constant fitting
values and originalfloat32 mean exactly equal to the first value. Nonfinite
means/SDs, negativeSDs, varying-zeroSD columns and inexact-mean constants all
have countzero. The registered guard is satisfied. Evidence:
`runs/brain_constant_discriminator/attempt002/report.json`.

This is first-fold numerical eligibility only: no fit, normalization, model,
score or tau-test access occurred; all-fold/full-panel eligibility remains
unproven. The whole parent is charged once toEP17. The conditional scientist
decision now admits isolated adapted implementation, retaining original code,
models and not-evaluable dispositions. Three agents/root are implementing and
reviewing separate normalization, artifact and single-pair execution modules.
Select fresh adapted fits for all20pairs rather than relabeling the12 original
models; numerical guard checks remain independent in every fold and final fit.
Separate finite resource admission and affected checks still precede any fit.
Full readout47382805 remains held; no evaluation release follows this result.
~~~~

## EP22

~~~~text
EP22 — Spectrum, eigenvectors or stimulus selection? Decomposing the VLM–DINOv2 brain-geometry gap
Current log entry (2026-10-12):
2026-10-12 20:49 UTC — d1/d2/d3 — representative cell passed; grids widened; full run selected (attempt02)
Representative cell. Job 47547277_0 (sub-01, fold 0) ran on sh04-01n14
and ended `COMPLETED 0:0`: 6 min 44 s, MaxRSS 3.4 GB, 4 CPUs.

- R0 reproduction: all 16 evaluable rows match EP04's observed native
  fisher z; the maximum absolute difference is 5.6e-17.
- D1: native geometry equals R0 exactly.
- Timing: D1 185 s, D2 181 s, D3 13 s.

The cell is preserved at
attempt01/cells/sub-01-fold-0.json.
It is superseded for aggregation because its α and λ grids were too narrow
(below). Its single-cell values are not interpreted.

Grid defect and fix. Both are open exploratory settings in a policy that
is not frozen.

- DINOv2's R1 exponent was selected at the upper bound, α = 2.0, so the
  optimum may lie beyond the grid. The α grid is widened to 0–4 in steps of
  0.1.
- R2's split-half-validated λ was often the largest value, 1.0. The λ grid
  is widened to {0, 1e-3, 1e-2, 1e-1, 1, 10, 100}.

These changes only fix optimizer bounds. The estimands are unchanged.

Full run. All 25 cells are rerun with the widened grids into
`runs/d123_development/20261012-attempt02/`.

- The array runs at most 8 tasks at once (`%8`, 32 cores), matching the
  policy's `max_parallel_cores`.
- A dependent collector (`afterany`) aggregates the cells and renders
  `figures/d123_development/20261012-attempt02/`. It also re-renders the D0
  figure, changing only the tick labels, and validates ASTRA.
- Estimated cost is about 25 × 9 minutes × 4 CPUs, roughly 15 allocated
  CPU-hours.

Submitted at 20:50 UTC.

- The array is job 47548992 (`--array=0-24%8`).
- The first collector submission, 47548994, was cancelled before it started.
  It was resubmitted as 47549215 with `sbatch --wait`
  (`--dependency=afterany:47548992`), so completion arrives as a notification
  instead of by polling.
~~~~

## EP23

~~~~text
EP23 — A conditional spectral-operator account of representational convergence
Current log entry (2026-10-14):
2026-10-14 03:15 UTC — a1 — rank-64 complete
- Ran: A1 array 47703628 (25 participant × fold cells, rank 64) and collect 47703629, all `COMPLETED`; a1_report.json.
- Primary, full − diagonal held-out Δz: +0.19 mean over models (DINOv2 0.13 to CLIP 0.23), 0.17–0.23 per participant; positive 5/5 for every model; exact sign p = 0.031 (the n = 5 floor).
- Mean held-out RSA z, primary ROIs: reference 0.11, native 0.15, power 0.18, diagonal 0.21, full 0.41. EVC: 0.04, 0.04, 0.04, 0.05, 0.13.
- Secondary, power − native: +0.023 (0.010–0.046), 5/5, p = 0.031. Selected α: CLIP 1.5–1.6, DINOv2 1.9–2.1, PEcore 1.8–2.1, SigLIP2 1.2–1.9; never at a grid edge.
- λ0 = 0 (no shrinkage toward native) chosen in 100/100 primary model-cells for both fitted families; this is the grid's lower limit, not a truncation.
- Reading: held-out brain geometry needs rotation within the models' top-64 PCs, not only per-axis reweighting. Per P4 this is predictive sufficiency, not native geometry. Exploratory, development data.
- Next: rank-32 sensitivity (same code, RANK=32, new RUN), then the stage figure and summary.
~~~~
