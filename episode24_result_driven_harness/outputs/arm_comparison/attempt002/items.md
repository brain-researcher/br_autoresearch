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

## EP05

~~~~text
EP05 — Does the LFP explain why one reach was different?
Current log entry (2026-10-04):
2026-10-04 21:48 UTC — `adaptive_search`: complete PLS panel; 44 comparisons, patience 11
Slurm46519201 completed with exit0 at21:40:35UTC after153seconds on one
CPU (0.0425 allocated core-hours). TotalCPU127.262s and batchMaxRSS1965528K
are separate process measurements. The remainder wrapper completed141.412s,
including18.218s full-panel replay/summary. The preserved resource pilot plus
fifteen fresh sessions complete all16. Compact evidence was released only
from the completed summary and passed its512KiB bound (469031bytes):
runs/adaptive_search/primary_history_pls_summary_attempt001/compact_evidence_attempt001.json.

PLS primary increment0.00036033660315523165 and identity0.001405621668816818
miss the original0.01/0.005 margins. Primary is0.00043220384738284584 in
Mihili and0.0002884693589276175 in Chewie, with5/6 and8/10 positive sessions.
Both animal identity means and every within-animal omission remain positive.
The full target-label-refit mean control is−0.0003127137807993053 primary,
0.0002822518517316857 identity. Candidate-minus-control is
+0.000673050383954537/+0.0011233698170851326; all eight individual draws
are worse in both animal summaries. This supports fitted target correspondence
under the fixed PLS recipe, not biological mechanism or matched effective
capacity. Every128 candidate and1024 control fold fits retained two components;
all16 outer-session selectors chose the existing strongest lambda10. No grid
or component-count extension follows from this result.

PLS loses both independently selected HW and the W/HW blend: primary contrasts
−0.0019809178700217526 and−0.0023185178279447544. The blend incumbent stays
0.0026788544310999856. Even lambda-matched HW is better on primary and identity.
Literal full-panel duplicate checks find neither new policy duplicates a
previous anchor. Candidate then mean control add two distinct comparisons:
42→43→44, patience9→10→11/12, post-initial36→38, alternative credit22→23,
literal-null14→15. Search remains open under its original stopping policy;
no promotion, mechanistic follow-up, unseen-session access or penalty amendment.

Independent compact-only interpretation agrees. The scalar figure renderer
initially lacked matplotlib in the active module environment; module discovery
found installed py-matplotlib/3.8.3_py312. Loading that module repairs plotting
only, with no installation or scientific rerun. The figure rendered successfully with24 plotted scalar rows and passed visual
inspection; PNG/SVG/CSV are under figures/primary_history_pls. Shared allocation46518769
preparation cost remains to be finalized separately and charged only once.
~~~~

## EP07

~~~~text
EP07 — What do earlier recording days preserve: task structure or trial-specific coupling?
Current log entry (2026-10-10):
2026-10-10 22:50 Pacific — matched_search — fourteen starting pairs scored; two convergence failures
Job 47336768 ended FAILED, exit1:0, at22:45:52 Pacific after256seconds
with four allocated CPUs. Charge 0.2844444444 allocated CPU-hours once to
EP07 and release the eight-hour reservation; retain the earlier preparation and
qualification charges. Batch MaxRSS was3079088K. This is an incomplete model
batch, not a scheduler fault. All36 focused implementation tests passed in4.763s,
and ASTRA schema/semantic validation passed before fitting.

The separate development feedback
records 14 complete paired contacts, or28 stream evaluations, over all five
new days and both regions. Starting08 and10 failed the elastic-net1000-iteration
limit at the specified absolute KKT tolerance1e-5 in both streams. Their partial
prediction files remain preserved and unscored. Four failed fits use one attempt
each; successful scored candidates will not be refitted or rescored for repair.
The numerical exception lacked final-KKT diagnostics, so source diagnosis is
selected before the second and final permitted attempt of each failed model.
No scientific threshold, penalty, L1 ratio, feature scaling, data role or model
opportunity changes are implied by a solver repair.

Among the14 completed comparisons, the largest global development selection
score is0.0657872387 for history starting13; the largest today-only score is
0.0343199661 for starting09. These are different provisional leaders from an
unfinished search, averaged over the two regions for selection only. They are
not held-out evidence, a scientific margin test, final model selection or proof
of trial-specific coupling. All held-out data remain closed. The sixteen-model
starting stage, adaptive successors, ten finalist checks and final qualification
remain incomplete.

The96-hour clock began at2026-10-11T05:41:59.230511+00:00 in
`runs/matched_search/search_clock.json`; repairs retain that clock and all costs.
The first aggregate batch finished sooner than forecast. Subsequent short checks
will be combined with substantive real fitting rather than submitted alone.
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
Current log entry (2026-10-14):
2026-10-14 — core_reference_calibration — complete real-data estimate released
47743173 completed 0:0 in 12m57s; the reporter accepted all 30 original people, 900 logical inner validations and 180 outer folds.
G = −7.574192738e-6, descriptive 95% participant interval [−5.604464579e-5, 4.089626032e-5].
T = −0.0004747710805, interval [−0.0007831690300, −0.0001663731310]; six folds are averaged within each person before t(df29) inference.
Report: runs/baseline_models/core_direct_checkpoint_batch_report001/core_reference.complete.json and exactly three lines in summary.md.
Figure: figures/core_direct_checkpoint_batch001/core_reference_G_T.png and PDF; both panels were visually inspected.
G <= 0 invokes the existing underidentified reference rule; this fixed core does not support a concept claim or absence-of-signal inference.
Same-account procedural separation is disclosed; concept/false scores and all 16 audit participants remain closed.
BR pipeline_plan_review returned approve/low with no findings or KG rules; its generic checklist is advisory, not statistical validation.
The focus goal's next EP18 estimate/uncertainty/figure/summary is delivered; preserve the broader unresolved study and advance other focus episodes.
Step 47377474.45 passed pinned ASTRA 0.0.14 schema/reference validation for the new finding after reusing an available YAML parser.
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
