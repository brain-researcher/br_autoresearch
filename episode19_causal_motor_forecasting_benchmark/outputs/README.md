# Outputs

**October 17 — v5 follow-up diagnostics (job 48281685): premature-R diagnostic O4 UNRESOLVED; control precision limits the linear near gate (calibration/diagnostic evidence, post hoc to v5).**
[PRD run](runs/source_views_and_timing/premature_r_diagnostic_job_48281685/) · [CPC run](runs/source_views_and_timing/control_precision_job_48281685/) · [result](source_timing_handoff.md#result-job-48281685) · [figure](figures/ep19_v5_followup_diagnostics_20261017.png) · [rules](source_timing_handoff.md#october-17-premature-r-diagnostic-and-control-precision-check--pre-declared-before-execution).

- **Summary — premature R:** pre-declared O4 UNRESOLVED in 14/15 people, with all known-positive checks passed. ΔH = 19 [6, 82] ms; ΔC_E = −0.149 [−0.292, −0.029]. No automatic detector or reference change follows.
- **Summary — control precision:** linear near sensitivity is 0.893 [0.884, 0.902] with all 2,545 backgrounds (v5's 32: 0.888) against the 0.90 gate. CP1: 1 below, 15 across, 29 above. Timing error explains every near classification. Sub-02's v5 sham of 0.070 is retained (all backgrounds: 0.051 [0.026, 0.078]).
- **Summary — status:** v5 D2/D1 stay UNDERIDENTIFIED and nothing is frozen. v1 labels are preserved and self-paced primary scoring stays blocked. No EEG or held-out person was used.

**Post hoc:** most tail R→L intervals are intermittent and fall back after R. 54/289 tail events show a substantial departure ≥ 288 ms before the label (mostly sub-06/20/14).
**Next:** pre-declare a fixed-label gap/H diagnostic. The daily batch gets three questions: estimator resolution versus gate logic, a continuity-revised acceptance reference (v6), and p95 versus worst case. WAY development non-neural baseline selection can proceed.

**October 17 — self-paced detector v2 / raw-onset v5: UNDERIDENTIFIED (calibration/diagnostic evidence).**
[Run](runs/source_views_and_timing/raw_onset_test_v5_job_48261284/) · [candidate detector outputs](runs/source_views_and_timing/detector_v2_job_48261284/) · [result](source_timing_handoff.md#result-job-48261284) · [figure](figures/ep19_detector_v2_acceptance_20261017.png).

- **Summary — decision:** D2 and D1 UNDERIDENTIFIED: 8/15 development people passed k=5 controls, and cross-k classifications disagreed. No candidate frozen.
- **Summary — timing:** D2 k=5 p95 bounds [324,324] ms (+12=336); D1 [360,360] (+12=372). Numeric FAIL classifications are descriptive; k=8's D2 total of 248 ms has 0/15 control passes.
- **Summary — readiness:** Preserve v1 labels/exposure from job 48219799; keep self-paced primary 300–600-ms scoring blocked. No EEG result, held-out validation or worst-case certification.

Job 48261284 completed in 83 s, exit 0; both regressions passed. The declared v5 rules were retained after the post hoc controls-only diagnostic. Both candidates passed noise-only admissibility at c=1, LOW/HIGH 0.027917/0.053088 g. D2 has 3,964 onsets and 2.153 h at risk versus v1's 4,189 and 2.612 h; the smaller lag tail is a descriptive cross-version comparison.

**Strongest alternative (post hoc):** low-amplitude activity/drift makes R premature. Tail R−provider median −260 ms versus −15 elsewhere motivates investigation, but selection on L−R and shared accelerometry limit the evidence.

**Next (post hoc):** prospectively declare fixed-label continuity/amplitude/drift diagnostics and control-precision checks at the same 288-ms margin. Retain all people and gates; development-derived slopes supplement the original controls. c=0.75 is a separate out-of-ladder engineering proposal, not a v5 result. Changing k or acceptance gates requires the scientist; any BELOW needs passed positive controls at the same margin.

WAY development non-neural baseline selection can proceed. Exclude held-out tuning/normalization initialization. The single daily scientist batch retains the timing-guarantee decision, minimum meaningful primary EEG gain and other unset margins; scored EEG remains uncleared. [Previous v4 result](source_timing_handoff.md#result-job-48230171).

**October 17 — self-paced raw-onset test: UNDERIDENTIFIED (calibration/diagnostic evidence).**
[Run](runs/source_views_and_timing/raw_onset_test_job_48230171/) · [result](source_timing_handoff.md#result-job-48230171) · [figure](figures/ep19_raw_onset_lag_test_20261017.png).

- Job 48230171: 47 s; 15 development people, 155 runs, 4,189 onsets; 3,661 stationary windows reproduced exactly.
- Full-run v4 included post hoc smoke-informed revisions; exposed sub-02/sub-15/sub-22 remain included.
- k=5 all-event p95 lag bounds: 696–696 ms (+12=708 ms). This challenges the 188-ms proxy; failed controls prevent a validated timing-failure verdict.
- Controls: 0/15 people passed all requirements; 14/15 exceeded the sham ceiling; pooled k=5 sham alarms were 11.3%. No control-passing subset exists.
- Post-lock agreement: R−provider median/p95 −16/60 ms at k=5; L−provider 228/564 ms. Shared accelerometry limits independence.
- Keep self-paced labels out of primary 300–600-ms scoring; preserve the original frozen artifacts. No EEG forecasting result or worst-case certification.
- Next (post hoc): predeclare a global detector redesign from all 15 development people and corrected acceptance controls—feasible targets, primary-bout-aligned shams, verified 1-kHz calibration; exclude held-out tuning and normalization initialization.
- WAY development non-neural baseline selection can proceed.
- One daily scientist batch: p95 versus GOAL's worst-case requirement, minimum meaningful primary EEG gain and other unset margins; scored EEG remains uncleared.

**October 17, 00:18 Pacific — development detector freeze complete (calibration, not a scientific result).**
Rules were fixed from development stationary noise before any detector output
([§1.2](component_details.md#12-detector-freeze-rules-october-16-written-before-any-detector-output)).
Final job 48219799: [summary](runs/source_views_and_timing/detector_freeze_job_48219799/detector_freeze_summary.json) ·
[figure](figures/ep19_detector_freeze_20261016.png) ·
[log](experiment_log.md#2026-10-17-0018-pacific--source_views_and_timing--development-detector-freeze-complete-calibration).
The generic replay matched `WayRiskStream` on 84/84 WAY series.

- **Labels and exposure for every development person:**
  - WAY: 2,030 of 2,712 cues armed, all with onsets; 0.21 h at risk.
  - Self-paced: 4,195 of 4,432 armed, 4,189 onsets; 2.61 h at risk.
  - Primary-horizon (300–600 ms) positives / scorable decisions: 3,527 / 14,847 (WAY) and
    24,005 / 187,972 (self-paced). WAY positives need reaction times > 300 ms, so slower people
    dominate (P3: 3, P7: 846).
- **Timing bound:** the pre-declared p95 term gives WAY 247 ms and self-paced 188 ms (< 300).
  - Post hoc: this is not a worst-case proof. Self-paced labels follow AccStartIndex by median
    228 ms (p95 564 ms; that alternative bound is 576 ms).
  - [Details](source_timing_handoff.md#october-1617-detector-freeze--measured-crossing-term-and-timing-bound).
- **Untouched:** no EEG, WAY S8–S9 or held-out person.

**Next:**
- Self-paced: pre-declare and run a provider-blind raw-onset lag test of the self-paced bound
  (the strongest alternative).
- WAY: development non-neural baseline selection can start.
- Daily question: may a p95-term budget stand for GOAL's worst case?
- Scored EEG still needs the scientist's minimum meaningful primary EEG gain.

**October 16, 18:40 Pacific — source facts resolved or assumed; no provider contact.**
All eleven open WAY and self-paced source items are classified in the
[source-fact table](source_timing_handoff.md#october-16-source-fact-classification-current-supersedes-the-open-items-below).
Checks were pre-declared and ran in job 48183086, with post hoc follow-ups
(labelled as such). [Figure](figures/ep19_source_facts_20261016.png) ·
[log entry](experiment_log.md#2026-10-16-1840-pacific--source_views_and_timing--all-eleven-open-source-facts-classified).

- **Resolved:**
  - WAY windows start at HS sample `500·trial_start_time − 1000`, so cues come
    from the HS LED channel.
  - Positions are in cm and forces in N; LED polarity holds in 84/84 files;
    surface codes are 1 = silk … 3 = sandpaper.
  - Self-paced sub-03/ses-02 bindings are a stale marker line; no development
    run is invalid; 1 g = 1.4465×10⁶ µV.
- **Assumed, with sensitivity plans:**
  - WAY filter delays and drift, giving a worst-case bound of 220 ms < 300 ms;
  - the actiCHamp filter and absolute TRIG latency;
  - unmarked gaps (31 ambiguous cue cycles).
- **Untouched:** no EEG model was fitted and no held-out file was opened.

**Next analysis step:** freeze the causal WAY and self-paced onset/stillness
detectors from development stationary noise, then produce labels,
eligible-stillness exposure and the detector-crossing term of the timing bound.
Provider HandStart/AccStartIndex serve only as agreement checks. The
development non-neural baseline selection follows. Scored EEG comparisons still
need the scientist's minimum meaningful primary EEG gain.

**October 12, 11:21 Pacific — three final-refit companion checks and ASTRA passed.**
All three selected `test_reference_final_refit` methods passed in 0.184 seconds,
with ASTRA schema/semantic validation also passing, in the closed EP19
companion of EP21 parent 47373800. These are separate from the three
development-driver checks below. Cost is included once in the EP21 parent's
**5.726667 allocated CPU-hours actual charge**; no separate allocation or EP19
empirical access occurred. That parent later received a recorded
scientist-directed stop; the separately closed checks and ASTRA remain passing. See
[completed evidence, methods and limits](experiment_log.md#2026-10-12-1121-pacific--reference_panel--final-refit-companion-checks-and-astra-passed).
Source timing/calibration, run bindings, genuine conditions, labels/prefixes,
margins, enforced access and production resource admission remain unresolved.
Earlier pending-check descriptions below preserve their observation-time status.

**October11,23:54Pacific — all three complete-source driver checks passed.**
The new fixed-recipe development composition passed all three focused methods
in5.558seconds, and changed ASTRA validation passed, inside the closed
1CPU/512MiB/noGPU step47375558.0. Slurm confirms COMPLETED0:0; its cost is
included once in the existing EP21 parent. Both original content fits continue
and their submitted runtime is unchanged. See
[verification, resource evidence and remaining prerequisites](experiment_log.md#2026-10-11-2354-pacific--reference_panel--complete-source-driver-checks-passed-in-closed-cpu-step).
This verifies software composition, not an empirical fit or source release.
The separate three final-refit checks in47373800 remain pending. Earlier
unrun descriptions below preserve their observation-time status.

**October11,23:49Pacific — complete-source fixed-recipe driver implemented and reviewed.**
The new driver connects existing fitting, matched prediction/calibration and
scoring across all72 WAY or five self-paced development folds. It reuses
frozen baselines and shared condition labels/offsets, releases each fold's
EEG models and returns only complete-source feedback. Root and independent
source review found no actionable defect. **Three focused checks and changed
ASTRA validation remain UNRUN**; no new compute or empirical fit was launched.
See [implementation, pending verification and remaining prerequisites](experiment_log.md#2026-10-11-2349-pacific--reference_panel--complete-source-fixed-recipe-driver-implemented-and-reviewed)
and the [component interface](component_details.md#complete-source-fixed-recipe-development-execution).
Source access/timing/calibration, genuine conditions, qualified labels/prefixes
and production resource admission remain unresolved. All submitted EP21
runtime and the separate pending final-refit checks remain unchanged.

**October11,17:12Pacific — complete-source development scoring connected and checked.**
The new per-recipe composition joins all72 WAY or all five self-paced
validation folds to the unchanged matched scorer, preserving every participant,
row/label identity, calibration mode and condition. Independent source review,
both focused tests and ASTRA validation passed in attended job **47434740**;
Slurm confirms COMPLETED0:0 and the compute shell is closed. Actual charge:
0.021389allocatedCPUh. See [implementation, verification and limits](experiment_log.md#2026-10-11-1712-pacific--reference_panel--complete-source-scoring-composition-verified)
and the [component convention](component_details.md#complete-source-development-scoring-composition).
No real-data fit or scientific result was produced. Source timing/calibration,
qualified labels/prefixes, run bindings, margins, data access and empirical
fitting orchestration remain unresolved. The separate final-refit checks in
47373800 are still pending; all submitted EP21 runtime is unchanged.

**October11,09:16Pacific — three development prediction checks and ASTRA passed.**
All three selected methods passed in **0.239 seconds**, and ASTRA schema and
semantic validation passed, in the closed EP19 software companion of EP21
parent **47375558**. Both exits are zero; the parent charge is counted once.
The parent was still running at the coordinator's observation. See the
[completed evidence and limits](experiment_log.md#2026-10-11-0916-pacific--reference_panel--development-prediction-checks-and-astra-passed).
No EP19 empirical access or fit occurred. The separate three changed
final-refit checks in **47373800** remain pending; source timing/calibration,
source/run-role bindings, qualified labels and calibration support, margins
and empirical orchestration remain unresolved. Earlier dated entries below
retain their observation-time status.

**October11,05:46Pacific — matched development prediction/calibration implemented.**
The new array-only bridge connects the frozen baseline and four EEG conditions
to one complete validation fold, preserving shared offsets and separate
prefix/suffix calibration. Independent source review passed after correcting
cross-condition filter matching. Three checks and ASTRA remain **unrun**,
assigned to submitted EP21 pair5 parent47375558 as an independent software
companion. See [implementation and
limits](experiment_log.md#2026-10-11-0546-pacific--reference_panel--matched-development-prediction-and-calibration-bridge-implemented).
No EP19 source release or empirical fit occurred.

**October11,05:32Pacific — companion parent47373800 submitted.**
Three changed ensemble checks and ASTRA remain **unrun**, waiting for
successful EP21 training47370327. EP19 results will be recorded separately;
no real-data fit is opened. See [submission record](experiment_log.md#2026-10-11-0532-pacific--reference_panel--independent-companion-parent-submitted).

**October11,05:30Pacific — changed ensemble checks assigned to an existing dependency chain.**
The three changed methods and ASTRA validation will run as an independent
software companion in EP21's selected first-readout parent, after successful
47370327. They remain **unrun**; no EP19 real-data execution is opened.
See [scope and shared accounting](experiment_log.md#2026-10-11-0530-pacific--reference_panel--changed-ensemble-checks-selected-as-independent-EP21-companion).

**October11,05:19Pacific — development-fold EEG ensemble composition implemented.**
The original pipeline now joins the selected baseline's fitting-fold offsets
to all three EEG seeds for either source, while preserving final refitting.
Independent source review passed; the three extended checks and changed ASTRA
remain **unrun**. See the [implementation and remaining execution blockers](experiment_log.md#2026-10-11-0519-pacific--reference_panel--development-fold-ensemble-composition-implemented)
and [component convention](component_details.md#final-eeg-reference-refit-composition).
No real-data fit or source release occurred. Earlier validation below applies
to its recorded code state; unchanged backend evidence remains reusable.

**October 11, 02:37 Pacific — eight final-refit checks and ASTRA passed.**
The [three new composition and five affected scope/report checks](runs/reference_panel/cuda_companion001/checks.txt)
passed in 0.034 seconds in the completed CPU child of EP21 parent **47360846**.
[ASTRA schema and semantic validation](runs/reference_panel/cuda_companion001/astra_validation.json)
also passed; the [EP19 child status](runs/reference_panel/cuda_companion001/status.txt)
is closed with both exits zero. This verifies the selected software interfaces,
not the full forecasting pipeline or empirical support. Source timing,
calibration, label qualification and empirical integration remain unresolved.
No source release, real-data fit or final evaluation occurred. The parent's
separate GPU measurement continues; all parent cost is charged once to EP21.
See the [result and limits](experiment_log.md#2026-10-11-0237-pacific--reference_panel--eight-focused-checks-and-astra-passed)
and [final-refit convention](component_details.md#final-eeg-reference-refit-composition).
Earlier dated entries retain their observation-time status.

**October 11, 01:52 Pacific — all155 metadata rows complete; three competing bindings remain.**
The [fresh report](runs/source_views_and_timing/column_stripe_job_47357234_metadata/development_stream_layout.json)
supports152 documentary file associations. Sub-03/session02/runs0009–0011
each reference the existing run0008 EEG from their marker files, so those
three associations remain unresolved. All nine original reference defects
are preserved; no runs were removed. Four focused checks and ASTRA passed.
See the [evidence, limits and parent accounting](experiment_log.md#2026-10-11-0152-pacific--source_views_and_timing--155-metadata-rows-complete-three-competing-bindings).
Physical timing/calibration and valid-run questions remain open. No EP19
rerun is selected; earlier preparation entries below are historical.

**October 11, 01:43 Pacific — documentary filename mapping selected.**
The publisher defines matching participant/session/run EEG, header and marker
filenames as corresponding records. Together with the frozen unique triplets,
this supports the [selected metadata reader repair](experiment_log.md#2026-10-11-0143-pacific--source_views_and_timing--documentary-triplet-mapping-selected),
preserving all nine embedded-reference discrepancies. Focused checks and a
bounded metadata continuation will share EP21's separately funded measurement
parent; they have not yet run. Calibration and timing qualification remain open.

**October 11, 01:36 Pacific — development layout report complete; nine bindings unresolved.**
The [report](runs/source_views_and_timing/job_47355564_stream_layout_retry02/development_stream_layout.json)
retains all155 runs:152 binary layouts and NewSegment records were inspected;
three header-reference conflicts and six marker-to-data bindings remain
unresolved. No run was excluded or silently remapped. Declared starts do not
prove gap-free acquisition. See the [results, exact affected runs and accounting](experiment_log.md#2026-10-11-0136-pacific--source_views_and_timing--stream-metadata-complete-with-nine-unresolved-bindings).
All24 unique selected EP19 software checks passed, including the21 previously
unrun return/risk/label/uncertainty checks below; changed ASTRA also passed.
Job47355564 completed0:0 and the compute session is closed. Sensor calibration,
delay bounds and WAY clock binding remain unresolved; no forecast result or
held-out access was produced. Earlier entries below are historical.

**October 10, 22:44 Pacific — nine refit/scope checks and ASTRA passed.**
The completed EP21 parent 47329203 verified final refits and the four affected
training-offset interfaces. See the [evidence](runs/scoring_composition_checks_job_47329203/tests.txt)
and [scope/limits](experiment_log.md#2026-10-10-2244-pacific--non_neural_selection--nine-refit-and-scope-checks-passed).
Later reducer tests remain unrun, including the four uncertainty checks below;
authentic timing/calibration and empirical integration remain unresolved.
No forecasting result or new EP19 job was produced. The parent is charged only
to EP21; earlier validation-pending descriptions are historical.

**October 10 — factorial-equivalence uncertainty reducer implemented; four checks unrun.**
The [new reducer](code/forecast_uncertainty.py) computes the registered joint
40-endpoint bootstrap intervals from supplied participant means, preserving
all twelve WAY and eight held-out self-paced participants. It returns separate
full-reference and interaction equivalence calls; missing margins stay
unresolved and interval containment does not establish the full explanation.
Root and independent source review found no concrete defects. Four focused
checks and changed ASTRA validation remain **unrun**, outside the unchanged
live EP21 companion tail. No empirical data, forecast or inference was produced.
See the [implementation and limits](experiment_log.md#2026-10-10--joint_final_evaluation-selects-registered-factorial-equivalence-reducer).
Earlier dated entries below are historical.

**October 9 — WAY streaming risk reducer implemented; six checks unrun.**
The [stream reducer](code/way_risk_stream.py) connects exact-start return
support to cue-scoped decisions and immutable label evidence. It continues
onset surveillance during requalification, a gap in the older conditional
pilot, and preserves one reach per cue and exclusive end/gap boundaries.
Implementation, separate test authorship and independent source review are
complete. Six focused checks and changed ASTRA validation remain **unrun**;
EP19 now has 26 pending methods, with the latest seventeen outside the unchanged
EP21 companion proposal. Authentic timing/calibration, source integration and
self-paced detection remain unresolved. No empirical forecast or new job was
produced. See the [implementation and limits](experiment_log.md#2026-10-09--source_views_and_timing-selects-way-streaming-risk-integration).
A [new public unit clue](source_timing_handoff.md#october-9-downstream-wrist-position-unit-clue)
narrows the source question without authorizing a conversion or resolving timing.
Earlier dated entries below are historical.

**October 9 — forecast-label projection implemented; five checks unrun.**
The [label helper](code/forecast_labels.py) converts caller-qualified risk and
onset facts into the fixed 19 categories, censoring bins and separate scoring
coverage. It preserves unknown status, suspended decisions and calibration
cutoffs, including earlier unresolved crossings. Root and independent source
reviews are complete. Five focused checks and changed ASTRA validation remain
**unrun**; EP19 has 20 pending methods, with the latest eleven outside the
unchanged EP21 companion proposal. Real source timing/calibration, qualified
risk/onset and prefix construction, and empirical integration remain unresolved.
No new empirical labels, forecast or job were produced. See the
[implementation and limits](experiment_log.md#2026-10-09-1850-pacific--past_only_controls-selects-forecast-label-projection).
Earlier dated entries below are historical.

**October 9 — completed-return predicate implemented; six checks unrun.**
The [new predicate](code/way_completed_return.py) checks the full 500 ms before
an authenticated start against digit clearance, all eight primary low paths,
supported object position/motion and each force plate separately. It preserves
unknown calibration/history, cue ordering and gap resets, and cannot arm late
from post-cue evidence. Independent and root source review are complete;
the six focused checks and changed ASTRA validation remain **unrun**. These six
are outside the unchanged EP21 companion proposal. Authentic clock/trim binding,
physical calibration and cue interpretation, the full timing bound and detector
integration remain unresolved. No empirical forecast or new finding was made.
See the [implementation and limits](experiment_log.md#2026-10-09-1835-pacific--source_views_and_timing-selects-completed-return-implementation).
Earlier dated entries below are historical.

**October 9 — final refits and frozen fitting offsets implemented; checks unrun.**
The [new composition](code/baseline_final_refit.py) supplies complete final
baseline fitting cohorts and a separate development-fold offset helper.
Training offsets are the log of the three-seed probability mean, uncalibrated
and explicitly not OOF. Root source review is complete; five new and four
affected scope checks plus changed ASTRA validation await coordinated compute.
No real data, model fit or final outcome was opened. See the
[implementation and prospective convention](experiment_log.md#2026-10-09-1707-pacific--non_neural_selection-selects-missing-final-refit-and-offset-composition).

**October 9, 14:56 Pacific — actual ensemble cost callback checks passed.**
The [new callback](code/baseline_ensemble_cost.py) fills the existing baseline
driver's missing parameter-count and inference-time measurement interface. It
uses actual fitted three-seed models and fixed label-free fitting-fold probes;
it does not read source data, fit models or resolve end-to-end sensor latency.
All three focused checks passed in 0.486 seconds and changed ASTRA validation
passed in shared parent 47133348, charged once to EP20. The earlier five driver
checks remain valid. Authentic
timing/calibration, qualified source inputs and production resource admission
still prevent empirical selection. See the
[execution evidence and limits](experiment_log.md#2026-10-09--non_neural_selection-selects-cost-callback-companion-in-47133348).

**October 9, 12:46 Pacific — complete development baseline driver checks passed.**
The [driver](code/development_baseline_driver.py) connects all nine non-neural
recipes and three seeds to the full source-specific folds, ensemble, calibration
and model selection. WAY preserves within-person pooling; self-paced uses the
registered calibrated32 primary and paired no-label suffix. Root and independent
source review closed, including the corrected 135-fit self-paced fixture count.
Allfive checks passed in1.135seconds, and changed ASTRA validation passed in
the shared EP20 parent47131800. No empirical forecast was run. Authentic timing/calibration, prepared
source inputs, cost measurement and production resources remain unresolved.
See the [execution evidence and limits](experiment_log.md#2026-10-09--non_neural_selection-selects-prepared-baseline-driver-companion).

**October 9, 00:42 Pacific — horizon scoring implemented; five checks and ASTRA passed.**
The [scoring checks](runs/scoring_checks_job_47086577/tests.txt) passed in
0.029 seconds in attended EP17 allocation47086577, charged once to EP17.
The scorer preserves full follow-up eligibility, matched rows and equal
participant weighting. Source timing/calibration and production data
integration still prevent empirical forecasting. See the
[evidence and scope](experiment_log.md#2026-10-09-0042-pacific--past_only_controls-all-five-horizon-scoring-checks-and-astra-pass).

**October 9, 00:27 Pacific — all five ensemble checks and ASTRA passed.**
The [closed checks](runs/ensemble_checks_job_47085635/tests.txt) passed in
0.112 seconds. Parent 47085635 stopped later at an independent EP17 permissions
check; its 43 CPU-seconds are charged once to EP20. No empirical forecast has
run. See the [evidence and limits](experiment_log.md#2026-10-09-0027-pacific--past_only_controls-all-five-ensemble-checks-and-astra-pass).
Earlier unrun descriptions below are historical.

**October 9, 00:19 Pacific — five ensemble checks accepted in EP20 parent 47085635.**
Startup and results remain unobserved; the parent is charged once to EP20.
No empirical forecast is included. See the
[accepted execution](experiment_log.md#2026-10-09-0019-pacific--past_only_controls-ensemble-checks-accepted-in-parent-47085635).

**October 9, 00:02 Pacific — three-seed ensemble/calibration integration implemented.**
The [adapter](code/ensemble_forecast_backend.py) now averages all three fixed
seed probability vectors before calibration and shares one reset evaluation
between paired suffix conditions. Five focused integration checks and changed
ASTRA validation remain unrun. Earlier 17 passing EEG backend checks remain
reusable; no empirical forecast or source qualification is claimed. See the
[implementation and remaining dependencies](experiment_log.md#2026-10-09-0002-pacific--past_only_controls-three-seed-ensemble-and-calibration-integration-prepared).

**October 8, 23:32 Pacific — all 17 EEG backend checks passed.**
Six GRU, five TCN and six EEGNet checks passed in **4.885 seconds** inside EP21
parent **47081340**; changed EP19 ASTRA schema and semantic validation also
passed. These are implementation results. No empirical forecast has run;
source timing/calibration and production integration remain unresolved.
The parent is still running and charged once to EP21. See the
[closed check evidence and limits](experiment_log.md#2026-10-08-2332-pacific--past_only_controls--all-seventeen-eeg-backend-checks-and-astra-pass).
Earlier pending-check statements below are historical.

**October 8, 23:26 Pacific — compact causal EEGNet implemented and source-reviewed.**
The [backend](code/eegnet_forecast_backend.py) preserves the selected kernel
grids, pooling phase and matched original/factorial capacity. Its six checks
join six GRU and five TCN checks: **all 17 remain unrun**, selected together
in the next EP21 work48 replacement's bounded component child. The replacement
has not yet been submitted, and changed ASTRA validation remains pending.
Source timing/calibration, production readers and empirical integration still
prevent forecasting. See the [method, selection and limits](experiment_log.md#2026-10-08-2326-pacific--development_specification--past_only_controls--compact-causal-eegnet-implemented-and-reviewed).
Earlier observations below are historical; no empirical forecast is claimed.

**October 8, 23:20 Pacific — existing EEG TCN implementation added to the handoff.**
The [backend](code/tcn_forecast_backend.py), prepared October 7, now has its
exact original/factorial method recorded and a bounded ASTRA compute recipe.
Its **five checks remain unrun**, alongside the six pending GRU checks; changed
ASTRA validation also remains pending. The named EEGNet backend file is absent.
No empirical forecast has run: source timing/calibration, readers and remaining
pipeline work are unresolved. See the
[reconciled implementation and next check](experiment_log.md#2026-10-08-2320-pacific--development_specification--past_only_controls--existing-eeg-tcn-handoff-reconciled).
Earlier observations below are historical; their passing evidence is retained.

**October 7, 18:35 Pacific — GRU correction implemented and source-reviewed.**
The [new backend](code/gru_forecast_backend.py) preserves the selected fixed
training schedule, past-only windows, participant weighting and matched
original/factorial capacity. Six focused tests and the changed ASTRA validation
await the next suitable compute allocation. No empirical forecast is running;
source timing/calibration and remaining pipeline work still prevent it.
See the [implementation and next check](experiment_log.md#2026-10-07-1835-pacific--development_specification--past_only_controls--gru-correction-implemented).
The earlier passing component results below remain reusable.

**October 7, 05:53 Pacific — boosted and temporal checks passed:** all
**ten** checks (five per backend) passed in **5.863 s** in EP21 job
**46891123** (array **46891123_6**); changed ASTRA schema and semantic
validation also passed. The parent was still running at the coordinator's
12:53:42 UTC observation and is charged once to EP21. See the
[completion record and remaining dependencies](experiment_log.md#2026-10-07-0553-pacific--past_only_controls--boosted-and-temporal-checks-pass),
[tests](runs/component_checks_job_46891123/tests.txt) and
[status](runs/component_checks_job_46891123/status.txt).
These fixture checks supersede the historical unrun descriptions below;
source timing/calibration, raw feature integration, production resource
qualification and the remaining forecasting controls are still unfinished.
No empirical forecast, onset qualification or held-out opening is implied.

**Historical October 7, 03:05 Pacific — temporal non-neural comparator implemented:**
the [new backend](code/temporal_hazard_backend.py) implements the registered
width-16 TCN over complete two-second past windows and prepared summaries,
with the selected left-zero boundary, 18 hazards and fixed 100-epoch schedule.
Its five focused checks and the earlier boosted five are now selected
together in the existing 120-second EP19 component slot; **all ten remain
unrun**, with no job submitted for this selection. This is EP19's portion of
the shared 35-check child under unchanged 540-second-plus-five-second limits;
the parent is charged once to EP21. Source calibration/timing, raw feature extraction, production memory
qualification and final refits remain unfinished. The array-only backend
materializes all training windows and does not open empirical forecasting.
See the [recipe and execution limits](experiment_log.md#2026-10-07-0305-pacific--development_specification-temporal-baseline-implementation-prepared).

**Historical October 7, 02:49 Pacific — five boosted checks selected:**
the new five-test module and changed ASTRA validation are selected for the
optional bounded child of the next eligible EP21 exact-remainder
`work_0007` / index **6**. No job is yet submitted for this selection.
That selection has since expanded to the ten EP19 checks above within the
same bounds. The existing 1-CPU/7-GiB parent is charged once to EP21; no separate EP19
allocation or empirical input is added. See the
[selected limits and output paths](experiment_log.md#2026-10-07-0249-pacific--past_only_controls-boosted-checks-selected-for-a-shared-remainder-allocation).

**Historical October 7, 02:41 Pacific — boosted non-neural comparator implemented:**
the [new backend](code/boosted_hazard_backend.py) implements the existing
18-head, 200-tree/depth-2/3/4 recipes over prepared context features, preserving
censored intervals and participant weighting. Its **five focused checks remain
unrun**; module availability is confirmed, with no install or empirical fit.
The [implementation record](experiment_log.md#2026-10-07-0241-pacific--development_specification-boosted-baseline-implementation-prepared)
states the selected tree initialization/settings and unresolved-head rule.
At that entry, raw feature extraction, the temporal baseline, final refits and source
timing/calibration remain unfinished; this does not open forecasting access.

**October 7, 01:22 Pacific — all 155 development headers extracted:**
the acquisition report completed in EP21 allocation **46877919** for all
**15 development participants**, with exit zero and **0.306229 s** extraction
time. It records **144 headers at 250 Hz / 69-Hz high cutoff** and **11 at
1,000 Hz / 280-Hz high cutoff**, with BrainVision Recorder **1.23.0003**,
actiCHamp Base Unit **5002**, software filters disabled, DC low cutoff and
notch off. Recorded resolution/unit tokens and auxiliary routing are now
available across the development cohort. Reference configuration, physical
acceleration calibration, filter delay and the full onset bound are not
verified by this extraction; forecasting readiness remains false. The parent
is charged once to EP21. See the [observed result and limitations](experiment_log.md#2026-10-07-0122-pacific--source_views_and_timing-all-development-acquisition-headers-extracted)
and [metadata report](runs/source_views_and_timing/job_46877919_acquisition_metadata/acquisition_metadata.json).

**Historical October 6, 23:49 Pacific — development-header extraction selected:**
the existing 155-header acquisition-metadata reader is selected for a separate
bounded child in the next eligible EP21 fresh refill, assignment index **10**
(`work_0011`), after the EP21 loss correction. It will reuse the four passing
checks and unchanged ASTRA validation. Limits are one CPU, 1 GiB virtual
memory, 60 seconds internally and 70 seconds to TERM plus five seconds to
KILL for the entire child. This selection changes no current job and requests
no separate allocation; the parent remains charged once to EP21. Extraction
is metadata qualification, with no forecasting result or timing/calibration
pass implied. The then-unrun status is superseded by the completion above.
See the [selected scope, inputs and output](experiment_log.md#2026-10-06-2349-pacific--source_views_and_timing-development-acquisition-extraction-selected-for-ep21-refill).

**October 6, 22:59 Pacific — acquisition-reader checks passed:**
all **four** prepared checks passed in **0.017 s** inside EP21 job
**46864089**; EP19 ASTRA schema and semantic validation also passed.
At that time the **155-header extraction was unrun**; these fixture results do not
establish cohort-wide acquisition settings, timing, calibration or forecasting
readiness. At that update the coordinator reported the parent active, with
its whole cost charged once to EP21 and no final charge then recorded. See the
[execution record](experiment_log.md#2026-10-06-2259-pacific--source_views_and_timing-acquisition-reader-checks-pass-in-ep21-job-46864089)
and [check receipts](runs/component_checks_job_46864089/status.txt).

**Historical October 6, 21:34 Pacific — acquisition metadata extraction prepared:**
two existing development headers expose previously unreported filter settings,
including 69-Hz and 280-Hz high cutoffs at their respective sampling rates.
A [focused reader](code/inspect_development_acquisition.py) is ready to extract
these settings from all 155 authorized development headers. Static review
passed; extraction, four checks and changed ASTRA validation await a suitable
substantive allocation. Calibration and timing qualification remain open. See
the [execution scope and evidence](experiment_log.md#2026-10-06-2134-pacific--source_views_and_timing-acquisition-metadata-extraction-prepared).
The checks and ASTRA validation have since passed as recorded above; actual
development-header extraction has since completed as recorded at the top.

[Public online resources, checked October 6](../DATASETS.md#public-online-resources--checked-october-6-2026): the official event generator, condition-code/event-time definitions and EEGLAB importer, with the upstream clock/calibration gap retained.

October 6 reconciliation: [DATASETS readiness](../DATASETS.md#current-readiness)
and GOAL now include the completed twelve-person S1 panel, fixed model/fold
definitions and passing backends. They distinguish these preparations from
the unresolved origin/calibration and empirical forecasting work; see the
[record](experiment_log.md).

October 6, 15:49 Pacific: the [published event-generator review](source_timing_handoff.md#october-6-1549-pacific-published-event-generator-and-remaining-exporter-gap)
documents direct `StartTime` copying, a **2-ms** first-sample convention and
force-sign/sensor identities. It does not authenticate a **2-second** clock
correction or physical-unit conversion. The upstream exporter/trim binding
and complete onset uncertainty remain unresolved; no empirical scope changes.
See the [record](experiment_log.md#2026-10-06-1549-pacific--public-event-generator-findings-added-to-the-source-handoff).

October 6, 11:52 Pacific: all **16 classical/non-neural backend checks passed**
in 0.871 s inside approved EP21 allocation 46801996; EP19 ASTRA validation
also passed. These source-free code checks leave empirical access, timing,
calibration and scientific thresholds unchanged. See the [execution record](experiment_log.md#2026-10-06-1152-pacific--prepared-backend-checks-pass-inside-ep21-job-46801996).
Earlier unexecuted-check descriptions below are historical.

October 6, 11:15 Pacific: the [official importer and origin review](source_timing_handoff.md#october-6-1115-pacific-official-importer-and-series-origin-construction)
finds no downstream two-second correction. The article documents trimming at
series boundaries, making the missing raw-to-HS/WS start-field assignment the
specific remaining origin question. This distinguishes a trimming-origin
hypothesis from padding without choosing either. No signal was reread or
shifted, and units and the full onset uncertainty remain unresolved. See the
[execution log](experiment_log.md).

October 6, 10:57 Pacific (PDT): the [timing, units and decision handoff](source_timing_handoff.md#october-6-timing-units-and-decision-review--recommendations-unadopted)
now distinguishes the observed two-second P1 discrepancy from an authenticated
clock correction and the required full onset bound. It records all nineteen
unset decision fields, prioritizes the minimum meaningful primary EEG gain
for scientific judgment, and preserves all twelve participants' S1 support
dispositions. A bounded development clock test is recommended only after an
explicit candidate origin binding. No correction, units amendment, threshold
or empirical job is adopted; see the [log](experiment_log.md).

October 6, 08:19 UTC: the [source-specific logistic-hazard model](code/non_neural_forecast_backend.py)
now connects prepared non-neural feature vectors to frozen 19-category baseline
probabilities for the existing EEG models. Independent static review passed,
including a correction that keeps each WAY model bound to its fitted
participant after reset. Four new [interface checks](tests/test_non_neural_forecast_backend.py)
remain unexecuted; they share the existing proposed 120-second EP19 prelude
with the twelve classical-backend checks. No job or empirical forecast ran.
Raw sensor-feature extraction, source admission, final refits and the other
two baseline families remain unfinished. See the
[implementation record](experiment_log.md#2026-10-06-0819-utc--logistic-baseline-model-interface-implemented-unexecuted).

October 6, 07:37 UTC: [two classical reference backends and a logistic-hazard
fitter](code/classical_forecast_backend.py) are implemented under the existing
recipes. Twelve [focused checks](tests/test_classical_forecast_backend.py) are
prepared but **not yet run**; the updated ASTRA program is also unvalidated.
There is no active EP19 compute or trained empirical forecast. Source timing,
calibration and cohort support remain unresolved; the
[current log](experiment_log.md#2026-10-06-0737-utc--classical-reference-backends-implemented-execution-pending)
separates the remaining model and source integration work.

October 6 prospective preparation: [finite model recipes and grouped folds](component_details.md#5-finite-model-recipes-and-grouped-development-folds)
are specified in the [registry](model_recipe_registry.json) and a source-free
enumeration/model interface. All five reference families, three non-neural
families, mandatory controls and fixed source roles are retained. Independent
static review and all3 [enumeration/fold checks](runs/model_recipe_checks001/tests.txt)
passed; the tests took0.003s and official
[ASTRA schema/semantic validation](astra/v0.0.14/validation_20261006_46721202.json)
exited0 in46721202. The parent completed0:0; its373CPU-seconds are charged
once to EP06, with no additional EP19 charge or active compute. This is implementation preparation, not a trained
model, onset/timing qualification or forecast. See the [current log](experiment_log.md).

Latest result (October 5, **20:21 UTC**): the original **12-person S1 support
panel is complete**. The [P3–P12 report](runs/source_views_and_timing/20261005-remaining-S1-46660232/remaining_s1_support.json)
returned all ten files in 119.174 seconds in step `46660232.6`, preserving the
[P1](runs/source_views_and_timing/20261003-peripheral-pilot02/peripheral_pilot_report.json)
and [P2](runs/source_views_and_timing/20261005-P2-support-46658777/p2_support_trace.json)
results without rereads. All remaining-file stages returned, primary sensor
values were finite, and bindings/handlers were restored.

Under the unchanged conditional support rule, **seven participants have
sufficient support; P1, P3, P6, P9 and P12 remain unresolved**. Their stationary
counts are respectively 13/34, 8/34, 16/34, 11/34 and 10/34; P3 also has only
three fitting-parity windows. Every participant remains in the
[full count table and execution record](experiment_log.md). No favorable subset
is selected, and no thresholds, onset/risk labels or EEG features/scores were
produced. These counts do not establish physiological rest, a clinical finding
or forecasting performance.

The selected support panel is finished. P2–P12 cue polarity, completed return
and total onset timing remain unresolved; P1's file-specific duration evidence
does not resolve them. No further pilot or forecasting is selected. The shared
parent's 0.9644444444 core-hours are accounted once under EP04.

The preserved October 3 eleven-file panel timed out as step `46430982.17` and
saved no report. Its underlying stalled operation and completed reads remain
unknown. The new one-file trace does not rerun that panel or imply a data/system
fault. The subsequent schema-only ASTRA repair passed at 22:20 UTC in `46433944`;
that result is not a scientific source-pass result.

Preserved earlier result (October 3, 21:45 UTC): the
[conditional peripheral pilot02](runs/source_views_and_timing/20261003-peripheral-pilot02/peripheral_pilot_report.json)
completed in1.22seconds after seven tests passed. Of34 eligible candidate
rest windows,13 passed the unchanged stationarity rule (38.24%, below50%).
Noise thresholds and onset/risk labels remain unresolved; no EEG feature or
score was computed. Its original failure of the support fraction is retained
in the completed fixed panel above, without threshold relaxation or a
dataset-wide failure claim.

Previous result (October3,21:35UTC): the
[fitting-only cue comparison](runs/source_views_and_timing/20261003-cue-polarity01/cue_polarity_report.json)
completed after three reader tests passed. All34 raw high intervals match the
official illuminated durations, supporting higher=illuminated for HS_P1_S1.
Absolute event comparisons share an unexplained2second origin discrepancy;
no timestamp correction is applied. Return completion and total onset timing
remain unresolved. The completed [conditional peripheral pilot](code/pilot_way_peripheral_onset.py)
used fixed0.5 raw cue decoding with every prior noise/onset setting unchanged
and read no WS timestamps.

Previous result: the
[LED-only diagnostic](runs/source_views_and_timing/20261003-led-diagnostic01/led_diagnostic.json)
completed in0.388seconds:119,398 of119,496 samples are exactly0 or1, and all
three retrospective thresholds give34 up/down crossings. Intermediate values
and small overshoots explain the failure of an exact-two-level assumption;
their acquisition mechanism and numerical on/off polarity remain unresolved.
The [official LED metadata comparison](code/check_way_cue_polarity.py) and
[scalar-field reader](code/read_way_cue_metadata.py) are retained for the
completed check above; windowed signal matrices were not numerically decoded.

Previous result (October3,21:22UTC): the
[single-file peripheral onset pilot](runs/source_views_and_timing/20261003-peripheral-pilot01/peripheral_pilot_report.json)
completed in 0.676 seconds after six component tests passed. Its LED signal
has 95 finite levels, so the selected exact two-level cue decoder returned
unresolved before threshold fitting or onset/risk-label construction. All
eight peripheral paths were finite. Numeric cue polarity, return-phase
semantics and the total onset timing bound remain unqualified; no EEG model,
feedback or held-out score was computed. See the [execution log](experiment_log.md).

The [diagnostic code](code/inspect_way_led_levels.py) and its completed result
are preserved; no production cue decoder has been selected.

Current preparation (October3,2026): EP19 has no EEG forecast or held-out
result. Existing archives remain read-only. A
[central-directory inventory](runs/source_views_and_timing/job_46425841_inventory/source_inventory.json)
classified1265 members. [Development header inspection](runs/source_views_and_timing/job_46425841_headers/development_headers.json)
verified all29 scientific channels across155 development runs. Eight selected
censoring/calibration and past-only input checks
[passed](runs/implementation_qualification/job_46425841_attempt002/qualification.json)
after an environment-only SciPy repair; their ASTRA validation passed.

The [exact 549-member development extraction](runs/source_views_and_timing/job_46425841_development_view/development_view.json)
completed in job `46425841`: 9.04 GB of continuous/raw development views,
with WAY fitting and feedback roles separated and no held-out or derived files
extracted. [One WAY fitting file](runs/source_views_and_timing/job_46425841_way_structure/way_development_structure.json)
has now been structurally inspected: all 29 required EEG channels are present,
with 500 Hz EEG and 4,000 Hz EMG. Nested development arrays were deserialized
without emitting signal values. Final ASTRA schema and semantic validation
[passed](astra/v0.0.14/validation_20261003_development_view.json).
See the [experiment log](experiment_log.md) and [source handoff](source_timing_handoff.md). Onset,
clock/filter timing, production models, source-role isolation for final
evaluation and scientific margins remain unqualified.

The files here have three purposes:

- [Current conceptual figure, v2](ep19_question_imagegen-v2.png) shows the
  past-only forecast, sensor/context comparison and recent-by-older-slow
  factorial with the older complement retained.
  [Exact image-generation prompts](ep19_question_imagegen-v2-prompt.md) include
  the inspected timing/table correction. Earlier `ep19_question.svg` and
  `ep19_question_imagegen.png` are historical illustrations, not current
  design figures; no trace is an observed result.
- [paper_plan.md](paper_plan.md) turns the question into a four-figure paper
  story with explicit next steps for each result.
- The remaining short files record current scientific readiness. They are not
  evidence that EEG forecasts movement.

Future results should keep the primary 300–600 ms EEG increment separate from
the secondary model-order comparison and the auxiliary new-model search. Large
temporary products belong in episode-specific scratch.
