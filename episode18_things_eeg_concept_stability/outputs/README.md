# EP18 outputs

**October 17, 01:20 Pacific — current registered concept comparison closes underidentified.**
The complete 30-person G/T/R control report has simultaneous one-sided 95%
lower bounds −0.0000605262, −0.000811683 and −0.000148457, below the frozen
0.002, 0.001 and 0 thresholds. This fails a necessary condition for the
biological comparison. The existing policy explicitly maps
`no_positive_reference_gain` to `underidentified`, and `underidentified` to
`closed_no_candidate`; its qualification and positive-control requirements
remain in force. See [the policy](../SEARCH_POLICY.yaml) and the controls
section of [GOAL.md](../GOAL.md).
The [three-line scientific stop](figures/repeated_image_control001/scientific_stop_3lines.md),
[existing G/T/R figure](figures/repeated_image_control001/repeated_image_G_T_R.png)
and [complete control report](runs/repeated_image_signal_recovery/report001/repeated_image_control.complete.json)
provide the empirical evidence and participant-level uncertainty.
This supplies the focus goal's scientific-reason branch for the current frozen
procedure. It is not a positive concept finding, an absence/equivalence result,
or a proof that every analysis of these data must fail. The repeated-image
diagnostics, including the selector sensitivity below, do not change the
original controls. Source inspection established no actionable reader,
time-grid, identity-design, whitening or scoring defect.
**No further execution is required for this registered route.** Concept/false
scores and the 16-person audit remain closed. A different primary target,
preprocessing or readout would require a separately specified scientific
amendment, preserving the exposure history; no such amendment is adopted here.
Earlier next-action statements below are historical.

**October 17, 01:14 Pacific — frozen selector sensitivity complete; improvement is not established.**
All 30 original people are retained: 16 changed directions were refitted at
their prospectively frozen penalties, and 44 original directions were reused.
Alternative R is −0.000442049, with simultaneous descriptive 95% interval
[−0.00122121, 0.000337115]. Its paired change from original R is −0.000396894
[−0.00114787, 0.000354078]. All three intervals in this report include zero.
See the [three lines](runs/repeated_selector_refit/report001/3lines.md),
[complete numeric report](runs/repeated_selector_refit/report001/repeated_selector_refit.complete.json),
and [participant figure](figures/repeated_selector_refit001/repeated_selector_refit.png)
([PDF](figures/repeated_selector_refit001/repeated_selector_refit.pdf)).
The single three-endpoint Bonferroni family describes this exploratory comparison;
it does not correct the preceding adaptive diagnostic history. The original
mean-of-fold-ratios selector was not a coding defect. Pooling the inner errors
has not established better prediction or positive repeated-image recovery here;
the interval does not prove equivalence, harm, or signal absence.
The unchanged nuisance losses and training whitening reproduced before each
alternative fit. Step47586225.3 completed successfully, and the coordinator
inspected the figure. Original R/G/T, their failed controls and all protected
evaluation boundaries remain unchanged.
No further selector search is activated. The current-route scientific stop is
recorded above; these diagnostics cannot authorize concept/false scores or the
16-person audit. Earlier snapshots below are historical.

**October 17, 00:58 Pacific — complete fixed lag curves do not establish localized recovery.**
All 30 people contribute to every endpoint. None of the 40 post-event,
40 pre-event or 40 paired post-minus-pre intervals excludes zero under the
single simultaneous two-sided 95% Bonferroni participant-t family; no bin has
jointly positive post and paired lower bounds. Temporal dilution remains
unresolved, not excluded; this is not evidence of absent EEG information.
See the [three lines](runs/repeated_signal_lags_job_47586225_step_1/report/summary.md),
[complete numeric report](runs/repeated_signal_lags_job_47586225_step_1/report/repeated_signal_lags.complete.json),
and [all fixed curves](figures/repeated_signal_lags_job_47586225_step_1/repeated_signal_lags.png)
([PDF](figures/repeated_signal_lags_job_47586225_step_1/repeated_signal_lags.pdf)).
Step47586225.1 completed in 4.93 s with 116,960 KiB peak RSS and exit0; the
coordinator inspected the figure. Only already released numeric records were
read. Acquisition lags retain the causal filter's 64-ms delay; these are not
corrected neural latencies. No peak, window, channel or person was selected.
Original R/G/T and their failed controls remain unchanged. The subsequent
frozen selector sensitivity is complete above; this dated lag snapshot is
historical. Changing the primary preprocessing or estimand requires a scientist decision.

**October 17, 00:53 Pacific — all 30 repeated-image metadata checks pass.**
In step47586225.0, every person's source `teststimnumber` identifies the same
image throughout all 12 recorded E3 sequences; every sequence contains indices
0–199 exactly once. This weakens a within-person source-label permutation as
an explanation for failed recovery. It does not verify absolute display timing,
establish image-specific EEG recovery, or assert identical index numbering
between people. The checker read no EEG amplitudes and changed no primary score.
See the [compact summary](runs/repeated_identity_metadata/job_47586225_step_0/diagnosis/summary.md)
and [complete metadata report](runs/repeated_identity_metadata/job_47586225_step_0/diagnosis/repeated_identity_metadata.complete.json).
All three focused tests passed. The check took 95.04 s internally, 1m37.16s
process walltime and 112,660 KiB peak RSS, inside the existing task allocation.
Next: review the remaining signal-recovery explanation before selecting another
EEG diagnostic; no new launch or preprocessing change is selected here.

**October 17, 00:41 Pacific — all 30 fixed detrending diagnostics complete in job48223571.**
Removing degree-0–3 sequence components did not clearly improve post-event
alignment: the paired raw-to-adjusted change is −0.0122613, with descriptive
95% participant interval [−0.0392444, 0.0147217]. Adjusted post-event
matched-minus-cyclic37 cosine is 0.00454881 [−0.0130629, 0.0221605].
The pre-event contrast is 0.0229474 [0.00685908, 0.0390358], while its matched
cosine alone is 0.00355618 [−0.00482643, 0.0119388]; the negative shifted
similarity contributes to that contrast. Post-minus-pre alignment is
−0.0183986 [−0.0438290, 0.00703176]. All intervals are descriptive, with 30 people.
See the [three lines](runs/repeated_signal_detrend/report001/summary.md),
[complete numeric report](runs/repeated_signal_detrend/report001/repeated_signal_detrend.complete.json),
and [four-panel diagnostic figure](figures/repeated_signal_detrend001/repeated_signal_detrend.png)
([PDF](figures/repeated_signal_detrend001/repeated_signal_detrend.pdf)).
Eight focused tests and the preceding ASTRA validation passed; the coordinator
inspected the figure. The job completed in 2m33s with exit0:0; driver work took
2m7.25s, with peak RSS226,776KiB.
Whole-sequence projection is not out-of-sample prediction. Original R and its
failed controls remain unchanged; no mechanism or concept-absence claim follows.
**October 17 source interpretation — the fixed shift is not an average mismatch.**
In [the template source](code/repeated_signal_structure.py), each sequence is
centered across all 200 images at every lag and sensor before averaging the six
sequences per half. If `c_k` is the cosine after cyclic image shift `k`, all shifts
have the same denominator and `sum(c_k, k=0..199) = 0`, up to roundoff. Thus the
mean of all 199 mismatches is `-c_0/199`, and matched-minus-mean-mismatch is exactly
`(200/199)*c_0`. This identity holds separately for both windows and each person.
Actual matched cosine remains near zero: adjusted post-event 0.000219077
[−0.0146914, 0.0151295], pre-event 0.00355618 [−0.00482643, 0.0119388]. The positive
pre-event matched-minus-shift37 contrast can therefore reflect that particular
negative comparator; it does not establish pre-event identity recovery. The
algebra does not identify why shift37 is negative or establish signal absence.
**Next: preserve this comparator caveat before selecting any pipeline change.**
No new analysis launch is selected. Concept/false scores and all 16 audit people
remain closed under same-account procedural separation. Earlier snapshots below are historical.

**October 17, 00:34 Pacific — all 30 failure diagnostics complete in 48220727.**
Degree-0–3 temporal components account for 81.7155% of nuisance residual energy
(descriptive 95% participant interval 78.3012–85.1297%). Matched-minus-cyclic37
odd/even template cosine is 0.0168101 [−0.00538739, 0.0390076], which does not
clearly establish image-aligned recovery. Polynomial and complementary gain
contributions are +0.000000412790 and −0.0000455676; they reconstruct unchanged
R = −0.0000451548. Slow-error concentration alone does not explain the failed gain.
See the [three lines](runs/repeated_signal_diagnostic/report001/summary.md),
[complete numeric report](runs/repeated_signal_diagnostic/report001/repeated_signal_diagnostic.complete.json),
and [diagnostic figure](figures/repeated_signal_diagnostic001/repeated_signal_diagnostic.png)
([PDF](figures/repeated_signal_diagnostic001/repeated_signal_diagnostic.pdf)).
All 11 focused checks and ASTRA validation passed; the coordinator inspected the
figure and confirmed the earlier R-axis label repair. Driver time was 9m6.7s.
These exploratory descriptions neither rescue the control nor establish signal absence.

**Earlier selected follow-up — fixed detrending/pre-event comparison, now complete above.**
For the same 30 people, 12 sequences and 200 identities, remove only the fixed
degree-0–3 sequence component for diagnostic templates, then compare the complete
0–800 ms and −800–0 ms windows using the same fixed identity shift. Report paired
raw-to-adjusted and post-minus-pre changes with descriptive participant intervals.
There is no cutoff, degree, channel, peak or feature search. The continuous
primary R/G/T, candidate selection, thresholds and data roles stay unchanged;
concept/false scores and all 16 audit participants remain closed. The selected
allocation is 1 CPU, 7 GiB and one hour; see the [prospective record](experiment_log.md).
Earlier job snapshots below are historical.

**October 17, 00:12 Pacific — all 30 repeated-image controls complete; positive control fails.**
The prescribed odd/even directions are complete for every original development
participant. Repeated-image gain is R = −0.0000451547619279, with descriptive
95% participant interval [−0.000139713671589, 0.0000494041477333]. Its simultaneous
one-sided lower bound is −0.000148456513660, below the required zero; the
leave-one-person requirement also fails. G/T estimates and thresholds are unchanged.
This is a failed calibration control, not evidence that EEG lacks image or concept
information. It does not rescue the main reference or complete the focus goal.
See the [three lines](runs/repeated_image_signal_recovery/report001/summary.md),
[complete numeric report](runs/repeated_image_signal_recovery/report001/repeated_image_control.complete.json)
and [G/T/R figure](figures/repeated_image_control001/repeated_image_G_T_R.png)
([PDF](figures/repeated_image_control001/repeated_image_G_T_R.pdf)).
The R-axis labels were repaired and visually checked in 48220727; ASTRA validation passed there.
The fixed detrending comparison is complete above; temporal-order/fixed-shift
alternatives remain unresolved. Concept/false scores and all 16 audit participants remain closed.
Jobs 48217491 and 48217947 are complete; earlier running entries below are historical.

**October 16, 23:51 Pacific — first repeated-image participant complete; full cohort running.**
Job48217491 passed all 16 focused tests and completed both directions for sub-07.
Authentic metadata confirms 12 E3 sequences × 200 identities; the first complete
fit took 133.9 seconds of driver work. Cohort job48217947 is running on
sh03-06n12 with 4 CPUs, 8 GiB and a 45-minute limit, using four one-thread
workers for the remaining 29 people and preserving the completed first fit.
Only after all 30 complete will the same job report participant uncertainty,
one figure and three lines. No cohort result is available yet; this control
cannot rescue the main-reference margins or open concept/false/audit comparisons.
See the [measured cost, submission and next action](experiment_log.md).

**October 16, 23:43 Pacific — prescribed real-data repeated-image control submitted.**
Job48217491 (1 CPU, 7 GiB, two hours) runs the focused reader/model/report checks,
verifies repeated-session metadata, then fits the first complete development
participant. The control retains all 200 identities and 12 repetitions, with
chronological odd/even sequence evaluation and training-only tuning/whitening.
All 30 original people are required before participant inference, one figure and
three lines. Positive repeated prediction would weaken inability to recover
reproducible image-specific EEG; it cannot rescue the main reference or establish
conceptual coding. See the [prospective interpretation and execution](experiment_log.md).

**October 16 interpretation of the October 14 result — complete 30-person real-EEG calibration.**
Image prediction gain is G = −0.000007574 (descriptive 95% participant interval
−0.00005604 to 0.00004090); target/response nuisance gain is T = −0.0004748
(−0.0007832 to −0.0001664). The fixed image reference shows no clear positive
incremental prediction, and the added task nuisance worsens prediction here.
The registered no-positive-reference rule leaves this calibration underidentified;
it does not establish absence of conceptual information in EEG.
All 30 people and 180 outer folds are complete; each person's six folds were
averaged before participant-level intervals. Separation was procedural within one account.
The completed calibration is available as [three lines](runs/baseline_models/core_direct_checkpoint_batch_report001/summary.md),
[numeric results](runs/baseline_models/core_direct_checkpoint_batch_report001/core_reference.complete.json),
and [one figure](figures/core_direct_checkpoint_batch001/core_reference_G_T.png) ([PDF](figures/core_direct_checkpoint_batch001/core_reference_G_T.pdf)).
This delivered the earlier estimate/figure milestone. Under the October 16 goal
amendment, calibration alone does not count as an explanation-separating result.
The broader concept question and matched-control comparison remain unresolved;
concept/false scores and the 16-person audit remain closed. Earlier entries below are historical.

**October 13, 23:20 Pacific — 28/30 core fits complete; four remaining folds and reporting queued.**
All original writers are terminal. Job 47743173 reuses their saved fits,
continues only sub-27/sub-31, and skips completed sub-47. Its fixed reporter releases G/T with
participant-level intervals, one figure and three summary lines only when all
30 original people have complete records. The matched-control search remains
unresolved and audit remains closed. See the [selection and release rule](experiment_log.md).

**October 13, 22:16 Pacific — 24/30 core fits complete; matched-control search exhausted.**
The remaining six core fits are running. The complete 30-person G/T estimate,
participant-level intervals and figure remain the next result; audit stays closed.
Control-search job 47692682 exhausted its selected 6.4 million proposals with no
accepted member. This does not establish that matching is impossible and does
not block the core estimate. See the [evidence and next action](experiment_log.md).

**October13,19:30Pacific — all30 core fits remain covered within the site CPU cap.**
Array47720344 replaces only the last two indices with8CPU workers sharing the
former single worker's16CPU allocation. Original indices0–26 and first-person
job47716305 continue. All models/folds and completed checkpoints remain; reporting
requires the complete30-person cohort. See the [log](experiment_log.md).

**October13,19:05Pacific — all30 development participants have submitted core fits.**
First-person job47716305 passed numerical checks and one complete paired grid;
array47717611 covers the other29, with all29 eligible to run as resources allow. Every person
keeps all30 inner validations and6 outer folds. No cohort estimate is published yet.
See the [measurement, checks and submissions](experiment_log.md).

**October13,18:42Pacific — unchanged core EEG calibration submitted as47716305.**
The16CPU/32GiB/4h job checks the joint direct solver, compares saved original
solutions and fits the complete first participant. All30 people remain required
before reporting cohort estimates. [Execution record](experiment_log.md).

**October 13 Pacific — unchanged real-data core continuation selected.**
Standing scientist authority reopens the complete 30-person image/task
calibration after its engineering time stop. A separate direct-ridge runner
preserves all models, folds, penalties and data roles; the selected first
sub-07 job is 16 CPUs/32 GiB/four hours. Submission and compute checks are
pending. The control search remains independent; audit is closed.
See the [current record](experiment_log.md), [runner](code/production_core_direct.py)
and [finite job](code/run_core_direct001.sbatch). Earlier clock-based closures
below are historical. No empirical result is claimed until complete-cohort inference.

**October 13, 14:49 Pacific — continuation 47692682 is running.**
Slurm confirms 13 CPUs/96 GiB on sh03-06n07, started at 14:47:30 Pacific.
The adapter accepted the separate authority and original configuration; it
uses the saved checkpoint and a cooperative stop at 23:47:30 Pacific tonight.
No completed control bank or EEG result is claimed. See the [execution record](experiment_log.md)
and [current segment](runs/false_partition_preparation/bounded_continuation_operations/segment_47692682/).

**October 13, 14:46 Pacific — adopted continuation submitted as 47692682.**
One 13-CPU/96-GiB/ten-hour job will resume the preserved sub-07 control search.
The separate search window ends October 14 at 14:43:13 Pacific. No completed
bank or EEG result is inferred from submission. See the [job and accounting](experiment_log.md)
and [adopted plan](proposed_bounded_generator_continuation.md).

**October 13, 14:43 Pacific — one bounded search continuation adopted and selected.**
The scientist approved one 13-CPU/96-GiB/ten-hour continuation from the saved
checkpoint, within a new search-only window ending October 14 at 14:43:13
Pacific. Original search criteria, state and charges are preserved. EEG fitting
and the full study clock remain closed. See the [adopted plan](proposed_bounded_generator_continuation.md),
[authority](config/bounded_generator_continuation_20261013_authority.json) and
[selection record](experiment_log.md). Earlier inactive statuses are historical.

**October 13, 14:40 Pacific — proposed continuation implemented and software-tested, still inactive.**
The adapter and finite wrapper preserve the frozen producer/configuration and
require a separately adopted execution deadline. Three lifecycle checks and
shell syntax passed on compute; no scientific payload or generator was run.
See the [reviewable proposal](proposed_bounded_generator_continuation.md),
[test evidence](runs/software_integration/proposed_continuation001/) and
[execution record](experiment_log.md). The original closeout remains in force.

**October 13, 14:30 Pacific — original execution window ended incomplete.**
The 21:30 UTC deadline has passed. The saved control search and 29/793 core
scalar fits remain incomplete; no final EEG comparison or held-out evaluation
was reached. This is a resource-limited closeout, with checkpoints preserved.
The separate continuation proposal below remains unadopted.
See the [deadline disposition](experiment_log.md).

**October 13 — bounded continuation proposal prepared; no new run selected.**
The [unadopted proposal](proposed_bounded_generator_continuation.md) describes
one ten-hour continuation from the preserved checkpoint. The first member's
remaining search budget would take about 7.5 hours at the observed rate;
completion is not guaranteed. It requires a separate search-only deadline
amendment and does not authorize EEG fitting or extend the full study clock.
See the [preparation record](experiment_log.md).

**October 13,14:02 Pacific — control search stopped at its time limit, checkpoint saved.**
Job47679999 preserved1,164,556 cumulative legal proposals, but no complete
matched-control bank is accepted. The cooperative stop returned its expected75
code and durable checkpoint. Charge12.815833333CPUh once; no automatic retry or
counter reset is selected. The EEG core remains incomplete at29/793 scalarfits;
there is no concept-specific EEG result or audit opening. The original21:30UTC
absolute end remains. See the [terminal evidence and accounting](experiment_log.md).
Earlier running statuses below are historical.

**October 13, 13:04 Pacific — saved-checkpoint control search running.**
The first segment preserved478,972 legal proposals before its time limit;
no complete matched-control group was accepted. Job **47679999**, running on
sh03-06n07 with 13 CPUs and 96 GiB, resumes
that exact state for up to one hour, retaining all search limits and the
original21:30UTC end. See the [checkpoint, accounting and continuation](experiment_log.md).
No cache rebuilding or counter reset is performed. Earlier submitted/running
states below are historical.

**October13,11:18Pacific — first matched-control search submitted as47672861.**
The new13CPU/96GiB/1h segment reuses the complete real-image/context cache;
it does not repeat preparation. The original eight-group menu, matching
requirements and21:30UTC absolute end remain. See the
[selected resources and submission](experiment_log.md) and
[public operations](runs/false_partition_preparation/search_operations_20261013_001/).
The EEG core remains incomplete; this search alone cannot establish a
concept-specific EEG effect or open the held-out audit.

**October13,10:38Pacific — false-group cache complete; core fit stopped at its time limit.**
Job47619998 completed all36 pair caches and6 context folds; false-partition
search has not started. Job47622384 preserved29 of793 scalar fits before
the planned deadline, with no complete parameter grid or validation score.
See the [terminal evidence, once-only accounting and next action](experiment_log.md).
The next steps are the ready search and a measured core-compute decision;
full-model affordability remains unresolved. Earlier running states below
are historical, and these observations establish no scientific result.

**October12,23:48Pacific — resumable real-EEG core cost submitted as47622384.**
The scalar checkpoint and integration checks now pass, including interrupted
recovery and the actual compute node's memory interface. This4CPU/15GiB/2h
attempt retains completed fits across stops and measures the original first
13-by61 parameter grid before both validation sides. See the
[passing continuation check](runs/software_integration/scalar_driver002/checks_stderr.txt),
[public operations](runs/baseline_models/core_scalar_operations_20261013_001/)
and [selection/accounting record](experiment_log.md). The false-group cache
continuation47619998 is independently running. Full-cohort fitting and the
scientific concept comparison remain incomplete; earlier statuses are historical.

**October12,23:13Pacific — saved preparation continuation submitted as47619998.**
Compute-side metadata confirms all36 pair caches and3of6 context folds are
complete. A prepare-only continuation is submitted for the remaining work:
4CPU/28GiB/4h, preserving the cache and original deadline. See the
[selection and next job ID](experiment_log.md). Core fitting still needs
finer checkpointing and cost measurement; full scientific comparison is incomplete.

**October12,23:07Pacific — both cost attempts reached stage limits.**
The core fit completed no whole validation grid; finer checkpointing and cost
measurement are needed before cohort expansion. False-group preparation can
continue saved work after its deliberate3h cutoff; search never started.
See [terminal evidence and accounting](experiment_log.md). Earlier running
statuses below are historical. No scientific result is established.

**October12,20:34Pacific — exact reuse of repeated training fits qualified on fixtures.**
The inactive [paired-inner fitter](code/core_reference_paired_inner.py) preserves
all30 inner and6 outer evaluations while reducing actual fitting calls from438
to228. Both focused methods pass, including exact agreement with the original
full fixture and separation of the two validation sides; independent source
review and changed ASTRA validation also pass. See the
[checks](runs/software_integration/paired_core_reuse001/checks_stderr.txt) and
[execution record](experiment_log.md). Real-data speedup remains unmeasured;
running code is unchanged and the helper does not provide checkpoint resumption.

**October12,20:21Pacific — complete-cohort and statistical components checked.**
Five focused methods passed in3.602seconds on the existing compute allocation;
the changed ASTRA program also validates. The new, inactive components enforce
all30 people/180 folds/1440 false matches and compute the registered
participant-level intervals and classification inequalities. See the
[check evidence](runs/software_integration/postfreeze_components001/checks_stderr.txt)
and [execution log](experiment_log.md). They do not yet provide the full control
pipeline, affordable full-model execution, an empirical result or scientific release.
The real-data reference and false-group preparation jobs continue.

**October12 — full-model capacity bottleneck clarified.**
The exact matching formulas and streamed baseline connection are implemented.
Full FIR factorizations and complete spectral/leverage diagnostics still lack
a demonstrated affordable backend; selected quantiles alone cannot satisfy
the adopted comparisons. The [source assessment and explicit cost assumptions](experiment_log.md)
preserve every scientific requirement. Current real-data core and false-bank
work continues; completing core calibration alone does not complete the full
concept comparison.

**October12,19:48Pacific — remaining29-person execution source prepared, inactive.**
The [fixed28-array plus final-person plan](proposed_core_reference_remaining001.md)
reuses the complete original30-person input binding and one final writer.
Source review is complete; no execution, new reservation or submission is
selected until first-person cost evidence supports resources. The current
reference and false-group jobs remain running with no completed cost result.
See the [preparation and live observations](experiment_log.md).

**October12,19:10Pacific — both reference fitting and control preparation running.**
Core-reference47591674 is fitting the first participant after all30 input
bindings completed. False-partition47584692 now runs onsh03-01n45; all30
participants' inputs and the full six-family image bank passed admission,
and sub07 cache preparation has started. No complete participant fit or
control search result is published yet. See the [current evidence](experiment_log.md).

**October12,19:07Pacific — all30 input bindings complete; first-person fit dispatched.**
The first reference job47584937 stopped before empirical access because a
test expected the wrong exception for a missing30th participant. Production
code correctly refused the incomplete cohort. Only that test expectation
changed; new job **47591674** is running with2CPU/15GiB/2h. Both selected
methods pass in19.083seconds. Initialization reports all30 participants and
five image families bound; the complete sub07 reference cost fit has started.
No completed-person fitting report or predictive result is yet available.
See the [diagnosis, passing evidence and current stage](experiment_log.md).
False-group cost47584692 remains queued; completed image assembly is preserved.

**October12,18:52Pacific — complete six-family image assembly published.**
All22,448 images occur exactly once in every family, as recorded in the
[complete assembly report](runs/feature_preparation/image_features_full_20261012_001/assembled/report.json).
Assembler47584687 completed successfully; the once-only charge is0.024444444CPUh.
The real-EEG core-reference and false-group cost jobs47584937/47584692 now
have satisfied dependencies and await scheduling. See the
[completion and accounting](experiment_log.md#2026-10-12-1852-pacific--feature_preparation--complete-six-family-image-assembly-published).
No EEG prediction result or scientific release is yet established.

**October12,18:42Pacific — complete CLIP replacement published.**
Job **47584681** completed all11,224 images and exited successfully in37m21s;
charge1.245 allocated CPU-hours once. The
[completion record](runs/feature_preparation/image_features_full_20261012_recovery001/recovery_complete.json)
preserves the original shard, model and full image range. See the
[execution log](experiment_log.md#2026-10-12-1842-pacific--feature_preparation--complete-clip-replacement-published).
The eight BLIP tasks were still running at the latest check. Existing
assembly47584687 and subsequent cost jobs47584692/47584937 retain their
dependencies. Full feature assembly and EEG prediction results remain pending.
Earlier dated running-state entries below are historical.

**October12,18:05Pacific — CLIP replacement running; real EEG reference queued.**
Worker47584681 has passed both recovery checks and is encoding the original
complete shard onsh03-06n11. Changed ASTRA validates. The independently
reviewed real-data reference job47584937 is queued after assembly47584687:
2CPU/15GiB/2h, complete30-person input binding, then the full first-person
36-split reference attempt. Its two focused checks are still unrun. See the
[selected numerical recipe, limits and execution evidence](experiment_log.md).
There is no new EEG prediction result or completed image assembly yet.

**October12,18:03Pacific — complete CLIP replacement submitted.**
The original task5 stopped at its two-hour deadline after7,638/11,224
encodings; its partial output remains unused. Worker47584681 repeats the
complete original shard with2CPU/15GiB/4h. New assembler47584687 requires
that worker and all original BLIP tasks; false-cost47584692 follows the
assembler. Their old predecessors were canceled before starting when the
dependency failed. All26 tasks, six families and22,448 images remain required.
See the [diagnosis, resources and submission record](experiment_log.md).
The new chain is pending, with focused checks to run inside the worker.
Earlier dated states below are historical; no EEG prediction result is claimed.

**October12,17:43Pacific — complete-cohort controller checks passed.**
Both focused methods pass in59.575seconds, with all36 split and30-person
requirements checked on existing fixtures. Changed EP18/EP21 ASTRA documents
also validate. [Evidence](runs/software_integration/companion003/checks.txt)
and [log](experiment_log.md) retain the distinction between software readiness
and unresolved full-scale compute/recipe/scientific-release requirements.
No actual baseline search has started.


**October12,17:39Pacific — the three affected integration checks passed.**
The repaired callback and two adapter checks pass in0.573seconds, using88MB
process memory and no empirical inputs. Earlier backend/full-panel passes are
reused. The [evidence](runs/software_integration/companion002/checks.txt) checks
software behavior; full-scale cost and a complete qualified controller remain
unresolved. Real-data preparation continues. See the [execution log](experiment_log.md).


**October12,17:27Pacific — streamed algorithm checks passed; integration remains inactive.**
All three streamed-backend checks and the full-panel callback check passed.
The remaining callback fixture's roundoff failure is repaired, along with a
reviewed candidate-context binding omission. Its recheck and two new adapter
checks remain unrun. The [inactive adapter](code/streamed_tuning_adapter.py)
preserves existing matching/diagnostics, but complete cohort controller wiring
and full-scale throughput remain unresolved. See the [execution and repair record](experiment_log.md).

**October12 — exact streamed capacity source reviewed; five software checks pending.**
The inactive [streamed backend](code/streamed_slot_capacity.py) passed independent
source review. It emits rows from the existing sequence operators and preserves
the full singular spectrum, all13 added-term EDF values and selected full-joint
leverages. The [production callbacks](code/production_continuous_designs.py)
are also prepared, with implementation/self-review only. Three backend and two
callback checks are selected as an independent serial companion in EP21's
first adapted-comparator job47579378, pair004; no check result is recorded
yet. The parent allocation is charged once to EP21. These fixture-only checks
read no empirical EP18 inputs or scores and do not gate the independent EP21
fit. A connected full-trial controller and full-scale throughput remain
unresolved; the new backend has not replaced the active dense tuning path.
See the [selected scope and limits](experiment_log.md#2026-10-12--baseline_models--inactive-streamed-backend-reviewed-five-software-checks-selected).

**October12,16:55Pacific — all30 participants' real EEG preprocessing complete.**
The [production report](runs/eeg_preprocessing/development_20261012_001/operational.json)
confirms all2160 intervals complete,9.56GB durable output,646.08seconds
preparation and0.86GB process peakRSS. Batch exit0 is recorded. Image feature
extraction continues; no encoding-model score or audit opening is claimed.
See the [current execution record](experiment_log.md). Earlier states below
are historical.

**October12,16:41Pacific — real EEG preprocessing running; first participant complete.**
Job47572924 is running onsh02-08n53. Both cache tests and changed ASTRA passed;
sub-07's complete cache preserves all72 sequences and22248 main events, with
586832 full-grid samples and583232 scored samples. See the
[first participant completion](runs/eeg_preprocessing/development_20261012_001/sub-07/complete.json)
and [running-stage record](experiment_log.md#2026-10-12-1640-pacific--development_eeg_preprocessing--allocation-running-and-checks-passed).
The all30-person manifest is not yet complete. Image extraction continues;
the full fitting backend's exact capacity computation remains unresolved.
These are preparation results, with no EEG prediction score or scientific claim.

**October12,16:33Pacific — all30-person EEG preprocessing submitted.**
Job47572924 requests1CPU/7GiB/2h for the fixed preparation of all2160 main
sequences, preserving62 channels and the original common masks. Cache and
wrapper source reviews passed; two cache checks precede actual EEG reads.
See the [selection and submission](experiment_log.md#2026-10-12-1632-pacific--development_eeg_preprocessing--complete-development-eeg-preparation-selected).
This is preprocessing, with no model fit or scientific score. Image extraction
and the dependent false-group cost job47571099 continue separately.

**October12,16:18Pacific — first production false-group cost job submitted.**
Job47571099 requests13CPU/96GiB/4h and depends on the existing complete-image
assembler47551339. It prepares all30 participants' structural inputs, then
builds the full first-participant cache and measures the fixed eight-group-bank
search. It uses one numerical thread; CPU count satisfies the scheduler's
memory requirement. Earlier resource-request issues were corrected before
scientific execution. See the [execution record](experiment_log.md).
All2160 development sequences are bound; image extraction/assembly continues.
No EEG fit or accepted false-group bank exists. Earlier entries are historical.


**October12,16:08Pacific — development geometry complete; image features still running.**
All30 development participants,2160 sequences,667440 main events and the
original62-channel order passed exact metadata binding. The causal filter and
recorded-E3 span are now fixed; three affected reader checks and ASTRA passed.
See the [complete geometry evidence](runs/feature_preparation/development_main_geometry_20261012_001/operational.json)
and [execution log](experiment_log.md). A first complete-participant cost run
for false-group construction is being prepared behind image assembly. No EEG
fit or accepted false-group bank exists. Earlier entries below are historical.

**October12,15:48Pacific — all22448 images delivered;22 feature workers running.**
Builder47551249 completed successfully:702 pixel packs cover every original
image, including the200 repeated images. At the15:48Pacific scheduler check,
22 workers in the original extraction arrays were running; four fast tasks
waited for their concurrency limit and the assembler waited on dependencies.
This completes pixel delivery; the six-family features and final assembly
are still in progress. See the
[completion and accounting](experiment_log.md#2026-10-12-1548-pacific--feature_preparation--all-image-delivery-complete-22-feature-workers-running).

**October12,15:44Pacific — development metadata/FIR inspection complete.**
All30 development recordings have1000Hz timing,62 common recorded channels
and sufficient recorded sequence margins for the proposed scoring/filter
support. The inspection took19.93seconds and121.84MiB peak RSS in EP15
parent47565811; it read headers/markers and EEG file sizes, not EEG amplitudes.
The prospective129-tap causal filter's64ms delay and numerical frequency
response are recorded; the filter is **not adopted**. Actual E3-to-first-image
gaps vary, so the reader's fixed1000-sample assumption needs replacement with
the authentic per-sequence join before EEG preparation. Main-sequence identity
and frozen channel-order checks still belong to that integration. See the
[completion and remaining boundary](experiment_log.md#2026-10-12-1544-pacific--input_role_preparation--development-geometry-and-prospective-fir-inspection-completed).
No EEG comparison or scientific result is available.

**October12,15:29Pacific — false-partition search implementation checks passed.**
Both focused search checks and changed ASTRA passed; earlier reader/design
checks also passed. These are software results, with no accepted empirical
partition bank or EEG score. At15:25Pacific the active image builder had
prepared18080/22448 pixel-pack images (80.5%); the26 feature tasks and assembly
still depend on completed preparation. See the [execution log](experiment_log.md).
EEG preparation and full-scale exact capacity computation remain unfinished.


**October12,14:39Pacific — tuning composition verified; sequence source resolved.**
The continuous EEG preparation bridge passed three checks; nested tuning and
capacity matching passed all four focused checks plus changed ASTRA. These
are software results. The original paper resolves E3 as sequence start; the
new explicit BrainVision reader binds recorded units and development-only paths,
but its focused tests and empirical execution remain pending. Full-image
preparation and its26 dependent feature tasks continue. No EEG comparison or
scientific claim is available. See the [checks, source and disclosure](experiment_log.md#2026-10-12-1439-pacific--baseline_search--tuning-checks-complete-sequence-trigger-source-resolved).

**October 12, 14:13 Pacific — full-image preparation running.**
Builder **47551249** is running on `sh03-06n09`. Configuration admission and
ASTRA validation passed; the operational stream confirms real pixel-pack
preparation has begun. The 26 feature tasks and assembler remain queued on
their successful dependencies. EP17's independent companion exited before
its operational directory was created; this did not stop EP18 preparation.
See the [startup observation](experiment_log.md#2026-10-12-1413-pacific--feature_preparation--builder-running-and-independent-companion-stopped).

**October12,14:07Pacific — full real-image extraction submitted.**
The chain is builder47551249 → fast array47551335 plus BLIP array47551337 →
assembler47551339. It covers **22448 images × six feature families**, with
26 feature tasks and at most22 concurrent workers, bounded to91CPUh total.
The builder was pending at the last observation; dependent tasks wait for
successful prerequisites. EP17's separately scoped32-image pilot shares a
bounded step inside the builder parent. See the
[selection, resource accounting and submission](experiment_log.md#2026-10-12-1405-pacific--feature_preparation--full-real-image-extraction-selected).
The completed small pilot below is the cost evidence; full features, false
partitions and empirical EEG results remain unfinished.

**October 12, 13:49 Pacific — all six real-image pilot families complete.**
The fixed32-image pilot finished successfully in299.14seconds; all six
singleton-repeat checks agreed exactly. The measured encoding costs project
to about37CPU-hours for22448images, before loading/I/O and variability
allowances. See the [completion, costs and next work](experiment_log.md#2026-10-12-1349-pacific--feature_preparation--all-six-real-image-pilot-families-complete).
Full extraction code passed three focused checks and independent source
review, including a duplicate-invocation history-preservation repair. The
attended parent completed and closed at13:55Pacific, using0.252222allocatedCPUh.
Full extraction is not launched. Full image features,
eight false partitions and the complete EEG fitting/evaluation pipeline
remain unfinished; no empirical EEG result is claimed.

**October 12, 13:46 Pacific — real-image cost pilot running; fold bridge tested.**
Parent47547023 is running the fixed32-image/six-family CPU pilot after all
five image-runner checks, four full-roster fold checks and changed ASTRA passed.
A private-notice format mismatch stopped the first attempt before decoding;
its narrow parser repair passed, and the fresh repaired attempt retains the
same allocation/deadline. See the [diagnosis, checks and selected repaired run](experiment_log.md#2026-10-12-1343-pacific--feature_preparation--private-notice-parser-diagnosis-and-bounded-repair-selected).
The [fold bridge](code/continuous_fold_runner.py) composes all six outer and
five inner splits with shared masks and training-only whitening. It still
requires raw preprocessing, the complete tuning/EDF caller and final scoring
convention; it is not an empirical EEG result. Image extraction is real-data
preparation. Full extraction, eight false partitions and EEG comparison remain
unfinished. The procedural path-listing disclosure is retained in the
[implementation entry](experiment_log.md#2026-10-12-1334-pacific--feature_preparation-and-baseline_search--implementation-resumed-under-adopted-procedural-boundary).

**October 12, 13:23 Pacific — procedural separation adopted:** disclosed same-account procedural separation replaces enforced EEG/pixel-worker isolation, so the SRCC draft is not needed and isolation requirements below are superseded; remaining features, partitions and the EEG runner still gate fitting ([decision](experiment_log.md#2026-10-12-1323-pacific--scientist-decision-procedural-same-account-separation-adopted)).

**October 11, 23:44 Pacific — current isolation support handoff prepared:**
the [unsent SRCC draft](experiment_log.md#2026-10-11-2344-pacific--input_role_preparation-current-unsent-isolation-support-handoff)
asks for a supported isolated image/EEG evaluator route using the retained
October 3 namespace failure. Exact runtime/assets are already qualified;
no new fault observation, retry, weaker boundary or support contact occurred.

**October 7, 02:21 Pacific — public-label geometry complete:**
job **46881096** prepared all **1,854 labels**, representing all **27 requested
rank triples / 81 bandwidth mappings** with **six geometries and 18 kernels**.
The runner took **33.25 seconds**, reported **310.72 MiB peak RSS**, and retained
**507,723,286 cached array bytes** within its 600 MB allowance. Its cost is
included once in the EP21 parent. Image/EEG access and empirical fitting remain
unreleased. See the [completion record](experiment_log.md#2026-10-07-0221-pacific--feature_preparation-public-label-geometry-completes)
and [closed report](runs/feature_preparation/label_geometry_46881096/label_geometry_report.json).
The earlier unrun statements below are historical snapshots.

**October 7, 00:47 Pacific — public-label geometry production selected, unrun:**
the coordinator selected the existing PCA/RBF runner for fresh EP21 global
index **12 / work_0013**, as a separate serial EP18 child bounded to **1 CPU /
7 GiB / 1,800 seconds / zero GPUs**. It reuses the 27 passing checks and reads
only public-label features and permitted preparation records. The full
eight-hour parent reservation and actual cost remain charged once to EP21;
EP18 records its child time as included, with a planned **600,000,000-byte**
storage allowance and actual bytes to reconcile. No parent startup or cache
completion is claimed. See the [selected scope and accounting](experiment_log.md#2026-10-07-0047-pacific--feature_preparation-public-label-geometry-selected-for-shared-worker).

**October 6, 22:59 Pacific — 27 component checks passed:**
the training transforms, continuous sequence model, slot-capacity calculation
and label RBF checks all passed inside EP21 job **46864089**, with **0.041 s**
summed test-suite runtime. EP18 ASTRA schema and semantic validation also
passed. At that documentation update the coordinator reported the parent
active; its full cost is charged once to EP21, with no final charge then recorded.
Label-geometry preparation, image/EEG access, empirical fitting and scoring
did not run. See the [component execution record](experiment_log.md#2026-10-06-2259-pacific--baseline_search-component-checks-pass-in-ep21-job-46864089)
and [check receipts](runs/component_checks_job_46864089/status.txt).
The historical pending-check statements below are superseded by this result;
the separate preparation and access dependencies remain open.

**Historical October 6, 21:32 Pacific — nonlinear label component reviewed:**
the [RBF component](code/label_rbf.py) now supplies the registered nonlinear
label geometry. Coordinator and independent static reviews found no blocking
issue; four new checks remain unrun. Identical training-concept inventories
permit reuse of label transforms across folds; the
[finite preparation runner](code/prepare_label_geometry.py) is now ready and
statically reviewed, with execution still pending. No new compute reservation
or EEG access was selected at that time; the October 7 selection above
supersedes this historical allocation status. See the
[review and next execution scope](experiment_log.md#2026-10-06-2132-pacific--baseline_search-nonlinear-label-component-reviewed).

**Historical October 6, 21:21 Pacific — capacity matching implemented, checks pending:**
the [new component](code/slot_capacity.py) measures added-slot flexibility
under the joint fit and lists eligible pairs on the existing penalty grid.
Static review is complete; five checks and changed ASTRA validation remain
unrun. Dense resource qualification, RBF construction and empirical fitting
remain unfinished. See the [implementation record](experiment_log.md#2026-10-06-2121-pacific--baseline_search-exact-capacity-component-prepared).

**Historical October 6, 21:08 Pacific — training transforms reviewed, checks pending:**
the [new component](code/training_transforms.py) implements the registered
training-only feature reductions, fixed sensor projection and explicitly
specified covariance whitening. Static review is complete; eight new checks,
ten affected sequence checks and changed ASTRA validation await the next
suitable substantive allocation. See the [review and execution scope](experiment_log.md#2026-10-06-2108-pacific--baseline_search-training-transforms-reviewed-compute-pending).
No image/EEG reader release or empirical fit has occurred.

**October 6 Pacific — continuous fitting core checks passed:**
the [sequence-local numerical model](code/continuous_sequence_model.py) handles
overlapping events, a direct joint nuisance/feature ridge fit and fitted-event
ablation. All **ten focused checks passed in 0.011 s** in the coordinated
one-CPU / 7-GiB session **46845646** on **sh04-18n32**; changed ASTRA validation
also passed. This supplies no EEG reader release or empirical fit. See the
[execution evidence and remaining pipeline](experiment_log.md#2026-10-06-pacific--baseline_search-sequence-local-fitting-checks-pass).

**October 6 Pacific — all concept-name features complete:** allocation
**46843116** completed the first-64-name pilot and, after explicit coordinator
release, all **1,854** prepared names. The [full label-feature report](runs/feature_preparation/all_name_text_46843116/label_feature_report.json)
records a **1,854 by 512 float32** CLIP matrix in **169.455 s**, peak
**609,144 KiB**, with zero token truncations or provider-name fallbacks.
The [exact runtime and public assets](runs/feature_preparation/exact_cpu_provision_46842156/model_runtime/model_runtime_qualification.json)
had qualified in **46842156**. Enforced image-worker and EEG isolation,
remaining image/caption features and the EEG fitting/evaluation pipeline are
unfinished. The label process read no pixels, private image routing, events
or EEG and emitted no complete feature freeze or access seal. See the
[execution log](experiment_log.md) for attempts and validation results.

The attended allocation is **COMPLETED 0:0** and closed, with **366 allocated
CPU-seconds** charged once. Updated workflow validation passed; its original
dependency failure remains preserved in the log. Together with model
provisioning, these two jobs used 713 CPU-seconds, excluding earlier history.

## Historical execution snapshots

The dated snapshots below preserve what was known at each point; the current
readiness statement above supersedes their pending/unobserved descriptions.

**Historical October 6, 18:30 Pacific — model provisioning complete:** job **46842156**
finished successfully, using **347 allocated CPU-seconds**. All fixed public
model assets passed source-integrity checks and all six selected model routes
loaded offline with the exact stable runtime. No inference, image or EEG access
occurred. See the [completion and remaining boundaries](experiment_log.md#2026-10-06-1830-pacific--feature_preparation-exact-runtime-and-all-public-model-assets-qualified).
The existing first-64-name label-only cost pilot is selected in a separate
attended session; image-worker and EEG audit isolation remain unresolved.
The two changed ASTRA documents await a retry with the diagnosed dependency
path correction, preserving the original missing-YAML failure logs.

**Historical October 6, 18:19 Pacific:** provisioning job **46842156** is submitted with
one CPU / 7 GiB / two hours. It will install the exact stable CPU runtime,
acquire the fixed public model assets and verify/load them offline. Shared
EP09 identity and EP20 circuit checks use this allocation serially. Startup
is pending; no feature inference or EEG access is released. See the
[submission and selected scope](experiment_log.md#2026-10-06-1819-pacific--feature_preparation-provisioning-job-46842156-submitted).

Historical October 6 Pacific: a [single finite provisioning job](code/provision_exact_cpu_runtime_001.sbatch)
is prepared for the exact CPU-wheel runtime and the fixed public model assets:
one CPU, 7 GiB, at most two hours. It preserves the old environment, checks
source integrity before offline model loading, and performs no feature/text
inference or EEG access. The [current execution route](paper_plan.md#october-6-pacific-selected-compute-provisioning-correction)
corrects the impossible interactive-DTN handoff; the fixed public files do not
require account credentials. Runtime success and source delivery remain
unobserved, and both pixel-worker separation and the EEG audit seal remain
unverified. See the [execution log](experiment_log.md) for actual launches.

[Public online resources, checked October 6](../DATASETS.md#public-online-resources--checked-october-6-2026): fixed-release THINGS-EEG documentation, exact PyTorch distributions and the pinned CLIP model configuration.

Historical October 6 reconciliation: [DATASETS readiness](../DATASETS.md#what-must-exist-before-model-comparison)
and GOAL now include the completed event join, 30/16 roles, human-feature and
private-routing metadata, and EP17 overlap. Authenticated assets, the exact
qualified runtime, reader isolation and scientific fitting remain unfinished;
see the [record](experiment_log.md).

Historical October 6 public-source update: the
[official release and asset-metadata note](paper_plan.md#october-6-public-source-update--exact-release-recipes-exist-runtime-untested)
links PyTorch's exact 2.4.1/0.19.1 distribution recipes and the existing pinned
CLIP configuration. Public release availability is confirmed; Sherlock
compatibility, checkpoint transfer and reader isolation remain unverified.
No install or new runtime test occurred; see the [log](experiment_log.md#2026-10-06-pacific--official-runtime-recipes-and-pinned-asset-metadata-recorded).

Historical October 6, 10:57 Pacific (PDT): the [then-unadopted route handoff](paper_plan.md#october-6-execution-route-recommendations--unadopted)
separates completed metadata/exposure work from the remaining authenticated
transfer, exact-runtime and reader-isolation dependencies. It corrects the
unsupported equation of namespace `ENOSPC` with Codex's sandbox limitation;
effective isolation through a suitably controlled runtime remains possible.
No compute-node transfer or alpha-build substitution is selected, and another
episode's runtime acceptance does not amend EP18. Recommendations are
unadopted; see the [log](experiment_log.md).

**Historical October5,15:04UTC:** the exact model-asset map and bounded
[DTN transfer](code/acquire_model_assets.py) are prepared and independently
reviewed. Agent SSH to the DTN was denied before transfer; the user has the
concrete authenticated-session command. Assets remain unacquired by EP18.
Seven project dependencies installed in the selected GroupHome environment,
including the diagnosed missing regex distribution. Actual imports then
showed that the advertised Torch2.4.1/Torchvision0.19.1 modules report
2.4.0a0/0.19.1a0. The module route is stopped pending supported matching
builds or provider clarification; no model or feature inference ran. See the
[evidence and support draft](experiment_log.md#2026-10-05-1504-utc--feature_preparation-regex-repaired-module-runtime-route-stopped).
The [runtime loader](code/qualify_model_runtime.py) remains prepared but
unexecuted. These tasks establish no pixel-worker boundary or EEG audit seal.

## Completed metadata and retained preparation records

This directory contains small, reviewable episode records and planned paper
artifacts. As of 2026-10-04, the
[30/16 participant roles and complete event join are frozen](runs/input_role_preparation/20261003-freeze_metadata01/frozen_inputs_and_roles.json),
using outcome-blind source work covering all 46 participants:
[structural report](runs/input_role_preparation/20261003-metadata01/structural_input_report.json)
and [explicit sample-origin repair](runs/input_role_preparation/20261003-sample_origin01/sample_origin_report.json).
The reports establish 62 common channels; the working join retains 1,133,808 events. Source
samples are one-based; original values remain alongside normalized recording
indices. The [clock diagnosis](runs/input_role_preparation/20261003-clock01/clock_segment_report.json)
locates the large sub-36/sub-38 clock discontinuities at recording-segment and
physical-sequence boundaries; all events remain present. The
[exact 30/16 role solution](runs/input_role_preparation/20261003-role_proposal01/role_proposal.json)
satisfies the three-tier balance objective, verified against an exhaustive
fixture and independently reviewed. Durable disjoint
[development](runs/input_role_preparation/20261003-freeze_metadata01/development_manifest.json)
and [audit](runs/input_role_preparation/20261003-freeze_metadata01/audit_manifest.json)
metadata views retain **739,440** and **394,368** events respectively.
The [source-only EP17/EP18 exposure record](config/ep17_ep18_exposure_record.json)
now records [exact source-inventory overlap](runs/input_role_preparation/20261003-freeze_metadata01/source_overlap.json):
**8,640 main-image identities / 720 concepts** and **100 repeated-image
identities** are shared. All 200 repeated concepts occur in the source
experimental concept inventory. The CNeuroMod CSV's 100 unmapped catch images
remain separate. This historical source inventory is distinct from the now
completed [EP17 qualified-universe comparison](runs/cross_episode_exposure/ep17_handoff001/exposure_report.json):
all **4,320 observed**, **3,840 common** and **3,634 repeat-qualified** EP17
images match EP18 main-image identities, across **720 shared concepts**.
None of those EP17 images matches an EP18 repeated-session image; **200
concepts** overlap that session. The coordinator supplied this immutable
metadata handoff after successful execution on job **46531931** in **4.210 s**.
Both designs and all role assignments remain frozen under the adopted
**correlated, design-exposed** decision. New EEG participants do not provide
independent stimulus-family confirmation; matching identities do not imply
identical pixels, identical features or clean pretraining exposure. See the
[October 4 closeout](experiment_log.md#2026-10-04--cross_episode_exposure-qualified-universe-and-claim-decision).
The required [development runtime separation](../inputs/README.md) remains a
distinct unverified prerequisite for EEG-driven candidate comparisons.
Actual attempts and access limits are in the
[execution log](experiment_log.md#2026-10-03--input_role_preparation-outcome-blind-source-qualification).
The freeze completed on job **46427768** in **49.539 s**; two focused identity
checks and [official ASTRA validation](astra/v0.0.14/validation_20261003_freeze_metadata.json)
passed. EEG/image values remain closed, the **audit seal is false**, and no EEG
result is claimed. Frozen roles and file manifests are not access controls.

The [sentinel-only isolated-runtime check](runs/input_role_preparation/20261003-runtime_sentinel01/runtime_feasibility.json)
failed before child launch with the platform's namespace `ENOSPC` response.
That runtime route is stopped; an existing supported isolated evaluator/runtime
is still needed before EEG-driven comparisons. No configuration changes,
weakened isolation or retry on another node are selected. Metadata and feature
registry preparation can continue independently; exact available property and
category tables and the remaining model-record prerequisites are listed in the
execution log.

The [human-feature crosswalk](runs/feature_preparation/20261003-human_metadata01/human_metadata_report.json)
also completed: all **1,854 concepts**, **12 property/size dimensions** and
**53 broad-category indicators** join without missing values. The 406 concepts
with no membership in those 53 categories remain included; they are not
missing records. Raw values are preserved without imputation, scaling or PCA.
The [complete prospective feature recipe](config/feature_registry.json) now
selects all required image, caption and label families, including the extra
independently trained encoder for the expanded panel. Its source references,
layers, dimensions, transforms and training-only reduction rules are explicit.
These are choices made before EEG access. The [pixel-only adapters](code/feature_extractors.py)
are implemented; [private routing/name preparation](runs/feature_preparation/20261003-mapping01/feature_metadata_report.json)
completed on job **46429138**, retaining **22,448 image identities** and
**1,854 name prompts**, with no provider-name fallbacks. The private source
crosswalk is for the trusted builder, never an image worker. Two focused
low-level formula checks passed in **1.543 s**, and the
[then-current ASTRA validation](astra/v0.0.14/validation_20261003_feature_recipe.json)
passed. Model assets and the exact runtime subsequently qualified, and all
concept-name features completed, as linked above. Effective image-worker
delivery remains unverified; no source-image pixels or EEG have been opened
by EP18, and no complete `frozen_feature_record` is yet emitted. The
[execution log](experiment_log.md#2026-10-03--feature_preparation-prospective-complete-panel)
distinguishes those engineering tasks from the existing access boundary.

- [Current conceptual figure, v2](ep18_conceptual_question-v2.png)
- [Exact image-edit prompt](ep18_conceptual_question-v2-prompt.md)

The current image distinguishes 10 Hz presentation from EEG sampling and
shows the shared image baseline; its caption makes explicit that all three
added-slot comparisons also share the linear-label baseline. Its alternative patterns are hypothetical;
the earlier `ep18_conceptual_question.png` is retained as historical material.

Future scientific outputs must respect the participant roles and access
boundary in [../DATASETS.md](../DATASETS.md).
