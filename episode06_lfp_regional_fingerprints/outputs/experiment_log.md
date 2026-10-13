# Experiment log

## 2026-10-11 01:20 Pacific — residual_fingerprint_development — stopped at thirty valid comparisons; no supported candidate

The four final robustness controls completed at **08:17:39 UTC**, all
available. Their focused omission/source-MAD check passed in0.049seconds.
Both seven-omission series preserve every slot and14 source fits. Original
LMP classification falls to **Chewie→Mihili0.5** when either Mihili February18
or March4 is omitted. Those omissions change that direction's target panel;
they do not diagnose a source-training cause. Five-band prototype accuracy
falls to **0.625/0.5** after the March4 omission. On selected-electrode units,
the five-band prototype scores **0.375/0.333333** with failed raw omissions;
under trial mismatch it scores **0.75/0.833333**, also with failed raw omissions.
Classification survival under mismatch does not establish trial-specific
coupling. All exact reports are linked by the [robustness completion](runs/residual_fingerprint_development/robustness001/scores/complete.json).

An independent reviewer checked the actual completed reports and the stopping
policy: **30 valid comparisons=12 starts+4 successors+14 controls**;
**14/18** post-starting comparisons are controls; the last **ten** valid
comparisons after the minimum produced no qualified improvement. Failed
scientific robustness does not make a complete control numerically invalid.
No unavailable omission or reference was discarded or counted.

Apply the existing stopping rule and close this development search as
**no_supported_candidate_under_design**. Preserve the exploratory M1 LMP
predictivity and its scope, without a neural regional, component-consequence
or absence-of-coupling claim. No candidate promotion, final-analysis freeze,
calibration-value opening or reserved-session evaluation occurred. Remaining
controls and empirical component/support follow-ups remain explicitly
untested. The [scientific closeout](runs/residual_fingerprint_development/development_closeout.md)
collects results, accounting, limitations and source links. This is an EP06
development terminal result, not completion of the overall EP01–EP21 goal.

The LMP [PNG](figures/lmp_controls/ep06_lmp_controls.png) and
[PDF](figures/lmp_controls/ep06_lmp_controls.pdf) were rendered and visually
reviewed. Final changed ASTRA schema and semantics passed. The four new
focused checks passed; unaffected earlier checks were reused.

The attended shell exited normally at **08:19:51 UTC**. Scheduler observation
at **08:20:02 UTC** reports **47354289 COMPLETED, exit0:0**,468 elapsed
seconds,AllocCPUS1,TotalCPU16.305seconds,stepMaxRSS143708K. Charge the entire
attended allocation **468/3600=0.13 allocated CPU-hours** to EP06, including
coordination time; extern471seconds is not another charge. Release the full
2/3CPU-hour reservation and preserve all earlier charges. PTY20032 is closed;
no job or compute session is live and no renewal or supervisor exists.

## 2026-10-11 01:17 Pacific — residual_fingerprint_development — robustness continuation ready

Independent review passed `run_robustness_controls.py` and its one focused
omission/source-MAD test. Root reviewed the matching launcher and all new
source is frozen for execution. Run `robustness_controls.sh` once inside
the existing47354289 compute PTY, preserving its08:32:06 UTC allocation end,
one CPU,7GiB, original search clock and reservation. This stage is capped at
the earliest of ten minutes, allocation end minus45seconds and search end.
The test remains unrun until this coordinated continuation.

The LMP figure already rendered successfully in the same allocation; root
visually inspected the [PNG](figures/lmp_controls/ep06_lmp_controls.png).
The two transfer directions, original/control scores, negative S values,
two-animal grouping and exploratory caveats are readable. The
[PDF](figures/lmp_controls/ep06_lmp_controls.pdf) preserves the same evidence.
No new scientific comparison or outcome read was added by plotting.

## 2026-10-11 01:15 Pacific — residual_fingerprint_development — four robustness controls selected

Adopt the [four fixed robustness controls](paper_plan.md#october-11-lmp-and-five-band-robustness-controls)
before their scores: original LMP classifier omissions, five-band prototype
classifier omissions, and the matched five-band selected-unit and mismatch
controls. This follows the independent result review: test the already
observed leads' robustness before another classifier choice. Four complete
negative controls are valid scientific comparisons; an unavailable omission
or reference cannot be dropped to manufacture the ten-comparison stopping
count. No automatic acceptance or opening follows.

Use the same verified attended allocation47354289 within its original end,
one CPU/7GiB and unchanged resource reservation. Bundle the one focused new
omission/source-MAD check, changed ASTRA validation, four empirical controls
and the LMP result figure; no new scheduler submission. The independent
figure source review passed after clarifying that signed-profile omissions
do not yet establish classifier omission robustness. Root now renders that
completed-evidence figure in the same compute PTY; implementation of the
four-control runner continues in separate source paths.

## 2026-10-11 01:13 Pacific — residual_fingerprint_development — twenty-six comparisons complete; LMP sensitivity localized

Job**47354289** was verified on **sh03-06n11.int**, one task CPU and7168MiB.
The magnitude stage returned0 and completed **08:12:27 UTC**. Three focused
checks passed in0.104seconds and changed ASTRA validated. No new neural fit or
outcome read occurred; all six full conditional references and raw signed
diagnostics are available. See [complete six-comparison results](runs/residual_fingerprint_development/representation001/scores/complete.json)
and [physical LMP comparisons](runs/residual_fingerprint_development/representation001/scores/physical_score_comparisons.json).

| Comparison | Mihili→Chewie BA | Chewie→Mihili BA | Raw sign and all omissions |
| --- | ---: | ---: | --- |
| Full-nine shape plus magnitude | 0.625 | 0.5 | Failed |
| Original CCA singleton LMP | 0.875 | 0.666667 | Passed |
| Selected-unit singleton LMP | 0.5 | 0.5 | Failed |
| Mismatched singleton LMP | 0.5 | 0.5 | Passed |
| PCA-ridge singleton LMP | 0.75 | 0.666667 | Passed |
| Five-band CCA, omit0.5–4Hz | 0.75 | 0.666667 | Passed |

Original M1 LMP recovery exceeds both the fixed within-direction mismatch
and fixed linear elapsed-time prediction on all seven dates. PMd differences
are mixed. This post-observation lead is exploratory, conditional on the
existing fits and two animals; no new inferential test or replication is
claimed. The classifier loses its advantage under mismatch and under the
selected-unit restriction. The latter changes the population target, so it
demonstrates sensitivity, not a causal artifact diagnosis. Singleton raw
cosines measure sign only, not precision or frequency geometry.
In particular, the mismatched scalar raw means reverse together:
M1−PMd is **−0.00119415/−0.000262863**, versus original
**+0.01104576/+0.01079659**. Its cosine+1 therefore does not mean the original
M1-favoring contrast survived. The original LMP conditional upper-tail
fractions are0.125/0.5, and the five-band prototype's are0.125/0.40625;
these are not strong two-direction reference evidence or adaptive-search
adjusted significance.

Both new successors remain below starting_10's equal-direction average;
neither is a qualified improvement or replaces the unaccepted comparator.
The direct PCA-ridge-versus-CCA singleton comparison also lowers Mihili→Chewie
accuracy from0.875 to0.75. Uniformly omitting0.5–4Hz fixes the raw-sign gate
for that declared feature subset without improving overall classification.
No required-control waiver, promotion, final analysis or reserved opening
follows. Calibration values and four reserved dates remain closed.

Ledger: **26 valid comparisons =12 starts+4 successors+10 controls**; ten of
fourteen post-starting comparisons are controls. Six valid comparisons have
elapsed after the minimum with no qualified improvement; patience is **6/10**.
The root-owned attended allocation remains open with its original twenty-minute
end for the next evidence-led comparison and result figure. Keep the existing
2/3CPU-hour reservation; final actual accounting follows shell exit. Source
files are no longer consumed by the completed magnitude stage.

## 2026-10-11 01:11 Pacific — residual_fingerprint_development — magnitude allocation queued

Attended srun returned **job47354289**, queued for the selected1CPU,7GiB,
20minute russpold allocation. Root owns PTY20032. Final independent runner
review also passed. No compute hostname has yet been verified and no test or
scientific command has run. Wait for this allocation; do not submit another.

## 2026-10-11 01:11 Pacific — residual_fingerprint_development — magnitude implementation ready for attended execution

Independent source review passed the new magnitude classifier, three focused
tests and attended launcher; root reviewed the score-only runner against its
actual producers. The runner retains original, mismatch and selected-unit
population designs separately, uses only completed score/state artifacts,
records source scaling and classifier parameters, and adds no neural fit or
outcome reader. All three tests remain unrun before allocation. Existing
unaffected checks are reused. Changed sources and ASTRA are frozen during
execution. Launch once via attended srun in russpold,1CPU,7GiB,20minutes;
resource discovery at08:10:35 UTC confirmed the partition and compatible
hardware. Logs belong to `launch003`; the original reservation and clock
recorded above apply. Root owns submission and subsequent commands.

## 2026-10-11 01:09 Pacific — residual_fingerprint_development — six magnitude and LMP comparisons selected

Adopt the [fixed six-comparison specification](paper_plan.md#october-11-magnitude-and-lmp-comparisons)
under the open development menu before classifier fitting. Reuse completed
physical scores: full-nine CCA magnitude normalization; singleton LMP; its
restricted-unit and trial-mismatch controls; a singleton PCA-ridge mapping
successor; and a uniform five-band feature successor. Source-only robust
magnitude scaling and the explicit singleton amplitude convention preserve
the old shape-only refusal. No raw neural refit or calibration/reserved read
is selected. Twenty comparisons and all consumed resources remain recorded;
there is no qualified improvement or patience reset.

The focused BR scientific-review call at08:08:08 UTC returned external-agent
directive **ext_review_dir_a1728d9b52b7**, containing BRKG review criteria.
BR did not read local artifacts or issue a result verdict. The independent
Codex reviewer applied the relevant correctness, completeness and judgment
criteria to the published plan and sources: PASS / batch specification
COMPLETE / exploratory judgment SOUND / PROCEED. Full scientific validation
is still incomplete. The review corrected the matched normalization anchor
to **starting_11** (full-nine CCA logistic1); starting_07 is mean-ridge and
cannot serve as that matched comparison. It also requires a direct
adaptive02-versus-control07 mapping contrast and explicit singleton sign-only
interpretation. Both are retained without changing promotion requirements.
This advisory workflow is not a canonical BR campaign launch or acceptance.

Three agents split classifier implementation, score-only runner integration
and independent scientific review; root owns records, scheduler and compute.
Prepare one attended **1CPU,7GiB,20minute** allocation after source review,
aggregating the six empirical comparisons, focused changed-code checks and
changed ASTRA validation. Reserve at most **2/3 allocated CPU-hour** to cover
site CPU rounding; charge actual allocation use and release unused reservation
after exit. No job has been submitted by this selection. Preserve the original
search deadline and bound this first stage to ten minutes. No automatic
renewal or duplicate pending job is permitted.

## 2026-10-11 01:02 Pacific — residual_fingerprint_development — twenty comparisons complete; attended allocation closed

Both support controls are available and completed at **07:59:20 UTC**. The
two focused helper checks passed in 0.075 seconds and changed ASTRA validated.
All fourteen unit-restricted and fourteen timing fits/predictions were fixed
before evaluation scoring. Evidence:
[complete support results](runs/residual_fingerprint_development/support001/scores/complete.json),
[unit restriction](runs/residual_fingerprint_development/support001/scores/control04_selected_electrode_units.json),
[linear timing baseline](runs/residual_fingerprint_development/support001/scores/control05_elapsed_time.json).

Restricting spike units to the selected LFP electrodes reduces balanced
accuracy to **0.5 in both directions**, with signed-profile stability still
failed (omitting Chewie October14 gives cosine −0.086655). Retained M1/PMd
unit counts by the fixed seven-date order are **7/15, 4/26, 7/6, 12/19, 9/31,
12/19, 21/27**. Both regions retain fifteen LFP electrodes and rank two.
This changes the target population/axes and does not identify an artifact's
cause or establish a complete bidirectional electrode match.

The timing model uses the original population axes and exactly the original
residual energy. Its fourteen regional scores range from −0.009014 to 0.007655;
the direction/bin-only residual predictor scores zero in every positive-energy
cell. **Original CCA LMP predictions in M1 are positive and exceed this fixed
linear timing baseline on all seven dates**, with original LMP scores between
0.003991 and 0.021427. This is a post-observation developmental lead, not a
prespecified success, independent replication, conditional-information estimate
or evidence that other behavior is ruled out. It motivates a bounded check of
whether the current centered-profile classifier discards useful magnitude.
Do not infer a supported regional fingerprint from this descriptive pattern.

Ledger: **20 valid comparisons = 12 starting models + 2 successors + 6
controls**. Six of eight post-starting comparisons are controls. Both explicit
challenger decisions retained the unaccepted starting_10 comparator. No
qualified improvement, accepted candidate, final analysis or reserved opening
exists. The ten-valid-comparison patience count after the minimum begins with
the next valid comparison; reaching 20 alone is not a stop. Remaining mandatory
controls, maximum48 and the original search clock still apply. Calibration and
all four reserved dates remain closed.

The attended shell exited normally at **08:01:36 UTC**. Scheduler observation
at **08:01:47 UTC** reports **47352945 COMPLETED, exit0:0**, 531 elapsed
seconds, one allocated CPU, TotalCPU25.733 seconds and step MaxRSS139496K.
Charge the full attended allocation **531/3600 = 0.1475 allocated CPU-hours**
to EP06, including coordination time; extern's533 seconds is not another
charge. Release the entire one-CPU-hour reservation. No job or compute PTY is
live, no source remains actively consumed, and no automatic renewal exists.

## 2026-10-11 01:02 Pacific — development_control_visualization — completed comparison figure rendered

The [PNG](figures/development_controls/ep06_development_controls.png) and
[PDF](figures/development_controls/ep06_development_controls.pdf) were rendered
in47352945 from six named completed reports. Root visually inspected the PNG:
both animal directions, exact accuracies, the named source-session omission,
chance-reference label and single-intervention limitations are readable.
This figure adds no fit, statistical test, biological replicate or opening.
The newly completed support-control results are reported separately above.

## 2026-10-11 00:59 Pacific — residual_fingerprint_development — support controls and completed-result figure ready

Both new array/metadata helpers and their one focused test each have independent
source-review PASS. The support runner also passed independent review of
unit restrictions, native basis/common energy, whole-control unavailability,
publication before outcomes and two-control counting. Root reviewed the
completed-report figure source and the bounded continuation script. No
scientific support-control execution is claimed yet.

Run `tools/support_controls.sh` once in the same verified attended allocation
47352945: two focused helper checks, changed ASTRA validation, support fit/score
and the already completed six-comparison figure. Its deadline is the earliest
of ten minutes, original allocation end minus45 seconds, and the unchanged
search deadline. It neither submits nor renews an allocation. Consumed new
source is frozen during execution. Preserve the single existing reservation.

## 2026-10-11 00:55 Pacific — residual_fingerprint_development — two support controls selected in the same allocation

Adopt the [selected-electrode-unit and elapsed-time controls](paper_plan.md#october-11-selected-electrode-units-and-elapsed-time-controls)
before either fit or score. Reuse the suitable root-owned, attended allocation
47352945 with its original 30-minute end and one CPU/7 GiB request; no new
scheduler submission or reservation is needed. Root will run only after the
two new focused checks and independent source reviews. New helper/runner
paths do not alter the completed eighteen comparisons. The proposed unit
restriction is explicitly narrower than a bidirectional electrode intersection;
the time baseline reports genuine residual-energy scores, not artificial band
profiles. Neither can convert failed signed stability into a promotion.

Both controls remain unrun at selection. Publish all 28 genuine fit/prediction
cells before outcomes, retain complete paired support, and preserve unavailable
results without rank fallback or session loss. The search clock, budgets and
all role boundaries persist. If both are available, the valid total becomes
20, with six controls among eight post-starting comparisons. This reaches only
the minimum; it is not an automatic scientific stopping point.

## 2026-10-11 00:54 Pacific — residual_fingerprint_development — eighteen comparisons complete; pairing and session alternatives remain

The batch in attended allocation **47352945**, verified on **sh03-06n11.int**
with one requested CPU and 7168 MiB, returned **0**. All fourteen focused
checks passed in 0.440 seconds; changed ASTRA validated. Both successor
families and the mismatch (42 full prediction cells) were published before
new scoring. Successor scores completed 07:53:31 UTC and controls 07:53:32 UTC.
All six scientific comparisons are available. The attended allocation remains
open for the next registered controls and figure; do not release or double
charge its at-most one CPU-hour reservation until it ends. Current batch
source is no longer actively consumed; preserve all completed artifacts.

| Comparison | Mihili → Chewie BA | Chewie → Mihili BA | Result |
| --- | ---: | ---: | --- |
| starting_10, fixed provisional comparator | 0.750 | 0.8333 | Raw signed omissions fail |
| adaptive00, two-PC ridge | 0.625 | 0.8333 | Mean improvement −0.0625; noise tolerance 0.08333; signed omissions fail |
| adaptive01, CCA covariance regularization 0.1 | 0.750 | 0.8333 | Improvement 0; tolerance 0; signed omissions fail |
| Recording-support metadata | 0.625 | 0.6667 | Lower than neural comparator; unmeasured recording confounding remains unresolved |
| Fixed per-session band-label permutation | 0.250 | 0.1667 | Frequency-label sensitivity, not a permutation significance test |
| Fixed within-direction whole-trial mismatch | 0.750 | 0.8333 | Classification survives this mismatch; trial-coupling interpretation unsupported |

The seven classifier omission comparisons are complete. Omitting **Mihili
March4** drops Mihili→Chewie BA to **0.5**, while the reverse is 0.75. This
and the previously failed signed omissions leave the no-single-session
requirement unmet for starting_10. The pairing intervention changes the actual
band recovery scores while preserving classifier accuracy; this does not prove
absence of coupling or quantify a null distribution. A single fixed band
permutation likewise gives sensitivity evidence only. Neither completed
intervention is automatically labeled a passed scientific control.

**Challenger decision 1:** retain starting_10 as provisional comparator over
adaptive00 because accuracy falls and the signed gate remains failed.
**Challenger decision 2:** retain starting_10 over adaptive01 because accuracy
is unchanged and the signed gate remains failed. Neither successor qualifies
as a material improvement or resets patience. Neither comparator nor challenger
is promoted. Evidence: [successors](runs/residual_fingerprint_development/adaptive001/scores/complete.json),
[control availability](runs/residual_fingerprint_development/controls001/scores/complete.json),
[profile controls](runs/residual_fingerprint_development/controls001/scores/profile_controls.json),
[metadata](runs/residual_fingerprint_development/controls001/scores/control00_metadata.json),
[pairing](runs/residual_fingerprint_development/controls001/scores/control03_mismatch.json).

Ledger: **18 valid scientific comparisons = 12 starts + 2 successors + 4
controls**; two completed challenger decisions are attached to those successor
trials, not counted again. Post-starting controls are **4/6**. The post-minimum
patience rule is not active below 20. No accepted candidate, final analysis or
reserved opening exists; calibration and all four reserved dates remain closed.
Next, execute the already registered same-electrode/unit and behavioral-time
controls, then use the remaining stopping rule without an unsupported rescue.

## 2026-10-11 00:52 Pacific — residual_fingerprint_development — attended Slurm request queued

The single selected request is **47352945**, `ep06-followup`, one CPU,
7 GiB, zero GPUs and 30 minutes. Slurm reports queued, waiting for resources.
The coordinator owns its live attended PTY and will verify the actual hostname
and job variables before executing the reviewed script. No scientific work or
tests have run in this request yet. Preserve the at-most one CPU-hour reserve;
no duplicate request or unrelated job is used. This is standalone Codex-to-
Slurm execution, not a canonical BR harness launch.

## 2026-10-11 00:51 Pacific — residual_fingerprint_development — all selected source reviews passed

Independent review passed the two successor and four-control runners,
two profile-control tests, paired-mismatch publication integration, new PCA
ridge family, exact-grid challenger calculation and attended shell script.
Root review agrees; there are no unresolved findings. All consumed source,
tests and analysis specifications are frozen for the selected allocation.
Proceed with the one attended request below, reserving at most one CPU-hour.
Scientific results and test execution are not yet claimed.

## 2026-10-11 00:51 Pacific — residual_fingerprint_development — bounded attended comparison session selected

Select one attended `srun --pty` shell on russpold with one CPU, 7 GiB,
zero GPUs and at most 30 minutes, running `tools/followup_profiles.sh`.
Discovery at 07:49 UTC confirmed the partition's seven-day/eight-GiB-per-core
limits and CPU hardware; quota has adequate scratch/Oak capacity. Reuse the
existing Python3.12/numpy/scipy/h5py modules. No separate tests-only job or
one-fit submissions are selected. Exit the shell when the comparison batch
finishes. Reserve at most **1 allocated CPU-hour**, conservatively allowing
two allocated CPUs for the 30-minute request; observed allocation/accounting
will replace that bound. No other EP06 job is live.

Root and independent source review pass the new genuine PCA2-to-two-target
ridge family and successor runner. Profile controls and the paired-mismatch
runner are undergoing final independent integration review; execute only
after it completes. The batch runs focused tests of changed dependencies,
validates changed ASTRA, publishes all 42 new predictions, and then scores
the two successors and four controls. All tests remain unrun at selection.
Write launch evidence under `runs/residual_fingerprint_development/launch002`,
new model results under `adaptive001` and controls under `controls001`.
The existing 07:40:05 UTC search clock continues without reset. Completion
does not waive any remaining control, final freeze or interpretation gate.

## 2026-10-11 00:47 Pacific — residual_fingerprint_development — successors and four controls adopted prospectively

Adopt the [two evidence-led successors and first four controls](paper_plan.md#october-11-evidence-led-successors-and-first-recording-controls)
under the active model space and continuation authority. Compare low-band
electrode-PCA ridge and stronger-regularized CCA with starting_10, preserving
the failed signed gate. Separately examine recording-support metadata,
leave-one-session classifier influence, frequency-label sensitivity and paired
whole-trial mismatch. All dates, native trial trajectories and paired regions
remain included. No successor/control scores have been opened and no compute
is currently reserved for this batch. The existing BR advisory criteria and
independent agent review inform the alternatives; neither is a canonical BR
verdict, harness launch or inspection of local artifacts by the BR server.

Three source-only agents own the new PCA-ridge family, successor runner and
profile controls; root owns the control runner and allocation. Metadata and
paired-mismatch helpers have independent source review PASS, tests unrun.
Use one bounded attended compute allocation for these short tasks. Six valid
scientific comparisons would yield 18 total; no fold/test/cell is an extra
trial. Remaining controls and the existing 20–48/patience rules stay active.

## 2026-10-11 00:42 Pacific — residual_fingerprint_development — twelve real comparisons complete; no candidate promoted

Job **47352162 completed, exit 0:0**, with 49 elapsed seconds, one allocated
CPU, TotalCPU 23.281 seconds and batch MaxRSS 455976K. All fourteen focused
checks passed in 1.360 seconds; both changed ASTRA programs validated. All
84 frozen fitted/prediction cells and 84 scored cells are complete, giving
**12 valid initial scientific comparisons**, not 84 independent experiments.
All per-candidate 128-assignment references are available. No calibration
values or reserved-session outcomes were requested. Evidence:
[complete initial results](runs/residual_fingerprint_development/starting001/scores/complete.json).

| Comparison | Mihili → Chewie BA | Chewie → Mihili BA | Mean BA |
| --- | ---: | ---: | ---: |
| Mean ridge, rank 2, full profile (00) | 0.625 | 0.6667 | 0.6458 |
| Mean ridge, rank 4, full profile (04) | 0.500 | 0.5000 | 0.5000 |
| CCA, full profile (09) | 0.625 | 0.8333 | 0.7292 |
| CCA, low six bands (10), provisional leader | 0.750 | 0.8333 | 0.7917 |

**None of the twelve passes signed-profile stability.** For starting_10,
the raw cross-animal signed cosine is 0.0735508; omitting Mihili February18
gives −0.389508 and omitting March4 gives −0.00787579. Its two conditional
label-reference fractions are 32/128 = 0.25 and 16/128 = 0.125. Accounting for
the fixed twelve-way maximum gives **44/128 = 0.34375**, with all assignments
retained. These are development references under unverified exchangeability,
not adaptive-search or population significance. Above-chance point accuracy
alone is insufficient, and no candidate is promoted or reserved data opened.

The [leader report](runs/residual_fingerprint_development/starting001/scores/starting_10.json)
also shows small, often nonpositive band recovery scores. Chewie October14
M1's 0.5–4 Hz score is −0.267828; retain it, without clipping or dropping the
band/session. This motivates testing multivariate predictive capacity versus
regularized-whitening stability, plus recording-support, trial-pairing and
session-influence controls. It does not justify changing the signed gate.
The initial minimum is complete; the overall minimum 20, patience and maximum
48 rules remain active. Successors must follow this observed evidence.

The persistent search clock began **2026-10-11 07:40:05.032915 UTC**, epoch
1791704405.0329149, in `runs/residual_fingerprint_development/search_clock.json`.
Its unchanged limit is 345600 seconds and is not reset by later jobs.
The EP07 plot took five seconds (1791704399–1791704404), charged once to EP07
as **0.0013888889 allocated CPU-hours**. Charge the remaining **44/3600 =
0.0122222222 hours to EP06**. The 0.0136111111-hour parent total is not duplicated.
Release the entire shared reservation; all source is now unfrozen. The full
initial batch was much faster than a long allocation needs, so subsequent
comparisons should share a bounded attended allocation or be aggregated,
not spawn a separate scheduler job for each tiny fit.

## 2026-10-11 00:39 Pacific — residual_fingerprint_development — first empirical comparisons submitted

Submitted **Slurm job 47352162**, using `tools/start_profiles.sbatch` with
1 CPU, 7 GiB, zero GPUs and a one-hour limit. This is the only writer for
`runs/residual_fingerprint_development/starting001`; launch/check logs use
`runs/residual_fingerprint_development/launch001`. Retain the at-most 2
allocated CPU-hour reservation until actual resources are observed. All
consumed source is frozen. Submission is standalone Codex-to-Slurm execution,
not a canonical BR harness launch or scientific acceptance.

Fourteen focused checks run first, followed by the separately scoped EP07
completed-report figure and the twelve initial empirical comparisons. No
scientific fit, score or search-clock start is claimed at submission.

## 2026-10-11 00:39 Pacific — residual_fingerprint_development — reviewed batch ready for submission

Independent final review passed for the complete runner and its two integration
tests, including the fixed-twelve maximum-mean selection reference. Root and
independent batch review agree on the `--mode` interface, exact six-family/
84-cell publication guard, evaluation-only scorer, partial-attempt preservation
and no score-stage refitting. All fourteen selected checks remain unrun until
compute. No EP06 outcome has been scored. Submit the selected single parent
now, preserving the at-most 2 CPU-hour shared reservation; all consumed source,
tests, contracts and ASTRA programs are frozen until the allocation ends.

## 2026-10-11 00:36 Pacific — residual_fingerprint_development — first twelve comparisons selected before scores

Adopt the [initial recipe/reference specification](paper_plan.md#october-11-initial-comparisons-and-development-references)
under existing implementation and continuation authority. This fills in the
registered mean/median/PCA representations, ridge/CCA mappings and prototype/
logistic classifiers before any EP06 neural fit or profile score. Twelve
comparisons reuse six genuinely distinct fitting recipes over all seven
dates and both regions: 84 fitted cells. Low-frequency subsets and classifier
variants reuse the relevant frozen fits/profiles. The broader adaptive menu
remains open. All initial predictions precede any evaluation-outcome read;
calibration values and four reserved dates remain closed to this stage.

Candidate-conditional 128-assignment label references retain all ties and
undefined assignments. They do not adjust for the full adaptive search or
establish two independent directional tests. Raw signed-profile agreement,
all seven signed omissions and a separately reported centered-shape diagnostic
are fixed prospectively. The paired-reference material-improvement heuristic
is also fixed before seeing initial scores. It does not change the 20–48 valid
trial limits, patience 10, two successor/two challenger requirements, minimum
40% post-starting control fraction or any final access/claim boundary.

The CCA primitive, median/PCA fitting integration and label-reference modules
have independent source review with no unresolved findings. Their seven new
checks remain unrun; the two new runner checks are being completed. Reuse
unaffected earlier checks, while rerunning five mean-ridge/profile checks whose
common fitting dependency changed. Root's selected `tools/start_profiles.sbatch`
aggregates these checks, changed ASTRA validation, all initial empirical fits
and scoring, plus one separately scoped EP07 completed-report figure. No new
synthetic research program is selected.

Proposed allocation: **1 CPU, 7 GiB, zero GPUs, at most one hour**, numerical
threads all 1, actual 1–2 CPUs accepted solely for memory rounding. Reserve
at most **2 allocated CPU-hours shared across the parent**, including the
at-most 95-second EP07 plotting interval, never an additional reservation.
On completion charge the actual plotting interval to EP07 and all other
parent time to EP06 once. Existing costs and the 1200 CPU-hour limit remain;
the persistent 96-hour search clock starts immediately before the first fit.
Estimate below 0.5 GiB additional retained fitting/prediction/report storage;
use six containers and node-local caches, with no intensive HOME writes.

Resource discovery at 07:35:47 UTC confirms russpold's seven-day maximum,
8 GiB/core limit and 24–32-core, 191–256 GiB hardware; this serial task requests
one core only. Quota remains HOME 14.5/15 GiB, Scratch 5.5/100 TiB and Oak
171.9/250 TiB. No test, fit, score or submission has occurred at this entry.
Submission follows final runner/batch source review; root owns the only launch.

## 2026-10-11 00:31 Pacific — source_session_support — all seven development exports complete

The [preparation report](runs/source_session_support/development001/report.json)
records a completed export of all **1954 trials across seven development
sessions**, with no dropped session, trial or native unit. Each region/session
has 96 training-finite electrodes and the fixed 15-electrode subset. Native
M1/PMd unit counts by session are 44/104, 38/121, 39/76, 74/114, 82/167,
70/137 and 88/190. Counts passed the native nonnegative-integer/finite rules.
The immutable role table and score-free EP07 handoff are complete. Four
reserved sessions remain closed under EP06 roles. This qualifies development
data support, not population rank or predictive success.

Job **47351138 completed, exit 0:0**, with 278 elapsed seconds and one allocated
CPU. EP06's measured preparation interval is 143 seconds, charged once as
**0.0397222222 allocated CPU-hours**. The remaining 135 seconds are charged to
EP07; the 0.0772222222-hour parent total is not duplicated. Release the shared
reservation. Batch MaxRSS was 3475652K. The actual neural export ran from
07:27:50.303895 to 07:30:11.860671 UTC, after EP07 primary completion.

Data: `$SCRATCH/br_autoresearch/episode06_lfp_regional_fingerprints/development001/development_views.h5`.
It is reproducible from retained public source, fixed role/electrode seeds and
durable metadata. No scientific EP06 fit or score has run and its 96-hour
search clock has not started. Source is now unfrozen. Next work connects the
reviewed fitting, scoring and classifier components to the first twelve
prospectively specified development comparisons; empirical support is still
unobserved.

## 2026-10-11 00:30 Pacific — source_session_support — EP07 opening complete; real development export running

The score-free EP07 operational opening record now certifies a complete
analysis freeze and finished single primary opening. The separately scoped
coordinator projected only this completed metadata and the closed trial-role
table to the episode-local handoff; no sibling model, prediction or score is
an EP06 input. Real seven-session development export started in parent47351138
at epoch1791703669. All eight exporter/reader/profile/classifier checks remain
passing from47350382. Preparation is still running at this observation; no
completed export, empirical fit, profile score or search-clock start is claimed.

The actual parent has1 CPU/7GiB/zero GPU, reducing its shared reservation to1
allocated CPU-hour and EP06's nested share to at most0.25 hours. Measure and
charge this preparation interval once on parent completion. Four reserved
sessions remain closed under EP06 roles. Same-corpus exposure is documented
and does not create independent confirmation.

New prospective `cca_mapping.py` and its two focused tests have independent
source review with no findings. Those tests remain unrun; the code is not
consumed by this preparation job and no CCA recipe has yet been adopted.

## 2026-10-11 00:24 Pacific — source_session_support — conditional companion carried to job47351138

Parent47350382 ended before its first primary invocation because of a Bash
empty-array launch defect; all eight EP06 checks remain passing and no EP06
preparation/handoff occurred. Its entire103-second cost is charged once to
EP07, with zero EP06 preparation charge and the old reservation released.
The diagnosed launch repair now submits parent **47351138**,1 CPU/7GiB/one hour,
reusing every completed check. This same conditional development001 export
retains the completed-EP07-opening guard and900-second bound; no new experiment
or repeated test is selected. Shared resource accounting and source freeze
remain as described below. No EP06 neural contact has occurred at submission.

## 2026-10-11 00:20 Pacific — source_session_support — eight component checks passed; export still conditional

The separately scoped companion in running parent47350382 passed all eight
new checks: two primary-profile, two classifier, two role-reader and two
exporter checks, in9.360seconds. Changed EP06 ASTRA also validated. Evidence:
[compute checks](runs/source_session_support/development001/component_checks_47350382.txt).
These checks used fabricated arrays/HDF files only; actual EP06 source export
has not yet occurred. The parent has actual1 CPU/7GiB, reducing this stage's
nested reservation to at most0.25 hours within the shared1-hour parent total.
EP07 qualification and its completed primary opening still precede neural
preparation. Reuse these passing checks for the unchanged implementations.

## 2026-10-11 00:19 Pacific — source_session_support — conditional companion submitted

Parent **Slurm job47350382** requests1 CPU/7GiB/one hour for EP07 continuation
and the separately scoped conditional EP06 preparation described below.
Final source review passed after replacing mixed-freeze deserialization with
four score-free completion facts and a timestamp in EP07's existing opening
status. EP06 reads only that completed operational record and the closed role
CSV. Eight new component checks run before any conditional EP06 neural export.
The preparation stage has a hard900-second bound including termination grace;
it is skipped if EP07 does not complete or EP06 checks fail. No EP06 data have
been opened at submission. All consumed EP06 source/tests/ASTRA are frozen.
The parent reservation includes this stage and will be accounted once.

## 2026-10-11 00:18 Pacific — source_session_support — conditional development export selected

The seven-session exporter, separate candidate/outcome reader, primary profile
scorer and classifier now have independent source review with no concrete
findings. Their eight new focused tests remain unrun. Reuse the three passed
bandwise checks and previously passed component checks; do not repeat the
existing metadata inventory. The coordinator's new `tools/prepare_after_ep07.py`
is under source review. It projects only closed EP07 operational timestamps
and1954 per-trial role records, then calls the episode-local exporter. It does
not import a sibling score, fitted model, prediction or neural array.

Select a separately scoped conditional companion in the forthcoming EP07
continuation allocation:1 CPU/7GiB/zero GPU, at most900 seconds for real EP06
development-role export, only after all eight new tests pass and EP07 has
actually completed its full analysis freeze and single primary opening.
The coordinator verifies those completed records before exclusively writing
`runs/source_session_support/ep07_primary_handoff.json`; a live or merely
submitted EP07 job cannot satisfy the condition. No scientific EP06 fit,
profile score or search-clock start is selected by this preparation.

Export all1954 trials from the seven fixed development dates to
`$SCRATCH/br_autoresearch/episode06_lfp_regional_fingerprints/development001/development_views.h5`.
Preserve full native units, source trial identities, fixed paired roles and
the training-selected15-electrode subsets. Store full96-electrode features
for later registered controls; ordinary candidate reads use only the fixed15.
Keep reserved dates closed. Copy small role/preparation records, including
an incomplete export report, to `runs/source_session_support/development001/`.
The neural view is reproducible from the retained public source and fixed
metadata; scratch is purgeable. Estimate below1GiB additional storage.

The parent batch shares one reservation; its total allocation is never charged
twice. Record conditional EP06 preparation start/end times, charge that portion
to EP06 and the remaining parent time to EP07. At most0.5 allocated CPU-hours
of the parent's at-most2-hour reservation cover this stage if memory rounding
allocates2CPUs. No reservation is additional to the parent total. No export,
metadata handoff, neural contact or new allocation has occurred at this entry.

## 2026-10-11 00:09 Pacific — residual_fingerprint_development — initial classifier conventions selected

Before any neural profile is scored, specify the existing centered-shape,
prototype and regularized-logistic branches. Center each regional profile
across its declared blocks and divide by its L2 norm. Prototype classification
uses the equal-session mean of source normalized rows per assigned class,
normalized again, with cosine scores. Logistic fitting minimizes mean row
logloss plus half the explicit positive candidate penalty times squared
coefficient norm, leaving the intercept unpenalized. Use Newton/Armijo with
at most100 iterations,40 line-search halvings and maximum absolute gradient
at most1e-9; numerical failure is unavailable without an automatic retry.

An exact prediction tie is explicitly unclassified and receives half credit
in balanced accuracy. A missing block, zero/nonfinite profile shape or
prototype, or numerical failure makes the complete comparison unavailable;
never remove a session/block or favor one region to resolve a tie. Bind all
source/target rows to physically identical ordered feature blocks and paired
session identities. Regional native population axes remain independent.
Labels are explicit complete M1=1/PMd=0 pairs or declared within-session
swaps, so future full-analysis nulls can refit every label-dependent step.
No target-fitted transformation or target-label use in fitting is allowed.

These are delegated prospective model conventions, not a new inferential
threshold, starting schedule or execution grant. Implement the array-only
classifier and two focused checks; tests remain unrun until useful compute.
EP06 neural data remain closed pending EP07's completed freeze and primary
opening. No new allocation, scientific trial or search-clock start occurs.

## 2026-10-11 00:02 Pacific — source_session_support — development role conventions selected from existing metadata

The permitted existing `runs/source_session_support/metadata001/session_metadata.json`
supports all seven development dates and1954 original trial positions. Each
trial declares30ms bins and successR; all eight directions and the selected
window are structurally present. No neural values were accessed for these
choices. Preserve original session positions and source MATLAB row identities.

Choose nearest circular octant with tolerance0.005radian while retaining the
original angle; observed maximum offset is0.0027871747297867344radian. Specify
the primary movement window as half-open[150,450)ms, offsets5--14 from
`idx_movement_on-1`, for both M1 and PMd. This is a prospective EP06 endpoint
convention, not a claim that metadata resolves upstream endpoint semantics.

Within each direction, select training=floor(n/2), calibration=floor((n-training)/2)
and evaluation=remainder. Require at least7 training,4 calibration and4 evaluation
trials per direction without dropping or merging a direction/session. Use local
PCG64 with SeedSequence[6006101,animal_index,original_session_position,direction],
animal indices Mihili0/Chewie-L1, on original-row-sorted trials. Assign the same
permutation to both regions and export roles in original source order. Alternate
the source-ordered assigned calibration rows within each direction between two
halves, retaining every row. Observed minima are8/4/5, so all seven dates meet
this structural rule before neural qualification.

| Development date | Original trials | Training | Calibration | Evaluation |
| --- | ---: | ---: | ---: | ---: |
| Mihili20140217 |208|102|51|55|
| Mihili20140218 |225|110|54|61|
| Mihili20140304 |203|100|49|54|
| Chewie-L20160929 |208|102|52|54|
| Chewie-L20161005 |202|99|48|55|
| Chewie-L20161007 |168|82|41|45|
| Chewie-L20161014 |740|369|184|187|

Every region has a stable guide with96 physical electrodes carrying all nine
bands. Actual neural finiteness/rank remain unqualified. On permitted training
data only, identify guide-complete electrodes finite across the training window
and draw15 without replacement using local PCG64 SeedSequence
[6006102,animal_index,original_session_position,region_index], with M1=0/PMd=1.
Sort selected physical IDs and retain that one subset across all candidate
branches. Regions keep independent electrode and native-unit axes. No repeated
subset campaign, evaluation-based selection or redraw is selected. Preserve
all native units and trials; invalid selected evaluation features or noninteger,
negative or nonfinite used-window spike counts stop visibly without silent
unit/trial/session removal. Further retained-session loss requires review.

Implement only an episode-local development exporter and separate candidate/
outcome readers. Actual export waits for an explicit episode-local metadata
handoff recording observed EP07 freeze/completed primary opening and exact
closed per-trial EP07 roles. The handoff contains no sibling scientific scores
or fitted state, and has not been created. Export only the seven development
sessions, preserving reserved outcomes. No source copy, export, allocation,
neural read, budget reservation or EP06 search-clock start occurs here.

The preceding primary-profile implementation and its two tests are now source-
reviewed with no concrete findings, but remain **unrun**. Classifiers, finite
starting recipes and quantitative control/search conventions are still separate
prospective work. The existing12-start/20--48-trial requirements remain unchanged.

## 2026-10-10 23:57 Pacific — residual_fingerprint_development — primary profile scoring selected

Specify the previously open primary denominator/weighting before neural
scoring, within the existing delegated model-development choices. Use pooled
zero-origin training-template residual energy, uniform whole-trial/native-bin
rows and pooled squared magnitudes over frozen population coordinates:
`S_block = 1 - sum((R - prediction_block)^2)/sum(R^2)`. Retain negative scores
and evaluation-versus-training mean shifts. Do not recenter evaluation
residuals, average coordinate-wise ratios, clip or add a denominator epsilon.
Zero/nonfinite residual energy is unavailable. The direction-balanced
component score and its later positive energy gate remain separate.

Implement an array-only `tools/primary_profile.py` that uses the existing
frozen-template transform and preserves exact evaluation trial/direction/time,
block and population bindings. A paired M1-minus-PMd profile requires the same
session/trials/directions/native times/blocks and equal rank, while retaining
each region's independent neuron/electrode axes. No classifier, profile
normalization, session reducer, empirical access or new numerical success
threshold is added by this component. Two focused independent mean-shift/
pairing checks will accompany it and remain unrun until substantive compute.

EP07 job47346999 is separately executing its controls and conditional primary
path. EP06 still requires EP07's completed freeze and primary opening before
any neural preparation/fitting/scoring. This source-only implementation creates
no job or resource reservation and does not start EP06's96-hour scientific clock.

## 2026-10-10 23:48 Pacific — bandwise_prediction — three component checks passed

The separately scoped companion in EP07 job47345111 passed all3 new bandwise
array checks and changed ASTRA validation. Evidence:
[checks](runs/bandwise_checks_job_47345111/checks.txt) and
[status](runs/bandwise_checks_job_47345111/status.txt).
These establish the analytic PCA/template/ridge arithmetic, preserved training
origin and native feature-only transport, plus role/support refusals. They are
not neural fits or EP06 scientific results. Charge the shared allocation once
to EP07; no independent EP06 reserve. Neural access remains closed until EP07's
full freeze and single primary opening. Remaining EP06 model assembly and
statistical choices are not established by these component checks.

## 2026-10-10 23:40 Pacific — residual_fingerprint_development — bandwise array checks bundled with EP07

Select the three already implemented bandwise predictor checks and changed
ASTRA validation as a companion in EP07's substantive development/control
allocation. Use only fabricated arrays and compute-local temporary files;
no EP06 neural view or score is opened. Results will be recorded under
`runs/bandwise_checks_job_<job_id>/`. Parent resource use is charged once to
EP07, with no duplicate EP06 CPU-hour reservation. The tests are currently
unrun; the adopted scientific roles and EP07 precedence remain unchanged.

## 2026-10-10 23:16 Pacific — residual_fingerprint_development — first bandwise fitting component selected

Advance the missing empirical modeling code while preserving EP07's full
freeze/primary-opening prerequisite for EP06 neural access. The existing
projector, scoring, nomination, stability and H reducers do not generate
the primary residual fingerprint. In particular, componentE subtracts
evaluation-cell means and cannot replace the required training-only template.

Select an array-only first implementation in `tools/bandwise_prediction.py`
for the registered15-electrode mean, training-only population PCA and ridge
branch. It receives admitted training arrays and explicit metadata; it has
no reader, data-role exporter, electrode selector, classifier or scorer.
All other registered model families remain available for later development.
The caller must provide a positive rank and ridge penalty, explicit registered
feature blocks and exactly15 admitted electrodes. No rank schedule is imported
from another episode; no rank, channel or session is silently dropped.

Declare these engineering conventions before empirical use: center
unresidualized training spike rows with equal whole-trial/native-bin weight
and fit orthonormal PCA loadings; estimate each direction/native-bin mean
from training spikes only, subtract it and project through those frozen axes.
Average the supplied15 electrodes separately for each of the nine native
feature classes. Fit feature centering/scaling only on training rows, assigning
scale1 to an exactly zero-variance predictor without deleting it. Fit separate
multioutput ridge maps for the explicitly declared single-band, contiguous
group or full-profile blocks, with an unpenalized intercept. The selected
positive penalty and rank remain fixed; numerical support failure is explicit.

Feature-only prediction cannot receive evaluation spikes. A separate frozen
training-template transformation may map supplied observations into the same
residual coordinates for a later scorer. Bind session, region, roles, disjoint
trial identities, native bins, electrodes/bands and axes before decoding
values. Identity checks do not establish source provenance or opening rights.
The primary profile denominator/weighting convention, finite search schedule,
electrode subset selection, trial-role publication and matched M1/PMd support
remain separate prospective runner work; no score definition is silently added.

Three focused independent arithmetic/role tests are selected with the module,
to run only in a subsequently selected substantive allocation. They are unrun;
source implementation and independent review are in progress. No scientific
payload, neural fit, reserved outcome or sibling fitted state is accessed,
and no new job or resource reservation is created by this preparation.

Implementation and independent source review are complete for the module
and three focused tests. No concrete defects were found. The tests include
independent analytic PCA/ridge calculations, an evaluation mean shift retained
by the frozen training template, physical-band permutation consistency and
metadata/rank refusals. All remain **unrun**; changed ASTRA validation is also
pending a substantive allocation. No empirical readiness or score is implied.

## 2026-10-10 22:44 Pacific — component_nomination_and_support — old-core checks passed; adopted roster migration selected

EP21 parent **47329203 completed, exit0:0**. Its companion evidence at
`runs/scoring_composition_checks_job_47329203/` reports all twelve selected
scoring/nomination checks passing in0.070seconds and ASTRA schema/semantic
validation passing. The parent5888seconds ×3allocated CPUs is charged once
to EP21, not here. These checks cover the historical nomination implementation;
they do not qualify the adopted eleven-session core. Seven roster-independent
scoring checks remain reusable while that source stays unchanged.

The live-writer restriction has ended. Select the already authorized executable
nomination migration: three Mihili/four Chewie development dates; every remaining
two/three nomination session scoreable for all candidate components and regions;
all-three/at-least-three-of-four outer stability; all-three/at-least-three-of-four
common final support; and final winners bound to actual stable outer-family
evidence. Final directional agreement will revalidate the retained nomination
and stability records; reserved H will require that agreement before reading
its scalars. Other-six/all-seven pooled references and fixed thresholds remain.
Replace the caller-only completion Boolean with the retained outer assessment.

One worker owns nomination, outer stability and H source/test changes; root owns
records and integration. The unchanged numerical scorer needs no migration.
Revised nomination and nine existing amended outer/H tests remain unrun until
substantive coordinated compute. No empirical access, scientific fit, score,
additional exclusion, budget reset or job is selected. EP07's full freeze and
single primary opening remain prerequisites for EP06 scientific fitting/scoring.

At22:52 Pacific, the executable migration is complete in nomination, outer
stability and H code and their existing five/four/five methods. Final nomination
now takes retained `outer_assessment` evidence, preserves its raw ranking winner
separately, and withholds the admissible nominee for unstable, unresolved or
mismatched outer families. Final directional agreement recomputes that evidence;
reserved H validates agreement before inspecting supplied support scalars.
ASTRA now records the adopted counts and these dependencies. Independent source
review found no concrete defect. Fourteen affected methods and changed ASTRA
validation are selected as scalar-only companions to submitted EP07 repair
job47338419, with evidence under `runs/nomination_migration_checks_job_47338419/`
when it executes. All allocation costs belong to that EP07 parent. No EP06
empirical reader or outcome opening is included. Its consumed source and ASTRA
are preserved until the parent terminates.

At22:55 Pacific, the [companion evidence](runs/nomination_migration_checks_job_47338419/)
records **all 14 revised methods passing in 0.204 seconds** and ASTRA schema
and semantic validation passing. Together with the seven unchanged numerical
scoring methods already passed in job47329203, the current 21 component methods
are covered. Parent47338419 is still running its separate EP07 real-data repair;
terminal cost will be charged once there. These are component arithmetic and
access-order checks, not a completed EP06 empirical pipeline or study result.

## 2026-10-10 Pacific — source_support — eleven-session amendment adopted

The scientist adopted the reviewed [eleven-session amendment](paper_plan.md#october-10-eleven-session-amendment-adopted),
jointly with EP07, before scores. Exclude `Mihili_CO_20140306.mat` in both
regions for native-count validity failure. Preserve the original date order,
three Mihili/four Chewie development sessions and all four named reserved
dates. Outer nomination uses the other two Mihili or three Chewie sessions;
stability requires all three Mihili folds or at least three of four Chewie
folds. Final nomination requires all three common Mihili sessions or at least
three of four Chewie sessions, includes every common scoreable session and
must match its stable outer family before bidirectional agreement. H retains
the pooled median/MAD calculation on other-six/all-seven references and its
fixed thresholds. Further loss requires another review, never renumbering or
another automatic reduction.

Applied the repository [run-episode guidance](../../.agents/skills/run-episode/SKILL.md) to GOAL, DATASETS, SEARCH_POLICY
and the existing paper plan, component details, source handoff and navigation.
Historical twelve-session metadata evidence, prior proposals and failed source
facts are preserved. EP07's full freeze and primary opening remain prerequisites;
the shared exposure record must retain exact episode/trial roles, actual
openings/freezes and transferred information. EP06 reserved results remain
same-corpus procedure checks, not independent confirmation. No model, neural
read, outcome opening, scheduler action, test execution or budget reset occurred.

The live EP21 companion still selects EP06 `component_scoring.py`,
`component_nomination.py`, their twelve tests and EP06 ASTRA. Those files remain
unchanged until explicit coordinator clearance. The nomination migration must
remove March 6, admit exact three/four rosters, nominate from all two/three
remaining target sessions and bind each final nominee to its stable outer
family. The dormant outer reducer now binds two/three-session nomination records
and exact three/four-fold assessments; H uses the other six/all seven canonical
reference sessions. Both reject the superseded nomination roster while this
implementation dependency is held. Their four outer-stability and five H test
methods now cover all-three Mihili support (two-of-three fails), the six/seven
reference counts and animal mixes, and explicit refusal of the excluded date.
These nine amended checks remain unrun and depend on the protected nomination
migration. Final-to-stable-family binding is not yet implemented. This records
adoption; no EP06 role export, neural opening, test or job was performed.
Existing resume authority persists without a new approval gate. Existing
resource charges and the first-scientific-trial clock rule are unchanged.

Coordinator and independent joint review found no concrete inconsistency in
the amended EP06/EP07 rules or dormant reducers. The nine amended methods remain
unrun; migration of the protected nomination core is still required after the
live companion finishes. EP07's separate real-data preparation has now started
under the existing resume instruction. That does not bypass EP06's full-freeze
and primary-opening dependency or introduce sibling outputs as research inputs.

## 2026-10-10 Pacific — source_support — eleven-session amendment prepared

Completed the prospective sample-size review in the [existing paper plan](paper_plan.md#october-10-eleven-session-amendment-for-decision-unadopted):
three/four source development sessions, two/three target nomination sessions
per fold, three-of-three/three-of-four stability, final-family agreement,
other-six/all-seven pooled H references, unchanged four reserved dates and
conditional two-animal interpretation. The joint EP07 proposal specifies
five new-day weights and support/omission rules. Recommendation is immediate
joint exclusion of invalid March 6 before scores, subject to the scientist's
choice; the earlier unsent request has no active response deadline.

Two read-only episode reviewers informed the concrete proposal. BR advisory
directive `ext_review_dir_12d86d49bdd4` supplied generic scientific review
criteria, not an artifact review, tailored verdict, approval or launch.
Root applied the relevant repeated-measures, selection and claim-limit
criteria locally. No scientific payloads, tests or outcomes were opened.
Operative contracts, code, roles, source records and access order remain
unchanged pending the joint decision. Existing four-session implementation
and its checks have not been revised to implement this proposal.

## 2026-10-09 — component_nomination_and_support selects fixed H arithmetic

Select the missing calibration-only H calculation under the existing formula:
standardize session reliability and profile distance separately by the reference
median and 1.4826 times its MAD, then subtract standardized distance from
standardized reliability. A development prediction uses exactly the other
seven canonical development sessions; a reserved prediction uses all eight.
Preservation is H >= 0.25, failure H <= -0.25, otherwise abstain. Missing or
nonfinite required inputs, zero standardization scales and invalid bandwise
distance scales cause abstention without dropping a reference session or
changing the fixed thresholds. No evaluation gains may enter this calculation.

The module consumes explicit component-specific calibration scalar records,
not source arrays. Bind development calls to the existing prefold nominee and
reserved calls to the existing same-component directional agreement; an absent
nominee cannot produce an H prediction. These in-memory bindings do not prove
actual freeze chronology or authorize an outcome opening. Source profiles and
the exact conversion of two regional reliability/distance inputs into one
session scalar remain unresolved upstream: neither an aggregation rule nor
a bandwise between-session variation estimator is silently chosen here.

One worker owns tools/component_support_h.py and focused tests; root owns
source review, integration and records. No neural data, H value, outcome score,
test or new allocation is selected. EP07's freeze/opening dependency remains.
The new checks are outside the unchanged pending EP21 companion; the final
reserved comparison and support-claim summary remain separate consumers of
the existing regional outcome classifications and these frozen predictions.

`tools/component_support_h.py` is complete. Its outer and reserved wrappers
consume existing nomination/agreement records and bind the exact canonical
target/component/calibration identities before numerical scalar access.
They do not reread the stored evaluation comparisons. No nominee returns
no prediction without inspecting H inputs. Missing reference keys or explicit
missing records abstain; extra target/reserved/unknown reference keys refuse
before value access. Finite negative reliability is retained. The output
keeps reference identities, medians/scales, both standardized terms, H,
prediction/abstention and reasons, without claiming preservation was observed.

Root independently reviewed the implementation and five methods in
`tests/test_component_support_h.py`, finding no concrete source defect. They
cover seven-versus-eight references in both directions and all four reserved
identities, inclusive thresholds and equal weights, missing/invalid/zero-scale
support, no-nominee no-read behavior, and forbidden session/component/role
bindings before scalar access. All five methods and changed ASTRA validation
are **unrun**. The unselected recipe uses one CPU / 1 GiB and30 seconds plus
five-second termination grace inside a future substantive allocation. It
operates on small scalar records with no model fit or scientific array; actual
runtime/RSS remain unmeasured.

There are now21 pending EP06 methods: the original12 stay in the pending
EP21 companion; the latest9 (four outer-stability and five H methods) stay
outside its unchanged tail. H arithmetic is implemented, while calibration
profile construction, regional aggregation, bandwise scale estimation, source
frequency-combiner fitting and the final reserved assessment remain unfinished.
Caller-supplied scalars must use comparable prospective prototypes/scales and
the applicable target-excluded7/all8 scope; matching keys cannot establish
that upstream fact. No neural value, empirical H prediction, outcome opening,
test, import, scheduler inspection or new job occurred.

## 2026-10-09 — component_nomination_and_support selects outer stability assessment

Select the missing record-only reducer of four existing outer nominations and
their matching nominee-only left-out regional comparisons. Apply the policy's
same-component-in-at-least-three-folds, no 0.01-tied fold and no left-out M1/PMd
sign contradiction rules. Preserve canonical target and left-out identities,
actual nomination records and all known failure or unresolved conditions.
Missing or unscoreable required comparisons cannot establish absence of a
contradiction. The reducer must not evaluate another left-out component,
re-nominate with its outcomes, infer access chronology, fit source weights,
choose an energy gate or open a reserved session.

Root owns the implementation, focused tests and records; the EP17 worker
independently reviews after its source work. No neural access, empirical score, allocation or
test execution is selected. New checks remain outside the unchanged pending
EP21 companion tail. EP07's freeze/opening prerequisite, the original session
roles and all existing margins remain unchanged. Full H prediction and source
frequency-combiner fitting remain separate empirical requirements.

`tools/component_outer_stability.py` is complete. It binds the exact four
canonical outer records and rechecks only their retained three-session
nomination arithmetic before obtaining any left-out comparison. Comparisons
must contain the prefold nominee alone. A fourth different untied nominee is
allowed; its left-out M1/PMd signs still enter the contradiction condition.
Strict sign comparison retains zero-versus-negative disagreement even when
the separate session-level rule labels both nonpositive gains a failure.
Jointly negative or small positive gains do not by themselves invalidate
selection stability, and stability does not establish preservation.

The reducer retains every individual condition, original record and unresolved
comparison. A known failed necessary condition yields unstable selection; when
the other conditions pass but required sign evidence is missing, unscoreable
or has no prefold nominee, outer stability remains unresolved. It never treats
missing evidence as sign agreement, alters a nomination using left-out gains,
or claims authentic access order, a completed H assessment or a final result.

Independent source review and root review of the four focused methods found
no remaining concrete issue. Checks cover both directions, three-of-four
selection, a different fourth nominee's conflict, ties, jointly negative and
zero gains, missing/unscoreable/nonominee states, forbidden session/component
inputs and source-record consistency. All four methods remain **unrun**.
The new unselected recipe uses one CPU, 1 GiB and 30 seconds plus5-second
termination grace within a substantive allocation. It uses only small scalar
score records; runtime/RSS are unmeasured. The existing12 methods and60+5-second
EP21 companion remain unchanged, so the new4 are outside that proposal.
All16 pending methods and changed ASTRA validation remain unrun. No empirical
data, source model, scientific score, scheduler call or new job was involved.

## 2026-10-09 — component_nomination_and_support selects component nomination

Select the next fully specified decision component around the new scorer:
outer nomination using exactly the other three target development sessions,
all-three-candidate numerical support, candidate-specific regional-sign
eligibility, median equal-region gains and the fixed0.01 tie margin. Final
nomination uses one common scoreable set of at least three of four development
sessions after the outer assessment is recorded. Numerically valid negative
or mixed-sign rivals must not be relabeled missing; negative sign-consistent
candidates remain eligible. Preserve the fixed directional/session identities,
left-out nominee-only access and reserved-session closure.

One worker owns `tools/component_nomination.py` and its focused tests; root
owns review and records. Caller-supplied prior-stage records cannot authenticate
actual access order, and this module creates no reader or fitting permission.
No neural data, source fit, scientific score, test execution or allocation is
selected. New checks are not yet admitted to the pending EP21 companion.
EP07's freeze/opening dependency and EP06's original source roles remain.

The source-animal frequency combiner is a separate missing empirical fitter.
Its provenance and exact M1/PMd swap are specified, but objective, coefficient
constraints, regularization and source-session pooling still require a
prospective internal recipe decision. This nomination implementation does not
silently choose that recipe or make the full empirical pipeline ready.

The nomination module and five focused methods are complete. Root independently
read both files and found no material defect: actual regional comparisons are
retained, exact wrong/left-out/reserved session keys refuse before record
lookup, and final eligibility uses common numerical support without deleting
valid mixed-sign or negative rivals. The bidirectional agreement helper checks
only the two final nominees; full outer stability and H predictions remain
outside this API. No empirical nomination has been produced.

Select these five methods together with the seven still-unrun scorer methods
inside the existing60+5-second EP06 companion of the unactivated EP21 proposal.
The parent request and tail allowance do not increase. All12 methods and
changed ASTRA validation remain **unrun**; this is source review and execution
preparation only. No original passed suite or metadata inventory is repeated.

## 2026-10-09 17:07 Pacific — component_nomination_and_support selects missing scoring implementation

Source inspection found only the completed metadata reader, while the fixed
component projectors and scoring equations remain unimplemented. Select an
array-only production component scorer from GOAL, policy and component_details:
direction-balanced P_C/P_D/P_E, separately projected observed and predicted
activity, normalized energy and S_k, and correct-versus-pooled/swapped gains.
The caller must supply a frozen design, genuine aligned prediction arrays,
one nominated component and an explicit positive energy gate. Choose no gate,
fit no weights and expose no other component of a left-out session. Preserve
negative scores and unresolved support/sign cases. A worker owns only the new
module and focused checks; root owns records and any later compute selection.

This is necessary missing scoring arithmetic, not another metadata inventory
or scientific simulation campaign. Prepare code and checks without executing
them on the login node. No allocation or neural access is selected. EP07's
freeze/primary-opening prerequisite, EP06 final roles, original margins and
budgets remain unchanged. Empirical readiness is not established by source
implementation or eventual finite-fixture checks.

The worker implemented `tools/component_scoring.py` and seven focused methods
in `tests/test_component_scoring.py`. Root independently read the source and
tests: the projector oracle uses explicit weighted indicator matrices,
predictions receive their own projection, the norm sums population-coordinate
energy and balances directions/trials/time, and regional classification
retains the fixed0.01 rule. No additional source defect was found in review.
Tests and changed ASTRA validation remain **unrun** pending coordinated compute;
there is no reason to repeat the earlier metadata inventory or open neural
payloads. The program records the exact focused check recipe and source
dependencies. No empirical component score or selection exists.

## 2026-10-06 Pacific — public source evidence added to existing handoff

Recorded the completed bounded online check in
[source_handoff.md](source_handoff.md#public-source-check--october-6-2026-pacific),
with exact Dryad/author-code URLs, the displayed release dates and the
distinction between downstream transformation code and an authentic source
correction. No corrected March 6 export was found; incomplete browser retrieval
is not evidence of absence. The completed 12-session metadata inventory is
unchanged and does not need another run. EP07's source remedy and prescribed
freeze/opening remain dependencies.

This edit reuses verified public findings without a new lookup. No source
payload, neural outcome, test, import, scheduler action, contact or job was
performed. Active policies, roles, thresholds and budgets remain unchanged;
the joint exclusion proposal is not adopted by this documentation update.

## 2026-10-06 Pacific — factual session mapping and unadopted joint fallback

Updated DATASETS from the completed metadata inventory: source names/dates
are known, while neural eligibility remains unqualified. March 6 is position
5, the fourth Mihili development session. The existing paper plan now gives a
concrete, unadopted joint exclusion fallback, its two-versus-three nomination
consequence, fixed reserved dates and proposed shared exposure record. The
correction-request window would begin only when the scientist sends it.
No six-session rule, threshold, role, access order, budget or job changed.
No payload read, test, scheduler action or source-route retry occurred.

## 2026-10-06 05:33 UTC — `source_session_support`: fixed metadata inventory complete

The [fixed report](runs/source_session_support/metadata001/session_metadata.json)
records **metadata_inventory_complete** for all **12 sessions / 3,090 trials**,
with no pending sessions or reported structural problems. The substantive
reader took **229.198 seconds**, starting 05:28:28 UTC, and wrote 16,751,220
bytes. All source trials are retained. Each has 0.03-second bin size and
provider success label `R`; acquisition dates match all twelve named files.
Every session contains the paired M1/PMd array headers, guides and declared
event metadata. No provider `trial_id`/`trialID` field is present: the report
uses the shared MATLAB trial-struct position as its explicit structural key,
without inventing an authenticated provider identifier.

Six focused reader checks passed in **0.056 seconds**, including a check
which forbids neural/velocity value reads. Official changed ASTRA schema and
semantic validation passed in
`astra/v0.0.14/validation_20261006_46721202.json`. Runtime output and test
evidence are under `runs/source_session_support/metadata001-job-46721202/`.
The same allocation also completed only the newly prepared EP02 (11),
EP15 (7), EP16 (3) and EP19 (3) small checks and changed ASTRA validations;
these are code checks, not scientific trials or a simulation campaign.

Slurm **46721202 COMPLETED 0:0**, actual start **05:26:56 UTC**, end
**05:33:09 UTC**, on **sh04-18n32**, one CPU / 2 GiB / no GPU.
The attended compute shell has exited. Parent elapsed **373 seconds** gives
**373 allocated CPU-seconds = 0.1036111111 CPU-hours**, charged once to EP06's
existing 1,200 CPU-hour allowance. Replace the 0.5 CPU-hour reservation with
this actual charge; reservation is now zero. Do not add the 367-second task,
375-second extern, reader time or companion checks again. Step MaxRSS is
**343,768 KiB**. The combined work finished sooner than the 30-minute cap;
no runtime padding or successor allocation was added.

No neural/velocity values, neural-quality measures, fits or scores were read
or produced. Scientific eligibility and session/trial roles remain unassigned;
zero scientific trials were consumed and the 96-hour search clock did not
start. Metadata completion does not resolve EP07's known count-representation
problem or its freeze/primary-opening dependency. Next substantive neural
work still requires that dependency (or a prospectively authorized compatible
joint-role design), plus the remaining EP06 support and recipe decisions.

## 2026-10-06 05:26 UTC — `source_session_support`: metadata inventory selected

Root selects the prepared fixed 12-session structural inventory, after its
six focused access/indexing checks, in one attended Slurm allocation:
1 CPU, 2 GiB, at most 30 minutes, no GPU. Aggregate the new EP02, EP15 and
EP16 arithmetic checks and changed ASTRA validation in the same allocation;
their inputs are episode-local code and small fixtures, never sibling outcomes.
All six numerical thread caps remain one. Root owns submission, execution
and monitoring; no duplicate job, automatic retry or neural successor.

Reserve at most 0.5 CPU-hours once against EP06's existing 1,200 CPU-hour
source-check allowance, including startup and the aggregate checks. Replace
the reservation with measured parent runtime after closure; do not charge
each episode or Slurm step again. Zero scientific trials and no start of the
96-hour scientific-search clock. The reader's fresh metadata001 directory
must not exist before its first execution. The single submitted attended
allocation is **46721202**, initially queued; runtime remains pending.

Independent static review passed. The read boundary remains identity, event,
direction and guide metadata plus neural/velocity headers; no neural values,
scoring, eligibility exclusions, session-role assignment or EP07 opening.
This entry records selection before scheduler side effects.

## 2026-10-06 UTC — `source_session_support`: metadata reader prepared

Under the scientist's continuing resume instruction, the coordinator selected
the concrete EP06 metadata-only reader on October 5 Pacific time. Implemented
`tools/inspect_session_metadata.py` for the exact six Mihili and six Chewie-L
sessions already named in the shared source. Reused the permitted sibling
reader's HDF5 field/reference conventions as source code only; no sibling
run report, fitted state, trial role or outcome was opened or imported.

The reader reads identity/event/direction metadata and guides, and neural and
velocity array headers only. It retains problems without removing a session
or trial, explicitly reports structural versus scientific eligibility, and
cannot perform neural-value qualification, fitting, scoring or role opening.
Added six focused compute-only checks covering forbidden reads, shared trial
indexing, reference-count mismatch, no whole-MAT fallback, atomic report
preservation and compute/thread guards. These are not a simulation campaign.

Updated the existing source handoff with exact CLI/input/output paths, remaining
delegated recipe work and a suggested 1 CPU / 2 GiB / 30-minute allocation
within the unchanged budget. Added only a metadata-inventory output/recipe to
the existing ASTRA source-support analysis; its qualified session-support
output remains unresolved. Independent static review passed the reader, its
six focused checks and the delegated-versus-unactivated recipe boundary.
The trial count is explicitly converted to a Python integer for JSON output.
No imports,
tests, source payload reads, ASTRA validation, scheduler calls or reservation
were made; no runtime or resource consumption is claimed. The coordinator
owns any substantive allocation and terminal accounting.

The next executable action is the structural inventory after the six small
checks inside that allocation. EP07's freeze/opening still controls later
neural fitting/scoring. No third-animal prerequisite, new scientific threshold,
joint-role amendment, session substitution or synthetic campaign was added.

## 2026-10-03 — `source_session_support`: named source handoff

The coordinator assigned outcome-blind EP06 readiness work under the scientist's
resume request. Full GOAL, DATASETS and the single search policy were read.
Inspected only the existing shared source's immediate filenames, provider
README, SOURCE.md and two provider metadata snapshots, plus the named root
location records. The complete 24-MAT-plus-README release already exists at
the path in `source_handoff.md`; stale episode statements that the recordings
were unavailable were corrected. No source acquisition/copy or MAT/neural
value access occurred, and no session was declared eligible or reassigned.

There is no identified separately sealed third animal with paired M1/PMd in
the named handoff. Chewie-R and Han/Lando are not substitutes. This is a limit
on the future external test only; the active two-animal component is gated by
EP07's frozen-analysis/single-primary-opening dependency. The coordinator's
current observation is that EP07 remains blocked on native integer counts for
Mihili 2014-03-06. No sibling outcome or fitted state was read or imported.

Read the provider's public Main.m as source-method documentation: its separate
filenames.mat mapping is not in the payload, and its unequal array-specific
latent-rank example cannot supply EP06's matched-rank recipe. The journal page
returned a browser challenge, so no claim of a new full article/supplement
review is made. This work does not revise the existing novelty conclusion.

No compute, synthetic test, scientific trial, scheduler action or canonical
mutation was selected. The next useful source step is an EP06 metadata-only
paired-session/trial/guide reader after coordination of the existing access
order; actual dates and structural eligibility must precede the role table.
Whole-trial fractions, model recipes, finite ranks and the positive energy gate
remain uninstantiated. No replacement source, downgraded replication or changed
numerical margin was introduced. Detailed permitted next actions and exact
missing external handoff are in `source_handoff.md`.

Created the previously absent `astra/v0.0.14/astra.yaml` for the existing
source-support, residual primary, frequency-rule consequence, component
nomination/support, final freeze, reserved test and bounded interpretation.
It has no runnable neural recipe, imported sibling outcome or new scientific
gate. Initial official schema/semantic validation is pending coordinator
compute; that documentation check does not unlock any outcome.

Initial official validation found one invalid uppercase option identifier,
`wait_for_EP07_freeze_and_primary_opening`; preserved its evidence in
`astra/v0.0.14/validation_20261003.json`. Mechanically renamed that identifier
and its default reference to lowercase `ep07`, leaving the access condition
unchanged. Corrected official schema and semantic validation **PASS**, verified
in `astra/v0.0.14/validation_20261003_attempt002.json`. No scientific recipe
has run and no outcome-access prerequisite changed.

## 2026-09-26 — scientific prelaunch rewrite

The question and analysis plan were rewritten around the scientific sequence:
cross-animal transfer, explanation of the signed profile, recording-confound
tests, and the correct-versus-pooled-versus-swapped prediction. A readable
acquisition-order rule now assigns four development and two reserved sessions
per animal once the source is available.

No neural or behavioral outcome was opened, no model was fit, and no reserved
session was evaluated.

## 2026-09-30 — EP07 component integration, not execution

EP06 now owns the recording-domain component consequence within EP07's paper:
source-animal correct/pooled/swapped frequency rules on common-time, direction-
contrast, and trial-residual projected activity. The residual-only EP06 primary,
session roles, controls, thresholds, nomination/support rules, and budgets
were not amended. No new experiment or scientific gate was added.

Different EP06/EP07 trial roles can expose the other episode's closed spikes.
EP06 neural fitting/scoring on this corpus follows EP07's full freeze and single
primary opening, unless a compatible joint-role design is approved prospectively
before relevant scores. EP06 still requires its own final freeze and one reserved
opening. No sibling fitted state or outcomes become a run input.

Only named design documents/local notes, the existing concept PNG, and primary
papers were inspected. No scientific payload, outcome, job, or model fit was
accessed; no analysis attempt, scheduler/process ID, or scientific artifact was
created. The source and exact field/session/trial support remain unprepared.
The next requested empirical action would remain the existing source/session
support step, with the cross-episode access order observed before neural reads.

## 2026-09-30 — details of the three existing components

- Added `component_details.md`: explicit balanced projection formulas, design-
  matrix construction, per-component examples, useful-gain interpretations and
  negative/unscorable boundaries. Linked it from the goal, data plan, paper
  plan, README and policy; no new component or experiment was added.
- Clarified that common time includes any level remaining in the frozen
  coordinates, direction-by-time includes static direction offsets, and trial
  residual is an evaluation-design projection, not the primary training-only
  mean-subtracted residual. No additional centering was introduced.
- Observed and predicted evaluation activity use the same fixed operator but
  their own projected means. Target-evaluation means remain score quantities,
  not fitted templates or target-adapted combining weights.
- Preserved all operative policy values, nomination/H/energy rules, 20–48
  trial budget, four reserved sessions and the EP07 access order. Unresolved
  role fractions and energy-gate values were not filled with invented numbers.
- No empirical analysis, attempt or scheduler/process ID; no payload, score,
  protected outcome, Git mutation or canonical action. Existing concept image
  retained. Next empirical action remains source/session support if requested.

## 2026-10-02 local scope review

Execution remains unstarted per the existing source/readiness record. Under the user's design-repair request, prospectively separated all-component numerical scoreability from nominee-specific M1/PMd sign consistency; measured unsupported rivals no longer veto another eligible nominee. Synchronized Goal, data, policy, plan and component details; see [review (consolidated)](paper_plan.md). Retained 0.01 margins, H, projectors, trial roles, EP07 access order and budget. No payload, fit, score, reserved opening or job; no ASTRA program exists at the standard path and no new gate was added.

## 2026-10-02 figure review

Viewed the former PNG and generated a less dense 1672 × 941 imagegen v2. Personally inspected the three task-defined components and corrected Panel D to apply all three rules to each component; independent visual review accepted it. GOAL/plan/output-index links and captions updated; exact prompts saved and old assets retained. No result, supported boundary, cortical-causal claim, policy change, runtime action or Git operation.
