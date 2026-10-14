# Experiment log

## 2026-10-13 13:59 Pacific — offline_information_value — complete, primary comparison negative

Parent47687188 completed76seconds on1CPU: charge0.021111111CPUh once,
release0.562222223CPUh from its reservation and retain every prior charge.
All eight sessions and35,160 fixed fits finished; the scientific process took
36.149seconds,22.299CPU-seconds and151,012KiB peakRSS. It read the original
768 fitting and320 evaluation trials (30,094,080 logical neural bytes), with
four reserved sessions closed. One complete offline evaluation is charged:
42/112 including the41 historical policy comparisons, whose conclusion stays.

The [complete report](runs/offline_information_value/attempt001/information_value.md)
and [JSON](runs/offline_information_value/attempt001/information_value.json)
retain all four fixed comparisons. Positive numbers below mean reduced MSE.

| Prefix / transfer | Chewie | Mihili | Equal-animal mean |
| --- | ---: | ---: | ---: |
| 32 / leave-session, primary | -5.3116184e-9 | -1.9957054e-9 | -3.6536619e-9 |
| 32 / opposite-animal | -4.0100248e-9 | +5.9855544e-9 | +9.8776480e-10 |
| 64 / leave-session | +1.7701567e-8 | +1.9948101e-8 | +1.8824834e-8 |
| 64 / opposite-animal | +2.0359297e-8 | +8.3103322e-8 | +5.1731310e-8 |

Primary baseline MSEs are1.5699732e-5 and1.2972797e-5 respectively. Only0/4
Chewie and1/4 Mihili primary sessions improve; every fixed-prediction session
omission stays negative. Thus this selected32-trial score provides no observed
incremental benefit under the fixed model. Both64-trial sensitivities have
positive animal means, but cannot replace the primary or establish a meaningful
acquisition gain. The analysis is exploratory on already exposed evaluation
trials; two animals, reused sessions/contexts and signal-quality proxies limit
interpretation. This is neither an equivalence result nor independent confirmation.

All three focused checks passed in0.257seconds; EP08 ASTRA schema and semantic
validation passed in this parent. Independent review found no numerical-source,
reporting or figure blocker. [PNG](figures/offline_information_value/attempt001/information_value.png)
and [PDF](figures/offline_information_value/attempt001/information_value.pdf)
show all four comparisons and session variation. No adaptive successor,
new policy search or reserved-session opening is selected. Small documentary
validation companions are included in this parent charge, not billed again.

## 2026-10-13 — offline_information_value — submitted as 47687188

Slurm accepted the selected 1 CPU/6 GiB/35-minute substantive analysis.
Root owns monitoring; no duplicate or automatic retry is selected. Scientific
completion and focused-check results await job artifacts. Reserve 0.583333334 CPUh.

## 2026-10-13 — offline_information_value — coordinator review and bounded launch selected

Root reviewed donor-only residualization, unchanged baseline shrinkage, whole-session
exclusion, matched removal denominators and complete reporting. Select fresh
attempt001 using tools/run_information_value.sbatch: 1 CPU, 6 GiB, 35 minutes,
1800-second scientific limit and 5-GiB RSS guard. Three focused checks precede
execution. Retain all historical charges; reserve 0.583333334 CPUh once for this
parent. A complete panel adds one offline evaluation (42/112), without reopening
the original 41-comparison policy search. Four reserved sessions stay closed.
Small changed ASTRA validations for EP06/08/17/20 accompany this substantive job;
charge the whole parent to EP08 only. No numerical result is implied by selection.

## 2026-10-13 20:44 UTC — offline_information_value — authorized exploratory branch implemented

The scientist approved the next EP08 step as a distinct real-data information-
value analysis: whether conditional complementarity predicts electrode-removal
loss beyond recorded signal quality and marginal association. Adopted the
complete [executable design](information_value_design.md) before new numerical
access. GOAL, DATASETS, the single policy and existing ASTRA
`offline_information_value` node now describe this branch. Preserve the original
41-comparison terminal policy conclusion and all four closed reserved sessions.
The development evaluation outcomes were already exposed during those comparisons;
this hypothesis and its eventual results are exploratory, not fresh confirmation.

The implementation uses only the existing eight-session reader and original
96 fitting/40 evaluation roles. Primary predictors use32 fitting trials;64 is
a fixed sensitivity. Full96 plus three outcome-independent partitions into
six16-electrode contexts each give384 complete electrode/context rows per state.
Fourfold conditional and marginal gains use fitting-only direction means and
normalizers. Fixed alpha1 removal models use all96 fitting trials and are frozen
before the evaluator opens40 trials. Leave-session-out and opposite-animal
regressions compare the same quality/marginal/redundancy covariates with and
without conditional gain, with equal animal/session/context-family weighting.
The primary endpoint is32-trial leave-session-out MSE reduction. All signs,
sessions, contexts, states and transfers are reported, without electrode-level
inference, policy promotion or automatic follow-up. No direction-value or
sensor–trial coupling analysis is activated.

Consequential design review reused the existing paper/design evidence and BR
retrieval dispositions (October4 05:42/06:44UTC supplied no relevant M1 evidence).
A bounded `pipeline_plan_review` of the supplied three-step summary returned
`findings=[]`, `risk_level=low`, with no KG rules consulted and enforcement none.
Its first call lacked the tool's required steps list and returned a schema error;
the corrected call completed. This generic advisory result is not evidence of
scientific validity, formal Society acceptance or execution permission.

New source files are `tools/electrode_information_value.py` and
`tools/development_information_value.py`. Three focused checks cover exclusion of
target-session/animal labels from value-regression fitting, and independent
primal-versus-dual removal arithmetic with a common denominator, plus refusal of
an apparent gain from duplicating a baseline predictor. Source review identified
that adding a correlated feature to fixed ridge can alter baseline shrinkage;
the adopted analysis therefore residualizes the conditional score against the
baseline predictor span using donor-only weighted least squares, without target
labels or residual rescaling. The baseline penalty is unchanged. These checks are prepared
in `tools/test_electrode_information_value.py`, **unrun**. No synthetic pilot or
new recovery campaign is selected. Changed ASTRA validation is also pending in
root's coordinated compute allocation; unchanged checks remain reusable.

The substantive command is in the design. Proposed ceiling:1CPU/6GiB/35minutes,
one numerical worker,1800s internal/1860s outer work limit and5GiB RSS guard.
There are35,160 small fixed ridge fits; the previous45,600 inner plus2,280 outer
fits took288.7s, a rough several-minute basis because dimensions differ. This is
not a measured cost guarantee. The complete panel charges one additional offline
evaluation against the shared maximum (42 including the historical41), without
restarting policy patience. Preserve prior CPU/wall charges and global ceilings;
root records applicable clock/budget admission and exact job/step before launch.
No resource reservation, scheduler call, source payload read, test or execution
was performed by the implementation worker on login node sh04-ln04 (no SLURM_JOB_ID).

Fresh intended output: `runs/offline_information_value/attempt001/`. It will retain
operational accounting, complete machine-readable rows/comparisons, a readable
summary and a saved-number figure when plotting is available. A resource or
technical stop remains incomplete; no selected-session result is substituted.

## 2026-10-06 — completed development evidence propagated to episode summaries

Reconciled GOAL, DATASETS, the paper plan and output navigation with the
October 4 adopted terminal report below. Removed stale claims that no neural
analysis or separated development handoff existed, and distinguished the
historical 19-comparison prefix from the completed 41-comparison search.
The policy's descriptive `status` now records the already-adopted
`development_only_no_initializer_robust_policy`; its rules, resources and
closed held-session boundary are unchanged. Existing observed reports and
their +0.003353-Q average/failed-robustness interpretation were reused. No
new source read, fit, check, proposal adoption or launch occurred.

## 2026-10-04 08:08 UTC — `final_report`: development-only terminal conclusion adopted

Root and the independent reviewer passed the final scientific wording and
stopping reconciliation. Root adopted
**`development_only_no_initializer_robust_policy`** at 08:08 UTC. Development
search is stopped at **41 valid comparisons**. The conservative 26-comparison
no-progress counter is explicitly a reconstruction after the completed 13–15
block; 11 of 21 post-coverage comparisons are conservative alternative checks.
No mandatory 60-comparison floor or new support/sensitivity reset is imposed.
Applicable coverage and actual adaptive/challenger decisions are sufficient
for this limited terminal route; no concrete additional required fit remains.

The adopted [development report](final_report.md) and
[machine-readable summary](development_summary.json) preserve the positive
average target-correspondence finding and the failed initializer robustness
screen. They do not establish a stable useful acquisition policy or a joint
electrode/trial principle. All four held sessions remain closed, no held
protocol is selected, and no final confirmation is claimed. No further EP08
method, source fit or held read is selected. Canonical BR state was not changed.

Changed ASTRA schema/semantics passed without warnings in verified allocation
46464685; evidence is
`astra/v0.0.14/validation_20261004_development_terminal.json`. The final edits
only clean prose and record adopted wording in existing descriptions; root
explicitly confirmed no further validation was needed for those description
changes. Stable analysis/output IDs and completed-development dependencies
are preserved. Terminal navigation is now updated. Root owns the separate
repository navigation and shared allocation accounting, with no duplicate
EP08 charge for the allocation.

## 2026-10-04 08:00 UTC — `final_report`: limited development conclusion and summary prepared for root wording review

Root reconciled the adopted stopping rule:41 total comparisons, a conservative
26 consecutive comparisons without qualifying progress after the completed
13–15 block, and11/21 post-coverage alternative checks, with broad applicable
coverage and multiple actual adaptive successor/challenger decisions. New
support's first score does not reset patience; raw+0.000664 is below0.001 and
the robustness screen introduces no new reset. No mandatory60-count floor
applies. On that reconciliation, no concrete applicable gap remains for the
limited development-only route; unactivated optional, held, explanatory and
external branches constrain claims rather than require another source fit.

Prepared [final_report.md](final_report.md) and
[development_summary.json](development_summary.json) for root's requested
wording review. The proposed terminal status is
`development_only_no_initializer_robust_policy`: preserve average
target-correspondence evidence, do not claim a stable acquisition policy, and
keep all four held sessions closed without final confirmation. The summary
explicitly identifies the patience count as a coordinator reconstruction.

The existing ASTRA`development_search.selection_report` and
`final_report.study_report` now describe these artifacts and the actual
completed-development dependencies. The final route no longer implies that
unselected held/coupling/resource-explanation outputs were completed or are
required for this limited report. Root still needs to review the exact wording
and validate changed ASTRA links in compute. Terminal navigation has **not**
been marked, no canonical state is changed, and no method, source fitting or
held protocol is selected by artifact preparation.

## 2026-10-04 07:55 UTC — `development_search`: initializer robustness evidence and terminal scientific recommendation

Root released the complete derived evidence from
[initializer_robustness_attempt001](runs/development_search/initializer_robustness_attempt001/initializer_robustness.json).
Exact run timing was07:48:08.567193–07:52:57.239909UTC,288.672735seconds wall,
283.796948seconds CPU and119,952KiB peakRSS. All2,280 actual target models,
45,600 inner fits,228,000 capacity scores,720 orders,4,800 direction maps,
69,120 logical marginal solves,120 normalization views and120 D fits
completed. Selection took0.924423seconds. Decoder constant-SD occurrences
were0 input/4,617 target, retained with SD-to-one. Source accounting is29,952
fitting-read calls and320 evaluation trials,324,176,640 logical bytes and
30,094,080 unique native bytes. Report size is18,571,736bytes. Within-order
full96 replay difference is zero; held access is false. The120 true selector
views have gain range approximately0.0180369373–0.0978145432. Three orders and
five seeds remain computational conditions, not separate valid comparisons;
the completed count is41.

Mean-three-order candidate Q is **0.206961554814**, control Q
**0.203608952102**, giving **+0.003352602713** overall. Animal contrasts
are Chewie **+0.005951295867**, Mihili **+0.000753909558**, so the
mean-order directional screen passes. Chewie has4/4 positive session effects
and omission means about+0.00139017 to+0.00759192. Mihili has **only1/4
positive session effects**, with omission means−0.000674772494 to+0.001517819548.
Thus the positive Mihili mean is not a consistently positive session result.
The mean-order session contrasts make the concentration explicit:

| Session | Candidate−control Q |
| --- | ---: |
| Mihili20140217 | −0.001537820410 |
| Mihili20140218 | −0.000285359402 |
| Mihili20140303 | −0.000201137670 |
| Mihili20140304 | +0.005039955715 |
| Chewie20160929 | +0.001139446440 |
| Chewie20161005 | +0.002001633884 |
| Chewie20161006 | +0.019634674138 |
| Chewie20161007 | +0.001029429006 |

Removing Mihili20140304 gives Mihili−0.000674772494; it is the only
positive mean-order session in that animal.

The stronger each-order/both-animal robustness screen **fails**:

| Initializer order | Chewie candidate−control Q | Mihili candidate−control Q |
| --- | ---: | ---: |
| 0 | +0.007628248045 | +0.001556363571 |
| 1 | +0.006146406697 | −0.000268015477 |
| 2 | +0.004079232860 | +0.000973380580 |

Order2 also has negative omission contrasts in each animal. Every overall
omission remains positive for each condition, illustrating why an equal-animal
overall mean alone is insufficient to establish animal/order robustness.
All five mean-three-order seed contrasts are positive in both animal means;
overall seed contrasts span+0.002213210212 to+0.004634906161. These seeds do
not supply independent animal replication or calibrated uncertainty.
Specifically, order2 loses in Chewie after omitting20161006
(−0.000251072263), and in Mihili after omitting20140304 (−0.000350888277).
Order1's Mihili contrast is negative under omissions20140218/20140303/20140304,
and positive only when20140217 is omitted. The old original initializer's
negative Mihili omission after excluding20140303 remains separate context.

Mean-three-order residual prediction beyond D is **+0.018001283899**
overall, Chewie+0.035391883749 and Mihili+0.000610684049. Mihili's minimum
omission is−0.000033816534. No original-versus-new Q improvement is claimed:
new prefixes, training normalizers and conditioned CV differ. No incumbent
promotion or patience reset follows this result.

Root rendered and visually checked the completed-report-only
[initializer comparison PNG](figures/initializer_robustness_attempt001.png),
[PDF](figures/initializer_robustness_attempt001.pdf) and
[50-row CSV](figures/initializer_robustness_attempt001.csv) at07:55UTC: PASS.
Original context is separate from the mean-three summary; each animal's axis
is explicitly labelled and the failed robustness screen is visible. Omission
ranges are fixed-prediction sensitivity summaries, not confidence intervals.

**Scientific recommendation to root:** subject to root's separate reconciliation
of stopping/coverage obligations, stop additional development fitting and
report limited, initializer- and session-dependent correspondence evidence.
The defensible statement is that fixed early residual marginal rankings beat
their matched within-direction target-label controls on average in these eight
development sessions, including the three prespecified initializer conditions.
A stable initializer-independent useful acquisition policy is **not
established**. Neither this contrast nor the earlier selector comparisons
establishes a joint electrode/trial principle, a biological mechanism, improved
deployment utility, or new-animal generalization.

The recommendation is to **keep the four held sessions closed for now**, not
spend the single opening to support an unsupported stronger policy claim.
If root selects a narrower internal held test of target-correspondence
usefulness, it needs a written prospective protocol before access: exact
retained policy/control and initializer conditions; metadata support and
fitting/evaluation assignments for all four sessions; full acquisition,
decoder/tuning and per-order-normalizer rules; one prespecified aggregation,
decision criterion and any meaningful-effect margin; session/animal/order
sensitivity reporting; and an honest terminal claim under failure. The0.001
search resolution is not a meaningful held-effect threshold, and the0.005
optional16-trial criterion cannot be silently transferred to this branch.
No new threshold, held read, source analysis or final study-completion claim
is selected by this recommendation. Root will integrate the literal patience
counter, coverage and remaining obligations and own the final stop/held
decision.

## 2026-10-04 07:54 UTC — `development_search`: three-initializer comparison complete; completed-report analysis selected

Root observed full exit0 completion in46464685 after all eight session groups:
status`completed_two_initializer_robustness_comparisons`,elapsed
288.672735071seconds,119,952KiB peakRSS. The candidate and mean-five-control
panel, each averaged over all three initializer orders, add exactly two valid
comparisons: **41 total**. The released artifact is
[initializer_robustness.json](runs/development_search/initializer_robustness_attempt001/initializer_robustness.json).
No partial order, seed or session was separately charged. No additional
scientific source work is selected and held sessions remain closed.

Before derived-result access, root selected execution of the already reviewed
[completed-report extractor](tools/extract_initializer_robustness.py) and
[figure helper](tools/plot_initializer_robustness.py) in its verified compute
allocation. Both root and independent static reviews passed; they read only
the complete released report, with no source or refit interface. Root will
send derived evidence for interpretation; no new worker-side numeric report
read is needed. Figure creation/viewing and scientific conclusions are not
yet claimed. Full resource details, contrasts and the prospective stop/held
recommendation await that evidence. Reaching41 does not itself grant study
completion or a held-session opening.

## 2026-10-04 07:47 UTC — `development_search`: new compute allocation verified; full robustness dispatch selected

Root observed job46464685 granted at07:47:16UTC with end08:17:21UTC,
normal7502,oneCPU/6GiB. It verified actualPTY40459 on`sh03-01n68.int` and
matching`SLURM_JOB_ID` at07:47:37UTC, with Requeue0/Restarts0. Root selected
the exact frozen600seconds internal/630seconds external full robustness
command recorded below, writing fresh`initializer_robustness_attempt001`,
after numerical-module/thread setup completes. The prior eleven passing
fixtures, ASTRA validation and complete metadata-only role artifact are reused.
This records verified allocation and selected dispatch before the scientific
side effect; actual start is still awaited. No outcome is released before
complete execution. Count remains39 and held sessions remain closed.

Root subsequently dispatched the exact full run at **07:48:08UTC** in46464685.
The first Mihili20140217 condition group completed285 models in32.788970seconds;
its scientific scores remain withheld. This partial progress adds no valid
comparison. Root statically reviewed both completed-report extraction/figure
helpers: PASS. Extraction and rendering will wait for whole-cohort completion;
no independent plotting-review delay is required while that reviewer owns
other scientific checks.

## 2026-10-04 07:44 UTC — `source_support`: separate initializer roles complete; next attended scientific allocation selected

Root observed metadata-role preparation complete in46463321 at approximately
07:44:29UTC: status`completed_initializer_robustness_roles`,24 session/order
roles,`neural_elements_read:false`. The new artifact is
[initializer_robustness_roles.json](runs/source_support/initializer_robustness_roles_attempt001/initializer_robustness_roles.json).
Original roles remain preserved. Eleven fixtures and ASTRA passed as recorded
below; no robustness scientific source read or score has occurred.

Before the next scheduler side effect, root selected a new **30-minute,
one-CPU,6GiB,normal partition,CPU_SKU7502** attended allocation for the exact
600/630-second EP08 run and aggregation, with separately scoped checks/pilot
work if ready. Root's07:42 resource discovery confirmed the partition limits
and feature. Root will finish current rendering, close46463321, request the
new allocation and verify its actual compute hostname/environment before
execution. This is a manual bounded request with sequential work, no overlap
or automatic renewal. No new job ID, grant, start/end or scientific launch is
claimed until observed. Count remains39 and held sessions stay closed.

Root subsequently observed the new request queued at07:45:19UTC as
**job46464685,PTY40459**, with the selected oneCPU/6GiB/30minutes and7502
feature. No scientific launch or compute grant is inferred from that pending
request. Prior46463321 is confirmedCOMPLETED,exit0:0. Its shared allocation
cost is recorded once in root's budget ledger; this EP08 note does not add a
second shared-job charge.

The EP08 owner also owns new completed-report-only extraction and plotting
helpers`tools/extract_initializer_robustness.py` and
`tools/plot_initializer_robustness.py`, to be prepared during execution.
These are analysis helpers, not source readers or fitting inputs; scientific
report access waits for complete-result release by root.
Both helpers are now written and undergoing independent static review. The
extractor verifies complete work and mean-three-Q aggregation; the figure
keeps the original initializer as separate context beside the three new orders
and their mean, with session dots and fixed session-omission ranges. No report
values have been opened and no rendering has occurred. The requested
post-result recommendation will distinguish development stopping from the
prospective claim/margin/held-protocol decision; neither count41 nor these
sensitivity screens automatically authorizes a held opening.

## 2026-10-04 07:43 UTC — `development_search`: robustness code and fixture review passed; checks and metadata-only role preparation selected

Root and independent static reviews passed all four initializer-robustness
modules, including the role-order match and full native-guide checks. Owner
peer review passed all eleven independent
[fixtures](tools/test_initializer_robustness.py), including explicit48/80
prefix/fold corruption rejection beyond the older metadata helper's32/64/96
checks. Formula/maps, fresh true-Y condition fits,285-before-evaluation,
mean-five-SSE then mean-three-Q with unequal denominators, within-order-only
full96 identity and2,280-before-reference parsing are covered. No tests have
yet been executed for this branch.

At07:43UTC, before the side effects, root selected all eleven fixtures,
changed ASTRA validation and the exact fresh metadata-only role preparation
command recorded below in current verified46463321/PTY27650. Its
07:50:17UTC end cannot contain the full630-second scientific reserve. Root
will use a new bounded attended allocation and verify its actual compute
shell before the scientific run. This selected sequence does not claim a
passing test, created role artifact or scientific start. Completed count
remains39; held sessions remain closed.

Root subsequently executed all **eleven fixtures successfully in4.773seconds**
on46463321; changed ASTRA schema/semantics passed without warnings. An initial
test invocation failed to import because root's command replaced the loaded
module's existing`PYTHONPATH` with a tools-only value. Preserving the existing
module path corrected the invocation; no software, implementation or model
change was made, and no scientific attempt ran. Root then dispatched the
already selected90-second-bounded fresh metadata-role preparation. Its
completion and the full scientific run are not yet claimed.

## 2026-10-04 07:37 UTC — `development_search`: initializer robustness adopted; separate-role implementation prepared

Root and independent written scientific reviews passed the
[initializer-robustness proposal](initializer_robustness_proposal.md). Root
adopted the exact three-order/five-label comparison at07:37UTC, retaining the
residual-marginal incumbent and the600seconds internal/630seconds external
full-cohort bound. No old role file is changed. New initializer/acquisition
roles remain entirely in each original96-trial whitelist, with the original40
evaluation identities and ordering preserved. The open-design authority for
new prospective initializer experiments is quoted in the adopted proposal.

New [role preparation](tools/initializer_robustness_roles.py),
[residual selector](tools/initializer_robustness.py),
[experiment/scoring](tools/initializer_robustness_experiment.py) and
[full runner](tools/development_initializer_robustness.py) are written.
The prepared-role validator reconstructs all three seeded queues and all five
32/48/64/80/96 prefix/fold records before neural reads. Main initializer folds
use position-mod4; actual later fitting calls the unchanged qualified
initializer-conditioned CV helper. The runner checks95 models/30 orders per
condition and285 models/90 orders before each session's one evaluation read,
with native-guide equality across the three conditions. Actual decoders use
true Y. Mean-five SSE is formed within an order's common denominator, then
three Qs are averaged. Averaged cells omit nonexistent common SSE and capacity
metadata. Exact full96 equality applies only within each order; original
results are separate context parsed after all2,280 new models.

The [ASTRA program](astra/v0.0.14/astra.yaml) records separate role preparation
as a dependency of the full comparison. The proposed fresh role artifact is
`source_support/initializer_robustness_roles_attempt001`; the scientific
attempt is`development_search/initializer_robustness_attempt001`.
These prepared commands have not yet run:

```bash
timeout 90s python3.12 outputs/tools/initializer_robustness_roles.py \
  --metadata-report outputs/runs/source_support/attempt002/metadata_support.json \
  --trial-roles outputs/runs/source_support/development_trial_roles_attempt001/development_trial_roles.json \
  --output-dir outputs/runs/source_support/initializer_robustness_roles_attempt001

timeout 630s python3.12 outputs/tools/development_initializer_robustness.py \
  --source-root "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
  --metadata-report outputs/runs/source_support/attempt002/metadata_support.json \
  --trial-roles outputs/runs/source_support/development_trial_roles_attempt001/development_trial_roles.json \
  --prepared-roles outputs/runs/source_support/initializer_robustness_roles_attempt001/initializer_robustness_roles.json \
  --reference-report outputs/runs/development_search/initializer_target_label_control_attempt001/initializer_target_label_control.json \
  --max-seconds 600 \
  --output-dir outputs/runs/development_search/initializer_robustness_attempt001
```

Root owns all tests, source execution and scheduling. Independent fixtures and
implementation review are in progress; no fixture/ASTRA pass or new role
artifact is claimed yet. Current parent46463321 ends07:50:17UTC. Its remaining
time cannot be assumed adequate after other work/checks: root must retain the
full630-second launch reserve or choose another bounded attended allocation.
Completed count remains39, and held sessions remain closed.

The stopping audit distinguishes a minimum-count requirement from patience:
there is **no mandatory60-comparison floor**. The historical no-progress
counter is reconstructed separately by root from prospective decisions.
Completing this pair would reach41 but would not itself grant completion.
The six-sign each-order robustness screen is a sensitivity interpretation,
not a new frontier reset or automatic promotion, and first new-order Q is
not directly comparable to incumbent Q under old roles. Remaining obligations
and a prospective held-session decision still require explicit closeout.

## 2026-10-04 07:27 UTC — `development_search`: raw initializer pair complete; residual marginal incumbent retained

Root released the [complete raw-initializer report](runs/development_search/initializer_raw_marginal_attempt001/initializer_raw_marginal.json)
after exit0 in46463321. Actual timing was07:25:13.374390–07:26:50.641198UTC,
97.266822seconds wall,96.943038seconds CPU,95,072KiB peakRSS. The completed
candidate and one mean-five control add two comparisons: **39 valid total**.
All760 actual models,15,200 inner fits,76,000 capacity scores,240 selector
orders,1,600 direction maps,23,040 logical marginal score solves,40 raw
normalization views and40 direction-only fits completed. Selection took
0.284541seconds. Decoder constant-SD occurrences were0 input/1,311 target,
retained with SD-to-one. Exposure was9,984 fitting-read calls,320 evaluation
trials,113,959,680 logical and30,094,080 unique native bytes. Report size is
5,764,025bytes. Both score/direction-null replay differences are zero; held
access remains false. No old panel was refitted or recharged.

Raw candidate Q is **0.205697477073**, own mean-five control Q
**0.201680704452**. Their contrast is **+0.004016772621** overall,
Chewie **+0.007383893948**, Mihili **+0.000649651295**. Three of four
sessions in each animal are positive, and the own-control screen passes.
Every Mihili omission remains positive, but omitting Chewie20161006 reverses
Chewie to−0.000187249950. All five seed contrasts are positive overall and
in Chewie; Mihili's seed2 contrast is−0.000308357736. This supports useful
within-direction pairing in the raw ranking with an important session limit;
the control retains direction association and is not a signal-free null.

The incumbent challenge is **+0.000664022442** overall, Chewie
**+0.001724452071**, Mihili **−0.000396407187**, so the both-animal
challenger screen fails. Two Mihili and three Chewie sessions are positive.
All overall omissions and Chewie omissions stay positive. Only omitting
Mihili20140303 flips Mihili positive, to+0.000752312637; its other omissions
remain negative. Root explicitly keeps the residual-marginal incumbent: the
raw branch fails the directional screen and its overall gain is below the
existing0.001-Q search-progress resolution. No automatic promotion occurs.

Raw residual prediction beyond D is **+0.021093150982**, positive in both
animal means under every omission. Pairing identity increment is
0.037759570171. True raw scores range approximately0.0351633–0.162361 and
are all positive; the40 true selector views have0 input and24 target zero-SD
column occurrences. No clipping was used. Keeping direction-related
association and changing X scaling are one preprocessing comparison, not an
isolated causal direction test or new named explanation.

Root requested the [prospective initializer-robustness draft](initializer_robustness_proposal.md) for the fixed
residual marginal candidate and matched target-label controls, using three
prespecified initializer/order replicates strictly inside the existing96-row
fitting whitelist and the unchanged40-row evaluation set. New role artifacts
must be separate from original frozen roles. Differing prefixes and
initializer-conditioned CV prohibit assuming historical full96 score replay,
even when the final training set is all96 rows; exact full96 equality applies
only within each new matched-order candidate/control group. This successor
remains unadopted and unimplemented pending written scientific reviews. The
written draft now freezes all three ordinary12-ID direction queue draws,
separate role artifacts, original initializer fold rule and later conditioned
CV. It requires285 models/session before the common evaluation read and
averages seed SSE within order, then Q across orders rather than pooling
different denominators. Open-design authority and the prior fixed-initializer
arrival-order comparison are quoted explicitly. Proposed full-cohort bounds
are600seconds internal/630seconds external; no run is selected by this draft.

## 2026-10-04 07:25 UTC — `development_search`: raw initializer checks passed and full comparison launched

Root executed all ten independent raw-initializer fixtures in verified
PTY27650/job46463321 on`sh03-01n68.int`: **PASS,1.409seconds**. Changed ASTRA
schema and semantics passed without warnings; evidence is
`astra/v0.0.14/validation_20261004_initializer_raw_marginal.json`.
Root dispatched the exact adopted full-eight-session command at
**07:25:13UTC**, with270seconds internal/300seconds external and more than
25minutes remaining before parent end07:50:17UTC. Output is the fresh
`runs/development_search/initializer_raw_marginal_attempt001`.

This records actual checks and dispatch only. The run is not yet complete,
scientific scores remain unreleased, and completed comparison count stays37.
Root owns process supervision and complete-report extraction. No additional
code or next scientific comparison is activated.

## 2026-10-04 07:18 UTC — `development_search`: direction-inclusive initializer ranking adopted; implementation and selected full-cohort recipe ready

Root and independent scientific review passed the
[raw-initializer proposal](initializer_raw_marginal_proposal.md), and root
adopted it at07:18UTC. The earlier raw method already included32-row selection
but reranked at32/64/96 with all electrodes visible and ordinary raw-target
heads. This comparison freezes the24/32-view initializer ordering through
later acquired-only histories and keeps the current residual reduced-rank
decoder. Global input scaling changes along with direction-inclusive
centering; five exact original within-direction maps preserve task means.
There are25 view/seed map sets, or200 direction-level maps, per session.

New [core](tools/initializer_raw_marginal.py),
[experiment](tools/initializer_raw_marginal_experiment.py) and
[full-cohort runner](tools/development_initializer_raw_marginal.py) preserve
all earlier files. Every decoder sees true targets. All30 orders precede
future source rows,95 distinct models precede each evaluation, and the sole
historical target-label report is read after all760 new models. Raw selector
normalization and23,040 logical marginal score solves have separate counters.
Legacy marginal cells' absent SSE fields remain explicit; recorded control
denominators are required. Root and independent preliminary static reviews
passed math/timing. Review identified and implementation tightened exact
within-new-run full96 score equality and direction-null agreement across
every old candidate/seed; historical full96 replay retains1e-10 tolerance.
The explicit computed-mean/SD finite check also prevents overflow from silently
becoming finite normalized zeros. Independent final static review passed all
three modules, and owner peer review passed all ten independently written
[fixtures](tools/test_initializer_raw_marginal.py). These include direct raw
formula equivalence, retained direction association, exact maps, main-view
exclusion, overflow, true-Y timing, mean-SSE aggregation, legacy/reference
checks and deferred reads. Root subsequently completed static review of both
reference corrections and the computed-normalizer finite guard: PASS. Before
execution, root selected this sequence in verifiedPTY27650/job46463321:
immediately after its currently running scoped job finishes, execute all ten
focused fixtures and validate the changed ASTRA program; after both pass,
launch the exact adopted full-eight command below into the fresh attempt001.
The observed parent end remains07:50:17UTC, with more than26minutes remaining
at selection. Root must preserve the full300-second launch reserve. These
are selected actions only; no test result or raw launch is claimed yet.
Root owns the separate completed-report extractor
`tools/extract_initializer_raw_marginal.py`; it is not a fitting input.

Root selected the eight-session branch at270seconds internal/300seconds
external, one numerical worker within4GiB branch memory, based on multiple
completed equivalent95-model/session workloads. Fresh output is
`initializer_raw_marginal_attempt001`; launch requires passing checks and the
full reserve in root's verified compute shell. Prior allocation46460577 was
closed07:18UTC. New request46463321,rootPTY27650, was queued07:18:48UTC for
oneCPU/6GiB/30minutes on normal7502. Root subsequently verified its grant on
`sh03-01n68.int`,start07:20:12UTC,end07:50:17UTC,Requeue0 and numerical threads
one. Root dispatched its other scoped remaining work at07:21:08UTC; raw
initializer checks/run follow its completion with the full300-second reserve.
Root owns all scheduler and execution actions. This is the selected command,
not an observed raw-initializer launch:

```bash
timeout 300s python3.12 outputs/tools/development_initializer_raw_marginal.py \
  --source-root "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
  --metadata-report outputs/runs/source_support/attempt002/metadata_support.json \
  --trial-roles outputs/runs/source_support/development_trial_roles_attempt001/development_trial_roles.json \
  --reference-report outputs/runs/development_search/initializer_target_label_control_attempt001/initializer_target_label_control.json \
  --max-seconds 270 \
  --output-dir outputs/runs/development_search/initializer_raw_marginal_attempt001
```

The [ASTRA program](astra/v0.0.14/astra.yaml) records this exact dependency and
command. Completed count stays37; only full candidate plus mean-five-control
completion can add two, reaching39. Held sessions and named explanatory
endpoints remain closed. No new source values or scientific scores were read
during proposal, implementation or static review.

## 2026-10-04 07:12 UTC — `development_search`: internally validated utility pair complete; both prospective screens fail

Root released the [complete CV-utility report](runs/development_search/initializer_cv_utility_attempt001/initializer_cv_utility.json)
after remaining-phase exit0 in46460577,07:10:32.504273–07:12:00.792907UTC.
The actual start resolves the earlier approximate07:10:30 dispatch record.
Remaining work took88.288653seconds wall,86.467870seconds CPU and101,980KiB
peakRSS. Including the reused pilot, total component runtime is99.593394seconds
wall and97.735878seconds CPU. The pilot was not refitted or charged separately.
Full completion adds the candidate and one mean-five control: **37 valid
comparisons total**. Held-session access remains false.

The pair completed760 actual target fits,15,200 actual inner ridge fits,
76,000 capacity scores and40 direction-only models. Separate selection work
was73,728 logical single-electrode proxy models,128 fitting-normalization
views,240 main orders and5,120 internal-fitting direction maps; selection
time was1.083486seconds. Zero input and1,311 target constant-SD fit occurrences
retained the declared SD-to-one rule. Combined exposure is9,984 fitting-read
calls and320 evaluation trials,113,959,680 logical bytes and30,094,080 unique
bytes. Main report size is10,807,538bytes plus the preserved899,987-byte pilot.
Full96 and direction-null reference replay differences are zero. No old
reference was refitted or charged; new fit-only maps were never claimed
identical to the older whole-main-view null.

Candidate Q is **0.202768202417**, compared with its own mean-five control
Q **0.200031696190**. Their contrast is **+0.002736506227** overall,
Chewie **+0.006802474956**, Mihili **−0.001329462503**, so the
correspondence screen fails. Three Chewie and two Mihili sessions are positive.
All overall session omissions remain positive, but only omitting
Mihili20140304 makes Mihili positive, at+0.000427953754; its other omissions
remain negative. All five seed contrasts are positive overall and in Chewie;
Mihili is positive only against seed1.

The marginal-incumbent challenge also fails: **−0.002265252214** overall,
Chewie **−0.000870585957**, Mihili **−0.003659918472**. Two Mihili
and one Chewie sessions are positive. Every overall omission remains negative
and every Mihili omission remains negative. Omitting Chewie20161006 makes
Chewie+0.000308786284, without reversing the overall loss. This fixed
internally validated ranking does not improve on the incumbent under the
declared matched decoder/exposure comparison.

Residual prediction beyond direction means is **+0.018163876325** overall,
Chewie+0.038423206952 and Mihili−0.002095454301. Omitting Mihili20140304
makes Mihili+0.002439982466. Internal utility diagnostics range approximately
−0.722527871 to+0.081616892. Each of the40 true selector views has62–96 of
96 electrodes with negative estimated validation benefit; its selected32
prefix contains0–32 negative utilities. These scores were retained without
clipping, exactly as specified. Small16/24-row proxy fits can produce noisy
utility estimates; this failure does not prove that the incumbent has no
fitting optimism or that these electrodes carry no jointly decodable signal.

All actual heads used true targets, with fit-only proxy normalization and
controls restricted to internal fitting rows. The result remains an exposed
development policy comparison, not a new removal-value or causal explanatory
test. The [next unadopted proposal](initializer_raw_marginal_proposal.md)
checks direction-inclusive raw marginal ranking under the same32-row acquired
initializer and unchanged residual decoder. Static inspection of the original
selector code/proposal confirms an important overlap: raw marginal ranking
already ran at a32-row prefix, as well as64/96. It was reselected at each
prefix with all96 electrodes available and used ordinary raw-target ridge
heads. The new question is the permanently frozen acquired-only ranking with
the residual reduced-rank head. Its global-versus-residual input scaling
change is explicit; original within-direction label maps retain direction
association. Root and independent scientific review are pending. No code or
new source read is activated.

## 2026-10-04 07:09 UTC — `development_search`: CV-utility pilot complete; finite remaining-seven bound selected from resources only

Root observed the predetermined [Mihili20140217 pilot](runs/development_search/initializer_cv_utility_pilot_attempt001/initializer_cv_utility.json)
start at07:08:58.418450UTC and complete in11.304741seconds, with68,908KiB
peakRSS. Its status is `completed_initializer_cv_utility_pilot_scores_withheld`,
scientific-score release false and comparison charge zero. No pilot scientific
scores or historical references were opened for interpretation.
The actual finish was07:09:09.723175UTC, with11.268008740seconds CPU.
The pilot completed9,216 proxy models,16 normalization views,640 maps,
30 orders,95 actual models,1,900 actual inner fits,9,500 capacity scores and
five null models. Selector time was0.127116120seconds. Zero input and19 target
constant-SD fit occurrences were retained. Exposure was1,248 fitting-read
calls and40 evaluation trials,12,828,160 logical versus3,612,160 unique bytes;
retained pilot size is899,987bytes. Held-session access is false.

Root selected **600seconds internal/630seconds external** for the seven
remaining sessions from this resource evidence: seven times11.305seconds is
approximately79seconds, with a further300-second conservative session
reserve and221seconds for variation. This is a finite pre-score bound within
the adopted ceiling, not a running-deadline or shared-budget extension.
Current allocation46460577 ends07:22:53UTC and had more than12minutes
remaining at selection; root must retain the full630-second reserve before
launch. The one-CPU branch remains inside the actual6GiB parent reserve.

The selected fresh output is`initializer_cv_utility_attempt001`. The completed
pilot is parsed only after all seven new sessions finish, validated and reused
once without refit. Its scores remain withheld until the complete cohort's
declared comparisons and replays finish. The changed ASTRA recipe is written
before the side effect; root will validate and launch this exact command:

```bash
timeout 630s python3.12 outputs/tools/development_initializer_cv_utility.py \
  --mode remaining \
  --source-root "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
  --metadata-report outputs/runs/source_support/attempt002/metadata_support.json \
  --trial-roles outputs/runs/source_support/development_trial_roles_attempt001/development_trial_roles.json \
  --pilot-report outputs/runs/development_search/initializer_cv_utility_pilot_attempt001/initializer_cv_utility.json \
  --reference-report outputs/runs/development_search/initializer_repeatability_attempt001/initializer_repeatability.json \
  --max-seconds 600 \
  --output-dir outputs/runs/development_search/initializer_cv_utility_attempt001
```

This records the selected remaining attempt, not an observed start or new
scientific result. The count remains35 and all held sessions remain closed.
Root subsequently observed changed ASTRA schema/semantics pass, evidence
`astra/v0.0.14/validation_20261004_initializer_cv_utility_remaining.json`,
and dispatched the exact remaining command at **07:10:30UTC** in46460577.
More than12minutes remained, exceeding the630-second external bound. The
runner was unchanged; scientific extraction waits for full completion.

## 2026-10-04 07:01 UTC — `development_search`: internal-CV electrode utility adopted; complete resource pilot selected before scores

Root adopted the [CV-utility recipe](initializer_cv_utility_proposal.md)
after root and independent written-proposal review passed. Every original
24/32-row selection view contains its own balanced internal3/4-fold CV,
with16/24 fitting and8 validation rows. Fitting-only X/Y normalizers and
alpha1 three-coordinate proxy heads produce signed validation-SSE benefit
without an optimistic zero floor. Five new ordinary target-vector maps are
restricted to internal fitting directions; true validation and all actual
decoder targets remain unchanged. The new null is explicitly distinct from
the older whole-main-view target-label controls.

New core,95-model orchestration and [pilot/remaining runner](tools/development_initializer_cv_utility.py)
are written without modifying older scientific modules. All30 orders precede
future rows and95 actual models precede evaluation/session. Internal proxy
counts are logical fitted electrode models, separate from actual decoder
inner/outer work; batched solves do not erase those costs. Completed pilot
scores and the historical repeatability reference are parsed by the remaining
runner only after all seven new sessions finish. Exact pilot recipe/counts/
roles and information barriers are required before reuse.

Root selected the complete first Mihili20140217 resource pilot on one CPU,
270seconds internal/300seconds external, fresh attempt below. Current attended
allocation46460577 is one CPU/6GiB and ends07:22:53UTC; the branch estimate is
within4GiB. Focused fixtures, static implementation review and changed ASTRA
validation precede root's actual launch. No source reads or tests were run
on the login node, and no pilot start is inferred here.

```bash
timeout 300s python3.12 outputs/tools/development_initializer_cv_utility.py \
  --mode pilot \
  --source-root "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
  --metadata-report outputs/runs/source_support/attempt002/metadata_support.json \
  --trial-roles outputs/runs/source_support/development_trial_roles_attempt001/development_trial_roles.json \
  --max-seconds 270 \
  --output-dir outputs/runs/development_search/initializer_cv_utility_pilot_attempt001
```

The pilot must complete95 actual models,1,900 actual inner fits,9,500 capacity
scores,9,216 proxy models,16 normalization views,30 orders,640 direction maps,
five nulls,1,248 fitting-read calls and40 evaluation trials. Its scientific
scores stay closed and it charges no comparison. Root chooses a finite
remaining-seven bound from resource-only pilot evidence before those scores
are released, within the adopted600/630-second ceiling and actual allocation
reserve. No remaining recipe or launch is selected yet. Full completion would
add two comparisons,35to37; current count remains35 and held/named endpoints
remain closed.

Root and independent static reviews subsequently passed the three new
modules, including the fitting-only raw-Y SD and residual-intercept handling.
All eleven focused fixtures passed in1.119seconds in verified allocation
46460577 on`sh03-01n68`. Changed ASTRA schema/semantics passed, evidence
`astra/v0.0.14/validation_20261004_initializer_cv_utility.json`.
Root dispatched the exact selected pilot at **07:08:58UTC** in attended
PTY69705. No scientific scores were opened; completion/resource observations
are pending and the completed-comparison count remains35.

## 2026-10-04 06:55 UTC — `development_search`: target-free logdet selector complete; both selection screens fail

Root released the [complete logdet report](runs/development_search/initializer_logdet_attempt001/initializer_logdet.json)
after exit0 in job46460577 on`sh03-01n68`. The run lasted
06:54:31.753905–06:55:04.678304UTC:32.924413seconds wall,32.852439seconds
process CPU and93,216KiB peakRSS. All nine focused fixtures passed in2.110seconds;
changed ASTRA schema/semantics passed, evidence
`astra/v0.0.14/validation_20261004_initializer_logdet.json`.
The completed candidate charges one comparison, moving34to**35 valid comparisons**.
All held sessions and named explanatory endpoints remain closed.

The run completed160 target fits,3,200 inner ridge fits,16,000 capacity scores,
40 selector orders,1,280 greedy selections and40 direction-null models.
Selector time was4.63894seconds. Source accounting is2,304 fitting-read
calls,320 evaluation trials,44,071,680 logical bytes and30,094,080 unique
bytes; retained report size is4,117,563bytes. There were zero input and276
target constant-SD fit occurrences, retained under the SD-to-one convention.
All40 selector views had no constant X columns, no exact-best ties beyond
one candidate and no numerical-negative gains; selected gains spanned
approximately0.4185956–2.07934297. Both historical full96 and direction-null
replays are exact. No prior reference was refitted or recharged.

Logdet Q is **0.197742382877**. Its matched-uniform contrast is
**−0.001800324348** overall, Chewie **−0.003688513279**, Mihili
**+0.000087864584**. Three Mihili and one Chewie sessions are positive;
every session omission preserves the negative Chewie and positive Mihili
animal-mean signs. The target-free-selection screen therefore fails. Against
uniform seed1 the candidate is positive in both animals, while its overall
contrast is negative against each of the other four seeds. Seed variation
does not replace the specified mean-five reference.

The incumbent challenge also fails: logdet-minus-marginal is
**−0.007291071754** overall, Chewie **−0.012846718908**, Mihili
**−0.001735424600**. Only one session per animal is positive; both animal
means remain negative under every omission. Compared with initializer
repeatability, logdet is **−0.004112199716** overall and negative in both
animal means; all Chewie omissions remain negative while Mihili's sign is
omission-sensitive. Thus residual-input volume diversity does not substitute
successfully for the observed target-linked marginal ranking under this recipe.

Residual prediction over direction means is **+0.013138056786** overall,
Chewie+0.026447074001 and Mihili−0.000170960429. Omitting Mihili20140304
makes Mihili approximately+0.001650048; its other omissions remain negative.
This is a weak and variable Mihili residual-prediction signal, not a successful
target-free selection explanation or evidence of biological complementarity.

The next proposal tests fitting optimism directly: rank single electrodes
using predictive utility from balanced validation folds strictly inside each
original24/32-row selection view. It will specify the score, fitting-only
normalizers and matched controls before review; no implementation, new
source access or named explanation endpoint is activated by this closeout.
The written [CV-utility proposal](initializer_cv_utility_proposal.md) specifies
16/24-row internal fits and8-row validation folds, signed standardized-SSE
benefit without clipping, fit-only whole-target maps and true validation Y.
It separates73,728 proxy fits from760 actual decoder fits and proposes a
predetermined complete first-session resource pilot. It remains unadopted;
no code, ASTRA recipe or new read is activated.

## 2026-10-04 06:47 UTC — `development_search`: target-free initializer logdet selector adopted; fixed implementation awaits tests and launch selection

Root adopted the [initializer logdet recipe](initializer_logdet_proposal.md)
after root and independent scientific review passed. Each original24/32-row
view uses only X to compute direction-residual means/population scales and
exactly32 alpha1 group-logdet choices. The implementation preserves exact
physical-ID ties, retained roundoff-negative gains and a declared technical
failure threshold, with no clipping or early stopping. Target values are
absent from the selector API but remain acquired and used by every decoder.

The new pure core, orchestration and [runner](tools/development_initializer_logdet.py)
reuse existing three-band reader, selected-rank fitter and later conditioned
fits without changing their files. Five orders precede future acquisition;
twenty models precede evaluation/session. All160 new fits complete before
reading the prior repeatability report's matched uniform, repeatability and
marginal panels. Exact roles/exposure and recorded target/full96 replay are
required; the legacy marginal missing-SSE limitation is explicit. No reference
fit or evaluation is charged again. A completed candidate adds one comparison,
34to35; current completed count remains34 and all held/named endpoints stay closed.

The adopted full-cohort bound is180seconds internal/210seconds external,
one CPU within4GiB. Root closed allocation46458807 at06:47UTC with no active
compute. A new attended one-CPU/6GiB/30-minute allocation is planned by root;
no job ID, compute node or start/end is inferred here. Focused fixtures,
ASTRA validation and root's actual attempt selection follow in verified
compute. The proposed fresh attempt command is:

```bash
timeout 210s python3.12 outputs/tools/development_initializer_logdet.py \
  --source-root "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
  --metadata-report outputs/runs/source_support/attempt002/metadata_support.json \
  --trial-roles outputs/runs/source_support/development_trial_roles_attempt001/development_trial_roles.json \
  --reference-report outputs/runs/development_search/initializer_repeatability_attempt001/initializer_repeatability.json \
  --max-seconds 180 \
  --output-dir outputs/runs/development_search/initializer_logdet_attempt001
```

This is the fixed recipe and proposed output path, not an observed launch or
new result. No source read or tests were performed by the login-node worker.

## 2026-10-04 06:42 UTC — `development_search`: high-power-only pair complete after finalization-only repair;34 valid comparisons

Root released [finalized attempt002](runs/development_search/high_power_correspondence_attempt002/high_power_correspondence.json)
after exit0 at06:42:53.769469UTC in allocation46458807. Root and independent
repair review passed; all thirteen fixtures passed in1.595seconds, including
the authentic legacy missing-field and no-source finalizer regressions.
Changed ASTRA schema/semantics passed without warnings, evidence
`astra/v0.0.14/validation_20261004_high_power_finalize.json`. The finalizer
took0.415660seconds wall,0.411558seconds CPU and76,424KiB peakRSS with
**zero new neural reads and zero new fits**. Its6,722,818-byte report preserves
the original95.225483-second/88.610553-CPU-second fit cost and references
the4,991,027-byte failed post-fit attempt. Nothing was overwritten or refitted.

The completed pair retains760 target fits,15,200 inner fits,76,000 capacity
scores,240 orders,1,600 maps,40 null models,9,984 fitting-read calls and320
evaluation trials. Original source exposure is91,841,280 logical bytes and
21,738,240 unique bytes; original peakRSS is100,412KiB. Zero input and1,311
target constant-SD fit occurrences retained the declared SD-to-one convention.
Both within-run full96 and historical direction-null replay differences are
zero. Legacy marginal SSE/denominator replay is explicitly not claimed;
those values were never serialized. Required denominator replay against the
original label-control and both LMP panels passed. Only final completion
charges the candidate/control pair, moving32to**34 valid comparisons**.
All four held sessions and named explanation endpoints remain closed.

High-power-only true marginal selection has Q **0.197552291224**, compared
with its own mean-five control Q **0.191158025310**: **+0.006394265915**
overall, Chewie **+0.012108453806**, Mihili **+0.000680078023**. The
prospective correspondence screen passes. Four Chewie and three Mihili
sessions are positive; every fixed-prediction session omission keeps both
animal means positive. Every seed contrast is positive overall and in
Chewie, while Mihili is negative against seed2 at approximately−0.000713438180.
This supports useful initializer correspondence under this two-band recipe,
with a much smaller Mihili effect and remaining seed sensitivity.

Residual prediction over direction means is **+0.012947965133** overall,
Chewie+0.025129717315 and Mihili+0.000766212952. Its separate full-mean screen
also passes, but omitting Mihili20140303 makes Mihili−0.000200728426.
High-power selection loses to the three-band model by **−0.007481163406**,
Chewie−0.014164075594 and Mihili−0.000798251219. All overall omissions
retain this loss; omitting Chewie20161006 makes Chewie's own difference
+0.000328601295. Against LMP-only, the difference is **−0.001780533493**,
Chewie−0.003777788622 and Mihili+0.000216721635. Omitting Chewie20161006
reverses that overall contrast to+0.003592316793, so the spectral ranking
is influential-session sensitive.

The high-power true-minus-own-control contrast exceeds the three-band
contrast by **+0.001013452474**, positive in both animal means. This larger
ablation contrast coexists with lower true prediction Q; it does not imply
greater total information or a physiological mechanism. Actual inputs are
2E versus3E or1E, changing both available features and regularized capacity.
The result neither isolates spike contamination nor establishes independent
band contributions, biological complementarity or a joint acquisition rule.

The next proposed comparison is target-free residual-input diversity under
the same original initializer and three-band decoder. It must pass scientific
review before code activation; no HF score is used to alter its fixed
objective or tune a proxy hyperparameter.

Root rendered and visually checked the [spectral comparison figure](figures/high_power_correspondence_attempt001.png),
[PDF](figures/high_power_correspondence_attempt001.pdf) and
[210-row CSV](figures/high_power_correspondence_attempt001.csv). Their fresh
figure prefix is attempt001, while the data source is finalized attempt002.
The independent plotter separates prediction Q from each spectrum's own
control contrast, displays session points and fixed-omission ranges, and
states that those ranges are not confidence intervals.

The [target-free residual-logdet proposal](initializer_logdet_proposal.md)
is ready for scientific review and remains unadopted. It uses fitting-only
X normalization and fixed32 greedy group-volume gains, with true-target
heads and reused matched uniform/repeatability/marginal references. It adds
one candidate only if completed, without recharging references or opening
the named explanatory endpoints.

Optional BR read-only retrieval at06:44UTC asked for sensor-selection
log-determinant evidence, Publication limit3. Results were MEG autoregression
DOI10.1002/hbm.21482, hippocampal recollection DOI10.1073/pnas.0908651106 and
wine/alcohol PMID25785844. None supplied relevant M1 selector evidence;
the retrieval did not endorse the method or change policy, and no retry is
needed for the independently specified prospective comparison.

## 2026-10-04 06:36 UTC — `development_search`: all HF fits complete; repair only known historical schema and finalize saved scores

Root observed [attempt001](runs/development_search/high_power_correspondence_attempt001/high_power_correspondence.json)
run06:33:02.451363–06:34:37.676832UTC in allocation46458807. It completed all
eight sessions,760 target fits,15,200 inner fits,76,000 capacity scores,
240 orderings,1,600 direction maps and40 null models before failing in
`attach_references` with `KeyError: 'null_SSE'`. This resolves the earlier
approximate06:32 launch record with actual run timestamps. The failure is
post-fit reference finalization, not a source, model or scheduler failure.
Root has not released scientific scores; count remains32 and held stays closed.

The preserved fit work cost95.225483seconds wall,88.610553seconds CPU and
100,412KiB peakRSS, with9,984 fitting-trial read calls,320 evaluation trials,
91,841,280 logical bytes,21,738,240 unique bytes and4,991,027 retained report
bytes. There were zero input and1,311 target constant-SD fit occurrences.
All these fits and saved scores are reused without recomputation.

Code diagnosis and root's compute-side key-only inspection agree: only the
legacy `initial_marginal` cells lack the denominator and standardized-SSE
fields. Their Q, wrong-pair Q, pairing increment, direction-null Q and
physical/numeric/target costs are present in both embedded packets. The
original target-label control and both LMP panels contain the required SSE
fields. The prior synthetic fixture had given the old marginal cells a
newer schema, so it missed this concrete compatibility defect.

Root approved the precise repair at06:36UTC. The corrected reference reader
compares every originally recorded cell field and fitted-score record where
present, and explicitly requires all base fields. Only the known legacy
marginal schema may omit all three saved SSE fields; control/LMP denominators
and SSE remain mandatory. Every new denominator is checked against those
three recorded historical panels, while direction-null and exposure checks
still cover every spectrum. No absent legacy denominator or SSE replay is
claimed or fabricated. Exact map, role, within-run full96 and twenty-cell
checks remain active.

The [new finalizer](tools/finalize_high_power_correspondence.py) requires the
exact post-fit failure, complete counts, original eight sessions, full role
records, all95-model/30-order barrier flags and saved seed panels before
reuse. It has no source-path argument or neural-read/fitting call. Original
fit costs and failure artifact remain intact; finalization has its own cost
record. The unchanged scoring summaries are shared with the main runner.
Focused legacy-schema and no-source finalizer regressions are pending root
execution, along with changed ASTRA validation. The selected repair command is:

```bash
timeout 90s python3.12 outputs/tools/finalize_high_power_correspondence.py \
  --completed-fit-report outputs/runs/development_search/high_power_correspondence_attempt001/high_power_correspondence.json \
  --reference-report outputs/runs/development_search/initializer_target_label_control_attempt001/initializer_target_label_control.json \
  --lmp-reference outputs/runs/development_search/lmp_correspondence_attempt001/lmp_correspondence.json \
  --max-seconds 60 \
  --output-dir outputs/runs/development_search/high_power_correspondence_attempt002
```

This fresh finalization attempt charges zero new target fits or neural reads;
the candidate/control pair charges two evaluations only when complete.
No finalizer launch or repaired scientific result is claimed yet.

## 2026-10-04 06:32 UTC — `development_search`: high-power-only full cohort launching

The EP05 parent returned exit1 after cleanly settling its active wave; no
EP05 worker remains active. Root now launches the already reviewed and tested
high-power-only full cohort in46458807, one numerical worker,270seconds
internal/300seconds external, into fresh
`runs/development_search/high_power_correspondence_attempt001`. Allocation
end06:51:41UTC leaves ample full-run reserve. All eleven focused fixtures and
changed ASTRA validation passed before this source action. Count32 remains
until all eight sessions and declared reference replays complete.

## 2026-10-04 06:18 UTC — `development_search`: native high-power-only comparison adopted; selected full run awaits checks and allocation

Root adopted the [high-power-only recipe](high_power_correspondence_proposal.md)
after root and independent scientific review passed without amendments.
This is a distinct spectral sensitivity: native100–200/200–400Hz values
without LMP, two actual coordinates per electrode, two-coordinate alpha1
marginal gain and the original five target-label maps. Decoders retain true
targets, actual2E ranks and fixed selected sets. All30 orders precede future
reads, and95 distinct models precede evaluation in every session. Old
three-band and LMP-only reports are parsed only after all new fits; target
quantities replay across spectra, while full96 capacity/scores replay within
the new run. Changes in spectral information and capacity are explicit.

The new reader, trajectory, selector, selected-rank fitter, orchestration and
[runner](tools/development_high_power_correspondence.py) preserve all old
implementation files. Focused fixture execution and changed ASTRA validation
remain pending in coordinator-owned compute. No tests or scientific source
reads were performed on the login node. The count remains **32 valid
comparisons**, and all four held sessions remain closed.
Root's subsequent static review passed for all five new modules, including
exact selected guide rows, actual2E indexing/ranks and20-value exposure,
fitting-only2x2 objective, true-target heads and both information barriers.
Independent static review also passed all five modules. The independent
[eleven focused fixtures](tools/test_high_power_correspondence.py) are ready
and passed a separate static review; they remain unexecuted. They cover
native numerical-row selection, actual dtype exposure, immutable two-band
trajectory, direct correlated2x2 objective and exact maps, actual2E ranks,
inner-view exclusion, true targets and both information barriers, mean-five
SSE, appropriate historical/within-run replay, and deferred reference reads.

Root selected a fresh full-cohort attempt,270seconds internal/300seconds
external, one CPU within the branch's4GiB estimate. Existing comparable
full-cohort runs took approximately87–107seconds. Root closed prior attended
allocation46457171 at06:18:07UTC and requested bounded allocation46458807,
four CPUs/12GiB/30minutes, no requeue. Its latest reported state is pending
resources; no start, end or compute hostname is inferred. The single HF
worker follows EP05's four-worker phase and never adds a fifth worker.
Root will verify actual remaining reserve before launching the exact command:

```bash
timeout 300s python3.12 outputs/tools/development_high_power_correspondence.py \
  --source-root "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
  --metadata-report outputs/runs/source_support/attempt002/metadata_support.json \
  --trial-roles outputs/runs/source_support/development_trial_roles_attempt001/development_trial_roles.json \
  --reference-report outputs/runs/development_search/initializer_target_label_control_attempt001/initializer_target_label_control.json \
  --lmp-reference outputs/runs/development_search/lmp_correspondence_attempt001/lmp_correspondence.json \
  --max-seconds 270 \
  --output-dir outputs/runs/development_search/high_power_correspondence_attempt001
```

The ASTRA branch requests one CPU/4GiB/five minutes; the prospective parent
allocation reserves four CPUs/12GiB for the sequential EP05/HF workload.
This records the selected attempt, not a launch, result or count increment.

Root subsequently observed actual allocation46458807 on`sh03-01n05.int`,
four CPUs/12GiB, no requeue,06:21:37–06:51:41UTC, verified in the execution
shell with hostname, environment and allocation limits. All eleven focused
HF fixtures passed there in0.777seconds. The changed ASTRA schema and
semantics passed without warnings, evidence
`astra/v0.0.14/validation_20261004_high_power_correspondence.json`.
EP05's four-worker phase started around06:23UTC; HF has not launched.
Its one-worker run follows only after that parent and all children finish
and root verifies at least300seconds of allocation reserve remain.

## 2026-10-04 06:11 UTC — `development_search`: conditional-gain pair complete; correspondence screen passes but incumbent challenge fails

Root released the [complete conditional-gain report](runs/development_search/initializer_conditional_attempt001/initializer_conditional_gain.json)
after remaining-phase exit0 in allocation46457171. That phase ran from
06:10:26.511606 to06:11:57.554902UTC,91.043310seconds wall and90.777167seconds
process CPU, with103,632KiB peakRSS. Including the preserved first-session
pilot, total component runtime was102.968421seconds wall and102.663921seconds
CPU. All760 models,15,200 inner fits,76,000 capacity scores,240 orderings,
7,680 greedy choices,1,600 direction maps and40 direction-only models
completed. There were zero constant-input fit occurrences and1,311 retained
constant-target occurrences under the existing zero-SD-to-one convention.

Combined exposure is9,984 fitting-trial views and320 evaluation trials,
113,959,680 logical bytes versus30,094,080 unique bytes. The complete report
is7,230,655bytes, with the579,453-byte pilot preserved separately; its cost
and evidence were reused without refit. Both historical/within-run full96
and direction-null replay differences are zero. Both full comparisons add
two evaluations: **32 valid comparisons total**, held-session access false.

Conditional selection has Q **0.202159022631**, compared with its own
mean-five target-label control Q **0.199744097712**, a contrast of
**+0.002414924919**. Chewie is **+0.004267958334** and Mihili
**+0.000561891504**; the prospective correspondence screen passes. Four
Chewie and three Mihili sessions are positive. Every fixed-prediction
omission preserves positive means in both animals, although omitting
Chewie20161006 reduces the overall contrast to approximately+0.000967830030
and Chewie to+0.001373768560. Every control seed loses overall and in
Chewie, while Mihili loses against seed2. This supports useful initializer
target correspondence under the fixed conditional proxy, with modest Mihili
effect size and remaining seed sensitivity.

The incumbent challenge fails. Conditional selection loses to true initial
marginal selection by **−0.002874432000** overall, Chewie
**−0.005648530074** and Mihili **−0.000100333926**. Two of four
sessions in each animal favor conditional selection; every overall omission
remains negative. All Chewie omissions remain negative. Omitting
Mihili20140303 makes Mihili positive at approximately+0.001026701680, while
its other omissions remain negative. The result therefore does not show
that this greedy conditioning improves the practical frozen-set policy.

Residual prediction beyond direction means is **+0.017554696540** overall,
positive in both animal means under every omission. Conditional selection
also exceeds the matched uniform-order panel by **+0.002616315406** overall
with both animal means positive, but omitting Chewie20161006 makes Chewie's
uniform contrast−0.000571433131. Against initializer repeatability, its
overall advantage is only **+0.000304440038**, with Chewie
−0.000969244607 and Mihili+0.001578124683; omitting either Mihili20140304 or
Chewie20161006 reverses that overall contrast. These references were reused
descriptively without new fits or charges and do not replace either screen.

The algorithm predicts some residual population signal, but changing the
ranking from independent marginal gains to these32 greedy conditional gains
does not beat the incumbent in either animal mean. This limits the specific
alpha1/32-row proxy; it does not establish absence of conditional population
information, biological complementarity or causal electrode value. The
named removal/complementarity/coupling endpoints remain unactivated, and
no influential session is excluded. Search coverage and the40-comparison
minimum remain incomplete. A scientifically distinct next proposal will be
reviewed before any new implementation or source reads.

Root rendered and visually checked the [conditional comparison figure](figures/initializer_conditional_attempt001.png),
[PDF](figures/initializer_conditional_attempt001.pdf) and
[68-row CSV](figures/initializer_conditional_attempt001.csv) in allocation46457171.
The independently authored [plotter](tools/plot_initializer_conditional.py)
uses only the completed report, shows each selector against its own control
and conditional-minus-marginal session contrasts, and states the different
nulls, two-animal support and omission limits.

The prospective [high-power-only proposal](high_power_correspondence_proposal.md)
asks whether the marginal correspondence result survives without LMP, using
actual two-coordinate native100–200/200–400Hz features. It is unadopted and
activates no implementation, ASTRA recipe, source read or new comparison.
The two-band own-label control is capacity matched; comparisons with the
completed three-band and LMP-only pipelines are descriptive feature-and-capacity
sensitivities, without a contamination or independent-physiology claim.

## 2026-10-04 06:08 UTC — `development_search`: conditional resource pilot complete; finite remaining-seven run selected before scientific release

Root observed the predetermined complete pilot exit0 in allocation46457171
from06:08:16.268928 to06:08:28.194022UTC. The earlier approximate06:08 launch
selection is now resolved by these actual execution timestamps. The
[preserved pilot](runs/development_search/initializer_conditional_pilot_attempt001/initializer_conditional_gain.json)
completed in11.925111seconds wall and11.886754seconds process CPU, with
68,376KiB peakRSS. It completed95 target models,1,900 inner fits,9,500
capacity scores,30 orders,960 conditional choices,200 direction maps and
five direction-only models; selector time was0.571231seconds. There were
zero constant-input fit occurrences and19 constant-target occurrences under
the existing retained-zero-SD-to-one rule. Source accounting records1,248
fitting-trial views,40 evaluation trials,12,828,160 logical bytes and3,612,160
unique bytes; retained pilot JSON is579,453bytes.

Root opened only resource/support fields, not candidate/control scientific
scores. The complete pilot is still uncharged and its scientific report
remains withheld from interpretation. Based on its approximately12-second
measurement, the previous107-second full same-model-count family and the
existing resource allowance, root selected **600seconds internal/630seconds
external** for the remaining seven sessions. This is a finite pre-score
selection within the existing budget, not a resource-budget extension.

The selected fresh attempt is`runs/development_search/initializer_conditional_attempt001`
in current attended allocation46457171, one CPU/6GiB, end06:23:20UTC.
The new ASTRA recipe and pilot artifact input are recorded before execution;
root will validate the changed program and then launch the exact command
below. This record establishes selection, not actual remaining-phase start.
Root subsequently observed the changed ASTRA schema and semantics pass
without warnings, evidence
`astra/v0.0.14/validation_20261004_initializer_conditional_remaining.json`,
then launched that exact remaining-phase command at approximately
**06:10UTC** in job46457171. Full completion and scientific scores remain
unobserved at this launch entry.
All665 new fits precede reading the pilot or old target/uniform JSON; the
pilot is then reused once with exact recipe/role/count equality. Combined
consumed resources remain explicit, and no old comparison is refitted or
recharged. Count30 stays unchanged until both full panels and replays finish.

```bash
timeout 630s python3.12 outputs/tools/development_initializer_conditional.py --mode remaining --source-root "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" --metadata-report outputs/runs/source_support/attempt002/metadata_support.json --trial-roles outputs/runs/source_support/development_trial_roles_attempt001/development_trial_roles.json --pilot-report outputs/runs/development_search/initializer_conditional_pilot_attempt001/initializer_conditional_gain.json --target-reference outputs/runs/development_search/initializer_target_label_control_attempt001/initializer_target_label_control.json --uniform-reference outputs/runs/development_search/initializer_repeatability_attempt001/initializer_repeatability.json --max-seconds 600 --output-dir outputs/runs/development_search/initializer_conditional_attempt001
```

## 2026-10-04 06:02 UTC — `development_search`: initializer conditional-gain pair adopted; complete first-session resource pilot selected

Root adopted the [conditional-gain recipe](initializer_conditional_gain_proposal.md)
after root and independent scientific review. The new
[pure selector](tools/initializer_conditional_gain.py) preserves the exact
qualified alpha1 conditional residual-ridge objective and rank-three updates,
with both right-hand sides computed before updating. It takes exactly32
choices, masks selected channels, uses ascending physical IDs for exact ties
and retains finite gains without clipping or a new selector penalty. Each
control uses the original within-direction target-ID maps but its own greedy
AX/AY state; the true target always trains the decoder.

New [orchestration](tools/initializer_conditional_experiment.py) and
[pilot/remaining runner](tools/development_initializer_conditional.py) preserve
the30-order-before-future and95-model-before-evaluation barriers. The original
three-band reader/fitter/scorer paths and completed artifacts remain unchanged.
The remaining phase fits all seven nonpilot targets before parsing any pilot
or historical scientific JSON. It then verifies the exact pilot recipe,
roles, counts and information-state flags, reuses its results without refit,
checks the25 old control maps and all96 reference capacity/scores, and keeps
current-phase costs separate from combined consumed pilot+remaining costs.
Completed repeatability/uniform panels are descriptive references only and
are not refitted or charged again.

Root selected **Mihili_CO_20140217.mat** as the predetermined complete
resource pilot, all six branches/views,95 actual models/120 references,
1,900 inner fits,9,500 capacity scores,30 orders,960 greedy steps,200
direction maps,1,248 fitting-trial reads and40 evaluation trials. This
measures the new32-step selection workload. Only timing, memory, support
and numerical validity may inform the finite remaining-seven bound;
scientific scores stay withheld until that bound is selected and the whole
cohort completes. The pilot alone charges no comparison.

The selected pilot bound is270seconds internal/300seconds external in
root's current verified attended allocation46457171, one CPU/**6GiB**,
ending06:23:20UTC. The scientific branch expects to fit within4GiB, which
does not change the parent's actually reserved6GiB. Root owns actual tests,
ASTRA validation, launch and scheduler checks. Root and independent review
passed all three implementation modules without actionable defects. The
implementation worker independently reviewed all nine scientific fixtures.
Root subsequently observed **all nine fixtures pass in1.124seconds** in
allocation46457171, followed by ASTRA schema/semantic validation passing
without warnings, exit0, evidence
`astra/v0.0.14/validation_20261004_initializer_conditional.json`.
The validator's actual filename omits`_gain`; the command below reflects the
observed artifact. Root selected immediate execution of the logged pilot at
approximately06:08UTC; actual launch timing is pending root's observation.
The worker has run no tests/source reads.

Full completion would contain240 orderings,7,680 greedy steps,1,600 maps,
760 target fits,15,200 inner fits,76,000 capacity scores,9,984 fitting views
and320 evaluation trials. Count30 stays unchanged until both complete panels
and references succeed, then two comparisons yield32. The prospective
incumbent-challenge and own-control screens are separate; no named
electrode-removal/complementarity endpoint or held session opens.

```bash
python3.12 -m unittest discover -s outputs/tools -p test_initializer_conditional_gain.py -v
env -u PYTHONPATH "$EP_AST_BIN" validate outputs/astra/v0.0.14/astra.yaml --skip-evidence --json > outputs/astra/v0.0.14/validation_20261004_initializer_conditional.json
timeout 300s python3.12 outputs/tools/development_initializer_conditional.py --mode pilot --source-root "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" --metadata-report outputs/runs/source_support/attempt002/metadata_support.json --trial-roles outputs/runs/source_support/development_trial_roles_attempt001/development_trial_roles.json --max-seconds 270 --output-dir outputs/runs/development_search/initializer_conditional_pilot_attempt001
```

## 2026-10-04 05:57 UTC — `development_search`: repeatability pair complete; modest uniform-control advantage fails the incumbent challenge

Root released the [complete repeatability report](runs/development_search/initializer_repeatability_attempt001/initializer_repeatability.json)
after exit0 in allocation46457171, from05:56:07.268357 to05:57:54.434490UTC.
Runtime was107.166152seconds wall and99.022705seconds process CPU, with
96,720KiB peakRSS. All760 target fits,15,200 inner systems,76,000 capacity
scores,240 orderings,40 repeatability fitting views and200 physical-ID
permutations completed. Logical reads were113,959,680bytes versus30,094,080
unique bytes, with9,984 fitting-trial views and320 evaluation trials. The
retained JSON is7,933,562bytes. Both full96 historical/within-run replay and
direction-null replay have maximum score difference zero. The completed pair
adds two evaluations: **30 valid comparisons total**, held access false.

Initializer repeatability selection has Q **0.201854582593**, compared with
uniform-order control Q **0.199542707225**, a difference of
**+0.002311875368**. Chewie is **+0.004478920162**, Mihili
**+0.000144830575**; the prospective two-animal usefulness screen passes.
All four Chewie sessions and two of four Mihili sessions favor repeatability.
Every overall session omission remains positive, approximately
+0.001906584214 to+0.002848266436. However, omitting Mihili20140217 or
20140218 makes the remaining Mihili mean−0.000124048941 or−0.000308132835.
All five individual uniform controls lose overall, but Mihili loses against
seeds3 and4. Its mean effect is small and sensitive despite the sign screen.

The incumbent challenge fails. Repeatability loses to true initial marginal
selection by **−0.003178872038** overall, Chewie **−0.004679285466**
and Mihili **−0.001678458610**. Five of eight individual session contrasts
favor repeatability, yet the larger losses dominate both animal means.
Omitting Chewie20161006 reverses Chewie to+0.002657319820 and the overall
contrast to+0.000489430605; every Mihili omission stays negative. This
highlights the incumbent advantage's influential-session concentration
without choosing an exclusion or declaring repeatability superior.

Repeatability's residual contrast beyond direction means is
Chewie **+0.034614507442** and Mihili **−0.000113994439**. Three of
four Mihili and all four Chewie sessions are individually positive; only
omitting Mihili20140304 makes that animal's mean positive (+0.002233512343),
while its other omissions remain negative. This residual readout is
descriptive, separate from the two adopted selector comparison screens.

Across40 fitting views there are no zero-SD LFP columns. Zero-score and
tied-electrode counts each range from0to15; each view's maximum mean
repeatability is between0.427386 and0.707506. The candidate's exact-tie
physical-ID preference and the control's uniform-ID rule therefore remain
relevant to interpretation. The experiment supports some practical value of
this target-free ordering relative to its paid uniform comparator, especially
in Chewie, but it does not beat the target-linked incumbent in both animal
means or establish a generic quality, causal or joint-acquisition principle.
Targets were still acquired for decoding; random selection does not
intrinsically need the matched all96 initializer exposure.

Root requested a prospective initializer-only greedy conditional-gain
candidate with five matched within-direction target-label controls next.
The complete [conditional-gain proposal](initializer_conditional_gain_proposal.md)
was prepared before adoption to challenge redundancy under
the same frozen information state, using
the exact earlier algorithmic alpha1 proxy. This does not activate the
separate named electrode-removal/complementarity endpoint. No new source or
fit is authorized by this follow-up description alone. Its32-step rankings
introduce meaningful new selection work, so the proposed first-session
resource pilot measures that cost while withholding scientific scores; root
may instead select a justified finite full-cohort bound before execution.
The completed matched uniform panel is reused only descriptively after all
new fits, without refitting or charging it again. The shared40-minimum and
broader search coverage remain incomplete. Root subsequently adopted the
recipe and selected the complete first-session pilot at06:02UTC; the
separate preparation entry above supersedes its earlier unadopted state.

## 2026-10-04 05:51 UTC — `development_search`: initializer-only repeatability and uniform control adopted and prepared

Root adopted [initializer_repeatability_proposal.md](initializer_repeatability_proposal.md)
after reviewing the full recipe and original selector formula; the independent
reviewer also passed it without requested changes. The new
[pure selector](tools/initializer_repeatability.py) accepts no population Y:
it uses exactly the original raw-X population-standardized direction ANOVA
repeatability, clips negative between-minus-within components to zero, maps
zero/zero to zero and averages all three feature scores without dropping
constant bands. Exact candidate ties retain ascending physical electrode IDs.

Five canonical full-ID uniform shuffles form the matched control, with the
fixed prospective literals and no identity rejection, score sorting or
outcome-dependent redraw. They are not called score-label permutations;
uniformity is preserved when candidate scores tie. The control pays the same
all96 initializer even though its choice does not need those observations.
Population targets remain acquired and train the true-target decoders.

New [orchestration](tools/initializer_repeatability_experiment.py) and
[runner](tools/development_initializer_repeatability.py) reuse the qualified
three-band reader, initial selected-rank and later conditioned fitting
helpers. All30 orders precede any future source row; all95 distinct models
precede evaluation. Existing source roles, target counts/exposure, all20
target denominators/direction-null scores and every branch's five full96
score/capacity records are checked after all new fitting. Old source files
and completed outputs are unchanged. No old score chooses a seed or model.

Root and the independent reviewer passed static review of all three new
modules without actionable defects. The implementation worker independently
reviewed all eight scientific fixtures without defects. Root subsequently
observed **all eight pass in0.328seconds at approximately05:55:15UTC** in
verified allocation46457171 on`sh03-01n68.int`, one CPU and6GiB. Its actual
start is05:53:16UTC and end06:23:20UTC, with no requeue. The source branch
retains its within4GiB envelope; the attended parent actually reserves6GiB.
Root subsequently observed ASTRA schema and semantics pass without warnings,
exit0, in the same shell, evidence
`astra/v0.0.14/validation_20261004_initializer_repeatability.json`.
The worker has run no source reads, tests or
scheduler calls; root owns those actions.

After those actual checks, root selected the exact full-eight-session command
below for fresh`runs/development_search/initializer_repeatability_attempt001`
in that verified allocation, **270seconds internal/300seconds external**,
to launch after the ASTRA result. This records selection before the process
side effect, not an actual start or scientific completion.
Root then observed actual full-eight-session launch at **05:56:07UTC** in
PTY65476/job46457171, with the exact selected fresh attempt and270/300-second
bounds. Root monitors the live process; no partial scientific report has been
read by this worker and count28 remains unchanged until completion.
Totals are40 repeatability fitting views,240 orderings,200 physical-ID
permutations,760 actual target fits,15,200 inner systems,76,000 capacity scores,
9,984 fitting reads and320 evaluation trials. Count28 remains until the
complete pair and all reference checks succeed, then two comparisons yield30.
The usefulness-vs-uniform and challenger-vs-marginal two-animal screens are
distinct;0.001 search resolution, all seeds, session omissions and exposure
caveats remain. No held session or named explanation endpoint opens.

```bash
python3.12 -m unittest discover -s outputs/tools -p test_initializer_repeatability.py -v
env -u PYTHONPATH "$EP_AST_BIN" validate outputs/astra/v0.0.14/astra.yaml --skip-evidence --json > outputs/astra/v0.0.14/validation_20261004_initializer_repeatability.json
timeout 300s python3.12 outputs/tools/development_initializer_repeatability.py --source-root "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" --metadata-report outputs/runs/source_support/attempt002/metadata_support.json --trial-roles outputs/runs/source_support/development_trial_roles_attempt001/development_trial_roles.json --reference-report outputs/runs/development_search/initializer_target_label_control_attempt001/initializer_target_label_control.json --acquired-reference outputs/runs/development_search/acquired_only_attempt001/acquired_only.json --max-seconds 270 --output-dir outputs/runs/development_search/initializer_repeatability_attempt001
```

## 2026-10-04 05:46 UTC — `development_search`: LMP-only pair complete; residual signal survives, selector advantage is fragile

Root released the [complete LMP report](runs/development_search/lmp_correspondence_attempt001/lmp_correspondence.json)
after exit0 in allocation46456142, from05:45:08.815162 to05:46:35.471850UTC.
The run took86.656703seconds wall and83.428825seconds process CPU, with
92,524KiB peakRSS. All760 models,15,200 inner fits,76,000 capacity scores,
240 orderings and1,600 control maps completed. Actual logical reads were
69,722,880bytes versus13,382,400 unique bytes, with9,984 fitting-trial views
and320 evaluation trials; the retained report is6,651,290bytes. Within-run
full96-LMP score/capacity replay and historical direction-null Q replay both
have maximum score difference zero. Both complete panels charge two
comparisons: **28 valid evaluations total**. Held-session access is false.

LMP true marginal selection has Q **0.199332824718**, compared with its
mean-five correspondence control Q **0.198620945726**, a contrast of
**+0.000711878991**. Both animal means are positive: Chewie
**+0.000994125761**, Mihili **+0.000429632221**. The strict directional
correspondence screen passes, but neither animal mean nor the overall mean
exceeds the unchanged0.001 search-progress resolution. Five of eight sessions
are positive: three Mihili and two Chewie. This is a small development
selection effect, not a new calibrated significance or equivalence claim.

The matched residual-information screen also passes. LMP true selection
exceeds the direction-only reference by **+0.014728498626** overall,
Chewie **+0.028907505936** and Mihili **+0.000549491317**. Both animal
residual contrasts remain positive under every fixed-prediction session
omission. This provides evidence of residual population prediction using the
native LMP representation under the current recipe, although the Mihili
contribution is modest.

The correspondence effect is less robust. Omitting Chewie20161005 reverses
the overall true-minus-control contrast to **−0.000117074364** and Chewie
to **−0.000663780949**; all other overall omissions remain positive, and
every Mihili omission remains positive. Against individual control seed3,
the overall contrast is−0.000019976427. Chewie loses against seeds1 and3;
Mihili loses against seed4. The five ordinary maps are not independent
biological replications or confidence bounds, and the mean-positive screen
does not remove these limitations.

The LMP true selector loses to the completed three-band true selector by
**−0.005700629913** overall, Chewie **−0.010386286972** and Mihili
**−0.001014972854**. Every overall session omission preserves this loss;
omitting Mihili20140303 reverses Mihili alone to+0.000150405949. The
LMP-minus-three-band *correspondence-contrast* difference is
**−0.004668934450** overall, Chewie−0.009377829993 and Mihili
+0.000039961094. These are whole-pipeline spectral sensitivities: removing
the two power features also reduces real input dimension and regularized
capacity. They do not isolate spike contamination, certify native LMP
spike-free, or establish causal electrode importance, conditional
complementarity or a useful joint electrode/trial policy.

No new source reads or fits are needed for this closeout. A compact plot of
the three principal contrasts, session influence and control-seed variation
was rendered from the completed report by
[plot_lmp_correspondence.py](tools/plot_lmp_correspondence.py).
At05:50UTC root rendered and visually checked the
[44-row figure](figures/lmp_correspondence_attempt001.png),
[PDF](figures/lmp_correspondence_attempt001.pdf) and
[plotted-value CSV](figures/lmp_correspondence_attempt001.csv) in allocation46456142;
all plotted animal means replayed, with no new fits or confidence intervals.
Root closed attended PTY83076/job46456142 at05:50:33UTC, exit0. That compute
session is no longer available for new tests or source work.

The next [initializer-only repeatability proposal](initializer_repeatability_proposal.md)
was prepared for root and independent review before scoring.
It preserves the original target-free LFP repeatability formula within24/32
initializer views and compares it with five uniform physical-ID orderings at
matched32×96 exposure. The uniform controls are explicitly not score-label
permutations; ties retain the candidate's physical-ID preference but cannot
make the control nonuniform. Targets remain acquired for true-target decoding,
despite being absent from the selector API. A separate incumbent comparison
would test whether this simpler rule improves on target-linked marginal
selection. No source read, ASTRA activation or new fitting is authorized by
this prospective document alone. Root subsequently adopted it at05:51UTC;
the separate preparation record above supersedes its earlier unadopted state.
Search coverage and the40-comparison minimum remain incomplete.

## 2026-10-04 05:40 UTC — `development_search`: native-LMP correspondence candidate/control adopted and prepared

Root adopted the clarified [LMP-only recipe](lmp_only_correspondence_proposal.md)
after root and independent scientific review. It removes the100–200/200–400Hz
power inputs from both selector and decoder and keeps one real native guide(0,0)
LMP coordinate per electrode. The new [reader/trajectory](tools/lmp_acquired_reader.py)
reads only selected LMP numerical rows and counts10 actual-dtype values per
unique electrode-trial. It preserves original trial/physical-electrode/unit
guides, ten-bin native targets and chosen-channel future access. The old
three-band code and all completed artifacts remain unchanged.

The [scalar selector](tools/lmp_correspondence.py) uses the same alpha1
mean-loss objective with one coordinate and exact existing five-seed ordinary
target-label maps; true selection uses unpermuted targets. The
[initial nested fitter](tools/lmp_selected_rank.py) uses real E-column input
and legal ranks, with the same qualified ridge/projection/scaling/tie rules.
No zeroed high-band padding or matched-three-band-capacity claim is made.
The [orchestration](tools/lmp_correspondence_experiment.py) freezes all30
orders before future reads, retains true Y in every decoder, shares five
identical all96-LMP fits and fixes all95 distinct models before evaluation.
The [runner](tools/development_lmp_correspondence.py) reads completed reference
packets only after all new fitting. It checks exact source-role/trial maps,
target counts/exposure and representation-independent denominator/direction
means, then compares the differing spectral predictions descriptively.
Historical unit-guide IDs were not serialized; only current within-run unit
identity and historical target-count/support equality are asserted.

Prospective correspondence and beyond-direction-mean screens separately
require positive contrasts in both animal means. The0.001 search resolution,
all20 cells, mean-five-SSE control, seed summaries and fixed-prediction
omissions remain. A result cannot certify native LMP spike-free, attribute
spectral differences solely to frequency/contamination, or establish coupling.

At05:42UTC root made one optional read-only BR retrieval for relevant M1 LMP/
spectral-method evidence (`Gallego Carracedo local field potentials`,
Publication limit3). It returned Parkinson LFP/MEG PMID21122819, epilepsy
LFP19362152 and accumbens reward26019312, with no relevant M1 method evidence.
The result had no decision impact and is not scientific endorsement, review
approval or canonical authority; no repeated retrieval is required.

Root reviewed all new code and relevant inherited helpers without actionable
defects. The EP05 reviewer authored seven independent scientific fixtures,
which the implementation worker independently reviewed without defects.
Root observed **all seven passed in0.100seconds at05:44UTC** in verified
allocation46456142 on`sh03-01n68`. Root subsequently observed ASTRA schema and
semantic validation pass without warnings, exit0, in the same PTY, evidence
`astra/v0.0.14/validation_20261004_lmp_correspondence.json`.
Root owns all validation, source execution
and scheduler checks. The worker has run no tests or scientific source reads.

After these checks, root selected the exact full-eight-session command below
for fresh`runs/development_search/lmp_correspondence_attempt001`, **270seconds
internal/300seconds external**, in the current one-CPU/4GiB attended
allocation46456142. Its verified end is05:55:37UTC, leaving approximately ten
minutes at selection. No new pilot is required by the adopted measured-family
resource rationale. This is the prelaunch selection record; actual launch and
completion await root's observation. Expected totals are760
models,15,200 inner fits,76,000 capacity scores,240 orderings,1,600 maps,9,984
fitting reads and320 evaluation trials. Only a complete candidate/control
pair adds two comparisons (26to28); count26 is unchanged at preparation.
Held sessions remain closed.

Root observed actual full-cohort launch at **05:45:08UTC** in attended
PTY83076/job46456142, with unchanged270/300-second bounds and the selected
fresh attempt. No immediate runner output is required to establish that
dispatch. Root owns process monitoring; the worker has not read a partial
scientific report. Count26 remains unchanged pending complete evidence.

```bash
python3.12 -m unittest discover -s outputs/tools -p test_lmp_only_correspondence.py -v
env -u PYTHONPATH "$EP_AST_BIN" validate outputs/astra/v0.0.14/astra.yaml --skip-evidence --json > outputs/astra/v0.0.14/validation_20261004_lmp_correspondence.json
timeout 300s python3.12 outputs/tools/development_lmp_correspondence.py --source-root "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" --metadata-report outputs/runs/source_support/attempt002/metadata_support.json --trial-roles outputs/runs/source_support/development_trial_roles_attempt001/development_trial_roles.json --reference-report outputs/runs/development_search/initializer_target_label_control_attempt001/initializer_target_label_control.json --acquired-reference outputs/runs/development_search/acquired_only_attempt001/acquired_only.json --max-seconds 270 --output-dir outputs/runs/development_search/lmp_correspondence_attempt001
```

## 2026-10-04 05:31 UTC — `development_search`: initializer correspondence control complete; positive but session-sensitive contrast

Root observed the [complete control](runs/development_search/initializer_target_label_control_attempt001/initializer_target_label_control.json)
exit0 in allocation46456142 on`sh03-01n68`, from
05:29:38.696708 to05:31:18.304743UTC. The earlier05:29:39 launch observation
was rounded. Wall time was99.608052seconds, process CPU85.620006seconds and
peakRSS86,108KiB. All required640 distinct models,12,800 inner fits,
64,000 capacity scores,200 orderings and1,600 direction maps completed.
Logical reads were99,982,080bytes versus30,094,080 unique bytes, with8,448
fitting-trial views and320 evaluation trials; retained JSON is5,438,937bytes.
Every common all-96 score/capacity replay passed with maximum score difference
zero. The complete five-seed control charges one comparison: **26 valid
evaluations total**. Held-session access remains false and search is incomplete.

True initial marginal selection has Q **0.205033454630**, compared with
mean-five target-label control Q **0.199652641190**. Their principal
true-minus-control contrast is **+0.005380813441**, with Chewie
**+0.010371955755** and Mihili **+0.000389671127**. The prospective strict
two-animal directional screen therefore passes. Five of eight session
contrasts are positive: three of four Chewie and two of four Mihili.
The Mihili mean is smaller than the unchanged0.001 search resolution;
that resolution is not a new biological margin or significance threshold.

Every overall fixed-prediction session omission remains positive,
approximately+0.001215596990 to+0.007433436530. However, omitting
Mihili20140303 makes the remaining Mihili contrast **−0.000497804739**.
Omitting Chewie20161006 reduces Chewie to+0.002041522858 and the overall
contrast to+0.001215596990. All five individual seed contrasts are positive
overall (approximately+0.0028660 to+0.0104171), while Mihili versus seed2 is
negative at−0.000403319. Five seeds are one randomized control, not five
independent animal replications or a confidence interval.

These results support useful within-direction target correspondence for this
fixed early marginal selector on the present development data, with a modest
and influential-session-sensitive Mihili contribution. They do not identify
conditional complementarity, spike-independent physiology, causal electrode
importance, a successful trial allocator or sensor–trial coupling. True
targets remain available to every decoder; ordinary selector-label maps can
preserve some associations and overlapping selected channels. The matched
initializer/decoder control is stronger evidence for this narrow question
than the earlier unequal-initializer-cost random reference, while the four
held sessions still provide no evidence because they remain closed.

Root requested a prospective low-frequency-only sensitivity next. Its
complete [LMP-only candidate/control specification](lmp_only_correspondence_proposal.md)
uses one actual guide(0,0) coordinate per physical electrode in both selection
and decoding, with true target correspondence versus the same five original-ID
permutation maps. Physical input dimensions and legal ranks are computed from E,
not3E; no padded/zeroed high bands or claim of matched three-band capacity.
The two prospective screens separately assess correspondence usefulness and
residual information beyond direction means. A positive result would not
certify native LMP as spike-free. The complete proposed pair would add two
comparisons only after760 models and all references finish, using an estimated
270/300-second bound on one CPU within4GiB. Root subsequently adopted it at
05:40UTC; the separate adoption/preparation record above supersedes this
entry's prospective state.

Root subsequently rendered and visually checked the report-backed
[initializer correspondence figure](figures/initializer_target_label_control_attempt001.png),
[PDF](figures/initializer_target_label_control_attempt001.pdf) and
[37-row plotted-value CSV](figures/initializer_target_label_control_attempt001.csv)
in allocation46456142. The figure makes the animal/session, five-seed and
fixed-prediction omission evidence visible without new fits or comparisons.
No confidence interval or independent-five-seed replication is asserted.

## 2026-10-04 05:22 UTC — `development_search`: initializer target-label control adopted and prepared

Root adopted the [fixed five-seed control](initializer_target_label_control_proposal.md)
after static review and before implementation or new source reads. It tests
whether the original paid initializer's true within-direction trial-to-target
correspondence contributes to its marginal electrode ordering. Each seed
permutes complete native-unit target vectors within each direction and only
within a matching 24-row inner or 32-row outer initializer view. Ordinary
one-shot maps use canonical original trial IDs, include identity/fixed points,
and preserve the true fitting-view normalization. Decoder training and CV
always use true targets. The original true marginal policy is reused, not
refitted or charged again; five seeds together are one control comparison.

New [core](tools/initializer_target_label_control.py),
[orchestration](tools/initializer_target_label_experiment.py) and
[runner](tools/development_initializer_target_labels.py) preserve prior files
and outputs. Every session freezes all25 orderings before future fitting reads,
then fits all80 distinct target models before its40 evaluation rows. The
five common all-96 models are shared across seeds. Scientific reference JSON
is parsed only after all eight sessions' new fits. Its recorded recipe,
session/animal identities, target exposure and twenty cells must agree;
every seed's five all-96 cells replay exact channels/rank/alpha/candidate count
and Q/Qwrong/pairing scores within1e-10. The old acquired-only report lacks
serialized source-role records and predictions; neither bytewise role nor
prediction-array identity is claimed. Current source roles are enforced by
the qualified reader and saved in the new report.

The principal contrast is true marginal Q minus mean-five-control Q, with
strict positive means required in both animals for the directional screen.
The unchanged0.001 search resolution, all20 cells, equal session/animal weights,
seed summaries and all fixed-prediction session omissions remain. This tests
correspondence usefulness under early marginal ranking, not causal electrode
importance, conditional complementarity or sensor–trial coupling. The cheaper
fixed-random initializer stays explicitly unequal in acquisition cost.

Root owns scientific fixtures, ASTRA validation and all compute/scheduler
actions. Root and the independent EP05 reviewer completed the full
core/orchestration/runner static review without actionable defects. Root
reported all seven [focused scientific fixtures](tools/test_initializer_target_label_control.py)
passed in0.071seconds in attended allocation46456142 on`sh03-01n68`.
The pinned validation
`astra/v0.0.14/validation_20261004_initializer_target_label_control.json`
passed schema and semantics without warnings, exit0. The allocation began
approximately05:25:5xUTC, with one CPU/4GiB and30minutes; root is its sole
compute owner. The worker's actual shell is login node`sh04-ln04` with no
Slurm allocation and performs bounded static reads and editing only.

After those observed checks, root selected the exact full-cohort270/300-second
command below to follow the already selected EP05 resource pilot, expected
approximately05:28–29UTC. This records selection, not actual empirical launch;
the true start and completion will be added from the coordinator's observation.

Root subsequently observed actual substantive launch at **05:29:39UTC** in
allocation46456142 on`sh03-01n68`, using the exact fresh attempt and270/300-second
bounds below. The preceding EP05 pilot completed before this launch; numerical
work remains sequential on the single allocated CPU. The allocation's actual
end is05:55:37UTC. At this launch observation the source run is live, no new
scientific scores have been released and the completed count remains25.

Selected recipe: one CPU within4GiB,270seconds internal/300seconds external,
numerical threads one, with at least five minutes remaining after checks.
The prior reverse-animal run measured72.198seconds for520 models; this fixed
control has640 models,12,800 inner fits,64,000 capacity scores,200 orderings,
1,600 direction maps,8,448 fitting-trial reads and320 evaluation reads.
Runtime for this implementation is not measured yet. The exact new attempt
path below was absent during the bounded resume inspection. Root will verify
the current attended allocation before any test or source command. There is
no automatic retry/requeue or partial-session scientific charge. Completed
count stays25 until the whole control and reference replay complete; then one
comparison yields26. The four held sessions and named endpoints remain closed.

```bash
python3.12 -m unittest discover -s outputs/tools -p test_initializer_target_label_control.py -v
env -u PYTHONPATH "$EP_AST_BIN" validate outputs/astra/v0.0.14/astra.yaml --skip-evidence --json > outputs/astra/v0.0.14/validation_20261004_initializer_target_label_control.json
timeout 300s python3.12 outputs/tools/development_initializer_target_labels.py --source-root "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" --metadata-report outputs/runs/source_support/attempt002/metadata_support.json --trial-roles outputs/runs/source_support/development_trial_roles_attempt001/development_trial_roles.json --reference-report outputs/runs/development_search/acquired_only_attempt001/acquired_only.json --max-seconds 270 --output-dir outputs/runs/development_search/initializer_target_label_control_attempt001
```

## 2026-10-04 00:35 UTC — `development_search`: reverse-animal comparison complete; transfer screens fail

The [complete reverse-animal report](runs/development_search/reverse_animal_electrode_attempt001/reverse_animal_electrode_value.json)
finished in attended allocation46441736 on`sh03-06n50.int`, exit0, from
00:34:21.397368 to00:35:33.595007UTC. Runtime was72.197650s and peakRSS
127,576KiB. Both declared full panels completed with exact common all-96
score/capacity replay difference **zero**, bringing the shared total to
**25 valid evaluations**. All four held sessions remain closed; no named
explanatory endpoint or global stopping condition is activated.

Other-animal learned selection has equal-animal Q **0.202759744022**. It loses
to initial marginal selection by **−0.002273710608**, with both Chewie and
Mihili negative (−0.003309746223 and−0.001237674994). Its three-seed own-label
control has Q0.202213554819, so the candidate's overall difference is
**+0.000546189204**. This small positive mean fails the prospective two-direction
screen: Chewie is+0.001125716743 while Mihili is−0.000033338336. Six of eight
sessions beat the own-control mean; four beat marginal selection. Both
prospective directional screens therefore fail, without altering their rules.

The candidate improves on the previous seven-donor learned rule by
**+0.000940500937** overall, positive in both animal means
(Chewie+0.000062168203; Mihili+0.001818833672). The overall improvement is
below the unchanged0.001 search-progress resolution, which is not a biological
margin or significance threshold. It still trails the previous seven-donor
label-control mean by−0.000662563321. The gain over cheaper fixed-random
selection is+0.000984618094 with Mihili negative; this is not an equal-cost win.

The three new control-seed Q means are approximately
0.20303181/0.20329726/0.20031159. Two of three individually exceed the candidate;
the low third seed affects their average. Omitting Chewie20161006 reverses
the candidate-minus-own-control overall contrast to approximately−0.00030471804,
with both remaining animal means negative, and reverses its contrast with the
previous learned rule to approximately−0.0001174001. These are fixed-prediction
omission summaries without donor/ranker refitting. Seeds and omissions do not
provide additional biological replication, independent tests or confidence
intervals.

Observed new work is40 shared ranker fits,160 target orderings,520 target
refits,10,400 inner systems,52,000 capacity scores,40 direction-only models,
6,912 fitting-trial reads and320 evaluation trials. Logical source reads total
86,004,480bytes with30,094,080 unique logical bytes; the retained report is
8,122,127bytes. The original897 target zero-SD fitting occurrences use the
qualified replacement-one rule. There are zero new donor-label fits. The
reused forty donor views retain their earlier3,840 full-pass label fits and
all7,200 historically incurred fits including the former pilot; these are
inherited calibration costs, not free information or newly charged fits.

This completes the explicit reverse-animal check for the fixed learned
ranking recipe. It does not establish reliable learned-label transfer or
advantage over paid within-session marginal selection. Changing from seven
mixed donors to four other-animal donors also changes training amount and
distribution, so the result does not identify a causal animal-identity effect.
Further work should address a distinct information/selection question. Search
coverage and the shared40–112 allowance remain incomplete; no held feedback
was used and no new independent-animal evidence is claimed.

## 2026-10-04 approximately 00:20 UTC — `development_search`: reverse-animal learned-value comparison adopted and prepared

Root adopted the exact [reverse-animal recipe](reverse_animal_electrode_value_proposal.md)
before implementation or new source reads/fits. The completed seven-donor
result motivates a distinct transfer check: train on exactly four sessions
from the other animal, in both directions, using unchanged predictors, donor
labels, ranker penalty, target acquisition, decoder/CV and three original
ordinary label permutations. Four equally weighted donor sessions give total
weight one, not one-half. The same-animal exclusion precedes numeric donor
statistics. Removing three donors changes training amount and composition;
this does not isolate an animal-identity effect or establish new-animal evidence.

New [core](tools/reverse_animal_electrode_value.py),
[orchestration](tools/reverse_animal_electrode_experiment.py) and
[runner](tools/development_reverse_animal_electrodes.py) preserve all previous
scientific files. The narrow loader reuses the forty completed donor views,
so the combined old JSON is physically parsed before fits. Only whitelisted
training records reach the ranker; old outcome fields are used for contrasts
after all new target fits. No claim of an unopened old packet, OS isolation,
free donor supervision or equal total-information cost is made. All 7,200
historically incurred donor fits remain disclosed; new donor fits number zero.

Root statically reviewed the core/orchestration and found no actionable defect.
The EP05 reviewer independently reviewed the core and prepared
[six focused fixtures](tools/test_reverse_animal_electrode_value.py), covering
same-animal and wrong-view exclusion before numeric access, unit-sum four-
session weights and analytic alpha1 coefficients, matched24/32 information,
missing-donor rejection, unchanged ordinary label maps, and exactly forty
shared ranker states. These checks are prepared, not yet observed executed.
Root owns all testing, ASTRA validation, source execution and scheduler checks.
The independent EP05 reviewer subsequently read the complete orchestration
and runner and found no actionable defect in the40 shared states/160 rankings,
both source/evaluation barriers, declared cached-record parsing, all520 fits
before old outcome comparisons, or common all-96 score replay. Implementation
and six fixtures are frozen for root's actual checks.

Selected full-cohort envelope: **one CPU within 4 GiB,270s internal and300s
external**, numerical threads one, with at least five minutes available after
review/checks. The previous full family cost83.538s included the same520 target
models plus donor-label fitting, which this run reuses. A new generic pilot is
unnecessary. Allocation46440428 was closed when insufficient time remained.
Root requested a concrete30-minute1CPU/4GiB normal CPU_SKU:7502 attended
allocation **46441736**, pending at its last reported observation; an allocation
request is not verified compute execution. The worker has not submitted or run
anything. No retry, requeue or favorable session/seed subset is authorized.

At00:28:02UTC root verified the actual attended PTY on`sh03-06n50.int`,
allocation46441736: one CPU,4096MiB and30minutes, with the qualified module
stack and explicit single-thread settings. The EP05 PCA pilot precedes the
EP08 checks in that session. Root's full runner and six-fixture static reviews
also passed; EP08 test execution, ASTRA validation and the substantive run
remain pending. The worker's own shell remains the login node and performs
only bounded documentation edits.

At approximately00:31UTC root reported **all six focused checks passed in
0.071seconds** in allocation46441736. The pinned ASTRA validation
`astra/v0.0.14/validation_20261004_reverse_animal_electrode_value.json`
passed schema and semantics with no warnings, exit0. Root's complete
core/orchestration/runner static review and the independent EP05 review also
passed. Root selected the exact logged270s internal/300s external command for
immediate full eight-session launch after the brief EP05 remaining-panel ASTRA
check, using fresh`runs/development_search/reverse_animal_electrode_attempt001/`.
This records the launch handoff, not an observed first-source-read timestamp
or completed comparison. No scientific scores have been released; the count
remains23 until all required fits, controls and reference replays complete.

Root subsequently confirmed the substantive command was actually launched at
**00:34UTC** in attended allocation46441736. The earlier approximately00:31UTC
observation was the prelaunch handoff, not execution. The exact selected
command, fresh output and270/300-second bounds are unchanged. Scientific
completion and scores remain unobserved at this launch entry.

The complete comparison expects40 shared ranker fits,160 target orderings,
520 actual target models,10,400 inner fits,6,912 fitting-trial reads and320
evaluation trials. Every target's20 orderings precede later source reads and
all65 target models precede its evaluation. Controls average correct/wrong
SSE, not predictions. Both old common all-96 references must replay. Two full
panels would advance23 to25 valid evaluations; no count changes before full
completion, and all four held sessions remain closed.

Concrete selected commands from the episode directory:

```bash
python3.12 -m unittest discover -s outputs/tools -p test_reverse_animal_electrode_value.py -v
env -u PYTHONPATH "$EP_AST_BIN" validate outputs/astra/v0.0.14/astra.yaml --skip-evidence --json > outputs/astra/v0.0.14/validation_20261004_reverse_animal_electrode_value.json
timeout 300s python3.12 outputs/tools/development_reverse_animal_electrodes.py --source-root "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" --metadata-report outputs/runs/source_support/attempt002/metadata_support.json --trial-roles outputs/runs/source_support/development_trial_roles_attempt001/development_trial_roles.json --training-report outputs/runs/development_search/learned_electrode_attempt001/learned_electrode_value.json --reference-report outputs/runs/development_search/acquired_only_attempt001/acquired_only.json --max-seconds 270 --output-dir outputs/runs/development_search/reverse_animal_electrode_attempt001
```

## 2026-10-04 00:17 UTC — `development_search`: learned selector and label control complete; both lose to marginal selection

The remaining seven targets completed in allocation46440428, exit0, from
00:14:25.420667 to00:15:33.468790UTC:68.048142s and97,876KiB peakRSS.
The [complete report](runs/development_search/learned_electrode_attempt001/learned_electrode_value.json)
reuses the completed pilot after all new fits and passes all-96 score/capacity
replay with exactly zero maximum score difference. Both declared comparisons
are complete: **23 valid evaluations total**, held sessions unopened, no named
explanatory endpoint activated. Neither the pilot nor the three control seeds
is an additional comparison.

Learned selection has equal-animal Q **0.201819243085**, versus initial marginal
selection0.205033454631: **−0.003214211546**. It also loses to its three-seed
donor-label control, Q0.203422307344: **−0.001603064259**. Both losses exceed
the existing0.001 progress resolution in magnitude; that resolution is not a
biological margin or significance test. Chewie/Mihili differences from marginal
are−0.003371914426/−0.003056508665, and from the label control
−0.000977466220/−0.002228662297. Four of eight sessions improve over marginal;
only two improve over the label-control mean. Learned-versus-cheaper-random
is only+0.000044117156 overall, with Mihili negative; it is not an equal-cost win.

Every fixed-prediction session omission preserves both **overall** losses:
learned-minus-marginal ranges−0.004258083092 to−0.001216989860 and
learned-minus-label-control−0.001998826674 to−0.000666464868. Animal signs are
less stable: omitting Chewie20161006 changes Chewie's corresponding contrasts
to+0.000622528946/+0.000441096417, while Mihili remains negative. All three
label-control seed means exceed learned overall; their Q values are
0.203980309071/0.203062074643/0.203224538317. Chewie's learned-minus-seed2 is
slightly positive(+0.000086803014), so do not claim every seed/animal loses.

Combined measured component walltime is83.538256s. Charged work includes
7,200 donor labels (pilot plus recomputed donor records),160 rankers,
520 target refits,10,400 inner systems,52,000 capacity scores,8,352 fitting-read
calls and320 evaluation trials. Logical reads125,940,480bytes have unique union
30,094,080bytes; pilot/complete reports retain2,233,270/15,185,287bytes.
There were897 target zero-SD fitting occurrences, retained using the qualified
replacement-one rule. No source arrays or decoder coefficients were written.

This rejects benefit from the specific three-predictor, fixed-penalty donor
value ranker under its explicitly larger donor-supervision budget. It does not
reject all transferable electrode rules or identify why the learned ranking
loses. The required reverse-animal transfer and remaining search families are
still open; no held score, new source, threshold or selection rule changes.

## 2026-10-04 00:14 UTC — `development_search`: learned-selector pilot complete; remaining seven selected

The predetermined complete pilot ran in allocation46440428 from
00:12:47.290079 to00:13:02.780176UTC, exit0. It completed in15.490114s with
79,436KiB peakRSS:7.691584s donor preparation and7.773981s target fit/evaluation
(7.250625s training reads/fits). Only these resource/support fields were read;
scientific scores remain withheld. All3,360 donor labels,20 rankers,65 target
models,1,300 inner systems and6,500 capacity scores completed. The original
44 native units,1,536 fitting-read calls and40 evaluation trials were retained;
zero-SD target occurrences13 use the existing replacement-one convention.
Logical source reads total28,449,280bytes, unique22,305,280bytes, with18,693,120
donor bytes. The compact report retains2,233,270bytes at
`runs/development_search/learned_electrode_pilot_attempt001/learned_electrode_value.json`.
No valid comparison is charged for this partial cohort.

Select the remaining seven targets now in the same1CPU/4GiB attended
allocation, **540s internal/600s external**. Seven pilot-sized target stages
plus eight donor preparations project about63s before differences in unit
count, source latency and report assembly; the ceiling allows the larger
sessions without changing science. Approximately15minutes remain in the
parent allocation. Recompute all eight compact donor records, charging the
additional3,840 labels, and reuse the completed pilot only after the other
seven target fits finish. Read the old acquired-only score reference only
after those fits, then require the complete all-96 score/capacity replay.
Preserve any failed partial report; no automatic retry or omitted session.
The added concrete ASTRA recipe passed schema/semantic validation before
launch (`astra/v0.0.14/validation_20261004_learned_electrode_remaining.json`,
exit0, no warnings). Root launched the exact command below in46440428.

```bash
timeout 600s python3.12 outputs/tools/development_learned_electrodes.py --mode remaining --source-root "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" --metadata-report outputs/runs/source_support/attempt002/metadata_support.json --trial-roles outputs/runs/source_support/development_trial_roles_attempt001/development_trial_roles.json --pilot-report outputs/runs/development_search/learned_electrode_pilot_attempt001/learned_electrode_value.json --reference-report outputs/runs/development_search/acquired_only_attempt001/acquired_only.json --max-seconds 540 --output-dir outputs/runs/development_search/learned_electrode_attempt001
```

## 2026-10-04 00:04 UTC onward — `development_search`: learned electrode ranker adopted; complete first-target pilot selected

Before new donor labels or fits, root adopted the fixed
[learned-electrode-value proposal](learned_electrode_value_proposal.md).
The implementation adds only the new [pure training core](tools/learned_electrode_value.py),
[trusted orchestration](tools/learned_electrode_experiment.py) and
[finite runner](tools/development_learned_electrodes.py). Existing scientific
readers, selectors and decoders remain unchanged. The target retains one
ten-bin time mean per original native unit, without neuron averaging. Donor
alpha1 uses the exact qualified mean-loss helper
`ridge_training.ridge_coefficients`, with `X'X+n*alpha*I`; the ranker uses the
separately declared weighted-mean-loss penalty1 and an unpenalized intercept.

This introduces the learned linear selector family, plus one three-seed
ordinary donor-label-permutation control. Target-session exclusion precedes
all donor feature/label/scaling use; matching24/32 initial views keep target
validation rows out of selection. All20 target rankers/orderings freeze before
any later target row; all65 actual target models precede evaluation. Scarce
trajectories reuse the qualified immutable selected-channel reader and later
initializer-conditioned CV. The comparison matches current target acquisition
to the initial marginal reference, while explicitly adding donor information,
reads, fits and storage. It is within-two-animal session cross-fitting, not
new-animal or chronological transfer, an OS seal, or a coupling endpoint.

Independent proposal/core/acquisition review **PASS** found no actionable
information-boundary or selected-channel/CV defect. Root ran the seven new
[scientific fixtures](tools/test_learned_electrode_value.py) in allocation
**46440428**: **all PASS in 0.027s**. They cover target/view exclusion before
numeric use, animal/session/electrode weighting and weighted ridge, analytic
24/32-row mean-loss donor gains with negative labels retained, exact three
predictors/raw-Y scaling, ordinary identity label draws and seed records,
physical-ID ties, and three-seed correct/wrong SSE means with a common
denominator. Root's full core/orchestration/runner static review **PASS**
agrees with the fixed proposal, including deadlines, complete target-model
barriers, count checks, timings-only stdout and preservation of partial reports.
Independent orchestration review also found no actionable barrier or CV defect.
Changed ASTRA schema/semantic validation passed with no warnings in the same
allocation (`astra/v0.0.14/validation_20261004_learned_electrode_value.json`).
Root starts the selected pilot at00:13UTC with approximately16minutes left in
the verified parent allocation. No worker executed tests or opened source
values. Root owns this log/navigation for launch and measured closeout.

Root selected the complete first declared target, `Mihili_CO_20140217.mat`,
with all seven donors/all views/all three control seeds: one CPU in the
attended4GiB parent,270s internal/300s external, only with at least five minutes
remaining after checks. This pilot computes3,360 donor labels,20 rankers,
65 target models/1300 inner systems,1536 fitting-read calls and40 evaluation
rows. Only completion, timing, memory and exposure are released for resource
selection. Its scientific scores remain withheld until the full cohort;
old outcome references are first read after all remaining new fits. The compact
pilot score record can be reused without serialized source arrays or decoder
coefficients. No remaining-session bound is selected yet.

```bash
env -u PYTHONPATH "$EP_AST_BIN" validate outputs/astra/v0.0.14/astra.yaml --skip-evidence --json > outputs/astra/v0.0.14/validation_20261004_learned_electrode_value.json
timeout 300s python3.12 outputs/tools/development_learned_electrodes.py --mode pilot --source-root "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" --metadata-report outputs/runs/source_support/attempt002/metadata_support.json --trial-roles outputs/runs/source_support/development_trial_roles_attempt001/development_trial_roles.json --max-seconds 270 --output-dir outputs/runs/development_search/learned_electrode_pilot_attempt001
```

The pilot alone charges no completed comparison; it is part of the eventual
two complete panels. Complete candidate/control panels would advance21to23,
not count seeds, donor fits, rankers or cells as extra evaluations. Shared
budget, required coverage, patience and the single closed held opening remain
unchanged. There is no automatic retry, requeue or extra source copy.

## 2026-10-03 23:57–23:58 UTC — `development_search`: grouping ablation reduces the mapping loss

Root released the [complete grouping-control report](runs/development_search/mapping_grouping_control_attempt001/mapping_grouping_control.json)
after its 23:57:07.105843 UTC observation. The runner completed in
**732.736110839s**, with process maximum RSS **401,552KiB**. Slurm job
**46439887** completed with exit0:0, elapsed12:28, CPU11:55.199 and accounting
maximum RSS891,340KiB. These process and job measurements have different
scopes and are retained separately. All required work completed: 2560 inner
systems, 4864 refits, 98,304 grouping decisions, 102,400 fitting reads and
320 evaluation reads. Endpoint score replay is zero; no held session was
accessed. Charge **one** complete control, bringing the shared count to **21**;
three Monte Carlo seeds are not three methods.

The three-seed mean control Q is **0.200320811382**: −0.001845134990 versus
balance, −0.002280833735 versus mean-value, and **+0.004502659106** versus the
original mapping heuristic. Against mapping, Chewie improves+0.004549424333
and Mihili+0.004455893880. Both remain below balance (−0.001991280885 and
−0.001698989094). Every session mean is below balance and mean-value. Seven
of eight sessions improve on mapping; Chewie20161006 instead loses
−0.001043043892. All four order means improve on mapping (original/0/1/2:
+0.004016622039,+0.005055283012,+0.005422610592,+0.003516120781), while all four
remain below balance. Every seed, session and resource cell remains in the
report. Root's final sensitivity extraction in allocation46440428 found seed
0/1/2 Q values0.200266620206/0.200245998850/0.200449815091. Their contrasts
against balance are−0.001899326166/−0.001919947522/−0.001716131281 and against
mapping+0.004448467930/+0.004427846574/+0.004631662815. For every seed, both
animal means improve on mapping and lose to balance, and every session loses
to balance. All eight fixed-prediction omissions preserve both-animal losses
to balance/mean-value and gains over mapping. Overall omission gains over
mapping range+0.003099420487to+0.005434737144; contrasts against balance range
−0.001934574334to−0.001710484522. Seeds are sensitivity draws, not biological
replication or independent method counts.

Randomizing the acquired-row utility groups reduces this heuristic's loss but
does not rescue it. This is evidence about the policy's direction-grouped
empirical utility estimates. The policies acquire different later histories,
and random groups also change count-dependent utility variance; the result
does not uniquely identify a biological mechanism, calibrated uncertainty or
sensor–trial coupling. It is not a permutation-tail significance test. The
40-evaluation minimum, diverse-family coverage, shared112ceiling and single
closed held opening remain unchanged.

The next [learned-electrode-value proposal](learned_electrode_value_proposal.md)
was prepared without opening this result and was still unadopted at this
closeout; root subsequently adopted it as recorded above. It proposes
session-cross-fitted donor supervision, with matched target-session acquisition
cost and explicitly greater donor training information/cost.

## 2026-10-03 23:47 UTC — `development_search`: completed mapping result visualized

Root rendered the frozen [plotting script](tools/plot_mapping_variance.py) in
allocation **46438625**, reading only the complete mapping report. The
[PNG](figures/mapping_variance_attempt001.png),
[PDF](figures/mapping_variance_attempt001.pdf) and
[source-value CSV](figures/mapping_variance_attempt001.csv) retain all session
and order effects and both animals' 32/48/64/80/96-trial profiles. The
[provenance sidecar](figures/mapping_variance_attempt001.json) records the
source report checksum and the 124 plotted rows. Support and contrast replay
checks passed; the source endpoint score replay difference is zero. Root
visually inspected the PNG: no clipping, and shared endpoint contrasts are
zero. No confidence intervals, new source values, fits or valid evaluations
were introduced; the development total remains 20.

```bash
python3.12 outputs/tools/plot_mapping_variance.py --report outputs/runs/development_search/mapping_variance_attempt001/mapping_variance.json --output-stem outputs/figures/mapping_variance_attempt001
```

## 2026-10-03 23:41–23:42 UTC — `development_search`: grouping-control checks and accepted batch

Root ran the six new scientific fixtures in the actual compute shell on
sh03-06n50, allocation **46438625**. The first attempt passed five fixtures
but the sixth could not import the new runner: `aggregate` was missing one
closing dictionary brace. The worker repaired only that syntax; the original
independent static review had missed it. Root then reran the complete focused
suite: **all six PASS in 0.040s**. Root's full static review is accepted.

The changed ASTRA packet passed schema and semantic validation with no
warnings; the saved result is
[validation_20261003_mapping_grouping_control.json](astra/v0.0.14/validation_20261003_mapping_grouping_control.json).
`bash -n outputs/tools/mapping_grouping_control01.sbatch` also passed.
No scientific protocol, seed, information boundary, output gate or resource
bound changed. These completed checks are reused for the selected substantive
batch, without duplicate tests in that batch. Its fresh main output directory
was absent before submission. At 23:42 UTC root submitted the exact selected
20-minute, one-CPU/2GiB script once with
`sbatch --parsable outputs/tools/mapping_grouping_control01.sbatch`;
Slurm accepted job **46439887**. The batch has no requeue or automatic retry,
and root alone monitors it. Acceptance is not a completed comparison: the
valid evaluation count remains 20 pending a complete report.

## 2026-10-03 — `development_search`: acquired-row grouping control adopted and selected

After the complete mapping result below, root adopted the unchanged
[grouping-control proposal](mapping_grouping_control_proposal.md). Its three
MC seed labels0/1/2 and complete scope were selected before that result;
adoption now asks whether the real direction grouping contributes to the
observed loss, with no rescue claim or new named explanatory endpoint.
The new [core](tools/mapping_grouping_control.py) and
[runner](tools/development_mapping_grouping.py) preserve every prior source,
decoder, allocator and result file. The new policy uses all eight development
sessions, all16fixed sets, all four orders and all20resource cells.

At every own-history choice, uniform pseudo-label assignment preserves the
actual direction-count multiset and all eight groups, including exhausted
directions. It changes only acquired-row utility grouping. True initializer
coordinates, M/eta/A at the same history, true labels/targets and decoder
fitting are unchanged. Canonical original-ID string order and a fresh full
filename/order/E/subset/MC/n seed key make assignments independent of prior
RNG calls and incidental row order. Actual queue legality and the old tie-break
apply after grouping. Histories may diverge across policies/seeds; this is a
policy ablation with count-dependent randomized utility variance, not an exact
significance null, calibrated uncertainty test or coupling result.

Identical initializer/final fits are shared across orders/seeds; intermediate
models use each history's actual acquired data. Three correct/wrong SSEs are
averaged within each matched set/order/checkpoint before subset/order/session/
animal aggregation; predictions are never averaged. Full actual group maps
are streamed to eight session JSONL files, with per-choice links in the result.
All old mapping/allocator/arrival packets enter trusted reporting only after
every new fit. Qualified Q-score replay uses the existing1e−10 tolerance and
scale-aware SSE/normalizer checks. Prior reports contain scores rather than
prediction arrays; no old prediction-array identity is claimed. Canonical
final X/Y/direction arrays match exactly within the new run before fit reuse.

An independent static review of the new core and full runner **PASS** found no
actionable defect in information timing, separate histories, true-target
preservation, grouped SSE aggregation or post-fit reference reuse. Six
independently authored [scientific fixtures](tools/test_mapping_grouping_control.py)
cover count/matrix/weighted-utility invariance, original-ID/RNG independence,
exhausted-group inclusion and legal ties, chance unchanged assignments without
redraw, chosen-next-only exposure with unchanged targets and separate histories,
and the exact three-seed SSE mean/normalizer. They have not run in a worker.

Root selected1CPU/2GiB/20minutes,1140s internal/1200s external, one numeric
thread, normal partition with the qualified`CPU_SKU:7502`, no GPU/requeue/retry.
The completed mapping cost271.200377s with273,288KiB RSS. Prospective control
cost is128initial CVs/2560inner systems,4864actual refits,102,400fitting reads,
320evaluation reads and98,304pre-choice updates; roughly2.7–3times the mapping
work plus group logging. All observation costs count in each history even when
identical initial/final computation is reused. The resource ceiling is a
bound, not a measured control runtime.

Root will run the new focused checks and changed ASTRA validation in its
attended compute PTY, then reuse them in the substantive batch:

```bash
PYTHONPATH="$PWD/outputs/tools${PYTHONPATH:+:$PYTHONPATH}" python3.12 -m unittest discover -s outputs/tools -p test_mapping_grouping_control.py -v
env -u PYTHONPATH "$EP_AST_BIN" validate outputs/astra/v0.0.14/astra.yaml --skip-evidence --json > outputs/astra/v0.0.14/validation_20261003_mapping_grouping_control.json
bash -n outputs/tools/mapping_grouping_control01.sbatch
mkdir -p outputs/runs/development_search/mapping_grouping_control_logs01
sbatch --parsable outputs/tools/mapping_grouping_control01.sbatch
```

The [new Slurm script](tools/mapping_grouping_control01.sbatch) contains only
the finite substantive command and the exact already qualified module stack.
Its logs have a separate sibling directory so the runner's fresh
`mapping_grouping_control_attempt001` output remains absent before launch.
Root alone submits and monitors. No test, source read, new score or scheduler
operation was performed by the worker. The new ASTRA output remains under
stable analysis ID`development_search`. No job ID exists at this selection
entry. One complete control comparison would bring20to21; three seeds do not
count as three methods. The40minimum/112ceiling, family-coverage obligations,
shared patience/resource limits and closed held sessions remain unchanged.

## 2026-10-03 23:27:17 UTC — `development_search`: mapping heuristic loses across all orders

The [complete mapping report](runs/development_search/mapping_variance_attempt001/mapping_variance.json)
in root allocation46438625 records elapsed271.200377s, maximum
RSS273,288KiB and observed23:27:17.410798UTC. All eight development sessions,
four orders and20resource cells completed:2560inner systems,1792actual refits,
32,768proxy updates,36,864fitting-trial reads and320evaluation reads.
Logical neural bytes were387,075,840, with30,094,080unique logical bytes;
these are element-accounting quantities, not physical filesystem traffic.
Endpoint score replay is exactly zero. No held session was accessed. Charge
one valid mapping-policy comparison, bringing the shared total to **20**.

Mapping Q is0.195818152276, versus balance0.202165946372 and mean-value
0.202601645117. Mapping-minus-balance is **−0.006347794096** and
mapping-minus-mean-value is **−0.006783492841**. Against balance, the Chewie
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

## 2026-10-03 23:22 UTC — `development_search`: mapping allocation started

The selected attended request returned job **46438625** and waited in queue
without a duplicate request. Its actual shell now reports sh03-06n50.int,
SLURM_JOB_ID=46438625 and one CPU. The previous allocation 46436630 exited
cleanly: COMPLETED, exit0:0, elapsed34:36, aggregate CPU13:43.100 across its
separate episode tasks. Launch the unchanged complete mapping command below
in the new allocation, with its existing840s/900s experiment bounds and
fresh output. Passing fixtures and ASTRA checks are reused.

## 2026-10-03 23:18 UTC — `development_search`: full mapping panel selected for compute

Select the complete adopted four-order `mapping_design_value` panel, using
the exact command and fresh `mapping_variance_attempt001` destination below.
Reuse its four passing fixtures and ASTRA validation. Root will request one
finite attended 1 CPU / 4 GiB / 40-minute allocation, with this experiment
bounded to 840 seconds internally and 900 seconds externally. The allocation
also accommodates the separately selected EP05 temporal-control pilot and
EP04 reliability/NC pilot if their implementation reviews are complete; their
scientific scopes and records remain in their own episodes. All work is
sequential, numerical threads one, with no GPU or automatic renewal. The
previous attended shell is closed before requesting the new allocation.
This entry selects execution, not completion or a twentieth valid result.

## 2026-10-03 23:16 UTC — `development_search`: mapping implementation checks passed

Root independently reviewed the adopted proposal, all new numerical and
reporting code, and the focused fixtures. The fixed-coordinate rank-one
updates, squared-factor utility, acquired-row-only interface, all-models-before-
evaluation ordering and post-fit reference reuse preserve the selected recipe.
All four focused tests passed in 0.774 seconds in allocation 46436630 on
sh03-06n50.int. The changed ASTRA program also passed its pinned schema and
semantic validation: `astra/v0.0.14/validation_20261003_mapping_variance.json`.
No scientific implementation changed after these checks. The complete
15-minute panel will use a subsequent bounded substantive allocation;
the present allocation has insufficient time for its unchanged full envelope.
The current valid count remains 19 until that panel completes.

## 2026-10-03 23:05 UTC — development_search arrival-order result and mapping successor

The full `arrival_order_attempt001/arrival_order.json` completed in root-owned
allocation 46436630, observed 22:58:55.617 UTC, elapsed 570.718449 seconds,
maximum RSS 203,268 KiB. It contains all eight development sessions and three
fixed post-initializer orders: 2,560 inner ridge systems, 3,712 actual refits,
77,824 fitting-trial reads and 320 evaluation-trial reads; 807,325,440 logical
neural bytes read versus 30,094,080 unique logical bytes. These are logical
element counts, not physical storage traffic. Shared endpoint replay is
exactly zero. No held session was accessed. Charge one valid robustness
comparison and zero new method families, bringing the shared total to **19**.

The complete three-order mean-value gain is +0.000363758426 Q, or
+0.000606264043 for the intermediate checkpoints, below the adopted 0.001
search resolution. Chewie and Mihili means are +0.000480528279 and
+0.000246988572. Order 0 Mihili is negative (−0.000707825246), as is order 2
Chewie (−0.000384332266). Omitting Chewie20161006 leaves +0.000029010236
overall and −0.000188968100 in Chewie. Mean-error is negative in both animals
and all three order aggregates, with −0.003908757943 overall. All order,
session, animal and resource-cell outcomes remain in the complete report.
These fixed-prediction omissions are sensitivity descriptions, not retraining
or independent animal confidence intervals. The original-order result remains
separate from the three-order mean. This does not establish robust useful trial
value or a coupling mechanism; required exploration and controls remain open.

After this complete result, root prospectively adopted
`mapping_variance_proposal.md`: `mapping_design_value` on all four fixed orders,
using only acquired LFP coordinates and fixed initializer M/eta, with stable
numerator `||Z0 A^-1 z||^2 / 32`. The proxy's full-linear, fixed-coordinate,
unit-noise interpretation differs explicitly from the refitted reduced-rank
decoder. Existing mean-value, mean-error and balance results are reused only
in trusted post-fit reporting. No calibrated uncertainty, coupling claim,
future-row access, threshold change or held opening is introduced. Completion
would charge one new evaluation, bringing 19 to 20; no partial panel is charged
as a successful comparison.

Prepared and frozen for independent review: `tools/mapping_variance.py`,
`tools/development_mapping_variance.py` and `tools/test_mapping_variance.py`.
Four focused checks cover independent trace-difference algebra through repeated
updates, fixed-coordinate/zero-design behavior, choosing before a source row is
revealed, and exact reference identity/capacity/endpoint failures. No test or
source operation ran in the worker shell. Root owns execution. The new panel
retains 2,560 inner systems, 1,792 actual refits, 32,768 pre-choice updates,
36,864 fitting reads and 320 evaluation reads. Based on fewer reads/refits than
the measured 571-second arrival panel plus bounded rank-one updates, prepare
1 CPU / 2 GiB / 15 minutes, 840 seconds internal; this is an unmeasured cost
estimate, not a time guarantee. No separate checking-only job is selected.

Exact commands, from the episode in a verified root-owned allocation, with
numeric threads one and the existing module PYTHONPATH preserved:

```bash
PYTHONPATH="$PWD/outputs/tools${PYTHONPATH:+:$PYTHONPATH}" python3.12 -m unittest discover -s outputs/tools -p test_mapping_variance.py -v
env -u PYTHONPATH "$EP_AST_BIN" validate outputs/astra/v0.0.14/astra.yaml --skip-evidence --json > outputs/astra/v0.0.14/validation_20261003_mapping_variance.json
timeout 900s python3.12 outputs/tools/development_mapping_variance.py \
  --source-root "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
  --metadata-report outputs/runs/source_support/attempt002/metadata_support.json \
  --trial-roles outputs/runs/source_support/development_trial_roles_attempt001/development_trial_roles.json \
  --original-report outputs/runs/development_search/trial_allocation_attempt001/trial_allocation.json \
  --arrival-report outputs/runs/development_search/arrival_order_attempt001/arrival_order.json \
  --max-seconds 840 \
  --output-dir outputs/runs/development_search/mapping_variance_attempt001
```

No mapping process has started. Existing source readers and completed allocator
implementations/results remain unchanged. The ASTRA program is changed and
frozen for one official validation after review.

## 2026-10-03 22:20 UTC — acquired-only development comparison selected

ASTRA `acquired_only_comparison`: run the already adopted and independently
reviewed acquired-only recipe in attended allocation `46433944`, one CPU,
2 GiB and a fifteen-minute step, with internal 840-second bound. Run the two
focused checks first, then `development_acquired_only.py` using the completed
metadata, frozen development trial roles, residual-selector and capacity
reports. Output is `runs/development_search/acquired_only_attempt001`.
This retains all eight development sessions and twenty fixed training/electrode
cells, exposes no held session, and charges one valid comparison only if the
whole panel completes. The initial 32-trial all-electrode screen and subsequent
restricted observations are counted separately from the fixed-random reference.

## 2026-10-03 — source_support / simple_comparisons shared fitter prepared

Before the unrun cost pilot, the coordinator selected a small shared-fitter
refactor so its measured implementation is the eventual fixed baseline fitter.
`tools/ridge_training.py` accepts only copied training arrays and relative
inner-fold indices. `tools/development_design.py` fixes the declared seeds,
resource cells and training-only row mappings. The existing selective reader
moved to `tools/development_reader.py`, with explicit disjoint training and
trusted-evaluation entrypoints restricted to the eight development filenames.
No scientific recipe or source identity changed.

`tools/development_baseline.py` is now prepared as the trusted evaluator.
It finishes all 48 models for a session before reading that session's reserved
40 evaluation rows, checks evaluation guides against the fitting guides,
and scores without another fit. The fixed fitter receives no source path,
source trial IDs, evaluation arrays or scoring callback; no candidate plugin
is accepted. This is actual callable/dataflow separation within trusted code,
not an OS seal against other code running with the same UID. Every source
path is one of the eight development files. Held-session access and the named
explanatory amendment remain closed. The current baseline does not implement
response-adaptive actions or claim their leakage check/efficiency result.

The full panel requires 48 final models and 960 inner fits per session,
8,064 ridge fits in total. Five subsets share the all-96 fit exactly; all cells,
sessions, both animals and paired resource-axis increments remain reported.
A completed panel charges one shared valid evaluation. The evaluator has not
run and its wall/resource selection awaits the unchanged bounded training-only
pilot. No global method search or held opening is selected. Partial failed
panels preserve read/fit counts and any completed session reports but do not
silently exclude sessions or count as complete scientific evaluations.

Three focused numerical tests are prepared, unrun: the original independent
augmented-least-squares/row-replication check; direct least-squares reference
for inner scaling/alpha selection and the training-mean-null score; and
rejection of out-of-array or duplicated validation folds. They test the
new shared API and scoring definition with synthetic arrays only. Independent source review of all five production modules and the three
focused tests found no concrete boundary, fold mapping, objective, reuse or
aggregation defect. Production source remains frozen for the coordinator
to execute the tests and still-unrun pilot. The ASTRA program now
links the adopted recipe, completed roles, cost pilot and fixed baseline;
validation of these latest recipe changes remains to run on compute.

## 2026-10-03 — prospective first development comparison and trial handoff

The coordinator requested a concrete comparison under the already-approved
open exploration scope. The inactive
[first-comparison recipe](development_comparison_proposal.md) specifies a
metadata-seeded five-trial-per-direction evaluation reserve, nested balanced
32/64/96-trial fitting sets, random nested electrode sets, training-only ridge
tuning/scaling, a fixed training-mean-null endpoint and a prospective search
progress rule. One complete method panel, including seeds and resource cells,
would charge one valid evaluation to the unchanged shared 40--112 allowance;
no scientific winner or held opening follows from this initial baseline.
The coordinator adopted the prospective recipe under the approved comparison
scope; adoption was recorded at 20:35:55 UTC. Its 0.001-Q progress threshold,
40--112 shared valid-evaluation allowance and 20-evaluation post-minimum
patience are explicit. No scored search, named explanatory amendment or held
opening was activated.

Prepared `tools/prepare_development_trial_roles.py` as a metadata-only consumer
of the completed source-support report. It preserves all eight assigned
development sessions, creates no held-session trial roles, and reads no MAT
file or neural element. The coordinator executed it in job 46429138 on
sh03-06n50.int: the report timestamp is 20:35:38.059729 UTC, with eight sessions,
320 reserved evaluation trials and zero source-MAT/neural access. Exact report:
`runs/source_support/development_trial_roles_attempt001/development_trial_roles.json`.
No source copy occurred. Ordinary file separation is not an OS-level
candidate/evaluator seal. The official ASTRA schema and semantic validator subsequently passed; see
`astra/v0.0.14/validation_20261003_development_roles.json`.

Prepared `tools/training_cost_pilot.py` and its focused independent mean-loss
ridge check. The selected next effect, after coordinator source review and
the current EP01 step, is a sequential one-CPU/two-GiB/ten-minute pilot in the
same owned allocation, with numeric threads one and a 540-second internal
deadline. It reads only the selected 96 fitting-trial windows of
Mihili20140217; all 40 evaluation trials and every held session remain closed.
Inner training-pool CV fits the prescribed ridge choices; its scores, alpha
choices and coefficients are not returned. The report contains source/logical
bytes, shapes, runtime/memory and numerical fit completion. No scientific
evaluation is charged for this unscored timing pilot, but actual resources
remain charged. Neither the check nor pilot has run as of this entry.
Independent source review found no read-boundary, fold, scaling, ridge-objective
or electrode-indexing defect. Before launch, it prompted finite/unique native
two-column unit-guide validation, cache-qualified own-process block-I/O counters,
and honest partial-fit counters on a timeout/failure. These changes preserve
the target, recipe and authorized reads; exact physical storage bytes remain
unavailable rather than being equated with logical HDF5 bytes.

The existing attempt002 metadata also establishes a concrete optional-reference
support failure: Chewie20161007 has 17 eligible trials in one direction, while
balanced 128 additional trials require 16 per direction plus 2 pilot trials,
or 18 even before untouched evaluation. Keep this development session and the
complete-grid failure; do not drop a cell or treat it as a failure of open
exploration. Bounded source-file stats give 3,315,332,161 bytes across the eight
development MATs. No bulk copy is selected without a numeric consumer.

## 2026-10-03 — source_support / attempt001 selected

The scientist authorized portfolio resumption and completion. The original
public Dryad source is present at
`$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source`;
the earlier absence statements describe the episode handoff at writing time.
No sibling output is a scientific input. Selected work is the episode-local
`tools/source_support.py` metadata inspection of the twelve fixed primary dates.
It reads scalar identities/events and electrode guides and inspects neural
array shapes, without reading LFP/spike elements or computing scores.

Before execution, fix chronological first-four development/last-two held
sessions within each animal. This implements the declared four/two split
without outcome selection. Keep all twelve even if support fails; do not
substitute a better-looking session. Inspection reports availability only;
0/0-band meaning, finite values, target consistency, untouched trial roles,
overlap history and the candidate/scorer access boundary remain to qualify.
Chewie-L is the contract's cohort label, not independently proven by a
`monkey=Chewie` scalar. The reference remains optional and new explanatory
endpoints remain unactivated. Selected compute: existing task-owned allocation
46424502, at most one CPU/four GiB and ten minutes for this inspection. Output:
`runs/source_support/attempt001/metadata_support.json`. Actual outcome follows.

The inspector completed in compute allocation46424502 without neural element
access. Its original exact-angle decoder misclassified source direction codes:
Mihili20140217 and Chewie20160929 contain eight distinct near-octant values,
e.g.0.0007963293 instead of zero and2.358981665 instead of3π/4. An inspected
Mihili20140218 session uses exact octant angles. This is an adapter error,
not evidence that the required targets are missing. Preserve attempt001;
attempt002 decodes the nearest octant within0.01radians, far below half the
45-degree spacing, and records each original angle and deviation. No trial
outcome or score informed the repair. Run the same bounded metadata command
with output `runs/source_support/attempt002/metadata_support.json`.

Attempt002 completed in allocation46424502, step2: all12sessions have eight
metadata-supported directions,3090trials in total, complete10-bin M1 movement
windows and96channels carrying all three required guide codes. The smallest
direction has17trials (Chewie20161007); no metadata exclusion remained. No
geometry field was found. These are availability facts, not neural quality,
decoder accuracy, complete reference support or an activated endpoint.

The provider's [figure-generation code](https://github.com/BeNeuroLab/Relationship_between_latent_dynamics_and_LFP/blob/master/Generate_all_figures.m#L73-L78)
applies a moving mean to the zero-lower-bound band and moving squared-signal
averages to the other bands. Its [Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC9470163/)
defines LMP using that moving-mean operation. Together with the observed unique
zero/zero guide pair, this supports its LMP interpretation. The source provides
post-processed features; no raw-voltage hardware claim follows. The12dates
overlap the planned EP05 M1 corpus and EP07 source; their outputs are not used,
and this is not independent source/animal replication. Source exposure history
must remain explicit in later held-session interpretation.

The updated ASTRA source-support recipe passed the official pinned schema
and semantic validator in allocation46424502, step5. Report:
`astra/v0.0.14/validation_20261003_attempt004.json`; evidence retrieval was
skipped. The validator used its own environment with module-injected
`PYTHONPATH` removed. No neural outcome or held-session opening occurred.

## 2026-10-02 — Concept figure v6

Built-in imagegen locally corrected [v5 to v6](figures/ep08_episode_series-v6.png):
equal schematic point counts in C and an actual equal-size electrode swap in D.
The [exact prompt](figures/ep08_episode_series-v6-prompt.md) is saved.
GOAL, paper plan and outputs README now display v6; prior assets remain.
No scientific endpoint, pilot requirement, contract, job or data access changed.

2026-10-02 — Local literature/design review clarified sensor–trial coupling versus aggregate factorial interaction; the optional reference decision is unchanged. See [review (consolidated)](paper_plan.md). Prior read-only BR retrieval for the cross-episode novelty question returned no relevant evidence; primary papers support this distinction. No scientific result, job, outcome access or runtime change.

No experiment has started. The source pack is absent, and no electrode value,
trial value, policy score, or held-session result has been computed.

## 2026-09-30 — Design milestone; no execution

The scientific writing now names conditional complementarity and reducible
uncertainty, with concrete linear proxies in
[acquisition_principle_proposal.md](acquisition_principle_proposal.md). Additional
explanatory scoring remains an unactivated amendment. The current concept PNG
and exact prompt are saved; old PNG/SVG assets are preserved.

The [ASTRA program](astra/v0.0.14/astra.yaml) now names existing required work:
`source_support`, `simple_comparisons`, `development_search`,
`internal_held_policy`, `conditional_electrode_value`,
`repeatable_direction_value`, `electrode_trial_coupling`, `resource_axes`,
`external_animal_confirmation`, and `final_report`. No node, attempt, or job
has started. The current scientific blocker for `source_support` is the absent
source pack; the next execution-stage action is acquisition and the stated
support checks when authorized. This writing task did not perform that action.
No third-animal source is available for `external_animal_confirmation`.

## 2026-09-30 — Scientist-approved open-scope amendment; no execution

[open_exploration_design.md](open_exploration_design.md) removes the episode-wide
separate-pilot prerequisite. The original 16-trial numerical protocol is an
optional reference. A new independent `offline_information_value` node and
branch-specific support/development descriptions reflect that scope.
Conditional/direction value work need not wait for a policy winner. No new
endpoint, numerical margin, recipe or finding is invented.

Current experiment/attempt: none. Process/Slurm IDs: none. Completed artifacts
are design documents and the built-in-imagegen v3 schematic/prompt, not data
or scores. Source data are still absent. Next execution-stage node:
`source_support`; exact next action, if separately authorized, is source
acquisition and branch-specific scientific support assessment. The former
pilot/grid requirements apply only to the reference. No experiments launched,
outcomes read, extra evaluation allowance granted, or Git mutation performed.

## 2026-09-30 — Concept-figure redesign

The scientist rejected the v3 workflow aesthetic and defensive in-image copy.
The new built-in-imagegen [v4 artwork](figures/ep08_scientific_context-v4.png)
uses three unboxed scientific illustrations; the
[prompt](figures/ep08_scientific_context-v4-prompt.md) is saved. GOAL, README and
paper-plan figure references/captions now use v4. Earlier artwork remains.
The experiment program, search policy and scientific design are unchanged.

## 2026-09-30 — Reference-guided episode-series figure

Following the scientist's request to refer to other episode images, EP05 and
EP12 were inspected and supplied directly to built-in imagegen as visual
references. The [v5 figure](figures/ep08_episode_series-v5.png) restores the
reaching/recording context and question-led four-panel composition. A focused
panel-B correction keeps candidate identity fixed and labels candidate versus
retained information correctly. The
[generation and correction prompts](figures/ep08_episode_series-v5-prompt.md)
are saved. Figure links/captions use v5; prior versions remain. Only artwork
and captions changed; the experiment program and scientific design did not.
## 2026-10-03 20:55 UTC — training pilot completed; full baseline selected

Coordinator allocation `46429138`, `sh03-06n50.int`, one CPU / 2 GiB:
the three focused shared-fitter/scoring tests passed (0.024 s). The unchanged
training-only pilot completed in 4.6323 s, including 4.5853 s selected-source
reading and 0.0470 s fitting, with 51,476 KiB peak RSS. All 240 inner fits and
12 training refits completed; 96 fitting trials / 2,549,760 logical bytes were
read. No evaluation rows, outer predictions/scores, or held sessions were read;
valid method evaluations charged: zero. Its compact report is preserved at
`runs/source_support/training_cost_pilot_job_46429138_attempt001/training_cost_pilot.json`.

Before scoring, select the adopted full eight-session random-subset baseline
at `runs/simple_comparisons/development_baseline_attempt001`, one CPU / 2 GiB
in this same allocation, with a 600-second internal bound. The full panel has
8,064 fits (32 times the pilot), at most 82 versus 44 native target columns,
and about 30.1 MB logical selected arrays. Even a doubled linear projection
of fitting cost is below six seconds; source I/O dominates and is variable,
so the ten-minute bound supplies ample measured-cost headroom without claiming
an exact physical I/O estimate. This is one development-method evaluation only
if all eight sessions finish. The four held sessions remain closed. The final
shared fitter/evaluator received independent code review with no concrete
defect. Source and prospective recipe are frozen for this attempt.

## 2026-10-03 — simple_comparisons — development baseline attempt 001 completed

The coordinator completed the adopted baseline in allocation **46429138** on
`sh03-06n50.int` at **20:55:33.366763 UTC**. The complete eight-session panel took
**25.73485 seconds**, with **54,952 KiB** peak RSS: **7,680 inner ridge fits +
384 refits**, **768 fitting + 320 evaluation trials**, and **30,094,080 logical
neural bytes**. All session models were fixed before that session's evaluation
rows were read. No held session was accessed. The report charges **one valid
method evaluation**, not 12 cells, five subsets, 384 models or 8,064 fits.
The three numerical checks and separate training-only pilot are recorded in
the immediately preceding entry and are not extra scientific evaluations.

Evidence: `runs/simple_comparisons/development_baseline_attempt001/development_baseline.json`.
The declared animal-balanced Q is **0.0887539298**: Chewie **0.0814713192** and
Mihili **0.0960365404**. All eight session means are positive. More electrodes
are not uniformly better: for Mihili_20140217 at 96 fitting trials, the
five-subset mean Q is **0.0564485881** with 16 electrodes and **0.0193279824**
with all 96. These are development observations under the existing complete
cell panel; no favorable session/cell is selected as a new endpoint. They
motivate a matched test of selection and redundancy, not a conclusion that
fewer electrodes are intrinsically superior. The full-source representation
and tuning can also explain a difference, and current Q may largely reflect
reach-direction means.

The proposed successor is five substantive selector/control panels under the
same endpoint and roles, with nested training-only selection. Additional
model/temporal/coverage comparisons and a direction-only null/residualization
check will follow development feedback. No successor has run. The optional
16-trial reference remains unavailable on its complete unchanged grid;
named conditional-value/next-direction/coupling endpoints remain unactivated.
The four held sessions stay closed and there is no scientific winner or
external-animal claim. Scheduler allocation-time charges remain coordinator
observations, distinct from this process runtime.

## 2026-10-03 — development_search — five selector panels prepared, unrun

The coordinator selected the successor after the completed baseline feedback:
direction-repeatability, marginal target association, target-agnostic block
D-optimal diversity, conditional target association, and a whole-trial target-
permuted conditional selector. Exact prospective definitions, fixed unit
shrinkage, stable tie rule, control seeds, information/cost boundary and the
unchanged Q are in `selector_comparison_proposal.md`. This is five family/
control panels, not an inflated hyperparameter or seed sweep. Four deterministic
selectors have one fit per scarce cell; the falsifier averages five whole-
multivariate-target permutations. Final decoders always use true targets.

`tools/selector_training.py` fits selection separately on each inner fitting
fold and then the full outer prefix; no validation/evaluation rows enter a
selector fit. `tools/development_selectors.py` fixes all five panels for a
session before its evaluation read, uses the existing eight-session reader,
and reuses the full-96 model/Q exactly across panels. The baseline report is
read for paired reporting only after all fitting/scoring finishes. Existing
baseline code and artifacts are unchanged. No source payload was read and no
new tests or numerical computation ran during implementation.

Three focused tests in `tools/test_selector_training.py` check the new
rank-three selector algebra against direct objective refits, perturb inner
validation rows to test the selection boundary, and compare fixed selection
against the unchanged ridge fitter with exact shared full-electrode identity.
Root's compute command is `python3 -m unittest discover -s outputs/tools
-p test_selector_training.py -v`. The earlier passing source/fitter checks
are reused, not repeated. The changed ASTRA program needs the normal official
compute validation, without making it an access or launch approval gate.

Proposed substantive command, following root review: one CPU, 2 GiB,
15 minutes, zero GPU and numeric threads one; internal bound 840 seconds.
The five panels contain 14,112 total decoder fits plus training-only selection,
versus the observed baseline's 8,064 fits/25.7 seconds; greedy-selection time
is not yet measured. Selection timing and completed counts are reported.

```bash
python3 outputs/tools/development_selectors.py \
  --source-root "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
  --metadata-report outputs/runs/source_support/attempt002/metadata_support.json \
  --trial-roles outputs/runs/source_support/development_trial_roles_attempt001/development_trial_roles.json \
  --baseline-report outputs/runs/simple_comparisons/development_baseline_attempt001/development_baseline.json \
  --max-seconds 840 \
  --output-dir outputs/runs/development_search/selectors_attempt001
```

Root owns execution and shared accounting. Completion would charge five valid
panels (six total including the baseline); failure retains partial exposure and
resources without pretending completed evaluations. Later temporal/model,
coverage/replay, direction-only-null/residual and required alternative checks
will be selected from meaningful feedback within the shared 40–112 allowance.
No named explanatory endpoint, causal/acquisition-efficiency claim, final
winner, additional held opening or fixed 40-panel menu is introduced.
## 2026-10-03 21:11 UTC — selected five-panel execution and grouped allocation

Root and an independent worker reviewed the prospective selector recipe and
its new pure training-array fitter/evaluator without a concrete launch blocker.
Select the three focused tests, then the unchanged five-panel command with
`--max-seconds 840`, one numerical thread, one CPU and 2 GiB. Preserve the
completed random baseline and its one evaluation; this successor charges five
only if all panels finish. No held session is opened.

Request one bounded attended normal/AMD7502 allocation, one CPU / 2 GiB /
40 minutes, grouping this substantive selector comparison with the already
selected EP19 peripheral pilot, EP04 metadata preparation, EP01 terminal-report
pooling, the EP08 baseline figure, and ASTRA validation. The preceding attended
job `46429138` was stopped after the unrelated EP03 metadata-read timeout;
that source route will not be retried here. Completed outputs from that job
remain valid. This request does not extend a prior job, rerun an existing
writer, or add any new EP01 recovery replicate. Root owns the PTY and all
scientific execution; no automatic renewal or background monitor is created.

The initial request used the descriptive hardware label `AMD7502` as a literal
constraint and was rejected before allocation (`Invalid feature specification`).
Fresh `sh_node_feat` discovery confirms the exact supported feature is
`CPU_SKU:7502`; `sh_part` confirms normal's limits. Correct only that request
spelling; no job was created or scientific code run by the rejected request.

Corrected submission returned attended job `46430982`; at 21:15 UTC the PTY
verified `sh03-06n50.int`, `SLURM_JOB_ID=46430982` and one allocated CPU.
After the focused tests, the selected full selector command runs in a separate
15-minute Slurm step, with an 840-second internal deadline and output
`runs/development_search/selector_comparison_attempt001`. Root owns its CPU;
other prepared science waits for this step to finish.

## 2026-10-03 21:16 UTC — development_search: five selector panels completed

The selected command completed in allocation **46430982**, step 1, on
`sh03-06n50.int`; the report timestamp is 21:16:24.032320 UTC. The coordinator
reported all three focused selector tests passed. The exact durable report is
`runs/development_search/selector_comparison_attempt001/development_selectors.json`
(the earlier proposed `selectors_attempt001` path was not the selected run).

| Panel | Animal-balanced Q | Gain from fixed random baseline | Chewie Q | Mihili Q |
| --- | ---: | ---: | ---: | ---: |
| Direction repeatability | 0.148469 | +0.059715 | 0.155663 | 0.141274 |
| Marginal target association | 0.144147 | +0.055394 | 0.155289 | 0.133006 |
| Conditional target association | 0.128715 | +0.039961 | 0.130232 | 0.127199 |
| Target-permuted conditional selector | 0.065023 | -0.023730 | 0.058244 | 0.071802 |
| Target-agnostic diversity | 0.046613 | -0.042141 | 0.043163 | 0.050063 |

The preserved random baseline is Q 0.088754. The first three selectors improve
the mean in every development session; diversity and the permuted-selector
control worsen it in every session. True-target conditional selection beats
its target-permuted control in all eight sessions. That comparison breaks
direction-level as well as individual-trial target linkage, so it does not
establish information beyond reach-direction means. Direction repeatability
is the strongest current aggregate and exceeds conditional selection in seven
of eight sessions. It meets the existing 0.001-Q search-progress resolution
relative to the baseline; it is not a final winner or a biological-margin test.
The target-agnostic diversity result indicates that covering variation in these
standardized features need not preserve useful population prediction.

Observed workload: **39.6713556 seconds**, **110,596 KiB** peak RSS,
13,440 inner ridge fits plus 672 refits = **14,112 ridge fits**, and **1,080
selector fits**, with 22.2629017 seconds attributed to selection. Input was
768 fitting trials and 320 reserved development evaluation trials, totaling
30,094,080 logical neural-array bytes. The 1,288 zero-SD target-column fit
occurrences are repeated fit occurrences, not distinct removed units; all
targets were retained. The single numerical thread occupied one allocated
step CPU. Its 39.671-second worker runtime is the attributable compute
observation; coordinator-owned terminal Slurm accounting includes step startup
and shared-allocation overhead without double-counting the other episodes.

Charge **five valid panels**, making **six total** with the completed baseline.
No held session was opened, no named explanatory amendment was activated, and
the global 40–112-evaluation search is incomplete. All96 models/scores were
shared as prescribed. The training-only array API and trusted evaluator
boundary were maintained, with the already disclosed same-UID filesystem
limitation. This is offline all96-training-electrode exposure, not evidence
for reduced acquisition cost.

The immediate next scientific question is whether this predictive advantage
survives accounting for reach direction. A prospective direction-mean null,
fixed within-direction prediction-pairing control, and direction-residual ridge
panel are proposed separately for coordinator review before implementation.
No new scientific fit, score, source access, scheduler query or test was run
while interpreting this completed report.

## 2026-10-03 21:25:58 UTC — development_search: D/W/R adopted and selected before execution

The coordinator adopted `direction_confounding_proposal.md` prospectively under
open exploration after the five-selector results. Its first timestamp was
corrected to the observed **21:25:58 UTC**; no recipe changed. Select one grouped
three-panel attempt: fitted direction-only target means (D), exact within-
direction wrong-trial loss on the two unchanged reference selectors (W), and
random-electrode direction-residual ridge (R). All prior baseline/selector
code, results, roles, seeds, target/window and four held sessions are preserved.

Prepared implementation: `tools/direction_training.py` receives fitting arrays
only, `tools/direction_control_metrics.py` belongs to the trusted evaluator,
and `tools/development_direction_controls.py` completes every selected model
before that session's evaluation read. The historical selector report enters
only after all new fits/evaluations to check correct-pair replay; no W values
are written unless all corresponding cell scores agree within absolute 1e-10.
D uses zero LFP electrodes and only three actual direction-mean models/session;
the grouped offline worker still reads all96 training electrodes for W/R.
R fits all means/scales inside each inner fitting fold, retains raw-Y SD target
weighting and true-target CV error, and refits on the complete allowed prefix.

Before source access, select exactly two focused checks: five-trial derangement
enumeration against the closed-form wrong-pair loss, and perturbation of inner
validation rows proving they cannot change fitting direction means/raw-Y SD.
The previously passed ridge/selector tests are reused, not rerun. Coordinator
compute command, from the EP08 directory with numeric threads one:

```bash
python3 -m unittest discover -s outputs/tools -p test_direction_controls.py -v
python3 outputs/tools/development_direction_controls.py \
  --source-root "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
  --metadata-report outputs/runs/source_support/attempt002/metadata_support.json \
  --trial-roles outputs/runs/source_support/development_trial_roles_attempt001/development_trial_roles.json \
  --selector-report outputs/runs/development_search/selector_comparison_attempt001/development_selectors.json \
  --max-seconds 540 \
  --output-dir outputs/runs/development_search/direction_controls_attempt001
```

Bound: one CPU, 2 GiB, ten-minute step, 540-second internal deadline, within the
coordinator's existing attended allocation if time remains. Estimated workload
is 11,592 ridge fits (11,040 inner, 552 outer), 240 selector fits and 24 D models;
prior 14,112-ridge/1,080-selector batch took 39.7 seconds. Exact runtime and
memory remain unobserved. Source access remains the fixed eight development
sessions' 768 fitting/320 evaluation trials through the qualified selective
reader; no source copies or held access. Completed D/W/R charges three panels,
bringing total six to nine only if the whole batch and replay check pass.
Partial failures preserve actual exposure/counters and do not invent completed
scientific evaluations. Root owns tests, launch and resource accounting. This
entry is prospective; no command above has yet been run by this worker.

Independent read-only review of the new recipe, four source/test files and
unchanged fitting/reader dependencies found no concrete launch blocker. It
confirmed the exact derangement mean, fitting-only direction means/raw-Y SD,
intercept reconstruction, unchanged reference replay, withheld W packet before
all tolerance checks, and the 11,592-ridge/24-direction-model accounting. This
was source review, not executed tests or source-payload access. The four new
files remain frozen for coordinator execution.

## 2026-10-03 21:32 UTC — development_search: D/W/R completed

Root completed the selected comparison in **46430982.9** on `sh03-06n50.int`,
exit zero; report timestamp 21:32:46.609812 UTC. Both focused tests passed in
0.031 seconds. Evidence is
`runs/development_search/direction_controls_attempt001/direction_controls.json`.

| Panel/readout | Animal-balanced Q | Wrong-pair Q | Correct-pair increment | Gain from direction-only |
| --- | ---: | ---: | ---: | ---: |
| Direction-only | 0.181856 | Same, up to rounding | 0 | 0 |
| Raw direction-repeatability selector | 0.148469 | 0.080148 | 0.068320 | -0.033387 |
| Raw conditional-target selector | 0.128715 | 0.060169 | 0.068546 | -0.053140 |
| Direction-residual random ridge | 0.188342 | 0.182093 | 0.006248 | 0.006486 |

Direction-only exceeds both raw selector means in seven of eight sessions;
Chewie 20161006 is the exception. However, both raw models have positive
correct-pair increments in all eight sessions: useful trial correspondence
survives within a direction. Thus the raw gains cannot be wholly explained by
direction labels, and their large pairing increments do not imply better
absolute prediction than the direction-only reference. The residual model
improves over direction-only in every session, with positive pairing increments
in every session. Its animal-average gain is 0.010520 for Chewie and 0.002452
for Mihili; pairing increments are 0.010574 and 0.001923 respectively. The
residual wrong-pair aggregate approximately returns to direction-only Q.
This supports modest within-direction predictive covariance in this representation;
it is not causal information, a sensor–trial coupling test, a prospective
acquisition-efficiency result, or an independent-animal confirmation.

Correct-pair replay maximum absolute difference was **2.220446049250313e-16**,
well below the prospectively fixed 1e-10 requirement. The batch used
**21.953039 seconds**, **110,608 KiB** peak RSS, **11,040 inner ridge fits +552
refits**, **240 selector fits** and **24 actual direction-only models**.
The 1,058 zero-SD target fit occurrences are repeated occurrences, not units
removed. Input was the same 768 training and 320 development evaluation trials,
30,094,080 logical neural bytes; source arrays/coefficients were not saved.
D itself used zero LFP electrodes. The grouped worker's all96 training-electrode
exposure is retained; same-UID callable separation is not an OS seal.

Charge **three** valid panels (D, W with two references, R), making **nine total**.
No held session opened; the named explanatory amendment remains inactive and
the shared search/coverage requirements are incomplete. At one allocated step
CPU, the worker runtime contributes 21.953039 CPU-seconds before coordinator
Slurm overhead accounting; do not double-count the shared allocation. The next
proposal compares temporal information loss and model capacity under the same
residual target/roles, rather than treating selector refinements as new evidence
for a coupling explanation. No successor was implemented or run during this
interpretation.

## 2026-10-03 21:36 UTC — development_search: temporal/rank/kernel comparison adopted

Root prospectively adopted `representation_capacity_proposal.md`, including
the exact constant-feature kernel limit and a common legal rank intersection
across inner folds, before any relevant scores. Independent methodological
review agreed with the regularized reduced-rank metric, centered-kernel formula
and whole-trial temporal representation. Code is prepared for source review;
no tests or source reads have occurred in this worker.

Select `capacity_comparison_attempt001`: T preserves all ten bins per trial;
L restricts residual response rank1/2/4/8/full using the regularized metric;
K uses a fixed training-median RBF bandwidth. Same native target means, direction
residualization/raw-Y SD, random electrode sets, roles, alpha grid and scoring.
All144 models for a session are fixed before evaluation access. The separate
trusted temporal reader differs from the frozen means reader only in retaining
its already permitted ten-bin tensor; the old reader and earlier fitters remain
unchanged. Fitting components receive copied fitting arrays; predictions belong
to the trusted evaluator. Historical D/W/R enters only final reporting.

Selected bounded compute commands from EP08 (all numeric threads one):

```bash
python3 -m unittest discover -s outputs/tools -p test_capacity_math.py -v
python3 outputs/tools/development_capacity.py \
  --source-root "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
  --metadata-report outputs/runs/source_support/attempt002/metadata_support.json \
  --trial-roles outputs/runs/source_support/development_trial_roles_attempt001/development_trial_roles.json \
  --direction-report outputs/runs/development_search/direction_controls_attempt001/direction_controls.json \
  --max-seconds 540 \
  --output-dir outputs/runs/development_search/capacity_comparison_attempt001
```

Two focused checks compare the reduced-rank regularized solution with an
independent augmented-design projection and centered-kernel prediction with
an explicit finite-feature ridge reference, including the constant limit.
No repeated role/reader/ridge preflight is selected. Root owns execution in
one CPU/2GiB/ten-minute step, only if its current attended allocation has time;
internal deadline540seconds. Expected24,192 ridge/kernel systems,1,152 outer
models and46,080 nested capacity candidate scores (the temporal model uses the
unchanged fitter and its inner scores are separately implicit in system counts).
L's up-to25 candidates versus5forT/K is recorded, not claimed equal tuning.
Same30,094,080 logical source-slice bytes are expected; actual cost remains
unmeasured. Whole-batch success charges3panels to move9to12; partial failure
retains exposure/counters and does not open held data or expand a failed grid.
The four new production files and two-test file are frozen for independent
source review before the coordinator runs them.

Independent static review of the four new production files and focused test file
found no concrete blocker: common legal rank intersection, regularized metric,
exact full-rank/constant-kernel limits, fit/predict centering, electrode/band/time
ordering, whole-trial roles and144fits-before-evaluation agree with the adopted
recipe. The two numerical references were reviewed but not executed by either
implementation/review worker. Root received the exact selected command; files
remain frozen for its compute step.

## 2026-10-03 21:40 UTC — development_search: temporal/rank/kernel completed

Root completed **46430982.12** on `sh03-06n50.int`, exit0; report timestamp
21:40:46.268403 UTC. Both independent objective/centering tests passed in0.002s.
Evidence: `runs/development_search/capacity_comparison_attempt001/capacity_comparison.json`.

| Panel | Q | Gain from mean residual ridge | Chewie gain | Mihili gain | Pairing increment |
| --- | ---: | ---: | ---: | ---: | ---: |
| Temporal residual ridge | 0.193312 | 0.004971 | 0.009570 | 0.000371 | 0.014180 |
| Reduced-rank residual ridge | 0.197790 | 0.009448 | 0.019881 | -0.000984 | 0.027249 |
| RBF residual kernel ridge | 0.191807 | 0.003465 | 0.007874 | -0.000944 | 0.016984 |

The unchanged mean residual ridge is0.188342 and direction-only is0.181856.
The reduced-rank model has the best animal-balanced aggregate; its gain is
uneven, especially large for Chewie20161006. It improves the mean-residual
reference in six sessions, temporal ridge in five, and kernel ridge in three.
Temporal ridge exceeds direction-only in all eight sessions; L and K each do
so in seven. All three have positive correct-pair increments in all sessions.
This supports examining retained temporal detail and response regularization,
not a universal decoder win or nonlinear biological mechanism. L also has up
to25nested candidates rather than5, a declared difference in tuning opportunity.
Full chosen rank/alpha/bandwidth records are retained for each fit; these are
not extra method evaluations or an invitation to expand a boundary grid.

Observed workload: **33.956798seconds**, **115,484KiB** peak RSS,
23,040inner+1,152outer = **24,192ridge/kernel systems**, and46,080nested candidate
scores for L/K. The unchanged temporal fitter has its standard inner scores in
addition. 2,208zero-SD target fit occurrences were retained without unit removal;
no zero-SD input columns occurred. The same768training+320evaluation trials and
30,094,080logical neural bytes were read, without source copies or saved neural
arrays/coefficients. Each temporal trial remained one row and all144models per
session preceded evaluation access. Charge3panels, making **12valid evaluations**;
held openings0, named explanatory amendment inactive, global search incomplete.
Single-CPU worker runtime is33.956798CPU-seconds before root's Slurm overhead
accounting. Source access and same-UID limitations remain unchanged.

Next informative comparison is residual-informed selection under one globally
fixed reduced-rank decoder: distinguish marginal residual association from
conditional residual gain and a direction-preserving target-link control.
This is proposed separately before implementation; no animal-specific winner,
new biological endpoint or sensor–trial coupling score is selected here.

## 2026-10-03 21:46 UTC — development_search: residual selector comparison adopted

Root adopted `residual_selection_proposal.md` before new fits. Prepare the three
selected panels under one global reduced-rank mean-feature decoder, preserving
raw-Y SD, fold-local residual selector orders, common rank grid, true-target
final decoder and direction-preserving whole-vector no-fixed-point selector
control. The code and focused boundary check are now frozen for independent
review; no tests, source access or scheduler activity by this worker.

Select the following commands from EP08, one numerical thread:

```bash
python3 -m unittest discover -s outputs/tools -p test_residual_selection.py -v
python3 outputs/tools/development_residual_selectors.py \
  --source-root "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
  --metadata-report outputs/runs/source_support/attempt002/metadata_support.json \
  --trial-roles outputs/runs/source_support/development_trial_roles_attempt001/development_trial_roles.json \
  --capacity-report outputs/runs/development_search/capacity_comparison_attempt001/capacity_comparison.json \
  --direction-report outputs/runs/development_search/direction_controls_attempt001/direction_controls.json \
  --max-seconds 540 \
  --output-dir outputs/runs/development_search/residual_selectors_attempt001
```

Resource request is one CPU/2GiB with at most ten-minute step/540-second internal
bound. Root may select a shorter finite internal/step bound to fit the remaining
attended allocation; this does not change any model, trial, seed or count.
If insufficient time remains, code stays prepared for a separately coordinated
substantive job; no worker submission/renewal or login test occurs. Expected
11,088systems,528actual refits,52,800nested scores,840selector fits. All72model
references (66actual fits) per session precede its evaluation read. All96scores
must replay within1e-10 before any new scientific packet is written. Source
access remains the same768training+320evaluation rows, all96offline exposure;
held opening0. Charge3only after complete success, moving12to15. Partial
technical failures preserve actual counters/exposure, never count as completed
panels or permit an outcome-adaptive retry. Existing numerical/source-reader
checks are reused; the one selected check addresses new raw-Y weighting,
fit-only ordering and the direction-preserving selector-only permutation.

**Prospective control correction before any execution:** root adopted ordinary
uniform within-direction whole-row permutations, including fixed points, in
place of the initial derangement proposal. No new tests, fits or outcomes had
been run. For centered direction groups of size m, derangement gives expected
partner -Y_i/(m-1), inducing negative covariance that squared selector scores
can exploit, particularly at m=3. Ordinary permutations have zero expected
centered covariance and better implement the intended null selector control.
The final third panel is `permuted_residual_conditional`; its seed prefix is
`ep08-residual-selector-permutation-v1`, with one uniform shuffle per group.
All other choices/counts and true-Y decoder fitting are unchanged. The focused
check now includes exact enumeration of the three-row ordinary-permutation
population, retaining the fit-only weighting and nonmutation checks. This does
not modify the earlier completed W evaluation-pairing control, whose estimand
is the descriptive expected wrong-trial loss, not a null training association.
The narrow three-file correction is frozen for independent review before launch.

## 2026-10-03 21:50 UTC — development_search: residual selection completed

Corrected ordinary-permutation source passed independent narrow review and its
one focused test (0.008s). Root selected **280-second internal / five-minute
step bounds** to fit the attended allocation and ran **46430982.15**, exit0,
report timestamp21:50:17.633588UTC. Evidence:
`runs/development_search/residual_selectors_attempt001/residual_selectors.json`.
No derangement-training recipe was tested or run; its pre-outcome correction
is preserved above. Earlier W evaluation wrong-pair controls remain unchanged.

| Panel | Q | Gain from matched random rank decoder | Chewie gain | Mihili gain | Pairing increment |
| --- | ---: | ---: | ---: | ---: | ---: |
| Residual marginal | 0.202708 | 0.004918 | 0.008954 | 0.000881 | 0.036600 |
| Residual conditional | 0.201434 | 0.003644 | 0.005946 | 0.001342 | 0.034348 |
| Within-direction permuted selector | 0.197046 | -0.000744 | -0.001133 | -0.000355 | 0.024364 |

Conditional selection exceeds its permutation control in all eight sessions,
but marginal selection leads in aggregate and in six of eight sessions. The
marginal method beats matched random rank decoding in six sessions; conditional
in four. The signal link matters to these selectors, but the additional
conditional update is not shown to improve over marginal residual association.
This is not evidence for a sensor–trial coupling mechanism. Both true-target
selectors have positive animal-average gains, with stronger gains in Chewie;
all96training exposure remains and no acquisition-efficiency claim follows.

All96replay max absolute difference **3.3306690738754696e-16** passed the fixed
1e-10requirement before report release. Observed52.232189seconds,110,608KiB peak
RSS,10,560inner+528outer=11,088systems,52,800nested rank/alpha scores,840selector
fits (21.389433seconds). 1,012zero-SD target fit occurrences retained all units;
zero input-SD occurrences. Same768training+320evaluation trials and30,094,080
logical neural bytes, no saved arrays/coefficients and no held opening.
Charge3, making **15valid evaluations**. The full search, required family/alternative
coverage and held-session selection remain incomplete. Single-CPU worker cost
52.232189CPU-seconds precedes root's distinct Slurm overhead accounting.

The next scientific gap is acquired-only behavior and explicit initializer
cost, followed by trial allocation/value. A separate proposal will keep the
current development evaluation boundary and report all initializer exposure;
it is unadopted and not part of the runnable ASTRA program. No new source access,
fit, test or policy choice follows automatically from these results.

## 2026-10-03 — development_search: acquired-only comparison adopted and prepared

Root adopted `acquired_only_proposal.md` prospectively after15completed evaluations.
The prior allocation46430982 expired at21:56UTC; no EP08 validation ran in its
remaining time, as the coordinator's other selected step occupied it. The old
PTY is not reused and no login-node test or automatic renewal occurs. Root
released the ASTRA edit freeze for this adopted implementation; final program
validation belongs in the next substantive allocation.

Prepared frozen files: `tools/acquired_only_reader.py`,
`tools/acquired_rank_training.py`, `tools/development_acquired_only.py` and
`tools/test_acquired_only.py`. A selects once from its32all96initializer rows,
then the trajectory reader slices only the frozen channels. B reads its fixed
random set from the first row. Both remain inside the exact existing96trial
pool; their next balanced checkpoint is the only allowed acquisition action.
After32, every initial row stays in every inner fitting set; validations
partition only newly acquired rows. All100model references/95actual rank fits
per session precede its40evaluation rows. Five distinct direction-only means
per session are shared, not repeatedly fit as extra methods. Past outcomes
enter reporting only for32-trial replay and are not policy inputs.

Selected commands for root's next coordinated substantive allocation:

```bash
python3 -m unittest discover -s outputs/tools -p test_acquired_only.py -v
python3 outputs/tools/development_acquired_only.py \
  --source-root "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
  --metadata-report outputs/runs/source_support/attempt002/metadata_support.json \
  --trial-roles outputs/runs/source_support/development_trial_roles_attempt001/development_trial_roles.json \
  --selection-report outputs/runs/development_search/residual_selectors_attempt001/residual_selectors.json \
  --capacity-report outputs/runs/development_search/capacity_comparison_attempt001/capacity_comparison.json \
  --max-seconds 840 \
  --output-dir outputs/runs/development_search/acquired_only_attempt001
```

One CPU/2GiB/15-minute step,840-second internal deadline, numerical threads1.
Estimated several-minute workload:760actual rank fits,15,960systems,76,000nested
scores,40initial selector fits and40direction-only models. Narrow repeated
views require13,824training read calls plus320evaluation calls; actual source
read sum and unique logical neural-byte union are reported separately. The
latter spans the same permitted96+40source trials/session. These estimates do
not replace observed cost; no tests or science have run for this attempt yet.

Two focused checks address the new selected-channel/exact-next-trial reader
boundary and prove every initializer row remains in all later fitting folds.
Passing earlier numerical projection/ridge/source checks are reused. The full
20cell comparison reports A exposure32*96+(n−32)*E and B exposure n*E, as well
as target observations; their initial acquisition cost is not matched. Its
one complete comparison charge would move15to16, not charge random replicas or
20cells separately. Four held sessions remain closed and no allocation/value
or coupling endpoint is silently activated. Independent source review pending.

Independent static review passed with no concrete launch blocker: exact chosen-
channel views, next16trial enforcement, initial-only/random source restrictions,
initializer-conditioned folds,95actual models before evaluation,32-trial replay,
and distinct policy/read-sum/unique exposure accounting agree with the recipe.
The header was updated to adopted/implemented/unrun; all scientific source and
test files remain frozen. Review did not execute tests or access payloads.

The coordinator reported schema/semantic ASTRA validation passed in new grouped
allocation46433944; evidence is
`astra/v0.0.14/validation_20261003_pre_recovery_group.json`. This validates the
adopted acquired-only program before its tests/run, not empirical completion.
The separate `trial_allocation_proposal.md` remains prospective and absent from
the runnable program. Its fixed set is conditional on the acquired-only result;
no allocator code, source access or ASTRA activation has been performed.

Root requested a plot consumer for this selected comparison. Prepared
`tools/plot_acquired_only.py`, unexecuted: after a complete report it renders
`figures/acquired_only_attempt001.png` and `.pdf`, with Q versus acquired trials
by retained electrode count and animal (four-session means plus faint session
traces), alongside report-derived LFP scalar exposure. It exposes the32×96
initial screen and unequal costs, retains the separate20-cell protocol and
adds no confidence intervals, refits, synthetic figures or source reads. The
scientific reader/fitter/evaluator/test files remain frozen for execution.

## 2026-10-03 22:22 UTC — development_search: acquired-only completed

Root executed both focused tests (2PASS,0.004seconds) and the complete selected
comparison in Slurm46433944 step4 onsh03-02n16.int. The durable
`runs/development_search/acquired_only_attempt001/acquired_only.json` finished
at22:22:23.163841UTC in133.452525seconds, peak RSS85,192KiB. All eight sessions
completed all twenty cells;95actual rank models/session were fixed before its
forty evaluation rows. Actual computation was15,200inner ridge fits,
760outer refits,76,000rank/alpha candidate scores,40initial selector fits and
40distinct direction-only means. Initial replay error was3.33e−16, within the
frozen1e−10 bound. The all96 branches share the same fit at every checkpoint.

The completed comparison charges **one** valid evaluation, moving15to16.
It read13,824fitting-trial views and320evaluation trials. Logical neural bytes
including repeated reads were148,903,680; their unique logical union was
30,094,080bytes. The OS block-input counter4,489,304 is cache-dependent and is
not an exact physical-byte estimate. No held session, named explanatory
amendment or held opening was used; no raw arrays or coefficients were written.
The same-UID trusted-reader boundary remains an information restriction, not
an OS seal. The current global search is incomplete.

All quantities below belong to this separate **twenty-cell** protocol, not the
earlier twelve-cell ranking. A denotes initial-only marginal selection; B is
the five-repetition fixed-random reference.

| Evidence unit | A mean Q | B mean Q | A minus B |
| --- | ---: | ---: | ---: |
| Animal-balanced |0.205033|0.201775|0.003258|
| Chewie |0.214867|0.209139|0.005728|
| Mihili |0.195200|0.194411|0.000789|
| Mihili20140217 |0.179159|0.180839|−0.001681|
| Mihili20140218 |0.123110|0.123372|−0.000262|
| Mihili20140303 |0.241937|0.239581|0.002356|
| Mihili20140304 |0.236594|0.233853|0.002742|
| Chewie20160929 |0.213415|0.215178|−0.001762|
| Chewie20161005 |0.205783|0.201586|0.004196|
| Chewie20161006 |0.289594|0.270123|0.019471|
| Chewie20161007 |0.150675|0.149669|0.001006|

A improves five of eight session means, with the largest contribution from
Chewie20161006. Individual scarce-electrode cells reverse sign across sessions
and checkpoints, rather than establishing one uniformly superior selected set.
All96 differences are exactly zero as required. A's within-direction pairing
increment is0.034792 versus B's0.031671; in Mihili alone it is smaller
(0.004920 versus0.005789). Thus a higher overall Q does not establish uniformly
stronger residual trial information. A and B improve the direction-only
reference by0.020429 and0.017171 respectively in the common twenty-cell mean.

Retained-electrode/trial matching does not match acquisition exposure. A uses
32×96+(n−32)×E electrode-trials, while B uses n×E. Each electrode-trial supplies
3bands×10native bins; population target observations are identical at each
checkpoint. The unpriced extra initializer exposure persists at every later
checkpoint:

| Retained E | A initial32 | B initial32 | Persistent extra | A at96 | B at96 |
| ---: | ---: | ---: | ---: | ---: | ---: |
|8|3072|256|2816|3584|768|
|16|3072|512|2560|4096|1536|
|32|3072|1024|2048|5120|3072|
|96|3072|3072|0|9216|9216|

**Recommendation, not adoption:** use the globally fixed-random branch with
its existing five scarce-set repetitions for the next trial-allocation
comparison. It provides a simpler, lower-observation-cost reference for
isolating trial choice; the present modest and heterogeneous Q gain does not
establish that screening all96electrodes is worth its extra cost. This is a
choice of the next informative experiment, not evidence that random sets are
universally better or that initial selection should be abandoned. Retain A as
a later robustness branch, without choosing differently by session or animal.
The recommendation uses the same repeatedly exposed development evaluation;
it is not an unbiased audit or a prospective acquisition claim.

`trial_allocation_proposal.md` remains unadopted and unimplemented pending
coordinator review. No new rank/alpha, utility, scoring rule or role is activated
by this result. Root is rendering the prepared source-report-only figure; the
figure will be linked after actual completion.

The coordinator subsequently rendered and viewed the source-report-only
`figures/acquired_only_attempt001.png` and`.pdf`; both are linked in navigation.
No new fit or uncertainty estimate was introduced by that display.

## 2026-10-03 — development_search: trial allocation adopted and prepared

Root adopted `trial_allocation_proposal.md` after the completed acquired-only
comparison, choosing fixed-random electrodes globally from trial1 with the
existing five scarce-set repetitions. The purpose is the explicit lower cost
and simpler conditioning history, not a claim of random-set superiority.
Mean-value and mean-error allocation compete with the balanced reference using
only already acquired target history, exactly the original finite96trial pool,
the same32balanced initializer and retained8/16/32/all96electrodes. The same
twenty checkpoints/cells and two animals are retained. Fixed-initial raw-Y SD
and global-mean null define common evaluator coordinates, so this comparison
cannot replace earlier rankings that used checkpoint-specific normalizers.

Rank and alpha are selected only at initialization and frozen thereafter.
The `full` alias resolves to the fixed structural integer min(3E,U,24), with
no new numerical rank threshold;1/2/4/8 remain literal. Initial coefficients
are preserved and later projections use that fixed cap. No later adaptive-row
CV occurs. Every trajectory stores rows in canonical direction/queue order;
initial and final fits are shared. Each policy must actually reach the exact
96row pool, and final acquired arrays must match before that shared fit is
reused. All240model references/176actual models and48complete trajectories per
session precede its40evaluation reads. Only two complete adaptive panels are
charged; successful completion would move16to18, with no charge for the
balanced reference, checkpoints, subsets or repetitions.

Prepared source: `tools/trial_allocation.py`, `tools/allocation_reader.py`,
`tools/development_trial_allocation.py`, `tools/test_trial_allocation.py`.
The trusted session reader opens one HDF handle and slices only the chosen next
trial on frozen channels; policies receive copied acquired targets, labels,
initial scales and counts, never future arrays, source paths or evaluation
data. Existing source roles and target/feature definitions are reused. Four
focused checks cover reducible-value versus total-error algebra/ties, exact
chosen-next-row access, non-growing rank with no later CV, and common evaluator
normalization despite policy-dependent fitting statistics. These are prepared,
not yet executed. Root source review passed; independent review is pending.

Selected root-only commands, episode cwd:

```bash
python3 -m unittest discover -s outputs/tools -p test_trial_allocation.py -v
python3 outputs/tools/development_trial_allocation.py \
  --source-root "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
  --metadata-report outputs/runs/source_support/attempt002/metadata_support.json \
  --trial-roles outputs/runs/source_support/development_trial_roles_attempt001/development_trial_roles.json \
  --max-seconds 540 \
  --output-dir outputs/runs/development_search/trial_allocation_attempt001
```

Envelope1CPU/2GiB/10minutes, internal540seconds, numerical threads1. Estimated
128initializer CV fits,2560inner ridge systems,12800candidate scores and
1408actual refits across eight sessions. Source observations total an estimated
28,672fitting-trial reads plus320evaluation rows. Compared with acquired-only's
observed133.45seconds/13,824fitting reads, this roughly doubles narrow source
reads but greatly reduces linear solves; the single session handle avoids
repeated file-open overhead. Unique/read-sum logical bytes and actual runtime
will be recorded separately. No source payload, test, scheduler check or fit
has run from the worker; code/log/ASTRA are frozen for coordinated execution.

## 2026-10-03 22:33 UTC — development_search: trial allocation completed

Root and independent static reviews passed before launch. The pinned ASTRA
validator passed schema/semantics in
`astra/v0.0.14/validation_20261003_trial_allocation01.json`. Root then launched
the four focused tests and full comparison at22:30:08UTC in Slurm46433944
step8 onsh03-02n16.int,1CPU/2GiB/10minutes with540seconds internal bound.
All four tests passed in0.068seconds; the scientific run completed at
22:33:32.827799UTC in203.444723seconds, peak RSS112,264KiB. The durable report
is `runs/development_search/trial_allocation_attempt001/trial_allocation.json`.

All eight sessions and all three twenty-cell profiles completed. There were
2560inner ridge fits,1408actual refits and12,800initializer candidate scores;
later rank/alpha tuning remained absent. Each session's176actual models and
48trajectories preceded evaluation. Initial32 and final96 model objects were
shared across policies after exact final canonical-array equality. Their
score contrasts are exactly zero. The two adaptive policies charge+2, moving
the shared total16to**18**, with no added charge for balance or repetitions.

Observed source access was28,672fitting-trial reads and320evaluation rows,
303,025,920logical neural bytes including repeats and30,094,080unique logical
bytes. The cache-dependent OS block-input counter was1,038,344; it is not an
exact physical-byte estimate. All policies use n×E electrode-trials and
n×U×10target values at the same cell; no all96initializer screening is hidden.
No held sessions, named explanatory endpoint, raw arrays or coefficients were
opened/activated/written by this run. Same-UID callable separation remains
weaker than an OS seal, and the global search remains incomplete.

The following are common-initializer-normalized Q values, distinct from prior
checkpoint-normalized rankings. Intermediate means cover48/64/80×four
electrode counts; the twenty-cell means additionally include shared32/96.

| Policy | Twenty-cell Q | Gain over balance | Intermediate gain | Chewie gain | Mihili gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Balanced |0.200114|0|0|0|0|
| Mean-value |0.200766|0.000652|0.001086|0.000993|0.000310|
| Mean-error |0.198233|−0.001882|−0.003136|−0.000446|−0.003318|

| Session | Mean-value twenty-cell gain | Mean-error twenty-cell gain |
| --- | ---: | ---: |
| Mihili20140217 |0.000431|−0.009548|
| Mihili20140218 |0|0.000372|
| Mihili20140303 |−0.000359|−0.008197|
| Mihili20140304 |0.001168|0.004103|
| Chewie20160929 |0.000146|−0.007934|
| Chewie20161005 |0.000168|−0.002643|
| Chewie20161006 |0.004331|0.007377|
| Chewie20161007 |−0.000672|0.001418|

Mean-value improves five session means, loses two and ties one. Mean-error
improves four and loses four despite its negative mean in both animals.
Mean-value exceeds mean-error in four of eight session means. The largest
mean-value gain is Chewie20161006; in that same session it is below balance
at48trials for16/32/96electrodes, then improves at64/80. Its48trial score equals
balance in the other seven sessions, and every checkpoint/cell in
Mihili20140218 is identical to balance. This limited, profile-dependent
advantage is not a universal uncertainty-allocation law. Initial/final equality
is designed and reported, not empirical convergence evidence.

The twenty-cell pairing increments are0.021508 for balance,0.021681 for
mean-value and0.022105 for mean-error. Error's higher pairing increment
coexists with lower overall Q, so the total-score ordering cannot be explained
as uniformly stronger trial-specific mapping. Utilities estimate reducible
direction-mean error versus total direction-mean prediction error; they do not
estimate the full decoder's information gain. The single fixed queue order,
initializer dependence, participant/session influences, original seed
sensitivities and final frozen held-session test remain outstanding.

Prepared `tools/plot_trial_allocation.py` reads only the completed report to
display paired intermediate session effects and the two animal checkpoint
profiles with faint session traces, without confidence intervals or refits.
Root owns rendering/viewing. The next recommendation is three prospectively
seeded permutations of only the eight post-initial rows within each direction,
preserving initial32, source pool, electrodes, fixed rank/alpha and common
evaluation scale. This arrival-order falsifier is unadopted; it is validation
of the existing policies, not new method aliases or trial-count padding.

The coordinator rendered and visually checked
`figures/trial_allocation_attempt001.png` and`.pdf`; navigation now links both.
The complete prospective successor is `trial_arrival_order_proposal.md`.
Following the coordinator's shared-accounting clarification, it proposes
**one additional scored robustness comparison** against the112ceiling, with
zero new method families (not orders times policies). This charge must be
adopted before launch; new evidence is not left outside shared accounting.
The proposal is not implemented, activated in ASTRA or executed.

The coordinator subsequently validated the final allocator-completion ASTRA
program in allocation46433944: schema and semantics passed in
`astra/v0.0.14/validation_20261003_trial_allocation_complete.json`. That
allocation is now closed; the next substantive execution uses a new coordinated
allocation, not the old PTY or a worker login shell.

## 2026-10-03 — development_search: arrival-order robustness adopted/prepared

Root adopted `trial_arrival_order_proposal.md` before any new order scores,
including **one** additional scored robustness comparison against the112
ceiling, **zero** new method families and all three orders/policies/sessions/
animals. This is validation of the existing direction-mean proxies, conditional
on the current initializer, feature definitions and evaluation pool. It neither
completes initializer/seed robustness nor opens the four held sessions.

Prepared `tools/arrival_order_reader.py`, `tools/development_arrival_order.py`
and `tools/test_arrival_order.py`, preserving all completed source files.
Three metadata-only, fixed seeded shuffles affect just eight post-initial rows
per direction. The exact original initializer stays unchanged. The trusted
reader exposes only the chosen next trial and frozen electrodes; acquired
arrays are returned in the original canonical ID order before every fit.
Initializer rank/alpha/normalizer and initial/final model objects are shared
across all policies/orders. Every final observed array must exactly match the
same original96row pool before shared final-fit reuse. All720model references/
464actual fits and144trajectories per session precede its40evaluation reads.

The prior completed report is read only after all new fitting, for original-
order display, all-seed rank/alpha equality and32/96endpoint replay within the
existing1e−10 tolerance. The original order is not refitted or called fresh.
Every new order is shown separately; the summary averages the three new orders
equally. Fixed-prediction session-omission summaries retain equal animal
weights without retuning. No favorable seed or successful subset is selected.

Three new focused tests cover fixed seeded tail support, chosen-next-row
access/original canonical final-array equality, and endpoint replay of every
order/seed with fixed numerical rank/alpha. Earlier utility, rank, scoring and
source-boundary tests are reused. Root owns execution after independent review;
the worker has not run tests, source values, fits or scheduler commands.

Selected attempt commands from the episode cwd:

```bash
python3 -m unittest discover -s outputs/tools -p test_arrival_order.py -v
python3 outputs/tools/development_arrival_order.py \
  --source-root "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
  --metadata-report outputs/runs/source_support/attempt002/metadata_support.json \
  --trial-roles outputs/runs/source_support/development_trial_roles_attempt001/development_trial_roles.json \
  --reference-report outputs/runs/development_search/trial_allocation_attempt001/trial_allocation.json \
  --max-seconds 840 \
  --output-dir outputs/runs/development_search/arrival_order_attempt001
```

Envelope1CPU/2GiB/15minutes, internal840seconds, threads1. Prospective counts:
128initializer CV fits,2560inner systems,12,800initial candidate scores,
3712actual refits,77,824fitting-trial reads and320evaluation rows. Unique logical
source support is unchanged. The observed203.44second one-order run motivates
this bound; actual runtime, peak memory and repeated/unique source bytes will
be reported. Only a complete comparison charges+1 (18to19); incomplete work
still consumes its actual shared resources. Source/test/log/ASTRA are frozen
for review and root's next coordinated execution.

The coordinator reported all three arrival-order tests passed in0.039seconds
and the pinned ASTRA schema/semantic validation passed in allocation46436630,
hostsh03-06n50, evidence
`astra/v0.0.14/validation_20261003_arrival_order01.json`. At that observation
the substantive source run had not started; it follows the coordinator's
already selected EP05 work. These checks do not constitute completed
arrival-order evidence, and the scientific sources remain frozen.

Separately, root requested an unadopted next mapping-uncertainty comparison.
`mapping_variance_proposal.md` now specifies one acquired-LFP regularized
design-variance rule, explicit algebra/units, same-direction acquired-only
feature-distribution estimates, fixed initial proxy transforms/penalty and
the mismatch to the actual reduced-rank decoder. It retains all four already
declared orders and reuses completed matching count/target-only controls,
with one proposed future comparison charge. No new source/outcome read,
implementation, ASTRA activation or numerical execution was used to prepare
it; root will consider it only after the pending arrival-order result.

At22:49:19UTC root dispatched the command sequence containing the selected
arrival-order attempt in allocation46436630.0 onsh03-06n50, after its three
tests and ASTRA passed. An already selected EP05 report-only plot precedes
the scientific command in that sequence; the dispatch timestamp is not an
observed first-source-read timestamp. Root will verify actual arrival start
and completion. The planned scientific envelope remains1CPU/2GiB/15minutes,
840seconds internal deadline and900seconds external timeout, numerical
threads1; the attended parent actually allocates **1CPU/4GiB**, which is the
available/reserved memory and must not be reported as a2GiB allocation.
The selected output is `runs/development_search/arrival_order_attempt001/`.
Scientific source remains frozen and completion/charge is not inferred from
dispatch. Root owns monitoring and result interpretation access.

## 2026-10-04 06:42 UTC — development_search: finalization-only repair checks

Root independently reviewed the narrow legacy-schema compatibility repair and saved-fit finalizer; owner and independent reviewer passed static review. Selected compute execution in verified46458807/sh03-01n05, with thirteen focused tests and changed ASTRA validation immediately preceding the60s internal/90s external finalizer. Fresh high_power_correspondence_attempt002 reuses the exact complete but uncharged760-fit attempt001 and only old qualified JSON references; zero new source reads/refits. Original fit costs and failed report stay preserved. Scientific outcomes/count remain unreleased until complete finalization.

## 2026-10-04 06:54 UTC — development_search: logdet checks passed and source run selected

Allocation46460577 granted, actual execution shell sh03-01n68.int/SLURM_JOB_ID46460577 verified;1CPU6GiB normal CPU_SKU:7502,30minutes, starts06:52:49UTC and ends07:22:53UTC, no GPU. Root explicitly disables the scheduler's default Requeue flag for this attended job. Nine focused logdet fixtures passed2.110s with numerical threads1; root/independent implementation review and owner fixture review passed. Changed ASTRA validation precedes source execution.

Select the adopted full eight-session source comparison now: `timeout210s python3.12 outputs/tools/development_initializer_logdet.py` with the original source/metadata/roles and completed initializer_repeatability_attempt001 reference, `--max-seconds180 --output-dir outputs/runs/development_search/initializer_logdet_attempt001`. Source fits remain subject to complete-panel checks; count remains34 until the complete report. No held access, reference refit, additional seed charge or new endpoint is activated.
