# Outputs

**October 16–17 pre-evaluation build (current entry point).** The [analysis specification](config/analysis_specification.json) fixes the role seal, packets, numbers, emulator, recalibration, uncertainty, decisions, positive control, search rules and the T9 command. Evidence:
- 49 [checks](runs/analysis_specification/20261017-synthetic_checks08/tests.txt) pass in Slurm.
- T11 [packets](runs/analysis_specification/20261016-t11_packets01/packets.json) resolve for 64 of 64 sessions.
- T11 calibration and known-positive control are in [calibration07](runs/analysis_specification/20261017-t11_calibration07_blocknorm/), with a [figure](figures/ep16_t11_positive_control.png).
- The sealed-evaluation [rehearsals on T11](runs/analysis_specification/20261017-t9_rehearsal04/), one from NPZ features and one through the archive reader, give identical primary results.
- An independent Codex review found 16 issues; the accepted fixes are applied and re-tested (spec `independent_review`).

**Result:** the T11 known-positive control **fails under the frozen readiness rule**. Four T11 sessions have no local signal, so every R bound is infinite; the injections themselves are recovered (+0.18 for +0.20).

T9 remains sealed. Four items for the scientist block the freeze: block normalization, the readiness rule, the J_min null, and the evaluation-only resampling. See the [log](experiment_log.md) and the spec's `proposed_pending_scientist`.

**October 16 case-series amendment (adopted):** develop on T11, then evaluate T9 once; claims concern one person. See the [GOAL amendment](../GOAL.md#adopted-case-series-amendment--october-16-2026), the [roles and final T9 exposure review](../DATASETS.md#case-series-roles-and-t9-exposure-review) (T9 confirmed as the evaluation case), the policy block `case_series_amendment_2026_10_16`, and the [remaining work](../DATASETS.md#current-readiness). Splitting into two time partitions leaves both T9 halves rank-deficient ([check](runs/structural_support/20261016-time_partition_support02/time_partition_support.json)). At adoption no neural value had been opened; T11 was opened on October 16–17 and T9 stays sealed.

**October 16 source-fact reassessment:** the missing size, layout, transition and precision facts were resolved or assumed before application ([items](schedule_definition.md#october-16-source-facts-and-adopted-assumptions); [rules](config/source_fact_rules.json)). Support did not change: only T9 is held-out eligible, and T9 and T11 are supported. This holds under the primary rules and nine sensitivities ([figure](figures/ep16_source_fact_support.png); [run](runs/structural_support/20261016-source_fact_support01/); [log](experiment_log.md#2026-10-16--structural_support--support-re-run-under-adopted-rules-48155786)). The closeout disposition stands. Next is a scientist decision on the T11-to-T9 case series.

**October 6 adopted closeout:** the original three-held-out-participant claim
has disposition **support_or_uncertainty_prevents_an_answer**. Completed
structural/preparation evidence is retained; no neural outcome or successor
is selected. See the [adoption record](experiment_log.md#2026-10-06--structural_support--scientist-adopted-support-limited-closeout).

[Public online resources, checked October 6](../DATASETS.md#public-online-resources--checked-october-6-2026): the Dryad field codebook and pinned provider identity/decoding methods, with the unresolved session-specific metadata distinguished.

October 6 reconciliation: [DATASETS readiness](../DATASETS.md#current-readiness)
now includes the completed all-session structural work, conditional T11/T9
support and passing analysis primitives, with the remaining session evidence
and reader/model work kept explicit. The [log](experiment_log.md) records this
documentation correction; no neural analysis or redesign was activated.

October 6 public-source update: the
[provider-field note](schedule_definition.md#october-6-public-source-update--documented-fields-and-unresolved-session-annotations)
links the exact Dryad field definitions and pinned provider metadata code.
Known units/index conventions are distinguished from still-missing session
target-size, controller and transition annotations. No support rerun or
neural opening occurred; see the [log](experiment_log.md#2026-10-06-pacific--public-provider-field-definitions-added).

October 6 factual clarification: **6/3** is the role allocation; **4+3** is
the minimum supported-participant requirement. Only T11 (development) and T9
(held-out candidate) currently pass the conditional numerical design, without
full qualification. The [support-limited closure and reopening condition](paper_plan.md#october-6-review--current-support-and-adopted-closure)
are **adopted**; the optional single-case evaluation draft remains **unadopted**. Missing metadata
does not provide a general monotonicity proof for design support; no unchanged
support rerun, role change or neural analysis is selected.

October 6 prospective preparation: the
[analysis/exposure definition](schedule_definition.md#october-6-prospective-analysis-and-exposure-definition)
now fixes raw score units, octant support, 30-bin window/source scaling,
direct recording-quality order and existing normalized anchors. The numerical
consumer's [three focused checks](runs/analysis_definition_checks001/tests.txt)
and changed ASTRA validation passed in allocation 46721202; see the
[execution record](experiment_log.md#2026-10-06-0529-utc--analysis_definition-focused-checks-pass).
Known exposure
for all nine participants is recorded; incomplete history review is not called
clean. Source size/layout facts, remaining scientific choices and actual
cohort/reader qualification remain unresolved. No unchanged support rerun or
neural job is selected. Earlier observations below are preserved.

[Current conceptual figure, v2](ep16_question_imagegen-v2.png) shows the
direction proxy, same-target-trial comparison and coexisting operational
signatures without invented scores or causal-mechanism claims.
[Exact image-generation prompts](ep16_question_imagegen-v2-prompt.md) include
the inspected correction. The earlier `ep16_question.svg` and
`ep16_question_imagegen.png` are historical illustrations, not current
design figures. No figure contains an observed result.

As of 2026-10-03, EP16 is preparing the authorized structural support check.
The exact v6 provider documentation and existing acquisition verification have
been inspected. The [format-only report](runs/structural_support/20261003-format01/structural_format_report.json)
from job 46425841 confirms MATLAB Level 5 for one already-development T10
session. The [field-selective extraction](runs/structural_support/20261003-fields01/structural_metadata_report.json)
passed on that session. The [complete structural view](runs/structural_support/20261003-fields_all01/structural_metadata_report.json)
now covers all 729 sessions from all nine participants, with no neural,
cursor or impedance values decoded. The [support description](runs/structural_support/20261003-support01/structural_support_description.json)
rules out T3 for held-out assignment: even its pair-count upper bounds are only
four near and five long pairs. The other participants still require schedule
and rank qualification. T3 can remain in the six-person development role;
the policy requires at least four supported development participants. See the
[current execution entry](experiment_log.md#2026-10-03--structural_support-source-and-selective-reader-preparation).
No neural comparison has run and no held-out participant result has been
examined. Participant assignment and protected role views remain unfinished.

The completed metadata-only step is the
[coordinate/conditional-rank description](schedule_definition.md), implemented
in `code/describe_schedule_design.py`. It uses the completed structural view,
permits translation and positive isotropic scale only, and does not yet qualify
task-size/transition compatibility or assign participants. See the selected
command in the execution log. The
[precision-corrected attempt02](runs/structural_support/20261003-schedule_design02/schedule_design_report.json)
completed on job46425841 at19:37:50UTC in1.689s, describing729sessions and
355geometrygroups; official ASTRA validation passed. Attempt01 is preserved.
Five people have a conditional full-rank component under this strict geometry
descriptor; this is not five qualified participants or terminal infeasibility.
The six-near/six-long held-out support gate applies within a common schedule
group, not separately to every connected component. See the
[aggregation clarification](schedule_definition.md#conditional-fixed-effect-design);
each contributing component still needs both gap classes and its own qualified
design. The current descriptor does not apply a six/six component filter.
The [completed target-storage census](runs/structural_support/20261003-target_storage01/target_storage_report.json)
on job46427768 at19:59:29UTC resolved all729target matrices as MATLAB doubles
(`<f8`), with zero numeric decoding/errors. It took404.088s; one node-local
temporary MAT at a time reached a maximum577,212,492bytes, with no raw copies
retained. Both focused access checks and updated
[ASTRA schema/semantic validation](astra/v0.0.14/validation_20261003_target_storage.json)
passed.
The [observed-schedule diagnostic](runs/structural_support/20261003-observed_schedule01/observed_schedule_report.json)
completed at 20:11:32 UTC in 7.836 seconds, retaining all 328,548 trials and
5,150 blocks. It reports observed transition/coverage differences without
equating missing sampled edges to forbidden-transition conflicts; ASTRA passed.
The [historical proposal](config/schedule_support_proposal.json) is preserved.
The coordinator adopted the [active tolerance/support configuration](config/schedule_support.json)
before application, including unresolved near duplicates, strict boundaries and
contrast-specific node information for influence. All six focused checks and
[conditional numerical application](runs/structural_support/20261003-conditional_support01/conditional_support_report.json)
completed on job 46429138 at 20:33:50 UTC; the application took 2.821s. It
retained 729 sessions / 328,548 trials, formed 368 geometry groups and evaluated
32 conditional designs, with two numerical passes (T11 and T9) and no
nontransitive groups. ASTRA validation passed. These are not qualified
participants: block/size classification and exposure review remain open, and
the current panel cannot support three held-out people or the
four-supported-development search. See the
[actual outcome and targeted remaining metadata](experiment_log.md#2026-10-03--structural_support-conditional-numerical-application-completed).
The earlier diagnostic is recorded in the
[execution entry](experiment_log.md#2026-10-03--structural_support-observed-schedule-diagnostic-selected).
Task/controller identity is not a universal matching requirement. The specific
missing size-dependence annotations and prospective matching/support rules are
discussed in the
[same-contract qualification route](schedule_definition.md#remaining-source-and-support-route).
Only the corresponding previously-null numerical policy fields were filled;
participant roles and scoring remain closed. The log also records a reviewer's
unintended public within-session/yield exposure without numerical outcomes;
the participant-specific exposure review remains necessary before assignment.
No further EP16 compute is selected. Failed pairs/components cannot be pruned
to restore support; this conditional-panel failure is not a universal source
infeasibility claim.

The files here have two purposes:

- [paper_plan.md](paper_plan.md) turns the scientific question into a
  four-figure paper story with explicit next steps for each possible result.
- The remaining short files record what has and has not happened during the
  episode. They do not themselves constitute scientific evidence.

Future result files should make the main comparison easy to trace: near-gap
versus long-gap transport loss, followed by the later-session signal,
recording-state, and 32-label recalibration tests. Large temporary products
belong in episode-specific scratch rather than this directory.
