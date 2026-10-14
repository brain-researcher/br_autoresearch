# Experiment log

## 2026-10-13 — v3_search — inactive child-side failure diagnostic prepared

The full-goal audit preserves the adopted original-M2 incomplete closeout.
A focused source review identified why the four old cap failures lack useful
terminal state: the qualification worker lets exceptions cross ProcessPool
without persisting their frame locals; only successful cases receive JSON.
Current cap messages already include richer scalars, so repeating that edit
would not repair the child-to-parent diagnostic loss or recover past values.

Prepared a separate [proposal and helper](proposed/m2_failure_telemetry_20261013/README.md)
that captures only allowlisted scalar failure state inside the child, writes a
fresh diagnostic record and reraises the original exception. It does not alter
an optimizer, successful return, scientific data role, threshold or stopping rule.
Independent source review found one output-boundary defect: unrestricted exception
text could include arrays. Root replaced it with exception classes/fixed reason
codes and omitted free-form diagnostic-error text. No tests were run; the helper
has no CLI, operational import, submission or activated scientific recipe.

No algebra defect or demonstrated acceleration was established. The qualified
M1's949 fits remain reusable. Original M2-minus-M1 scope is unchanged, and the
≥1900 additionalCPUh lower bound still exceeds1064.288778CPUh remaining; this is
not a complete-search upper bound. No empirical retry, synthetic expansion,
budget reset, reactivation or outcome opening is selected by this preparation.

## 2026-10-12 — v3_search — inactive original-M2 engineering handoff

Two bounded, independent source reviews support the
[inactive original-M2 reactivation proposal](proposed_original_m2_reactivation.md).
All four preserved iteration-cap failures arise in the same pure low-rank
solver; the hybrid rank-8 case fails in its prerequisite anchor before hybrid
alternation begins. Neither review established an objective, gradient,
projection, retraction, callback or stopping-rule defect. The old public
exceptions lack the terminal gradients, relative objective change and accepted
step needed to diagnose slow convergence. Reaching the cap does not establish
divergence or identify which convergence predicate failed.

The October 6 incomplete closeout remains operative. Standing willingness to
increase compute is recorded; it does not establish a repaired solver or a
finite full-search cost. Preserve 735.711222 allocated CPU-hours and 54.240833
charged execution wall-hours, all original scope and closed outcomes. The
proposal identifies the common engine and missing diagnostic evidence without
selecting an optimizer change, new test, job, budget reset or successor. Only
permitted source and public historical runtime exceptions were inspected; no
scientific payload or canonical state was opened or changed.

## 2026-10-06 — v3_search — scientist-adopted local closeout

The scientist replied to the outstanding-judgment recommendations:
"i agree with your judgement; EP20 is fine with simulation because we dont have real data".
For EP10, this adopts closure of the original M2 objective as **incomplete
under computational and numerical limits**. The EP20-specific simulation
permission does not select an EP10 successor.

Preserve all 949 qualified M1 fits on 194 cells/12 conservative groups and the
secondary exploratory M1-minus-M0 gain of 0.0026378637 bits/cell, all failed
attempts, and prior charges. The documented M2 lower bound is at least 1,900
CPU-hours against 1,064.288778 remaining; required optional-family numerical
cases also remain unresolved. This is neither a biological null nor evidence
that M1 is sufficient. No further probe, M2 run, replacement study, budget
reset or final/unassigned outcome opening is selected.

The local contract, plan and navigation now record that adopted disposition;
the ASTRA descriptions retain the original program without inventing completed
search outputs. No payload, test, job or scheduler action occurred, and no
canonical BR state or scientific acceptance changed. Earlier unadopted wording
below records the proposal before this explicit acceptance.
Bounded readback also marked the older active-execution and reservation
descriptions as preserved history, consistently with this adopted closeout.

## 2026-10-06 — critique revision: factual clarification and unadopted closure

Reviewed the supplied critique against the current producer, diagnostic and
accounting descriptions without opening result payloads. Clarified that all
completed real M1 fits passed, whereas four optional higher-rank synthetic
cases remain unresolved. Retained the measured 0.0026378637-bit/cell
exploratory gain, the at-least-1,900-CPU-hour lower bound and 1,064.288778
remaining CPU-hours; a much larger runtime extrapolation is not established.
Proposed closing M2 as incomplete, not as a biological negative. That
disposition is unadopted. No contract, budget, recipe, access rule, scheduler
state or scientific artifact changed; no compute or follow-up was selected.

## 2026-10-03 20:04 UTC — triangular solve agrees numerically but does not accelerate the warm path

Job **46428039** completed with exit 0 on AMD EPYC 7502 node sh03-06n50,
using the frozen NumPy 1.26.4/ILP64 OpenBLAS Zen backend and one thread.
The six focused direction, conditioning, buffer-layout and unchanged-gate
tests passed in **0.086 s**. Both versions completed the same 12-lambda first
synthetic permutation warm path with the same per-lambda iteration counts,
107 exact evaluations and 159 factored-solve calls. The preserved
[paired report](stage0/v3_m1_triangular_prefix_job_46428039/paired/paired_prefix.json)
records original/repaired wall times **98.6347/98.5940 s**. Factored-solve time
fell **0.18930 → 0.02688 s**, but exact-evaluator time remained
**70.0233/70.1197 s**. The single original-then-repaired pair supplies no useful
whole-path speedup; its approximately 0.04% wall-time difference is not a
reliable performance gain.

Across all 12 completed fits, maximum objective and parameter absolute
differences are **2.66e-15** and **6.24e-13**. At the fixed last common lambda,
full candidate-distribution comparison gives maximum log-probability
difference **1.03e-13**, maximum probability difference **1.39e-15**, marginal
MAE **6.93e-16** and mean KL **7.00e-18 bits**. These results support local
numerical agreement on this workload and the focused tests. They are not a
complete production requalification or a whole-permutation timing unit, do
not establish full-search feasibility, and do not repair the four failed
optional-family cases. No real M2 or protected outcome was opened.

The coordinator reports **217 elapsed seconds × 1 allocated CPU**, charging
**0.060277778 core-hour** in place of the 0.5-core-hour reservation. The
[reconciled resource snapshot](stage0/resource_accounting_20261003_after_triangular_prefix.json)
also charges the completed ASTRA step **46427768.3** at **8 seconds × 1 CPU**
(19:58:36–19:58:44 UTC), giving **225 CPU-seconds = 0.0625 core-hour** total
new work. It records **735.711222 core-hours used**, **1,064.288778 remaining**,
and 54.240833 charged execution wall-hours in the unchanged epoch, leaving
89.759167. No further EP10 compute charge remains pending in allocation
46427768. Official schema and semantic validation
passed in allocation 46427768; exact output:
`astra/v0.0.14/validation_20261003_triangular_prefix.json`.

Read-only evaluator review finds the main remaining work is the required
exhaustive two-pass candidate evaluation, exact 21-moment Hessian construction,
and separately streamed direct directional derivative. The latter is outside
the 70-second evaluator counter, so the remaining 28.6 seconds cannot be
labelled expendable overhead. Reusing a rounded gradient derivative, persistent
soma features, fewer candidates/passes, or a different summation rule would
violate the frozen method. Batch-local arithmetic rearrangement is possible
in principle, but this profile identifies no dominant redundant operation with
a credible saving large enough to justify another probe against the remaining
budget. Do not chase further small linear-algebra changes from this result.
Current M2 remains an incomplete search under technical and resource limits;
no further compute is selected by this closeout.

## 2026-10-03 — selected bounded M1 triangular-solve engineering comparison

The coordinator authorized a semantics-preserving performance repair following
the failed M2 resource and optional-family diagnostics. Code inspection finds
that each M1 Newton direction factors its Hessian with Cholesky, then calls
six general `numpy.linalg.solve` operations on those triangular factors across
the initial solve and exactly two refinement steps. NumPy's documented general
solver uses LAPACK `gesv`, repeating work unnecessary for triangular matrices
([NumPy 1.26 documentation](https://numpy.org/doc/1.26/reference/generated/numpy.linalg.solve.html)).
The selected helper `stage0/tools/v3_m1_triangular_solve.py` instead calls
`cblas_dtrsv64_` from the **same existing NumPy ILP64 OpenBLAS library**, with
explicit binary64 contiguous copies and correct lower/transpose semantics.
No SciPy installation or second BLAS is introduced. Cholesky, exactly two
refinement steps, diagonal shifts, residual/descent/predicted-decrease/Wolfe
gates, tolerances, caps, lambda paths and all scientific rules remain unchanged.
There is no cell-by-candidate or soma-feature cache. Root's independent
code-only review found no concrete C-ABI or unchanged-gate defect; execution
qualification is still pending.

Selected compute: run six focused direction/conditioning/buffer-layout tests,
then `stage0/tools/benchmark_v3_m1_triangular_prefix.sbatch` runs one pair on
AMD EPYC 7502/Zen with the frozen environment and one numerical thread. Both
variants use the unchanged deterministic 194-row synthetic falsifier fixture,
same permutation and first actual descending 12-lambda warm path. Original
general solves run first, repaired triangular solves second; each stops after
that path or **300 seconds**. Aggregate timings separate factored solves from
exact evaluations. Mutually completed fits are compared without exporting or
reusing optimizer state, followed by one full candidate-distribution comparison
at the last mutually completed lambda (120-second cap). No objective, family,
caps or acceptance thresholds are tuned from the comparison. An interrupted
prefix remains incomplete; its timing is not the full permutation unit used
in the earlier 1,900-core-hour bound and cannot replace that bound.

The proposed dedicated normal-partition request is **1 CPU, 4 GiB, 15 minutes**
with the discovered `CPU_SKU:7502` feature. Reserve **0.5 allocated core-hour**
conservatively for possible memory-driven rounding to two CPUs; reconcile the
actual allocation and elapsed time on completion. Against the latest preserved
resource snapshot, this leaves 1,063.851278 unreserved core-hours before any
other new EP10 charge; no lifetime ceiling or wall epoch resets. Root owns the
submission. Fresh scratch/durable paths will be
`v3_m1_triangular_prefix_job_<jobid>` under the existing EP10 roots. The changed ASTRA recipe requires its
normal schema/semantic check on compute; it does not create an access gate.

The coordinator submitted the reviewed wrapper and Slurm accepted job
**46428039**. Source is frozen for this attempt. Its exact scratch directory is
`$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/v3_m1_triangular_prefix_job_46428039/`,
and the wrapper preserves its aggregate reports and focused test output under
`outputs/stage0/v3_m1_triangular_prefix_job_46428039/` on ordinary exit.
The paired report will be `paired/paired_prefix.json` within those directories;
stdout/stderr are `$SCRATCH/ep10-triangular-prefix-46428039.out` and `.err`.
Acceptance is not evidence of execution, test success or numerical agreement.
The coordinator will supply terminal observations and actual resource usage.

Separately, only the terminal low-rank iteration-cap error text now includes
rank/rho, final U/d gradient norms, relative objective change, last accepted
step and line-trial count. This exposes the concrete diagnostic quantities
missing from the failed attempt without altering its optimizer. No optional
requalification is selected. Four required optional-family cases still failed,
and no real M2, null search, SWC, final or unassigned access is authorized by
this engineering comparison. The valid completed M1 producer/results remain
preserved and are not rerun or invalidated by preparation of the repair.

## 2026-10-03 — terminal M2 diagnostic resource reconciliation

Slurm accounting confirms optional-family job 46237243 ended CANCELLED after
160,622 seconds with 9 allocated CPUs. Replace its unused reserved tail with
actual usage, releasing 30.445 allocated core-hours. Charge both attended
runtime-bound diagnostics conservatively at 600 seconds × 4 allocated CPUs
each, including their evidence preservation and closeout. The reconciled
[resource record](stage0/resource_accounting_20261003_m2_diagnostics_terminal.json)
retains all historical usage and the existing October 1 wall epoch: 735.648722
lifetime allocated core-hours used, with 1,064.351278 remaining; 54.178333
epoch wall-hours charged, with 89.821667 remaining. This replaces reservations,
not consumed history. The operative 1,900-core-hour projected lower bound
still exceeds both the remaining allowance and the entire lifetime ceiling.
No real M2 or full-search-null job is launched.

## 2026-10-03 18:51 UTC — `v3_search` resource bound observed

The prespecified target-free diagnostic completed on AMD EPYC 7502 node
sh03-06n50 in Slurm job 46425841 using the frozen numerical backend and one
effective thread. The required fresh-permutation M1 prefix was still running
at the 300-second deadline. Its incomplete coefficients were discarded. The
unchanged combined-resource formula therefore projects at least **1,900
allocated core-hours**, even with every other cost set to zero, exceeding the
entire **1,800-core-hour** lifetime ceiling before subtracting prior usage.
Preserved report:
`stage0/v3_m2_resource_lower_bound_job_46425841/resource_lower_bound.json`.

The current frozen implementation cannot pass its preaccess launch gate.
Do not submit real M2 or full-search null jobs under this configuration.
This is an incomplete search under the existing budget, not positive or
negative biological evidence or a lower bound for every possible algorithm.
No real M2 membership, K, outcomes, final/unassigned data or SWC was opened.
The diagnostic is not a completed unit-timing report or valid model trial.
Preserve already-authorized optional-family qualification 46237243 within
its existing reservation and original timeout; its completion alone cannot
override this separate resource failure. Any later engineering route must
preserve the scientific estimator and requalify against the same remaining
budget; no larger ceiling or revised biological endpoint is activated.

### 18:56 UTC — optional qualification failures and selected stop

The dedicated monitoring worker found four explicit `RECOVERY_CASE_FAIL`
events in job 46237243: low-rank ranks 4, 6 and 8, and sparse-plus-low-rank
rank 8. Each hit the frozen iteration cap without simultaneous gradient and
relative-objective convergence. The coordinator confirmed that
`qualify_v3_m2_optional.py` retains these failures and unconditionally rejects
the aggregate after its worker pool exits; remaining workers cannot make this
attempt PASS. Only the three reused low-rank rank 1/2/3 cases are present in
the completed-case directory. Scheduler still showed RUNNING at 44h36m32s.

This new failure evidence supersedes the earlier instruction to preserve the
job until its original timeout. Selected action: stop only job 46237243, then
preserve its exact scratch directory and log in a distinct terminal archive.
This prevents further spending on an irrecoverably failed attempt; unfinished
workers are not scientific negatives or reusable fits. No automatic retry,
iteration-cap increase, family deletion or resource-ceiling change is selected.

The stop command succeeded. Slurm accounting confirms CANCELLED at
44h37m02s with 9 allocated CPUs (401.555 allocated core-hours); the batch step
ended by signal 15. Preserved the exact directory, runtime log and stderr under
`stage0/v3_m2_optional_family_qualification_job_46237243_terminal_20261003/`.
No new optional-family qualification is submitted.

Root's compute-process monitoring also found that overlapping one-CPU Slurm
steps default to the same physical CPU. The original runtime-bound diagnostic
could run across CPUs 0–3, so a conservative interpretation treats its resource
projection as provisional pending a check with enforced distinct CPU placement.
Selected one repeat of the unchanged diagnostic, bound to allocated physical
CPU 3, while other scientific steps use CPUs 0 or 2. Its output is
`v3_m2_resource_lower_bound_job_46425841_isolated_cpu3_attempt002`; reserve another
10 minutes × 4 allocated CPUs conservatively. This directly checks a concrete
timing concern; it does not retry the failed optional numerical qualification.

The repeat with physical CPU 3 reserved for the diagnostic confirmed the same
300-second interruption and 1,900-core-hour lower bound. Its exact report is
preserved in
`stage0/v3_m2_resource_lower_bound_job_46425841_isolated_cpu3_attempt002/`.
The resource conclusion therefore holds with distinct physical CPU placement;
the current M2 configuration remains unlaunchable under the original ceiling.

## 2026-10-03 — ASTRA and resource diagnostic checks

The official pinned ASTRA validator passed schema and semantic validation
in compute allocation46424502 (evidence retrieval skipped); report:
`astra/v0.0.14/validation_20261003.json`. The three focused runtime-bound tests
passed in0.012s. The coordinator independently checked the original combined
resource formula: cap80, coverage16 leaves57 slots, contributing at least
22800T/3600 allocated core-hours over the observed controller and99 replays.
Thus a300-second lower bound exceeds even the full1800-core-hour ceiling.
The current attended node is AMD8224P; the diagnostic retains its frozen
AMD7502/Zen/one-thread guard and has not run on incompatible hardware.

The next attended portfolio allocation will explicitly request the discovered
`CPU_SKU:7502` feature, normal partition,4CPUs,16GiB and1hour. EP10 uses one
numeric thread and its300-second interruption; its reserved10minutes×4CPUs
remains conservative. Other portfolio work shares the remaining capacity.
This is a hardware-specific allocation change, not an automatic renewal or
a relaunch of the optional-family job.

## 2026-10-03 11:20 PDT — `v3_m1_compare_m0` and `v3_search`: current results and bounded resource diagnosis

The coordinator executed the frozen comparator and M1 adequacy calculation in
allocation `46424502` and preserved their complete directories on Oak. Both
**PASS**, using repaired M1 producer `46332190`. Exact reports:
`outputs/stage0/v3_m1_compare_m0_v1_job_46424502/v3_m1_minus_m0_comparison.json`
and `outputs/stage0/v3_m2_m1_adequacy_gate_job_46424502/v3_m2_m1_adequacy.json`.
The equal-group M1-minus-M0 gain is `0.002637863687897285` bits/cell, a secondary
exploratory comparison, not a biological or M2 claim. M1 adequacy gives
`1.5953602605961161` bits/cell over fixed-K uniform, Brier improvement
`0.029304747266512`, target RMS `0.12149700112033555` and maximum absolute target
residual `0.2564803858876762`; all original thresholds pass. Historical M0
exposure remains recorded; this new comparison separately opened only M0
summary and outer predictions after the current independent M1 postflight.
No M0 full-refit/telemetry, SWC, final or unassigned outcome was opened.

The exact timeout directory and stdout/stderr for `46325527` were preserved by
the coordinator using non-overwriting compute-node copies. Its last progress
records completed fresh permutation M1/cache and **two completed M2 folds**,
with the third starting at `2026-10-02T21:43:56.125Z`. Slurm killed it at the
four-hour limit. There is no complete unit timing or resumable numerical
checkpoint. The roughly 30-minute interval from third-fold start to kill is
diagnostic context, not a substituted complete measured unit.

The six existing preaccess reports are: core qualification PASS `45832824`,
actual-K PASS `45847880`, current M1 adequacy PASS `46424502`; optional-family
report pending current job `46237243`; scale absent (old `45924664` cancelled);
complete falsifier timing absent. The observed wrapper's old producer paths
remain unlaunchable until all required current reports and resource gates pass.
No real M2 or full-search null was launched, and stale M1 evidence is not reused.

Budget reconciliation retains the Oct1 wall reset, all lifetime usage and the
optional job's full 48-hour allocation reservation. Charge clean M1 actual
7540 seconds × 3 CPUs, diagnostic actual 14424 seconds × 2 CPUs, and a
conservative attended EP10 600 seconds × 4 CPUs. This leaves `86.77222222`
wall hours and `1035.23961111` allocated core-hours unreserved before the next
diagnostic. Details: `outputs/stage0/resource_accounting_20261003_after_M1_closeout.json`.

Selected before the next compute effect: a single narrow target-free timing
lower-bound diagnostic, with the unchanged synthetic fixture, permutation and
fresh-M1 fit. Time only the required M1 prefix and interrupt it at 300 seconds;
unmeasured fixture/assignment setup remains outside the bound. Reserve at most
another 10 attended minutes × 4 allocated CPUs. Under the existing combined
formula, cap80/coverage16 has 57 remaining slots, each priced at least one
permutation unit T. Ignoring every other nonnegative cost gives
`2*(57*T+99*57*T)*2/3600 = 22800*T/3600` allocated core-hours. T=300 therefore
already projects 1900 core-hours, exceeding even the original full1800 ceiling.
The coordinator independently read the original formula and confirmed this
derivation. This is a bound for the current implementation and frozen backend,
not a proof about every implementation or hardware platform. An interrupted
prefix is never a fit result, completed timing report or valid trial; a resource
stop means incomplete search, neither success nor negative biology. No budget,
model, threshold, estimator, search/null count or access contract is changed.

The three focused lower-bound/timer tests passed in `0.012s` in allocation
`46424502`, step2. They compare the simplified bound to the original combined
resource validator and exercise actual signal interruption, cleanup and
numerical-error propagation. That node is AMD EPYC8224P, so the scientific
diagnostic was **not** run there; it still requires the frozen AMD7502 backend.
The new ASTRA output preserves the existing analysis ID; schema/semantic
validation and the actual scientific resource diagnostic remain pending with
the coordinator. No elapsed prefix or resource-failure result is inferred.

## 2026-10-03 11:10 PDT — `v3_m1`: independent postflight passes; comparison and adequacy selected

Root preserved the complete exact producer directory `46332190` under Oak,
leaving Scratch unchanged. Actual independent postflight completed at
`2026-10-03T18:10:40.792104Z` in allocation `46424502`: **PASS**, 91 checkpoints,
949 fits,194cells,12groups. Every fit used the ordinary frozen gradient-and-
relative-objective witness; there were no stall-certificate returns. Saved
one-SE decisions and all score/calibration identities were recomputed; maximum
inclusion-marginal sum-to-K error was `4.440892098500626e-15`.
Evidence: `outputs/stage0/v3_m1_real_fit_single_thread_v3_job_46332190/independent_postflight.json`.
Historical M0 access is retained, while this audit opens no M0 and confirms the
fitting process's closed M0/SWC/final/unassigned flags. Its only extra outcome
read is the exact permitted development membership projection for calibration.

The repaired comparator's 11 focused tests passed in0.165s on compute. Tests
include original score/access invariants plus current producer binding and
rejection of erased M0 history or an obsolete postflight. No score, threshold,
tuning or model changed. Readiness now separately records the actual current
postflight; old producer/gate evidence remains historical.

Selected next process effects: execute the frozen M1-minus-M0 comparator and
the unchanged pre-M2 M1 adequacy calculation against `46332190` in allocation
`46424502`. Both write new job-scoped outputs. The comparison may now open only
the existing M0 summary and outer-cross-fitted predictions, with the existing
atomic process-local access record; prior historical exposure is never reset.
No real M2/full-search null/final/terminal-anatomy access is released by this
selection. Preserve the actual timeout diagnostics before considering any
bounded continuation of the separate permutation timing stage.

## 2026-10-03 — `v3_m1` and `v3_m1_compare_m0`: completed clean producer, selected postflight

The scientist authorized resuming possible work and finishing the questions.
The coordinator's fresh scheduler evidence reports clean M1 job `46332190`
COMPLETED, elapsed `02:05:40`, exit `0:0`. Its terminal summary/manifest reports
194 development cells in 12 groups, 78 warm paths, 13 zero refits and zero
boundary folds, with new-process M0/SWC/final/unassigned access false. These
are producer observations; independent postflight is still pending.

Selected next actions before any new M0 comparison: preserve the exact
completed job directory from episode Scratch at
`outputs/stage0/v3_m1_real_fit_single_thread_v3_job_46332190`, without changing
or deleting its source; independently verify current-job artifacts, all solver
witnesses, one-SE choices, score/calibration identities and access/backend
bindings; then mechanically repair the existing comparator's stale producer
and readiness bindings. Root owns the compute shell and transfers; this worker
performs source edits only on login. The comparison remains the frozen equal-
group paired score, with no new tuning or modeling.

Historical M0 opening `45760290` remains recorded truthfully. The repaired
fitter's process-local M0 access is separately false. Old M1/comparison
artifacts remain historical after the derivative correction; they are neither
overwritten nor used as initialization. The comparator may read M0 summary and
outer predictions only after the new producer's independent postflight passes.

Existing optional job `46237243` remains running under its 48-hour limit.
Permutation diagnostic `46325527` timed out at `04:00:24`; no successor or
complete timing result is inferred. Real M2/full-search nulls remain gated;
final/unassigned trees and the proposed terminal-anatomy follow-up stay closed.
The root-owned attended Slurm allocation is `46424502`. No new submission or
postflight/comparison execution has occurred in this preparation entry.

Preparation continuation: the existing comparator now binds current producer
`46332190` and a separate current postflight gate; its regression fixture keeps
the actual historical M0 access string and explicitly rejects erased history
and obsolete producer postflight. Its scoring/pairing formulas are unchanged.
The independent postflight checks current-job checkpoints and convergence
witnesses and recomputes one-SE decisions from saved grids plus calibration
from selectively projected existing development membership. It reuses only
the trusted field-projection parser, not the fitter/evaluator/optimizer.
The M1-adequacy wrapper also binds the repaired current producer. After the
postflight passes, select both the paired comparison and unchanged M1 adequacy
calculation as short stages in the same attended allocation. They create new
job-scoped results and do not release M2 before its other required gates.

## 2026-10-02 — Concept figure v2; writing only

Saved the built-in imagegen [compact terminal-layout schematic](ep10_within_target_implementation-v2.png)
and [prompt](ep10_within_target_implementation-v2-prompt.md). The drawing makes
positive-context overlap explicit and removes the checklist/disclaimer banner.
GOAL, paper plan and outputs README now display it. Prior figures remain.
No endpoint, live contract, ASTRA, scientific payload, job or access changed;
this is not a new scheduler observation. Existing execution entries remain
the runtime handoff.

## 2026-10-02 — Scope/novelty review; design retained

Added a linked [primary-literature review (consolidated)](paper_plan.md).
The planned terminal follow-up is scientifically interpretable but has no
source-supported new A/B/C candidate selected here. The note explains why
development support, not another prose rewrite, is the remaining choice.
Prior read-only BR projection retrieval returned no relevant evidence;
checked primary papers informed the assessment. No new scheduler observation,
payload access, endpoint, execution-contract, ASTRA or runtime change.
The dated execution entries below remain the runtime handoff.

## 2026-09-30 — Scientific-primary redesign (documentation only)

- `[episode10_single_cell_coprojection]` The scientist requested parallel
  scientific-design repairs. EP10's planned next biological study now makes
  within-A terminal-profile shift with positively recorded co-target context
  its scientific primary, beyond flexible soma/layer/continuous geometry.
  Falasconi 2026 source modules, Gao 2025 terminal diversity, and Yuan 2024
  co-target/laminar organization are explicit prior-art competitors.
- B-positive and C-positive contexts may overlap. A+B+C-positive cells retain
  identity and overlap covariance; unknown zeros never define exclusivity.
- Rewrote GOAL, DATASETS, and paper_plan; saved the built-in imagegen schematic
  `ep10_within_target_implementation.png` and its final prompt. No empirical
  value, profile, candidate, or held-out prediction was observed or selected.
- The active v3 policy and ASTRA executable program are unchanged. No Stage-0
  tool/contract, scientific payload, threshold, search decision, job, or role
  changed. Existing execution attempts remain as recorded below; this turn
  did not inspect or change scheduler state.
- The terminal follow-up is planned, not activated: its profile measurement,
  primary contrast/direction/margin, group inference, selection/multiplicity,
  and evaluation-access contract require a separate prospective freeze.
  Existing final/unassigned data remain closed; no new execution or canonical
  action occurred. Runtime handoff still follows the latest dated v3 entries.

Before the 2026-09-26 Stage-0 launch, no outcome-modeling experiment had
started under the v2 `GOAL.md`, `DATASETS.md`, and `SEARCH_POLICY.yaml` design.
The evidence-chain figure contains synthetic marks only and is not an
experiment. The 2024 axonal BARseq novelty overlap and the conditional
A+B-versus-A+C terminal-profile endpoint were added as design constraints; no
morphology outcome was inspected.

The paper plan was added as a design document. Creating it did not open data,
run an analysis, or change any scientific outcome.

On 2026-09-24, the first analysis stage was clarified as scientific data
qualification. No outcome was opened during this documentation change.

A final consistency pass clarified development-only A+B/A+C eligibility,
pooling-discrepancy inference, model-search anchors, negative-result reporting,
and route-specific held-out predictions. No data or outcome was opened.

## 2026-09-26 — Stage 0 launch and SEU qualification

- Run: `ep10-stage0-20260926T063634Z`.
- Authority: the scientist explicitly requested EP10 launch in this Codex
  task. The run is a standalone episode-managed exploration; no matching
  canonical Brain Researcher loop or Goal handoff was observed, and no MCP
  mutation was made.
- Scope: read the frozen Goal, dataset contract, and search policy; inventory
  the three EP10 steward assets; execute outcome-blind qualification of the
  compact SEU metadata workbook, CCFv3 ZIP member names, Allen NRRD headers,
  and structure hierarchy.
- Durable code and contract:
  `outputs/stage0/qualification_contract.yaml`,
  `outputs/stage0/source_manifest.json`, and
  `outputs/stage0/tools/qualify_resource.py`.
- Leakage tests: 9/9 passed. The durable qualifier skips `Projection class`
  cells before value decoding and rejects duplicate identities, identity-set
  mismatches, and ZIP path traversal.
- Engineering repair: an initial pre-artifact identity-join trial treated
  SpreadsheetML `_x005F_` escapes literally and reported 596 false mismatches.
  The parser was corrected to decode one pass of SpreadsheetML escapes; the
  same inputs and scientific configuration were rerun. The repaired result has
  zero identity issues and a 1,876/1,876 filename join. No SWC member was
  opened.
- Prelaunch probe deviation: an earlier one-off trusted probe materialized
  complete workbook rows transiently while emitting only allowed aggregate
  fields. It printed and persisted no `Projection class` value. The durable
  qualifier eliminates that unnecessary decoding. The workbook is therefore
  treated as touched by a trusted builder, not as pristine or candidate-visible.
- Observed support: 1,876 cells and 39 conservative `fMOST Brain ID` specimen
  groups overall. MOp has 75 cells across 17 groups; after the required 12
  development groups, at most five groups remain for final evaluation rather
  than the required eight. No listed source reaches 20 groups.
- Decision: `revise_resource_strategy_before_outcome_search`. This is a
  resource-feasibility decision, not a biological null result. The shared
  EP09--EP11 role ledger remains unfrozen, and no morphology, target-set,
  bouton, voxel, or final-group outcome was opened.
- Leakage hardening: the first mechanically clean result was retained as
  `qualification_run001`, then superseded after review showed that a global
  shared-string table was retained in memory. Run 002 resolved only header and
  allowed-cell shared-string indexes. Run 003 additionally rejects every
  named non-allowlisted metadata field, instruments ZIP-member non-read
  behavior, and omits forbidden-column occupancy. Run 004 closes the remaining
  schema edge case by rejecting populated columns with no header before
  shared-string lookup. Before handoff, the superseded run-002 durable copy was
  redacted so no retained result publishes forbidden-column occupancy. Run 005
  binds the executable explicitly to module Python 3.12.1 and upgrades the
  NRRD test to fail on any read beyond the header terminator. All five runs
  give the same support decision; run 005 is the current evidence.
- Evidence:
  `outputs/stage0/launch_identity.json`,
  `outputs/stage0/qualification_run005/qualification.json`, and
  `outputs/stage0/qualification_run005/source_group_support.csv`.

## 2026-09-26 — Lean Gao MOp qualification

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

## 2026-09-26 — Gao MOp soma geometry and prospective role freeze

- Soma access: Slurm job `45396282` requested only bytes 0--4095 for each of
  the 837 selected canonical SWCs and retained only the first validated
  46--51-byte soma-root row. It did not materialize or inspect any axon tree.
- Geometry: Slurm job `45398661` found all 837 cells atlas-geometry eligible
  (833 primary `Axon_and_dendrite`, 4 `Axon_only`). Of 48 prespecified
  radius/depth conditions, 40 were feasible and 28/36 finite-depth conditions
  were feasible. The Pareto-minimal finite conditions were 500 um/0.20 (825
  candidate cells) and 750 um/0.10 (828); the prespecified radius-first rule
  selected 500 um/0.20. Exact-distance and LineString sensitivity agreed at
  that condition.
- Role freeze: outcome-blind Slurm job `45401314` froze 12 development, 8
  final, and 56 unassigned fMOST/sample units. Development contains 209 primary
  cells, 200 supported (95.69%); final contains 54 primary cells, all 54
  supported. A same-build rerun reproduced the assignment exactly.
- Cross-episode boundary: all 15 units shared with EP11 were forced unassigned
  before optimization. EP11 roles remain unfrozen.
- Access boundary: no complete axon tree, projection target, branch, terminal,
  voxel payload, or role-specific outcome was opened. Final-role morphology
  remains sealed.
- Evidence: `outputs/stage0/gao_mop_soma_inventory.csv`,
  `outputs/stage0/mop_geometry_support.csv`,
  `outputs/stage0/mop_geometry_summary.json`,
  `outputs/stage0/gao_mop_frozen_roles.csv`, and
  `outputs/stage0/gao_mop_role_freeze_summary.json`.

## 2026-09-26 — Projection-observation freeze and development handoff

- The outcome-blind observation contract was frozen before any complete SWC
  body was fetched. It uses a 56-coordinate Allen atlas/QC map, replacing the
  coarse `MO` anchor with `MOs` so local MOp axon is not treated as a target.
- The primary exact model is prospectively limited to 16 coordinates:
  MOs, SS, ACA, STR, TH, MB, P, and MY crossed with soma-relative
  ipsilateral/contralateral side. This anatomy-defined motor-routing panel has
  at most 12,870 fixed-K subsets; it was not chosen from observed target
  prevalence or candidate effects.
- A target is detected only when a positive-length target component contains a
  rooted axon leaf/final leaf edge. Cable without a terminal-bearing component
  is recorded as `passing_only`. A zero means only `not observed in the eligible
  released reconstruction`, never biological or synaptic absence. A tree,
  registration, or complete-observation failure makes all coordinates unknown
  and excludes the cell; unknown is never zero.
- Primary modeling is restricted to the frozen 200-cell development
  common-support domain. Structural QC must retain at least 190 overall and the
  prespecified 90%-ceiling minimum in each of all 12 groups; failure means
  revise or stop without replacement, promotion, or threshold relaxation.
- The four `Axon_only` cells cannot support a sensitivity under the frozen roles
  (0 development, 1 final, 3 unassigned), so none will be opened for
  development and that sensitivity is omitted.
- The contract pins the positive-diagonal AP/DV/ML NRRD affine, Fortran-order
  loading, the pre-laterality six-neighbor erosion rule, and a complete
  node-transition rule that cannot turn an axon-to-dendrite/marker transition
  into a terminal.
- The role-safe manifest contains exactly the 200 common-support primary
  development SWCs from 12 frozen fMOST/sample units. It contains none of the
  nine unsupported development-role cells and no final, unassigned,
  `Axon_only`, or EP11-shared row. The final 55 and unassigned 573 files remain
  absent.
- A header-only provider check confirmed direct file length/type support. No
  SWC body was fetched. The current agent session cannot authenticate
  noninteractively to the Sherlock DTN, so the attended DTN transfer remains
  pending rather than being rerouted through the login node.
- Evidence: `outputs/stage0/projection_observation_contract.yaml`,
  `outputs/stage0/allen_target_vocabulary.csv`,
  `outputs/stage0/gao_mop_development_manifest.csv`, and
  `outputs/stage0/gao_mop_development_acquisition.md`.

## 2026-09-26 — Stage-0 reproducibility and exact-normalization preflight

- Slurm job `45412269` completed in 16:57 with exit 0 on a compute node. No
  SWC body, projection outcome, final cell, or network resource was opened.
- The safe soma-geometry derivation reproduced 209 development-role rows with
  exactly 200 common-support cells and nine excluded cells. The generated
  200-row acquisition manifest matched the frozen manifest exactly, contained
  all 12 development groups, and contained zero EP11-shared files.
- The metadata/geometry-only role solver reproduced the frozen 12 development,
  8 final, and 56 unassigned allocation exactly; all 15 EP11-overlap units
  remained unassigned.
- Synthetic terminal-constructor checks passed all 13 frozen tree, transition,
  laterality, terminal-versus-passing, unknown, and sealed-sensitivity cases.
- Ninety-nine outcome-blind exact-normalization suites took 650.99 seconds
  total (median 6.58 seconds, range 6.43--6.65). Each suite exhaustively scored
  all 65,536 states across the 16-coordinate panel and the bounded
  56-coordinate, `K=5` case of 3,819,816 states. This is a throughput benchmark,
  not a complete 99-null search benchmark.
- Evidence: `outputs/stage0/exact_normalization_benchmark.json` and Scratch
  run directory `stage0_preflight_v1/job_45412269`.

## Remaining uncertainty

- Gao MOp provides 76 joined released brain/specimen units and therefore passes
  the 12/8 count arithmetic, but distinct-animal identity is not certified by
  the public metadata. Exact soma-position/common-support geometry now passes;
  the complete-target observation rule is frozen but remains untested on full
  trees, and full-tree structural completeness remains unknown.
- SEU-to-Allen transform compatibility and target-calling reliability remain
  unverified because Stage 0 deliberately did not decode morphology or voxel
  payloads.

## 2026-09-27 — Gao development execution and frozen stop decision

- Sherlock DTNs expose transfer protocols but no shell, so the public-HTTP
  downloader ran as bounded Slurm `service` job `45438445`. Its pre-network
  validation reproduced exactly 200 model-eligible development cells in 12
  groups. The job downloaded 200/200 files, wrote 625,064,526 bytes, and left
  no partial, unexpected, final, unassigned, unsupported-development,
  `Axon_only`, or EP11-shared SWC.
- Job `45438679` built the constructor runtime under
  `$GROUP_HOME/users/zijiao/envs/ep10_targets_py312` with Python 3.12.1,
  NumPy 1.26.4, PyYAML 6.0.3, pynrrd 1.1.3, and typing_extensions 4.16.0.
  The clean import-location gate and `pip check` passed without a user-site or
  Scratch environment.
- Constructor job `45438680` completed with exit 0 under the prospectively
  frozen observation contract. It retained 5/200 globally eligible cells and
  excluded 195: 141 `tree_or_construction_failure`, 22
  `axon_annotation_zero`, 6 `axon_outside_volume`, and 26
  `terminal_leaf_nongrey`. Eligibility counts were below every one of the 12
  frozen per-group minima.
- Parser diagnostic job `45439110` resolved all 141 generic tree failures:
  79 transitions were `10->0`, 3 `10->10`, 1 `10->11`, 33 `10->2`, 2
  `10->3`, 12 `11->2`, 9 `12->0`, 1 `12->2`, and 1 `12->3`. Zero of the
  141 parsed successfully. Provider tags 10/11/12 therefore occur internally,
  while the frozen contract treats them as terminal-only markers. The
  constructor enforced the contract as written; it did not crash.
- The frozen attrition gate required at least 190 eligible cells and every
  group minimum. Even ignoring all 141 parser failures could yield at most
  146/200 because 54 cells independently failed atlas or terminal criteria.
  The terminal Stage-0 decision is
  `revise_or_stop_before_candidate_search` without cell/group replacement,
  promotion, tag reinterpretation, rule change, or threshold relaxation.
  M0/M1/M2, full-search nulls, candidate anatomy, final acquisition, and final
  evaluation were not run. Final and unassigned trees remain absent.
- Evidence: `outputs/stage0/gao_mop_development_structural_qc_summary.json`,
  Scratch `gao_mop_development_v1/download_log.csv`, Scratch
  `gao_mop_development_targets_v1/aggregate_summary.json`, and Slurm logs for
  jobs `45438445`, `45438679`, `45438680`, and `45439110`.

## 2026-09-27 — Provider-native v2 qualification and second stop decision

- Public SWC semantics were resolved before a revised run. The INCF SWC
  specification defines types above 7 as provider-custom, and the provider's
  pinned `pyswcloader` code includes an edge iff neither endpoint type is 3/4
  and defines terminals as topology leaves excluding types 3/4. Types 10/11/12
  were therefore preserved rather than relabelled.
- Outcome-blind topology profiler job `45440950` covered the same 200 frozen
  development files and found 1,700 custom nodes: 850 internal and 850 leaves.
  All 1,690 connected custom components had a single boundary class (1,623
  axon and 67 dendrite); no target vocabulary or projection call was computed.
- The v2 revision retained the exact manifest and froze a revised capacity
  ceiling of 146, an overall minimum of 139, and 90%-ceiling minima for the 11
  positive-capacity groups. Group `221539` remained an assigned zero-capacity
  stratum. V1 remained failed and was not relabelled.
- Structural-only Slurm job `45442349` completed with exit 0 and emitted no
  target outcomes. Only 11/200 cells were globally eligible: 82 first failed
  annotation-zero axon, 27 outside-volume axon, 74 nongrey terminal, and 6
  soma-not-MOp. All 11 positive-capacity group minima failed. All five v1
  eligible cells stayed eligible; six former parser failures recovered.
- Target-blind overlap job `45443462` evaluated all structural flags jointly.
  Strict eligibility was 11; tolerating only segment mapping failures gave 12,
  tolerating only terminal mapping failures gave 85, and tolerating both gave
  194. The last policy discards both completeness safeguards and still leaves
  group `221588` at 7/10 versus its accepted minimum of 9. It is a diagnostic,
  not authorization to relax the endpoint or gate.
- Decision: v2 target construction and candidate search remain locked. A next
  attempt would require an explicit new observation claim (for example a
  positive-only/unmapped-tolerant endpoint) and a new group gate; this is a
  material redesign, not a parser fix. Final and unassigned trees remain
  sealed.
- Evidence:
  `outputs/stage0/projection_observation_contract_v2.yaml`,
  `outputs/stage0/gao_mop_development_structural_qc_summary_v2.json`, and
  `outputs/stage0/gao_mop_development_structural_qc_overlap_v2.json`.

## 2026-09-27 — Scientist-authorized v3 positive-only submission

- The scientist explicitly authorized an exploratory positive-only,
  unmapped-tolerant redesign with 194 development cells and accepted group
  `221588` support of 7. The exact authorization and observation semantics are
  recorded in `outputs/stage0/v3_user_authorization.yaml` and
  `outputs/stage0/projection_observation_contract_v3.yaml`; v1 and v2 remain
  failed historical runs and were not relabelled.
- The target-blind overlap evidence retained 194/200 cells across the frozen
  12 development groups and excluded exactly six `soma_not_MOp` cells. It did
  not load the target vocabulary or compute projection outcomes. Under v3,
  unmapped content is omitted from positive target assignment and every zero
  means no mapped positive terminal observation, not a negative or verified
  absence.
- Constructor and validator synthetic tests and the target-blind v3 preflight
  passed. After independent code, launch, and science audits found no launch
  blocker, Slurm job `45463647` was submitted exactly once as
  `ep10-v3-targets`. It completed with exit 0 in 7 minutes 28 seconds and used
  1.10 GB peak memory. Exact postflight passed: 194 eligible cells, six and
  only six `soma_not_MOp` exclusions, the frozen 12-group support vector with
  `221588=7`, and all 16 per-cell zero-semantics columns. Final/unassigned and
  EP11-shared files were not opened. The eligible `K_i_obs` range is 1--10;
  188/194 cells have at least two recorded positives.
- The standalone plan milestone was exported through
  `bin/astra-milestone` and the deployed `br-export-astra` launcher to
  `outputs/astra/v0.0.14/astra.yaml`, with no validation warnings. The ASTRA
  document is a non-authoritative plan projection with no findings; it grants
  no execution or scientific authority.
- Before inspecting emitted target summaries or calls, the top-level contracts
  were amended to freeze the v3 recorded-positive-pattern estimand, all-194
  scoring denominator, unknown-zero semantics, null interpretation, and
  inactive biological-promotion/final endpoints. Independent science and
  machine-policy audits of that amendment passed. M0/M1/M2 remain locked until
  a validated implementation completes. Final and unassigned morphology
  remains sealed.
- Durable result summary:
  `outputs/stage0/gao_mop_development_targets_summary_v3.json`. The four
  development output tables remain in the episode-specific Scratch output
  root recorded there.
- A subsequent launch-path audit found that job `45463647`, although
  numerically successful, read its atlas through a live EP11 Scratch output
  path. That violates the episode rule against live sibling outputs as run
  inputs. The job and its results are preserved as history but are not
  eligible for downstream modeling input.
- Target-blind model-interface and synthetic-math job `45470073` completed
  with exit 0 in 6 minutes 16 seconds and used 1.06 GB peak memory. Independent
  postflight passed: 194 rows, exactly six `soma_not_MOp` exclusions, the exact
  group vector (`212074=6`, `220187=35`, `220188=25`, `220190=16`,
  `221091=33`, `221225=39`, `221418=12`, `221514=5`, `221539=1`,
  `221588=7`, `221751=13`, `233281=2`), segment-unmapped yes/no support of
  109/85, and all six synthetic interface/math checks. The job accepted no
  target calls and opened no outcome, final, or unassigned data.
- Initial replay job `45471384` failed with exit `2:0` after 13 seconds, before
  target access, because the fresh rich overlap summary lacked the compact
  frozen policy-by-group view expected by the validator. An adapter added that
  view without changing any count or gate. Replacement job `45472058` then
  completed with exit `0:0` in 9 minutes 26 seconds using 1.10 GB, only the
  EP10-local steward atlas, fresh target-blind evidence, and a new output root.
  Independent postflight passed exactly: 194 eligible and six `soma_not_MOp`
  exclusions; group support `212074=6`, `220187=35`, `220188=25`,
  `220190=16`, `221091=33`, `221225=39`, `221418=12`, `221514=5`,
  `221539=1`, `221588=7`, `221751=13`, and `233281=2`; 16 zero-semantics
  columns; `K_i_obs` distribution `1:6, 2:24, 3:24, 4:31, 5:28, 6:49,
  7:17, 8:7, 9:5, 10:3`; 188 pair-informative cells; and identities exactly
  matching the model-interface preflight. No final or unassigned data were
  opened. Historical job `45463647` remains barred; the next boundary is the
  M0 contract and synthetic preflight.

## 2026-09-27 — Frozen v3 M0 and development-fit launch

- The development-only M0 contract and engine were frozen before the real fit.
  M0 is the additive exact fixed-`K_i_obs` recorded-positive-pattern reference,
  with 12 outer and 11 inner leave-one-conservative-group-out folds, equal-group
  scoring, training-fold-only design standardization, and deterministic ridge
  tuning. It is not a biological absence or complete-target-set model.
- The engine accepts only the canonical EP10 Scratch roots
  `gao_mop_development_targets_v3_ep10_assets` and
  `v3_model_interface_preflight_v1`, joined one-to-one over the exact 194
  development identities. The sibling-backed historical root
  `gao_mop_development_targets_v3` is explicitly rejected and cannot enter a
  model fit.
- The target-free deterministic self-test passed exact dynamic-programming,
  analytic-gradient, optimizer, planted-recovery, and fold-accounting checks.
  Maximum absolute errors were `1.7763568394002505e-15` for log partition,
  `1.9984014443252818e-15` for inclusion marginals, and
  `4.107825191113079e-15` for common-offset invariance. Gradient error was
  `4.1300318e-10` absolute and `1.3729139e-9` relative; repeated optimizer
  probabilities differed by zero and the fitted gradient infinity norm was
  `1.7838188e-9`. Planted recovery reached `8.0512457e-8` bits mean conditional
  KL and `6.5401164e-5` inclusion-marginal MAE. Fold accounting reproduced
  exactly 194 rows, 12 outer folds, and 11 inner folds per outer split.
- Independent implementation review initially found incomplete binding of the
  passed interface. After the engine was changed to require the canonical
  sealed roots and passed upstream summaries, the independent audit passed.
- Slurm job `45475171` passed its in-job upstream gate and target-free self-test,
  then failed closed before writing any model output because deterministic
  L-BFGS did not meet both frozen tolerances in 500 iterations. Diagnostic rerun
  `45476204` reproduced the failure and localized it to outer-held group
  `212074`, inner-held group `220187`, and ridge `lambda=1e-5`
  (`gradient_inf=5.92721e-5`, relative objective change `2.66655e-7`). The
  absent `v3_m0_fit_v1` output was not relabeled as a result.
- Two independent numerical reviews classified this as weak-ridge,
  high-dimensional underconvergence rather than a likelihood, gradient, data,
  or resource failure. They converged on a versioned solver-only probe with
  full-dimension L-BFGS memory while preserving the exact objective, ridge
  grid, Armijo search, and both convergence tolerances.
- Solver addendum v2 preserves every v1 scientific setting and changes only
  L-BFGS history `10 -> 255`, the fail-closed cap `500 -> 2000`, and scalar
  telemetry. It requires the worst fit across two deterministic warm-path
  repeats to finish in fewer than 1,000 iterations. Existing v1 tests and a new
  255-parameter ill-conditioned stress passed; two independent code reviews
  passed after exact path sealing, recoverable launch logging, and the headroom
  gate were enforced. Targeted job `45483386` was submitted once for the exact
  outer `212074` / inner `220187` training split. It emits no held-out scores,
  coefficients, prediction vectors, target contrasts, final, or unassigned
  information.
- Targeted solver-v2 job `45483386` failed closed in 1:31. The full-memory
  change converged through `lambda=1e-5` (434 iterations), but `lambda=1e-6`
  required 1,161 iterations and therefore violated the frozen `<1000`
  production-headroom rule; `lambda=1e-7` then exhausted 2,000 iterations at
  `gradient_inf=2.59427e-6` and relative objective change `3.19515e-9`.
  Accepted steps had median 1.0, all 2,000 curvature pairs updated, and none
  were skipped. This rules out a cap-only production resubmission and activates
  the predeclared solver-only review of an expanding strong-Wolfe line search.
  No score, coefficient, prediction vector, target contrast, final, or
  unassigned information was emitted.
- The target-blind M1 atlas-geometry materializer passed its independent audit
  after output confinement and symlink rejection were made fail-closed. Job
  `45481507` stopped in launcher preflight because the virtual-environment
  interpreter symlink was incorrectly treated as a scientific-input symlink;
  no input or output was opened. The corrected replacement `45481754` completed
  `0:0` in 16 seconds. Independent postflight found exactly four files, 16
  anchors, 3,104 cell-target rows (194 cells x 16), 120 target-pair rows, no
  imputation/drop, and all target-call/SWC/realized-route/final/unassigned flags
  false.
- Solver addendum v3 replaced only the shrink-only Armijo search with
  deterministic expanding strong-Wolfe bracketing and bisection zoom; history
  255, cap 2,000, the ridge path, and both convergence tolerances were
  unchanged. Independent path/sealing and numerical audits passed after
  nonfinite objective/gradient trials were made fail-closed. Synthetic tests
  exercised expansion to step 16, bisection zoom, NaN/+Inf rejection, and the
  255-parameter stress; every accepted stress step met both Wolfe conditions.
- Exact-split targeted job `45485783` then failed closed in 17 seconds at the
  first `lambda=1000` fit after two accepted steps. Its current gradient
  infinity norm was already `2.9196468e-8`, but the preceding relative change
  was `1.2808392e-7`; the next search exhausted 60 zoom bisections. It emitted
  no coefficient/prediction/target-contrast vectors or held-out scores.
  Independent diagnosis reproduced a floating-point objective plateau: a
  trial satisfying both actual Wolfe inequalities can be discarded because a
  bracket-only equal-objective comparison runs before the curvature test.
  The next versioned diagnostic changes only that predicate order, checking
  exact Wolfe acceptance before nondecrease bracketing. It does not add a
  no-move convergence rule or change any model, grid, cap, or tolerance.
- Versioned v4 applied that acceptance-precedence rule in both zoom and outer
  expansion. Two independent audits, the full synthetic suite, and rounded
  plateau regressions passed before exact-split job `45488803` was submitted.
  The real probe nevertheless failed closed in 19 seconds with scalar
  telemetry identical to v3: the third `lambda=1000` search again used 61
  evaluations and exhausted 60 zoom iterations after the same first two
  accepted steps. Therefore predicate order was not sufficient; no observed
  trial both passed Armijo and met strong curvature. Before any further solver
  proposal, the next bounded action is a one-fit, scalar-only trace of that
  exact search. V4 emitted no model output, vector, held-out score, final, or
  unassigned information.
- Scalar trace job `45491203` completed `0:0` in 17 seconds and reconstructed
  the frozen v4 trajectory exactly: 11 evaluations accepted step `2^-10`, one
  evaluation accepted step 1, and the third search used one outer plus 60 zoom
  evaluations before the ceiling. In that failed search, the outer and first
  five zoom candidates failed Armijo by 3--4 objective ULPs; the first four
  still met curvature. At `alpha=0.015625`, the objective and Armijo bound both
  rounded exactly to the current objective, but curvature ratio was
  `0.98438 > 0.9`; smaller steps approached ratio 1. The candidate became an
  exact no-move at trial 39. Thus the deterministic search had no evaluated
  representable candidate satisfying both Wolfe conditions. The conservative
  ridge strong-convexity gap bound from dimension 255, `gradient_inf`, and
  `lambda=1000` is `1.08685e-16` absolute (`1.61641e-17` normalized), far
  below the unchanged `1e-10` relative threshold. Independent review of a
  versioned certificate rule is next; ULP-relaxed Armijo and a bare no-move
  success rule are not authorized.
- Final and unassigned data remain sealed. Neither M0 nor M1 has been fit.

## 2026-09-27 — Certified solver path and production v3 M0 pass

- V5 added only the independently reviewed ridge strong-convexity certificate
  for the two exact all-finite v4 zoom-stall failure classes. Narrow job
  `45493228` passed the frozen `lambda=1000` case without a pseudo-step or
  state/history mutation: `gradient_inf=2.9196468e-8` and the conservative
  relative gap bound was `1.6164119e-17`. Full-path job `45493559` then failed
  closed at `lambda=1e-7` after 2,000 iterations, distinguishing true
  weak-ridge conditioning from finite-precision line-search stall.
- V6 introduced the exact 255-dimensional Hessian and deterministic Cholesky
  Newton branch only for `lambda` values `1e-6`, `1e-7`, and `1e-8`; all
  stronger-ridge paths remained bitwise V5. Pair moments agreed with brute
  force to numerical precision and the analytic Hessian passed finite
  differences. Exact-prefix job `45495272` converged the formerly slow
  `1e-6` and `1e-7` fits in 6 and 7 iterations. Full probe `45495354` completed
  every fit but failed the frozen warm-versus-zero prediction check only at
  `lambda=1e-5`; diagnostic job `45495722` localized the maximum marginal and
  log-probability differences to `5.1281e-5` and `1.1302e-4`, while objectives
  agreed to `1.56e-11`. No threshold was relaxed.
- V7 therefore extended the already audited exact-Newton branch to precisely
  `lambda=1e-5`, with the other eight grid values still delegated unchanged to
  V5. Narrow job `45496608` passed the frozen equivalence thresholds, and full
  probe `45496696` passed all 36 fits, deterministic repeats, and all 12
  warm-versus-zero checks; every accepted step satisfied exact strong Wolfe,
  no Cholesky shift was needed, and the worst warm-path fit used 201
  iterations. Independent routing, numerical, and sealing audits passed.
- Production development-only M0 job `45497012` then completed `0:0` in 17:30
  using the exact EP10-local target replay and target-blind interface roots. It
  emitted the four expected files under Scratch `v3_m0_fit_v7`: 194 unique
  cross-fitted cell rows across the frozen 12 groups, a passed fit summary, the
  full-development refit, and scalar solver telemetry. All 12 outer folds had
  11 training groups and 11 inner folds; no selected value was a grid boundary
  (`outer_boundary_fold_count=0`). The full-development one-standard-error
  rule selected `lambda=0.1`.
- Independent mechanical audit passed all 1,741 recorded fits: 1,718 ordinary
  relative-change witnesses and 23 preauthorized strong-convexity stall
  certificates. The maximum gradient infinity norm was
  `9.9793248e-8`, the maximum iteration count was 216, every accepted step
  passed exact strong Wolfe, and all branch-routing and no-vector-emission
  checks passed. The 16 inclusion marginals summed to each cell's frozen `K`
  within `2.7533531e-14`; the reported gain identity was exact. Summary,
  refit, and telemetry all record `final_or_unassigned_opened=false`, and zero
  remains unknown/non-negative rather than absence.
- M0 is now a passed local exploratory reference, not a biological finding.
  M1 has not been fit. Only its target-blind geometry is frozen; the next gate
  is a separately frozen joint M1 statistical contract followed by target-free
  synthetic recovery and real-shape exhaustive-enumeration runtime/memory
  probes. Final and unassigned data remain sealed.

## 2026-09-27 — Frozen v3 M1 statistical contract

- The minimal global M1 recommendation was formed while production M0 was
  still running, from the pre-existing M0 and target-blind geometry contracts.
  After M0's mechanical pass it was written durably as
  `outputs/stage0/v3_m1_model_contract.yaml`; realized M0 scores, selected
  lambdas, coefficients, residuals, and calibration did not choose the design.
- M1 is the same exact fixed-`K_i_obs` set softmax with a jointly refit
  `B` (`17 x 15`) plus one global six-feature atlas-geometry vector `gamma`.
  `gamma=0` must reproduce M0 for the same `B`. A single positive ridge value
  penalizes all 261 parameters, and each outer, inner, and full-development fit
  recomputes both M0 design transforms and geometry scales using training rows
  only. M0 predictions are not an offset or warm start.
- Each feature is centered over the cell's full candidate-set universe, then
  divided by an equal-group/equal-cell/equal-candidate RMS learned only from
  the training fold. Exact exhaustive normalization and stable log-sum-exp are
  required because SD, pair, and MST features are nonadditive. The
  implementation must stream 21-dimensional candidate moments and expand the
  261-dimensional Hessian by Kronecker structure; candidate-by-261 matrices,
  pruning, sampling, and pseudolikelihood are forbidden.
- Before any real M1 fit, the contract requires exact K-edge and brute-force
  checks, gamma-zero M0 parity, all-coordinate gradient and Hessian-vector
  checks, a frozen 544-row planted recovery, a separate 68-row all-lambda
  warm/zero gate, and a target-free 194-row real-geometry benchmark. The latter
  uses a frozen synthetic K schedule and an all-`K=8` stress case, with an
  initial 8-hour/4-CPU/16-GB Slurm request. Joint and geometry-residual rank and
  condition remain unverified stop gates, not asserted results.
- Only after every target-free gate passes may a narrowly scoped first target
  access read `provider_neuron_id` and physical column `K` to repeat rank and
  condition checks for every actual training fold. It may not inspect realized
  set membership or fit an M1 objective. Real M1 remains unlaunched, and final,
  unassigned, SWC, and realized-route data remain sealed.

## 2026-09-27 — Target-free v3 M1 probe submitted

- The frozen M1 target-free engine, focused regression suite, and Slurm
  launcher passed two independent read-only implementation audits. The engine
  uses exact exhaustive fixed-K normalization, streamed 21-dimensional
  candidate moments, the complete 261-dimensional joint Hessian, and the
  prospectively frozen deterministic solver. The launcher has no target, M0
  artifact, SWC, final, or unassigned input route.
- Lightweight engine gates and the focused 11-test suite passed before
  submission; shell syntax and Slurm test-only validation also passed. The
  heavy 544-row recovery, 68-row all-lambda warm/zero gate, 79-training-set
  target-free rank gate, and all-K=8 real-shape benchmark were not run on the
  login node.
- One real target-free probe was submitted as Slurm job `45534222` with the
  frozen 8-hour, 4-CPU, 16-GB request. Its stdout and job-scoped result/log
  paths are under the EP10 Scratch tree. No duplicate job or pre-existing
  output root was present.
- Job completion alone will not authorize target access. Postflight must check
  whole-job elapsed time below 6.4 hours, batch-step MaxRSS below 12.8 GiB,
  every scientific/numerical gate, and remaining EP10 policy budget. Actual K
  remains sealed until those pass; realized sets, final, and unassigned remain
  sealed beyond that.

## 2026-09-27 — Synthetic design failure, addendum, and narrow rank pass

- Target-free probe `45534222` failed closed after 11/11 focused tests and its
  in-probe unit gates, before the planted recovery fit or real-shape benchmark.
  The failure was the full 544-row synthetic rank gate. Slurm recorded exit
  `2:0`, elapsed 1:50, and batch MaxRSS 119,780 K. No real K, target call, M0
  artifact, SWC, final, or unassigned file was opened.
- The frozen synthetic generator had assigned layer by `row_index mod 4` and
  hemisphere by `row_index mod 2`. Under the frozen Helmert coding, hemisphere
  was therefore an exact linear combination of the three layer columns. The
  17-column design had rank 16, bounding the B block at 240 and the complete
  joint Gram at 246 rather than the required 261. This made the gate
  mathematically impossible; no rank or condition threshold was relaxed.
- `v3_m1_synthetic_design_addendum_v1.yaml` prospectively changed only the
  target-free synthetic hemisphere schedule to `(row_index // 4) mod 2` and
  kept the physical ML coordinate consistent with side. It consumes no random
  draw. Seeds, normal-then-uniform order, frozen uniforms, K schedules,
  candidate counts, planted parameters, model, six geometry features, rank and
  condition thresholds, solver, resources, and all seals remain unchanged.
  Two independent read-only audits passed before revised execution.
- Narrow job `45535744` then passed 12/12 tests and the revised synthetic-only
  rank gate. The 544- and 68-row designs both have rank 17. For the 544-row
  joint Gram, B rank is 255, geometry rank is 6, geometry-beyond-M0 Schur rank
  is 6, and joint rank is 261; their condition numbers are approximately
  2,950, 55.6, 86.6, and 11,445 respectively, all relevant criteria passing.
  No signed block had a negative eigenvalue beyond its tolerance, and the
  B-to-geometry cross-range relative defect was `3.98e-14`. This authorizes
  only the complete target-free v2 probe, not actual-K access or real M1.

## 2026-09-27 — Sampled-recovery failure and exact-population addendum

- Full target-free v2 probe `45536046` passed 12/12 focused tests, its unit
  gates, and the revised 544-row rank gate, then failed closed at the planted
  recovery comparison before the 68-row warm/zero gate and real-shape
  benchmark. Slurm recorded `FAILED`, exit `2:0`, elapsed 1:55, and batch
  MaxRSS 119,252 K. The one-categorical-draw fit had mean conditional KL
  `0.3335485051` bits, inclusion-marginal MAE `0.0513892757`, and planted-versus-
  recovered gamma cosine `0.9671683871`; the frozen hard limits were 0.01,
  0.02, and 0.90. No real K, realized target set, M0 artifact, SWC, final, or
  unassigned file was opened.
- For 261 fitted parameters and 544 equal-weight categorical observations, the
  first-order regular estimation scale is `261 / (2*544*ln(2)) =
  0.3460876890` bits. The observed KL is 96.38% of that scale. This diagnoses
  the high-precision one-hot recovery criterion as a population target
  misapplied to ordinary finite-sample estimation error; it does not justify
  relaxing the threshold to 0.334 bits and does not demonstrate that the 194
  real development cells are adequate for M1, cross-validation, M2, or any
  scientific inference.
- `outputs/stage0/v3_m1_synthetic_recovery_addendum_v2.yaml` prospectively
  changes only the 544-row hard-acceptance reference. The retained PCG64
  categorical sample and ridge fit remain an unchanged descriptive diagnostic,
  including the existing solver convergence witness. Hard recovery now fits
  from zero to the exact planted population cross-entropy using exhaustive
  expected sufficient statistics. Its objective, gradient, and Hessian are
  respectively exact expected NLL plus ridge, model-minus-planted expected
  statistics plus ridge, and exact conditional covariance plus ridge.
- The original acceptance values remain unchanged: population conditional KL
  at most 0.01 bits, inclusion-marginal MAE at most 0.02, and gamma cosine at
  least 0.90. Independently, `||theta_star||^2=2.9275`, `lambda=1e-6`, 261
  parameters, and the frozen `1e-7` gradient tolerance imply a raw
  ridge-plus-solver population-KL upper bound of `2.76875e-6` nats
  (`3.99446e-6` bits). A prospectively frozen 10% numerical allowance gives an
  implementation-consistency witness ceiling of `3.045625e-6` nats
  (`4.39391e-6` bits). This witness is not a new scientific threshold; its
  Pinsker marginal-MAE implication is report-only.
- Seeds, rows, groups, K schedules, categorical draws, planted parameters,
  model, six geometry features, exact normalization, lambda, optimizer,
  thresholds, 68-row warm/zero gate, real-shape benchmark, resources, and all
  data seals remain unchanged. The next action is an independently audited,
  narrow target-free exact-population recovery probe. A narrow pass authorizes
  only a complete target-free v3 probe, not actual-K access or a real M1 fit.
- The implementation then passed 15/15 focused tests, including direct
  exhaustive population objective/gradient/Hessian parity, finite differences,
  Hessian-vector checks, ridge-at-planted behavior, parser sealing, and Newton
  correction threading. Independent policy/launcher and code audits found no
  narrow-probe blocker. After a successful scheduler test-only check, narrow
  synthetic-only job `45539044` was submitted with 30 minutes, 4 CPUs, and 8
  GB. It has no real interface, geometry, K, target, M0 artifact, SWC, final,
  or unassigned input route.
- Job `45539044` completed `0:0` in 2:03 with batch MaxRSS 118.50 MiB and
  passed 15/15 tests. The exact population fit converged in five unshifted
  Newton steps. Its mean conditional KL was `8.73054e-10` bits, inclusion
  marginal MAE `2.41900e-6`, and gamma cosine `0.99999999988`; its
  `6.05155e-10`-nat KL also passed the `3.045625e-6`-nat implementation
  witness. The full joint rank was 261 with condition number about 11,445.
  The sampled diagnostic exactly reproduced job `45536046` and remained
  non-gating. Independent postflight passed, all data-access seals remained
  false, and this authorizes only the complete target-free v3 probe.
- The complete target-free launcher was advanced to the engine's v3 output
  guard, given the frozen recovery addendum explicitly, and otherwise retained
  the 8-hour, 4-CPU, 16-GB request and exact target-blind interface/geometry
  inputs. Independent launcher audit and scheduler test-only validation passed
  with no duplicate queued job. Complete target-free v3 job `45540327` was
  then submitted; actual K, realized sets, M0 artifacts, final, and unassigned
  remain sealed pending its full postflight and budget reconciliation.
- Full v3 job `45540327` passed 15/15 focused tests and every in-engine unit
  gate, then failed closed before a synthetic-gates payload or real-shape
  benchmark was emitted. Slurm recorded `FAILED 2:0`, elapsed 2:31, and batch
  MaxRSS 127,624 K. The error was an exact strong-Wolfe zoom bracket reaching
  floating-point resolution without satisfying the frozen 261-dimensional
  certificate. No real K, target call, M0 artifact, SWC, final, or unassigned
  input was opened. The complete job will not be rerun unchanged: the next
  action is a synthetic-only scalar diagnostic that identifies the exact
  68-row path/lambda and records gradient/gap telemetry without changing any
  numerical behavior or threshold.
- The logging-only diagnostic implementation passed 16/16 focused tests,
  independent code review, shell syntax, and scheduler test-only validation.
  It atomically checkpoints every fit in the exact 68-row order
  `warm_repeat1`, `warm_repeat2`, then zero-start fits, while emitting scalars
  only and accepting no real input path. Diagnostic job `45585270` was
  submitted with 30 minutes, 4 CPUs, and 8 GB; no solver behavior, model,
  threshold, or data-access rule changed.
- Diagnostic job `45585270` completed its intended localization and then
  failed closed with Slurm exit `2:0` in 59 seconds; batch MaxRSS was 122,648 K.
  The very first fit (`warm_repeat1`, lambda `1000`, all-zero start) accepted
  one Newton step and stalled at optimizer iteration 2 when the strong-Wolfe
  zoom bracket reached binary64 resolution. All values were finite. The
  current gradient infinity norm was `6.9385468515204e-7`, so the frozen
  `1e-7` gradient predicate failed, while the upward-rounded strong-convexity
  relative objective-gap bound was `1.010793330691067e-14`, well below the
  frozen `1e-10` gap tolerance. The last Newton system needed no diagonal
  shift, had scaled backward error `2.378422688082158e-18`, and predicted only
  `5.583062496243228e-16` objective decrease. The final Wolfe bracket width
  was `6.938893903907228e-18`; Armijo passed by
  `3.552713678800501e-15`, but the curvature ratio was
  `0.9687194823250145` versus the frozen `0.9` requirement.
- This is a localized finite-precision line-search/convergence conflict, not a
  resource, rank, recovery, or nonfinite-value failure. The diagnostic opened
  no real K, target call, M0 artifact, SWC, final, or unassigned input. No EP10
  job is currently active. The complete target-free probe will not be rerun
  unchanged; independent mathematical and contract audits will first select
  the smallest principled repair, followed by the exact 68-row synthetic gate.
- `[m1_target_free_qualification]` Three independent audits rejected a
  gap-only certificate because it would remove the frozen gradient predicate.
  They instead approved `outputs/stage0/v3_m1_solver_addendum_v1.yaml`, which
  retains the exact model, Wolfe constants, `1e-7` gradient criterion, `1e-10`
  relative-objective criterion, and existing gradient-AND-gap stall
  certificate. The sole numerical change evaluates the same exhaustive
  softmax line-objective difference and directional derivative directly with
  stable likelihood-ratio identities, avoiding subtraction of two rounded
  total objectives.
- `[m1_target_free_qualification]` The implementation uses `log1p/expm1` for
  centered increments no larger than 0.5 and independent shifted baseline and
  combined normalizers otherwise, including a regression where a baseline-
  underflow candidate is revived by the trial step. It keeps accepted-state
  absolute objective and full gradient unchanged; stable `delta_F` is used
  only for the exact Armijo/bracketing predicates and the unchanged relative-
  change calculation. Focused lightweight tests passed, including observed-
  score centering, K=0/16, ridge and population correction, sub-ULP Wolfe,
  nonfinite negatives, and certificate invariance. Independent final static
  review found no numerical blocker.
- `[m1_target_free_qualification]` Before submission, the single canonical
  Scratch basename was aligned across addendum, engine, and launcher. Shell
  syntax, Python compilation, YAML loading, and scheduler test-only validation
  passed; test-only reported hypothetical job `45591115`. At
  `2026-09-28T00:12:17Z`, no EP10 job was queued or running. The selected next
  attempt is the synthetic-only launcher
  `outputs/stage0/tools/probe_v3_m1_synthetic_warm_zero_solver_v1.sbatch`
  with 30 minutes, 4 CPUs, and 8 GB. It will run all 22 focused tests and then
  the exact 68-row warm/warm/zero gate. No real K, target call, M0 artifact,
  SWC, final, or unassigned input is accepted.
- `[m1_target_free_qualification]` Submitted that one narrow attempt as Slurm
  job `45591145` at `2026-09-28T00:12:48Z`. Babysit it to a terminal state;
  completion alone does not authorize the complete target-free relaunch or any
  real-data access. First audit all focused tests, every 68-row comparison and
  convergence witness, scalar Wolfe telemetry, access seals, elapsed time, and
  batch-step MaxRSS.
- `[m1_target_free_qualification]` Job `45591145` failed closed with exit
  `2:0` after 2:44; batch MaxRSS was 122,272 K. All 22 focused tests passed in
  38.777 seconds, including the exact seed-20260928 lambda-1000 regression.
  All 36 requested fits then completed with valid ordinary convergence
  witnesses; warm-repeat1 and warm-repeat2 were bit-identical at every lambda,
  and all Wolfe, stable-line, and access-seal telemetry remained valid.
- `[m1_target_free_qualification]` The sole failed comparison was the frozen
  zero-start equivalence check at lambda `1e-8`: maximum candidate log-
  probability difference was `0.0011745975798476138` versus `1e-4`. The same
  comparison passed its maximum probability difference
  (`5.579142176714299e-10` versus `1e-5`) and objective difference
  (`1.9539925233402755e-14` versus `0.00014355`). The zero-start fit converged
  in 18 unshifted Newton steps with gradient infinity norm
  `4.9176049732301205e-11`; the two warm paths remained exactly reproducible.
  Thus the stable-line repair fixed the original strong-ridge stall, but the
  complete gate exposed a distinct weak-ridge path-equivalence precision
  failure. No real K, target call, M0 artifact, SWC, final, or unassigned input
  was opened. The job will not be rerun unchanged; independent audits will
  derive the smallest prospective numerical refinement without relaxing the
  frozen `1e-4` log-probability gate.
- `[m1_target_free_qualification]` Independent contract and numerical audits
  diagnosed the lambda-`1e-8` zero-start result as approximately one Newton
  step behind the reproducible warm endpoint. They rejected a new gradient or
  gate-aware stopping threshold and approved the prospective
  `outputs/stage0/v3_m1_solver_addendum_v2.yaml`: after the first accepted-step
  ordinary witness, every fit makes one additional unchanged exact-Hessian
  Newton/strong-Wolfe attempt. The original `1e-7` and `1e-10` predicates,
  finite-stall certificate, warm/zero thresholds, lambda grid, and exact
  stationary-initial witness remain unchanged.
- `[m1_target_free_qualification]` The v2 implementation records the first
  witness and one-shot confirmation telemetry, never reads a comparison metric
  inside optimizer stopping, and preserves an explicitly reviewed exact-zero
  exception where no strict descent direction exists. Independent review found
  no blocker. Python compilation, YAML and contract loading, shell syntax, and
  scheduler test-only validation passed; test-only reported hypothetical job
  `45594338`. At `2026-09-28T00:28:41Z`, no EP10 job was queued or running.
  The selected next attempt is the unique synthetic-only launcher
  `outputs/stage0/tools/probe_v3_m1_synthetic_warm_zero_solver_v2.sbatch` with
  30 minutes, 4 CPUs, and 8 GB. Real K, target calls, M0 artifacts, SWCs,
  final, and unassigned data remain sealed.
- `[m1_target_free_qualification]` Submitted exactly that one attempt as Slurm
  job `45594960` at `2026-09-28T00:29:06Z`. Babysit it to terminal; a narrow
  pass authorizes only the complete target-free v4 relaunch, not actual-K or
  any M0-artifact, SWC, final, or unassigned access.
- `[m1_target_free_qualification]` Job `45594960` completed `0:0` in 2:51
  with batch MaxRSS 121,964 K (119.11 MiB). All 22 focused tests passed, all
  36 fits made and accepted exactly one confirmation step, and all 12 frozen
  warm/zero comparisons passed without a stall certificate or exact-zero
  exception. At lambda `1e-8`, maximum log-probability difference fell from
  `0.0011745976` to `2.7850345e-7` against the unchanged `1e-4` limit;
  probability difference was `1.4220570e-13` against `1e-5`, and the zero-start
  terminal gradient infinity norm was `1.2528197e-14`. Warm repeats remained
  bit-identical and every real-input, target, M0-artifact, SWC, final, and
  unassigned access seal remained false. This authorizes exactly one complete
  target-free v4 probe after launcher review; it does not authorize actual-K
  access or a real M1 fit.
- `[m1_target_free_qualification]` The engine output/stage guard and a new
  launcher were advanced to v4. The launcher requires both solver addenda with
  v2 active, retains the frozen 8-hour/4-CPU/16-GB request, and accepts only the
  exact target-blind interface and geometry roots. Independent review found no
  blocker and confirmed coverage of unit gates, exact population recovery, the
  full 68-row gate, all 79 real-shape training sets, and the all-K8 benchmark.
  Bash, contract/output-guard checks, and scheduler test-only validation passed;
  test-only reported hypothetical job `45596001`. At
  `2026-09-28T00:36:32Z`, no EP10 job was queued or running. The selected next
  attempt is `outputs/stage0/tools/probe_v3_m1_target_free_v4.sbatch`; all
  target, actual-K, M0-artifact, SWC, final, and unassigned inputs remain sealed.
- `[m1_target_free_qualification]` Submitted exactly that one full attempt as
  Slurm job `45596378` at `2026-09-28T00:36:55Z`. Babysit it to terminal and
  perform independent postflight plus remaining-budget reconciliation before
  any actual-K access. Completion by itself does not authorize a real M1 fit.
- `[m1_target_free_qualification]` Refreshed
  `outputs/astra/v0.0.14/astra.yaml` into the new skill's experiment-program
  form: seven stable root analyses plus nested adaptive search, falsification,
  null, and assessment analyses. The target-free analysis consumes only the
  frozen contracts/addenda, target-blind model interface, and geometry; the
  first target-derived edge is `m1_actual_k_gate`, and final/unassigned data do
  not appear as inputs. The pinned ASTRA 0.0.14 schema and semantic validators
  passed with 0 errors and 0 recommendations. Runtime state remains here in the
  ledger: active attempt `45596378`; exact next action is to babysit it and, only
  after a passed postflight and budget reconciliation, select the actual-K gate.
- `[m1_target_free_qualification]` Job `45596378` completed `0:0` in 4:50
  with batch MaxRSS 127,452 K (124.46 MiB). All 22 focused tests, exact
  population recovery, all 36 warm/warm/zero fits and 12 comparisons, all 79
  real-shape training-fold rank/condition checks, and the all-K8 benchmark
  passed. The benchmark projected 17.56749 core-hours and 4.39187 wall-hours
  for the frozen production scope. Every actual-K, target-call, M0-artifact,
  SWC, final, and unassigned access seal remained false.
- `[m1_target_free_qualification]` Independent accounting reconciled all 40
  recorded jobs. Conservative allocated use is 4.34389 of 1800 core-hours;
  the strictest episode wall-clock basis is 42.09833 of 144 hours; Scratch use
  is 0.651084 of 1536 GB. Even after the benchmark projection, 1778.08862
  core-hours and 97.50979 strict wall-hours remain. Compute, memory, wall-time,
  and storage therefore impose no blocker.
- `[m1_actual_k_gate]` At `2026-09-28T00:54:35Z`, selected the next authorized
  experiment: a separate, narrow actual-K-only geometry gate. It may read only
  `provider_neuron_id` and physical `K` from the frozen development target-call
  asset, align them to the 194-row target-blind interface, and apply the same
  frozen transforms, Gram construction, and 79 training-fold rank/condition
  checks used by the passed target-free gate. It may not read or emit realized
  membership, M0 artifacts, SWCs, final, or unassigned data. No target-derived
  field has yet been opened and no job has yet been submitted; code and launcher
  audit plus scheduler test-only must pass first.
- `[m1_actual_k_gate]` Implemented a separate rank-only gate and launcher. A
  fixed-index binary RFC4180 scanner assembles only physical field 0
  (`provider_neuron_id`) and, after interface membership is known, physical
  field 8 (`K`) for the exact 194 modeled cells. The other 88 fields are never
  decoded or assembled. It requires the canonical 200-row replay, exact 194
  interface matches, six excluded rows with blank K, the passed target-free v4
  artifact, and the recorded postflight plus budget seals before target access.
  A one-past-end candidate index poisons every realized-set route.
- `[m1_actual_k_gate]` Ten synthetic tests passed, including quoted commas,
  embedded newlines, escaped quotes, malformed quoting, symlink rejection,
  200-to-194 projection, noncanonical K, aggregate redaction, exact 79-fold
  coverage, and independent recomputation of every rank/condition predicate.
  Python compilation, frozen-contract/prerequisite checks, Bash syntax, ASTRA
  schema and semantic validation, and scheduler test-only passed; test-only
  reported hypothetical job `45602180`. Three independent reviews found no
  blocker. At `2026-09-28T01:08:44Z`, no actual-K job was queued or running and
  the real target CSV had not been opened. The selected next attempt is exactly
  `outputs/stage0/tools/gate_v3_m1_actual_k_v1.sbatch`, 30 minutes, 4 CPUs,
  8 GB. It may open only ID/K; membership, M0, SWC, final, and unassigned remain
  sealed.
- `[m1_actual_k_gate]` Submitted exactly that one attempt as Slurm job
  `45602325` at `2026-09-28T01:09:10Z`. Babysit it to terminal and audit the
  aggregate-only 79-fold result plus scheduler accounting. This submission
  authorizes ID/K access only; it does not authorize realized membership, a
  real M1 fit, M0 artifacts, SWCs, final, or unassigned data.
- `[m1_actual_k_gate]` Job `45602325` completed `0:0` in 25 seconds with batch
  MaxRSS 103,468 K (101.04 MiB). All 10 focused tests and all 79 frozen
  training-fold gates passed. Minimum joint rank was 261, minimum raw and
  Schur-residual geometry ranks were both 6, and maximum joint condition number
  was `3665038.1162885157` against the unchanged `1e10` ceiling. The report is
  aggregate-only: it contains no provider IDs, K values/distribution, arrays,
  or fold rows, and no candidate score or model fit was computed. Independent
  postflight found no blocker. Only development provider ID and physical K were
  opened; realized membership, M0 artifacts, SWCs, final, and unassigned data
  remain sealed.
- `[m1_actual_k_gate]` This PASS satisfies the final pre-fit geometry gate. It
  authorizes preparation of a separate, independently audited development-only
  real M1 fit that may read realized positive-set membership. M0 remains
  forbidden to design, tuning, initialization, and fitting and may be opened
  only after M1 outer-cross-fitted predictions exist; SWCs, final, and
  unassigned remain forbidden.
- `[v3_m1]` At `2026-09-28T01:15:46Z`, selected the next authorized experiment:
  implement and independently qualify the development-only real M1 runner.
  It will use the frozen 12-outer/11-inner LOGO controller, descending
  12-lambda warm paths, largest-lambda one-SE rule, zero-start selected outer
  refits, the boundary-review stop, and a zero-start full-development
  coefficient refit. The existing target-free engine remains target-free.
  Realized membership may be opened only by the eventual audited runner; no
  membership has yet been opened in this stage. M0 has no input route and stays
  sealed until all 194 M1 outer predictions and the complete M1-only fit exist.
  SWCs, final, and unassigned remain forbidden. Resource approval continues to
  use the conservative 157-path projection even if exact identical-fold reuse
  reduces executed paths.
- `[v3_m1]` At `2026-09-28T01:50:20Z`, selected the first real-M1 execution
  attempt after implementation and pre-submit qualification. The separate
  production runner projects only development `provider_neuron_id`, exact
  `fMOST_brain_id`, physical `K`, and the frozen 16 `detected__...` fields after revalidating actual-K job
  `45602325`; it has no M0, SWC, final, or unassigned input route. It executes
  66 exact unordered leave-two warm paths, 12 zero-start outer refits, 12
  leave-one full-tuning warm paths, and one zero-start full refit, while budget
  approval remains based on the conservative uncached 157-path projection.
  Python compilation, all 11 focused synthetic tests, Bash syntax, the pinned
  ASTRA 0.0.14 schema and semantic validators, and scheduler test-only passed;
  the latest test-only job was hypothetical `45609712`. No same-name or
  episode-path job was active at this check. The exact selected launcher is
  `outputs/stage0/tools/fit_v3_m1_real_v1.sbatch` with 8 hours, 4 CPUs, and
  16 GB on `russpold,normal`; Slurm stdout will be directed to the job-scoped
  EP10 Scratch path. No new target membership has yet been opened and no real
  M1 job has yet been submitted. The next side effect, after the final
  independent verdict and one immediate duplicate check, is exactly one
  `sbatch --parsable` submission.
- `[v3_m1]` At `2026-09-28T01:56:43Z`, repaired the final independent-audit
  findings before submission. The fixed-index target projection now also reads
  physical `fMOST_brain_id` for the exact 194 matched rows and requires its
  exact agreement with the target-blind interface group; a mismatch regression
  fails closed. M1 reporting now includes the full `K_i_obs` count table and
  `2 <= K_i_obs <= 15` pair-informative support by group. ASTRA now declares
  all three synthetic/solver addenda consumed by the runtime. Same-job requeue
  recovery is limited to an empty manifest-bootstrap state or validated
  checkpoints, and a pass summary plus all four ordinary artifacts returns
  idempotently before membership is reopened. Python compilation, all 11
  focused tests, Bash syntax, ASTRA schema/semantic validation, and scheduler
  test-only all passed after these repairs; the new hypothetical job is
  `45609712`. No target membership was opened and no real job was submitted
  during repair.
- `[v3_m1]` Submitted exactly one M1-only production attempt as Slurm job
  `45609750` at `2026-09-28T01:57:23Z`, after a final exact-name duplicate
  check returned empty and the independent final audit reported no blocker.
  The request is 8 hours, 4 CPUs, and 16 GB on `russpold,normal`; Slurm stdout
  is job-scoped in EP10 Scratch and the runner uses an append-only sibling log
  without `tee` or a watcher. The job was initially `PENDING`. Its only new
  authorized target access is development ID/group/K/detected16 for the 194
  frozen interface cells. M0, SWCs, final, and unassigned remain sealed. The
  exact next action is to babysit job `45609750` at scheduler-safe intervals,
  inspect its phase/checkpoints without opening sealed paths, and diagnose any
  failure before considering a rerun.
- `[v3_m1]` By `2026-09-28T02:00:27Z`, job `45609750` was `RUNNING` on
  `sh03-06n07`; all 11 in-job focused tests passed and the job-scoped manifest
  reached `membership_loaded_outer_tuning`. Development ID/group/K/detected16
  has therefore been opened for the authorized 194-cell M1 fit. The manifest
  still records M0, SWC, final, and unassigned access as false. No boundary or
  solver failure is present; continue scheduler-safe babysitting.
- `[v3_m1]` Job `45609750` failed closed `2:0` after 49:21, with batch MaxRSS
  108,772 K (106.22 MiB). It durably completed 53 of 66 unordered leave-two
  warm paths and six outer selections; those selections were all interior
  (`1, 1, 1, 1, 0.01, 1`) with boundary count zero. The first absent path in
  the frozen execution order is exactly `leave_two__221418__221588`; its warm
  path raised `UncertifiedLineSearchFailure` before a checkpoint could be
  committed. The failure summary confirms M0, SWC, final, and unassigned stayed
  sealed. Do not rerun unchanged. The exact next action is one narrow scalar
  diagnostic of that training set and lambda/certificate context, followed by
  a reviewed contract-preserving solver repair if supported by the evidence.
- `[v3_m1_diagnostic]` At `2026-09-28T02:55:51Z`, selected exactly one narrow
  replay of failed path `leave_two__221418__221588`. It validates the exact
  failed job `45609750` and its first 53 atomic checkpoints, reconstructs only
  that authorized development training set, and stops at the first solver
  failure while emitting scalar lambda, initialization, iteration, gradient,
  line-search, and certificate context. It emits no coefficients, cell IDs,
  membership vectors, predictions, K distribution, M0 artifact, SWC, final,
  or unassigned content. Python compilation, Bash syntax, and the production
  runner's 11 focused `unittest` cases passed; scheduler test-only reported
  hypothetical job `45615918`. Independent audit returned no blocker after
  verifying that membership access is durably marked immediately before the
  only authorized target read. No same-name diagnostic or production job was
  active at this check. The selected launcher is exactly
  `outputs/stage0/tools/diagnose_v3_m1_real_failure_v1.sbatch`, requesting 30
  minutes, 4 CPUs, and 8 GB on `russpold,normal`; the next side effect is one
  `sbatch --parsable` submission followed by scheduler-safe babysitting.
- `[v3_m1_diagnostic]` Submitted exactly that one scalar diagnostic as Slurm
  job `45616376` at `2026-09-28T02:56:19Z`, after the immediate exact-name
  duplicate check returned empty. The scheduler accepted the frozen request
  of 30 minutes, 4 CPUs, and 8 GB on `russpold,normal`; it was initially
  `PENDING`. This does not authorize any solver change or production rerun.
  Development membership is the only target-derived input; M0 artifacts,
  SWCs, final, and unassigned remain sealed. Babysit at scheduler-safe
  intervals through terminal state, then inspect only the scalar diagnostic.
- `[v3_m1_diagnostic]` Job `45616376` reached its designed fail-closed terminal
  state `2:0` after 2:49 with batch MaxRSS 105,588 K (103.11 MiB), but the
  original solver failure did not reproduce. The exact
  `leave_two__221418__221588` replay completed all 12 descending lambdas from
  zero at `1000` through warm-started `1e-8`; every fit returned the unchanged
  gradient-and-relative-objective convergence witness plus its required one
  accepted confirmation step. The diagnostic then set
  `status=fail_failure_not_reproduced` and exited nonzero as specified. This
  rules out the prior guess of a stable lambda-1000 failure and does not itself
  authorize accepting or rerunning around the original uncertified stall.
  Development ID/group/K/detected16 was reopened for this authorized replay;
  coefficients, cell IDs, membership vectors, predictions, M0 artifacts,
  SWCs, final, and unassigned were not emitted or opened. The exact next action
  is independent numerical-reproducibility and policy review before any solver
  change, checkpoint reuse, or production requeue.
- `[v3_m1_diagnostic]` At `2026-09-28T03:15:07Z`, selected one original-backend
  causal replay after the Intel diagnostic failed to reproduce the AMD
  production stall. The launcher is pinned to `CPU_SKU:7502` (AMD EPYC Rome),
  requests and runtime-verifies four effective OpenBLAS threads, rejects
  inherited OpenBLAS/NumPy CPU overrides, and records the dispatched backend.
  It executes two independently zero-started copies of the exact failed
  `leave_two__221418__221588` 12-lambda warm path, matching the frozen
  `warm_path_repeats: 2` rule. Any expected solver failure is retained as
  scalar context while repeat 2 still runs; if both paths complete, only
  objective, coefficient-maximum, and probability-maximum differences are
  persisted against the unchanged `1e-12`, `1e-10`, and `1e-10` tolerances.
  Coefficients, membership vectors, predictions, and candidate arrays remain
  memory-only. Python compilation, Bash syntax, the 11 runner tests, and six
  focused diagnostic tests passed. Scheduler test-only reported hypothetical
  job `45619002` on AMD 7502 node `sh03-06n05`. Three independent reviews
  returned no blocker; no same-name or production job was active. The exact
  selected launcher is
  `outputs/stage0/tools/diagnose_v3_m1_real_repro_v2.sbatch`, 30 minutes,
  4 CPUs, and 8 GB. M0 artifacts, SWCs, final, and unassigned remain sealed.
- `[v3_m1_diagnostic]` Submitted exactly that one original-backend replay as
  Slurm job `45619301` at `2026-09-28T03:15:30Z`, after the immediate
  exact-name duplicate check returned empty. The scheduler accepted the
  `CPU_SKU:7502` constraint and frozen 30-minute/4-CPU/8-GB request; the job
  was initially `PENDING` with reason `None`. This authorizes only the two
  scalar-diagnostic repetitions and does not authorize a solver change or
  production resume. Babysit at scheduler-safe intervals through terminal.
- `[v3_m1_diagnostic]` Job `45619301` completed `0:0` in 28 seconds on
  `sh03-06n05` with batch MaxRSS 107,728 K (105.20 MiB). Runtime telemetry
  confirmed AMD 7502, OpenBLAS core `Zen`, and four effective threads. Both
  independent repetitions reproduced the original failure at the first fit,
  `lambda=1000`, optimizer iteration 3, after two accepted steps and the
  ordinary convergence witness. In both, the forced confirmation Newton solve
  had a negative ordinary dot-product direction but the separately stable
  direct directional derivative rounded to exact `0.0`; strong-Wolfe therefore
  rejected it as non-descent before any evaluation. Gradient infinity norm was
  `6.661338147750939e-16`; the unchanged gradient and relative strong-convexity
  gap predicates both passed, but the old certificate classified this failure
  type as ineligible and correctly failed closed. The diagnostic decision is
  `trigger_single_thread_arm`. No coefficient, prediction, membership vector,
  M0 artifact, SWC, final, or unassigned output was emitted; all sealed flags
  remain false. The next experiment is the same two-repeat path on the same
  AMD SKU with one BLAS thread, without changing model or solver mathematics.
- `[v3_m1_diagnostic]` At `2026-09-28T03:21:25Z`, selected exactly that
  single-thread arm. It inherits the identical failed training set, two
  independently zero-started 12-lambda paths, solver code, objective, Hessian,
  Wolfe rules, convergence thresholds, certificate, and scalar-only repeat
  comparisons from the audited four-thread diagnostic. The only numerical
  change is `cpus-per-task=1` with effective OMP/OpenBLAS/MKL threads required
  to equal one on `CPU_SKU:7502`; inherited backend overrides are rejected.
  A two-repeat pass is diagnostic evidence only and requires a prospective
  single-thread execution addendum plus the frozen 68-row qualification and a
  clean production root; no four-thread checkpoint may be reused. Python
  compilation, Bash syntax, the 11 runner tests, six v2 tests, and three v3
  tests passed. Scheduler test-only reported hypothetical job `45619860` on
  AMD 7502 node `sh03-06n12`. Three independent audits returned no blocker,
  and no same-name or production job was active. The selected launcher is
  exactly `outputs/stage0/tools/diagnose_v3_m1_real_repro_1thread_v3.sbatch`,
  30 minutes, 1 CPU, and 8 GB. M0 artifacts, SWCs, final, and unassigned stay
  sealed.
- `[v3_m1_diagnostic]` Submitted exactly that one single-thread arm as Slurm
  job `45619990` at `2026-09-28T03:21:56Z`, after the immediate exact-name
  duplicate check returned empty. The scheduler accepted `CPU_SKU:7502`, one
  requested CPU, 8 GB, and 30 minutes; it was initially `PENDING` with reason
  `None`. The runtime itself must still prove both `SLURM_CPUS_PER_TASK=1` and
  effective OpenBLAS threads=1 before membership access. Babysit at
  scheduler-safe intervals; this submission does not authorize checkpoint
  reuse or production execution.
- `[v3_m1_diagnostic]` Job `45619990` failed pre-access `2:0` after 14 seconds
  with batch MaxRSS 48,008 K (46.88 MiB). All 11+6+3 in-job tests passed, but
  Sherlock's `CR_CORE_MEMORY` scheduling rounded the one-CPU TRES request to a
  two-CPU task environment. The pre-access backend gate therefore rejected
  `SLURM_CPUS_PER_TASK=2` before recording backend metadata or opening
  membership; `realized_development_membership_opened=false` and every M0,
  SWC, final, and unassigned seal remains false. This is a launcher/preflight
  defect, not a solver result. The narrow repair is to set OMP/OpenBLAS/MKL
  threads explicitly to one and gate the effective OpenBLAS count at one while
  recording the scheduler allocation separately; the requested TRES and AMD
  SKU constraint remain unchanged. Do not rerun unchanged.
- `[v3_m1_diagnostic]` At `2026-09-28T03:25:45Z`, completed that narrow
  launcher repair and selected one clean new job-scoped attempt. The numerical
  libraries are now explicitly assigned one thread; the backend gate requires
  the exact OMP/OpenBLAS/MKL/NUMEXPR environment and effective OpenBLAS count
  to equal one, while rounded Slurm CPU allocation is telemetry only. A direct
  regression with `SLURM_CPUS_PER_TASK=2` and effective OpenBLAS=1 passed; a
  mismatched thread environment failed closed. Python compilation, Bash
  syntax, the 11 runner tests, six v2 tests, and four v3 tests passed.
  Scheduler test-only reported hypothetical job `45620345` on AMD 7502 node
  `sh03-06n12`. Three independent re-reviews returned no blocker, and no
  same-name or production job was active. The failed job `45619990` root will
  not be reused or overwritten; submission must receive a fresh job ID.
- `[v3_m1_diagnostic]` Submitted exactly that fresh repaired attempt as Slurm
  job `45620682` at `2026-09-28T03:26:07Z`, after the immediate exact-name
  duplicate check returned empty. The scheduler accepted one requested CPU,
  8 GB, 30 minutes, and `CPU_SKU:7502`; as expected, its core allocation view
  rounds to two while `TresPerTask=cpu=1`. Numerical threads are independently
  pinned and verified at one. The job was initially `PENDING` with reason
  `None`; babysit through terminal without duplicate submission.
- `[v3_m1_diagnostic]` Job `45620682` completed `0:0` in 3:30 on AMD 7502
  node `sh03-06n05`, with batch MaxRSS 47,728 K (46.61 MiB). Runtime telemetry
  proved OpenBLAS core `Zen`, effective OpenBLAS threads=1, all four numerical
  thread variables=1, and separately recorded Sherlock's rounded allocation.
  Both independently zero-started warm paths completed all 12 lambdas, for
  24/24 accepted fits. At every lambda, repeat objective difference,
  coefficient maximum absolute difference, and candidate-probability maximum
  absolute difference were each exactly `0.0`, passing the unchanged
  `1e-12`, `1e-10`, and `1e-10` gates. The durable decision is
  `single_thread_backend_pass`. Development membership was opened only for
  this authorized diagnostic; M0 artifacts, SWCs, final, and unassigned stayed
  sealed. This is diagnostic evidence, not production authorization: next
  freeze a prospective single-thread numerical-execution addendum, run the
  frozen 68-row warm-repeat/zero qualification at one effective thread, and
  then use a clean production job/root with no four-thread checkpoint reuse.
- `[m1_single_thread_execution_qualification]` At `2026-09-28T18:13:53Z`,
  froze and selected the prospective AMD-7502 one-thread execution gate under
  `outputs/stage0/v3_m1_numerical_execution_addendum_v3.yaml`. The model,
  objective, exact Hessian, Newton/strong-Wolfe rules, convergence and
  certificate thresholds, lambda order, initializations, and all comparison
  tolerances remain unchanged. Four-thread job `45609750` checkpoints are
  ineligible and will not be read, copied, linked, or reused. The selected
  launcher is exactly
  `outputs/stage0/tools/probe_v3_m1_synthetic_warm_zero_single_thread_v3.sbatch`:
  deterministic synthetic data only, 68 rows, 36 fits, one requested and one
  effective numerical thread, AMD `CPU_SKU:7502`, 8 GB, and 30 minutes. The
  same Python process verifies OpenBLAS core `Zen`, AMD EPYC 7502, all four
  thread variables equal one, and effective OpenBLAS threads equal one before
  calling the unchanged engine. Python compilation, Bash syntax, policy-only
  validation, seven focused execution tests, and all 22 unchanged M1 engine
  tests passed. Scheduler test-only reported hypothetical job `45732780` on
  AMD 7502 node `sh03-06n11`; the immediate exact-name and related-production
  active-job checks were empty. This selection authorizes one clean
  qualification submission only. Real K, realized membership, M0, SWCs,
  final, and unassigned data remain sealed for this gate.
- `[m1_single_thread_execution_qualification]` Submitted exactly that one
  frozen qualification as Slurm job `45734258` at `2026-09-28T18:18:59Z`,
  after the final atomic exact-name duplicate check returned empty. Slurm
  accepted the AMD-7502, one-requested-CPU, 8-GB, 30-minute launcher. This job
  has no route to real K, realized membership, M0, SWCs, final, or unassigned
  data. Babysit at scheduler-safe intervals through terminal status; do not
  submit a duplicate.
- `[m1_single_thread_execution_qualification]` Job `45734258` completed
  `0:0` in 2:30 on AMD EPYC 7502 node `sh03-01n08`, with batch MaxRSS
  117,412 K (114.66 MiB). In-job engine tests passed 22/22 and execution tests
  passed 7/7. Runtime telemetry proved OpenBLAS core `Zen`, one effective
  OpenBLAS thread, and all four numerical thread variables equal one while
  recording Sherlock's rounded two-CPU allocation separately. The frozen gate
  completed 36/36 fits as exactly 12 warm-repeat-1, 12 warm-repeat-2, and 12
  zero-start fits; all 12/12 comparisons passed. Every repeat objective,
  coefficient, and candidate-probability difference was exactly `0.0`; the
  maximum zero-start objective, probability, and log-probability differences
  were respectively `1.7763568394002505e-15`, `1.4114464144573654e-13`, and
  `2.785074570965662e-07`, within the unchanged gates. Real K, realized
  membership, M0, SWCs, final, and unassigned remained sealed. This pass
  authorizes only repair, audit, and one clean AMD-7502/effective-one-thread
  development M1 production submission under the frozen execution addendum;
  it does not authorize reuse of job `45609750` checkpoints or M0 opening.
- `[v3_m1]` At `2026-09-28T18:39:18Z`, completed the clean production repair
  and selected exactly
  `outputs/stage0/tools/fit_v3_m1_real_single_thread_v3.sbatch` for one new
  submission. The runner validates the frozen numerical-execution addendum,
  passed qualification job `45734258`, matching AMD-7502/OpenBLAS-Zen
  effective-one-thread backend, and the recorded first membership opener
  `45609750` before the only authorized development-membership read. It uses
  a fresh job-scoped `v3_m1_real_fit_single_thread_v3` root and rejects all
  four-thread or foreign-job checkpoints. Bash syntax, all 13 focused runner
  tests, and ASTRA 0.0.14 schema and semantic validation passed; two
  independent stabilized-code audits found no launch blocker. The exact-name
  active-job check was empty, and scheduler test-only reported hypothetical
  job `45738897` on `sh03-06n12`. The frozen request is one requested and one
  effective numerical thread, AMD `CPU_SKU:7502`, 16 GB, and 8 hours on
  `russpold,normal`. This selection authorizes one `sbatch --parsable`
  submission only. M0 artifacts, SWCs, final, and unassigned remain sealed.
- `[v3_m1]` Submitted exactly that clean single-thread production launcher as
  Slurm job `45739355` at `2026-09-28T18:41:18Z`, after the final atomic
  exact-name duplicate check returned empty. Slurm accepted the frozen
  AMD-7502, one-requested-CPU, 16-GB, 8-hour request. The job has a fresh
  job-scoped root and cannot reuse job `45609750` or any other foreign
  checkpoint. Its runtime must still pass the numerical backend,
  qualification, actual-K, and pre-membership access gates. Babysit job
  `45739355` at scheduler-safe intervals through terminal state. M0 artifacts,
  SWCs, final, and unassigned remain sealed.
- `[v3_m1]` By `2026-09-28T18:45:37Z`, job `45739355` was `RUNNING` on
  `sh03-01n15`. The in-job numerical gate passed on AMD EPYC 7502 with
  OpenBLAS core `Zen`, one effective OpenBLAS thread, and all four numerical
  thread variables equal one; all 13 focused runner tests passed. The
  job-scoped manifest reached `membership_loaded_outer_tuning` with zero warm
  paths and zero refits yet checkpointed. Authorized development
  ID/group/K/detected16 membership is now open for this fit. M0 artifacts,
  SWCs, final, and unassigned remain false in both the runtime gate and
  manifest. Continue scheduler-safe babysitting without duplicate submission.
- `[v3_m1]` By `2026-09-28T19:41:42Z`, job `45739355` had completed 56/66
  unique leave-two warm paths and seven outer selections, all interior. It
  successfully passed the former four-thread failure path
  `leave_two__221418__221588`; the `221418` outer fold completed with selected
  lambda `0.1`, and no failure summary exists. The preceding selected lambdas
  were `1, 1, 1, 1, 0.01, 1`. Continue through the remaining outer tuning,
  outer refits, and full-development fit; all protected seals remain closed.
- `[v3_m1]` By `2026-09-28T19:50:37Z`, job `45739355` completed all 66/66
  unique outer-tuning warm paths and all 12/12 outer selections with boundary
  count zero. The selected-lambda vector is
  `1,1,1,1,0.01,1,0.1,1,1,1,1,1`. It then completed 9/12 zero-start outer
  refits while remaining `RUNNING`; no failure summary exists. Continue the
  last three outer refits, prediction artifact, and full-development
  tuning/refit before any postflight or M0 access.
- `[v3_m1]` Job `45739355` completed `0:0` in 1:18:38 on AMD-7502 node
  `sh03-01n15`, with batch MaxRSS 120,748 K (117.92 MiB). Its manifest is
  terminal `complete` with 78 warm paths and 13 zero-start refits, matching 66
  unique leave-two outer-tuning paths, 12 leave-one full-tuning paths, 12
  outer refits, and one full-development refit. The pass summary reports 194
  cells, 12 groups, outer boundary count zero, full-development lambda `1.0`,
  and `checkpoint_resume_used=false`; runtime was one effective OpenBLAS
  thread on the qualified AMD/Zen backend. No failure summary exists and M0,
  SWC, final, and unassigned flags are false. At `2026-09-28T20:05:36Z`,
  three independent read-only postflight audits were started. Do not open M0
  until artifact, numerical, and seal postflight all pass.
- `[v3_m1]` At `2026-09-28T20:22:41Z`, independent artifact, numerical, and
  access-seal postflight all passed for job `45739355`, with no blocker. The
  artifact audit found exactly 194 unique outer-cross-fitted rows across the
  frozen 12 groups (including group `221588=7`) and exactly 91 current-job
  checkpoints: 66 leave-two warm paths, 12 leave-one warm paths, 12 outer
  zero-start refits, and one full zero-start refit. The numerical audit
  independently checked all 949 fits, the one-SE selections, calibration, and
  score identities. It classified the two lambda-1000 zoom-resolution stalls
  as valid frozen-contract certificate returns: all values finite and both
  gradient and relative-gap predicates passed during the required v2
  confirmation attempt. Maximum inclusion-marginal sum-to-K error was
  `4.440892098500626e-15`; gain and uniform-score identity errors were zero.
  The seal audit confirmed exact job/qualification/opener/backend binding and
  no old checkpoint reuse. M0, SWC, final, and unassigned remained unopened.
  This releases only the already frozen mechanical paired M1-minus-M0
  comparison; it does not authorize M0 influence on M1 or any protected-data
  opening.
- `[v3_m1_compare_m0]` At `2026-09-28T20:36:02Z`, implemented, independently
  audited, and selected the frozen mechanical M1-minus-M0 comparator. The
  exact launcher is
  `outputs/stage0/tools/compare_v3_m1_m0_v1.sbatch`, requesting 10 minutes,
  one CPU, and 2 GB on `russpold,normal`. Its runner first requires the exact
  job-`45739355` postflight gate and validates only that M1 job's terminal
  manifest, summary, and 194 prediction rows. It then writes an atomic
  `M0_opened=false` access record and changes it to `true` immediately before
  the first operation on the exact existing M0 v7 root. It reads only the M0
  summary and outer-cross-fitted predictions, pairs the exact 194 cells and 12
  group/K assignments, and computes the frozen one-twelfth mean of paired
  group-mean M1-minus-M0 log2 gains. It has no target/interface/geometry,
  full-refit, telemetry, SWC, final, or unassigned route. Python compilation,
  Bash syntax, 11/11 focused synthetic/access-order tests, and ASTRA 0.0.14
  schema and semantic validation passed; a stabilized independent re-audit
  found no blocker. Scheduler test-only reported hypothetical job `45760078`,
  and the exact-name active-job check was empty. Real M0 has not yet received
  any filesystem operation from this comparator work. This selection
  authorizes one `sbatch --parsable` submission only.
- `[v3_m1_compare_m0]` Submitted exactly that comparator as Slurm job
  `45760290` at `2026-09-28T20:37:51Z`, after the final atomic exact-name
  duplicate check returned empty. Slurm accepted the frozen one-CPU, 2-GB,
  10-minute request. The returned ID was recorded before any status poll. Real
  M0 remains unopened until the running job durably records its access marker;
  SWCs, final, and unassigned remain sealed. Babysit job `45760290` without a
  duplicate submission.
- `[v3_m1_compare_m0]` Job `45760290` completed `0:0` in 8 seconds on
  `sh04-04n24`, with batch MaxRSS 24,960 K. All 11/11 in-job comparator tests
  passed. After the required durable access marker, the job opened only the
  frozen M0 summary and outer-cross-fitted prediction table and emitted a
  terminal pass report with exactly 194 one-to-one cells across 12 groups,
  exact group/K pairing, and no degenerate K=0 or K=16 rows. The equal-group
  M0 and M1 means were respectively `-8.218544398137748` and
  `-8.215906534449852` bits per cell, giving paired M1-minus-M0 gain
  `0.002637863687897285` bits per cell. A separate local recomputation matched
  the aggregate and every group mean exactly and found zero fixed-K uniform
  identity error. Group effects are strongly heterogeneous, so this secondary
  exploratory observation-pattern score is not biological promotion and
  cannot rescue a failed primary. M0 full-refit and telemetry, SWCs, final,
  and unassigned stayed sealed. Independent paired-result and access-seal
  postflight are now in progress before `v3_search` preparation.
- `[v3_m1_compare_m0]` At `2026-09-28T20:46:47Z`, both independent
  postflights passed with no blocker. The numerical audit independently read
  only the authorized M1/M0 summaries and outer-cross-fitted predictions,
  confirmed all 194 exact provider/group pairs, all group and K assignments,
  the frozen group-size vector, and zero uniform-baseline error, and reproduced
  every per-group and aggregate value exactly. The access audit confirmed
  terminal Slurm state `COMPLETED 0:0`, durable false-marker then true-marker
  ordering before the first M0 filesystem operation, and an access surface
  limited to the M0 summary and outer-cross-fitted predictions. No partial or
  failure artifact remained. The comparator gate is passed; its small positive
  aggregate amid large cross-group heterogeneity remains descriptive secondary
  evidence only. M0 full-refit and telemetry, SWCs, final, and unassigned are
  still sealed. The next dependency-satisfied experiment is prospective
  development-only `v3_search` contract and implementation preparation.

## 2026-09-28 — v3 M2 prospective qualification and repair

- `[v3_search.m2_m1_adequacy]` Slurm job `45773980` completed `0:0` in
  9 seconds after running all 11 focused tests. Independent audit reproduced
  the exact 194-row/12-group inventory and the equal-group gate: M1 beats the
  fixed-K uniform reference by `1.5953602605961161` bits per cell, marginal
  Brier is `0.13959451726019392` versus `0.16889926452670592`, target
  calibration RMS is `0.12149700112033555` versus `0.21972497328552726`
  (ratio `0.5529503510848226`), and maximum absolute target residual is
  `0.2564803858876762` versus `0.48738076593545343` (ratio
  `0.5262423218433764`). All frozen thresholds pass. The gate read only the
  M1 summary and outer-cross-fitted predictions; real M2 outcomes, M0
  full-refit/telemetry, SWCs, final, and unassigned data remained unopened.
- `[v3_search.m2_core_qualification]` Target-free Slurm job `45772526`
  completed `0:0` in 3:10 and its then-current nine tests passed, but an
  independent post-job conformance audit rejected it as qualification for the
  frozen contract. The audit found mismatched solver limits/initialization,
  majorization slack, Dykstra and lambda-maximum procedures, plus incomplete
  frozen recovery, derivative, KKT, range-null, and deterministic-repeat
  coverage. The contract/addendum also changed after that job ended. This is a
  failed qualification attempt, not authority for actual-K or real-M2 access.
- `[v3_search.controller_conformance]` The stable 16/16 controller tests and
  built-in target-free replay self-test pass, but independent adversarial
  review found four remaining bypasses: coverage cycle/phase was not bound,
  optional rank could be selected from a separate score stream, fabricated
  cell keys could satisfy the pair selector, and the integrated RNG replay did
  not exercise fitted-M1 sampling or more than one fold ordinal. Repairs and
  focused regressions are in progress before re-audit.
- `[v3_search]` No real M2 outcome, candidate pair effect, final, unassigned,
  or SWC data has been opened. The next executable work is the smallest
  affected target-free repair and rerun; the actual-K-only rank-104 gate stays
  closed until the core and controller re-audits pass.
- `[v3_search.m2_core_qualification]` Focused repaired-core job `45777447`
  was submitted with one CPU, 8 GiB, and 30 minutes and failed `1:0` after
  4:38. Eight of nine target-free tests passed. The sole failure was a newly
  added cold-versus-warm Dykstra raw-vector assertion: two of 120 coordinates
  differed by at most `8.09835705e-12` against an over-strict test-only
  `2e-12` bound, while both prox paths independently converged and certified
  under the frozen `1e-12` iterate tolerance and its `20x` range/invariant
  certification envelope. This is a regression-test defect, not authority to
  change the algorithm or tolerances; no protected path was available.
- `[v3_search.m2_core_qualification]` After diagnosis, the algorithm was left
  unchanged and the regression was narrowed to require both certificates and
  raw-vector agreement within the existing `2e-11` certification envelope.
  Python compilation and diff whitespace checks passed. The selected smallest
  affected rerun is one synthetic-only nine-test job named
  `ep10-m2-core-repair-r2`, requesting one CPU, 8 GiB, and 30 minutes, with
  append log `m2_core_repair_r2_%j.log`. No real K, membership, M0 artifact,
  SWC, final, or unassigned input route exists. The next side effect is exactly
  one submission after an immediate exact-name duplicate check.
- `[v3_search.m2_core_qualification]` The exact-name duplicate check was empty
  and the selected focused rerun was submitted exactly once as Slurm job
  `45778895`. Its append log is
  `$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/m2_core_repair_r2_45778895.log`.
  Babysit this job to a terminal scheduler state; completion still does not
  replace the separate comprehensive frozen qualification and independent
  re-audit.
- `[v3_search.m2_core_qualification]` Focused rerun `45778895` completed
  `0:0` in 2:39; all 9/9 repaired-core tests passed in 150.206 seconds and
  the batch step used 146,512 K MaxRSS. Numerical libraries were pinned to one
  thread; Sherlock reported its normal two-core allocation rounding separately.
  This validates the focused algorithm repair only. Comprehensive frozen
  qualification coverage and an independent stabilized-code audit remain
  required before actual-K access.
- `[v3_search.controller_conformance]` At `2026-09-29T00:58:54Z`, the
  stabilized target-free controller passed 34/34 tests and its built-in replay
  self-test. Independent adversarial re-audit then rejected every tested
  missing or altered trial-16/post-core evidence binding, same-branch
  diagnostic, contrast, wrong-core, signed-zero, and omitted-evidence attack,
  while preserving the frozen B1/B6/B8 behavior. This closes the prior four
  controller bypasses without opening actual K, real M2 outcomes, M0 artifacts,
  SWCs, final, or unassigned data.
- `[v3_search.m2_core_qualification]` At `2026-09-29T00:58:54Z`, root smoke
  checks and final independent read-only audit found no blocker in the repaired
  frozen core implementation and comprehensive qualifier. The exact selected
  attempt is `outputs/stage0/tools/qualify_v3_m2_core.sbatch`, requesting one
  CPU, 8 GiB, and two hours, and running the core suite once, the full
  qualification once, and the saved-report validator once. ASTRA v0.0.14 now
  explicitly orders controller conformance, this comprehensive target-free
  qualification, the still-closed actual-K rank-104 gate, and only then the
  adaptive real-M2 search; pinned schema and semantic validation passed. The
  next side effects are an exact-name duplicate check, one scheduler test-only,
  a second exact-name duplicate check, and at most one real submission. Actual
  K, realized M2 memberships, M0 full-refit/telemetry, SWCs, final, and
  unassigned data remain sealed.
- `[v3_search.m2_core_qualification]` At `2026-09-29T00:59:44Z`, the first
  exact-name active-job check for `ep10-v3-m2-qual` was empty and scheduler
  test-only accepted the frozen request as hypothetical job `45807730` on one
  processor. No workload ran. The selected launcher is unchanged; perform one
  final exact-name duplicate check immediately before its sole real submission.
- `[v3_search.m2_core_qualification]` Submitted exactly the selected
  comprehensive target-free launcher once as Slurm job `45808280` at
  `2026-09-29T01:00:14Z`, after the final exact-name active-job check returned
  empty. The returned job ID was recorded before any status poll. Babysit this
  job without duplicate submission through terminal scheduler state, then
  independently validate its saved report. Actual K, realized M2 memberships,
  M0 full-refit/telemetry, SWCs, final, and unassigned data remain sealed.
- `[v3_search.m2_core_qualification]` Job `45808280` failed `2:0` after
  3:24 on `sh04-04n23` with batch MaxRSS 146,748 K. Its first phase passed all
  10/10 core tests in 151.379 seconds. The subsequent comprehensive qualifier
  stopped before emitting a report because the zero-KKT dual-feasibility solver
  hit its iteration cap with residual `7.249804822985446e-05`, stationarity
  `5.2189417185848871e-09`, relative change `3.2829091434453252e-13`, and
  threshold `1e-12`. This is a failed qualification attempt, not authority for
  actual-K or real-M2 access. Diagnose the exact case and root cause before any
  rerun; all protected seals remain intact.
- `[v3_search.m2_core_qualification]` At `2026-09-29T02:11:48Z`, the exact
  failure from job `45808280` was repaired without changing the model,
  `1e-12` proximal tolerance, 10,000-iteration cap, `1e-7` fit KKT tolerance,
  or `1e-12` relative lambda bracket. The solver now moves a bracket endpoint
  only from an explicit direct feasible witness or a range-space support
  separator; the latter carries a homogeneous decision term, a dimension-aware
  binary64 lower bound including `B @ a` span-construction error, and inward
  group-ball rounding. A certified feasible upper correction is reused only as
  the deterministic initialization for nested midpoints, and the final allowed
  quasi-Newton update is checked without exceeding the cap. Focused exact
  midpoint, five-penalty two-sided, and boundary fail-closed smoke tests passed;
  the core suite now also runs the full frozen failing group-lasso fold twice
  and checks exact correction propagation and determinism. Independent static
  audit of the frozen snapshot returned PASS with no must-fix and confirmed the
  launcher remains synthetic-only and job-ID scoped. The selected smallest
  affected rerun is the same comprehensive launcher,
  `outputs/stage0/tools/qualify_v3_m2_core.sbatch`: its fail-fast first phase
  executes all regressions before the full qualifier and report validator.
  The next side effect is one exact-name active-job check immediately followed
  by at most one submission. Actual K, realized M2 memberships/effects, M0
  full-refit/telemetry, SWCs, final, and unassigned data remain sealed.
- `[v3_search.m2_core_qualification]` The immediate exact-name active-job check
  was empty and the repaired comprehensive target-free launcher was submitted
  exactly once as Slurm job `45816758` at `2026-09-29T02:12:28Z`. Its durable
  output directory and runtime log are job-ID scoped under
  `$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/`. Babysit this
  job without duplicate submission through a terminal scheduler state, then
  independently validate the saved report before opening the actual-K gate.
  Actual K, realized M2 memberships/effects, M0 full-refit/telemetry, SWCs,
  final, and unassigned data remain sealed.
- `[v3_search.m2_core_qualification]` Job `45816758` failed `1:0` after
  7:12 on `sh04-04n21` in its fail-fast 13-test core phase; the full qualifier
  and report validator did not run. Four tests passed before nine lambda-max
  cases failed at the frozen 10,000-iteration cap. At the failures, null-space
  stationarity was already `6.1e-18` to `6.4e-14`, but the newly added generic
  BLAS `B @ a` error envelope made every support-separation lower bound
  negative (for example `-2.6654480104387851e-25` for lasso and
  `-3.2173283235925854e-14` for the exact frozen group fold). The direct
  boundary and oblique certificate tests still passed. This localizes the
  defect to an overly pessimistic span-construction arithmetic bound, not the
  real-arithmetic separator, warm propagation, penalty model, or frozen
  tolerances. Do not rerun yet: replace only that BLAS envelope with a tighter
  per-coordinate `math.fsum` product-error bound, retain fail-closed boundary
  coverage, and rerun the smallest affected tests. All protected seals remain
  intact.
- `[v3_search.m2_core_qualification]` At `2026-09-29T02:38:39Z`, the
  overconservative span envelope from job `45816758` was replaced by a
  per-coordinate `math.fsum` reconstruction with explicit multiplication,
  summation, and subnormal-product error bounds. A normalized smooth-minimax
  fallback was added inside the same frozen 10,000-iteration budget for
  tangential affine/dual-ball intersections: feasible results still require a
  direct dual witness, while infeasible results use a softmax KKT direction
  only after it passes the same guarded range-space support separator. The
  previously ambiguous exact group midpoint now certifies infeasible in 609
  total iterations with positive margin `3.318511123062902e-08`; synthetic
  lasso full-fold smoke and ordinary plus discriminating subnormal boundary
  checks pass. No tolerance, bracket rule, penalty, or model changed. The
  selected smallest affected scheduler check is
  `outputs/stage0/tools/qualify_v3_m2_zero_kkt_repair.sbatch`, one CPU, 4 GiB,
  30 minutes, running only the five zero-KKT/fold/recovery regressions before
  any comprehensive rerun. Await the final frozen-snapshot static audit, then
  perform test-only, an immediate exact-name duplicate check, and at most one
  submission. All protected seals remain intact.
- `[v3_search.m2_core_qualification]` At `2026-09-29T02:48:33Z`, two
  independent read-only static reviews returned PASS with no must-fix on the
  frozen focused r3 repair. They confirmed conservative arithmetic and
  subnormal guards, direct-witness or guarded-separator-only acceptance,
  shared 10,000-iteration accounting, explicit minimax-route and cap-boundary
  regressions, qualifier telemetry, and a synthetic-only job-ID-scoped
  launcher. The selected next execution remains exactly
  `outputs/stage0/tools/qualify_v3_m2_zero_kkt_repair.sbatch`; scheduler-test
  it, perform one immediate exact-name active-job check, and submit at most
  once. Actual K, real M2 outcomes, M0 full-refit/telemetry, SWCs, final, and
  unassigned data remain sealed.
- `[v3_search.m2_core_qualification]` At `2026-09-29T02:49:17Z`, scheduler
  test-only accepted the selected one-CPU focused request, the immediately
  following exact-name active-job check for `ep10-m2-kkt-r3` was empty, and
  the launcher was submitted exactly once as Slurm job `45821735`. Its log is
  job-ID scoped at
  `$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/m2_zero_kkt_repair_r3_45821735.log`.
  Babysit this job to a terminal scheduler state without duplicate submission;
  only a passing focused result can authorize the comprehensive target-free
  rerun. Actual K, real M2 outcomes, M0 full-refit/telemetry, SWCs, final, and
  unassigned data remain sealed.
- `[v3_search.m2_core_qualification]` Focused synthetic job `45821735` failed
  `1:0` after 5:50; 2/5 tests passed and three near-boundary zero-KKT calls
  failed closed. The all-penalty fold-local route test passed. The remaining
  failures had already reached null-space stationarity between `4.3e-17` and
  `4.6e-18`, but direct-witness residuals (`1.82e-14`, `3.29e-14`, and
  `1.68e-12`) remained above their binary64 decision scales (`6.20e-15`,
  `2.75e-15`, and `2.62e-15`), while guarded separation margins were
  nonpositive. This is evidence for a smaller final near-boundary certificate
  refinement, not authority to relax a tolerance or rerun unchanged. Diagnose
  and repair that exact synthetic failure before another focused submission;
  all protected seals remain intact.
- `[v3_search.m2_core_qualification]` At `2026-09-29T03:06:31Z`, the three
  job-`45821735` failures were traced to the reject-only rough-gap shortcut
  examining the raw smooth-minimax direction before its exact range-space
  reconstruction. The repair disables only that shortcut for the minimax
  candidate; the existing range reconstruction, conservative arithmetic lower
  bound, and strict positive-margin acceptance remain mandatory. A direct
  regression proves a raw-negative range-plus-null direction is accepted only
  when its rigorously guarded range projection separates; its focused smoke
  passed 1/1 in 6.818 seconds. Two independent static reviews returned PASS,
  and the selected synthetic-only launcher now runs six focused tests. Next:
  scheduler-test it, perform one immediate exact-name active-job check, and
  submit at most one rerun. No tolerance, phase budget, iteration cap, penalty,
  model, or protected-data boundary changed.
- `[v3_search.m2_core_qualification]` At `2026-09-29T03:07:07Z`, scheduler
  test-only accepted the repaired six-test request, the immediate exact-name
  active-job check was empty, and the focused launcher was submitted exactly
  once as Slurm job `45823728`. Its job-scoped log is
  `$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/m2_zero_kkt_repair_r3_45823728.log`.
  Babysit it to terminal state without duplicate submission. Only a passing
  focused result and independent log inspection can authorize selection of the
  comprehensive target-free rerun; all protected seals remain intact.
- `[v3_search.m2_core_qualification]` Focused synthetic rerun `45823728`
  failed `1:0` after 11:59. The new discriminating projection regression and
  the all-penalty fold-local route test passed, but the same three tangential
  full-fold/recovery calls failed with numerically identical residuals,
  stationarity, minimax iterations, constraints, and final margins as job
  `45821735`. Therefore the raw minimax prefilter was a real fail-closed
  completeness defect but not the cause of these three cases. Do not rerun the
  same solver. Capture the minimax KKT-gradient, projected candidate gap, and
  direct-witness ball/affine residuals for these exact synthetic calls in one
  bounded diagnostic, then repair the evidenced certificate path. Frozen
  thresholds, cap, penalties, model, and protected seals remain unchanged.
- `[v3_search.m2_core_qualification]` At `2026-09-29T03:26:23Z`, two
  independent static reviews cleared one bounded synthetic certificate
  diagnostic. It adds observational failure telemetry only: minimax gradient
  and null leakage, weight concentration, projected nominal and fully guarded
  separator gaps with guard-loss components, and affine equality plus nominal
  and outward-guarded ball violations. The selected launcher is
  `outputs/stage0/tools/diagnose_v3_m2_zero_kkt_certificates.sbatch`, one CPU,
  4 GiB, 30 minutes, running exactly the two known reproducers and requiring
  all three expected diagnostic records. Scheduler-test it, perform one
  immediate exact-name active-job check, and submit at most once. This is a
  diagnostic, not a qualification, and all protected seals remain intact.
- `[v3_search.m2_core_qualification]` At `2026-09-29T03:27:02Z`, scheduler
  test-only accepted the selected diagnostic, the immediate exact-name
  active-job check was empty, and it was submitted exactly once as Slurm job
  `45825768`. Its job-scoped log is
  `$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/m2_zero_kkt_certificate_diagnostic_45825768.log`.
  Babysit it through terminal state without duplication, then interpret the
  three expected synthetic records before editing or launching another
  qualification. All protected seals remain intact.
- `[v3_search.m2_core_qualification]` Synthetic diagnostic job `45825768`
  completed `0:0` in 4:03 with all three expected records. The smooth-minimax
  continuation had not solved its final active face: the two group cases
  collapsed to one constraint with gradient infinity norms `0.351` and
  `0.303`, and the elastic case retained four equal weights but gradient
  `0.00153`. Their separator null leakages were `0.0592`, `0.0788`, and
  `0.000485`, producing genuinely negative projected gaps. Conservative
  arithmetic guard loss was only about `1.3e-14`, so it was not causal.
  Conversely, each affine candidate satisfied the range equation to about
  `1e-16` and exceeded its dual ball by only `6.50e-14`, `1.81e-14`, or
  `1.26e-12`. This distinguishes an under-solved active face from a tolerance,
  cap, or rounding-guard problem. Implement and audit a bounded active-face
  normal polish whose candidate can certify only through the unchanged guarded
  separator; do not relax direct-witness or acceptance thresholds. Protected
  seals remain intact.
- `[v3_search.m2_core_qualification]` At `2026-09-29T03:54:10Z`, the
  diagnostic-localized defect was repaired with a deterministic active-face
  minimum-norm multiplier polish at the final smooth-minimax coordinates.
  Face growth, KKT solves, add/drop events, and candidate evaluations all share
  the existing 1,024-iteration minimax budget and the frozen 10,000 total cap;
  no model, penalty, tolerance, or decision margin changed. A polished normal
  can certify only through the unchanged fully guarded range-space separator.
  Two independent read-only static reviews returned PASS on the exact snapshot,
  including the KKT signs, deterministic support handling, fail-closed paths,
  iteration accounting, and non-vacuous full-fold regressions for certified,
  positive-margin, nonsingleton, materially nonuniform active faces. Lightweight
  Python AST and launcher shell checks also passed. The exact selected smallest
  affected execution is
  `outputs/stage0/tools/qualify_v3_m2_zero_kkt_repair.sbatch`, one CPU, 4 GiB,
  30 minutes, running six synthetic-only focused regressions. Next: scheduler
  test-only, one immediate exact-name active-job check for `ep10-m2-kkt-r3`,
  and at most one submission. Actual K, real M2 outcomes, M0 full-refit and
  telemetry, SWCs, final, and unassigned data remain sealed.
- `[v3_search.m2_core_qualification]` At `2026-09-29T03:54:56Z`, scheduler
  test-only accepted the selected one-CPU focused request, the immediately
  following exact-name active-job check for `ep10-m2-kkt-r3` was empty, and
  the launcher was submitted exactly once as Slurm job `45828606`. Its
  job-scoped log is
  `$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/m2_zero_kkt_repair_r3_45828606.log`.
  Babysit this job to terminal scheduler state without duplicate submission,
  then inspect the complete log. Only a passing focused result can authorize
  selection of the comprehensive target-free rerun. Actual K, real M2
  outcomes, M0 full-refit and telemetry, SWCs, final, and unassigned data
  remain sealed.
- `[v3_search.m2_core_qualification]` Focused synthetic job `45828606` failed
  `1:0` after 17:37 on `sh02-06n03`; its batch step used 149,232 K MaxRSS.
  Four of the six named tests passed, while the tangential full-fold test and
  the elastic-net and group-lasso recovery subcases produced three fail-closed
  zero-KKT errors. The active-face repair did solve the diagnosed normal
  equations: its best null-space leakage was only `6.38e-15`, `1.58e-16`, and
  `3.34e-16`. However, the unchanged fully guarded separator margins remained
  slightly nonpositive at `-1.33e-14`, `-6.18e-15`, and `-4.29e-15`, so none
  was accepted. Each call then exhausted the frozen 10,000 total iteration cap.
  This is evidence of a final boundary-certification precision gap, not a
  scheduler failure or authority to relax a tolerance, change the model, or
  rerun unchanged. Independently compare primal-coordinate refinement,
  higher-precision certificate arithmetic, and frozen-bracket handling, then
  repair only the evidenced path and rerun the smallest affected synthetic
  tests. All protected seals remain intact.
- `[v3_search.m2_core_qualification]` At `2026-09-29T04:25:03Z`, the three
  job-`45828606` failures were identified as late dyadic bisection queries at
  the finite-precision zero-KKT decision boundary, not failed active-normal
  convergence. The terminal no-proof outcome now has a dedicated
  indeterminate exception. Only `fold_lambda_maximum` catches it: the unknown
  point never moves either endpoint, and the fold instead probes the fixed
  one-quarter and three-quarter points, retaining only certified infeasible
  lower and direct-feasible upper endpoints. Every midpoint and quarter probe
  is charged to the existing 256-call safety cap; monotonic contradiction,
  no shrink, or an unresolved band fails closed, and the final upper endpoint
  is recertified. No certificate guard, tolerance, penalty, model, or solver
  cap changed. A mocked tri-state regression passed in 6.853 seconds and proves
  that unknown midpoint and lower-quarter queries cannot enter the bracket or
  replace the feasible warm state. Two independent static audits returned PASS.
  The exact selected smallest affected rerun is the same job-ID-scoped focused
  launcher, now with seven synthetic tests. Next: scheduler test-only, one
  immediate exact-name check for `ep10-m2-kkt-r3`, and at most one submission.
  Actual K, real M2 outcomes, M0 full-refit and telemetry, SWCs, final, and
  unassigned data remain sealed.
- `[v3_search.m2_core_qualification]` At `2026-09-29T04:25:41Z`, scheduler
  test-only accepted the selected seven-test focused request, the immediately
  following exact-name active-job check for `ep10-m2-kkt-r3` was empty, and
  the launcher was submitted exactly once as Slurm job `45831090`. Its
  job-scoped log is
  `$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/m2_zero_kkt_repair_r3_45831090.log`.
  Babysit this job to terminal state without duplicate submission, then inspect
  its complete log. Only a pass can authorize selection of the comprehensive
  target-free rerun. All protected seals remain intact.
- `[v3_search.m2_core_qualification]` Focused synthetic job `45831090`
  completed `0:0` in 12:52 on `sh04-04n24`; its batch step used 147,040 K
  MaxRSS. The complete 11-line job-scoped log reports 7/7 tests passing in
  761.493 seconds and the explicit launcher PASS marker. This includes the
  mocked indeterminate-oracle regression, arithmetic boundary and projected
  separator checks, oblique certificates, the exact tangential full fold,
  all-penalty fold-local lambda maxima, and deterministic planted recovery.
  Await one independent complete-log and next-launcher postflight before
  selecting the comprehensive target-free rerun. All protected seals remain
  intact.
- `[v3_search.m2_core_qualification]` At `2026-09-29T04:45:36Z`, two
  independent postflights returned PASS on focused job `45831090` and the exact
  next launcher. The complete focused log has no traceback or error and was
  written after the repaired engine and test files. The selected comprehensive
  target-free execution is exactly
  `outputs/stage0/tools/qualify_v3_m2_core.sbatch`, requesting one CPU, 8 GiB,
  and two hours. It uses unique job-ID-scoped Scratch paths, runs all 15 current
  core tests once, the full eight-gate synthetic qualifier once, and the saved
  report validator once under fail-fast shell semantics. Python AST and launcher
  shell syntax checks pass. No actual K, real M2 outcome, M0 full-refit or
  telemetry, SWC, final, or unassigned path is referenced. Next: scheduler
  test-only, one immediate exact-name active-job check for `ep10-v3-m2-qual`,
  and at most one submission.
- `[v3_search.m2_core_qualification]` At `2026-09-29T04:46:05Z`, scheduler
  test-only accepted the selected comprehensive request, the immediately
  following exact-name active-job check for `ep10-v3-m2-qual` was empty, and
  the launcher was submitted exactly once as Slurm job `45832824`. Its unique
  result directory and runtime log are
  `$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/v3_m2_core_contract_qualification_job_45832824/`
  and the same path with `.log`. Record this ID before polling, babysit it to a
  terminal state without duplication, then independently validate the saved
  report. All protected seals remain intact.
- `[v3_search.m2_core_qualification]` At `2026-09-29T06:26:33Z`, job
  `45832824` remained healthy but slow on `sh02-06n03`: its 15/15 core tests
  had passed and the synthetic qualifier was still consuming one full CPU with
  about 94 MB resident memory. The `russpold` partition permits longer jobs,
  but one attempt to extend this already-running job from two to four hours was
  denied by Slurm access control; its request and state were unchanged. Do not
  retry or bypass that control. Continue babysitting the existing job. If it
  reaches its time limit, treat that as an evidenced walltime-sizing failure,
  increase only the next launcher's explicit walltime, and rerun once through
  the normal duplicate-check workflow. All protected seals remain intact.
- `[v3_search.m2_core_qualification]` Comprehensive target-free job `45832824`
  completed `0:0` in 1:57:03 on `sh02-06n03`, just within its original two-hour
  request; its batch step used 93,520 K MaxRSS. The complete 601-line log
  reports all 15 core tests passing in 2,173.122 seconds, the eight-gate report
  with both `status` and `overall_gate_status` equal to `pass`, the saved-report
  validator passing with required-gate count 8, core-penalty count 3, and
  lambda-penalty count 5, and the explicit comprehensive PASS marker. The saved
  report is
  `$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/v3_m2_core_contract_qualification_job_45832824/v3_m2_core_contract_qualification.json`.
  Its emitted scope flag says real target membership or M2 effects were not
  opened. Complete an independent read-only report postflight before opening
  or launching the separately defined actual-K gate. Final, unassigned, SWCs,
  real M2 outcomes, and M0 full-refit or telemetry remain sealed.
- `[v3_search.m2_core_qualification]` At `2026-09-29T06:49:48Z`, an
  independent postflight read the complete 601-line job log and saved JSON and
  returned PASS. It confirmed exactly 15/15 named tests, all eight required
  gates in the frozen order, the 65,536-row candidate inventory, 120 physical
  pairs, three core and five lambda penalty cases, the saved-report validator,
  and every protected-access flag false. This satisfies the ASTRA
  `m2_core_qualification_report` dependency. The next selected experiment is
  the separately bounded `v3_search.m2_actual_k_gate`; implement and audit its
  sealed four-field rank-104 route before any submission. Realized positive-set
  identity, real M2 scores/effects, M0 full-refit or telemetry, SWCs, final,
  and unassigned data remain sealed.
- `[v3_search.m2_actual_k_gate]` At `2026-09-29T07:00:31Z`, contract review
  selected exactly the 12 frozen leave-one-conservative-group-out training
  folds and no extra full-development pre-access gate. The implementation uses
  the existing byte-level physical-ID/K projection, exact 194-cell inventory
  join, poisoned realized-membership index, and the frozen M2 engine to derive
  each projector and require computed rank 104 at tau/10, tau, and 10 tau. It
  adds no condition-number gate and emits aggregate diagnostics only. Ten
  inherited projection-sealing tests and eight new M2 gate tests pass; a
  separate pre-access smoke passed the contract, job-45832824 core report,
  inventory, target-blind interface/geometry join, and exact 12-fold inventory
  without opening actual K. The selected one-thread launcher requests one CPU,
  8 GiB, and two hours. Independent code, sealing, and operational audits are
  in progress before ASTRA recipe update, scheduler test-only, duplicate check,
  and at most one submission. Actual K, realized memberships, M2 scores/effects,
  M0 full-refit or telemetry, SWCs, final, and unassigned data remain sealed.
- `[v3_search.m2_actual_k_gate]` At `2026-09-29T07:05:01Z`, independent code,
  sealing, and operational audits all returned PASS on the exact gate, tests,
  and launcher. They confirmed 12 LOGO folds, outcome-free residualization,
  numerically derived stable rank 104, fail-closed behavior, aggregate-only
  output, no invented condition-number threshold, no broad readiness read, no
  protected input route, exact job-scoped Scratch output, and one-thread
  Slurm resources. All 20 named launcher dependencies are present as regular
  non-symlink files; Python 3.12.1, NumPy 1.26.4, and PyYAML 6.0.3 load. The
  ASTRA `m2_actual_k_gate_report` recipe now names the exact one-CPU, 8-GiB,
  two-hour launcher and parses with the intended values. This selects one
  actual-K-only execution. Next: one scheduler test-only, one immediate
  exact-name active-job check, and at most one submission with a unique Scratch
  log. Actual K remains unopened at this instant; realized memberships, M2
  scores/effects, M0 full-refit or telemetry, SWCs, final, and unassigned data
  remain sealed.
- `[v3_search.m2_actual_k_gate]` At `2026-09-29T07:05:54Z`, scheduler
  test-only accepted the exact one-CPU launcher, the immediately following
  exact-name active-job query was empty, and the gate was submitted exactly
  once as Slurm job `45847880`. Its job-scoped log and result directory are
  `$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/v3_m2_actual_k_gate_45847880.log`
  and
  `$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/v3_m2_actual_k_gate_v1_job_45847880/`.
  Babysit this job to terminal state without duplicate submission, then inspect
  the complete aggregate-only report. Realized memberships, M2 scores/effects,
  M0 full-refit or telemetry, SWCs, final, and unassigned data remain sealed.
- `[v3_search.m2_actual_k_gate]` At `2026-09-29T07:07:58Z`, job `45847880`
  was running on `sh04-04n21`. Its complete log so far shows all 10 inherited
  projection-sealing tests and all eight new M2 actual-K gate tests passing;
  the process then entered the 12-fold residual-rank computation. Record actual
  K as opened only within this authorized four-field gate. No realized
  membership, candidate score or M2 fit, M0 full-refit or telemetry, SWC,
  final, or unassigned route has opened. Continue at scheduler/file polling
  intervals of at least 60 seconds without duplicate submission.
- `[v3_search.m2_actual_k_gate]` Job `45847880` completed `0:0` in 44 seconds
  on `sh04-04n21`. The complete 46-line log reports all 18 launcher tests and
  the gate passing. Its aggregate report records 194 cells, 12 groups, 12/12
  LOGO training folds, computed rank 104 at tau/10, tau, and 10 tau in every
  fold, no thirteenth gate, and no emitted fold rows, IDs, groups, segment
  flags, or K values. Actual K opened only inside the gate; realized positive-
  set membership, candidate scoring or M2 fitting, M0 full-refit or telemetry,
  SWCs, final, and unassigned flags are all false. Obtain resource telemetry
  and one independent complete-log/report postflight before selecting the next
  outcome-accessing execution.
- `[v3_search.m2_actual_k_gate]` Independent postflight returned PASS on the
  exact 46-line log and 44-line report for job `45847880`; it independently
  confirmed 12/12 stable rank-104 folds, computed rather than hard-coded rank,
  the qualified core job `45832824`, aggregate-only output, and every protected
  flag false except the authorized actual-K-only access. Final accounting is
  `COMPLETED 0:0`, 44 seconds, batch MaxRSS 140,588 K; Sherlock allocated two
  cores while all numerical thread variables remained pinned to one. The ASTRA
  `m2_actual_k_gate_report` dependency is now satisfied. Select the exact next
  adaptive-search execution without opening realized memberships until its
  remaining pre-outcome dependencies and launcher are cleared.
- `[v3_search.adaptive_model_search]` At `2026-09-29T11:52:18Z`, the named
  optional-family adapter, repaired physical Rademacher primitive, and fresh
  permutation-M1 primitive reached stable target-free implementations. Before
  releasing held scale job `45870818`, run one aggregate prelaunch regression
  on a compute node covering the scale scaffold, controller, executor,
  observed session, optional solver/adapter, replay/diagnostic, Rademacher,
  permutation mapping/M1, and the three affected physical-core regressions.
  This opens no real M2 outcome, M0 full-refit or telemetry, SWC, final, or
  unassigned data. Record the returned job ID, keep `45870818` held until this
  regression and the independent adapter audits pass, and do not duplicate it.
- `[v3_search.adaptive_model_search]` At `2026-09-29T11:54:04Z`, the aggregate
  target-free prelaunch regression was submitted exactly once as Slurm job
  `45876007`, corrected in place before allocation to the discovered eligible
  `russpold` partition, with one CPU, 4 GiB, and 30 minutes. Its log is
  `$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/v3_m2_prelaunch_regression_45876007.log`.
  Job `45870818` remains held. Babysit `45876007` to a terminal PASS and finish
  the two independent adapter audits before releasing the scale benchmark.
- `[v3_search.adaptive_model_search]` Prelaunch regression job `45876007`
  failed `1:0` after 10 seconds, before any Python test, because its transient
  launcher used `/oak/.../envs/.../python` while the verified group environment
  is under `$GROUP_HOME/users/zijiao/envs/...`. Batch MaxRSS was 17,860 K. The
  launcher now resolves the existing executable through `$GROUP_HOME`; do not
  rerun yet, because the concurrent independent Rademacher audit also found a
  real cached-statistics/candidate-basis mutation gap that must be repaired
  first. Scale job `45870818` remains held and no protected data opened.
- `[v3_search.adaptive_model_search]` At `2026-09-29T12:16:33Z`, optional
  qualification job `45862377` remained CPU-active with stable memory at
  3:30:43 of its four-hour limit. One attempt to extend the live job to six
  hours was denied by Slurm access control; its limit remains four hours. Do
  not retry or bypass that denial. Continue babysitting the same job; if it
  times out, treat that as an evidenced walltime-sizing failure before one
  corrected rerun. All protected seals remain intact.
- `[v3_search.adaptive_model_search]` At `2026-09-29T12:19:27Z`, the repaired
  Rademacher path passed independent adversarial re-audit (15/15 focused plus
  cache mutation/relabel, signed-zero, validation-only-K, no-recompute, and
  exact-projector probes). The optional-authority author matrix is also green
  after its three audit repairs, with independent re-audit still running. The
  corrected aggregate prelaunch regression is now the next bounded compute;
  submit it once with the verified `$GROUP_HOME` Python path while keeping
  scale job `45870818` held through both the regression and optional re-audit.
- `[v3_search.adaptive_model_search]` At `2026-09-29T12:20:05Z`, the corrected
  aggregate prelaunch regression was submitted exactly once as Slurm job
  `45877235` on the discovered `russpold,normal` batch partitions. Its log is
  `$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/v3_m2_prelaunch_regression_45877235.log`.
  Keep scale job `45870818` held; release requires this job to complete PASS
  and the optional-authority independent re-audit to clear.
- `[v3_search.adaptive_model_search]` At `2026-09-29T12:26:50Z`, prelaunch
  regression job `45877235` completed `0:0` in 2:43 with 414,188 K batch
  MaxRSS and the explicit `EP10_V3_M2_PRELAUNCH_REGRESSION_PASS` marker. The
  scale scaffold, units, controller, executor, observed session, optional
  solver/adapter/diagnostic, replay, Rademacher, permutation mapping/M1, and
  three affected physical-core regressions all passed. Independent audits also
  cleared the repaired Rademacher and optional authorities. The user hold on
  scale job `45870818` was therefore released; it remains pending behind exact
  dependency `afterok:45862377` and cannot start unless optional qualification
  succeeds. Continue babysitting both jobs without duplication.
- `[v3_search.adaptive_model_search]` Optional qualification job `45862377`
  reached its unchanged four-hour limit and ended `TIMEOUT` at 4:00:19; the
  batch step was cancelled `0:15` with 152,500 K MaxRSS. It remained CPU-active
  to the limit, all 25 launcher tests had passed before the long population
  recovery, no report was emitted, and the complete scheduler log contains only
  the time-limit cancellation. Dependent scale job `45870818` was consequently
  cancelled with zero elapsed and no allocation. This is an evidenced
  walltime-sizing failure, not a numerical failure. Increase only the optional
  launcher's explicit limit from four to six hours, retain all code, seeds,
  thresholds, resources, and output guards, then scheduler-test, duplicate-check,
  and submit exactly one corrected attempt. Submit one fresh scale dependency
  against that new optional job; do not revive or duplicate the cancelled pair.
- `[v3_search.adaptive_model_search]` At `2026-09-29T07:17:33Z`, a read-only
  execution audit found no observed adaptive-M2 runner or launcher and no
  contract-satisfying search-scale benchmark. The existing M2 engine has only
  core sparse penalties and a synthetic self-test; the controller is pure
  grammar with fabricated test evidence; the old fixed-K normalizer benchmark
  explicitly excludes complete null-search replay. The already-passed M1
  adequacy report is job `45773980` and will be bound, not rerun. Before any
  realized membership opens, implement a membership-blind benchmark on the
  allowed real-K/group/covariate/geometry shape with deterministic synthetic
  sets, using the exact shared executor and the still-missing low-rank and
  sparse-plus-low-rank solvers. It must project both 52-trial and 80-cap
  observed branches plus 99 complete replays, fresh synthetic M0/M1 refits,
  memory, Scratch, parallelism, and wall time below every frozen ceiling. The
  benchmark runs first; only its independent PASS plus a separately audited
  observed launcher can authorize real M2 outcomes. M0 artifacts/telemetry,
  SWCs, final, and unassigned remain sealed.
- `[v3_search.adaptive_model_search]` At `2026-09-29T12:50:08Z`, corrected
  optional-family qualification job `45879373` was submitted exactly once with
  the evidenced walltime-only repair: six hours, one CPU, 8 GiB, unchanged
  code, seeds, thresholds, numerical-thread pins, and output guards. Fresh
  scale benchmark job `45879484` was then submitted exactly once with
  `afterok:45879373` and an explicit binding to that job's optional report;
  it cannot start before the corrected qualification succeeds. The cancelled
  jobs `45862377` and `45870818` were not revived. No protected data opened.
- `[v3_search.adaptive_model_search]` At `2026-09-29T12:56:26Z`, the new v4
  target-free falsifier timing supplement passed independent review with no
  critical, high, or medium finding and 8/8 lightweight tests. Slurm job
  `45879960` was then submitted exactly once with one CPU, one numerical
  thread, 8 GiB, and 48 hours. It measures only the four frozen incremental
  units on a deterministic synthetic 194-cell/12-group fixture and writes an
  aggregate-only Scratch report; real M2 outcomes, M0 artifacts/telemetry,
  SWCs, final, and unassigned data remain sealed.
- `[v3_search.adaptive_model_search]` Timing job `45879960` failed `1:0`
  after six seconds in the shell preflight, before creating its run directory
  or entering Python. All other guards were independently true; scheduler
  accounting showed the known Sherlock rounding of the one requested CPU to
  two allocated CPU threads, while the launcher incorrectly required
  `SLURM_CPUS_PER_TASK == 1`. The identical doomed assertion was removed from
  the still-dependent scale launcher, and pending scale job `45879484` was
  cancelled with zero elapsed and zero allocation. Numerical thread variables
  remain pinned to one in both launchers. After syntax, scheduler, and exact
  duplicate checks, corrected timing job `45880194` and fresh scale job
  `45880215` with `afterok:45879373` were each submitted exactly once at
  `2026-09-29T13:00:45Z`. No numerical failure or protected access occurred.
- `[v3_search.adaptive_model_search]` Corrected timing job `45880194` passed
  all 8 launcher tests but then failed closed after 14 seconds because the
  Python entry point independently repeated the same exact-one
  `SLURM_CPUS_PER_TASK` assertion. Its safe aggregate report records only
  `FalsifierTimingError`, `status: blocked`, and both real/protected access
  flags false; it never built the synthetic fixture or entered a numerical
  timing unit. The runtime now accepts Sherlock's positive allocated CPU count
  while still requiring every numerical thread variable to equal one; a new
  test covers the observed value two and the suite remains 8/8. After syntax,
  scheduler, and duplicate checks, job `45881554` was submitted exactly once
  at `2026-09-29T13:11:33Z`. This is the first attempt able to enter the
  synthetic numerical body.
- `[v3_search.adaptive_model_search]` At `2026-09-29T13:13:06Z`, timing job
  `45881554` was RUNNING and CPU-active after its 8/8 launcher tests, with
  175,484 K MaxRSS. It has therefore crossed both corrected preflight guards
  and entered the target-free synthetic numerical body. Continue babysitting;
  no protected access has occurred.
- `[v3_search.adaptive_model_search]` At `2026-09-29T13:38:14Z`, both named
  falsifier production adapters and their shared observed-session/executor
  integration passed target-free review. The permutation adapter passed 7/7
  focused, 6/6 mapping, and 4/4 fresh-M1 tests plus independent adversarial
  probes; the Rademacher adapter passed 11/11 focused and 15/15 primitive
  tests. An exact shared-session probe then bound both adapters while pristine,
  committed four Rademacher references and one whole-set permutation through
  their one-shot capability/receipt/ack ledgers, and rejected cross-scope use.
  Independent shared-code audit found and reverified two atomicity repairs:
  invalid-attempt rollback now restores its counter, and integrity failures
  override retry/ordinary dispositions. No real M2 outcome, final, unassigned,
  SWC, or forbidden full-refit/telemetry input was opened. The remaining code
  blocker is the sealed M1-outer importer plus real observed runner; null
  replay and selected full-development refit execution remain postselection,
  not first-outcome-access gates.
- `[v3_search.adaptive_model_search]` At `2026-09-29T13:57:56Z`, the sealed
  194-cell observed-input bridge, producer-job-45739355 outer-M1 importer,
  serial observed runner, and prerequisite-first CLI are implementation
  complete. Target-free verification is 8/8 importer, 5/5 loader, 11/11
  runner, and 3/3 launcher tests; independent loader/importer and launcher
  audits returned PASS. The runner persists aggregate attempt logs separately
  from mode-0600 numerical execution records, retains sufficient trial and
  coverage detail for downstream reconstruction, keeps candidate-pair detail
  out of the aggregate summary, and makes any incomplete/nonordinary terminal
  exit nonzero. Select one aggregate target-free integration execution through
  `outputs/stage0/tools/test_v3_m2_observed_launch_integration_v1.sbatch`
  with one numerical thread, one requested CPU, 8 GiB, and one hour. It opens
  no membership, outcome, final, unassigned, SWC, or forbidden M0/full-M1
  artifact. Submit only after the exact-name active-job duplicate check, then
  babysit it alongside optional job `45879373`, timing job `45881554`, and
  dependent scale job `45880215`.
- `[v3_search.adaptive_model_search]` At `2026-09-29T14:00:10Z`, scheduler
  test-only accepted the target-free observed-launch integration script, the
  exact-name duplicate query was empty, and the job was submitted exactly once
  as Slurm job `45885787`. It requests one CPU, 8 GiB, and one hour; Sherlock
  currently shows two allocated CPUs while every numerical thread remains
  pinned to one. Its stdout/stderr are
  `$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/ep10-v3-m2-obs-integ-45885787.out`
  and `.err`; the target-free pycache is under job-scoped
  `v3_m2_observed_launch_integration_job_45885787/`. Babysit this job to a
  terminal PASS without duplication. Real membership and outcomes, final,
  unassigned, SWCs, M0 full refit/telemetry, and M1 full refit remain sealed.
- `[v3_search.adaptive_model_search]` Integration job `45885787` completed
  `0:0` in 2:25 with 208,048 K batch MaxRSS and the explicit target-free PASS
  marker. Its nine suites passed with counts 18, 12, 5, 7, 11, 8, 5, 11, and
  3. However, an independent runner audit completed after this job started and
  found two launch-gate defects in the tested snapshot: combined resource
  reconciliation undercounted the coverage-length-16/24 controller branches,
  and the persistence loop conflated non-ledgered integrity failures with
  committed numerical invalid attempts. Both are being repaired with focused
  regressions. Because the code changed for evidenced reasons after allocation,
  this PASS is retained as useful shared-path evidence but is not the final
  observed-launch integration gate. Submit exactly one stable-snapshot rerun
  only after both repairs pass independent re-audit. No real or protected data
  opened in job `45885787`.
- `[v3_search.adaptive_model_search]` At `2026-09-29T14:20:30Z`, the repaired
  observed runner reached a stable target-free snapshot. Author verification
  passed Python compilation and 19/19 focused tests; an independent full audit
  and its separate numeric/artifact audit each returned PASS with independent
  19/19 runs. The stable snapshot now covers all three coverage lengths
  (16/24/28) under caps 52/80, authenticated invalid versus resource commits,
  real fixed-M1 serialization, trial-16 coverage-result linkage, aggregate
  hybrid-identity redaction, and terminal persistence before any postselection
  callback. This evidenced repair supersedes the snapshot tested by job
  `45885787` and selects exactly one final target-free integration rerun through
  `outputs/stage0/tools/test_v3_m2_observed_launch_integration_v1.sbatch`.
  Perform scheduler test-only and one exact-name active-job check immediately
  before submission; do not submit a duplicate. Real M2 membership/outcomes,
  final, unassigned, SWCs, M0 full refit/telemetry, and M1 full refit remain
  sealed.
- `[v3_search.adaptive_model_search]` At `2026-09-29T14:21:06Z`, scheduler
  test-only accepted the stable-snapshot target-free integration launcher and
  the immediately following exact-name active-job query was empty. The final
  integration rerun was therefore submitted exactly once as Slurm job
  `45887906`, requesting one CPU, 8 GiB, and one hour with every numerical
  thread pinned to one; Sherlock displays its normal two-CPU allocation
  rounding. Its stdout/stderr are
  `$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/ep10-v3-m2-obs-integ-45887906.out`
  and `.err`, with job-scoped target-free pycache under
  `v3_m2_observed_launch_integration_job_45887906/`. Babysit this exact job to
  terminal state without duplicate submission. Real M2 membership/outcomes,
  final, unassigned, SWCs, M0 full refit/telemetry, and M1 full refit remain
  sealed.
- `[v3_search.adaptive_model_search]` Stable-snapshot integration job
  `45887906` completed `0:0` in 4:39 with 206,020 K batch MaxRSS and the
  explicit target-free PASS marker. Its nine suites passed with counts 18, 12,
  5, 7, 11, 8, 5, 19, and 3. Independent postflight confirmed the complete
  169-line stdout and 81-line stderr, `TotalCPU 04:22.061`, one node, 8 GiB,
  Sherlock's two-CPU allocation rounding with numerical threads pinned to one,
  and no traceback, protected route, or real-data access. This closes the
  observed launcher/integration gate on the repaired snapshot. The remaining
  first-access blockers are the still-running optional-family qualification
  `45879373`, dependent scale benchmark `45880215`, and falsifier timing
  `45881554`, followed by their independent aggregate/resource postflights.
  Real M2 membership/outcomes, final, unassigned, SWCs, M0 full
  refit/telemetry, and M1 full refit remain sealed.
- `[v3_search.adaptive_model_search]` At `2026-09-29T14:43:40Z`, a read-only
  production-launch audit found that the passed Python launcher still supplies
  `postselection_callback=None`. That is safe for target-free integration but
  is not safe for the first consequential observed run: the selected-refit
  authorization is process-local, weakly bound to the exact live session,
  owner, cache, history, and coverage evidence, cannot be serialized, and the
  runner refuses nonempty-root resume. Exiting after search would therefore
  strand the contract-required selected full-development M1/M2 cold/warm refit
  and its validity gates. The 99 full-controller nulls remain a later separate
  array and are not a first-access gate. Before real M2 membership opens,
  implement and independently qualify one named in-process selected-refit
  coordinator supporting every selected family, including rank-changing
  hybrid successors, and persist an explicit result linkage. This is a
  concrete completion-path repair, not an added SHA, schema, or attestation
  gate. Optional job `45879373`, scale dependency `45880215`, and timing job
  `45881554` continue unchanged; all protected seals remain intact.
- `[v3_search.adaptive_model_search]` At `2026-09-29T15:00:33Z`, two
  independent read-only audits rejected the first selected-refit coordinator
  draft before any real M2 access. The draft could issue more than one terminal
  hybrid receipt, authenticated a caller-supplied full theta/PreparedM2 only
  after construction instead of prospectively authorizing the fresh full-M1
  builder, and authenticated a publicly fitted hybrid precursor only after the
  fit. Repair is therefore constrained to one atomically reserved live-terminal
  procedure capability with the chronology `fresh objective -> one precursor
  when required -> one cold fit -> one warm fit -> complete`, burning on any
  exception. The report must also retain independently recomputable raw
  evidence for the frozen 66+12+1 M1 execution, rank-104 transform checks,
  all-candidate and all-120-pair repeat gates, 12 group contributions, and the
  target-blind octant diagnostic. These are enforcement of the frozen
  completion contract, not new SHA/schema/attestation gates. Optional job
  `45879373`, scale dependency `45880215`, and timing job `45881554` continue
  unchanged; real M2, final, unassigned, SWC, and forbidden M0-full routes
  remain sealed.
- `[v3_search.adaptive_model_search]` At `2026-09-29T15:10:07Z`, the
  independent completion-metric audit found that the first coordinator draft
  incorrectly computed dominance, aggregate MOp gain, leave-one score gains,
  and selected-pair direction stability from in-sample all-194 refit
  predictions. The frozen estimands instead use the authenticated winner's
  unique 194 outer-cross-fitted rows; the all-194 refit is coefficient-only and
  supplies only the cold/warm reproducibility comparison. The repair must bind
  the exact authenticated winner execution, compute all 12 group and leave-one
  metrics from those outer rows, and retain the full target-blind position
  diagnostic without filtering. Before outcome access, resolve the addendum's
  otherwise-unstated median-tie convention target-blind as octant bit 1 when a
  coordinate is greater than or equal to its exact full-development median and
  bit 0 otherwise, ordered `xyz` from `000` through `111`; all 96
  group-by-octant support cells remain reported, including explicit empty
  strata. The primary occupied-octant gain is the equal mean of its nonempty
  group-by-octant means, with a pooled-cell mean retained descriptively; a
  primary octant gain below -0.01 records only the frozen scope limitation and
  never filters cells, reselects a model, or fails a selected-refit gate. A
  second authority audit also confirmed that `PreparedTrialCache` does not
  expose the original moments directly. The narrower repair reuses the existing
  closure-private physical-fold provenance: all 12 cached training PreparedM2
  objects must reauthenticate the exact cells, moments, libraries, and training
  indexes before issuance, and the new all-194 PreparedM2 must pass the same
  provenance check. No redundant source registry is added. No real M2 data was
  opened while finding either defect, and the active resource jobs continue
  unchanged.
- `[v3_search.adaptive_model_search]` At `2026-09-29T15:16:43Z`, dependent
  scale benchmark job `45880215` was placed on a reversible user hold while it
  was still pending with zero compute. Its sbatch reads the shared executor,
  observed-session, and optional-family tests only when the dependency releases;
  allowing automatic startup during the selected-refit edits could therefore
  execute a mixed intermediate snapshot. Preserve the exact job and its
  `afterok:45879373` dependency, then release the hold only after the repaired
  code reaches a compile-clean, independently audited target-free snapshot.
  This is not a duplicate submission and opens no additional data.
- `[v3_search.adaptive_model_search]` At `2026-09-29T16:26:03Z`, the mandatory
  in-process selected-refit completion path reached a frozen target-free
  snapshot after independent math, position-strata, and authority/lifetime
  audits. The final repairs prevent hybrid cold/warm publication from
  resurrecting a burned phase, reauthenticate the full objective and exact
  cold/warm pair for every selected family immediately before atomic report
  commit, reauthenticate the complete receipt chain for hybrids, and burn a
  duplicated or off-chronology full-M1 begin. Independent login-light
  verification passed 13/13 selected-refit tests and 8/8 focused observed
  authority/lifetime tests; the broader loader/runner/launcher regression had
  already passed 7/20/3. Select exactly one new compute-node target-free
  integration through
  `outputs/stage0/tools/test_v3_m2_observed_launch_integration_v1.sbatch`.
  Perform scheduler test-only and an exact-name active-job duplicate query
  immediately before submission, then babysit the submitted job to terminal
  PASS. Keep scale benchmark `45880215` held until that PASS. Real M2
  membership/outcomes, final, unassigned, SWCs, and forbidden M0 full-refit
  inputs remain sealed.
- `[v3_search.adaptive_model_search]` At `2026-09-29T16:26:49Z`, scheduler
  test-only accepted the frozen selected-refit target-free integration script,
  and the immediately following exact-name active-job query returned zero
  matches. The integration was therefore submitted exactly once as Slurm job
  `45899040`, requesting one CPU, 8 GiB, and one hour; Sherlock displays its
  normal two-CPU allocation rounding while all numerical thread variables stay
  pinned to one. Its stdout/stderr are
  `$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/ep10-v3-m2-obs-integ-45899040.out`
  and `.err`, with job-scoped target-free pycache under
  `v3_m2_observed_launch_integration_job_45899040/`. It was pending for initial
  scheduling at the first poll. Babysit this exact job to terminal PASS without
  duplicate submission; keep scale job `45880215` held meanwhile. Real M2
  membership/outcomes, final, unassigned, SWCs, and forbidden M0 full-refit
  inputs remain sealed.
- `[v3_search.adaptive_model_search]` At `2026-09-29T16:34:43Z`, frozen
  selected-refit integration job `45899040` was independently postflighted as
  `COMPLETED 0:0` in 2:57 with `TotalCPU 02:45.244`, one node, 8 GiB requested,
  and 208,924 K batch MaxRSS. All 11 configured suites passed, with counts
  25, 18, 20, 13, 5, 7, 11, 8, 7, 20, and 3 (137 tests total), and stdout
  contains exactly one explicit target-free PASS marker. The complete
  169-line stdout and 116-line stderr contain no traceback, failed/error test,
  crash, protected route, membership/outcome, candidate-pair detail, or real
  data path; only benign locale/module warnings are present. The job-scoped
  runtime contains only the expected 16 compiled files. This closes the
  repaired observed-launch/selected-refit integration gate. Subject to the
  independent held-job binding check, release the existing scale benchmark
  `45880215` without resubmission; its preserved `afterok:45879373` dependency
  must continue to prevent execution before the optional-family report exists.
  Real M2 membership/outcomes, final, unassigned, SWCs, and forbidden M0
  full-refit inputs remain sealed.
- `[v3_search.adaptive_model_search]` At `2026-09-29T16:35:10Z`, an
  independent read-only held-job audit and a fresh scheduler inspection both
  confirmed that scale job `45880215` is still the original zero-compute job,
  with exact command
  `benchmark_v3_m2_search_scale_v1.sbatch`, exported
  `OPTIONAL_QUALIFICATION_JOB_ID=45879373`, and unfulfilled
  `afterok:45879373`. Every shared source/test file changed since its
  submission is covered by integration job `45899040`, and none changed after
  that passing integration began. Select release of this exact existing job
  only via `scontrol release 45880215`; do not resubmit or alter its dependency.
  It must remain dependency-pending until optional job `45879373` completes
  successfully and writes its exact sealed report.
- `[v3_search.adaptive_model_search]` At `2026-09-29T16:37:07Z`, the user
  hold on the original scale job `45880215` was released exactly once without
  resubmission or job mutation. The active queue now reports reason
  `Dependency`, zero elapsed compute, and the original unfulfilled
  `afterok:45879373`; the older `scontrol` reason string remained stale as
  `JobHeldUser`, but the scheduler-facing queue confirms the hold is cleared.
  Continue babysitting optional job `45879373`; scale may start only after its
  exact prerequisite succeeds. Timing job `45881554` remains independently
  CPU-active.
- `[v3_search.adaptive_model_search]` At `2026-09-29T16:46:07Z`, an
  independent route-level snapshot-drift audit cleared the post-start mtimes
  of the two long-running target-free resource jobs without adding a hash or
  schema gate. Both processes import their dependencies exactly once and never
  reload or reread source. Timing job `45881554` reaches the executor only
  through `logo_fold_specs`, `prepare_trial_cache`, and
  `penalty_from_configuration`; recovered prelaunch bytecode and the current
  module have identical normalized semantics on those paths, while every
  other timing numerical dependency except the optional module predates the
  job. Optional job `45879373` reaches only the frozen low-rank/hybrid value
  types, effective-beta maps, synthetic scale calculation, and synthetic
  qualification fits. Its tool, tests, diagnostic, controller, M1, and M2
  dependencies all predate launch; the later optional-module work is confined
  to selected-refit terminal authorization, receipt/lifetime, and phase
  enforcement, with no qualification constant, seed, threshold, solver
  kernel, or recovery-call-graph change. Therefore both live jobs retain
  coherent numerical snapshots that can authorize the current numerical and
  resource paths if their actual reports and terminal postflights pass. Do not
  rerun solely for mtime drift; rerun only for an actual numerical, report, or
  scheduler gate failure.
- `[v3_search.adaptive_model_search]` At `2026-09-29T17:03:17Z`, independent
  persistence accounting replaced the benchmark reports' deliberately
  in-memory `scratch_gib=0` measurements with a conservative nonzero manual
  envelope. One observed controller retains at most 92 attempt pairs (80
  valid plus 12 invalid), while its synchronous selected refit creates 79
  fresh M1 checkpoint bundles containing 937 fits. Allowing 8 MiB per attempt
  pair, 32 GiB for selected-refit checkpoints/report, 64 MiB fixed output,
  concurrent atomic temporaries, and a twofold contingency gives 66 GiB for
  the first observed job and 220.75 GiB for the complete observed-plus-99-null
  campaign. This is 14.37 percent of the frozen 1,536-GiB ceiling, leaving
  1,315.25 GiB (6.96-fold) headroom without deletion credit. Current Scratch
  quota is 5.5/100 TB and 2.0/20.0 million inodes. This clears Scratch for the
  first observed launch; the later null array separately requires its writer
  to be frozen and one target-free full-cap persistence smoke plus a fresh
  quota/scoped-usage check. A null writer is not a first-observed launch gate.
- `[v3_search.adaptive_model_search]` At `2026-09-29T17:06:35Z`, the
  first-observed production wrapper was added at
  `outputs/stage0/tools/launch_v3_m2_observed_search_v1.sbatch` and passed
  `bash -n` plus an independent exact-path/argument/access-order audit. It
  requests one CPU, 8 GiB, 24 hours, and no GPU; pins all six numerical thread
  variables to one; passes the exact four sealed-development roots and six
  prerequisite reports; leaves output derivation to the job-scoped Python
  launcher; and keeps the selected full-development refit synchronous in that
  same process. It contains no null-array route and no additional hash,
  schema, or attestation gate. Do not submit it until optional job `45879373`,
  timing job `45881554`, and dependent scale job `45880215` all complete and
  pass aggregate/scheduler/resource postflight. Immediately before submission,
  recheck `sh_part`, run scheduler test-only, and require an empty exact-name
  active-job query; then submit exactly once.
- `[v3_search.adaptive_model_search]` At `2026-09-29T18:17:03Z`, optional
  qualification job `45879373` remained CPU-active at 5:25 of its six-hour
  limit, with stable 162,144 K MaxRSS, no error, and no report yet. A focused
  read-only walltime audit confirmed that its 30 recovery fits are serial and
  expose no checkpoint or progress artifact; the sole report is written
  atomically only after all fits finish. Preserve and wait for the current job:
  do not cancel, duplicate, or retry the previously access-denied in-place
  time-limit extension. If it actually times out while still CPU-active,
  classify that as an evidenced walltime-sizing failure and submit one clean,
  materially longer qualifier changing only walltime after fresh partition,
  scheduler-test, and exact-name duplicate checks. Preserve dependent scale
  job `45880215` unchanged while `45879373` lives; if the exact `afterok`
  predecessor fails, replace the resulting zero-compute dependent with exactly
  one fresh scale job bound to the replacement qualifier rather than repointing
  or duplicating it early.
- `[v3_search.adaptive_model_search]` At `2026-09-29T18:52:00Z`, optional
  qualification job `45879373` ended `TIMEOUT` after 6:00:11; its batch step
  was cancelled `0:15` with `TotalCPU 5:58:34` and 162,144 K MaxRSS. All 25
  optional and five diagnostic launcher tests had passed, the numerical body
  remained CPU-active to the limit, the log contains only the time-limit
  cancellation, and no report was emitted. Dependent scale job `45880215`
  consequently cancelled with zero elapsed and no allocation. This is the
  second evidenced walltime-sizing failure, not a numerical or scientific
  failure. Increase only the optional launcher's explicit limit from six to
  twelve hours; keep code, seeds, thresholds, one requested CPU, 8 GiB,
  numerical-thread pins, and output guards unchanged. After fresh `sh_part`,
  shell syntax, scheduler-test, and exact-name active-job checks, submit one
  replacement optional qualifier and one fresh scale job bound by `afterok`
  and `OPTIONAL_QUALIFICATION_JOB_ID` to that new job. Timing job `45881554`
  continues independently and must not be duplicated.
- `[v3_search.adaptive_model_search]` At `2026-09-29T18:54:52Z`, current
  `sh_part` confirmed that both eligible `russpold` and public `normal`
  partitions allow the corrected 12-hour qualifier, with one requested CPU,
  8 GiB, and no GPU. Shell syntax and scheduler test-only passed, and the exact
  active job-name query was empty. The walltime-only replacement optional
  qualification was therefore submitted exactly once as Slurm job `45924290`.
  Its job-scoped report will be
  `$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/v3_m2_optional_family_qualification_job_45924290/v3_m2_optional_family_qualification.json`.
  Submit exactly one fresh scale job with `afterok:45924290` and exported
  `OPTIONAL_QUALIFICATION_JOB_ID=45924290` after its own scheduler-test and
  exact-name active-job check; do not reuse cancelled scale job `45880215`.
- `[v3_search.adaptive_model_search]` At `2026-09-29T18:56:40Z`, scheduler
  test-only accepted the unchanged scale benchmark with the replacement
  dependency, and the exact active `ep10-v3-m2-scale` query was empty. A fresh
  scale job was therefore submitted exactly once as `45924664`, bound to
  `afterok:45924290` and exported
  `OPTIONAL_QUALIFICATION_JOB_ID=45924290`. It must remain zero-compute and
  dependency-pending until the replacement optional report succeeds. Cancelled
  scale job `45880215` remains historical and must not be revived. Continue
  babysitting replacement optional `45924290`, timing `45881554`, and dependent
  scale `45924664` without duplicates.
- `[v3_search.adaptive_model_search]` At `2026-09-29T18:58:18Z`, replacement
  optional job `45924290` was RUNNING on `russpold` with the exact corrected
  12-hour limit, one requested CPU, Sherlock's two-CPU allocation rounding,
  and 8 GiB. Replacement scale job `45924664` remained zero-compute PENDING
  with exact unfulfilled `afterok:45924290` and exported report binding to the
  same job ID. Timing job `45881554` remained independently CPU-active. All
  replacement reports were still absent at this first confirmation, as
  expected.
- `[v3_search.adaptive_model_search]` At `2026-09-29T19:20:50Z`, the
  first-observed wrapper's two superseded prerequisite paths were updated
  mechanically from timed-out/cancelled jobs `45879373` and `45880215` to
  replacement optional job `45924290` and replacement scale job `45924664`.
  `bash -n` and an independent read-only re-audit passed: the exact four input
  roots, six prerequisite reports, validate-before-access order, synchronous
  selected refit, one-thread pins, and sealed final/unassigned/SWC/M0 routes
  are unchanged. Live accounting continued to advance for optional
  `45924290` and timing `45881554` with low RSS; scale `45924664` remained
  zero-compute with exact unfulfilled `afterok:45924290`. No prerequisite
  report existed yet, so no postflight or observed submission was attempted.
- `[v3_search.adaptive_model_search]` At `2026-09-29T19:44:11Z`, the earlier
  submission-timing plan was superseded operationally, without changing any
  scientific or data-access gate. Sherlock policy and the episode skill prefer
  scheduler dependencies to a login-node watcher; a persistent polling loop
  was therefore stopped. Select exactly one zero-compute observed attempt using
  the already-audited production wrapper, both a user hold and the explicit
  dependency `afterok:45924290:45881554:45924664`. Submission only reserves a
  scheduler ID: the user hold must prevent allocation and execution even after
  dependencies pass, and release—not submission—is the protected-access gate.
  Before release, independently postflight the exact optional, timing, and
  scale jobs/reports/logs/sacct records; reproduce all frozen resource formulas
  and manual 24-hour/8-GiB checks; reconcile cumulative EP10 use, a fresh quota
  and scoped Scratch use, the 66-GiB first-job bound, and the controlling
  220.75-GiB campaign bound. Immediately before release, require the same job
  to remain zero-compute with unchanged dependencies, command, one-CPU/8-GiB/
  24-hour/no-GPU request, and no duplicate observed job; then release that ID
  once. If any predecessor or postflight fails, never release it. No hash,
  generic schema, attestation, ASTRA, or scientific-code change is introduced,
  and final, unassigned, SWC, and M0-full routes remain sealed.
- `[v3_search.adaptive_model_search]` At `2026-09-29T19:45:05Z`, fresh
  `sh_part` confirmed the selected one-CPU, 8-GiB, 24-hour, no-GPU request is
  valid on `russpold` and `normal`; shell syntax and an identical scheduler
  test-only with both hold and dependency passed; and the exact active
  `ep10-v3-m2-observed` name query was empty. The already-audited production
  wrapper was then submitted exactly once as held observed job `45933922` with
  explicit `afterok:45924290:45881554:45924664`. This ID is not authorization
  to run and must not be duplicated or released early. First verify it remains
  zero-compute, unallocated, user-held, and unchanged. Then preserve every
  path-loaded observed Python/helper file until terminal completion; if repair
  requires a change, cancel this zero-compute job before re-audit rather than
  release it. Release `45933922` exactly once only after all three producer
  postflights and the frozen cumulative resource/Scratch reconciliation pass.
- `[v3_search.adaptive_model_search]` At `2026-09-29T19:45:55Z`, the first
  scheduler confirmation of held observed job `45933922` passed. It is
  `PENDING` for `JobHeldUser`, has `RunTime=00:00:00`, no node and
  `AllocTRES=(null)`, and preserves the exact three unfulfilled dependencies
  `afterok:45924290`, `afterok:45881554`, and `afterok:45924664`. Its command is
  the audited production wrapper, its limit is 24 hours, and `ReqTRES` remains
  one CPU and 8 GiB with no GPU; Sherlock's two-CPU node rounding does not
  change the one-thread numerical pins. Optional `45924290` and timing
  `45881554` remain running, while scale `45924664` remains zero-compute and
  dependency-pending. Do not release or duplicate `45933922` before the full
  terminal postflights and resource reconciliation.
- `[v3_search.adaptive_model_search]` At `2026-09-30T22:09:03Z`, authoritative
  accounting showed optional qualifier `45924290` TIMEOUT after 12:00:08,
  batch CPU 11:59:59 and MaxRSS 162,072 K. Its 25 optional and five diagnostic
  tests passed, but the silent serial 30-fit recovery body emitted neither
  report nor PASS. Dependent scale `45924664` and held observed `45933922`
  cancelled with zero elapsed, zero CPU and no allocation; neither is live or
  may be released. Timing `45881554` remains RUNNING and CPU-active at 32:56.
  All three new reports are absent. This is the third serial walltime failure,
  not a scientific result. Final/unassigned/SWC/M0-full access remains closed.
- `[v3_search.adaptive_model_search]` Selected the same required optional
  recovery experiment with a focused execution repair: twelve independent
  family/rank cases use twelve spawned single-thread workers. Both repeated
  cold fits, and each hybrid's live initializer, remain inside one child;
  all 30 fits, seed 2026092810, fixture, thresholds and final report fields are
  unchanged. Each completed case is immediately atomically saved under the
  new job's `completed_cases/` and logged; a failed or missing case prevents
  final PASS while other successful cases survive. Shared numerical helpers
  and the running timing producer are unchanged. Four focused orchestration
  regressions and a narrow independent code review passed. The replacement
  requests 12 CPUs, 8 GiB and 12 hours, with all six numerical thread variables
  set to one and no GPU. Prior 162-MiB per-process usage suggests about 2.1 GiB
  for the parent plus workers; the parallel peak is not yet measured.
- `[v3_search.adaptive_model_search]` Before replacement submission, cumulative
  conservative allocated use was approximately 127.824/1800 core-hours and
  102.761/144 strict wall-hours, including the live timing elapsed; calendar
  elapsed from first job 45396282 is smaller. Missing ledger job 45866787 and
  cancelled observed 45933922 both have zero cost. Reserving 144 core-hours
  and 12 wall-hours for the replacement plus timing's remaining approximately
  15.06 wall-hours / 30.12 core-hours stays below the frozen ceilings. Fresh
  sh_part supports the request on russpold/normal; Scratch is 5.5/100 TB.
  Submit exactly one replacement after shell syntax, scheduler test-only and
  an empty exact-name active-job query. Do not queue an automatically running
  24-hour scale replacement yet: reconcile remaining time again when the
  producer reports exist. Nulls and any observed release still require their
  original report/resource gates; no new scientific prerequisite is added.
- `[v3_search.adaptive_model_search]` At `2026-09-30T22:10:15Z`, shell syntax
  and scheduler test-only accepted the repaired 12-CPU/8-GiB/12-hour qualifier;
  the exact active `ep10-v3-m2-optqual` query was empty. Submitted exactly once
  as job `46115632`. Its final report is
  `$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/v3_m2_optional_family_qualification_job_46115632/v3_m2_optional_family_qualification.json`;
  completed-case summaries and progress are in the same job-scoped root and
  adjacent `.log`. Separate Slurm diagnostics use
  `$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/v3_m2_optional_job_46115632.slurm.out`
  and `.slurm.err`. Confirm this existing ID, then babysit it and timing
  `45881554`; do not submit duplicates. Scale and observed have no live
  replacement yet. Reconcile remaining frozen budget before the next stage.
- `[v3_search.adaptive_model_search]` At `2026-09-30T22:16:56Z`, replacement
  `46115632` is RUNNING on sh02-06n04 with twelve allocated CPUs and its
  12-hour limit; elapsed 1:10 and batch CPU 1:01 prove execution. The original
  optional tests are still in progress, with no failure marker yet. Timing
  `45881554` remains RUNNING at 33:04, batch CPU 33:04 and MaxRSS 368,580 K.
  Continue these exact handles; verify the repaired recovery body starts
  before claiming any numerical qualification result.
- `[v3_search.adaptive_model_search]` At `2026-09-30T22:19:15Z`, all 25 original
  optional tests, five diagnostic tests and four execution regressions passed
  inside job `46115632`. Its log explicitly reached
  `RECOVERY_START workers=12 independent_cases=12`. At 3:28 elapsed, batch
  aggregate CPU was 21:35 and MaxRSS 1,348,404 K (approximately 1.29 GiB),
  supporting real parallel execution within the 8-GiB request. No recovery
  case or final report has completed yet. Timing `45881554` remains CPU-active
  at 33:06. Observe these exact jobs and the new completed-case progress; do
  not treat test PASS or parallel startup as optional qualification PASS or
  empirical M2 launch. The Codex thread Goal record itself currently reports
  `blocked`; this continuation resumes authorized execution but the available
  goal API has no active/resume mutation. No goal completion is claimed.
- `[v3_search.adaptive_model_search]` At `2026-09-30T22:23:07Z`, the parallel
  qualifier emitted its first real completed case, low-rank rank 1 PASS, and
  its atomic summary exists under
  `$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/v3_m2_optional_family_qualification_job_46115632/completed_cases/low_rank_population_recovery_rank_1.json`.
  At 7:22 elapsed, aggregate batch CPU is 1:05:55 and MaxRSS remains
  1,348,404 K. The other eleven cases are not yet complete; the final report
  is absent. Timing `45881554` remains RUNNING at 33:10 with its supplement
  absent. Continue these exact handles; this is retained synthetic recovery
  evidence, not empirical M2 launch or scientific acceptance.

## 2026-09-30 — Existing v3 continuation and partial recovery QC

- `[v3_search.adaptive_model_search]` Resumed the existing episode without a
  new Goal, execution amendment, duplicate job, outcome access, or scientific
  experiment. The current chat Goal API returned no Goal; the local existing
  GOAL.md and episode run identity remain unchanged. The terminal-profile
  follow-up remains planned and inactive.
- At `2026-09-30T23:17:02Z`, optional recovery `46115632` remained RUNNING at
  1:01:15 with 12 allocated CPUs; batch CPU 10:57:55 and MaxRSS 1,348,404 K
  support active parallel computation within its existing 8-GiB request.
  Exactly one of twelve recovery cases is complete: low-rank rank 1. Its mean
  conditional KL is 3.4449865019472765e-08 bits against the frozen 1e-05 limit,
  and its deterministic repeat is bit-exact. This is partial contract-required
  synthetic QC, not whole optional qualification PASS or empirical evidence.
- Preserved the actual completed-case JSON under
  `outputs/stage0/v3_m2_optional_family_qualification_job_46115632/completed_cases/`.
  Rendered, inspected, and shared
  `outputs/figures/v3_m2_required_recovery_partial_job_46115632.png` and its SVG.
  No Neurofig skill was found in available/user/project skill locations; the
  figure uses Sherlock's existing Matplotlib module, with no installation.
- At the same snapshot, timing `45881554` remained RUNNING at 34:04:31;
  batch CPU 34:04:13 and MaxRSS 368,580 K. Neither final optional report nor
  timing supplement exists. Scale `45924664` and observed `45933922` remain
  historical zero-compute cancellations, with no live replacement.
- Required resource reconciliation at `2026-09-30T23:13:51Z` returned all 80
  recorded parent jobs. Current accounting did not reproduce the prior strict
  wall basis. Preserve, rather than reduce, the older conservative estimate:
  104.80878/144 strict wall-hours and 141.59733/1800 allocated core-hours,
  after carrying forward only new live-job use. Existing remaining reservations
  consume 25.01 wall-hours, leaving 14.18122 unreserved wall-hours and
  1498.06044 core-hours. Evidence:
  `outputs/stage0/resource_accounting_20260930_resume.json`. Prior records are
  untouched; lower present accounting totals confer no extra budget/authority.
- Exact next authorized action: let these two existing producers finish,
  preserving each actual case/report. Postflight their exact reports and job
  outcomes once available. Reconcile actual remaining resources before
  selecting a fresh scale attempt bound to optional job `46115632`; do not
  automatically reserve its current 24-hour request against the 14.18-hour
  unreserved bound. Apply the unchanged combined scale/timing gate and
  cumulative budget before any empirical M2 launch. The stale observed wrapper
  report IDs need mechanical rebinding only after successful producers exist.
- The saved ASTRA experiment program remains at
  `outputs/astra/v0.0.14/astra.yaml`; it was not overwritten by the lossy
  milestone exporter. Its unchanged validation is reused. No new finding,
  terminal outcome, canonical action, or successor is asserted. Final,
  unassigned, SWC, and forbidden M0 routes remain closed.

- `[v3_search.adaptive_model_search]` At `2026-09-30T23:20:32Z`, existing
  optional `46115632` is still RUNNING at 1:04:46 (batch CPU 11:36:44;
  MaxRSS 1,348,404 K), and timing `45881554` is RUNNING at 34:08:02
  (batch CPU 34:07:44; MaxRSS 368,580 K). Both required final reports remain
  absent, with only the same low-rank rank-1 case complete. This is an ongoing
  producer stage, not an episode terminal result. No fresh scale/observed job
  was submitted and no empirical M2 outcome was accessed. The next action and
  resource gates above remain operative.

## 2026-10-01 — focused interpretation review during authorized babysit

- At `2026-10-01T05:30:29Z`, timing `45881554` remained RUNNING at
  40:17:59 with steady batch CPU and MaxRSS 368,580 K. The optional watcher
  last reported `46115632` RUNNING at 6:51:01, one of twelve required cases
  complete, no final report or error. The same producers remain the next
  execution step; no new scientific run, contract change, or data opening.
- The scientist instructed active use of relevant exposed BR retrieval,
  focused critique, and scientific review at consequential scientific choices,
  reusing existing evidence and retaining restricted-data and formal Society
  boundaries. The existing M1 aggregate provides a substantive interpretive
  question: how far could a future M2 gain discriminate residual pair
  organization from a finite source/observation reference?
- BR source-backed retrieval was actually used. The initial long query
  returned unrelated Crossref components and was rejected. One focused query
  identified Gao PMID 41253150; five unrelated hits were excluded. The primary
  public Gao abstract was separately verified, as were Han's visual-cortex
  prior art and Yuan's different-assay observation limitations. These support
  retaining source/population/observation competitors; they neither validate
  EP10 independently nor quantify Gao missingness. Falasconi metadata was
  accessible but fresh abstract/full-text verification was unavailable, so its
  specific module count was not a premise of this review. Raw BR returns and
  relevance decisions are saved in
  `outputs/stage0/scientific_review_20261001/`.
- No standalone BR focused-critique endpoint is exposed. A focused native
  scientific critique used the BR external rubric and the existing frozen
  specification/aggregate log only. BR validated and stored verdict
  `ext_review_verdict_7c31bcf5fcb9` at `2026-10-01T05:34:46Z`, bound to
  directive `ext_review_dir_2a3addc6254a`. The verdict's limited correctness
  pass and sound narrow judgment are not a whole-pipeline certificate;
  completeness is incomplete and overall `explore_more` means already-required
  pending evidence only. BR did not independently read external artifacts.
  This is advisory scientific review, not formal Society or canonical
  acceptance. Saved `review.md`, `verdict.json`, and `br_submitted_review.json`
  record its scope and evidence.
- Scientific implication and decision: M1-minus-M0 is incremental to an
  informative soma/source reference. Its small heterogeneous gain does not
  establish geometry's irrelevance. M1's adequacy against uniform prediction
  does not establish an exhaustive nuisance model. A future calibrated M2 gain
  would challenge the fitted M1 observation generator, while omitted nonlinear
  source effects, population structure, and correlated recording remain
  competitors. The review supports the existing restricted continuation and
  claim limits; it changes no frozen estimand, execution gate, stopping
  decision, data role, or authority. M2 and all 99 full-controller nulls remain
  pending; the terminal-profile follow-up remains unactivated.
- Rendered and visually inspected the smallest useful **exploratory** figure
  from existing authorized aggregate values only:
  `outputs/figures/v3_m1_reference_gains_exploratory_20261001.{png,svg,json}`.
  The source script is in the review directory. It shows two different
  reference comparisons, with no invented uncertainty or new experiment.
  Neurofig plotting remains unavailable; the existing Matplotlib module was
  used. The earlier partial recovery figure remains QC and unchanged.
- The native thread Goal is now authoritatively present and active; it was
  resumed without creating a replacement. The existing ASTRA program/export
  path and passing unchanged validation remain preserved. No successor,
  scientific promotion, or episode completion is asserted.

- `[v3_search.adaptive_model_search]` At `2026-10-01T08:08:45Z`, existing
  optional job `46115632` remains RUNNING at 9:52:58 with 2:07:02 left.
  Its low-rank rank-2 required recovery case completed PASS and was preserved
  at `outputs/stage0/v3_m2_optional_family_qualification_job_46115632/completed_cases/low_rank_population_recovery_rank_2.json`.
  Actual case KL is `4.503538504900079e-08` bits against the frozen `1e-05`
  ceiling; the repeat is bit-exact, gradient infinity norm is
  `9.990211589313595e-07`, and the fit took 2580 iterations. Exactly 2/12
  required cases are now complete; the full report is absent. A case-local
  `every_authorized_rank_qualified` flag applies only to its recorded singleton
  rank and is not overall qualification. No empirical M2 result was accessed.
- Rendered, inspected, and shared the smallest updated **QC** figure from
  the actual saved rank-1 and rank-2 reports:
  `outputs/figures/v3_m2_required_recovery_partial_job_46115632_cases2.{png,svg,json}`,
  with source script in `outputs/stage0/scientific_review_20261001/`.
  These prescribed fixtures support recovery at those two ranks only. Timing
  `45881554` last remained RUNNING at 42:53:53 (`2026-10-01T08:06:23Z`),
  with stable MaxRSS 368,580 K and no supplement/error. Continue these same
  attempts; all existing overall gates, budget reservations, and claim limits
  remain operative.

- `[v3_search.adaptive_model_search]` At `2026-10-01T10:01:56Z`, the
  low-rank rank-3 population recovery case completed PASS with actual KL
  `3.029344341289948e-08` bits, bit-exact repeat, gradient infinity norm
  `9.987512484675673e-07`, and 3151 iterations. The actual case JSON was
  preserved and the 3/12 **QC** figure was rendered, inspected, and shared:
  `outputs/figures/v3_m2_required_recovery_partial_job_46115632_cases3.{png,svg,json}`.
- At `2026-10-01T10:19:15Z`, exact Slurm accounting establishes optional
  `46115632` **TIMEOUT**, elapsed 43,221 seconds (12:00:21), 12 allocated
  CPUs; batch CANCELLED `0:15`, MaxRSS 1,348,404 K. Slurm stderr explicitly
  attributes cancellation to the time limit. Parent `0:0` is not success when
  its scheduler state is TIMEOUT. All 34 in-job focused tests passed, but only
  low-rank ranks 1–3 completed, nine required cases are incomplete, and the
  final qualification JSON is absent. Overall qualification did not pass.
  Actual case reports, logs, raw accounting, and a clearly separate
  `terminal_diagnostic.json` are durable in
  `outputs/stage0/v3_m2_optional_family_qualification_job_46115632/`; no
  producer result was fabricated. The empirical M2 gate remains closed.
- Timing `45881554` remains RUNNING at 45:06:44 with no final supplement.
  A focused read-only timeout diagnosis and required remaining-resource gate
  are determining repair versus stop; no resubmission, resource expansion,
  threshold/family change, new test cycle, or sealed data access occurred.
  Continue the existing timing watch while resolving this concrete issue.

- Required post-timeout resource reconciliation retains the prior conservative
  accounting floor and counts the optional job's 21-second overrun. At
  `2026-10-01T10:19:15Z`, used strict wall is 126.936833 hours and allocated
  core time is 296.234000 hours. The existing timing reservation consumes
  another 2.887778 wall-hours, leaving 14.175389 unreserved wall-hours and
  1497.990444 unreserved core-hours. Even immediate timing completion would
  leave only 17.063167 wall-hours; the unchanged 24-hour scale request cannot
  be reserved. The all-99-replay combined gate remains **unresolved**, since
  optional, scale, and timing reports are not complete; total timing duration
  is not a substitute for measured units. Saved evidence:
  `outputs/stage0/resource_accounting_20261001_optional_timeout.json`.
- Focused static/log diagnosis found substantial prescribed solver runtime,
  not a demonstrated wrong solver or deadlock. The three passes required
  complete cold repeats; rank 2/3 used thousands of iterations per fit.
  Reusing an identical hybrid evaluation could save one evaluation per sweep,
  but no measured saving would resolve the nine incomplete cases or the
  resource gate, and it cannot explain unfinished pure low-rank cases. No
  code change, test/profiling cycle, or automatic resubmission followed.
- One bundled disposition question was asked: recommended closeout as
  **INCOMPLETE SEARCH** after existing timing finishes, or retention at a human
  gate for a separately specified continuation. The question grants no new
  execution authority, and no answer is inferred from elapsed time. Continue
  the already reserved timing job; fresh empirical M2 work remains closed.

## 2026-10-01 — Both producer attempts terminal; resource/human gate

- `[v3_search.adaptive_model_search]` At `2026-10-01T13:14:08Z`, exact UTC
  accounting establishes timing job `45881554` TIMEOUT at `48:00:29`, end
  `13:13:00Z`, two allocated CPUs. Batch CANCELLED `0:15`, MaxRSS 368,580 K;
  stderr explicitly records the time limit. Eight focused tests passed, but
  the final timing supplement and all intermediate timing artifacts are
  absent. Parent `0:0` does not make this TIMEOUT a pass. Actual logs, captured
  UTC accounting and a separate root diagnosis are preserved in
  `outputs/stage0/v3_m2_falsifier_timing_v1_job_45881554/`.
- Optional `46115632` remains terminal TIMEOUT with three actual passing
  low-rank cases and nine incomplete cases, no overall report. Both producer
  watches are complete. No substitute qualification/timing report was written.
  No reported exception identifies a numerical defect to repair; total timing
  duration cannot replace the four missing measured units.
- Final required accounting retains the conservative floor: strict wall
  `129.832667/144` hours, allocated core `302.025667/1800`; remaining
  `14.167333` wall and `1497.974333` core hours, no live reservations.
  Evidence: `outputs/stage0/resource_accounting_20261001_producers_terminal.json`.
  Independent focused arithmetic agrees. The unchanged 24-hour scale request
  cannot be reserved; the all-99-replay gate remains unresolved.
- Search evidence is **INCOMPLETE at the preaccess resource/human gate**.
  Empirical M2 valid trials and full-controller replays launched: zero. This
  supports no success or negative biological result. The pending bundled
  disposition question has no answer; no new authority is inferred. No new
  compute, code/test cycle, model/threshold/budget change, outcome opening or
  successor followed. The next authorized action is the human disposition,
  not a new producer or scale attempt.
- The substantive existing M1 scientific review and returned evidence are
  reused: `ext_review_verdict_7c31bcf5fcb9` supports the current restricted
  claim boundary, with competing explanations retained. No ceremonial review
  was added for timeout/QC. Review/source artifacts and the three-case QC
  figure are preserved. Formal Society and restricted-data boundaries remain
  unchanged.
- Terminal handoff: `outputs/stage0/producer_terminal_handoff_20261001.md`.
  All seven local status projections now carry this material milestone. The
  existing native Goal was resumed without replacement; scientific completion
  is not asserted. ASTRA program `outputs/astra/v0.0.14/astra.yaml` remains
  unchanged with passing pinned validation reused. Separately, the legacy
  single-analysis exporter remains lossy and was not used to overwrite it;
  no new complete export is claimed.

- Required BR session snapshot persisted at `2026-10-01T13:17:16Z` in the
  existing run `br_20261001_053007_beed42e3b1` and canonical Codex session.
  Initial `session_not_started` error was repaired by transparently
  late-recording the missing start event (`research_evt_0004`), then retrying
  once. Actual returned closeout is saved in
  `outputs/stage0/br_session_closeout_20261001.json`. BR logging status
  `succeeded` denotes snapshot persistence, not scientific episode success,
  acceptance, Goal completion or new authority. No transcript, external trace
  or protected artifact was uploaded. The resource/human gate remains pending.

## 2026-10-01 — User-authorized wall-time reset and same-episode continuation

- `[v3_search.adaptive_model_search]` At `2026-10-01T22:10:05Z`, recorded the user's
  explicit authorization to reset the wall time. This authorizes a
  fresh 144-hour strict-wall window at the existing ceiling. The prior
  129.83266667 wall-hours remain historical consumption; CPU is not reset:
  302.02566667/1800 allocated core-hours used, 1497.97433333 remaining.
  Durable authorization: `outputs/stage0/v3_wall_time_reset_authorization_20261001.json`.
  SEARCH_POLICY now names this accounting epoch. Only wall accounting changed;
  model families, thresholds, data access, 99 nulls, other resource ceilings
  and stopping rules retain their prior meanings.
- The pending closeout/continuation choice is resolved in favor of continuation.
  The prior terminal handoff remains accurate history. Existing native Goal
  identity is retained; its API currently reports blocked and exposes no resume
  operation, so no replacement Goal is created. This does not prevent carrying
  out the user's new authorized instruction in the existing episode.
- Next authorized work is the same prerequisite evidence: reuse the three
  complete recovery cases and resume only nine incomplete cases. The known
  timing defect is loss of all partial evidence on timeout; retain complete
  probe measurements and aggregate progress outside timed calls, with each
  probe's twelve folds and single dictionary together. Focused orchestration
  repairs/regressions precede replacement launch; no extra scientific
  experiment or broad audit is added. Scientific BR review already returned
  support for the restricted interpretation and is reused unchanged.

- `[v3_search.adaptive_model_search]` Selected replacement producers at
  `2026-10-01T22:13:35Z` after the diagnosed timeout-preservation repairs. Optional
  reuses exactly the three durable job46115632 rank1/2/3 case artifacts;
  copied legacy evidence must match those originals, and nine incomplete
  cases restart at their cold case boundaries. All30 frozen fits remain
  represented: six reused and24 still required. Ten focused orchestration
  regressions passed; the actual legacy loader accepted3cases without fitting.
  The unchanged25+5 numerical test passes are reused.
- Timing now runs eight independent complete probes with process-local
  dictionary/caches and single numerical threads; each probe retains12 serial
  folds and its original measurement/aggregation rules. Aggregate phase/count
  and completed-probe measurements are saved outside measured calls. All17
  relevant lightweight tests passed (original8 plus9 orchestration tests);
  no new numerical fit or scientific experiment was used to test the repair.
- Launch intent: optional48h/9requestedCPUs/8GiB and timing48h/8CPUs/8GiB,
  noGPU. Partition discovery supports48h in russpold/normal; changed launchers
  pass shell syntax and scheduler test-only. Test-only IDs46237201/46237202
  are hypothetical, not submitted jobs. Exact-name duplicate check was empty.
  Conservative reservation is96 new-window wall-hours and864core-hours,
  leaving48wall-hours and633.974333core-hours unreserved. Evidence:
  `outputs/stage0/resource_accounting_20261001_wall_reset_launch.json`.
  These are required target-free producers; real M2 remains closed.

- `[v3_search.adaptive_model_search]` Optional resume submitted as **46237243** at `2026-10-01T22:13:46.783215+00:00`. Same frozen qualifier, 3 saved cases reused /9new cases; 48h,9requestedCPUs,8GiB. Returned scheduler ID recorded before continuing.

- `[v3_search.adaptive_model_search]` Timing replacement submitted as **46237247** at `2026-10-01T22:14:19.105394+00:00`. Eight independent complete probes,48h,8CPUs,8GiB, one numerical thread per worker. Required units and final report remain unchanged; aggregate-only completed/progress artifacts survive timeout.

- `[v3_search.adaptive_model_search]` At `2026-10-01T22:16:02Z`
  (15:16:02 Pacific), both actual jobs46237243 and46237247 are PENDING,
  elapsed0 and no allocatedCPUs yet. Their reservations remain active;
  there is no producer error or justification for duplicate submission.
  Wall-reset accounting has0used/96reserved/48unreserved hours. Prior
  failures remain historical. Existing ASTRA program is retained with its
  unchanged validation; its SEARCH_POLICY input now points to the explicit
  reset authorization. No new scientific result, figure or ceremonial BR
  review is generated from scheduling state. The existing native Goal API
  still reports blocked; no available Goal tool can set it active, and no
  replacement Goal or canonical Brain Researcher action was created.

## 2026-10-01, 21:20 Pacific — Approved continuation; concrete timing failure

- `[v3_search.adaptive_model_search]` The user's “approve” confirms continued
  work under the already-recorded wall reset; it does not reset time again or
  reset CPU usage. Optional46237243 is RUNNING with9CPUs and21662s elapsed
  at2026-10-02T04:20:55Z, successfully reusing the3saved cases; no additional
  complete case or final report yet. CPU activity and~1.02GiB peak RSS support
  live computation. Continue this same job.
- Timing46237247 FAILED1:0 after5868s (end2026-10-02T00:01:20Z),8CPUs,
  peak RSS1807964KiB. Producer blocked report says BrokenProcessPool;
  permutation progress records UncertifiedLineSearchFailure during fresh
  permutation M1. Core2/core4 each recorded one complete fold; no complete
  probe exists. Actual report/logs/progress/accounting are preserved under
  outputs/stage0/v3_m2_falsifier_timing_v1_job_46237247/.
- Focused source diagnosis identifies an exception transport defect: the M1
  exception requires scalar_context but its serialized args contain message
  only. Failure reporting lost the numerical scalar context and broke the pool.
  Repair error transport/reporting without changing solver math; use its
  focused regression, then run only the identical failing permutation probe
  to recover actual scalar evidence. No generic qualification cycle.
- The timing launcher also omitted the frozen M1 CPU/backend binding.
  Actual failed node sh03-06n12 is CPU_SKU:7502; therefore CPU-family mismatch
  is NOT established as the numerical cause. Add the required binding and
  effective backend attestation for the isolated continuation; do not infer
  solver success or alter tolerances. Final/M2 outcome boundaries remain closed.
- Updated required resource accounting is
  outputs/stage0/resource_accounting_20261002_timing_failure.json; release only
  the unused reservation of the terminal timing job, preserve its actual use
  and the live optional reservation. Native Goal identity remains preserved.

- `[v3_search.adaptive_model_search]` At `2026-10-02T04:26:20.609063+00:00`, the exception transport
  defect is repaired.21/21 focused tests pass, including an actual spawned
  process carrying a real UncertifiedLineSearchFailure from mocked fitting
  without destroying another probe. Failures retain allowed scalar solver
  telemetry and fit ordinal; other complete probes remain preservable. The
  optimizer, fixture, seeds, tolerances and measurement rules are unchanged.
- Selected only the identical failed permutation probe for a bounded4-hour,
  one-requested-CPU,8GiB diagnostic on CPU_SKU:7502 with the existing exact
  OpenBLAS-Zen/one-thread backend attestation before fitting. Prior observed
  failure occurred after1:37:48; four hours bounds recapture with headroom.
  No full8-probe rerun is selected. A pass is one complete probe only;
  failure preserves its actual scalar context for the next repair decision.
  Request syntax and scheduler test-only pass (hypothetical ID46270445);
  exact-name duplicate check is empty. Reserve4wall/8conservative core-hours,
  leaving90.37wall and1044.934333core hours after the live optional reservation.
- Rendered, inspected and shared actual partial/incomplete **QC** figure:
  outputs/figures/v3_m2_falsifier_partial_checkpoint_job_46237247.png (SVG and
  source script retained). It reports recorded fold-completion counts only;
  zero full probes means no valid overall timing projection or empirical M2
  finding. Existing Matplotlib was used; no new data were generated to plot.

- `[v3_search.adaptive_model_search]` Isolated permutation diagnostic submitted as **46270474** at `2026-10-02T04:26:49.559790+00:00`. Actual returned ID recorded. Continue optional46237243 independently; no other producer resubmission or empirical M2 access.

- `[v3_search.adaptive_model_search]` Initial same-handle scheduler check
  completed at~2026-10-02T04:27:50Z: optional46237243 remains RUNNING on
  sh03-06n12 at6:07:41; isolated diagnostic46270474 is PENDING, scheduler
  display2CPUs for1requested (the conservative reservation already counts2).
  The observation command was awaited to completion after a delayed response;
  no process was restarted from an observation timeout. No diagnostic output
  exists before allocation. Keep the same two handles. ASTRA program remains
  preserved with unchanged validation, and the original native Goal remains
  the same object; scientific completion is not asserted.

## 2026-10-02, 11:01 Pacific — Confirmation failure localized; recovery still running

- `[v3_search.adaptive_model_search]` At 2026-10-02T18:01:03Z, optional
  recovery46237243 remains RUNNING:70870s elapsed,9allocatedCPUs,48h limit.
  Three reused low-rank cases remain complete; no additional complete case or
  final report is present. Remaining job allowance is28h18m50s. Keep this job.
- Isolated permutation diagnostic46270474 FAILED1:0 after4353s, ending
  2026-10-02T05:44:49Z. It successfully retains the actual optimizer failure;
  CPU7502/OpenBLASZen/effective1thread attestation passes. At warm-path
  ordinal64, lambda1000, iteration3, mandatory post-witness confirmation sees
  direct directional derivative0.0 although Newton gradient-dot-direction is
  -9.23649169756876e-34. Gradient infinity norm6.661338147750939e-16 is below
  1e-7; previous relative objective change4.1397441992912907e-16 is below1e-10.
  Strong-Wolfe rejects the non-descent value before evaluating a trial.
- The finite-stall certificate is explicitly ineligible and not attempted;
  certificate_pass=false. This is consistent with cancellation at numerical
  convergence, not a passing fit or proof that relaxing acceptance is valid.
  No optimizer/threshold modification or blind retry is made. Bounded source
  assessment will determine whether an implementation-only repair exists.
- Actual diagnostic, backend report, progress and Slurm logs are preserved in
  `outputs/stage0/v3_m2_falsifier_permutation_diagnostic_job_46270474/`.
  Resource accounting is `outputs/stage0/resource_accounting_20261002_diagnostic_terminal.json`:
  93.160833wall-hours and1050.516000allocatedcore-hours remain unreserved after
  retaining the live optional job's full reservation. Prior usage remains
  recorded; the user's existing wall reset is unchanged.
- Real M2 remains unopened; no full timing report, scale result or combined
  99-replay resource projection is available. This numerical QC result is
  not biological evidence. Existing BR scientific interpretation remains
  unchanged; no scientific acceptance or Formal Society transition occurs.
  ASTRA program and its unchanged passing validation are retained; runtime
  state belongs here. No new ASTRA export is claimed or needed for this update.

## 2026-10-02, 11:11 Pacific — Direct-derivative conformance repair and bounded replay

- `[v3_search.adaptive_model_search]` Source assessment found a concrete
  discrepancy with solver addendum v1 lines110–129: both alpha-zero and trial
  derivatives contracted rounded per-cell feature residuals instead of directly
  summing candidate-centered directional contributions. This is distinct from
  the unchanged, correctly enforced confirmation and certificate rules.
- Repaired only these derivative paths in tools/fit_v3_m1.py. Candidate-level
  projection precedes expectation; math.fsum combines contributions within
  batches, across batches, and across cells/ridge/linear correction. The trial
  path shares existing candidate passes; alpha-zero uses two new streamed
  passes counted in time and candidate-visit telemetry. No candidate cache,
  fallback, tolerance change, acceptance relaxation or optimizer replacement.
- Six bounded array regressions passed in0.131s in
  tests/test_v3_m1_direct_derivative_regression.py. They reproduce rounded-away
  signal, compare against an independent full-feature expectation, check initial
  and trial paths, unchanged objective/gradient, late observed batch, singleton,
  constant increment, cancellation and nonfinite failure. Existing helper-call
  test signature was updated. Native implementation review plus root's focused
  source review found no blocker to the bounded replay. No fitted experiment
  ran on the login node. New production timing overhead is unmeasured.
- Preserved earlier M1 qualifications, predictions and comparisons as historical
  artifacts; they do not qualify the changed evaluator. Affected numerical
  qualification and clean M1 production must be revalidated before downstream
  reuse; actual outcome data were not opened or refitted here. Optional recovery
  uses its own optimizer and continues unchanged in its existing processes.
- Submitted only the same isolated permutation diagnostic as **46325527** at
  2026-10-02T18:12:27.802611+00:00: unchanged4h,1requestedCPU (2conservatively reserved),8GiB,
  AMD7502/OpenBLASZen/effective1thread; fresh job-scoped output root. The repair
  has not yet been shown to resolve the original numerical failure. If the
  compliant calculation still fails, preserve its scalar evidence and stop for
  the consequential decision rather than silently relaxing acceptance.
- Existing launcher validation is reused because the launcher is unchanged;
  exact-name duplicate check was empty. New budget evidence is
  outputs/stage0/resource_accounting_20261002_derivative_repair_launch.json:
  89.160833wall-hours and1042.516core-hours remain unreserved after this request
  and the ongoing optional reservation. No additional wall reset.
- Last independent recovery observation was19h47m49s RUNNING,9CPUs,
  recorded step CPU172h54m32s and peak RSS1072480KiB, with the same3completed
  cases. This supports active calculation but no additional successful case.
- This is a mechanical conformance repair, not a new scientific design/result
  decision or Formal Society action. Prior BR source-backed critique remains
  the interpretation boundary for historical M1 comparisons, with numerical
  reuse now pending. No new scientific BR review is warranted by mock success.
  ASTRA program remains unchanged and no new complete export is asserted.

- `[v3_search.adaptive_model_search]` At~2026-10-02T18:12:53Z,
  recovery46237243 is RUNNING at19h53m00s with9CPUs; repaired diagnostic
  46325527 is PENDING for Resources, scheduler2CPUs and4h limit. No duplicate.
  Rendered and inspected the actual failed-run numerical QC figure
  outputs/figures/v3_m1_permutation_confirmation_qc_job_46270474.png
  (SVG and plotting source retained). It contrasts passed ordinary tolerances
  with the rejected mandatory confirmation; no biological evidence or claim
  that the repair has passed. Existing Matplotlib was used because Neurofig
  skill/renderer availability was previously found unavailable.

## 2026-10-02, 11:30 Pacific — Continue existing attempts

- `[v3_search.adaptive_model_search]` User requested continued work; preserve
  the same episode, wall-reset epoch, CPU history, scientific gates and Goal
  identity. At18:29:36UTC recovery46237243 is RUNNING72583s/9CPUs and
  repaired diagnostic46325527 is RUNNING952s/2allocatedCPUs. Diagnostic
  started18:13:44UTC; actual CPU7502/Zen/effective1thread attestation passes.
  It is still in permutation M1 with no complete probe or failure result.
  Recovery still has the same3reused cases and no final report. Continue these
  existing attempts; no new job or duplicated qualification is submitted.
- Backend artifact is retained at
  outputs/stage0/v3_m2_falsifier_permutation_diagnostic_job_46325527/numeric_backend.json.
  The exact unchanged job reservations leave89.160833wall-hours and1042.516
  allocatedcore-hours unreserved. A bounded read-only check of the existing
  frozen revalidation sequence prepares the result-dependent next action;
  it adds no prerequisite or new synthetic experiment.
- Native Goal API still reports the same object blocked and has no resume
  operation; no replacement or false completion was created. This does not
  alter the user's authorization for these already-running Slurm attempts.

## 2026-10-02, 11:34 Pacific — Existing affected M1 qualification resumed

- `[v3_search.adaptive_model_search]` Submitted existing frozen one-thread
  68-row/36-fit M1 qualification as **46328672** at18:34:31UTC:30minutes,
  one requestedCPU (two conservatively reserved),8GiB,AMD7502/Zen/one thread,
  seed20260928,all12lambdas,warm1/warm2/zero. This qualification is required
  after the direct-derivative evaluator change. It does not depend on a PASS
  from the separate permutation diagnostic; the previous sequential intent
  added no scientific dependency and is superseded to allow parallel work.
- Launcher probe_v3_m1_synthetic_warm_zero_single_thread_v3.sbatch retains
  mandatory per-process backend attestation and affected M1 engine regressions.
  Only its unchanged backend-policy test invocation is skipped by reusing its
  prior PASS. Shell syntax passed; no scheduler constraints/resources changed,
  so prior request validation is reused. Exact-name duplicate check was empty.
- Reserve0.5wall/1allocatedcore hour, leaving88.660833wall and1041.516core hours
  unreserved after all current requests. No new wall reset, no real-data access.
- BR external review directive ext_review_dir_ce90de996e4f provides the rubric
  for focused native critique of qualification/result reuse. Actual directive
  is retained under stage0/scientific_review_20261002_m1_reuse/. The server has
  not read local artifacts or produced a scientific verdict. Returned tailored
  checks are empty; native critique applies only relevant criteria. The current
  exposed pipeline reviewer is imaging-specific and cannot substitute for this
  methodological critique. Formal Society remains untouched.
- The tool automatically opened research logging under the returned binding
  codex:codex:01a0f492-d6d3-79c0-a4b3-992037a3b843 (client prefix was normalized
  twice). This is a research-log binding, not a new episode or Goal. Preserve
  returned binding for verdict/closeout; do not manufacture canonical state.
- Early focused critique supports parallel target-free qualification but flags
  that v3's execution-only amendment cannot waive numerically affected prior
  population recovery/throughput evidence after an evaluator repair. Resolve
  that exact dependency before production. Neither the old45734258 binding nor
  falsifying the truthful historical M0 post-fit opening45760290 is permissible.
  Existing static geometry, fold definitions and uniform-moment rank evidence
  remain reusable. No additional fit or new production job is authorized by review.

## 2026-10-02 — Focused scientific review resolves reuse scope

- `[m1_single_thread_execution_qualification]` Job46328672 is the existing
  ASTRA experiment of this ID, currently running the affected engine regressions
  before its frozen36fits. Its runtime backend attestation passed. The earlier
  broad v3_search log key described the blocked downstream context.
- BR directive ext_review_dir_ce90de996e4f was applied by independent native
  critic /root/m1_reuse_scientific_critique. Verdict accepted by BR as
  ext_review_verdict_4fe160ae4283; actual directive, verdict and response are
  preserved under outputs/stage0/scientific_review_20261002_m1_reuse/.
  This is source/code-grounded focused methodological critique, not hosted
  access to local data, Formal Society, or scientific acceptance.
- Decision impact: supports the parallel36-fit qualification and corrected
  derivative as contract-determined engineering. CHANGES the initial proposed
  downstream reuse: new36-fit PASS alone is insufficient after evaluator change.
  Earlier v1/v2 population-recovery and throughput obligations remain in force;
  v3 changed execution mode only. Refresh the two frozen544-row sampled and
  exact-population fits and194-row all-K8 numerical throughput. Reuse existing
  unchanged79uniform-moment ranks, geometry, designs and independent M2 optional
  evidence. Do not repeat36fits inside a monolithic wrapper.
- Historical M0 opening is not automatically a scientific amendment gate:
  design_timing.choice_independence explicitly preserves design despite
  operational exposure, while M0_comparator_root remains forbidden to fitting,
  tuning, initialization and repair-selection code. Retain lifetime opening
  truthfully, keep the new process closed to M0, prohibit comparator-driven
  repairs, and repair stale runner/history representation prospectively before
  any new production. No new thresholds or selection based on M0 are permitted.
- Preparation of the minimal existing affected-numerical subset is authorized;
  its launch still requires46328672 PASS and a fresh resource reservation.
  Current failure diagnostics remain unresolved. On unchanged numerical failure,
  preserve actual scalar evidence and stop the dependent branch for a decision.

## 2026-10-02, 11:47 Pacific — Qualification PASS; affected refresh selected

- `[m1_single_thread_execution_qualification]` Job46328672 COMPLETED0:0 in178s,
  batchMaxRSS118780KiB.22affected engine tests and all36fits/12comparisons PASS.
  Independent result inspection confirms exact phase/grid/seed, required backend,
  finite metric thresholds and protected-access flags false. Maximum gradient
  infinity1.2333060015835216e-14; all repeat objective/coefficient/probability
  differences0. Warm-versus-zero maximum probability difference1.4114464144573654e-13
  and maximum log-probability difference2.785074570965662e-7 (<1e-4). Actual report,
  logs and terminal accounting retained under
  stage0/v3_m1_synthetic_warm_zero_diagnostic_v2_job_46328672/.
- Rendered, inspected and shared QC figure
  outputs/figures/v3_m1_requalification_qc_job_46328672.png, with SVG/source.
  It supports required numerical qualification only; no biological result.
- `[m1_numerical_revalidation]` Selected the existing affected544-row sampled
  and exact-population recovery plus194-row all-K8 numerical-throughput gates.
  New wrapper invokes unchanged model helpers and frozen numeric block, reuses
  the actual46328672 qualification, and omits unchanged79rank sets/36fit rerun.
  Six focused mocked orchestration regressions and shell syntax pass. Root's
  focused source review agrees with exact prescribed operations and sealed roots.
  No scientific optimizer ran on login; new evaluator overhead is measured here.
- Reserve inherited8h/16GiB with frozen1requestedCPU (2allocated conservatively),
  AMD7502/Zen/one thread. Fresh current accounting is
  outputs/stage0/resource_accounting_20261002_m1_affected_refresh_launch.json.
  It releases only unused terminal46328672 reservation and retains historical
  floors, actual usage and both live job ceilings. After the refresh reservation,
  81.111389wall-hours and1026.417111core-hours remain unreserved;13concurrent CPUs,
  32GiB requested memory and at most1GiB new scratch remain inside original caps.
- Required result gates remain exact recovery thresholds, complete12lambda path,
  conservative157path time/cost and25%headroom. SchedulerMaxRSS/currentbudget
  postflight is explicitly pending even if numerical report passes. No real M1/M2
  launch follows automatically. ASTRA program now declares this independently
  runnable required subset and its downstream artifact dependency; validation
  is being refreshed for this actual program change.

- `[m1_numerical_revalidation]` Submitted **46330031**, returned by sbatch;
  recorded at2026-10-02T18:48:26.004478+00:00. Same frozen one-thread AMD7502 backend,
 8h/16GiB, qualification46328672, explicit resource-accounting input.
  Prior jobs46237243 and46325527 continue. No real-data refit submitted.
- ASTRA validator is not exposed as a hosted BR capability in the current
  search result; locating the already-used local pinned validator without
  blocking computation. A filename-only lookup mistakenly used the broader
  outputs root with filename filters; it returned only the already-known
  ASTRA YAML path. No protected artifact content or outcome value was read.
  Further lookup is restricted to explicitly authorized tool/configuration paths.

- `[m1_numerical_revalidation]` Pending46330031 reported3schedulerCPUs for
  the unchanged16GiB/1requestedCPU request. It was cancelled at18:49:12UTC
  before allocation/start: actual0elapsed,0allocatedCPUs,0CPUtime. The prior
  two-CPU reservation is preserved as historical preparation, not charged as
  work. Repair the concrete allocation-accounting guard; numerical thread count
  remains1 and no model/solver rule changes. Two focused allocation/projection
  regressions pass for actual3/4CPU, reject above reservation and stale2CPU,
  and preserve157path wall/headroom rules. Root confirms the scoped change.
- Replacement reserves up to4roundedCPUs for the same8h/16GiB request and
  budgets future production core-hours using that explicit conservative count.
  New current record: outputs/stage0/resource_accounting_20261002_m1_affected_refresh_relaunch.json.
  After all reservations81.111389wall-hours and1010.417111core-hours remain,
  with15concurrent reservedCPUs and32GiB memory. Same episode/reset epoch and
  frozen1requestedCPU/effective1thread backend; no additional authority used.

- `[m1_numerical_revalidation]` Replacement submitted as **46330399** at
  approximately18:51:38UTC; actual sbatch-returned ID recorded. Same numerical
  experiment,8h/16GiB,1requestedCPU and1effective thread; four roundedCPUs now
  conservatively reserved. The cancelled46330031 attempt used zero compute.
  A transient job-ID transcription in the reservation record was corrected
  to46330399 immediately before any downstream use.
- ASTRA official0.0.14 schema passed; semantic validation found an existing
  shell-brace placeholder in the M2 actual-K recipe. Changed `${SCRATCH}` to
  equivalent `$SCRATCH` so the ASTRA command-template parser accepts it, without
  changing the shell command's meaning. Refreshing validation; this export
  repair did not delay or authorize any scientific execution.

- `[m1_numerical_revalidation]` Current ASTRA program passes official pinned
  astra-spec0.0.14 schema and semantic validation at18:52:22UTC:0errors,
 0recommendations. Actual validation artifact:
  outputs/astra/v0.0.14/validation_20261002.json. Existing full experiment program
  is preserved; no lossy single-analysis export was used. Prior semantic
  placeholder defect was repaired separately from scientific outcomes.

## 2026-10-02 19:02 UTC — numerical refresh passes fits, stops at frozen resource gate

- `[m1_numerical_revalidation]` Actual job46330399 ran18:54:09–18:58:26UTC,
  257s,3allocatedCPUs,TotalCPU04:04.191,batchMaxRSS104984KiB. Exit2 is the
  intended resource-gate stop; this is not a numerical-fit failure. The actual
  JSON, runtime log and Slurm logs are preserved under
  outputs/stage0/v3_m1_affected_numerics_refresh_v1_job_46330399/.
- Required544-row recovery PASS: exact-population mean conditional KL
  8.730540159154818e-10bits (maximum0.01), inclusion MAE2.4189973210442507e-6
  (maximum0.02), gamma cosine0.9999999998799624 (minimum0.9). Implementation
  consistency witness passes. Sampled one-hot KL0.3335485051322913bits versus
  descriptive first-order reference0.3460876890367826bits; no population
  threshold is applied to that finite-sample diagnostic. These are QC only.
- Required194-row all-K8 numerical path PASS,12lambdas,187.18363271886483s.
  The frozen157-path projection is8.163286204683827h, exceeding the6.4h
  headroom bound and eight-hour request. No real targets/K/M0/final/unassigned/
  SWC were opened by this refresh. 91-path exact reuse remains report-only.
- Focused follow-up to the existing BR-directed native critic confirms this
  gate is correctly implemented: v3 execution addendum lines120–136 preserves
  157paths,25%headroom and8h. Decision impact: STOP dependent M1 production;
  preserve all passing numerical evidence, do not rerun or silently substitute
  91paths. This mechanical authority decision needs no new biological review.
  Prior BR directive ext_review_dir_ce90de996e4f and verdict
  ext_review_verdict_4fe160ae4283 remain the scientific reuse review.
- Current accounting outputs/stage0/resource_accounting_20261002_m1_refresh_terminal.json
  releases only unused terminal-job reservations and charges3*257/3600 actual
  core-hours. Existing live jobs retain full ceilings. Unreserved89.04wallh and
  1042.202944coreh remain. Historical CPU/wall usage and the sole144h reset
  epoch are preserved. No second reset or ceiling increase.
- Prepared, NOT approved/frozen: outputs/stage0/v3_m1_wall_limit_amendment_proposal_20261002.md.
  It changes only clean M1 production8h→12h, within existing cumulative
  budget, retaining one requestedCPU/effective thread,AMD7502/Zen,16GiB,
  up-to-four allocatedCPU accounting,157paths,headroom,all science/access rules.
  Conditional reservation would leave77.04wallh and994.202944coreh. Asked user
  once through the asynchronous question; no answer assumed. Required source
  of gate: frozen execution addendum production_boundary.requested_wall_time.
- QC figure rendered, inspected and shared:
  outputs/figures/v3_m1_refresh_resource_gate_qc_job_46330399.png (+SVG and source).
  It uses only the actual report and frozen limits, not new experiment values.
  Neurofig skill/renderer unavailable; existing Matplotlib used. Initial plot
  command picked inheritedPython3.13 againstPython3.12modules; corrected to the
  explicitpython3.12 interpreter and rendered successfully. No science changed.
- At19:02:12UTC optional46237243 RUNNING20h42m19s,still3/12completed cases
  (historicallow-rank1/2/3),no final report; diagnostic46325527 RUNNING48m28s,
  phasepermutation_M1_started,zero completed folds,no terminal result. Both
  remain within their preexisting48h/4h limits. No duplicate/replacement jobs.
- Next action: human decision on proposed M1 wall-limit amendment; preserve
  monitoring of already-authorized jobs. If approved, bind fresh evidence and
  truthfully separate historical M0 opening from current-process access, run
  focused runner regression and current resource reconciliation before clean
  M1 execution. No old fitted coefficients may initialize repaired M1.
  Real M2, full-controller nulls and final/unassigned outcomes remain gated.
- ASTRA full program unchanged since official0.0.14 validationPASS18:52:22UTC;
  reuse outputs/astra/v0.0.14/validation_20261002.json. No outstanding export
  failure. Native Goal identity is preserved with its preexistingblocked
  status; available tools cannot resume it. Do not replace or markcomplete.

- Closeout: BR snapshot initially returned `session_not_started` for its existing
  logging run. Logged a transparent late start event (`research_evt_0003`), then
  successfully saved snapshot `research_evt_0004` in the same run/session at
  19:03:55 UTC. Local copy: outputs/stage0/session_snapshot_20261002_m1_resource_gate.json.
  BR's logging-run `succeeded` means snapshot persistence, not episode success
  or scientific acceptance. Its heuristic validation-evidence advisory does
  not change the actual preserved test/qualification results above. No further
  logging audit or optional summary action was launched. ASTRA has no unresolved
  export issue; the M1 resource approval remains the substantive open gate.

## 2026-10-02 19:08 UTC — approved twelve-hour clean M1 request

- `[v3_m1]` The user replied `ok` to the immediately preceding explicit
  twelve-hour amendment request. Recorded that approval narrowly in
  outputs/stage0/v3_m1_wall_limit_authorization_20261002.json and froze
  outputs/stage0/v3_m1_production_resource_amendment_v4.yaml. The historical
  proposal and eight-hour failure remain unchanged. No episode budget, CPU
  ceiling, data role, solver/selection rule, or stopping criterion changed.
- Current resource accounting reserves12h and up to4rounded allocated CPUs,
  retaining1requestedCPU/effective thread and16GiB. After existing live-job
  ceilings and the new reservation,77.04wall-hours and994.202944core-hours
  remain unreserved. Frozen157-path projection8.163286h is below9.6h; measured
  producerMaxRSS104984KiB is below80% of16GiB. See
  outputs/stage0/resource_accounting_20261002_m1_clean_refit_launch.json.
- Runner preparation is limited to newqualification46328672, actual affected
  numerical refresh46330399, the approved request/resource assessment, and the
  truthful distinction between historicalM0opening45760290 and forbidden new
  process access. Existing BR directive/verdict and focused critique are reused;
  no new scientific design choice is being made. Passing ranks, geometry and
  numerical fits are not repeated. All old fitted M1 outputs remain historical.
- At19:08:34UTC authorized optional46237243 and diagnostic46325527 remained
  RUNNING with no new completed case/fold or final report; no duplicate jobs.
- ASTRA program now declares the resource amendment/accounting dependencies
  and12h M1 recipe. Official astra-spec0.0.14 schema/semantic validation PASS
  at19:10:58UTC,zero errors/recommendations. Artifact:
  outputs/astra/v0.0.14/validation_20261002_m1_12h.json. This does not itself
  grant scientific execution authority; the user's narrow amendment does.

- `[v3_m1]` Pre-submission selection: execute the clean real-data M1 refit under
  approved12h amendment,1requestedCPU/16GiB/AMD7502/Zen/one numerical thread.
  Five focused qualification/resource/history tests PASS0.071s and six affected
  checkpoint/CLI tests PASS0.054s; launcher syntax PASS. Root and independent
  focused reviewer found no concrete blocker; no numerical fit gate rerun.
  Scheduler at19:14:14UTC still shows46237243 and46325527 RUNNING. Existing
  reservations remain intact. New run must retain historical M0 opening, keep
  its own M0 access closed, and start in a clean job-scoped output directory.

- `[v3_m1]` Submitted **46332190**, the actual sbatch-returned job ID; recorded
  at 2026-10-02T19:15:05.923423+00:00. This is the clean real-data M1 refit, not a synthetic
  qualification. No old fitted M1 coefficients are reused. Output root is
  $SCRATCH/br_autoresearch/episode10_single_cell_coprojection/
  v3_m1_real_fit_single_thread_v3_job_46332190. Prior authorized jobs continue.

- `[v3_m1]` Job46332190 started19:17:18UTC with3allocatedCPUs. Its thirteen
  launcher-owned orchestration tests PASS0.306s. Backend confirmsAMD7502/Zen/
  one effective thread; qualification/resource gates PASS before membership.
  Actual manifest records membership_loaded_outer_tuning on194cells/12groups,
  original M0 history preserved and current-processM0/final/unassigned/SWCfalse.
  No complete scientific result exists at19:19:39UTC. Runtime logs contain
  mock test events before actual fitting; those are not real completed folds.

- At19:23:44UTC, active monitoring is blocked after two `exec_command` attempts
  failed before shell creation: `No file descriptors available (os error24)`.
  The single retry was separated by one minute. No scheduler query executed
  in either failed attempt, so no later job state is asserted. Last verified
  scheduler state remains19:19:39UTC: all three jobsRUNNING. Jobs are unchanged.
  Saved exact resume handoff in
  outputs/stage0/monitoring_handoff_20261002_clean_M1.md. This is a command-tool
  failure, not evidence of failed science or Slurm outage. Existing ASTRA PASS
  and durable numerical artifacts remain valid; no further export is needed.
