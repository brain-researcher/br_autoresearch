# Experiment log

## 2026-10-12 16:23 Pacific — source_roles — scientist explicitly confirms EP02's procedural separation

At about 16:22 Pacific the scientist (zijiao) wrote "record the EP02 separation
amendment too". This confirms EP02's inclusion directly, in addition to the
13:10 portfolio reply cited below. The amendment itself is already recorded by
the 13:54 entry and `scientist_amendment_20261012` in `SEARCH_POLICY.json`
revision 5, which was checked against the EP15, EP17 and EP18 amendments and
parses as valid JSON. No file other than this log was changed, and no access
flag, threshold, handoff or job changed. Provider semantics, the four raw
margins and the mechanism readout still gate empirical fitting.

## 2026-10-12 13:54 Pacific — source_roles — adopted procedural separation propagated

At about 13:10 Pacific the scientist, zijiao, answered “follow your
recommendation” to a single portfolio evaluation-separation recommendation
explicitly naming EP02, EP15, EP17 and EP18. The exact recommendation and reply
are preserved in the [EP15 decision record](../../episode15_mdtb_topography_vs_geometry/outputs/experiment_log.md#2026-10-12-1310-pacific--scientist-adopts-procedural-same-account-evaluation-separation),
with the same scope confirmed in the [EP17 record](../../episode17_cneuromod_model_ranking_stability/outputs/experiment_log.md#2026-10-12-1310-pacific--source_roles--scientist-adopts-procedural-same-account-separation).
This entry propagates that already adopted decision to EP02; it does not ask
for a new decision. The cited October 3 namespace/ENOSPC failure was EP18 job
46427768, not an EP02 execution. No support request is sent.

Same-account procedural separation replaces permission-enforced candidate,
steward and evaluator separation. Development may use only the separately
released nine-control handoff and permitted outcome-free aggregate structure.
Candidate work must not open, list, search or reacquire the mixed source or
either audit role. Record responsible roles and permitted paths. Freeze
analysis choices and the versioned final configuration before audit access;
log every audit-role access here, disclose unplanned access and mark affected
results. The single evaluator transaction and atomic aggregate-only output
remain; the optional boundary stage cannot change the primary conclusion.
Evaluation results must disclose the weaker procedural boundary and must never
call it enforced or blinded.

Updated only GOAL, DATASETS, SEARCH_POLICY revision 5, README and this log.
The enforced identity, storage-denial and network-denial wording in the legacy
firewall documents, source-semantics note, paper plan and ASTRA is superseded
by the [GOAL amendment](../GOAL.md#adopted-procedural-evaluation-separation--october-12-2026).
Those files and all runners remain unedited. Their role restrictions,
no-reacquisition rule, final lock and aggregate-only evaluator requirements
continue to apply. ASTRA must be synchronized and its affected validation run
in a coordinated compute session before execution relies on the new boundary.

The separation requirement is satisfied procedurally; physical isolation is
unverified. No development handoff or access/scoring flag is released; all
existing access and positive-terminal authorization flags remain false and
all four raw thresholds remain null. Source codebook/time/trigger definitions,
assignment/exclusion support or the already allowed noncausal sensitivity,
mechanism readout, complete empirical implementation, endpoint-specific binding
calibration, final-lock authorization, opening counts and resource limits are
unchanged. Existing passing component/reader checks remain software evidence,
not an empirical result.

Documentation editing and bounded inspection only on sh04-ln04 with no
SLURM_JOB_ID. No empirical payload, role pack, outcome, scientific execution,
test, scheduler action or Git mutation was used for this propagation.

## 2026-10-11 01:26 Pacific — implementation_qualification — development reader selected

Implement a dormant reader for the existing role materializer's
`development_full/MANIFEST.json` and NPY layout. Preserve the exact nine
development identities and expose only `trialID`, `seshID`, `lickState` and
`lick`, with original dtype and axes. Admit the externally provisioned root
before opening its manifest or arrays; validate role and identities before
mapping any array. Use an explicit memory cap, one mouse at a time, readonly
maps and deterministic closure. No mixed-source fallback, audit reader,
endpoint definition, time-axis inference, empirical CLI or self-issued access
grant is selected. A callback is an integration hook, not OS isolation or
evidence that the policy's pre-development conditions have been met.

One worker owns code and focused fixtures, another independently reviews
source; root owns records and any compute. Checks remain unrun until a
coordinated allocation is selected. This source-only implementation does not
open any actual pack or resolve provider codebook/clock/trigger facts,
mechanism readout, enforced role separation or the four binding margins.
Passing observation-component checks are reused; no simulation sweep or
empirical fit is selected.

Select the two reader fixtures and changed ASTRA validation as a serial
companion in EP19's bounded attended integration allocation (1 CPU / 7 GiB /
20 minutes). Root owns the PTY; no actual EP02 source/pack access, repeated
observation tests or additional batch job. The parent is charged once to
EP19; this selects no extra EP02 reservation and does not reset budget history.

Parent47355564 was verified on `sh04-04n22.int`, one CPU and7,168MiB at
08:29UTC. The first fixture invocation failed before loading the tests:
setting `PYTHONPATH=outputs/code` replaced the module-provided NumPy path.
This is a local wrapper error, not a cluster/module fault or reader failure.
Preserve `runs/development_reader_checks_job_47355564/tests.txt`; retry only
these two tests with the module path retained and save `tests_retry.txt`.
Correct the new ASTRA recipe the same way. No scientific data was opened.

The retry passed both tests in0.019seconds. ASTRA schema validation passed,
but semantic validation rejected shell `${PYTHONPATH...}` as a command
placeholder. The test file already inserts its code directory, so remove
that unnecessary override from the ASTRA command. Preserve the first
validator report and revalidate only the corrected program; do not rerun
the passing tests.

At08:30UTC, the corrected ASTRA program passed schema and semantic validation.
Both reader checks are complete; source review also passed. Evidence is in
[`runs/development_reader_checks_job_47355564/`](runs/development_reader_checks_job_47355564/):
`tests_retry.txt`, `astra_validation_retry.json` and `status.txt`, with the
two initial wrapper/configuration failures preserved alongside them. The
parent remains live for EP19 integration and is charged once there. The new
reader grants no data access; codebook/time/trigger, mechanism readout,
presigned margins and enforced role separation remain unresolved.

At01:36Pacific, parent47355564 completed0:0 and its attended shell closed.
The entire438-second allocation is charged once to EP19
(0.121666667 allocated CPU-hours), including this companion; no duplicate
EP02 charge or reservation remains.

## 2026-10-09 — implementation_qualification — required final-nine observation refit implemented

The existing policy requires `all_nine_development_mice_after_recipe_lock`.
The observation fitter previously supplied only the eight-control training
fold. Added a distinct `FinalObservationFit` and
`refit_final_observation_model` in the [existing component](code/logistic_normal_observation.py),
requiring exactly the nine fixed development controls, caller-frozen recipe
and explicit budget callback. The constrained profile algorithm is shared
without changing the eight-control API, likelihood, grid, offset bounds or
selection rule. Callback failure aborts without a partial fit, including when
raised inside the root solver. Final fits cannot enter the LOSO prediction
helper. Conditional plug-in uncertainty limits remain unchanged.

Three new [focused checks](tests/test_logistic_normal_observation.py) cover a
known nine-control optimum and preserved fold result, exact roster/type
separation and mid-root resource abort. The six prior checks passed for the
old source; all nine and changed ASTRA validation are **unrun** for this
revision. Independent source review found no actionable defect in the shared
statistical algorithm, nine/eight-control separation, resource abort or new
fixtures. No source payload, empirical
fit, new scientific choice, recipe lock or reader release was created.

Select these nine checks (120 seconds) and pinned ASTRA validation (30 seconds)
as a sequential companion only after the live EP20 primary process exits in
parent **47123089**, and only if at least 180 seconds remain. Root alone owns
that compute PTY and execution. Do not run concurrently with its one-CPU
primary or submit a separate short job. The parent stays charged once to
EP20 within its existing two-hour reservation; EP02 gains no extra reservation
or duplicate charge. The future evidence directory is
`runs/final_refit_checks_job_47123089/`; no execution is implied by selection.

Actual empirical work still needs authenticated codebook/time/trigger facts,
mechanism-to-session readout, enforced data-role separation and binding
margins. The final refit API becomes scientifically usable only after the
already-required development search and legitimate recipe lock. It does not
replace those unresolved dependencies or activate a new study.

At12:34Pacific, after the EP20 primary exited, root reverified
`sh04-04n22.int`, parent47123089, one CPU and7,168MiB in the same live PTY.
All **nine checks passed in0.787s**; changed ASTRA schema and semantic validation
also passed. Evidence is retained in
[tests](runs/final_refit_checks_job_47123089/tests.txt) and
[ASTRA validation](runs/final_refit_checks_job_47123089/astra_validation.json).
Both process exits were zero. These close the component's unrun status above;
the empirical dependencies remain. Parent resources remain charged once to
EP20, with no separate EP02 allocation or duplicated charge.

## 2026-10-07 05:53 Pacific — implementation_qualification — population observation checks pass

All **six** selected logistic-normal observation checks passed in **0.674 s**
inside physical job **46891123** (EP21 array **46891123_6**) on
`sh04-04n24.int`. The [test output](runs/component_checks_job_46891123/tests.txt)
covers shared-intercept quadrature, exact zero-variance behavior, constrained
training-only population fitting, fixed new-mouse prediction, numerical/fold
refusals and unchanged complete-panel seed/equal-mouse scoring. These are
focused analytic/interface checks, not empirical fitting or complete model
qualification. The [ASTRA result](runs/component_checks_job_46891123/astra_validation.json)
also records schema and semantic **PASS** for the changed program, including
`implementation_qualification.population_observation_checks`.

The [wrapper record](runs/component_checks_job_46891123/wrapper.txt) verifies
the compute host, one CPU, 7 GiB and Python 3.12.1 with NumPy 1.26.3,
SciPy 1.12.0 and scikit-learn 1.5.1. The
[closed component status](runs/component_checks_job_46891123/status.txt)
records runtime, test and ASTRA exits zero and `empirical_execution=false`.
The coordinator observed the EP21 parent running at **12:53:42 UTC**, after
its **05:52:16 Pacific** start. The entire parent remains charged once to
EP21; this component completion is not parent-job completion or a second
EP02 charge.

This result supersedes the then-unrun check/ASTRA statuses in the preparation
entries below. No failure calls for another run. The common mechanism
readout/trial-to-session mapping, authenticated codebook/time/trigger facts,
isolated development reader and four presigned margins remain unfinished.
The accepted article-primary choice and noncausal scope of the 462-label
sensitivity remain unchanged. No empirical payload, intervention outcome,
additional simulation program or new access stage was opened. The next work
remains those source/access and empirical-pipeline dependencies, using these
passing components without treating them as biological evidence.

## 2026-10-07 09:52 UTC — source_semantics_design — official release metadata retrieved

The browser tool again failed to open the public Figshare API. Bounded HTTPS
requests from the coordinator then successfully read its current article,
published-version list and version1 metadata, each capped at twenty seconds
and one MiB. These are small public JSON records, not scientific computation
or payload downloads. The [article record](https://api.figshare.com/v2/articles/21816054)
lists one attachment, `seshMerge.mat` (file38710665; 271,014,539 bytes), and the
[version list](https://api.figshare.com/v2/articles/21816054/versions) contains
only version1. Its description and empty custom fields supply no release
codebook, sample-time vector, online-cap implementation or exporter link.

The earlier unverified-attachment state is now resolved for this specific
record: no separate documentation file or later published version is listed.
This is not proof of absence elsewhere, nor authentication of `trialID`,
`seshID`, `lickState`, timing or trigger/cap semantics. Updated DATASETS and
the existing source handoff point to the direct primary records. The exact
provider request remains the next useful source action and remains unsent.
No MAT, empirical Figure-1 source, outcome or new dataset was opened; no stage
or simulation was activated. Existing observation-component checks remain
selected and unrun in the future shared allocation.

## 2026-10-07 09:48 UTC — implementation_qualification — shared remainder-worker checks selected

Following coordinator and independent static review with no blocking source
defect, select the existing six logistic-normal observation checks and changed
ASTRA validation for the prospective EP21 **work_0007 / array index 6**
continuation, via its `full_remainders_001.sbatch` wrapper. Tests receive a
**120-second** limit and ASTRA **30 seconds**, within an outer **540-second**
serial child shared with separately scoped EP15 and EP19 work. These replace
the earlier provisional test-time envelope; they do not select additional
tests or a simulation/recovery program.

The parent allocation retains **1 CPU / 7 GiB / zero GPUs**, with numerical
thread limits one. Charge the whole parent once to **EP21**; this preparation
adds no separate job or resource reservation. The coordinator owns submission,
scheduler observations, logs and any diagnosed repair. At this selection,
all six checks and ASTRA validation remain **unrun** and no job ID or result
is inferred. No payload read, source reader release, empirical fit, margin
adoption or access approval follows from this selection.

## 2026-10-07 09:43 UTC — implementation_qualification — prospective population observation recipe implemented

The coordinator selected the [common logistic-normal mouse observation
recipe](paper_plan.md#october-7-selected-prospective-population-observation-recipe)
under the scientist's delegated judgment, before empirical outcome access.
This explicitly instantiates the previously unspecified pooling law within
the existing session-binomial/training-only/population-predictive grammar.
No policy fields, margins, model families, endpoint or access roles changed.
The BR advisory rubric `ext_review_dir_70ff8d44da21` (09:37:57 UTC) supplied
review criteria, not artifact review, a verdict or canonical acceptance.

New [fitter and predictor](code/logistic_normal_observation.py) fit one common
offset and a caller-frozen SD grid using exactly eight development training
mice. They integrate one new-mouse Gaussian intercept jointly across all
sessions, preserve exact zero-SD behavior, and retain zero-information sessions
with positive total mouse support. The constrained profile and offset-boundary
diagnostic are returned. Fixed mechanism logits cannot carry a simultaneously
free second observation intercept. Population hyperparameters are plug-in
estimates; this does not supply posterior parameter uncertainty or complete
mechanism/readout identification.

Bounded Brent solves and fixed increasing Gauss–Hermite orders have explicit
resource/iteration/tolerance failures. A failed grid point refuses the fit.
Prediction construction accepts no held outcome; scoring never refits or
changes integration orders using outcomes. The integrated complete-panel
entrypoint preserves the existing five-seed mean and equal-mouse ranking and
requires the same observation recipe across candidates, nulls, folds and seeds.

Six [focused mathematical checks](tests/test_logistic_normal_observation.py)
and changed ASTRA validation are **prepared, unrun**. The tests use analytic
count examples and independent scalar quadrature, not simulated cohorts or
empirical records. Proposed coordinator-owned check inside the next suitable
substantive allocation: **1 CPU, 2 GiB estimated memory, 60-second timeout,
zero GPUs**, all numerical thread limits one. This is an estimate for these
fixtures, not empirical fitting throughput. No standalone job is requested;
charge shared parent compute once to its assigned episode.

```sh
timeout --signal=TERM --kill-after=5s 60s python3.12 -B -m unittest discover -s outputs/tests -p test_logistic_normal_observation.py -v
```

The task remained on verified login shell sh04-ln04 without an allocation;
only bounded source inspection and editing occurred. No imports, tests,
scientific computation, source payload reads, scheduler action or simulation
sweep occurred. The actual common readout/trial-to-session map, codebook/time/
trigger definitions, isolated development reader and four presigned margins
remain separate unfinished dependencies. Existing passing components and
historical records are preserved.

## 2026-10-07 — implementation_qualification — grouped binomial optimizer checks pass

The coordinator-owned attended session **46845646** ran all six
[grouped-binomial MAP checks](runs/shared_integration_46845646/tests_initial.txt)
in **1.893 s**, with every test passing. The
[status artifact](runs/shared_integration_46845646/status.txt) identifies
`sh04-18n32.int`, one allocated CPU, 7168 MiB, `initial_checks_exit=0` and
`astra_exit=0`. These are observed component and ASTRA results, not a completed
empirical fit. Shared allocation cost is assigned once to EP18; exact terminal
accounting remains with the coordinator and is not duplicated in EP02.

The tests cover known stationary solutions, coupled fixed designs/offsets and
penalties, exact session denominators and splitting equivalence, invalid
fold/count/penalty refusals, fixed held-mouse prediction and explicit failure
on iteration exhaustion. Together with independent static review, this supports
numerical-component readiness under the supplied-input contract. No failure
requires a repair or additional test run at this point.

The optimizer does not select an observation design or shrinkage setting,
provide an integrated population-predictive distribution, authenticate the
provider's codebook/time/trigger facts, or deploy the isolated development
reader. Those requirements and the remaining shared state/readout and
trial-to-session mapping are unchanged. No source payload, outcome or recovery
cohort was used. The preparation record below is preserved as history; this
entry supersedes its then-unrun test status only. This documentation update
made no code, ASTRA, policy, scheduler or test changes.

## 2026-10-06 18:55 Pacific — implementation_qualification aggregated attended implementation session selected

Select the prepared grouped binomial optimizer checks and changed ASTRA validation within
one coordinator-owned attended session aggregating EP02/15/18/20 integration,
diagnosis and any required focused repairs. Request **one CPU / 7 GiB /
30 minutes / zero GPUs** in the discovered development partition. Reuse the
qualified EP18 runtime and run each episode serially with all thread limits
one. The coordinator owns the shell; no independent agent submits or runs tests.
This replaces the provisional wait for a future EP21 batch without touching
those occupied workers. It is one combined implementation session, not four
short scheduler submissions or a simulation sweep.

Reserve at most **1,800 allocated CPU-seconds once to EP18**, then replace that
reservation with the actual terminal allocation charge; no duplicate cost is
assigned to EP02/15/20. Existing consumed budgets and results remain intact.
No empirical payloads, reader-release or scientific recipe change is selected.
Verify the actual compute shell before commands, preserve initial failures,
and exit the attended shell as soon as selected work is finished, without
padding runtime or adding an automatic successor.

## 2026-10-07 — implementation_qualification — grouped binomial optimizer prepared

Continued the authorized empirical-pipeline engineering with an array-only
[quadratic-MAP solver](code/grouped_binomial_map.py) and
[six focused analytic tests](tests/test_grouped_binomial_map.py). The previous
development scorer evaluates supplied logits but does not fit them. The new
component fits a caller-fixed binomial design/offset with an explicit SPD
penalty, prior center and convergence settings. It preserves session counts
and enforces exactly the eight training mice of the declared development fold.
The new-mouse helper has no outcome input and returns explicitly conditional
plug-in logits, not an integrated predictive distribution.

Contract basis: `SEARCH_POLICY.json:365–370` supplies the binomial family and
training-only/predictive-only limits; `GOAL.md:103–111` supplies the development
identities and biological unit. No intercept/random-intercept structure or
shrinkage strength is specified there, so none is selected here. Numerical
optimizer implementation is not adoption of an empirical observation recipe.
See the [method scope and remaining definitions](source_semantics_design.md#october-7--generic-grouped-binomial-optimizer-prepared).

Tests and changed ASTRA validation are **unrun** in this editing scope. Proposed
coordinator-owned compute check, from the episode directory, with one CPU and
BLAS/OpenMP threads fixed to one:

```sh
timeout --signal=TERM --kill-after=5s 60s python3.12 -B -m unittest discover -s outputs/tests -p test_grouped_binomial_map.py -v
```

This is six tiny analytic checks, estimated below 60 seconds and 2 GiB; it can
share the next substantive owned allocation. No empirical reader, source/outcome
access, synthetic cohort, recovery program, scheduler action or computation
was performed on login node sh04-ln04. No policy, margin, access or budget changed.
Production observation structure, new-mouse predictive integration, the shared
state/readout and session mapping, authentic codebook/time/trigger semantics
and existing role isolation remain the next substantive dependencies.

## 2026-10-07 — implementation_qualification — adopted article component checks pass

Within numeric job **46839671** (array 46839670, task 0), the EP02 subprocess
ran only the existing ACTR component test module: **11 tests passed in 0.051 s**, including the three
new explicit-trigger article-multiplier checks and the retained synthetic/
pinned-variant checks. See the
[test output](runs/component_checks_job_46839671/tests.txt),
[status artifact](runs/component_checks_job_46839671/status.txt) and
[ASTRA validation](runs/component_checks_job_46839671/astra_validation.json).
The status records `checks_exit=0`, `astra_exit=0` and parent resource cost
charged once to EP21; schema and semantic validation both passed.

These results validate the tested arithmetic and preserve separate 25-ms
article versus 30-ms pinned-code behavior. They do not authenticate the online
750-ms trigger/cap, trial/session/lick codes or sample-time axis, qualify a
complete empirical fitter, adopt rival operators or the four margins, or open
an outcome. No synthetic cohort or recovery sweep was launched for this check.
The parent fitting batch's completion is not inferred from this subprocess
success. This entry records exact existing artifacts; no new implementation,
test, scheduler action or payload access was performed by this documentation
pass. Source and complete-fitter dependencies remain as recorded below.

## 2026-10-06 — implementation_qualification — adopted article multiplier separated from synthetic trigger

Continued the explicitly authorized EP02 engineering work under the adopted
article/task-primary choice. The existing article simulator's stimulation
helper used a documented first-eligible 50% cap scenario, which cannot stand
in for the unresolved online implementation. Added
`actr_port.article_stimulation_multiplier`: it accepts the current source
anchor's beta and an explicit resolved Boolean trigger, doubles beta only
when triggered and otherwise preserves it. It has no performance-error,
arm-code, lick-state, trial-order or cap fallback. Caller provenance remains
an external requirement; the function cannot authenticate a supplied Boolean.

The existing synthetic helper delegates only its final article multiplication
to this function, retaining its historical scenario and the pinned-code rule.
No scientific trigger convention, parameter search, model family, access role,
margin or budget changed. Existing article dynamics remain 25 ms; pinned-code
dynamics remain 30 ms with their separate set-four sensitivity interpretation.

Added three focused analytic checks to `tests/test_actr_port.py`, making eleven
in that module: accepted multiplication versus set-four, refusal of missing
or non-Boolean trigger inputs, and source-domain/overflow behavior. These are
prepared and **unrun**. The coordinator can run only that unittest module in
compute, without the `qualify_actr_port.py` synthetic-fixture driver, plus the
changed ASTRA validation. No test, import, simulation, scheduler action,
payload or outcome read occurred in this editing scope on sh04-ln04.

The remaining complete-fitter definitions are concrete: shared state/readout
and trial-to-session probability mapping, training-only hierarchical or
partial-pooling specification, and the still-unadopted rival production
operators/configurations. Provider trial/session/lick codes, the time axis and
actual trigger/cap also remain required. No generic fitter or extra
qualification sweep is invented to conceal those dependencies.

## 2026-10-06 Pacific — scientist adopts article primary and pinned-code sensitivity

Authority: the scientist replied “i agree with your judgement” after the
recommendation to use authenticated task/article primary and pinned code as
separately labeled sensitivity. Updated the policy's variant resolutions and
current GOAL/data/design/navigation prose accordingly. Primary uses article
25-ms dynamics and doubles current beta_DA under the authenticated task trigger;
the 30-ms, set-four pinned-code version stays report-only, without selection,
rescue or promotion. Existing source-difference history is retained.

This agreement resolves the model choice only. Provider trigger/cap, codebook,
sample-time and assignment facts remain unresolved; optional rival operators
and all four raw margins remain unadopted. Role isolation, full fitting and
applicable qualification remain our implementation work. No payload, outcome,
test, simulation, provider contact, scheduler action or resource change occurred.

## 2026-10-06 Pacific — verified public-source lookup recorded

At the scientist's request, added the completed bounded web lookup to the
[existing source handoff](source_semantics_design.md#public-source-check--october-6-2026-pacific),
with exact official URLs and found-versus-unverified distinctions. The author
repositories are accessible implementation references; no release-specific
field codebook, 701-sample origin or online-cap definition was found there.
The Figshare DOI/API retrieval failed in the browser tool, so no claim is made
that current documentation or a correction is absent. Further useful work is
the release-linked metadata/codebook lookup or the existing unsent provider
request, not another numerical qualification run.

This documentation update reuses that lookup without a new search. No payload,
protected outcome, test, code execution, scheduler action, contact, policy,
role, budget or scientific adoption changed. Earlier records remain intact.

## 2026-10-06 Pacific — source-first design revision; no activation

Revised the existing `source_semantics_design.md` and GOAL to distinguish
unresolved provider field meanings from model implementation. The raw time
axis independently blocks the required replication; the noncausal 462-label
fallback was already permitted. The provider request is drafted but unsent.
Article-primary/author-code-report-only sensitivity remains unadopted; the
four grid-derived margin proposals are explicitly not endorsed on their
present rationale. Active policy, thresholds, roles, budget and access are
unchanged. No payload, test, simulation, scheduler action or job was run.

> Local projection only; it grants no execution, audit, review, reward, or
> scientific authority.

## 2026-10-06 05:26 UTC — implementation_qualification: aggregate checks selected

Root selects the eleven new rival-interface checks and changed ASTRA
validation inside the attended 1 CPU / 2 GiB / maximum 30-minute allocation
whose substantive task is EP06's fixed source-metadata inventory. The parent
allocation is charged once to EP06; no duplicate EP02 charge or scientific
trial. Root owns scheduling and the compute shell. Only EP02 code and its
small algebra fixtures are inputs, with all six thread caps one; no empirical
source, sibling outcome, production model choice or search is activated.
The submitted allocation is **46721202**. Root observed execution on
`sh04-18n32` during05:27–05:28UTC: all11 tests passed in0.066s, recorded in
[`runs/rival_model_checks001/tests.txt`](runs/rival_model_checks001/tests.txt).
Official ASTRA schema and semantic validation exited0 in
[`validation_20261006_46721202.json`](astra/v0.0.14/validation_20261006_46721202.json).
Previously passing components were not rerun. Parent **46721202 completed
0:0**, from05:26:56 to05:33:09UTC:373 seconds at1CPU, or **0.1036111111CPUh**,
with MaxRSS343,768KiB. The entire cost is charged once to EP06's source-check
allowance; its reservation is released, with zero additional EP02 charge or
duplicate share. No compute session remains active. These are arithmetic checks of unadopted
interfaces, not production selection, model qualification or empirical evidence.

## 2026-10-06 — implementation_qualification: concrete rival interfaces prepared

Implemented `code/rival_model_core.py` and
`tests/test_rival_model_core.py` within the existing finite seven-family
grammar. The existing [source handoff](source_semantics_design.md#october6-outcome-blind-rival-interfaces--implemented-unadopted)
records the exact unadopted low-parameter equations and still-required
scientific selections: shared state/action basis and behavioral readout,
separate performance-error/rate/bias updates, canonical SARSA/max-Q targets,
and direct reinforcement of the supplied preceding detected action without
neural learning. All source meanings, event triggers and numerical parameters
are caller inputs. There is no empirical adapter, data access, model fitting,
simulation sweep, search activation or policy/margin amendment.

Independent source-only review of the operators and all11 hand-computable
tests found no actionable numerical or scientific-interface defect; no code
was executed during review. Runtime checks subsequently passed as recorded
above. The
previously passing ACTR, pinned-critic and development-score components are
unchanged and their checks are reused. Selected coordinator-owned aggregate
compute verification was, from this
episode, `python3.12 -B -m unittest discover -s outputs/tests -p
test_rival_model_core.py -v`, bounded to60s plus5s termination with one CPU and
thread caps. No standalone short job is submitted. Current optional model
equations are not a qualified production rival panel; source semantics,
scientist-selected model choices, binding calibration/margins and physical
reader separation still gate empirical scoring.

## 2026-10-05 00:33 UTC — pinned critic checks pass

All nine focused source-equation tests pass in0.034s in root-owned
allocation46535708 on sh03-01n68, after its MV fitting process finished.
Evidence: `runs/pinned_critic_checks001/tests.txt`. The changed ASTRA schema
and semantics validate in `astra/v0.0.14/validation_20261005_score_and_critic.json`.
Independent static review also passed against the pinned MATLAB source.
These checks are grouped and charged once with EP01's attended allocation.

The critic remains an author-parity arithmetic component. Production rival
representation/readout definitions, empirical source meanings, approved
binding calibration/margins and physical reader separation remain incomplete.
No source outcome, adaptive trial or scientific gate was opened by this work.

## 2026-10-05 00:12 UTC — missing pinned-critic arithmetic selected

The allowed MIT author source `dlRNN_criticEngine.m` at commit
3627746957975632cddb05bf8f6b26a4b6f901b6 defines a missing numeric component.
Root selects its exact array-only port, `code/pinned_critic_core.py`, and focused
independent fixtures in `tests/test_pinned_critic_core.py`. The worker owns only
those files. This is author-code parity, not a complete on-/off-policy rival,
behavioral readout, new task simulator or empirical endpoint.

Prospective conventions: supplied finite binary64 weights, basis[feature,time]
and reward[time], explicit nonnegative alpha, gamma/lambda in[0,1], and source
stimulation condition; no task defaults. Reset eligibility at each call and
process exactly T-1 transitions. Compute both values from the current
pre-update weights, delta=r_eff+gamma*(V_next-V_current),
eligibility=gamma*lambda*eligibility+alpha*x_current, then
weights+=delta*eligibility. The source condition>=20 multiplies supplied reward
by10. Preserve the unused final reward and nonzero final bootstrap; do not
invent a terminal update, clip an overflow or mutate caller arrays.

This delta differs from canonical r+gamma*V_next-V_current when gamma!=1.
The adjacent permitted source caller initializes gamma=lambda=1 but does not
call the critic engine, so it cannot establish an action/readout mapping.
BR read-only KG query `a21acb52` for these distinctions returned unrelated
behavioral-addiction paths; it supplies no usable scientific evidence and has
no decision impact. No BR verdict or canonical mutation occurred.

Tests will run only in a coordinator-owned allocation after independent review,
grouped with substantive work and within its remaining time. Current46535708
is occupied by MV; no concurrent test uses its single CPU. Provider meanings,
binding calibration/margins and role separation remain unchanged blockers.

The component and nine focused fixtures are now implemented. Independent
review against the exact24-line MATLAB source passes, including sequential
pre-update values, eligibility reset, source value overwrites and the unused
final reward. Root static review also passes. Compute execution remains pending
until MV releases allocation46535708's CPU; no standalone short test job is
submitted. Selected test command: `python3.12 -B -m unittest discover -s
outputs/tests -p test_pinned_critic_core.py -v`, at most60s plus5s termination,
followed by changed ASTRA validation within the allocation's remaining time.

## 2026-10-05 00:10 UTC — development-score arithmetic checked

Independent static review passes. All12 focused tests pass in0.007s under
Python3.12.1 in root-owned allocation46535708 on sh03-01n68. Evidence:
`runs/development_score_core_checks001/tests.txt`. Changed ASTRA schema and
semantic validation also pass in `astra/v0.0.14/validation_20261005_score_core.json`.
The brief grouped checks are included once in EP01's attended allocation cost.
This completes this arithmetic component only; model fitting, endpoint mapping,
provider definitions, scientist-approved margins and physical reader separation
remain incomplete. No empirical payload, development score or trial is opened.

## 2026-10-05 00:04 UTC — required development-score arithmetic selected

A scoped implementation review confirms that the existing ACTR components and
endpoint-shaped calibration are not the required session-binomial LOSO scorer.
The current goal explicitly delegates outcome-blind implementation before
source semantics/calibration are complete (GOAL's stage-specific rules).
Root selects the source-free production arithmetic in
`code/development_score_core.py` and focused tests, with one worker owning
those two files. This adds no simulation search, source reader, endpoint
definition, trigger default, empirical threshold or candidate run.

Prospective conventions: include the binomial log-combination term and use
stable log-sigmoid arithmetic on supplied prediction logits. At exact p=0/1,
compatible counts have log mass0 and incompatible counts have negative-infinite
log mass; skip zero-count logarithm terms, never clip probabilities or add
pseudocounts. Higher fold/ranking arithmetic refuses undefined/nonfinite scores
without dropping any observation or interpreting that refusal as a biological
finding. Counts and unique session identities must come from a future verified
adapter; the core cannot establish their endpoint meaning.

For every held mouse and seed, sum session logPMFs and divide by that mouse's
total eligible trials. Average exactly five explicit shared seed scores within
mouse before subtracting the constant-rate null; rank equally across exactly
the nine frozen development source-record IDs by minimum, median, mean gain,
negative pre-frozen complexity, then lowest configuration ID. Preserve session
order, fixed record order and sorted explicit seed IDs; use float64/math.fsum
without an invented epsilon tie. Each declared training roster must contain
the other eight mice, but array annotations do not prove fit isolation.

The unchanged empirical blockers remain: provider codebook/axis/trigger and cap
semantics, scientist-approved binding rule/four margins, and real reader
separation. The contract's noncausal462-label sensitivity is already permitted
when randomization records are missing; it is not design-exact or an ITT claim.
No extra permission gate is invented for these numerical implementation choices.

Root may run the focused new-core checks and changed ASTRA validation in the
already-requested EP01 allocation46535708, at most60s+5s termination per check,
one thread/at most2GiB, after static review. It remains pending at selection;
no login-node test, separate EP02 compute request or observed-data access occurs.


## 2026-10-03 — `source_semantics_design` and `implementation_qualification`: authorized outcome-blind preparation

The scientist authorized resuming possible work across episodes. This stage
opens documentary sources and the pinned MIT author simulator only, never the
mixed MAT, empirical Figure-1 script, role packs or intervention outcomes.
The existing contract leaves trial/session/trigger/time-axis definitions,
assignment scope, runtime role separation and four raw thresholds unresolved;
no development scoring, configuration lock or audit is selected.

Selected preparation: preserve the article and code simulation variants
separately (25 versus30 ms; contingent multiplier2 versus code assignment4),
implement outcome-blind simulator components, and qualify only the necessary
equations on compute. The exact pinned simulator remains an oracle, not an
authoritative online lick-trigger codebook. Record concrete missing source
definitions and recommended primary/sensitivity handling; proposed thresholds
remain unactivated until endpoint-faithful independent calibration and the
existing scientist approval. No synthetic result can self-bind the policy.

The source handoff is `outputs/source_semantics_design.md`. It recommends a
later article/task primary and pinned-code parity sensitivity, while preserving
both explicit variants now. It proposes—but does not activate—raw thresholds
`0.15`, `0.05`, `0.06`, `0.00` in canonical practical/arm/scale/LOO order.
They are selected from the existing provisional grid and require independent
endpoint-faithful calibration and scientist review, not empirical tuning.

Retrieved only the MIT-pinned simulator's training, recurrent engine,
behavioral plant and comparison source plus license into
`outputs/code/author_oracle/`. The separately pinned TONIC Gaussian was read as
a mathematical oracle; its source was not copied. The implemented helper keeps
the actual `4*sigma^2` denominator and sample-index trapezoidal normalization.

Selected compute command is `outputs/code/qualify_actr_port.py`: eight focused
source-equation/variant tests followed by six four-trial,50-unit synthetic
fixtures (two variants times three calibrated/control arms), one numeric CPU,
8GiB and at most10 minutes in a coordinator-owned allocation. This exercises
recurrent dynamics, eligibility orientation, the behavioral plant, source
modulation branches and learning updates. It does not count as any adaptive
trial or full model recovery. The replacement initializer, short input pulses,
and article first-eligible cap scenario remain explicitly provisional. No
independent-equation qualification or800-trial production parity is inferred.
Actual execution results are pending; this entry precedes the process effect.

Actual attempt001 in compute allocation `46424502`, step8: all eight focused
tests passed in0.051s and the six synthetic loops returned, but terminal JSON
serialization failed on a NumPy `bool_` stimulation flag. No completed
`qualification.json` was produced. The original attempt directory is retained.
The source stimulation function now converts its returned trigger to a plain
Python bool; branch equations and random draws are unchanged. The existing
branch test now covers NumPy-valued errors and checks the returned bool type.
Selected retry uses a fresh `actr_components_job_46424502_attempt002` directory
under the same bounded attended allocation, with the same eight tests and six
fixtures. This is an engineering retry, not another adaptive trial.

Actual attempt002 in the same allocation: all eight tests passed in0.055s and
all six synthetic fixtures completed with a saved
`synthetic_component_qualification_pass` report. The coordinator is preserving
the exact output directory. Production qualification and audit authorization
remain false. ASTRA validation was invoked; its actual report still awaits
coordinator verification. A fresh worker is independently authoring the
equation checker from `outputs/code/actr_checker_interface.md` and pinned
sources only, without access to the main port, tests or expected fixture values.

The coordinator preserved the completed report without overwrite at
`outputs/runs/implementation_qualification/job_46424502_attempt002/qualification.json`.
The official ASTRA0.0.14 schema and semantic validator also **PASS**; actual
validator output is `outputs/astra/v0.0.14/validation_20261003.json`.
Independent equation comparison remains pending; full implementation
qualification and empirical access are not inferred from these passes.

Source-handoff preparation also removed the unsupported Sherlock
`--account=russpold` directive from the four existing inventory, materialization
and calibration wrappers; original CPU/memory/time limits and commands remain.
No wrapper was submitted and no legacy receipt helper was promoted to an
active gate. This mechanical repair does not provision candidate/steward
separation or resolve endpoint semantics.

Independent worker `inventory_ep15_21` authored and froze
`outputs/code/independent_actr_checker.py` from the pinned source equations and
interface only, without reading primary code, tests or outcomes. It uses pure
standard-library scalar loops and supplies39 independently chosen cases:
3Gaussian,4recurrent,4learning,24stimulation,4plant. Selected next effect is
`outputs/code/compare_actr_implementations.py` in the coordinator's allocation,
using those inputs for both implementations. A component comparison tolerance
of absolute1e-12 plus relative1e-12 is declared prospectively; this does not set
an empirical/calibration margin or production score-replay tolerance. The
primary author wrote the adapter only after independent reference authorship.
Both ports remain partial preparation even if the component comparison passes.

Actual first independent replay in allocation `46425841` at
2026-10-03T18:47:07.355431Z failed on six of39 cases. The preserved report is
`outputs/runs/implementation_qualification/job_46425841_independent/independent_replay.json`.
All33 source-domain comparisons passed (maximum absolute difference
3.122502256758253e-17). The six failures are original stimulation inputs with
base multiplier0.4, rejected by the main source-domain guard. Source trainer
line50 constructs a nonnegative cumulative Gaussian DA response and line259
adds1 to two response entries; thus these six inputs exercise a raw algebraic
extension outside the source-derived model. The initial interface omitted
this domain distinction. The independent author separately confirmed it from
source, without editing their original cases or reading main implementation.

Selected repair: clarify the interface, preserve the original39 fixtures and
failed report, retain the main beta>=1 guard and nonnegative learning guard,
and classify the six source-domain rejections explicitly rather than counting
them as numerical agreement. The independent author will append a separate
six-case source-valid boundary suite. The selected retry is the same comparison
entrypoint with a fresh `actr_independent_job_46425841_attempt002` scratch
directory, one numeric CPU and2GiB within the coordinator's attended session.
It will report numerical comparisons and source-domain rejections separately.
No scientific model, threshold, empirical access rule or independent expected
value changes. The pending ASTRA validation records the added independent
comparison recipe separately from the already-passing first program version.

Actual corrected independent replay in allocation `46425841` **PASS**:
39 numerical-equation comparisons (33 original source-domain cases plus six
independently added boundary cases), and six explicitly reported
out-of-source-domain rejections,45 cases total. The first39-case failure and
all original fixtures remain preserved; neither source guard nor reference
equation was weakened. The actual result is
`outputs/runs/implementation_qualification/job_46425841_independent_attempt002/independent_replay.json`.
Official ASTRA schema and semantic validation also **PASS**, saved at
`outputs/astra/v0.0.14/validation_20261003_job46425841.json`.
This closes partial synthetic component preparation only. Production model
qualification, source task/callback authentication, role separation, binding
calibration and empirical execution remain incomplete; zero adaptive trials
and zero audit openings have occurred.

## 2026-09-22 — Specification frozen

- Selected the Dudman learning-rate question and the 9/6/5/4 evidence roles.
- Defined a whole-mouse primary contrast and a one-opening audit boundary.
- Recorded two legacy structural diagnostics as exposed and permanently
  excluded them from scientific decisions.
- Added a reusable outcome-blind endpoint-calibration engine and tests.
- Recorded that implementation existed but scientific audit opening remained
  blocked.

No adaptive trial or audit has occurred. Runtime inventories, calibration
packets, receipts, signatures, and checksums are intentionally not tracked.

## 2026-09-23 — Portable pre-outcome implementation migrated

- Added the sanitized, outcome-free cohort-map contract.
- Added portable Phase-0 inventory, deterministic role materialization,
  generic calibration, provisional endpoint-shaped calibration, and external
  firewall verification tools with job-scoped `$SCRATCH` launchers.
- Hardened the firewall verifier to require distinct Ed25519 signer identities
  and keys, complete hash bindings, and a bounded valid time interval.
- Made calibration prerequisite inspection explicitly nonbinding without
  trusted cryptographic authorities and a separate signed policy amendment.
- Kept every runtime packet, role pack, assessment, signature, UID, and job
  record outside the tracked specification.

This was an engineering/specification migration only. It did not read an
audit outcome, run adaptive search, or authorize candidate scoring or audit
access.

## 2026-09-26 — Readiness machinery simplified

- Removed generic checksum, strict-schema, manifest-chain, signed-receipt, and
  attestation requirements from the active policy and access contract.
- Retained the release/index cohort identity, mouse-level independence,
  development/audit separation, endpoint and randomization gaps, one-shot
  aggregate evaluator, and versioned write-once final lock.
- Kept the older hash/receipt implementation files as nonbinding history; they
  are no longer required for qualification, development work, or audit
  authorization.

No outcome was inspected, and the earlier dated log entries were left intact.

## 2026-09-28 — Primary-estimand units reconciled

- Made the raw between-arm probability-gain contrast `delta_rate_raw` the
  primary estimand in the Goal, dataset contract, and policy revision 3.
- Reserved `delta_rate_std` for the exact companion obtained by dividing the
  raw contrast by the frozen development-control standard deviation.
- Required explicit raw and standardized fields in the future aggregate audit
  packet and tied every standardized margin to the presigned raw-unit margin.
- Normalized all future binding thresholds to the calibration-compatible names
  `practical_margin_raw`, `arm_margin_raw`, `development_scale_floor_raw`, and
  `loo_delta_floor_raw`; all four remain null and unbound.

This was a pre-outcome contract repair. No development or audit outcome was
opened, no calibration result was adopted, and audit authorization remains
disabled.

## 2026-09-30 — Question-first reanalysis design and conceptual figure

- Reframed the paper around what the closed-loop protocol contrast identifies,
  not rediscovery of the source paper's adaptive-rate mechanism.
- Added `paper_plan.md` and the built-in imagegen schematic
  `ep02_conceptual_question.png` with its final prompt set. Inspected timing,
  lick/no-lick triggers, biological group counts, session cap, gain formula
  and absence of empirical values. Removed ambiguous pre-reward water icons
  and model curves in one targeted revision.
- Preserved the prior raw-unit estimand repair, four null thresholds, exact
  standardized companion, fixed roles, active model grammar, 28–48-trial
  budget and closed outcome access. Policy additions are narrative claim
  limits only, not executable-contract changes.
- Added an ASTRA worklist of existing requirements with no executable
  commands, invented gates, runtime fields or scientific findings.

ASTRA-linked handoff: `source_semantics_design`, `role_separation`,
`implementation_qualification`, `small_n_binding_calibration`,
`development_search`, `evaluator_qualification`, `configuration_lock`,
`primary_audit` and optional `boundary_diagnostic` are all unlaunched; no
attempt or job ID exists. Protected empirical data were not read.

The next scientific action, if separately authorized, is
`source_semantics_design`: resolve endpoint/assignment documentation and
choose the supported causal scope. A formal rival-exclusion test remains a
proposal requiring a pre-outcome scientific decision, not part of the active
audit. No empirical/synthetic experiment execution, Slurm submission, MCP
action or Git mutation occurred; checks were limited to changed documents,
local references and the saved schematic.

## 2026-10-02 — Scope and current-literature review

For `development_search`/`primary_audit` interpretation, retained the existing
finite-panel reanalysis scope and added the 2025 tail-striatal action-prediction
error study as a circuit/task boundary, not a new operative model. The bounded
read-only BR publication search returned neighboring learning-rate studies,
without direct additional release evidence. See [review (consolidated)](paper_plan.md).
No endpoint, threshold, access role, policy or ASTRA stage changed; no data or
experiment was opened. Other status notes remain accurate and unchanged.

## 2026-10-03 21:09 UTC — source_semantics_design documentary blocker

A bounded review of the [provider methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC9908546/),
[Figshare v1 metadata](https://api.figshare.com/v2/articles/21816054/versions/1),
and the two pinned code repositories did not resolve the release-specific
`trialID` codebook, `seshID` training-day mapping, `lickState` predicate, or
701-sample event/time axis. The article supplies task timing and the broad
750-ms stimulation contingency, but its preparatory summary uses one second;
that summary cannot replace the required 750-ms endpoint. The Figshare record
lists only `seshMerge.mat`, without a separate codebook. The inspected TONIC
photostimulation/alignment helpers are generic; its continuous-behavior
extractor is a declaration without an implementation. They do not establish
this release's exporter or online trigger. The article describes custom
Arduino control and video postprocessing code available on request.

The smallest useful next step is the authoritative exporter/online-control
source or provider definitions covering trial codes, training sessions,
axis/alignment/resampling, lick-state boundaries/missingness, plus-arm cap
counter/reset/order, and assignment/exclusion roster. The coordinator has
requested the source/codebook location from the user. No provider contact was
made. Re-running the existing monolithic MAT inventory cannot establish these
meanings and would materialize mixed-role values; it was not run. The empirical
Figure-1 script and all source values remained closed.

The four proposed margins in `source_semantics_design.md` remain inactive.
`SEARCH_POLICY.json` at `small_n_calibration.prerequisite_inventory` explicitly
states `explicit_scientist_approval_required: true`; Goal's binding-rule clause
also requires resolved semantics, calibrated criteria and scientist approval.
This is an explicit scientific requirement, not an inferred approval gate for
routine implementation. Physical steward/candidate/evaluator separation is
independently unresolved. No contract change, new test, scientific run, or
scheduler action occurred; passing component qualifications are reused.
