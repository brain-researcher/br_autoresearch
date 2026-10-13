# Necessary recovery checks for the declared MV and MZ alternatives

Adopted by the coordinator October5,2026 at00:01UTC, after independent
contract/code review and before any MV/MZ fitted recovery score. This advances
the existing policy-v6 alternatives; it does not amend the question, select a
new follow-up, erase the failed MC cell or unlock observed behavior/BOLD.

The original full-family recovery conjunction cannot pass:25 valid MC-at-SESOI
failures leave at most175/200 successes. The goal nevertheless explicitly
permits a narrower single-process result and qualifying group-only inference.
The execution log preserves those unevaluated alternatives. Their necessary
recovery checks can therefore proceed without claiming that the full-family
route has recovered or removing MC from future scientifically relevant rivals.

## Two prospectively fixed cells

| Generating family | Focal values for every participant | Seed for index i |
| --- | --- | --- |
| MV | eta=log(1.15), z=0.5, c=0 |810031+10007*i|
| MZ | eta=0, z=0.45, rho=log(0.45/0.55), c=0 |910031+10007*i|

Each cell retains the existing target of200 fixed indices,0–199. There is no
seed search or replacement of a valid failure. Only representative index0 of
each cell is selected initially. Measured convergence and cost determine
whether/how to schedule the remaining fixed indices within the same contract.

For each participant independently, draw kappa, log_a and nondecision_logit
from the existing implementation's truncated normals using its population
centers, individual scale centers and bounds. Preserve the2% contaminant
channel, its1% miss probability, and model-generated four-second deadline
censoring. Use the frozen deadline likelihood, priors, optimizer and numeric
support unchanged. Constant focal truth defines a uniform-SESOI necessary
corner of the recovery grid; it cannot establish individual rank/RMSE or
robustness to heterogeneous focal effects.

Reuse only the design columns from the completed, preserved MC attempt003
`runs/behavioral_mechanism_kill_test/20261003-MC-SESOI-replicate000-attempt003/simulated_events.csv.gz`:
participant, task_version, run, trial_order, onset, gain and loss. Its simulation
retained the qualified original offers and trial identities. Project that
explicit allowlist and generate all response fields afresh; do not consume MC
scores or fitted parameters, or reopen the observed response table. Validate
108 participants (54 EI/54 ER),432 runs and27,648 distinct original offer slots,
four runs and64 ordered trials per participant/run, with no dropped/replaced
row. The copied initial onset-sensitive schedule IDs are not the qualified
eight-sequence identities and must not be used for later sequence sensitivity.

## Necessary comparison and limits

Fit M0 and the generating focal model separately in each primary direction:
training{1,4}→scoring{2,3}, then training{2,3}→scoring{1,4}. Every subject and
population parameter comes from its training half. Keep all108 participants
and256 trials/person across the two held-out halves, including deadline misses.
Compute joint choice/RT/censoring log predictive density with the unchanged
plugin predictive implementation; no posterior uncertainty claim is made.

The core screen is mean focal-minus-M0 gain at least0.01nat/trial and strictly
greater than two participant standard errors. Preserve EI/ER signs, participant
median and its existing9,999-draw stratified bootstrap as separate diagnostics.
Only valid core failures can support deterministic curtailment:21 failures
make the fixed200-draw90% criterion unreachable. Technical/nonconverged or
nonfinite attempts do not count, and richer diagnostic failures cannot inflate
the core failure count. A single representative is neither a cell verdict nor
a full recovery pass. No narrower empirical launch follows from passing it.

All later obligations remain: other admissible competing families including MC,
z/c confusion, null/half/twice-SESOI and heterogeneous-effect regimes,
group-effect bias/interval coverage, individual recovery where claimed,
urgency and nonlinear-value rivals, all registered partitions, missingness
stress and the actual eight-sequence sensitivity. The generic-bias stacked
ensemble is a separate training-only comparison, not a relabeled MZ or MC win.

## Selected bounded execution

Reuse the existing Python3.12.1/NumPy1.26.3/scikit-learn1.5.1 module stack and
the qualified AMD7502 CPU feature, with one BLAS/OpenMP thread per worker.
First select one attended100-minute allocation,1CPU/2GiB, for the focused
new-runner checks, changed ASTRA validation and the complete MV representative.
After checks and independent review pass, a separate95-minute1CPU/2GiB MZ
representative may run concurrently. Each fitting process has a90-minute
external limit plus10seconds for termination, and writes only a fresh episode
attempt. This reserves at most3.25 additional allocated core-hours; previous
consumption remains charged. No GPU, automatic requeue, renewal or follow-on
array is selected. No new node-local or shared bulk source copy is needed;
planned durable outputs are below20MiB per representative.

The existing MC fitter/runner/reports remain unchanged. Each complete fit is
preserved; partial optimizer state is not a converged checkpoint. If a run
fails or times out, diagnose its saved attempt before selecting a continuation.

## October5 scheduling update after the scientist requests more parallel jobs

At00:23UTC the scientist's explicit speed preference selects fixed indices1–20
for each cell before the index0 representatives have fully completed. Both
baseline fits have converged; focal runtime/convergence uncertainty remains.
The tested implementation, scientific cells and200-draw targets stay fixed.

Two20-task arrays may run concurrently, each task retaining1CPU/2GiB/95minutes,
5400s process limit plus10s termination, AMD7502, one numeric thread and
no-requeue. This adds at most63.3333333333 allocated core-hours,40CPUs/80GiB
peak and800MiB outputs, beyond the initial3.25-core-hour representative envelope.
Only these40 new draws are selected. Root monitors and pools unique valid
indices, stops pending work when a cell is unreachable, and diagnoses failures
before any retry. No source, scientific threshold, seed or failed MC result
changes. The initial representative-only reservation above is historical.

## October5 full fixed MZ cell scheduling selection

At00:37UTC, after MZ0 completed all four fits in1224.255s and passed the
necessary0.01nat/>2SE screen, root selects remaining MZ indices21–199 under
the scientist's explicit preference for more parallel jobs. Together with
index0 and already-running1–20, these are exactly the existing200 indices.
One passing draw is feasibility evidence, not a recovery-rate or full-gate pass.
MV's selected indices0–20 remain; its first valid failure is preserved.

The additional179 MZ tasks retain1CPU/2GiB/95minutes,5400s+10s termination,
AMD7502, one numeric thread and no-requeue. All179 may be scheduled concurrently.
Their extra reservation is283.4166666667 allocated core-hours. This brings
the current representatives and both expanded waves to350 reserved core-hours,
excluding preserved earlier costs and without pretending reservation is usage.
Peak arrays can use219CPUs/438GiB plus the attended coordinator allocation.
The same21-valid-failure pending-only curtailment, technical exclusion and
no-automatic-retry rules apply. No seed, scientific threshold, outcome role,
generating cell or prior failure changes.

## Same-cohort MZ0 fixed-bound rival comparison

After the necessary MZ0 screen completes, fit the four missing fixed-bound
rival components on that same preserved simulated cohort: MC trained on1,4
and on2,3, then MV trained on1,4 and on2,3. Reuse the completed M0/MZ
fits and original held-out scores. Do not regenerate events, refit those
models, read observed responses or create a new recovery index.

The separate entrypoint uses the existing fitting and likelihood APIs because
the general CLI's score export requires an absent `sequence_template` column.
It validates the saved source as completed MZ index0 and uses its exact
implementation. Every completed fit is checkpointed. Each model scores the
same108 participants with256 held-out trials each over both directions.
Report MC/MV-versus-M0 using the existing necessary comparison and paired
MZ-minus-MC/MV mean differences and participant standard errors as diagnostics.
Nonnested comparisons have no new selection or curtailment threshold.

Family selection, confusion rate, group recovery and full recovery remain
unevaluated: the admissible menu also includes MVZ/MVC, and the frozen
urgency, nonlinear-value, effect-regime and sensitivity obligations remain.
For a later group-bias calculation the effect-scale denominator must be
explicit; error relative to the tested0.05 starting-point departure must not
be obscured by normalization to raw z=0.45. Individual rank/RMSE cannot be
qualified by this constant-focal-parameter generating cell.

The selected representative add-on has a3000-second process cap plus10seconds
for external termination,1CPU/2GiB, one numeric thread and noGPU. It may use
the existing attended allocation only with adequate remaining time after its
focused checks. Any incomplete fit is a technical partial result, not a
recovery-cell failure. No automatic retry, successor or allocation renewal
is selected.

## Prospective necessary MZ group-bias estimand

Before group estimates are inspected, root adopts the physical rejection
displacement0.5 minus mean(expit(fitted rho)), averaging equally over the54
participants in each task version and their two primary fitted halves. The
uniform MZ cell's true displacement is0.05 separately in EI and ER. For each
version compute abs(mean across200 replicate estimates minus0.05)/0.05;
both must satisfy the existing10% relative-bias maximum. Do not substitute
mean absolute error, expit of a mean rho, a population-marginal estimate or
a denominator of0.45. All valid draws enter, including predictive-screen
failures. Technical/missing draws remain unresolved, and partial means have
no final bias verdict. A null effect has undefined relative bias rather than
an invented positive denominator.

This component uses already-completed fits. It supplies no uncertainty
interval, coverage qualification, post-selection correction or full-gate
verdict. The required participant-resampling interval still needs its own
prospective convention and measured resource selection. No bootstrap refits
are selected by this estimand definition.

## October5 required half- and twice-SESOI MZ representatives

At09:51UTC, before generating either new cell, root instantiates the
registered half- and twice-SESOI levels for the same uniform MZ necessary
corner. GOAL's behavioral recovery section and SEARCH_POLICY's simulation
levels already require these effect sizes. Continue the rejection direction
of the existing MZ cell; scale the physical starting-point displacement,
not rho. These representatives do not depend on the outstanding nonnested
arbitration or interval-method choices.

| Level | Focal truth for every participant | Fixed seed for index i |
| --- | --- | --- |
| half (0.5 SESOI) | eta=0, z=0.475, rho=log(0.475/0.525), c=0 |1700047+10007*i|
| twice (2 SESOI) | eta=0, z=0.40, rho=log(0.40/0.60), c=0 |1900057+10007*i|

Each namespace has200 prospectively fixed indices0–199. Only index0 of each
is initially selected; any later wave needs a separate coordinator selection
based on measured feasibility and cost. Seeds are not replacements for prior
valid or technical results. The new cells preserve the exact design-only
source projection, all108 participants/432 runs/27,648 offer slots, original
nuisance sampler, deadline, contamination, priors, numeric supports and
original500-sweep fitting procedure. Freshly fit M0/MZ in both primary
directions, with unchanged held-out scoring and complete checkpoints.

The0.01-nat/two-SE nested diagnostic is unchanged. Half-SESOI is below the
registered at-or-above-SESOI90% requirement: its outcomes cannot count as
failures of that requirement or trigger its21-failure curtailment. Twice-SESOI
is subject to the existing necessary-screen semantics, but a representative
does not establish its cell rate, family selection, confusion, group bias,
interval coverage or full recovery. Both levels retain zero focal-effect
heterogeneity and cannot establish individual rank/RMSE recovery. All broader
generating regimes, robustness competitors, splits and missingness obligations
remain required; observed behavior/BOLD remain closed.

A separate thin adapter reuses the checked single-process loop without
editing its scientific functions or the completed/live production runners.
Focused checks cover new truth/seed identities, half-level failure semantics,
checkpoint claims and restoration of scoped bindings after success/failure.
The two representatives share a bounded2CPU/4GiB/100-minute allocation,
normal/CPU_SKU:7502/no-requeue, with one exclusive exact1CPU/2GiB step each,
all numerical thread counts1 and5400-second process caps plus10s termination
grace. Checks and changed ASTRA validation share that substantive allocation.
Maximum reservation is3.3333333333 allocated core-hours; all previous costs
remain charged and actual new cost will be recorded after termination.

## October5 preserved MZ0 complementary-partition sensitivity

The registered other complementary partitions are independently required by
GOAL's held-out behavioral prediction section. Root selects their preparation
on the prospectively designated first MZ cohort, index0/seed910031, while
preserving its completed primary fits and all later cohorts. This is the same
saved simulation used for the fixed-bound rivals; no responses are generated
or copied, and observed behavior/BOLD remain closed.

The two selected partition processes are12_34: training{1,2}→scoring{3,4}
and its reverse, and13_24: training{1,3}→scoring{2,4} and its reverse.
Each freshly fits M0/MZ in both directions: four original500-sweep fits,
all108 participants,128 held-out trials per direction and256 per participant
combined. Priors, likelihood, numeric supports, optimization, nuisance/focal
truth, participant weighting and the0.01-nat/two-SE diagnostic are unchanged.
All subject and population parameters are learned only from that direction's
training runs. Original primary scores cannot substitute for alternate scores.

Validate the completed MZ0 report, generating truth and saved cohort through
the existing source checks and replay its original primary scores before
starting separate alternate-partition accumulators. Preserve each returned
fit before validation/scoring; any numerical failure is technical unknown.
These are within-cohort sensitivity diagnostics, not independent replicates,
cell rates, family selection, curtailment, confusion, group-interval or gate
verdicts. Four leave-one-run-out directions and broader recovery remain open.

Two exclusive exact1CPU/2GiB steps share a2CPU/4GiB/100-minute allocation,
normal/CPU_SKU:7502/no-requeue. Each process is capped at5400seconds plus10s
termination grace. Focused source/split/scoring/failure checks and changed
ASTRA validation share the substantive allocation; numerical threads stay1.
The primary MZ0 four-fit cost1224s is analogous planning evidence, not a runtime
guarantee for another partition. Reserve at most3.3333333333 allocated
core-hours, with actual charge after termination and no automatic successor.

## October5 remaining MZ0 leave-one-run-out sensitivities

Before any LOO fit, root instantiates the four registered omitted-run
sensitivities on the same preserved MZ0 cohort. Each held run1/2/3/4 is
scored by fresh M0/MZ fits trained on the other three runs. Retain192
training and64 held-out trials per participant, all108 participants,
original500, unchanged priors/likelihood/support and train-only subject/
population fitting. Original primary source validation/replay is unchanged;
no original score is substituted into a LOO panel and no simulation is made.

Each direction reports its per-participant sum(log predictive density)/64.
Pool only the four explicitly selected completed, valid, source-matched
directions: average their four scores within each participant, then apply
the existing MZ-minus-M0 nested diagnostic to108 paired differences. Equal
averaging is exact because every held run has64 trials; together they cover
all256 original trials once. Do not treat432 participant-folds as independent.
Missing/technical directions leave the combined diagnostic unknown; no
partial renormalization or zero filling is allowed. Preserve all fit/scoring
prefixes and return malformed purported-complete inputs as errors.

The pooled report may describe per-run overall/EI/ER advantage signs and
whether at least three of four run means are strictly positive. These are
the already-declared robustness diagnostics, not new gates or independent
replications. A one-cohort M0/MZ sensitivity cannot qualify the full menu,
family recovery, confusion, interval coverage, individual recovery or the
complete behavioral gate. Other generating regimes and mandatory nonlinear/
urgency rivals remain open; observed behavior/BOLD stay closed.

Four1CPU/2GiB workers share4CPU/8GiB/100minutes, normal/CPU_SKU:7502,
no-requeue, numerical threads1. Each process has5400s plus10s termination
grace. Combined focused checks190s, changed ASTRA40s, parallel workers5410s
and final pooling130s total5770<6000s. The wrapper waits on all four selected
children and pools only if every child exits0; it does not poll, retry or
select a successor. Reserve6.6666666667 allocated core-hours, preserving
all prior charges and deferring actual charge until the allocation ends.

## October5 preserved MZ0 anchored nonlinear-value sensitivity

The mandatory nonlinear-value rival is already specified in GOAL136–149 and
SEARCH_POLICY150–162. Prepare nonlinear M0 and MZ on preserved MZ0, one
process per model, each freshly fitting both original primary directions.
Use the existing analytic fixed-bound deadline likelihood with separate
gain/loss powers in[0.5,1.5], anchored at ten dollars, and their originally
recorded priors, starts, supports and original500 optimizer. All participant
and population parameters are learned only from each training half. There is
no new simulation, observed-data access, urgency approximation or formula change.

Validate/replay the preserved fixed-primary source unchanged before creating
separate nonlinear score accumulators. Nonlinear M0 has five parameters and
MZ six; use explicit rival-aware fit checks and likelihood calls rather than
relabeling nonlinear fits as fixed. Retain all108 participants,128 held-out
trials per direction and256 combined. Each directional log-likelihood sum
is divided by256 before adding the two contributions. Preserve returned
fits before validation/scoring and retain technical-failure prefixes. Missing,
nonconverged or nonfinite components leave the combined result unknown.

After both exact selected reports are complete and source-matched, the
existing nested contrast evaluates nonlinear MZ minus nonlinear M0 using
108 participant differences. Unequal powers permit magnitude-dependent
gain/loss asymmetry even with eta fixed at zero: M0 is a curvature-adjusted
reference, not a globally symmetric-value null. This comparison measures
incremental starting bias conditional on that flexibility. Fixed-primary
scores remain separate descriptive references; no nonnested arbitration,
family/rate/interval/rank/recovery or full-gate verdict is supplied. Numerical
convergence and cost have not yet been measured for this nonlinear pair.

Two exact exclusive1CPU/2GiB workers share2CPU/4GiB/100minutes on normal/
CPU_SKU:7502 with numerical threads1 and no requeue. Each whole process has
5400s plus10s grace; combined new checks190s, ASTRA40s and conditional
pooling130s give5770<6000s. Fixed MZ0's four-fit1224s and fixed MVC0's
five-parameter pair2464s inform planning only. Reserve3.3333333333 allocated
core-hours once; preserve prior consumption and charge actual allocation cost
after termination. No automatic extension, retry or wave is selected.

## October5 remaining twice-SESOI fixed draws

After representative0 completes all four original500 fits in1126.328423s
with a valid necessary-screen success, select the remaining fixed twice-SESOI
indices1–199. Keep seeds1900057+10007*i, physical z0.40, eta=c=0 and all
original nuisance/simulation/fitting/scoring rules. No seed replacement,
effect change, new generating regime or half-SESOI wave is selected here.

A separate thin entrypoint reuses the frozen effect-level adapter unchanged,
adding only truthful external-wave dispatch metadata at every checkpoint.
It accepts twice indices1–199 only and cannot select subsequent work.
This preserves the original representative and its active half-SESOI sibling.
All valid necessary failures remain failures; technical/nonfinite/nonconverged
attempts remain unknown. The coordinator retains the existing deterministic
21-valid-failure curtailment rule for the fixed200 at-or-above-SESOI cell.
Richer diagnostic failures cannot inflate that count. No family/rate/coverage
or complete gate is supplied by individual baseline components.

Submit index1 first with focused tests of the new dispatch metadata and scoped
binding restoration, reusing prior scientific checks. Once these pass, root
separately releases2–199 without waiting for index1 completion. Every task uses
1CPU/2GiB/95minutes, normal7502/no-requeue, threads1 and5400s plus10s grace.
Index1's130s new checks and40s ASTRA leave total5580<5700s. All199 tasks
reserve315.0833333333 allocated core-hours once,199 CPUs/398GiB peak and
below4GiB durable outputs; actual costs are charged only after termination.

## October5 remaining half-SESOI fixed draws

Half representative0 completes all four original500 fits in2144.299226s.
Its0.0039358058nat/trial gain is below the fixed margin; the90% criterion does
not apply below SESOI, and this cannot trigger21-failure curtailment. Select
the remaining fixed half indices1–199, seeds1700047+10007*i,z0.475,eta=c=0,
with all original nuisance/design/simulation/fitting/scoring settings. No
scientific-failure substitution or seed replacement is permitted.

A separate half-only1–199 entrypoint changes only truthful external dispatch
metadata around the frozen effect adapter. Every checkpoint retains false
90%-criterion applicability, null required fraction and false curtailment
failure, regardless of its valid necessary-screen result. Baseline diagnostics
remain distinct from full-menu family recovery, intervals and the complete gate.

First index1 uses1CPU/2GiB/100minutes to combine its focused new checks and
the seven new twice-summary checks under190s, ASTRA40s, completed-result
plotting130s and science5410s,5770<6000 total. Plot failure is recorded without
blocking independent scientific fitting. The other198 tasks use95minutes each;
all use normal7502, numerical threads1, no requeue and5400-second process caps
plus10s grace. Reserve315.1666666667 allocated core-hours once,199 CPUs/
398GiB peak, below4GiB outputs. Root releases2–199 only after changed-path
checks pass. Prior costs remain charged; no automatic subsequent selection.

## October5 urgency likelihood cost and resolution pilot

Before selecting full urgency fits, measure the actual registered numerical
likelihood. Its separate finite-volume solve for each distinct drift on every
objective call prevents extrapolating urgency costs from analytic fixed-bound
fits. Select the first sorted EI and ER participant from preserved MZ0, both
primary training halves, M0/MZ, and the three exact deterministic first-sweep
starts already used by the fitter. Validate and replay the complete108-person
fixed-primary source first. The selected two participants are a prospective
cost/accuracy pilot, not a reduced recovery cohort or fitted urgency result.

At each of the24 parameter/data combinations evaluate the existing base mesh
(dt0.002s,200 spatial intervals) and the already required refined mesh
(dt0.001s,400 intervals), at most48 likelihood vectors with128 trials each.
Retain actual unique-drift counts, consumed time and per-trial/mean paired
log-likelihood differences. Do not optimize, invent a numerical acceptance
threshold, select a model, or qualify precision at future fitted support.
No observed responses or new simulations are used. The original likelihood,
fitter, implementation and active worker files remain unchanged.

The whole pilot, including source reads/replay and partial checkpoints, is
bounded to1800s plus10s termination grace. A timeout preserves the completed
prefix and consumed cost; it is neither scientific failure nor authorization
to retry or extend. This measurement informs the next urgency resource and
numerical decision; full fitted-support mesh checks remain required.

Bundle this pilot with terminal MZ/null/twice/half summaries, the new half-pool
and pilot checks, and changed ASTRA validation in one attended1CPU/2GiB/
60-minute normal7502 allocation. Numerical threads remain one. Bound checks
to130s each, ASTRA40s, MZ/null pools310s each, twice/half pools250s each and
the pilot1810s, total3230s before setup within3600s. Reserve1 allocated
core-hour once and charge actual elapsed allocation after exit. Reuse passing
old summary checks; this allocation has no supervisor, retries or successor.

### Isolated batching candidate for the same urgency equations

The scalar pilot's first center likelihood takes20.73s for86 exact unique
drifts at the base mesh and44.69s at the refined mesh. Prepare an isolated
batched implementation while retaining the original solver as reference.
Stack independent drift systems as uncoupled tridiagonal blocks; retain the
same coefficients, implicit time steps, initialization, boundary flux,
response-density convention and SciPy `solve_banded((1,1))`. Cross-block
couplings must be zero and mass conservation checked separately for every
drift. No rounding of drift values, surrogate, truncation, new dependency,
scientific formula or active-file modification is permitted.

Before any adoption, test block independence/order, single-block behavior,
boundary flux/survival and response/miss likelihood against the scalar solver
at both registered meshes. If the complete48-call scalar pilot finishes,
benchmark the candidate on that exact saved initialization panel, reusing
its recorded scalar values/times rather than repeating the costly reference.
Validate/replay the unchanged full source and match every selected training
row, parameter vector and mesh. Report actual timing and per-vector maximum
absolute log-score discrepancies. The existing1e-12 replay tolerance is an
engineering-equivalence check only, never a numerical-inference threshold
or proof of fitted-support precision. No optimizer or full fit is selected.

The benchmark has a600s whole-process cap plus10s grace and preserves
incomplete results. It may run in the same attended allocation only after
static review and focused compute checks, with at least15 minutes remaining
for checks250s, changed ASTRA40s and benchmark610s. Otherwise defer it to
later coordinated compute without extending this allocation. No extra
core-hour reservation, automatic retry, adoption or successor is selected.

## October5 diagnosed nonlinear M0 first-half sweep exhaustion

The original nonlinear M0 process returns a finite first-half fit after all500
sweeps, with no subject-optimizer failures but without satisfying the unchanged
joint-relative tolerance. Its final objective change is0.002432152163 versus
the required0.000787274527. Retain this technical failure and its five lower
population-scale support contacts; neither supplies a scientific recovery
failure or a reason to change priors, support or model equations.

Select one fresh M0-only numerical repair: increase only the maximum-sweeps
ceiling to1500, retain all original starts and stopping/tolerance rules, and
fit both original primary halves anew. No failed optimizer state or parameter
warm start is reused. The original500 implementation still validates/replays
the saved fixed-primary source; a private runtime copy governs only the new
nonlinear M0 fits and their convergence checks. Preserve both specifications,
the exact failed-source identity, original support contacts and new fit evidence
in a distinct repair report. All108 participants and128/256 score normalization
remain unchanged. The original nonlinear MZ process continues independently.

One1CPU/2GiB/4h30m normal7502 job gives a15600s whole-process cap plus10s
grace, focused new checks190s and ASTRA40s,15840s within16200s. Original
first-half cost2459.72s informs this finite3×sweep estimate; completion is not
guaranteed and no automatic extension/retry follows. Reserve4.5 allocated
core-hours once. Keep parent46606082 allocation accounting separate from the
repair's prior-process-plus-new-process diagnostic cost to avoid double charge.

Only after both reports are terminal may an explicit repaired summary validate
the fresh M0 at runtime1500 together with the original MZ at500. A technical
component leaves the combined comparison unknown. Do not relabel repaired
fits as original500 or erase population-scale boundary contacts; no formal
family/rate/interval/full-recovery verdict follows from this comparison.

## October5 diagnosed nonlinear MZ reverse-half sweep exhaustion

The original nonlinear MZ process subsequently returns a converged153-sweep
first fit and a finite, nonconverged500-sweep reverse fit, with no subject
optimizer failures. Its final reverse objective change0.001055468812 exceeds
the unchanged0.000770483788 tolerance. Preserve all seven lower population
scale contacts in each original fit; this is technical unknown.

Select a separate reverse-only1500 numerical repair. Validate the exact
terminal46606082 attempt and the unchanged full fixed-primary source/replay.
Copy the complete first fit under original500, and rescore its held-out rows
because the technical report did not preserve the first score accumulator.
Fit only reverse freshly under1500, without failed-state warm starts. Model,
priors, support, starts, tolerance and all108 participants remain unchanged.
The report identifies each half's actual sweep specification and distinguishes
the copied first fit's historical elapsed time from new rescore overhead.
Prior process3434.922692s plus new process is counted once diagnostically;
allocation charges remain separate.

Use1CPU/2GiB/2h20m normal7502, one numerical thread and no requeue. A7800s
whole-process cap includes source validation/replay, rescoring and fresh
reverse fitting. New checks190s, ASTRA40s and fit/grace7810s sum to8040s,
within8400s. Reserve2.3333333333 allocated core-hours once. Completion is not
guaranteed; no automatic retry or walltime extension follows.

A separate both-repair pool may join the explicitly selected M0 repair46611338
with this new MZ attempt only after both are terminal. It must retain each
actual phase, original/runtime specifications, failed-attempt contacts and
costs. The existing M0-only pool remains frozen and correctly leaves the
original failed MZ unknown. Only complete valid components permit the existing
nested participant contrast; no formal family or full-gate verdict follows.

## October5 urgency conditional first-sweep cost pilot

After all48 batched likelihood vectors match the saved scalar reference
exactly, adopt the isolated batched solver only for a bounded optimization
cost pilot. Select eight tasks: first sorted EI and ER participant, both
primary training halves, and M0/MZ. Each uses all128 training rows, the exact
three existing starts, original parameter bounds and L-BFGS-B options, and
the original first-sweep conditional prior (mean=population centers,
sd=exp(log(individual_scale_center)), retaining the literal original arithmetic).
Starts still use individual_scale_center directly. No population hyperparameter is updated, no
held-out urgency score is computed and no full cohort/model fit is claimed.

Every task validates/replays the unchanged full108-person MZ0 source and
requires the completed matching scalar/batched benchmark. Record each start's
elapsed time, optimizer nfev/nit/status/message/parameters/objective, measured
likelihood call counts/times and the original minimum-finite selection rule.
Preserve a partially attempted start and its consumed cost on interruption;
an incomplete3000s task is censored resource evidence, not a recovery failure.
Use only the original .002/200 mesh here. Fitted-support .001/400 qualification
and full108-person hierarchical fitting remain separate requirements.

One8CPU/16GiB/60-minute normal7502 job bundles focused new checks under250s,
ASTRA40s and eight concurrent1CPU/2GiB steps with3000s whole-process caps plus
10s grace:3300<3600s before setup. Numerical threads stay one per worker.
Reserve8 allocated core-hours once; report actual parent cost after exit,
without charging child process times again. No automatic retry, solver-wide
adoption, full-fit dispatch, or new recovery cell follows from this pilot.

## October5 common-gain-range MZ0 sensitivity

GOAL's EI/ER section and DATASETS explicitly require a common gain range
10–20 sensitivity. Select this on the preserved synthetic MZ0 cohort now,
with observed responses/BOLD still closed. Validate/replay the full original
source, then retain rows with10<=gain<=20, keeping their original responses,
deadline misses, loss magnitudes, run and trial identities. Do not match,
resample or equalize the two versions: this is a common range, not identical
offer grids or a within-person context experiment.

Retain all108 participants and both primary directions. Fit fresh original500
M0 and MZ on each restricted training half; preserve the original likelihood,
priors, supports, starts, tolerances and contamination channel. Record each
participant's actual retained training/held-out slots and counts. Full-session
offer matrices imply96 retained EI and176 ER rows per participant, but
per-half counts must be validated from the actual retained rows, not assumed
equal or128. No completed full-range fit is reused as a restricted fit.

The two direction components save held-out log-likelihood sums and actual
counts. Only after both complete may the pool divide each participant/model's
sum across directions by their total retained count, then apply the existing
nested contrast across108 participant differences. Do not average unequal
half means, use256 as denominator, weight participants by retained trial count
or add the original full-range primary scores. Preserve EI/ER diagnostic signs
and the usual claim limits; one sensitivity is not full recovery or independent
replication and does not resolve missing heterogeneity/stress/template settings.

Use one2CPU/4GiB/100-minute normal7502 parent with two simultaneous1CPU/2GiB
direction steps, each capped at5400s plus10s grace. New focused checks190s,
ASTRA40s and terminal-only pool130s leave5770<6000s. Reserve3.3333333333
allocated core-hours once; actual parent cost is charged after termination.
No automatic retry, follow-on wave or observed-data opening is selected.

## October5 isolated parallel hierarchical fitter preparation

The original conditional participant optimizations use fixed version means
and scales throughout each sweep. Prepare an isolated parallel candidate so
these independent blocks can use multiple allocated CPUs, followed by the
unchanged serial version-mean, global-mean and scale updates. Collect results
in the original sorted participant order. If final likelihood evaluations run
in parallel, accumulate the joint objective in the original order, adding each
participant likelihood and then its prior separately. Preserve original starts,
optimizer settings, bounds, first-minimum-finite selection, failure handling,
support contacts, convergence criterion and scientific return fields.

Keep the original fitter and all live worker files frozen. The candidate may
use the isolated batched urgency likelihood, but preparation is not adoption
for a full scientific fit. Qualify concurrency and multi-sweep arithmetic
against the original fitter with a cheap deterministic likelihood in bounded
coordinated compute; require allocated worker/thread limits and cleanup on
failure or deadline. Actual urgency cost and fitted-support precision remain
separate requirements. No full-fit submission, new core-hour reservation or
inference is selected by this preparation entry.

## October5 diagnosed common-gain M0 iteration-limit repair

Both original46613139 direction components ended at the first M0 fit after500
sweeps, with finite estimates, no subject-optimizer failures and EI/ER kappa
population scales at their original lower supports. Neither satisfies the
original joint-relative convergence threshold. No MZ fit or held-out score was
computed. Preserve those terminal reports and their actual parent cost.

Select four independent model/direction components on exactly the same MZ0
source and common-gain10–20 rows. Fresh M0 fits may run up to1500 sweeps,
with only maximum_sweeps changed; retain original starts, likelihood, priors,
bounds, tolerances and first-convergence rule. MZ is previously unattempted
and stays at original500. Failed states and full-range fits are never warm
starts. Each component validates the exact corresponding original failure,
full source/replay and retained support before fitting. Retain original and
runtime specifications and the old failure's contacts/cost separately from
the new component; four consumers do not imply four original cost charges.

Only all four valid completed components can supply a pooled nested contrast:
M0[1500,1500] and MZ[500,500], each participant/model's two held-out log sums
divided by the actual96 EI/176 ER total, then equal weighting across108 people.
Missing/technical components remain unknown without partial contrasts;
malformed completed components abort. Existing scientific claim limits remain.

One4CPU/8GiB/90-minute normal7502 parent bundles new repair/pool checks250s,
the eight isolated parallel-fitter checks in a2CPU/4GiB step130s, changed ASTRA40s,
four simultaneous1CPU/2GiB components4200s plus10s grace, and terminal-only
pool130s:4760<5400s before setup. Reserve6 allocated core-hours once; charge
actual parent cost after termination. The scientific components still use the
original serial fitter. No automatic retry, further iteration increase,
urgency full-fit adoption or observed-data access is selected.

## October5 terminal urgency probe2 endpoint diagnostic

Probe2 in46612782 completed its three original conditional starts, all with
unsuccessful line-search returns. Select its actual minimum-finite start2
endpoint, after validating the complete terminal report and unchanged full
MZ0 source/replay, for a local numerical diagnostic on the same128 training
rows and original conditional prior. This is not a full cohort fitted result.

Recover the installed unchanged L-BFGS-B bound-aware nondecision-logit
finite-difference perturbation using a cheap constant recording objective with
the same method, bounds and options. Evaluate exactly three declared points:
the endpoint, that actual perturbation and an explicitly diagnostic opposite
perturbation of the same effective size within the original bounds. Evaluate
the original scalar and isolated batched likelihood once each at each point,
on the original .002/200 mesh. Record every trial difference, conditional
objective, local slope, actual solver step count/time step, response-bin and
active-state assignments, and spatial initialization. Retain recomputed
endpoint objective versus optimizer-reported fun without assuming they agree
after an unsuccessful return. No scientific optimizer is rerun.

Matching scalar/batched values can locate shared reference behavior but cannot
establish smoothness along the unrecorded line-search path or full numerical
precision. Disagreement needs diagnosis before adoption. No new acceptance
threshold, refinement, optimizer change, full urgency fit or repeated48-call
benchmark is selected. Bound the whole producer to300s plus10s grace and
preserve partial calls on failure. Focused checks and the diagnostic may share
a suitable existing task allocation under root coordination; no extra parent
allocation or core-hour reservation is selected here.

## October5 unchanged optimizer trace after endpoint diagnosis

The selected endpoint diagnostic completes with exact scalar/batched equality
at all three points, matching local slopes and1864 time steps, but its
recomputed objective differs from the unsuccessful optimizer's reported fun.
That local panel cannot reconstruct the historical failed line search.
Select only original probe2/start2 for a recorded rerun from its unchanged
original initialization, using the same128 training rows, initial conditional
prior, batched coarse likelihood, bounds and optimizer settings. Use the
unchanged conditional optimizer with this one original start; retain original
start index2 separately from local single-start index0. Previous starts are
independent and are not repeated; no full cohort or population update occurs.

The likelihood adapter records each actual candidate,128 returned trial
values, exact summed negative likelihood/prior/objective, elapsed cost and
compact grid/bin/active geometry, then returns the original values unchanged.
It adds no objective evaluations. Preserve partial attempted/returned records
on errors or the1200s whole-process cap, plus10s external grace. This envelope
exceeds the original selected call's749.164759961s while remaining bounded.
Use one allocated CPU with numeric threads1 and fresh output, and retain the
original probe and completed endpoint artifacts unchanged.

Compare the new returned record with the original and locate existing trace
evaluations of its returned x and reported fun. Reproduction is an observed
fact, not a requirement to force with new tolerances or optimizer settings.
No new initialization, refined mesh, success filter, precision criterion or
full-fit adoption is selected. Focused tests and this trace should reuse
available capacity in the current substantive allocation; an additional
parent allocation is not reserved by this selection.

## October5 reverse common-gain MZ repair and combined execution

Parent46615238 ended with three complete components: direction0 M0 at512
sweeps, direction0 MZ at494, direction1 M0 at674. Preserve these fits/scores.
Only direction1 MZ exhausted500, with finite estimates, no subject failures,
EI/ER kappa/rho population-log-scale lower contacts, and the exact terminal
context recorded at13:09 in the execution log. Select that component alone
for a fresh1500-sweep maximum, leaving likelihood, priors, original starts,
tolerances, first convergence, support and original500 source/scoring fixed.

Use the isolated parallel hierarchical implementation with4 workers for this
fixed-bound MZ component only. Its eight engineering checks passed under the
previous substantive allocation. Participant blocks run independently at the
same fixed version parameters; original population updates and sorted
objective arithmetic remain serial. This selection does not adopt batched
urgency for any full fit. Bound the whole new producer to3600s with10s grace;
reserve cleanup time inside the fitter's remaining-time limit.

The terminal pool validates the three preserved components, exact failed
reverse MZ and new repair against the same full source and original two M0
failure contexts. Only four complete compatible score panels permit the
existing nested contrast with actual96 EI/176 ER total trials per participant,
then equal108-person weighting. Per-direction ceilings become M0[1500,1500]
and MZ[500,1500]. No partial contrast or broad recovery/gate claim is allowed.
Count the original two M0 processes once, all four46615238 component processes
once, and the new repair process once as descriptive durations; allocated
costs stay separate in the execution log.

The previous allocation is terminal. One new5CPU/10GiB/75-minute normal7502
parent will run this4CPU repair and the independent selected1CPU unchanged
urgency-start trace concurrently. New repair/pool tests190s, trace tests70s,
ASTRA40s, maximum concurrent science3610s, and conditional pool130s total
4040s below4500s. Reserve6.25 allocated core-hours once; actual parent charge
follows termination. Each branch requires its own focused tests, and one
branch's numerical/technical failure must not erase the other's result.

## October5 nonlinear M0 direction-specific second numerical selection

The fresh1500-sweep nonlinear M0 first half in46611338 remains technical:
its last objective change0.000835900079 exceeds the unchanged0.000787131418
criterion, with no final-sweep subject failures. The last50 changes show
continued slow descent. Preserve the original500 failure and this1500 failure,
including their five lower population-scale contacts. No score or failed
optimizer state from either attempt may be reused.

Select two independently runnable fresh nonlinear M0 directions: first half
maximum2000, reverse half maximum1500. Only the first ceiling is extended
from the previous selection; the reverse half was never reached. Keep original
initialization, first-convergence rule, priors, bounds, likelihood and all
tolerances unchanged. Validate/replay the full original500 MZ0 source and
exact two-failure lineage before fitting. Score each direction's128 held-out
rows with the original likelihood, retaining its log sum divided by256 per
participant so complementary directions combine by addition.

Adopt the qualified isolated parallel fitter for these two nonlinear M0
components only: four independent participant workers per direction, original
nonlinear likelihood, serial population updates and ordered objective sums.
Use actual4CPU steps and all numerical thread limits1; preserve interrupted
progress and reserve bounded worker cleanup inside each4800s whole-process
cap. Failed state is not a warm start. Actual nonlinear parallel speedup is
unknown; the serial first-half measurement projects about8402s for2000 sweeps,
so the selected bound may still yield technical unknown rather than extension.

The new pool requires both valid M0 directions and the preserved completed
MZ reverse-repair46612046, whose original153-sweep first fit and fresh736-sweep
reverse fit remain unchanged. Validate M0[2000,1500]/MZ[500,1500] ceilings,
identical original source/primary scores, all108 participants and256 trials.
Missing or technical M0 directions withhold the nonlinear contrast. Keep all
historical process costs once, separately from new process and allocation
costs; do not count repeated lineage references or copied fits again.

One normal7502 parent requests8CPUs/16GiB/90m for two concurrent4CPU/8GiB
steps. New producer/pool checks250s including grace, ASTRA40s, maximum
concurrent science4810s and pool130s total5230s below5400s. Reserve12 allocated
core-hours once; actual charge follows termination. No automatic retry,
further ceiling, changed optimizer or scientific verdict is selected.

## October5 crossed nondecision-time diagnostic after the saved trace

The saved urgency trace reproduces its unsuccessful return and exposes
different recorded states for returned x and reported fun. Repeat evaluations
at x are identical. The objective change to the final trial is dominated by
one bin-changing row, but other parameters and exact drift grouping also
change. Select a bounded diagnostic using A=trace evaluation270 and B=485,
C=A with only nondecision_logit from B, and D=B with only that coordinate
from A. The A→C and D→B contrasts hold the remaining parameters fixed.
Horizon and actual time step remain coupled to nondecision time, precluding
a claim that this alone isolates the bin-lookup operation.

Exactly five new unchanged coarse urgency calls are selected: scalar B,
scalar/batched C and scalar/batched D. Reuse saved batched A/B and the
completed endpoint's matching scalar A; no repeated A, optimizer, refined
mesh or new step size is introduced. Full original-source/replay, prior,
selected participant/training slots and terminal anchor identities must match.
Retain all128 trial vectors and geometry, solver differences, conditional
objective/prior differences, two within-background contrasts and descriptive
interaction. Numerical diagnosis supplies no full-fit or inference criterion.

One1CPU/2GiB step may use available capacity in46618614. Its180s whole
producer cap,190s external timeout+10s grace, new checks60s+10s and ASTRA
30s+10s total310s within6m. Preserve all prior attempts and partial new calls.
No new parent allocation, reservation, automatic continuation, interpolation
change or optimizer adoption is selected.

## October5 two-point registered refined-mesh comparison

The completed crossed diagnostic reproduces the nondecision-coordinate effect
under both fixed backgrounds with exact scalar/batched agreement. Stop unchanged
coarse repeats. Select only A and C from that completed report for two batched
calls at the already registered .001-second/400-interval mesh. Preserve their
parameters, original prior, selected ER participant and128 training slots.
Reuse saved coarse vectors and report refined-minus-coarse vectors/objectives,
both A→C contrasts, trial geometry and partial completion if interrupted.
No scalar repeats, additional points, optimizer or new simulation are selected.
This tests resolution dependence at failed conditional-candidate states, not
precision at a qualified fitted support. No acceptance threshold, density
interpolation change, optimizer revision or production adoption is introduced.

After parent46618614 ends, one bounded attended1CPU/2GiB/15m allocation may
run new focused checks45s+10s, ASTRA30s+10s and the whole diagnostic120s with
external130s+10s. Also render a saved-result figure comparing primary,
common-gain and nonlinear MZ0 contrasts within30s+10s, retaining their distinct
trial supports and nonlinear reference meaning. Use means±2 participant SE,
never label these as qualified confidence intervals or independent cohorts.
All work uses one numerical thread and original artifacts remain unchanged.
Reserve0.25 allocated core-hours once and charge actual time after prompt exit;
no automatic successor, allocation extension or renewal is selected.
