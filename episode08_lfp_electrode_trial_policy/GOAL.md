# EP08 — learning electrode-retention and trial-allocation policies

This episode is governed by [the common adaptive protocol](../ADAPTIVE_SEARCH_PROTOCOL.md).

## Authority and history boundary

This is the current local contract. It does not create canonical state,
authorize compute, expose neural outcomes, or establish a result. Earlier LFP
access must be recorded, and no prior score, preferred subset, threshold, or
conclusion may initialize the search. Canonical loop, Goal, registered program,
and search-policy bindings are required before launch.

Because only the existing two primary animals are currently available, the
locked test described here is internal held-session evidence. A third,
lineage-independent animal is required for an external generalization claim.

## Adaptive scientific question

After one fixed all-electrode calibration pilot, can an acquisition policy
learn **which physical electrodes to retain and which trial stratum to sample
next**, under a fixed electrode-by-trial budget, so that an LFP-to-population
mapping preserves more unseen-trial motor information than cost-matched
random, geometric, or reliability-only acquisition?

The scientific output is one transferable acquisition policy and its
budget-performance frontier, not a post hoc list of the best electrodes in
each recorded session. The policy must operate from information genuinely
available at the stated acquisition step and transfer unchanged to unseen
sessions.

## Data roles and firewall

Use the six Mihili and six Chewie-L M1 execution sessions as the primary
corpus; Chewie-R is implant sensitivity. Assign whole sessions within each
primary animal to adaptive development or a sealed internal audit before
neural utilities are computed. The manifest should target four development
and two audit sessions per animal, subject to a frozen feasibility rule.

Within every session, freeze a direction-stratified evaluation set that is
never available to the selector or decoder. Every policy then receives the
same **16-trial pilot**—two hash-selected calibration trials in each of the
eight frozen directions—recorded on every structurally eligible electrode.
The pilot reveals LFP and calibration spike targets, is charged and reported
separately, and cannot contain evaluation trials. A policy locks its retained
electrode set after this pilot. It may then request only the next trial stratum;
an electrode added later is forbidden and no retained electrode may backfill
an earlier non-pilot trial.

Development sessions can train policy parameters and proposal logic. On audit
sessions the policy sees only static geometry/QC, the identical pilot summary,
and—after each permitted acquisition action—the selected-electrode LFP and its
calibration target. Evaluation spikes never create policy reward. Audit
evaluation outcomes are opened once after the complete action log is locked.

## Action space and bounded scientific grammar

Electrode retention and trial allocation are ordered stages. The retention
stage selects a subset from the pilot-observed physical electrodes and then
closes permanently. A trial action chooses only a direction/time stratum; the
trusted replay returns the next trial in that stratum's frozen hash order.
Whole electrodes carry all three primary LFP features: LMP, 100–200-Hz power,
and 200–400-Hz power. The decoder, neural target roster, epoch (`+150` to
`+450 ms` after movement onset), metric, and evaluation denominator are shared
across policies.

Candidate policies may compose only:

1. **Reliability selectors:** artifact fraction, missingness, line-noise,
   stationarity, and repeatability estimated from currently acquired LFP;
2. **Spatial-diversity selectors:** farthest-first, coverage, or D-optimal
   selection using authenticated electrode geometry, with an explicit
   geometry-missing fallback;
3. **Redundancy-aware selectors:** correlation pruning, facility-location,
   log-determinant, or conditional-variance gain computed from acquired LFP;
4. **Learned electrode selectors:** linear/ridge ranker, shallow gradient
   boosting, or offline contextual ranking trained on cross-fitted
   development-session marginal-utility labels, using only pilot-time
   features;
5. **Trial allocators:** direction-balanced, uncertainty-weighted,
   residual-diversity, D-optimal, or bounded upper-confidence allocation over
   direction/time strata. For adaptive allocators, reward is the change in
   nested-CV calibration loss computed by the trusted evaluator using acquired
   calibration data only; and
6. **Combination rules:** a fixed weighted pilot score or learned pilot
   ranker followed by one trial allocator. Searchable allocator memory and
   lookahead are capped at two acquisition steps.

The primary common grid is whole-electrode counts `E in {4, 8, 16}` crossed
with post-pilot calibration-trial counts `N in {32, 64, 128}`. The pilot is an
identical additional cost for every policy. Full-electrode/full-trial data are
a ceiling/equivalence diagnostic, not a primary grid cell. A fixed reduced-rank
ridge decoder with development-selected global rank/penalty is used for every
selector so decoder flexibility cannot masquerade as acquisition-policy gain.
Learned marginal-utility labels for session `s` must be generated by a model
trained on other development sessions; the final audit ranker is refit once on
all eight development sessions. Learned policies may not use audit evaluation
targets, hidden trials, neuron identities unavailable prospectively, numeric
channel identity across sessions, realized future utility, free-form code, or
a per-session policy choice.

## Comparators and objective

Every candidate is compared at identical `(E,N)` costs with:

- repeated uniform random electrode and balanced-random trial selection;
- geometry-only farthest-first selection;
- reliability-only ranking;
- the fixed full-resource decoder; and
- a development-only hindsight oracle, reported strictly as an upper bound.

For policy `p`, session `s`, and grid cell `(E,N)`, let
`Delta_p,s(E,N)` be held-out `R2_SSE(p) - R2_SSE(b*)`, where `b*` is the
strongest frozen cost-matched baseline selected on development evidence only.
The primary objective is the equal-cell, animal-balanced mean: average the nine
grid-cell deltas within session, sessions within animal, then the two animal
means equally. A meaningful improvement is `0.005` mean `Delta R2_SSE`.

Promotion also requires positive improvement at `(4,32)` and `(8,64)` in both
primary animals, no material loss versus the full-resource ceiling under a
frozen equivalence margin, leave-one-animal-out transfer stress, and a majority
of development sessions improved in each animal. Complexity and acquisition
latency are secondary objectives. No ambiguous two-dimensional AUC is used.

## Search stages

1. **Preflight:** authenticate sources, freeze session/trial roles, verify
   geometry and resource support, and pass sequential-replay/leakage fixtures.
2. **Baseline stage:** run the identical pilot, then evaluate all random,
   geometric, reliability, and fixed trial-allocation baselines with common
   seeds and the same no-backfill rule.
3. **Coverage stage:** run at least 20 diverse policies spanning every
   selector and allocator family.
4. **Adaptive stage:** propose policy compositions and bounded parameters from
   the append-only ledger; each trial states a mechanism and parent.
5. **Stress stage:** rerun finalists across subset seeds, budget anchors,
   animals, and required falsifiers; retain a Pareto archive over utility,
   robustness, and cost.
6. **Lock/audit:** select one global policy, freeze its fitted-development
   state and sequential action code, replay from immutable inputs, then apply
   it unchanged to the sealed sessions and open evaluation outcomes once.

## Required falsifiers and ablations

- many-seed random selection with identical action counts and compute;
- permutation of development marginal-utility labels for learned selectors;
- shuffled electrode geometry for spatial policies;
- reliability, diversity, learned-score, and trial-allocation ablations;
- redundancy-matched and signal-quality-matched baselines;
- reverse animal transfer and leave-one-development-session influence;
- leave-one-animal-out selector and allocator transfer;
- action-log replay proving that every decision used only information then
  available;
- spike-contamination and low-frequency-only sensitivities; and
- synthetic saturating, spatial-clustered, redundant, trial-limited, and null
  acquisition worlds.

## Incumbent, budget, and stopping

The incumbent is a nonterminal policy selected only from development evidence.
It is not a candidate-ready result and cannot open audit data. Run at least
**40** and at most **112** valid policy trials. Patience is **20** consecutive
valid trials after the minimum without at least `0.005` improvement in the
primary mean `Delta R2_SSE` or a new feasible Pareto point. All selector and
allocator families, baseline repetitions, and required falsifiers must be
covered before patience can stop the search. At most 16 engineering failures
may be retried outside scientific patience.

Resource envelope: CPU only; at most 1,200 aggregate CPU-hours, 120 wall-clock
hours, 48 concurrent cores, and 750 GB scratch. Hitting a resource limit yields
`incomplete_search`, not evidence that a policy does or does not work.

## Lock, one-shot audit, and terminal boundary

The lock bundle must hash the source and role manifests, exposure ledger,
grammar, full trial/action ledger, baseline seeds, decoder, selected policy,
all policy-time features, fitted development state, resource grid, metrics,
intervals, falsifiers, environment, and terminal rules. No adaptation to
sealed-session evaluation outcomes is allowed. The audit opens once; any
post-audit improvement belongs to a future dataset and policy version.

`candidate_ready` requires the locked policy to exceed the strongest
cost-matched baseline by at least the frozen meaningful margin in the
animal-balanced nine-cell mean, improve both animal means, improve at least
three of the four audit sessions, pass `(4,32)` and `(8,64)` anchor rules,
survive influence and falsifier tests, and satisfy the full-resource
equivalence check. With only four audit sessions this is descriptive internal
evidence, not population inference. Otherwise return `closed_no_candidate`,
`unresolved`, or `technical_failure` exactly as locked.

A positive result supports only retrospective, internal held-session
generalization for **post-pilot electrode retention and calibration-trial
allocation** in these two animals and this feature representation. It does not
establish from-scratch electrode placement, a universal electrode count,
online clinical performance, causal electrode importance, a new implant
design, or external-animal generalization. The latter requires a third animal
collected or sequestered independently of this search.
