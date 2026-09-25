# Can a short pilot tell us what to record next?

At the start of a recording session, every electrode may be available, but it
may be impractical to keep all of them active or collect a large calibration
set. After a few reaches, the experimenter has to make two choices: which
electrodes are worth keeping, and which kind of reach would be most useful to
record next.

EP08 asks whether the same short pilot can guide both choices. Every policy
first sees 16 trials—two reaches in each of eight directions—recorded on all
eligible electrodes. It then keeps 4, 8, or 16 whole electrodes and chooses the
direction of each additional calibration trial. The final decoder uses the
selected data to predict motor-population activity on untouched trials.

The first test compares these choices at exactly the same electrode and trial
budgets. The main score gives equal weight to all nine points on the budget
grid and compares the pilot-guided policy with random selection, broad spatial
coverage, and simple signal-quality ranking. A positive result must also hold
at the two prespecified scarce-resource settings, in both animals, and in at
least three of four held-out sessions. The rule is learned on complete sessions
and then applied without change to those four sessions.

A winning policy is not yet a scientific explanation. It may simply retain
the cleanest electrodes, even when several carry the same information. It may
request trials only to restore equal direction counts. It may also win on
average because of one favorable session or one budget setting. The individual
sessions, both animals, and every point on the budget grid therefore remain
visible in the result.

This is the entry point to a larger question:

> Can a 16-trial pilot reveal which electrode or next reach will add information
> that is not already present in the data collected so far?

The intended paper should identify what the pilot actually learned. For
electrodes, a pilot score should predict how much performance is lost when that
electrode is removed from the retained set, not merely whether its signal is
clean. For trials, the errors available before a choice should predict which
reach direction will produce the largest improvement after its next trial is
added. The study should also show whether a session benefits more from another
electrode or from more calibration trials, without pretending that those two
resources have the same physical cost.

Even a positive result would not show that a selected electrode is biologically
special, that the replayed trial request would work online, or that fewer
channels save a measured amount of power. It would support a narrower claim:
in these recordings, information from a 16-trial pilot can guide a fixed
electrode-and-trial acquisition rule that predicts unseen population activity
better than cost-matched simple rules.

The [paper plan](outputs/paper_plan.md) compares this claim with prior work and
specifies the electrode-removal and next-trial predictions. The four current
held-out sessions test the acquisition policy itself. Because the explanatory
tests are designed afterward, confirming those explanations requires newly
sealed sessions; they cannot rescue a failed first-round policy.

## At a glance

| Question | EP08 design |
| --- | --- |
| What does every policy see first? | The same 16 trials: two from each of eight reach directions, recorded on all structurally eligible electrodes. |
| What is decided next? | Which 4, 8, or 16 whole electrodes to keep, followed by the reach direction of each new calibration trial. |
| What is compared? | Pilot-guided choices versus random, spatial-coverage, and signal-quality choices with the same data budget and decoder. |
| What is held out? | Complete sessions and a fixed set of untouched trials within each session. Those trials never guide an acquisition choice. |
| What must repeat? | The nine-cell average and the two scarce-resource settings must pass, both animal averages must improve, and at least three of four held-out sessions must improve. Every grid cell is still reported. |
| What can the first test conclude? | Whether one fixed pilot-based acquisition rule beats simpler cost-matched rules in these recordings. |
| What would make a deeper finding? | Pilot measurements predict an electrode's added value and the benefit of the next trial before either outcome is revealed. |
| What is the next test? | In the eight development sessions, cross-fit electrode-removal and alternative next-direction replays. Confirming that explanation requires newly sealed third-animal data; the four current held-out sessions test only the acquisition policy. |

## From a winning policy to an acquisition principle

Imagine two electrodes with equally clean LFP. If their signals carry the same
information, retaining both wastes a scarce channel. A third, slightly noisier
electrode may be more valuable because it captures population activity the
first electrode misses. The follow-up therefore asks whether a pilot-only
score predicts an electrode's **conditional value**: how much held-out
prediction worsens when that electrode is removed from an otherwise fixed set.

The same distinction applies to trials. If rightward reaches are already well
predicted but upward reaches have large calibration residuals, an adaptive
allocator should predict that the next upward trial will reduce future error
more than another rightward trial. That prediction must be made before the
trial's target or evaluation outcome is seen and compared with a balanced
allocator under the same replay queue.

The tested electrode counts and trial counts have different physical units.
EP08 may compare how performance changes along each axis of the frozen grid,
but it may not declare an electrode equivalent to a number of trials without a
separate, prespecified cost model.

The possible scientific outcomes are:

| Result | Interpretation and next step |
| --- | --- |
| Electrode and trial choices both add reproducible gain, and pilot-only scores predict their later conditional value | Test a joint acquisition principle in a newly sequestered third animal |
| Only electrode retention helps | Focus the claim on nonredundant sensor selection; balanced trial collection remains adequate |
| Only trial allocation helps | Focus the claim on calibration sampling; a simpler electrode rule is sufficient |
| Reliability alone explains the selected set | Report a quality-control result, not a new spatial or information-selection principle |
| A policy wins one grid cell, session, or animal only | Narrow the scope or stop; do not average away the failure |
| No policy clears the primary rule | Report the bounded negative or unresolved result; do not search post hoc for a mechanism story |

## Authority and history boundary

This is the current local contract. An explicit scientist task may start
bounded episode work, but it does not expose held-out neural outcomes or
establish a result. Earlier LFP access must be recorded, and no prior score,
preferred subset, threshold, or conclusion may initialize the search. Data
roles, search rules, and the configuration lock must be fixed before
candidate-discriminating development.

Because only the existing two primary animals are currently available, the
locked test described here is internal held-session evidence. A third,
lineage-independent animal is required for an external generalization claim.

## Exact first-round policy test

After one fixed all-electrode calibration pilot, can an acquisition policy
learn **which physical electrodes to retain and which reach-direction stratum to sample
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
electrode set after this pilot. It may then request only the next reach direction;
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
closes permanently. A trial action chooses only one of the eight reach-direction
strata; the trusted replay returns the next whole trial in that direction's
frozen hash order. The fixed movement epoch is part of every trial and is not a
separate action dimension.
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
   reach-direction strata. For adaptive allocators, reward is the change in
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
