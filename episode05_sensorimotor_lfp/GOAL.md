# Does the LFP explain why one reach was different?

A monkey can reach to the same target many times without making exactly the
same movement. One reach may be faster, another may start later, and another
may bend farther to the side. Target direction and time since movement began
can describe the typical reach, but they cannot explain why two trials to the
same target differ.

EP05 asks whether the M1 LFP recorded on an individual trial contains that
missing information. Within each session, we train on seven of the eight reach
directions and test on the eighth. From the seven training directions, we
estimate the movement expected from direction and elapsed time. The LFP model
must then predict how the actual held-out reach departs from that expectation.

The important comparison is not whether LFP can reconstruct hand movement at
all. A decoder can look accurate simply by repeating the typical trajectory.
The first test asks whether adding LFP improves the prediction of the complete
X/Y velocity trace when that direction was absent from the session's fit.

Even that improvement may not belong to individual trials. Suppose every
reach in the held-out direction curves a little more than the seven-direction
estimate. The LFP model could learn that common correction without knowing
which neural recording belongs to which reach. We therefore deliberately pair
each LFP prediction with a different reach to the same target. If the score
barely changes, the result is a direction-level correction. If the correct
pairing is clearly better in both animals, the LFP carries information about
the particular reach.

This is the entry point to a larger biological question:

> When a reach is faster or more curved than expected, does the useful LFP
> signal reflect the same moment-to-moment motor-population activity that is
> visible in spikes?

The intended paper should show more than a winning decoder. It should locate
the extra prediction in speed, curvature, and time; show whether it repeats
across sessions and animals; and distinguish a low-frequency population signal
from task timing, high-frequency spike contamination, or a few unusually good
recordings. If the first result holds, the next test asks whether the chosen
LFP features predict held-out changes in the simultaneously recorded spike
population and whether both signals predict the same correction to the reach.

Even a positive result would not establish that LFP causes the movement or
that this decoder is ready for online BCI control. It would support a narrower
claim: in these recordings, the LFP from the correct trial helps explain how
that reach differs from the movement expected from its direction and timing.

The [paper plan](outputs/paper_plan.md) compares this claim with prior work and
specifies the follow-up predictions, alternatives, and figure-level evidence.
The held-out-direction test remains the first result; the spike-population
follow-up cannot change its answer.

## At a glance

| Question | EP05 design |
| --- | --- |
| What varies? | The speed and shape of individual reaches after accounting for target direction and elapsed time. |
| What is compared? | A direction-and-time prediction versus the same prediction with information from that trial's LFP. |
| What is held out? | One entire reach direction at a time within each session. Neither model sees those trials while that session's model is being fit. |
| What must repeat? | The improvement must survive across sessions and be present in both animals, not come from one favorable recording. |
| What checks the individual-trial claim? | Pair each LFP prediction with the wrong reach to the same target. Correct pairings must predict better than wrong pairings. |
| What can the first test conclude? | Whether LFP helps predict reaches in a direction held out from that session's fit, and whether that help follows the individual trial. |
| What would make a deeper finding? | Evidence that the useful LFP signal tracks the same trial-varying population activity seen in spikes and is not explained by timing or high-frequency spike contamination. |
| What is the next test? | Keep the selected LFP representation fixed, then fit the separate follow-up maps and ask whether it predicts held-out spike-population changes and the same speed or curvature correction. |

## From a prediction gain to a scientific finding

Consider two reaches to the same target. Their average direction and timing are
nearly identical, but one accelerates later and bends more. A decoder can look
good by reproducing the average trajectory without knowing anything about that
difference. EP05 first asks whether LFP features predict the difference itself.

If they do, the next analysis asks what the useful signal is. Using only
development sessions and a separately frozen follow-up contract, derive a
spike-population latent from training trials and ask whether the chosen LFP
representation predicts held-out fluctuations in that latent beyond the
latent's own training-derived direction-by-time mean. Pass the LFP-predicted latent through a separately
trained spike-latent-to-velocity map, then ask whether that chain and the direct
LFP decoder make the same signed correction on the same held-out trials and
time bins. The concrete readouts are along-path speed error,
perpendicular/curvature error, and a time course fixed in advance; they are not
new search targets.

Four explanations must remain distinguishable:

| Possible result | What it would mean |
| --- | --- |
| Low-frequency/LMP features predict both the spike latent and the same kinematic residual | The released LFP features contain a compact view of trial-varying motor-population activity that is useful beyond the average reach. |
| LFP improves velocity prediction but does not track the spike-population readout | Keep the engineering prediction result; do not claim that the mechanism is shared population dynamics. |
| The gain exists only in 100--400 Hz features or same-electrode spike-rich channels | Bound the result to a spike-contaminated/high-frequency recording feature; do not make a broad field-potential claim. |
| The gain disappears after held-out directions, whole-session selection, or a fresh audit | The apparent decoder success did not establish a condition-general LFP signal. |

The follow-up uses the already assigned development evidence for explanation;
it is not a second independent replication. Before it runs, its models,
readouts, margins, multiplicity, budget, and stopping rule must be frozen. A
future sealed session tests the locked prediction once. No result in the
follow-up may alter the original search ledger, winner, terminal class, or
audit opening.

## Reproduction infrastructure is not search

Before any adaptive trial, run a fixed publication-fidelity suite for the M1
spike-latent/LFP profile and M1 X/Y velocity decoder using the pinned paper
code, best-recoverable dependency boundary, and synthetic fixtures. Its output
is `pass`, `mismatch`, or `technical_failure`.

- Compatibility work, scheduler retries, and exact reproductions never count
  toward the 28 scientific trials.
- Neural outcomes may not be used to tune the reproduction target or tolerance.
- A mismatch is a reportable scientific limitation and blocks a positive
  adaptive candidate; it is not permission to optimize the benchmark.
- Search begins only after the source is technically evaluable and the
  reproduction result is frozen.

## Evidence roles and independence

- **Adaptive development/selection:** all structurally eligible M1 sessions in
  the historical Dryad release. The corpus is outcome-exposed from the earlier
  EP05 work and therefore exploratory. Candidate policies are assessed with
  nested whole-session outer folds; within each session, complete trials and
  entire directions remain grouped.
- **Selection unit:** one session estimate after collapsing X/Y, directions,
  trial folds, channels, and time bins. Animals are the biological replication
  level; Chewie implants remain nested in one animal.
- **Prospective audit:** compatible future whole sessions whose neural outcomes
  and comparison scores are sealed until lock. New sessions from an existing
  animal test session robustness; a new animal is required to strengthen
  biological generalization. No such data are claimed to exist here.

Randomly withholding trials or renaming sessions from the exposed Dryad
release does not create independent confirmation.

## Bounded scientific operator grammar

Every scientific trial is one declarative, session-invariant pipeline from:

1. **LFP block:** LMP; one authenticated stored power band; a registered
   low-frequency, high-frequency, or all-band block; or LMP plus one registered
   band block. Stored band identities come from guides, not numeric columns.
2. **Channel/electrode transform:** fixed matched-electrode mean/median,
   train-only channel PCA, train-only reliability weighting, or deterministic
   top-`k` selection from a finite `k` schedule using training trials only.
3. **Temporal representation:** current bin or causal-looking historical
   windows from a finite lag schedule. Because source features may contain
   offline/zero-phase processing, every result remains offline regardless of
   lag sign.
4. **Kinematic target:** X/Y velocity residual after a training-only cyclic
   direction-by-elapsed-time baseline; registered speed or acceleration
   targets are secondary diagnostic branches and cannot win the primary
   search.
5. **Feature transform:** train-only standardization, PCA, PLS, or a registered
   tensor product of LFP block and elapsed-time basis.
6. **Predictor:** ridge, elastic net, reduced-rank regression, PLS regression,
   linear kernel ridge, or RBF kernel ridge with a finite capacity schedule.
7. **Ensemble:** convex blend of at most two already evaluated complementary
   pipelines; weights are learned inside training sessions/folds.

Models must use the same eligible trials and score full velocity after adding
the neural residual prediction back to the baseline. Negative `R2_SSE` values
are retained. Arbitrary windows, per-session winner selection, test-direction
scaling, bin-level random splits, foundation-model training, and arbitrary code
mutation are forbidden.

## Objective and constraints

For each session, compute held-out-direction
`delta_R2 = R2_SSE(candidate full velocity) - R2_SSE(direction-by-time baseline)`
on identical complete trials. The primary objective is the animal-balanced
mean of session-level `delta_R2`, with the exact within-animal aggregation and
minimum practical gain frozen before search.

A promotable policy must:

- have a positive prespecified summary in both eligible M1 animals and the
  frozen majority of their eligible sessions;
- survive a within-direction whole-trial evaluation-pairing null with all
  fitted transforms held fixed;
- remain positive under leave-one-session influence and matched trial/channel
  budgets;
- show an increment beyond behavior-only and elapsed-time-only baselines;
- pass temporal-offset and feature-label negative controls; and
- not rely solely on 100--400 Hz power or disappear under the registered
  spike-bleed-through sensitivity.

Random-trial performance is a within-condition ceiling. Publication
`r_corr_squared`, phase atlases, speed/acceleration, and reduced-resource curves
are diagnostics and cannot rescue a failed held-out-direction objective.

## Multi-stage adaptive loop

1. **Infrastructure gate:** authenticate source/code, freeze eligibility and
   event conventions, pass synthetic index/leakage/metric fixtures, and freeze
   the publication reproduction result without search.
2. **Session-role freeze:** create immutable animal/implant/session manifests
   and nested whole-session folds. Within each outer session, all transforms
   and tuning use only the remaining sessions and that session's permitted
   training trials/directions.
3. **Coverage stage:** run direction-by-time, LMP, canonical low/high bands,
   linear regularized, reduced-rank, and one nonlinear anchor before adaptive
   exploitation.
4. **Adaptive stage:** make one mechanism-led operator change per trial;
   append hypothesis, parent, full configuration/hash, session predictions,
   objective, constraints, runtime, and failure reason. Maintain a nonterminal
   incumbent and complexity/robustness Pareto archive.
5. **Successive fidelity:** early pruning may use a frozen subset of inner
   development sessions. Every finalist is rerun across every eligible
   development session, both animals, all held-out directions, fixed seeds,
   and mandatory nulls.
6. **Stress stage:** incumbent plus at most three challengers undergo
   leave-one-session and leave-one-animal description, trial/channel matching,
   low-frequency-only, spike-bleed-through, temporal-offset, and target
   residualization ablations.
7. **Configuration lock:** choose one global policy, then hash source/split
   manifests, code, environment, operator DAG, feature guides, windows, model
   rule, thresholds, seeds, and output schema.
8. **One-shot prospective audit:** run the locked policy on all sealed audit
   sessions once. Session-specific coefficients may be fit under the locked
   training recipe; no policy choice, threshold, exclusion, or candidate may
   change after any audit score is visible.

The incumbent cannot terminate the loop early. Minimum coverage, falsifiers,
stress tests, patience, and audit-opening qualification are separate gates.
At least 40% of valid post-coverage trials must be falsifiers, ablations,
negative controls, influence guards, synthetic recovery, or direct
replications. At least two outcome-adaptive successor cycles and two recorded
incumbent/challenger decisions are required before lock.

## Mandatory falsifiers and ablations

- fixed publication-fidelity result and method fixtures;
- direction-by-time and elapsed-time-only behavioral baselines;
- within-direction whole-trial evaluation-pairing permutation;
- whole-trial LFP/behavior shuffle and registered temporal offsets;
- LMP-only, each registered band block, low-frequency-only, high-frequency-only,
  and all-band ablations;
- matched electrode/channel and trial-count analyses;
- same-electrode/unit-intersection spike-bleed-through sensitivity;
- train-only target-residualization versus full-target fit;
- leave-one-session influence and both-animal sign report; and
- capacity-matched random features or phase-randomized surrogate permitted by
  the released representation.

A failed mandatory falsifier blocks promotion rather than becoming another
optimizable metric.

## Budget and stopping

- minimum valid scientific trials: **28**;
- maximum valid scientific trials: **64**;
- patience after the minimum: **12** valid trials without material constrained
  primary improvement;
- CPU ceiling: **2,000 core-hours**;
- GPU ceiling: **0 GPU-hours**;
- wall-clock ceiling: **120 hours** from the first scientific trial;
- finalists: at most **4** including the incumbent;
- audit openings: **1**;
- maximum parallel CPU cores: **32**;
- per-trial memory ceiling: **128 GB**;
- scratch-storage ceiling: **750 GB**.

Reproduction, structural QC, synthetic fixtures, and exact infrastructure
retries are ledgered separately and do not inflate trial depth. The search may
stop before 28 trials only for resource exhaustion, technical impossibility,
or policy violation.

## Terminal classes

- `candidate_ready`: the reproduction gate passes, the locked global policy
  meets every development constraint, and it passes the prospective one-shot
  audit.
- `closed_no_candidate`: technically valid deep search finds no feasible
  policy or the locked policy fails audit.
- `search_exhausted_no_audit`: search completes on the exposed public corpus
  but no sealed future whole-session audit exists.
- `technical_failure`: source authenticity, guides/events, publication
  compatibility, session eligibility, or executable scoring cannot be
  established after bounded recovery.
- `policy_violation`: outcome-dependent benchmark tuning, session cherry
  picking, audit leakage, undeclared operators, or post-audit adaptation.

For canonical outer status, `candidate_ready` maps to `candidate_ready`;
`closed_no_candidate` and `search_exhausted_no_audit` map to
`closed_no_candidate`; and `technical_failure` or `policy_violation` map to
`technical_failure`.

## Claim boundary

Development success supports an exploratory statement about stable predictive
information within this released M1 corpus. A successful fresh-session audit
supports the locked policy's session robustness only for represented animals
and task; population-level, cross-region, real-time, raw-LFP, causal, and
cross-session weight-transfer claims remain out of scope. A new-animal audit
is required before extending biological generalization.

The stronger paper interpretation—compact LFP features track the
trial-varying motor-population state that explains departures from an average
reach—requires the separately locked spike-latent and residual-error evidence
in the paper plan. A positive `delta_R2` alone does not establish that
interpretation. Conversely, a useful LFP-to-spike association cannot rescue a
failed primary kinematic test. No real EP05 analysis, audit, or follow-up was
run while preparing this revision.
