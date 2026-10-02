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

## The scientific logic

![EP05 conceptual figure showing same-target reach variation, seven-direction training, the correct-versus-wrong-trial control, and the conditional population follow-up](outputs/ep05_question_imagegen.png)

This schematic starts with two different reaches to the same target. It then
separates the direction/time baseline from the LFP branch and shows how pairing
a fixed prediction with the wrong reach tests trial identity. The population
follow-up is conditional on the kinematic result and cannot rescue a failed
first test. All trajectories, traces, and outcomes are illustrations, not EP05
observations; the study text defines the numerical decision rules. The
[generation and correction prompts](outputs/ep05_question_imagegen_prompt.md)
are saved, and the [earlier SVG](outputs/ep05_question.svg) is retained as a
historical design asset.

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
follow-up cannot change its answer. No real EP05 analysis has run under this
revised plan.

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
development sessions and a separate follow-up plan chosen in advance, derive a
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
| The gain disappears after held-out directions, whole-session selection, or a future-session test | The apparent decoder success did not establish a condition-general LFP signal. |

The follow-up uses the already assigned development evidence for explanation;
it is not a second independent replication. Before it runs, choose its models,
readouts, margins, multiplicity, budget, and stopping rule in advance. A future
sealed session then tests the final prediction once. No follow-up result may
rewrite the answer to the first kinematic test.

## First reproduce the published benchmark

Before trying new models, rerun the published M1 LFP-to-population and X/Y
velocity analyses with the reference paper code and small known-answer test
cases. Record whether the published result is reproduced, differs materially,
or cannot be run for a technical reason.

- Compatibility work, scheduler retries, and exact reproductions never count
  toward the 28 scientific trials.
- Neural outcomes may not be used to tune the reproduction target or tolerance.
- A mismatch is a reportable scientific limitation and blocks a positive
  adaptive candidate; it is not permission to optimize the benchmark.
- Search begins only after the source is technically evaluable and the
  reproduction result is recorded before search.

## Evidence roles and independence

- **Adaptive development/selection:** all structurally eligible M1 sessions in
  the historical Dryad release. The corpus is outcome-exposed from the earlier
  EP05 work and therefore exploratory. Candidate policies are assessed with
  nested whole-session outer folds; within each session, complete trials and
  entire directions remain grouped.
- **Selection unit:** one session estimate after collapsing X/Y, directions,
  trial folds, channels, and time bins. Animals are the biological replication
  level; Chewie implants remain nested in one animal.
- **Future-session test:** compatible future whole sessions whose neural outcomes
  and comparison scores remain unseen until the final analysis is chosen. New sessions from an existing
  animal test session robustness; a new animal is required to strengthen
  biological generalization. No such data are claimed to exist here.

Randomly withholding trials or renaming sessions from the exposed Dryad
release does not create independent confirmation.

## Models we will compare

Every scientific trial uses one analysis recipe for all sessions. A recipe may
choose from:

1. **LFP block:** LMP; one documented stored power band; a registered
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

## What counts as a convincing result

For each session, compute held-out-direction
`delta_R2 = R2_SSE(candidate full velocity) - R2_SSE(direction-by-time baseline)`
on identical complete trials. The primary objective is the animal-balanced
mean of session-level `delta_R2`, with the exact within-animal aggregation and
minimum practical gain chosen before search.

A promotable policy must:

- have a positive prespecified summary in both eligible M1 animals and the
  prespecified majority of their eligible sessions;
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

## How the study moves from first test to conclusion

1. **Check the source and the benchmark.** Confirm animal, session, event, trial,
   and feature identities; run small indexing and leakage tests; then record the
   publication-reproduction result.
2. **Set the splits before looking for a winner.** Write one readable
   animal/implant/session table and the whole-session folds. Within a scored
   session, fit every transformation on the allowed training sessions, trials,
   and directions only.
3. **Cover the main explanations first.** Compare the direction-and-time
   baseline, LMP, canonical low- and high-frequency blocks, a regularized linear
   model, a reduced-rank model, and one nonlinear reference.
4. **Improve one scientific idea at a time.** Each later trial changes one
   interpretable choice and records why it was tried, what it changed, and how
   every session responded. The current best model is provisional.
5. **Retest every finalist fairly.** Run each finalist on every eligible
   development session, both animals, every held-out direction, and all required
   controls.
6. **Challenge the strongest models.** Test dependence on one session, the
   number of trials and channels, high-frequency activity, timing, and the way
   the movement baseline is removed.
7. **Fix one final analysis.** Before any new session is opened, record the
   exact source sessions, splits, features, windows, model, thresholds, and
   required outputs in one dated plan.
8. **Test it once on sealed future sessions.** Session-specific coefficients
   may be fit by the fixed recipe, but the scientific choices cannot change
   after any future-session result is visible.

The current leading model cannot end the study early. The initial model set,
alternative-explanation checks, stress tests, and the 12-trial patience period
must all be completed. At least 40% of trials after the initial model set must
test alternatives, remove model components, run negative controls, measure
session influence, recover known answers, or directly repeat a result. Before
choosing the final model, complete at least two rounds in which the next model
is motivated by the preceding development result, and compare the leader with
a plausible challenger at least twice.

## Checks that can overturn a positive result

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

A failed required alternative-explanation check blocks promotion rather than
becoming another quantity to optimize.

## Budget and stopping

- minimum valid scientific trials: **28**;
- maximum valid scientific trials: **64**;
- patience after the minimum: **12** valid trials without material constrained
  primary improvement;
- CPU ceiling: **2,000 core-hours**;
- GPU ceiling: **0 GPU-hours**;
- wall-clock ceiling: **120 hours** from the first scientific trial;
- finalists: at most **4** including the current leading model;
- future-session evaluations: **1**;
- maximum parallel CPU cores: **32**;
- per-trial memory ceiling: **128 GB**;
- scratch-storage ceiling: **750 GB**.

Reproduction, structural QC, small known-answer tests, and exact infrastructure
retries are recorded separately and do not inflate trial depth. The search may
stop before 28 trials only for resource exhaustion, technical impossibility,
or policy violation.

## How the study can end

- **Supported in a future-session test:** the publication benchmark is
  reproduced, one global analysis passes every development check, and its
  prediction succeeds when evaluated once on sealed compatible sessions.
- **No supported candidate:** the study is technically valid, but no model
  survives the comparisons and controls, or the final model fails on the
  future sessions.
- **Development result only:** development finishes on the previously exposed
  public recordings, but no sealed future session is available. The result is
  explicitly exploratory.
- **No scientific answer:** the source, event meanings, session eligibility,
  reproduction, or scoring cannot be established reliably, or outcomes were
  used in a way that invalidates the planned comparison.

## Claim boundary

Development success supports an exploratory statement about stable predictive
information within this released M1 corpus. A successful future-session test
supports the final analysis's session robustness only for represented animals
and task; population-level, cross-region, real-time, raw-LFP, causal, and
cross-session weight-transfer claims remain out of scope. A new-animal test
is required before extending biological generalization.

The stronger paper interpretation—compact LFP features track the
trial-varying motor-population state that explains departures from an average
reach—requires the separately planned spike-latent and residual-error evidence
in the paper plan. A positive `delta_R2` alone does not establish that
interpretation. Conversely, a useful LFP-to-spike association cannot rescue a
failed primary kinematic test. No real EP05 analysis, future-session test, or
follow-up was run while preparing this revision.
