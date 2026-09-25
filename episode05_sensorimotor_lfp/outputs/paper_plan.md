# EP05 paper plan: can LFPs explain why this reach differs from the average?

Status: proposed study design, 2026-09-24. No real-data result, prospective
audit, or working follow-up executor is claimed here.

## The paper question in plain language

In an eight-direction reaching task, direction and elapsed time can give a
strong guess of where the hand should move. A decoder can reproduce that
template and still know little about an individual reach.

EP05 asks a harder question: after subtracting a cyclic direction/time template
fit on the seven permitted directions, can M1 LFP features predict that one
reach is a little faster, later, or more curved than the template—even when its
direction was withheld from model fitting? No held-out-direction trial is used
to calculate that template. The exact circular basis and interpolation rule
must be frozen before search; otherwise this comparison is undefined.

If the answer is yes, the paper should then explain what the useful LFP signal
tracks. The preferred explanation to test is that a compact LMP or
low-frequency representation follows the same trial-varying motor-population
state that is visible in simultaneously recorded spikes. The serious rivals
are task timing, a few idiosyncratic sessions or electrodes, high-frequency
spike contamination, and flexible model fitting.

The adaptive search in [GOAL.md](../GOAL.md) answers the first question. The
explanatory work below is a separate fixed-comparison follow-up. It cannot
alter the primary `delta_R2`, winner, terminal class, or audit rule.

## What would and would not be new

| Prior work | What is already known | What EP05 would still need to add |
| --- | --- | --- |
| Gallego-Carracedo et al., eLife 2022, [doi:10.7554/eLife.73155](https://doi.org/10.7554/eLife.73155) | In these recordings, LFP features relate to low-dimensional spike-population dynamics; that relationship depends on frequency and cortical region and was similar across planning and execution. The paper and code also provide the direct reproduction target. | Show that LFPs predict **trial-specific continuous velocity left over after direction and time**, on withheld directions, under one cross-session policy; then connect the gain to held-out fluctuations in the spike-population state rather than merely restating an LFP–spike association. |
| Flint et al., Journal of Neural Engineering 2013, [doi:10.1088/1741-2560/10/5/056005](https://doi.org/10.1088/1741-2560/10/5/056005) | Two monkeys used LFP-based decoders for stable online cursor control; offline prediction of hand movement could still vary. | EP05 uses a different, offline dataset and cannot claim online control. It must show condition-general residual prediction and state exactly what survives across sessions and animals. |
| Wang et al., Journal of Neural Engineering 2014, [doi:10.1088/1741-2560/11/3/036009](https://doi.org/10.1088/1741-2560/11/3/036009) | Motor-cortical LFPs can retain directional and kinematic decoding value over long periods, but preferred directions and signal quality can change. | Separate a pipeline policy that is stable across sessions from session-specific fitted coefficients, and test whole-session influence rather than presenting the best session or feature band. |
| Gallego et al., Nature Neuroscience 2020, [doi:10.1038/s41593-019-0555-4](https://doi.org/10.1038/s41593-019-0555-4) | Low-dimensional cortical population dynamics can remain stable across time and support kinematic decoding despite turnover in recorded units. | Test whether the useful LFP residual is aligned with those trial-varying population dynamics in this task; do not infer this merely because both signals decode movement. |

This table is a starting benchmark, not an exhaustive novelty review. Before a
paper claim is written, record for the retained candidate: the closest decoder
and LFP–population studies, the exact overlap, the new statement, the strongest
rival explanation, and the observation that would refute it.

The following outcomes are **not** enough novelty by themselves:

- reproducing the publication's M1 X/Y decoder;
- reporting that one more model or frequency block wins on the same exposed
  sessions;
- showing an ordinary LFP–spike correlation;
- decoding target direction that is already known to the behavioral baseline;
  or
- calling a cross-validation fold an independent replication.

## Study sequence

### 1. Establish trial-specific kinematic information

Run the existing primary contract unchanged. For every evaluated session and
held-out direction, fit the frozen cyclic direction-by-time function only on
the other seven directions, then evaluate it at the held-out angle. This is an
interpolated behavioral template, not the held-out direction's empirical
average. Compare it with the candidate on exactly the same complete evaluation
trials, retain negative `R2_SSE`, collapse to one session estimate, and
aggregate with Mihili and Chewie balanced.

The primary result must show where the gain occurs, not only its average:

- observed and predicted X/Y trajectories for examples selected by a
  development-only rule;
- all-session `delta_R2`, with both animals and leave-one-session influence;
- along-path speed error and perpendicular/curvature error from the same X/Y
  residual; and
- time-resolved error using bins fixed before candidate-discriminating search.

These component plots explain the fixed primary outcome. They do not become
alternate objectives and cannot select the winner.

Also quantify how much `delta_R2` depends on pairing each LFP trajectory with
the correct behavioral trial. With all fitted transforms and predictions held
fixed, permute whole predicted trajectories only among evaluation trials of the
same held-out direction. For each session report

```text
trial_identity_gain = observed_delta_R2
                      - frozen_center(null_delta_R2).
```

Freeze the permutation count, null center, practical margin, session/animal
aggregation, multiplicity, and invalid-cell rule before use. This removes the
part of the score obtainable from a direction-wide mean correction. A positive
primary `delta_R2` may still be reported if this increment fails, but the paper
must then say “held-out-direction prediction” rather than “trial-specific
information.”

### 2. Ask whether the useful LFP signal tracks population activity

After the primary development result is fixed and before any prospective audit
is opened, freeze one explanatory comparison. Use the primary locked LFP
representation without follow-up tuning. Construct the spike-population latent
inside each training fold only; freeze its rank schedule, smoothing, temporal
support, alignment, and regularization before inspecting explanatory scores.

Evaluate four models on the same held-out trials and directions:

| Model | Inputs beyond the behavioral baseline | Question answered |
| --- | --- | --- |
| B | none | How much does target direction plus elapsed time explain? |
| L | the locked LFP representation | Does LFP predict the individual reach? |
| S | the train-only spike-population latent | Does population activity predict the same deviation? |
| L→S | an LFP-predicted spike latent passed through the frozen S-to-velocity map | Can the measured population relationship account for the LFP velocity correction without fitting this path directly to evaluation behavior? |
| LS | both LFP and spike latent, with capacity controlled | Does LFP add information not already present in the measured population state, or mostly provide a compact proxy? |

The LFP representation must also predict held-out spike-latent fluctuations
beyond their own direction-by-time average. This is a prediction problem, not
a same-bin correlation computed after looking at all trials.

### 3. Test the prediction generated by that explanation

Let `r(t)` be observed X/Y velocity minus the training-derived cyclic template
evaluated at that direction and elapsed time. Let `r_L(t)`, `r_S(t)`, and `r_LSchain(t)` be the
residual-velocity predictions from L, S, and the L→S chain. Both stages of the
chain are fit on training trials; evaluation velocity is never used to fit or
align its intermediate latent.

The proposed explanation predicts all three of the following on held-out data:

1. L predicts the spike latent beyond direction and time.
2. `r_L(t)`, `r_S(t)`, and `r_LSchain(t)` improve prediction of the observed
   `r(t)` under their separately frozen margins.
3. `r_L(t)` and `r_LSchain(t)` make the same signed correction: when a reach is
   faster than its template, both predict a positive along-path correction;
   when it bends to one side, both predict the same perpendicular correction.

Record the mean held-out dot product between `r_L(t)` and `r_LSchain(t)` after
training-only scaling, separately for along-path and perpendicular components.
A positive dot product alone is not success: each model must also predict the
observed residual and survive the fixed nulls. LS-versus-S and LS-versus-L
comparisons state whether the information is redundant or complementary; they
do not turn “uniqueness” into a causal claim.

Use a whole-trial pairing null within direction, registered temporal offsets,
and capacity-matched features. The null must rerun every fitting step that
could learn the pairing. Freeze the meaningful gains, simultaneous intervals,
number of permutations, and joint decision rule before explanatory outcomes.

### 4. Test the locked prediction in future sessions

If a compatible future-session audit is available, add the explanatory
readouts to the output schema **before** the single primary configuration lock.
Open all sealed audit sessions once. Session-specific coefficients may be fit
under the locked recipe, but the LFP block, spike-latent rule, decomposition,
thresholds, exclusions, and model comparisons cannot change.

New sessions from Mihili or Chewie test session robustness. Only a separately
declared and sealed new animal can strengthen biological generalization. If no
future sessions exist, finish as `search_exhausted_no_audit`; the exposed Dryad
corpus cannot be relabeled as confirmation.

## Decisive controls and what failure means

| Rival explanation | Direct test | Interpretation if it wins |
| --- | --- | --- |
| The decoder repeats the average trajectory | Direction-by-time baseline; withheld directions; observed-versus-template residual plots | No trial-specific LFP result |
| LFP only corrects a systematic seven-direction interpolation error | Observed `delta_R2` minus the frozen within-direction whole-trial pairing-null center | Retain a held-out-direction decoder result, but drop the trial-specific claim |
| A few sessions or one animal drive the mean | Both-animal sign requirement and leave-one-session influence | Narrow or reject the stable-policy claim |
| High-frequency power is filtered spiking | Low-frequency/LMP-only analysis, 100--400 Hz exclusion, and same-electrode/unit-intersection sensitivity | Describe a spike-rich recording feature, not a general LFP mechanism |
| Timing or leakage drives both signals | Whole-trial pairing, temporal-offset, indexing, and train-only transform fixtures | Invalid explanatory association |
| More parameters create the gain | Matched trials, channels, ranks, effective capacity, and random/phase surrogate | No evidence for the proposed representation |
| Good spike prediction does not matter for behavior | Require S and L to predict the observed kinematic residual, not merely each other | LFP–population association without the proposed movement consequence |
| Offline processed features look causal or real-time | Preserve source filtering and temporal-support audit; label every result offline | No real-time or causal BCI claim |

## When to deepen, narrow, or stop

| Evidence | Next action and permitted claim |
| --- | --- |
| Primary LFP gain, shared LFP/spike corrections, low-frequency robustness, and prospective transfer | Develop the claim that compact released LFP features track a trial-varying motor-population state useful beyond the average reach, within the recorded task and animals. |
| Primary LFP gain but no held-out spike-latent relationship | Report a bounded decoder result; do not tell a shared-population-state story. |
| Primary `delta_R2` is positive but trial-identity gain misses its margin | Report correction beyond the cyclic template without claiming that the right LFP trial predicts the right reach. |
| LFP and spike latent predict each other but LFP does not improve the primary residual outcome | Report the relationship if independently useful, but EP05's kinematic question is negative. |
| Gain survives only in high-frequency or spike-rich features | State that boundary explicitly; stop the broad low-frequency LFP direction. |
| Development succeeds but the one-shot future-session audit fails | Report the exposed-corpus result and failed transfer; do not swap a candidate or open another audit. |
| Intervals are wide or no meaningful-margin sensitivity is established | Call the result unresolved; lack of significance is not evidence that LFP adds nothing. |
| No sealed audit exists | End the current search without an audit claim; specify, but do not simulate, the needed future acquisition. |

Before the explanatory round runs, its own contract must specify support rules,
latent construction, fixed models, meaningful margins, uncertainty,
multiplicity, null count, compute budget, stopping events, and result labels.
The primary search's 28--64 trial budget does not automatically pay for this
follow-up.

## Paper figures, contingent on the evidence

| Figure | Scientific judgment | What must be shown |
| --- | --- | --- |
| 1. Beyond the average reach | LFPs predict trial-specific velocity on withheld directions | Cyclic template, observed individual trajectories, locked prediction, all-session `delta_R2`, trial-identity gain and its pairing-null distribution, negative values, both animals |
| 2. What information was added | The gain concerns speed/curvature deviations and has a specific temporal and feature source | Along/perpendicular residuals, frozen time bins, LMP/low/high ablations, matched channel and trial budgets |
| 3. Link to population activity | LFP and spike-population state predict the same held-out deviations | B/L/S/L→S/LS contrasts, LFP-to-latent score, signed correction agreement, temporal and pairing nulls |
| 4. Does it survive another session? | The whole policy and explanatory prediction transfer as locked | Every sealed session, no partial reveal, effect sizes and intervals, failure accounting |
| 5. What this changes for BCI design | A compact offline feature set has a defined benefit and boundary | Performance versus channel/feature budget, session influence, spike-bleed boundary, explicit online/raw-signal limits |

The abstract should eventually say what was predicted beyond the behavioral
template, which LFP features carried it, whether it tracked the spike
population state, where it replicated, and which BCI claim remains untested.
A better model score by itself is not the paper conclusion.
