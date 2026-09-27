# Which population components can the M1–PMd LFP fingerprint recover across days?

M1 and PMd were recorded at the same time while each animal performed the same
reaching task. In each region we have two views of local activity: LFP features
and spikes from nearby neurons. EP06 asks whether the relationship between those
two signals differs between M1 and PMd in a way that repeats across animals.

For each session and region, the primary analysis builds one frequency profile:
how well does LMP or each registered LFP band predict the held-out
**single-trial spike residual** after a direction-by-time mean learned on
training trials has been removed? This residual-only score is fixed before any
component analysis. We learn the M1-versus-PMd difference from Mihili and test
whole sessions from Chewie, then reverse the animals. Both directions must
work; a strong average cannot hide failure in one animal.

A classifier score is only the first result. Because each region is tied to a
different implanted array, a model might recognize missing channels, recording
quality, or spike leakage rather than a reusable neural relationship. The study
must therefore show the signed frequency difference, test the strongest array
and recording alternatives, and ask which kind of held-out population activity
the difference actually helps recover. A separate, unresidualized follow-up
asks whether the useful part is the common time course, the direction-specific
departure from that course, or trial-to-trial activity. It then makes a frozen
prediction for sessions held out across the acquisition dates.

## The scientific logic

![EP06 conceptual figure showing the residual-only regional fingerprint, three frozen population projectors applied to held-out observed and predicted activity, a calibration-visible support boundary, and held-out pass or fail predictions](outputs/ep06_question_imagegen.png)

Like the EP12 concept figure, this mockup makes the scientific alternatives
visible with synthetic patterns. It shows the M1 and PMd frequency profiles,
their reciprocal cross-animal transfer, and how correct, pooled, and swapped
weights distinguish a useful fingerprint from a transferable label alone. It
then shows three fixed projectors applied in the same way to observed and
predicted held-out activity, the calibration-visible boundary used before a
held-out score is seen, and the pass/fail prediction for sessions spread across
acquisition dates. These are illustrative patterns, not EP06 results; the
numerical definitions are given below.

Even the strongest positive result would remain a **recording-domain** result.
This dataset cannot separate cortical region from its implanted array. A third
animal can test whether the rule repeats again, but a design that breaks the
region–array link would be needed to claim a pure cortical-area effect.

No real EP06 analysis, reserved-session test, or third-animal test has run under
this plan.

## At a glance

| Question | EP06 design |
| --- | --- |
| What varies? | In the primary test, how well LMP and each registered LFP band predict single-trial spike residuals in M1 and PMd. |
| What is the first comparison? | Learn the complete M1–PMd profile difference in one animal and classify whole sessions from the other animal, in both directions. |
| What is held out? | Whole sessions from the other animal, followed by two separately reserved sessions per animal for one final test. |
| What must repeat? | The signed profile difference and classification must agree in both animal-transfer directions. |
| What could mislead us? | Region is tied to array, so hardware, missing channels, reliability, or high-frequency spike leakage may reveal the label. |
| What makes the result useful? | M1- and PMd-specific frequency weights must predict held-out population activity better than one pooled rule; swapping the regional weights should hurt. |
| What makes it explanatory? | A separate unresidualized projection must identify the same population component in both transfer directions, and a rule fixed on development sessions must predict its outcome in sessions held out across acquisition dates. |
| What remains unproven? | Pure cortical identity, causality, online BCI value, and generalization beyond these animals and this recording pipeline. |

## The first result: does the fingerprint transfer?

For each session and region, training trials define the spike-population axes
and the mean response for every reach direction and time bin. Subtract that
frozen mean from both training and held-out spike responses. For each registered
LFP feature block, fit a model on training trials to predict this residual and
score untouched whole trials. Trial identities, time bins, and population
coordinates are pooled within a session; the resulting held-out residual score
is one profile value for that session, region, and feature block.

The component follow-up described below cannot replace this residual score or
change the primary conclusion. In particular, a strong common time course does
not retroactively make a failed residual-transfer test positive.

The primary test has two equal parts:

1. learn the M1–PMd rule from Mihili's development sessions and classify
   Chewie's development sessions; and
2. learn the rule from Chewie and classify Mihili.

Report the two balanced accuracies separately and their equal-weight mean above
chance. Also show the signed M1-minus-PMd profile in every session. A result is
not convincing if the classifier transfers but the apparent frequency
difference changes sign between animals.

## The deeper result: does the fingerprint help prediction?

After the first result is fixed, use source-animal development sessions to fit
three fixed ways of combining the same bandwise predictions in the other
animal:

| Rule | What it asks |
| --- | --- |
| Correct region | Do source-learned M1 weights help target M1, and source-learned PMd weights help target PMd? |
| Pooled | Is one common frequency rule sufficient for both regions? |
| Swapped region | Does deliberately giving M1 the PMd rule and PMd the M1 rule make prediction worse? |

Source development data alone fit the M1, PMd, and pooled frequency weights.
The swapped rule is exactly the source M1/PMd weight permutation. A target
session may use its training trials to fit population axes, scaling, and one
base predictor per band, but it cannot refit the frequency-combining weights.
Calibration trials are used only for the support prediction described below;
evaluation trials are used only for the final score. All three rules therefore
use the same target trials, electrodes, axes, and bandwise predictions. The
proposed explanation predicts that the correct-region rule beats both pooled
and swapped rules. If the source weights are nearly identical, or the target
bandwise predictions are too similar for the rules to differ, the follow-up is
inconclusive rather than evidence for or against a regional effect.

This follow-up explains the first result; it is not a new animal replication
and cannot change whether the original classifier transferred.

## The paper-level prediction: which signal is recovered, and when?

This is a separate analysis of the **unresidualized** spike-population response.
Training trials freeze the population axes and three linear projectors:

1. **Common time, `P_C`:** the direction-invariant time subspace.
2. **Direction × time, `P_D`:** direction-by-time contrasts after the common
   time subspace has been removed.
3. **Trial residual, `P_E`:** the remaining within-direction, within-time trial
   activity.

The trial-space inner product gives every reach direction equal total weight,
every trial within a direction equal weight, and every analyzed time bin equal
weight. With sum-to-zero direction contrasts, `P_C` and `P_D` are orthogonal;
`P_E = I - P_C - P_D` is their orthogonal complement. Thus
`P_C + P_D + P_E = I` on the declared analysis space. The matrices depend only
on the frozen direction/time design and training-defined population axes, not
on evaluation spikes. Training fixes the direction set, time grid, contrast
coding, and axes; applying the operators to evaluation rows uses only their
preassigned direction labels and time bins.

For each projector, source-animal development sessions alone fit the
component-specific M1, PMd, and pooled frequency weights. Swapped is the exact
M1/PMd permutation. Target training fits only its axes, scaling, and bandwise
base predictors under the frozen recipe; it cannot update a combining weight.

For component `k`, the held-out target is `P_k Y_eval` and a rule's held-out
prediction is `P_k Yhat_eval`. Both are obtained by applying the same frozen
projector to genuine evaluation activity. The score is
`1 - sum||P_k Y_eval - P_k Yhat_eval||^2 / sum||P_k Y_eval||^2`, using the same
direction-balanced denominator for correct, pooled, and swapped rules. This is
a fraction-of-component-energy prediction score, not explained variance; it
may be negative.

A positive minimum component-energy gate is fixed from development and
known-answer synthetic tests before the final analysis. The gated quantity is
the target denominator divided by the total direction-balanced evaluation
weight, so a session does not pass merely by containing more trials. A
nonfinite score or normalized energy below that gate makes the
region/session/component unscorable and therefore unresolved. It is never
converted to zero or failure. For every scoreable component, first report
`G_region = score(correct) - max(score(pooled), score(swapped))` separately for
M1 and PMd. Average the regions only when their `G_region` signs agree. A
component is preserved when the equal-region mean is at least 0.01 and both
regional gains are positive; it fails when both regional gains are at or below
zero. Opposite signs or an average between 0 and 0.01 are unresolved.

For each transfer direction, an outer leave-one-target-session-out loop tests
the selection rather than merely recomputing it. In each fold, the other three
target development sessions nominate the component with the largest median
equal-region gain. Every candidate must be scoreable with sign-consistent
regional gains in all three nomination sessions; top medians within 0.01 are a
tie. If any candidate fails this eligibility check, that fold has no nominee.
Only a valid prefold nominee receives a frozen support prediction and regional
evaluation gains in the left-out session. The fold does not inspect the other
two evaluation components. After all four folds are recorded, the previously
unopened development components may
be scored for a final nomination using all four sessions; they cannot alter the
outer-fold record. A component claim requires the same nominee in at least
three of four outer folds, no tied fold within the 0.01 nomination margin, and
no M1/PMd sign contradiction. Otherwise selection is unstable. The two transfer
directions must then independently make the same untied final nomination, and
both are frozen before any reserved-session outcome is opened.

The prediction for each reserved session is made from two quantities visible in
its calibration trials: split-half reliability of the nominated component's
frequency profile, and its standardized distance from the source-animal
profile. For a cross-fitted development prediction, pool the other seven
development sessions to set the reference median and scale; for a reserved
prediction, use all eight. Standardization subtracts that median and divides by
1.4826 times the median absolute deviation. If either scale is zero, the rule
abstains rather than inventing a cutoff. A zero or nonfinite bandwise
between-session scale in the profile-distance calculation, or a nonfinite
split-half reliability, also forces an abstention. The single support index is
`H = standardized reliability - standardized profile distance`; its equal
weights are fixed, not learned from eight session outcomes. `H >= 0.25` predicts
preservation, `H <= -0.25` predicts failure, and the interval between them is a
declared abstention. Every reserved prediction is written down before its
held-out trials are scored.

The main competing explanation is generic recording quality. It predicts that
all three component families rise or fall together and that recording metadata
predicts the regional gains and preservation/failure status at least as well as
the component-specific support index. If that pattern wins, EP06 has a
transferable recording label, not a new fact about population organization.

There are only four reserved sessions, and they may not populate both sides of
the support boundary. That is enough to falsify a bad prediction but not to
estimate a transferable boundary precisely. Unless separately sealed sessions
or a third animal provide adequate precision, this component result is planned
as part of EP07 rather than as a stand-alone paper.

Report both regional gains, the resulting session status, each support
prediction, and every abstention. Resample whole trials within a session to
show measurement uncertainty, then resample sessions within each animal and
average the two animal-transfer directions equally. Also report each direction
and every leave-one-session result separately. Trial bins, electrodes, and
population coordinates never count as independent biological replicates. For
the support rule, report the exact number correct, wrong, and abstained with an
interval over sessions. A within-corpus prediction counts as supported only if
at least three of the four reserved sessions receive a
non-abstaining prediction and none is wrong. Even four correct predictions from
only one outcome class do not establish the boundary. Wide uncertainty is an
unresolved result, not confirmation.

## Evidence roles and the session split

The current release reports six simultaneous M1/PMd sessions for Mihili and six
for Chewie-L. After structural eligibility is checked, order each animal's six
sessions by acquisition date, using the provider session name only to break a
tie. The roles are fixed by position:

| Animal | Development and model selection | Reserved final test |
| --- | --- | --- |
| Mihili | positions 1, 2, 4, and 5 | positions 3 and 6 |
| Chewie-L | positions 1, 2, 4, and 5 | positions 3 and 6 |

Record the resulting session names in `DATASETS.md` before any regional score is
inspected. Do not move a session because its result is inconvenient. If either
animal has fewer than six structurally eligible paired sessions, stop for a
scientific review of sample size instead of reallocating sessions after looking
at outcomes.

The eight development sessions support model comparison. The four reserved
sessions are opened together once after one analysis is final. Because related
historical work already exposed this release, that reserved-session test checks a fixed procedure
within the same corpus; it is not independent confirmation.

Chewie-R is another implant in the same animal, not a third animal. Han and
Lando contain area-2 recordings under a confounded animal/task design and may be
used only for a clearly labelled recording-domain sensitivity analysis. A true
external transfer test requires a separately sealed third animal with
simultaneous M1 and PMd recordings.

## Models we will compare

Every trial uses one rule for all sessions. The bounded choices are:

1. **Time window:** the primary execution window is 150–450 ms after movement
   onset; a small behavior-defined preparation/execution set may be compared.
2. **LFP features:** LMP and the registered 0.5–4, 4–8, 8–12, 12–25, 25–50,
   50–100, 100–200, and 200–400 Hz bands, either alone, in registered contiguous
   groups, or as the full profile.
3. **Electrode summary:** the same 15-electrode budget in both regions, using a
   mean, median, training-only reliability weighting, or training-only PCA.
4. **Spike-population summary:** training-only PCA or reduced-rank summaries
   with a finite dimension schedule supported in both regions.
5. **LFP-to-population model:** ridge, reduced-rank regression, PLS, or
   regularized CCA, always evaluated on held-out whole trials.
6. **Profile comparison:** centered shape, reliability-adjusted shape, or a
   fixed combination of shape and overall magnitude.
7. **Across-session alignment:** none, training-session robust scaling, or a
   training-only orthogonal alignment.
8. **Region classifier:** nearest prototype, shrinkage Mahalanobis, or
   regularized logistic classification.
9. **Ensemble:** at most two already evaluated complementary rules.

Band edges, reserved-session-specific alignment, separate winners for each animal,
unpaired M1/PMd trials, electrodes chosen from held-out outcomes, raw-phase
analyses, and arbitrary new model families are outside this study.

## What a convincing result must survive

- classification above the complete-analysis label-shuffle reference in both
  animal-transfer directions;
- the same signed M1–PMd profile direction in both animals;
- leave-one-session analysis showing that no single session determines the
  result;
- paired trials and equal trial, electrode, spike-rank, reliability, and model
  capacity in the two regions;
- a model using recording metadata alone;
- low-frequency-only, high-frequency-exclusion, and same-electrode/unit checks;
- a whole-trial LFP-to-spike pairing shuffle;
- direction-and-time behavioral baselines;
- known-answer checks that the three projectors are orthogonal under the
  declared weights, sum to identity, and do not change under direction-count
  imbalance;
- source-only combining-weight provenance, the exact M1/PMd swap, and no target
  weight refit;
- the positive component-energy gate on every reported regional gain;
- the fixed unresidualized component-family ordering in both animal-transfer
  directions and the frozen support prediction on sessions held out across
  acquisition dates;
- band-label shuffles and smoothness/dimensionality-matched surrogate
  population activity; and
- the area-2 sensitivity reported without claiming a three-region hierarchy.

Failure of a required control blocks the neural regional interpretation even if
the headline classifier accuracy is high.

## Study sequence

1. Complete a focused novelty check against the source article, supplements,
   code, and later work using these recordings. If component-specific
   cross-day recovery is already established, or cannot be distinguished from
   EP07, plan EP06 as part of EP07 rather than as a separate paper.
2. Confirm the source, the paired M1/PMd trials, events, electrodes, units, and
   frequency labels. Run small known-answer tests for indexing, held-out
   prediction, and the planned shuffles.
3. Write the session-role table from the predeclared acquisition-order rule.
4. Run the fixed low-frequency, full-profile, ridge, CCA, prototype, and
   regularized-classifier starting comparisons.
5. Change one scientific choice at a time. Every trial records its question,
   full analysis choices, per-session results, controls, runtime, and failure
   reason. The current best model is provisional.
6. Retest each finalist on all eight development sessions, both transfer
   directions, and all required controls.
7. Challenge the best model and at most three alternatives with session
   influence, matching, frequency, spike-contamination, time-window, mapping,
   and profile ablations.
8. Cross-fit the component-family nomination and the one-dimensional support
   rule. Record the 0.01 component margin, the `H` thresholds, every required
   output, and one final analysis before opening the reserved sessions.
9. Open all four reserved sessions together and run that analysis once. No
   candidate, exclusion, or threshold may change after a reserved result is
   visible.
10. If a suitable third animal later becomes available, test the unchanged rule
   in a separately planned study.

At least 40% of valid trials after the starting comparisons must be controls,
ablations, influence checks, known-answer recovery tests, or direct repeats.
At least two evidence-led successor trials and two explicit best-model versus
challenger decisions are required before the final analysis is chosen.

## Budget and stopping

- minimum valid scientific trials: **20**;
- maximum valid scientific trials: **48**;
- after the minimum, stop after **10** valid trials without a meaningful
  improvement that also passes the controls;
- CPU ceiling: **1,200 core-hours**;
- GPU ceiling: **0 GPU-hours**;
- wall-clock ceiling: **96 hours** from the first scientific trial;
- finalists: at most **4**;
- reserved-session evaluations: **1**;
- maximum parallel CPU cores: **32**;
- memory ceiling per trial: **128 GB**; and
- scratch-storage ceiling: **750 GB**.

Source checks, role assignment, small known-answer tests, and a retry after a
proven technical failure do not count as scientific trials, but they still use
the resource budget.

## How the study can end

| Result | Meaning |
| --- | --- |
| One final rule passes development, every required control, and the reserved-session test | Supports only the bounded two-animal, within-corpus recording-domain claim. |
| The label and average prediction benefit transfer, but the component family or frozen support prediction does not | Keep EP06 as a bounded companion result for EP07; do not claim a separate population-organization paper. |
| The component prediction is correct but only four reserved sessions are available | Report the exact sessions and uncertainty within EP07; do not claim that the support boundary is independently established. |
| A valid search finds no rule that passes, or the final rule fails on the reserved sessions | No supported candidate under this design. |
| Development finishes but the reserved sessions cannot be opened | Report the development result only; do not imply confirmation. |
| The source, pairing, feature meanings, or scoring cannot be established | The scientific question was not tested. |
| Outcomes influenced the split, model choices, or later adaptation | Treat the comparison as invalid, not as scientific evidence. |

## Claim boundary

A successful reserved-session test supports a reproducible recording-domain
fingerprint across Mihili and Chewie in this task and feature pipeline. The
correct-versus-pooled-versus-swapped comparison is additionally required to say
that the fingerprint has a useful population-prediction consequence. The
unresidualized component analysis may explain that consequence, but it cannot
rewrite the residual-only primary result. A separate population-organization
paper would additionally require the same nominated family, a frozen support
prediction that succeeds on adequately many sessions spanning both sides of
its boundary, and uncertainty that excludes a chance-level rule. The four
currently reserved sessions do not provide that precision by themselves.

Neither result establishes that cortical area caused the difference, that a
particular band has a unique biological origin, that the relationship works
online, or that it generalizes to raw LFP, other behaviors, species, or
recording technologies. A separately sealed third-animal test is required for
any claim beyond these two animals, and even that would not by itself separate
region from array hardware.
