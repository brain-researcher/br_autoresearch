# EP06 paper plan: is the M1–PMd fingerprint useful, or just recognizable?

Status: proposed study design, 2026-09-26. No real-data result, reserved-session test,
or third-animal test is claimed here.

## The question in plain language

For each recording session, we ask how well LMP and each registered LFP band
predict the nearby spike-population activity. This produces one frequency
profile for M1 and one for PMd.

The first question is whether the M1–PMd difference learned in Mihili labels
whole Chewie sessions, and whether the reverse direction also works. The second
question is whether the difference is useful: do frequency weights learned for
the correct region improve prediction of held-out population activity compared
with one pooled rule, while deliberately swapping the region labels makes
prediction worse?

The array is different in M1 and PMd, so the paper must always distinguish a
reusable recording-domain result from a pure cortical-area claim.

## What would be new?

| Prior work | What is already known | What EP06 must add |
| --- | --- | --- |
| Gallego-Carracedo et al., eLife 2022, [doi:10.7554/eLife.73155](https://doi.org/10.7554/eLife.73155) | LFP-to-population relationships in these recordings depend on frequency and differ across M1, PMd, and area 2. | Do not present the basic regional difference as new. Show a signed M1–PMd profile that transfers in both animal directions, survives recording controls, and predicts a benefit for the correct regional frequency rule. |
| Ince et al., PLOS ONE 2010, [doi:10.1371/journal.pone.0014384](https://doi.org/10.1371/journal.pone.0014384) | Frequency-dependent spatial LFP patterns in motor cortex carried reach-direction information. | Test a region-linked LFP-to-population profile rather than target direction, using whole-session cross-animal transfer. |
| Hall et al., Nature Communications 2014, [doi:10.1038/ncomms6462](https://doi.org/10.1038/ncomms6462) | Multichannel motor-cortical LFPs estimated neuronal firing and supported real-time biofeedback. | Ask whether the useful frequency rule differs reproducibly between the two recording domains; EP06 does not test online control. |
| Gallego et al., Nature Neuroscience 2020, [doi:10.1038/s41593-019-0555-4](https://doi.org/10.1038/s41593-019-0555-4) | Low-dimensional population dynamics can remain stable over long periods. | Directly test whether a region-linked LFP rule transfers across animals rather than inferring it from latent stability. |

Reproducing the source paper's regional plot, finding one good band, or
classifying folds from the same session would not be enough. Before asserting
novelty, compare the final analysis with the source paper, its supplements and
code, and later reports using the same dataset.

## Study sequence

### 1. Test whole-session transfer in both directions

Use four development sessions from each animal. Learn one complete M1–PMd
profile rule from Mihili and test all four Chewie sessions, then reverse. Fit
all scaling, population summaries, LFP-to-population models, reliability
weights, alignment, and classifier parameters on the permitted training side.

Report both balanced accuracies separately, their equal-weight mean, and every
session. Trials, time bins, electrodes, bands, and population coordinates are
measurements within a session; they do not increase the number of animals.

### 2. Show what the classifier recognized

Plot the signed M1-minus-PMd difference for every registered feature block and
every session. Any highlighted band must be chosen by a development-only rule,
not because it looks clean in reserved sessions.

The figure should let a reader answer:

- Does the same part of the profile have the same sign in both animals?
- Does the difference survive equal trial, electrode, population-rank,
  reliability, and model-capacity budgets?
- Does it remain without 100–400 Hz power and on matched electrodes and units?
- Can recording metadata or missing channels classify the sessions just as
  well?
- Does one session or one profile component control the result?

If only the final classifier score is stable, the biological interpretation is
unknown.

### 3. Ask whether the difference improves prediction

After the first result is fixed, use the source animal's M1 and PMd profiles to
create three fixed ways of combining frequency-specific predictions in the
other animal:

| Rule | Scientific question |
| --- | --- |
| Correct region | Do source-learned M1 weights help target M1 and source-learned PMd weights help target PMd? |
| Pooled | Would one common frequency rule work just as well for both regions? |
| Swapped region | Does deliberately giving each region the other region's weights hurt? |

Fit each band's base predictor once on the allowed target training trials.
Then combine the same held-out bandwise predictions with the three fixed weight
sets. No rule receives extra coefficients, different trials, extra electrodes,
or a different population dimension.

The proposed explanation predicts a meaningful correct-minus-pooled advantage
in both animal directions and worse performance after the weights are swapped.
If the regional weights are almost identical, or the bandwise predictions are
too similar for the mixtures to differ, call the test inconclusive. That is not
evidence that a pooled biological rule is sufficient.

### 4. Use the reserved sessions once

The two reserved sessions from each animal can test the final classifier and
the prediction consequence only if both results are specified before they are
opened. Reveal all four sessions together. The fixed recipe may fit its planned
session-specific base models on training trials, but it cannot change feature
support, model family, weights, thresholds, or exclusions after any reserved
result is visible.

Because related work has already exposed the release, this is a procedural
within-corpus test rather than independent confirmation.

### 5. Carry the unchanged rule to a third animal

A later third-animal study must use the same profile construction, classifier,
frequency rules, thresholds, and support. It is a separately planned study and
must report every compatible session. If no suitable simultaneous M1/PMd source
exists, the paper must retain its two-animal scope.

## Rival explanations and decisive checks

| Rival explanation | Direct check | If it wins |
| --- | --- | --- |
| Ordinary reach direction explains the result | Direction-and-time baseline and whole-trial LFP-to-population pairing shuffle | No evidence for the proposed regional population relationship |
| Array quality reveals the region label | Metadata-only classifier and matched trial, channel, rank, reliability, and missingness support | Recording or hardware fingerprint; no neural regional interpretation |
| High-frequency spike leakage creates the profile | Low-frequency-only analysis, high-frequency exclusion, and matched electrode/unit checks | Limit the result to a spike-rich recording feature |
| Flexible population summaries or models create separability | Capacity-matched alternatives and smoothness/dimensionality-matched surrogate activity | No profile-specific evidence |
| Frequency labels leak the answer | Verify feature guides and shuffle band labels through the full analysis | Invalid regional interpretation |
| One animal or session drives the score | Both transfer directions and leave-one-session analysis | Narrow or reject the transfer claim |
| Correct-region weighting has more freedom | Apply all three fixed weight sets to the same bandwise predictions | No benefit attributable to the regional rule |
| The three mixtures cannot meaningfully differ | Check weight separation and diversity of the bandwise predictions | Consequence is inconclusive |

Matching measured hardware features does not remove unmeasured array effects.
The strongest wording remains about the two recording domains unless a future
design breaks the region–array link.

## When to deepen, narrow, or stop

| Evidence | Next action and permitted claim |
| --- | --- |
| Both classification directions pass, the signed contrast repeats, the correct-region rule beats pooled, and the reserved sessions pass | Develop a reusable recording-domain organization claim with a concrete population-prediction consequence. |
| Classification transfers but correct-region does not beat pooled | Report a stable label with no demonstrated prediction benefit; stop the stronger interpretation. |
| The regional rules or bandwise predictions cannot be distinguished | Call the consequence inconclusive and seek a better-powered prespecified test. |
| The two animal directions disagree | Treat the result as animal-specific; do not average away the disagreement. |
| Recording metadata or matching explains the result | Reframe as a hardware or recording-quality result, or stop. |
| Only high-frequency or spike-rich features survive | State that boundary and stop the broad LFP claim. |
| The reserved sessions fail | Preserve the development result and the failed transfer; do not switch candidates. |
| No third-animal source exists | End with a two-animal scope and a concrete future-data requirement. |
| Precision is too low to distinguish meaningful effects | Call the result unresolved; a wide interval is not evidence for equivalence. |

## Paper figures, if supported by the evidence

| Figure | Scientific judgment | Evidence shown |
| --- | --- | --- |
| 1. Does the label transfer? | Whole-session M1/PMd discrimination works in both directions | Paired design, both directional accuracies, chance reference, every session |
| 2. What is the fingerprint? | A signed frequency-profile difference repeats | All bands and sessions, effect direction, reliability, session influence |
| 3. Neural profile or array artifact? | The difference survives its strongest measured alternatives | Metadata model, matched support, low/high-frequency and spike-contamination checks, surrogate and label shuffles |
| 4. Does the fingerprint help? | The correct regional frequency rule improves held-out population prediction | Correct, pooled, and swapped rules on the same bandwise predictions, both directions, inconclusive cases shown |
| 5. Where does it generalize? | The complete statement survives reserved sessions and, if available, a third animal | Every opened session, failures, third-animal result, explicit region–array limitation |

The abstract should state the signed profile difference, both animal-transfer
directions, the prediction consequence, the hardware limitation, and the scope
of replication. “M1 and PMd differ” is already too broad and too close to the
source publication to be the conclusion.
