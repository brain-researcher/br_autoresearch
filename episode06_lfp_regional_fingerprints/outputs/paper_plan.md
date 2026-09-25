# EP06 paper plan: is the M1–PMd LFP fingerprint reusable, or just an array label?

Status: proposed study design, 2026-09-24. No real-data result, internal audit,
third-animal data, or working follow-up executor is claimed here.

## The paper question in plain language

For each session, EP06 builds a profile: how well does LMP or each registered
frequency band predict the simultaneously recorded spike-population dynamics?
The first test asks whether the M1-versus-PMd difference in that profile is
consistent enough that a rule learned in Mihili labels whole sessions from
Chewie, and vice versa.

That classification is only an entry point. A classifier could recognize a
noisy array, a missing-channel pattern, or one high-frequency spike-rich band.
The paper should identify the signed profile difference and ask whether it has
a concrete consequence: does using source-learned **M1-specific** versus
**PMd-specific** frequency weighting improve recovery of held-out population
activity in the other animal, compared with one pooled rule? A deliberately
swapped-region rule supplies a direct negative control.

The estimand remains a recording-domain fingerprint because region and array
are tied together. Neither metadata adjustment nor a third animal with the
same layout proves pure cortical identity.

The adaptive classifier in [GOAL.md](../GOAL.md) and the explanatory
correct-region/pooled comparison below are separate rounds. The latter cannot
change the primary score, winner, audit, or terminal class.

## What would and would not be new

| Prior work | What is already known | What EP06 would still need to add |
| --- | --- | --- |
| Gallego-Carracedo et al., eLife 2022, [doi:10.7554/eLife.73155](https://doi.org/10.7554/eLife.73155) | In the source recordings, LFP-to-latent relationships were frequency dependent and differed across M1, PMd, and area 2; M1/PMd profiles were also similar between planning and execution. | Do not claim the basic regional difference as new. Show a locked signed M1–PMd contrast that transfers in both animal directions, survives the strongest recording confounds, and predicts a held-out advantage for correct-region rather than pooled frequency weighting. |
| Ince et al., PLOS ONE 2010, [doi:10.1371/journal.pone.0014384](https://doi.org/10.1371/journal.pone.0014384) | Spatial LFP correlation patterns in M1 and PMd decoded reach direction; information depended on frequency and was stable over about a week. | Distinguish region-linked LFP-to-population profiles, not target direction, and show whole-session/animal transfer under paired regional capacity. |
| Hall et al., Nature Communications 2014, [doi:10.1038/ncomms6462](https://doi.org/10.1038/ncomms6462) | Multichannel motor-cortical LFPs can estimate individual-neuron firing and were used for real-time biofeedback. | Test whether the frequency rule linking LFP to population activity differs reproducibly between the two recording domains; EP06 does not test online control. |
| Gallego et al., Nature Neuroscience 2020, [doi:10.1038/s41593-019-0555-4](https://doi.org/10.1038/s41593-019-0555-4) | Low-dimensional population dynamics can remain stable across long periods in sensorimotor areas. | Show that a region-linked LFP rule transfers across animals rather than inferring transfer from latent stability alone. |

This is a starting comparison, not an exhaustive review. Before asserting
novelty, compare the retained profile with the source paper's figures,
supplements, code, and any later report using the same data. Record which exact
bands, windows, animals, and transfer tests are already published.

The following are not sufficient contributions by themselves:

- reproducing the publication's frequency-by-region plot;
- classifying M1 and PMd in folds drawn from the same session;
- finding the highest-scoring band after trying many band labels;
- showing that array/QC metadata correlate with region; or
- calling the four outcome-exposed reserved sessions fresh replication.

## Study sequence

### 1. Establish whole-session, cross-animal transfer

Run the existing deterministic 4+2 contract unchanged. In development, train
the complete global profile and classifier policy on four Mihili sessions and
classify the four Chewie development sessions, then reverse. All scalers,
latents, LFP-to-latent mappings, reliability estimates, alignment, and
classifier parameters must be fit on the permitted training side.

Report the two directional balanced accuracies separately as well as their
mean. A high mean cannot hide one failed direction. Keep one estimate per
region/session until the animal-level summary; trials, time bins, electrodes,
bands, and latent coordinates are measurements, not additional sessions or
animals.

### 2. Show what the classifier actually recognized

For every registered feature block and latent readout, report the source-fitted
M1-minus-PMd contrast with its sign. The main profile plot must include every
development and, once opened, every audit session. Highlighted bands are
chosen by a source-development rule fixed before target scores, not because
they look clean in the held-out animal.

The contrast must answer simple questions a reader can inspect:

- Is the same band higher for M1 than PMd in both animals?
- Does the effect remain after equal electrode, trial, spike-rank, and
  reliability budgets?
- Is it present without 100--400 Hz power and on the same-electrode/unit
  intersection?
- Can array/QC/impedance/missing-channel metadata classify the sessions just
  as well?
- Does one session or one profile coordinate determine the decision?

If only the final classifier score is stable, the mechanism is unknown. The
paper must not invent a band story after the audit is opened.

### 3. Turn the contrast into a held-out prediction

After the primary development result is fixed, retain the objects that actually
generated its signed contrast. Let `P[r,b,l]` be the source-development,
session-balanced, out-of-fold, undifferenced region prototype in the exact
normalized/aligned profile representation supplied to the locked primary
classifier. Here `r` is region, `b` is band, and `l` is a prespecified common
latent coordinate. Use the primary pipeline's locked normalization, alignment,
latent rank, reliability rule, and support. Define

```text
D[b,l]      = P[M1,b,l] - P[PMd,b,l]
P[pool,b,l] = 0.5 * (P[M1,b,l] + P[PMd,b,l]).
```

`D` alone is not enough to recover two undifferenced profiles; the immutable output
must retain `P[M1]`, `P[PMd]`, `D`, and `P[pool]`. On the structurally frozen
common band set `B*` and latent set `L*`, apply the same parameter-free
band-marginal projection to each prototype:

```text
a[r,b] = mean over l in L* of P[r,b,l]
w[r,b] = exp(a[r,b]) / sum over b' in B* exp(a[r,b']).
```

Missing bands are handled by a source-independent structural support rule, not
by dropping whichever band performs poorly. Freeze nondegeneracy margins for
`||w[M1] - w[PMd]||`, correct-versus-pooled weight separation, and the resulting
target prediction-mixture separation before follow-up scores. If these margins
fail or the bandwise prediction tensor is too collinear to distinguish the
mixtures, run the comparison for completeness but label the consequence
non-identifiable; it cannot support a positive mechanism claim. `D = 0` must
make M1, PMd, and pooled weights identical, though not necessarily uniform; an
independent uniform-weight rule is reported as a reference. This projection
tests one prespecified band-marginal consequence of the primary profile, not
every latent-specific feature of `D`.

The resulting rules are:

| Rule | How it is learned | Explanation tested |
| --- | --- | --- |
| Fixed pooled-prototype | `w[pool]` from the equal-region source prototype | One common mixture is sufficient |
| Correct-region | `w[M1]` for target M1 and `w[PMd]` for target PMd | The source profile's band-marginal projection predicts which mixture is useful |
| Swapped-region | Exchange `w[M1]` and `w[PMd]`, changing nothing else | A direct falsifier for the regional interpretation |

For each `(target session, region, fold)`, fit one target-training latent basis
and outcome scaler, shared by every band and rule. Then fit each supported band's base
LFP-to-latent predictor once, without giving the fit a rule identity or source
weight. Restore all base predictions to the same target-latent units and freeze
and hash the common held-out tensor `Yhat[region,b,t,l]`. Only then form

```text
Yhat_rule[t,l] = sum over b in B* w[rule,b] * Yhat[b,t,l].
```

No rule-specific coefficient, intercept, gain, standardization, penalty,
alignment, support change, or exclusion is fit after mixing. Thus correct,
pooled, and swapped rules have exactly the same trials, base predictions,
electrodes, latent axes, capacity, and output dimensions; only the locked
source weights differ. This output-level mixture cannot be absorbed into a
freely refit target coefficient. Because target-training spikes still fit the
base predictors, the transferred object is the frequency-weight policy—not a
zero-shot decoder or a source coefficient matrix.

The follow-up endpoint is session-level:

    delta_R2_region = R2_SSE(correct-region) - R2_SSE(fixed pooled-prototype)

Retain negative `R2_SSE`. Freeze how latent dimensions are aggregated and give
sessions and the two animal-transfer directions equal weight. Also report the
swapped-minus-correct contrast. The proposed explanation predicts a positive
`delta_R2_region` by a meaningful prespecified margin in both directions, a
penalty for swapping, and a prespecified positive cosine agreement between the
two animals' projected M1-minus-PMd weight-difference vectors.

This is not evidence that M1 or PMd “generates” a frequency band. It says only
that, under the released feature pipeline, knowing the recording domain makes
a source-learned LFP feature rule more useful for predicting local population
activity.

### 4. Use the one internal opening correctly

The four hash-reserved sessions can test both the primary classifier and the
follow-up consequence only if all follow-up outputs and decision rules are
declared before the single configuration lock. Reveal all four together.
Do not inspect classifier scores first and then decide whether to compute the
correct-region contrast. There is no second opening and no replacement session.

One planned fit is required inside each reserved session: the locked recipe
uses its permitted training trials to estimate the shared target latent basis,
outcome scale, and one base predictor per band before untouched trials are
scored. This is not permission to retrain the policy after reveal. All support,
hyperparameters, source weights, fitting code, and mixture rules are already
locked, and no score is visible during fitting. The core YAML currently says
`retraining_policy: forbidden`; the executable follow-up contract must make
this distinction machine-readable before the audit can open. If its runtime
interprets that field as forbidding the mandatory within-session base fit, the
follow-up cannot share the current audit and must move to a successor round.

Because the release was already outcome-exposed by related work, this is a
procedural within-corpus audit. It can show that the locked code and policy
transfer to held-out sessions, but not independent confirmation.

### 5. Carry one locked prediction to a third animal

A direct third-animal test uses the same profile construction, classifier,
signed contrast, frequency weights, thresholds, and support rules. No
target-driven Procrustes alignment, reliability threshold, band selection, or
model choice is permitted. All compatible sessions are reported.

This must be a separately frozen successor round because the current common
protocol ends after its internal audit. If no suitable simultaneous M1/PMd
source exists, retain the two-animal scope. Additional Chewie sessions or the
Chewie-R implant are not a third animal.

## Decisive controls and what failure means

| Rival explanation | Direct test | Interpretation if it wins |
| --- | --- | --- |
| Ordinary reach-direction information | Direction-by-time baselines and within-direction whole-trial LFP–latent pairing null | No evidence for the proposed regional population relationship |
| Region labels are learned from array quality | Metadata-only classifier plus equal channel, trial, rank, reliability, and missingness support | Recording/hardware fingerprint; no neural regional interpretation |
| High-frequency spike leakage creates the profile | Low-frequency-only, high-frequency exclusion, same-electrode/unit-intersection, and spike-quality matching | Bound the result to a spike-rich recording feature |
| Flexible latents or mappings create separability | TME/smoothness/dimensionality-matched surrogate and capacity-matched mapping/null procedure | No profile-specific evidence |
| Band names or numeric column order leak the label | Authenticated guides and full band-label/profile permutation through fitting | Invalid regional interpretation |
| One animal or session drives the result | Both directional transfers and leave-one-session influence | Narrow or reject transfer claim |
| Correct-region weighting only has more freedom | Hash one common bandwise prediction tensor, then apply all fixed convex mixtures without any post-mixture fit | No consequence beyond model flexibility |
| Feature units or band labels manufacture the mixture result | Replay positive diagonal raw-feature unit changes through the full locked train-only preprocessing/base fit, source paired-region label swap through `P -> w -> mixture`, source band-label permutation against the fixed target tensor, a `D = 0` collapse sanity check, and a separate uniform-weight reference | Reject or narrow the claimed profile-to-mixture link |
| Base band predictions are too collinear to distinguish mixtures | Freeze a prediction-tensor identifiability diagnostic and weight-distance margin | Call the consequence non-identifiable rather than evidence for or against regional utility |
| Area 2 appears to extend the hierarchy | Keep Han/Lando as recording-domain specificity only because area, animal, and task are confounded | No three-region ordering or cortical hierarchy claim |

Hardware adjustment is not magic. Region-specific arrays may differ in
unmeasured ways, and region labels remain perfectly tied to those arrays. The
strongest permitted wording is therefore about the measured recording domains
unless a new design breaks that link.

## When to deepen, narrow, or stop

| Evidence | Next action and permitted claim |
| --- | --- |
| Bidirectional classification, repeated signed contrast, correct-region advantage, and locked transfer all pass | Develop a reusable recording-domain organization claim and name the bands, windows, animals, and exact population-prediction consequence. |
| Classifier transfers but correct-region does not beat pooled | Report a stable label with no demonstrated population-recovery benefit; stop the stronger interpretation. |
| Profile-to-weight transform is uniform/degenerate, regional weights are too similar, or base predictions are collinear | Call the consequence non-identifiable and seek a better-powered prespecified test; do not count it as pooled adequacy or a positive mechanism result. |
| Correct-region helps within one animal but cross-animal direction disagrees | Treat it as animal/session-specific and do not average away the disagreement. |
| Metadata-only prediction or matching explains the result | Reframe as a hardware/recording-quality result or stop; do not claim cortical area identity. |
| Only high-frequency/spike-rich features survive | State the contamination boundary and stop the broad LFP claim. |
| Internal audit fails | Preserve the development result and failure; do not adapt or swap candidates. |
| Third-animal source is absent | End with a two-animal, exposure-tainted scope and a concrete acquisition requirement. |
| Precision or meaningful-margin sensitivity is insufficient | Call the result unresolved; nonsignificance does not prove a common pooled rule is adequate. |

Before running the follow-up, freeze support, weighting formulas, model
capacity, meaningful margins, uncertainty, multiplicity, synthetic cases,
null count, compute budget, stopping events, and result labels. The primary
20--48 trial budget is not inherited automatically.

## Paper figures, contingent on the evidence

| Figure | Scientific judgment | What must be shown |
| --- | --- | --- |
| 1. Does the label transfer? | Whole-session M1/PMd discrimination works in both animal directions | Paired session design, both directional accuracies, permutation reference, every session |
| 2. What is the fingerprint? | A named signed frequency-by-latent contrast repeats | All bands and sessions, effect direction, reliability, influence, development-only highlight rule |
| 3. Neural profile or array artifact? | The contrast survives its strongest measured alternatives | Metadata-only model, matched support, low-frequency/high-frequency and spike-bleed tests, TME and band-label nulls |
| 4. Does the fingerprint matter? | A fixed band-marginal projection of the profile improves held-out population recovery | Undifferenced region prototypes and equation, one hashed bandwise prediction tensor, pooled/correct/swapped `R2_SSE`, identifiability and rescaling/null checks, both transfer directions |
| 5. Where does it generalize? | The complete locked statement survives reserved sessions and, if available, a third animal | One-shot audit accounting, all failures, third-animal result, explicit unresolved region–array confounding |

The abstract should ultimately state the actual signed contrast, the
cross-animal directions, the population-recovery consequence, the hardware
limitation, and the replication scope. “M1 and PMd differ” is already too broad
and too close to the source publication to be the conclusion.
