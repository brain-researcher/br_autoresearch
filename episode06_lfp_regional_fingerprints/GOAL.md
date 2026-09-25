# Does the LFP–population relationship differ reproducibly between M1 and PMd?

During the same reaching task, electrodes in M1 and PMd record two views of
local activity: the LFP and the spikes of nearby neurons. In one session, a
low-frequency LFP feature might predict the local spike-population activity
better in M1, while another frequency might work better in PMd. A difference
in one session, however, could just reflect a noisy array or a few unusually
good channels.

EP06 asks whether the complete band-by-band pattern repeats across animals.
For every session, we make one profile showing how well LMP and each registered
LFP power band predict the simultaneously recorded population activity. We
learn the M1-versus-PMd difference from Mihili and ask whether it labels whole
sessions from Chewie, then reverse the animals. The signed profile contrast—
not every band in isolation—must align across the two transfer directions, and
any band singled out for interpretation must have the same sign in both
animals. A high average score cannot hide a failed transfer in one direction.

Even successful classification may be misleading. In these recordings, each
brain region is tied to its own implanted array. A classifier could therefore
recognize missing channels, recording reliability, one spike-rich
high-frequency band, or another hardware difference rather than a reusable
neural relationship. The first result must show the actual signed frequency
contrast and test it under equal channel, trial, spike-rank, and model budgets,
without high-frequency bands, and against recording-metadata controls.

This is the entry point to a larger biological question:

> If M1 and PMd have a reproducible LFP–population difference, does that
> difference tell us which LFP frequencies will recover held-out population
> activity better?

The follow-up turns the profile into a direct prediction. M1 and PMd profiles
learned from the source animal produce two fixed sets of frequency weights. In
the other animal, the correct-region weights, one pooled set of weights, and a
deliberately swapped-region set combine exactly the same bandwise predictions.
If the profile captures a useful regional difference, the correct-region
combination should predict unseen population activity better than the pooled
combination, and swapping M1 with PMd should make prediction worse.

The intended paper must therefore show more than an M1/PMd label. It must show
which parts of the profile differ, that their signs repeat in both animals,
that ordinary recording-quality explanations do not account for the result,
and that the difference has the predicted consequence for population-state
recovery. A separately locked third-animal test would be needed to show that
the rule extends beyond Mihili and Chewie.

Even a positive result would not isolate pure cortical identity, because
region and array cannot be separated in this dataset. It would support a
narrower claim: under this reaching task and recording pipeline, an M1–PMd
difference repeats in both animals and can guide prediction of local
population activity, while remaining confounded with the two arrays. It would
not establish a causal mechanism, a unique biological origin for any frequency
band, or generalization to other tasks and populations.

The [paper plan](outputs/paper_plan.md) compares this claim with prior work and
specifies the follow-up prediction, alternatives, and figure-level evidence.
The cross-animal classification remains the first result; the follow-up cannot
change its answer.

## At a glance

| Question | EP06 design |
| --- | --- |
| What varies? | How well LMP and each LFP power band predict the local spike-population activity in M1 and PMd. |
| What is compared? | One complete M1 profile and one complete PMd profile for each recording session, rather than isolated bands chosen after seeing the result. |
| What must transfer? | A rule learned from Mihili must distinguish M1 from PMd in Chewie, and the rule learned from Chewie must work in Mihili. |
| What is held out? | Whole sessions from the other animal during cross-animal transfer, followed by two separately reserved sessions per animal for one internal audit. |
| What is the main score? | The average of the two cross-animal balanced accuracies, reported relative to chance and alongside each direction separately. |
| What could give a misleading positive result? | The two regions use different arrays, so missing channels, reliability, hardware metadata, or high-frequency spike contamination could reveal the label. |
| What can the first test conclude? | Whether these recordings contain a signed M1–PMd LFP–population fingerprint that repeats between two animals. |
| What would make a deeper finding? | The M1- and PMd-specific frequency rules must improve held-out population prediction over one pooled rule, while a swapped-region rule performs worse. |
| What is the next test? | Freeze the regional frequency weights, compare correct, pooled, and swapped mixtures on identical predictions, then carry the unchanged rule to a separately locked third-animal study if suitable data become available. |

## From a region label to an interpretable consequence

Suppose the classifier calls a held-out session “M1.” That answer alone does
not show what was learned. It could depend on one noisy band, missing channels,
or array quality. A useful scientific result must show the actual contrast—for
example, that the same registered low- or mid-frequency part of the profile is
higher for M1 than PMd in both animals—while reporting every band and session.

The proposed follow-up makes a prediction from the two source-region profile
prototypes that produced that signed contrast. The contrast alone is not
enough: subtracting PMd from M1 discards their shared level. Therefore retain
the undifferenced M1 and PMd frequency-by-latent prototypes in the exact locked
normalized/aligned representation supplied to the primary classifier, their
signed difference, and their equal-region pooled prototype. A fixed,
parameter-free softmax projection turns each prototype into one convex set of
band weights; the exact equation is in the paper plan.

In a target session, fit one base LFP-to-latent predictor per band exactly once
on permitted training trials under the same locked recipe. Freeze those
bandwise predictions before applying any regional rule. The correct-region,
pooled, and swapped rules then form different fixed weighted averages of the
**same predictions at the output level**; no coefficient, intercept, scaling,
or regularization is refit after weighting. If the fingerprint captures a
reusable difference, the correct-region mixture should beat the pooled mixture
by a fixed meaningful margin and swapping M1 and PMd weights should hurt. This
cannot be explained by a separate region-supervised model or by raw feature
weights being absorbed into freely refit coefficients.

This yields distinct outcomes:

| Possible result | What it would mean |
| --- | --- |
| Classification transfers, the signed contrast repeats, and correct-region weights improve held-out latent prediction | The recording domains have a reusable difference with a concrete modeling consequence. Region remains confounded with array/hardware. |
| Classification transfers but correct-region weights do not beat pooled weights | A stable label exists, but it has not shown a useful consequence for population-state recovery. |
| Regional weights collapse to the same vector or bandwise target predictions are too collinear to distinguish mixtures | The proposed consequence is non-identifiable; do not treat a null contrast as evidence that the regional profile has no effect. |
| Metadata-only or deliberately reliability-mismatched controls classify equally well | The result is compatible with an implant/recording-quality fingerprint; no neural regional interpretation. |
| Only high-frequency or spike-rich channels carry the result | Bound the claim to a spike-contaminated recording feature, not a broad LFP profile. |
| The two transfer directions disagree or the reserved sessions fail | No general two-animal fingerprint; report the narrower session result or stop. |

The follow-up uses the same exposed recordings and simultaneous spikes, so it
is explanatory evidence, not another animal replication. Its support rules,
models, margins, multiplicity, budget, and stopping rule must be frozen before
its outcomes are inspected. Its results cannot rewrite the original audit or
terminal class.

“Fit once in a target session” means the session-specific training step already
required to construct each bandwise profile under a locked recipe. It does not
mean choosing a new policy after audit outcomes appear. The executable
follow-up contract must distinguish this mandatory within-session fit from the
core YAML's forbidden post-reveal retraining before any shared audit opens; if
it cannot, the follow-up moves to a successor round.

## Evidence roles and internal split

The current source has six simultaneous M1/PMd sessions for Mihili and six for
Chewie-L. After structural eligibility is frozen, rank canonical session IDs
within each animal by
`SHA256("ep06-adaptive" || animal || canonical_session_id)`:

- the lowest four eligible sessions per animal are adaptive development and
  nested session-level selection;
- the highest two eligible sessions per animal are a one-shot **internal
  audit** opened only after configuration lock; and
- if all six sessions per animal are not eligible, stop at the frozen
  sample-size/identifiability gate rather than outcome-guided reallocation.

All current-release outcomes are conservatively exposure-tainted by related
historical work. Thus this internal audit tests procedure and within-corpus
session transfer; it is not independent confirmation.

An **external generalization confirmation round** requires an independently
sealed third animal recorded simultaneously from M1 and PMd under a compatible
reaching task with adequate sessions, trials, electrodes, units, and
authenticated LFP feature semantics. That animal is absent from the current
release and is not claimed to have been acquired. Under the common protocol,
the current round ends when its one-shot internal audit opens; a third-animal
test must be a separately frozen successor round with a new exposure record. A
direct generalization test must reuse the same locked candidate without
adaptation; an adapted search would answer a different question.

## One session-level fingerprint

For each session, region, and registered feature block, fit a training-trial
mapping from LFP features to a training-derived spike-population latent. Score
untouched trials with `R2_SSE`, preserving negative values, and subtract the
training-derived direction-by-elapsed-time baseline. Collapse trials, bins,
channels, folds, and seeds into one feature profile per region/session before
regional classification.

Profile shape and magnitude remain separate. The primary discrimination uses
shape after the registered centering/shrinkage rule; magnitude is a secondary
axis and cannot rescue unstable shape.

## Bounded scientific operator grammar

One trial is a declarative global fingerprint policy chosen from:

1. **Analysis window:** the primary `+150 ms` to `+450 ms` execution interval,
   or one of a small set of event-defined preparation/execution windows frozen
   from behavior before neural search. The primary question may not migrate to
   whichever window scores best.
2. **Feature block:** LMP and authenticated power classes 0.5--4, 4--8, 8--12,
   12--25, 25--50, 50--100, 100--200, and 200--400 Hz; individual classes,
   registered contiguous low/mid/high groupings, or the full nine-class
   profile. Numeric matrix order is never semantic identity.
3. **Electrode aggregation:** fixed 15-electrode matched budget, train-only
   reliability shrinkage, robust mean/median, or train-only PCA across eligible
   physical electrodes. M1 and PMd receive identical capacity rules.
4. **Spike latent:** train-only PCA or reduced-rank latent with dimension from
   a finite schedule capped by shared session support. Region-specific outcome
   dimensions cannot be selected from audit performance.
5. **LFP-to-latent mapping:** ridge, reduced-rank regression, PLS, or regularized
   CCA evaluated on held-out whole trials. Scaling and regularization are
   training-only.
6. **Profile normalization:** centered unit-L2 shape, reliability-shrunk shape,
   or joint shape-plus-magnitude with a frozen magnitude weight.
7. **Session/animal alignment:** none, training-session robust scaling, or a
   train-only orthogonal/Procrustes alignment using region labels only in the
   training animals/sessions.
8. **Region decision rule:** nearest prototype with correlation/cosine
   distance, shrinkage Mahalanobis, or regularized logistic classification.
9. **Ensemble:** convex combination of at most two already evaluated
   complementary policies.

Arbitrary band edges, audit-specific alignment, per-animal winners, unpaired
M1/PMd trials, bin-level splits, outcome-chosen electrodes, raw-phase analyses,
and arbitrary code mutation are forbidden.

## Objective and constraints

The primary development objective is the mean of the two cross-animal
balanced accuracies minus 0.5: train the global fingerprint rule on development
sessions from Mihili and classify M1 versus PMd development sessions from
Chewie, then reverse. Nested session folds provide policy selection without
using a scored session to tune its transforms.

A policy is feasible only if:

- balanced accuracy is above chance in both transfer directions under the
  complete-procedure region-label permutation null;
- M1-minus-PMd contrast vectors align in sign/direction across animals;
- no single development session determines the result;
- paired trial identity, equal regional capacity, and minimum reliability pass;
- low-frequency-only and same-electrode/unit-intersection sensitivities retain
  the frozen minimum direction; and
- TME/smoothness-matched surrogate and band-label nulls do not explain the
  effect.

Area-2 profiles from Han and Lando are a prespecified recording-domain
specificity analysis only. Region is confounded with animal/task there, so
area 2 cannot rescue or strengthen the primary M1/PMd decision.

## Multi-stage adaptive loop

1. **Preflight:** authenticate source, guides, paired M1/PMd trial IDs, events,
   electrodes, spike ranks, licenses, and synthetic indexing/leakage/null
   fixtures; then freeze the deterministic 4+2 session split.
2. **Coverage stage:** evaluate fixed-profile, low-frequency, full-profile,
   ridge, CCA, prototype, and regularized-classifier anchors before
   exploitation.
3. **Adaptive development:** make one mechanism-led operator change per trial;
   record parent, hypothesis, complete DAG/hash, per-session profiles,
   cross-animal scores, constraints, runtime, and failure reason. Maintain a
   nonterminal incumbent and robustness/complexity Pareto archive.
4. **Successive fidelity:** prune only on frozen inner session folds. Every
   finalist runs on all eight development sessions, both animal-transfer
   directions, fixed seeds, and all mandatory falsifiers.
5. **Stress stage:** incumbent plus at most three challengers undergo
   leave-one-session influence, reliability matching, electrode/trial budget
   matching, low-frequency-only, spike-bleed-through, window, mapping, and
   profile-block ablations.
6. **Configuration lock:** choose exactly one policy and hash source/session
   manifests, code, environment, feature semantics, folds, transforms,
   hyperparameter rule, nulls, thresholds, seeds, and output schema.
7. **One-shot internal audit:** reveal the four reserved current-release
   sessions together and score the locked policy once. No candidate or rule
   changes follow this reveal.
8. **External-confirmation handoff:** end the current round, preserve its lock
   hash and audit receipt, and specify a future third-animal successor round.
   That successor may apply the identical locked policy once; the new animal
   cannot update the representation, alignment, decision rule, thresholds, or
   stopping logic in a direct generalization test.

An incumbent is never a terminal result. Minimum trials, operator coverage,
falsifiers, stress tests, patience, configuration lock, and the audit gate are
independent requirements.
At least 40% of valid post-coverage trials must be falsifiers, ablations,
negative controls, influence guards, synthetic recovery, or direct
replications. At least two outcome-adaptive successor cycles and two recorded
incumbent/challenger decisions are required before lock.

## Mandatory falsifiers and ablations

- complete-procedure M1/PMd label permutation at paired-session/trial level;
- band-label/profile permutation through prototype/classifier construction;
- within-direction whole-trial LFP-to-latent pairing permutation;
- TME or smoothness/dimensionality-matched surrogate latent;
- direction-by-time and elapsed-time-only baselines;
- low-frequency-only and high-frequency-exclusion analyses;
- same-electrode/unit-intersection spike-bleed-through sensitivity;
- matched electrode, trial, spike-rank, and reliability budgets;
- array/QC/impedance/missing-channel metadata-only prediction and matching,
  reported as a hardware-confound falsifier rather than a neural comparator;
- feature-group, mapping-family, normalization, and alignment ablations;
- leave-one-development-session influence and both transfer directions; and
- Area-2 recording-domain specificity reported without regional inference.

A mandatory falsifier failure makes the pipeline infeasible regardless of
headline accuracy.

## Budget and stopping

- minimum valid scientific trials: **20**;
- maximum valid scientific trials: **48**;
- patience after the minimum: **10** valid trials without material constrained
  primary improvement;
- CPU ceiling: **1,200 core-hours**;
- GPU ceiling: **0 GPU-hours**;
- wall-clock ceiling: **96 hours** from first scientific trial;
- finalists: at most **4** including the incumbent;
- internal audit openings: **1**;
- maximum parallel CPU cores: **32**;
- per-trial memory ceiling: **128 GB**;
- scratch-storage ceiling: **750 GB**.

Structural QC, deterministic session allocation, fixtures, and exact reruns
after proven infrastructure failure do not count as scientific hypotheses but
remain ledgered and consume resource ceilings.

## Terminal classes

- `candidate_ready`: one locked policy passes development and the one-shot
  internal audit. It is eligible only for the bounded within-corpus,
  two-animal claim and carries a mandatory third-animal confirmation
  requirement for any generalization claim.
- `closed_no_candidate`: a technically valid bounded search finds no feasible
  policy, or the locked policy fails the current round's single internal audit.
- `search_exhausted_no_audit`: development finishes without a permissible
  internal-audit opening; it maps outward to `closed_no_candidate` but is not
  evidence that the scientific effect is absent.
- `technical_failure`: source, pairing, guide, event, electrode/rank,
  reliability, or executable evaluation gates cannot be satisfied.
- `policy_violation`: leakage, audit-specific adaptation, outcome-driven
  reallocation, undeclared operators, or post-reveal candidate swapping.

For canonical outer status, `candidate_ready` maps to `candidate_ready`,
`closed_no_candidate` and `search_exhausted_no_audit` map to
`closed_no_candidate`, and `technical_failure` or `policy_violation` maps to
`technical_failure`.

## Claim boundary

Passing the current-release internal audit supports only within-corpus
cross-session and two-animal transfer under known exposure. A later sealed
third-animal successor round may support the locked fingerprint's transfer to
that animal under the compatible task and measurement pipeline, but its result
cannot rewrite this round's terminal class. Neither result estimates
population prevalence, separates cortical area from its implanted array or
other task/hardware confounds, assigns mechanism to frequency bands, or
generalizes to raw LFP,
other behaviors, species, or recording technologies.

The stronger paper interpretation—a reusable regional difference in how LFP
features recover local population dynamics—requires the separately locked
correct-region versus pooled and swapped-region comparison in the paper plan.
Classification accuracy alone does not establish that consequence. Even a
successful third-animal transfer does not fully separate cortical area from
array placement and other region-linked hardware without a design that breaks
that confounding. No real EP06 search, audit, or follow-up was run while
preparing this revision.
