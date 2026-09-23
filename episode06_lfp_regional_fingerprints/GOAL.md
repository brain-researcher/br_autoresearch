# EP06 — adaptive regional recording-domain fingerprints

## Status, protocol, and exposure boundary

This is the current local episode contract, governed by
[`../ADAPTIVE_SEARCH_PROTOCOL.md`](../ADAPTIVE_SEARCH_PROTOCOL.md). It is not
authority to read outcomes, run compute, create a canonical loop, submit a
candidate, or change campaign state.

Outcomes from the same Dryad release were accessed previously, so a clean
directory cannot restore freshness. The current episode inherits no prior
action, decision, or scientific acceptance.

## Scientific question

Can a bounded adaptive search learn a frequency-by-latent profile policy that
distinguishes simultaneously recorded M1 from PMd across sessions and transfers
between Mihili and Chewie, and does that locked regional fingerprint generalize
without modification to a third animal with matched simultaneous M1/PMd
recordings?

The estimand is a predictive **M1-versus-PMd recording-domain fingerprint**
under the released task, implanted arrays, and measurement pipeline. Region
and array are structurally paired in these recordings, so this is not a pure
cortical-identity estimand, universal cortical identity, causal mechanism,
population prevalence estimate, or evidence that any frequency band has a
uniquely synaptic origin.

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
test must be a newly registered successor round with a new exposure record. A
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
  policy, or the locked policy fails either audit.
- `technical_failure`: source, pairing, guide, event, electrode/rank,
  reliability, or executable evaluation gates cannot be satisfied.
- `policy_violation`: leakage, audit-specific adaptation, outcome-driven
  reallocation, undeclared operators, or post-reveal candidate swapping.

For canonical outer status, `candidate_ready` maps to `candidate_ready`,
`closed_no_candidate` maps to `closed_no_candidate`, and `technical_failure`
or `policy_violation` maps to `technical_failure`.

## Claim boundary

Passing the current-release internal audit supports only within-corpus
cross-session and two-animal transfer under known exposure. Passing the sealed
third-animal audit supports the locked fingerprint's transfer to that animal
under the compatible task and measurement pipeline. Neither result estimates
population prevalence, separates cortical area from its implanted array or
other task/hardware confounds, assigns mechanism to frequency bands, or
generalizes to raw LFP,
other behaviors, species, or recording technologies.
