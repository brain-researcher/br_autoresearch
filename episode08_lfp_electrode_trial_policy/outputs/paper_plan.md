# EP08 paper plan: what can a 16-trial pilot tell us to record next?

Status: proposed study design, 2026-09-24. No policy result, acquisition
mechanism, hardware saving, or external-animal validation is claimed here.

## The intended discovery

EP08 starts with a matched-resource policy test. Every method receives the same
16 all-electrode pilot. It must then retain 4, 8, or 16 physical electrodes and
allocate 32, 64, or 128 additional calibration trials. One global policy is
compared with random, geometry-only, and reliability-only acquisition on the
same untouched trials.

The paper should then answer a more useful question: **what did the short pilot
reveal about future recording value?** Specifically:

1. Can it distinguish a clean but redundant electrode from one that adds
   population information not present on the other retained electrodes?
2. Can the current calibration errors predict which reach-direction stratum
   will benefit most from the next trial?
3. Does a session appear more sensitive to increasing electrode count or to
   increasing calibration-trial count on the fixed grid, and can the pilot
   predict that pattern?

A black-box policy that wins the average score is an engineering candidate. A
policy whose pilot scores predict held-out electrode and trial value in new
sessions provides an acquisition principle. The second claim requires the
mechanism and external tests below.

The adaptive round in [GOAL.md](../GOAL.md) remains the primary study. The
follow-up cannot change its `3 x 3` grid, strongest-baseline comparison,
animal-balanced estimand, search budget, or one-shot audit conclusion.

## A concrete example

Suppose electrodes 11 and 12 are both clean and close together. The pilot shows
that their LFP traces are highly correlated. Electrode 37 is slightly noisier
but captures a fluctuation that neither 11 nor 12 captures. Reliability-only
ranking keeps 11 and 12; a redundancy-aware rule keeps 11 and 37. If removing
37 later causes a larger loss on untouched trials than removing 12, the pilot
made a correct prediction about **conditional value**, not just signal quality.

Now suppose the early decoder predicts seven reach directions well but has
large cross-validated residuals for upward reaches. If the policy requests an
upward trial and that next hash-queued trial reduces untouched-trial error more
than the balanced allocator's next request, the trial decision also has a
testable explanation. Neither conclusion follows merely from inspecting which
electrodes or directions the winning policy happened to choose.

## What prior work already established

| Prior work | What it already showed | What EP08 must add |
| --- | --- | --- |
| Flint et al., Journal of Neural Engineering 2012, [doi:10.1088/1741-2560/9/4/046006](https://doi.org/10.1088/1741-2560/9/4/046006) | Motor-cortical LFP features can decode reaching, and performance can plateau after a subset of electrodes/features selected by their individual kinematic correlation. | Prospective pilot-only selection of whole physical electrodes, a test of conditional rather than marginal value, transfer to held-out sessions, and matched trial allocation. |
| Gallego-Carracedo et al., eLife 2022, [doi:10.7554/eLife.73155](https://doi.org/10.7554/eLife.73155) | The source corpus contains region- and frequency-dependent relationships between LFPs and population latent dynamics. | Determine how much of that relationship can be retained under a fixed electrode/calibration budget and why a selected subset works. |
| Even-Chen et al., Nature Biomedical Engineering 2020, [doi:10.1038/s41551-020-0595-9](https://doi.org/10.1038/s41551-020-0595-9) | Intracortical BCI performance can be studied as neural-interface parameters and channel resources are reduced, motivating power-saving hardware designs. | A session-transferable post-pilot acquisition rule for these LFP features, with sequential information boundaries; EP08 still cannot claim actual power or bandwidth savings without those measurements. |
| Sussillo et al., Nature Communications 2016, [doi:10.1038/ncomms13749](https://doi.org/10.1038/ncomms13749) | Historical neural variability can make spike-based decoders robust to future recording conditions and electrode loss. | Choose sensors and calibration trials from current pilot information under identical budgets, rather than train one decoder on a larger historical dataset. |
| Li et al., Computers in Biology and Medicine 2025, [doi:10.1016/j.compbiomed.2025.110231](https://doi.org/10.1016/j.compbiomed.2025.110231) | An active-learning/domain-adaptation decoder used uncertainty to select a small target-domain calibration set and was evaluated with recordings from three monkeys. | Adaptive calibration-sample selection is therefore not itself new. EP08 must add the joint pilot-guided sensor-and-trial policy, predict conditional electrode and next-direction value, and enforce a strict matched-resource sequential replay. |

Active-learning and channel-selection literatures already contain many ways to
rank samples or sensors. A new optimizer is therefore not enough. EP08's
distinct contribution must be a jointly tested statement about which
pilot-observable properties predict future LFP-to-population value under a
strict sequential replay.

## Study sequence

### 1. Complete the primary matched-resource policy test

Keep the current whole-session split: four development and two sealed sessions
per primary animal, if structural qualification passes. Give every policy the same
16-trial all-electrode pilot and evaluate every retained policy at all nine
cells of

```text
E in {4, 8, 16} whole electrodes
N in {32, 64, 128} post-pilot calibration trials.
```

Use the same fixed reduced-rank ridge decoder, neuron roster, evaluation trials,
and full-resource check for all selectors. Preserve the animal-balanced,
equal-cell primary score and the `(4,32)` and `(8,64)` anchors. Report each grid
cell and each session even when the aggregate passes.

This round answers whether the policy works on four internal audit sessions.
It does not show that a retained electrode is biologically special, that a
requested trial caused an online improvement, or that the rule generalizes to
a new animal.

### 2. Test whether the pilot predicts unique electrode value

For a fixed acquired data state and retained set `S`, define the conditional
value of retained electrode `e` using the trusted evaluator:

```text
V_e = evaluation_loss(S without e) - evaluation_loss(S)
```

Both losses use the same acquired calibration trials, untouched evaluation
trials, target-neuron roster, and frozen decoder procedure. A positive `V_e`
means the electrode adds information not fully supplied by the other retained
electrodes. This is a mechanism label, not a new primary endpoint and not a
causal effect of implanting that electrode.

Before these labels are available for a session, predict their ordering from a
small set of pilot-only quantities:

- reliability and missingness;
- distance from and geometry coverage relative to the other retained
  electrodes;
- correlation or conditional variance relative to the retained set; and
- the frozen learned pilot score, if the learned selector survived the primary
  controls.

Use whole-session cross-fitting. A model predicting values for development
session `s` must be trained without `s`. Limit the explanatory comparison to
the prespecified reliability, geometry, redundancy, and learned-score families;
do not search arbitrary electrode descriptors after seeing removal losses.

The decisive question is not whether selected electrodes have high scores.
It is whether the pilot-only prediction ranks held-out conditional values
better than reliability alone and whether that relation repeats in both
animals. If reliability alone performs equally well, the correct conclusion is
a quality-control rule, not a new principle of spatial complementarity.

### 3. Test whether pre-action state predicts next-trial value

At a frozen acquisition state `h`, each legal action requests the next
hash-queued whole trial from one reach-direction stratum. For mechanism evaluation,
the trusted replay can branch from the same state, add exactly one legal next
trial for each candidate stratum, refit the fixed decoder, and score all
branches on the same untouched evaluation trials. Define

```text
V_a(h) = loss_before(h) - loss_after(h, next trial from stratum a).
```

The policy must predict the ordering of `V_a(h)` before receiving those trials
or evaluation outcomes. Candidate predictors are limited to current
direction-wise calibration counts, nested-CV error, uncertainty, and residual
diversity. Compare the chosen action with direction-balanced and frozen-random
actions at identical queue depth.

This is retrospective replay. It assumes trials within a direction are
exchangeable under the frozen hash order. It does not show how a person or
animal would respond if an online system requested a different movement. Any
future real-time or human-burden claim needs a prospective interaction study.

### 4. Describe the electrode and trial axes without inventing an exchange rate

Use the locked nine-cell surface to report explicit step contrasts, for
example:

```text
S_E(4->8 | N)   = R2(8,N)  - R2(4,N)
S_N(32->64 | E) = R2(E,64) - R2(E,32).
```

Also report whether the electrode step changes the trial step and vice versa.
These contrasts show sensitivity along each axis and possible interaction.
They do not say that four electrodes cost the same as 32 trials. A statement
about where to spend one common budget requires separately measured channel
power, bandwidth, recording time, user burden, and a cost function frozen
before outcomes.

The follow-up prediction is modest and testable: given the 16-trial pilot, can
a rule predict the shape of the session's nine-cell surface or at least the
sign of the prespecified step contrasts? With only eight development sessions,
use a simple, frozen rule and report all leave-one-session and
leave-one-animal-out failures.

### 5. Freeze the mechanism and test a third animal

Before any external evaluation outcome is opened, freeze:

- the third-animal source and session roles;
- physical-electrode and feature crosswalks;
- pilot and queue construction;
- selector, allocator, and decoder state;
- conditional electrode and next-trial value definitions;
- mechanism scores, margins, multiplicity, and uncertainty;
- handling of missing geometry, invalid actions, and insufficient support; and
- compute budget, stopping events, and the complete result table.

The external test should ask whether the pilot-only rules predict conditional
electrode and trial value in every eligible sealed session, and whether the
joint policy beats the same baselines across the complete supported grid. If
the third animal needs development sessions to fit a within-animal policy,
those sessions must be declared in advance and kept separate from its audit
sessions.

Chewie-R can reveal sensitivity to another implant in the same animal but
cannot fill this role. The four current audit sessions also cannot become fresh
confirmation of a mechanism chosen after their outcomes are seen. No suitable
third-animal pack is currently provisioned.

Accordingly, the current four audit sessions belong only to Figure 1's primary
policy result. Figures 2--4 may use cross-fitted development evidence to choose
and explain a mechanism, but their first confirmatory test is Figure 5 on the
new sealed external source. Generic audit summaries frozen before the primary
opening remain descriptive and cannot be relabeled as that test.

## Decisive alternatives and what they mean

| Alternative explanation | Test on identical data and budgets | Consequence if supported |
| --- | --- | --- |
| The cleanest electrodes are enough | Reliability-only ranking and quality-matched subsets | Drop the novelty claim about spatial or conditional information |
| More spatial spread is enough | Geometry-only farthest-first and shuffled-geometry control | Claim spatial coverage only if geometry transfers and the shuffle fails |
| The learned ranker memorizes session utility | Whole-session cross-fitting, utility-label permutation, and leave-one-animal-out transfer | Reject the learned mechanism if it does not transfer |
| Trial gain is just direction balancing | Balanced allocator with identical queue depth and counts | Use the simpler balanced rule; do not claim uncertainty targeting |
| One grid cell creates the average | Full nine-cell report, required anchors, and session-level influence | Narrow or reject the global policy claim |
| LFP selection exploits spike contamination | Spike-contamination and low-frequency-only sensitivities | Limit the signal interpretation or reject the policy |
| Offline queue replay stands in for online acquisition | Explicit exchangeability checks and prospective follow-up | Keep the claim retrospective until a real sequential experiment exists |

## When to deepen, narrow, or stop

| Evidence | Next action and permitted claim |
| --- | --- |
| Joint policy passes; both pilot-only mechanism predictions repeat and transfer externally | Develop a joint acquisition principle for the declared LFP representation and protocol |
| Electrode prediction succeeds but adaptive trials do not beat balance | Focus on post-pilot electrode retention; recommend balanced calibration trials |
| Trial prediction succeeds but complex electrode selection does not beat reliability | Focus on adaptive calibration sampling; retain a simple quality rule for electrodes |
| Policy wins but neither mechanism prediction works | Report a bounded black-box engineering result; do not tell a story about unique electrodes or informative directions |
| Gains disappear outside one animal, session, or budget cell | Narrow to that domain or stop; do not average across the failure |
| Full-resource equivalence fails | Do not claim the reduced acquisition preserves the target information |
| Primary audit fails or remains unresolved | Preserve that outcome; mechanism exploration cannot create `candidate_ready` |
| No third animal is available | Stop at internal held-session evidence and state the missing validation directly |

## Paper figures, contingent on evidence

| Figure | Scientific judgment | Required content |
| --- | --- | --- |
| 1. Does the acquisition policy work? | One global policy beats all cost-matched baselines | Sequential design, full nine-cell surface, all sessions/animals, random-seed distribution, full-resource ceiling |
| 2. Which electrodes are uniquely useful? | Pilot-only information predicts conditional electrode value | Example physical array, quality and redundancy, predicted versus held-out removal loss, reliability-only comparison |
| 3. Which trial should come next? | Pre-action state predicts next-trial value beyond balance | Action timeline, direction-wise errors, counterfactual queue replay, chosen/balanced/random comparisons |
| 4. What is the bottleneck? | Electrode and trial axes have a reproducible interaction or separable effects | Prespecified grid-step contrasts, no unsupported common-cost conversion, animal/session influence |
| 5. Does the rule transfer? | The same pilot interpretation and joint policy work in a new animal | Frozen third-animal roles, every sealed session, failures, mechanism and performance predictions |

The abstract should eventually name the pilot-observable property that
predicted future value, say whether electrode selection, trial allocation, or
both mattered, and state the exact external scope. It must not describe
retrospective retention as electrode placement, channel count as measured
power savings, replay as an online experiment, or population-spike prediction
as demonstrated clinical BCI control.
