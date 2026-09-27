# EP08 paper plan: what can a 16-trial pilot tell us to record next?

Status: proposed study design, 2026-09-26. No acquisition result, explanatory
result, hardware saving, online benefit, or third-animal validation is claimed.

## The intended discovery

Every method receives the same 16-trial all-electrode pilot: two reaches in
each of eight directions. It must then retain 4, 8, or 16 physical electrodes
and choose the directions of 32, 64, or 128 additional calibration trials.
One global method is compared with random, spatial-coverage, and signal-quality
rules on the same untouched trials using the same decoder.

The first question is whether pilot-guided choices work at matched budgets. The
paper should then ask what the pilot actually revealed:

1. Can it distinguish a clean but redundant electrode from one that adds
   population information missing from the other retained electrodes?
2. Can the current calibration errors predict which reach direction will
   benefit most from the next trial?
3. Can it predict whether a session is more sensitive to adding electrodes or
   adding calibration trials, while keeping those resources in separate units?

A method that wins only the average score is an engineering candidate. A
method whose pilot measurements predict later electrode and trial value in new
sessions supports an acquisition principle.

## A concrete example

Suppose electrodes 11 and 12 are both clean and close together, and their
pilot LFP signals are highly correlated. Electrode 37 is slightly noisier but
captures variation missing from 11 and 12. Signal-quality ranking keeps 11 and
12; a redundancy-aware method keeps 11 and 37. If removing 37 later harms
untouched-trial prediction more than removing 12, the pilot predicted
**conditional value**, not just signal quality.

Now suppose seven reach directions are already predicted well, while upward
reaches have large cross-validated calibration errors. If the method predicts
that another upward trial will reduce future error more than the balanced
choice, that prediction explains the trial decision. The prediction must be
made before the additional trial or its later benefit is known.

Neither conclusion follows merely from listing the electrodes or directions
chosen by the best-performing method.

## What earlier work already established

| Earlier work | What it showed | What EP08 must add |
| --- | --- | --- |
| Flint et al., Journal of Neural Engineering 2012, [doi:10.1088/1741-2560/9/4/046006](https://doi.org/10.1088/1741-2560/9/4/046006) | Motor-cortical LFP features can decode reaching, and performance can plateau after selecting a subset by individual kinematic correlation. | Pilot-only selection of whole electrodes, conditional rather than marginal value, transfer across sessions, and matched trial allocation. |
| Gallego-Carracedo et al., eLife 2022, [doi:10.7554/eLife.73155](https://doi.org/10.7554/eLife.73155) | This source contains region- and frequency-dependent relationships between LFP and population dynamics. | Determine how much of that relationship survives a fixed electrode/calibration budget and why the selected subset works. |
| Even-Chen et al., Nature Biomedical Engineering 2020, [doi:10.1038/s41551-020-0595-9](https://doi.org/10.1038/s41551-020-0595-9) | Neural-interface performance can be studied as channel resources are reduced, motivating lower-power hardware. | A session-transferable post-pilot rule; EP08 still cannot claim measured power or bandwidth savings without those measurements. |
| Sussillo et al., Nature Communications 2016, [doi:10.1038/ncomms13749](https://doi.org/10.1038/ncomms13749) | Historical neural variability can improve decoder robustness to electrode loss. | Choose electrodes and calibration trials from current pilot information under identical budgets, rather than simply train on more historical data. |
| Li et al., Computers in Biology and Medicine 2025, [doi:10.1016/j.compbiomed.2025.110231](https://doi.org/10.1016/j.compbiomed.2025.110231) | An active-learning/domain-adaptation decoder used uncertainty to select a small target-domain calibration set across three monkeys. | Adaptive calibration selection alone is not new; EP08 must jointly test sensors and trials, predict their future value, and respect what information was available at each choice. |

Channel selection and active learning already offer many ranking algorithms. A
new optimizer is therefore insufficient. EP08's contribution must be a clear
statement about which pilot-visible properties predict future
LFP-to-population value.

## Study sequence

### 1. Test the joint acquisition rule at matched budgets

Use four development and two untouched sessions per animal if the data support
that split. Give every method the same pilot and evaluate the complete grid:

```text
E in {4, 8, 16} whole electrodes
N in {32, 64, 128} additional calibration trials
```

Use the same reduced-rank ridge decoder, population target, and evaluation
trials for every method. Preserve the equal-cell, animal-balanced primary score
and the `(4,32)` and `(8,64)` scarce-resource settings. Report every grid cell
and every session, including failures.

This step answers whether one global method works on four internal held
sessions. It does not show that a retained electrode is biologically special,
that a requested trial caused an online improvement, or that the method
generalizes to a new animal.

### 2. Ask whether the pilot predicts unique electrode value

For a fixed retained set `S`, define the conditional value of electrode `e`:

```text
V_e = evaluation_loss(S without e) - evaluation_loss(S)
```

Both losses use the same acquired calibration trials, untouched evaluation
trials, target population, and decoder procedure. A positive value means that
the electrode adds predictive information not fully supplied by the others;
it is not a causal effect of implanting that electrode.

Before those removal values are available for a session, predict their
ordering from a small set of pilot-only quantities:

- reliability and missingness;
- geometry and coverage relative to the other retained electrodes;
- correlation or conditional variance relative to the retained set; and
- the selected learned pilot score, if it survived the primary controls.

Use whole-session cross-fitting: a model predicting values in one development
session is trained without that session. The decisive comparison is whether
the pilot-only rule ranks conditional value better than reliability alone and
whether the relation repeats in both animals.

### 3. Ask whether the current state predicts the best next direction

At a fixed acquisition state, consider the next prespecified trial from each
legal reach direction. Add one such trial, refit the common decoder, and score
each branch on the same untouched trials:

```text
V_direction = loss before - loss after adding that direction's next trial
```

The method must predict the ordering before receiving those trials or their
evaluation benefits. Predictors are limited to current direction counts,
cross-validated calibration error, uncertainty, and residual diversity.
Compare the selected direction with balanced and random choices at the same
point in the recorded trial order.

This is retrospective recorded-data analysis. It assumes that trials within a
direction are reasonably exchangeable in the prespecified order. It does not
show how an animal would respond if a real-time system requested a movement.

### 4. Keep electrode and trial resources in separate units

Use the nine-cell surface to report explicit step contrasts, for example:

```text
electrode step: R2(8,N)  - R2(4,N)
trial step:     R2(E,64) - R2(E,32)
```

Also report whether the electrode step changes with trial count and vice versa.
These contrasts describe sensitivity along two axes. They do not say that four
electrodes cost the same as 32 trials. A common-cost conclusion would require
direct measurements of channel power, bandwidth, recording time, participant
burden, and a cost function chosen before outcomes.

The explanatory prediction is modest: from the 16-trial pilot, can a simple
rule predict the shape of a session's nine-cell surface or at least the sign of
the prespecified step contrasts? With only eight development sessions, every
leave-one-session and leave-one-animal failure must be reported.

### 5. Test the explanation in a third animal

Before viewing outcomes in an external session, choose:

- the third-animal session roles;
- electrode and LFP-feature correspondence rules;
- pilot and within-direction trial order;
- electrode selector, trial allocator, and decoder;
- conditional electrode and next-direction value definitions;
- prediction scores, effect margins, uncertainty, and multiple-testing plan;
- behavior when geometry or trial support is missing; and
- the complete result table.

The external study should ask whether pilot-only rules predict conditional
electrode and trial value in every eligible held session and whether the joint
method beats the same comparisons across the supported grid. If development
sessions from the third animal are needed, they must be declared and kept
separate from its final test sessions.

Chewie-R cannot fill this role because it is another implant in the same
animal. The four current held sessions also cannot freshly confirm an
explanation chosen after their results are known. No suitable third-animal data
source is currently available.

## Decisive alternatives

| Alternative explanation | Fair comparison | Consequence if supported |
| --- | --- | --- |
| The cleanest electrodes are sufficient | Reliability-only ranking and quality-matched subsets | Drop the novelty claim about complementary information |
| Spatial spread is sufficient | Geometry-only farthest-first and shuffled locations | Claim spatial coverage only if it transfers and the location shuffle fails |
| A learned ranker memorizes sessions | Whole-session cross-fitting, permuted value labels, and leave-one-animal transfer | Reject the learned explanation if it does not transfer |
| Trial gain is only direction balancing | Balanced allocation with the same counts and position in the trial order | Use the simpler balanced rule; do not claim uncertainty targeting |
| One grid cell creates the average | Full nine-cell surface and session-level influence | Narrow or reject the global policy claim |
| High-frequency features exploit spike-rich content | Low-frequency-only and spike-contamination sensitivities | Limit the signal interpretation or reject the method |
| Recorded data stand in for an online experiment | State the exchangeability assumption and require a prospective interaction study | Keep the claim retrospective |

## When to deepen, narrow, or stop

| Evidence | Next action and permitted claim |
| --- | --- |
| Joint method passes; both value predictions repeat and transfer to a third animal | Develop a joint acquisition principle for the declared LFP representation and protocol |
| Electrode prediction succeeds but adaptive trials do not beat balance | Focus on post-pilot electrode retention and recommend balanced calibration trials |
| Trial prediction succeeds but complex electrode selection does not beat reliability | Focus on calibration sampling and keep a simple quality rule for electrodes |
| Method wins but neither explanatory prediction works | Report a bounded engineering result without a story about unique electrodes or informative directions |
| Gains disappear outside one animal, session, or budget cell | Narrow to that domain or stop; do not average across the failure |
| Reduced acquisition is materially worse than the full-resource ceiling | Do not claim that the reduced design preserves target information |
| The primary held-session result fails or remains ambiguous | Preserve that result; later explanation work cannot turn it positive |
| No third animal is available | Stop at internal held-session evidence and state the missing validation directly |

## Figures, contingent on evidence

| Figure | Scientific judgment | Required content |
| --- | --- | --- |
| 1. Does the acquisition rule work? | One global method beats all cost-matched simple rules | Sequential design, full nine-cell surface, all sessions and animals, random distribution, full-resource ceiling |
| 2. Which electrodes add unique information? | Pilot-only measurements predict conditional electrode value | Example array, signal quality and redundancy, predicted versus observed removal loss, reliability-only comparison |
| 3. Which direction should come next? | Pre-choice information predicts next-trial value beyond balance | Choice timeline, direction-wise errors, all alternative next directions, balanced and random comparisons |
| 4. What is the bottleneck? | Electrode and trial axes show reproducible main effects or interaction | Prespecified grid-step contrasts, no unsupported common-cost conversion, animal/session influence |
| 5. Does the explanation transfer? | The same pilot interpretation and joint method work in a new animal | Third-animal roles chosen in advance, every held session, failures, mechanism and performance predictions |

The abstract should name the pilot-visible property that predicted later value,
say whether electrode selection, trial allocation, or both mattered, and state
the exact validation scope. It must not describe post-pilot retention as
electrode placement, electrode count as measured power savings, retrospective
recorded-data analysis as online interaction, or population-spike prediction
as demonstrated clinical BCI control.
