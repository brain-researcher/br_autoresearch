# EP08 paper plan: conditional signal value and learnable calibration deficits

Status: scientific-design revision, 2026-09-30. No acquisition result,
explanatory result, hardware saving, online benefit, or third-animal validation
is claimed. Exploration of models, summaries and training/calibration states is
open. The original 16-trial protocol is now an **optional reference experiment**,
with its numerical decisions unchanged. New branch endpoints and named-rule
scoring remain unspecified/unactivated, not a new scored primary experiment.

![EP08: conditional electrode value, trial value and a real electrode swap](figures/ep08_episode_series-v6.png)

Conceptual illustration, using EP05 and EP12 as visual references. Center-out
reaching links LFP recordings to population activity. A shared candidate
electrode has different value beside different retained sets; mapping
uncertainty is contrasted with trial variability; a same-size electrode swap
raises a prediction about the next direction, with calibration data held fixed.
Panel C uses equal numbers of schematic observations, not a fixed pilot size.
Panel D removes one selected electrode and adds a different one; its question
is sensor-dependent trial value, not aggregate factorial interaction.
[Built-in imagegen edit prompt](figures/ep08_episode_series-v6-prompt.md).

## The intended discovery

Which properties of an electrode set and a training/calibration state predict
conditional population information and the value of another observation?
There is **no compulsory separate pilot**, no prescribed first test of whether
16 trials suffice, and no preselected proxy architecture. The
[open-exploration design](open_exploration_design.md) distinguishes offline
information-value studies from sequential acquisition claims. Models, temporal
summaries, fitting amounts/coverage, source-trained/static selectors and
response-adaptive initializers can all be developed within the shared budget.

Offline value prediction can use a declared training pool without claiming
limited-acquisition efficiency. A sequential rule must use only actually
available observations and count initialization and subsequent exposure. Match
decoder, target and tuning within selection comparisons; zero-shot selection
does not remove population-decoder calibration requirements. Freeze the selected
final analyses before the single shared held-session opening.

The paper should distinguish a **matched-budget policy result** from a
**predictive acquisition principle**. One candidate principle is conditional
complementarity and reducible uncertainty: quality-matched electrodes differ
in population-linked information left after conditioning on the retained set;
equally high-error directions differ in how much parameter uncertainty another
trial can reduce. The core forward predictions are:

1. Can it distinguish a clean but redundant electrode from one that adds
   population information missing from the other retained electrodes?
2. Can pre-choice, reducible uncertainty predict which reach direction will
   benefit most from the next trial, beyond direction counts and persistent
   noise, rather than simply targeting the largest error?
3. Does changing the retained electrode set predictably change which reach
   direction is worth sampling next?
4. Can it predict whether a session is more sensitive to adding electrodes or
   adding calibration trials, while keeping those resources in separate units?

A policy win is not the discovery statement. The intended statement is that a
small acquired-data proxy predicts later conditional electrode loss and
repeatable next-direction benefit, and that recomputing it after a
quality-matched electrode swap predicts a change or stability of the direction
ordering. Different fitting information states may support those estimates
differently; no reliable ranking, interaction, or generalization is presumed.
Sixteen trials are one optional constraint, not the defining problem.
Sensor–trial coupling means electrode choice changes conditional trial value;
factorial interaction means non-additivity of selected policies on a particular
aggregate score scale. Either can occur without the other. The optional
reference retains its inherited resolved-interaction requirement for its
stronger joint claim; this is not a universal definition of coupling.
Two positive component gains alone establish neither explanation.

## A concrete example

Suppose electrodes 11 and 12 are both clean and close together, and their
training/calibration LFP signals are highly correlated. Electrode 37 is slightly noisier but
captures variation missing from 11 and 12. Signal-quality ranking keeps 11 and
12; a redundancy-aware method keeps 11 and 37. If removing 37 later harms
untouched-trial prediction more than removing 12, the available data predicted
**conditional value**, not just signal quality.

Now suppose upward and downward reaches have equally large calibration errors.
Upward error reflects an uncertain mapping; downward error mostly reflects
persistent trial noise around a stable mapping. The proposed uncertainty
reduction score predicts another upward trial will help more. If repeated
branch trials show no such ordering, or direction counts explain it equally
well, the rule is falsified or unsupported. The prediction must be made before
the trial's LFP, target, or evaluation benefit is available.

Neither conclusion follows merely from listing the electrodes or directions
chosen by the best-performing method.

## What earlier work already established

| Earlier work | What it showed | What EP08 must add |
| --- | --- | --- |
| Cohn, Ghahramani and Jordan, JAIR 1996, [primary manuscript](https://arxiv.org/abs/cs/9603104) | Statistical active learning can select observations to reduce prediction uncertainty. | No first uncertainty-sampling claim; test whether an acquired-only proxy predicts repeatable benefit in this recording task. |
| Krause, Singh and Guestrin, JMLR 2008, [primary article](https://jmlr.org/papers/v9/krause08a.html) | Information-based sensor selection already accounts for what a selected set makes predictable. | No first redundancy-aware selection claim; test target-linked conditional value across declared information states and its relation to calibration choice. |
| Flint et al., Journal of Neural Engineering 2012, [doi:10.1088/1741-2560/9/4/046006](https://doi.org/10.1088/1741-2560/9/4/046006) | Motor-cortical LFP features can decode reaching, and performance can plateau after selecting a subset by individual kinematic correlation. | Pilot-only selection of whole electrodes, conditional rather than marginal value, transfer across sessions, and matched trial allocation. |
| Gallego-Carracedo et al., eLife 2022, [doi:10.7554/eLife.73155](https://doi.org/10.7554/eLife.73155) | This source contains region- and frequency-dependent relationships between LFP and population dynamics. | Determine how much of that relationship survives a fixed electrode/calibration budget and why the selected subset works. |
| Even-Chen et al., Nature Biomedical Engineering 2020, [doi:10.1038/s41551-020-0595-9](https://doi.org/10.1038/s41551-020-0595-9) | Neural-interface performance can be studied as channel resources are reduced, motivating lower-power hardware. | A session-transferable post-pilot rule; EP08 still cannot claim measured power or bandwidth savings without those measurements. |
| Sussillo et al., Nature Communications 2016, [doi:10.1038/ncomms13749](https://doi.org/10.1038/ncomms13749) | Historical neural variability can improve decoder robustness to electrode loss. | Choose electrodes and calibration trials from current pilot information under identical budgets, rather than simply train on more historical data. |
| Li et al., Computers in Biology and Medicine 2025, [doi:10.1016/j.compbiomed.2025.110231](https://doi.org/10.1016/j.compbiomed.2025.110231) | An active-learning/domain-adaptation decoder used uncertainty to select a small target-domain calibration set across three monkeys. | Adaptive calibration selection alone is not new; EP08 must jointly test sensors and trials, predict their future value, and respect what information was available at each choice. |

Channel selection and active learning already offer many ranking algorithms. A
new optimizer is therefore insufficient. EP08's contribution must be a clear
statement about which permitted training/calibration properties predict future
LFP-to-population value. This is a proposed empirical distinction, not a
confirmed literature gap or a claim that conditional selection theory has
been invented here. The closest LFP prior already selected/dropped whole
electrodes using marginal correlations; the stricter distinction is
**conditional target-linked value and sensor-dependent learnability**, not
whole-electrode selection by itself.

## The proposed rule is concrete, but is not yet frozen

The [acquisition-principle proposal](acquisition_principle_proposal.md) specifies
two simple linear proxies. The electrode score uses residual target–electrode
cross-covariance after accounting for the retained set, not signal variance or
cleanliness alone. The trial score uses expected reduction in parameter
uncertainty averaged over the requested direction's **already acquired**
features, not its unseen next feature row and not total predictive error.

In the optional 16-trial linear example, summaries have one row per whole
trial; each fold contains the eight
directions. Fitting-only scaling/projections, a rank cap below the fitting-trial
count, ridge/shrinkage, and pooled residual-noise estimates are essential with
only 16 reaches. Even then the proxy may be too uncertain or may miss the
full decoder's dynamic information. These are testable approximations, not
mutual-information estimates or exact expectations of held-out improvement.

The proxy fits existing redundancy-aware, linear-ranking, uncertainty and
information-gain candidate families, so it does not replace family coverage
or automatically become the global winner. Any new rule-value scoring
endpoint, margin, predictable-pair selection, or confirmation criterion
requires a separate prospective amendment. Those choices remain open here;
none may change the original held-session decision or rescue its failure.

## Study paths: not a pilot-first gate

Start supported information-value analyses using declared training/calibration
states. Compare estimators and information amounts on development sessions;
test short-initializer feasibility only where it matters to a proposed
acquisition rule. Candidate predictors below are illustrative, not a closed
menu. Steps 2–4 need not wait for a policy win in step 1.

New-state endpoints, meaningful margins, multiplicity and stopping rules must
be specified before their corresponding scoring/final stage; none inherits
`0.005` automatically. All branches share the existing evaluation and resource
allowance. No resource extension or additional held opening is granted.

### 1. Optional reference: original matched-budget acquisition experiment

Only this reference fixes the 16-trial initializer, reduced-rank ridge decoder,
nine-cell grid, `0.005` primary margin and scarce-resource conditions. Its
terminal rule remains intact; failure or absence does not decide every other
supported branch. The v2 concept figure depicts this reference case only.

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

At each of the nine fixed budgets, run this complete comparison:

| | Balanced trials | Adaptive trials |
| --- | --- | --- |
| Simple quality-and-reliability electrodes | `Q_SB`: reference | `Q_SA`: trial-selection contribution |
| Conditional-value electrodes | `Q_CB`: electrode-selection contribution | `Q_CA`: combined contribution |

Report `Q_CB - Q_SB`, `Q_SA - Q_SB`, and `Q_CA - Q_SB`, together with the
factorial interaction `Q_CA - Q_CB - Q_SA + Q_SB`. Also show the electrode
contrast under adaptive trials and the trial contrast under conditional-value
electrodes. Random and spatial selectors remain useful additional benchmarks,
but they cannot replace one of these four cells.

A combined-policy result in the optional reference requires `Q_CA` to beat the strongest cost-matched
simple rule under the primary decision. If the two component gains are
positive but the interaction is practically absent, the paper reports two
individually useful methods. It does not yet claim a joint acquisition
principle under that inherited decision. This does not negate a separately
supported sensor-dependent trial-value effect whose aggregate interaction
cancels; report that effect under its own prospectively specified analysis.

This step answers whether one global method works on four internal held
sessions. It does not show that a retained electrode is biologically special,
that a requested trial caused an online improvement, or that the method
generalizes to a new animal.

### 2. Ask whether available fitting data predict unique electrode value

For a fixed retained set `S`, define the conditional value of electrode `e`:

```text
V_e = evaluation_loss(S without e) - evaluation_loss(S)
```

Both losses use the same acquired calibration trials, untouched evaluation
trials, target population, and decoder procedure. A positive value means that
the electrode adds predictive information not fully supplied by the others;
it is not a causal effect of implanting that electrode.

Before those removal values are available for a session, predict their
ordering from declared allowed-data quantities, for example:

- reliability and missingness;
- geometry and coverage relative to the other retained electrodes;
- correlation or conditional variance relative to the retained set; and
- a development-trained value score, compared with appropriate simple controls.

Use whole-session cross-fitting: a model predicting values in one development
session is trained without that session. The decisive comparison is whether
the allowed-data rule ranks conditional value better than reliability alone and
whether the relation repeats in both animals. The proposed target-linked
conditional score must also add predictive information beyond marginal
pilot–target association and target-agnostic conditional variance. Neither
high variance nor low correlation with the retained set establishes useful
complementarity by itself.

### 3. Ask whether the current state predicts the best next direction

At a fixed acquisition state, consider the next prespecified trial from each
legal reach direction. Add one such trial, refit the common decoder, and score
each branch on the same untouched trials:

```text
V_direction = loss before - loss after adding that direction's next trial
```

The method must predict the ordering before receiving those trials or their
evaluation benefits. Candidate predictors include current direction counts,
cross-validated calibration error, uncertainty, and residual diversity. The
proposed rule decomposes uncertainty in the fitted mapping from persistent
residual noise and approximates expected parameter-variance reduction using
only already acquired features. It must not condition its choice on the unseen
next trial's LFP, even if those features are accessible in a retrospective file.
Compare the selected direction with balanced and random choices at the same
point in the recorded trial order.

Do not infer learnability from a large current error or from one fortunate
next trial. Fix common, balanced replay states before neural scoring and repeat
the branch test across several ordinary within-direction trial orders.
For the reference, candidate checkpoints are after 16, 32, 64, and 96 additional
trials; retain
only checkpoints supported by every primary session, and freeze that common
list before looking at policy performance. The value score must predict
average future benefit and beat balance across states and orders. Persistent
error without repeatable reduction is evidence for irreducible or unmodelled
variation, not for adaptive sampling.

This is retrospective recorded-data analysis. It assumes that trials within a
direction are reasonably exchangeable in the prespecified order. It does not
show how an animal would respond if a real-time system requested a movement.

### 4. Test whether electrode choice changes next-trial value

This is the decisive prediction for a joint acquisition principle. At a common
replay state, keep the calibration trials fixed and change only the retained
electrode set. Compare the simple quality-and-reliability set with the
conditional-value set at the same electrode count. Also make one-for-one swaps
matched on pilot reliability, artifact rate, and missingness but differing in
pilot-predicted complementarity.

Before revealing any branch trial, predict how each swap will change the
ordering of `V_direction`. The preferred explanation is that the retained set
changes which population components are recoverable, creating a different
learnable deficit across reach directions. It predicts that the direction
ordering changes in the stated way and that the corresponding realized
benefit repeats across replay orders.

Not every swap is expected to reverse the best direction. The forward claim is
that the pre-choice score predicts the change or stability of ordering. Any
minimum score-gap or direction-pair eligibility rule must be fixed before
realized branch values; selecting a pair for a favorable reversal is invalid.

The competing explanation is simpler: clean electrodes and direction balance
are independently useful. It predicts additive performance in the four-cell
comparison and little or no change in next-direction ordering after a
quality-matched swap. That outcome still supports useful acquisition methods,
but not a joint principle.

The coupling claim requires enough trials to branch all legal directions from
the same states, enough future trials to repeat the comparison across orders,
physical electrode identities and pilot quality measures for matched swaps,
and the same untouched population target for every branch. If those data are
not available, keep the supported policy result and drop the coupling claim.

### 5. Keep electrode and trial resources in separate units

For the reference, use the nine-cell surface to report step contrasts below.
Other supported branches choose their resource contrasts prospectively, keeping
the axes separate; they need not implement this exact grid. For example:

```text
electrode step: R2(8,N)  - R2(4,N)
trial step:     R2(E,64) - R2(E,32)
```

Also report whether the electrode step changes with trial count and vice versa.
These contrasts describe sensitivity along two axes. They do not say that four
electrodes cost the same as 32 trials. A common-cost conclusion would require
direct measurements of channel power, bandwidth, recording time, participant
burden, and a cost function chosen before outcomes.

The explanatory prediction is modest: from the declared allowed fitting state, can a simple
rule predict the shape of a session's nine-cell surface or at least the sign of
the prespecified step contrasts? With only eight development sessions, every
leave-one-session and leave-one-animal failure must be reported.

### 6. Test the explanation in a third animal

Before viewing outcomes in an external session, choose:

- the third-animal session roles;
- electrode and LFP-feature correspondence rules;
- fitting/calibration information state, initialization and within-direction order;
- electrode selector, trial allocator, and decoder;
- conditional electrode and next-direction value definitions;
- common replay states, replay orders, quality-matched swaps, and the coupling
  prediction;
- prediction scores, effect margins, uncertainty, and multiple-testing plan;
- behavior when geometry or trial support is missing; and
- the complete result table.

The external study should ask whether allowed-data rules predict conditional
electrode and trial value in every eligible held session, whether changing the
retained set predicts the change in next-direction ordering, and whether the
combined method beats all four factorial comparisons across the supported grid. If development
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
| A high-error direction is irreducibly noisy | Repeat every legal next-direction branch across prespecified common states and trial orders | Do not call error learnable unless predicted benefit repeats beyond one future trial |
| Electrode and trial methods help independently | Four-cell factorial contrasts plus quality-matched electrode swaps at fixed replay states | Report two methods; reserve “joint principle” for a predicted change in next-direction ordering |
| One grid cell creates the average | Full nine-cell surface and session-level influence | Narrow or reject the global policy claim |
| High-frequency features exploit spike-rich content | Low-frequency-only and spike-contamination sensitivities | Limit the signal interpretation or reject the method |
| Recorded data stand in for an online experiment | State the exchangeability assumption and require a prospective interaction study | Keep the claim retrospective |

## When to deepen, narrow, or stop

| Evidence | Next action and permitted claim |
| --- | --- |
| Joint method passes; both value predictions repeat and transfer to a third animal | Develop a joint acquisition principle for the declared LFP representation and recording setup |
| Electrode and trial contributions pass but their interaction and coupling prediction do not | Report two individually useful acquisition methods, not a joint principle |
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
| 1. What information state supports useful choices? | Value prediction and acquisition efficiency are distinguished | Training/calibration state comparisons, models and initializers, actual exposure, matched simple rules; full nine-cell/factorial surface if the reference is used |
| 2. Which electrodes add unique information? | Allowed fitting measurements predict conditional electrode value | Example array, signal quality and redundancy, predicted versus observed removal loss, reliability-only comparison |
| 3. Which direction should come next? | Pre-choice information predicts repeatable next-trial value beyond balance | Choice timeline, direction-wise errors, all alternative next directions, prespecified replay states and orders, balanced and random comparisons |
| 4. Are the choices coupled? | A quality-matched electrode swap predictably changes which direction is worth sampling | Fixed replay states, four factorial contrasts, predicted and observed direction-order changes, repeated future-trial branches |
| 5. What is the bottleneck? | Electrode and trial axes show reproducible main effects or interaction | Prespecified grid-step contrasts, no unsupported common-cost conversion, animal/session influence |
| 6. Does the explanation transfer? | The same allowed-data interpretation and selected method work in a new animal | Third-animal roles chosen in advance, every held session, failures, mechanism and performance predictions |

The abstract should name the permitted-information property that predicted later value,
say whether electrode selection, trial allocation, or both mattered, and state
the exact validation scope. It must not describe post-pilot retention as
electrode placement, electrode count as measured power savings, retrospective
recorded-data analysis as online interaction, or population-spike prediction
as demonstrated clinical BCI control.
