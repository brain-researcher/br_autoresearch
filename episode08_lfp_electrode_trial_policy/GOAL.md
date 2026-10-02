# What makes the next electrode or calibration trial useful?

[Scope and novelty review, 2026-10-02](outputs/scope_novelty_review_20261002.md).

EP08 keeps its exploration open: **what makes electrode and calibration
information valuable, and when can that value be predicted?** One candidate
explanation is conditional complementarity and reducible uncertainty. At
comparable signal quality, an electrode may be valuable when its available
training/calibration signal predicts population activity not recoverable
from the other retained electrodes. A reach direction should be valuable when
another trial is expected to reduce uncertainty about that mapping, not merely
when its current error is large. The retained set changes the mapping being
estimated, so a quality-matched electrode swap can change the predicted order
of next-direction value. That last prediction distinguishes a joint rule from
two independently useful acquisition methods.

This is a testable application of established conditional sensor selection and
experimental design, **not a first algorithm or a demonstrated biological
mechanism**. The public source is absent and no neural analysis has run. The
original matched-budget protocol is now an **optional reference scenario**,
not the research-wide prerequisite. Its numerical decision remains unchanged.
Additional named-rule scoring remains proposed and unactivated, detailed in the
[acquisition-principle proposal](outputs/acquisition_principle_proposal.md).

There is **no mandatory separate 16-trial pilot**. Explore models, temporal
summaries, training/calibration amounts and coverage, static/source-trained
selection, and response-adaptive initialization on development sessions.
Neither low-rank linear proxies nor a fixed initializer are preselected.
Zero-shot electrode selection does not imply zero-shot population decoding.

The [open-exploration amendment](outputs/open_exploration_design.md) separates
two questions. Offline information-value analysis can use a declared training
pool to predict conditional electrode/trial value. Sequential acquisition can
use only what was actually available before each action and must count all
initializer and later acquisition exposure. The first does not automatically
support the second's efficiency claim. Within a selection comparison, decoder,
target, tuning opportunity and evaluation support must be matched.

![EP08: conditional electrode value, trial value and a real electrode swap](outputs/figures/ep08_episode_series-v6.png)

Conceptual illustration: center-out reaching links electrode LFPs to population
activity; the same candidate electrode can overlap or complement different
retained sets; mapping uncertainty differs from trial variability; an electrode
swap raises a prediction about the next useful direction with calibration data
held fixed. The five schematic observations in each uncertainty illustration
are not a prescribed pilot size. The swap changes electrode membership, not
just color; it asks sensor-dependent trial value, not factorial score
interaction. The composition retains the EP05/EP12-inspired v5 style. The
[exact imagegen prompt](outputs/figures/ep08_episode_series-v6-prompt.md) and
[earlier protocol overview](outputs/ep08_open_exploration-v3.png) are retained. The
[v2 figure](outputs/ep08_conditional_value_principle-v2.png) and its
[prompt](outputs/ep08_conditional_value_principle-v2-prompt.md) now illustrate
only the optional 16-trial reference case. Earlier PNG/SVG assets are retained.

All branches share the existing search/resource allowance and the single
held-session opening. Development can revise candidates; the chosen final
protocol, comparisons, endpoints and decisions are fixed before held scoring.
The source and external third-animal data remain unavailable; no analysis or
Git action is launched by this amendment.

## Candidate predictions, not a closed hypothesis menu

| Quantity known before the choice | Proposed prediction | Discriminating failure |
| --- | --- | --- |
| Allowed training/calibration target-linked signal left after conditioning on the retained set, with quality matched | More complementary electrodes have larger later conditional removal loss than equally clean redundant electrodes | Reliability or marginal target correlation ranks removal loss equally well |
| Calibration geometry and uncertainty in fitted parameters, with noise and direction counts accounted for | Larger expected reduction in parameter uncertainty predicts greater repeatable next-direction benefit | High-error directions remain noisy; the proxy does not predict benefit beyond balance |
| The same calibration trials under a quality-matched electrode swap | Recomputed pre-choice scores predict the change, or stability, of direction-value ordering | Two policy gains occur, but the predicted ordering change does not repeat |

The named candidate scores use regularized, trial-cross-fitted linear proxies, not an
estimated high-dimensional information-theoretic quantity. Sixteen pilot
trials do not establish that these rankings are identifiable or precise.
Whole trials are the fitting/validation units; time bins do not create extra
pilot replicates. Fitting-only centering and target scaling, rank limits,
shrinkage, pooled noise estimation, and direction support are specified in the
proposal for that candidate. Other supported estimators remain open. Decoder
capacity is matched within each selection comparison; reduced-rank ridge is
the unchanged decoder of the optional reference scenario.

## The scientific question

Can information in a declared training/calibration state reveal future value?

1. **Nonredundant electrode value:** an electrode is useful because it adds
   population information not already present on the other retained
   electrodes, not merely because its signal is clean.
2. **Next-trial value:** the current calibration state predicts which reach
   direction will benefit most from one additional trial.

Value analysis need not wait for short-pilot feasibility or a policy win.
Independently, a sequential study can ask whether an available-information
rule improves neural prediction at matched total resources. EP08 also asks a
third, stricter question: **does changing the retained electrode set
predictably change which reach direction is worth sampling next?** Without
that dependency, electrode selection and trial allocation may be two useful
methods, but they are not yet a joint acquisition principle.

## At a glance

| Question | EP08 design |
| --- | --- |
| What information is explored? | Declared training/calibration pools, different initial amounts and coverage, static/source-trained and response-adaptive rules. No compulsory separate pilot. |
| What is chosen? | Development explores electrode sets, trial value and information states. Sequential action rules must be specified before comparison. |
| What is the 16-trial design? | One optional reference: retain 4/8/16 electrodes and add 32/64/128 trials, preserving its nine-cell decision intact. |
| What is predicted? | Motor-population spike activity on untouched trials from the same session. |
| What is the fair comparison? | Match decision-time information, actual total exposure, target, decoder and tuning; keep offline-value and sequential-efficiency claims distinct. |
| What must repeat? | The selected branch's prospectively specified predictions and decisions across sessions/animals. The old `0.005`, 3/4 and scarce-cell conditions apply only to the reference. |
| What would explain a win? | Allowed-data predictions of later conditional value, repeatable next-direction benefit and, for a coupling claim, sensor-dependent trial ordering. These remain hypotheses. |
| What would confirm the explanation? | The same predictions and acquisition rule work in newly sequestered sessions from a third animal. |

## Optional reference scenario: original 16-trial acquisition protocol

**Scope of the remaining protocol sections:** pilot-only information, the
fixed grid/decoder, family coverage, `0.005` primary decision, replay settings
and terminal conditions below describe `reference_16_trial_acquisition` only.
They do not gate other supported offline analyses or initializers. Global
data separation, shared resource ceilings, one held opening and honest claim
limits still apply to every branch. New endpoints/stopping rules need a
prospective specification; no automatic transfer of the reference margin.

### Why a winning policy is not yet an explanation

Two electrodes can be equally clean yet carry nearly identical information.
Keeping both may waste a scarce channel. A slightly noisier electrode may be
more useful if it captures population variation missing from the first one.
The explanatory test therefore asks how much prediction worsens when each
retained electrode is removed from an otherwise unchanged set.

The same distinction applies to trial selection. Two directions can have the
same large calibration error: one because its mapping is poorly estimated,
the other because trial-to-trial noise remains high even after its mapping is
stable. The proposed rule favors expected reduction in the first uncertainty,
not the largest error. It must predict benefit before the additional trial or
its later evaluation benefit is known.

A large current error is not enough. It could be irreducible trial-to-trial
noise, in which case one lucky next trial may appear helpful but the benefit
will not repeat. EP08 therefore evaluates next-direction predictions at
prespecified replay states and across several trial orders. A direction is
learnable only when its predicted benefit repeats beyond one favorable draw.

The two decisions must also be separated experimentally. At every fixed
electrode count and trial count, EP08 evaluates all four combinations:

| | Balanced trials | Adaptive trials |
| --- | --- | --- |
| Simple electrode selection | Reference | Trial-selection contribution |
| Conditional-value electrode selection | Electrode-selection contribution | Combined contribution |

Two positive component gains alone support two useful methods. Evidence for
sensor–trial coupling additionally requires the retained
electrode set to alter the ordering of next-direction value in a way predicted
before those future trials are revealed.

Coupling and aggregate factorial interaction are different quantities: changed
direction values can cancel in an aggregate score, and an interaction alone
does not establish the proposed explanation. The optional reference retains
its inherited stronger joint-claim decision; it is not a universal definition
of coupling.

Electrode count and trial count have different physical meanings. EP08 reports
performance changes along each axis of the 3-by-3 grid. It does not claim that
one electrode is worth a particular number of trials or a measured amount of
power without a separate cost study.

## Data and session roles

The primary corpus is six Mihili and six Chewie-L M1 execution sessions from
the public Foundation LFP source. If the sessions meet the structural
requirements, four sessions per animal are used to develop one global policy
and two per animal remain untouched until the policy and analysis are final.
Whole sessions, not trials from the same session, separate development from
the final test.

The public outcomes have been used before. This study can therefore provide a
careful internal held-session test, but not an independent animal-level
replication. Chewie-R is another implant in Chewie and may be used only as an
implant-sensitivity analysis. A genuinely independent third animal is needed
for an external generalization claim.

Within every session, evaluation trials are set aside and never used to choose
electrodes, train the acquisition rule, or fit the decoder. The 16 pilot trials
and the later calibration trials come from a separate pool. Within each reach
direction, that pool is placed in one ordinary seeded order before any policy
score is known. Asking for a direction reveals only the next trial in that
direction's order.

## What happens after the pilot

The electrode decision comes first and is permanent for that session. A method
may retain 4, 8, or 16 physical electrodes, each carrying LMP, 100--200 Hz
power, and 200--400 Hz power. It may not add an electrode later or retroactively
use a discarded electrode on a previously collected non-pilot trial.

After electrode selection, each trial decision chooses one of the eight reach
directions. The analysis then advances to the next prespecified recorded trial
from that direction. This retrospective walk-through preserves what would have
been known at each choice, but it does not show how an animal would respond to
a real-time request.

Every method uses the same reduced-rank ridge decoder, population target,
movement window from 150 to 450 ms after movement onset, and untouched
evaluation trials. Decoder flexibility therefore cannot masquerade as an
acquisition advantage.

## Acquisition rules considered

Candidate methods may combine one electrode selector with one trial allocator.

The factorial explanation uses two named selectors. The **simple selector**
ranks electrodes by pilot signal quality and reliability. The
**conditional-value selector** also uses pilot-visible redundancy and
complementarity to predict the loss caused by removing an electrode from the
retained set. Geometry-only and repeated-random selectors remain benchmarks,
but they do not replace either named row of the factorial comparison.

Electrode selectors may use:

- signal reliability, artifact fraction, missingness, line noise,
  stationarity, and repeatability from data already acquired;
- spatial diversity or array coverage, with a stated fallback when geometry is
  unavailable;
- redundancy measures such as correlation, conditional variance, or
  log-determinant gain; or
- a simple learned ranking rule trained across development sessions from
  pilot-visible features.

Trial allocators may use:

- balanced direction counts;
- current calibration uncertainty;
- residual diversity;
- information gain; or
- a bounded upper-confidence rule.

Lookahead is limited to two acquisition steps. One global method is selected
for all sessions and animals; choosing a different winner for each session is
not allowed. Learned value labels for one development session must be produced
without using that session to train the ranker.

## Fair comparisons and primary measure

At every `(electrodes, additional trials)` point, fit the same decoder for four
prespecified combinations. Write their held-out performances as `Q_SB`
(simple electrodes, balanced trials), `Q_SA` (simple, adaptive), `Q_CB`
(conditional-value, balanced), and `Q_CA` (conditional-value, adaptive).

Report these contrasts in every grid cell:

```text
electrode-selection contribution = Q_CB - Q_SB
trial-selection contribution     = Q_SA - Q_SB
combined contribution            = Q_CA - Q_SB
algorithm interaction            = Q_CA - Q_CB - Q_SA + Q_SB
```

Also report the electrode contrast under adaptive trials (`Q_CA - Q_SA`) and
the trial contrast under conditional-value electrodes (`Q_CA - Q_CB`). The
interaction distinguishes a combined winner from two individually useful
components; it does not by itself explain why they interact.

Additional comparisons are:

- repeated uniform-random electrode selection plus balanced-random trials;
- geometry-only farthest-first electrode selection;
- reliability-only electrode ranking;
- the full-resource decoder as a ceiling check; and
- a development-only hindsight best case, reported only as an upper bound.

For each session and grid cell, compute the difference in held-out `R2_SSE`
between the candidate and the strongest simple rule chosen from development
evidence. Give all nine grid cells equal weight within a session, average
sessions within each animal, and give Mihili and Chewie-L equal weight. The
prespecified meaningful improvement is `0.005 R2_SSE`.

The complete nine-cell surface, every session, both animals, and negative
effects are reported even when the overall average is favorable.

## Search plan and budget

The study proceeds in five scientific stages:

1. Confirm the animals, sessions, physical electrodes, reach directions,
   pilot support, trial pools, LFP features, and population targets.
2. Establish repeated random, spatial, reliability, and fixed trial-allocation
   comparisons using common seeds and identical budgets.
3. Evaluate at least 20 diverse methods spanning every selector and allocator
   family.
4. Use development results to propose bounded combinations and parameters.
5. Stress-test the best method across subset seeds, budget settings, animals,
   and alternative explanations, then choose one global policy for the
   untouched sessions.

Run at least 40 and at most 112 valid method evaluations. After the minimum,
stop when 20 consecutive valid evaluations improve neither the primary score
by at least `0.005` nor the feasible utility/robustness/cost frontier. Coverage
of every selector and allocator family, the simple comparisons, and all
required alternative-explanation checks must be complete before patience can
stop the study. At least 40% of post-coverage work is reserved for those
checks. Up to 16 failed runs caused by engineering problems may be retried
without counting as scientific patience.

The resource limit is CPU only: at most 1,200 aggregate CPU-hours, 120
wall-clock hours, 48 concurrent cores, 256 GB memory, and 750 GB temporary
storage. Reaching a resource limit makes the study incomplete; it is not a
positive or negative scientific result.

## Alternative explanations that must remain visible

The selected method must be compared with:

- many random selections with identical action counts and compute;
- learned electrode rankings after their development value labels are
  permuted;
- spatial selection after electrode locations are shuffled;
- versions with reliability, diversity, learned score, or adaptive trial
  allocation removed;
- one-for-one electrode swaps matched on pilot signal quality, artifact rate,
  and missingness but differing in predicted complementarity;
- next-direction benefits at prespecified common replay states and across
  several ordinary trial orders, rather than one favorable next trial;
- transfer in the reverse animal direction and omission of each development
  session;
- training on one animal and applying the rule to the other;
- a step-by-step check that every decision used only information available at
  that time;
- low-frequency-only and spike-contamination sensitivities; and
- simulated worlds that are saturating, spatially clustered, redundant,
  trial-limited, or null.

These comparisons distinguish a genuine pilot-guided acquisition principle
from signal quality alone, spatial spread alone, direction balancing,
irreducible direction-specific noise, one fortunate future trial, one
favorable session, or information that would not have been available at the
time of the decision.

The named conditional-complementarity / reducible-uncertainty proposal does
not replace this required comparison set or automatically become the selected
policy. Its linear proxy lies within the existing redundancy-aware,
linear-ranking, uncertainty, and information-gain candidate families. Any
additional rule-prediction score, pair-selection criterion, effect margin, or
confirmation decision must be specified prospectively in a separate amendment
before its relevant outcomes are accessed. It cannot amend the primary
`0.005 R2_SSE` decision after a failed or ambiguous policy result.

## Held-session decision

Before the four untouched sessions are scored, choose one global acquisition
method, its fitted development state, the common decoder, pilot and trial
ordering, all grid cells, comparisons, effect margins, uncertainty method,
missing-geometry behavior, and result table. The method sees only the common
pilot and the calibration information available after each permitted choice.

A positive primary result requires all of the following:

- mean improvement of at least `0.005 R2_SSE` over the strongest
  cost-matched simple rule across the nine-cell grid;
- positive mean improvement in both Mihili and Chewie-L;
- improvement in at least three of the four untouched sessions;
- success at both `(4 electrodes, 32 trials)` and `(8 electrodes, 64 trials)`
  in both animals;
- no material loss relative to the full-resource ceiling under the
  prespecified equivalence margin;
- robustness when one animal or influential development session is omitted;
  and
- all required alternative-explanation and information-timing checks pass.

All four factorial combinations remain visible regardless of which method
wins. A positive `Q_CA - Q_SB` establishes a combined-policy result, not a
joint principle. In this optional reference, the stronger joint interpretation additionally requires a
resolved algorithm interaction and the electrode-dependent next-direction
prediction described below. If both component contrasts are positive but the
interaction is practically absent and direction order is unchanged, report
two individually useful acquisition methods.

Possible study outcomes are: a reproducible joint policy, a positive but
ambiguous result, no policy clearing the prespecified rule, an incomplete
study, or an invalid analysis caused by broken data separation or scoring.
Once untouched-session results are visible, they cannot guide another search
within the same study.

## Tests that explain why the policy works

### Does the pilot predict unique electrode value?

For a retained set `S`, define an electrode's conditional value as:

```text
V_e = evaluation loss without electrode e - evaluation loss with the full set S
```

Both losses use the same calibration trials, untouched evaluation trials, and
decoder procedure. Before seeing these removal losses for a session, predict
their ordering from pilot-only reliability, geometry, redundancy, and the
selected pilot score. A useful explanation must outperform reliability alone
and repeat in both animals.

### Does the current state predict next-trial value?

At one acquisition step, separately evaluate what would have happened after
adding the next available trial from each reach direction:

```text
V_direction = loss before the trial - loss after adding that direction's next trial
```

The method must predict this ordering before receiving those trials or their
evaluation benefits. Candidate predictors are current direction counts,
cross-validated calibration error, uncertainty, and residual diversity. The
chosen direction is compared with balanced and random choices at the same
point in the recorded trial order.

The proposed reducible-uncertainty proxy averages a possible next trial over
the direction's **already acquired** feature distribution. It never reads the
next recorded trial's LFP or target in order to decide whether to request it.
Its parameter-uncertainty reduction is not the total predictive variance or an
exact expectation of the final decoder's held-out improvement. A shrinking
proxy is meaningful only if it predicts repeatable realized benefit in the
branch tests; a formula alone does not establish learnability.

The test is repeated from common, outcome-independent replay states under
several prespecified within-direction orders. A useful score must predict the
average ordering of future benefit and beat balance across states and orders.
Large error without repeatable improvement is classified as irreducible or
unmodelled variation, not as evidence that the direction should be sampled.

These are predictive labels, not causal effects. Electrode-removal loss does
not prove that an electrode is biologically special, and the recorded-trial
analysis does not establish that real-time trial requests would have the same
effect.

### Are electrode and trial choices actually coupled?

At each common replay state, hold the calibration trials fixed and compare a
simple retained set with a conditional-value set of the same size. Include
one-for-one swaps whose electrodes are matched on pilot quality but differ in
predicted complementarity. Before revealing any branch trial, predict how the
swap will change the ordering of `V_direction`.

The preferred explanation predicts that the retained set changes which neural
population components remain recoverable, so the direction with the greatest
learnable deficit should change in a predictable way. The competing
explanation is that electrode quality and direction imbalance help
independently; under that account, quality-matched swaps should leave the
next-direction ordering essentially unchanged. This prediction must repeat
across replay states, trial orders, sessions, and both animals before EP08 uses
the phrase **joint acquisition principle**.

### Which resource is limiting?

Report changes along the electrode and trial axes separately, for example:

```text
electrode step at N trials = R2(8,N) - R2(4,N)
trial step at E electrodes = R2(E,64) - R2(E,32)
```

Also test whether the benefit of more electrodes depends on trial count and
vice versa. Do not convert these axes into a common cost without direct
measurements of power, bandwidth, recording time, or participant burden.

## External confirmation and claim boundary

The four current untouched sessions test the acquisition policy itself. The
electrode-removal and next-direction explanations are developed afterward and
therefore require newly sequestered sessions for confirmation. The preferred
test is a third animal with compatible M1 LFP, population spiking, physical
electrode geometry, eight reach directions, and enough trials for the same
pilot and grid.

A successful internal result would show that, in these recordings, a common
16-trial pilot can guide post-pilot electrode retention and calibration-trial
allocation better than cost-matched simple rules. It would not establish
from-scratch electrode placement, causal electrode importance, measured power
savings, online behavioral control, a universal electrode count, or
generalization to a new animal. The [paper plan](outputs/paper_plan.md)
specifies how each result branch changes the final claim.
