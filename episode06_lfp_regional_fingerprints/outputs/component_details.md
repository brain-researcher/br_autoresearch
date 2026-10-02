# EP06: what each population component means

Design clarification, 2026-09-30. These definitions expand the existing
unresidualized component analysis; they add no component, endpoint, threshold,
model restriction or outcome access. No recording or component score has been
examined. The residual-only primary and the EP07 access order remain unchanged.

## One response, three task-design projections

Work within one session and region in its training-defined population axes.
Let `Y[d,i,t]` be the population vector on trial `i` of direction `d` at native
time bin `t`. There are `D` registered directions, `n_d` trials per direction
in the role being evaluated, and `T` fixed bins. Each trial retains all bins.
These formulas require the declared complete direction/time support; they
do not authorize imputation, a new window or dropping an inconvenient direction.

The declared inner product gives equal total weight to directions, equal
weight to trials within a direction, and equal weight to bins:

```text
<U,V>_W = (1/D) sum_d (1/n_d) sum_i (1/T) sum_t U[d,i,t]' V[d,i,t]

m_d(t) = (1/n_d) sum_i Y[d,i,t]
m(t)   = (1/D) sum_d m_d(t)

(P_C Y)[d,i,t] = m(t)
(P_D Y)[d,i,t] = m_d(t) - m(t)
(P_E Y)[d,i,t] = Y[d,i,t] - m_d(t)
```

The directions are averaged equally, not in proportion to trial count. The
identical operators act on predictions: form their own `mhat_d(t)` and
`mhat(t)` from `Yhat`, never insert observed component means into a prediction.
They are weighted orthogonal projectors:

```text
P_k^2 = P_k;   P_k' W = W P_k
P_j P_k = 0 for j != k;   P_C + P_D + P_E = I
||Y||_W^2 = ||P_C Y||_W^2 + ||P_D Y||_W^2 + ||P_E Y||_W^2
```

This is a partition by the declared task design, not a claim that the three
pieces have distinct biological generators or distinct neuron identities.
Time-varying population vectors and their dimension remain those of the frozen
training-defined axes; evaluation does not fit a new basis.

An equivalent implementation uses the time-indicator design `C` and the full
direction-by-time cell design `F`, with diagonal row-weight matrix `W`:

```text
P_C = C (C' W C)^(-1) C' W
P_F = F (F' W F)^(-1) F' W
P_D = P_F - P_C;   P_E = I - P_F
```

All declared cells must have support. These are weighted projections; with
unequal counts, they need not be symmetric under an unweighted Euclidean norm.
The fixed construction rule, not a response-estimated contrast, determines
them for each role's preassigned rows.

## C: common-time activity

**Question:** does the source recording-domain frequency rule help predict the
population trajectory common to all reach directions?

`P_C` repeats the equally direction-averaged vector `m(t)` on every trial.
It retains the direction-invariant trajectory across the analyzed epoch and
any constant level already present in the declared response coordinates.
There is no extra within-epoch grand-mean subtraction in this projector.
Training-only scaling or centering, if part of the frozen recipe, is not
changed here. It removes direction-specific departures and within-direction
trial differences.

For example, if all directions share a rise and fall after movement onset,
that shared pattern belongs here. A source M1 rule helping target M1 more than
pooled or swapped weights would identify a predictive consequence for this
common-time component, not for the identity of a particular trial.

The existing elapsed-time and direction/time baselines matter: a highly
predictable average can reflect task timing or an offset without needing
trial-specific LFP information. Neither a large `S_C` nor the largest component
energy by itself establishes a useful domain rule; the correct-versus-pooled-
versus-swapped gains and existing controls must carry that claim. This is not
necessarily a stimulus-evoked response or a distinct neural mechanism.

## D: direction-by-time activity

**Question:** does the source frequency rule help predict how the population
trajectory for one reach direction differs from the common trajectory?

`P_D` repeats `m_d(t)-m(t)` within a direction. At every bin its equal-direction
average is zero. It retains direction-dependent trajectories, including a
direction-dependent level across the epoch, and removes the common-time
piece and trial-specific deviations within the direction. This is the full
sum-to-zero direction-by-time subspace, not a newly defined pure interaction
with all direction main effects removed.

For example, two rightward reaches share a rightward departure from the common
trajectory; a leftward reach can share a different departure. Those average
contrasts belong here even when the individual reaches differ in amplitude.

A correct-domain gain would locate the source frequency rule's usefulness in
direction-dependent task structure. It does not show decoding of individual
reach deviations, and it does not isolate pure movement intention. Direction
labels can accompany speed, force, timing and other task differences. Keep the
existing behavioral baselines, equal support, smoothness/dimension controls
and both animal-transfer directions visible; do not turn the component name
into a causal physiological explanation.

## E: within-direction, within-time trial activity

**Question:** does the source frequency rule help predict which particular
reach departs from its direction's usual population response?

`P_E` subtracts the role-specific direction/time average and retains the
trial differences left at each bin. Its within-direction trial average is
zero. It removes both common-time and average direction-dependent structure,
not merely the grand mean. It can retain real trial-varying activity, unmodeled
behavior, fluctuations in state, recording variation and noise; “residual”
does not identify which of these generated it.

For example, two rightward reaches at the same time can have the same `C` and
`D` components but opposite deviations along a population coordinate. Predicting
the correct deviation from that trial's LFP addresses `E`; repeating a
direction/time template on every trial yields `P_E Yhat = 0`.

Whole-trial within-direction pairing disruption is especially informative
for this distinction. For fixed predicted trajectories, relabeling trials
inside a direction leaves their `C` and `D` averages unchanged but changes which
trial residual is paired to a particular response. This algebra explains the
control; it does not add a new endpoint or replace the registered shuffle
procedure. A positive `G_E` still needs the existing pairing, high-frequency,
electrode/unit and quality controls. It is neither causal LFP-to-spike influence
nor EP07's history-minus-today-only pairing increment.

### E is not the primary mean-subtracted residual

The primary uses `R_primary = Y_eval - m_train,d(t)`, with its mean learned on
training only. The component target uses `P_E Y_eval = Y_eval-m_eval,d(t)`
in the declared unresidualized response space. In matching coordinates:

```text
R_primary = P_E Y_eval + [m_eval,d(t) - m_train,d(t)]
```

The second term is constant across evaluated trials within direction/time
and can reflect mean drift or estimation error. It belongs outside the
evaluation `E` projection. Therefore the component result cannot be substituted
for the primary profile or used to rescue its result. A strong primary residual
score and a strong `E` score need not coincide.

## Target-evaluation means are score quantities, not fitted templates

Direction labels, role-specific trial counts, native bins, contrast coding
and the frozen axes define the operators before scores. Different trial-role
sizes can give different-sized design matrices; the same frozen construction
rule is applied, not a matrix of the wrong size copied between roles.
Applying a known linear operator to `Y_eval` necessarily computes observed
component means. That is evaluator-only target construction, not a learned
prediction template. Those target means cannot enter `Yhat`, fit axes, refit
target combining weights or repair a model. Permitted development score
feedback can still compare candidates; reserved scores cannot revise choices.

Each rule's `P_k Yhat_eval` is formed independently from its frozen predictions
on the same evaluation trial IDs. Evaluation spikes never define a projector
or its contrast scheme. These batch task projections do not establish causal
or online single-trial decoder performance.

## The same fair consequence test for all three

For each component, source-animal development sessions alone fit the M1, PMd
and pooled combining weights. Swapped is the exact M1/PMd permutation. Target
training may fit axes, scaling and bandwise base predictors, but not a
combining weight. Correct, pooled and swapped use identical target trials,
bandwise predictions, support, population coordinates and projectors.

```text
S_k = 1 - ||P_k Y_eval - P_k Yhat_eval||_W^2 / ||P_k Y_eval||_W^2
G_region,k = S_correct,k - max(S_pooled,k, S_swapped,k)
```

`S_k` measures fraction of component energy predicted, not explained variance.
Zero prediction gives score zero for a scoreable component; wrong predictions
can give negative scores. Do not clip those scores. A large component energy,
a high absolute score or a classifier label is not the correct-domain gain.
Because denominators differ by component and the rules can use different
component-specific weights, do not average the three scores, add their gains
as a single “R2”, or select a family by its raw absolute score.

The existing development-defined positive normalized-energy gate applies to
every region/component. The displayed row weights sum to one, so
`||P_k Y_eval||_W^2` is already the normalized energy. An equivalent implementation
using unnormalized row weights divides by their total weight; neither convention
lets extra trials manufacture support. Low energy or a nonfinite result is
unscorable and unresolved, never recoded as failure. No numerical gate is
chosen by this writing clarification.

Report M1 and PMd gains before averaging. Preserve the existing decision:
both gains positive and equal-region mean at least `0.01` is preservation;
both at most zero is failure; mixed signs, unscorable support or a smaller
positive mean is unresolved. Also show each rule's score and both pairwise
contrasts so a gain is interpretable; these displays add no promotion test.

## Selection and manuscript interpretation stay conditional

The 2026-10-02 [review](scope_novelty_review_20261002.md) prospectively clarifies
nomination eligibility. All three components must be numerically scoreable on
the common nomination sessions; an unscoreable rival leaves the full comparison
unresolved. Regional sign consistency is candidate-specific: a measured
mixed-sign rival cannot be nominated for a joint-region consequence, but does
not veto another eligible candidate. Choose the largest median among eligible
candidates under the same tie margin; this is not global superiority over
mixed-sign rivals. Negative and mixed-sign measurements remain visible.

The `H`, outer-fold stability, and reserved-test rules are unchanged. Do not
examine all three left-out
evaluation components before nominating a family. Additional explanatory
detail is not permission to change that order, fit target combining weights,
introduce a new band, or expand the model/resource budget.

In the combined EP07 manuscript, the supported family would say **where the
source recording-domain rule has predictive utility**:

| Family | Bounded interpretation of supported utility | Not established |
| --- | --- | --- |
| C | Common time/level structure benefits from the source-domain rule | Individual-trial identity or a distinct task-evoked mechanism |
| D | Direction-dependent population structure benefits from that rule | Pure movement intention, trial deviations or cortical causality |
| E | Within-direction trial activity benefits from that rule | Pure biological noise, causal coupling or EP07's history-specific increment |

If several families are similarly useful, report the nomination tie or
instability under the existing margin rather than claiming a unique mechanism.
If recording metadata explains their behavior together, retain the generic
recording-quality alternative. Two animals, four reserved sessions, the array-
region confound and the exposed public corpus remain the evidence limits.

Source/trial support, role fractions, model implementation, weight-separation
criteria and the energy-gate value remain unresolved before empirical work.
The [paper plan](paper_plan.md), [goal](../GOAL.md) and
[policy](../SEARCH_POLICY.yaml) retain the operative rules. This note documents
the three components without activating an analysis or a new confirmation.
