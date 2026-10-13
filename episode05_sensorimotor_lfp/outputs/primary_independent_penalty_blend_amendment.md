# Proposed exception for one independently regularized W/HW blend

**UNADOPTED — explicit human amendment required, 2026-10-04.** This is a
prospective permission decision, not authorization to implement, compute
scores, read source values or launch jobs. No new mixture score has been
derived. The original proposal and operative GOAL/search policy remain
unchanged and binding. Scientific review of this draft does not adopt it.

## Decision being requested

The adopted [original constant-blend proposal](primary_history_blend_proposal.md),
lines35–36, explicitly states:

> Do not add independent component penalties, negative weights, extra blend components or a finer weight grid after outcomes.

Approve or decline a **single explicit exception for independent component
penalties**, applied only to the separately specified exploratory challenger
below. This releases a restriction after development outcomes have been seen;
it is not an implementation repair, a reinterpretation of the old result, or
permission implied by the broad request to finish the episode. A new recipe
name alone would not remove the prohibition. The restrictions on negative
weights, extra components and a finer alpha grid remain in force.

An affirmative human decision would authorize this scientific amendment to
be recorded and subsequently implemented through the normal episode workflow.
This draft itself grants none of those actions. Any execution still needs a
separately specified bounded resource plan; no runtime, memory, allocation,
pilot or remaining-panel claim is made here.

## Evidence boundary and scientific question

Released evidence ends at40 valid comparisons, patience7. The original
full-support constant blend remains the comparable incumbent: primary
0.0026788544310999856, identity0.004559394718773168, both practical margins
failed. There are21 alternative/stress checks and13 literal null controls
among34 post-initial comparisons. Neither this draft nor an eventual
amendment completes the currently unmet patience stopping requirement.

The question is whether requiring W and HW to share the same nominal ridge
penalty limits their useful error complementarity. Their different covariance
spectra mean that equal penalties do not imply equal shrinkage or effective
degrees of freedom. Changing a component's penalty can change the direction
of its prediction error, not only its amplitude, so alpha generally cannot
absorb the change. Mixture loss contains the cross-term between component
errors; independently optimal endpoint penalties do not establish the optimal
joint pair.

This motivates testing the constraint, not expecting a gain. The released
support-stress results distinguish a benefit over HW at the new mixture's
penalty from a loss against the previously selected HW endpoint. They do not
prove that independent penalties help on original full support. No new
outcomes or per-bin errors were inspected to choose this candidate.

## Exact finite challenger if the exception is approved

Use only the existing complete original full-support W and HW five-penalty
prediction arrays under `runs/adaptive_search/`:

- `primary_repeatability_weight_control_attempt001/` for W.
- `primary_history_repeatability_weight_control_attempt002/` for HW.

Both penalties use the unchanged grid `[0.001,0.01,0.1,1,10]`; historical
weight uses the unchanged grid `[0,0.25,0.5,0.75,1]`. The proposed full native
XY prediction is

`P(lambdaW,lambdaHW,alpha)=(1-alpha)*PW(lambdaW)+alpha*PHW(lambdaHW)`.

There are125 nominal parameter triples. Canonicalize alpha0 by setting
lambdaHW=lambdaW and returning the literal W array; canonicalize alpha1 by
setting lambdaW=lambdaHW and returning the literal HW array. This leaves at
most85 distinct predictions:75 interior triples and ten endpoint arrays.
Retain all85 canonical parameter labels for the declared selector and tie
rule, even when additional literal prediction coincidences allow cached
computation. Do not prune labels by session, outcomes or array coincidence.
Inactive endpoint penalties never participate as spurious tuning choices.

Use the original blend helper for literal endpoints and unchanged interior
arithmetic. Both components already contain the same B; do not add it again.
No new neural fit, source read, feature, target, interval, component, alpha
value or numerical preprocessing is introduced. Replay original trial IDs,
directions, physical support, Y/B values and memory-layout convention,
original D and all five correct/wrong-pair component score rows exactly.
No tolerance or fallback permits a failed replay.

For every held outer session, select the canonical cell using only the other
fifteen sessions' original primary increments, averaging sessions equally
within each animal and then the two animals equally. Apply the exact maximum
tuple

`(training_primary_score, diagonal, min(lambdaW,lambdaHW), max(lambdaW,lambdaHW), lambdaW, -alpha)`

where diagonal is1 when lambdaW=lambdaHW and0 otherwise. Exact ties first
retain the original diagonal policy, then use the stated deterministic
regularization and alpha order. On the diagonal this tuple is
`(score,1,lambda,lambda,lambda,-alpha)`, exactly preserving the old
`(score,lambda,-alpha)` ordering. Thus an off-diagonal cell must strictly beat
the best diagonal training score to replace it. No near-tie tolerance is
introduced. The entire old25-cell policy remains nested; its old selectors,
saved selected arrays and scores must replay independently.

The held session, identity score, animal-specific signs and omission results
cannot choose either penalty or alpha. The same global selection procedure
applies to all sessions; no separate animal-specific model choice is added.

## Evaluation, reuse and scientific count

Preserve original whole-direction fits, all sixteen complete sessions,
native joint-XY denominator, behavioral baseline, negative scores, and exact
within-direction whole-trial wrong-pair mean. Keep primary0.01/identity0.005
margins and all original animal, majority, omission and finalist requirements.

Report direct common-D SSE and identity differences against the original
selected constant blend and original independently selected W/HW endpoints.
Also report W at the selected lambdaW and HW at the selected lambdaHW as
descriptive component anchors. Preserve both animal means, all session
signs, all fixed-prediction within-animal omissions and every selected triple.
These contrasts add no threshold or count. A nested training-grid advantage
does not guarantee an advantage in held outer sessions.

Before charging, compare the complete selected prediction panel literally
and independently against each already evaluated original selected W, HW,
constant blend, H2W and time-varying blend panel. Replay each frozen anchor's
original selector, exact target/support and selected prediction scores; do
not retune it to the new pair. The latter two are duplicate references only,
not mixture components or additional performance objectives. All sixteen
selected arrays must equal one same fixed anchor for a global duplicate;
different matching anchors by session, parameter tokens, equal means or
numerical closeness are insufficient. Name every exact global match and
charge zero. Do not force an off-diagonal or interior choice to earn depth.

Otherwise one complete distinct original-support policy contributes at most
one candidate, zero alternative/stress credit and zero literal-null credit.
The nominal triples, canonical cells, anchors and diagnostics add no trials.
Apply the existing patience rule using the actual incumbent at completion:
only a strictly higher complete comparable animal-balanced primary score
can reset patience. Identity improvement or a diagnostic contrast cannot.
Exact duplicates, incomplete panels and technical failures add no count.
No minimum/maximum depth, patience, control-allocation or resource ceiling
is amended by this exception.

## Main tradeoff and limits

The proposal enlarges selection from25 to at most85 distinct prediction
choices on the same exposed sixteen sessions and two animals. The outer
fifteen-session rule controls each session's parameter selection; it does
not undo repeated adaptive choice of recipes on this corpus or make the
overlapping outer folds independent replication. A gain could reflect this
added selection flexibility. Equal nominal penalties, matched grid sizes
or common alpha do not establish matched model capacity.

Any result remains exploratory development. It would test the shared-penalty
constraint, not identify a neural mechanism, establish contamination-free
LFP, provide online validity or replace absent future sessions. No benefit,
promotion, patience reset or further candidate is promised. Pending an
explicit human amendment, the correct action is to preserve this draft and
perform no implementation or mixture-score computation.
