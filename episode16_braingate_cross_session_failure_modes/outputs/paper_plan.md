# EP16 paper plan

## Paper question

Across longer gaps, does a later session contain less locally recoverable
direction signal, does an earlier-session mapping lose a larger fraction of
the signal that remains, and—when that extra transport loss exists—can
observable recording change reproduce it or 32 labelled trials repair it?

The paper should not begin with a model catalogue. It should begin with one
earlier-session mapping, one later session, and the fact that a poor transferred
score has several possible meanings.

## Figure 1 — Is there an extra long-gap problem?

**Scientific judgment:** mappings transferred across at least 180 days lose
more of the later session's recoverable above-null performance than mappings
transferred across 1–30 days.

**Panels:**

- the source-session to target-session design;
- the null, later-session, and transported scores on the same trials;
- the participant-level near-versus-long contrast after accounting for general
  source and target difficulty; and
- the target-balanced median robustness check.

**Adjusted contrast:** within each connected participant-by-schedule component,
use chronological pairs only. Let `B_t`, `L_t`, and `F_st` be the null,
target-local, and transported target-trial scores after each has been averaged
equally across the two reversed outer folds. Define
`R_st = (L_t - F_st) / (L_t - B_t)`, with no clipping or epsilon replacement.
If the denominator is nonfinite or fails its pre-set positive readiness
minimum, the pair and registered connected component are unresolved rather
than being dropped. Fit `R_st` with a component average,
source-session effects, target-session effects, and one long-gap indicator.
The long-gap coefficient is
interpretable only if the pairing pattern contains both gap classes and that
indicator retains variation after source and target identity are removed. The
constrained design must also have full rank and pass pre-set conditioning,
residual-information, and single-session-leverage limits. Six pairs in each
class alone are not enough. Session-node resampling rebuilds all pairs and nests
block/trial resampling within sessions; pairs are not treated as independent.
Use 9,999 draws. A draw that loses rank or a required denominator is not redrawn
or dropped: it is assigned the adverse infinite tail for the bound being
formed. If this makes a 95% bound unbounded, the participant is unresolved.
For the robust contrast, take the median across chronological sources within
each target and gap class, then the median across targets, then long minus near
within component; retain the same equal schedule-group and component weights.

**Decision:** both the primary and robust contrasts must clear the pre-set
positive margin in each of the three held-out participants.

**If the result is not supported:** conclude only that the positive extra-loss
margin was not cleared. This is not evidence of equivalence or absence. The
later-session signal comparison is still reported, but the recording-state and
remapping explanations are not presented as explanations of a deficit that was
not established.

## Figure 2 — Does the later session itself contain less usable signal?

**Scientific judgment:** distinguish a transported mapping that fails while a
later-session mapping remains strong from a later session in which even the
local benchmark weakens.

**Panels:**

- later-session model minus null across time;
- transported model minus null for the same chronological pairs;
- near-versus-long fractional change; and
- learning curves showing whether a weak local score is simply caused by too
  little fitting data.

**Decision:** define `J_u` as a session-local model score minus that session's
null. For each source-to-target transition, calculate
`(J_target - J_source) / J_source` and the transported score minus the target
null. Use chronological source-to-target pairs; reciprocal transport is a
separate sensitivity and cannot enter this decline estimand. Within each lag
class and connected schedule component, give every target
equal total weight and divide it equally among that target's eligible sources.
Compare the long mean with the near mean for both quantities, use the identical
component set, and then weight schedule groups and components equally. Both
simultaneous upper bounds must pass their negative margins.

This route is adjudicated even when Figure 1 does not establish extra
normalized transport loss. A later session may contain less locally recoverable signal
while the transported mapping loses the same fraction of what remains.

**If only this route passes:** claim less locally recoverable direction signal
under the specified analysis, not neural loss or biological degeneration.

## Figure 3 — What can reproduce or repair the deficit?

**Scientific judgment:** test two explanations side by side rather than choosing
one after seeing the data.

**Panels:**

- observed versus emulated changes in local signal and transport loss;
- identity-aware recording-state emulation versus matched-random channel
  degradation;
- 16/32/64-label recalibration curves; and
- output-only, orthogonal, low-rank, and bounded supervised recalibration at
  equal data budgets.

**Decisions:** recording-state sufficiency requires the emulator to reproduce
both near- and long-gap changes and the extra long-gap deficit. Low-budget
remappability requires the selected 32-label method to recover at least 50% of
the gap and at least 0.10 of the locally recoverable above-null signal.

**If both pass:** report a composite operational signature. Do not divide the
loss into causal percentages.

**If neither passes:** the long-gap deficit remains real but unexplained by the
registered mechanisms.

## Figure 4 — Does the explanation repeat across people?

**Scientific judgment:** determine whether one explanation holds in all three
held-out participants or whether fully resolved differences repeat within each
person.

**Panels:**

- a compact signature map for the three participants;
- simultaneous uncertainty for every conclusion-driving comparison;
- two time-based within-person replication partitions when heterogeneity is
  eligible; and
- sensitivity results for `tx_4_5`, survivor electrodes, reciprocal session
  direction, and target-schedule groups.

**Decision:** a shared explanation requires the same result in all three
participants. A heterogeneity result requires at least two distinct profiles,
with each person's profile reproduced in both time partitions and the full
analysis.

**If profiles differ but do not repeat:** conclude that the mechanisms did not
replicate; do not call them participant subtypes.

## Result-dependent next steps

| Result | Next scientific step |
| --- | --- |
| Extra normalized transport loss not established | Still adjudicate the local-signal route; make no absence or equivalence claim, and do not present recording-state or recalibration findings as explanations of an excess transport deficit |
| Locally weaker signal | Seek raw-voltage or richer-behaviour data in a new study that can distinguish recording quality from representation and task effects |
| Recording-state sufficiency | Prospectively measure channel quality and evaluate a recording-aware normalization strategy without claiming hardware causation |
| Thirty-two-label remappability | Test the selected low-budget method prospectively in a separately designed closed-loop study |
| Composite signature | Design an intervention study that changes one factor at a time; do not infer additive shares from EP16 |
| Repeatable participant differences | Recruit additional participants and predefine profile classification before estimating prevalence |
| Unresolved | Report which denominator, support, or uncertainty requirement failed; narrow the next experiment to that bottleneck |

## Main claim boundary

The strongest EP16 paper can make an operational claim about offline transfer
in this publication-exposed release. It cannot identify hardware failure,
neuron loss, representational drift, conscious intention, historical online
decoder behaviour, clinical benefit, or population prevalence.
