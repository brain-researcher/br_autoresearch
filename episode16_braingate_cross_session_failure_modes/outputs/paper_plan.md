# EP16 paper plan

## Paper question

When an offline human iBCI mapping transfers poorly to a later session, is the
failure explained by less locally recoverable signal, by observable changes in
the recording, or by a mismatch that 32 labelled trials can repair?

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

**Decision:** both the primary and robust contrasts must clear the pre-set
positive margin in each of the three held-out participants.

**If the result is absent:** the main conclusion is that this analysis finds no
additional long-gap mapping deficit. The later-session signal comparison is
still reported, but the recording-state and remapping explanations are not
presented as explanations of a deficit that was not established.

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

**Decision:** both the fractional local decline and raw transported above-null
decline must pass their negative margins.

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
| No extra long-gap loss | Test whether the absence is specific to the released feature and direction proxy; do not continue mechanism hunting in this episode |
| Locally weaker signal | Seek raw-voltage or richer-behaviour data in a new study that can distinguish recording quality from representation and task effects |
| Recording-state sufficiency | Prospectively measure channel quality and evaluate a recording-aware normalization strategy without claiming hardware causation |
| Thirty-two-label remappability | Test the selected low-budget method prospectively in closed loop with a separately governed protocol |
| Composite signature | Design an intervention study that changes one factor at a time; do not infer additive shares from EP16 |
| Repeatable participant differences | Recruit additional participants and predefine profile classification before estimating prevalence |
| Unresolved | Report which denominator, support, or uncertainty requirement failed; narrow the next experiment to that bottleneck |

## Main claim boundary

The strongest EP16 paper can make an operational claim about offline transfer
in this publication-exposed release. It cannot identify hardware failure,
neuron loss, representational drift, conscious intention, historical online
decoder behaviour, clinical benefit, or population prevalence.
