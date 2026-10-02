# EP20 paper plan

[Current conceptual figure](ep20_question_imagegen-v2.png) · [Exact image-generation record](ep20_question_imagegen-v2-prompt.md).

See the [2026-10-02 scope review](scope_novelty_review_20261002.md) for the
prior-work boundary and the unchanged promotion requirement.

## Paper question

Does changing NeuroCam's sensing pads recover cortical detail that better
software on the published hardware cannot? EP20 tests pad geometry and scan
allocation for 1–100 Hz cortical-surface voltage recovery under matched
resources. Optimized published-hardware reconstruction tests software
sufficiency; conventional reconstruction on the same candidate hardware and
simple geometry/schedule controls test whether the joint design adds value.

The explanatory hypothesis is that spatial and temporal demands can be
separated: spatial scale and pad noise/impedance predict when geometry helps;
temporal scale and settling/crosstalk predict when the time-focused recipe
helps. These inputs should predict the published-like, time-focused,
geometry-only or joint family **before reconstruction**. The rival is that
wave orientation, space–time phase and coupled electronics change the preferred
family even when the rule's inputs match. Existing challenges test this
distinction; a large interaction alone does not refute an accurate rule.

A second question is whether published measurements identify a design
preference at all. Different contact and switching mechanisms may reproduce
the same anchors yet rank candidates differently. A material reversal would
identify a missing device measurement, not an optimum. Stable rankings and
successful independent predictions could support one virtual design for
fabrication testing and explain where it should help. No design has been scored.

Methods are in the [component design](component_details.md),
[starting field/pad/scan catalogue](parameter_candidate_catalogue.md) and
[shared electrical fit and probes](electrical_model_and_probes.md). Keep broad
literature-context controls beside deliberate fine/rapid challenges, and
include coverage, ADC work and computation in the matched resource comparison.

## Figure 1 — The five-way fair comparison

**Scientific judgment:** define what must be beaten before a result can be
called co-design rather than better software, different hardware, or a simple
schedule rule.

**Panels:**

- the published 64 × 64 NeuroCam architecture and its three reported dynamic
  settings;
- the five arms in plain language, with C0–C4 shown only as shorthand;
- identical cortical fields, device conditions, masks, and resource budgets
  feeding every arm; and
- the design scope: legal pad geometry, legal scan allocation, and
  measurement-aware reconstruction;
- the set of reference models that all fit calibration and held-aside published
  measurements, with an explicit candidate-rank-reversal check.

**Decision:** the candidate co-design must be compared with the full
published-hardware conventional and optimized frontier, the same candidate
hardware with conventional software, and every registered simple control.

**If resource parity cannot be established:** the study cannot make a co-design
claim, even if reconstruction scores differ.

**If accepted reference models reverse the candidate ranking:** the published
measurements do not identify an optimum. Stop selection and obtain the device
measurement that separates the models.

## Figure 2 — Does the co-design beat the full frontier?

**Scientific judgment:** one candidate improves 1–100 Hz cortical-surface field
R² against every eligible published-hardware setting and every matched control
on paired worlds.

**Panels:**

- member-by-member C3-minus-C0/C1 paired gains;
- C3 versus matched C1, C2, and all C4 controls;
- the least favourable gain across signal and device conditions; and
- resource use before and after fabrication rounding.

**Decision:** lower bounds must reach 0.010 against every C0/C1 setting and
0.005 against C1, C2, and each C4 control. Every condition-specific comparison
must remain above −0.005.

**If C1 matches C3:** optimized software is sufficient under the registered
comparison.

**If C2 matches C3:** the hardware change may help, but special co-designed
software is not supported.

**If C4 matches C3:** the complicated search did not beat a simple rule.

## Figure 3 — Which conditions make each design preferable?

**Scientific judgment:** absolute field and device measurements predict whether
the published-like, time-focused, geometry-only, or joint family will recover
the field best, rather than merely describing an average winner. The
time-focused factor bundles scan and reconstruction software; it does not
isolate temporal acquisition alone.

**Panels:**

- the two field inputs measured from truth or a separate known-input pilot
  before reconstruction: `1/(2*k90)` in micrometres and `1/(2*f90)` in seconds,
  where the 90% cutoffs come from common-mask spatial and temporal field power;
- one absolute characterization table for each frozen D0–D3 representative:
  equal-site input-referred 1–100-Hz RMS noise over the common panel P; maximum
  site-level discrete Q90 impedance at integer 1–100-Hz tones; worst-panel/history
  conditional-mean settling to 1%; and maximum same-victim input-equivalent
  adjacent-line multisine ratio over P victims and physical neighbours. Repeat
  for every device condition, retain unknown/censored values, and identify these
  as panel characterizations rather than full-array maxima;
- observed absolute utilities `U_mdk` and frozen family-specific predictors
  `Uhat_md(x_mdk)` within every accepted reference model, with no observed
  candidate score, margin-adjusted score, or post hoc "penalty" used as an
  input;
- the explicit two-by-two map: neither spatial nor temporal benefit predicts
  D0, temporal only predicts D1, spatial only predicts D2, and both predict D3;
  and
- held-out conditions labelled by predicted family before all four observed
  family scores are revealed.

The four representatives are factorially constrained: D0 is published-like,
D1 changes the time-focused scan/software recipe only, D2 changes geometry only,
and D3 combines the D1 and D2 changes. C4 remains a required simple-control set
outside the predicted four-family endpoint.

**Decision:** fit one shallow monotone utility for each D0–D3 family on
development conditions crossed with every accepted reference model. Define
**S** as short spatial scale plus pad-noise and pad-impedance costs below frozen
thresholds; each cost is the worst of the D2/D0 and D3/D1 ratios. Define **T**
as short temporal scale plus schedule-settling and schedule-crosstalk costs
below frozen thresholds; each cost is the worst of the D1−D0 and D3−D2
increases. A spatial cost failure may switch off only **S**; a temporal cost
failure may switch off only **T**. The family mapped by S/T must lie within
0.005 of the frozen predicted-utility maximum or the separable rule fails.

Freeze the utilities and six thresholds and require them to predict held-out
conditions. A threshold interval yields its adjacent-family prediction set;
an abstention on an observed non-tie counts as an error. The simultaneous 95%
upper bound on equal-weight mean regret—best observed arm R² minus the predicted
arm R²—must be at most 0.005 in every accepted model and every S/T cell, with
at least five pre-outcome conditions in each cell. For a threshold-tie
prediction set, regret uses its worst member. Confusion, balanced accuracy,
gain-sign accuracy, and abstention are descriptive because winner-class counts
cannot be guaranteed prospectively. Accepted models are scored separately,
never averaged; fix these gates before independent outcomes.

**Competing explanation:** the apparent winner exploits one simulator's
unidentified scaling law. That explanation wins if equally well-fitting
reference models reverse rankings or the frozen condition rule fails.

The geometry, schedule, software, uncertainty, reversal, and random-control
ablations remain in this figure as mechanism checks. Every secondary and
robustness requirement still has to pass.

Show conditional geometry gains `U_D2-U_D0` and `U_D3-U_D1`, and bundled
time-focused scan/software gains `U_D1-U_D0` and `U_D3-U_D2`. Their factorial
interaction remains descriptive. D2 is not generally D3's matched C2: matched
C2 retains the joint candidate's geometry and scan and changes reconstruction
only. D1 comes from the published-setting C1 frontier, not a silently expanded
set of novel schedules.

Matched direction/phase conditions can retain the same radial spectral scales
and device probes while interacting differently with a rolling scan. Test
those conditions through the existing challenges and unchanged regret rule;
do not add direction or phase as predictor inputs after seeing their outcomes.
Large interaction alone does not invalidate a predictive rule, while an average
device win cannot rescue a rule that fails its registered condition test.

**If the primary gain passes but a robustness check fails:** report a fragile
development result and do not advance it to fabrication.

## Figure 4 — Does the gain survive independent models?

**Scientific judgment:** test the frozen D0, D1, D2, and D3 family
representatives with separately implemented cortical-field, source-to-surface,
and electronics models rather than merely new random draws from the development
code.

**Panels:**

- development versus independent model lineage;
- at least three hidden signal mechanisms, four device shifts per mechanism,
  64 paired fields per condition, and at least five pre-outcome conditions in
  every S/T cell under every accepted model;
- one representative or deterministic selection rule frozen for each D0–D3
  family across every accepted reference model;
- all four representatives and required C4 controls repeated on the same
  independent worlds under every accepted reference model;
- predicted versus observed winner sets, regret in every accepted-model and
  S/T cell, a four-class confusion matrix, abstention, balanced accuracy, and
  the sign of gain over D0; and
- the separate catastrophic-plausibility physical-signal or phantom check.

**Decision:** record the frozen rule's family and gain-sign prediction before
reconstruction. The observed winner set contains every family within 0.005 of
the best R²; intervals that cannot separate leaders are unresolved. The
selected joint C3/D3 candidate must still pass every primary, secondary,
resource, and robustness requirement. No runner-up, changed family
representative, or changed rule may replace a failure.

**If accepted reference models produce different unique winners, with each
simultaneous one-sided lower bound more than 0.005 above its runner-up:**
optimization was not identified by the published measurements. Stop design
nomination and obtain the noise, impedance, settling, or crosstalk measurement
that best separates the models.

**If development passes but this figure fails:** conclude that the apparent
gain did not generalize beyond the development models.

**If the device passes but the rule fails:** retain a descriptive model-specific
comparison only. Do not nominate or advance a specification under the current
promotion rule, which already requires the frozen condition-to-design test.
No general acquisition principle is established.

**If both pass:** advance the one virtual specification to fabrication or
device testing and state the measurable domain in which it is predicted to
help; do not claim physical superiority.

## Result-dependent next steps

| Result | Next scientific step |
| --- | --- |
| Virtual co-design passes twice and every promotion requirement, including the frozen rule, passes | Propose fabrication or bench testing of exactly that design against a matched physical reference |
| Frozen design rule also predicts independent winners | Test the predicted spatial, temporal, and absolute device-response boundary directly on a bench |
| Accepted reference models reverse rankings | Measure the unresolved pad-noise, impedance, settling, or crosstalk response before any candidate search |
| Candidate wins but the frozen rule fails | Report a descriptive model-specific comparison; no design promotion under the current rule |
| Software is sufficient | Improve reconstruction on existing hardware before redesigning the array |
| Hardware alone is sufficient | Test the simpler hardware change with conventional reconstruction |
| Simple rule is sufficient | Prefer the simple rule and test its manufacturing tolerance |
| Gain is condition-specific | Expand or narrow the intended operating domain before fabrication |
| Independent-model failure | Investigate which field or electronics assumption drove the development gain; do not select a replacement from final results |
| Reference model fails held-aside checks | Obtain raw traces or additional device measurements before comparing candidates |
| All legal designs violate resources | Reconsider the fixed hardware constraints in a new study rather than weakening them after results |
| Evidence remains uncertain | Spend new simulations or measurements on the uncertainty bottleneck, not on choosing from point estimates |

## Main claim boundary

The strongest EP20 paper can recommend one **virtual** design for fabrication
testing and state a frozen, independently tested D0–D3 rule linking spatial
scale and event timescale to absolute, independently characterized noise,
impedance, settling, and crosstalk. If plausible reference models reverse
rankings or the four-family prediction fails, the transferable claim is not
supported. No result here can claim that the unbuilt device is physically
superior, manufacturable, chronically safe, reliable, biocompatible, or
effective in vivo.
