# EP20 paper plan

## Paper question

Under which measurable spatial, temporal, and device conditions can a legal
change to NeuroCam pad geometry and scan allocation, paired with
measurement-aware reconstruction, improve cortical-surface voltage recovery
without receiving extra wiring, conversion, latency, bitrate, power, thermal,
training, or model-capacity resources?

The deeper question is whether measurable conditions tell us **which** design
should win: spatial correlation length, temporal timescale, and absolute
noise, impedance, settling, and crosstalk characterization. A paper should
deliver a rule that predicts the preferred design family under new conditions,
not only one promising virtual device.

The paper should begin with the spatial-versus-temporal measurement trade-off
and the five matched comparisons. It should not begin with optimization
machinery or imply that a virtual design has already been fabricated.

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
the published-like, time-only, geometry-only, or joint family will recover the
field best, rather than merely describing an average winner.

**Panels:**

- the two field inputs measured from truth or a separate known-input pilot
  before reconstruction: `1/(2*k90)` in micrometres and `1/(2*f90)` in seconds,
  where the 90% cutoffs come from common-mask spatial and temporal field power;
- one absolute characterization table for each frozen D0–D3 representative:
  input-referred 1–100 Hz RMS noise under a zero-input probe, the 90th
  percentile of impedance magnitude across 1–100 Hz, settling time after a unit
  step to 1% error, and maximum adjacent-line crosstalk amplitude ratio under a
  1–100 Hz multisine scan, repeated for every device condition;
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

**If the device passes but the rule fails:** advance, at most, one model-specific
virtual specification; do not claim a general acquisition principle.

**If both pass:** advance the one virtual specification to fabrication or
device testing and state the measurable domain in which it is predicted to
help; do not claim physical superiority.

## Result-dependent next steps

| Result | Next scientific step |
| --- | --- |
| Virtual co-design passes twice | Fabricate or bench-test exactly that design against a matched physical reference |
| Frozen design rule also predicts independent winners | Test the predicted spatial, temporal, and absolute device-response boundary directly on a bench |
| Accepted reference models reverse rankings | Measure the unresolved pad-noise, impedance, settling, or crosstalk response before any candidate search |
| Candidate wins but the frozen rule fails | Treat it as a model-specific virtual device, not a transferable acquisition principle |
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
