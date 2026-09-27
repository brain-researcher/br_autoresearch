# EP20 paper plan

## Paper question

Can one legal change to NeuroCam pad geometry and scan allocation, paired with
measurement-aware reconstruction, improve cortical-surface voltage recovery
without receiving extra wiring, conversion, latency, bitrate, power, thermal,
training, or model-capacity resources?

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
- the five arms in plain language, with D0–D4 shown only as shorthand;
- identical cortical fields, device conditions, masks, and resource budgets
  feeding every arm; and
- the design scope: legal pad geometry, legal scan allocation, and
  measurement-aware reconstruction.

**Decision:** the candidate co-design must be compared with the full
published-hardware conventional and optimized frontier, the same candidate
hardware with conventional software, and every registered simple control.

**If resource parity cannot be established:** the study cannot make a co-design
claim, even if reconstruction scores differ.

## Figure 2 — Does the co-design beat the full frontier?

**Scientific judgment:** one candidate improves 1–100 Hz cortical-surface field
R² against every eligible published-hardware setting and every matched control
on paired worlds.

**Panels:**

- member-by-member D3-minus-D0/D1 paired gains;
- D3 versus matched D1, D2, and all D4 controls;
- the least favourable gain across signal and device conditions; and
- resource use before and after fabrication rounding.

**Decision:** lower bounds must reach 0.010 against every D0/D1 setting and
0.005 against D1, D2, and each D4 control. Every condition-specific comparison
must remain above −0.005.

**If D1 matches D3:** optimized software is sufficient under the registered
comparison.

**If D2 matches D3:** the hardware change may help, but special co-designed
software is not supported.

**If D4 matches D3:** the complicated search did not beat a simple rule.

## Figure 3 — Why does it help, and where does it fail?

**Scientific judgment:** distinguish geometry, scan allocation, software, and
their complementarity while testing whether the gain survives realistic
variation.

**Panels:**

- geometry-only, schedule-only, software-only, and uncertainty-model
  ablations;
- large/small pad reversal and random geometry controls;
- focal, broad, travelling-wave, silent-region, and multiscale field families;
  and
- localization, propagation, transient score, uncertainty, and downstream
  outcomes.

**Decision:** every secondary noninferiority margin, leave-one-field-family and
leave-one-device-condition check, fabrication rounding check, and development
physical-signal or phantom stress must pass.

**If the primary gain passes but a robustness check fails:** report a fragile
development result and do not advance it to fabrication.

## Figure 4 — Does the gain survive independent models?

**Scientific judgment:** test exactly one selected design with separately
implemented cortical-field, source-to-surface, and electronics models rather
than merely new random draws from the development code.

**Panels:**

- development versus independent model lineage;
- at least three hidden signal mechanisms, four device shifts per mechanism,
  and 64 paired fields per condition;
- the full D0–D4 comparison repeated on the same independent worlds; and
- the separate catastrophic-plausibility physical-signal or phantom check.

**Decision:** the selected D3 design must again pass every primary, secondary,
resource, and robustness requirement. No runner-up or changed method may
replace it.

**If development passes but this figure fails:** conclude that the apparent
gain did not generalize beyond the development models.

**If it passes:** advance the one virtual specification to fabrication or
device testing; do not claim physical superiority.

## Result-dependent next steps

| Result | Next scientific step |
| --- | --- |
| Virtual co-design passes twice | Fabricate or bench-test exactly that design against a matched physical reference |
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
testing under a paper-derived NeuroCam model and independently implemented
virtual challenge. It cannot claim that the unbuilt device is physically
superior, manufacturable, chronically safe, reliable, biocompatible, or
effective in vivo.
