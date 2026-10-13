# Research memory

2026-10-02 — [Scope/novelty review (consolidated)](paper_plan.md): Aligned narrative promotion with the policy: a rule-failing candidate remains descriptive, not promotable; replaced time-only shorthand with the bundled scan/software factor. The contribution remains virtual design identifiability and conditional prediction, not fabricated-device superiority.

## Reusable lessons from the design

- A co-design claim requires comparisons with optimized software, hardware
  alone, and simple resource-matched controls—not only old hardware with old
  software.
- Every arm must see the same cortical fields, device conditions, masks, and
  resource budget.
- Unknown power, thermal, routing, or fabrication cost is not zero.
- A different random seed in the same simulator is not an independent model
  test.
- A successful virtual result selects a design for fabrication testing; only a
  fabricated comparison can establish physical superiority.

## Open questions

- Can aggregate published measurements constrain a useful reference model?
- Are the selected virtual pad dimensions compatible with actual via and
  routing clearances?
- Can all five arms be matched for conversions, latency, bitrate, modeled
  power, and training allowance?
- What physical-signal or phantom source can support separate development and
  final plausibility checks?

These are design questions, not scientific results.

## Component-design lessons — 2026-09-30

- lambda/tau are spectral 90%-power scales, not autocorrelation decay scales.
- Pad sensitivity, contact gaps and reference each need one forward-chain owner.
- Rolling scans couple field direction to timing even at matched marginal scales.
- D2 is geometry-only; D3's matched C2 keeps D3's scan as well as geometry.
- A descriptive interaction does not replace the predictive regret criterion.
- `component_details.md` keeps implementation choices open and states the
  still-needed field/device assumptions rather than inventing numeric values.

## Starting catalogue — 2026-09-30

`parameter_candidate_catalogue.md` selects half-/quarter-area subfootprints and
finite field/scan contrasts while keeping legal follow-ups open. Correlation
distance is not lambda90, Gaussian width is not event duration, and correlated
mixture power includes cross terms. Equal source-stream work is not verified
internal ADC work. Coverage removed by a scan stays in the scoring mask.

## Electrical mechanisms — 2026-09-30

The four probe outputs must come from one coupled circuit. Passive thermal
noise uses Re[Z], not |Z|; active TFT noise and shared reference terms need their
own owners. A static cutoff does not identify switched settling. Fix a common
victim panel before outcomes, include off-gate aggressors, and distinguish
same-victim leakage from the paper's still-unverified crosstalk normalization.
Unknown gains/tails are bounds or undefined inputs, not epsilon-repaired zeros.

## Compact source/inverse choices — 2026-09-30

Use a small source-sensitivity panel, not a Cartesian grid. Causal history
limits recurrent state too; two initial history bundles mean two tuning
configurations, with the short-history control reserved for matched follow-up.

## Anchor roles — 2026-09-30

Holding public summaries aside from fitting is not blind validation. A
source-count noise law is not identified by adding a constant for the held mode.

## Shared fit and inverse — 2026-09-30

Equal nominal row-dwell rates do not isolate a switching-noise coefficient.
RMS-summary centres cannot be squared and subtracted as component variances.
Inverse history limits posterior/recurrent state, not physical circuit memory;
delayed reporting does not make post-target acquisitions past-only inputs.
