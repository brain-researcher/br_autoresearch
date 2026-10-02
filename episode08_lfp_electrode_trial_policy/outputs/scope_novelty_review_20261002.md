# EP08 scope, novelty and clarity review — 2026-10-02

Design/literature review only; no new data access, analysis or empirical claim.
The source-availability statements below come from existing episode documents,
not a new filesystem or scheduler survey.

## Question and contribution

Which properties of available training/calibration information predict an
electrode's conditional population value and a direction's next-trial value?
The stronger candidate is sensor-dependent learnability: changing retained
electrodes changes which future observation is useful. Models, information
states, summaries and initializers remain open; there is no compulsory pilot.

This is an empirical explanation in LFP-to-population prediction, not a new
theory of conditional selection, active learning or uncertainty reduction.
A policy win alone is insufficient. Prospective prediction of conditional
removal loss, repeatable direction benefit and sensor-dependent direction value
would be the substantive evidence, with matched decoder/tuning opportunity.

## Closest checked primary literature

- [Cohn, Ghahramani and Jordan 1996](https://arxiv.org/abs/cs/9603104):
  uncertainty-reducing active learning is established.
- [Krause, Singh and Guestrin 2008](https://jmlr.org/papers/v9/krause08a.html):
  sensor value depends on the selected set; redundancy-aware selection is known.
- [Flint et al. 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3429374/):
  LFP decoding and selection by marginal kinematic information already motivate
  using electrode subsets. Whole-electrode selection is not itself new.
- [Gallego-Carracedo et al. 2022](https://elifesciences.org/articles/73155):
  the source already establishes structured LFP–population relationships.
- [Li et al. 2025](https://pubmed.ncbi.nlm.nih.gov/40262392/):
  uncertainty-based target-domain decoder calibration is already demonstrated;
  adaptive calibration alone is not the novelty.

## Repair and remaining empirical burden

The writing now distinguishes **sensor–trial coupling** from **factorial score
interaction**. Direction-specific changes can cancel after aggregation;
algorithmic non-additivity need not support the proposed biological/statistical
explanation. The optional reference's numerical decision remains unchanged.

Test future value using only information allowed before the choice. Offline
value studies need not claim sequential acquisition efficiency; sequential
claims count all acquired exposure. Existing documents record an absent source
pack and no newly sequestered third-animal source. Within-source evidence and
external-animal transfer remain different claims. No paper-level discovery,
hardware saving or online benefit is established by this review.

The next scientific work is supported development exploration when its inputs
and execution are authorized—not another mandatory pilot or review gate.
