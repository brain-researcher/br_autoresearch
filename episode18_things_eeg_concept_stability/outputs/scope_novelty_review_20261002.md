# EP18 scope, novelty and clarity review — 2026-10-02

Question: does sharing EEG responses by named concept predict unseen exemplars better than an equally flexible continuous-label basis and matched false groupings, under the same image model and RSVP overlap model?

## Closest work and the remaining contribution

[Holm et al. (2024)](https://doi.org/10.1016/j.neuroimage.2024.120626) makes residual visual structure a direct alternative to semantic interpretations. [Kim et al. (2026)](https://doi.org/10.1167/jov.26.8.2) already relates perceptual and conceptual structure to object EEG. [Rong et al.](https://elifesciences.org/reviewed-preprints/108915v1) compares vision/language representations in THINGS-EEG2, while [Watson et al. (2026)](https://doi.org/10.1016/j.neuroimage.2026.122012) motivates continuous dimensions as a serious alternative. Cross-exemplar or semantic-model decoding alone cannot carry this paper.

The potential contribution is the conjunction of continuous overlapping-response prediction, capacity-matched label readout, eight visually/temporally matched false partitions and frozen participant-level repetition, including the distant-exemplar prediction.

## Decision and repair

Keep the broad question and its strong alternatives. The opening no longer contrasts concept with pixels as if those sources were separable. Named labels may already identify concepts in the continuous inputs: the comparison concerns sharing and regularization structure, not newly supplied nominal information.

The distant-event post-fit ablation measures predictive importance within a particular fitted decomposition. Correlated bases can redistribute fitted contributions, so the contrast is neither unique variance nor a causal concept contribution. It remains useful as the registered prediction, without adding a new test.

All claims are relative to the selected feature bases, fixed concept inventory and meaningful margins. Neither a win nor equivalence proves abstract semantic coding, feature completeness, or transfer to unseen concepts. A standalone paper remains conditional on the complete comparison being consequential, not merely positive.

No control, threshold, participant role, search space or outcome-access rule changes; this was a documentation-only review.
