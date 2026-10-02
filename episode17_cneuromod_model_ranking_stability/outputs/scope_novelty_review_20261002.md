# EP17 — scope, novelty and clarity review

## Decision

Keep the three-operation question as a methods/measurement study. Broaden the
title beyond self-supervision, keep named checkpoints as starting contrasts,
and align eligibility with the actual intervention: changing how the same
responses/predictions are measured or aggregated. This is a pre-score design
amendment, not a completed comparison. A standalone paper remains conditional
on an informative empirical result; global novelty is not established.

## Closest evidence checked

- [Conwell et al. 2024](https://www.nature.com/articles/s41467-024-53147-y):
  controlled comparisons already examine visual-model properties and linking
  methods. Another SSL/supervised leaderboard or general warning that methods
  matter is insufficient.
- [Prince et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9708069/):
  GLMsingle already establishes improvements in single-trial response
  reliability and downstream uses. EP17 tests a relation between fixed models,
  not the invention of better beta estimation.
- [Official DINO records](https://github.com/facebookresearch/dino)
  and [official SwAV records](https://github.com/facebookresearch/swav)
  identify distinct training recipes. Common backbone and nominal image corpus
  do not equate their augmentation, optimization or duration.
- [Methods for computing the maximum performance of computational models of fMRI responses](https://pmc.ncbi.nlm.nih.gov/articles/PMC6426260/)
  already treats noise ceilings as part of computational-model evaluation.
  Positive-weight reaggregation is not a newly discovered biological mechanism.

## What the present experiment can add

A repeatable map of which sampled cortical population carries each fixed
representation's advantage, with response estimation, selection and weighting
separated under matched neural fitting. Whole-concept holdout tests whether
the comparison survives new stimuli in the same people.

For fixed weights, `sum((w_R-w_A)*d_dev)` is exactly the development support
edge. Carrying that number to audit is empirical repeatability, not an
independent explanatory prediction. A successful audit cannot establish why
the spatial advantage arose, that reliable voxels are the uniquely correct
population, or that self-supervision caused a better neural representation.

## Open directions, not new execution requirements

The beta OLS has a related limitation: spatial-bin intercepts force
`sum_bin(y-y_hat)=0`, so its mean fitted prediction equals the mean fitted
development target on the same support. This is the mean of the training
blockwise changes, not automatically pooled-image R². Aggregate agreement
cannot demonstrate a gain-map increment over bin intercepts. The registered
high/low test describes a reliability association; a direct held-out spatial
comparison to bin-only predictions would be needed for the stronger claim.

If the initial measurements show consequential heterogeneity, compare whether
it follows anatomical position, repeat reliability, or externally defined
stimulus/functional organization. Useful competitors include spatial-bin-only
advantage maps and within-bin rearrangements that preserve broad anatomy while
breaking local correspondence. An explanatory model should predict a held-out
pattern beyond the corresponding coarse comparator, not simply rediscover a
weighted mean. These are prospective design options, not extra audit criteria
or outcome-driven opportunities to rescue the current endpoint.

The existing broad development search remains available. There is no reason
to make every result a sign reversal or every future hypothesis about SSL;
continuous shifts and stable relations are scientifically reportable even
when they do not satisfy a candidate-ready terminal.

## Changes and boundary

Replaced the single-training-property admission rule with claim-specific
fixed-pair measurement eligibility. Preserved three required pairs, five edges,
readout matching, exposure exclusions, 480/120/120 roles, four participants,
budgets, all numerical margins, and one-shot audit. The ASTRA registry
description is synchronized; no new analysis, data access or execution is added.

BR hypothesis verification returned a Lucene nested-clause error rather than
evidence. Primary sources and mathematical inspection supported the revision;
no BR scientific verdict is claimed. The retained v2 image is a historical
named-pair illustration and its SSL wording is qualified in the main caption.

An independent read-only review verified fixed-pair eligibility and support
algebra, and identified the beta-intercept identity above. GOAL, policy and
paper plan now state it explicitly; no new forecast gate or experiment was added.

A second reviewer identified a remaining score-target ambiguity. Beta's
observed forecast target is now explicitly the mean of twelve audit-block
changes, matching the blockwise fitting/cross-fit target. The primary pooled-
image beta edge and high/low endpoint remain separate and unchanged; support
and ceiling use their aggregate edge. This is a pre-score endpoint-definition
clarification, not permission to choose whichever aggregation succeeds.
