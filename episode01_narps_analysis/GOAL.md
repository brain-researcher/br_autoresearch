RPS Analysis-Choice Variance Attribution v1

## Research Question

Using the public NARPS ds001734 dataset, determine how sensitive gain- and
loss-related task-fMRI results are to:

1. Confound handling
2. Outcome-blind subject exclusion
3. Spatial smoothing

Determine which analysis choices contribute most strongly to variation in
continuous effect-map geometry and statistical conclusion stability.

## Scientific Scope

This is an exploratory methods study. It is not an independent confirmation
study and must not be presented as establishing a causal neural mechanism for
gain or loss processing.

Analyze `gain_demean` and `loss_demean` separately.

The equalIndifference and equalRange task versions have different gain
supports. Therefore, a pooled EI/ER gain-minus-loss comparison must not be used
as the primary analysis. Any EI-versus-ER difference must be labeled as a
task-version-associated exploratory result.

## Frozen Analysis Grid

Use the analysis grid defined in `inputs/frozen_grid.json`:

- Three confound models
- Three QC subject sets
- Three smoothing levels
- 27 total `confound × QC × smoothing` cells

QC thresholds, subject membership, confound definitions, and smoothing levels
must not
