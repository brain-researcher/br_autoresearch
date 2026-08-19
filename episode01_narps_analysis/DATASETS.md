# NARPS Analysis-Choice Variance Attribution v1

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
must not be changed after neural outcome access begins.

## Primary Analysis

The primary endpoint is analysis-choice sensitivity of continuous,
unthresholded group effect maps within one frozen common-brain mask.

Before reading any neural outcome map, write the following to:

- `outputs/frozen_analysis_contract.json`
- `outputs/frozen_analysis_contract.md`

The frozen contract must define:

- The exact primary estimand
- Common analysis mask
- Effect-map normalization
- Variance-decomposition method
- Included main effects and interactions
- Uncertainty-estimation procedure
- Missing-cell policy
- Stopping rules
- Primary and secondary visualization targets

After these files are written, the primary contract must not be modified in
response to observed results.

BH-FDR at `q <= 0.05` may be reported as a secondary conclusion-stability
readout. Threshold must not be treated as an additional multiverse factor in
this episode.

## Required Controls

Run a cohort-stratified matched-size random-deletion calibration:

- `Q0 → Q1`: randomly remove 0 EI and 2 ER participants
- `Q1 → Q2`: randomly remove 0 EI and 3 ER participants
- Replicates: 1,000
- Random seed: 20260814
- Reuse the same sampled subject sets across all confound and smoothing cells

Additional requirements:

- Treat `std_dvars` as a diagnostic only
- Do not censor individual volumes
- Do not exclude a participant solely because nuisance regressors are not full
  rank when the gain and loss contrasts remain estimable
- Do not treat overlapping multiverse cells or bootstrap samples as independent
  observations
- Interpret QC-set differences as analysis-set or sample-composition
  sensitivity, not as a causal effect of QC
- Run at least one reasonable exploratory diagnostic after the primary result
- Label all post-hoc findings as exploratory

## Reproducibility Requirements

The final analysis must preserve:

- Exact subject membership for every QC set
- Exact confound columns
- Exact smoothing parameters
- Exact random seeds
- A manifest of all completed and failed grid cells
- Executable analysis code
- Software and runtime information
- Failed attempts and deviations from the frozen plan
- Enough information for an independent rerun

Do not modify the source dataset or overwrite an existing result.

## Terminal Outcomes

### `candidate_ready`

Use this outcome only when:

- The frozen analysis grid has been completed
- Matched-size calibration has been completed
- The primary result survived the required methodological checks
- A reproducible and scientifically defensible candidate finding exists
- A specific prediction, falsifier, primary confirmation test, and required
  controls can be stated

### `closed_no_candidate`

Use this outcome when the analysis completed successfully but no sufficiently
stable, informative, or reproducible candidate finding was identified.

A valid null or inconclusive result is preferable to relaxing the frozen rules.

### `technical_failure`

Use this outcome when data availability, runtime limitations, model
identifiability, or another technical blocker prevents completion of the frozen
analysis.

Do not change QC thresholds, omit failed cells, or substitute a different
scientific question to avoid a technical failure.

## Claim Boundary

Every result from this workspace remains exploratory.

The terminal candidate bundle must state:

```text
scientific_status: exploratory_only
confirmation_eligible: false
scientific_acceptance: false
confirmation_requirement: independent_fresh_confirmation_required
