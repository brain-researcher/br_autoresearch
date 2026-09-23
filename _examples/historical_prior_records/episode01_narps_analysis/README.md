# Episode 01 retrospective example guide

> **Status: historical exploratory episode, closed without a candidate.**
>
> This guide was added retrospectively on **2026-09-13** to make the episode
> easier to understand and reuse as an example. It was not present when the
> analysis was frozen, run, or submitted. It is not part of the frozen analysis
> contract, the terminal candidate bundle, or canonical Brain Researcher state.
> Existing historical files remain the evidence; the MCP service remains the
> authority for canonical state.

## Status at a glance

| Item | Status |
|---|---|
| Scientific mode | Exploratory methods study |
| Dataset | Public NARPS OpenNeuro `ds001734` |
| Analysis completion | Complete: 54/54 contrast-by-grid cells |
| Terminal outcome | `closed_no_candidate` |
| Scientific status | `exploratory_only` |
| Independent confirmation | Still required |
| Canonical outer loop | `COMPLETE@3`, completion kind `native_goal_terminal` |
| Society review | Not eligible and not called |
| Reward or launch approval | Not opened |
| Server confirmation execution | None |
| Scientific acceptance | None |
| Landscape transition | None |

The primary scientific result is in [outputs/RESULT.md](outputs/RESULT.md).
The frozen, pre-outcome specification is in
[outputs/frozen_analysis_contract.md](outputs/frozen_analysis_contract.md).
The final observed control-plane status is summarized in
[outputs/governance.md](outputs/governance.md).

## What this example teaches

This episode is useful because it shows a complete exploratory analysis whose
correct endpoint was a negative terminal rather than a promoted claim. In
particular, it demonstrates how to:

1. freeze a multiverse and its decision rule before neural-outcome access;
2. define a group estimand that remains stable under unequal cohort sizes;
3. separate continuous-map sensitivity from thresholded conclusion stability;
4. calibrate subject-set sensitivity against matched random deletion;
5. repair engineering failures without changing the scientific contract;
6. preserve failed attempts, diagnostics, and uncertainty rather than tuning
   the analysis after seeing results; and
7. distinguish local scientific work from canonical lifecycle bookkeeping.

It also illustrates a scientifically healthy stopping decision: a complete,
reproducible analysis need not yield a candidate.

## Read this episode in this order

For a fast but accurate review, use the following sequence:

1. This retrospective guide for orientation.
2. [outputs/frozen_analysis_contract.md](outputs/frozen_analysis_contract.md)
   for the binding scientific design.
3. [outputs/RESULT.md](outputs/RESULT.md) for the human-readable result.
4. [outputs/tests/math_review.md](outputs/tests/math_review.md) for the
   independent synthetic mathematics audit.
5. [outputs/experiment_log.md](outputs/experiment_log.md) for execution and
   handoff chronology.
6. [outputs/governance.md](outputs/governance.md) for observed canonical state.
7. [outputs/memory.md](outputs/memory.md) for reusable lessons and approaches
   that should not be carried forward.

The historical [GOAL.md](GOAL.md) is truncated at its final line and refers to
an absent `inputs/frozen_grid.json`. The historical
[DATASETS.md](DATASETS.md) contains much of the intended protocol but is not a
conventional source manifest. Do not infer missing requirements from those two
files; use the frozen contract for the implemented analysis.

## Research logic

### Question

How sensitive are gain- and loss-related task-fMRI group maps to three common
analysis choices?

- confound handling (`C`);
- outcome-blind subject inclusion (`Q`); and
- spatial smoothing (`S`).

The broader methods question was which of these choices contributes most to
variation in continuous effect-map geometry and to the stability of
thresholded conclusions.

### Scope and nonclaims

The two contrasts, `gain_demean` and `loss_demean`, were analyzed separately.
The equalIndifference (EI) and equalRange (ER) task versions have different
gain supports, so a pooled EI/ER gain-minus-loss contrast was not a primary
analysis. Any EI-versus-ER coefficient is task-version-associated and
exploratory.

This episode does **not** establish:

- a causal neural mechanism for gain or loss processing;
- a causal effect of quality-control exclusion;
- an independently confirmed result;
- scientific acceptance by Brain Researcher; or
- a reason to change the frozen analysis after observing the maps.

### Unit of analysis and primary map

Each technically eligible participant contributed a four-run,
inverse-variance fixed-effect map. For every multiverse cell, the group model
was ordinary least squares with an intercept and a task-version code of `+0.5`
for EI and `-0.5` for ER. The primary intercept is therefore

```text
(mean_EI + mean_ER) / 2
```

rather than the unstratified participant mean when cohort sizes differ.
Subject-level precision was not used as a group weight.

Primary maps were continuous and unthresholded. They were not spatially
centered, z-scored, winsorized, thresholded, or L2-normalized. Mask voxels had
equal weight. BH-FDR at `q <= 0.05` was a secondary conclusion-stability
readout, not another multiverse factor.

## Data and cohorts

The frozen contract records the following read-only sources:

| Source | Historical path |
|---|---|
| Raw participants and events | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/input/ds001734` |
| fMRIPrep derivatives | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/fmriprep/ds001734/derivatives` |
| FitLins run-level designs | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/analyses/ds001734/task-MGT/node-runLevel` |

The required imaging grid was MNI152NLin2009cAsym at 2 mm. The inventory was
108 participants—54 EI and 54 ER—with four usable runs each, or 432 runs.
Technical eligibility required both frozen contrasts to remain estimable in
all four runs under every confound model; nuisance rank deficiency by itself
was not an exclusion criterion.

The outcome-blind QC sets were nested:

| Set | Frozen rule | EI | ER | Total | Removed relative to Q0 |
|---|---|---:|---:|---:|---|
| Q0 | All technically eligible participants | 54 | 54 | 108 | None |
| Q1 | Weighted mean FD <= 0.30 mm and every run's proportion of FD > 0.5 mm <= 0.20 | 54 | 52 | 106 | `sub-030`, `sub-100` |
| Q2 | Q1 plus weighted mean FD <= 0.20 mm and every run's proportion of FD > 0.5 mm <= 0.10 | 54 | 49 | 103 | Q1 removals plus `sub-016`, `sub-018`, `sub-088` |

All five Q2 exclusions were ER participants. Consequently, Q sensitivity is
analysis-set or sample-composition sensitivity, not a causal effect of QC.
`std_dvars` was diagnostic only, and no individual volumes were censored.

The episode's local [inputs/](inputs/) directory is empty. A rerun therefore
depends on access to the historical external paths above and cannot be
reconstructed from this directory alone. A new episode should instead record
immutable source identifiers, versions, hashes, licensing, acquisition steps,
and a verified local layout in its dataset contract.

## Exact multiverse

The frozen grid was `3 confounds × 3 QC sets × 3 smoothing levels = 27` cells
per contrast. With two contrasts, the required total was **54**
contrast-by-grid cells.

### Confound axis

| Level | Definition |
|---|---|
| C0 | Six native motion parameters, native run-specific cosine/DCT columns, unchanged task regressors, and intercept |
| C1 | C0 task/DCT/intercept terms with true Friston24: six originals, six first derivatives, six squared originals, and six squared derivatives |
| C2 | C1 plus `a_comp_cor_00` through `a_comp_cor_05`, each metadata-verified as combined-mask and retained |

The first row of each motion derivative was set to zero, and squares were
formed after originals and derivatives. Nuisance regressors stayed in native
units. Missing, nonfinite, non-retained, or incorrectly masked required
aCompCor components were technical-failure conditions.

### QC and smoothing axes

The Q axis was Q0/Q1/Q2 as defined above. The smoothing axis was:

| Level | Gaussian smoothing |
|---|---:|
| S0 | 0 mm FWHM |
| S1 | 5 mm FWHM |
| S2 | 8 mm FWHM |

Smoothing was applied to 4D BOLD data before the first-level model. Estimation
used a Nilearn voxelwise AR(1) GLM with `drift_model=None`,
`standardize=False`, and `signal_scaling=0`.

No cell could be imputed, omitted, replaced, or silently simplified. A missing
cell required `technical_failure` rather than a smaller post-hoc multiverse.

### Common mask

Mask construction was outcome-blind. A voxel had to be covered by at least
95% of all 432 run masks and at least 95% of the 216 run masks within each task
version. The same mask was used for every cell and both contrasts. The final
[common analysis mask](outputs/common_analysis_mask.nii.gz) contained 221,116
voxels.

## Attribution and decision rules

For each contrast, the 27 continuous group maps were decomposed with a
balanced, sum-to-zero functional ANOVA into seven orthogonal terms:

```text
C, Q, S, C×Q, C×S, Q×S, C×Q×S
```

Interactions were allocated equally among their participating factors using
the frozen Shapley rule. This produced attributed C, Q, and S sums of squares,
shares of total across-cell map variation, and the ratios `C/S`, `Q/S`, and
`(C+Q)/S`.

Uncertainty used a paired Bayesian bootstrap over Q0 participants. The same
raw participant weight draw was reused across every C/Q/S cell and both
contrasts, with within-cohort renormalization for each eligible Q set. The run
used 2,000 PCG64 replicates with seed `20260815`.

The frozen candidate rule required, for at least one contrast:

1. both `C/S` and `Q/S` point estimates to exceed 1; and
2. both corresponding 97.5% simultaneous interval lower bounds to exceed 1.

This complete rule was evaluated as written. Partial patterns were not enough.

The required matched-size deletion calibration used 1,000 PCG64 replicates
with seed `20260814` and reused sampled sets across all C×S cells and both
contrasts:

- Q0→Q1 randomly removed 0 EI and 2 ER participants;
- Q1→Q2 randomly removed 0 EI and 3 ER participants.

## Workflow and checkpoints

The workflow can be understood as six checkpoints. These are a reading and
verification map, not a promise that this historical workspace is portable.

| Checkpoint | Purpose | Principal evidence |
|---|---|---|
| 0. Freeze | Specify sources, mask, estimand, grid, uncertainty, decision rule, and stopping outcomes before outcome access | [Frozen contract](outputs/frozen_analysis_contract.md), [JSON contract](outputs/frozen_analysis_contract.json) |
| 1. Audit | Verify source inventory, run designs, estimability, QC membership, and common mask | [Source audit](outputs/source_audit.json), [run/design audit](outputs/run_design_qc_audit.tsv), [QC sets](outputs/qc_subject_sets.tsv), [mask](outputs/common_analysis_mask.nii.gz) |
| 2. Pilot | Exercise one bounded subject fit before the full array | [Pilot record](outputs/pilots/sub-001_pilot_1000vox.json) |
| 3. Subject fits | Produce all 108 immutable subject checkpoints through Slurm | [Subject maps](outputs/subject_maps/), [fit code](outputs/code/fit_subject.py), [array script](outputs/code/fit_subjects.sbatch), [logs](outputs/logs/) |
| 4. Aggregate | Require every checkpoint, build all 54 cells, attribute variance, and run matched deletion | [Aggregation provenance](outputs/aggregation_provenance.json), [multiverse manifest](outputs/multiverse_manifest.tsv), [aggregation code](outputs/code/aggregate_results.py) |
| 5. Diagnose and verify | Audit numerical edge behavior and frozen mathematics without redefining the analysis | [Edge audit](outputs/edge_scaling_diagnostic_summary.json), [math review](outputs/tests/math_review.md), [tests](outputs/tests/) |
| 6. Finalize | Render figures/result and create the exact negative terminal bundle | [Finalizer](outputs/code/finalize_candidate.py), [result](outputs/RESULT.md), [bundle](outputs/candidate_bundle.json) |

Before moving between checkpoints, verify counts and manifests rather than
assuming a scheduler exit code implies scientific completeness. The final
aggregation explicitly required all 108 valid subject checkpoints.

## Results

### Primary continuous-map attribution

| Contrast | C share (95% interval) | Q share (95% interval) | S share (95% interval) | C/S (97.5% simultaneous interval) | Q/S (97.5% simultaneous interval) | (C+Q)/S (95% interval) |
|---|---:|---:|---:|---:|---:|---:|
| `gain_demean` | 0.06782 [0.05683, 0.08654] | 0.03821 [0.003381, 0.1277] | 0.8940 [0.8062, 0.9314] | 0.07587 [0.06087, 0.1025] | 0.04274 [0.002440, 0.2070] | 0.1186 [0.07365, 0.2405] |
| `loss_demean` | 0.07731 [0.06218, 0.09560] | 0.02723 [0.002338, 0.09135] | 0.8955 [0.8341, 0.9287] | 0.08634 [0.06618, 0.1119] | 0.03041 [0.001731, 0.1354] | 0.1168 [0.07682, 0.1989] |

No ANOVA interaction had an observed sum-of-squares share of at least 0.05.
For this estimand, smoothing accounted for roughly 89% of across-cell map
variation in both contrasts. However, neither C/S nor Q/S approached the
frozen threshold of 1, so neither contrast met the complete candidate rule.

### Descriptive map and FDR stability

Median matched-pair summaries changed most strongly along the smoothing axis:

| Contrast | Axis changed | Pearson r | Cosine | NRMSD | FDR Jaccard |
|---|---|---:|---:|---:|---:|
| `gain_demean` | C | 0.9737 | 0.9665 | 0.09160 | 0.6427 |
| `gain_demean` | Q | 0.9829 | 0.9818 | 0.07596 | 0.7186 |
| `gain_demean` | S | 0.6070 | 0.6082 | 0.8197 | 0.002235 |
| `loss_demean` | C | 0.9530 | 0.9551 | 0.1233 | 0.6364 |
| `loss_demean` | Q | 0.9837 | 0.9814 | 0.06852 | 0.7369 |
| `loss_demean` | S | 0.5859 | 0.5941 | 0.8225 | 0.004190 |

All 27 cells had at least one BH-FDR rejection for each contrast. Rejection
counts ranged from 14 to 29,721 for gain and from 12 to 20,029 for loss. These
thresholded summaries were secondary and were not used to expand the grid.

### Matched-size calibration

For Q0→Q1, all 9/9 C×S cells in both contrasts had upper-tail empirical
`p <= 0.05` for both normalized RMS difference and `1 - Pearson r`. Median
observed/null-median ratios were:

| Contrast | NRMSD | 1 - Pearson r |
|---|---:|---:|
| `gain_demean` | 1.679 | 2.593 |
| `loss_demean` | 1.649 | 2.689 |

Q1→Q2 was weaker. The corresponding ratios were 1.142 and 1.349 for gain and
1.198 and 1.458 for loss; significant-cell counts were respectively 2/9 and
2/9 for gain, and 3/9 and 6/9 for loss. These are sample-composition
calibrations, not causal QC estimates.

### Exploratory influence and edge diagnostics

Across participants and C×S cells, median ER leave-one-out influence was
0.09076 for frozen-excluded versus 0.07740 for retained participants for gain,
and 0.1487 versus 0.1101 for loss.

The edge-scaling audit covered all 1,944 subject × C × S × contrast variance
arrays. It found 30,186 array-voxel entries below `1e-20`, 18,057 below
`1e-30`, and a global minimum of `6.89e-33`; every stored variance remained
finite and strictly positive. Diagnostic clamp counts were available for
94/108 participants because 14 older compatible sidecars predated that
logging. No post-hoc mask trimming or variance floor was introduced.

These diagnostics leave an important uncertainty: some apparent smoothing
sensitivity may reflect low-intensity boundary scaling behavior rather than
spatial regularization alone. That caveat strengthens the decision not to
promote a candidate.

## Failure and repair history

The scientific grid was completed, but execution was not failure-free:

1. An extra near-zero temporal-mean guard, stricter than the frozen mask
   contract, caused six array tasks to stop before creating checkpoints. The
   extra guard was removed, affected tasks were rerun, and the scientific
   contract was not changed.
2. A full-4D development diagnostic exceeded a 4 GiB interactive allocation.
   It was replaced by bounded checks and Slurm-scale retries rather than
   increasing login-node work.
3. Dependencies tied to failed array tasks left some pending aggregation jobs
   unsatisfiable. Final aggregation independently checked for all 108 valid
   checkpoints before proceeding.
4. During later terminal replay, `pytest` was unavailable in the declared
   environment. The two self-contained synthetic test entrypoints were run
   directly and both passed.

The detailed operational record is
[outputs/engineering_repairs.json](outputs/engineering_repairs.json), and the
evidence-completeness record is
[outputs/aggregation_provenance.json](outputs/aggregation_provenance.json).
No frozen scientific deviation was introduced by these repairs.

## Canonical lifecycle and nonacceptance boundary

Local analysis and canonical state should not be conflated:

1. The scientific contract was frozen at `2026-08-14T14:22:00Z`, before neural
   outcome access.
2. The complete client-native Sherlock analysis subsequently produced an
   exploratory `closed_no_candidate` bundle.
3. A fresh Brain Researcher handoff was prepared later to validate and replay
   that existing terminal evidence; this was not a new scientific run.
4. The first fresh submission encountered an
   `autoresearch_goal_submission_conflict` because the identical negative
   bundle was already owned by an earlier handoff.
5. Once production exposed `retry_terminal_reconciliation`, the original owner
   replayed the exact persisted bundle. The server accepted it idempotently.
6. At the last recorded observation (`2026-08-16T14:16:59Z`), outer loop
   `arl_8cbd25a7a8610a008c2830ba624052c4` was `COMPLETE@3` with terminal status
   `closed_no_candidate`. The fresh sibling handoff was non-owning and reported
   `closed_by_other_handoff`.

This reconciliation closed bookkeeping only. There was no Society outcome,
scientist reward, launch approval, server confirmation execution, canonical
confirmation episode, scientific acceptance, or Landscape transition. The
exact observed IDs and states are preserved in
[outputs/governance.md](outputs/governance.md); they must not be copied into a
new episode.

## Reproduction and verification notes

### Historical environment assumptions

The result records Python `3.12.1` loaded through the Sherlock module system
and the historical environment `/home/users/zijiao/pcvenv`. Those references
are site- and time-specific, not a portable lockfile. Before rerunning:

- confirm that all three external dataset paths still exist and are read-only;
- compare the source/run/QC audits with the frozen contract;
- inspect script help and Slurm resource requests rather than assuming old
  scheduler settings remain appropriate;
- keep transient intermediates on scratch and expensive work on Slurm; and
- never overwrite the existing result, contract, bundle, maps, or checkpoints.

After all immutable subject checkpoints exist, the exact commands recorded in
the historical result are:

```bash
module load python/3.12.1
/home/users/zijiao/pcvenv/bin/python outputs/code/audit_edge_scaling.py
/home/users/zijiao/pcvenv/bin/python outputs/code/aggregate_results.py --gram-voxel-chunk 8192
/home/users/zijiao/pcvenv/bin/python outputs/code/finalize_candidate.py
```

These are **post-checkpoint** commands, not a complete data-acquisition or
from-scratch runbook. The full fitting implementation and scheduler entrypoints
are under [outputs/code/](outputs/code/).

### Verification checklist

A faithful verification should confirm all of the following before reading the
terminal decision:

- 108 technically eligible participant checkpoints are present;
- Q counts are 108, 106, and 103 with `Q2 ⊂ Q1 ⊂ Q0`;
- both contrasts exist for every one of the 27 C×Q×S cells;
- the common mask is unchanged and contains 221,116 voxels;
- the functional-ANOVA term sums partition total across-cell sum of squares;
- the same paired bootstrap draws are reused across cells and contrasts;
- the same random deletion sets are reused across all C×S cells and contrasts;
- seeds are `20260815` for 2,000 bootstrap replicates and `20260814` for 1,000
  matched-deletion replicates;
- all declared output artifacts are regular, in-workspace files; and
- the two synthetic test entrypoints and production self-test pass.

The independent review in
[outputs/tests/math_review.md](outputs/tests/math_review.md) found no blocking
mathematical defect. It specifically tested factor ordering, projection
algebra, Shapley allocation, unequal-cohort coefficients, reduced-Gram
bootstrap calculations, and deletion metrics.

## Artifact map

### Contracts, result, and lifecycle projections

| Artifact | Role |
|---|---|
| [GOAL.md](GOAL.md) | Historical, truncated goal draft; preserved for provenance |
| [DATASETS.md](DATASETS.md) | Historical combined protocol/dataset draft |
| [Frozen contract, Markdown](outputs/frozen_analysis_contract.md) | Human-readable pre-outcome scientific contract |
| [Frozen contract, JSON](outputs/frozen_analysis_contract.json) | Machine-readable frozen contract |
| [Result](outputs/RESULT.md) | Primary human-readable synthesis |
| [Candidate bundle](outputs/candidate_bundle.json) | Exact exploratory negative-terminal bundle |
| [Experiment log](outputs/experiment_log.md) | Local execution and handoff chronology |
| [Governance](outputs/governance.md) | Projection of observed canonical state |
| [Memory](outputs/memory.md) | Local reusable lessons and nonportable conclusions |

This historical episode predates the current seven-projection convention. It
does not contain `society.md`, `loop.md`, `landscape.md`, or `verification.md`.
Their absence must not be disguised by creating documents that appear
contemporaneous with the run.

### Primary and secondary analysis evidence

| Artifact | Role |
|---|---|
| [Variance attribution](outputs/variance_attribution.tsv) | Primary attribution estimates and intervals |
| [Conclusion stability](outputs/conclusion_stability.tsv) | Continuous-pair and secondary FDR stability |
| [Multiverse manifest](outputs/multiverse_manifest.tsv) | Completed-cell accounting |
| [Matched-size calibration](outputs/matched_n_calibration.tsv) | Observed-versus-random deletion summaries |
| [Matched random sets](outputs/matched_n_random_sets.tsv) | Exact sampled deletion sets |
| [Exploratory ER influence](outputs/exploratory_er_loo_influence.tsv) | Post-primary leave-one-out diagnostic |
| [QC subject sets](outputs/qc_subject_sets.tsv) | Exact frozen membership |
| [Run/design audit](outputs/run_design_qc_audit.tsv) | Run eligibility and design checks |
| [Source audit](outputs/source_audit.json) | Input inventory audit |
| [Aggregation provenance](outputs/aggregation_provenance.json) | Completeness and aggregation provenance |

### Numerical and engineering diagnostics

| Artifact | Role |
|---|---|
| [Edge diagnostic summary](outputs/edge_scaling_diagnostic_summary.json) | Compact boundary-scaling audit |
| [Edge diagnostic table](outputs/edge_scaling_diagnostic.tsv) | Detailed edge-scaling results |
| [Engineering repairs](outputs/engineering_repairs.json) | Failures, retries, and contract-preserving repairs |
| [Mathematics review](outputs/tests/math_review.md) | Independent synthetic review |
| [Executable tests](outputs/tests/) | Synthetic and black-box test entrypoints |

### Generated maps, checkpoints, code, and operations

| Family | Contents |
|---|---|
| [Group maps](outputs/group_maps/) | 162 images: 2 contrasts × 27 cells × effect/t/FDR maps |
| [Subject maps](outputs/subject_maps/) | 216 files: JSON and NPZ checkpoint for each of 108 participants |
| [Common mask](outputs/common_analysis_mask.nii.gz) | Frozen 221,116-voxel analysis mask |
| [Figures](outputs/figures/) | Five summary figures |
| [Code](outputs/code/) | Six Python programs and two Slurm scripts |
| [Logs](outputs/logs/) | 230 historical scheduler output/error files |
| [Pilot](outputs/pilots/sub-001_pilot_1000vox.json) | Bounded subject-level pilot record |

Large generated maps, checkpoints, logs, caches, and temporary diagnostics are
present on disk, but many are not tracked by Git. Their presence must not be
mistaken for a portable dataset package, and they should not be deleted merely
to simplify this example.

## Limitations and unresolved questions

- Every scientific result came from `ds001734`; there is no independent
  confirmation dataset.
- The multiverse cells overlap heavily in participants, and bootstrap samples
  are not independent observations.
- All five strict QC exclusions were ER participants, so QC-set sensitivity is
  inseparable from sample composition and task-version imbalance.
- EI and ER have different gain supports, limiting direct task-version
  interpretation.
- Bayesian-bootstrap intervals describe participant-sampling sensitivity of
  map-level summaries; they are not voxelwise inferential intervals.
- The 95%-coverage mask admits low-intensity boundary voxels in some runs.
  Nilearn's scaling behavior at those voxels may contribute to the large S
  attribution.
- Fourteen participant sidecars predate clamp-count logging; they were not
  recomputed merely to complete a post-hoc diagnostic.
- The local `inputs/` directory does not contain the source data or the
  originally referenced `frozen_grid.json`.
- The historical Python environment is not captured as a portable lockfile or
  container.
- A negative terminal closes this exploration; it does not create a confirmed
  null claim.

## Adaptation checklist for a new episode

Use the structure and discipline of this example, not its historical facts.

- [ ] Create a new `episodeNN_topic` with its own `GOAL.md`, `DATASETS.md`,
      read-only `inputs/`, and writable `outputs/`.
- [ ] Write one precise research question and explain why the answer would be
      informative.
- [ ] Define the population, unit of analysis, conditions, contrasts, and
      primary estimand in mathematical or operational terms.
- [ ] State explicit nonclaims and distinguish exploration from confirmation.
- [ ] Record immutable dataset identifiers, versions, hashes, licenses, access
      dates, source URLs/paths, expected layout, and validation checks.
- [ ] Freeze every analysis factor and level before outcome access; state the
      exact total cell count.
- [ ] Freeze cohort membership rules, missing-data policy, and technical
      eligibility criteria without inspecting neural outcomes.
- [ ] Specify the primary endpoint, normalization, mask, uncertainty method,
      multiple-comparison treatment, and interaction allocation.
- [ ] Define a quantitative candidate rule, a falsifier, required controls, and
      `candidate_ready`, `closed_no_candidate`, and `technical_failure`
      conditions.
- [ ] Run synthetic mathematics/software tests and one bounded pilot before
      expensive execution.
- [ ] Use immutable checkpoints, manifests, fixed seeds, and Slurm for
      expensive work; verify artifacts before each downstream stage.
- [ ] Log failed attempts and explain every repair, including why it does or
      does not alter the scientific contract.
- [ ] Keep primary, secondary, and post-hoc diagnostics visibly separate.
- [ ] Treat a complete negative result as valid; never relax a frozen rule to
      manufacture a candidate.
- [ ] Query canonical state immediately before every canonical mutation and
      preserve human gates and nonacceptance boundaries.
- [ ] Maintain the current required workspace projections, but never copy this
      episode's loop IDs, handoff IDs, paths, participant sets, numerical
      results, or terminal status into another episode.
- [ ] End with a plain-language result, an artifact index, an executable
      verification recipe, known limitations, and the exact next scientific
      question—if any.
