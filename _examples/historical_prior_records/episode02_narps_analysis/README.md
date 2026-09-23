# Episode 02 retrospective example guide

> **Added 2026-09-13. Retrospective, local, and non-canonical.** This guide was
> written after the episode to make the historical work easier to understand
> and reuse as an example. It is not part of the frozen analysis contract or
> immutable Society review packet, and it does not change any scientific or
> governance state. When acting on the episode, query the Brain Researcher MCP
> service again; never replay an action from this document.

## Episode at a glance

| Field | Historical record |
|---|---|
| Question | Why did spatial smoothing account for about 89.5% of the continuous gain- and loss-map variation in Episode 01? |
| Dataset | Public NARPS `ds001734`; 108 participants, 54 EI and 54 ER; four expected task runs per participant |
| Scope | Bounded exploratory successor; two frozen mechanism tests, not a repeat of the parent 3 × 3 × 3 grid |
| Selected local candidate | `residual_substantive_spatial_sensitivity`—a material analysis-induced spatial-pattern change, **not** a biological mechanism |
| Scientific status | `exploratory_only`; no independent confirmation or scientific acceptance |
| Last recorded local loop state | `AWAITING_REWARD`, revision 3 |
| Next actor | Human scientist |
| Next action | Re-read canonical state. If it is still `AWAITING_REWARD`, decide the reward gate; do not launch confirmation or additional science unless the canonical action subsequently authorizes it. |

Historical lineage and review identifiers:

- Parent loop: `arl_8cbd25a7a8610a008c2830ba624052c4`
- Parent Goal handoff: `goal_handoff_e3f79fc20a54776ebf304655`
- Episode 02 loop: `arl_b622657e9fbd000075c5ad5f4cf30c44`
- Episode 02 Goal handoff: `goal_handoff_daf31dc2e8aa7a9965a9a8e7`
- Review packet: `goal_review_packet_47c3c6255a89cbdc7e5692a2`
- Direction: `goal_direction_b1af31a162679fc821c548a2`
- Frozen successor contract v2 SHA-256:
  `27a58a09ee043e71b7d1fca93de0ef727b79fc6ecbd9fef2eaf800208317829f`

The identifiers above are for audit and lookup, not permission to mutate the
loop. The current canonical state always outranks this historical snapshot.

## How to use this example

Choose the route that matches your purpose:

1. **Understand the science:** read this guide, then the frozen
   [Goal](GOAL.md), [result](outputs/RESULT.md), and
   [Society projection](outputs/society.md).
2. **Audit the design:** read the prospective
   [mechanism selection](outputs/mechanism_selection.md), the
   [frozen contract](outputs/frozen_successor_contract.md), and the
   [analysis manifest](outputs/analysis_manifest.tsv) in that order.
3. **Audit provenance:** read [DATASETS.md](DATASETS.md), the
   [source inventory](outputs/source_inventory.md), its
   [machine-readable record](outputs/source_inventory.json), and the
   [per-run inventory](outputs/source_inventory_runs.tsv).
4. **Reproduce computation:** first pin a replacement for the historical
   environment, revalidate all source references, and use the scripts and
   Slurm entry points listed below. Do not overwrite the reviewed outputs.
5. **Make a governance decision:** ignore the status in this guide until the
   canonical MCP state has been re-read. The local Markdown is only a
   projection.

## Why this successor exists

Episode 01 asked whether continuous gain- and loss-related group maps were more
sensitive to confound handling and outcome-blind subject exclusion than to
spatial smoothing. Its frozen support rule required, for at least one contrast,
both `C/S > 1` and `Q/S > 1`, with simultaneous interval lower bounds above 1.
That directional hypothesis was falsified:

| Contrast | C/S | Q/S | Smoothing-attributed share |
|---|---:|---:|---:|
| `gain_demean` | 0.0758674 | 0.0427427 | about 0.8940 |
| `loss_demean` | 0.0863375 | 0.0304137 | about 0.8955 |

Episode 02 therefore did not ask whether smoothing dominated again. It asked
*why* it could dominate. The successor separated four plausible explanations:

- global amplitude or offset changes;
- boundary-mask or inverse-variance-estimator behavior;
- broad residual spatial-pattern sensitivity; and
- a disconnect between continuous-map geometry and thresholded conclusions.

Task-version/sample composition was retained as a stratified internal control,
not promoted to a causal explanation. All data remained inside `ds001734`, so
EI/ER strata, bootstraps, split halves, and leave-one-out checks are internal
robustness—not independent confirmation.

## Prospective selection and frozen decision hierarchy

Before reading new neural outcome values, the episode ranked five mechanisms
in [mechanism_selection.md](outputs/mechanism_selection.md). It retained two
tests because they shared one bounded recomputation: a fixed first-level C1
model at only 0 and 8 mm FWHM. Residual spatial sensitivity and threshold/rank
behavior were predefined alternatives or controls rather than extra searches.

The frozen v2 hierarchy was:

| Priority | Candidate | Complete frozen support rule | Primary falsifier |
|---:|---|---|---|
| 1 | Edge/weighting mechanism | For both contrasts: full IVW edge enrichment and its bootstrap lower bound ≥ 2.0; full IVW core/full ratio and its upper bound ≤ 0.75; equal-run or stabilized-IVW attenuation and its lower bound ≥ 25%; EI and ER each show enrichment ≥ 1.5 and core ratio ≤ 0.85 | Persistence in the core or failure of matched aggregation controls to attenuate the difference |
| 2 | Amplitude/offset mechanism | For both contrasts: scale attenuation ≥ 0.50 and spatial `r ≥ 0.80`; full-sample estimates and paired-bootstrap lower bounds clear both thresholds; EI and ER point estimates do likewise | Large residual after affine alignment or low spatial correlation |
| 3 | Residual spatial sensitivity | Both mechanisms above fail; in the equal-run core, both contrasts and both EI/ER strata have `D_raw ≥ 0.25` and `r ≤ 0.80` | Small core differences, high correlation, or failure in a task-version stratum |
| 4 | Continuous-threshold disconnect | For both contrasts and EI/ER: `r ≥ 0.80`, FDR Jaccard `< 0.25`, and matched-cardinality top-absolute-t Jaccard `≥ 0.50` | Low rank overlap as well as low FDR-set overlap, showing broad relocation rather than stable-rank threshold crossing |

Every candidate also required agreeing deterministic split-half directions and
the relevant leave-one-subject-out guard. If none passed, the required outcome
was `closed_no_candidate`; a source, estimability, runtime, storage, or
checkpoint blocker would instead produce `technical_failure`. No replacement
candidate could be introduced after outcome access.

The full estimands, controls, masks, seeds, model settings, resource cap, and
stop rules are authoritative in the [frozen Markdown contract](outputs/frozen_successor_contract.md)
and its [JSON counterpart](outputs/frozen_successor_contract.json).

## Data readiness and provenance

The episode used site-local, read-only references:

| Reference | Intended source | Historical Stage-0 finding |
|---|---|---|
| [`inputs/raw`](inputs/raw) | OpenNeuro raw checkout | 432 events files; 323/432 raw BOLD links readable and 109 annex links dangling |
| [`inputs/fmriprep`](inputs/fmriprep) | fMRIPrep derivatives | Complete 432/432 MNI 2 mm BOLD images, masks, and confounds for the bounded test |
| [`inputs/fitlins`](inputs/fitlins) | FitLins analyses/designs | Complete 432/432 run designs with `gain_demean`, `loss_demean`, and intercept |
| [`inputs/prior_episode`](inputs/prior_episode) | Curated Episode 01 packet | Prior exploratory evidence and lineage only; not a fresh test set |

Stage 0 found 108 participants—54 equalIndifference (EI) and 54 equalRange
(ER)—with four expected runs each. The complete intersection required for the
selected test was 432 runs. The relevant images shared one
`MNI152NLin2009cAsym` 2 mm spatial signature. The FitLins record reported
version `0.11.0.post0.dev16`, estimator `nilearn`, and configured smoothing
`5:run:iso`.

The incomplete raw annex did not block the frozen analysis because complete
fMRIPrep BOLD data supported the bounded recomputation. It remains a provenance
fact, not permission to download replacements. Absolute OAK symlinks are
site-specific and can drift; anyone rerunning this example must re-inventory
them instead of assuming this retrospective count is still current.

The curated parent packet's usage rules are recorded in
[`inputs/prior_episode/LINEAGE.md`](inputs/prior_episode/LINEAGE.md). It includes
the parent [result](inputs/prior_episode/RESULT.md),
[contract](inputs/prior_episode/frozen_analysis_contract.md), and
[candidate bundle](inputs/prior_episode/parent_candidate_bundle.json), among
other audit tables. It may explain lineage and motivate mechanisms, but it
cannot make Episode 02 independent of Episode 01.

## Outcome-blind workflow

The episode's ordering is as important as its calculations:

1. **Initialize governance.** Confirm there is no active native Goal, create a
   fresh loop and Goal exactly once, and record identifiers.
2. **Inventory without neural outcomes.** Inspect filenames, metadata, tables,
   NIfTI headers, source compatibility, compute, and storage. Produce the
   source inventory.
3. **Select mechanisms prospectively.** Rank the five candidate explanations
   and retain no more than two before reading new neural voxel values.
4. **Freeze the contract.** Specify estimands, predictions, SESOIs, masks,
   controls, strata, seed, uncertainty, stop conditions, and terminal rules in
   both Markdown and JSON.
5. **Repair feasibility only while still blind.** The initial 100%-coverage
   core failed its predeclared size guard. Before neural outcome access, the
   contract was transparently re-frozen as v2 using one six-neighbor erosion
   of the fixed 95%-coverage common mask. Endpoints and SESOIs did not change.
6. **Execute bounded compute through Slurm.** Fit the fixed C1 model at 0 and
   8 mm, retain subject/run provenance, and stop after the two selected tests
   and their controls.
7. **Aggregate and falsify.** Apply the negative control, EI/ER strata,
   deterministic halves, 1,000 paired cohort-stratified bootstraps with PCG64
   seed 20260817, and leave-one-subject-out guards.
8. **Finalize exactly one terminal bundle.** Produce `candidate_ready`,
   `closed_no_candidate`, or `technical_failure` without widening the search.
9. **Submit for governed review, then stop.** A candidate-ready submission may
   reach Society and a human reward gate; it is not scientific acceptance.

The detailed historical execution ledger is in
[experiment_log.md](outputs/experiment_log.md). Governance boundaries and the
last local state projection are in [governance.md](outputs/governance.md).

## Supported evidence summary

The terminal result did not re-estimate the parent's approximately 89.5%
variance share. It tested candidate explanations for the observed smoothing
sensitivity under the frozen successor contract.

### Continuous-map tests

| Contrast | IVW full `D_raw` | Scale attenuation (95% CI) | Pearson `r` (95% CI) | Edge enrichment (95% CI) | Core/full `D` ratio (95% CI) | Equal-run attenuation | Stabilized-IVW attenuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gain_demean` | 0.9039 | 0.0089 [0.0081, 0.0110] | 0.4410 [0.3658, 0.4281] | 3.026 [2.703, 3.468] | 1.000 [0.998, 1.003] | -0.030 | -0.000 |
| `loss_demean` | 0.9101 | 0.0068 [0.0062, 0.0111] | 0.4024 [0.3459, 0.4022] | 2.985 [2.769, 3.378] | 0.998 [0.996, 1.002] | -0.030 | 0.000 |

Under the frozen rules:

- affine alignment removed less than 1% of the normalized change and spatial
  correlations were low, so amplitude/offset dominance failed;
- the periphery was enriched by about threefold, but the difference persisted
  almost unchanged in the core and matched aggregation controls did not
  attenuate it, so the complete edge/weighting rule failed; and
- the stored residual rule, EI/ER checks, deterministic halves, and
  small-subject guard passed, producing the local candidate
  `residual_substantive_spatial_sensitivity`.

These are internal exploratory readouts. In particular, the displayed
bootstrap intervals require audit: some do not contain the corresponding
full-sample estimate, a concern later raised by review.

### Fixed threshold/rank control

| Contrast | S0 BH-FDR voxels | S8 BH-FDR voxels | FDR Jaccard | Matched `k` | Top-absolute-t Jaccard |
|---|---:|---:|---:|---:|---:|
| `gain_demean` | 20 | 19,238 | 0.0010 | 20 | 0.0811 |
| `loss_demean` | 21 | 20,029 | 0.0010 | 21 | 0.0769 |

Both FDR-set and matched-rank overlap were low. That falsified the predefined
stable-rank threshold-crossing explanation; it was more consistent with broad
rank relocation under this analysis.

The detailed values and decision audit live in
[mechanism_results.tsv](outputs/mechanism_results.tsv),
[robustness_results.tsv](outputs/robustness_results.tsv), and
[threshold_stability.tsv](outputs/threshold_stability.tsv). The concise
historical narrative is [RESULT.md](outputs/RESULT.md), and the summary figure
is [mechanism_summary.png](outputs/figures/mechanism_summary.png).

## Review disagreement and the open human gate

The [Society projection](outputs/society.md) records two distinct judgments
that must not be collapsed:

| Reviewer/router | Recorded judgment | Meaning |
|---|---|---|
| Supplemental ten-role integrator | `revise`; `reward_eligible=false` | The exploratory S0–S8 difference was credible enough to discuss, but the mechanistic label and inferential details needed revision. |
| Server-owned strict router | `eligible`; `reward_eligible=true` | The direction was allowed to reach the scientist reward decision under the workflow. This was not scientific acceptance. |

The supplemental review's main objections were:

- directly applying a matched 8-mm filter made S0 closely predict observed S8;
- smoothing-dependent scaling-floor behavior remained unresolved;
- a one-voxel, 2 mm erosion was not kernel-aware for an 8 mm operation;
- the stored residual split rule used near-tautological range bounds, even
  though the observed halves passed the intended SESOIs;
- percentile-bootstrap calibration and the mapwise BH scope needed audit; and
- AR(1), reliability, independent-pipeline reproduction, and transportability
  were not established.

The last recorded local state is therefore `AWAITING_REWARD`, not “accepted”
and not “ready to confirm.” The next valid operation belongs to a human
scientist after a fresh canonical-state read. If canonical state differs from
this guide, follow canonical state.

## Limitations and claim boundary

This episode supports only a narrow methods statement: under one frozen
within-`ds001734` analysis, the S0–S8 group-map difference was large, poorly
removed by affine map alignment, persistent in the chosen core and aggregation
controls, and accompanied by low matched-rank overlap.

It does **not** establish:

- a biological or neural mechanism;
- an independently confirmed smoothing effect;
- generality beyond `ds001734`, the fixed C1 model, or the selected contrasts;
- a causal effect of QC, task version, or sample composition;
- that every residual difference exceeds the deterministic consequence of
  applying an 8 mm kernel;
- calibrated uncertainty until the bootstrap anomaly is resolved;
- reliability across sessions, estimators, packages, or acquisition sites; or
- scientific acceptance, confirmation eligibility, or permission to launch
  more computation.

`gain_demean` and `loss_demean` were analyzed separately. Their magnitudes
should not be compared or pooled into an EI/ER gain-minus-loss claim.

## Reproduction and verification notes

The historical commands, from this episode directory, were:

```bash
module load python/3.12.1
/home/users/zijiao/pcvenv/bin/python outputs/code/inventory_sources.py
/home/users/zijiao/pcvenv/bin/python outputs/code/build_masks.py
mkdir -p /scratch/users/zijiao/episode02_narps_analysis/{logs,checkpoints}
sbatch outputs/code/fit_subjects.sbatch
# Continue only after exactly 108 valid subject checkpoints.
sbatch outputs/code/aggregate_results.sbatch
/home/users/zijiao/pcvenv/bin/python outputs/code/finalize_results.py
```

These commands document history; they are not a portable environment
specification. Before any rerun:

1. work in a new scratch/output namespace so the reviewed files stay intact;
2. record the current source targets and re-run the source inventory;
3. capture Python, Nilearn, NumPy, SciPy, NiBabel, and scheduler versions;
4. verify the frozen JSON hash shown at the top of this guide;
5. run the mathematical checks in
   [test_pipeline_math.py](outputs/tests/test_pipeline_math.py)—the historical
   environment lacked `pytest`, so six functions were invoked directly;
6. require exactly 108 complete, valid subject checkpoints before aggregation;
7. compare the new manifest and output hashes to the historical
   [analysis manifest](outputs/analysis_manifest.tsv); and
8. treat any mismatch as a reproducibility finding, not as permission to edit
   the frozen record.

The executable pipeline is split into
[inventory_sources.py](outputs/code/inventory_sources.py),
[build_masks.py](outputs/code/build_masks.py),
[fit_subject.py](outputs/code/fit_subject.py),
[aggregate_results.py](outputs/code/aggregate_results.py), and
[finalize_results.py](outputs/code/finalize_results.py). The two scheduler
entry points are [fit_subjects.sbatch](outputs/code/fit_subjects.sbatch) and
[aggregate_results.sbatch](outputs/code/aggregate_results.sbatch).

## Artifact map

| Role | Start here | Other relevant artifacts |
|---|---|---|
| Scientific scope | [GOAL.md](GOAL.md) | [DATASETS.md](DATASETS.md) |
| Parent lineage | [prior-episode LINEAGE](inputs/prior_episode/LINEAGE.md) | [parent result](inputs/prior_episode/RESULT.md), [parent contract JSON](inputs/prior_episode/frozen_analysis_contract.json), [parent candidate bundle](inputs/prior_episode/parent_candidate_bundle.json) |
| Outcome-blind preparation | [source inventory](outputs/source_inventory.md) | [source inventory JSON](outputs/source_inventory.json), [per-run inventory](outputs/source_inventory_runs.tsv), [mechanism selection](outputs/mechanism_selection.md) |
| Frozen design | [successor contract](outputs/frozen_successor_contract.md) | [contract JSON](outputs/frozen_successor_contract.json), [mask summary](outputs/mask_summary.json) |
| Implementation | [code directory](outputs/code) | [pipeline math tests](outputs/tests/test_pipeline_math.py) |
| Evidence | [result](outputs/RESULT.md) | [mechanism results](outputs/mechanism_results.tsv), [robustness](outputs/robustness_results.tsv), [threshold stability](outputs/threshold_stability.tsv), [group maps](outputs/group_maps), [summary figure](outputs/figures/mechanism_summary.png) |
| Terminal handoff | [candidate bundle](outputs/candidate_bundle.json) | [decision record](outputs/decision.json), [analysis manifest](outputs/analysis_manifest.tsv) |
| Human-readable history | [experiment log](outputs/experiment_log.md) | [memory](outputs/memory.md), [governance](outputs/governance.md), [Society](outputs/society.md) |

The candidate bundle uses a historical `CandidateBundleV1` schema. Preserve it
as historical evidence; do not silently modernize it into the current schema.
Likewise, Episode 02 predates the current seven-projection local policy and has
no `outputs/loop.md`, `outputs/landscape.md`, or `outputs/verification.md`.
Their absence should be taught as a legacy difference, not backdated away.

## Prioritized follow-up study design

This is a planning recommendation, not an authorized experiment. It may begin
only after the human reward decision and any subsequent canonical approval.

### 1. Resolve the deterministic matched-kernel explanation

Freeze a primary comparison between the observed S8 result and the result
predicted by applying the exact 8 mm spatial operator to the S0 pathway. Define
the operator, boundaries, resampling, normalization, and comparison space
before neural outcome access. The primary estimand should isolate change beyond
the deterministic filtering prediction. Failure to exceed a prospectively
chosen SESOI should reject the stronger “residual substantive” interpretation.

### 2. Repair numerical and spatial controls prospectively

- Harmonize or explicitly model smoothing-dependent signal-scaling floors.
- Replace the one-voxel core with kernel-aware interior definitions and report
  a fixed sensitivity series chosen without outcomes.
- Use synthetic images with known amplitude, edge, and kernel effects to test
  all estimands and negative controls.
- Define non-vacuous split-half inequalities directly in terms of the intended
  SESOIs, not broad range checks.
- Audit percentile-bootstrap construction; require intervals to satisfy basic
  consistency checks or predeclare another calibrated interval procedure.
- State the family over which BH-FDR is applied and test reliability separately
  from cross-condition similarity.

### 3. Reproduce with an independent implementation

Implement the frozen design in a second validated pipeline or package without
copying numerical intermediates. Compare subject-level maps, group estimands,
and decision booleans under predeclared tolerances. Disagreement should pause
claim promotion and become the result of the replication audit.

### 4. Confirm on fresh data

Select and version-pin an independent gain/loss task-fMRI dataset with no
participants or derived maps shared with `ds001734`. Freeze eligibility,
contrasts, acquisition constraints, exclusions, missing-data rules, kernels,
masks, estimands, uncertainty, SESOIs, and stop rules before accessing its
neural outcomes. Do not tune the confirmation design to reproduce the Episode
02 effect size, and do not count EI/ER or resampling inside `ds001734` as fresh
confirmation.

### 5. Bound the claim after confirmation

Report whether the effect transports across datasets, pipelines, spatial
resolutions, and task versions. If it does not, narrow the claim to the
conditions that were actually supported rather than adding more exploratory
branches to the confirmation run.

## What to copy—and what not to copy

### Copy this pattern

- Start with an explicit parent result and a non-repetition rule.
- Rank competing explanations before outcome access.
- Retain only a small number of tests with explicit falsifiers.
- Freeze human-readable and machine-readable contracts together.
- Separate feasibility repair from outcome-driven adaptation and log any
  pre-outcome re-freeze.
- Use matched negative controls, subject-level provenance, fixed randomness,
  explicit resource caps, and terminal stop rules.
- Separate exploratory evidence, workflow eligibility, human reward, launch
  approval, confirmation, and scientific acceptance.
- Run expensive work through Slurm and keep high-frequency intermediates on
  scratch.
- Preserve negative and contested results instead of widening the search.

### Do not copy these episode-specific details

- historical loop, Goal, review-packet, or direction identifiers;
- OAK symlink targets, user-specific Python paths, scratch paths, or Slurm IDs;
- participant counts, contrasts, masks, kernels, thresholds, seeds, or effect
  sizes without a new scientific justification;
- the legacy CandidateBundleV1 schema;
- the assumption that internal strata or resampling are independent
  confirmation;
- the label “substantive spatial sensitivity” without first resolving the
  matched-kernel and numerical-control objections;
- a Society `eligible` result as evidence of scientific acceptance; or
- any action from this guide without a fresh canonical-state read.

## Preservation note

All files that existed before this retrospective guide remain the historical
record. In particular, do not fix wording, formatting, or even harmless-looking
typos inside reviewed artifacts: their byte identity may be part of local-to-
server provenance. Add future explanation in a separately dated document and
make its non-canonical status explicit.
