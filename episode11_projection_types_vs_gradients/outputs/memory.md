# Memory

## Latest babysitting observation — 2026-09-30

The user requested continued babysitting. The same Slurm job46143347 is
RUNNING, startedSeptember30 18:21:20 PDT, with3CPUs/16GB/72h and no
requeue. Its allocation cutoff isOctober3 18:21:23 PDT; the renewed episode
window endsOctober5 18:12:23 PDT. The two unchanged baseline starts converged
at objectives4.57373073603 and4.44457198869. The revised empirical start is
still computing; CPU/read counters advanced and no error is logged. No fresh
verification or terminal result exists. Follow this job to its result/cutoff,
preserve actual artifacts and assess frozen gates. Real outcomes remain closed.


## Current continuation — explicit reset and submitted attempt

Clock-origin blocker cleared by explicit user reset. Window ends October5,18:12:23 PDT. Single H revision job46143347 submitted; initial state PENDING. Reuse passing regression; follow this job and adjudicate original gates, preserving failed attempt.

Actual authority/submission records: `outputs/stage0/episode_wall_clock_reset_20260930.json` and `outputs/stage0/population_h_empirical_perturbation_v1_submission.json`.


## Latest continuation — 2026-09-30

The empirical_perturbation_v1 initializer is implemented and its one-replicate array regression passed. No optimizer ran. The failed H diagnostic and research/data contracts are unchanged. Next action: verified remaining global allowance, then a prospectively bounded H attempt.

Actual regression: `outputs/stage0/population_h_empirical_perturbation_v1_regression.json`. H recovery remains untested for this revision; launch awaits resource accounting.


## 2026-09-30 — Terminal H attempt

`population_h_optimizer_recovery`, job `45736553`, completed `0:0` at
16:20:59 PDT after 52:48:32, but its frozen recovery diagnostic failed:
`revise_population_H_optimizer_iteration_cap`. Empirical start reached 1200
iterations; best-two fresh score gap 0.129459 exceeds 0.002. Best truth-based
H passes its truth-deficit bound, localizing the failure to initialization
recovery. Actual summary/log/accounting and terminal QC are preserved; see
`stage0/population_h_recovery_report.md`. Panel A and all real outcomes remain
closed. Stop before another attempt at the prospective start-strategy and
existing-budget gate. Terminal ASTRA export succeeded; its snapshot is kept
separately and the validated active C/H program was restored.

## 2026-09-30 — Resumed H diagnostic

`population_h_optimizer_recovery` remains the single existing attempt
`45736553`; observed RUNNING at 16:14 PDT after 52.695 hours. Two starts have
logged KKT convergence, the empirical start and final gates remain pending.
No new execution or real-outcome access occurred. A bundled clarification of
the existing global 120-hour clock/remaining budget is pending; no ceiling was
changed. Actual observations, partial QC, and the durable C checkpoint are
linked from `experiment_log.md`. The active C/H ASTRA program validates;
legacy snapshot export separately returned `astra_projection_conflict`.

## 2026-09-30 — Paper-question revision

- Ask whether reusable allocation patterns add useful information beyond a
  flexible continuum among nearby `SSp-tr` neurons, not whether clustering
  discovers new cell types.
- One principal named target-allocation rule is the intended paper object;
  the current 56-coordinate primary, model grammar, role counts, thresholds,
  and access boundary are unchanged.
- Latent labels are integrated for prediction; post-score membership is
  evidence about recurrence, not an independent label predictor.
- 206 cells do not establish 206 independent replicates; the 36 brain/sample
  pairs lack animal certification, and the intended audit has eight units.
  Production recovery/precision remain unresolved. A narrower target primary
  would require a prospective redesign, not a rescue test.
- Winnubst 2019 and Timonidis 2023 are added prior boundaries. Conceptual
  figure: `ep11_conceptual_question.png`; it contains no observations.
- Recorded synthetic job state below was not refreshed in this writing task.

## Active version 2

- The Gao whole-cortex targeted `SSp-tr` acquisition contains 214 SWCs from 36
  unique one-to-one fMOST brain/sample-ID pairs (332,357,190 bytes).
- The axon/dendrite-distinguished primary scope contains 206 provider-labeled
  `Axon_and_dendrite` reconstructions; the eight `Axon_only` files are
  sensitivity-only. All 36 pairs retain at least one primary-scope cell. This
  provider distinction does not establish axon completeness, which needs
  separate QC.
- Provider connectivity classes and projection summaries are excluded. Raw
  files are held in episode scratch storage.
- All 206 primary-scope cells and all 36 pairs are cortex-geometry eligible.
  The final 96-row screen crosses two reconstruction scopes with exact-polyline
  and `LineString3D` endpoint-handling depth projections. Each of the four
  grids independently has 16/18 feasible finite-depth settings, the same
  boundary, and no indeterminate result. A line-balanced 12-development/
  8-audit allocation exists at 500 µm pial geodesic distance with normalized-
  depth tolerance 0.10 and every broader tested finite setting. At 250 µm,
  tolerances 0.10/0.20 are infeasible and 0.30 is feasible. The projections'
  absolute normalized-depth difference has median 0 and maximum 0.0323.
- The screen requires two distinct development-brain neighbors per cell,
  development leave-one-out, 0.90 per-brain and 0.95 per-role retention, eight
  multi-cell development brains, both roles for each non-singleton line, and
  an unassigned singleton `Tle4` brain. It emitted no allocation witness.
- Stage-0 requalification remains active pending independent-animal identity,
  reconstruction-completeness QC, and production model qualification. The
  outcome is now fixed as 28 named targets by two lateralities, but no real
  outcome has been constructed.
- Current synthetic evidence says sample reallocation is not the first repair:
  D27/A8 did not improve recovery, D12/A23 still had a 0.076 half-width, and
  geometry permits at most 33 assigned units. True development labels rescued
  only about 0.04 nats/neuron under the old estimator.
- The old absolute proxy scale is unqualified because its calibration and
  recovery standardized synthetic covariates on different index scopes. The
  repaired generator is subset-invariant. The identified full-family optimizer
  and exact truth anchors now pass, but a 32-draw C MLE overfits and is not a
  population calibration. The first complete positive-amplitude C ladder
  failed only its two-start `R=128` stability gate because the empirical start
  persisted in a bad local basin. The prospectively frozen
  `r128_three_fresh_v1` schedule retained all rungs and thresholds, used fresh
  validation/terminal streams, and passed all ten C pilot gates with exact
  status `pass_population_C_SAA_pilot_gates_not_Panel_A_calibration`. Its
  separate durable summary is
  `outputs/stage0/population_saa_r128_three_fresh_full_ladder_summary.json`.
  One fitted-H population optimizer-recovery job (`45736553`) is now running;
  the finite-pool H probe still establishes execution only. Panel A remains
  incomplete, and no roles or projection outcomes exist yet.

## Historical version 1

- Liu et al. (2024) Supplementary Data 1 supplies the Image ID-to-Animal ID
  crosswalk. The aggregate join resolved all 39 workbook families and 1,876
  cells with no ambiguity or count mismatch: 38 match Image ID directly and
  one 43-cell family uses its paired Animal ID as an alias.
- A provisional screen of all 92 exact provider source labels found zero with
  at least 20 animals; the maximum was 17. The follow-up used only predefined
  Allen-ontology parents.
- The parent screen found two minimal count-qualified candidates: `SS` with 22
  animals/324 cells and `STR` with 21/355. `STR` had no same-label 13/8 split
  through 1,500 µm and passed only by pooling distinct striatal labels, so it
  was rejected.
- Under the stronger rule requiring two distinct development-animal neighbors
  per retained same-label cell, the provisional 3-D Euclidean result required
  broad 1,250--1,500-µm radii and never reached a 0.95 per-animal floor.
- Safe coordinate checks support provider `X/Y/Z` as Allen `AP/DV/ML` in
  micrometres. SS is bilateral (149/175 cells; 13/22 animals span both halves),
  so the final screen folded homologous hemispheres and used official 10-µm
  streamlines, pial-surface geodesics, and normalized cortical depth. Provider
  labels remained fixed; atlas proximity never relabeled a cell.
- Five of 324 SS cells lacked a valid streamline and were excluded before
  retention denominators. Generic `SSp` then failed the global necessary
  five-animal rule, leaving 318 scored cells in eight fixed exact provider-
  label strata across all 22 animals.
- The final 384-row grid exhaustively compared 12/8+2 and 14/8 across six
  tangential radii, three finite depth tolerances plus depth-unrestricted
  diagnostics, four retention floors, and two depth-projection conventions.
  Independent audits passed its arithmetic and implementation.
- Fully valid counts were zero everywhere. For finite cortex-aware depth,
  support-only counts were also zero even at the 0.80 sensitivity floor. Best
  unconstrained primary maximin retention was 0.778 for 12/8+2 and 0.500 for
  14/8.
- `SSp-tr` is a deterministic bottleneck: five singleton animals admit no
  supported 3-development/2-audit assignment at 1,500 µm, even with depth
  unrestricted and under either depth-projection convention.
- Version 1 closed as `closed_insufficient_source_support`. No projection
  outcome was opened, no roles were assigned, and audit open count is zero.
  Precision and model simulations were not run after the necessary support
  failure.
- This closure is a data--design mismatch, not evidence for a continuum or
  against projection groups. Removing `SSp-tr` would define a new seven-
  stratum, 21-animal design requiring 12/8/1-versus-13/8 requalification.
