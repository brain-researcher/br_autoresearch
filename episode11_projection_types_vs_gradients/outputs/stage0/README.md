# EP11 Stage-0 artifacts

This directory retains the completed v1 A1876 qualification and contains the
active v2 Gao `SSp-tr` requalification. Historical files are not rewritten as
if the added resource had been available in v1.

- `readiness_status.yaml`: current v2 readiness state and retained v1 result.
- `redacted_metadata_contract.yaml`: fields allowed for identity and coverage
  assessment, plus fields that must remain hidden.
- `steward_handoff.md`: the minimal data-separation request.
- `tools/validate_redacted_metadata.py`: trusted-side validator that emits
  identifier-free source-support summaries.
- `fixtures/` and `synthetic_validation.yaml`: small synthetic checks of the
  allow-list and whole-animal role rule.
- `provisional_source_brain_proxy_census.*`: exact provider-label count screen.
- `provisional_allen_parent_proxy_census.*`: fixed-ontology parent count screen.
- `identity_join_check.md`: aggregate verification of the 39-family animal join.
- `shared_ep09_ep10_exposure_check.md`: outcome-free sibling status check.
- `candidate_common_support_screen.*`: exhaustive SS/STR spatial-support and
  whole-animal split feasibility screen.
- `ss_two_neighbor_support_screen.*`: exhaustive SS comparison of 12/8 plus
  two unassigned animals versus forced 14/8 when each retained cell requires
  direct support from two distinct development animals.
- `ss_cortex_geometry_support_screen.*`: final hemisphere-folded pial-surface
  geodesic and normalized-depth qualification, including the fixed depth-
  projection sensitivity and exact provider-label role gate for v1.
- `tools/download_whole_cortex_ssptr.py`: targeted, resumable download and SWC
  validation for the 214 provider-labelled Gao `SSp-tr` cells.
- `whole_cortex_ssptr_inventory.csv` and
  `whole_cortex_ssptr_download_summary.json`: v2 targeted acquisition inventory
  and structural QC, explicitly excluding the published projectome `class`.
- `whole_cortex_ssptr_acquisition.md`: concise acquisition, eligibility, and
  SWC node-type interpretation for 214 downloaded and 206 primary-eligible
  cells.
- `whole_cortex_ssptr_geometry_support.*`: v2 cortex-aware common-
  support and line-balanced 12/8 feasibility screen. It passes at 500 µm
  pial-geodesic distance with 0.10 normalized-depth tolerance, without
  emitting or locking a role assignment.
- `whole_cortex_ssptr_identity_unit.md`: strict identity result. The 36
  fMOST IDs and 36 provider sample IDs form a one-to-one set of distinct
  brain/sample units; the public release does not certify that they are 36
  independent animals.
- `projection_outcome_contract.yaml` and `allen_target_vocabulary.csv`: the
  frozen outcome-blind definition of 28 named Allen targets crossed with
  ipsilateral/contralateral side, scored as one 56-coordinate relative axon-
  length composition with exact zeros.
- `tools/build_allen_target_vocabulary.py`: deterministic builder and ontology
  partition self-test for the fixed target vocabulary.
- `tools/construct_projection_outcome.py` and
  `tools/test_construct_projection_outcome.py`: segment-level atlas/midline
  clipping constructor and synthetic tests. These tests do not construct or
  inspect any real projection outcome.
- `mixed_support_56d_qualification_scope.yaml` and
  `tools/run_mixed_support_56d_qualification.py`: executable common-score
  subset for 56-D exact-zero compositions and B/P/C/D/H. The accompanying
  `mixed_support_56d_qualification_{report,summary}` record a pre-repair run
  with 280/280 valid synthetic comparisons but zero reusable-group recovery at
  the maximum tested effect. The comparator has since been factorized so H is
  exactly C plus the D/H-shared group term. Its current smoke implementation
  also passes four-fold whole-unit separation, freezes a C-selected noise
  scale into H, fits the shared group term conditional on C, and independently
  verifies a projected-oracle effect proxy.
- `mixed_support_56d_crossfit_smoke_{report,summary}`: the current 400-
  comparison result. All comparisons were finite and the no-group false-
  predictive-gain rate was zero, but strong-group predictive-gain recovery was
  zero and the overall 0.05-nat precision rate was 0.0175. Status is `revise`;
  this is not the full registered qualification.
- `mixed_support_design_sensitivity_{report,summary,replicates}` and
  `tools/run_mixed_support_design_sensitivity.py`: a 32-replicate
  common-random-number diagnostic. The infeasible D27/A8 estimator ceiling did
  not improve recovery; the infeasible D12/A23 precision ceiling reached only
  a 0.076 mean one-sided half-width. True training labels produced only about
  0.04 nats/neuron fitted gain, so estimator, signature, gating, and full-C
  calibration repair precede more role-count work.
- `mixed_support_group_module_ablation_{report,summary,replicates}` and
  `tools/run_mixed_support_group_module_ablation.py`: the repaired-generator
  256-replicate synthetic attribution. Exact truth arms pass, but the joint
  MC-SE rule stopped unresolved at the replicate cap. Its point estimates are
  consistent with losses in discovery, signature estimation, and fitted
  gating; they are not a formal component attribution or qualification.
- `mixed_support_estimator_repair_design.md` and
  `tools/run_mixed_support_full_family_calibration.py`: the next estimator
  repair contract and an executable synthetic-only finite-pool diagnostic of
  separately optimized bounded C/H smoke families. Identified-coordinate,
  likelihood, gradient, nesting, symmetry, and extreme-logit tests pass, and a
  representative C optimizer run converges. The 32-draw C MLE nevertheless
  overfits its finite pool and is not an accepted population calibration;
  positive-amplitude C was therefore evaluated through the increasing
  group-stratified SAA path described below.
- `population_saa_full_ladder_summary.json`,
  `population_saa_r128_three_fresh_full_ladder_summary.json`, and
  `tools/run_mixed_support_population_saa.{py,sbatch}`: the original failed
  two-start full-ladder C result, the separate passing three-fresh revision,
  and the synthetic-only bounded-memory runner. The prospectively frozen
  revision retained every `128/512/2048/8192` rung and threshold, used fresh
  validation/terminal streams, and passed all ten C pilot gates with status
  `pass_population_C_SAA_pilot_gates_not_Panel_A_calibration`.
- `tools/run_mixed_support_population_h_recovery.{py,sbatch}`: the synthetic
  fitted-H optimizer-recovery consumer. Its strict passing-C intake and 18
  structural checks pass; one attempt is running as Slurm job `45736553`.
  This is not Panel A calibration and opens no real outcome.
- `whole_cortex_ssptr_nested_role_design_sensitivity.*` and
  `whole_cortex_ssptr_nested_frontier_sensitivity.*`: outcome-blind geometry
  diagnostics. All-35 nested ceilings are infeasible; each depth convention
  assigns at most 33 units. Separately feasible counts do not establish one
  common allocation, and no role redesign is selected.
- `synthetic_recovery_qualification.*`: completed five-target, outcome-blind
  surrogate for the fixed 12-development/8-audit/16-unassigned design. It
  failed recovery and precision checks and cannot authorize real-outcome
  search.

The v1 terminal status remains `closed_insufficient_source_support`; v2 is in
Stage-0 requalification. Independent-animal identity, reconstruction
completeness/truncation, and full production recovery/precision remain open.
No development/audit split is locked and audit open count remains zero.
