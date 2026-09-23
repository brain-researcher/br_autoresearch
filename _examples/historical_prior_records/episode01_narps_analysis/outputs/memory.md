# Research memory

> Client-maintained workspace projection only. This is not Brain Researcher
> memory and must not be promoted outside this workspace without a separate,
> scientist-requested memory write.

## Reusable local lessons

- A completed exploratory multiverse may validly close without a candidate.
  The frozen rule required both C/S and Q/S point estimates and simultaneous
  lower bounds to exceed 1; neither contrast met it. Evidence:
  `outputs/frozen_analysis_contract.md`, `outputs/RESULT.md`,
  `outputs/variance_attribution.tsv`.
- Engineering repair must remain separate from scientific reinterpretation.
  The extra temporal-mean guard was removed without changing the frozen
  analysis contract, and failed tasks had produced no immutable checkpoint.
  Evidence: `outputs/engineering_repairs.json`,
  `outputs/aggregation_provenance.json`.
- The matched-size deletion calibration can reveal unusual sample-composition
  changes without turning them into a causal QC claim. Evidence:
  `outputs/matched_n_calibration.tsv`, `outputs/RESULT.md`.

## Approaches not carried forward

- Do not carry forward a claim that confound and subject-exclusion sensitivity
  both exceed smoothing for this estimand; the complete frozen rule was not
  met. Evidence: `outputs/RESULT.md`, `outputs/variance_attribution.tsv`.
- Do not treat Q differences as a causal effect of QC because all five frozen
  exclusions are ER participants. Evidence: `outputs/qc_subject_sets.tsv`,
  `outputs/RESULT.md`.
- Do not use a pooled EI/ER gain-minus-loss contrast as the primary result
  because task versions have different gain support. Evidence:
  `outputs/frozen_analysis_contract.md`, `outputs/RESULT.md`.
- Do not introduce post-hoc mask trimming or a variance floor to rescue the
  frozen decision. Evidence: `outputs/edge_scaling_diagnostic_summary.json`,
  `outputs/RESULT.md`.

## Observed failure modes

- An extra per-run near-zero temporal-mean guard was stricter than the frozen
  95% coverage-mask contract and caused six array tasks to fail before
  checkpoint creation. Evidence: `outputs/engineering_repairs.json`.
- A full-4D development diagnostic exceeded the 4 GiB interactive allocation
  and was replaced by bounded checks and Slurm-scale retries. Evidence:
  `outputs/engineering_repairs.json`.
- Dependency patterns tied to failed array tasks left pending aggregation jobs
  unsatisfiable; final aggregation independently required all 108 valid
  checkpoints. Evidence: `outputs/engineering_repairs.json`,
  `outputs/aggregation_provenance.json`.
- A terminal bundle persisted before the durable negative-terminal reservation
  contract can own a loop revision while remaining unreconciled. A fresh
  handoff must not steal that ownership, and `terminal_no_review` must not be
  misreported as `COMPLETE`. Evidence: `outputs/governance.md`.

## Constraints and data gaps

- All findings use ds001734 and remain `exploratory_only`; independent fresh
  confirmation is required. Evidence: `outputs/candidate_bundle.json`,
  `outputs/RESULT.md`.
- The edge-scaling audit is diagnostic only, and 14 compatible sidecars predate
  clamp-count logging. Evidence:
  `outputs/edge_scaling_diagnostic_summary.json`, `outputs/RESULT.md`.
- Analysis cells overlap in subjects and are not independent observations.
  Evidence: `outputs/frozen_analysis_contract.md`, `outputs/RESULT.md`.

## Open questions

- How much of the observed smoothing sensitivity reflects boundary scaling
  behavior versus spatial regularization remains unresolved.
- Any independent confirmation would require a separately frozen and approved
  episode; this negative replay creates none.
- The original handoff requires an explicit, server-validated legacy adoption
  path before its exact persisted submission can close the outer loop.

## Promotion status

- Not requested.
