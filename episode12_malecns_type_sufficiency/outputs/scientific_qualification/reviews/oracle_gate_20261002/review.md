# Focused oracle-gate review — 2026-10-02

The review supports continuing the existing bounded refinement. It identifies
unresolved oracle precision, not a diagnosed T/U/M operating-characteristic
failure. The decision and execution authority are unchanged.

BR supplied external-review directive `ext_review_dir_c87e4d3047c6`, including
its four-axis rubric and BR-KG criteria. The local reviewer
`/root/interpret_gate` read the listed episode artifacts; the root agent
cross-checked the numerical facts and descriptive scaling. BR did not read
these artifacts or independently execute the scientific judgment. The submitted
external verdict was accepted by BR; the actual directive, verdict and receipt
are preserved alongside this report. This is not Formal Society or canonical
acceptance.

## Evidence and scope

- `../../stochastic_oracle_v3/qualification_attempt001/qualification_look06.json`:
  4,608 rows per scenario; 35 scenarios; 25 meet the 0.002 tolerance;
  `all_cells_precision_met=false`; `next_action.action=refine_oracle_bank`
  to 9,216. Worst maximum half-width: 0.00584555890821066 in
  `curved_continuum_sparse_intermediates__n7__N100__s1p8__m0p1`.
- `../../../executor/SYNTHETIC_QUALIFICATION_SPEC_POST36_V3.yaml`:
  expected fixed finite-procedure-gain estimand, independent oracle namespace,
  error-budget allocation, and frozen terminal bank size of 12,500.
- `../../../executor/SCIENTIFIC_CONTRACT_POST36_V3.yaml`:
  simultaneous Gaussian-multiplier enclosure retains a multivariate-CLT
  assumption. Error-budget accounting is not proof of exact finite-sample
  coverage.
- `../../../executor/slurm/run_stochastic_oracle_refinement.sbatch`:
  follows complete-aggregate decisions within the frozen look schedule.
- `../../../executor/slurm/qualify_after_stochastic_oracle.sbatch`:
  oracle precision releases generated qualification only. Conformance and
  operating-characteristic qualification must pass before development replay.

At 11:02 PDT, root observed job 45736217 RUNNING with 31/35 look7 shards and
job 45736218 PENDING on its original dependency. Partial look7 shards are runtime
progress only; no global precision decision is inferred from them.

## Focused critique

Look6 reports uncertainty in the oracle calibration target. It does not report
outer false-M, false-adequacy, coverage or numerical-acceptance results. Thus
neither a method-validity pass nor a method failure follows from this report.
No matching BR-KG rule violation was established. MRI, brain-map and subject-
manifest checks are inapplicable to this generated-data gate and were not
introduced as new requirements.

A descriptive scaling calculation from existing look6 widths, holding variance
and the max-t critical value fixed, gives:

| Rows per scenario | Worst scaled half-width | Scaled widths above 0.002 |
| --- | ---: | ---: |
| 9,216 | 0.0041334343438211895 | 5 |
| 12,500 | 0.0035491738265535037 | 3 |

Formula: observed look6 width times sqrt(4608 / proposed rows). These are
extrapolations, not observations, confidence bounds, new experiments or an
early-stop rule. Variances, correlations and Monte Carlo-calibrated critical
values can change. They indicate that the frozen cap may remain unresolved,
so completion of the running computation does not promise a precision pass.

## Decision effect

The evidence supports the current action: allow the same bounded job to finish
look7 and, if the aggregate requires it, the already-authorized final look8.
The BR verdict is `explore_more` because completeness is still incomplete;
here this means only that existing chain. It grants no new experiment,
threshold change, resource extension or data access.

A precision pass releases the existing generated-qualification dependency.
Passing frozen conformance and operating-characteristic qualification is
required before replaying the development prefix with the corrected fitter.
If the final bounded aggregate remains unresolved, fail closed and request a
scientific amendment before proceeding past that gate. Final connectivity
remains outside current authority.

