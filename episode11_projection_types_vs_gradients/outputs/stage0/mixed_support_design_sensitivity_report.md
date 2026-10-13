# EP11 synthetic estimator and design sensitivity

- Analysis class: `synthetic_design_and_estimator_sensitivity_not_qualification`
- Real projection outcomes read: **no**
- Geometry constraints enforced: **no**
- Outcome-bearing search authorized: **no**
- Recorded execution predates the full-layout synthetic-covariate
  standardization repair. Its nested arms share each full-36 draw, but the
  absolute projected-proxy effect label is not qualified and must be rerun.

The fixed-split rows are metadata-only prospective allocations. Bootstrap
rows are estimator ceilings based on synthetic clones of complete metadata
units, not evidence that more independent animals exist.
The three observed-unit arms share the identical D12/A8 core and differ
only by moving the same 15 reserve units into development or audit.
Outer-CV rows are descriptive only: overlapping training sets invalidate
the independent-audit t interval, so their power and precision are `NA`.

| Design | Fixture | n | Dev | Audit | Fitted H-C | Training-label-only H-C | Oracle-density H-C | Unit SD | Halfwidth | Power | Precision | H-group accuracy | Required audit n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nested_baseline_12_dev_8_audit_16_unassigned | strong_2x_reusable_groups | 32 | 12 | 8 | -0.037 | 0.039 | 0.117 | 0.208 | 0.139 | 0.000 | 0.062 | 0.584 | 49 |
| nested_baseline_12_dev_8_audit_16_unassigned | zero_group_null_groups_plus_continuum | 32 | 12 | 8 | -0.034 | NA | 0.000 | 0.213 | 0.143 | 0.000 | 0.000 | NA | 52 |
| nested_estimator_ceiling_27_dev_8_audit_1_unassigned | strong_2x_reusable_groups | 32 | 27 | 8 | -0.043 | 0.044 | 0.117 | 0.211 | 0.141 | 0.000 | 0.031 | 0.581 | 51 |
| nested_estimator_ceiling_27_dev_8_audit_1_unassigned | zero_group_null_groups_plus_continuum | 32 | 27 | 8 | -0.055 | NA | 0.000 | 0.228 | 0.153 | 0.000 | 0.094 | NA | 59 |
| nested_precision_ceiling_12_dev_23_audit_1_unassigned | strong_2x_reusable_groups | 32 | 12 | 23 | -0.032 | 0.047 | 0.103 | 0.212 | 0.076 | 0.000 | 0.219 | 0.554 | 51 |
| nested_precision_ceiling_12_dev_23_audit_1_unassigned | zero_group_null_groups_plus_continuum | 32 | 12 | 23 | -0.030 | NA | 0.000 | 0.224 | 0.080 | 0.000 | 0.094 | NA | 57 |

The observed strong-fixture reference SD of 0.217 implies a 36-unit
one-sided 95% halfwidth of 0.061.
Therefore alternative allocations alone may not repair precision if the
unit-level variability remains unchanged.

## Interpretation and stopping rule

The D27/A8 estimator ceiling did not improve the fitted strong-fixture
contrast or group accuracy relative to D12/A8. Supplying true synthetic
training labels raised `H - C` only to about 0.04 nats/neuron, far below the
projected-oracle proxy near 0.12. The current bottleneck therefore extends
beyond label discovery to the signature, gating, likelihood, or calibration
path. The D12/A23 precision ceiling reduced the mean half-width to about
0.076, but did not reach 0.05; any feasible design with no more than 20 audit
units cannot repair this by reallocation alone if the observed unit-level
variability is unchanged.

The companion outcome-blind geometry frontier proves at most 33 assigned
units under each depth convention and does not establish one shared role
assignment across conventions. These synthetic arms are diagnostic ceilings,
not implementable role designs or registered qualification. The next step is
to repair and isolate group discovery, signature estimation, smooth gating,
and full population-optimal `C`-versus-`H` calibration before testing more role
counts or outer cross-validation.
