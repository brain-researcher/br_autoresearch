# EP11 outcome-blind five-target recovery surrogate screen

**Completed:** 2026-09-26

## Scope

This screen used the observed 206-cell, 36-brain/sample metadata layout 
(cell counts, soma coordinates, provider layer, line, hemisphere, age, and lab), 
but every five-target projection composition was simulated. No real axon 
projection value, provider projection class, or development/audit identity was 
read or emitted. Role allocations were newly sampled from metadata for each 
replicate and therefore do not lock the eventual real-data roles.

The comparison used one common five-target compositional outcome and its exact 
additive-log-ratio Jacobian. `C` was a flexible continuous kernel density after 
the shared position model; `H` was a development-fitted finite mixture after the 
same position model. Candidate bandwidths and mixture configurations were chosen 
by grouped development cross-validation. Audit brains were scored directly: no 
audit refit, reclustering, label matching, or prevalence recalibration occurred.

This is a recovery-screen surrogate for the planned evaluator. A surrogate pass 
cannot activate outcome-bearing search and is not proof that the frozen 56-D 
mixed-support likelihood, production B/P/C/D/H implementation, or 99-replicate 
full-search null is qualified.

## Registered synthetic regimes

No-group stress cases: smooth spatial, curved, strongly nonuniform, 
depleted-middle, and animal-confounded continua. Positive cases: reusable groups 
without a within-group continuum, reusable groups plus a continuum, and 
spatially interdigitated reusable groups. Positive cases used effect multipliers 
0, 0.5, 1, 1.5, and 2; the fixed screening margin was 
0.050 nats/neuron. These are generator-amplitude multipliers, 
not oracle predictive gains calibrated to multiples of that margin; therefore 
the maximum-effect route rates below are diagnostics rather than power at 2x 
the scientific margin.

## Design-level surrogate result

| Design | Valid | False-group rate (95% MC upper) | Max hard-null rate | Core group route at max multiplier | Stress group route at max multiplier | Coverage | Precision < margin | Continuous adequacy at zero | Surrogate check | Registered gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 12_dev_8_audit_16_unassigned | 4000/4000 | 0.000 (0.003) | 0.000 | 0.000 | 0.000 | 0.947 | 0.000 | 0.681 | fail | not_full_registered_qualification |

Failed surrogate checks are reported literally; a failure is a design result, not evidence against the biology.

- `12_dev_8_audit_16_unassigned` failed gates: `core_recovery_at_maximum_effect_multiplier;stress_recovery_at_maximum_effect_multiplier;continuous_adequacy_zero;precision`

## Scenario readout

| Design | Scenario | Effect | N | Group route | Continuous adequate | H-C | Coverage | Half-width < margin |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 12_dev_8_audit_16_unassigned | animal_confounded_continuum | 0.0 | 200 | 0.000 | 0.975 | -10.137 | 0.920 | 0.000 |
| 12_dev_8_audit_16_unassigned | curved_continuum | 0.0 | 200 | 0.000 | 0.595 | -2.534 | 0.945 | 0.000 |
| 12_dev_8_audit_16_unassigned | depleted_middle_continuum | 0.0 | 200 | 0.000 | 0.580 | -2.338 | 0.940 | 0.000 |
| 12_dev_8_audit_16_unassigned | groups_plus_continuum | 0.0 | 200 | 0.000 | 0.610 | -2.327 | 0.965 | 0.000 |
| 12_dev_8_audit_16_unassigned | groups_plus_continuum | 0.5 | 200 | 0.000 | 0.640 | -2.656 | 0.945 | 0.000 |
| 12_dev_8_audit_16_unassigned | groups_plus_continuum | 1.0 | 200 | 0.000 | 0.620 | -2.676 | 0.920 | 0.000 |
| 12_dev_8_audit_16_unassigned | groups_plus_continuum | 1.5 | 200 | 0.000 | 0.645 | -3.241 | 0.950 | 0.000 |
| 12_dev_8_audit_16_unassigned | groups_plus_continuum | 2.0 | 200 | 0.000 | 0.600 | -3.447 | 0.930 | 0.000 |
| 12_dev_8_audit_16_unassigned | nonuniform_continuum | 0.0 | 200 | 0.000 | 0.640 | -2.251 | 0.980 | 0.000 |
| 12_dev_8_audit_16_unassigned | reusable_groups | 0.0 | 200 | 0.000 | 0.665 | -2.443 | 0.960 | 0.000 |
| 12_dev_8_audit_16_unassigned | reusable_groups | 0.5 | 200 | 0.000 | 0.630 | -2.667 | 0.940 | 0.000 |
| 12_dev_8_audit_16_unassigned | reusable_groups | 1.0 | 200 | 0.000 | 0.660 | -3.130 | 0.955 | 0.000 |
| 12_dev_8_audit_16_unassigned | reusable_groups | 1.5 | 200 | 0.000 | 0.585 | -3.371 | 0.950 | 0.000 |
| 12_dev_8_audit_16_unassigned | reusable_groups | 2.0 | 200 | 0.000 | 0.600 | -3.914 | 0.965 | 0.000 |
| 12_dev_8_audit_16_unassigned | smooth_spatial_continuum | 0.0 | 200 | 0.000 | 0.720 | -2.667 | 0.970 | 0.000 |
| 12_dev_8_audit_16_unassigned | spatially_interdigitated_groups | 0.0 | 200 | 0.000 | 0.660 | -2.403 | 0.915 | 0.000 |
| 12_dev_8_audit_16_unassigned | spatially_interdigitated_groups | 0.5 | 200 | 0.000 | 0.700 | -2.716 | 0.950 | 0.000 |
| 12_dev_8_audit_16_unassigned | spatially_interdigitated_groups | 1.0 | 200 | 0.000 | 0.620 | -3.345 | 0.975 | 0.000 |
| 12_dev_8_audit_16_unassigned | spatially_interdigitated_groups | 1.5 | 200 | 0.000 | 0.585 | -3.484 | 0.945 | 0.000 |
| 12_dev_8_audit_16_unassigned | spatially_interdigitated_groups | 2.0 | 200 | 0.000 | 0.590 | -3.713 | 0.925 | 0.000 |

## Interpretation boundary

A valid comparison may end on the continuous route. Separation, minimum 
component prevalence, and recurrence are applied only to a positive reusable-
group route. Failure to select groups is never recorded as an invalid comparison.

Before outcome-bearing search, the frozen 56-D mixed-support likelihood, the full 
production B/P/C/D/H evaluator, and the 99-replicate full-search null must receive 
registered qualification. Any failed recovery or precision check must be resolved 
prospectively. These simulations neither expose 
nor answer the real SSp-tr groups-versus-continuum question.

## Reproduction

- Primary cells: 206 in 36 brain/sample units
- Replicates requested per condition: 200
- Oracle redraws per replicate: 16
- Root seed: 20260926
- Detailed condition metrics: `synthetic_recovery_qualification.csv`
- Script: `tools/run_synthetic_recovery_qualification.py`
