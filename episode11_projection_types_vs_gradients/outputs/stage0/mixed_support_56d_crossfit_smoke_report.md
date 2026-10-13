# EP11 56-coordinate mixed-support synthetic smoke subset

- Subset status: `revise`
- Qualification class: `synthetic_smoke_subset_not_registered_qualification`
- Full registered gate: `not_evaluated`
- Outcome-bearing search authorized: **no**
- Real projection outcomes read: **no**

All B/P/C/D/H values are nats per neuron on the same measure: counting
measure over nonempty support sets times face-specific Hausdorff measure.
The continuous route is evaluated whenever the common comparison is valid;
it does not require a group-separation or group-reuse gate. This subset does
not call a continuous conclusion because absolute C/P calibration is not yet
implemented, and it does not call a reusable-group conclusion because the full
position-standardized reuse policy is not evaluated.

## Condition summary

| Scenario | Target margin multiple | Calibrated amplitude | n | Valid | Predictive group gain | No meaningful group gain | H-C | H-D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| animal_confounded_continuum | 0.00 | 0.0000 | 20 | 1.000 | 0.000 | 0.400 | -0.038 | -1.326 |
| curved_horseshoe_continuum | 0.00 | 0.0000 | 20 | 1.000 | 0.000 | 0.350 | -0.016 | -0.949 |
| depleted_middle_continuum | 0.00 | 0.0000 | 20 | 1.000 | 0.000 | 0.350 | -0.048 | -1.040 |
| groups_plus_continuum | 0.00 | 0.0000 | 20 | 1.000 | 0.000 | 0.350 | -0.011 | -0.889 |
| groups_plus_continuum | 0.50 | 0.1117 | 20 | 1.000 | 0.000 | 0.350 | -0.025 | -1.257 |
| groups_plus_continuum | 1.00 | 0.1392 | 20 | 1.000 | 0.000 | 0.400 | -0.010 | -0.600 |
| groups_plus_continuum | 1.50 | 0.2400 | 20 | 1.000 | 0.000 | 0.250 | 0.005 | -1.000 |
| groups_plus_continuum | 2.00 | 0.3019 | 20 | 1.000 | 0.000 | 0.350 | -0.023 | -0.942 |
| nonuniform_continuum | 0.00 | 0.0000 | 20 | 1.000 | 0.000 | 0.500 | 0.026 | 1.005 |
| reusable_groups | 0.00 | 0.0000 | 20 | 1.000 | 0.000 | 0.600 | -0.048 | -2.671 |
| reusable_groups | 0.50 | 0.1105 | 20 | 1.000 | 0.000 | 0.350 | 0.006 | -2.635 |
| reusable_groups | 1.00 | 0.1905 | 20 | 1.000 | 0.000 | 0.400 | -0.016 | -2.489 |
| reusable_groups | 1.50 | 0.2574 | 20 | 1.000 | 0.000 | 0.500 | -0.023 | -2.766 |
| reusable_groups | 2.00 | 0.3080 | 20 | 1.000 | 0.000 | 0.300 | -0.020 | -2.574 |
| smooth_spatial_continuum | 0.00 | 0.0000 | 20 | 1.000 | 0.000 | 0.400 | -0.026 | -1.583 |
| spatially_interdigitated_groups | 0.00 | 0.0000 | 20 | 1.000 | 0.000 | 0.600 | -0.053 | -2.160 |
| spatially_interdigitated_groups | 0.50 | 0.2292 | 20 | 1.000 | 0.000 | 0.350 | -0.027 | -2.590 |
| spatially_interdigitated_groups | 1.00 | 0.2789 | 20 | 1.000 | 0.000 | 0.500 | -0.034 | -2.404 |
| spatially_interdigitated_groups | 1.50 | 0.3212 | 20 | 1.000 | 0.000 | 0.600 | -0.061 | -2.441 |
| spatially_interdigitated_groups | 2.00 | 0.3581 | 20 | 1.000 | 0.000 | 0.300 | -0.010 | -2.552 |

## Interpretation of this smoke

The factorization repair removed the pre-repair multi-nat `H-C` failure, but
it did not establish recovery. Across the three 2x grouped fixtures, mean
`H-C` was -0.0176 nats/neuron, the predictive-gain lower-bound rate was zero,
and planted-label accuracy averaged 0.570. The one-sided 95% interval
half-width was below 0.05 in only 0.0175 of all repeats; its mean was 0.145 in
the 2x fixtures. Independent verification of the projected-oracle calibration
had maximum absolute target error 0.0476 nats/neuron and maximum Monte Carlo
SE 0.0349, so it is not sufficiently precise for registered qualification.
Repeating this same smoke with more recovery seeds would not repair either the
estimator or the fixed-design precision problem.

## Remaining blockers

- provider certification that each brain/sample unit is an independent animal
- a locked real development/audit role assignment
- the final per-cell pial-geodesic and streamline-depth common-support solver
- real-axon completeness and truncation eligibility
- marginalization of unknown, missing, censored, or truncated target coordinates
- exact conditional-nonempty maximum likelihood for support parameters
- the final hierarchical new-animal random-effect integral
- production qualification of the fixed unit-held-out nuisance/noise protocol
- equal-animal weighting inside coefficient residual-PCA and group-signature fitting
- absolute calibration and trajectory adequacy for P and C
- the full position-standardized reuse ESS overlap influence and technical-decoding gates
- smooth position-dependent group gating beyond this subset's fixed constant gate
- precision qualification of the synthetic oracle-gain calibration
- projection of each grouped fixture onto the full registered C family
- separate oracle calibration of the H-minus-D continuum amplitude
- the full registered hyperparameter and K search grammar
- development-family reference selection under equal adaptive-search budgets
- calibrated animal-level intervals for the final evaluator
- the 99 complete full-search null controllers
- deterministic replay and the global compute envelope

Passing this subset would validate the executable score direction and
precision diagnostics only. It cannot substitute for the registered evaluator,
complete null controllers, role lock, real-data eligibility, or independent-
animal certification.
