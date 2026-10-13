# EP11 SSp-tr nested role-design geometry sensitivity

This is an outcome-blind feasibility diagnostic on the 206 primary
axon/dendrite-distinguished reconstructions. It uses only soma geometry,
brain/sample-unit identity, and strain/Cre line. No regional projection
outcome, role identity, or MILP witness is emitted.

The common core is D0=12 and A0=8. The remaining 15 non-singleton
units form R. The joint core system requires the baseline D12/A8, the
estimator ceiling (D0+R)/A0 = D27/A8, and the precision ceiling
D0/(A0+R) = D12/A23 to pass simultaneously. The preferred compromise
adds D16/A19 with D0 contained in D16, A0 contained in A19, and exactly
two D16 units in each of the eight non-singleton lines. The singleton
Tle4 unit is unassigned throughout.

Every design separately enforces direct support from two development
units, at least 0.90 retention in every assigned role unit, at least
0.95 retention in each role overall, at least eight multi-cell
development units, and non-singleton-line representation in both roles.
The two depth conventions are solved separately. Passing both is not
yet proof that one common hidden allocation satisfies both conventions;
that stronger robustness check would require one cross-convention MILP.

## Aggregate geometry

- Primary cells / brain-sample units: 206 / 36.
- Geometry-excluded primary cells: 0.
- Exact-versus-LineString3D normalized-depth absolute difference
  median / maximum: 0.0000 / 0.0323.

## Aggregate feasibility

| Depth projection | Joint system | Cells with >=2 candidate neighbor units | Binary variables | Constraints | Solver status | Feasible |
|---|---|---:|---:|---:|---|---|
| exact_polyline_closest_point_primary | core_plus_estimator_and_precision | 203 | 1344 | 2806 | infeasible | no |
| exact_polyline_closest_point_primary | core_plus_estimator_precision_and_exact2_compromise | 203 | 1828 | 3847 | infeasible | no |
| ccf_streamlines_LineString3D_project_sensitivity | core_plus_estimator_and_precision | 203 | 1344 | 2806 | infeasible | no |
| ccf_streamlines_LineString3D_project_sensitivity | core_plus_estimator_precision_and_exact2_compromise | 203 | 1828 | 3847 | infeasible | no |

Feasibility establishes only that a nested allocation exists under
the fixed outcome-blind support rule. It does not choose or lock role
identities, establish animal independence, authorize projection-outcome
access, or support either groups or a continuum.
