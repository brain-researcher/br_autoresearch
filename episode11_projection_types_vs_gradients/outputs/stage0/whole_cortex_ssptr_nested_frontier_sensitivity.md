# EP11 SSp-tr nested role-count frontier

This outcome-blind diagnostic enumerates all 136 additions (x,y) with
x>=0, y>=0, and x+y<=15 at 500 um / 0.10 normalized depth. For each
pair, one hidden D0=12/A0=8 baseline and its expanded D=(12+x) and
A=(8+y) roles must simultaneously satisfy the registered geometry rules.
It reads soma geometry, brain/sample-unit identity, and strain/Cre line
only; it does not read a regional projection outcome.

Each count pair is solved independently, so feasible pairs may use
different hidden allocations. The two depth conventions are also solved
independently. This report does not claim one role assignment works across
different pairs or across both depth conventions. No role identity or MILP
witness is emitted.

## Fixed rules

- The 35 non-singleton units are partitioned into D0, A0, D_add, A_add,
  and unassigned; the singleton Tle4 unit is unassigned.
- Baseline and expanded roles separately require two direct development
  neighbors per retained cell, at least 0.90 retention per assigned unit,
  at least 0.95 retention in each role overall, at least eight multi-cell
  development units, and every non-singleton line in both roles.
- The Pareto set below is coordinate-wise maximal among proved-feasible
  count pairs only. Any indeterminate row limits interpretation.

## Aggregate geometry

- Primary cells / brain-sample units: 206 / 36.
- Geometry-excluded primary cells: 0.
- Exact-versus-LineString3D normalized-depth absolute difference
  median / maximum: 0.0000 / 0.0323.

## Aggregate frontier

| Depth convention | Tested | Feasible | Infeasible | Indeterminate | Max dev-only x | Max audit-only y | Max x+y | Max assigned | Pareto pairs | Pareto additions (x,y) | Expanded counts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| exact_polyline_closest_point_primary | 136 | 105 | 31 | 0 | 13 | 13 | 13 | 33 | 14 | (0,13); (1,12); (2,11); (3,10); (4,9); (5,8); (6,7); (7,6); (8,5); (9,4); (10,3); (11,2); (12,1); (13,0) | (D12,A21); (D13,A20); (D14,A19); (D15,A18); (D16,A17); (D17,A16); (D18,A15); (D19,A14); (D20,A13); (D21,A12); (D22,A11); (D23,A10); (D24,A9); (D25,A8) |
| ccf_streamlines_LineString3D_project_sensitivity | 136 | 104 | 32 | 0 | 13 | 12 | 13 | 33 | 13 | (1,12); (2,11); (3,10); (4,9); (5,8); (6,7); (7,6); (8,5); (9,4); (10,3); (11,2); (12,1); (13,0) | (D13,A20); (D14,A19); (D15,A18); (D16,A17); (D17,A16); (D18,A15); (D19,A14); (D20,A13); (D21,A12); (D22,A11); (D23,A10); (D24,A9); (D25,A8) |

The companion CSV contains all aggregate pairwise statuses. Feasibility
does not choose or lock roles, establish animal independence, authorize
projection-outcome access, or support either groups or a continuum.
