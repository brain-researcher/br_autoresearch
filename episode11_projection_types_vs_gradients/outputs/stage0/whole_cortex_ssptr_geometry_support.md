# EP11 Gao whole-cortex SSp-tr geometry-support screen

## Scope and rule

This is an outcome-blind soma-geometry feasibility screen for the Gao
whole-cortex SSp-tr cohort. The observed provider layer labels are harmonized
to one SSp-tr source; layer is retained as an observed covariate and is
not treated as a discrete support stratum. No axon geometry, projection
class, role identity, or solver witness is emitted.
The primary track uses the 206 cells whose provider reconstruction
distinguishes axon from dendrite. This flag alone does not certify
reconstruction completeness. The eight Axon_only reconstructions enter
only the explicitly labelled 214-cell sensitivity track and never enter
primary retention or neighborhood denominators.

A feasible allocation contains 12 development and 8 audit brain/sample
units. Their interpretation as animals remains conditional on confirming
that the units are independent. Every retained cell needs direct neighbors
from at least two distinct development
brains within the stated pial-geodesic radius and normalized-depth
tolerance. Each role brain/sample unit must retain at least 90% of its
cells, each role overall must retain at least 95%, and at least eight
development
brains must contain two or more retained cells. Every non-singleton
strain/Cre line must occur in both roles; the singleton Tle4 brain is
forced unassigned. The unrestricted-depth rows are tangential-only
diagnostics.

## Input and geometry diagnostics

- Downloaded inventory: 214 cells; primary axon/dendrite-distinguished set: 206; Axon_only sensitivity additions: 8; all span 36 fMOST brains. sample-ID check: one_to_one_36_brains.
- Geometry eligible: 214/214 cells across 36 brains; minimum eligible cells per brain: 1.
- Layer cell counts: L1=7, L2/3=58, L4=34, L5=91, L6a=24.
- Strain/Cre-line brain counts: C57BL/6J=9, Cux2-Cre=3, Ntsr1-Cre_GN220=4, PlexinD1-2A-CreER=3, Rbp4-Cre_KL100=4, Rorb-IRES2-Cre-D=3, Tle4-2A-CreER=1, Tlx3-Cre_PL56=4, Wfs1-Tg2-CreERT2=5.
- Streamline offset median / 95th percentile / maximum: 4.2 / 11.1 / 19.4 um.
- Exact-versus-LineString3D normalized-depth absolute difference median / maximum: 0.0000 / 0.0323.
- Bounded pial graph nodes / undirected edges / components: 278,822 / 1,410,538 / 1.

## Allocation-existence results

| Analysis track | Depth projection | Cells | Tangential radius (um) | Depth tolerance | Cells with >=2 candidate neighboring brains | MILP status | Feasible allocation exists |
|---|---|---:|---:|---:|---:|---|---|
| primary_206_axon_dendrite_distinguished | exact_polyline_closest_point_primary | 206 | 250 | 0.10 | 184 | infeasible | no |
| primary_206_axon_dendrite_distinguished | exact_polyline_closest_point_primary | 206 | 250 | 0.20 | 196 | infeasible | no |
| primary_206_axon_dendrite_distinguished | exact_polyline_closest_point_primary | 206 | 250 | 0.30 | 203 | feasible | yes |
| primary_206_axon_dendrite_distinguished | exact_polyline_closest_point_primary | 206 | 250 | unrestricted_diagnostic | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | exact_polyline_closest_point_primary | 206 | 500 | 0.10 | 203 | feasible | yes |
| primary_206_axon_dendrite_distinguished | exact_polyline_closest_point_primary | 206 | 500 | 0.20 | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | exact_polyline_closest_point_primary | 206 | 500 | 0.30 | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | exact_polyline_closest_point_primary | 206 | 500 | unrestricted_diagnostic | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | exact_polyline_closest_point_primary | 206 | 750 | 0.10 | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | exact_polyline_closest_point_primary | 206 | 750 | 0.20 | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | exact_polyline_closest_point_primary | 206 | 750 | 0.30 | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | exact_polyline_closest_point_primary | 206 | 750 | unrestricted_diagnostic | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | exact_polyline_closest_point_primary | 206 | 1000 | 0.10 | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | exact_polyline_closest_point_primary | 206 | 1000 | 0.20 | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | exact_polyline_closest_point_primary | 206 | 1000 | 0.30 | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | exact_polyline_closest_point_primary | 206 | 1000 | unrestricted_diagnostic | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | exact_polyline_closest_point_primary | 206 | 1250 | 0.10 | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | exact_polyline_closest_point_primary | 206 | 1250 | 0.20 | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | exact_polyline_closest_point_primary | 206 | 1250 | 0.30 | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | exact_polyline_closest_point_primary | 206 | 1250 | unrestricted_diagnostic | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | exact_polyline_closest_point_primary | 206 | 1500 | 0.10 | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | exact_polyline_closest_point_primary | 206 | 1500 | 0.20 | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | exact_polyline_closest_point_primary | 206 | 1500 | 0.30 | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | exact_polyline_closest_point_primary | 206 | 1500 | unrestricted_diagnostic | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | ccf_streamlines_LineString3D_project_sensitivity | 206 | 250 | 0.10 | 184 | infeasible | no |
| primary_206_axon_dendrite_distinguished | ccf_streamlines_LineString3D_project_sensitivity | 206 | 250 | 0.20 | 196 | infeasible | no |
| primary_206_axon_dendrite_distinguished | ccf_streamlines_LineString3D_project_sensitivity | 206 | 250 | 0.30 | 203 | feasible | yes |
| primary_206_axon_dendrite_distinguished | ccf_streamlines_LineString3D_project_sensitivity | 206 | 250 | unrestricted_diagnostic | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | ccf_streamlines_LineString3D_project_sensitivity | 206 | 500 | 0.10 | 203 | feasible | yes |
| primary_206_axon_dendrite_distinguished | ccf_streamlines_LineString3D_project_sensitivity | 206 | 500 | 0.20 | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | ccf_streamlines_LineString3D_project_sensitivity | 206 | 500 | 0.30 | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | ccf_streamlines_LineString3D_project_sensitivity | 206 | 500 | unrestricted_diagnostic | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | ccf_streamlines_LineString3D_project_sensitivity | 206 | 750 | 0.10 | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | ccf_streamlines_LineString3D_project_sensitivity | 206 | 750 | 0.20 | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | ccf_streamlines_LineString3D_project_sensitivity | 206 | 750 | 0.30 | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | ccf_streamlines_LineString3D_project_sensitivity | 206 | 750 | unrestricted_diagnostic | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | ccf_streamlines_LineString3D_project_sensitivity | 206 | 1000 | 0.10 | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | ccf_streamlines_LineString3D_project_sensitivity | 206 | 1000 | 0.20 | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | ccf_streamlines_LineString3D_project_sensitivity | 206 | 1000 | 0.30 | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | ccf_streamlines_LineString3D_project_sensitivity | 206 | 1000 | unrestricted_diagnostic | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | ccf_streamlines_LineString3D_project_sensitivity | 206 | 1250 | 0.10 | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | ccf_streamlines_LineString3D_project_sensitivity | 206 | 1250 | 0.20 | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | ccf_streamlines_LineString3D_project_sensitivity | 206 | 1250 | 0.30 | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | ccf_streamlines_LineString3D_project_sensitivity | 206 | 1250 | unrestricted_diagnostic | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | ccf_streamlines_LineString3D_project_sensitivity | 206 | 1500 | 0.10 | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | ccf_streamlines_LineString3D_project_sensitivity | 206 | 1500 | 0.20 | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | ccf_streamlines_LineString3D_project_sensitivity | 206 | 1500 | 0.30 | 206 | feasible | yes |
| primary_206_axon_dendrite_distinguished | ccf_streamlines_LineString3D_project_sensitivity | 206 | 1500 | unrestricted_diagnostic | 206 | feasible | yes |
| sensitivity_214_including_8_axon_only | exact_polyline_closest_point_primary | 214 | 250 | 0.10 | 191 | infeasible | no |
| sensitivity_214_including_8_axon_only | exact_polyline_closest_point_primary | 214 | 250 | 0.20 | 204 | infeasible | no |
| sensitivity_214_including_8_axon_only | exact_polyline_closest_point_primary | 214 | 250 | 0.30 | 211 | feasible | yes |
| sensitivity_214_including_8_axon_only | exact_polyline_closest_point_primary | 214 | 250 | unrestricted_diagnostic | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | exact_polyline_closest_point_primary | 214 | 500 | 0.10 | 211 | feasible | yes |
| sensitivity_214_including_8_axon_only | exact_polyline_closest_point_primary | 214 | 500 | 0.20 | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | exact_polyline_closest_point_primary | 214 | 500 | 0.30 | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | exact_polyline_closest_point_primary | 214 | 500 | unrestricted_diagnostic | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | exact_polyline_closest_point_primary | 214 | 750 | 0.10 | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | exact_polyline_closest_point_primary | 214 | 750 | 0.20 | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | exact_polyline_closest_point_primary | 214 | 750 | 0.30 | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | exact_polyline_closest_point_primary | 214 | 750 | unrestricted_diagnostic | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | exact_polyline_closest_point_primary | 214 | 1000 | 0.10 | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | exact_polyline_closest_point_primary | 214 | 1000 | 0.20 | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | exact_polyline_closest_point_primary | 214 | 1000 | 0.30 | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | exact_polyline_closest_point_primary | 214 | 1000 | unrestricted_diagnostic | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | exact_polyline_closest_point_primary | 214 | 1250 | 0.10 | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | exact_polyline_closest_point_primary | 214 | 1250 | 0.20 | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | exact_polyline_closest_point_primary | 214 | 1250 | 0.30 | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | exact_polyline_closest_point_primary | 214 | 1250 | unrestricted_diagnostic | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | exact_polyline_closest_point_primary | 214 | 1500 | 0.10 | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | exact_polyline_closest_point_primary | 214 | 1500 | 0.20 | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | exact_polyline_closest_point_primary | 214 | 1500 | 0.30 | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | exact_polyline_closest_point_primary | 214 | 1500 | unrestricted_diagnostic | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | ccf_streamlines_LineString3D_project_sensitivity | 214 | 250 | 0.10 | 191 | infeasible | no |
| sensitivity_214_including_8_axon_only | ccf_streamlines_LineString3D_project_sensitivity | 214 | 250 | 0.20 | 204 | infeasible | no |
| sensitivity_214_including_8_axon_only | ccf_streamlines_LineString3D_project_sensitivity | 214 | 250 | 0.30 | 211 | feasible | yes |
| sensitivity_214_including_8_axon_only | ccf_streamlines_LineString3D_project_sensitivity | 214 | 250 | unrestricted_diagnostic | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | ccf_streamlines_LineString3D_project_sensitivity | 214 | 500 | 0.10 | 211 | feasible | yes |
| sensitivity_214_including_8_axon_only | ccf_streamlines_LineString3D_project_sensitivity | 214 | 500 | 0.20 | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | ccf_streamlines_LineString3D_project_sensitivity | 214 | 500 | 0.30 | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | ccf_streamlines_LineString3D_project_sensitivity | 214 | 500 | unrestricted_diagnostic | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | ccf_streamlines_LineString3D_project_sensitivity | 214 | 750 | 0.10 | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | ccf_streamlines_LineString3D_project_sensitivity | 214 | 750 | 0.20 | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | ccf_streamlines_LineString3D_project_sensitivity | 214 | 750 | 0.30 | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | ccf_streamlines_LineString3D_project_sensitivity | 214 | 750 | unrestricted_diagnostic | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | ccf_streamlines_LineString3D_project_sensitivity | 214 | 1000 | 0.10 | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | ccf_streamlines_LineString3D_project_sensitivity | 214 | 1000 | 0.20 | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | ccf_streamlines_LineString3D_project_sensitivity | 214 | 1000 | 0.30 | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | ccf_streamlines_LineString3D_project_sensitivity | 214 | 1000 | unrestricted_diagnostic | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | ccf_streamlines_LineString3D_project_sensitivity | 214 | 1250 | 0.10 | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | ccf_streamlines_LineString3D_project_sensitivity | 214 | 1250 | 0.20 | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | ccf_streamlines_LineString3D_project_sensitivity | 214 | 1250 | 0.30 | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | ccf_streamlines_LineString3D_project_sensitivity | 214 | 1250 | unrestricted_diagnostic | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | ccf_streamlines_LineString3D_project_sensitivity | 214 | 1500 | 0.10 | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | ccf_streamlines_LineString3D_project_sensitivity | 214 | 1500 | 0.20 | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | ccf_streamlines_LineString3D_project_sensitivity | 214 | 1500 | 0.30 | 214 | feasible | yes |
| sensitivity_214_including_8_axon_only | ccf_streamlines_LineString3D_project_sensitivity | 214 | 1500 | unrestricted_diagnostic | 214 | feasible | yes |

Feasibility means that at least one allocation satisfies the fixed
metadata-only support rules. It does not select or lock an allocation,
and it is not evidence for a projection group or a continuum.
