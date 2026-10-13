# EP11 SS cortex-aware common-support screen

## Status and rule

This outcome-blind screen folds homologous hemispheres, assigns each soma to
an official Allen 10-um cortical streamline, uses shortest-path distance on
the pial-endpoint voxel surface as tangential distance, and uses pia-to-soma
path length divided by local streamline length as normalized depth. Support
requires the same exact provider label and direct neighbors from at
least two distinct development animals. No projection outcome, role identity,
or maximizing split is emitted.
Exact provider labels with fewer than five geometry-eligible animals overall
are excluded by the necessary 3-development/2-audit support rule before
role enumeration. Every fixed scored label must then have retained cells in
at least three development and two audit animals in a qualifying split.
The primary depth projection is the exact closest point on the streamline
polyline. The ccf-streamlines LineString3D endpoint-handling behavior is
evaluated as a fixed sensitivity using the same voxel-center coordinates.
Normalized-depth tolerances 0.10/0.20/0.30 are qualification candidates;
the unrestricted-depth rows are tangential-only diagnostics and cannot
qualify the final cortex-aware design.

A qualifying split must retain at least 95% of cells overall within
development and within audit, and at least eight development animals must
retain two or more cells. The table below uses the primary 90% per-animal
floor; 80%, 95%, and all-cell rows are retained in the CSV.

## Geometry diagnostics

- Valid streamline coordinates: 319/324 (0.985).
- Geometry exclusions before retention denominators: 5 cells across 3 animals; all 22 animals retain at least one eligible cell (minimum 1).
- Exact-label inventory / fixed scored scope: 9 / 8 labels and 318 cells. Labels below the necessary five-animal total: SSp.
- Geometry-eligible animal counts by exact provider label: SSp=1, SSp-bfd=9, SSp-ll=11, SSp-m=13, SSp-n=9, SSp-tr=5, SSp-ul=8, SSp-un=7, SSs=16.
- Absolute normalized-depth difference, exact closest point versus LineString3D, median / 95th percentile / maximum: 0.00000 / 0.00803 / 0.01535.
- Streamline offset median / 95th percentile / maximum: 4.6 / 11.7 / 15.5 um.
- Bounded pial graph nodes / undirected edges / components: 745,797 / 3,631,743 / 1.
- Component containing all SS endpoints: 745,797 nodes.

## Primary-floor results

| Depth convention | Role design | Tangential radius (um) | Depth tolerance | Support-only splits | Fully valid splits | Fraction | Unconstrained maximin | Unconstrained maximin with label representation |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| exact_polyline_closest_point_primary | 12_development_8_audit_2_unassigned | 250 | 0.10 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| exact_polyline_closest_point_primary | 12_development_8_audit_2_unassigned | 250 | 0.20 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| exact_polyline_closest_point_primary | 12_development_8_audit_2_unassigned | 250 | 0.30 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| exact_polyline_closest_point_primary | 12_development_8_audit_2_unassigned | 250 | unrestricted_diagnostic | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| exact_polyline_closest_point_primary | 12_development_8_audit_2_unassigned | 500 | 0.10 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| exact_polyline_closest_point_primary | 12_development_8_audit_2_unassigned | 500 | 0.20 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| exact_polyline_closest_point_primary | 12_development_8_audit_2_unassigned | 500 | 0.30 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| exact_polyline_closest_point_primary | 12_development_8_audit_2_unassigned | 500 | unrestricted_diagnostic | 0 | 0 | 0.00000000 | 0.500 | 0.000 |
| exact_polyline_closest_point_primary | 12_development_8_audit_2_unassigned | 750 | 0.10 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| exact_polyline_closest_point_primary | 12_development_8_audit_2_unassigned | 750 | 0.20 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| exact_polyline_closest_point_primary | 12_development_8_audit_2_unassigned | 750 | 0.30 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| exact_polyline_closest_point_primary | 12_development_8_audit_2_unassigned | 750 | unrestricted_diagnostic | 0 | 0 | 0.00000000 | 0.500 | 0.000 |
| exact_polyline_closest_point_primary | 12_development_8_audit_2_unassigned | 1000 | 0.10 | 0 | 0 | 0.00000000 | 0.419 | 0.000 |
| exact_polyline_closest_point_primary | 12_development_8_audit_2_unassigned | 1000 | 0.20 | 0 | 0 | 0.00000000 | 0.500 | 0.000 |
| exact_polyline_closest_point_primary | 12_development_8_audit_2_unassigned | 1000 | 0.30 | 0 | 0 | 0.00000000 | 0.750 | 0.000 |
| exact_polyline_closest_point_primary | 12_development_8_audit_2_unassigned | 1000 | unrestricted_diagnostic | 53,880 | 0 | 0.00000000 | 0.955 | 0.000 |
| exact_polyline_closest_point_primary | 12_development_8_audit_2_unassigned | 1250 | 0.10 | 0 | 0 | 0.00000000 | 0.500 | 0.000 |
| exact_polyline_closest_point_primary | 12_development_8_audit_2_unassigned | 1250 | 0.20 | 0 | 0 | 0.00000000 | 0.750 | 0.000 |
| exact_polyline_closest_point_primary | 12_development_8_audit_2_unassigned | 1250 | 0.30 | 0 | 0 | 0.00000000 | 0.778 | 0.000 |
| exact_polyline_closest_point_primary | 12_development_8_audit_2_unassigned | 1250 | unrestricted_diagnostic | 790,895 | 0 | 0.00000000 | 0.971 | 0.000 |
| exact_polyline_closest_point_primary | 12_development_8_audit_2_unassigned | 1500 | 0.10 | 0 | 0 | 0.00000000 | 0.500 | 0.000 |
| exact_polyline_closest_point_primary | 12_development_8_audit_2_unassigned | 1500 | 0.20 | 0 | 0 | 0.00000000 | 0.750 | 0.000 |
| exact_polyline_closest_point_primary | 12_development_8_audit_2_unassigned | 1500 | 0.30 | 0 | 0 | 0.00000000 | 0.778 | 0.000 |
| exact_polyline_closest_point_primary | 12_development_8_audit_2_unassigned | 1500 | unrestricted_diagnostic | 805,977 | 0 | 0.00000000 | 0.971 | 0.000 |
| exact_polyline_closest_point_primary | 14_development_8_audit_all_animals | 250 | 0.10 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| exact_polyline_closest_point_primary | 14_development_8_audit_all_animals | 250 | 0.20 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| exact_polyline_closest_point_primary | 14_development_8_audit_all_animals | 250 | 0.30 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| exact_polyline_closest_point_primary | 14_development_8_audit_all_animals | 250 | unrestricted_diagnostic | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| exact_polyline_closest_point_primary | 14_development_8_audit_all_animals | 500 | 0.10 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| exact_polyline_closest_point_primary | 14_development_8_audit_all_animals | 500 | 0.20 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| exact_polyline_closest_point_primary | 14_development_8_audit_all_animals | 500 | 0.30 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| exact_polyline_closest_point_primary | 14_development_8_audit_all_animals | 500 | unrestricted_diagnostic | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| exact_polyline_closest_point_primary | 14_development_8_audit_all_animals | 750 | 0.10 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| exact_polyline_closest_point_primary | 14_development_8_audit_all_animals | 750 | 0.20 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| exact_polyline_closest_point_primary | 14_development_8_audit_all_animals | 750 | 0.30 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| exact_polyline_closest_point_primary | 14_development_8_audit_all_animals | 750 | unrestricted_diagnostic | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| exact_polyline_closest_point_primary | 14_development_8_audit_all_animals | 1000 | 0.10 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| exact_polyline_closest_point_primary | 14_development_8_audit_all_animals | 1000 | 0.20 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| exact_polyline_closest_point_primary | 14_development_8_audit_all_animals | 1000 | 0.30 | 0 | 0 | 0.00000000 | 0.500 | 0.000 |
| exact_polyline_closest_point_primary | 14_development_8_audit_all_animals | 1000 | unrestricted_diagnostic | 0 | 0 | 0.00000000 | 0.750 | 0.000 |
| exact_polyline_closest_point_primary | 14_development_8_audit_all_animals | 1250 | 0.10 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| exact_polyline_closest_point_primary | 14_development_8_audit_all_animals | 1250 | 0.20 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| exact_polyline_closest_point_primary | 14_development_8_audit_all_animals | 1250 | 0.30 | 0 | 0 | 0.00000000 | 0.500 | 0.000 |
| exact_polyline_closest_point_primary | 14_development_8_audit_all_animals | 1250 | unrestricted_diagnostic | 0 | 0 | 0.00000000 | 0.750 | 0.000 |
| exact_polyline_closest_point_primary | 14_development_8_audit_all_animals | 1500 | 0.10 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| exact_polyline_closest_point_primary | 14_development_8_audit_all_animals | 1500 | 0.20 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| exact_polyline_closest_point_primary | 14_development_8_audit_all_animals | 1500 | 0.30 | 0 | 0 | 0.00000000 | 0.500 | 0.000 |
| exact_polyline_closest_point_primary | 14_development_8_audit_all_animals | 1500 | unrestricted_diagnostic | 0 | 0 | 0.00000000 | 0.750 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 12_development_8_audit_2_unassigned | 250 | 0.10 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 12_development_8_audit_2_unassigned | 250 | 0.20 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 12_development_8_audit_2_unassigned | 250 | 0.30 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 12_development_8_audit_2_unassigned | 250 | unrestricted_diagnostic | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 12_development_8_audit_2_unassigned | 500 | 0.10 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 12_development_8_audit_2_unassigned | 500 | 0.20 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 12_development_8_audit_2_unassigned | 500 | 0.30 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 12_development_8_audit_2_unassigned | 500 | unrestricted_diagnostic | 0 | 0 | 0.00000000 | 0.500 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 12_development_8_audit_2_unassigned | 750 | 0.10 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 12_development_8_audit_2_unassigned | 750 | 0.20 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 12_development_8_audit_2_unassigned | 750 | 0.30 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 12_development_8_audit_2_unassigned | 750 | unrestricted_diagnostic | 0 | 0 | 0.00000000 | 0.500 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 12_development_8_audit_2_unassigned | 1000 | 0.10 | 0 | 0 | 0.00000000 | 0.419 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 12_development_8_audit_2_unassigned | 1000 | 0.20 | 0 | 0 | 0.00000000 | 0.500 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 12_development_8_audit_2_unassigned | 1000 | 0.30 | 0 | 0 | 0.00000000 | 0.750 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 12_development_8_audit_2_unassigned | 1000 | unrestricted_diagnostic | 53,880 | 0 | 0.00000000 | 0.955 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 12_development_8_audit_2_unassigned | 1250 | 0.10 | 0 | 0 | 0.00000000 | 0.500 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 12_development_8_audit_2_unassigned | 1250 | 0.20 | 0 | 0 | 0.00000000 | 0.750 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 12_development_8_audit_2_unassigned | 1250 | 0.30 | 0 | 0 | 0.00000000 | 0.778 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 12_development_8_audit_2_unassigned | 1250 | unrestricted_diagnostic | 790,895 | 0 | 0.00000000 | 0.971 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 12_development_8_audit_2_unassigned | 1500 | 0.10 | 0 | 0 | 0.00000000 | 0.500 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 12_development_8_audit_2_unassigned | 1500 | 0.20 | 0 | 0 | 0.00000000 | 0.750 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 12_development_8_audit_2_unassigned | 1500 | 0.30 | 0 | 0 | 0.00000000 | 0.778 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 12_development_8_audit_2_unassigned | 1500 | unrestricted_diagnostic | 805,977 | 0 | 0.00000000 | 0.971 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 14_development_8_audit_all_animals | 250 | 0.10 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 14_development_8_audit_all_animals | 250 | 0.20 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 14_development_8_audit_all_animals | 250 | 0.30 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 14_development_8_audit_all_animals | 250 | unrestricted_diagnostic | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 14_development_8_audit_all_animals | 500 | 0.10 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 14_development_8_audit_all_animals | 500 | 0.20 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 14_development_8_audit_all_animals | 500 | 0.30 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 14_development_8_audit_all_animals | 500 | unrestricted_diagnostic | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 14_development_8_audit_all_animals | 750 | 0.10 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 14_development_8_audit_all_animals | 750 | 0.20 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 14_development_8_audit_all_animals | 750 | 0.30 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 14_development_8_audit_all_animals | 750 | unrestricted_diagnostic | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 14_development_8_audit_all_animals | 1000 | 0.10 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 14_development_8_audit_all_animals | 1000 | 0.20 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 14_development_8_audit_all_animals | 1000 | 0.30 | 0 | 0 | 0.00000000 | 0.500 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 14_development_8_audit_all_animals | 1000 | unrestricted_diagnostic | 0 | 0 | 0.00000000 | 0.750 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 14_development_8_audit_all_animals | 1250 | 0.10 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 14_development_8_audit_all_animals | 1250 | 0.20 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 14_development_8_audit_all_animals | 1250 | 0.30 | 0 | 0 | 0.00000000 | 0.500 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 14_development_8_audit_all_animals | 1250 | unrestricted_diagnostic | 0 | 0 | 0.00000000 | 0.750 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 14_development_8_audit_all_animals | 1500 | 0.10 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 14_development_8_audit_all_animals | 1500 | 0.20 | 0 | 0 | 0.00000000 | 0.000 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 14_development_8_audit_all_animals | 1500 | 0.30 | 0 | 0 | 0.00000000 | 0.500 | 0.000 |
| ccf_streamlines_LineString3D_project_sensitivity | 14_development_8_audit_all_animals | 1500 | unrestricted_diagnostic | 0 | 0 | 0.00000000 | 0.750 | 0.000 |

The exact-provider-label development/audit representation requirement is included
here because it is outcome-blind. Effective group-membership requirements
remain later claim gates. This artifact emits no role assignment, projection
group, or biological result.

## Qualification decision

The current eight-stratum, 22-animal design has no qualifying role split. In
the cortex-aware primary grid, support-only split counts are zero even at the
0.80 sensitivity floor for both designs. The best unconstrained primary
per-animal maximin is 0.778 for 12 development / 8 audit / 2 unassigned and
0.500 for forced 14/8. Fully valid counts remain zero under the exact-polyline
primary depth, the `LineString3D` sensitivity, and tangential-only diagnostics.

`SSp-tr` is a necessary-condition bottleneck: its five singleton animals have
no 3-development/2-audit assignment that gives every role the required direct
support, even at 1,500 um with depth unrestricted. The current design is
therefore `closed_insufficient_source_support`. This is not evidence for a
continuum or against reusable projection groups; no projection outcome was
opened.
