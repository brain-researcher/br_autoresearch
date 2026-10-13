# EP11 SS two-distinct-development-neighbor support screen

## Status and fixed rule

This is an outcome-blind, soma-only aggregate sensitivity screen for SS. It
does not select roles, identify prospective exclusions, or choose a final
design. All 324 eligible cells from 22 verified animals and 9 exact provider
labels are retained at the source-data stage.

A cell counts as supported only when cells from at least **two distinct
development animals** share its exact provider label and fall within the
inclusive radius. For a development cell, those must be two other development
animals; same-animal cells never contribute, and audit animals do not support
development leave-one-animal-out retention. For an audit cell, both supporting
animals must be in development. One supporting cell may serve multiple cells.

Distances are three-dimensional Euclidean distances in the provider CCF fields.
This geometry is explicitly **provisional pending tangential/depth validation**;
the screen does not establish transform or laminar-coordinate validity.

The two exhaustive role designs are:

- 12 development / 8 audit / 2 prospectively unused: 29,099,070 whole-animal
  labeled-role splits. The two unused animals vary by hypothetical split; no
  membership or maximizing witness is emitted.
- 14 development / 8 audit, all 22 retained: 319,770 whole-animal splits.

For each split, supported-cell retention is computed separately for every
development and audit animal. The split score is the minimum retention across
all role animals. Valid-split counts use fixed prospective floors 0.80, 0.90,
0.95, and 1.00 (all cells). The maximin is the largest split score over the
complete role-split universe. Counts and fractions are aggregate and exact
apart from displayed decimal rounding. Floor decisions use exact integer
cross-multiplication; their counts and existence flags are authoritative if a
displayed, rounded maximin lies near a floor.

## Results

First radius with at least one qualifying role split:

- 12/8+2 unused: >=0.80: 1250 um; >=0.90: 1500 um; >=0.95: none; all-cell: none.
- 14/8 all-retained: >=0.80: 1500 um; >=0.90: none; >=0.95: none; all-cell: none.

| Role design | Radius (um) | Best both-role maximin | Valid split counts by floor |
|---|---:|---:|---|
| 12/8+2 unused | 500 | 0.143 | >=0.80=0, >=0.90=0, >=0.95=0, all-cell=0 |
| 12/8+2 unused | 750 | 0.286 | >=0.80=0, >=0.90=0, >=0.95=0, all-cell=0 |
| 12/8+2 unused | 1000 | 0.500 | >=0.80=0, >=0.90=0, >=0.95=0, all-cell=0 |
| 12/8+2 unused | 1250 | 0.800 | >=0.80=16,257, >=0.90=0, >=0.95=0, all-cell=0 |
| 12/8+2 unused | 1500 | 0.943 | >=0.80=1,072,616, >=0.90=13,827, >=0.95=0, all-cell=0 |
| 14/8 all-retained | 500 | 0.000 | >=0.80=0, >=0.90=0, >=0.95=0, all-cell=0 |
| 14/8 all-retained | 750 | 0.000 | >=0.80=0, >=0.90=0, >=0.95=0, all-cell=0 |
| 14/8 all-retained | 1000 | 0.000 | >=0.80=0, >=0.90=0, >=0.95=0, all-cell=0 |
| 14/8 all-retained | 1250 | 0.500 | >=0.80=0, >=0.90=0, >=0.95=0, all-cell=0 |
| 14/8 all-retained | 1500 | 0.857 | >=0.80=27,870, >=0.90=0, >=0.95=0, all-cell=0 |

The two-neighbor rule is stronger than the earlier one-neighbor screen, so its
retention values are not direct reproductions of the earlier 0.971 versus 0.714
comparison. The CSV contains the full 40-row design/radius/floor grid with
existence flags and split fractions. These are provisional geometry-support
diagnostics only, not common-support qualification, a role assignment, an
exclusion decision, or audit access.
