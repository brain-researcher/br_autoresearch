# EP11 candidate common-support screen for SS and STR

## Status and frozen scope

This is a soma-only, outcome-blind aggregate feasibility screen for the two
ontology parents that passed the verified-animal count screen. It neither
selects a final source nor assigns development/audit roles. No real audit is
opened. The fixed grid is two support scopes, radii 250/500/750/1000/1500 um,
and minimum source-total cell counts per animal of 1/2/3/5/10. The 12/8
minimum-floor design is retained throughout that grid; at the one-cell
threshold only, an all-retained 14/8 SS design and 13/8 STR design are also
evaluated.

All 679 included cells
have complete provider CCFv3 soma coordinates. Distance is three-dimensional
Euclidean distance in the provider's 1-um CCF coordinate fields, with
`distance <= radius`. The local CCF binding remains provisional; this screen
does not validate orientation or transform compatibility. No morphology
archive or outcome-derived field was accessed.

## Support and partition definitions

- `parent_wide`: an audit cell may be supported by any development cell in the
  same SS or STR parent, regardless of exact provider label.
- `same_exact_provider_label`: the supporting development cell must have
  the same exact provider soma-region label. Broad same-node labels remain
  their own labels; no manual regrouping is used.
- Same-animal cells never support one another. One development cell may support
  multiple audit cells. The minimum-cells threshold filters animals by their
  total cells in the parent; it is not a required neighbor count.
- Animal-level partition criterion: every one of 8 audit animals has at least
  one cell supported by at least one development animal.
- Cell-level partition criterion: every cell in every audit animal is supported
  by at least one development animal. Support is evaluated audit-to-development;
  reverse development-to-audit support is not required.
- Development leave-one-animal-out criteria are reported separately. At the
  animal level, every development animal must have at least one cell supported
  by another development animal. At the cell level, all of its cells must be
  supported by another development animal. Thus the support pool has 11 other
  animals in the 12/8 floor and 13 SS or 12 STR animals in the all-retained
  design. Audit animals never support development LOO. Joint counts require
  the matched audit and development criteria simultaneously.
- The best-achievable minimum support is
  `max_partition min_audit_animal(supported_cells / that_animal_cells)`.
  A second maximin summary takes the minimum over both audit animals and
  leave-one-animal-out development animals.

At the one-cell threshold, every unordered whole-animal assignment is evaluated.
The 12/8 minimum-floor universes contain 29,099,070 labeled-role partitions for
SS and 2,645,370 for STR; 2 SS animals and 1 STR animal are unused in each such
hypothetical partition, not declared excluded. The all-retained universes contain
319,770 SS 14/8 splits and 203,490 STR 13/8 splits, with no unused animal.
No witness, membership, or maximizing assignment is saved or emitted.
Source-specific existence does not imply a compatible shared EP09--11 role table.

Aggregate counts and fractions are reported directly. Distance quantiles are
reported only with at least 5 finite distances and are rounded to 10 um;
optimized worst-support fractions are rounded to 0.001.

## Results

- **SS (Somatosensory areas).** 324 cells from 22 verified animals; retained animals at minimum 1/2/3/5/10 cells are 22/18/13/11/7. At the one-cell threshold, the 12/8 floor's first parent-wide animal-level/cell-level feasible radii are 250 um / 500 um; the same-label radii are 250 um / 750 um.
  For that 12/8 floor, with development leave-one-animal-out support also required, the first parent-wide animal-level/cell-level radii are 500 um / 1000 um; the same-label radii are 500 um / 1500 um.
  In the all-retained 14/8 split, the first audit-only parent-wide animal-level/cell-level radii are 250 um / 500 um; the same-label radii are 250 um / 750 um.
  In the all-retained 14/8 split, with development LOO also required, the first parent-wide animal-level/cell-level radii are 500 um / none through 1500 um; the same-label radii are 1000 um / none through 1500 um.
- **STR (Striatum).** 355 cells from 21 verified animals; retained animals at minimum 1/2/3/5/10 cells are 21/17/14/13/8. At the one-cell threshold, the 12/8 floor's first parent-wide animal-level/cell-level feasible radii are 250 um / 750 um; the same-label radii are 250 um / 750 um.
  For that 12/8 floor, with development leave-one-animal-out support also required, the first parent-wide animal-level/cell-level radii are 1000 um / 1500 um; the same-label radii are none through 1500 um / none through 1500 um.
  In the all-retained 13/8 split, the first audit-only parent-wide animal-level/cell-level radii are 250 um / 750 um; the same-label radii are 250 um / 750 um.
  In the all-retained 13/8 split, with development LOO also required, the first parent-wide animal-level/cell-level radii are 1000 um / 1500 um; the same-label radii are none through 1500 um / none through 1500 um.

With a rule requiring at least 2 cells per animal, SS retains 18 animals and
STR retains 17. Neither can form a 12-development/8-audit partition before any
spatial criterion is applied. Thresholds 3, 5, and 10 are therefore also
count-infeasible.

Parent-wide versus exact-label feasibility differences:

- SS, 12/8 floor, audit-only animal-level: parent-wide-only feasibility at no tested radius.
- SS, 12/8 floor, audit-only cell-level: parent-wide-only feasibility at 500 um.
- SS, 12/8 floor, development-LOO plus audit animal-level: parent-wide-only feasibility at no tested radius.
- SS, 12/8 floor, development-LOO plus audit cell-level: parent-wide-only feasibility at 1000 um.
- SS, all-retained 14/8, audit-only animal-level: parent-wide-only feasibility at no tested radius.
- SS, all-retained 14/8, audit-only cell-level: parent-wide-only feasibility at 500 um.
- SS, all-retained 14/8, development-LOO plus audit animal-level: parent-wide-only feasibility at 500, 750 um.
- SS, all-retained 14/8, development-LOO plus audit cell-level: parent-wide-only feasibility at no tested radius.
- STR, 12/8 floor, audit-only animal-level: parent-wide-only feasibility at no tested radius.
- STR, 12/8 floor, audit-only cell-level: parent-wide-only feasibility at no tested radius.
- STR, 12/8 floor, development-LOO plus audit animal-level: parent-wide-only feasibility at 1000, 1500 um.
- STR, 12/8 floor, development-LOO plus audit cell-level: parent-wide-only feasibility at 1500 um.
- STR, all-retained 13/8, audit-only animal-level: parent-wide-only feasibility at no tested radius.
- STR, all-retained 13/8, audit-only cell-level: parent-wide-only feasibility at no tested radius.
- STR, all-retained 13/8, development-LOO plus audit animal-level: parent-wide-only feasibility at 1000, 1500 um.
- STR, all-retained 13/8, development-LOO plus audit cell-level: parent-wide-only feasibility at 1500 um.

The CSV contains the full fixed 100-row aggregate grid, including cells/animal
and provider-label coverage distributions, nearest-other-animal distance summaries,
radius-wise cell/animal neighbor coverage, exhaustive 12/8 floor and
all-retained partition existence/counts/fractions, development
leave-one-animal-out filters, and the rounded best-achievable minimum audit-only
and both-role support. These are count/support diagnostics only, not full
common-support qualification, eligibility, a final-source decision, or a role
assignment.
