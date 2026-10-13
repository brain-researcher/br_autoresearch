# EP11 provisional source-by-verified-animal census

## Status

This is an outcome-blind animal-count feasibility descriptor. An aggregate-only
identity join matched 38 workbook families directly to provider Image IDs and
matched the one remaining family through its paired Animal ID alias. After that
deterministic correction, all 39 families and all 1,876 rows map one-to-one,
with zero ambiguous, unmatched, or count-mismatched families. No identifier or
alias value is emitted.

The join verifies animal grouping for this count screen. It does **not** assign
development or sealed-audit roles, assess prior exposure or frozen eligibility,
or qualify overlapping soma-position/layer support. Counts alone therefore do
not establish 12 development plus 8 usable sealed-audit animals.

## Input and boundary

- Resource: SEU-A1876 `Full_morphometry.xlsx`
- Records: 1,876; exact trimmed provider soma-region labels: 92
- Verified animal groups across the workbook: 39
- Only allow-listed grouping, CCF soma-coordinate, manual-check, soma-region, and
  cortical-lamination fields were resolved. Row identifiers were not resolved,
  and the omitted mixed-role column was skipped before data-value
  resolution. No morphology archive was opened.

## Aggregation and disclosure controls

Each CSV row is one provider soma-region label. Cells-per-verified-animal
summaries include quartiles plus coarse count bands. No raw Brain, Animal, or
Image ID, alias value, cell identifier, absolute coordinate, layer label,
row-level filename, or role appears in either artifact. CCF overlap diagnostics use 500-um bins but
release only bin counts, never bin addresses. Position spans are the 5th-to-95th
percentile span of within-animal coordinate medians and are rounded to 100 um.
Position completeness and diagnostics are suppressed below 5
coordinate-complete verified animals; layer completeness and overlap diagnostics are
suppressed below 5 layer-observed verified animals.
This aggregate-release threshold is a disclosure control, not validation of
eligibility, role balance, common support, or biological sampling adequacy.

## Headline results

- 0 of 92 source labels have at least 20 verified animals; the largest source has 17.
- 0 source labels have at least 20 verified animals with complete CCF soma coordinates.
- 0 source labels have at least 20 verified animals with a nonmissing provider layer value.
- Safe aggregate overlap diagnostics are released for 30 source labels by CCF coverage and 14 by layer coverage; smaller groups are suppressed.
- Manual-check metadata: 1,736 checked, 140 unchecked, 0 missing.

Any source below 20 verified animals cannot reach the current design's 12+8
count floor. No exact provider label passes that preliminary count screen.
Frozen eligibility, whole-animal role assignment, exposure checks, and
overlapping support remain unevaluated.
