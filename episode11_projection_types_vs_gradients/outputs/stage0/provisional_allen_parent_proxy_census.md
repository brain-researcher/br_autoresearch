# EP11 provisional Allen-parent verified-animal census

## Status

This is an outcome-blind ontology roll-up for diagnosing label granularity. It
does not select a final source. An aggregate-only identity join verifies a
one-to-one animal grouping for all 39 workbook families and 1,876 rows: 38
families match provider Image IDs directly, and one uses its paired Animal ID
alias. There are zero ambiguous, unmatched, or count-mismatched families, and
no identifier or alias value is emitted.

The verified grouping supports animal-count aggregation only. These results do
**not** assign development or sealed-audit roles or qualify frozen eligibility,
prior exposure, or common spatial support.

## Inputs and mapping

- Metadata resource: SEU-A1876 `Full_morphometry.xlsx`.
- Ontology: Allen adult mouse structure graph 1, acquired 2026-09-22 from
  `https://api.brain-map.org/api/v2/structure_graph_download/1.json`.
- Identity join: 39 of 39 workbook families map one-to-one to animal groups;
  zero ambiguous, unmatched, or count-mismatched families.
- Exact mappings: 92 of 92 provider labels
  (92 exact acronyms; 0 exact names). No case folding,
  fuzzy matching, coordinate matching, count-driven merging, or manual override
  was used.
- Unmapped labels (0): none.
- Ambiguous labels (0): none.

The local Allen acquisition is still marked as source-only and not a validated
episode-ready atlas binding. Here it is used only for deterministic graph
ancestry. The coordinate summaries remain provisional and do not validate
orientation, transform compatibility, or a final source definition.

## Parent construction and disclosure controls

The CSV contains 54 canonical Allen nodes that are proper graph
ancestors of at least one provider label, spanning Allen structure levels
0, 1, 2, 3, 4, 5, 6, 7, 8. Each population includes every mapped provider
label at or below that node; same-node cells are included when present. Unique
verified-animal counts are recomputed from the internal union and are never obtained by
summing children. The smallest qualifying flag means no stricter descendant
parent row also reaches 20 verified animals.

No raw Brain, Animal, or Image ID, alias value, cell identifier, row coordinate,
absolute location, layer value, or outcome-derived field is emitted. Spatial summaries use 500-um CCF
bins but expose only non-additive aggregate counts/fractions, never bin
addresses. Robust within-animal centroid spans are rounded to 100 um. All spatial
fields are blank below 5 coordinate-complete verified animals;
coordinate-completeness counts are not released, preventing complementary
subtraction across hierarchy rows. No morphology archive was opened.

## Results

- 9 of 54 ontology parent populations reach the 20-verified-animal count screen.
- 2 are minimal in the Allen ancestor partial order.

The minimal qualifying parents are:

- `SS` (Somatosensory areas): 22 verified animals, 324 cells, 9 included provider labels.
- `STR` (Striatum): 21 verified animals, 355 cells, 7 included provider labels.

These parent populations are anatomically defined roll-ups, not inferred cell
types, eligibility sets, or audit roles. Reaching 20 verified animals only
distinguishes an overly fine exact-label partition from an immediate count
failure. It does not select either parent as a final source or establish role
balance, common support, or feasibility of the current EP11 design.
