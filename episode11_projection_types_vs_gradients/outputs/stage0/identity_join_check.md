# EP11 identity join check

**Result:** pass after one deterministic ID-column correction. The raw workbook
`fMOST Brain ID` set is not exactly equal to the supplementary `Image ID` set.

This aggregate-only check used only `Morphology Name` and `fMOST Brain ID` from
the SEU-A1876 workbook, and `Animal ID`, `Image ID`, and `#full-morphology` from
Liu et al. (2024) Supplementary Data 1. No `Projection class`, axon field, or
projection outcome was loaded or inspected.

| Check | Result |
| --- | ---: |
| Workbook rows | 1,876 |
| Unique morphology names | 1,876 |
| Missing or duplicate morphology names | 0 |
| Unique workbook brain IDs | 39 |
| Missing workbook brain IDs | 0 |
| Supplement rows with full morphologies | 39 |
| Unique supplement Image IDs | 39 |
| Unique supplement Animal IDs | 39 |
| Image-to-animal mapping | one-to-one |
| Supplement full-morphology total | 1,876 |

## Strict and resolved joins

Strict equality on workbook `fMOST Brain ID == Image ID` matches 38 families.
It leaves one workbook-only ID, one supplement-only Image ID, and 43 unmatched
workbook rows. Counts agree for every directly matched family.

The single workbook-only value is the official `Animal ID` paired with the
single missing `Image ID` in the supplement; both sides contain 43
morphologies. Resolving an ID first as an `Image ID`, and otherwise as a unique
supplement `Animal ID`, gives:

- 39 workbook families mapped one-to-one to 39 `(Animal ID, Image ID)` pairs;
- zero ambiguous or unresolved families;
- zero unmatched workbook rows;
- zero workbook-only or supplement-only image families;
- zero per-family count mismatches; and
- an exact total of 1,876 morphologies on both sides.

EP11 may therefore use the joined `(Animal ID, Image ID)` pair as its animal
key. It must not equate raw `fMOST Brain ID` with `Image ID` without applying
the one documented Animal-ID alias. This check establishes identity grouping
only and carries no projection result.
