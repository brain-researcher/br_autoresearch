# EP11 historical A1876-v1 minimal data-steward handoff

**Historical v1 status:** not executed. The A1876-v1 design closed for
insufficient source support before role assignment or projection access. The
requirements below describe that historical handoff and are not the active
Gao-v2 Stage-0 checklist. Gao v2 remains blocked in Stage 0; it has not opened
real projection outcomes.

EP11 needs a trusted data-separation step because the available workbook mixes
safe identity/soma metadata with `Projection class`, and the morphology
archives have not yet been divided into development and audit animals. The
candidate analysis must not inspect those mixed-role sources directly.

The handoff should produce five things:

1. **Use the verified animal join.** Resolve workbook `fMOST Brain ID` against
   Liu et al. (2024) Image ID first and Animal ID second. Preserve the observed
   38 direct Image-ID matches and one unique 43-cell Animal-ID alias; reject any
   future unmatched, ambiguous, duplicate, or conflicting reconstruction.
2. **Shared EP09--11 role table.** Assign each whole animal to development,
   sealed audit, or excluded after checking whether it was already exposed in a
   sibling episode. Current EP09/EP10 records show no real outcome opening, but
   contain no animal-level exposure table, so every animal remains
   `exposure_unknown` until this check is completed.
3. **Source-support table.** For every candidate source, report development and
   audit animal counts, cells per animal, soma position/depth/layer coverage,
   cross-animal overlap, unsampled intermediate locations, missingness, and
   relevant experimental strata. Candidate-facing audit information must be
   aggregate only.
4. **Separated data views.** Provide development-only metadata and morphology
   files for model development, while keeping the audit files evaluator-only.
5. **Atlas and spatial-support check.** Confirm coordinate orientation, axes,
   spacing, ontology, and compatibility between the SEU coordinates and the
   Allen CCF assets. Define cortex-aware tangential and depth coordinates
   separately. If a revision retains the present strata, re-evaluate 12/8 plus
   two prospectively unassigned animals against the appropriate alternative;
   otherwise evaluate the newly versioned allocation. In either case, require
   direct same-provider-label support from at least two distinct development
   animals per retained audit cell and per development cell under
   leave-one-animal-out.

A short methods note describing these transformations is enough.

Under the historical A1876-v1 design, Stage 0 could have succeeded only if at
least one source passed the predeclared identity, cell-support,
spatial-overlap, missingness, precision, and stratum rules. The
12-development/8-audit minimum was specific to that design family and would
still have needed precision/simulation justification after source support
passed. Failure to pass meant that those data could not execute that design;
it was not evidence against projection groups and did not prevent the active
Gao-v2 redesign or future additional data.
