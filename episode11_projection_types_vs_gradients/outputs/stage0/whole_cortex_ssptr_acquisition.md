# Gao whole-cortex SSp-tr acquisition

**Completed:** 2026-09-25

EP11 design v2 uses a targeted subset of the Gao whole-cortex projectome
(Zenodo record [16909126](https://zenodo.org/records/16909126)). The subset was
selected only from the provider's source-area labels: `SSp-tr1`,
`SSp-tr2/3`, `SSp-tr4`, `SSp-tr5`, `SSp-tr6a`, and the allowed but currently
empty `SSp-tr6b`. Published projection classes and regional projection
summaries were not used.

## Downloaded cohort

- 214 SWC files in 36 unique one-to-one fMOST-brain/sample-ID pairs
- 332,357,190 bytes (316.96 MiB)
- 206 `Axon_and_dendrite` reconstructions eligible for the primary analysis
  because axon and dendrite are distinguished; completeness remains a separate
  outcome-construction QC requirement
- 8 `Axon_only` reconstructions retained only for sensitivity analysis because
  axon and dendrite are not distinguished
- all 36 brain/sample-ID pairs retain at least one primary-eligible cell

The 36 pairs are sampling units for the current geometry screen. They will not
be called independent animals, nor assigned development/audit roles, until the
provider's one-brain/one-animal interpretation is confirmed.

## Validation

All selected files passed the download and SWC checks: expected cell and brain
counts, unique neuron IDs, seven-column rows, finite coordinates, one root,
valid parent references, and at least one soma and axon node. In these files,
the provider's axon tree can be encoded as SWC type `0` or `2`; type `3` is the
distinguished dendrite and types `10`--`12` are marker nodes. Primary outcome
construction must therefore accept axon types `0` and `2`, exclude type `3`,
and ignore marker nodes.

The compact inventory deliberately omits provider fields `class`, `class1`,
`class2`, and all provider-derived regional projection summaries. It contains
identity, source/layer, line, sample descriptors, soma coordinates, and file-
integrity/QC fields only.

## Locations

- Raw files: `/scratch/users/zijiao/br_autoresearch/episode11_projection_types_vs_gradients/gao_whole_cortex_ssptr_v1/swc/`
- Inventory: `outputs/stage0/whole_cortex_ssptr_inventory.csv`
- Machine-readable summary: `outputs/stage0/whole_cortex_ssptr_download_summary.json`
- Reproducible downloader: `outputs/stage0/tools/download_whole_cortex_ssptr.py`

Acquisition does not lock a source-population analysis, model, target
vocabulary, or development/audit split. The subsequent outcome-blind geometry
screen retained all 206 primary cells and found the planned 12/8 support design
feasible at 500 µm pial-geodesic distance and 0.10 normalized depth. It emitted
no role assignment. The remaining pre-outcome gates are the brain/sample-unit
interpretation, target/observation contract, precision, and simulation
qualification.
