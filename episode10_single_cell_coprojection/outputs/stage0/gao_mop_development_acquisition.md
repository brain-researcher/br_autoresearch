# Gao MOp development-only acquisition

The frozen EP10 role ledger contains 209 primary `Axon_and_dendrite` rows from
12 development fMOST/sample units. The acquisition and constructor manifest is
the exact 200-row frozen common-support subset. It is derived only from the
frozen role ledger, outcome-blind geometry inventory, and emitted 209-row
common-support ledger. It contains no final or unassigned row and no unit
shared with EP11.

The other nine development rows are absent from the acquisition manifest and
must not be downloaded or opened. Structural QC must retain at least 190 of
the 200 acquired common-support cells and
the frozen per-group minimum in all 12 groups, or the endpoint is revised or
stopped without replacement. The four release-wide `Axon_only` cells have no
development representation and are not part of this acquisition or a viable
sensitivity.

Do not run the older transient `download_mop_swcs_on_dtn.sh`: it consumes the
all-837-file MOp list and would cross the frozen role boundary.

Before any normal downloader run, freeze the EP10 target-observation and
structural-QC contract. A normal run reads complete development SWC bodies.
Final and unassigned files must remain absent.

## Common-support provenance

The authoritative outcome-blind support input is the 209-row
`gao_mop_development_assignment_support.csv`. It records all frozen primary
development assignments, including 200 `primary_common_support=yes` /
`primary_model_eligible=yes` rows and nine `no` rows under
`exact_polyline_radius_500um_depth_tolerance_0.20`. Its per-group supported
counts reconcile exactly to `gao_mop_frozen_roles.csv`.

The nine unsupported provider IDs are `2001`, `3291`, `4201`, `5531`, `9987`,
`10544`, `10545`, `17288`, and `17307`; their SWCs are absent from the
acquisition manifest. An independent targeted confirmation on the login node
completed in 5.1 seconds. It used only the safe soma inventory plus selected
pre-existing Allen geometry HDF5 lookups: exact streamline depth and an exact
26-neighbor pial-surface graph bounded around the three geometry-ambiguous
development groups (97,328 nodes, 382,896 undirected edges, 74 cells). It read
no SWC row or projection outcome, made no network request, and submitted no
job.

The reproducible full outcome-blind derivation is
`tools/derive_mop_development_common_support.py`; its explicit-time
`tools/derive_mop_development_common_support.sbatch` wrapper requests 30
minutes, two CPUs, and 8 GiB. The wrapper has not been submitted. The
dependency-free preflight below checks the 837-row safe inventory and 209-row
development assignment boundary without opening geometry HDF5 or writing an
output:

```bash
python3 /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/outputs/stage0/tools/derive_mop_development_common_support.py \
  --roles /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/outputs/stage0/gao_mop_frozen_roles.csv \
  --inventory /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/outputs/stage0/gao_mop_geometry_inventory.csv \
  --output /scratch/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/gao_mop_common_support_v1/gao_mop_development_assignment_support.csv \
  --validate-only
```

The durable manifest can be regenerated deterministically without network
access:

```bash
ml devel
ml python/3.12.1
python3 /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/outputs/stage0/tools/build_mop_development_manifest.py \
  --roles /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/outputs/stage0/gao_mop_frozen_roles.csv \
  --inventory /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/outputs/stage0/gao_mop_geometry_inventory.csv \
  --common-support /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/outputs/stage0/gao_mop_development_assignment_support.csv \
  --output /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/outputs/stage0/gao_mop_development_manifest.csv
```

From an attended Sherlock session, connect to the data-transfer node and run:

```bash
ssh dtn.sherlock.stanford.edu
ml devel
ml python/3.12.1
python3 /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/outputs/stage0/tools/download_mop_development_swcs.py \
  --manifest /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/outputs/stage0/gao_mop_development_manifest.csv \
  --roles /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/outputs/stage0/gao_mop_frozen_roles.csv \
  --inventory /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/outputs/stage0/gao_mop_geometry_inventory.csv \
  --common-support /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/outputs/stage0/gao_mop_development_assignment_support.csv \
  --output-root /scratch/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/gao_mop_development_v1 \
  --workers 8 \
  --execute-development-transfer
```

For a no-network check on any Sherlock host, use `--validate-only` instead of
`--execute-development-transfer`. That mode does not create the destination.
The downloader itself recomputes the exact allowed rows through the shared
builder from the role, inventory, and common-support inputs and compares every
manifest row and field. It refuses a normal run outside a DTN, refuses any
manifest row not explicitly marked frozen
development/non-shared/primary/common-support/model-eligible/unexposed, writes
only below `$SCRATCH` or `$GROUP_SCRATCH`, and rejects a destination containing
any unexpected SWC.
It records provider `Content-Length` and actual byte counts in the plain
`download_log.csv` acquisition log.
