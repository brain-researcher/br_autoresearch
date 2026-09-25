# Episode 17 inputs

This directory is reserved for durable Episode 17 input contracts and becomes
read-only after provisioning. It currently contains no neural or stimulus
payload.

The required CNeuroMod-THINGS 1.0.1 subset is stored outside this canonical
checkout. The root
[`DATA_LOCATION_MANIFEST.json`](../../DATA_LOCATION_MANIFEST.json) identifies
the canonical restricted tree as `ep17_restricted_raw` at
`/oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/restricted/cneuromod-things-1.0.1-restricted-raw`.
The CNeuroMod stimulus archive is under `private_steward_acquisition` at
`/oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/steward_acquisition`.
Both trees were relocated by same-filesystem rename on 2026-09-24 and are
`canonical_in_place`.

It includes the neural/structural subset and official release-provenance
archive described in [`../DATASETS.md`](../DATASETS.md). Operational transfer
logs, receipts, and checksum inventories remain with the canonical sources;
they are not tracked in this repository, and a code-only checkout contains no
data payload. A canonical storage location is not a verified episode handoff.

The verified raw source still mixes development, calibration, and audit rows.
It is restricted acquisition material, not a search-worker handoff. The trusted
builder must create four materialized, non-symlinked handoffs:

1. metadata only;
2. builder-only support/calibration neural data;
3. development neural data plus frozen calibration-derived artifacts; and
4. evaluator-only sealed-audit neural data.

Until those handoffs and their permissions are verified, no MAT/NIfTI neural
array may be parsed or summarized. Source-integrity checks are permitted on the
mixed raw source; scientific value access is not.

No candidate output, cache, fitted readout, neural score, or audit result may
be written under `inputs/`.
