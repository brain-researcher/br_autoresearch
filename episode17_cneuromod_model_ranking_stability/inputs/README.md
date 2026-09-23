# Episode 17 inputs

This directory contains durable Episode 17 inputs and becomes read-only after
provisioning.

The required CNeuroMod-THINGS 1.0.1 subset is already provisioned in
steward-managed, read-only OAK storage. Its exact location is an operational
record outside Git and is disclosed only to the trusted builder.

It includes the neural/structural subset and official release-provenance
archive described in [`../DATASETS.md`](../DATASETS.md). Source integrity and
read-only permissions were verified during provisioning. Operational transfer
logs, receipts, and checksum inventories remain with that steward workspace;
they are not tracked in this repository, and a code-only checkout contains no
data payload.

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
