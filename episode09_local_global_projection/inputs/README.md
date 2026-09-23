# Read-only inputs

This directory is an instruction-level read-only boundary. Do not place
generated files, logs, mutable caches, or edited source data here.

No scientific input is provisioned at contract-preparation time. Before an
authorized launch, human-approved immutable references must expose the exact
SEU-A1876 RAW and CCFv3 morphology payloads, Allen CCF assets, brain/calibration
metadata, and provider-method provenance specified in `../DATASETS.md`.

Record source identity, version, bytes, local SHA-256, access time, license,
and provisioning method for every input. Never consume a mutable sibling
episode's live `outputs/` directory.
