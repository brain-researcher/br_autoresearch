# EP11 Stage-0 artifacts

The launch manifest and its referenced initial readiness snapshot are immutable
local provenance. Later state is written as numbered snapshots and append-only
experiment-log events; it does not rewrite the launch record.

- `launch_manifest.yaml` and `launch_manifest.sha256`: frozen local launch.
- `readiness_status.yaml`: readiness snapshot at launch.
- `readiness_status.0002.yaml`: state after synthetic metadata-validator checks.
- `redacted_metadata_contract.yaml`: trusted allow-list and role-safe outputs.
- `steward_handoff.md`: required permission-separated deliverables.
- `tools/validate_redacted_metadata.py`: trusted-side validator that turns a
  content-addressed, allow-listed row table into identifier-free aggregate
  support; only that aggregate may cross to the candidate process, and the
  tool must never receive the raw workbook.
- `fixtures/`: synthetic-only positive and leakage-rejection cases.
- `synthetic_validation.yaml`: exact code/fixture hashes and observed results.

No file in this directory is a projection outcome or a canonical Brain
Researcher artifact.
