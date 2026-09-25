# Experiment log

Append-only local log for `ep11-stage0-20260925T000506Z`. Event hashes are
SHA-256 of the exact `canonical_payload` string. They are local provenance, not
canonical Brain Researcher receipts.

## Event 0001 — Stage-0 launch

- UTC: `2026-09-25T00:05:06Z`
- Type: `stage0_launch`
- Previous event: `GENESIS`
- User authorization: “好 那就继续？” after the metadata-only launch boundary
  was explained.
- Scope: metadata/readiness qualification, synthetic checks, and trusted
  handoff only.
- Manifest: [`stage0/launch_manifest.yaml`](stage0/launch_manifest.yaml),
  SHA-256 `e57fa907df46cca10d628d97a3dfd43ce67ec51facc07663c0f360e3c35b7aa3`.
- Outcome access: none.
- Audit open count: `0`.
- `canonical_payload`: `event_id=0001|utc=2026-09-25T00:05:06Z|type=stage0_launch|prev=GENESIS|manifest_sha256=e57fa907df46cca10d628d97a3dfd43ce67ec51facc07663c0f360e3c35b7aa3|scope=metadata_readiness_only|outcome_access=false|audit_open_count=0`
- Event SHA-256: `b0fd2991878d038550583a8b6c1afe9de15114c2736b5c132bc8b5346c398f00`.

## Event 0002 — Mechanical acquisition receipt verification

- UTC: `2026-09-25T00:07:54Z`
- Type: `mechanical_source_receipt_verification`
- Previous event SHA-256:
  `b0fd2991878d038550583a8b6c1afe9de15114c2736b5c132bc8b5346c398f00`.
- `Full_morphometry.xlsx` byte length and SHA-256 matched its acquisition
  receipt. The Allen annotation, template, and structure-graph hashes also
  matched their receipts.
- Only byte hashing, file metadata, quarantine documentation, and checksum
  receipts were read. No workbook cells, archive contents, axonal geometry,
  projection summary, or audit outcome was decoded or displayed.
- `canonical_payload`: `event_id=0002|utc=2026-09-25T00:07:54Z|type=mechanical_source_receipt_verification|prev=b0fd2991878d038550583a8b6c1afe9de15114c2736b5c132bc8b5346c398f00|source_workbook_sha256=7cec6a78b83f27570c524ba76db637383c6e596e919c2eb7c11ff09a0009d0a1|ccf_receipts=match|decoded_values=false|outcome_access=false`
- Event SHA-256: `688f265a81a3340edee607f8a5672a10b807c001fc06c4049e2c97e3d1935be0`.

## Event 0003 — Readiness gate assessment

- UTC: `2026-09-25T00:10:56Z`
- Type: `readiness_gate_assessment`
- Previous event SHA-256:
  `688f265a81a3340edee607f8a5672a10b807c001fc06c4049e2c97e3d1935be0`.
- Result: Stage 0 remains active and is blocked on an external trusted handoff,
  not terminal. No valid census yet exists, so
  `closed_insufficient_source_support` has not been adjudicated.
- Missing: SEU-specific redactor, authenticated Brain ID→animal mapping,
  duplicate lineage, shared EP09--11 role ledger, source-support census,
  role-separated views, and validated CCF/SEU binding.
- Artifacts: [`stage0/readiness_status.yaml`](stage0/readiness_status.yaml),
  [`stage0/redacted_metadata_contract.yaml`](stage0/redacted_metadata_contract.yaml),
  and [`stage0/steward_handoff.md`](stage0/steward_handoff.md).
- Outcome access: none. Audit open count: `0`.
- `canonical_payload`: `event_id=0003|utc=2026-09-25T00:10:56Z|type=readiness_gate_assessment|prev=688f265a81a3340edee607f8a5672a10b807c001fc06c4049e2c97e3d1935be0|status=blocked_on_trusted_handoff|terminal=false|outcome_access=false|audit_open_count=0`
- Event SHA-256: `7d8cc20e75d5378cec62f81a656e95f36a8d1dc8b8576a3220566a842eb7dab2`.

## Event 0004 — Synthetic trusted metadata-aggregate validator qualification

- UTC: `2026-09-25T00:13:29Z`
- Type: `synthetic_metadata_validator_qualification`
- Previous event SHA-256:
  `7d8cc20e75d5378cec62f81a656e95f36a8d1dc8b8576a3220566a842eb7dab2`.
- Tool: [`stage0/tools/validate_redacted_metadata.py`](stage0/tools/validate_redacted_metadata.py),
  SHA-256 `274bf5d1bf4c9b9f02d8f012557d3060622f2056a7a4ebfbe0c8d73423123ecf`.
- Positive synthetic input with 12 development and 8 audit animals passed the
  count-floor check while explicitly leaving full support unadjudicated.
- A fixture placing one animal in both roles and a fixture containing
  `Projection class` were both rejected with exit code 2.
- Real row-level audit metadata is restricted to the trusted steward process;
  only identifier-free aggregates may reach candidates.
- No real metadata or projection outcome was read.
- Full record: [`stage0/synthetic_validation.yaml`](stage0/synthetic_validation.yaml).
- `canonical_payload`: `event_id=0004|utc=2026-09-25T00:13:29Z|type=synthetic_metadata_validator_qualification|prev=7d8cc20e75d5378cec62f81a656e95f36a8d1dc8b8576a3220566a842eb7dab2|tool_sha256=274bf5d1bf4c9b9f02d8f012557d3060622f2056a7a4ebfbe0c8d73423123ecf|positive=pass|role_leak=rejected|forbidden_field=rejected|real_data_access=false`
- Event SHA-256: `5fd5b9d117feff67a32fbcafcdf99eb837d2775902b24c9ea258745398969e2e`.
