# EP21 inputs

## Stable interfaces

The ASTRA program refers to four role-specific input interfaces. Their names
are stable; their current availability is recorded only in
`outputs/experiment_log.md`.

| ASTRA input | Expected path | Role |
| --- | --- | --- |
| `laion_regular_structural_manifest` | `inputs/laion_regular_structural_manifest.json` | Redacted, pre-lock structural metadata |
| `laion_regular_feature_handoff` | `inputs/laion_regular_feature_handoff/` | Non-neural regular-image stimuli, captions, and features |
| `laion_regular_scoring_handoff` | `inputs/laion_regular_scoring_handoff/` | Post-lock, role-filtered regular-only neural scoring data |
| `upstream_model_asset_inventory` | `inputs/upstream_model_asset_inventory.json` | Outcome-blind code, model, checkpoint, tokenizer, parser, and weight identities |

These paths define interfaces, not evidence that a payload has been
provisioned. Once provisioned, `inputs/` is read-only.

## Ownership and access

- `source_qualification` reads the structural manifest and model-asset
  inventory. Its lock-facing report may contain only IDs and roles, shapes,
  dtypes, finite/NaN masks without values, coverage, missingness, provenance,
  and stimulus-universe labels.
- The caption/control, RCNN, and cross-modal feature-preparation analyses read
  the non-neural feature handoff only after a valid protocol lock exists.
- Neural-scoring analyses read the scoring handoff only after the protocol
  lock and their required feature manifests exist.
- Neither the feature handoff nor the scoring handoff is an input to source
  qualification or protocol locking.

A mixed-role handoff is invalid. All 371 OOD raw images, derived features,
neural responses, predictions, scores, and comparison summaries remain inaccessible
to EP21. Neural values, neural summaries, predictions, and model--brain scores
are withheld from the lock resolver.

The root `DATA_LOCATION_MANIFEST.json` is an inventory pointer only. It does
not constitute a handoff, resolve the stimulus DUA, authorize protected-data
access, or make EP04's legacy Scratch tree an EP21 input.

Before provisioning, record the release identity, data-risk and DUA decision,
role-filtered path, allowed slices, and exact upstream asset identities or
verified absence. Missing assets may lock an affected module or subclaim as
not evaluable; they cannot trigger a post-result substitute. Do not write
generated code, features, predictions, caches, or reports under `inputs/`.
