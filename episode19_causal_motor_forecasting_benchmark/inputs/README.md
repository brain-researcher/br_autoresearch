# Episode 19 inputs

`inputs/` is read-only after provisioning. No role-filtered WAY-EEG-GAL or
self-paced EEG handoff is currently present here. Their verified, unextracted
mixed source archives remain outside the episode under the canonical private
steward root. The deferred AJILE12 materialization also remains outside the
episode and is not a primary-task fallback.

The root
[`DATA_LOCATION_MANIFEST.json`](../../DATA_LOCATION_MANIFEST.json) names the
canonical private steward root `private_steward_acquisition` at
`/oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/steward_acquisition`
and groups these assets as `asset_ep19_public_sources`,
`asset_ep19_safe_commit`, and `asset_ep19_ajile12`. The steward tree was
relocated by same-filesystem rename on 2026-09-24 and is
`canonical_in_place`. None of those logical locations is a provisioned EP19
input or role-qualified handoff.

`../DATASETS.md` requires content-addressed, role-filtered structural,
development, no-feedback lock, and evaluator-only audit handoffs. Do not link a
mixed public download or shared source tree into this directory. Public access
does not grant held-out access to a candidate or controller.

No candidate output, cache, fitted model, onset result, audit QC, or score may
be written under `inputs/`.

Before candidate scoring, this directory must also contain
`FROZEN_ADMISSIBLE_SPACE.yaml` and its signed SHA-256 record. That manifest
enumerates every legal search operator and exact configuration range; its hash
is frozen before any candidate EEG outcome is returned. It is intentionally
absent until the admissible space is frozen.
