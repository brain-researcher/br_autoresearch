# Input state at initialization

No neural, behavioral, stimulus-image, model, or original-analysis data are
materialized here.

The root
[`DATA_LOCATION_MANIFEST.json`](../../DATA_LOCATION_MANIFEST.json) records the
current quarantined source as `legacy_steward_acquisition`, the planned durable
external root as `planned_steward_acquisition`, and the existing legacy runtime
copy as `scratch_ep04`. The external target is `absent_planned_target`, while
the source assets remain `staged_not_moved`. `scratch_ep04` is a
`duplicate_review` cleanup candidate after durable verification, not an input
handoff or final destination. This directory contains none of those bytes.

The documented candidate sources, planned read-only references, access terms,
and unresolved readiness checks are listed in `../DATASETS.md`. Before launch,
an operator may provision immutable references to separately stored sources and
record their versions and hashes. The active episode must not write into this
directory or replace a sealed source after outcome access.

A location-manifest entry alone does not establish a role-filtered handoff,
permissions, integrity, scientific readiness, or launch authorization.

Outcome-blind source manifests and all generated artifacts belong under
`../outputs/`.
