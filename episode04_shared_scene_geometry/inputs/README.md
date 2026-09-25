# Input state at initialization

No neural, behavioral, stimulus-image, model, or original-analysis data are
materialized here.

The root
[`DATA_LOCATION_MANIFEST.json`](../../DATA_LOCATION_MANIFEST.json) records the
canonical private steward root as `private_steward_acquisition` at
`/oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/steward_acquisition`
and the existing legacy runtime copy as `scratch_ep04`. The steward tree was
relocated by same-filesystem rename on 2026-09-24 and is
`canonical_in_place`. `scratch_ep04` remains a `duplicate_review` cleanup
candidate after durable verification, not an input handoff or final
destination. This directory contains none of those bytes.

The documented candidate sources, planned read-only references, access terms,
and unresolved source-qualification checks are listed in `../DATASETS.md`.
During episode setup, an operator may provision immutable references to
separately stored sources and record their versions and hashes. The active
episode must not write into this directory or replace a sealed source after
outcome access.

A location-manifest entry alone does not establish a role-filtered handoff,
permissions, integrity, scientific qualification, or access to protected
outcomes.

Outcome-blind source manifests and all generated artifacts belong under
`../outputs/`.
