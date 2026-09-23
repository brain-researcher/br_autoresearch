# Inputs

This directory is read-only during an authorized Episode 01 run. The present
draft contains only a lineage manifest; it contains no development or audit
neural payload and is not launch-ready.

Required pre-launch surfaces are:

- `prior_lineage/materialized/`: verified content-addressed copies of the small
  legacy artifacts declared in `PRIOR_LINEAGE_MANIFEST.json`;
- `development/`: immutable `ds001734` payload references, folds, masks,
  operators, environments, containers, and synthetic fixtures; and
- `audit_manifest/`: metadata, schema, counts, and commitments for `ds000005`,
  with no candidate-discriminating array.

The true audit payload must remain outside this workspace and be reachable only
by the trusted one-shot runner after configuration lock. Do not add symlinks or
hardlinks to either legacy output tree or to the audit data.
