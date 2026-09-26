# Verification projection

> Specification verification only; this file grants no candidate-scoring,
> audit-access, or scientific authority.

## Tracked checks

- `GOAL.md`, `DATASETS.md`, and `SEARCH_POLICY.json` define one consistent
  question, estimand, record split, and audit boundary.
- The mouse is the sole inferential and resampling unit.
- Candidate mounts exclude the mixed source and both audit payloads.
- The primary audit is one-shot, aggregate, and nonadaptive.
- The boundary group cannot alter or rescue the primary terminal.
- Audit opening and positive mechanistic terminals are currently disabled.
- The tracked cohort map is outcome-free and agrees with the policy's disjoint
  9/6/5/4 source-index partition.
- `DATASETS.md` and the policy identify the Figshare release, while the cohort
  map records the published source indices; no generated digest is needed to
  establish that mapping.
- Role-materialization and calibration helpers remain available, but their
  manifests, schemas, hashes, receipts, and exact runtime details are not
  qualification gates.
- The audit access boundary is checked directly once provisioned and repeated
  only if the access layout changes or a concrete failure is observed.
- The final configuration is a versioned write-once record frozen before audit
  access; a lock hash is not required.

## Remaining runtime verification

Endpoint semantics, the randomization/exclusion roster, physical access
separation, implementation qualification, calibration operating
characteristics, evaluator behavior, and final authorization must be verified
at the stage where each becomes relevant. A short pass/revise note is enough;
no extra schema, checksum, or attestation layer is required.
