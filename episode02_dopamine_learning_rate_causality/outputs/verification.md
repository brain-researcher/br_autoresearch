# Verification projection

> Specification verification only; this file grants no candidate-scoring,
> audit-access, or scientific authority.

## Tracked checks

- `GOAL.md`, `DATASETS.md`, and `SEARCH_POLICY.json` define one consistent
  question, estimand, record split, and audit boundary.
- The primary estimand is `delta_rate_raw` in probability-gain units;
  `delta_rate_std = delta_rate_raw / development_scale` is a secondary
  scale-free companion, and the evaluator must report both explicitly.
- The four unbound future thresholds use the calibration-compatible canonical
  names `practical_margin_raw`, `arm_margin_raw`,
  `development_scale_floor_raw`, and `loo_delta_floor_raw`.
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

## 2026-09-30 — Focused writing/figure coherence

- GOAL, DATASETS, README and paper plan distinguish control prediction,
  complete-case protocol contrast and learning-rule identification.
- Policy narrative additions do not alter the raw primary contrast, exact
  standardized transform, numerical rules, family grammar, role assignments,
  budget, thresholds or authorization flags. The four thresholds remain null.
- No formal all-rival exclusion statistic or new endpoint is active.
- The saved imagegen PNG depicts reward-time lick/no-lick trigger rules,
  6/5 primary mice and 9 controls, the 750-ms window, the `stimLick+` cap and
  two-epoch gain. It has no empirical values or winner. The final prompt set
  records its targeted correction; GOAL/README/paper-plan links use local files.
- Focused checks passed: JSON parsing; unchanged primary units, four null raw
  thresholds, 28–48 budget and closed authorization flags; YAML parsing and
  ASTRA internal artifact references; 19 local links in GOAL/README/paper plan;
  saved PNG presence; and scoped `git diff --check`.
- ASTRA is a compact worklist of existing requirements. Official ASTRA 0.0.14
  schema/semantic conformance remains unverified; no validator was installed.

These are writing and asset checks, not scientific qualification or audit
opening. No protected outcome, runtime result or code executor was inspected.
