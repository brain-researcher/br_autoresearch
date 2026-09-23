# Verification projection

> Client-maintained record of mechanical verification only. This file grants
> no scientific, Society, reward, approval, execution, ClaimCard, memory, or
> Landscape authority.

## Verification scope

- verifier: local contract checks
- state: formal local specification; no scientific run or audit has started
- exact prior-artifact identities live only in the anti-leakage lineage
  contract and are not duplicated in this projection

## Mechanical verification

| Check | Status | Scope or limitation |
| --- | --- | --- |
| Required tree and seven projections | passed | Exact seven-file projection set plus top-level contracts and guards; no scientific payload |
| Unique-key YAML and JSON parsing | passed | PyYAML unique-key loader for changed YAML and standard JSON parser for schemas/manifest |
| Search-policy schema | passed | JSON Schema Draft 2020-12 validation with `jsonschema`; structure, not scientific validity |
| Budget, coverage, ledger, audit, and terminal assertions | passed | 30/60/12 trials, 12/12/6 stage minima, 15 failures, 2,500 CPU-hours, 0 GPU-hours, 96 wall-hours, 1,536 GiB scratch; 12 coverage slots; common ledger fields; nine terminal classes; one scientific audit transaction |
| Decision-contract cross-check | passed | Lock prerequisites equal the stop-policy depth gates; family-selection precedence, nonempty nonkernel comparator set, patience reset, incomplete routes, and no-output retry semantics are explicit |
| Registry and evidence-policy identity | passed | Exactly one current EP01 at the new path, formal EP02 is the independent Dudman contract, EP18 is the sole incomplete and uncounted draft, two NARPS prior IDs remain unambiguous, and the current portfolio count is 19 |
| Prior artifact identity | passed | All 23 declared files match the identities in the lineage contract at both their current archive paths and frozen historical revision; the packet remains metadata-only and is not launch-ready |
| Legacy relocation integrity | passed | All 70 tracked blobs are byte-identical after the path-only move; file counts, byte totals, and three historical input symlinks are unchanged; both old root paths are absent |
| Ignore-boundary check | passed | All 646 legacy local-only payload entries remain ignored; prior manifest is trackable; future `inputs/prior_lineage/materialized/` payload remains ignored |
| Audit-input exclusion | passed locally | New inputs contain no neural-array extension; repository inspection cannot substitute for an OS/service firewall |
| Cross-contract links and whitespace | passed | New relative Markdown links resolve; checked files have no trailing whitespace |

## Scientific boundary

- Scientific validity, source readiness, calibration, audit independence, and
  canonical registration remain unresolved launch blockers.
