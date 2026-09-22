# Verification projection

> Client-maintained record of mechanical verification only. This file does not
> grant scientific, Society, reward, approval, execution, ClaimCard, memory, or
> Landscape authority. It is not a CandidateBundle review artifact and must
> never be declared in a frozen CandidateBundle's `output_artifacts`.

## Binding

- observed_at: `2026-09-22T16:09:15-07:00`
- verifier: local contract checks run by Codex
- source revision or environment: validated on a local branch from `brain-researcher/br_autoresearch` base `9c4396bbd6add1b537907dd330ba76a54363439a`; shared contract snapshots came from committed OAK revisions `a499b40` and `debfba4`; ignored OAK inputs remain external; Episode 20 remains an unregistered local draft
- `GOAL.md` SHA-256: `4a788471673d956cc9b46d0af05a8a1f48f782b15de8b685cee624f1a7cc8d67`
- `DATASETS.md` SHA-256: `e89642d4bab7fb107a6e8a09cb2dbc72d6955ea41278a2445985c2e69dd91360`
- `SEARCH_POLICY.yaml` SHA-256: `0cfd3dd18056c8d6849967ffd5ae5c77762c2fc960187ec68ab5e7f4332a539a`

## Mechanical verification

| Check | Status | Evidence ref | Scope or limitation |
| --- | --- | --- | --- |
| Artifact presence | passed | [`../GOAL.md`](../GOAL.md), [`../DATASETS.md`](../DATASETS.md), [`../SEARCH_POLICY.yaml`](../SEARCH_POLICY.yaml) | Required top-level contracts, input guard, output guard, seven workspace projections, and supporting read-only validation code are present; no scientific payload is present |
| Policy parse and schema validation | passed | [`../SEARCH_POLICY.yaml`](../SEARCH_POLICY.yaml), [`../../SEARCH_POLICY.schema.json`](../../SEARCH_POLICY.schema.json) | PyYAML unique-key loader plus the schema-selected `jsonschema` Draft 2020-12 validator; validates structure, not scientific adequacy |
| Cross-contract assertions | passed | [`../GOAL.md`](../GOAL.md), [`../SEARCH_POLICY.yaml`](../SEARCH_POLICY.yaml) | Exact 16-arm coverage; 14 ordered terminal rules and outer mappings; 32/64/12/12 trial, patience, and engineering-failure limits; deterministic stop routes; explicit pre-audit development lock |
| Ledger compatibility | passed | [`../SEARCH_POLICY.yaml`](../SEARCH_POLICY.yaml), [`../../TRIAL_LEDGER.schema.json`](../../TRIAL_LEDGER.schema.json) | Policy requires every common ledger field plus Episode 20 ladder, arm, hardware, software, environment, and resource hashes |
| Estimand and role assertions | passed | [`../GOAL.md`](../GOAL.md), [`../DATASETS.md`](../DATASETS.md), [`../SEARCH_POLICY.yaml`](../SEARCH_POLICY.yaml) | Member-specific eligible supports and weight renormalization; `s1`/resource matching; all eligible `D4` controls; public fit-holdout; replay-to-hardware firewall; disjoint empirical partitions; 17 pre-run manifests |
| Numerical recomputation | passed | [`../DATASETS.md`](../DATASETS.md), [`../SEARCH_POLICY.yaml`](../SEARCH_POLICY.yaml) | Recomputed pixel counts, raw conversion rates, effective pixel-sample rates, 4-ms full scan, 250-S/s rate, and 125-Hz Nyquist from documentary values |
| Reference-example boundary checks | passed with external-input limitation | [`code/validate_reference_bundles.py`](code/validate_reference_bundles.py), [`code/test_reference_bundles.py`](code/test_reference_bundles.py), [`../inputs/README.md`](../inputs/README.md) | Validator and all three tests passed against the separately provisioned ignored OAK bundles; a clean Git checkout passes two synthetic boundary tests and explicitly skips the provisioned-bundle test, so full validation is not standalone-rerunnable without those inputs |

## Scientific boundary

- Scientific validity: not mechanically verified; reference qualification,
  simulator identifiability, statistical power, and audit independence remain
  unresolved launch blockers.
- Mechanical agreement does not establish novelty, causal interpretation,
  independent confirmation, or scientific acceptance.
