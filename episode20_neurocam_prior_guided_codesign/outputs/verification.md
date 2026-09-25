# Verification projection

> Client-maintained record of mechanical verification only. This file does not
> grant scientific, Society, reward, approval, execution, ClaimCard, memory, or
> Landscape authority. It is not a CandidateBundle review artifact and must
> never be declared in a frozen CandidateBundle's `output_artifacts`.

## Binding

- observed_at: `2026-09-21T22:30:07-07:00`
- verifier: local contract checks run by Codex
- state: formal local specification; no experiment or audit has started

## Mechanical verification

| Check | Status | Evidence ref | Scope or limitation |
| --- | --- | --- | --- |
| Artifact presence | passed | [`../GOAL.md`](../GOAL.md), [`../DATASETS.md`](../DATASETS.md), [`../SEARCH_POLICY.yaml`](../SEARCH_POLICY.yaml) | Required top-level contracts, input guard, output guard, and exactly seven workspace projections are present; no scientific payload is present |
| Policy parse and schema validation | passed | [`../SEARCH_POLICY.yaml`](../SEARCH_POLICY.yaml), [`../../SEARCH_POLICY.schema.json`](../../SEARCH_POLICY.schema.json) | PyYAML unique-key loader plus the schema-selected `jsonschema` Draft 2020-12 validator; validates structure, not scientific adequacy |
| Cross-contract assertions | passed | [`../GOAL.md`](../GOAL.md), [`../SEARCH_POLICY.yaml`](../SEARCH_POLICY.yaml) | Exact 16-arm coverage; 14 ordered terminal rules and outer mappings; 32/64/12/12 trial, patience, and engineering-failure limits; deterministic stop routes; explicit pre-audit development lock |
| Ledger compatibility | passed | [`../SEARCH_POLICY.yaml`](../SEARCH_POLICY.yaml), [`../../TRIAL_LEDGER.schema.json`](../../TRIAL_LEDGER.schema.json) | Policy requires every common ledger field plus Episode 20 ladder, arm, hardware, software, environment, and resource hashes |
| Estimand and role assertions | passed | [`../GOAL.md`](../GOAL.md), [`../DATASETS.md`](../DATASETS.md), [`../SEARCH_POLICY.yaml`](../SEARCH_POLICY.yaml) | Member-specific eligible supports and weight renormalization; `s1`/resource matching; all eligible `D4` controls; public fit-holdout; replay-to-hardware firewall; disjoint empirical partitions; 17 pre-run manifests |
| Numerical recomputation | passed | [`../DATASETS.md`](../DATASETS.md), [`../SEARCH_POLICY.yaml`](../SEARCH_POLICY.yaml) | Recomputed pixel counts, raw conversion rates, effective pixel-sample rates, 4-ms full scan, 250-S/s rate, and 125-Hz Nyquist from documentary values |

## Scientific boundary

- Scientific validity: not mechanically verified; reference qualification,
  simulator identifiability, statistical power, and audit independence remain
  unresolved scientific requirements before candidate scoring and audit.
- Mechanical agreement does not establish novelty, causal interpretation,
  independent confirmation, or scientific acceptance.
