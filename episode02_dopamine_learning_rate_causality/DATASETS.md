# Episode 02 data and exposure contract

## Status

This document defines evidence roles. It does not certify a local data copy,
open candidate or audit access, or record runtime receipts.

## Primary release

| Field | Value |
| --- | --- |
| Dataset | Data supporting *Mesolimbic dopamine adapts the rate of learning from action* |
| Provider | Janelia Figshare article `21816054`, version 1 |
| DOI | `10.25378/janelia.21816054.v1` |
| License | CC BY 4.0 |
| Trusted local source | `/oak/stanford/groups/russpold/data/br_autoresearch_data/dudman_learning_rate_2023/figshare-21816054-v1/source/seshMerge.mat` |
| Bytes | `271014539` |
| MD5 | `23b0b229d92ab9bf26ab9989946eafb4` |
| SHA-256 | `1039b9555ef50b80e01cc923b372a7960daade774fdb6a236c34cafcedf75866` |
| Container | MATLAB v5; `seshMerge`, `1 x 24` struct |

The provider MAT contains every role in one compressed object. It must never
be copied, linked, or mounted into a candidate workspace.

## Documentary and code sources

| Source | Permitted use |
| --- | --- |
| Nature article, DOI `10.1038/s41586-022-05614-z` | Task, acquisition, intervention, assignment prose, and reported exclusions; full-text XML SHA-256 `0e228404c08fc41a4d1715a6107a0741216b8cb5b508080ea0d6478bb1248c8d` |
| Supplementary Table 1 | SHA-256 `bb637651bd82b454b05cafe82e662012937524afa6a0d54ac9edac8fb7011a6d`; identifies records 21–24 as high-amplitude `stim+Lick+` |
| Reporting Summary | SHA-256 `388145418aa3a74a4be2254bbc4dd2b3977514e6a573d1204291fcab6d68ca12`; documents randomization within repeated 2–4-mouse cohorts, four removals after initial collection for poor signal associated with targeting/expression, and no blinding during collection |
| `dudmanj/RNN_learnDA`, commit `3627746957975632cddb05bf8f6b26a4b6f901b6` | Mechanism specification and synthetic oracle only |
| `DudLab/TONIC`, commit `f78fdbf4bca0e5c16859f885550b4e274fc3459d` | Documentary oracle; reuse requires license resolution |

The empirical Figure-1 script contains outcome-derived ordering and labels.
Only its published record mapping may be used.
That outcome-free mapping is frozen in [`COHORT_MAP.json`](COHORT_MAP.json);
the full empirical script remains prohibited from candidate workspaces.

## Record roles

| Role | Records | Mice | Trial rows | Access |
| --- | --- | ---: | ---: | --- |
| Development controls | `1,2,4,6,9,11,15,16,19` | 9 | 8,318 | Candidate-visible after role and firewall qualification |
| Primary `stimLick-` | `3,5,10,14,18,20` | 6 | 5,509 | Evaluator-only |
| Primary `stimLick+` | `7,8,12,13,17` | 5 | 5,016 | Evaluator-only |
| High-amplitude boundary | `21,22,23,24` | 4 | 2,434 | Evaluator-only, post-primary |

Identity is release version plus one-based struct index, never an inference
from outcome arrays. Observed fields are:

```text
trialID seshID lickState stimState latency cueLatency
nac vta ds pupil lick nose whisk body iti
predVars cuePredVars rewDA prepLick baseLick
```

There are 21,277 trial rows. NAc, lick, and body streams have 701 samples per
trial; VTA and DS coverage is partial. Records 21–24 have narrower support.

## Endpoint and design gaps

- `trialID` lacks an authoritative codebook.
- `seshID` must be tied to training-day semantics.
- The 100-Hz streams lack zero-time and event-alignment metadata.
- The relation between `lickState` and the online 750-ms contingency needs
  authoritative documentation.
- Missing blocks, interpolation, eligible denominators, and abstention must be
  frozen before intervention outcomes are opened.
- The randomization-block roster, allocations, and exclusion details are
  absent.

Provider-derived features (`predVars`, `cuePredVars`, `rewDA`, `prepLick`, and
`baseLick`) are parity checks only.

## Exposure ledger

| Evidence | Current status | Consequence |
| --- | --- | --- |
| Paper and aggregate figures | Public | Publication-exposed reanalysis only |
| Nine controls | Cohort identity and author-derived ordering exposed | Development-only forever |
| Eleven primary mice | Identity and structural support inspected | Evaluator-only |
| Four boundary mice | Identity and structural support inspected | Evaluator-only; cannot rescue primary |
| Legacy structural diagnostics | Per-record stimulated-trial count and contingency-consistency indicator emitted before search | Permanently excluded from scientific decisions |
| Mixed-role MAT | Trusted-steward source | Never candidate-mounted |

## Runtime provisioning

The tracked `inputs/` directory contains instructions only. A trusted steward
must create physically separate development, candidate-structural, primary-
audit, and boundary-audit surfaces. Candidate workers may see only the first
two.

Portable source tools for structural inventory and deterministic role
materialization are tracked under `outputs/code/`. The tools require an
explicit `DUDMAN_SOURCE_MAT` and job-specific `$SCRATCH` and never publish
generated packets into the tracked episode. Materialization integrity does
not establish confidentiality.

Before candidate or audit access, the runtime must enforce distinct steward,
candidate, and evaluator principals; source and audit-payload denial to the
candidate; disabled network egress; evaluator-only audit mounts; and one atomic
aggregate return packet. The stable contract is
[`outputs/firewall/FIREWALL_CONTRACT.json`](outputs/firewall/FIREWALL_CONTRACT.json).
The tracked contract is intentionally incomplete: the audit-pack, policy, and
configuration-lock hashes and two distinct Ed25519 signer pins must be added
after runtime objects are frozen. A current, doubly signed receipt is still
required.

Generated inventories, role packs, calibration tables, receipts, signatures,
and checksum files belong in scratch or another runtime store. Their absence
from Git is intentional.

## Access boundary

Data availability does not grant role access. Endpoint semantics, randomization
or exchangeability scope, physical separation, endpoint-faithful calibration,
a raw-unit margin, implementation qualification, evaluator behavior, and the
audit-opening criteria remain unresolved.
