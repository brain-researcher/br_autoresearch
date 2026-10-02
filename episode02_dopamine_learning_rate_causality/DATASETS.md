# EP02 — Evidence for a protocol contrast, not a unique learning rule

This release is used for a publication-exposed mechanistic-discrimination
reanalysis of naive trace conditioning. It cannot supply a new independent
discovery cohort. The [scientific question](GOAL.md) is whether control-
qualified rules make distinguishable locked predictions and what the existing
mouse-level protocol contrast can identify.

The source paper's adaptive-rate interpretation is prior knowledge, not an
unseen target. Other tasks and learning stages do not enter this release.
Adding modern alternative model families, new trajectory endpoints, or a
formal rival-exclusion test would require a separate pre-outcome amendment;
the current model grammar and terminal rules remain unchanged.

## Status

This document defines evidence roles. It does not itself certify role
separation or open candidate or audit access.

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
| Container | MATLAB v5; `seshMerge`, `1 x 24` struct |

The provider MAT contains every role in one compressed object. It must never
be copied, linked, or mounted into a candidate workspace.

## Documentary and code sources

| Source | Permitted use |
| --- | --- |
| Nature article, DOI `10.1038/s41586-022-05614-z` | Task, acquisition, intervention, assignment prose, and reported exclusions |
| Supplementary Table 1 | Identifies records 21–24 as high-amplitude `stim+Lick+` |
| Reporting Summary | Documents randomization within repeated 2–4-mouse cohorts, four removals after initial collection for poor signal associated with targeting/expression, and no blinding during collection |
| `dudmanj/RNN_learnDA`, commit `3627746957975632cddb05bf8f6b26a4b6f901b6` | Mechanism specification and synthetic oracle only |
| `DudLab/TONIC`, commit `f78fdbf4bca0e5c16859f885550b4e274fc3459d` | Documentary oracle; reuse requires license resolution |

The empirical Figure-1 script contains outcome-derived ordering and labels.
Only its published record mapping may be used.
That outcome-free mapping is frozen in [`COHORT_MAP.json`](COHORT_MAP.json);
the full empirical script remains prohibited from candidate workspaces.

## Record roles

| Role | Records | Mice | Trial rows | Access |
| --- | --- | ---: | ---: | --- |
| Development controls | `1,2,4,6,9,11,15,16,19` | 9 | 8,318 | Candidate-visible after development-role separation |
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

### Estimand unit convention

The mouse-level `raw_gain` is the session-8 minus session-1 preparatory lick
probability. The primary contrast `delta_rate_raw` is the between-arm difference
in mean raw gain and is measured in probability-gain units. The companion
`delta_rate_std` equals `delta_rate_raw` divided by the frozen development-
control standard deviation and is measured in development-SD units. It is not
an independent estimand, and any standardized margin must be derived exactly
from the presigned raw-unit margin.

Despite the historical symbol `delta_rate_raw`, this is an acquisition-gain
contrast, not an estimate of a latent learning-rate parameter. The two epochs
alone do not identify rate separately from asymptote or expression. The
continuous-lick, latency, motion and neural summaries keep their existing
replication/falsifier/secondary roles; none becomes a new primary endpoint.

The assigned interventions are whole closed-loop protocols: reward-time
stimulation after no lick versus a lick in the preceding verified 750-ms
window, with a 50% session cap in `stimLick+`. The protocol contrast is not a
fixed-dose or per-stimulation causal effect. Realized stimulation counts
remain excluded rather than being adjusted as if they were baseline features.

Nine control mice provide model development and a reference center, not
randomized arm-versus-control confirmation. Without the assignment and
attrition roster, the released complete cases support only explicitly
assumption-dependent comparisons. An unblocked label calculation is not the
documented design, an ITT effect, or a heterogeneous average-effect confidence
bound. A rate-gate-consistent pattern is not unique mechanism identification.

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

Before development access, the runtime must separate steward and candidate
roles and deny the candidate access to the mixed source and both audit
payloads. Before audit access, it must additionally separate the evaluator,
disable candidate and evaluator network egress, mount audit data only after
the final lock, and return one atomic aggregate packet. Record the people or
runtime identities responsible, the source release and role map used, and the
versioned final lock in a short run note. No signature, checksum bundle, or
custom receipt format is required. The access controls are summarized in
[`outputs/firewall/FIREWALL_CONTRACT.json`](outputs/firewall/FIREWALL_CONTRACT.json).

Generated inventories, role packs, and calibration tables belong in scratch
or another runtime store. Their absence from Git is intentional.

## Access boundary

Data availability does not grant role access. Endpoint semantics, randomization
or exchangeability scope, physical separation, endpoint-faithful calibration,
a raw-unit margin, implementation qualification, evaluator behavior, and the
audit-opening criteria remain unresolved.
