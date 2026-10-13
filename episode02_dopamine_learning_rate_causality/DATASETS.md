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

This document defines evidence roles. The scientist's October 12 decision
adopts [procedural same-account separation](GOAL.md#adopted-procedural-evaluation-separation--october-12-2026)
and replaces the enforced-isolation requirement. Physical isolation remains
unverified; this document opens neither candidate nor audit access. Development
still needs its separately recorded handoff release and existing prerequisites.

The scientist's October 6 agreement adopts authenticated article/task primary
and separately labeled pinned-code sensitivity; see the
[accepted choice](outputs/source_semantics_design.md#adopted-model-choice--october-6-2026).
Provider field meanings, actual online trigger/cap logic and sample times remain
unverified source facts. This adoption changes neither data roles nor the
inactive margin proposals and does not open the mixed-role MAT.

As reconciled October 6, the [source and implementation handoff](outputs/source_semantics_design.md)
already distinguishes the article's task rule from the pinned-code simulator,
the approved noncausal 462-label sensitivity from randomized inference, and
checked arithmetic components from a complete empirical fitter. ACTR/parity,
development-score and pinned-critic component checks have passed; eleven
optional rival-interface checks also passed, but those interfaces remain an
unadopted proposal. See the [dated execution evidence](outputs/experiment_log.md)
and [current navigation](outputs/README.md). The exact provider request is
drafted and unsent. Four grid-derived margin proposals remain inactive and
need a substantive rationale; they are not missing data fields. The
[October 6 public-source check](outputs/source_semantics_design.md#public-source-check--october-6-2026-pacific)
did not resolve the release codebook/time origin or verify current Figshare
attachments. None of this opens the mixed-role MAT or intervention outcomes.

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

### Public online resources — checked October 6, 2026

- [Official RNN_learnDA repository](https://github.com/dudmanj/RNN_learnDA): public MIT-licensed RNN reinforcement-learning code, including the recurrent trainer, critic and model-comparison source. It supplies implementation oracles, not the missing empirical field meanings. EP02 keeps its pinned commit and prohibits the outcome-derived empirical Figure-1 script from candidate workspaces.
- [Official TONIC repository](https://github.com/DudLab/TONIC): public MATLAB documentation/code includes event alignment, data-loading and event-detection utilities. Its README explains that raster alignment consumes caller-supplied event timestamps/windows; that generic interface cannot identify the released 701-sample time origin or trial codes. Existing license and source-reuse restrictions remain.
- [Figshare release v1](https://doi.org/10.25378/janelia.21816054.v1): the October 6 browser failure was resolved by a bounded public metadata request on October 7. The [article API](https://api.figshare.com/v2/articles/21816054) lists only `seshMerge.mat` (file38710665, 271,014,539 bytes), and the [version list](https://api.figshare.com/v2/articles/21816054/versions) lists only version1. Its [version1 description](https://api.figshare.com/v2/articles/21816054/versions/1) has no field codebook, sample-time vector or task/exporter link; custom fields are empty. No separate documentation attachment or later published version is listed. This finding is limited to that record, not all possible author resources. The already provisioned MAT remains the selected source and was not downloaded or opened.

The inspected repository documentation does not resolve `trialID`, `seshID`,
`lickState`, the exact online 750-ms/cap implementation or the assignment and
exclusion roster. The [specific source request](outputs/source_semantics_design.md)
therefore remains useful; no additional simulation or re-extraction can supply
these semantics. No empirical array or outcome-derived script was opened.

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

The tracked `inputs/` directory contains instructions only. The steward role
must provide separate development, candidate-structural, primary-audit and
boundary-audit surfaces. Under the October 12 amendment the same account may
hold these procedural roles; the separation is not permission-enforced.
Candidate work may use only the released first two surfaces, and must not open,
list, search or reacquire mixed-source or audit paths.

Portable source tools for structural inventory and deterministic role
materialization are tracked under `outputs/code/`. The tools require an
explicit `DUDMAN_SOURCE_MAT` and job-specific `$SCRATCH` and never publish
generated packets into the tracked episode. Materialization integrity does
not establish confidentiality.

Before development access, record the steward and candidate roles, their
permitted paths and the development handoff release. Before audit access,
record the evaluator role and freeze the analysis choices and versioned final
lock. Audit data may be opened only in the single recorded evaluator
transaction, which returns one atomic aggregate packet without partial
feedback. Its optional boundary stage follows the committed primary conclusion.
Log every audit-role access in `outputs/experiment_log.md`; disclose unplanned
access and mark affected results. Record the responsible operator, source
release and role map. No signature, checksum bundle or custom receipt format
is required. Same-account permissions do not demonstrate confidentiality or
blinding; every evaluation result must disclose this weaker procedural boundary.

The enforced identity, storage-denial and network-denial requirements in
[`outputs/firewall/FIREWALL_CONTRACT.json`](outputs/firewall/FIREWALL_CONTRACT.json)
and `FIREWALL.md` are superseded by the October 12 terms in GOAL and
SEARCH_POLICY. Their role restrictions, no reacquisition, final lock, single
opening and aggregate-only output remain operative. No handoff, access flag or
scoring authorization is released by this documentation change.

Generated inventories, role packs, and calibration tables belong in scratch
or another runtime store. Their absence from Git is intentional.

## Access boundary

Data availability does not grant role access. Endpoint semantics, randomization
or exchangeability scope, the development handoff, endpoint-faithful calibration,
a raw-unit margin, implementation qualification, evaluator behavior, and the
audit-opening criteria remain unresolved.
