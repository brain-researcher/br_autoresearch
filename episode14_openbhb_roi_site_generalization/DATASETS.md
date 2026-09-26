# Dataset Contract — Episode 14

This contract uses the pinned, prepared OpenBHB ROI material in this episode.
It provisions nothing and authorizes no audit-label
access. Shared adaptive rules are in
[`../ADAPTIVE_SEARCH_PROTOCOL.md`](../ADAPTIVE_SEARCH_PROTOCOL.md).

## Fixed upstream identities

| Item | Fixed value |
| --- | --- |
| Dataset | OpenBHB |
| Hugging Face repository | `benoit-dufumier/openBHB` |
| Dataset revision | `8508cda68fea74f217926acbf46ee5863f8879d1` |
| RAMP kit | `ramp-kits/brain_age_with_site_removal` |
| RAMP commit | `0e79d8b400b08afc2d25ff299f92c994aa775cf7` |
| Conservative license rule | CC BY-NC-SA 3.0 plus the most restrictive source-cohort DUA |
| Prepared clean root | `inputs/openbhb/` |

The Hugging Face metadata badge and dataset-card text disagree, so the more
restrictive dataset-card license remains controlling. Data and derived
matrices may not be committed or redistributed under the code license.

## Participant and feature inventory

The complete benchmark is described as 5,330 scans. The pinned public labelled
material used here contains 3,984 participants:

| Role | Participants |
| --- | ---: |
| Official train / adaptive development | 3,227 |
| Public internal validation / one-shot audit | 362 |
| Public external validation / one-shot audit | 395 |
| Total available to this contract | 3,984 |

The pinned files contain 395 external validation rows; an older README value
of 396 is not used.

| Atlas | Development rows | Audit rows | Numeric ROI features |
| --- | ---: | ---: | ---: |
| Desikan | 3,227 | 757 | 476 |
| Destrieux | 3,227 | 757 | 1,036 |

All pinned ROI values were reported finite with no missing cells. Train and
validation feature headers must match exactly within atlas, and Desikan/
Destrieux rows must align by the opaque row mapping before concatenation.

## Prepared-bundle readiness

Before search, verify directly that the prepared bundle matches the declared
OpenBHB and RAMP revisions, row counts, atlas feature counts, headers, opaque
row alignment, and three development split definitions above. Stop on a
concrete mismatch. No checksum inventory or separate source manifest is a
launch gate. Participant IDs and sessions are never model inputs.

## Development roles

The official RAMP JSON defines three overlapping robustness splits:

| Fold | Fit | Internal test | External test |
| --- | ---: | ---: | ---: |
| 0 | 2,622 | 289 | 316 |
| 1 | 2,619 | 290 | 318 |
| 2 | 2,615 | 292 | 320 |

External-test `siteXacq` values in each fold are absent from its fit role;
internal-test values are present in fit. Only 1,185 of 3,227 participants enter
any held-out role, and held-out sets overlap substantially. Scores therefore
measure robustness across predefined views, not three independent cohorts.

The exposed development bundle contains train ROI matrices, opaque row IDs,
training age and `siteXacq`, and the converted development split definitions.
All transforms and predictors are fitted anew on each fold's fit role.

Identifiers, session, age, `site`, `siteXacq`, study, sex, scanner variables,
global tissue volumes, and QC variables are excluded as model features. Age and
fit-role `siteXacq` may be supplied only to the declared supervised fit steps
and trusted probes. No held-out or audit site label is passed to `transform`.

## One-shot audit firewall

The original clean bundle contains 757 audit feature rows with opaque IDs and
no age/site labels. Upstream `participants.tsv`, original participant mapping,
and validation targets were deliberately not retained in the episode
workspace. This is an accidental-leakage boundary rather than cryptographic
secrecy because labels are public upstream.

Before treating those rows as audit evidence, record a dated exposure review
covering prior human access, browser/download caches, upstream scripts,
participant mappings, model notes, quick baselines, and every agent context.
Any row-level target/domain join or validation score visible to a
candidate/controller actor consumes this audit. The rows may then be retained
only as exposed development evidence until a new audit source exists.

For a valid audit:

1. place one winning configuration, evaluator revision, feature headers,
   metrics, thresholds, seeds, and command in a dated write-once record;
2. start a clean process with network egress disabled;
3. do not mount browser caches, Hugging Face caches, the upstream raw source,
   `participants.tsv`, or participant-ID mappings;
4. expose audit ROI features to the locked transform/model, but expose age,
   `siteXacq`, and internal/external roles only inside the trusted evaluator;
5. run all 757 participants once and atomically emit the sealed metrics; and
6. revoke the label mount and prohibit tuning, model selection, or rerun.

The audit evaluator alone creates fixed-seed five-fold probe predictions and
10,000 hierarchical bootstrap draws that resample `siteXacq` and then
participants. Candidate jobs see neither replicate-level results nor partial
metrics. The baseline/candidate refits, folds, seeds, point margins, one-sided
interval rules, and terminal table are fixed in the write-once lock record.

ComBat, site-wise centering/scaling, target-domain normalization, transductive
PCA, or any transform that requires audit-site identity or audit-distribution
statistics is prohibited. Training-site labels may be used to learn a fixed
projection only if its `transform(X)` requires features alone.

## Role table and access status

| Role | Rows | Current exposure | Permitted use |
| --- | ---: | --- | --- |
| Development | 3,227 | exposed/prepared | adaptive search across the three fixed splits |
| Locked-probe tournament | same 3,227 | exposed/prepared | expensive probe evaluation of at most 12 selected configurations |
| Audit features | 757 | prepared without targets | one locked transform/predict call only |
| Audit targets/domain roles | 757 | absent from episode workspace; public upstream | trusted evaluator only after lock |
| Retired RAMP/private leaderboard | not an active endpoint | historical only | no selection or claim |

The clean feature bundle and quick evaluator are already reported ready. A
compliant no-network audit-label handoff is not yet created, so the audit
cannot open. Candidate work must also implement and qualify the adaptive
controller before scored search.
