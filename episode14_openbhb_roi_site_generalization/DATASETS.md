# Dataset and Evidence Roles — Episode 14

EP14 uses the pinned OpenBHB FreeSurfer ROI bundle already provisioned for this
episode. This document declares data roles; it does not authorize target
access, provision new data, or turn public labels into fresh audit evidence.

## September 30 representation refocusing — design only

The main question returns to transferable age representations built from the
allowed FreeSurfer ROIs. Proposed axes include bilateral shared/contrast
components, provider-declared ROI/measurement-family grouping, atlas
granularity and shallow predictor interactions. The original 3,227/757 roles
and frozen four-arm reference remain unchanged. No new source, feature payload,
outcome access or executable search is provisioned by this writing revision.

Construct new features only from allowed ROI columns and declared mappings.
Do not import excluded global tissue-volume/QC/sex fields or infer connectivity
from a regional table. ROI grouping is not physical cortical coarse-graining;
that would require surface data outside the current bundle. A complete
bilateral basis contains the same information as its original pairs.

The [representation design](outputs/representation_research_design.md) keeps
search broad while separating anatomical organization from generic shrinkage,
compression and predictor capacity. Search ranges and confirmation scope
remain prospective choices, not newly authorized holdout endpoints.

### Optional prior site diagnostic

The retained optional diagnostic would use development ages to define fit-only shared
age support and to fit auxiliary discarded-score age models and age-conditional
site probes. Seen-domain rows can evaluate conditional site prediction; unseen-
domain rows can evaluate age transfer. Within-domain centered age scoring is an
evaluator-only description, not a target-dependent representation or the
operative age-MAE endpoint. A domain's mean age cannot enter the predictor.

Both unseen-domain and within-domain utility on the same declared common age
support are required to distinguish useful discarded information from a
domain-average age shortcut. Support and age variation must be reported per
domain; insufficient overlap is a scientific limitation, not a reason to
silently drop inconvenient domains. All fitting, weights, directions, and any
residualization remain inside training roles. The diagnostic is unactivated
until its prospective amendment is frozen.

The current conditional holdout authorizes no new diagnostic endpoint. A new
endpoint on those rows requires an explicit pre-outcome amendment, a valid
exposure review, and a compatible evaluator scope. Otherwise these diagnostics
are development or retrospective evidence and need genuinely new confirmation
material. Conditional site prediction cannot identify causal scanner effects.

## Fixed upstream material

| Item | Fixed value |
| --- | --- |
| Dataset | OpenBHB |
| Hugging Face repository | `benoit-dufumier/openBHB` |
| Dataset revision | `8508cda68fea74f217926acbf46ee5863f8879d1` |
| RAMP kit | `ramp-kits/brain_age_with_site_removal` |
| RAMP commit | `0e79d8b400b08afc2d25ff299f92c994aa775cf7` |
| Conservative license rule | CC BY-NC-SA 3.0 plus the most restrictive source-cohort DUA |
| Prepared root | `inputs/openbhb/` |

The upstream badge and dataset-card license text disagree, so the more
restrictive dataset-card rule controls. Participant data and derived matrices
must not be committed or redistributed under the repository's code license.

## Participant and feature inventory

The pinned public labelled material contains 3,984 participants:

| Role | Participants |
| --- | ---: |
| Official training / exposed development | 3,227 |
| Public internal validation / conditional holdout | 362 |
| Public external validation / conditional holdout | 395 |
| Total | 3,984 |

The source benchmark describes 5,330 scans, but rows outside the pinned 3,984
are not part of this contract. The prepared bundle contains 395 external rows;
an older README value of 396 is not used.

| Atlas | Development rows | Holdout feature rows | Allowed numeric ROIs |
| --- | ---: | ---: | ---: |
| Desikan | 3,227 | 757 | 476 |
| Destrieux | 3,227 | 757 | 1,036 |

The primary analysis uses Desikan. Destrieux is a prespecified sensitivity
analysis and cannot determine the primary winner. Atlas rows must align through
the prepared opaque mapping; participant identifiers are never model inputs.

Identifiers, session, age, `site`, `siteXacq`, study, sex, scanner variables,
global tissue volumes, and QC variables are excluded from predictor features.
Age and fit-role `siteXacq` may be supplied only to their declared supervised
fit steps and fixed probes.

## Exposed development roles

The official RAMP definitions provide three overlapping robustness views:

| View | Fit | Internal test: seen `siteXacq` | External test: unseen `siteXacq` |
| --- | ---: | ---: | ---: |
| 0 | 2,622 | 289 | 316 |
| 1 | 2,619 | 290 | 318 |
| 2 | 2,615 | 292 | 320 |

Only 1,185 of the 3,227 development participants enter any held-out role, and
the held-out sets overlap. These views are not independent cohorts. For the
primary aggregate, compute each appearance-level age loss or site-classification
correctness first, then average those values within repeated participants.
Predictions or class probabilities are not averaged into a participant-specific
ensemble. Each view and held-out domain remains visible as a robustness
description.

Every learned step—standardization, PCA, site-direction estimation, ridge
tuning, age fitting, and site probing—is fit anew inside the corresponding
fit role. Internal-test and external-test rows supply no fitted state and no
hyperparameter choice. A held-out transform receives ROI values only.

The primary external age estimand weights held-out `siteXacq` domains equally
after participant-level loss aggregation. Participant-pooled MAE is secondary.
The domain count, support, every domain effect, and leave-one-domain-out
sensitivity must be reported. All inference is conditional on the observed
domains; the bootstrap does not create a population-of-sites sample.

## Prepared input boundary

The following paths have fixed roles:

| Path | Role | Current access |
| --- | --- | --- |
| `inputs/openbhb/exposed/` | development ROI matrices, development age/`siteXacq`, split definitions, atlas reference text | available for an authorized development run |
| `inputs/openbhb/audit_features/` | 757 public-validation feature rows with opaque identities | feature-only; use only after candidate lock |
| public-validation age/`siteXacq`/role labels | trusted evaluator input | absent from the episode workspace and closed |

`inputs/` is read-only. Durable code, reports, predictions, and figures belong
under `outputs/`; transient execution belongs under the episode-specific
Scratch directory.

Before development execution, the lightest necessary checks are row counts,
atlas feature counts, headers, finite values, opaque row alignment, bilateral
mapping, and exact split membership. Stop on a concrete mismatch. A new
checksum inventory, source mirror, or provenance chain is not a launch gate.

## Conditional public holdout

The 757 feature rows are public upstream, so their audit role depends on actual
outcome exposure rather than nominal file placement.

Before any public-validation target is retrieved or mounted, a dated exposure
review must cover every designer, controller, process, accessible cache,
browser/download history, participant mapping, prior artifact, and agent
context that could have seen a row-level target join, candidate score, or
partial outcome.

- If freshness is established, the 395 external rows may support one paired
  age-error comparison and the 362 internal rows may support the fixed linear
  site-probe guardrail after one candidate is locked.
- If exposure is uncertain or failed, all 757 rows are retrospective
  validation evidence. They may not be called a one-shot audit, and a new
  source is required for confirmation.

For an eligible one-shot evaluation:

1. lock the candidate, R0, code revision, feature headers, atlas mapping,
   transforms, seeds, metrics, margins, bootstrap rule, and evaluator command;
2. use a clean process with network egress disabled and without upstream raw
   caches or participant mappings;
3. expose ROI features to the locked transform and predictor, but expose age,
   `siteXacq`, and roles only inside the trusted evaluator;
4. compute the 395-row external age comparison and 362-row internal probe once;
5. emit the complete result atomically and revoke the label mount; and
6. prohibit model, threshold, representation, or analysis changes after any
   candidate-discriminating outcome becomes visible.

A pre-outcome technical failure may be repaired without consuming the holdout.
Once a score, partial metric, prediction-target join, or candidate-specific
outcome is visible, freshness is consumed.

## Evidence roles

| Evidence | Permitted role |
| --- | --- |
| three RAMP views | development comparison and robustness description |
| fixed sanity controls | implementation and falsification checks |
| Destrieux repeat | prespecified sensitivity only |
| fresh one-shot 757-row evaluation | narrow confirmation of one locked comparison |
| exposure-compromised 757-row evaluation | retrospective validation only |

Reduced linear probe accuracy is not proof of scanner invariance. External age
error is not evidence of biological aging mechanism. Neither development nor
public holdout performance establishes generalization to diseases, raw MRI,
clinical settings, or acquisition domains absent from the pinned material.
