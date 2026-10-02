# Experiment log

Recorded on: 2026-09-28.

## Identity correction and current handoff

- Identity correction: before the first empirical attempt, the scientist
  replaced the mistakenly supplied Doerig proposal with Wang et al. (2025),
  *Inter-individual and inter-site neural code conversion without shared
  stimuli*, DOI `10.1038/s43588-025-00826-5`.
- Attempt count: zero. No analysis has started, completed, failed, or been
  retried under either identity.
- Empirical access: none. No EP21 handoff, neural value, image feature,
  prediction, reconstruction, identification score, OOD payload, EP04 output,
  or legacy Scratch object has been opened or produced.
- Source review only: the public paper and public code-release metadata were
  inspected to draft the contract. This did not mount or authorize empirical
  data.
- Authorization: contract writing only. Empirical provisioning, empirical
  access, feature extraction, model fitting, scoring, and reconstruction are
  not authorized.
- Active analysis: none.
- Active jobs or local processes: none. No Slurm job has been submitted.
- Current gate: `source_qualification` is blocked before its first attempt on
  explicit authorization and its two outcome-blind input files.
- Neural scoring gate: a valid `outputs/protocol_lock.yaml` is required, and
  that lock would still not grant execution authority.
- Program: `outputs/astra/v0.0.14/astra.yaml`.

## Stable input state

All six declared interfaces are currently unprovisioned:

| Input ID | Expected path | Current state |
| --- | --- | --- |
| `conversion_structural_manifest` | `inputs/conversion_structural_manifest.json` | absent |
| `upstream_pipeline_asset_inventory` | `inputs/upstream_pipeline_asset_inventory.json` | absent |
| `laion_regular_feature_handoff` | `inputs/laion_regular_feature_handoff/` | absent |
| `laion_regular_scoring_handoff` | `inputs/laion_regular_scoring_handoff/` | absent |
| `external_site_feature_handoff` | `inputs/external_site_feature_handoff/` | absent |
| `external_site_scoring_handoff` | `inputs/external_site_scoring_handoff/` | absent |

The first two are the only payload inputs allowed at source qualification.
The external interfaces are independent and optional for completing the
20-directed-pair LAION branch. There is deliberately no OOD input interface.
No execution recipe is declared because no empirical stage is executable now.

## source_qualification

Status: blocked before first attempt.

Blockers: empirical work is not explicitly authorized, and both
`inputs/conversion_structural_manifest.json` and
`inputs/upstream_pipeline_asset_inventory.json` are absent.

Outcome-blind qualification items, not resolved method choices:

- authenticate repository tag `V1.0.0` at commit
  `edbff02edd6a08d3c6c829af45a957fd14bdc460` and the corresponding Zenodo
  archive identity, DOI `10.5281/zenodo.14910040`;
- record the manuscript's ReLU description against the tagged converter's
  `LeakyReLU(0.2)` implementation;
- record the manuscript content-loss weighting description against the tagged
  code's batch/feature-block normalized inverse-squared-norm weighting;
- record the manuscript eight-layer reconstruction description against the
  tagged configuration's nineteen-layer list; and
- verify the completeness of released inter-site implementations and assets,
  including Deeprecon-specific path and configuration assumptions.

Next action: only after an explicit scientist instruction authorizes this
work, provision those two read-only inputs and perform source qualification
without mounting a feature handoff, scoring handoff, neural value, outcome
summary, reconstruction, or OOD payload.

## pipeline_conformance

Status: waiting for `source_qualification`.

Blocker: no qualification report exists. The paper-code discrepancies above
remain qualification facts; none has been adopted as an implementation choice.

Next action: after qualification, resolve every shared field and every field
for each runnable module from outcome-blind evidence, test synthetic or
permuted fixtures, and emit a conformance report plus protocol lock. Missing
assets or unresolved fields must lock only the affected branch or subclaim as
`not_evaluable` unless the unresolved field is shared by all scoring.

## laion_feature_preparation

Status: waiting for the protocol lock, LAION branch qualification, separate
execution authorization, and `laion_regular_feature_handoff`.

Blocker: none of those execution prerequisites exists.

Next action: after all gates pass, prepare only authenticated regular-image
VGG19/AlexNet features and reconstruction-asset references. OOD material
remains physically unavailable.

## laion_decoder_training

Status: waiting for the protocol lock, LAION feature manifest, separate
execution authorization, and `laion_regular_scoring_handoff`.

Blocker: no fitting prerequisite exists.

Next action: fit and freeze each qualified participant's target decoder on
that participant's authenticated subject-unique regular training role only.

## interindividual_converter_training

Status: waiting for the protocol lock, LAION decoder and feature manifests,
separate execution authorization, and `laion_regular_scoring_handoff`.

Blocker: no fitting prerequisite exists.

Next action: fit each qualified directed content-loss converter on the source
participant's disjoint subject-unique images and each brain-loss comparator on
`tau` train only; do not inspect or tune on `tau` test.

## finding_a_feature_decoding

Status: waiting for the protocol lock, fitted LAION objects, feature manifest,
separate scoring authorization, and `laion_regular_scoring_handoff`.

Blocker: no scoring prerequisite exists.

Next action: score within-source, content-loss, and brain-loss decoded VGG19
features on the locked common `tau` test support and emit both the Finding A
report and decoded-feature manifest.

## finding_b_natural_reconstruction

Status: waiting for the protocol lock, authenticated reconstruction assets,
decoded-feature manifest, fixed display manifest, and separate execution
authorization.

Blocker: no reconstruction prerequisite exists.

Next action: reconstruct only the fixed nonselected natural-image set from
decoded features and record every success and failure. Do not use true test
images or their features in optimization, and do not assign a quantitative
success label to this qualitative finding.

## finding_c_natural_identification

Status: waiting for the protocol lock, LAION feature manifest, complete
reconstruction manifest, and separate scoring authorization.

Blocker: no scoring prerequisite exists.

Next action: apply the locked pixel and AlexNet pairwise-identification rule to
the Finding B reconstructions using the complete locked candidate universe.

## artificial_analogue_disposition

Status: waiting for the protocol lock only; this is a non-empirical
disposition, not an artificial-image analysis.

Blocker: no protocol lock exists. No OOD input is requested or permitted.

Next action: emit `blocked_by_ep04_reservation` or
`not_evaluable_under_current_contract` from the locked contract without
opening any raw or derived OOD object or any EP04 state.

## external_feature_preparation

Status: independent external branch; waiting for a runnable external lock,
separate authorization, and `external_site_feature_handoff`.

Blocker: the external handoff is absent and NSD/THINGS have not qualified.
This does not block any LAION analysis.

Next action: if provisioned and qualified, prepare source-compatible external
features and held-out candidate universes independently by dataset and
direction. Otherwise retain the locked `not_evaluable` disposition.

## external_decoder_training

Status: independent external branch; waiting for the protocol lock, external
feature manifest, separate authorization, and
`external_site_scoring_handoff`.

Blocker: no external fitting prerequisite exists. This does not block the
LAION branch.

Next action: if runnable, fit each qualified NSD/THINGS participant's target
decoder on that participant's native training role only.

## intersite_converter_training

Status: independent external branch; waiting for the protocol lock, qualified
LAION and external decoder/feature manifests, both scoring handoffs, and
separate authorization.

Blocker: no inter-site fitting prerequisite exists. The planned 70 directed
pairs remain contingent, not promised.

Next action: if runnable, fit only the four declared LAION-to-external or
external-to-LAION direction blocks with zero cross-training image overlap and
no manufactured shared-stimulus brain-loss comparator.

## finding_d_intersite_reconstruction

Status: independent external branch; waiting for the protocol lock, qualified
intersite converters and decoders, feature manifests, both scoring handoffs,
fixed displays, and separate execution authorization.

Blocker: no reconstruction prerequisite exists. Its absence does not block
Findings A-C or the LAION final report.

Next action: if runnable, reconstruct the fixed held-out images separately by
dataset and direction and record all failures. Finding D remains qualitative.

## finding_e_intersite_identification

Status: independent external branch; waiting for the protocol lock, complete
Finding D reconstruction manifest, both feature manifests, and separate
scoring authorization.

Blocker: no scoring prerequisite exists. Its absence does not block Findings
A-C or the LAION final report.

Next action: if runnable, report the locked pixel/AlexNet identification
endpoints separately by source dataset, target dataset, direction, pair, and
layer; do not pool directions to conceal heterogeneity.

## laion_reproduction_report

Status: waiting for qualification, conformance, the protocol lock, Findings
A-C, and `artificial_analogue_disposition`.

Blocker: no upstream analysis has run.

Next action: once the LAION prerequisites exist, report the ordered Findings
A-C verdicts, artificial-image disposition, external availability, deviations,
failures, evidence dependencies, and claim limits without a scalar episode-wide
label in `laion_report`. This is the complete primary-branch report, not an
integrated A-E artifact. Findings D-E remain separately exported if their
independent branch is provisioned; absent external inputs cannot block this
LAION report. If the external branch is locked runnable, EP21 completion and
the final manuscript synthesis additionally require its separate Findings D-E
reports.


### pipeline_conformance — redundant harness checks removed — 2026-10-02

Cached reuse now consumes only current parent authorization and the completed public result; it does not recompute source/worker/harness hashes, archive stats, interpreter/package-root bindings or parameter-shape inventories. New actual-worker reports use v3; completed v2 artifacts retain their original provenance. Known relevant code/input/runtime/protocol changes must be recorded as stale here and passed as `relevant_change` or retained via `EP21_PUBLIC_CONFORMANCE_RELEVANT_CHANGE` in launch configuration until diagnosis/new authorized attempt; no automatic deletion or retry.

Actual worker asset authentication, strict load/mean/required analytic fixture, source/step resource authority, cache accounting, and the unchanged feature lock/context gate remain. Focused mocked regression: 19 tests passed locally; this is orchestration evidence, not live public-model or scientific validation. See `pipeline_conformance/harness_repair_2026-10-02.md` and the updated three tool files.

No job, empirical payload access, lock, scientific/resource contract amendment, successor, or ASTRA re-export was performed. Existing completed work and unchanged ASTRA program/export are reused. The previously recorded combined-allocation proposal and unresolved real lock/context integration remain the next execution conditions; this harness edit supplies neither.
