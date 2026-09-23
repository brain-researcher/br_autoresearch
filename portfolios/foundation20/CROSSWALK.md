# Foundation-20 crosswalk

This crosswalk separates primary scientific questions from local episode
numbers. It records direct, bounded, partial, and adjacent coverage, plus topics
that have no active episode yet.

Terminology note: retained phrases such as `pre-launch`, `launch-blocked`, or
`not launch-ready` describe readiness for a scored/confirmatory stage in this
historical inventory. They do not block explicit `bin/codex-episode` startup;
the corresponding checks now run inside the episode.

## Coverage summary

- Direct/exact pre-launch operationalization: F20-02, F20-04, F20-05, and
  F20-17.
- Bounded adaptive operationalizations: F20-01, F20-11, F20-12, F20-13, and
  F20-19.
- Adjacent but not equivalent: ep04, ep05, ep09, ep10, and ep12.
- Formal direct or bounded coverage: 9 of 20 questions; 11 have no formal
  direct or bounded episode yet.
- Data readiness is not episode readiness: F20-02 has an acquired but
  role-mixed derivative release; F20-03 has a partially usable local
  derived-data canary; F20-16, F20-17, and F20-18 have acquisition-verified
  shared source releases; F20-20 has its source tables; and F20-05 has a
  acquisition-verified CNeuroMod restricted raw source but no role-filtered
  beta/model/audit handoff. None is currently launch-ready.

| ID | Foundation question | Current coverage | Physical episode, historical evidence, or missing asset |
| --- | --- | --- | --- |
| F20-01 | Analysis variability vs weakly identifiable effects | **Bounded adaptive pre-launch operationalization** | The new ep01 adaptively searches exact-kernel, stage, masking, scaling, estimation, aggregation, residual, and reliability explanations for smoothing-induced map change, then performs one locked `ds000005` transport audit. The two old NARPS runs are fully exposed priors and count as no new evidence. This episode addresses one important analysis-flexibility mechanism but does not adjudicate weak identifiability in general. |
| F20-02 | Shared cross-task representational geometry | **Exact pre-launch** | ep15 distinguishes hierarchical parcel membership, smooth relocation, certified isometry, and bounded stable non-isometry using reliable between-person Task-B-only map and condition-geometry estimands. The source release is acquired, but the 12/12 role manifest, role-filtered handoffs, exposure verification, scientific margins, and a permission-separated audit runner are absent. ep04 remains adjacent, not equivalent. |
| F20-03 | Stable individual functional fingerprints | **No episode; bounded source partially provisioned** | Local `ds000114` arrays can support only a short-interval, same-scanner exploratory canary. The 280 effect rows are 20 participant-session observations x 14 uneven task-contrast cells, not 280 independent samples; their source NIfTI paths are stale and affine, masks, confounds, QC, completeness, and provenance still need recovery. Exact IBC validation remains unprovisioned. |
| F20-04 | Condition semantics to effect-map geometry | **Exact pre-launch** | ep03; NeuroEffect corpus exists remotely, but immutable Sherlock handoff and taxonomy are absent. |
| F20-05 | Measurement and model-ranking stability | **Exact pre-launch** | ep17 uses CNeuroMod-only B/C/D, support, and ceiling edges with 480 development, 120 calibration, and 120 sealed concepts. Its no-refit audit tests held-out concepts in the same four people, not external transport. Restricted raw acquisition and the separately governed original/CNeuroMod stimulus archives are verified; controlled image extraction and exact event alignment, the taxonomy, role-filtered handoffs, model/ROI/ceiling/margin and terminal-contract manifests, exposure governance, and the evaluator remain blockers. |
| F20-06 | More unique images vs more repetitions | **Missing** | NSD scan-budget episode not created. |
| F20-07 | Session scaling vs longitudinal drift | **Missing** | NSD session/drift episode not created. |
| F20-08 | Shared prior vs personalized encoding | **Missing** | NSD cross-subject calibration episode not created. |
| F20-09 | Pre-fixation semantic selection | **Missing** | AVS data not provisioned. |
| F20-10 | Concept representation vs image-specific features | **Incomplete draft; source EEG and images acquired** | Proposed ep18 has the complete 59.4-GB THINGS-EEG1 release plus the original THINGS image archive, metadata, and THINGSplus-CC0 control archive acquisition-verified and quarantined, and its draft search policy now exists. Input/output guards, the seven workspace projections, controlled image extraction, the exact image-event join, participant roles, role-filtered views, and the shared EP17/EP18 exposure ledger remain absent. It is not a current episode or launch-ready. |
| F20-11 | Region-specific spectral-latent fingerprints | **Bounded pre-launch operationalization** | ep06; M1/PMd is tested in only two animals, while area2 cross-region attribution is task/animal-confounded. |
| F20-12 | Stable mapping vs session-specific fitting | **Bounded pre-launch operationalization** | ep07 tests within-animal transfer in behavior-anchored residual coordinates; common LFP asset pack and frozen alignment specification are absent. |
| F20-13 | Electrode/trial scaling law | **Bounded adaptive operationalization** | ep08 learns post-pilot electrode-retention and calibration-trial allocation policies across a fixed resource grid; common asset pack and sequential replay evaluator remain absent. |
| F20-14 | Shared motor-imagery geometry across people | **Missing** | Human BCI dataset and split not provisioned. |
| F20-15 | Pre-movement intention vs movement leakage | **Missing** | All 55 published AJILE12 assets are materialized in steward quarantine with final aggregate verification pending at the 2026-09-22 inventory update, but the public derivatives lack the raw filtering and reconstructable onset provenance needed for this question. No qualifying episode or onset/leakage contract is provisioned; EP19 treats AJILE12 only as a deferred exploratory extension. |
| F20-16 | Model-free learning vs latent-state inference | **No episode; shared source acquired** | The Rajagopalan *Drosophila* operant-matching release (Zenodo 7449214) is local and acquisition-verified. Episode-specific extraction, trial-schema QC, an immutable handoff, and a split/exposure ledger remain missing. |
| F20-17 | Adaptive learning rate causal mechanism or correlate | **Direct/exact pre-launch; launch blocked** | Formal ep02 pins the Dudman Janelia Figshare 21816054 v1 source and official cohort map and freezes a 28--48-trial adaptive policy. Exact role-pack materialization passed integrity checks for 9 development controls, an intended atomic 6 `stimLick-` versus 5 `stimLick+` primary audit, and a non-rescuing 4-mouse boundary diagnostic, but the handoff remains `materialized_unsealed` because same-UID source access and network reacquisition defeat confidentiality. Generic calibration v2 returned `selected_rule: null` (0/216 eligible), so no audit or positive terminal is authorized. Endpoint/time semantics, an assignment-mechanism or presigned conditional-exchangeability receipt, endpoint-faithful calibration with an absolute raw-unit margin and scientist signoff, the author-code port/configuration and resource profile, a permission-separated evaluator, and canonical binding remain blockers. |
| F20-18 | Working-memory content vs behavioral state | **No episode; shared source acquired** | The Kathman processed *Drosophila* two-photon/behavior release (Zenodo 20053990) is local and acquisition-verified. Neural-behavior synchronization QC, nuisance/state definitions, an immutable handoff, and a split/exposure ledger remain missing. |
| F20-19 | Discrete neuron types vs continuous anatomical gradients | **Bounded pre-launch operationalization** | ep11 can test reusable residual morphology modes versus continuous-unimodal gradients, not molecular neuron types; SEU-A1876/CCF payload and biological identity mapping are absent. |
| F20-20 | Biological constraints vs spatial autocorrelation | **Source corpus available; maps/design not ready** | Neurosynth v7 coordinates, metadata, and abstract-term TF-IDF are local, but these are not frozen cognitive maps. A canonical source manifest, term/concept curation, version-pinned map-generation contract, biological annotations and transforms, publication-disjoint held-out-family ledger, and representation-matched spatial-null bank are absent. |

## Foundation-adjacent episodes retained on purpose

- ep04 asks a distinct scene-level cross-modal geometry question.
- ep05 is an active technical/scientific LFP precursor, not one of the three
  Foundation LFP questions.
- ep09 tests incremental local-to-distal projection prediction.
- ep10 tests residual single-cell target combinations.
- ep12 tests whether curated cell-type aggregation is predictively sufficient
  for individual-neuron wiring in the single-specimen MaleCNS graph. It is not
  a projection-morphology substitute for F20-19 and is unrelated to the
  Rajagopalan learning-mechanism question in F20-16.

These are legitimate primary questions but must not be counted as completed
Foundation-20 topics.
