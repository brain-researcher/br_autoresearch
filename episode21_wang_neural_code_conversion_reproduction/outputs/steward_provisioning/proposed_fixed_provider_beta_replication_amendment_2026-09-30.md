# Proposed EP21 fixed-provider-beta replication amendment

Status: **proposed, not adopted**. No beta array has been opened; this document does not authorize export, model fitting or scoring, and changes no resource allocation.

## Purpose and correction

The scientist clarified: “我们的目的只是replication”. EP21 is a fixed adapted retrospective reproduction on LAION-fMRI. Requiring a bespoke release with provably training-only upstream beta estimation is a stricter alternative preprocessing design, not an automatic requirement of replication. Provider GLMsingle measurement estimation and EP21 decoder/converter fitting must be distinguished. Nevertheless, the current written contract expressly forbids OOD contributions to upstream normalization and has not adopted a tau-test exception. The original scientist instruction preserved that boundary, so it cannot be silently reinterpreted.

## Concrete proposed choice

Use regular-image rows from the already acquired, fixed provider GLMsingle v1.2 final TYPED `betasmd` release in native space, in percent signal change. Preserve the authenticated trial/image join, tau variant 0, repetitions, native voxel identities and the approved independent masks. Treat those supplied betas as fixed measurement inputs to this adapted replication. Do not require a custom replacement beta release or raw-BOLD reprocessing merely to prove upstream train/test exclusions.

Accept and disclose the limitation that the provider's original HRF/library, nuisance-PC, ridge/autoscaling and response-scaling pools have not been established to exclude regular tau-test or OOD responses. The representative metadata lists mixed-role runs and a mean BOLD reference across runs; it does not establish the actual exclusion pools. This proposal accepts possible inherited upstream dependence. It does not establish clean estimation, prove contamination, quantify its effect, or establish equivalence to the source study's preprocessing.

## Scope of the exception

The exception applies only to inherited dependence in the fixed, previously supplied provider betas. EP21 may export/read only selected regular-response rows through a role-isolated steward interface under separately approved resources. No OOD images, response rows, predictions, features, maps, preprocessing arrays, reference statistics or results may enter an EP21 handoff, scorer, protocol resolver or chat. The blanket prohibition on every OOD derivative gains only this explicit ancestry exception for fixed regular-row provider measurements. These measurements must not be labeled OOD-independent. No new OOD-dependent preprocessing may be fitted by EP21. No EP04 or legacy Scratch input is permitted.

Decoder/converter response and feature normalization, voxel ranking, learned models, tuning and stopping remain confined to their declared training roles. Tau-test responses remain evaluation-only for these EP21 fitted objects. The exact source/target training ID disjointness and tau split requirements remain intact. The independent mask amendment is unaffected.

## Required contract changes if adopted

1. Scope `SEARCH_POLICY.yaml:preprocessing_policy`'s blanket upstream normalization exclusion to a documented fixed-provider-measurement exception; preserve strict exclusions for every EP21 learned transform and direct OOD access.
2. Set the provider preprocessing disposition in `PROTOCOL_TABLE.yaml:response_preprocessing` and `outputs/protocol_decisions_2026-09-30.yaml:delegated_LAION_adaptations.response_and_ROI` to accepted fixed upstream measurement with mixed-role exclusions unverified, instead of automatically requiring an eligible replacement or making an endpoint not evaluable for that uncertainty alone.
3. Update DATASETS/GOAL/input documentation and the pipeline-conformance ASTRA dependency consistently; retain original reports and their unknown eligibility facts as historical evidence.

The affected table is `PROTOCOL_TABLE.yaml:response_preprocessing`; no amendment is applied by this proposal.

## Reporting and next step

Call results an adapted retrospective replication conditional on standard provider betas. Do not claim an independently estimated test response, fresh confirmation, complete upstream train/test isolation, absence of inherited OOD influence, or an OOD/generalization result. No comparability margin is introduced. No successor, different model family, new scientific budget or relaxed EP21 train/test fit rule is proposed.

After an explicit scientist decision, record the chosen measurement definition and proceed to a concrete bounded regular-only neural steward export proposal. A protocol lock and the applicable separate execution/resource authorization are still required. This amendment alone cannot start empirical execution.
