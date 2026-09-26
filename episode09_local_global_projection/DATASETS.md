# EP09 dataset contract

This episode follows [the common adaptive protocol](../ADAPTIVE_SEARCH_PROTOCOL.md).

## Fixed real sources

| Source/file | Release | Role |
| --- | --- | --- |
| SEU-A1876 | Zenodo `13944322` | primary same-cell dendrite and axon source |
| `Full_morphometry.xlsx` | file in Zenodo `13944322` | identities, brain IDs, soma, layer, QC |
| `Full_morphology_RAW.zip` | file in Zenodo `13944322` | primary native dendrites |
| `Full_morphology_CCFv3.zip` | file in Zenodo `13944322` | CCF full axons and transform checks |
| `Axonal_arbor_CCFv3.zip` | file in Zenodo `13944322` | candidate distal-arbor outcome source |
| `Dendritic_arbor_CCFv3.zip` | file in Zenodo `13944322` | secondary method sensitivity |
| Allen Mouse CCF | exact CCFv3 annotation/ontology to be pinned | atlas assignment and geometry |
| CCF-ME | Zenodo `13801372`, version 2 | optional M2 sensitivity after lineage audit |

Record the provider release, file names, reuse terms, and read-only local
locations. Released metadata report 1,876 unique morphologies across 39
`fMOST Brain ID` values, 92 soma regions, 1,736 manually checked cells, and 308
cells missing `Projection class`; these are inventory counts, not an eligible
sample.

## Exposure and shared-outcome caveat

EP09, EP10, and EP11 can expose the same SEU axonal targets. Maintain one
shared pre-outcome ledger with stable cell IDs, conservative biological groups,
duplicates, source lineage, exposure, and whole-group roles. Audit groups are
not independent across those episodes. Opening them for any one episode
contaminates the others unless all three contracts and configurations were
already locked. Historical output, cached features, human inspection, and
external publications/models with overlapping neurons must be declared.

## Development and audit roles

Split by the most conservative verified animal/brain/specimen unit, never by
neuron. The primary design requires at least 12 development and 8 sealed audit
groups with source-population support on both sides. Group roles must be
assigned from identity, coverage, and technical feasibility before any
dendrite–axon association or target prevalence is inspected.

Development groups build target-support rules, tune the observation pipeline,
and drive adaptive search. Audit-group axonal outcomes and derived targets live
in a permission-separated store until lock. If group provenance or the minimum
counts fail, downgrade prospectively to grouped development evidence or stop;
do not redefine a brain ID as an animal or use random cell splits.

## Outcome and predictor construction

For each cell, independently parse native dendrites and CCF axons, validate
tree semantics, coordinate units/orientation, soma registration, clipping,
disconnected branches, terminal capture, and provider-arbor linkage. Pin a
non-overlapping CCF target vocabulary and a qualifying distal-arbor rule using
development groups only.

Every cell-target pair must be labeled `detected`, `not_detected`, or
`uncertain`. A valid zero requires an observable reconstruction and target;
uncertain stays missing. Preserve target support by positive and negative
independent groups. `Projection class` and every other axon-derived label are
forbidden predictors, split fields, and imputation aids.

Predictors must come from native dendrites plus prospectively available soma,
source, layer, independent label, quality, and acquisition records. Any CCF
dendrite feature is a locked sensitivity. External morphology representations
require verified cell/SWC identity mapping, topology and geometry comparison,
and publication-lineage deduplication.

## Firewall

Provision read-only payloads for SEU-A1876, Allen CCFv3, provider methods/code,
biological metadata, and the shared split ledger. A trusted evaluator
constructs development predictions and scalar/group diagnostics; candidate
jobs do not receive group-audit outcomes or target prevalence. Seal audit
mappings, outcomes, feature caches derived from them, metrics, and inference
artifacts under separate permissions. Use the verified cell/group ledger and
frozen coordinate transform for every join.

## Missing blockers

The primary archives are not yet provisioned episode-locally; exact
CCF annotation/structure graph and registration code are unpinned; biological
animal/specimen mapping and batch metadata are unresolved; qualifying-arbor
and valid-nondetection rules are unvalidated; and the shared EP09/10/11 role
and exposure ledger does not exist. Each blocks candidate scoring or protected
audit access, not explicit task startup. Optional CCF-ME absence does not block
the primary analysis.

Large payloads remain outside Git or behind immutable read-only references.
Transient expansions and feature matrices belong in
`$SCRATCH/br_autoresearch/episode09_local_global_projection/`; durable
outputs are source and role records, parsers/tests, ledgers, the write-once lock,
and reports.
