# Dataset Contract — Episode 11

This contract does not provision, expose, or authorize any data. Role and
access rules follow
[`../ADAPTIVE_SEARCH_PROTOCOL.md`](../ADAPTIVE_SEARCH_PROTOCOL.md).

## Pinned primary resource

The intended resource is SEU-A1876, Zenodo record `13944322`, associated with
Jiang et al. (2025) and reported by the API as CC BY 4.0. Exact API and license
responses must be retained at provisioning.

| File | Bytes | Provider MD5 | Intended role | Current state |
| --- | ---: | --- | --- | --- |
| `Full_morphometry.xlsx` | 181,376 | `dcad84366865aa6ffc4d5b010159dd9b` | identity, soma, layer, QC metadata | absent |
| `Full_morphology_RAW.zip` | 1,679,789,577 | `40139b77ca8752aa28cff487e0517bae` | native-morphology sensitivity | absent |
| `Full_morphology_CCFv3.zip` | 1,536,231,503 | `a5d2242516268a301e0efc16438990dd` | primary full registered axons | absent |
| `Axonal_arbor_CCFv3.zip` | 24,463,661 | `6b6d903884043d574320f07b8f220451` | validated representation sensitivity only | absent |

Allen Mouse CCFv3 annotation, ontology, axes, resolution, and hashes are also
required and currently absent. The peta-voxel source images are not required.

## Known inventory, not assumed replication

Prior metadata inspection reported 1,876 unique morphologies, 39 `fMOST Brain
ID` values, and 92 soma-region labels. Counts ranged from 1 to 225 cells per
brain ID (median 22); 1,736 cells were manually checked, 140 were unchecked,
and 511 had cortical-layer values. These are release descriptors, not the
eligible sample or number of animals.

`fMOST Brain ID` may define an evidence group only after its
animal/specimen/experimental mapping is authenticated. IDs from the same
animal are collapsed; linked or unresolved IDs are collapsed or excluded by a
rule frozen before outcomes. Neurons, arbors, branches, atlas regions, and
spatial blocks are never independent biological groups.

## Shared EP09–11 role ledger

After identity resolution and before axonal outcomes are inspected, one
immutable ledger must assign every eligible **complete biological group** to:

- `development`: visible to group-held-out adaptive search;
- `audit_sealed`: unavailable to candidates and opened once by the trusted
  evaluator; or
- `excluded`: with a prospective reason.

The assignment algorithm, seed, duplicate families, group mapping, and
exposure timestamps are shared across EP09, EP10, and EP11. There is no fresh
audit if a relevant outcome was previously exposed through any sibling. The
audit cannot be synthesized by hiding random neurons from an already exposed
group. The common ledger must retain at least **12** complete development
groups and **8** complete sealed-audit groups, using the same assignments for
EP09--EP11. If verified groups do not support both roles and these minima, this
episode stops before search.

## Outcome and predictor provenance

Match each neuron across metadata, RAW morphology, CCFv3 morphology, and any
provider-arbor file. The record must carry immutable source ID, verified
biological group, RAW/CCF soma positions, source anatomy, quality/completeness,
acquisition/registration batch, duplicate lineage, source paths, and exclusion
reason.

The primary outcome is an axon-only representation constructed from
authenticated full axons. Provider arbor files are derived by spectral
clustering and are not automatically terminal arbors; they require validation
against matched full axons. Passing fibers, ambiguous cells, and truncated
cells follow frozen rules and are not silently encoded as absent targets.

The released axon-derived `Projection class` is forbidden as a predictor,
split rule, target rule, imputation field, or validation label. Any
axon-derived completeness or review field used to decide outcome observability
cannot also be a predictor unless independent prospective provenance is
established.

## Access boundary

| Role | May be accessed for | Must remain hidden |
| --- | --- | --- |
| Metadata/readiness | identity, grouping, duplicates, source/QC provenance, feasibility | all real development and audit axonal outcome summaries |
| Synthetic qualification | grammar recovery, calibration, threshold and failure-rule design | all real outcomes |
| Development | group-held-out fitting, representation learning, trial selection, null design | every `audit_sealed` outcome and score |
| Lock | manifest, code/environment, winner, thresholds, seeds, decision rules | audit outcomes |
| One-shot audit | trusted evaluation of the locked package | labels and outcomes from candidate/search processes |

Audit outputs may be emitted only after the final lock hash is recorded. Any
audit read during development invalidates fresh-audit status and is logged as
an exposure, not repaired by relabeling.

## Required readiness artifacts

Provisioning must preserve provider bytes/MD5 values, compute local SHA-256,
and keep archives unchanged. Before a launch decision, the episode needs:

- exact resource and license manifests;
- verified cell-to-file and cell-to-biological-group tables;
- duplicate/overlap checks against any other morphology source used by the
  campaign;
- authenticated CCF assets and coordinate conventions;
- a full-axon/arbor interpretation report;
- the shared EP09–11 exposure and role ledger; and
- a feasibility report showing that the declared grammar and audit have
  enough independent groups and spatial support.

CCF-ME v2 (`13801372`) is dendrite-focused and cannot serve as an external
audit of this axonal claim. No other audit asset is asserted by this contract.
