# Dataset Contract — Episode 11

This contract does not provision, expose, or authorize any data. Role and
access rules follow
[`../ADAPTIVE_SEARCH_PROTOCOL.md`](../ADAPTIVE_SEARCH_PROTOCOL.md).

## Pinned primary resource

The intended resource is SEU-A1876, Zenodo record `13944322`, associated with
Jiang et al. (2025) and reported by the API as CC BY 4.0. Exact API and license
responses must be retained at provisioning. The official files have been
acquired and hash-verified in steward mixed-role quarantine, but none is
provisioned to this episode or safe for candidate access. “Acquired” below does
not mean “episode-ready.”

| File | Bytes | Provider MD5 | Intended role | Current state |
| --- | ---: | --- | --- | --- |
| `Full_morphometry.xlsx` | 181,376 | `dcad84366865aa6ffc4d5b010159dd9b` | identity, soma, layer, QC metadata after trusted redaction | acquired in steward quarantine; absent episode-locally; mixed-role |
| `Full_morphology_RAW.zip` | 1,679,789,577 | `40139b77ca8752aa28cff487e0517bae` | native-morphology sensitivity | acquired in steward quarantine; absent episode-locally; mixed-role |
| `Full_morphology_CCFv3.zip` | 1,536,231,503 | `a5d2242516268a301e0efc16438990dd` | primary full registered axons | acquired in steward quarantine; absent episode-locally; mixed-role |
| `Axonal_arbor_CCFv3.zip` | 24,463,661 | `6b6d903884043d574320f07b8f220451` | validated representation sensitivity only | acquired in steward quarantine; absent episode-locally; mixed-role |

Allen Mouse CCFv3 annotation, ontology, axes, resolution, and hashes are also
required. A 25-µm 2017 CCF asset is present in steward quarantine but has not
been validated or provisioned for EP11. The peta-voxel source images are not
required.

## Known inventory, not assumed replication

Prior metadata inspection reported 1,876 unique morphologies, 39 `fMOST Brain
ID` values, and 92 soma-region labels. Counts ranged from 1 to 225 cells per
brain ID (median 22); 1,736 cells were manually checked, 140 were unchecked,
and 511 had cortical-layer values. These are release descriptors, not the
eligible sample or number of animals.

The workbook exposes cell name, `fMOST Brain ID`, RAW and CCFv3 soma
coordinates, manual-check status, soma region, cortical layer, and the forbidden
axon-derived `Projection class`. It does not provide an authenticated mapping
from brain ID to animal/specimen or the shared development/audit ledger. A
trusted builder must redact outcome-adjacent fields before emitting any
metadata-only view.

`fMOST Brain ID` may identify an independent animal only after its
animal/specimen/experimental mapping is authenticated. IDs from the same
animal are collapsed; linked or unresolved IDs are collapsed or excluded by a
rule frozen before outcomes. A conservative unresolved specimen group can
support feasibility or a separately redesigned cross-group analysis, but it
cannot count toward EP11's cross-animal 12/8 minima. Neurons, arbors, branches,
atlas regions, and spatial blocks are never independent animals.

## First action: source-specific support census

Before opening projection outcomes, a trusted builder must produce a redacted
source-by-independent-animal support table for every candidate source
population. For each animal it must report eligible and excluded cells,
soma-coordinate range and density, cortical layer or depth when applicable,
sex, age, strain/genotype or Cre line, labeling strategy, imaging modality,
hemisphere, source laboratory, acquisition/registration batch, reconstruction
version, derived-feature lineage, and overlap with other animals. It must
explicitly mark positions or layers where plausible intermediate neurons were
not sampled. The source definition must state which experimental strata it
pools and which are adjustment or falsification variables.

A provisional source-by-`fMOST Brain ID` table may diagnose what lineage
information is missing, but brain ID is only a proxy until its animal/specimen
mapping is authenticated. It cannot establish cross-animal support, choose the
development/audit split, or satisfy the 12/8 minima.

One source population may proceed only if at least 12 development and 8 sealed-
audit verified independent animals contain eligible cells within a frozen
overlapping soma-position/depth domain. These counts are evaluated inside the
selected source, not over the full dataset, and are necessary but not
sufficient. Outcome-blind precision and synthetic work must freeze non-null
minimum effective cells per animal, development/audit animals per spatial
neighborhood or layer, mutual-support and cross-animal-neighbor metrics, a
maximum position/depth gap, tolerable missing/unknown/truncated fraction,
animal--batch aliasing, and precision for the meaningful effect. Each putative
projection label must have multi-animal development support. The same rules are
applied to every candidate source and may not use apparent projection gaps.
Until this artifact passes every rule, EP11 makes no claim that cross-animal
adjudication is feasible. A valid census with no passing source deterministically
ends as `closed_insufficient_source_support`; corrupt identity or lineage data
is instead a technical failure.

The 12/8 minimum is a requirement of the current EP11 design, not a universal
biological threshold. Its precision and recovery basis must be documented. A
failed census may motivate a prospectively versioned narrower question, revised
sample-size rule, or added resource before projection outcomes are opened; it
cannot justify relaxing the threshold after looking at projection patterns.
`closed_insufficient_source_support` describes the present data--design match
and carries no conclusion about whether projection groups exist.

## Shared EP09–11 role ledger

After identity resolution and before axonal outcomes are inspected, one
immutable ledger must assign every eligible **verified independent animal** to:

- `development`: visible to animal-held-out adaptive search;
- `audit_sealed`: unavailable to candidates and opened once by the trusted
  evaluator; or
- `excluded`: with a prospective reason.

The assignment algorithm, seed, duplicate families, group mapping, and
exposure timestamps are shared across EP09, EP10, and EP11. There is no fresh
audit if a relevant outcome was previously exposed through any sibling. The
audit cannot be synthesized by hiding random neurons from an already exposed
animal. The common ledger must retain at least **12** complete development
animals and **8** complete sealed-audit animals, using the same assignments for
EP09--EP11. If verified animals do not support both roles and these minima,
this episode stops before search.

## Outcome and predictor provenance

Match each neuron across metadata, RAW morphology, CCFv3 morphology, and any
provider-arbor file. The record must carry immutable source ID, verified
animal, RAW/CCF soma positions, source anatomy, quality/completeness, the
experimental strata listed above, acquisition/registration batch, duplicate
and reconstruction lineage, source paths, and exclusion reason.

The primary outcome is one fixed quantitative regional projection-distribution
vector constructed from authenticated full axons on a frozen non-overlapping
target vocabulary. Every adjudicative model predicts this same object under the
same fixed coordinate map, dimension, and density measure. Target resolution,
laterality, normalization, zero/missingness handling, and the distinction among
terminal arbor, passing fiber, unknown, and absence are selected from atlas
anatomy, measurement provenance, reconstruction QC, and outcome-blind
precision only, then frozen before any real regional projection summary is
opened. “Target observability” means complete-axon and registration QC, not
regional target mass. Any future outcome-informed construction must instead be
replayed inside every development fold and full-search-null replicate.

The observed-data likelihood is also common: it fixes whether total mass is
modeled or normalized away, joint probability for structural-zero target sets
and positive values, any transform and original-measure Jacobian, and a frozen
observation operator that marginalizes missing, unknown, censored, or truncated
coordinates rather than filling them with zero. Every adjudicative model uses
the same masks, support, base measure, and scoring units.

Provider arbor files are derived by spectral clustering and are not
automatically terminal arbors; they require validation against matched full
axons. Topology/path summaries, provider arbors, alternative target
resolutions, and alternative normalizations are separately scored sensitivity
tracks. They cannot enter primary model selection or be compared numerically
with primary-outcome log density.

The released axon-derived `Projection class` is forbidden as a predictor,
split rule, target rule, imputation field, or validation label. Any
axon-derived completeness or review field used to decide outcome observability
cannot also be a predictor unless independent prospective provenance is
established.

Before outcome access, freeze the exact `B` predictor whitelist. It is limited
to named design/technical strata and prospectively validated QC. Layer and
subregion are source-defining or belong to `P`; axon-derived variables and
unregistered dendritic or other morphometric features are excluded. Retain the
whitelist and provenance hash in every matched-model record and the final lock.

## Access boundary

Possession or filesystem readability is not itself recorded as outcome
exposure. A trusted, non-interpretive process may hash bytes, validate container
format, join authenticated identities, or emit an allow-listed metadata view
without exposing projection patterns to candidates. Such operations must log
the actor/process, exact input and output hashes, fields emitted, and whether
any values or diagnostics were visible outside the trusted boundary. Candidate
viewing of axonal geometry, projection-derived values, summaries, errors, or
plots is outcome exposure. If access records cannot establish which case
occurred, the affected animals cannot be claimed as fresh audit; uncertainty is
resolved conservatively in the shared ledger rather than equating mere file
availability with viewing.

| Role | May be accessed for | Must remain hidden |
| --- | --- | --- |
| Metadata/readiness | identity, grouping, duplicates, source/QC provenance, feasibility | all real development and audit axonal outcome summaries |
| Synthetic qualification | grammar recovery, calibration, threshold and failure-rule design | all real outcomes |
| Development | animal-held-out model fitting, internal latent fitting, trial selection, null design | every `audit_sealed` outcome and score |
| Lock | manifest, code/environment, winner, thresholds, seeds, decision rules | audit outcomes |
| One-shot audit | trusted evaluation of the locked package | labels and outcomes from candidate/search processes |

Audit outputs may be emitted only after the final lock hash is recorded. Any
audit read during development invalidates fresh-audit status and is logged as
an exposure, not repaired by relabeling.

## Required readiness artifacts

Provisioning must preserve provider bytes/MD5 values, compute local SHA-256,
and keep archives unchanged. An explicit launch may perform metadata-only
qualification, but before candidate-discriminating outcome search the episode
needs:

- exact resource and license manifests;
- verified cell-to-file and cell-to-independent-animal tables;
- duplicate/overlap checks against any other morphology source used by the
  campaign;
- authenticated CCF assets and coordinate conventions;
- a full-axon/arbor interpretation report;
- the shared EP09–11 exposure and role ledger;
- the redacted source-by-animal-by-position/depth support census and frozen
  common-support decision;
- a cell-, animal/brain-, acquisition-, reconstruction-version-, and derived-
  feature-lineage overlap and novelty dossier for Peng et al. (2021), Yufeng
  Liu et al. (2024; exact SEU-A1876 analysis), Lijuan Liu et al. (2025), and
  Xiong et al. (2025; near-identical F1877 resource); and
- a feasibility report showing that the declared grammar, 99 full-search-null
  reruns, and audit have enough independent animals, cells, and spatial support.

CCF-ME v2 (`13801372`) is dendrite-focused and cannot serve as an external
audit of this axonal claim. No other audit asset is asserted by this contract.
