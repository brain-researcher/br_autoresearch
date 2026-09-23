# EP10 dataset contract

This episode follows [the common adaptive protocol](../ADAPTIVE_SEARCH_PROTOCOL.md).

## Fixed real sources

| Source/file | Identity | Role | Current state |
| --- | --- | --- | --- |
| `Full_morphometry.xlsx` | Zenodo `13944322`; 181,376 B; MD5 `dcad84366865aa6ffc4d5b010159dd9b` | cell, brain, soma, layer, QC metadata | absent |
| `Full_morphology_CCFv3.zip` | Zenodo `13944322`; 1,536,231,503 B; MD5 `a5d2242516268a301e0efc16438990dd` | full axons and completeness checks | absent |
| `Axonal_arbor_CCFv3.zip` | Zenodo `13944322`; 24,463,661 B; MD5 `6b6d903884043d574320f07b8f220451` | distal-arbor outcome candidate | absent |
| `Axonal_bouton_CCFv3.zip` | Zenodo `13944322`; 1,419,124,590 B; MD5 `156bbaaf31896d21c6541542cda1b86c` | optional putative-bouton sensitivity | absent |
| Allen Mouse CCF | exact CCFv3 annotation and structure graph to be pinned | vocabulary and geometry | absent |

Verify provider byte counts/MD5 values, compute local SHA-256 hashes, pin
licenses and parsers, and preserve original archives. Released metadata report
1,876 morphology names, 39 `fMOST Brain ID` values, 92 soma regions, 1,736
manually checked cells, and 140 unchecked cells; these counts do not establish
eligibility or biological independence.

## Exposure and correlated-episode boundary

EP09, EP10, and EP11 may reuse identical cells and axonal outcomes. One shared,
pre-outcome ledger must resolve stable cell identities, conservative
animal/brain/specimen groups, duplicate/transformed lineage, batches, exposure,
and whole-group roles. Their audits are correlated. If any shared audit target
is opened before all intended contracts lock, later uses are outcome-exposed
and cannot be called sealed evaluation.

Search initialization may not use historical Q0/Q1 scores, pair coefficients,
target vocabularies chosen by performance, or sibling predictions. External
sources and pretrained representations require cell/SWC hash, topology,
geometry, and publication-lineage deduplication.

## Required observation table

For every eligible neuron construct:

- `V_i`: targets that could have been scored, fixed from non-overlapping atlas,
  source, hemisphere, support, and observation rules—not realized targets;
- `S_i`: complete qualifying target set within `V_i`;
- `K_i = |S_i|`, given equally to Q0 and Q1; and
- `Omega_i`: every size-`K_i` subset of `V_i`, exactly normalized by both
  models.

The primary cohort requires `2 <= K_i <= |V_i|-2`, complete target readability,
positive residual-pair design rank, and tractable exact normalization. Unknown
is never zero. Validate full-axon tree/compartment semantics, clipping,
boundary-reaching branches, coordinate transforms, matched provider arbors,
terminal evidence, and negative observability. A passing fiber is not
automatically a target.

`Projection class` is axon-derived and prohibited as predictor, split field,
candidate-space rule, imputation field, or calibration target. Permitted
covariates are prospectively available source/soma/layer/independent-label,
quality, acquisition, and frozen atlas geometry fields.

## Development and audit roles

Assign whole conservative biological groups from structural identity and
coverage before target identities are inspected. Require at least 12
development and 8 locked audit groups with supported source populations and
recurring pairs. Development groups may define supported vocabulary and tune
Q0/Q1 in nested group folds. Audit `S_i`, target frequencies, choice spaces
derived from outcomes, and metrics remain in a permission-separated store
until the configuration lock.

If biological grouping, valid nondetection, recurring-pair support, or exact
normalization fails, the primary episode is blocked. It cannot be rescued by
random-neuron splits, clustering, pseudolikelihood, treating unknown as zero,
or shrinking the vocabulary after score inspection.

## Firewall and fitted-Q0 simulations

Candidate jobs may receive development tables and approved scalar/group
diagnostics only. The trusted evaluator owns group folds, audit identities and
outcomes, exact enumerations, and metric/inference code. Hash all vocabulary,
geometry, residualization, and choice-space artifacts.

Fitted-Q0 null datasets must preserve the development group sizes, covariates,
`V_i`, `K_i`, choice spaces, and fitted Q0 heterogeneity. Each null replicate
must invoke the same search controller from an empty ledger and consume the
same proposal, family-selection, stopping, and promotion rules as the real
development search.

## Missing blockers

The listed sources are not provisioned locally; the Allen CCF identity and
geometry are unpinned; animal/specimen mapping and batch metadata are
unresolved; target readability and nondetection are unvalidated; minimum
group/pair support and exact-normalization cost are unknown; the shared
EP09/10/11 split/exposure ledger and audit store do not exist; and canonical
bindings are null. All block launch. Optional bouton sensitivity does not
block the primary complete-arbor analysis.

Large sources remain outside Git or behind immutable read-only references.
Expanded archives, exact choice-space caches, and simulations belong in
`$SCRATCH/autoresearch/episode10_single_cell_coprojection/`; durable
outputs are manifests, ledgers, validated tables/specifications, lock bundles,
and reports.
