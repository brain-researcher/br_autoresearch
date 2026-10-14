# EP23 data contract

status: proposed_not_frozen (design of 2026-10-13)

The counts below come from permitted metadata checks recorded in the
[experiment log](outputs/experiment_log.md). No EP23 neural or model outcome has
been computed.

## LAION-fMRI (initial stage)

- **Dataset:** LAION-fMRI, 7T, five participants (sub-01, sub-03, sub-05,
  sub-06, sub-07). The source, licensing and Low-Risk record are those in
  [EP22's DATASETS](../episode22_geometry_gap_decomposition/DATASETS.md#source).
  - fMRI data, metadata and derivatives are CC0.
  - EP23's initial stage needs no raw images.
- **Physical location:** resolve it only through the root
  [DATA_LOCATION_MANIFEST.json](../DATA_LOCATION_MANIFEST.json), logical ID
  `asset_ep04_laion_mixed_role`. EP04, EP21 and EP22 use the same bytes, so
  EP23 results on LAION are correlated with theirs, not independent of them.

| Slice | EP23 role | Exposure |
| --- | --- | --- |
| Development: 4,712 subject-unique images × 5 participants. Each participant has 4,246 LAION, 322 THINGSplus and 144 THINGS images. | Exploratory: A1, A2 | Exposed by EP04 and by EP22 D0–D3 |
| Regular shared: 1,121 images | Excluded | Exposed in EP04 |
| OOD: 371 images | Historical motivation only | Consumed by EP22's single confirmation opening (2026-10-13); no new EP23 analysis |

Units:

- Participants are the biological units.
- Images, RDM edges, voxels, sectors, folds, model families and repetitions
  are not independent replications.
- No development image is shared by all five participants.

### Reused frozen definitions (read-only)

EP23 reads the following items in place and never writes to EP04 or EP22. Both
episodes are terminal.

- **EP04's frozen development definitions,** as listed in
  [EP22 DATASETS](../episode22_geometry_gap_decomposition/DATASETS.md#reused-frozen-ep04-definitions-read-only):
  - the image, trial and group tables;
  - the five image-disjoint, component-grouped folds;
  - the development embedding rows;
  - the preprocessing rules: training-only session standardization, at
    least two valid presentations, and same-index ROI mapping.
- **Primary sectors:** laion-ventral, laion-lateral and laion-dorsal.
  laion-EVC is descriptive.
- **EP22's audited conventions:**
  - native RSA uses model cosine distance and neural correlation distance;
  - Spearman on average ranks, then Fisher z;
  - equal weights over folds, then sectors, then participants;
  - E_val/E_vec decomposition.

  The code is
  [`d123_cell.py`](../episode22_geometry_gap_decomposition/outputs/code/d123_cell.py).
  EP22's spectral power convention is Φ_α = U_k D_k^α in the leading-k
  training-PCA basis of centered features, with X = U D Vᵀ. So α = 1 is the
  truncated native geometry, the Gram eigenvalues of Φ_α are d^{2α}, and α = 2
  squares the feature Gram matrix.

### Repetitions and reliability

- Development images have four planned presentations. EP04 records 23,536
  four-repeat and 24 three-repeat images.
- Split-half brain reliability is therefore available within the development
  role. A2 uses it at each scale.

### Spatial measurement metadata (A3, deferred spatial test)

- **Responses:** GLMsingle-tedana betas are stored in each participant's T1w
  volume space, so voxel coordinates are available from the image affines.
- **Noise ceilings:** per-voxel maps exist for 4-repetition, 12-repetition and
  all-repetition stimulus sets, using the NSD estimator.
- **ROI masks:** stored as fsnative surface labels (`.label`, `.func.gii`).
- **Not verified:** surface meshes, preprocessing smoothing, and a
  neural-topography model.

A real spatial-pooling test is therefore outside the initial stage. A3 uses
synthetic inputs only.

### Model embeddings

There are four released provider embeddings. They are described in
[EP22 DATASETS](../episode22_geometry_gap_decomposition/DATASETS.md#model-embeddings):

| Model | Released representation |
| --- | --- |
| CLIP | OpenCLIP ViT-H/14 projection, 1,024-d |
| PEcore | PE-Core L/14 final embedding, 1,024-d |
| SigLIP2 | SigLIP2 SO400M/14 final embedding, 1,152-d |
| DINOv2 | ViT-L/14 layer-23 patch-token mean, 1,024-d |

The stage mismatch between DINOv2 and the VLMs is a competing explanation, not
a controlled objective contrast. New embeddings, layers or checkpoints are
outside the initial stage.

## Semantic labels for A2

- **Source:** THINGS concept metadata (Hebart et al. 2019, OSF `jum2f`),
  steward asset `asset_ep18_things_metadata` (`things_metadata_osf_jum2f_20260922`).
  EP23 reads only concept-level label tables, which contain no outcomes.
- **Taxonomy:** `category27_manual.tsv`. This holds the THINGS authors' manual
  membership of the 1,854 concepts in 27 categories. Its rows align with the
  row order of `concepts-metadata_things.tsv`, which was checked on 10 concepts.
- **Mapping:** THINGS and THINGSplus image names
  (`unique_THINGS[+]_<concept>_pNN.jpg`) map to concept IDs. Only concepts with
  exactly one category27 label are used.

Per-participant counts of single-label images (466 THINGS-stratum images per
participant):

| | sub-01 | sub-03 | sub-05 | sub-06 | sub-07 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Categories with ≥20 images | 3 | 4 | 2 | 2 | 4 |
| Categories with ≥10 images | 9 | 7 | 10 | 8 | 8 |

- **Model–model analysis:** pooled over participants there are 2,330
  THINGS-stratum images, of which 1,166 are single-label. Of the categories,
  19 have ≥20 images and 10 have ≥40.
- **Fine scale:** within a participant, each concept contributes one or two
  images (2,118 single and 106 double concept–participant pairs). The "fine"
  scale is therefore mostly between-concept structure within a category, not
  exemplar structure within a concept.
- **category53:** the THINGS 53-category long-format table gives only one or
  two categories with ≥20 images per participant, so it is not used.
- **Not semantic:** photographic source categories such as "selfmade" are
  not semantic scales.

The label table, the single-label rule and both thresholds were chosen from
label counts alone, before any EP23 outcome.

## Co-occurrence operator sources (A0)

The checks of 2026-10-13 used annotation files only (no images), stored as
transient, read-only copies under
`$SCRATCH/br_autoresearch/episode23_spectral_operator_convergence/feasibility/downloads/`.
URLs and dates are in `download_record.tsv`. Full results are in the
[log](outputs/experiment_log.md) and in
[feasibility_report.json](outputs/runs/operator_feasibility/20261013-check01/feasibility_report.json).
Access and licence details are in the
[source audit](outputs/design/operator_source_audit.md).

| Source | Licence | Role | Pooled S (≥10 images) | S_p | Caveats |
| --- | --- | --- | ---: | --- | --- |
| Visual Genome v1.4 objects, 108,077 images | CC BY 4.0 | Primary operator | 615 | 166–181 | Labels are not exhaustive. Synsets come from automatic canonicalization with human checks, using the most frequent sense. Exact-name matching undercounts (e.g. glass, glasses); a WordNet 3.0 offset crosswalk is needed. |
| COCO 2017 captions, 123,287 images | CC BY 4.0 | Secondary (description channel) | 880 | 234–257 | The current matcher over-counts from WordNet synonyms and verbs ("stands", "background", "picture", "light", "hold"). Match `Word`/`uniqueID` only and audit frequent matches. |
| LVIS v1, 119,979 images | CC BY 4.0 annotations | Robustness only | 473 | 121–151 | Federated labels: frequent categories are capped near 2,300 images, and 9.8% of positives are non-exhaustive. Use verified-pair denominators. Fails S_p ≥ 150. |
| Open Images V7 class names | CC BY 4.0 | Not used | 1,117 by name | — | Human-verified labels (several GB) and a MID crosswalk would be needed. |
| ADE20K full, 2021 index | Annotations BSD-3; registration | Deferred | — | — | `index_ade20k.pkl` is a pickle and would need a restricted unpickler. DINOv2's LVD-142M used ADE20K as a retrieval seed. |

Independence:

- **Model training data:** the annotations were produced by humans, not by
  the evaluated representations. COCO and VG photographs, or similar ones,
  may still be in the models' training data. Exact overlap is unverified.
- **Between sources:** COCO, LVIS and part of VG share images, so
  agreement between these sources is not independent replication.
- **Exclusions:** THINGSplus secondary-object labels are excluded, because
  they come from the THINGS photographs themselves. CLIP-filtered LAION
  captions are excluded as not independent of CLIP.

Name mapping:

- 166 THINGSplus development images have truncated multiword names (e.g.
  `air`, `bok`). They are unmatched, unless provider stimulus metadata
  resolves them.

## Candidate independent confirmation (A4, inactive)

No eligible source is currently available. Every candidate needs a decision by
the scientist before any of its outcomes are read.

| Candidate | Status on 2026-10-13 | Constraint |
| --- | --- | --- |
| THINGS-EEG1 (OpenNeuro ds003825) | Acquired. EP18 assigns 30 development and 16 audit participants; its audit is closed and its development work is active. | EP18's audit participants are reserved. EP18 asks whether a categorical basis predicts held-out exemplars, which overlaps EP23's coarse/fine question, so its results would be correlated evidence. |
| CNeuroMod-THINGS fMRI | Acquired. EP17 assigns 480 development, 120 support and 120 sealed audit concepts in the same four participants; the audit is closed. | Four participants cannot reach α = 0.05 with an exact one-sided sign test (minimum p = 0.0625). EP17's audit concepts are reserved. |
| An untouched THINGS dataset, e.g. THINGS-EEG2 or THINGS-data [verify availability and terms] | Not acquired | Requires acquisition scope, new embeddings (GPU extraction) and a frozen endpoint first. |
| NSD | Not present | Requires a separate DUA, acquisition and resource decision. |

- A confirmation source needs at least five independent participants for the
  proposed sign test.
- EEG cannot isolate fMRI pooling: modality, generators and sampling differ.
- No LAION split, renewed seal or reanalysis of consumed OOD outcomes can
  supply confirmation.

## Experienced operators

- **What the annotation operators are.** They are proxies measured from
  photographs, not records of what any model or person experienced.
- **What is missing.** An operator of a learner's actual experience is not
  available: egocentric or child corpora need access approvals. Neither is a
  temporal transition operator.
- **What is not evidence.** An operator inferred from the embeddings being
  explained is not independent evidence.
- **Visual-similarity comparator for A0.** The scientist approved binding the
  raw stimuli on 2026-10-13. A0 uses the least access that meets the need:
  - It reuses EP04's frozen RGB block-moment features, read-only:
    `../episode04_shared_scene_geometry/outputs/runs/source_readiness/20261005-low_level_full01/`,
    with `low_level_features.npy` (23,560 × 3,150) and `image_ids.json`.
  - EP04 computed them from the DUA-covered raw images under its approved
    [control specification](../episode04_shared_scene_geometry/outputs/low_level_control_proposal.md).
  - A0 reads no pixels.
  - The features cover colour, luminance layout and local contrast at three
    grid scales. They do not cover orientation, spatial frequency or shape.
    A Gabor/GIST sensitivity analysis would read raw images under the
    approved binding.

## Storage

- **Durable:** proofs, code, compact estimates, figures and the log go under
  `outputs/`. The proposed ceiling is 2 GiB.
- **Transient:** caches go under
  `$SCRATCH/br_autoresearch/episode23_spectral_operator_convergence/`. The
  proposed ceiling is 20 GiB.
- **Inputs:** `inputs/` holds no copies. Inputs are read in place, as listed
  in [inputs/README.md](inputs/README.md).
- **Not inputs:** legacy `$SCRATCH/autoresearch/` trees.
