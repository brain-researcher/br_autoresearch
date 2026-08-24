# Sherlock data catalog

This page lists shared data that can be used to propose autoresearch episodes.
The data themselves are on Sherlock and are not stored in this Git repository.

The catalog is an inventory, not a claim that every dataset is ready for every
question. An episode still needs to inspect the observation unit, task and
event structure, preprocessing, confounds, file coverage, access terms, and
possible leakage. No dataset is assigned to discovery, validation, or
confirmation at ingestion time.

## Processed human neuroimaging roots

These are broad working collections rather than frozen releases.

| Collection | Current inventory | Sherlock path | Notes |
| --- | ---: | --- | --- |
| OpenNeuro FitLins results | 54 dataset directories | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/analyses` | Directory presence does not guarantee that every map is materialized or that contrasts are comparable across datasets. |
| OpenNeuro fMRIPrep derivatives | 9 dataset directories | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/fmriprep` | Each episode checks subject, run, confound, and file coverage. Dataset-level licenses still apply. |
| HCP-YA derivatives | Not yet counted here | `/oak/stanford/groups/russpold/data/HCP_YA/HCP-YA-BIDS` | HCP access and data-use terms apply. |
| HCP-YA connectivity data | Not yet counted here | `/oak/stanford/groups/russpold/data/HCP_YA/HCP1200_PTN` | Connectivity products are a separate resource from the task-fMRI derivatives. |

## Curated public releases

The curated source root is:

```text
/oak/stanford/groups/russpold/data/br_autoresearch_data
```

Each release below is kept as an intact, read-only source. The relative path in
the tables is below that root. Archives may be extracted or indexed in scratch
space for an episode, but the source copy should not be modified.

This inventory was checked against the live OAK catalog on 2026-08-23. The 14
ready releases contain 9,825 source files and 97.8 GB in total.

### Human neuroimaging, brain maps, and meta-analysis

| Dataset and source | Release | Contents | Files / size | License | Relative path |
| --- | --- | --- | ---: | --- | --- |
| [NARPS results](https://zenodo.org/records/3634120) | v2.0.1 | Multi-team task-fMRI analysis results | 1 / 52.1 MB | CC BY 4.0 | `narps_results/zenodo-3634120-v2.0.1` |
| [Neurosynth data](https://github.com/neurosynth/neurosynth-data) | v0.7 | Coordinate meta-analysis data, study metadata, and topic features | 22 / 51.4 MB | ODbL 1.0 | `neurosynth/neurosynth-data-v0.7` |
| [neuromaps annotations](https://github.com/netneurolab/neuromaps/tree/0.0.7) | v0.0.7 public-only | Receptor, microstructural, functional, and developmental brain annotations | 123 / 323 MB | Mixed by annotation | `neuromaps/neuromaps-data-v0.0.7-public` |
| [Functional Fusion MDTB](https://zenodo.org/records/16788784) | v1.0 | Multi-task, multi-session subject-level fMRI derivatives | 27 / 14.6 GB | CC BY 4.0 | `functional_fusion_mdtb/zenodo-16788784-v1.0` |
| [Individual Brain Charting maps](https://neurovault.org/collections/6618/) | NeuroVault collection 6618, snapshot 2026-08-21 | Single-subject, unthresholded task-fMRI Z maps across many tasks | 9,532 / 29.2 GB | CC0 | `ibc_neurovault/neurovault-collection-6618-snapshot-2026-08-21` |

Important details:

- NARPS v2.0.1 is retained because existing episode lineage refers to that
  exact release. It is not silently replaced by a later Zenodo version.
- Neurosynth and neuromaps contain meta-analytic summaries and brain
  annotations rather than participant-level observations.
- The neuromaps copy contains 86 logical public annotations represented by 123
  files. One restricted OSF object was excluded. Citation and license metadata
  must be checked for each annotation; the collection does not have one blanket
  license.
- The public MDTB release contains derivatives, including 24 subject archives.
  Resting-state data and time series are not part of this upstream release.
- The IBC snapshot freezes a collection that can otherwise change. All 9,532
  upstream records currently carry `is_valid=false`, while also carrying
  `not_mni=false`; those flags are preserved for investigation rather than used
  as an automatic exclusion rule. The 9,532 maps are repeated observations from
  a much smaller participant set, not 9,532 independent participants.

### Neuron morphology and spatial data

| Dataset and source | Release | Contents | Files / size | License | Relative path |
| --- | --- | --- | ---: | --- | --- |
| [NeuroXiv](https://download.neuroxiv.org/) | Portal snapshot 2026-08-21 | Atlas-mapped mouse neuron morphology, metadata, and portal additions | 10 / 20.8 GB | CC BY-NC 4.0 plus provider academic-use terms | `neuroxiv_portal/portal-snapshot-2026-08-21` |

NeuroXiv is available for internal academic research under its provider terms.
This repository does not redistribute its data. The snapshot includes
post-paper hypothalamus and MERFISH additions, so those files are not part of
the original paper's analysis denominator. The morphology files are atlas-
mapped reconstructions, not raw microscopy images.

### Animal behavior and neural recording

| Dataset and source | Release | Contents | Files / size | License | Relative path |
| --- | --- | --- | ---: | --- | --- |
| [Rajagopalan operant matching](https://zenodo.org/records/7449214) | v1.0.0 | Drosophila sequential choices, rewards, and changing contingencies | 1 / 6.49 GB | CC BY 4.0 | `rajagopalan_operant_matching_2023/zenodo-7449214-v1.0.0` |
| [Dudman learning-rate data](https://figshare.com/articles/dataset/21816054) | Figshare v1 | Mouse behavior, photometry, and perturbation data in an aligned session file | 1 / 271 MB | CC BY 4.0 | `dudman_learning_rate_2023/figshare-21816054-v1` |
| [Kathman working-memory data](https://zenodo.org/records/20053990) | Zenodo record 20053990 | Drosophila behavior, two-photon imaging, olfaction, and navigation | 4 / 1.27 GB | CC BY 4.0 | `kathman_working_memory_2026/zenodo-20053990` |
| [Kenyon-cell odor imaging](https://zenodo.org/records/8166598) | v1.0 | Drosophila calcium responses to repeated odors | 3 / 15.6 MB | CC BY 4.0 | `kc_odor_imaging/zenodo-8166598-v1.0` |
| [Odor-experience plasticity](https://zenodo.org/records/5781484) | Zenodo record 5781484 | Drosophila antennal-lobe calcium imaging and experience manipulation | 2 / 208 MB | CC0 | `odor_experience_plasticity/zenodo-5781484` |
| [Foley reversal learning](https://zenodo.org/records/5010248) | Zenodo record 5010248 | Drosophila reversal-learning observations and official analysis code | 3 / 582 kB | CC0 | `foley_reversal_learning/zenodo-5010248` |
| [Matheson wind circuit](https://zenodo.org/records/6863832) | v1 | Drosophila behavior, neural imaging, perturbation, and navigation | 91 / 23.8 GB | CC BY 4.0 | `matheson_wind_circuit_2022/zenodo-6863832-v1` |
| [Siliciano edge-vector data](https://zenodo.org/records/20751812) | Zenodo record 20751812 | Drosophila behavioral logs and two-photon imaging | 5 / 758 MB | CC BY 4.0 | `siliciano_edge_vector_2026/zenodo-20751812` |

These releases preserve the provider payloads. For example, the Kathman source
is a processed public release rather than raw microscope movies, the Matheson
release remains an unextracted split archive, and the Siliciano entry excludes
local extracted copies and locally derived tables.

## Using a dataset in an episode

`DATASETS.md` should identify the exact release and describe what still needs
to be checked. It should not infer independent sample size from file count or
decide a dataset's scientific role merely because it is convenient.

Before reading the outcomes that will distinguish candidates, the episode
should record which observations are available for search, which evidence will
be used for later evaluation, and what would count as failure. That split is a
scientific decision made for the question, not a permanent property imposed by
this catalog.
