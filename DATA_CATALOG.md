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
| OpenNeuro FitLins results | 55 `ds*` dataset directories | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/analyses` | Directory presence does not guarantee that every map is materialized or that contrasts are comparable across datasets. |
| OpenNeuro fMRIPrep derivatives | 9 dataset directories | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/fmriprep` | Each episode checks subject, run, confound, and file coverage. Dataset-level licenses still apply. |
| HCP-YA derivatives | Not yet counted here | `/oak/stanford/groups/russpold/data/HCP_YA/HCP-YA-BIDS` | HCP access and data-use terms apply. |
| HCP-YA connectivity data | Not yet counted here | `/oak/stanford/groups/russpold/data/HCP_YA/HCP1200_PTN` | Connectivity products are a separate resource from the task-fMRI derivatives. |
| NARPS event-table pool | 4,646,421 logical bytes | `/oak/stanford/groups/russpold/data/NARPS` | Legacy shared support tables; distinct from the curated NARPS results release and not a role assignment. |
| NARPS pupillometry pool | 9,382,338,780 logical bytes | `/oak/stanford/groups/russpold/data/NARPS_pupillometry` | Shared ASC/EDF material and archives; access terms, participant coverage, and episode role must be verified before use. |

## Curated public releases

The curated source root is:

```text
/oak/stanford/groups/russpold/data/br_autoresearch_data
```

Each release below is intended to remain an intact, immutable source. The
relative path in the tables is below that root. Archives may be extracted or
indexed in scratch space for an episode, but the source copy should not be
modified. Actual permissions vary; `DATA_LOCATIONS.md` records the currently
writable EP16 release whose write seal is still pending.

This inventory was checked against the live OAK catalog on 2026-09-23. The 28
catalogued releases declare 10,817 source files and 396,139,712,889 source bytes
(396.1 GB) in total.

### Human neuroimaging, electrophysiology, brain maps, and meta-analysis

| Dataset and source | Release | Contents | Files / size | License | Relative path |
| --- | --- | --- | ---: | --- | --- |
| [NARPS results](https://zenodo.org/records/3634120) | v2.0.1 | Multi-team task-fMRI analysis results | 1 / 52.1 MB | CC BY 4.0 | `narps_results/zenodo-3634120-v2.0.1` |
| [Neurosynth data](https://github.com/neurosynth/neurosynth-data) | v0.7 | Coordinate meta-analysis data, study metadata, and topic features | 22 / 51.4 MB | ODbL 1.0 | `neurosynth/neurosynth-data-v0.7` |
| [neuromaps annotations](https://github.com/netneurolab/neuromaps/tree/0.0.7) | v0.0.7 public-only | Receptor, microstructural, functional, and developmental brain annotations | 123 / 323 MB | Mixed by annotation | `neuromaps/neuromaps-data-v0.0.7-public` |
| [Functional Fusion MDTB](https://zenodo.org/records/16788784) | v1.0 | Multi-task, multi-session subject-level fMRI derivatives | 27 / 14.6 GB | CC BY 4.0 | `functional_fusion_mdtb/zenodo-16788784-v1.0` |
| [Individual Brain Charting maps](https://neurovault.org/collections/6618/) | NeuroVault collection 6618, snapshot 2026-08-21 | Single-subject, unthresholded task-fMRI Z maps across many tasks | 9,532 / 29.2 GB | CC0 | `ibc_neurovault/neurovault-collection-6618-snapshot-2026-08-21` |
| [THINGS-EEG1](https://openneuro.org/datasets/ds003825/versions/1.2.0) | OpenNeuro v1.2.0 | Raw continuous and provider-preprocessed EEG for object vision | 412 / 59.4 GB | CC0 | `things_eeg1/openneuro-ds003825-v1.2.0` |

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
- The THINGS-EEG1 snapshot contains recordings from 50 participants and
  provider derivatives. The 22,248 stimulus images are not included, and file
  count is not a trial or participant count.

### Neuron morphology, connectomics, and spatial data

| Dataset and source | Release | Contents | Files / size | License | Relative path |
| --- | --- | --- | ---: | --- | --- |
| [NeuroXiv](https://download.neuroxiv.org/) | Portal snapshot 2026-08-21 | Atlas-mapped mouse neuron morphology, metadata, and portal additions | 10 / 20.8 GB | CC BY-NC 4.0 plus provider academic-use terms | `neuroxiv_portal/portal-snapshot-2026-08-21` |
| [FlyEM Male CNS](https://male-cns.janelia.org/download/) | v1.0 flat connectome | Drosophila synaptic-connectivity, neuron-annotation, and neurotransmitter-prediction tables | 11 / 31.3 GB | CC BY 4.0 | `flyem_male_cns/gcs-male-cns-v1.0-flat-connectome` |

NeuroXiv is available for internal academic research under its provider terms.
This repository does not redistribute its data. The snapshot includes
post-paper hypothalamus and MERFISH additions, so those files are not part of
the original paper's analysis denominator. The morphology files are atlas-
mapped reconstructions, not raw microscopy images.

The FlyEM release is the complete observed flat-connectome table prefix. It
does not include image or segmentation volumes, skeletons, or Neo4j data.

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

### Intracortical recording, BCI, and interface data

| Dataset and source | Release | Contents | Files / size | License | Relative path |
| --- | --- | --- | ---: | --- | --- |
| [LINK long-term intracortical activity](https://dandiarchive.org/dandiset/001201/0.251023.2336) | DANDI 0.251023.2336 | Macaque processed neural features, kinematics, and behavior | 312 / 12.6 GB | CC BY 4.0 | `link_long_term_intracortical/dandi-001201-0.251023.2336` |
| [FALCON M1-A](https://dandiarchive.org/dandiset/000941/0.241029.1405) | DANDI 0.241029.1405 | Macaque reach-to-grasp multiunit spikes, iEMG, and behavior | 11 / 312 MB | CC BY 4.0 | `falcon_m1a/dandi-000941-0.241029.1405` |
| [Hennig neural engagement](https://github.com/mobeets/neural-engagement/tree/ab2f1ec3f425d81bf2972a64fdd2231e947b3e23) | GitHub commit ab2f1ec3f425 | Analysis code and two preprocessed macaque BCI-learning example sessions | 54 / 6.00 MB | BSD 3-Clause repository license; data terms not separately stated | `hennig_neural_engagement/github-ab2f1ec3f425` |
| [Brochier multielectrode grasp](https://doi.gin.g-node.org/10.12751/g-node.f83565/) | Fixed DOI release | Raw 30 kHz macaque waveforms, LFP, spikes, behavior, and electrode metadata | 1 / 15.5 GB | CC BY 4.0 record; embedded files state CC BY-SA 4.0 | `brochier_multielectrode_grasp/g-node-f83565` |
| [BrainGate long-term array performance](https://datadryad.org/dataset/doi:10.5061/dryad.x0k6djj1h) | Dryad v6 | Deidentified human intracortical features, impedance, and closed-loop BCI behavior | 24 / 84.7 GB | CC0 | `braingate_long_term_array_performance/dryad-x0k6djj1h-v6` |
| [Sensorimotor LFP](https://datadryad.org/dataset/doi:10.5061/dryad.xd2547dkt) | Dryad v5 payload | Macaque post-processed LFP, spike counts, kinematics, and behavior | 25 / 9.57 GB | CC0 | `sensorimotor_lfp/dryad-xd2547dkt-v5-payload` |
| [EPIC PtIr coating](https://data.mendeley.com/datasets/7p3cxn7jtn/1) | Mendeley v1 | Rat electrode coating, impedance, noise, unit-yield, and histology workbook | 1 / 132 kB | CC BY 4.0 | `epic_ptir_coating/mendeley-7p3cxn7jtn-v1` |
| [FALCON H2](https://dandiarchive.org/dandiset/000950/0.241029.1403) | DANDI 0.241029.1403 | Human attempted-handwriting behavior and 20 ms binned spikes | 47 / 1.22 GB | CC BY 4.0 | `falcon_h2/dandi-000950-0.241029.1403` |
| [FALCON M1-B](https://dandiarchive.org/dandiset/001209/draft) | DANDI draft snapshot 2026-09-16 | Macaque reach-to-grasp multiunit spikes, iEMG, and behavior | 12 / 234 MB | CC BY 4.0 | `falcon_m1b/dandi-001209-draft-snapshot-2026-09-16` |
| [Mindful iBCI stability](https://datadryad.org/dataset/doi:10.5061/dryad.n2z34tn5s) | Dryad v6 | Deidentified human fixed-decoder recordings and closed-loop cursor behavior | 2 / 412 MB | CC0 | `mindful_ibci_stability/dryad-n2z34tn5s-v6` |
| [Long-term unsupervised recalibration](https://datadryad.org/dataset/doi:10.5061/dryad.1jwstqk6g) | Dryad v4 | Anonymized human decoder-recalibration recordings, behavior, and simulations | 2 / 1.69 GB | CC0 | `long_term_unsupervised_recalibration/dryad-1jwstqk6g-v4` |
| [O'Doherty NHP reaching](https://zenodo.org/records/3854034) | Zenodo 3854034 plus broadband snapshot 2026-09-16 | Macaque reaching data with sorted spikes, kinematics, and linked raw-broadband supplements | 78 / 81.3 GB | CC BY 4.0 | `odoherty_nhp_reaching/zenodo-3854034-plus-broadband-2026-09-16` |

Important details:

- Published DANDI versions are frozen here, but FALCON M1-B is a dated snapshot
  of a mutable draft and has no immutable version or DOI.
- Public FALCON packages contain calibration and mini-validation material while
  private evaluation portions remain withheld. Mini-validation files are not
  automatically independent sessions.
- Most releases contain processed features rather than raw voltage. The
  Brochier archive and the 30 linked O'Doherty supplements provide raw
  broadband recordings; the supplements do not cover every O'Doherty session.
- The Hennig codepack contains only two example sessions, not the full study
  dataset. Its repository license does not resolve separate data-republication
  terms.
- Human releases remain subject to their stated deidentification and use
  constraints. The recalibration release excludes the T11 personal-use data
  identified upstream as potentially containing protected health information.
- The EPIC release measures electrode and material performance in rats and has
  no BCI behavior.

## Using a dataset in an episode

`DATASETS.md` should identify the exact release and describe what still needs
to be checked. It should not infer independent sample size from file count or
decide a dataset's scientific role merely because it is convenient.

Before reading the outcomes that will distinguish candidates, the episode
should record which observations are available for search, which evidence will
be used for later evaluation, and what would count as failure. That split is a
scientific decision made for the question, not a permanent property imposed by
this catalog.
