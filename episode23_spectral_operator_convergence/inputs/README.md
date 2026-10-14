# EP23 inputs

This directory holds no copies of data. Every input is read in place and is
read-only. Physical locations of steward assets resolve through the root
[DATA_LOCATION_MANIFEST.json](../../DATA_LOCATION_MANIFEST.json).

| Interface | Source | Role |
| --- | --- | --- |
| LAION-fMRI development betas and ROIs | `asset_ep04_laion_mixed_role` (EP04's single-trial development cache under `$SCRATCH/br_autoresearch/episode04_shared_scene_geometry/neural_full01/`, resolved through EP04's participant handoffs) | A0 brain patterns; A1 and A2 (exploratory) |
| EP04 frozen development tables, folds and embedding rows | `../../episode04_shared_scene_geometry/outputs/runs/source_readiness/` (`20261003-metadata_folds04/`, `20261003-development_embeddings01/`) | Fixed definitions shared with EP22; A0 model embeddings |
| EP04 RGB block-moment features | `../../episode04_shared_scene_geometry/outputs/runs/source_readiness/20261005-low_level_full01/` | A0 low-level visual operator. No pixels are read. |
| EP22 RSA, E_val/E_vec and spectral-power code | [`../../episode22_geometry_gap_decomposition/outputs/code/`](../../episode22_geometry_gap_decomposition/outputs/code/) | Audited conventions. Copy the code into `outputs/code/` before modifying it. |
| THINGS concept labels (`category27_manual.tsv`, `concepts-metadata_things.tsv`) | `asset_ep18_things_metadata` | A0 crosswalk and taxonomy control; A2 labels. The tables hold concept metadata only. |
| Visual Genome v1.4 objects, COCO 2017 captions, NLTK WordNet 3.0 | Public annotation and lexical files in transient scratch (`$SCRATCH/br_autoresearch/episode23_spectral_operator_convergence/feasibility/downloads/`, read-only; URLs and dates in `download_record.tsv`) | A0 operators. LVIS and Open Images were used for feasibility only. |

LAION-fMRI OOD rows (images, embeddings, responses or summaries) are not
inputs. They were consumed by EP22. No THINGS-EEG1, CNeuroMod or other
confirmation data is bound to EP23. See
[DATASETS.md](../DATASETS.md#candidate-independent-confirmation-a4-inactive).
