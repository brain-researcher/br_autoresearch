# EP22 inputs

Every object placed here is read-only.

| Interface | Content | Status |
| --- | --- | --- |
| `laion_ood_embeddings_corrected/` | The provider's 2026-09-28 corrected CLIP, DINOv2, PEcore and SigLIP2 embedding files. They are full 25,052-row HDF5 files from `https://laion-fmri.s3.us-west-2.amazonaws.com/stimuli/`, with their `download_record.json` (ETag, Last-Modified, bytes). | Provisioned 2026-10-12 by job 47545539. The binding passed: exactly 116 OOD rows changed, and all other rows are identical to the 2026-05-12 release. See [the binding report](../outputs/runs/d0_outcome_blind_spectra/20261012-attempt01/results/ood_embedding_binding.json). |

The outcome-blind OOD category table is a derived D0 output, not an input:
[ood_category_table.csv](../outputs/runs/d0_outcome_blind_spectra/20261012-attempt01/results/ood_category_table.csv).

Development inputs are not copied here. EP22 reads EP04's frozen development
definitions and saved development mean patterns in place (see
[DATASETS.md](../DATASETS.md#reused-frozen-ep04-definitions-read-only)), and it
resolves source paths through the root location manifest.

OOD response rows are never placed in `inputs/`. Only the locked evaluator
reads them, directly from the canonical source.
