# Cleanup A completion record

Cleanup A was completed on 2026-09-24 after explicit authorization. The five
redundant source paths were permanently removed. Their retained copies remain
under the shared `br_autoresearch_data` root:

`/oak/stanford/groups/russpold/data/br_autoresearch_data`

## Completed cleanup

| Record | Removed path | Retained path |
| --- | --- | --- |
| `scratch_things_eeg1_ingest` | `/scratch/users/zijiao/autoresearch_ingest/things_eeg1` | `things_eeg1/openneuro-ds003825-v1.2.0/source` |
| `scratch_ep05_payload` | `/scratch/users/zijiao/autoresearch/episode05_sensorimotor_lfp/source/zenodo_7055668/payload` | `sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source` |
| `personal_link_001201_full` | `/oak/stanford/groups/russpold/users/zijiao/data/bci_poc/link_001201_full` | `link_long_term_intracortical/dandi-001201-0.251023.2336/source` |
| `personal_link_001201_sentinels` | `/oak/stanford/groups/russpold/users/zijiao/data/bci_poc/link_001201_sentinels` | `link_long_term_intracortical/dandi-001201-0.251023.2336/source/sub-Monkey-N` |
| `personal_epic_ptir_coating` | `/oak/stanford/groups/russpold/users/zijiao/data/bci_poc/epic_ptir_coating` | `epic_ptir_coating/mendeley-7p3cxn7jtn-v1/source` |

The estimated reclaimed space is 82,027,835,392 bytes. This is a historical
estimate recorded before removal, not a new measurement.

## Read-only report

The helper now has only two read-only commands:

```bash
bin/data-cleanup list
bin/data-cleanup show scratch_things_eeg1_ingest
```

It accepts only the five fixed logical IDs. It cannot write receipts, move
data, remove data, or accept a source or destination path.

Older detailed planning receipts under the external `manifests/` directory
remain untouched as historical artifacts. The helper does not read or update
them.
