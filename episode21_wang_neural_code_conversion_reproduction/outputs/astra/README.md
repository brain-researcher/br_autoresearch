# EP21 ASTRA export

The scientific program remains `v0.0.14/astra.yaml`, with 17 analyses.
The legacy exporter tried to write its different, single-analysis snapshot to
that same filename. This caused the observed plan conflict; a legacy terminal
export could overwrite the program. EP21 now uses the episode-local adapter:

```bash
"$OAK/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/outputs/astra/export_milestone.sh" plan
```

It reuses the existing pinned Brain Researcher builder, official ASTRA
validator, and atomic writer. Compatibility files are isolated at
`compatibility/v0.0.14/plan/astra.yaml` or
`compatibility/v0.0.14/terminal/astra.yaml`. Repeated identical plan export is
idempotent; a changed plan fails visibly rather than overwriting its history.
The terminal arguments remain `--output-ref`, `--finding`, and `--created-at`.
For an approved contract amendment, supply a named revision to preserve the
earlier snapshot without a content collision:

```bash
"$OAK/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/outputs/astra/export_milestone.sh" plan --snapshot-id independent-mask-amendment-2026-09-30
```

This exports under the additional `plan/independent-mask-amendment-2026-09-30/`
directory. Repeating the identical named revision is idempotent.
No terminal export is appropriate while EP21 is incomplete at the regular-neural export allocation and subsequent protocol-lock gates.

Use this adapter for EP21 rather than shared `bin/astra-milestone` or
`br-export-astra`, whose old destination would collide with the full program.
Shared tools and the pinned runtime have not been modified. The historical
failure log is retained. This compatibility file is not a complete scientific
handoff, execution authorization, or scientific finding. The complete handoff
is the full program plus `outputs/experiment_log.md` and the result artifacts.
