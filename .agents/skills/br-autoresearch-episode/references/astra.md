# ASTRA projections

Read this file only when an episode reaches its plan or terminal ASTRA
milestone. ASTRA is a non-authoritative interoperability projection. It is not
a launch, scientific gate, reward, confirmation, or Landscape action.

## Access rule

The adapter must obey the episode's current positive read allowlist or run with
equivalent filesystem isolation. If its read footprint is unknown or unsafe,
do not invoke it. Report `ASTRA compliance: open` and continue the authorized
episode work.

## Plan milestone

After reading the episode contracts, run:

```bash
episode_dir=$(cd -P -- "<absolute-episode-dir>" && pwd)
repo_root=$(dirname -- "$episode_dir")
"$repo_root/bin/astra-milestone" plan "$episode_dir"
```

The adapter chooses the deployed `br-export-astra` launcher when available.
Do not activate its environment or rewrite `PYTHONPATH` yourself on Sherlock.
If export fails, report the error and continue; ASTRA is not an execution gate.

## Terminal milestone

Export terminal ASTRA only when the episode is genuinely terminal and the
scientific artifacts already exist. Use only already-open, terminal-stage
allowlisted files under `outputs/`; do not browse closed outputs for stronger
evidence. State the observed finding or limitation without strengthening it,
and use the actual UTC closeout time.

```bash
episode_dir=$(cd -P -- "<absolute-episode-dir>" && pwd)
repo_root=$(dirname -- "$episode_dir")
"$repo_root/bin/astra-milestone" terminal "$episode_dir" \
  --output-ref outputs/<actual-artifact> \
  --finding "<observed finding, refutation, or limitation>" \
  --created-at <actual-UTC-timestamp>
```

Repeat `--output-ref` when several artifacts jointly support the finding. Never
invent a path, finding, or timestamp. A completed Slurm job is not necessarily
an episode terminal. If export fails or cannot run safely, preserve the real
scientific outcome and report `ASTRA compliance: open`.
