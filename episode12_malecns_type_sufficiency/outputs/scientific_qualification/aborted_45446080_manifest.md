# Preserved scientific-qualification failure: Slurm 45446080

- Preserved at: `2026-09-27T02:20:04-0700`
- Original durable directory: `outputs/scientific_qualification/qualification/`
- Preserved directory: `outputs/scientific_qualification/aborted_45446080_scientific_qualification_failed/`
- Qualification job: `45446080`, terminal state `FAILED`, exit `1:0`, elapsed `06:43:33`
- Readiness job: `45446082`, terminal state `CANCELLED`, elapsed `00:00:00`
- Post-36 tail job: `45447999`, terminal state `CANCELLED`, elapsed `00:00:00`
- Classification: designed scientific-gate failure, not an execution crash
- Records: 2,520/2,520 generated-data replicates; no real connectivity accessed
- Downstream action: blocked; an unchanged rerun or readiness bypass is forbidden

The six qualification artifacts were moved together only after the jobs were
terminal. Their content hashes before and after the move matched:

| Artifact | SHA-256 |
| --- | --- |
| `cost_profile.json` | `dcdeb9b082f3310be0127bdcfce0098b45044b58209501c22b1f4b0907605253` |
| `integration_stability.json` | `97f8af42e52b98da039a6e0a0b01ca9500623bec2f73a00de66b16ba38cd390b` |
| `qualification_report.json` | `5e2ec5b05e89ac53a6c313c819509b3507cc32a6441766a5e8fce372263caa1d` |
| `replicate_results.jsonl` | `b1b8db0b81d1223d54f3ee7d96b432e18f05fea42b7fc3fe201e95d4953a4632` |
| `result_table.csv` | `263c518eea09eaf07aa4b7ce89b4b891227aae16ecbebb6d1f10467e7f72a7b5` |
| `result_table.md` | `461772d54252bbaac736b489201def7e168b82ca32cd0fd0027ad13faf5076de` |

The failure evidence is complete and internally consistent. Contract and cost
gates passed, but 25/35 cells failed operating-characteristic acceptance and
the integration-stability gate failed. The preserved run must remain immutable
evidence while a separately versioned repair is tested on generated data.
