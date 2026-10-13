# Fitted-H continuation observation — 2026-09-30

Analysis ID: `population_h_optimizer_recovery`. This is a partial observation
of the existing synthetic attempt, Slurm job `45736553`, not its final result.
The scheduler snapshot was collected at 16:14:09 PDT.

The job was `RUNNING` after 189,702 seconds (52.695 hours), with three CPUs
and a 72-hour limit. Its scheduled end is October 1, 2026, at 11:32 a.m. PDT.
CPU and read counters increased between observations. The copied runner emits
each start's result only after that start returns, so the quiet third start is
not evidence of a stall. No implementation change or new attempt was made.

Two logged starts satisfy the physical KKT rule:

| Start | Training negative mean log density |
| --- | ---: |
| Embedded fitted C | 4.57373073603 |
| Truth-based nonzero signature | 4.44457198869 |

The empirical start has no logged result. The logged difference is about
0.12916 nats/neuron on the fitting objective; it is not a fresh verification
score or evidence about real projection organization. The remaining start,
polish, fresh verification, and full frozen conjunction must finish before H
recovery can be adjudicated. The runner has no intermediate H parameter
checkpoint; a timeout would leave partial scalar logs, not a recovered model.

`partial_optimizer_progress.json` and
`../../figures/population_h_partial_optimizer_qc_20260930.png` contain only
these logged values. The figure is **QC, partial synthetic optimizer evidence**.
It was inspected after rendering with the existing Sherlock Matplotlib module;
the Neurofig skill was unavailable. `scheduler_accounting.txt`,
`runtime_counters.txt`, and the copied log preserve the actual observations.

The four named attempts in this limited scheduler snapshot sum to 76.7694
active job hours and 230.3028 allocated CPU core-hours. These are partial
accounting, not the episode's total usage or its global wall-clock origin.
Earlier qualification work is not included. The policy retains a global
120-hour wall ceiling and 1,500-core-hour ceiling including qualification.
The ledger does not define the global clock origin or remaining allowance.
One bundled scientist clarification is pending. No extension, resubmission,
successor, role change, or real-outcome access is authorized by this resume.
The existing job was left unchanged; it has not been cancelled or declared
complete.

The actual fitted-C input checkpoint was copied from this job's `c_input/` to
`../population_saa_r128_three_fresh_C_model_state.npz` for durable retention.
Its passing intake and prior checks were reused. The expected H result remains
`$SCRATCH/br_autoresearch/episode11_projection_types_vs_gradients/population_h_recovery/run_45736553/results/population_h_recovery_summary.json`.
Preserve that actual final artifact under `outputs/stage0/` when available,
then adjudicate without altering the frozen gates. A pass is limited to
`pass_population_H_optimizer_recovery_not_Panel_A_calibration` and grants no
successor authority.

ASTRA is separate from scientific status. The existing compatibility export
command returned `astra_projection_conflict` because its generated snapshot
differed from the existing file. The old snapshot is retained as
`../../astra/v0.0.14/legacy_single_analysis_snapshot.yaml`. The active
`../../astra/v0.0.14/astra.yaml` now records the existing C/H experiments and
their dependencies, with no finding or runtime fields. Pinned 0.0.14 schema
and semantic validation passed; the actual validation result is
`../../astra/v0.0.14/validation_20260930.json`. This active diagnostic program
does not reconstruct the entire historical episode worklist or imply export
adapter support for nested experiments.
