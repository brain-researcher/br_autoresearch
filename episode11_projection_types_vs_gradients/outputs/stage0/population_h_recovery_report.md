# Fitted-H optimizer recovery — failed frozen diagnostic

Analysis ID: `population_h_optimizer_recovery`. Slurm job `45736553`
completed on September 30, 2026, at 16:20:59 PDT, with exit code `0:0` and
elapsed time 52:48:32. Process completion produced a valid diagnostic report;
the frozen recovery conjunction did not pass. Exact status:
`revise_population_H_optimizer_iteration_cap`.

## Actual result and interpretation

Exactly two acceptance gates failed:

| Frozen gate | Observed result | Requirement |
| --- | --- | --- |
| Every main start avoids its iteration cap | Empirical start stopped at 1,200 iterations / 1,230 evaluations | No capped start |
| Best-two converged H fits agree on fresh draws | Gap 0.129458594 nats/neuron, MCSE 0.000104397 | Absolute gap below 0.002 |

The empirical start's message is `STOP: TOTAL NO. of ITERATIONS REACHED LIMIT`.
Its physical projected-gradient norm was 0.0001247525, above 0.00002. The
embedded-C and truth-based starts converged after zero and 11 iterations.
Their training negative mean log densities were 4.5737307360 and 4.4445719887;
the empirical start remained at 5.0957142090. No numerical parameter bound
was active. Polishing the best fit took zero iterations and gave zero gain.

Fresh paired verification used the frozen stream 5601 and 512 generated
replicates. The best fitted H improved over C by 0.129458594 nats/neuron.
Its truth-minus-H deficit was 0.001455587 (MCSE 0.000328496), with one-sided
95% upper bound 0.001996895, passing the 0.010 requirement. Maximum monitored
MCSE was 0.000343083, below 0.005. The exact truth anchor, retained C intake,
polishing, numerical-boundary and other gates passed.

Thus the best truth-initialized solution represents this planted distribution
well, while recovery across the frozen starts is unreliable. The fresh
best-two gap is about 64.7 times its tolerance and far larger than its Monte
Carlo uncertainty. Increasing verification precision would not address the
observed initialization failure. The result concerns the implemented
synthetic family at diagnostic amplitude 1.25 on provisional A8 metadata
support. It does not calibrate the required 0.100-nat fixture, complete
Panel A, evaluate D12-to-A8 transport, or support a biological grouping or
continuity claim. All real regional projection outcomes remain closed.

## Artifacts and resource accounting

- [Actual terminal summary](population_h_recovery_summary.json)
- [Actual job log](population_h_recovery_45736553.out)
- [Terminal scheduler accounting](population_h_recovery_45736553_scheduler.txt)
- [In-job structural-check artifact](population_h_recovery_45736553_self_test.json)
- [Terminal QC figure](../figures/population_h_recovery_terminal_qc_20260930.png)

The **QC** figure shows the capped empirical start and fresh best-two score
gap, directly from the terminal summary. Its error bar is ±1.96 MCSE over
synthetic draws, not uncertainty over biological animals. The figure was
rendered with the existing Matplotlib module and visually inspected.

The attempt retained its original three CPUs / 16 GB / 72-hour request and
used 158.4267 allocated CPU core-hours. Its reported total CPU time was
53:25:21 and maximum RSS 2,246,348 K. No extension or new submission occurred.
The episode's global 120-hour clock origin and complete earlier qualification
accounting remain undocumented; this report does not reset or increase either
global ceiling. The user's subsequent instruction to babysit was followed
through this existing attempt's completion.

## Next authority gate

Stop before another launch. A revised H initialization strategy must be
prospectively decided and its budget reconciled before a new attempt, retaining
this failure and the acceptance thresholds. Any approved follow-up needs a
fresh frozen verification stream; stream 5601 is now inspected. Do not silently
discard the empirical start, raise the iteration cap, relax agreement, or open
real outcomes. No successor was launched and no episode scientific terminal
or acceptance is asserted.

## ASTRA export, separate from the diagnostic result

The existing terminal export path succeeded without validation warnings.
Its single-analysis snapshot is retained at
`../astra/v0.0.14/legacy_terminal_snapshot_20260930.yaml`. The adapter replaced
the active experiment program, so the C/H program and its supported synthetic
finding were restored at `../astra/v0.0.14/astra.yaml`. Both pinned schema and
semantic validation passed; the actual result is
`../astra/v0.0.14/terminal_validation_20260930.json`. The compatibility snapshot
does not preserve experiment dependencies or grant scientific authority.
