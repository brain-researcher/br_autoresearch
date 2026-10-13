# Proposed M1 production wall-limit amendment — approval pending

Status: DRAFT, not frozen and not execution authority.

Change only the clean M1 production job's requested wall time from 08:00:00 to
12:00:00, superseding the wall-time field of
`outputs/stage0/v3_m1_numerical_execution_addendum_v3.yaml`.
This does not reset or enlarge the episode budget.

Evidence: job 46330399 completed the frozen recovery and all-K8 numerical fits
successfully, then correctly stopped at the resource gate (exit 2).
The complete 12-lambda path took 187.18363271886483 seconds.
The required 157-path projection is 8.163286204683827 hours.
The existing eight-hour request permits at most 6.4 projected hours under
the retained 25% headroom rule. A request of at least 10.204107755854784 hours
is required; 12 hours gives a 9.6-hour projection limit.

Preserve the one requested CPU, AMD EPYC 7502/OpenBLAS Zen backend, one effective
numerical thread, 16 GiB request, all model/solver/selection/acceptance rules,
157-path budget basis, 25% headroom, data boundaries, and stopping criteria.
Charge up to four rounded allocated CPUs conservatively (48 core-hours maximum).
All actual historical usage and the sole wall-reset epoch remain recorded.
The existing live jobs retain their full reservations.

Current resource reconciliation is
`outputs/stage0/resource_accounting_20261002_m1_refresh_terminal.json`.
After existing jobs' full remaining reservations, 89.04 wall-hours and
1042.202944 allocated core-hours remain unreserved. A 12-hour/four-allocated-CPU
M1 reservation would leave 77.04 wall-hours and 994.202944 allocated core-hours.
These are conditional planning values, not a new reservation or launch.

Reuse the passing 36-fit qualification, population recovery and all-K8 numerical
results; do not rerun them to relabel the failed eight-hour resource decision.
Preserve the original failed report and evaluate the same measured projection
against the prospective approved limit in a separate execution record.

Before a clean real-data refit, repair the stale qualification binding and the
runner's distinction between historical M0 opening and current-process access,
run only the focused regression, and reconcile current resources.
Historical M0 access stays true; the new fitting process cannot open M0 artifacts.
Never initialize from historical fitted M1 coefficients.

This approval would cover only the M1 production wall-limit amendment.
It would not authorize real M2, null replays, final/unassigned/SWC access,
biological promotion, a successor episode, or a second wall-time reset.

