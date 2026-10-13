# Memory

## Monitoring unavailable — 2026-10-02, 12:23 Pacific

Clean M1 job46332190 started and passed runtime/backend/access gates. Last
verified at19:19:39UTC: M1, optional46237243 and diagnostic46325527 were RUNNING.
The command tool then failed twice before shell creation with
`No file descriptors available (os error 24)`, one minute apart. No scientific
failure or Slurm outage is inferred; no jobs were cancelled or duplicated.
Active monitoring requires restored command execution. Exact resume handoff:
outputs/stage0/monitoring_handoff_20261002_clean_M1.md. ASTRA validation PASS;
this tool failure is separate from scientific outcomes and export.

## Approved clean M1 refit submitted — 2026-10-02, 12:14 Pacific

User approved the proposed12h M1 wall limit. The narrow amendment and current
resource reservation are recorded; historical budgets and all scientific/data
rules remain unchanged. Clean real-data M1 job46332190 is submitted after
5gate and6checkpoint/CLI regressions passed and focused review found no blocker.
It binds qualification46328672 and numerical refresh46330399, preserves
historicalM0opening45760290, and forbids M0 access by the new fitting process.
Optional46237243 and diagnostic46325527 continue. All reservations leave77.04
wall-hours and994.202944core-hours unreserved. ASTRA12hprogram validationPASS.

## M1 resource decision required — 2026-10-02, 12:02 Pacific

Job46330399 passed population recovery and all-K8 numerical fits, then correctly
stopped at the frozen eight-hour resource gate:157-path projection8.1633h versus
6.4h with headroom. Actual results and scheduler telemetry are preserved.
The drafted12h M1 limit amendment awaits user approval; it stays inside existing
episode ceilings. Passing numerical evidence will be reused. No clean M1 or
real M2 launched. Optional46237243 and diagnostic46325527 remain running within
existing authorizations. After their full reservations,89.04wallh and
1042.202944coreh remain unreserved. ASTRA validation remainsPASS; see execution
ledger and resource_accounting_20261002_m1_refresh_terminal.json for details.

## Affected numerical refresh submitted — 2026-10-02, 11:49 Pacific

Qualification46328672 PASS:22engine tests,36fits,12comparisons; actual artifacts
are preserved. Required544-row recovery and all-K8 throughput refresh46330399
is submitted (8h,1requestedCPU,16GiB,frozen backend), reusing passed qualification
and unchanged ranks. Recovery46237243 and permutation diagnostic46325527 continue.
All current reservations leave81.111389wall-hours and1010.417111core-hours
unreserved. Real M1/M2 remain gated. See execution ledger for actual artifacts,
BR review impact, minor filename-search scope deviation and ASTRA validation.

## Required M1 requalification — 2026-10-02, 11:34 Pacific

Existing36-fit target-free qualification46328672 is submitted alongside running
recovery46237243 and permutation diagnostic46325527. Fresh affected numerical
qualification is required by the frozen contract after the derivative repair.
BR-directed focused native critique is resolving affected population/timing
reuse and truthful historical M0 handling before any clean real M1 refit.
BR verdict ext_review_verdict_4fe160ae4283 supports target-free continuation
but requires affected544-row recovery and all-K8 throughput refresh before
production; unchanged geometry/ranks remain reusable. No real M1/M2 job is launched. All current reservations leave88.660833wall-hours
and1041.516allocatedcore-hours unreserved; no additional reset.

## Derivative repair replay — 2026-10-02, 11:11 Pacific

The candidate-level direct-derivative defect is repaired; six focused tiny-array
regressions pass. Acceptance and confirmation rules are unchanged. Bounded
permutation replay46325527 is RUNNING at11:30Pacific,15m52s elapsed,
with actual CPU7502/Zen/effective1thread attestation passed
(4h,1requestedCPU,8GiB,frozen backend).
It has not yet established that the observed failure is resolved. Recovery
46237243 continues with3/12cases complete. Preserve all prior M1 artifacts,
but require affected numerical revalidation before reusing them under the
changed evaluator. M2 remains closed.89.160833wall-hours and1042.516core-hours
remain unreserved after both live requests; same wall-reset epoch.
See the execution ledger and stage0/readiness_status.yaml for exact next action.

## Current status — 2026-10-02, 11:01 Pacific

Recovery46237243 remains RUNNING at19h41m10s: three reused cases complete,
nine without completed results,28h18m50s left. Diagnostic46270474 is terminal
FAILED after1h12m33s and now durably preserves the real numerical failure on
the frozen one-thread backend. Ordinary convergence passed, but its mandatory
confirmation had direct derivative exactly0; the frozen certificate rejects
this failure class. No solver change, acceptance relaxation or retry is made.
Actual artifacts: `stage0/v3_m2_falsifier_permutation_diagnostic_job_46270474/`.
The next step is the bounded contract-preserving defect assessment and the
same live recovery job. M2 remains closed. Resource accounting now leaves
93.160833wall-hours unreserved with all previous CPU usage retained; see
`stage0/resource_accounting_20261002_diagnostic_terminal.json` and the execution
ledger. Existing scientific interpretation, Formal Society boundary and ASTRA
program remain unchanged. Earlier dated entries retain their historical scope.

## Approved continuation — 2026-10-01, 21:27 Pacific

Optional recovery46237243 is running and retains three reused passing cases.
Timing46237247 failed during permutation M1; its numerical exception also
broke process-pool transport. Actual failed outputs and progress are durable
under `stage0/v3_m2_falsifier_timing_v1_job_46237247/`. The transport repair
passed21 focused regressions and preserves scalar diagnostics and other probes.
Only the identical failed permutation probe was resubmitted as46270474:
4hours, one requestedCPU,8GiB, frozen AMD7502/Zen/one-thread backend check.
The solver and scientific thresholds are unchanged. Follow these two active
attempts in `experiment_log.md` and `stage0/readiness_status.yaml`.
The wall reset remains the same epoch; this approval does not reset it again.
After accounting for the failed job, live optional reservation and new4-hour
diagnostic,90.37wall-hours remain unreserved. Empirical M2 remains closed.
A partial/incomplete numerical QC figure is retained at
`figures/v3_m2_falsifier_partial_checkpoint_job_46237247.png`; it supports no
full timing projection or biological conclusion. No new scientific decision
or formal Society transition follows; prior BR review remains applicable.

## Resumed producers after wall reset — 2026-10-01, 15:14 Pacific

User-authorized wall-time reset is recorded; optional resume46237243 and timing
replacement46237247 are submitted and were pending at22:14:19UTC. Both request
48hours and8GiB; numerical worker counts are9 and8. New-window wall reservations
total96hours, leaving48hours. Cumulative CPU accounting remains in force.
Optional reuses the three original passing cases; timing saves complete-probe
measurements and phase/count progress. The focused repair regressions passed
10/10 and17/17. Follow these same job IDs; no duplicate or empirical M2 launch.
See `stage0/resource_accounting_20261001_wall_reset_launch.json` and latest
`experiment_log.md` for exact resource/evidence bindings.

## Wall-time reset — 2026-10-01

The user explicitly authorized resetting wall time. A fresh 144-hour strict-wall
window now applies under `stage0/v3_wall_time_reset_authorization_20261001.json`;
129.832667 historical wall-hours and 302.025667 cumulative allocated core-hours
remain recorded. The pending disposition is resolved in favor of same-episode
continuation. Required missing producer evidence still gates empirical M2;
three completed recovery cases are retained for reuse. Scientific contracts,
closed data, formal Society and other resource ceilings retain their scope.
Use the latest `experiment_log.md` and readiness block for execution state.
Earlier dated blocked/terminal entries describe the previous accounting epoch.

## Terminal producer handoff — 2026-10-01, 13:14 UTC

Both recorded producers are terminal: optional 46115632 TIMEOUT with 3/12
cases complete; timing 45881554 TIMEOUT at 48:00:29 without a supplement.
The current search is incomplete at the preaccess resource/human gate. Final
conservative accounting leaves 14.167333 strict wall-hours; the unchanged
24-hour scale request cannot be reserved. No empirical M2 or full-controller
replay launched. Actual evidence, the scientific review and QC figures are
preserved; see `stage0/producer_terminal_handoff_20261001.md` and
`experiment_log.md` under `v3_search.adaptive_model_search`.

The pending disposition question is the next action. Reuse passing unchanged checks and the returned BR review; do not substitute infrastructure work or an identical rerun for this resource gate. The planned terminal-profile study remains inactive.

Earlier dated entries below are historical snapshots.

## Continuation — 2026-09-30, 23:17 UTC

Preserve the prior strict wall-budget basis as a conservative floor when current
Slurm totals do not reproduce it; do not treat the discrepancy as new authority.
All 80 recorded job IDs resolve. The carried-forward snapshot leaves 14.18122
wall-hours after existing-job reservations. One low-rank recovery case passes;
eleven cases and both final producer reports remain pending. No new experiment
is justified while those required producers run.

## Execution lesson — 2026-09-30

The optional qualifier lost all recovery progress at 4-, 6- and 12-hour serial
timeouts because it only wrote its final report. Replacement 46115632 executes
the identical twelve family/rank cases in twelve spawned workers and saves
each completion. Keep both repeated fits and each hybrid anchor in one process;
solver identities cannot be passed between workers. Timing 45881554 is unchanged.
Scale 45924664 and observed 45933922 cancelled at zero compute. Continue from
the dated execution log and current readiness block.

## Current design lesson — 2026-09-30

Source modules, spatial wiring, generic morphology diversity, and co-target
laminar association are already published (Falasconi 2026, Gao 2025, Liu 2024,
Yuan 2024). The proposed EP10 contribution is a named within-A terminal
implementation prediction across recorded-positive contexts, beyond flexible
continuous source space, with branch-level explanation and a calibrated pooling
consequence. Contexts can overlap; unknown is not negative. The biological
primary is planned separately from the frozen active v3 statistical primary.
No outcome or runtime state was changed by this writing repair.

This is a local research projection, not Brain Researcher memory, and it has
not been promoted outside EP10.

## Reusable local lessons

- SpreadsheetML cell identities may contain `_x005F_` escapes. A one-pass
  decode is required before joining `Morphology Name` to SWC basenames. The
  repaired 1,876/1,876 join is recorded in
  `stage0/qualification_run005/qualification.json`.
- A redactor should skip forbidden cells before shared-string dereferencing;
  filtering only after whole-row decoding is an avoidable exposure.
- Large neuron count is not biological replication. SEU has 1,876 complete
  morphologies and 39 conservative specimen groups overall, yet the priority
  MOp source has only 17 groups and cannot meet the frozen 12/8 split.
- In the Gao portal, the SWC filename prefix joins one-to-one to released
  `fMOST_id`/sample records and is defensible as a brain/specimen unit, but the
  public records do not certify distinct animals. The official API supports
  outcome-blind source filtering and targeted per-neuron SWC acquisition, so
  EP10 does not need the 13.6-GB whole-release archive.
- Strict HTTP ranges can recover only the canonical soma root without opening
  an axon tree. All 837 MOp roots were atlas-geometry eligible. On the
  prespecified finite grid, 500 um/0.20 and 750 um/0.10 are incomparable
  minimal feasible conditions; EP10 froze 500 um/0.20 by a radius-first rule.
- Reserving every EP11-overlap unit as EP10-unassigned preserves EP11's future
  role options. The remaining metadata/geometry-only allocation freezes 12
  development and 8 final fMOST/sample units before projection outcomes.
- For MOp, replacing the coarse Allen `MO` target anchor with `MOs` prevents
  local source cable from becoming an apparent projection target. A full
  56-coordinate atlas map remains useful for QC, while an outcome-blind
  16-coordinate motor-routing panel keeps exact fixed-K normalization bounded
  at 12,870 subsets.
- A role-safe acquisition manifest should be joined at the cell level after
  group roles freeze. EP10's manifest contains the 200 common-support
  development trees and zero unsupported, final, unassigned, or EP11-shared
  files; keeping the other files physically absent is clearer than relying on
  labels in a mixed directory.
- The role scope is broader than the acquisition and inferential scope: EP10's
  development role has 209 cells, but the nine outside frozen common support
  remain unopened. Only the 200-cell common-support domain was acquired. The
  v1 190-cell complete-observation gate failed historically; it is not the v3
  gate. Reconstruction zeros never mean biological absence.
- Re-deriving safe soma common support and the role split in one compute job is
  a useful pre-transfer check: job `45412269` reproduced the 200/9 cell boundary
  and 12/8/56 roles exactly. Ninety-nine synthetic exact-normalization suites
  required 650.99 seconds, which bounds normalizer throughput but not the full
  adaptive-search/null cost.
- Provider-specific SWC tags require authoritative semantics before a
  terminal rule is frozen. In the Gao development trees, node types 10/11/12
  occur internally with children; EP10's prospective terminal-only marker rule
  consequently rejected 141/200 trees. An otherwise clean metadata, soma, and
  group-support screen cannot substitute for a small outcome-blind format
  compatibility sample when custom node types determine global eligibility.
- A frozen attrition gate prevented post-outcome repair: only 5/200 cells were
  eligible, all 12 group minima failed, and even rescuing every parser failure
  could yield at most 146/200. The correct action is a new prospectively frozen
  revision, not reinterpretation or threshold relaxation in the current run.
- Provider-native semantics later resolved types 10/11/12 without relabelling:
  an axon edge is included when neither endpoint is type 3/4, and a terminal is
  a topology leaf whose raw type is not 3/4. Complete-observation v2 still
  retained only 11/200 cells.
- The scientist prospectively authorized v3 using the target-blind overlap
  count: exactly 194 development cells and group `221588` support of 7. V3 is
  positive-only and unmapped-tolerant. Unmapped content is omitted from target
  assignment, and all zeros are unknown/non-negative observations. This
  changes the scientific estimand; complete-set M0/M1/M2 cannot be reused
  unchanged.
- A scientifically compatible shared atlas is not an acceptable run input when
  reached through a live sibling episode's output path. Stage the authorized
  steward/common source into the active episode's Scratch namespace, preserve
  the sibling-backed run only as history, and replay to a new episode-local
  output before downstream use.
- A converged toy optimizer does not qualify a 255-parameter weak-ridge fit.
  Preserve cap exhaustion as a numerical failure, localize the exact fold and
  ridge value, then test a versioned solver-only change on that complete warm
  path before a full nested-CV resubmission. Do not respond by relaxing the
  scientific grid or convergence tolerances.
- A narrow strong-convexity certificate is appropriate only for an exact
  all-finite line-search resolution stall; it must leave the iterate, gradient,
  history, counters, and prior relative-change witness unchanged. Genuine
  weak-ridge conditioning required an exact-Hessian Newton branch, not a looser
  stopping rule. Extending that branch prospectively through `lambda=1e-5`
  restored warm-versus-zero agreement without changing the objective, grid,
  or tolerances; production M0 then passed all 1,741 fit witnesses.
- Passing M0 does not license an outcome-adaptive M1. Freeze the global
  geometry design before inspecting M0 scientific gains, refit every parameter
  within each fold, and benchmark exhaustive fixed-K enumeration before using
  the already materialized target-blind geometry.

## Approaches not carried forward

- Directly mounting or unpacking the mixed-role SEU archives was rejected by
  the steward quarantine contract.
- Pooling MOp with MOs or SSp to manufacture group support was rejected by the
  episode policy.
- A shared SEU EP09--EP11 split was not frozen because no SEU source population
  passed EP10's minimum group gate. EP10 instead uses Gao and reserves all 15
  Gao units shared with EP11 as unassigned.

## Open questions

- Can the provider or authors certify the animal relation among Gao fMOST/sample
  units? Until then, claims remain at the conservative fMOST/sample-unit level.
- Can a development-frozen positive-only observation-pattern model provide a
  useful exploratory estimand without implying verified non-detection? This
  must be resolved before M0/M1/M2, while final and unassigned data stay sealed.
- Can a future trusted builder establish SEU-to-Allen transform compatibility
  and complete-target observability without exposing final roles?
