# Governance

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

Contracts, data boundaries, budgets and stopping criteria remain frozen. No new execution authority or canonical acceptance was granted. Formal episode disposition awaits the already-asked question.

Earlier dated entries below are historical snapshots.

## Continuation — 2026-09-30, 23:17 UTC

This continuation adds no execution authority and preserves all frozen v3
contracts, data roles, budget and stopping rules. Only existing job recovery,
required resource accounting, actual completed QC preservation and plotting
occurred. The proposed within-target biological endpoint remains inactive.
Final/unassigned/SWC/M0-full routes stay closed. No canonical mutation occurred.

## Authorized execution repair — 2026-09-30

The active v3 continuation repaired an evidenced serial walltime failure using
twelve independent single-thread case workers, within the unchanged frozen
resource ceilings. Fixture, model families, thirty fits, thresholds and report
semantics are unchanged. Replacement optional job is 46115632; timing 45881554
continues unchanged. Final/unassigned/SWC/M0-full access remains sealed. This
does not activate the separately planned terminal-profile endpoint.

## Scientific-design revision — 2026-09-30

The user authorized scientific-question/paper rewrites, not final-data opening,
scientific promotion, canonical action, or mutation of frozen execution.
GOAL/DATASETS/paper_plan now distinguish the planned terminal-profile primary
from the operative v3 M2-minus-M1 observation-pattern primary. SEARCH_POLICY,
the executable ASTRA program, Stage-0 tools/contracts, data roles, and existing
jobs are unchanged. The proposed positive-context endpoint is not activated;
it needs a separate prospective measurement/evaluation contract. Historical
control-state entries below remain historical and do not substitute for the
latest execution-ledger handoff or fresh scheduler observation.

This file is a local projection and grants no Brain Researcher, Society,
reward, confirmation, ClaimCard, scientific, memory, or Landscape authority.

## Observed control state

- Local run: `ep10-stage0-20260926T063634Z`.
- Local phase: Stage 0 v3 development-only M0 strong-Wolfe solver revision
  preparation after v2 failed closed; target-blind M1 geometry passed.
- Local decision: v1 and v2 failed their frozen gates; exploratory v3 target
  construction job `45463647` completed with exit 0 and exact postflight
  passed at 194/200 cells. Independent science and machine-policy audits of
  the positive-only amendment passed. Because job `45463647` used a live EP11
  Scratch atlas path, it is retained only as historical numerical evidence and
  is barred from downstream modeling input. Target-blind interface/math job
  `45470073` and its independent postflight passed. Replay attempt `45471384`
  failed before target access because the fresh overlap summary lacked the
  validator's compact policy-by-group view; an adapter added that view without
  changing counts or gates. Replacement replay `45472058` and independent
  postflight passed using EP10-local inputs. The M0 contract/engine and exact
  194-row, 12-outer/11-inner LOGO design are now frozen. Target-free exact-DP,
  gradient, deterministic-optimizer, planted-recovery, and fold-accounting
  tests passed; independent audit passed after the interface was bound to the
  canonical sealed roots. Jobs `45475171` and `45476204` both failed closed
  before output on the frozen 500-iteration L-BFGS cap; the diagnostic localized
  the first failure to outer `212074`, inner `220187`, `lambda=1e-5`. Independent
  reviews authorized only a versioned solver-only probe with the objective,
  grid, line search, and tolerances unchanged. Solver v2 and its `<1000`
  warm-path headroom gate passed independent implementation review. Exact-split
  job `45483386` failed closed at low ridge: `lambda=1e-6` exceeded the frozen
  headroom and `lambda=1e-7` exhausted 2,000 iterations. No production fit is
  authorized; the predeclared expanding strong-Wolfe review is active. The M1
  materializer passed audit
  after path sealing was strengthened. Launcher-preflight failure `45481507`
  opened no input; corrected target-blind geometry job `45481754` completed and
  its exact 16/3,104/120-row sealed postflight passed. M1 has not been fit.
- Canonical EP10 loop: none observed in the complete 19-record owner-scoped
  loop listing re-read at 2026-09-26T07:13:00Z.
- Canonical Goal handoff: none observed; no identifier was guessed.
- Society: not called.
- Scientist reward: not requested or recorded.
- Confirmation execution: not authorized or started.
- Scientific acceptance and Landscape transition: false.
- ASTRA: non-authoritative plan projection exported to
  `outputs/astra/v0.0.14/astra.yaml`; no findings or authority asserted.

## Data and outcome gate

The SEU metadata-to-SWC name join passed, but MOp has 17 conservative specimen
groups rather than the 20 required for 12 development and 8 final groups. The
Gao metadata contain 837 MOp cells across 76 joined fMOST/sample units, which
passes the released brain/specimen-count screen. Public Gao records provide no
animal, mouse, or donor key, so the maximum current interpretation is cross-
brain/specimen-unit support, not cross-animal replication. Canonical soma-root
rows were consumed for all 837 cells and all were atlas-geometry eligible. The
frozen 500-um/0.20 common-support condition supports 12 development and 8 final
whole units. Fifteen units overlap EP11 and are all forced EP10-unassigned; the
remaining assignment is recorded in `stage0/gao_mop_frozen_roles.csv`.

The target vocabulary and reconstruction-level target observation rule are now frozen in
`stage0/projection_observation_contract.yaml`, and the exact 200-file
development-only handoff is frozen in
`stage0/gao_mop_development_manifest.csv`. Service job `45438445` acquired
exactly those 200 development files through the Slurm `service` partition;
final, unassigned, unsupported, and EP11-shared files remained absent.
Constructor job `45438680` retained 5/200 globally eligible cells and passed
none of the 12 frozen group minima. The failures were 141 tree/parser, 22
annotation-zero axon, 6 outside-volume axon, and 26 nongrey terminal. Diagnostic
job `45439110` resolved every parser failure to a transition out of provider
node type 10, 11, or 12. Those tags occur internally in the provider trees but
the frozen contract permits them only as terminal markers. This is a
resource/contract incompatibility rather than a constructor crash.

That predeclared gate remains the historical v1 result. Provider-native v2
subsequently retained 11/200 cells under complete-observation requirements.
Its target-blind overlap diagnostic showed that tolerating both segment and
terminal mapping failures would retain 194/200 cells, including 7 cells in
group `221588`; the diagnostic itself did not authorize a change or inspect
target outcomes.

The scientist then explicitly authorized v3 as an exploratory positive-only,
unmapped-tolerant endpoint with that exact 194-cell gate. V3 does not turn
unknown zeros into negatives: unmapped content is omitted and a zero means no
mapped positive terminal observation. Job `45463647` constructed only the
development observations and passed exact postflight: 194 eligible, six
`soma_not_MOp` exclusions, the frozen group vector with `221588=7`, and all 16
zero-semantics columns. The positive-only estimand was frozen in the top-level
contracts before emitted target summaries or calls were inspected, and its
independent science and machine-policy audits passed. A later provenance audit
found that job `45463647` used a live EP11 Scratch atlas path, so it remains a
historical result and cannot supply downstream inputs. Target-blind job
`45470073` completed with exit 0 in 6:16 using 1.06 GB, and independent
postflight passed its exact 194-row, six-`soma_not_MOp`, 12-group,
segment-unmapped 109/85, and six synthetic-check gates. It accepted no target
calls and opened no outcomes. Replay job `45471384` failed with exit `2:0` in
13 seconds before target access because the fresh rich overlap summary lacked
the frozen validator's compact policy-by-group view. An adapter supplied that
view without changing counts or gates. Replacement job `45472058` completed
with exit `0:0` in 9:26 using 1.10 GB, the EP10-local steward atlas, fresh
target-blind evidence, and a new output root. Independent postflight passed
194 eligible / six excluded, the exact 12-group vector, all 16 zero-semantics
columns, `K_i_obs` distribution `1:6, 2:24, 3:24, 4:31, 5:28, 6:49, 7:17,
8:7, 9:5, 10:3`, 188 pair-informative cells, and exact interface-identity
agreement. The canonical model inputs are the direct episode Scratch children
`gao_mop_development_targets_v3_ep10_assets` and
`v3_model_interface_preflight_v1`; the sibling-backed historical
`gao_mop_development_targets_v3` root remains rejected. The frozen M0 engine's
target-free synthetic tests passed exact DP (maximum log-partition and marginal
absolute errors `1.7763568394002505e-15` and `1.9984014443252818e-15`), gradient
(maximum absolute error `4.1300318e-10`), deterministic optimization, planted
recovery (`8.0512457e-8` bits mean conditional KL), and exact 194-row,
12-outer/11-inner fold accounting. Independent audit passed after sealing was
strengthened. Jobs `45475171` and `45476204` preserved that gate but failed
closed on numerical convergence before any M0 output; the exact diagnostic
location is outer `212074`, inner `220187`, `lambda=1e-5`. Solver-only v2
passed implementation review, but targeted job `45483386` failed closed at low
ridge before completing one path. Strong-Wolfe v3 is under solver-only review;
production M0 remains blocked on a passed probe. Target-blind M1 geometry job
`45481754` completed after launcher-only preflight failure `45481507`; exact
postflight passed without target calls, SWCs, realized routes, final, or
unassigned access. M1 remains unfit. Final acquisition/evaluation and all
unassigned trees remain sealed. Local authorization and ASTRA export are not
canonical Society or scientific acceptance.

Outcome-blind compute preflight job `45412269` completed with exit 0 and
reproduced the support and role boundaries exactly. Its 99-suite synthetic
exact-normalization result is a throughput measurement only and grants no
permission to open final outcomes or begin candidate search.

## Development M0 milestone — 2026-09-27

Versioned V5--V7 solver probes preserved the frozen likelihood, ridge grid,
LOGO design, and tolerances while resolving separately the finite-precision
stall and true weak-ridge conditioning. Production job `45497012` completed
`0:0`; independent mechanical audit passed 194 cells, 12 outer folds, zero
boundary selections, full-development `lambda=0.1`, and 1,741 valid convergence
witnesses. This authorizes only the local development M0 reference. The
separate minimal joint M1 model contract is now frozen, but it does not
authorize real M1 until its target-free and actual-K-only pre-fit probes pass,
and it grants no final/unassigned, candidate, Society, reward,
confirmation, ClaimCard, or Landscape authority.
