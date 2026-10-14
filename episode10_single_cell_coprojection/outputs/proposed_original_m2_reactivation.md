# Original M2 reactivation — inactive source review

Prepared October 12, 2026, Pacific. **Not adopted; no execution selected.**
This note addresses the original Gao v3 M2 question, not the separately planned
terminal-profile study. It changes no operative contract, numerical source,
budget, data role or historical attempt. The user's standing approval to
increase compute answers the funding preference; it does not supply a completed
solver repair or a measured full-search cost.

## What is already settled

The [October 6 closeout](experiment_log.md#2026-10-06--v3_search--scientist-adopted-local-closeout)
is an adopted incomplete disposition. [SEARCH_POLICY.yaml](../SEARCH_POLICY.yaml)
lines 4–19 explicitly leave resumption and successors unselected. The October 1
wall-reset continuation was actually executed and subsequently closed; it is
not an unanswered approval request.

Retain **735.711222 allocated CPU-hours** consumed and **54.240833 charged
execution wall-hours** in the October 1 accounting epoch. The last ledger
retains **1,064.288778 CPU-hours** and **89.759167 execution wall-hours** under
the existing ceilings. Earlier consumed wall time remains historical. The
permitted records do not establish a later expired absolute calendar deadline;
do not invent a fresh window or reset any consumption. These values are the
last recorded ledger, not a new accounting measurement
([log, lines 64–74](experiment_log.md)).

Qualified M1 evidence remains usable: all 949 fits on the same 194 development
cells and 12 conservative specimen groups, with the recorded secondary
exploratory M1-minus-M0 gain. It is neither an M2 result nor evidence that M1 is
sufficient. No real M2 search, final group, unassigned tree or terminal-profile
outcome is opened by this proposal.

## The failure is narrower than four different broken models

The preserved public
[runtime log](stage0/v3_m2_optional_family_qualification_job_46237243_terminal_20261003/runtime.log)
lines 11–14 records four `LowRankError` iteration-cap failures:

| Qualification case | Actual failed solver |
| --- | --- |
| Low-rank rank 4, 6 and 8 | Pure low-rank fit at recovery rho 0.001 |
| Sparse-plus-low-rank rank 8 | Its prerequisite pure low-rank anchor at rho 0.1 |

The hybrid caller fits that anchor before entering its alternating optimizer
([qualify_v3_m2_optional.py, lines 366–382](stage0/tools/qualify_v3_m2_optional.py)).
The reported hybrid case therefore does **not** demonstrate failure of hybrid
alternating sweeps. All four observed failures reach the same low-rank engine.
The remaining interrupted cases are incomplete, not passed or scientifically
negative. Three completed rank-1/2/3 cases remain preserved under their original
qualification context; changes to the solver must not silently relabel their
evidence as qualifying the new implementation.

The engine uses the exact conditional objective plus the registered diagonal
penalty, a projected Stiefel gradient for U, and a Euclidean gradient for d.
Each iteration starts Armijo backtracking at step 1, halves it as necessary,
and uses deterministic QR retraction/canonicalization. It requires both
gradient infinity norm at most 1e-6 and relative objective change at most 1e-9,
within 20,000 iterations and 80 line trials per iteration. See
[fit_v3_m2_optional.py](stage0/tools/fit_v3_m2_optional.py), constants 35–55,
gradient 2451–2510 and iteration loop 2884–2975.

Independent source review by `ep21_finish_handoff` found no concrete algebra
defect: the half-weight symmetric edge adjoint gives the stated U/d gradients;
the tangent projection, signed QR and canonicalization are coherent; the
Armijo derivative is minus the joint squared gradient norm; and both accepted-
state convergence predicates and matching synthetic/production objectives are
retained. This is source review, not executed numerical qualification.

The U gradient contains the eigenvalue scale d, while the d
gradient uses a different scale; fixed unit-step steepest descent can be slow
in such coordinates. That is a plausible numerical explanation, **not a
diagnosis of these recorded runs**. Their old exceptions do not retain separate
U/d gradients, relative objective change, accepted step or iteration trajectory.
Consequently they cannot distinguish slow convergence, one lagging parameter
block, objective-rounding effects or another numerical cause. The richer
terminal error text now in the source was added after this failed attempt;
it does not retrospectively supply missing evidence.

## Smallest meaningful next implementation decision

First resolve the common low-rank solver, using its exact objective and caller
above. No concrete convergence-preserving source repair is established yet.
A larger job with the identical 20,000-iteration solver would reproduce the
same deterministic cap failure if it reaches that point. Increasing the cap,
relaxing either convergence predicate, deleting high ranks, changing the
initializer or adding multistart is not an already-authorized repair.

An analytically justified acceleration, if identified, must preserve the
model, exact normalization, fold-local penalty, original data support and both
convergence predicates. A changed search direction or step rule would still
change the frozen optimizer and require an explicit prospective numerical
selection and affected verification before empirical use; preservation of the
objective alone does not establish identical nonconvex solutions. This note
selects neither that change nor another qualification run. If source analysis
cannot determine a repair, the specific missing evidence is the failed
low-rank solver's progress and terminal convergence state, not another broad
simulation campaign or a new data inventory.

Any prospective reactivation must explicitly supersede the recorded closeout,
retain the 194 cells/12 groups and frozen 16-target positive-only semantics,
and retain every triggered family, 40–88 valid trials, 16-anchor coverage,
required falsifiers and all 99 complete controller nulls. Development-only
access and the prohibition on final/unassigned access remain intact. The
terminal-profile study is not a substitute for completing this question.

## Resource admission remains unresolved

The source-backed lower bound is at least **1,900 additional allocated
CPU-hours** for the existing implementation, obtained from an interrupted
required timing unit with other nonnegative costs omitted
([log, lines 166–188 and 288–302](experiment_log.md)). It is a lower bound,
not a complete timing estimate or an upper bound for a new solver.

Combined with retained consumption, merely accommodating that lower bound
would require a lifetime ceiling of at least **2,635.711222 CPU-hours**, before
unmeasured additional work. That arithmetic does not justify selecting that
ceiling as sufficient, requesting a 1,900-hour run, or resetting the ledger.
No finite new production reservation is supported by the current evidence.
The previously measured triangular-solve change did not accelerate the whole
path and does not supply an improved cost basis.

The proposal is therefore ready as a concrete inactive engineering handoff:
one common failed solver and missing diagnostic information are identified;
the repair, affected evidence and complete cost admission remain unfinished.
No new budget question, job, test, data read or scientific result follows.
