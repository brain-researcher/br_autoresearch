# Minimal remaining model-family coverage on the fixed LMP anchor

Prepared after the complete four-trial development panel on2026-10-03.
**Adopted prospectively by the coordinator; implemented, unrun.** No new source values or fit informed this proposal. Root selected
both recipes and the RBF-only constant-coordinate rule below before scoring.

## Why these two comparisons

Full-channel LMP is the strongest current global development anchor, with
positive primary increments in15/16sessions and positive identity increments
in all16, but it falls short of both practical margins. Keep that entire
representation fixed and test the two still-missing minimum model families:
one reduced-rank linear reference and one bounded nonlinear RBF reference.
Do not add band, window, lag, electrode-count or baseline variants here. This
is a result-informed development choice of an anchor, not a new confirmation.
The prior scalar/LMP/low/high results remain charged, intact and reported.

Reuse the original source, sixteen session identities, every frozen trial,
eight complete held-direction folds, current-bin LMP electrode vector and
0:+14 native window. Preserve training-only per-feature mean/SD (zero SD→1),
the equal-direction first harmonic and shared residual intercept, native XY
residual target, full-trajectory denominator and exact within-direction
wrong-pair mean. The same five normalized-loss penalties and exact stronger-
shrinkage tie rule apply. Each outer session's policy parameter selection uses
only the other fifteen banks. Both animal grouping and0.01/0.005 margins stay
unchanged. No population-spike values or future-session source are needed.

## R: one rank-one regularized response map

Let Z be the standardized fitting-bin matrix, R the centered native XY
residual, N its fitting rows, and H=Z'Z+N*lambda*I. Write

`C = H^{-1} Z'R`,
`S = R' Z C`,
`C_rank1 = C v1 v1'`,

where v1 is the leading orthonormal eigenvector of the symmetric2×2 response
metric `(S+S')/2`. This minimizes the same regularized least-squares objective
subject to coefficient rank at most1. It is the response metric of the
regularized problem, not unweighted Euclidean truncation of C. Rank2 is the
already executed full-response ridge anchor and is reused, not counted as a
new scientific trial. Fix rank1 here rather than choosing rank from the outer
session's score. Eigenvector sign has no effect; an exact tied eigenvalue uses
the native X response axis, and an all-zero metric yields a zero map. No
numerical eigenvalue threshold or score-tuned rank repair is introduced.

Fit the response projection only from the seven fitting directions for each
lambda; predict the eighth. Select lambda through the same outer15-session
selector. Preserve all native XY outputs after adding back the shared
behavior prediction. This tests whether one shared linear residual output
direction improves shrinkage/generalization; it does not infer a one-dimensional
biological population state.

## K: one explicitly approximate finite-capacity RBF reference

Use an RBF kernel on exactly the same standardized fitting LMP vectors Z.
Choose a fixed, training-only Nyström approximation so the nonlinear reference
does not require a dense roughly15,000×15,000 kernel for large sessions. This
is explicitly **Nyström RBF ridge**, not exact full-kernel ridge.

For each held-direction fit, choose one complete fitting trial in each of the
seven legal directions. Within direction, sort original trial IDs, initialize
Python `random.Random` with seed version2 and literal
`ep05-rbf-landmark-v1|20261003|<session>|<held_direction>|<training_direction>`,
and take one `randrange` index. Use all fifteen native bins from each chosen
trial, in canonical direction/trial/time order: exactly105 landmarks. This
selection uses identities only and is shared by every lambda. No trial or bin
is split between fitting and evaluation; landmarks are basis points drawn
from the already legal fitting pool. No favorable seed, landmark count or
outcome-dependent subsample is chosen.

Fix `sigma_squared = D_active`, the number of fitting feature columns with
strictly positive original training SD before zero→1 replacement. Each
nonconstant standardized column has population variance1, so independent
training draws have mean squared separation2*D_active; the kernel

`k(z,z') = exp(-||z-z'||² / (2*sigma_squared))`

has a dimension-based distance scale set by the existing standardization.
This is a fixed bandwidth rule, not a median-distance search or a claim that
the empirical distribution is Gaussian. For **RBF only**, set coordinates of
training-zero-SD columns identically zero in both fitting and evaluation kernel
inputs, retaining their identities and counts. An evaluation value can vary
even when a feature was constant in training; zero-SD→1 scaling alone would
not remove that difference. Linear recipes keep zero-SD→1 unchanged. If
D_active=0, use the exact zero centered-kernel feature branch,
which predicts the common behavior baseline. Do not fit a bandwidth floor.

Let L be the lower Cholesky factor of `K_landmark + 1e-8*I`, and define
`Phi = K_to_landmark L^{-T}`. The1e−8 diagonal regularizer is fixed relative
to the unit diagonal of this kernel; it handles duplicate/highly redundant
landmarks without deleting them, an eigenvalue cutoff or outcome-based
capacity repair. It is part of the approximation, not claimed mathematically
identical to unregularized Nyström. Failure of the specified finite positive
factorization stops for numerical diagnosis, never an adaptive jitter sweep.

Subtract the fitting-row mean of each Phi column, use that same mean for
evaluation features, and fit the centered native XY residual with the same
`N*lambda` ridge penalty. No subsequent feature standardization, whitening,
rank grid, feature selection or target scaling is added. The feature metric
is the declared approximate RBF kernel; it is not capacity-matched to the
linear input metric merely because both use the same penalty grid. That
limitation is reported and cannot be hidden by changing the grid afterward.

## Shared evaluation, controls and count

The full LMP ridge result is the fixed reference for both recipes, including
its out-of-direction common behavior baseline. Report every session, both
animals and the complete correct/wrong-pair increment for all three families;
do not pick a different family for each session. Validate exact baseline and
trial/order identity when reusing reference scores. New fits must not read
the reference's scientific score until their parameters are fixed. Existing
development exposure is acknowledged; the numerical boundary does not reset it.

Two complete sixteen-session panels add **two valid scientific trials**,
bringing the running total4→6. The rank2 anchor, five lambda values,
landmark identities, 128direction folds and technical pilot do not add trials.
These two panels would cover the listed initial predictor families; they do
not complete28–64trials,12-trial patience, the post-initial40% alternative-
check requirement, required falsifiers or candidate promotion. Reuse of the
anchor does not establish a second independent replication or make this
one change sufficient for all adaptive/challenger requirements.

## Concrete implementation and measured-cost decision after adoption

Use a separate capacity runner and preserve every current recipe/bank. Reuse
existing LMP/XY source identity logic, not another episode's representations
or learned states. A future selected source handoff may cache only the exact
LMP/XY window if it reduces repeated I/O; no cache or source replay is created
by this proposal. The RBF fitting feature matrix is at most about15,570×105
float64 entries (about13MB), plus the existing LMP arrays and small105×105
systems. Kernel construction uses bounded row chunks. This is a working-array
estimate, not a measured source-I/O runtime or a guarantee about BLAS workspace.

The coordinator selected the same complete Chewie_20150313 representative for
both new recipes at1CPU/4GiB/5minutes, before selecting the remaining-panel
deadline from its actual cost. Retain its full banks and never abandon a
recipe using pilot scores. Focused tests: rank-one ridge against the augmented-
design best-rank-one objective; Nyström kernel/feature orientation and duplicate-
landmark handling; held-direction changes cannot alter training landmark/
bandwidth/centering parameters; and constant features reduce to the shared
baseline. Reuse exact pairing/session-selection fixtures. Root owns all
tests, source access and allocation decisions. The separate implementation is
`tools/primary_capacity.py`, `capacity_lmp_reader.py` and
`run_primary_capacity.py`; six focused fixtures are in
`tools/test_primary_capacity.py`. No source cache is required or created:
one authorized LMP-only read supplies both fixed recipes, avoiding rereads
of unused low/high-band values. The complete pilot preserves its banks for
the later full panel. Reference target/baseline/trial identity is compared
exactly only after every new session fit is fixed; prior scores never enter
the new fitter.
