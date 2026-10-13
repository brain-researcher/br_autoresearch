# EP12 bounded-U contract-conformance audit

Date: 2026-09-27  
Scope: generated data and source code only; no real or final connectivity access

## Frozen definitions

The active scientific contract defines:

- `U_linear`: a bounded low-rank linear logistic-normal with latent dimension
  at most two;
- `U_curved`: a bounded smooth connected one-dimensional logistic-normal curve
  with a cubic probabilists-Hermite decoder and a unimodal Gaussian latent.

The development grammar separately varies representation/rank and covariance/
covariance-rank. The transition specification explicitly states that
covariance rank is not the representation/component parameter rank. No frozen
file says that U may add an unrestricted full-rank ambient Gaussian residual
around its declared low-dimensional manifold.

## Operational mismatch

The current default qualification path uses `count_aware_log_ratio` and
`shrinkage`:

- count-aware representation retains all 10 ILR coordinates for the 11-bin
  synthetic vocabulary;
- shrinkage returns a full positive-definite 10-dimensional covariance;
- U-linear uses that covariance as its complete Gaussian latent law rather than
  a rank-at-most-two factor;
- U-curved fits a one-dimensional cubic mean curve, then adds a full-rank
  Gaussian `noise_covariance` and scores a nine-node Gaussian mixture;
- `score_neurons` never reads the stored continuous `latent_offsets` or
  `log_weights` fields.

A generated curved-continuum probe (`n=19` per side, count total 100, strength
1.8, seed 1202) confirmed:

- U-linear prior covariance had 10/10 eigenvalues above `1e-8`, although its
  diagnostics recorded `covariance_rank: 2`;
- U-curved residual covariance had 10/10 eigenvalues above `1e-8`;
- replacing every stored latent offset and log weight by extreme values changed
  held-out scores by exactly `0.0` for both U implementations.

Thus the V3 numerical and interval pilots tested a stable implementation, but
not the bounded U family declared by the contract. They remain useful defect
evidence and runtime measurements; they cannot qualify the repaired family.

## Correct scoring architecture after a decision

For a pure bounded family, U-linear should use `eta = A z`, with
`rank(A) <= 2` and Gaussian `z`; U-curved should use
`eta = C [z, z^2-1, z^3-3z]` with scalar Gaussian `z`. Predictive scores should
directly integrate the raw multinomial likelihood over deterministic common-
random latent-space quadrature nodes. A singular manifold must not be converted
to a full-rank ambient Gaussian by eigenvalue flooring.

M must use the same allowed within-group continuous law and add only the frozen
two or three group offsets, so its advantage cannot come from an unshared
continuous/noise capacity. Continuous-law parameters, not a duplicate cache,
must also drive fitted-null generation.

If an ambient residual Gaussian is scientifically desired, its covariance
family, rank, floor, sharing across U/M, and parameter accounting must be
explicitly frozen. That is a different model family, not an engineering detail.

## Boundary and guard

Pure bounded manifolds versus manifolds plus a fitted residual tube can change
the best U, M-minus-U gain, null generator, and terminal classification. The
choice is consequential. Until it is made, the active qualification entry
point fails before expensive synthetic fitting, and no successor qualification
or readiness launch is allowed.

The CLI guard was exercised with a one-replicate smoke request. It returned
nonzero at the conformance check and did not create the requested output
directory. The focused scientific, interval, and Slurm-handoff suites then
passed 40/40 tests.
