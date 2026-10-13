# Formal Inputs and Evidence Roles — Episode 13

EP13 has no participant dataset, images, BOLD time series, or statistical
train/test split. Its inputs are exact matrices, theorem statements, proofs,
and replayable certificates.

## Public online resources — checked October 6, 2026

The primary methods reference is [S. R. Searle (1965), *Additional Results
Concerning Estimable Functions and Generalized Inverse Matrices*](https://academic.oup.com/jrsssb/article/27/3/486/7026850),
JRSS B 27(3), 486–490, DOI
[10.1111/j.2517-6161.1965.tb00608.x](https://doi.org/10.1111/j.2517-6161.1965.tb00608.x).
The publisher's public record supplies the bibliographic identity, abstract
and references. Its abstract concerns estimating linear-model estimable
functions and their variance/covariance. The full article is a separate PDF;
this check did not assume unrestricted full-text access or retrieve it.

This gives a directly citable classical-methods foundation for EP13's
estimability framing and its novelty limitation. It is not a BOLD dataset,
an empirical validation set or proof that finite numerical examples establish
the general theorem. EP13's explicit separable criterion, zero branches and
constructive certificates are established by the retained episode proof and
exact replay, under the fixed domain below. No empirical source or additional
simulation is required for the completed scope.

## Fixed mathematical domain

All dimensions T, Q, V, and K are positive integers. The known objects are

| Object | Shape | Status |
| --- | --- | --- |
| A | T x V | fixed temporal observation operator |
| P | Q x K | fixed spatial observation operator |
| c | length K | fixed spatial target direction |
| s | length V | fixed temporal target direction |
| X = B^T | V x K | unknown and free over all rational matrices |
| Y = A X P^T | T x Q | exact observed matrix |
| tau = s^T X c | scalar | prespecified target |

The phrase “admissible latent matrix” means every element of
Q^(V x K). Nonnegative, low-rank, normalized, sparse, or singleton latent
families are not part of this episode.

The theorem is stated for arbitrary compatible A and P. An application may
construct A = R S H with

| Object | Shape |
| --- | --- |
| H | F x V |
| S | T x F |
| R | T x T |

R must be square. Idempotence is an application-model validity rule; symmetry
is not required. Neither idempotence nor the factorization A = R S H is needed
for the identifiability theorem after A is fixed.

## Frozen coordinate convention

Use column-major vectorization:

\[
x=\operatorname{vec}(X),\quad
y=\operatorname{vec}(Y),\quad
M=P\otimes A,\quad
\ell=c\otimes s.
\]

Every exact fixture must directly replay y = Mx and tau = ell^T x. A fixture
that checks only a Boolean identifiability label is incomplete.

## Evidence roles

| Role | Contents | Epistemic use |
| --- | --- | --- |
| exposed history | 13 previously described valid cases and four invalid controls; no retained local payload | design context only |
| proof | exact derivation for arbitrary compatible finite dimensions | universal support |
| examples | fixed exact positive, negative, zero, and orientation cases | implementation regression only |
| certificates | exact readout or target-changing invisible perturbation | constructive support or refutation |
| independent replay | separately authored rational checker and frozen fixture suite | transcription/orientation audit only |

Finite examples never prove the theorem. The proof does not become stronger by
running more randomly generated matrices.

## Required exact suite

The implementation regression suite must include:

- rank-deficient M with an identifiable nonzero target;
- temporal-only, spatial-only, and simultaneous loss;
- c = 0, s = 0, and both zero;
- realized tau = 0 in a nonidentifiable model;
- non-square asymmetric orientation cases;
- a non-symmetric idempotent R that requires R^T in the readout equation;
- nonunique valid readouts;
- normalized target-changing null witnesses; and
- malformed shapes, rationals, zero dimensions, and invalid application-model
  inputs as explicit rejections.

## Independence and access boundary

The current package completed on October 3 with local status
`theorem_verified`. The [proof](outputs/exact/proof.md),
[attempt lock](outputs/exact/runs/attempt001/package_lock.json),
[12-case orientation and seven-rejection report](outputs/exact/runs/attempt001/orientation_report.json)
and [independent replay](outputs/exact/runs/attempt001/independent_replay_report.json)
are retained. The independent checker was authored from the frozen contracts
and interface before its author inspected the main implementation, fixtures
or results; it imported no historical evaluator code, constants, expected
labels or precomputed answers. The [execution record](outputs/experiment_log.md)
documents that authorship boundary and actual completion.

There is no protected biological outcome to open. The separation is between
the authored theorem package and its separately authored replay. A replay
can find an implementation or orientation error; it
cannot create universal evidence beyond the proof.

Inputs remain read-only. Durable proof documents, exact fixtures, checker
source, replay reports, and the conceptual figure belong under outputs/.
Transient execution belongs under the episode-specific Scratch directory.

## Scope exclusions

This contract does not cover:

- restricted or nonlinear latent families;
- multiple runs sharing one latent X;
- nonseparable or vector-valued targets;
- uncertain A or P;
- noise, conditioning, or estimator variance;
- empirical HRF fitting; or
- biological or causal interpretation.

Each changes the mathematical question and requires a separate contract.
