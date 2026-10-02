# Formal Inputs and Evidence Roles — Episode 13

EP13 has no participant dataset, images, BOLD time series, or statistical
train/test split. Its inputs are exact matrices, theorem statements, proofs,
and replayable certificates.

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

No current checker, proof lock, or independent replay output exists. A future
independent checker may use this frozen statement and its own implementation,
but it must not import historical evaluator code, constants, expected labels,
or precomputed answers.

There is no protected biological outcome to open. The only future separation
is between the authored theorem package and a separately authored replay of
that package. A replay can find an implementation or orientation error; it
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
