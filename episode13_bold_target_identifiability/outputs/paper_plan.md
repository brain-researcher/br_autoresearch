# Paper plan — Exact identifiability of a latent BOLD target

[Scope and novelty review, 2026-10-02](scope_novelty_review_20261002.md).

## Manuscript type

A reusable proof-carrying technical reference and lemma package, not an
empirical benchmark or an adaptive-search report. A standalone research paper
would require a substantive contribution beyond the present result.

## One-sentence result

For Y = A X P^T and tau = s^T X c with X free over the full finite-dimensional
latent space, tau is determined by Y exactly when c = 0, s = 0, or both target
directions survive their respective observation row spaces.

## Honest contribution

The underlying linear-functional estimability theorem is classical; see
[Searle (1965), Additional Results Concerning Estimable Functions and
Generalized Inverse Matrices](https://academic.oup.com/jrsssb/article/27/3/486/7026850).
The
episode-specific value is to make its consequence for a separable BOLD
observation model explicit and constructive:

- a zero-aware necessary-and-sufficient criterion;
- a direct readout tau = v^T Y u when identifiable;
- an invisible rank-one perturbation when nonidentifiable;
- an orientation-safe vectorization convention; and
- a proof-carrying exact implementation contract.

This is useful as a technical reference, but the present result does not
establish sufficient standalone research-paper novelty. Finite matrix
exploration or exact implementation certificates do not make the classical
criterion a new theorem.

## Reader-facing question

Can one exact latent contrast be recovered even when the latent matrix itself
cannot be recovered?

The answer is yes. Full injectivity of the observation operator is unnecessary.
Only the prespecified target functional must lie in its row space.

## Main theorem

Freeze column-major x = vec(X), y = vec(Y),
M = P kron A, and ell = c kron s. Then the following are equivalent:

1. tau is constant on every fiber of X -> Y;
2. ell is in row(M);
3. ker(M) is contained in ker(ell^T);
4. adding ell^T does not increase rank;
5. M^T w = ell has an exact solution.

For the separable target, this becomes

c = 0 or s = 0 or
[c in row(P) and s in row(A)].

## Proof structure

### Proposition 1 — General linear criterion

Two latent vectors have the same observation exactly when their difference is
in ker(M). The scalar target is unique on an observation fiber exactly when
ell annihilates that kernel. Row-space/null-space duality gives the remaining
equivalences.

### Proposition 2 — Separable corollary

row(P kron A) = row(P) tensor row(A). A nonzero pure tensor c tensor s belongs
to that product subspace exactly when both factors belong to the corresponding
row spaces. Zero factors require their own branch because tau is then
identically zero.

### Proposition 3 — Positive certificate

If P^T u = c and A^T v = s, then tau = v^T Y u. This is the target readout and
can exist even when M has a large null space.

### Proposition 4 — Negative certificate

When either nonzero target factor misses its row space, construct
Delta X = z d^T so that A Delta X P^T = 0 but
s^T Delta X c = 1. The two latent matrices X and X + Delta X then have the same
observation and different targets.

## Main figure

![EP13 scalar-target identifiability](ep13_conceptual_question-v2.png)

The three panels separate the compressed observation, one prespecified scalar
and the question of equal targets for observationally indistinguishable latent
matrices. For known operators, exact observations and unrestricted X,
identifiability holds iff `c = 0`, `s = 0`, or both `s ∈ row(A)`
and `c ∈ row(P)`. Matrix/vector glyphs are symbolic, not numerical examples.
This illustrates classical estimability, not a new theorem or empirical
result. Certificates and stability limits remain in the text.
[Exact imagegen prompt set](ep13_conceptual_question-v2-prompt.md).

## Exact examples table

The paper should give a compact table rather than a large benchmark:

| Example | Purpose | Required artifact |
| --- | --- | --- |
| rank-deficient but target-identifiable | separate scalar recovery from full recovery | exact readout |
| temporal loss | show failure through ker(A) | normalized Delta X |
| spatial loss | show failure through ker(P) | normalized Delta X |
| c = 0 or s = 0 | expose the zero-branch exception | zero readout |
| swapped Kronecker order | catch vectorization error | direct y and tau replay |
| non-symmetric idempotent R | catch R versus R^T error | exact temporal witness |

## Section outline

1. **Problem.** A scalar scientific target may be recoverable without the
   latent matrix.
2. **Exact model and convention.** Define X, Y, tau, dimensions, latent domain,
   and column-major vectorization.
3. **General theorem.** Prove the row/kernel/rank/readout equivalence.
4. **Separable BOLD corollary.** Prove the component row-space criterion and
   zero branches.
5. **Proof-carrying certificates.** Give positive readouts and negative
   rank-one perturbations.
6. **Orientation and edge cases.** Show the minimal exact suite and why labels
   alone cannot detect all coordinate mistakes.
7. **Limits.** Separate exact algebra from stability, noise, uncertain
   operators, and biological interpretation.

## Reviewer-facing safeguards

- Define admissible X as the full rational vector space; do not leave it
  implicit.
- Do not claim that finite enumeration proves a universal statement.
- Do not require both row conditions when c or s is zero.
- Do not infer target identifiability from one realized tau value of zero.
- Do not use R where R^T is required.
- Do not equate exact identifiability with a useful noisy estimator.
- Do not claim novelty for the general row-space theorem.

## Potential later extensions

The following could support a larger paper, but they are outside EP13:

- shared-X multi-run operators whose stacked rows identify a target jointly;
- restricted latent classes;
- nonseparable or vector-valued targets;
- uncertain A or P; and
- quantitative stability and variance under noise.

Adding any of them changes the model family and requires a new scientific
decision rather than silent expansion of this contract.

## Current status

The theorem statement and proof architecture have been independently reviewed.
The exact regression suite, independent checker, and locked replay have not
been implemented or run. No empirical or protected outcome has been accessed.
