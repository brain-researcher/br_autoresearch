# When Does a BOLD Observation Determine a Latent Target?

[Scope and novelty review, 2026-10-02](outputs/scope_novelty_review_20261002.md).
EP13 is a reusable technical reference and lemma package based on classical
linear estimability; the present result does not establish sufficient
standalone research-paper novelty.

## The question in plain language

A BOLD analysis can lose most of a latent neural matrix and still determine one
prespecified scalar contrast exactly. EP13 asks when that happens.

For one fixed group or run, let

\[
X=B^\top,\qquad Y=A X P^\top,\qquad \tau=s^\top Xc.
\]

X is latent. Y is observed. A is the known temporal observation operator, P is
the known spatial observation operator, and tau is the one target fixed before
seeing Y.

The scientific question is not whether X can be reconstructed. It is:

> Do all latent matrices that produce the same observed Y also produce the
> same value of tau?

![EP13: one scalar target and observationally indistinguishable latent matrices](outputs/ep13_conceptual_question-v2.png)

Classical linear-estimability illustration for known operators, exact
observations and unrestricted X. The target is identifiable iff
`c = 0`, `s = 0`, or both `s ∈ row(A)` and `c ∈ row(P)`.
Panel C asks whether every pair producing the same Y has the same scalar;
colored rectangles are symbolic matrices/vectors, not numerical data.
[Exact imagegen prompt set](outputs/ep13_conceptual_question-v2-prompt.md).

## Exact setup

All dimensions are positive integers. The known objects are

| Object | Shape | Role |
| --- | --- | --- |
| A | T x V | temporal observation operator |
| P | Q x K | spatial observation operator |
| s | length V | temporal target direction |
| c | length K | spatial target direction |
| X = B^T | V x K | latent matrix, free over all rational matrices |
| Y | T x Q | exact observation A X P^T |

The rational field is used so certificates can be replayed exactly. The
mathematical equivalence holds over any field for which the stated finite-
dimensional linear algebra is valid.

If an application constructs A = R S H, that decomposition is part of the
application model, not an assumption needed by the theorem. A square
idempotent R may be non-symmetric. Whenever a temporal readout is expanded
through the decomposition, the correct equation uses
A^T = H^T S^T R^T.

The theorem below requires X to range freely over the full V x K vector space.
A restricted latent family would be a different problem: for an affine family
X0 + D, the target need only annihilate invisible directions inside D.

## Coordinate convention

Use column-major vectorization and freeze

\[
x=\operatorname{vec}(X),\qquad
y=\operatorname{vec}(Y),\qquad
M=P\otimes A,\qquad
\ell=c\otimes s.
\]

Then

\[
y=Mx,\qquad \tau=\ell^\top x.
\]

The order matters. With vec(B) or vec(Y^T), a commutation matrix or a different
Kronecker order is required. An implementation audit must replay the two
identities above, not merely compare yes/no identifiability labels.

## Main theorem

For known compatible A, P, c, and s, with X ranging over all rational V x K
matrices, the following statements are equivalent:

1. tau is uniquely determined by Y for every latent X.
2. ell belongs to the row space of M.
3. ker(M) is contained in ker(ell^T).
4. Appending ell^T does not increase rank:
   rank([M; ell^T]) = rank(M).
5. There is an exact readout vector w such that M^T w = ell; then
   tau = w^T vec(Y).

Because both the observation and target are separable, these equivalent
conditions reduce further to

\[
\boxed{
c=0\quad\text{or}\quad s=0\quad\text{or}\quad
\bigl(c\in\operatorname{row}(P)\ \text{and}\
s\in\operatorname{row}(A)\bigr).
}
\]

This disjunction is the complete result. The zero branches are not cosmetic:
if c = 0 or s = 0, then tau is identically zero and therefore identifiable
even when the other row-space condition fails.

## Why the theorem is true

Two latent vectors x1 and x2 give the same observation exactly when their
difference lies in ker(M). Their targets agree exactly when ell^T kills that
difference. This proves the kernel criterion. Finite-dimensional row-space/
null-space duality gives the row-space condition, rank preservation, and
existence of a readout.

For nonzero c and s,

\[
\operatorname{row}(P\otimes A)
=\operatorname{row}(P)\otimes\operatorname{row}(A).
\]

A nonzero pure tensor c ⊗ s lies in that tensor-product subspace if and
only if each factor lies in its corresponding row space. The zero-target
branches are handled separately.

The proof is short enough to write directly. Finite case search can test an
implementation, but it is neither the proof nor a substitute for it.

## Constructive positive certificate

In the nonzero identifiable branch, choose exact vectors u and v satisfying

\[
P^\top u=c,\qquad A^\top v=s.
\]

Then

\[
\tau=v^\top Y u.
\]

Equivalently, w = u ⊗ v is a vectorized readout certificate. The readout
need not be unique; any exact solution that replays correctly is valid. If
c = 0 or s = 0, use the zero readout.

## Constructive negative certificate

If the row-space condition fails, exact linear algebra supplies a target-
changing invisible perturbation.

- If s is outside row(A), choose z in ker(A) with s^T z != 0 and a vector d
  with c^T d != 0.
- If c is outside row(P), choose d in ker(P) with c^T d != 0 and a vector z
  with s^T z != 0.

After exact normalization, the rank-one perturbation

\[
\Delta X=z d^\top
\]

satisfies

\[
A\Delta X P^\top=0,\qquad s^\top\Delta Xc=1.
\]

Thus X and X + Delta X produce the same Y but targets that differ by one. That
pair is a complete refutation of identifiability.

## Required exact examples

The proof-carrying implementation must include a small fixed suite whose role
is to detect transcription, orientation, and certificate errors:

1. a rank-deficient M with an identifiable nonzero target;
2. temporal loss only;
3. spatial loss only;
4. simultaneous temporal and spatial loss;
5. c = 0, s = 0, and both zero;
6. an observed numerical value tau = 0 that is nevertheless nonidentifiable;
7. non-square asymmetric matrices that distinguish P kron A from A kron P;
8. a non-symmetric idempotent residualizer that exposes an erroneous use of R
   instead of R^T;
9. nonunique valid readouts; and
10. incompatible shapes, malformed rationals, zero dimensions, and invalid
    application-model inputs as parser/model rejections rather than theorem
    counterexamples.

The suite must replay the observation identity, target identity, readout, and
null witness. Agreement on a Boolean label alone is insufficient because a
simultaneous coordinate permutation can preserve that label while reading the
wrong scalar.

## What is genuinely learned

The useful conclusion is a target-level statement:

- the full latent matrix may be nonidentifiable while one scalar target is
  exactly identifiable;
- temporal and spatial information loss are separately necessary only in the
  nonzero target branch;
- a positive conclusion includes an explicit formula for reading the target
  from Y; and
- a negative conclusion includes two observationally indistinguishable latent
  matrices with different targets.

This is more informative than reporting the rank of the full observation
operator or a finite battery of passing examples.

## Work program

EP13 is now a direct theorem-and-verification episode, not an adaptive theorem
search.

1. **Statement freeze:** freeze dimensions, the full latent domain, the
   column-major convention, the theorem, and application-model validity rules.
2. **Proof package:** write the general linear criterion, separable corollary,
   zero cases, constructive readout, and constructive null witness.
3. **Independent exact replay:** implement a small rational checker from the
   frozen statement without importing historical labels or evaluator code.
4. **Orientation and edge-case suite:** replay the fixed examples above and
   inspect certificates, not only labels.
5. **Lock and review:** freeze one proof package and one checker revision before
   any separately authored replay. A replay failure invalidates the package;
   it does not trigger an adaptive trial campaign.

There is no scientific reason to require 16--64 conjecture trials, patience,
or hundreds of CPU-core-hours after the exact proof is available.

## Completion and claim boundary

The local theorem package is ready for scientific review only when:

- every equivalence is proved over the declared full latent domain;
- the zero branches are explicit;
- the readout and null certificates replay exactly;
- the orientation examples verify the actual vectorization identities;
- an independently authored checker agrees with the frozen package; and
- no finite suite is described as universal proof.

A valid positive conclusion is an exact theorem for this fixed linear model. A
valid negative conclusion is an exact counterexample to a proposed statement
plus a proved corrected statement. An implementation or specification failure
is technical failure, not scientific refutation.

The result does **not** establish numerical stability, efficiency under noise,
robustness to uncertain A or P, HRF validity, biological validity, neural
mechanism, reverse inference, or recoverability of the full latent matrix.
Those are separate questions and require a new contract.

This episode remains local and non-canonical. No Society decision, reward,
confirmation, or Landscape transition is implied.
