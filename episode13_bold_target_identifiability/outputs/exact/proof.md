# Exact target identifiability and constructive certificates

This package proves the frozen EP13 statement. It is a technical reference for
classical linear estimability, not a new biological finding. The finite replay
is an implementation and orientation audit, not the proof.

## Domain and coordinates

Work over the rational field. Let all four dimensions be positive, with
`A ∈ Q^(T×V)`, `P ∈ Q^(Q×K)`, `s ∈ Q^V`, and `c ∈ Q^K` known. The unknown
`X` ranges over **all** of `Q^(V×K)`. Set `Y=AXPᵀ` and `τ=sᵀXc`.
With column-major vectorization, `x=vec(X)`, `y=vec(Y)`, `M=P⊗A`, and
`ℓ=c⊗s`. For indices starting at zero,

`Y[t,q] = Σ_v Σ_k A[t,v] X[v,k] P[q,k]`

is the `(t+Tq)` coordinate of `(P⊗A)vec(X)`, since the corresponding
matrix entry is `P[q,k]A[t,v]` in column `(v+Vk)`. Similarly,
`sᵀXc=Σ_k Σ_v c[k]s[v]X[v,k]=(c⊗s)ᵀvec(X)`.
These establish the actual scalar and observation identities, including order.

## General linear criterion

For every compatible known tuple, the following are equivalent:

1. `τ` is constant on every observation fiber `{x: Mx=y}`.
2. `ker(M) ⊆ ker(ℓᵀ)`.
3. `ℓ ∈ row(M)`.
4. `rank([M;ℓᵀ])=rank(M)`.
5. There is a rational `w` with `Mᵀw=ℓ`.

For (1)⇒(2), use the fiber of zero: zero and every `h∈ker(M)` must have
the same target, so `ℓᵀh=0`. For (2)⇒(1), two vectors on a fiber differ by
an element of the kernel, and hence their target difference is zero.

Each row of M annihilates its kernel, so `row(M)` is included in the
annihilator of `ker(M)`. Rank-nullity gives both spaces dimension
`rank(M)`: the annihilator has dimension `VK-dim ker(M)`. Thus the spaces
coincide, proving (2)⇔(3) over Q without an inner-product or positivity
assumption. Appending a row preserves rank exactly when it is in the original
row space, giving (3)⇔(4). A linear combination of rows with coefficients w
is precisely `Mᵀw`, giving (3)⇔(5). Gaussian elimination stays in Q.
Whenever (5) holds, `wᵀy=wᵀMx=ℓᵀx=τ` for every latent matrix.

## Separable criterion, with its zero branches

Write `U=row(P)` and `W=row(A)`. The rows of `P⊗A` are all products of
a row of P and a row of A. Their span is `U⊗W`, directly proving
`row(P⊗A)=U⊗W`.

If `c=0` or `s=0`, then `ℓ=0` and `τ=0` for every X. No row condition on
the other factor is necessary; take w=0.

Now suppose c and s are both nonzero. If c∈U and s∈W, bilinearity shows
`c⊗s∈U⊗W`. Conversely, if c∉U, the annihilator argument above supplies
a rational d in ker(P) with `cᵀd≠0`. Contract the spatial factor of any
element of `U⊗W` with d: the result is zero. For `c⊗s` it is
`(cᵀd)s≠0`, a contradiction. Therefore c∈U. The corresponding contraction
with `z∈ker(A)` proves s∈W. This proves the complete criterion

`c=0 OR s=0 OR (c∈row(P) AND s∈row(A))`.

## Positive certificate

In the nonzero identifiable branch solve `Pᵀu=c` and `Aᵀv=s` exactly.
Then `vᵀYu=vᵀAXPᵀu=sᵀXc` and `w=u⊗v` satisfies `Mᵀw=ℓ`.
Solutions need not be unique. Every additional vector in `ker(Mᵀ)` may be
added to w; the resulting readout gives the same value on every attainable Y.
The zero branch uses the zero w directly, even if a component equation is
inconsistent.

## Negative rank-one certificate

Nonidentifiability implies c and s are nonzero. If s∉row(A), choose a kernel
basis vector z of A for which `sᵀz≠0` (at least one exists by the annihilator
criterion), and replace it by `z/(sᵀz)`. Choose a coordinate j with `c[j]≠0`
and set `d=e_j/c[j]`. Then `ΔX=zdᵀ` has rank one,
`AΔXPᵀ=0`, and `sᵀΔXc=1`.

Otherwise s∈row(A) and c∉row(P). Choose d in ker(P) with `cᵀd≠0` and
divide it by that scalar; choose a coordinate i with `s[i]≠0` and set
`z=e_i/s[i]`. The same two equations hold. This covers simultaneous loss as
well, using the first branch. All divisions have stated nonzero denominators.
For **every** X, X and X+ΔX have identical observations and targets differing
by exactly one. A realized value τ=0 in one X cannot establish identifiability.

## Optional application factorization

An application may supply `A=RSH`, where R is T×T, S is T×F, and H is F×V.
The declared application rule requires `R²=R`, but not `Rᵀ=R`. Transposing
the product gives `Aᵀ=HᵀSᵀRᵀ`; replacing Rᵀ by R is generally wrong.
The linear theorem itself needs only A. An invalid factorization, shape or
rational token is rejected as invalid input, not a counterexample.

## Replay and interpretation

`fixtures.json` contains only fixed input tuples with neutral case IDs. The
main implementation constructs component certificates and checks the complete
row/kernel/rank/readout equivalence. The independently authored checker receives
only those inputs; it does not receive expected labels, main code or computed
certificates. The orchestrator replays the checker's generated certificates and
compares derived observations, targets and decisions. `package_lock.json` in
each attempt binds the proof, fixture and both implementation revisions before
the independent replay. The orientation suite additionally demonstrates that
the wrong Kronecker order and wrong residualizer transpose fail their stated
identities; it verifies nonunique readouts and normalized rank-one witnesses.

The theorem establishes no numerical conditioning, uncertainty robustness,
noise efficiency, HRF validity, latent-state recovery, biological mechanism,
or causal conclusion. Restricted latent families, nonlinear models and joint
multiple-run inference are outside this package. No canonical scientific
acceptance follows from a successful local replay.
