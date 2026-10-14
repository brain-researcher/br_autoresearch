# EP23 conditional theory: P1–P5

These are conditional mathematical propositions, not explanations established for EP23.
Every substantive modeling assumption is identified below.
The accompanying script ([theory_checks.py](../code/theory_checks.py)) attempts numerical refutation;
its 2026-10-13 run (job 47703623) passed all 17 checks ([results](../runs/p1_p5_theory/20261014-attempt01/theory_checks.json)).
Simulations, SMT searches, and symbolic checks on special cases can refute claims,
but cannot prove the general propositions.

## Definitions and scope

**Assumption A1 — common domain and measure.**
Use a finite common event set $\Omega=\{1,\ldots,n\}$ and a fixed sampling measure.
For the symmetric stochastic statements, the measure is uniform.
Matrices act on column vectors of functions; stochastic matrices have row sums one.
All matrix norms and eigenspaces use the stated measure and normalization.

Write $e=n^{-1/2}\mathbf1$ and $H=I-ee^\top$.
Removing the constant mode means restricting to $e^\perp$, or applying $H$ on both sides.
This removes the constant stationary vector, not necessarily every stationary vector:
a reducible chain can have additional eigenvectors with eigenvalue one.

**Assumption A2 — symmetry.**
Unless an extension is explicitly stated, $T=T^\top$ is entrywise nonnegative
and satisfies $T\mathbf1=\mathbf1$.
It is therefore doubly stochastic.
For a reversible transition matrix with positive stationary measure $\pi$,
detailed balance makes
$$
S=D_\pi^{1/2}TD_\pi^{-1/2},\qquad D_\pi=\operatorname{diag}(\pi),
$$
symmetric; the constant mode becomes $\sqrt\pi$ in these coordinates.
Results must then use this similarity transform and its corresponding inner product.
Directed, nonnormal, partially observed, and policy-dependent cases need additional arguments.

**Assumption A3 — spectral representation, when invoked.**
Although $T_S=U_S\Lambda_SU_S^\top$ follows from symmetry,
a representation
$$
\Phi_S=U_{S,I}\,g_S(\Lambda_{S,I})R_S
$$
is a modeling hypothesis unless derived from a specified objective.
Here $I$ is the retained index set and $R_S$ is an orthogonal feature-coordinate change.
Shared subspaces, shared neighbor identities, and shared distances are different claims.

For centered empirical features $X=UDV^\top$, retain $k$ positive singular values.
The power representation $\Phi_\alpha=U_kD_k^\alpha$ has Gram matrix
$U_kD_k^{2\alpha}U_k^\top$.
Thus $\alpha=1$ gives the centered rank-$k$ native geometry, and
$\alpha=2$ squares that rank-$k$ Gram matrix.
It squares the full feature Gram only when the retained rank is the full rank.
A preference for $\alpha\approx2$ does not identify a two-step mechanism.

## Standard results used, with conditions

**Eckart–Young–Mirsky.**
For $A\in\mathbb R^{m\times n}$ with singular values
$\sigma_1\ge\cdots\ge\sigma_\ell\ge0$, $\ell=\min(m,n)$, and $0\le r<\ell$,
a truncated SVD $A_r$ satisfies
$$
\min_{\operatorname{rank}(C)\le r}\|A-C\|_F
 =\left(\sum_{i>r}\sigma_i^2\right)^{1/2},
\qquad
\min_{\operatorname{rank}(C)\le r}\|A-C\|_2=\sigma_{r+1}.
$$
The same truncation minimizes every unitarily invariant norm.
The Frobenius minimizer is unique when $0<r<\operatorname{rank}(A)$
and $\sigma_r>\sigma_{r+1}$; it is also unique when $r=0$
or $r\ge\operatorname{rank}(A)$.
A positive singular-value tie across a truncated boundary permits nonuniqueness.
A gap does not generally imply uniqueness for operator-norm approximation.
References: [Eckart–Young (1936)](https://doi.org/10.1007/BF02288367)
and [Mirsky (1960)](https://doi.org/10.1093/qmath/11.1.50).

**Weyl.**
For real symmetric $A,\widehat A=A+E$, with eigenvalues in descending algebraic order,
$$
|\widehat\lambda_i-\lambda_i|\le\|E\|_2
\quad\text{for every }i.
$$
Indeed, every unit-vector Rayleigh quotient changes by at most $\|E\|_2$;
applying the variational characterization of each ordered eigenvalue proves the claim.

**Davis–Kahan: the one-sided mixed-gap form used here.**
Let $U,\widehat U$ span the leading $q$ eigenspaces of symmetric $A,\widehat A$,
where $1\le q<d$.
If
$\Delta=\lambda_q(A)-\lambda_{q+1}(\widehat A)>0$,
then, for either operator or Frobenius norm,
$$
\|\sin\Theta(\widehat U,U)\|
 \le \frac{\|EU\|}{\Delta}.
$$
The condition separates the entire reference leading interval from the perturbed
complement on one side; arbitrary pairwise eigenvalue distances are insufficient.
This specialization is proved below.
Reference: [Davis–Kahan (1970)](https://doi.org/10.1137/0707001).

**Yu–Wang–Samworth (2015), Theorem 2.**
Let $A,\widehat A=A+E$ be real symmetric $d\times d$ matrices.
Let $U,\widehat U$ contain orthonormal eigenvectors indexed $a,\ldots,b$
in descending algebraic order, and let $q=b-a+1$.
Set $\lambda_0=+\infty$, $\lambda_{d+1}=-\infty$, and assume
$$
\delta=\min\{\lambda_{a-1}-\lambda_a,\lambda_b-\lambda_{b+1}\}>0.
$$
Then
$$
\|\sin\Theta(\widehat U,U)\|_F
 \le \frac{2\min\{\sqrt q\,\|E\|_2,\|E\|_F\}}{\delta},
$$
and some orthogonal $Q\in\mathbb R^{q\times q}$ satisfies
$$
\|\widehat UQ-U\|_F
 \le \frac{2^{3/2}\min\{\sqrt q\,\|E\|_2,\|E\|_F\}}{\delta}.
$$
No small-perturbation assumption is required.
Reference: [published author PDF, Theorem 2](https://personal.lse.ac.uk/wangt60/publication/DKvariant.pdf).
Our proof below independently establishes the needed leading-subspace specialization.

## P1 — Operator relations share an eigenbasis under stated conditions

**Statement and assumptions.**
Under A1–A2, write $T=U\Lambda U^\top$.
For $0\le\gamma<1$,
$$
TT^\top=T^2=U\Lambda^2U^\top,\qquad
(I-\gamma T)^{-1}
 =\sum_{j=0}^\infty\gamma^jT^j
 =U\operatorname{diag}\!\left((1-\gamma\lambda_i)^{-1}\right)U^\top.
$$
Thus these operators admit a common orthonormal eigenbasis.
For an arbitrary symmetric, nonstochastic $T$, the series assertion instead
requires $\gamma\|T\|_2<1$ as a sufficient condition.

**Proof.**
Jensen's inequality and double stochasticity give
$\sum_i(Tf)_i^2\le\sum_{i,j}T_{ij}f_j^2=\sum_jf_j^2$.
Hence $\|T\|_2\le1$ and $|\lambda_i|\le1$.
The spectral theorem gives the displayed product formula.
The geometric series converges in operator norm because $\|\gamma T\|_2<1$.
Multiplying its partial sums by $I-\gamma T$ gives
$I-(\gamma T)^{L+1}$, which tends to $I$.
Diagonalizing the convergent series gives the stated inverse.

The vector $e$ is a common eigenvector.
Symmetry makes $e^\perp$ invariant, so the same proof applies after restriction.
For real spectral functions $f$, $f(T)=U\operatorname{diag}(f(\lambda_i))U^\top$.
A common leading subspace additionally requires the same selected indices:
strictly increasing functions on the relevant spectral support preserve its order.
Squaring is increasing on nonnegative support, but not on all of $[-1,1]$.

**Explicit reordering counterexample.**
The symmetric stochastic matrix
$$
T=\frac1{15}\begin{pmatrix}1&10&4\\10&1&4\\4&4&7\end{pmatrix}
$$
has eigenvalues $1,1/5,-3/5$, with respective directions
$(1,1,1)$, $(1,1,-2)$, and $(1,-1,0)$.
After removing the constant mode, the leading algebraic eigenvalue is $1/5$.
Squaring gives $1/25$ and $9/25$, reversing these two modes.

**Explicit nonsymmetric counterexample.**
For
$$
T=\begin{pmatrix}1&0\\1/2&1/2\end{pmatrix},
\qquad TT^\top=\begin{pmatrix}1&1/2\\1/2&1/2\end{pmatrix},
$$
$(1,1)^\top$ is a right eigenvector of $T$, but not of $TT^\top$.
The shared-context kernel uses left singular vectors in general.
Visual similarity is a separate candidate operator, with no automatic common basis.

**Observation-map statement and assumptions.**
Let $(Z_0,Z_1)$ have stationary marginal $\mu$, with transition operator $T$,
and let $X_t=h(Z_t)$ have pushforward measure $\mu_X$.
The pullback $J:L^2(\mu_X)\to L^2(\mu)$ is $Jf=f\circ h$.
Then the observed one-step conditional-expectation operator is
$$
T_X=J^*TJ,\qquad \|T_X\|\le\|T\|\le1.
$$
If $h$ is bijective on the positive-measure supports, $J$ is unitary.
Otherwise $J$ remains an isometry, and $T_X$ is a compression and a contraction;
it need not be a strict contraction.

**Proof.**
Pushforward integration gives $\|Jf\|_{L^2(\mu)}=\|f\|_{L^2(\mu_X)}$.
The adjoint satisfies $J^*g(x)=\mathbb E[g(Z_0)\mid X_0=x]$.
Iterated conditioning therefore gives
$J^*TJf(x)=\mathbb E[f(X_1)\mid X_0=x]$.
Stationary Jensen gives $\|T\|\le1$, establishing the norm bound.
Bijectivity makes the isometry onto, hence unitary.
This is unitary equivalence of operators; it does not make $T_X$ itself unitary.

For $T=\left(\begin{smallmatrix}4/5&1/5\\1/5&4/5\end{smallmatrix}\right)$
and $h(1)=h(2)$, normalized coordinates give $J=(1,1)^\top/\sqrt2$.
Thus $J^\top TJ=[1]$: the eigenvalue $3/5$ disappears.
Bijectivity assumptions, including PRH's, matter at precisely this step.
The observed process need not itself be Markov.
For self-adjoint $T$,
$J^*T^2J-(J^*TJ)^2=J^*T(I-JJ^*)TJ$;
equality requires invariance of the observable subspace.

**Other limits.**
Diagonal removal and normalization are separate operations.
Removing the diagonal from the $3\times3$ example gives row sums
$14/15,14/15,8/15$, so even its constant eigenvector is lost.
An entrywise logarithm is not spectral functional calculus:
$$
A=\begin{pmatrix}2&1\\1&1\end{pmatrix},\qquad
\log_{\rm entrywise}A=\begin{pmatrix}\log2&0\\0&0\end{pmatrix}.
$$
Their commutator is
$\left(\begin{smallmatrix}0&-\log2\\\log2&0\end{smallmatrix}\right)\ne0$.
Consequently, PMI or log-density-ratio results require their own operator bridge.

## P2 — An exact learning bridge for one surrogate objective

**Statement and assumptions.**
Let $K=K^\top\in\mathbb R^{n\times n}$, with
$\lambda_1\ge\cdots\ge\lambda_n$, and let $0\le r\le n$.
Consider only the unweighted objective
$$
\min_{\Phi\in\mathbb R^{n\times r}}\|K-\Phi\Phi^\top\|_F^2.
$$
An optimal Gram matrix is
$$
G_*=\sum_{i=1}^r\max(\lambda_i,0)u_iu_i^\top.
$$
If $s=\#\{i:\lambda_i>0\}$ and $t=\min(r,s)$, its loss is
$\sum_i\lambda_i^2-\sum_{i=1}^t\lambda_i^2$.
All optima retain the largest positive eigenvalues, allowing choices inside
a positive eigenspace tied across the cutoff.

**Proof without assuming the optimizer's eigenvectors.**
A matrix is $\Phi\Phi^\top$ for such a $\Phi$ exactly when it is PSD with rank at most $r$.
The forward implication follows from quadratic forms and rank.
The converse follows by taking square roots of its nonnegative eigenvalues
and padding its factor with zero columns.

Write any feasible Gram as $G=\sum_i x_iw_iw_i^\top$, where
$x_1\ge\cdots\ge x_n\ge0$ and $x_i=0$ for $i>r$.
For any rank-$j$ orthogonal projector $Q$,
$$
\operatorname{tr}(KQ)=\sum_i\lambda_i a_i
 \le\sum_{i=1}^j\lambda_i,\qquad
a_i=u_i^\top Qu_i,\quad 0\le a_i\le1,\quad\sum_i a_i=j.
$$
The inequality follows by moving any available mass from a smaller
$\lambda_i$ to a larger one.
With $Q_j=\sum_{i\le j}w_iw_i^\top$ and $x_{n+1}=0$,
$$
\operatorname{tr}(KG)
 =\sum_j(x_j-x_{j+1})\operatorname{tr}(KQ_j)
 \le\sum_i\lambda_i x_i.
$$
Therefore
$$
\|K-G\|_F^2\ge\sum_i(\lambda_i-x_i)^2.
$$
Minimizing the right side gives $x_i=\max(\lambda_i,0)$ for $i\le r$
and zero otherwise; this sequence is already nonincreasing.
Choosing $w_i=u_i$ attains every inequality and proves the result.
Equality in the mass-transfer inequalities gives the stated freedom only
within tied eigenvalue blocks.

**Uniqueness.**
Uniqueness concerns the retained subspace and its optimal Gram, not feature coordinates.
If $0<r<s$, the retained subspace is unique exactly when
$\lambda_r>\lambda_{r+1}$.
If $r\ge s>0$, all positive modes are retained and $G_*=K_+$ is unique:
the positive-to-nonpositive boundary is separated, or the whole space is retained.
If $r=0$ or $s=0$, the unique Gram is zero.
With positive retained eigenvalues $\theta_1,\ldots,\theta_t$, all its factors have form
$W\operatorname{diag}(\sqrt{\theta_i})O$, where $OO^\top=I_t$ and
$O\in\mathbb R^{t\times r}$.
Thus no particular nonzero factor or basis is identified.

**Explicit limits and counterexamples.**
For $K=\operatorname{diag}(2,-3)$ and $r=1$, the PSD optimum is
$\operatorname{diag}(2,0)$, with loss $9$.
The unrestricted rank-one approximation $\operatorname{diag}(0,-3)$
has loss $4$ but is infeasible.
For $K=\operatorname{diag}(1,1,-1)$ and $r=1$,
$\operatorname{diag}(1,0,0)$ and $\operatorname{diag}(0,1,0)$
both have loss $2$, demonstrating boundary nonuniqueness.

Weighting the squared $(1,1)$ residual by $1/100$ changes even this surrogate:
for $K=\operatorname{diag}(2,1)$ and $r=1$, keeping the first mode costs $1$,
whereas $\operatorname{diag}(0,1)$ costs $4/100$.
A unit-diagonal constraint also excludes the unweighted optimum
$\operatorname{diag}(2,0)$.

This proposition does not cover weighted SGNS logistic loss, general InfoNCE,
unit-diagonal constraints, or nonnegative-feature constraints.
For $K=f(T)$ it derives square-root weighting of retained positive **kernel**
eigenvalues; it does not assume that kernel order equals transition-operator order.
It does not establish that CLIP, PEcore, SigLIP2, DINOv2, or brains optimize this loss.
Each published factorization result needs its own bridge to $T$.

## P3 — Coarse stability can coexist with fine variability

**Statement and assumptions.**
Assume $T_0$ is symmetric stochastic and
$T_S=T_0+E_S$ remains entrywise nonnegative, where
$E_S=E_S^\top$, $E_S\mathbf1=0$, and $\|E_S\|_2\le\varepsilon$.
The constant vector is then an exact common eigenvector.
Let $L$ have orthonormal columns spanning $e^\perp$ and define
$A=L^\top T_0L$, $F_S=L^\top E_SL$.
Symmetry and $E_Se=0$ imply $\|F_S\|_2=\|E_S\|_2$
and $\|F_S\|_F=\|E_S\|_F$.

Let $1\le q<n-1$, and assume the reference algebraic gap
$\delta=\lambda_q(A)-\lambda_{q+1}(A)>0$.
Write $P_0,\widehat P_S$ for the leading rank-$q$ projectors in this complement.
Then
$$
\|\sin\Theta(\widehat P_S,P_0)\|_F
 \le\frac{2\min\{\sqrt q\,\varepsilon,\|E_S\|_F\}}{\delta}.
$$
If additionally $\varepsilon<\delta/2$, the perturbed leading cluster is separated and
$$
\|\widehat P_S-P_0\|_2\le\frac{\varepsilon}{\delta-\varepsilon}.
$$
Two learners satisfying the same assumptions obey
$$
\|\widehat P_1-\widehat P_2\|_2
 \le\frac{2\varepsilon}{\delta-\varepsilon}.
$$

**Proof of the Frobenius bound.**
For equal-rank projectors, let the principal angles be $\theta_i$ and put
$s^2=\sum_i\sin^2\theta_i=q-\operatorname{tr}(P_0\widehat P)$.
The two-dimensional principal-angle decomposition gives
$$
\|\widehat P-P_0\|_2=\max_i\sin\theta_i,\quad
\|\widehat P-P_0\|_F=\sqrt2\,s,\quad
\|\widehat P-P_0\|_*=2\sum_i\sin\theta_i.
$$
These identities follow by diagonalizing the difference of the two
rank-one projectors in each principal plane; its eigenvalues are
$\pm\sin\theta_i$.

In an eigenbasis of $A$, the reference gap gives
$\operatorname{tr}(A(P_0-\widehat P))\ge\delta s^2$:
mass $s^2$ removed from the top block has eigenvalues at least $\lambda_q$,
while the same mass added below has eigenvalues at most $\lambda_{q+1}$.
Since $\widehat P$ maximizes $\operatorname{tr}((A+F)P)$ over rank-$q$ projectors,
$$
\delta s^2\le\operatorname{tr}(F(\widehat P-P_0)).
$$
Operator/nuclear norm duality bounds the right side by
$2\sqrt q\,\|F\|_2s$; Frobenius Cauchy–Schwarz bounds it by
$\sqrt2\,\|F\|_Fs$.
Dividing when $s>0$, and treating $s=0$ directly, proves the stated YWS bound.
Orthogonal Procrustes alignment gives squared error
$2\sum_i(1-\cos\theta_i)\le2s^2$, proving its stated alignment consequence.

**Proof of the mixed-gap and two-learner bounds.**
Weyl gives
$\widehat\lambda_q-\widehat\lambda_{q+1}\ge\delta-2\varepsilon>0$
and
$\Delta=\lambda_q-\widehat\lambda_{q+1}\ge\delta-\varepsilon>0$.
For completeness, let $U$ span the reference leading space and let
$\widehat U_\perp$ span the perturbed complement.
Set $S=\widehat U_\perp^\top U$, $R=\widehat U_\perp^\top FU$,
$\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_q)$,
and $\widehat\Lambda_\perp$ equal the complementary perturbed diagonal.
The eigenvector equations imply
$S\Lambda-\widehat\Lambda_\perp S=-R$.
With $c=\lambda_q$, the unique solution is
$$
S=-\int_0^\infty
 e^{t(\widehat\Lambda_\perp-cI)}R\,e^{-t(\Lambda-cI)}\,dt.
$$
Differentiating the integrand verifies the equation.
The two exponential norms are bounded by $e^{-\Delta t}$ and $1$.
Integration gives $\|S\|\le\|R\|/\Delta$ for operator or Frobenius norm.
Since $\|S\|_2=\|\widehat P-P_0\|_2$, the mixed-gap bound follows.
The triangle inequality through $P_0$ proves the two-learner bound.

The stronger assumption $\varepsilon<\delta/4$ is sufficient but conservative:
$$
\|\widehat P-P_0\|_2\le\frac{4\varepsilon}{3\delta}<\frac13.
$$
Errors and gaps must refer to the same operator.
For $K=h(T)$, its relevant ordering and spectral separation must be checked anew.

**Stochastic block construction.**
Take $B\ge2$ blocks of size $m\ge2$ and $n=Bm$.
Let within-block weight be $p$ and between-block weight be $q_b$,
where $0<q_b<p\le1$; $q_b$ denotes the usual block-model parameter $q$.
Include self-loops and normalize by $z=m[p+(B-1)q_b]$.
For the block-indicator matrix $C\in\mathbb R^{n\times B}$,
$$
T_0=\frac{q_b\mathbf1\mathbf1^\top+(p-q_b)CC^\top}{z}.
$$
Its row sums are one.
The block-constant space $\mathcal C=\operatorname{col}(C)$ contains $e$.
On $\mathcal C\cap e^\perp$, $T_0$ acts by
$\beta=(p-q_b)/(p+(B-1)q_b)>0$.
On the fine space $\mathcal F=\mathcal C^\perp$, it acts by zero.
The fine space is the direct sum of the within-block zero-sum contrast spaces.

Let $W$ be an orthonormal basis for $\mathcal F$ and take
$E=WHW^\top$, $H=H^\top$, $\|H\|_2\le\rho<\min(q_b/z,\beta)$.
Then $EC=0$, so every coarse vector and eigenvalue is preserved exactly.
Also $E\mathbf1=0$, and $|E_{ij}|\le\rho<q_b/z$
ensures entrywise positivity of $T_0+E$.
All fine eigenvalues lie in $[-\rho,\rho]$, below the coarse value $\beta$.
Thus the leading $B-1$ nonconstant coarse projector is exactly unchanged.

Strictly block-local perturbations are obtained by taking $H$ block diagonal
in bases for the individual within-block contrast spaces.
General $H$ can couple those contrasts, but still cannot affect any block mean.
For $B=m=2$, $p=3/4$, $q_b=1/4$, an explicit block-local example is
$$
T_0=\frac18\begin{pmatrix}3&3&1&1\\3&3&1&1\\1&1&3&3\\1&1&3&3\end{pmatrix},
\quad
E=\frac1{40}\begin{pmatrix}1&-1&0&0\\-1&1&0&0\\0&0&-1&1\\0&0&1&-1\end{pmatrix}.
$$
The nonconstant coarse eigenvalue remains $1/2$; fine eigenvalues become $\pm1/20$.

**Additional distributional assumption — independent Haar fine orientations.**
Let $d=\dim\mathcal F$, fix fine eigenvalues $\theta_i$ with
$|\theta_i|\le\rho$ and $\theta_k>\theta_{k+1}$, and set
$H_S=O_S\operatorname{diag}(\theta_i)O_S^\top$.
Assume the $O_S$ are independent Haar orthogonal matrices.
The selected fine rank-$k$ projectors are then independent Haar subspaces.

If $Q=OQ_*O^\top$, rotational invariance implies
$R\,\mathbb E[Q]R^\top=\mathbb E[Q]$ for every orthogonal $R$.
Coordinate sign flips force off-diagonal entries to vanish;
coordinate permutations make all diagonal entries equal.
Because $\operatorname{tr}(Q)=k$, $\mathbb E[Q]=(k/d)I$.
For an independent rank-$k$ projector $P$,
$$
\mathbb E[\operatorname{tr}(PQ)\mid P]=k^2/d,\qquad
\mathbb E[\operatorname{tr}(PQ)/k]=k/d.
$$
This proves chance-level expected normalized overlap while coarse invariance is exact.
Under strictly block-local Haar perturbations, the same argument applies per block.
Selecting $k_b$ modes in fine block dimension $d_b$ gives aggregate expectation
$\bigl(\sum_b k_b^2/d_b\bigr)/\bigl(\sum_b k_b\bigr)$;
it equals total-rank/total-dimension when the selected fractions are equal.

Small fine gaps alone do not imply disagreement.
For the explicit $4\times4$ example, choosing $E_1=E_2=E$
gives identical fine projectors despite its small fine separation.
Internal ties require comparing whole eigenspaces.
For $\operatorname{diag}(2,1,1)$, rank-two selections
$\operatorname{diag}(1,1,0)$ and $\operatorname{diag}(1,0,1)$ differ by norm one
even though the matrix is unchanged; the boundary gap is zero.
Human taxonomic labels are proposed analysis scales, not established experience eigenmodes.

**kNN margin lemma and proof.**
Let $K,\widehat K$ be PSD Gram matrices on the same stimuli and
$\|\widehat K-K\|_2\le\eta$.
Define $d_{ij}^2=(e_i-e_j)^\top K(e_i-e_j)$.
Then
$$
|\widehat d_{ij}^2-d_{ij}^2|
 \le\|\widehat K-K\|_2\|e_i-e_j\|^2\le2\eta.
$$
For a fixed anchor, exclude the anchor itself.
If its ordered squared distances satisfy $d_{(k+1)}^2-d_{(k)}^2>4\eta$,
every originally selected distance remains smaller than every excluded distance.
Indeed, each selected distance increases by at most $2\eta$,
and each excluded distance decreases by at most $2\eta$.
Thus its kNN set is unchanged.
The quadratic-form inequality itself only requires symmetry.

Subspace stability alone does not establish this margin.
For $K=\operatorname{diag}(0,1,4)$ and
$\widehat K=\operatorname{diag}(0,9,4)$, the eigenbasis and rank-two range agree,
but stimulus 1's nearest neighbor changes from stimulus 2 to stimulus 3.

## P4 — Readout invariance has a limited domain

**Statement and assumptions.**
Let training features $X\in\mathbb R^{n\times d}$ have full column rank,
let $Y$ be the training responses, and let $R\in\mathbb R^{d\times d}$ be invertible.
Use $XR$ in training and $X_{\rm eval}R$ in evaluation.
Unregularized least-squares fitted and evaluation values are invariant.

**Proof.**
The map $C\mapsto B=RC$ is a bijection, and $XRC=XB$.
Thus minimizing $\|Y-XRC\|_F^2$ is exactly the original minimization.
Full column rank makes $\widehat B$ unique, giving
$\widehat C=R^{-1}\widehat B$ and
$X_{\rm eval}R\widehat C=X_{\rm eval}\widehat B$.
Training fitted values are the orthogonal projection onto $\operatorname{col}(X)$.
This training projection invariance also holds without full column rank,
but arbitrary coefficient-selection conventions need not preserve evaluation values.

For penalty $\lambda\operatorname{tr}(B^\top LB)$, $L\succeq0$,
the corresponding transformed penalty is
$\lambda\operatorname{tr}(C^\top R^\top LR\,C)$.
Substitution gives exactly the same objective, proving invariance.
In particular, an original isotropic ridge penalty transforms to $R^\top R$.

**Explicit isotropic-ridge counterexample.**
Take $X=I_2$, $Y=(1,1)^\top$, $\lambda=1$, and $R=\operatorname{diag}(2,1)$.
Original isotropic ridge predicts $(1/2,1/2)^\top$.
Isotropic ridge fitted anew to $XR$ predicts $(4/5,1/2)^\top$.
Using the transformed penalty $\operatorname{diag}(4,1)$ restores the original fit.

Rank truncation also fails to commute with arbitrary rescaling:
$X=\operatorname{diag}(2,1)$ has first-coordinate leading rank-one range,
whereas $XR=\operatorname{diag}(2,3)$ for $R=\operatorname{diag}(1,3)$
has second-coordinate leading rank-one range.
A zero weight, such as $R=\operatorname{diag}(1,0)$, is not invertible.
For rank-deficient training $X=(1,1)$, $Y=1$, and $R=\operatorname{diag}(2,1)$,
minimum-norm coefficients are $(1/2,1/2)^\top$ and $(2/5,1/5)^\top$.
At $X_{\rm eval}=(1,0)$, predictions are $1/2$ and $4/5$.

Finite sample size alone does not invalidate the exact full-rank OLS identity.
Regularization, estimated preprocessing, feature selection, and truncation
can remove its conditions; fixed-geometry RSA directly depends on spectral weights.
“Encoding is insensitive to the spectrum” is therefore not unconditional.

**Latent nonidentifiability and induced metric.**
With unrestricted readout coefficients $B$,
$$
ZB=(ZR)(R^{-1}B),\qquad
(ZR)(ZR)^\top=ZRR^\top Z^\top.
$$
For $Z=B=I_2$ and $R=\operatorname{diag}(2,1/2)$, observations stay $I_2$,
but latent Grams are $I_2$ and $\operatorname{diag}(4,1/4)$.
Their two-row squared distances are $2$ and $17/4$.
Thus observations alone cannot identify native latent geometry.

A linear readout induces the PSD metric $M=BB^\top$:
$v^\top Mv=\|B^\top v\|^2\ge0$, and
$\|z_iB-z_jB\|^2=(z_i-z_j)M(z_i-z_j)^\top$.
Conversely, every PSD metric has a square-root factorization;
a fixed output dimension bounds its rank.
A full readout matching brain geometry, as contemplated by EP22 R3,
establishes predictive sufficiency under its evaluation, not native geometry.

## P5 — Pooling yields pure weighting only under a diagonal condition

**Statement and assumptions.**
Let the noise-free responses have compact SVD
$N=UAV^\top$, where $U^\top U=V^\top V=I_r$ and
$A=\operatorname{diag}(a_i)$ with $a_i>0$.
Let measurement be a fixed linear map $M$, and let $B=NM$.
Then
$$
BB^\top=UA(V^\top MM^\top V)AU^\top.
$$
Writing $C=V^\top MM^\top V\succeq0$, a sufficient condition for weighting
without mixing the specified columns of $U$ is that $C$ be diagonal.

**Proof and commutation condition.**
Multiplication gives the identity directly.
If $C=\operatorname{diag}(d_i)$, then
$BB^\top=U\operatorname{diag}(a_i^2d_i)U^\top$.
Conversely, since all $a_i>0$, diagonality in this fixed basis requires
$a_ia_jC_{ij}=0$ for $i\ne j$, hence $C_{ij}=0$.

If $[C,A^2]=0$, then $(a_j^2-a_i^2)C_{ij}=0$.
For distinct $a_i^2$, this forces $C$ to be diagonal.
With repeated neural eigenvalues, it only makes $C$ block diagonal:
an orthogonal rotation within each degenerate block diagonalizes $C$.
The same rotation of $U$ and $V$ leaves the SVD valid.
Thus commutation preserves distinct neural eigenspaces, but not arbitrary
chosen axes inside a degenerate eigenspace.
For example, $A=I_2$ and
$C=\left(\begin{smallmatrix}1&1/2\\1/2&1\end{smallmatrix}\right)$ commute,
yet mix the fixed coordinate axes.

**Product nonidentifiability.**
Assume additionally $d_i>0$.
The measured Gram eigenvalues identify $\mu_i=a_i^2d_i$.
For every $c_i>0$, replacing
$a_i$ by $c_ia_i$ and $d_i$ by $d_i/c_i^2$ leaves every $\mu_i$ unchanged.
These alternatives are realizable under unrestricted measurement:
take $N'=U\operatorname{diag}(c_ia_i)V^\top$ and
$M'=V\operatorname{diag}(c_i^{-1})V^\top M$; then $N'M'=NM$.
Independent restrictions on neural amplitudes or measurement could change identifiability.

An explicit example is
$$
N=\operatorname{diag}(1,2),\ M=\operatorname{diag}(2,1),
\qquad
N'=\operatorname{diag}(2,1),\ M'=\operatorname{diag}(1,2).
$$
Both measured responses are $\operatorname{diag}(2,2)$,
while the neural and pooling factors differ.

**Exact affine machine-checking scope.**
For $\theta=(\log a_1,\ldots,\log a_r,\log d_1,\ldots,\log d_r)^\top$,
$$
y_i=\log\mu_i=2\log a_i+\log d_i,\qquad
y=M_{\rm aff}\theta,\qquad M_{\rm aff}=[\,2I_r\ \ I_r\,].
$$
This rational-coefficient affine matrix has rank $r$ and nullity $r$.
Its nullspace contains, and is spanned by, $(e_i,-2e_i)$.
For $r=2$,
$M_{\rm aff}=\left(\begin{smallmatrix}2&0&1&0\\0&2&0&1\end{smallmatrix}\right)$.
P5's factor nonidentifiability in these log coordinates is the one statement
here inside BR's finite rational affine-identifiability scope.
No BR machine check is claimed to have been executed.
Zero factors lie outside these log coordinates and cannot restore unrestricted identifiability.

**Explicit Gaussian-blur counterexample.**
On the one-dimensional grid $\{0,1\}$, choose Gaussian width
$\sigma^2=1/(2\log2)$.
The Gaussian affinities are $1$ on the diagonal and $1/2$ off it.
Row normalization gives the symmetric blur
$$
M=\frac13\begin{pmatrix}2&1\\1&2\end{pmatrix}.
$$
It preserves the constant direction and attenuates the contrast by $1/3$.
Take $U=V=I_2$ and $A=\operatorname{diag}(2,1)$.
The neural modes are not blur eigenvectors, and
$$
C=MM^\top=\frac19\begin{pmatrix}5&4\\4&5\end{pmatrix},
\qquad
BB^\top=\frac19\begin{pmatrix}20&8\\8&5\end{pmatrix}.
$$
The off-diagonal entries prove mode mixing despite genuine Gaussian smoothing.
Even if $V$ is the blur eigenbasis, neural amplitudes $(2,1)$ give measured
Gram eigenvalues $(4,1/9)$, not values proportional to $(16,1)$.
Thus low-pass filtering establishes neither the diagonal condition nor $\alpha=2$.

**Cross-repeat Gram statement and proof.**
Assume aligned repeats satisfy $B^{(t)}=S+E^{(t)}$ for the same signal $S=NM$.
Assume finite second moments, conditional zero means
$\mathbb E[E^{(t)}\mid S]=0$, and conditional independence across repeats.
Zero cross-second moment would suffice in place of independence.
Expansion and conditioning give
$$
\mathbb E[B^{(1)}B^{(2)\top}\mid S]=SS^\top,
\qquad
\mathbb E[B^{(t)}B^{(t)\top}\mid S]
 =SS^\top+\mathbb E[E^{(t)}E^{(t)\top}\mid S].
$$
The added same-repeat term is PSD because every quadratic form is
$\mathbb E[\|E^{(t)\top}v\|^2\mid S]\ge0$.
Symmetrizing the cross-repeat Gram preserves unbiasedness, not sample PSD.
For $S=(0,0)^\top$ and independent Rademacher noises
$E^{(t)}=\xi_t(1,0)^\top$, opposite signs give
$B^{(1)}B^{(2)\top}=\operatorname{diag}(-1,0)$.
This occurs with probability $1/2$ despite satisfying all noise assumptions.

Signal pooling, noise, and response estimation therefore require separate models.
Contextual precedents supplied by the episode include measurement-induced geometry
changes from voxel averaging/sampling (Kriegeskorte & Diedrichsen, 2016)
and eigenvector delocalization under finite neuron sampling
(Kang, Canatar & Chung, 2025).
They are not used as proof lemmas here; P5 is a conditional pooling restatement.

## Numerical refutation protocol

The script uses seeded NumPy draws and reports each check's instance count,
maximum left-hand-side/bound ratio, and pass/fail.
For identities and explicit counterexamples, the numerator is numerical residual
and the bound is a stated floating-point tolerance.
For Haar overlap, it compares sample means with the proved expectation.
For independent values in $[0,1]$, Hoeffding's inequality gives
$\Pr(|\bar X-\mathbb EX|>t)\le2e^{-2Lt^2}$.
The script uses this with a union bound across batches and nominal level $10^{-6}$.
This is a stochastic diagnostic under ideal independent Haar sampling,
not a finite-run guarantee, a proof of rotational invariance, or empirical brain evidence.
No theorem's assumptions about real learners are established by these checks.

## Relation to EP23 A0 results

The following are empirical, exploratory observations on exposed data, not proofs.

- A measured Visual Genome co-occurrence operator's leading 8-dimensional subspace
  overlapped participants' brain subspaces above chance, in 5/5 participants.
  P1–P2 specify possible operator/objective bridges; P3 concerns conditional subspace stability.
- That overlap did not persist beyond a 27-category taxonomy for brains, in 3/5.
  P3 distinguishes coarse and fine subspaces; its kNN margin is a separate condition
  needed for any inference about neighbor identities.
- Model subspaces aligned with the operator beyond taxonomy.
  P2–P3 bear on objective and perturbation conditions, without establishing those conditions.
- Brain alignment was confined to the leading 8 modes, while brain split-half
  self-consistency was 0.10–0.20.
  P3 bears on perturbation sensitivity; P5 bears on measurement and noise limits.

None of these propositions is claimed to explain these observations.

## What is proved, what is assumed, what remains empirical

| Proposition | Exact conditional result | Assumptions and checks required | What remains empirical or unestablished |
|---|---|---|---|
| P1 | Spectral functions share a basis; observation gives $J^*TJ$. | Fixed measure, symmetry or valid reversible transform; observation invariance and spectral ordering when needed. | Similar experience operators across learners; applicable observation maps; relation to visual similarity or entrywise PMI. |
| P2 | Unweighted Frobenius PSD rank-constrained fitting retains the largest positive kernel eigenvalues. | The stated objective and constraints; kernel-to-operator bridge; positive boundary gap for a unique retained subspace. | Whether actual model objectives or brains implement a comparable spectral bridge. |
| P3 | Gap-dependent bounds, exact block coarse invariance, Haar chance overlap, and the kNN margin guarantee. | Same operator for gap/error; stochasticity; fine-support structure; independent Haar orientations for the expectation; distance margin for neighbors. | Similar learner operators, sufficient fine variability, taxonomy/eigenmode correspondence, and real neighborhood stability. |
| P4 | OLS and transformed-penalty invariance; unrestricted-readout latent nonidentifiability; PSD induced metric. | Invertible consistently applied transform; full rank for unique coefficients and the stated evaluation guarantee; matching penalty. | Native geometry cannot be inferred from predictive sufficiency without additional restrictions. |
| P5 | Pooling identity, conditional pure weighting, affine product nonidentifiability, and cross-repeat unbiasedness. | Diagonal/commuting measurement action with degeneracy handled; positive factors for logs; aligned repeats and stated noise moments. | Approximate brain-mode preservation, spatial frequency/order correspondence, and an $\alpha=2$ mechanism. |
