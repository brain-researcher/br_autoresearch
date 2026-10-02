# EP13 scope, novelty and clarity review — 2026-10-02

Mathematical/design review only; no new proof replay or computation is claimed.

## Exact question

For known A and P, unrestricted rational X, observation Y = A X P^T and a
prespecified target tau = s^T X c, is tau constant on every observation fiber?
After column-major vectorization, this is linear-functional estimability for
M = P ⊗ A and target direction c ⊗ s.

## Closest primary reference and contribution assessment

[Searle (1965), Additional Results Concerning Estimable Functions and
Generalized Inverse Matrices](https://academic.oup.com/jrsssb/article/27/3/486/7026850)
is a primary reference for classical linear estimability and generalized
inverse results. Row-space/null-space duality and exact linear readout are
established mathematics, not a new BOLD identifiability theorem.

The separable criterion, explicit zero branches, rank-one null witnesses and
orientation-safe certificates make a useful reference implementation and
exposition. They do not by themselves supply sufficient standalone
research-paper novelty. That assessment is now explicit in GOAL and the
manuscript plan, rather than implying a publishable methods contribution.

## Keep the useful scope

Retain the exact theorem, constructive positive/negative certificates and
small exact-rational implementation examples. A rank-deficient observation
can determine a particular scalar even when it cannot reconstruct all of X.
Finite tests check implementation; the proof establishes the universal claim.

Noise, uncertain operators, physiological validity and numerical stability
are different questions. They are neither solved by this theorem nor
automatically added to the episode to manufacture novelty. Restricted latent
families would also change the question.

There is no empirical biological claim to validate in the present scope.
The recorded next work is exact certificate/orientation checking and
independent replay. This review does not declare those tests executed, add
an experiment, or change the theorem, policy or ASTRA program.
