# Formal Corpus Contract — Episode 13

This episode has no empirical dataset. Its inputs are exact rational matrices,
vectors, structured conjectures, proofs, finite regression cases, and
replayable certificates. The common adaptive governance rules are in
[`../ADAPTIVE_SEARCH_PROTOCOL.md`](../ADAPTIVE_SEARCH_PROTOCOL.md), but finite
case separation here is an implementation audit, not statistical validation.

## Prior-exposure boundary

Prior work exposed 13 contract-valid cases and four invalid-input controls.
Their evaluator, cases, labels, constants, operator construction, solver, and
precomputed answers are development-history facts only; no local legacy file
is retained or required by the current verifier. No new corpus, verifier, or
audit output is asserted to exist by this contract.

## Exact declared family

Each valid finite case supplies compatible exact-rational objects for

`A_g = R S H_g`, `Y = A_g B_g^T P^T`, and `tau = c_g^T B_g s_g`.

| Object | Shape |
| --- | --- |
| `H_g` | `F x V` |
| `S` | `T x F` |
| `R` | `T x T` |
| `P` | `Q x K` |
| `B_g` | `K x V` |
| `c_g` | length `K` |
| `s_g` | length `V` |

`R` is square and idempotent; symmetry is not required. Rational entries use
exact strings such as `"3"`, `"-2"`, and `"5/7"`. Floating-point
approximations cannot support a proof or certificate. Generated regression
cases may retain the historical dimension-at-most-four convention, but that
finite bound never limits or proves the universal theorem.

## Evidence roles

| Role | Contents | Epistemic use |
| --- | --- | --- |
| `history_exposed` | declared prior cases, theorem candidate, and known outcomes; no local artifact dependency | development and regression only |
| `cegis_development` | every generated exact case, failed proof obligation, minimized counterexample, and repair | adaptive conjecture search |
| `proof` | exact derivation valid for arbitrary compatible finite dimensions | universal support or refutation |
| `audit_sealed` | independently authored checker plus a frozen deterministic/exhaustive case rule, hidden from controller feedback until lock | one-shot implementation and orientation audit only |

The audit generator may construct new cases from the frozen rule, but these
are generated fixtures rather than an independent biological dataset. Once
run, they join exposed history. A second run cannot restore a fresh audit.

## Independence boundary

Before comparison with prior results, the episode must freeze an independently authored:

- operator/target construction and column-major vectorization convention;
- exact row-space, rank, null-space, and certificate replay implementation;
- deterministic generated-case rule and any seed;
- exhaustive bounded enumeration domain, if used;
- malformed-shape/rational and non-idempotence rules; and
- output schema for proof obligations, counterexamples, and certificates.

The audit checker may not import, execute, copy constants from, or use expected
labels from any prior evaluator. Agreement with prior results is a regression
comparison after the independent lock, not a source of truth.

## Required case coverage

Development and locked audit rules together must exercise:

- non-injective `M` with an identifiable nonzero target;
- temporal-only, spatial-only, and dual losses;
- zero `s_g`, zero `c_g`, and both zero;
- non-square asymmetric orientation cases;
- non-symmetric idempotent `R`;
- exact readout certificates and normalized target-changing null witnesses;
- compatible full-rank and rank-deficient cases; and
- incompatible shapes, malformed rationals, and non-idempotent `R`.

Case labels must be derived by an exact independent oracle or directly checked
certificate, never by floating-point tolerance or the incumbent conjecture.

## Storage and execution boundary

All current outputs, if later authorized, must include the conjecture ledger,
exact case corpus, minimized
counterexamples, proof versions, independent checker, lock manifest, and
one-shot audit record. CPU execution belongs
on an authorized compute node; no participant data, images, raw BOLD, or GPU
training is part of this contract.
