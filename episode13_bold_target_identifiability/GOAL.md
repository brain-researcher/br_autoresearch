# Counterexample-Guided Exact Theorem Discovery for BOLD Targets

## Authority and exposure boundary

This is the current local Episode 13 contract. It defines the scientific
question and constraints; it grants no compute or locked-audit access and does
not award a reward or establish a result. Prior exact cases and their answers
are exposed development facts; no deleted local artifact is a runtime
dependency.

The episode follows
[`../ADAPTIVE_SEARCH_PROTOCOL.md`](../ADAPTIVE_SEARCH_PROTOCOL.md). It treats
AI-scale iteration as counterexample-guided inductive synthesis (CEGIS), not
as an empirical benchmark. Exact proof or exact refutation—not the number of
tested matrices—is the scientific terminal.

## Adaptive mathematical question

For the exact rational finite-dimensional model

\[
A_g=RSH_g,\qquad Y=A_g B_g^\top P^\top,\qquad
\tau=c_g^\top B_gs_g,
\]

what are the weakest sufficient-and-necessary conditions under which the
prespecified scalar `tau` is uniquely determined by `Y` for every admissible
latent matrix `B_g`?

With a declared vectorization convention, the starting conjecture family uses
`M = P kron A_g` and `ell = c_g kron s_g` and relates:

- target membership in `row(M)`;
- containment `ker(M) subset ker(ell^T)`;
- rank preservation after appending `ell^T`;
- exact readout or target-changing null certificates; and
- factorization into temporal and spatial row-space conditions, including all
  zero-target and non-square cases.

The system may discover a shorter proof, remove an unnecessary assumption,
find an exact counterexample, or repair the theorem to a correct bounded
statement. It may not replace exact algebra with floating-point agreement.

## Development and audit roles

- **Exposed development evidence:** the historical V3 bundle, including 13
  valid exact cases and four invalid-input controls, plus every counterexample
  generated during CEGIS. These cases guide conjecture repair and can never be
  called confirmation.
- **Exact proof workspace:** symbolic derivations and machine-replayable
  rational certificates. This is the only source of universal support.
- **Locked implementation audit:** a separately authored exact verifier and a
  frozen generated-case rule, held out from controller feedback until one
  post-lock run. It can detect implementation or orientation errors, but its
  finite cases cannot establish a universal theorem.

There is no empirical participant dataset, biological replication, or
statistical train/test split.

## Bounded conjecture grammar

Each trial is one structured conjecture package, not an arbitrary essay. Its
grammar is limited to:

- **base operator statement:** row-space membership, kernel annihilation, or
  rank-augmentation equality for `M` and `ell`;
- **certificate form:** exact readout `w` with `M^T w = ell`, exact
  observation-preserving witness `delta` with `M delta = 0` and
  `ell^T delta = 1`, or a factorized `u,v` construction;
- **factorization clause:** temporal condition on `s_g` and `A_g`, spatial
  condition on `c_g` and `P`, their conjunction, or an explicit statement that
  only the full operator criterion is valid;
- **case partition:** both targets nonzero, `s_g = 0`, `c_g = 0`, and shape/
  validity rejection cases;
- **assumption set:** compatible finite dimensions, exact rational entries,
  declared vectorization, and selected model-validity assumptions such as
  idempotence of `R`; and
- **repair operator:** add/remove one assumption, split one case, correct a
  Kronecker orientation, weaken an equivalence to an implication, or replace
  a false factorized clause while retaining the strongest proven full-operator
  result.

The grammar excludes empirical HRF fitting, numerical conditioning claims,
approximate tolerances as proof, participant data, and unconstrained theorem
generation outside this observation model.

## Objective, constraints, and nonterminal incumbent

Candidate conjectures are ordered lexicographically:

1. no known exact counterexample and all proof obligations discharged;
2. greatest valid scope with the fewest assumptions;
3. complete handling of zero, asymmetric, non-square, and invalid-input cases;
4. independently replayable certificates; and
5. simplest theorem and proof among equally general correct candidates.

Until a complete exact proof or exact refutation is independently replayed,
the incumbent is a nonterminal conjecture. Passing thousands of finite cases
does not promote it. Every failed proof obligation and counterexample is
append-only evidence and must generate either a grammar-valid repair or a
recorded dead end.

## CEGIS stages

1. **Specification lock:** freeze dimensions, vectorization, admissibility,
   exact serialization, proof obligations, grammar, and malformed-input rules.
2. **Independent tooling:** implement a rational checker and certificate
   replay path without importing the historical evaluator, cases, expected
   labels, constants, solver, or precomputed answers.
3. **Coverage stage:** instantiate conjectures for all base criteria,
   factorization clauses, and zero/nonzero partitions and challenge them with
   constructive exact cases.
4. **CEGIS loop:** for each conjecture, attempt a symbolic proof and in
   parallel search exact small-dimensional counterexamples. A counterexample
   is minimized, replayed, entered into the permanent corpus, and used to
   produce a single declared repair.
5. **Theorem lock:** freeze one theorem or precise refutation, its assumptions,
   proof, certificates, verifier implementation, generated-case rule, and
   invocation environment.
6. **Audit once:** run the separately authored verifier and frozen generated
   cases exactly once against the locked package. No audit result may silently
   alter the theorem; failure invalidates the lock and requires a new episode.
7. **Terminal adjudication:** accept only an exact proof or exact
   counterexample/refutation with a proved corrected statement.

## Mandatory falsifiers

- a rank-deficient full operator with a nonzero identifiable target;
- temporal-only, spatial-only, and simultaneous information loss;
- `s_g = 0`, `c_g = 0`, and both zero;
- non-square asymmetric matrices that reveal Kronecker/vectorization order;
- a non-symmetric idempotent residualizer;
- valid cases with nontrivial null spaces and independently replayed readout
  or target-changing certificates;
- incompatible shapes, malformed rationals, and non-idempotent residualizers;
- exhaustive enumeration over a frozen bounded small-integer matrix family;
  and
- property-based dual construction: generate row-space targets from `M^T w`
  and nonidentifiable targets with a certified null direction.

Ablations must separately remove the row, kernel, rank, certificate, and
factorization clauses so that no equivalence is accepted only because another
checker supplied the answer.

## Numeric search contract

- minimum conjecture trials: **16**;
- maximum conjecture trials: **64**;
- patience: **12** consecutive completed conjecture trials without a strictly
  better lexicographic incumbent, active after trial 16;
- total compute ceiling: **256 CPU-core-hours**;
- per-trial ceiling: **8 CPU cores**, **32 GiB RAM**, **4 wall-hours**;
- overall wall-clock ceiling: **72 hours**;
- GPU allocation: **0**;
- all conjectures, assumptions, proof attempts, counterexamples, repairs,
  exact certificates, checker outputs, and resource records are retained.

Patience or resource exhaustion may close the search as unresolved; it can
never turn finite coverage into a theorem.

## Lock, one-shot audit, and terminals

After theorem lock, the audit verifier runs once. A failing audit produces an
invalidated-lock record and a non-success terminal for this episode; further
repair requires a newly registered episode with a new lock.

Valid terminals are:

- `candidate_ready_theorem`: a complete exact proof covers the declared
  finite-dimensional family, all case partitions, and independently replayed
  certificates; the finite audit is consistent but is not the proof;
- `candidate_ready_refutation`: an exact minimized counterexample refutes the
  starting conjecture and a corrected bounded statement is itself exactly
  proved;
- `closed_unresolved`: search budget ends without proof or complete bounded
  refutation; or
- `technical_failure`: the exact contract, checker independence, certificate
  replay, or lock/audit integrity is ill-defined.

The result concerns only a fixed exact rational linear observation operator.
It does not establish numerical stability, robustness to HRF or preprocessing
uncertainty, biological validity, neural mechanism, reverse inference, or the
assumptions of any real experiment. The two exact positive terminals map to
canonical `candidate_ready`; unresolved maps to `closed_no_candidate`; an
invalid exact contract or audit maps to `technical_failure`.
