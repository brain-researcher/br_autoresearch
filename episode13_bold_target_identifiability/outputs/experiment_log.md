# Experiment log

## 2026-10-03 11:03 PDT — `formal_theorem_derivation`, `exact_orientation_suite`, `independent_certificate_replay`: locally complete

Actual execution at `2026-10-03T18:03:23.714204Z` completed in root-owned Slurm
allocation `46424502` on `sh04-18n31.int`, Python 3.12.1. The original selected
command returned **`theorem_verified`**, with all 12 valid exact cases and seven
invalid-input rejections passing. No repair or repeat was needed.

Durable evidence under `exact/runs/attempt001/`:

- [package_lock.json](exact/runs/attempt001/package_lock.json) binds the proof,
  fixed input tuples and both implementation revisions before independent replay.
- [orientation_report.json](exact/runs/attempt001/orientation_report.json)
  records exact observation/target identities, positive readouts and normalized
  rank-one invisible witnesses. It detects the wrong Kronecker order and wrong
  residualizer transpose, replays nonunique readouts, and distinguishes realized
  zero target from structural identifiability.
- [independent_replay_report.json](exact/runs/attempt001/independent_replay_report.json)
  agrees on every valid/invalid input decision and exact observation, scalar and
  identifiability result. Its independently generated certificates also replay.
- [completion.json](exact/runs/attempt001/completion.json) records the actual
  local outcome, execution host/allocation and non-canonical claim boundary.

The root coordinator authored `exact/independent_checker.py` from GOAL,
DATASETS, SEARCH_POLICY and the input/output interface only, before inspecting
main implementation, fixtures, expected labels or certificates. The independent
implementation uses a full Kronecker exact solve for its decision; the main
implementation constructs component certificates and cross-checks the general
criteria. Neither imported historical evaluator code or historical constants.
Only input tuples were passed to the independent checker. This is separately
authored implementation replay, not an independent biological replication.

After checker authorship, root reviewed [proof.md](exact/proof.md) and found
all declared equivalences, zero branches, coordinate identities, normalized
rank-one witnesses and the `Rᵀ` expansion correctly derived. The proof covers
the full rational latent space; finite examples establish implementation and
orientation agreement, not universal truth by enumeration. No noise, stability,
HRF, latent-recovery, biological or causal claim is added. No Society decision,
reward, confirmation or canonical state mutation occurred.

All current scientific requirements are complete; no additional EP13 compute
is needed. The existing pinned ASTRA 0.0.14 validator was located for the
changed program. Root executed its official schema and semantic checks in
Slurm allocation `46424502`; both passed. Actual CLI output:
[validation_20261003.json](astra/v0.0.14/validation_20261003.json). This validates
the program representation, not a second theorem or canonical acceptance.
Earlier planned/blocked descriptions below are historical and superseded by
this completed attempt.

## 2026-10-03 — `formal_theorem_derivation`, `exact_orientation_suite`, `independent_certificate_replay`: authorized preparation

The scientist authorized resuming possible work and completing jobs for all
questions. This task owns EP13 only. Read the complete frozen GOAL, DATASETS
and search policy; there are no empirical datasets or sealed BOLD outcomes.
The executing preparation shell is `sh04-ln04` without `SLURM_JOB_ID`; source
editing only occurred here. Root owns compute and scheduler observations.

Authored [the exact proof](exact/proof.md), [fixed input-only suite](exact/fixtures.json),
[component certificate implementation](exact/target_certificate.py), and
[replay runner](exact/run_exact.py), preserving all three stable ASTRA analysis
IDs. The suite covers the required nonzero/zero, orientation, nonunique readout,
normalized invisible perturbation and malformed/application-input cases. It
uses Python standard-library rational arithmetic, with no installation.

The separate checker path `exact/independent_checker.py` is reserved for an
independent author. That author receives only the frozen theorem contracts and
an input/output interface, not main implementation, expected labels, historical
evaluators or computed certificates. The orchestrator supplies only neutral-ID
input tuples to that checker and replays its independently generated certificates.
Each attempt locks the proof, suite and both implementation revisions before
independent replay, as required by the frozen contract. A failed attempt is
retained and requires a distinct repaired attempt.

Selected execution: root-owned Slurm allocation, one CPU, one GiB, expected
seconds; no separate short scheduler job. Command from the episode root:
`python3 outputs/exact/run_exact.py --run-dir outputs/exact/runs/attempt001`.
The two ASTRA recipes expose the same orientation and independent phases
separately. No process/job has been launched for EP13, no exact replay has
passed yet, and the proof package is not canonically accepted. Changed ASTRA
schema/semantic validation remains to be performed on compute if the existing
pinned tooling is available; its availability does not block exact science.

Next action: receive the independently authored checker, execute the fixed
package in root's allocation, inspect actual certificates/rejections and record
the result. No adaptive conjecture search or extra scientific endpoint is added.

## 2026-10-02 — Concept figure v2

Built-in imagegen produced the [scalar-versus-latent-state illustration](ep13_conceptual_question-v2.png),
with its [exact prompt set](ep13_conceptual_question-v2-prompt.md). The figure
uses dimension-agnostic matrix glyphs after targeted visual correction; exact
formulas and observation/target arrows were inspected. Reader captions retain
the row-space condition and zero exception. Earlier artwork remains.
No theorem, policy, ASTRA, proof replay or execution state changed.

2026-10-02 — [Scope/novelty review (consolidated)](paper_plan.md) adds the classical Searle estimability reference and classifies the output as a reusable technical reference/lemma package, not an established standalone paper contribution. Existing theorem and exact-check program are retained; no new proof replay, experiment or runtime observation.

## conceptual_figure

Status: complete.

The explanatory figure in `ep13_conceptual_question.png` summarizes the frozen
model, the zero-aware decision rule, constructive certificates, and the claim
boundary. It is a conceptual aid, not empirical evidence or a replay result.

## astra_program_validation

Status: complete.

The pinned ASTRA 0.0.14 schema and semantic validators accept
`astra/v0.0.14/astra.yaml` with zero warnings. Its internal analysis-output
references also resolve. This validates the experiment description, not the
theorem or its future replay artifacts.

## formal_theorem_derivation

Status: design complete; exact machine replay not started.

The contract now states the full latent domain, column-major convention,
general row/kernel/rank/readout equivalence, separable zero-aware corollary,
constructive readout, and constructive null witness.

## exact_orientation_suite

Status: planned.

Next action: implement the fixed rational examples from GOAL.md and directly
replay observation, target, readout, and null-witness identities.

## independent_certificate_replay

Status: blocked on a separately authored checker.

No historical evaluator, expected label, proof lock, or replay result has been
opened. No scheduler job or process is active.
