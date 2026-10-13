# EP13 outputs

[Public online methods reference, checked October 6](../DATASETS.md#public-online-resources--checked-october-6-2026): Searle's original estimability paper and the distinction between published theory and empirical data.

**Locally complete: `theorem_verified`, October 3, 2026, 11:03 PDT.** The
[proof package](exact/proof.md) establishes the full row/kernel/rank/readout
equivalence, zero-aware separable criterion and constructive certificates.
The scalar target is identifiable exactly when `c=0`, `s=0`, or both
`c ∈ row(P)` and `s ∈ row(A)`.

The [orientation report](exact/runs/attempt001/orientation_report.json) passes
all 12 valid exact fixtures and seven invalid-input rejections. The
[independent replay](exact/runs/attempt001/independent_replay_report.json)
agrees on exact observations, targets and identifiability and supplies its own
replayed certificates. The [attempt lock](exact/runs/attempt001/package_lock.json)
binds the authored proof, input suite and both implementations before replay;
the [completion record](exact/runs/attempt001/completion.json) records Slurm
allocation `46424502`, Python 3.12.1 and the actual result. The
[execution log](experiment_log.md) records authorship independence and review.

![EP13 scalar-target identifiability](ep13_conceptual_question-v2.png)

The figure illustrates classical linear estimability, not a new theorem or
result artifact. With known operators, exact observations and unrestricted X,
the scalar is identifiable iff `c = 0`, `s = 0`, or both
`s ∈ row(A)` and `c ∈ row(P)`.
[Built-in imagegen prompt set](ep13_conceptual_question-v2-prompt.md).
The earlier conceptual PNG remains historical.

The independent checker was authored from the frozen contracts and interface
before its author inspected the main implementation, fixtures or results. Both
implementations use standard-library rational arithmetic. Source:
[input suite](exact/fixtures.json), [component checker](exact/target_certificate.py),
[independent checker](exact/independent_checker.py), [runner](exact/run_exact.py).
The finite suite audits implementation and orientation; the general proof is
the universal evidence. This is a classical technical reference, not an
empirical BOLD finding or canonical scientific acceptance. No further EP13
scientific compute is required by the current contract.
