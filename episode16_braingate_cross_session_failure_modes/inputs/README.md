# Episode 16 inputs

`inputs/` is read-only. It intentionally contains no symlink or copy of the
84.73-GB mixed Dryad source.

`../DATASETS.md` requires three operator-created, content-addressed handoffs:
development full data, audit structural metadata, and evaluator-only audit
data. None exists yet. The target-schedule-only structural support scan and 6/3
participant role manifest have not been instantiated. The actual sequence is:
regenerate the
terminal-relevant exposure ledger, freeze the target-schedule-only structural
support and 6/3 roles, then apply the immutable scorable-mask algorithm inside
the role-filtered development builder or one-shot audit evaluator. Scorability
cannot move a participant after the role hash.

Do not extract archives here, mount a mixed audit archive into a candidate
worker, or infer launch authority from the verified shared source. After an
authorized launch, transient extraction belongs under
`$SCRATCH/autoresearch/episode16_braingate_cross_session_failure_modes/`.
