# Episode 16 inputs

This directory is read-only and intentionally empty. The 84.73-GB Dryad
release remains in shared research storage rather than being copied into the
episode.

Before neural scoring, the study needs three clearly separated data views:

1. full data for the six development participants;
2. target-schedule and trial-count information for choosing the three held-out
   participants without neural scores; and
3. full data for those three participants, available only for the final
   pre-specified evaluation.

None of those episode-specific views has been prepared. The next data step is
the outcome-blind target-schedule and near/long pair-support check described in
`../DATASETS.md`. That check determines whether a six-person development set
and a three-person held-out set are scientifically possible.

Do not extract the complete release here or give development code access to
held-out neural data. Temporary extraction for authorized episode work belongs
in `$SCRATCH/br_autoresearch/episode16_braingate_cross_session_failure_modes/`.
