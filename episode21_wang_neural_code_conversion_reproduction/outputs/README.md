# Outputs

EP21 was corrected on 2026-09-28 to reproduce Wang et al.'s content-loss
neural code conversion study, not the superseded Doerig et al. proposal. The
correction occurred before any EP21 empirical access, model fit, score,
process, or scheduler job. The earlier draft remains recoverable in Git
history but has no scientific standing in the active episode.

The durable workspace is
`/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction`.
Large reconstructible intermediates belong under
`$SCRATCH/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/`
only after separate execution authorization. `inputs/` remains read-only.

`../PROTOCOL_TABLE.yaml` is the unresolved outcome-independent design table.
`source_qualification` and `pipeline_conformance` must resolve every shared
field and every field for a runnable branch, or lock the affected branch
`not_evaluable`, before writing `protocol_lock.yaml`. Every model-fitting or
scoring analysis consumes that lock. The fixed experiment program is
`astra/v0.0.14/astra.yaml`; mutable attempts, jobs, blockers, and next actions
belong in `experiment_log.md` under the same analysis IDs.

The primary LAION branch exposes one non-neural regular feature handoff and one
regular neural scoring handoff. Within the scoring handoff, manifest-enforced
role slices separate subject-unique fitting rows, shared `tau` train rows, and
shared `tau` test rows; no extra undeclared path is implied. The NSD/THINGS
inter-site branch is independently provisioned and may remain `not_evaluable`
without blocking the LAION branch. Live EP04 outputs and legacy EP04 Scratch
are forbidden inputs. The entire 371-image OOD payload—including raw images,
derived VGG/AlexNet features, neural values, reconstructions, predictions,
scores, and summaries—remains closed to EP21.

`ep21_neural_code_conversion_design.png` is a conceptual design mockup linked
from `../GOAL.md`. It contains no observed data and is not evidence of
execution, completion, or a scientific result.
