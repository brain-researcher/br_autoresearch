# EP10 Stage 0 — outcome-blind resource qualification

EP10 was started on 2026-09-26 UTC in the episode-managed qualification
stage defined by `SEARCH_POLICY.yaml`. This directory contains the preserved
SEU qualification history and the current lean Gao MOp readiness records. It
is not a model-search result or a scientific claim.

The mixed-role SEU archives remain in steward quarantine. They are not mounted,
symlinked, or unpacked here. Stage 0 may read the metadata workbook through the
trusted redaction path, inspect ZIP member names without reading SWC contents,
and inspect atlas headers without reading image volumes. It must not expose the
`Projection class` column, decode morphology coordinates, assign development
and final roles, or open final outcomes.

The first qualification question is whether one source population can support
12 development and 8 final conservative specimen groups. Failure of that gate
means revise the resource strategy or narrow/stop the endpoint before outcome
search. It is not evidence against biological co-projection organization.

`qualification_run001/` through `qualification_run005/` are preserved as the
historical SEU engineering record. They produced the same scientific support
decision: SEU MOp has only 17 conservative specimen groups and cannot support
the frozen 12-development/8-final design. Their mechanical launch machinery
is nonbinding for the selected Gao resource; no additional schema, hash, or
interpreter gate is required. The run-002 durable copy was redacted before
handoff, and no retained result publishes forbidden-column occupancy.

The later [lean Gao MOp qualification](gao_mop_qualification.md) intentionally
does not extend the SEU launch machinery with another schema, hash manifest, or
test harness. Its 76-row [source/group table](gao_source_group_support.csv)
shows that Gao MOp passes the released brain/specimen-count screen (837 cells,
76 fMOST/sample units). Canonical soma-root rows, and no axon-tree rows, were
opened for geometry: 837/837 cells were eligible, and the outcome-blind
radius-first rule froze 500 um/0.20 common support. The
[role ledger](gao_mop_frozen_roles.csv) fixes 12 development, 8 final, and 56
unassigned units; all 15 units shared with EP11 are unassigned. Independent-
animal identity, full-tree reconstruction QC, and complete-target observation
remain unresolved. No complete Gao tree or projection outcome was opened.

The outcome-blind [projection-observation contract](projection_observation_contract.yaml)
is now frozen. Its full 56-coordinate atlas map replaces `MO` with `MOs`, and
its primary exact model uses the prespecified 16-coordinate motor-routing
panel MOs/SS/ACA/STR/TH/MB/P/MY crossed with soma-relative side. Detection
requires terminal-bearing target cable; passing-only cable is not a target,
globally unreliable cells are unknown, and zero means only not observed in the
eligible released reconstruction rather than biological absence. Primary
modeling is restricted to the frozen 200 common-support cells among 209
development-role cells;
structural QC must retain at least 190 plus every frozen per-group minimum
across all 12 groups or the endpoint is revised/stopped without replacement.
The four `Axon_only` cells cannot support a sensitivity under the frozen roles
(0 development, 1 final, 3 unassigned) and will not be opened for development.
The [development manifest](gao_mop_development_manifest.csv) contains exactly
200 common-support primary trees from the 12 frozen development groups and no
unsupported, final, unassigned, or EP11-shared row. Full-tree acquisition is pending an attended
authenticated DTN session; no tree body was fetched while preparing these
artifacts.

Outcome-blind Slurm preflight job `45412269` completed successfully. It
re-derived the exact 200-supported/9-excluded development-cell boundary,
reproduced the frozen 12/8/56 role allocation, and passed the downloader and
synthetic terminal-constructor checks without opening an SWC body. The
[exact-normalization benchmark](exact_normalization_benchmark.json) ran 99
synthetic suites in 650.99 seconds (6.58-second median); each suite exhaustively
enumerated all 65,536 states across `K=0..16` and the bounded 56-coordinate,
`K=5` case with 3,819,816 states. This measures the exact-normalizer workload,
not the cost of a complete 99-null model-search rerun.

Canonical Brain Researcher state is not bound to this standalone launch. No
Society review, reward, confirmation, ClaimCard, or Landscape transition is
created by these files.
