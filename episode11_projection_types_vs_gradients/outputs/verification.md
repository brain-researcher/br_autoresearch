# Verification

- Current Goal, dataset contract, search policy, and prospective paper plan:
  present.
- Source archives: acquired only in mixed-role steward quarantine; not
  provisioned to EP11.
- Source-specific verified-animal support census: not yet available.
- The current design's 12-development/8-audit requirement within one selected
  source is not yet demonstrated; it is not treated as a universal threshold.
- Local Stage-0 run `ep11-stage0-20260925T000506Z` is active. Its frozen launch
  manifest hash is
  `e57fa907df46cca10d628d97a3dfd43ce67ec51facc07663c0f360e3c35b7aa3`.
- Contract/lint checks rerun on 2026-09-25 after the Stage-0 amendments:
  - PyYAML duplicate-key parse: passed;
  - `SEARCH_POLICY.schema.json` validation with `jsonschema`: passed;
  - revised access, development-selection, 12/8 interpretation, and redesign
    semantics: passed;
  - all Stage-0 YAML duplicate-key parsing and validator Python AST parsing:
    passed;
  - five-event experiment-log hash chain and launch-manifest hash: passed;
  - local Markdown-link resolution: passed; and
  - `git diff --check -- episode11_projection_types_vs_gradients`: passed.
- Synthetic trusted-side metadata-validator checks:
  - allow-listed 12-development/8-audit fixture: accepted while full support
    remained explicitly unadjudicated;
  - one animal assigned to both roles: rejected; and
  - forbidden `Projection class` column: rejected; and
  - trusted-steward mode with a matching synthetic safety receipt: accepted,
    emitting identifier-free aggregates only.
- Checked artifact SHA-256 values:
  - `GOAL.md`: `b95d9f3e8ed2f7315326f00ee47c9143f31cf58c62ba80fa928ef4c753b0f93a`;
  - `SEARCH_POLICY.yaml`: `9abf97f6f97ac0b061b4070e6337a3eb0a80c7a67e7eea87b7186b62a7d234d5`;
  - `DATASETS.md`: `89d189099b64d71a2ec558444a9ea1b3840b7790ba95d6ef8b18cdf111ec2aae`;
  - `outputs/paper_plan.md`: `f090ffbf2540b3b55979bc3249afeb77e87e09c5f89ee1b62039ac8b7e2aea10`.
- These are contract and lint checks only. They do not establish scientific
  validity, model qualification, source readiness, or cross-animal support.
- Outcome computation: not run.
- One-shot audit: not opened.
- Only mechanical source/CCF byte hashes were checked. Workbook cells,
  morphology archive contents, projection outcomes, and audit outcomes were
  not opened in this run.
- Stage 0 is waiting for the trusted handoff specified in
  `stage0/steward_handoff.md`. Projection-outcome development remains blocked.
