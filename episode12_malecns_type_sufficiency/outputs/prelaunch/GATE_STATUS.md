# EP12 pre-outcome gate status

Status date: 2026-09-25. Overall status: **development launched; final locked / not qualified**.

This table distinguishes an executed check from the existence of a file. A
gate passes only when the linked execution produced its required evidence.

| Gate | Status | Executed evidence |
| --- | --- | --- |
| Episode/storage boundary | pass | Executor paths are restricted to this EP12; durable scientific outputs are under `outputs/`; temporary work is under the EP12 scratch root. |
| Synthetic control-plane qualification | pass | `outputs/executor_qualification_v3/qualification_report.json` passed against the current executor code hash; scope is explicitly synthetic control-plane only. |
| Development/final connectivity firewall | pass for first development configuration | `development_run001/materialization/development_exposure.json` records development body-statistic/weight access and explicitly records that final focal connectivity was not materialized or summarized. The contract authorizes `development_only` and keeps final access false. |
| Source schema authentication | pass | `source_preflight/source_schema_manifest.json` checked required annotation, body-statistic, and weight schemas against the pinned provider manifest without opening body-statistic or weight record values. |
| Annotation-only eligibility and whole-type roles | pass | `source_preflight/annotation_role_manifest.json` records 1,020 eligible whole types, split exactly into 816 development and 204 final types with a content hash. |
| Canonical counts and endpoint-mass bins | pass for first real branch | The hashed development sparse matrix retains raw integer weights/totals and 11,771 raw partner keys; trial001 deterministically retained 11 typed partners and seven explicit pooling/endpoint bins without dropping mass. |
| Actual bounded T/U/M fitter | pass for one real grammar configuration | `development_run001/trial001_reconciled/type_direction_receipts.jsonl` accounts for all 816 development types: 807 complete numerical T/U/M fits and nine prespecified zero-count unscorable types. |
| Source-only transfer / no target refit | pass for fixed method | Fitted model objects are immutable during target scoring; tests and receipts use source-only CV and marginalize target latent coordinates and labels. |
| Numerical integration stability | pass | `scientific_qualification_dev007/integration_stability.json`: 256 versus 512 draws changed directional M-minus-reference gain by at most 0.001652, below the 0.002 derived tolerance, with identical selected model IDs. |
| Numerical failures | pass for reconciled first real configuration | The original run misclassified nine zero-count reason-code types as technical failures. Snapshot-verified reconciliation changed no score and records zero remaining technical failures. |
| Fixed-method cost | pass, limited scope | `scientific_qualification_dev007/cost_profile.json` projects about 336 CPU-core-hours for the fixed method under one numeric thread per worker, below the 1,000-hour policy ceiling. This is not a grammar-wide cost proof. |
| Real-development cost | **user-authorized for the minimum full search** | `development_run001/real_cost_projection.json` measures 0.764 core-hours for one complete real configuration. At 36 trials × 100 observed/null searches, the direct projection is about 2,750 core-hours. On 2026-09-25 the user authorized this planning scale and removed the original 1,000-hour stop for completing that scope; see `outputs/executor/DEVELOPMENT_BUDGET_OVERRIDE.yaml`. |
| Synthetic operating characteristics | **fail / insufficient** | The smoke has only one replicate per cell and cannot bound 5% error rates. The qualified 72-replicate run has not passed. Current grouped scenarios also expose an adequacy-resolution problem at the fixed margin. |
| Coverage at nominal 95% | **fail / insufficient** | No completed 72-replicate generated run currently establishes the required coverage interval. |
| Adequacy sensitivity at or above 0.02 | **fail** | Generated grouped-with-continuum probes do not yet demonstrate reliable bilateral M gain above 0.02 across the declared sample-size/count grid; adequacy must therefore remain unavailable. |
| Full adaptive grammar | **fail / incomplete** | One declared branch has executed on real development data. The remaining partner-vocabulary, representation, rank, covariance, regularization, and covering-array branches are not implemented/executed. |
| Mode separation, prevalence, partner identifiability, and cross-side matching intervals | **fail / unimplemented** | Model diagnostics exist, but the required simultaneous lower-bound and matching decision code is not complete. |
| Anatomy, endpoint, type, partner, and high-strength influence controls | partial | Endpoint permutation and leave-one-type-out controls execute on generated data; the complete policy control set is not implemented or qualified. |
| Complete 99-search null | **fail / not run** | No observed adaptive search exists to calibrate; asserted controller fixtures do not count as executed scientific null searches. |
| Deterministic full scientific replay | **fail / not run** | Unit-level deterministic scoring is tested, but a complete adaptive search plus null replay has not run. |
| Slurm real execution | pass for first configuration | Materialization job `45279289` completed in 6:11. Real T/U/M job `45280169` completed all 816 receipts in 46:16; its nonzero exit reflected the subsequently reconciled zero-count classification. |
| Locked launch/evaluator manifest | **fail / absent by design** | A launch manifest cannot be issued while required scientific gates fail. |

Consequently EP12 has genuinely launched development connectivity execution,
but the adaptive search and qualification are incomplete, final connectivity
remains locked, and no EP12 scientific terminal has been produced.
