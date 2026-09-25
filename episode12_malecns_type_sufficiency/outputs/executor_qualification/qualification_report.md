# EP12 synthetic executor qualification

Date: 2026-09-25

Episode:
`/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency`

Temporary root:
`/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency`

## Result

The episode-specific executor passed 21/21 synthetic tests and 16/16 durable
qualification checks. This result qualifies only the synthetic control plane.
It does not qualify the scientific T/U/M models, enable real-data execution,
permit scientific interpretation, or represent an executed full-search null.

| Fixture | Detailed terminal | Outer status | Valid synthetic trials | Final opens |
| --- | --- | --- | ---: | ---: |
| Positive | `candidate_ready_residual_modes` | `candidate_ready` | 52 | 1 |
| Adequate | `closed_single_population_adequate_within_margin` | `closed_no_candidate` | 52 | 1 |
| Unresolved | `closed_unresolved` | `closed_no_candidate` | 52 | 1 |
| No valid comparison | `closed_no_valid_comparison` | `closed_no_candidate` | 0 | 0 |
| Budget exhaustion | `incomplete_search` | `closed_no_candidate` | 12 | 0 |
| Input failure | `technical_failure` | `technical_failure` | 0 | 0 |

The three complete-search fixtures each retain 99 asserted synthetic null
receipts and report protocol arithmetic of 15,600 model-family fits. The
receipts state that complete controller reruns and deterministic null replay
were **not** executed. They are contract fixtures, not scientific evidence or
a runtime profile.

## Integrity

- Policy hash:
  `d64c5c9c56b70f3312472e056b9394f519107eda268902ceb586771d435d8b8d`
- Executor code hash:
  `dad853859b17e6d426a07b85aa92723df330a79c492cd0c702c3ac7d083dbe38`
- Canonical JSON report hash recorded by the manifest:
  `dc428ee18d013e2dd636d45e2b1f010995d71a7ff0cd09bc0c8d04124875d111`
- Report-file SHA-256:
  `73f9e9fb22b2f77102c332785b0674a46a4f0e939eb143ce607a5f23ea65a785`
- Manifest-file SHA-256:
  `86efe2714a2917a0e1d725d14c01c71e5740583f1dc2bfe01af65dbb6e6cc557`

The qualification command was immediately repeated. It re-verified all six
hash-chained journals and every referenced synthetic final vault and null
receipt file, then returned the existing report idempotently.

## Boundaries

- Connectivity outcome values inspected: 0.
- MaleCNS declared source accessed: no.
- `inputs/` modified: no.
- Real-data command available: no.
- Scientific shape fixtures passed: no; all five remain pending.
- Scientific final evaluation opened: no.
- Follow-up study started: no.
- Canonical review or reward requested: no.

The unresolved real-data enablement decisions are listed in
`../executor/REAL_DATA_GATES.md`.
