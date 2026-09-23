# Dataset Contract — Episode 12

This episode relies only on the already acquired MaleCNS v1.0 flat-connectome
release described below. It creates no new data asset and grants no graph
access. The common role and locking rules are in
[`../ADAPTIVE_SEARCH_PROTOCOL.md`](../ADAPTIVE_SEARCH_PROTOCOL.md).

## Fixed release

| Item | Fixed value |
| --- | --- |
| Shared dataset ID | `flyem_male_cns` |
| Release | `gcs-male-cns-v1.0-flat-connectome` |
| Provider ID | `male-cns:v1.0` |
| Release date | 2026-06-08 |
| Source prefix | `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/` |
| Read-only local source | `/oak/stanford/groups/russpold/data/br_autoresearch_data/flyem_male_cns/gcs-male-cns-v1.0-flat-connectome` |
| License | CC BY 4.0 |
| Paper | Berg et al., *Cell* (2026), `10.1016/j.cell.2026.08.015` |

The release comprises 11 Feather files totaling 31,318,683,398 bytes. Existing
source metadata reports provider sizes/MD5, local SHA-256, and successful Arrow
IPC opening. This episode must reference those immutable records rather than
copying or mutating the shared source.

### Required graph core

| File | Bytes | Provider MD5 | Role |
| --- | ---: | --- | --- |
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 14,483,314 | `50a7718770c57220f160ba4f431ab89e` | node, type, side, anatomy annotations |
| `body-neurotransmitters-male-cns-v1.0.feather` | 43,282,834 | `3d842b12fe5c49eefade528d7dd24a1f` | optional annotation/QC context only |
| `body-stats-male-cns-v1.0-minconf-0.5.feather` | 778,062,826 | `404c3349c28580148e16815eb99f382a` | strength/status support; connectivity-derived |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 1,051,241,946 | `f30e9dcca25cfd021bf1e7b3d975599e` | directed weighted-graph outcome |

The primary outcome uses the full `minconf-0.5` weight table. Colocated
`traced-only` and `significant-only` variants are not primary inputs unless
their semantics are independently authenticated and added before lock. Point
synapse, synapse-partner, and t-bar neurotransmitter tables are not required.

## Exposure history

Schemas and annotation category counts were previously inspected. No
connection-weight, body-stat, synapse, or neurotransmitter values were
reported as inspected for the original question. Existing annotation-only
artifacts include:

- 211,577 annotation rows;
- 164,506 rows with non-null `type` and 11,751 distinct raw type strings;
- provisional left/right counts of 80,785/82,932;
- 1,224 types with at least five provisional left and five right neurons;
- 1,184 at that threshold after rejecting types with any non-lateral row; and
- a conservative scenario with 656 fully lateral types and 13,806 typed rows.

These are feasibility counts, not the eligible cohort or mixture-support
threshold. The broad fallback used in the provisional counts is not the final
sensory-aware side rule. Preserve the existing annotation inventory and first
access history as exposed development metadata.

## Annotation-only 80/20 assignment

Before body stats or graph weights are opened, derive eligible type candidates
using a frozen annotation-only rule. Assign **whole provider types**, not
neurons or edges, 80% to `development` and 20% to `final_sealed` with a fixed
deterministic seed. Any stratification may use only prespecified annotation
scope/support fields and must be recorded before outcomes.

The primary brain stratum excludes prospectively identified optic-column,
retinotopic, sensory-receptor, and serial-homologue families. VNC types are a
separate analysis matched or adjusted within `somaNeuromere`. Medial, unknown,
missing, and left/right-conflict cases follow frozen quarantine/exclusion
rules. A failure to retain adequate bilateral whole-type support stops the
episode; it cannot be repaired by relaxing thresholds after graph access.
Every type assigned to `final_sealed` remains in the audit accounting. A
prospectively frozen graph-support formula may return `unscorable`, but no
replacement final type may be chosen after any final graph value is opened.

## Field restrictions and outcome ownership

Provider `type` defines focal and partner categories, with explicit caveat that
the taxonomy was connectivity-informed. `instance` is side-encoded and
forbidden as a primary category. `group` may encode finer
connectivity-informed distinctions and is forbidden for eligibility,
splitting, initialization, fitting, or model selection. Both are available
only in a post-result novelty audit.

Outgoing profiles own rows by focal source neuron. Incoming profiles own rows
by focal target neuron. Keep each focal neuron and all of its
direction-specific incident counts in its whole-type role. Synapses and edges
are not samples. Untyped, unproofread, fragment, missing-annotation, and
out-of-vocabulary endpoint mass remain explicit; no analysis silently
conditions on typed partners.

During outgoing development, only development-type source-owned profiles may
be materialized. Final-type identities may be used for endpoint-to-type lookup
but their focal profiles, totals, support, and graph-derived summaries remain
sealed. Incoming profiles remain unmaterialized until the outgoing result is
immutable.

## Staged role boundary

| Stage | Allowed | Hidden |
| --- | --- | --- |
| Annotation lock | provider methods and annotation fields; eligibility and 80/20 type assignment | body stats, graph weights, synapses, neurotransmitters |
| Synthetic qualification | generated fixtures only | all real connectivity-derived values |
| Outgoing development | body stats as declared; development-type source-owned rows | final-type focal profiles; all incoming summaries |
| Outgoing lock | development ledger and selected triplet | all final-type outcomes |
| One-shot final evaluation | trusted evaluation of final-type source-owned rows in both directions | final labels/outcomes from controller and candidate |
| Incoming scope | target-owned rows under a separately fixed workflow after outgoing freeze | no claim of independent evidence |

The final evaluator must log first access, prevent search-code label access,
and refuse a second opening. Any premature access invalidates the audit role.

## Launch blockers and required records

The shared source is acquisition-ready, but the original contract reports no
canonical non-BIDS handoff. A future run requires operator-managed read-only
registration for `ds:manual:flyem_male_cns` and an appropriate
`connectome-graph` readiness profile. This contract does not create either.

Required durable records are the immutable source manifest; annotation and
exposure history; neuron/type/endpoint eligibility ledger; 80/20 whole-type
assignment; endpoint-coverage report; outgoing and incoming vocabularies;
trial ledger; frozen T/U/M configurations; every type/direction score; nulls,
component matching and influence results; resource accounting; and the final
lock/audit manifests. High-I/O transforms must remain reconstructable from the
pinned release and durable code.
