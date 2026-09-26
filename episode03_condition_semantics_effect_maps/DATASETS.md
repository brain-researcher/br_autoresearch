# EP03 adaptive data contract

## Status and source exposure

This contract does not assert that any absent asset has been acquired. The
intended development source is the NeuroEffect/OpenNeuro collection:

| Reported asset | Current knowledge | Exposure classification |
| --- | --- | --- |
| 51 OpenNeuro dataset groups | Remote handoff not yet bound here; identity and independence must be regenerated | Fully exposed by prior corpus-wide development |
| 577 analysis/contrast programs | Repeated observations nested within dataset groups | Fully exposed; never independent sample size |
| approximately 45 GB of statistical maps | Source locations, stable release identifiers, and map eligibility not yet verified for this episode | Source outcomes previously used in pilots |
| reported 36-dataset / 341-contrast repeatability subset | Lineage and meaning of repeats require authentication | Development/reliability estimation only |
| NeuroEffect/FitLins provenance | Reported for many programs | Exact commit/container coverage unresolved |

Historical reports of approximately 2.2% improvement over a global mean map
and 40--46% reduction in repeated-map disagreement are prior feasibility
signals, not current results and not thresholds to optimize against.

## Read-only development handoff

Before search, bind a read-only handoff under the episode input path. It must
contain only the information needed to establish scientific eligibility and
run the frozen analysis:

1. OpenNeuro accession/snapshot, participant and derivative lineage, license,
   and exposure flag for every dataset group;
2. one row per map with a deterministic map ID, group, participant and
   derivative lineage, task, canonical signed contrast, target observability
   (`detected`, `verified_absent`, or `unknown`), statistic, threshold status,
   sample size, source release, space, affine, mask coverage, smoothness, and
   path;
3. raw methods/events/contrast excerpts, standardized signed description,
   redacted model input, and provenance;
4. task-family, condition-cluster, duplicate, near-duplicate, and repeatability
   tables made without target performance;
5. stable code, container, and provider/model version identifiers needed to
   execute the frozen pipeline, plus map-generation provenance.

Only unthresholded, sign-authenticated group Z maps that can be placed in the
frozen common space and mask are primary eligible. Do not silently mix Z, T,
beta, effect-size, posterior, or thresholded maps.
`unknown` must not be recoded as `verified_absent`; only `detected` eligible
map targets enter geometry scoring, while verified absences and unknowns are
reported separately.

## Dataset roles

| Role | Source | Allowed use | Freshness |
| --- | --- | --- | --- |
| structural/QC fixtures | synthetic maps, permuted labels, metadata-only tables | implementation, leakage, sign, and split tests | no neural claim |
| adaptive development and selection | all authenticated members of the historically exposed 51 dataset groups | grouped nested CV, operator search, falsifiers, ablations, influence and reliability analysis | exposed exploratory evidence |
| prospective audit | future independent dataset groups, sealed before any target-map or result-text access | exactly one locked evaluation after search and config lock | potentially fresh only after exposure audit |

No subset of the 51-group source may be promoted to audit. Releases sharing
participants, first-level inputs, or derivative lineage remain one group. A
future audit group may supply methods-only text for locked inference, but its
map, reliability, result text, and model-comparison score remain inaccessible
until audit opening.

## Audit firewall

- Store audit bytes and decryption/access authority outside the adaptive
  worker's readable paths until the immutable, write-once configuration lock
  exists.
- Maintain separate development and audit group tables and verify that no
  participant, derivative, map, semantic-cluster duplicate, or source file
  crosses them.
- Do not use audit metadata distributions to select encoders, ranks,
  nuisance handling, stopping, or thresholds beyond the fields required to
  establish eligibility.
- Permit one deterministic execution of the locked container. Preserve all
  outputs even if it fails. A software failure may be retried only from the
  identical pre-outcome lock and may not reveal partial scores.
- After any score is revealed, the audit set is permanently exposed and no
  alternative pipeline may be substituted.

## Missing assets and blockers

- The cross-host 51-group handoff is not provisioned or scientifically
  qualified here.
- Exact independent-group, duplicate, sign, statistic, space, task-family,
  semantic-cluster, and license eligibility is unresolved.
- The methods-only redaction audit and frozen text/model cache do not yet
  exist for the episode.
- No prospective independent audit dataset group has been acquired, sealed,
  or authenticated.
- Audit power and represented-family requirements must be calculated after the
  development inventory; this document does not fabricate a sample count.
- The executable evaluator has not been qualified. This blocks candidate
  scoring and audit opening even if source bytes exist elsewhere; it does not
  block explicit task startup.

## Storage boundary

Large immutable inputs must remain outside Git and be exposed read-only.
Durable group/map tables, ledgers, code, compact results, and the final lock
record belong in the episode workspace; transient matrices belong under a
dedicated `$SCRATCH/br_autoresearch/episode03_condition_semantics_effect_maps/`
path. The old EP03 directory and any remote live cache are provenance sources
only, never mutable episode storage.
