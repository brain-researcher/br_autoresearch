# EP03 adaptive data contract

## Status and source exposure

This contract does not assert that any absent asset has been acquired. The
intended development source is the NeuroEffect/OpenNeuro collection:

| Reported asset | Current knowledge | Exposure classification |
| --- | --- | --- |
| 51 OpenNeuro dataset groups | Remote handoff not yet bound here; identity and independence must be regenerated | Fully exposed by prior corpus-wide development |
| 577 analysis/contrast programs | Repeated observations nested within dataset groups | Fully exposed; never independent sample size |
| approximately 45 GB of statistical maps | Exact paths, bytes, hashes, and map eligibility not yet verified for this episode | Source outcomes previously used in pilots |
| reported 36-dataset / 341-contrast repeatability subset | Lineage and meaning of repeats require authentication | Development/reliability estimation only |
| NeuroEffect/FitLins provenance | Reported for many programs | Exact commit/container coverage unresolved |

Historical reports of approximately 2.2% improvement over a global mean map
and 40--46% reduction in repeated-map disagreement are prior feasibility
signals, not current results and not thresholds to optimize against.

## Immutable development handoff

Before search, bind a read-only, content-addressed handoff under the episode
input path. It must contain:

1. OpenNeuro accession/snapshot, participant and derivative lineage, license,
   and exposure flag for every dataset group;
2. one row per map with group, task, canonical signed contrast, statistic,
   threshold status, sample size, space, affine, mask coverage, smoothness,
   path, byte count, and SHA-256;
3. raw methods/events/contrast excerpts, standardized signed description,
   redacted model input, and provenance;
4. task-family, condition-cluster, duplicate, near-duplicate, and repeatability
   tables made without target performance;
5. exact code commits, containers/lockfiles, and map-generation provenance;
   and
6. a top-level manifest hash binding every object.

Only unthresholded, sign-authenticated group Z maps that can be placed in the
frozen common space and mask are primary eligible. Do not silently mix Z, T,
beta, effect-size, posterior, or thresholded maps.

## Dataset roles

| Role | Source | Allowed use | Freshness |
| --- | --- | --- | --- |
| structural/QC fixtures | synthetic maps, permuted labels, metadata-only tables | implementation, leakage, sign, split, and serialization tests | no neural claim |
| adaptive development and selection | all authenticated members of the historically exposed 51 dataset groups | grouped nested CV, operator search, falsifiers, ablations, influence and reliability analysis | exposed exploratory evidence |
| prospective audit | future independent dataset groups, sealed before any target-map or result-text access | exactly one locked evaluation after search and config lock | potentially fresh only after exposure audit |

No subset of the 51-group source may be promoted to audit. Releases sharing
participants, first-level inputs, or derivative lineage remain one group. A
future audit group may supply methods-only text for locked inference, but its
map, reliability, result text, and model-comparison score remain inaccessible
until audit opening.

## Audit firewall

- Store audit bytes and decryption/access authority outside the adaptive
  worker's readable paths until the configuration-lock receipt exists.
- Maintain separate hashed development and audit manifests and prove no
  participant, derivative, map, semantic-cluster duplicate, or source file
  crosses them.
- Do not use audit metadata distributions to select encoders, ranks,
  nuisance handling, stopping, or thresholds beyond the schema required to
  establish eligibility.
- Permit one deterministic execution of the locked container. Preserve all
  outputs even if it fails. A software failure may be retried only from the
  identical hash-bound specification and may not reveal partial scores.
- After any score is revealed, the audit set is permanently exposed and no
  alternative pipeline may be substituted.

## Missing assets and blockers

- The cross-host 51-group handoff is not provisioned or hash-verified here.
- Exact independent-group, duplicate, sign, statistic, space, task-family,
  semantic-cluster, and license eligibility is unresolved.
- The methods-only redaction audit and frozen text/model cache do not yet
  exist for the episode.
- No prospective independent audit dataset group has been acquired, sealed,
  or registered.
- Audit power and represented-family requirements must be calculated after the
  development inventory; this document does not fabricate a sample count.
- The search policy and executable evaluator are unregistered, so launch is
  blocked even if source bytes exist elsewhere.

## Storage boundary

Large immutable inputs must remain outside Git and be exposed read-only.
Durable manifests, ledgers, code, compact results, and lock receipts belong in
the episode workspace; transient matrices belong under a dedicated
`$SCRATCH/br_autoresearch/episode03_condition_semantics_effect_maps/` path. The
old EP03 directory and any remote live cache
are provenance sources only, never mutable episode storage.
