# EP03 data: separating task-family recognition from contrast information

The data must allow a question that a familiar-looking map cannot answer:
does a signed comparison described in methods predict another dataset's map
better than knowing only its task family? The number of independent dataset
groups, the support within each family, and the reliability of their maps
matter more than the number of contrast files.

This is a design contract, not a report that the collection has been imported
or reanalyzed. Only the episode's plain-text contracts and local status docs
were read for the 2026-09-30 writing milestone; no map payload or protected
outcome was accessed.

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

In particular, a small gain over a **global mean** does not establish gain over
a **task-family prototype**, the current primary comparator. The reported
repeatability change does not establish the reliability of a family-residual
map or a noise ceiling for semantic prediction. The meaning of those reports
has to be recovered from the original methods before it can inform precision.

## What the source can identify

| Data object | Scientific role | What it does not establish |
| --- | --- | --- |
| Methods/events and signed contrast definitions | Model input and a map-independent family/semantic taxonomy | Neural results, intended psychological mechanism, or causal intervention validity |
| Unthresholded group Z contrast map | Sign-preserving normalized statistical-map geometry | Effect amplitude, an individual's brain state, or causal neural implementation |
| Independent dataset group | Unit of exclusion, equal weighting, uncertainty, and influence | Independence of each contrast, voxel, paper, or derivative |
| Authenticated repeated maps | Development reliability and sensitivity analysis | A fresh audit or interchangeable repeats without participant/processing lineage |
| Prospective independent group | One locked evaluation of transfer after search | Freshness merely from a new filename or another release of the same participants |

Z-map normalization is not an effect-size transformation and does not remove
all processing or sampling differences. A prototype-dominated outcome is
interpretable only with adequate group support and informative uncertainty;
otherwise the source yields an unresolved comparison. The independent-group
inventory and source handoff remain missing, so no such conclusion is now
available.

The 2026-10-02 sign-consistency amendment applies one methods/ontology-defined
canonical comparison ordering to every eligible map and description. With
source-to-canonical polarity `s`, fitting uses `Z*=s Z`; both the family
prototype and prototype-plus-residual prediction are multiplied by `s` on
return to the requested comparison. Reversal changes only that polarity,
not family membership or evidence roles. Canonical ordering cannot use map
values or scores; unresolved ordering remains sign-ineligible. The family
baseline receives the same orientation convention as the candidate. See
[the exact rule](GOAL.md#one-orientation-rule-for-the-baseline-model-and-evaluator).

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

The task-family taxonomy must come from methods and task definitions rather
than target maps. Every prototype, feature vocabulary, residualization,
semantic/map basis, and fitted readout is an outer-training-fold object.
Splitting contrast rows at random, retaining another derivative of a held-out
group, or creating family labels after seeing prediction performance would
answer a different, leaky question. Semantic-cluster exclusion remains a
required falsifier in addition to independent-dataset exclusion.

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
