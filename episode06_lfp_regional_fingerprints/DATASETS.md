# EP06 adaptive data contract

## Status and source identity

This contract does not assert acquisition of absent data. The intended current
source is the public Dryad release for Gallego-Carracedo et al. (2022), DOI
`10.5061/dryad.xd2547dkt`, dataset ID 93309, metadata version 7 / internal ID
194056, with version-5 file payload / internal ID 192175. Historical inventory
reports 24 MATLAB sessions plus a 1,020-byte README (25 objects totaling
9,571,571,204 bytes). Reverify every provider/local byte and digest.

The reference publication code is BeNeuroLab commit
`cbda8e2e6106f5eb5ff98e18a689c595179ac5db`. It is provenance, not proof of a
complete licensed runtime.

## Recording blocks and exact roles

| Block | Reported sessions/regions | Adaptive role |
| --- | --- | --- |
| Mihili | 6, simultaneous M1 + PMd | four deterministic-hash-ranked development/selection sessions; two reserved internal-audit sessions, conditional on eligibility |
| Chewie-L | 6, simultaneous M1 + PMd | four deterministic-hash-ranked development/selection sessions; two reserved internal-audit sessions, conditional on eligibility |
| Chewie-R | 4, M1 only, separate implant | sensitivity only; never a third animal or paired-region audit |
| Han | 5, area 2 | recording-domain specificity only |
| Lando | 3, area 2 | recording-domain specificity only |
| prospective third animal | matched simultaneous M1 + PMd | required sealed external generalization audit; absent |

Within each primary animal, eligible canonical session IDs are ranked by
`SHA256("ep06-adaptive" || animal || canonical_session_id)`. Lowest four are
development/selection and highest two are internal audit. This rule is frozen
before neural outcome access. If fewer than six sessions in either animal pass
structural eligibility, the planned 4+2 design is not silently repaired; stop
for scientist review of power and identifiability without inspecting regional
scores.

Chewie-L and Chewie-R are the same biological animal. Sessions, regions,
electrodes, bands, trials, bins, folds, seeds, and profile elements are not
independent animals.

## Exposure classification

The same release was outcome-exposed in historical EP05 work. Treat every
current-release neural outcome as exposed unless an itemized ledger proves a
narrower fact. The deterministic four-session internal holdout is therefore a
procedural internal audit, not fresh confirmation. Neither a new split nor the
fact that the historical EP06 proposal was never executed resets exposure.

## Required immutable asset pack

Provision a read-only, content-addressed pack containing:

1. exact provider/local manifests, SHA-256 values, versions, and licenses for
   all source objects and code/dependencies;
2. animal, implant, session, array, region, trial, event, direction, physical
   electrode, unit, and feature-guide tables;
3. source-to-derived trial IDs proving simultaneous M1/PMd fields share trials,
   behavior, event times, and time bases;
4. authenticated `bin_size`, result status, `tgtDir`, target/go/movement/end
   indices, velocity, spike, LFP, spike-guide, and LFP-guide fields;
5. per-session structural support for 15 complete-nine-class physical
   electrodes, at least the frozen minimum spike rank, balanced directions,
   and enough whole trials;
6. historical-outcome exposure ledger; deterministic development/internal
   audit session manifest; and per-file access controls; and
7. synthetic fixtures for MATLAB indexing, history direction, smoothing,
   held-out prediction, profile normalization, region permutation, and nulls.

The released LFP matrices are post-processed features, not raw voltage. Raw
phase, PAC, new filtering, coherence, traveling waves, and re-referencing are
not available operators.

## Development, selection, and internal-audit firewall

- Only the eight hash-ranked development sessions may expose regional
  candidate scores during adaptive search.
- Fit all scalers, latents, reliability weights, mappings, alignment, and
  classifiers inside the applicable training trials/sessions/animals.
- Keep simultaneous M1/PMd trials paired, complete trials intact, and session
  estimates separate until animal-level summaries.
- The four internal-audit sessions reside outside the search worker's readable
  outcome path until the configuration-lock receipt exists.
- Reveal all four internal-audit sessions together for one deterministic run;
  hide partial results and permit no candidate/rule change afterward.
- Historical exposure means a firewall can prevent new adaptive leakage but
  cannot make these sessions independently fresh.

## External third-animal confirmation firewall

No qualifying external data are currently identified or acquired. A future,
separately frozen confirmation source must provide simultaneous M1 and PMd
recordings from a third animal under a compatible task and authenticated
feature pipeline. Before any neural outcome access:

- bind animal/task/hardware/session provenance, license, hashes, eligibility,
  trial pairing, guides, and exposure statement;
- keep neural bytes and comparison summaries inaccessible to the completed
  round's search and internal-audit workers;
- freeze how the original two animals form prototypes/reference objects and
  how the third animal is scored; and
- create a new round with its own exposure record and execute the already
  locked policy once, with no representation, alignment, hyperparameter, threshold,
  exclusion, or stopping change for a direct generalization test.

New sessions from Mihili or Chewie cannot satisfy the third-animal requirement.
An exact infrastructure retry is allowed only if no score was released and all
hashes remain unchanged.

## Data needed to explain a transferable fingerprint

The [paper plan](outputs/paper_plan.md) proposes a later fixed comparison. It
does not create another opening of the four internal-audit sessions and does
not make any current-release outcome fresh.

| Source | Proposed role | Condition before use |
| --- | --- | --- |
| Eight hash-ranked development sessions | Retain the undifferenced M1 and PMd prototypes in the exact locked normalized/aligned profile representation that produced the primary signed contrast; derive correct-region and equal-region pooled convex output-mixture weights | Begin only after the primary development result is fixed; keep both cross-animal directions; freeze the prototype support, projection equation, bandwise base-predictor recipe, and margins before target-animal follow-up population-recovery scores |
| Four reserved current-release sessions | In the same single internal-audit transaction, test the locked signed contrast and correct-region versus pooled/swapped prediction if that output was declared before lock | No partial score release, second opening, new alignment, band selection, or threshold change |
| Array/QC/impedance/channel metadata | Test whether non-neural recording properties reproduce the region label or the apparent advantage | Use a frozen metadata-only model and matched-support analysis; never call metadata adjustment proof of pure cortical origin |
| Prospective third animal with simultaneous M1/PMd | Apply the unchanged classifier, signed contrast, and region-specific consequence in a successor round | Bind task, hardware, sessions, trials, guides, region labels, and exposure before outcomes; use no target-driven alignment or selection |

The follow-up target is the held-out spike-population latent already defined
from simultaneous spikes. A signed difference alone cannot recover two
undifferenced regional profiles, so the immutable pack must retain the
source-only M1 prototype, PMd prototype, their signed difference, and their
equal-region pooled prototype on one common band-by-latent support. These are
the exact normalized/aligned profile coordinates supplied to the locked
primary classifier, not a different pre-normalization precursor. The paper plan
gives the deterministic prototype-to-convex-weight equation.

For every `(target session, region, fold)`, fit one target-training latent basis
and outcome scale shared across every band and rule. Then fit one base predictor
per supported band once on its permitted training trials with the same trial
support, electrode budget, mapping family, regularization recipe, and output
dimension. Restore predictions to the shared target-latent units and freeze the
complete held-out tensor. Correct-region, pooled, and swapped rules combine those
same bandwise predictions at the output level; no joint coefficient, intercept,
bandwise scaling, or penalty is learned after the fixed source weights are
applied. The swapped rule exchanges M1 and PMd weight vectors only. Common-band
support, weight/effective-mixture separation margins, a uniform reference, and
the raw-feature unit-rescaling replay are frozen before target follow-up scores.

Every transform that sees neural values—including reliability shrinkage,
latent axes, scaling, band weighting, and any Procrustes map—must be fit on the
permitted source/training data. Held-out target outcomes or scores cannot select bands, resolve
feature-guide ambiguity, align components, choose support, or decide which
sessions to report. A zero or missing band is not evidence of a regional
absence; missingness and unsupported profile mass must be reported.

M1 and PMd are recorded by different arrays. Metadata matching can identify
measured hardware explanations but cannot remove unmeasured region-linked
array effects. Even a third animal recorded with the same region/array layout
repeats that structural confounding. A pure cortical-area claim would require
a design that breaks or independently measures it; it cannot be obtained by
stronger classifier performance in this release.

## Missing assets and blockers

- No adaptive content-addressed source pack is bound here.
- Provider/local hashes, trial pairing, guide semantics, events, electrode and
  spike-rank support, reliability, dependencies, and licenses must be verified.
- The exact deterministic 4+2 session-ID manifest has not been generated from
  an authenticated eligibility inventory.
- The four current-release internal-audit response objects are not yet placed
  behind a permission boundary.
- No third matched M1/PMd animal dataset has been identified, acquired, or
  sealed; population/generalization candidacy is therefore blocked.
- The search evaluator has not been qualified against the frozen policy; this
  blocks scored search and audit opening, not explicit task startup.

## Storage and compute boundary

Large immutable inputs remain outside Git and are exposed read-only. Durable
manifests, code, trial ledger, predictions, lock receipts, and reports belong
in the episode workspace; transient arrays belong in a dedicated
`$SCRATCH/br_autoresearch/episode06_lfp_regional_fingerprints/` path.
Historical EP05/EP06 directories may supply
explicitly declared provenance only and must not be modified.
