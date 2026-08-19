# Finding a Reliable and Transferable N-Back Signature

## The question

Can we find a whole-brain response signature that reliably distinguishes
2-back from 0-back, remains stable in an independent session, and transfers to
a task-compatible cohort without mainly reflecting motion, stimulus category,
site, or general task difficulty?

The object of study is a **measurement**, not a mechanism. A successful result
would define a reproducible mapping from task-fMRI data to a working-memory
load-related score, together with the conditions under which that mapping
holds or fails. It would not establish a general working-memory network, a
clinical biomarker, or a causal theory of working memory.

## Why this is worth testing

The usual 2-back minus 0-back contrast is easy to compute but hard to
interpret. It changes load, attention, vigilance, response demands, errors,
reaction time, and sometimes stimulus composition at the same time. A highly
reliable signature can therefore be reliably wrong.

The scientific opportunity is to search for an assay that improves several
properties together, rather than maximizing a single statistic. If that fails,
the failure should reveal which part of the measurement is tied to a session,
task, category, cohort, scanner, or site.

## Starting point

Begin with an outcome-blind inventory of the candidate HCP-YA, AOMIC, and NDA
resources described in `DATASETS.md`. Determine whether they can support four
non-overlapping roles:

1. discovery and model selection;
2. a genuine same-subject independent session or retest;
3. a task-compatible external transport test; and
4. a defensible control for general task difficulty.

Two phase-encoding runs from one visit count as run-to-run replication, not an
independent session. Two datasets do not become compatible merely because both
use the term `n-back`.

Do not inspect confirmation outcomes while assigning these roles. If a true
retest, compatible external cohort, or useful difficulty control cannot be
identified, report that limitation rather than weakening the claim after the
fact.

## What autoresearch should explore

Search for an assay, not for a significant map. Plausible choices include:

- how the high-load versus low-load response is formulated and whether
  stimulus categories are balanced;
- canonical HRF, derivative, or bounded FIR models;
- voxelwise, parcel, network-summary, or fixed pattern-expression
  representations;
- univariate or multivariate aggregation and regularization;
- motion, censoring, physiological, and other nuisance treatment;
- run/session aggregation and reliability estimation; and
- a predeclared FPN-connectivity comparator derived from the nearest prior BR
  direction.

The search must be structured and bounded. Use no more than two rounds and 36
total configurations. Round 1 should cover a predeclared subset of the space;
Round 2 may refine only the non-dominated region using a rule written before
Round-1 outcomes are inspected. Do not add a new axis or threshold to rescue a
weak result.

The campaign runner may use relevant Landscape, Self-evolvable Questions, the
fifteen research moves, or OOD-KG context while formulating candidates. That
context is optional and does not count as evidence. The runner is responsible
for preserving the ideation trace, reflections, lifecycle documents, and any
eligible `candidate_ready` Society handoff; negative and technical terminal
states do not enter Society. This Goal does not duplicate those protocol
details.

## How candidates are judged

Evaluate every configuration with a reward vector, not one score:

1. run-to-run and, when genuinely available in development data, test-retest
   reliability;
2. held-out 2-back versus 0-back discrimination;
3. a predeclared held-out relationship with a working-memory behavior measure;
4. margins against motion, stimulus category, label permutation, general
   difficulty, and site where site can be tested;
5. robustness to the frozen nuisance and exclusion alternatives; and
6. compute cost, missing-data burden, and implementation complexity.

Use subject- and group-aware out-of-fold estimates with uncertainty. Freeze
the practical margins and pass/fail rules before looking at discovery results.
Report the Pareto frontier. A strong coordinate cannot compensate for a failed
negative control, and an unavailable test is not a pass.

Compare candidates with two locked baselines:

- a simple, interpretable ROI/network summary;
- a conventional whole-brain 2-back minus 0-back representation.

A nominated candidate must weakly improve or match both baselines on every
indispensable dimension, strictly improve at least one predeclared dimension,
and avoid a meaningful loss of construct validity. If several remain, use a
predeclared tie-breaker favoring fewer fitted degrees of freedom, wider control
margins, lower cost, and simpler cross-dataset implementation.

## Confirmation firewall

Before opening independent-session or external-cohort outcomes, lock one assay:

- preprocessing, mask/atlas, contrast/HRF, feature order, and nuisance model;
- all learned parameters and score definition;
- subject/session eligibility and exclusions;
- behavior endpoint and covariates;
- metrics, practical margins, uncertainty method, and failure rules; and
- a one-shot execution manifest.

Confirmation permits no target-cohort feature selection, retraining,
recalibration, threshold choice, or replacement by another Pareto-frontier
member. Discovery can prepare this protocol, but Society review, scientist
reward, confirmation approval, and confirmation execution remain separate
steps.

## What would count as a useful terminal result

### Candidate ready

Nominate one assay only if it passes every evaluable discovery gate, survives
the negative controls, exceeds the locked baselines under the frozen rules, and
has eligible sealed sources for the mandatory retest and transport tests.

The discovery claim is limited to a **provisional n-back load signature ready
for independent confirmation**. It does not claim that retest or transport has
already succeeded.

### Closed without a candidate

Stop after the bounded search if no assay passes all hard gates, or if any of
the following is true:

- the discovery gain disappears in a properly held-out split;
- reliability improvement is mainly explained by motion, category, site, or
  numerical instability;
- reliability rises while construct validity falls beyond the frozen margin;
- no candidate stably exceeds both locked baselines;
- no genuine retest, compatible external cohort, or useful difficulty control
  can be secured; or
- keeping a candidate alive would require a post-hoc threshold, new search
  axis, or confirmation outcome.

A well-localized failure is still a useful result. Distinguish a scientific
failure from a technical failure such as inaccessible data, broken software,
or exhausted storage.

## Expected outputs

The run should leave:

- an outcome-blind data and compatibility inventory;
- the frozen data roles, search space, metrics, baselines, and stop rules;
- a complete search ledger with failures and resource use;
- a readable Pareto frontier and round-by-round reflection;
- `RESULT.md` explaining what was tested and what survived;
- when warranted, one locked signature and a one-shot confirmation protocol;
- the terminal `candidate_bundle.json` and the runner-maintained lifecycle
  documents.

All new artifacts belong under `outputs/`. Treat `inputs/` and the external OAK
data roots as read-only. Use
`$SCRATCH/autoresearch/episode03_nback_signature/` for transient work and Slurm
for expensive computation.
