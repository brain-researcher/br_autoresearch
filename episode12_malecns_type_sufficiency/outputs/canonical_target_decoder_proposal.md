# Proposed canonical-target decoder for corrected outgoing development

**Status: ADOPTED prospectively, 2026-10-20.** The renewed animal-question
instruction and standing scientist authority adopt the source-training count
decoder specified below, with 0.5 per canonical category and per-fold refitting.
Scoring/CV, fitted-null sampling and shared observed/null controls are repaired;
14 generated-only tests passed. Readiness remains false pending affected-procedure
qualification and legacy procedure-lock/launch bindings. No old qualification is
inherited, no original comparison is declared complete, and no estimand,
threshold, data role, circuit endpoint or incoming/final opening changes.
The prior remains category-count dependent and within-pool sensitivity remains
limited. See the [prospective adoption and test record](experiment_log.md#2026-10-20--canonical-scoring-repair--prospective-decoder-adopted).

**Historical proposal status, 2026-10-04: UNADOPTED.** The text below records
the reviewed proposal at that date; its old pending-approval and clock language
does not override the October 20 engineering adoption. This was a separately prepared scientific
proposal, not an operative contract, implementation, qualification result or
launch decision. It does not change the running generated chain or renew the
observed-search deadline of 2026-10-01 05:57:34 UTC. Any later execution needs
both a resolved scientific procedure and an explicit resource amendment that
preserves consumed costs. Final focal connectivity remains closed.

## Problem and recommended choice

V3 requires every candidate to predict and score the same canonical outgoing
partner-count vector. The current empirical runner pools partner columns and
scores the pooled counts directly. A decoder is missing; neither V3 nor the
method freeze specifies its conditional probabilities.

Recommend a fixed, model-independent conditional-composition decoder estimated
from each transfer source's training rows. It restores canonical predictions
and scores while retaining the declared search over internal partner pooling.
Its smoothing, weighting and empty-pool behavior are prospective choices.
They must not be presented as values already authorized by V3.

For canonical category j in internal pool g, with source training rows A,

    c_j = sum over i in A of y_ij
    r(j | g, A) = (c_j + 0.5) / (sum over k in g of c_k + 0.5 * |g|)
    p_model,j(z) = q_model,g(z) * r(j | g, A)

Here q is the model's internal pooled probability at latent state z. The
decoder r is shared by T, U and M for the same training rows and is independent
of z. The proposed decoder pseudocount is **0.5 per canonical category**,
fixed across all search configurations. It is separate from the grammar's
`composition_pseudocount`; matching the numerical value of V3's fit
pseudocount does not make this a previously frozen decoder rule.

This choice gives a pool with many categories more total prior mass
(0.5 times its category count). That is a material modeling assumption, not an
implementation detail. Source counts weight decoder fitting, so high-total
neurons contribute more to r; this does not change the equal-neuron scoring or
make synapses biological replicates.

## Fixed target and internal pooling

The canonical dictionary consists of all annotation-defined `typed::<type>`
categories, individually, followed by the six explicit semantic endpoints:
unknown, untyped, fragment, proofreading, missing annotation, and out of
vocabulary. It is fixed before replay and is common to every configuration.
Known partner identities do not authorize final-type focal outcome access.

The current materializer already constructs its raw vocabulary from all
annotation-defined partner keys plus `missing_annotation`, rather than only
nonzero observed edges (`executor/ep12_executor/development_data.py`,
`_partner_key_table` and `materialize_development_view`). Reuse that annotation
namespace and the existing deterministic endpoint mapping. Multiple raw
status/statusLabel keys can map into one semantic endpoint; they are not new
canonical prediction targets. Unrecognized keys route to the fixed
out-of-vocabulary endpoint, never to a category discovered from final outcomes.

`other_typed` and hierarchy categories are internal pools, not replacements
for the individually defined canonical known types. Preserve the existing
development-derived support/mass/hierarchy partition rule and its fitting
scope for each configuration. Identical development inputs and configurations
can reuse the same partition; this does not freeze observed outcome-derived
memberships into a generated null search. The rule uses all development focal
rows before reciprocal side slicing. A
source-trained decoder does not make that global preprocessing source-only;
that prior development exposure must remain explicit. Changing the partition's
fitting scope would require a separate decision.

Remove an internal group only when its canonical preimage is structurally
empty, and do so before fitting every member of the triplet. The current
helper can retain an empty `other_typed` cell in the hierarchical branch.
Positive probability in a group with no canonical children cannot be decoded
while conserving mass. This removal changes model fitting and is not a
score-preserving relabeling.

Retain groups with canonical children even when all their source counts are
zero: the proposed smoothing gives a uniform conditional composition. Retain
all six canonical endpoints at zero observed mass. Singleton groups have
r = 1. Neither source-zero categories nor unseen target categories justify
discarding counts or renormalizing to typed mass.

## Fitting, scoring and selection

Fit r separately for each provider type and each transfer source hemisphere.
Inside source CV, recompute it using that fold's training rows only. Reuse r
among candidates sharing a training subset. Do not estimate it from validation
rows or the transfer target. Preserve existing family-specific fold seeds,
candidate ordering, standard-error selection, reference selection and
null-generator preference; do not change these rules to force invariance.

For canonical target counts y with original positive total N, score

    s_model(y) = log[sum over z of w_model(z)
                         * Multinomial(y; N, p_model(z))] / N.

Use the existing model-specific latent support and integration rules. Preserve
all counts, N and endpoint mass. Aggregate neuron scores equally within type
and direction, then use the existing equal-type/equal-direction rule. Existing
zero-total and source-support accounting remains unchanged.

For pooled counts Y_g = sum over j in g of y_j, this score equals the pooled
score plus

    [sum_g log(Y_g!) - sum_j log(y_j!) + sum_j y_j * log r(j | g, A)] / N.

The correction factors outside latent integration. It cancels in a fixed
held-target M-minus-reference contrast when the fitted models use the same
decoder. It does not prove end-to-end selection invariance: folds differ
between families, fold corrections vary, and selection uses fold-score
standard errors. Canonical scores must therefore enter source CV and the
T/U reference and null-generator decisions, not merely the final result table.

Any fitted null must generate canonical probabilities with the same decoder.
Each complete null search must recompute outcome-dependent support/mass
partitions from its own generated development data under the same search
rules, then refit its own decoders under the identical training rules. The
annotation hierarchy and canonical dictionary remain fixed. Reusing the
observed data's selected pool memberships across all nulls would omit part of
the required complete search; storing only a pooled generator is likewise
insufficient. This proposal does not authorize a null launch or alter its
99-search requirement.

## Scientific limits and alternatives

Every competing model has the same within-pool conditional composition.
Canonical scoring restores the common target; it does **not** recover group
differences confined within a pool. Pooling can therefore limit sensitivity.
Existing adequacy requirements still apply: a missed within-pool alternative
cannot be turned into an adequacy conclusion by adding a decoder.

| Alternative | Benefit | Tradeoff |
| --- | --- | --- |
| Fixed uniform within each pool | No estimated decoder parameters | Ignores unequal source partner frequencies |
| Neuron-normalized source composition | Reduces high-total-neuron influence on fitting | Changes the fitting objective and needs an explicitly chosen smoothing scale |
| Fixed total prior mass per pool | Avoids total prior strength growing with pool size | Is a different smoothing model from the recommended per-category rule |

The recommended count-based, per-category-smoothed rule is a concrete choice
for review, not a claim that alternatives are scientifically invalid or that
it will improve observed performance. No outcomes were opened to select it.

## Qualification and later execution boundary

The live oracle and generated-method qualifier pass a fixed 11-column
synthetic target directly to `scientific_models.reciprocal_transfer`. They do
not call the empirical vocabulary collapse or this decoder. Their synthetic
`other_typed` column is a generated category, not an empirical pool with
canonical children. An eventual pass retains its existing fixed-procedure
meaning; it does not automatically qualify this new empirical pipeline.

The later validation obligation is specific to the changed boundary:
canonical mapping and total conservation; the likelihood factorization;
exclusion of validation/target rows from decoder fitting; structurally empty
versus zero-observed support; canonical-score CV/reference selection; and
canonical fitted-null parity. Scientific sensitivity to distinctions hidden
inside a pool also remains subject to the existing adequacy framework.
These are proposed obligations, not executed checks or permission for a new
sweep. Identity/singleton cases may preserve existing generated results, but
that alone cannot establish equivalence for pooled cases.

If the entire finite fitted/CV/selection/scoring procedure is not shown
equivalent, affected qualification must cover the corrected procedure using
the frozen scenario and acceptance framework. Decide the exact affected scope
and measured resource request after the existing chain's result. Neither a
blanket reuse nor a blanket rerun is adopted here.

Implementation, if authorized later, belongs in the separately reviewed
isolated package rather than files imported by the live jobs. The corrected
18+9+9 replay must regenerate adaptive successors from corrected outcomes.
Historical nonconforming-U runs remain attempts. The existing latent-rank cap,
scientific margins, final access restrictions and consumed budgets remain
unchanged. No circuit-round execution follows from this proposal.

## Evidence and review

This proposal was prepared from existing contracts and code only, with
independent decoder/selection and generated-qualification-path reviews.
Relevant sources are `executor/SCIENTIFIC_CONTRACT_POST36_V3.yaml`,
`prelaunch/METHOD_FREEZE.md`, `../SEARCH_POLICY.yaml`, and the executor modules
`development_data.py`, `development_trial.py`, `scientific_models.py`,
`scientific_synthetic.py`, `scientific_qualification.py` and
`coverage_successor.py`. No scientific payload, historical numerical report,
live aggregate or SCRATCH log was read; no code, job or operative contract was
changed. Final review is recorded in the execution log.
