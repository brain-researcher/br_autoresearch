# V3 discovery and frozen exploration trace

Use this reference for a native Goal whose frozen CandidateBundle contract is
V3. Complete the outcome-blind ideation prelude before accessing
candidate-discriminating target outcomes. Dataset and QC inventory that is
outcome-blind, plus already-observed prior-episode phenomenon evidence, may be
used.

The prelude is selection provenance. It is not a separate Goal, Society review,
confirmation, or execution program.

## Outcome-blind ideation

1. Start from the phenomenon, available prior evidence, and outcome-blind
   dataset or QC inventory. Record references and every unavailable or
   unresolved premise.
2. Inspect the live schemas for kg_hypothesis_workflow and
   kg_hypothesis_candidate_cards. Attempt one high-level candidate-ideation call
   permitted by the schema. Do not invent arguments or replace it with a
   lower-level route.
3. If neither tool is exposed, the schema rejects the request, or the call
   fails, record KG ideation as unavailable with the observed reason in
   outputs/idea_search.json. This is non-blocking workspace provenance, not
   evidence for or against a candidate.
4. Screen the canonical fifteen PatternId values exactly once and in this
   order:
   - assumption_audit_and_pivot
   - architectural_operator_substitution
   - generative_process_redesign
   - controlled_diagnostic_design
   - unify_into_shared_representation
   - reframe_as_solvable_object
   - self_supervised_signal_engineering
   - structural_prior_encoding
   - algebraic_equivalence_unification
   - heterogeneous_decomposition
   - decompose_and_delegate
   - relax_discrete_search_to_continuous
   - adapt_via_conditioning
   - characterize_limit_then_surpass
   - targeted_self_supervised_objective
5. Do not repeat this screen after target outcomes, score it after target
   outcomes, or treat an omitted pattern as screened.
6. Record one to three advisory ideation-pattern steps for each candidate.
   official_submode is optional and is allowed only when an actual C00--C30
   card informed formulation and an authoritative mapping to a canonical
   PatternId is available.
7. Otherwise use parent_pattern_only and omit submode_id. Use unknown_pattern
   only when no canonical PatternId fits. Never fabricate C00--C30 lineage.
8. Freeze GoalExplorationTraceV2 and the pre-outcome selection portion of
   CandidateBundleV3 before accessing candidate-discriminating outcomes.

A later terminal record may add observations, but it must not change the frozen
ideation record, candidate set, selected steps, or screen result.

outputs/idea_search.json is client-maintained workspace provenance. It is not a
top-level server schema, authority store, Goal gate, PatternProgram, or eighth
Markdown projection. V3 authority is GoalExplorationTraceV2 embedded in the
CandidateBundle. An unavailable KG attempt does not become server-validated
lineage.

## Exploration trace

Retain the V1 exploration fields:

- Use phenomenon_driven unless the scientist explicitly chose theory_driven or
  instrument_validation.
- Record the phenomenon statement, source, and required evidence references.
- For bounded_ranked_candidates, include at least two competing explanations
  and two contiguous ranked candidates, except for explicit instrument
  validation.
- Bound the set to at most eight candidates.
- For each candidate, record an explanation, discriminating test, falsifier,
  input readiness, estimated cost, selection bit, and pre_outcome_timing.
- Select no more than two candidates, and only candidates whose input readiness
  is ready.
- Mark timing assurance as client_attested, not server-verified
  preregistration.

A registered program may instead freeze registered_program_policy with its
search_policy_ref. It is not an implicit escape to unbounded exploration.

Record novelty_classification, nearest_prior_refs, novelty_delta, and
revision_requirement. A novel_increment must name its nearest prior references
and a concrete delta.

A complete trace finishes with belief_update,
strongest_surviving_alternative, failed_assumptions, proposed_disposition, and
a successor_question for every non-stop disposition. A candidate_ready
terminal_candidate_id must be a pre-outcome selected candidate. A
closed_no_candidate terminal must not name one.

This trace is advisory and client-attested. It neither proves novelty nor
grants a gate.

## Bounded coding-agent exploration

The coding-agent harness may inspect, code, execute, observe, and repair within
the episode boundary. Follow Sherlock storage and compute policy: durable
artifacts go under outputs/, transient intermediates go to the episode scratch
directory, and expensive work goes through Slurm.

Before terminal submission:

1. Maintain the seven files described in
   [workspace projections](workspace_projections.md).
2. Write outputs/candidate_bundle.json using the exact contract frozen by the
   handoff. For an enhanced V3 handoff, use CandidateBundleV3 and
   GoalExplorationTraceV2; never place a V1 or V2 body under a V3 handoff.
3. Parse and validate the JSON locally.
4. Preserve failed attempts, deviations, limitations, reproduction
   information, and the evidence for the terminal status.
5. Do not rewrite the frozen pre-outcome record in response to later results.

An already-frozen V1 or V2 handoff keeps its historical contract, including
GoalExplorationTraceV1 where applicable.

## Usage telemetry

Before closing the native Goal, submit the host's one discovery invocation
through autoresearch_goal_usage_submit:

- use the server-issued goal_handoff_id;
- use a stable invocation_id and idempotency key;
- submit only observed host token, LLM-cost, compute-actual, or reservation
  telemetry;
- never submit caller-asserted owner, episode, run, reward, scientific outcome,
  or authority;
- if a value was not observed, use null and mark it unavailable rather than
  converting it to zero.

If the native Society panel later runs, submit its separate
direction_society_native invocation the same way. Use
autoresearch_goal_usage_get only to inspect coverage, never to infer an absent
charge.

## Route onward

When the bundle has reached a genuine terminal status and parses against the
frozen contract, read
[terminal submission and review](terminal_submission_review.md). A successful
analysis alone does not authorize submission, Society, reward, or confirmation;
those steps remain state-routed.
