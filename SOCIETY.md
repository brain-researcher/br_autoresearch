# Brain Researcher Society

Brain Researcher Society is a multi-agent scientific review system. Its job is
to expose different ways a candidate can be wrong before a scientist spends a
confirmation dataset or authorizes a larger experiment.

For Goal-candidate review, the current relationship is:

```text
frozen candidate and evidence
    -> six independent primary reviews
    -> two cross-reviews + one adversarial red team
    -> one advisory integrator
    -> server-side strict eligibility gate
    -> scientist reward and launch decision
```

This is the implemented full panel. It is not required for an episode that ends
with `closed_no_candidate` or `technical_failure`, and it is not evidence that
ten agents are the optimal panel size. The public research loop should remain
simple even when the internal review is detailed.

## What the agents review

The six primary roles look at different failure modes:

| Role | Main question |
| --- | --- |
| Lineage and commitment | Is this the candidate that was actually frozen, and were its commitments made before the decisive result? |
| Result and gate | Did the result pass the gate that the candidate claimed it would pass? |
| Statistics and spin | Are uncertainty, multiplicity, effect size, and null results interpreted honestly? |
| Method assumptions | Which analysis choices or hidden assumptions could produce the result? |
| Neuroimaging validity | Are task, space, acquisition, preprocessing, and measurement claims valid? |
| Scope and overclaim | Does the language go beyond the population, data, or test that was run? |

Two cross-reviewers compare related objections. A red team looks for conflicts,
hidden assumptions, and decisive missing evidence. The integrator summarizes
the disagreements and required changes. It does not authorize anything.

## How it is implemented

The Goal review topology is fixed in
`brain_researcher.autoresearch.society.codex_native_topology`. When native
collaboration is available, a root Codex task creates ten child tasks with
fixed roles and model settings.

Each child produces a typed memo bound to the same `review_packet_id` and
`direction_id`. The host validates role coverage, parent and child identities,
fixed model settings, completed turns, memo bindings, and the three required
peer-review edges. Missing structure becomes `panel_incomplete` rather than
agreement.

The native panel is supplemental. A separate server-side strict router reads a
bounded projection of the frozen evidence and the role-labelled objections. It
is instructed to ignore any client-side claim that a candidate is already
eligible. Only the server route can open `AWAITING_REWARD`, and even that is not
scientific acceptance or permission to execute.

## Information exposure

The intended information contract is staged:

| Stage | Intended input |
| --- | --- |
| Primary reviewer | Frozen packet plus its own role contract, without the other primary memos. |
| Cross-reviewer or red team | Frozen packet plus only the memos from its declared dependencies. |
| Integrator | Frozen packet plus the three cross-review memos, after the required peer challenges. |
| Strict router | Bounded candidate and artifact evidence plus role-labelled objections. |
| Scientist | Persisted review, unresolved objections, and a separate reward and launch gate. |

This design tries to reduce information cascades. Primary reviewers should form
their first judgments without copying one another; downstream agents receive
more information because comparison is their job.

There is an important current limitation: this exposure pattern is an
orchestration and prompt contract. The persisted trace can attest roles, turns,
bindings, and peer interactions, but it cannot prove that a child read,
understood, or causally used every supplied memo. The ten roles also currently
share one model lineage, so role diversity is not the same as independent
model evidence.

## Power and incentives

```text
reviewer / red team -> voice and objection, no decision
integrator          -> synthesis and recommendation, no authority
strict router       -> eligibility gate, no reward or launch
scientist           -> reward, selection, and launch approval
executor            -> computation, no scientific acceptance
result closeout     -> reviewed finding or refusal, no automatic Landscape update
```

This separates information power from action power. The integrator has some
agenda-setting power because it chooses what to emphasize. The strict router
has gatekeeping power. Neither can spend data, run an experiment, or promote a
claim by itself.

The Goal panel is reward-blind. Its reviewers are not rewarded for accepting a
candidate, rejecting it, sounding confident, or agreeing with the integrator.
The scientist rewards the scientific direction only after the review. This
limits the principal-agent problem in which a reviewer advances the work that
benefits itself.

## Credit across episodes

The current ten-agent Goal panel stores role-bound memos, but it does not score
those reviewers against later confirmation outcomes. Its child tasks are fresh
for each panel, so an individual Goal reviewer does not yet accumulate
cross-episode reputation.

Other opt-in parts of the broader Brain Researcher Society contain
system-supplied seat or juror calibration and experimental credit staking.
Those mechanisms are separate from this Goal panel. Calibration is persistent
only when a caller deliberately reuses a path-backed store, and canonical
cross-episode Goal-reviewer scoring and durable market balances do not yet
exist.

For the Goal panel, a future version should use **reviewer calibration**, not a
single prestige score. Before seeing the outcome, a reviewer would make a
typed forecast about a failure mode and cite its evidence. After an externally
settled outcome, the system could update a domain-specific record for:

- calibration and confidently wrong objections;
- material failures caught before confirmation;
- false vetoes and important failures missed;
- evidence quality; and
- whether the proposed discriminating test resolved the disagreement.

The stable unit should be a role, model version, scientific domain, and claim
type, not an ephemeral task name. Scores should retain uncertainty at small
sample sizes and weaken under domain shift.

Credit should change attention, not authority. A well-calibrated reviewer may
receive harder cases or trigger an extra check. Its score must never suppress a
minority memo, approve a candidate, grant reward, launch compute, or create a
finding.

## Does the panel need to be this large?

We do not know yet. The ten-role topology buys independent first-pass reviews,
explicit adversarial challenge, and a visible separation between advice and
authority. It also costs tokens, time, and coordination, and multiple agents
from one model family can repeat the same mistake.

The campaign should compare the full panel with a smaller review on matched
candidates. Useful outcomes include unique material objections found, missed
scientific errors, false blocks, calibration after confirmation, latency, and
token cost. If the cross-review layer adds no new information, it should be
removed. Complexity has to earn its place.

## Non-negotiable rules

1. Review the same frozen candidate and evidence.
2. Keep the panel reward-blind.
3. Treat missing roles or peer edges as incomplete review.
4. Keep the native panel advisory and the server decision separate.
5. Keep eligibility, scientist reward, launch approval, execution, scientific
   acceptance, and Landscape transition as different events.
6. Do not call Society for `closed_no_candidate` or `technical_failure`.
7. Never let reviewer credit turn into self-authorization.
