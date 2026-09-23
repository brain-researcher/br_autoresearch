# Native Codex Society panel

For Goal review, use this reference only after an immutable Society packet
exists and current next_action routes to panel work. The registered Sydnor
continuation section has its own COMPLETE-state prerequisites. Society is
reward-blind and supplemental in both routes.

## Fixed topology

When native collaboration tools are exposed, use
codex-native-society-topology-v1. The current task is the root conductor and is
not counted as a child.

Spawn exactly ten child tasks in dependency-ordered waves:

1. six primary reviewers;
2. three cross-review or red-team tasks; and
3. one integrator.

Use the exact roles, dependencies, and peer edges published by
brain_researcher.autoresearch.society.codex_native_topology. Do not reconstruct
the topology from memory.

Use spawn_agent, followup_task, and wait_agent. Each child must have:

- task_name equal to its exact role ID;
- model gpt-5.6-sol;
- reasoning_effort xhigh;
- fork_turns none; and
- a message containing its role contract, immutable packet, and only its
  declared dependency memos.

Requested calls express intent, not runtime evidence.

## Observed provenance only

Register and display only observed native state:

- derive agentRole only from the persisted Thread source;
- count a peer edge only from V2 SubAgentActivity whose kind is interacted;
- use the enclosing notification thread ID as sender and the activity child
  thread ID as target;
- verify each child's actual model and reasoning effort through thread/resume;
- report real thread ID, role, parent ID, native status, display phase, model,
  reasoning effort, and completed-turn state;
- report each observed native interaction or scheduling edge.

Respect the client concurrency limit. With a four-slot tree, the root runs no
more than three children concurrently, waits, and reuses freed slots. The final
tree must still contain exactly ten descendants and all three required peer
followup_task routes:

- evidence_cross_reviewer to /root/result_gate_auditor
- methods_scope_cross_reviewer to /root/method_assumptions_auditor
- adversarial_red_team to /root/statistics_spin_auditor

Never invent a child, status transition, memo, or review exchange from the
requested topology.

## Execution surfaces

Keep these surfaces distinct:

- Native children started in the current Codex task belong to its task tree.
- Children started by a GCE or pod-owned codex app-server belong to that remote
  parent.

Remote state may be mirrored into Brain Researcher events, but do not claim the
trees are shared unless both clients are verified to use the same app-server
session.

The native panel is collaboration-tools-only: no shell, file edits, MCP, web,
apps, or compute. The strict zero-tools CodexCLIRouter is a separate typed
closeout layer.

If native collaboration is unavailable or the observed tree is incomplete,
report PANEL_INCOMPLETE. Do not replace missing children with prose personas or
treat panel failure as a scientific null.

## Registered Sydnor continuation

For a completed registered Sydnor episode:

1. Persist the panel at society/codex-native-panel-v1.json.
2. Call sydnor_episode_ingest_native_society_review with only server-issued
   episode_id and frozen source_loop_revision.
3. Require the bridge to revalidate registered adjudication, canonical packet
   binding, and observed native topology.
4. Its automatic canonical roll-forward remains admission_required while no
   registered successor question exists.
5. Continue only through sydnor_episode_admit_next_round using the exact
   scientist-approved next_question, frozen source revision, and
   scientist_confirmation true.

The server derives owner, next-round identities, and the one-candidate
discovery policy. Candidate materialization stops at AWAITING_REWARD. Neither
ingestion nor admission authorizes compute or scientific acceptance.

Never replace the scientist gate with a caller path, panel prose, a
chat-derived question, or another spin run.
