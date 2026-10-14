#!/usr/bin/env python3
"""Separate goal-trigger metadata from exported user-facing messages.

This corrects the first parser's classification; source history is untouched.
It also preserves exact request counts used in the baseline report.
"""
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys

if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("Run in the existing task-owned compute session.")
root = Path(sys.argv[1])
visible = root / "visible_messages.jsonl"
counts, goals = Counter(), Counter()
messages, requests, runtime = [], [], []
for line in visible.open():
    row = json.loads(line)
    if row["role"] != "user":
        messages.append(row)
        continue
    text = row["text"]
    if text.lstrip().startswith('<codex_internal_context source="goal">'):
        counts["goal_continuation"] += 1
        match = re.search(r"<objective>(.*?)</objective>", text, re.S)
        event = {k: row[k] for k in ("line", "ordinal", "timestamp")}
        event["event"] = "explicit_goal_continuation"
        if match:
            objective = match.group(1).strip()
            goals[objective] += 1
            event["user_provided_objective"] = objective
        runtime.append(event)
        continue
    if row["context_only"]:
        counts["context_injection"] += 1
        continue
    kind = "question_reply" if text.lstrip().startswith('<send_user_message_question_reply>') else "other_user_role"
    counts[kind] += 1
    messages.append(row)
    requests.append({**row, "kind": kind})
if not counts["goal_continuation"]:
    raise SystemExit("Already refined or no goal packets; preserve existing counts.")
temporary = visible.with_suffix(".tmp")
temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in messages))
temporary.replace(visible)
(root / "user_requests.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in requests))
(root / "goal_continuations.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in runtime))
report = dict(source_summary="summary.json", user_record_classes=dict(counts),
              goal_objectives=[dict(record_count=n, objective=t) for t,n in goals.items()],
              refinement="Goal trigger packets and environment-only injections removed from visible messages; only user-provided objectives retained separately.")
(root / "continuation_counts.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(dict(counts), indent=2))
