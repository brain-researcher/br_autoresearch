#!/usr/bin/env python3
"""Count a fixed private rollout prefix without exporting conversation bodies.

Run on compute. The only output is summary.json; raw staging is temporary
node-local storage. Counts are descriptive event counts, not judgments of
wasted work or human authorship.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile


def message_text(payload):
    return "\n".join(item.get("text", "") for item in payload.get("content", [])
                     if isinstance(item, dict) and item.get("type") in ("input_text", "output_text", "text"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("L_SCRATCH_JOB"):
        raise SystemExit("Use a verified task-owned Slurm compute session.")
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    byte_limit = args.source.stat().st_size
    captured_at = datetime.now(timezone.utc).isoformat()
    counts, events, calls, daily = Counter(), Counter(), Counter(), defaultdict(Counter)
    call_ids, message_ids, objectives = set(), set(), set()
    user_classes = Counter()
    configurations = Counter()
    first_time = last_time = None
    line_no = 0
    malformed = []
    omitted = Counter()
    function_classes = Counter()
    # Copy exactly the observed prefix once; concurrent appends are out of scope.
    with tempfile.TemporaryDirectory(prefix="ep24-history-", dir=os.environ["L_SCRATCH_JOB"]) as stage:
        snapshot = Path(stage) / "rollout.jsonl"
        remaining = byte_limit
        with args.source.open("rb") as src, snapshot.open("wb") as dst:
            while remaining:
                chunk = src.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise RuntimeError("Source shortened during snapshot; analysis abandoned.")
                dst.write(chunk)
                remaining -= len(chunk)
        with snapshot.open("rb") as src:
            for line_no, raw in enumerate(src, 1):
                if not raw.endswith(b"\n"):
                    omitted["incomplete_final_record_bytes"] += len(raw)
                    break
                try:
                    row = json.loads(raw)
                except (ValueError, UnicodeError):
                    malformed.append(line_no)
                    continue
                kind, p = row.get("type", "unknown"), row.get("payload", {})
                counts[kind] += 1
                ts = row.get("timestamp")
                first_time = first_time or ts
                last_time = ts or last_time
                day = (ts or "unknown")[:10]
                daily[day]["records"] += 1
                if kind == "turn_context":
                    policy = p.get("sandbox_policy", {})
                    if isinstance(policy, dict):
                        policy = policy.get("type", "unknown")
                    configurations[(str(p.get("approval_policy", "unknown")), str(policy), str(p.get("model", "unknown")))] += 1
                elif kind == "event_msg":
                    event = p.get("type", "unknown")
                    events[event] += 1
                    daily[day]["event:" + event] += 1
                    # Never serialize event payloads, including free-text fields.
                elif kind == "response_item":
                    item = p.get("type")
                    if item == "message":
                        role, channel = p.get("role"), p.get("channel")
                        if role not in ("user", "assistant") or channel == "analysis":
                            omitted["nonvisible_message"] += 1
                            continue
                        mid = p.get("id")
                        if mid and mid in message_ids:
                            omitted["duplicate_message_id"] += 1
                            continue
                        if mid:
                            message_ids.add(mid)
                        text = message_text(p)
                        if not text:
                            continue
                        if role == "user" and text.lstrip().startswith('<codex_internal_context source="goal">'):
                            match = re.search(r"<objective>(.*?)</objective>", text, re.S)
                            user_classes["goal_continuation"] += 1
                            if match:
                                objectives.add(match.group(1).strip())
                            daily[day]["explicit_goal_continuations"] += 1
                            continue
                        # Environment-only injections are identifiable metadata;
                        # don't infer automatic/human origin from other wording.
                        context_only = text.lstrip().startswith(("<environment_context>", "# AGENTS.md instructions", "<external_codex_apps_open_page>"))
                        if context_only:
                            omitted["context_only_user_message"] += 1
                            user_classes["context_injection"] += 1
                            continue
                        if role == "user":
                            category = "question_reply" if text.lstrip().startswith('<send_user_message_question_reply>') else "other_user_role"
                            user_classes[category] += 1
                        daily[day]["message:" + role] += 1
                    elif item in ("function_call", "custom_tool_call"):
                        cid, name = p.get("call_id"), p.get("name", "unknown")
                        if cid and cid in call_ids:
                            omitted["duplicate_call_id"] += 1
                            continue
                        if cid:
                            call_ids.add(cid)
                        calls[name] += 1
                        daily[day]["tool_calls"] += 1
                        argument = p.get("arguments", p.get("input", ""))
                        if not isinstance(argument, str):
                            argument = json.dumps(argument)
                        for label, pattern in {
                            "scheduler_query": r"\b(?:squeue|sacct|scontrol\s+show\s+job|seff)\b",
                            "scheduler_submit": r"\b(?:sbatch|srun)\b",
                            "documentation_reference": r"(?:AGENTS\.md|SKILL\.md|GOAL\.md|SEARCH_POLICY\.(?:yaml|json))",
                            "validation_reference": r"(?:pytest|unittest|validate|validation|conformance|smoke)",
                            # Historical Chinese approval terms; preserve matching behavior.
                            "approval_reference": r"(?:approval|permission|authorize|confirm|\u6279\u51c6|\u6388\u6743|\u540c\u610f)",
                        }.items():
                            if re.search(pattern, argument, re.I):
                                function_classes[label] += 1
                    elif item not in ("function_call_output", "custom_tool_call_output"):
                        omitted["response_item:" + str(item)] += 1
        summary = dict(source=str(args.source), snapshot_bytes=byte_limit,
                       captured_at_utc=captured_at,
                       first_timestamp=first_time, last_timestamp=last_time,
                       physical_lines_seen=line_no, record_types=dict(counts),
                       event_types=dict(events), unique_tool_calls=dict(calls),
                       argument_keyword_counts=dict(function_classes), daily=dict(daily),
                       configurations=[dict(approval_policy=a, sandbox_policy=s, model=m, contexts=n)
                                       for (a, s, m), n in configurations.items()],
                       user_record_classes=dict(user_classes),
                       distinct_goal_objective_count=len(objectives),
                       malformed_complete_line_numbers=malformed, omitted=dict(omitted),
                       compute_job=os.environ["SLURM_JOB_ID"],
                       cautions=["Keyword counts are overlapping mentions, not executed commands or wasted-work judgments.",
                                 "User-role messages do not by themselves establish human authorship.",
                                 "Session duration does not establish continuous active or unattended execution.",
                                 "Distinct objectives count parsed strings, not human intentions.",
                                 "Conversation bodies, objectives, arguments and tool/event payloads are not exported."])
        (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k: summary[k] for k in ("snapshot_bytes", "first_timestamp", "last_timestamp", "record_types", "event_types", "unique_tool_calls", "malformed_complete_line_numbers")}, indent=2))


if __name__ == "__main__":
    main()
