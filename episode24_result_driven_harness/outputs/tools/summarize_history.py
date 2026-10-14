#!/usr/bin/env python3
"""Inspect a fixed prefix of a live rollout; never export internal reasoning.

Run on compute. The only raw copy is temporary node-local storage. Counts are
descriptive event counts, not judgments of wasted work or human authorship.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile


def redact(text):
    text = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)((?:api[_-]?key|access[_-]?token|password|secret)\s*[=:]\s*)[^\s,;]+", r"\1[REDACTED]", text)
    return text


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
    call_ids, message_ids = set(), set()
    configurations = Counter()
    first_time = last_time = None
    last_usage = None
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
        with snapshot.open("rb") as src, \
             (args.output / "visible_messages.jsonl").open("w") as messages, \
             (args.output / "actions.jsonl").open("w") as actions, \
             (args.output / "runtime_events.jsonl").open("w") as runtime:
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
                base = dict(line=line_no, ordinal=row.get("ordinal"), timestamp=ts)
                if kind == "turn_context":
                    policy = p.get("sandbox_policy", {})
                    if isinstance(policy, dict):
                        policy = policy.get("type", "unknown")
                    configurations[(str(p.get("approval_policy", "unknown")), str(policy), str(p.get("model", "unknown")))] += 1
                elif kind == "event_msg":
                    event = p.get("type", "unknown")
                    events[event] += 1
                    daily[day]["event:" + event] += 1
                    if event == "token_count":
                        last_usage = (p.get("info") or {}).get("total_token_usage", last_usage)
                    # Only explicit runtime fields: no system prompts/reasoning.
                    if event not in ("agent_message", "agent_reasoning", "user_message", "token_count"):
                        out = {**base, "event": event}
                        for key in ("turn_id", "status", "reason", "trigger", "source", "duration_ms"):
                            value = p.get(key)
                            if isinstance(value, (str, int, float, bool)):
                                out[key] = redact(str(value))[:500]
                        runtime.write(json.dumps(out, ensure_ascii=False) + "\n")
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
                            out = {**base, "event": "explicit_goal_continuation"}
                            if match:
                                out["user_provided_objective"] = redact(match.group(1).strip())
                            runtime.write(json.dumps(out, ensure_ascii=False) + "\n")
                            daily[day]["explicit_goal_continuations"] += 1
                            continue
                        # Environment-only injections are identifiable metadata;
                        # don't infer automatic/human origin from other wording.
                        context_only = text.lstrip().startswith(("<environment_context>", "# AGENTS.md instructions", "<external_codex_apps_open_page>"))
                        if context_only:
                            omitted["context_only_user_message"] += 1
                            continue
                        daily[day]["message:" + role] += 1
                        out = {**base, "role": role, "channel": channel,
                               "phase": p.get("phase"), "context_only": context_only,
                               "original_chars": len(text), "text": redact(text[:16000])}
                        messages.write(json.dumps(out, ensure_ascii=False) + "\n")
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
                        classes = []
                        for label, pattern in {
                            "scheduler_query": r"\b(?:squeue|sacct|scontrol\s+show\s+job|seff)\b",
                            "scheduler_submit": r"\b(?:sbatch|srun)\b",
                            "documentation_reference": r"(?:AGENTS\.md|SKILL\.md|GOAL\.md|SEARCH_POLICY\.(?:yaml|json))",
                            "validation_reference": r"(?:pytest|unittest|validate|validation|conformance|smoke)",
                            "approval_reference": r"(?:approval|permission|authorize|confirm|批准|授权|同意)",
                        }.items():
                            if re.search(pattern, argument, re.I):
                                classes.append(label)
                                function_classes[label] += 1
                        out = {**base, "call_id": cid, "tool": name, "classes": classes,
                               "argument_chars": len(argument),
                               "episode_mentions": sorted(set(re.findall(r"(?:episode|EP)(\d{2})", argument, re.I)))}
                        # Questions and goal mutations are externally relevant.
                        # Other arguments may embed source, prompts, or raw data.
                        if any(t in name for t in ("request_user_input", "create_goal", "update_goal")):
                            out["visible_request"] = redact(argument[:16000])
                        actions.write(json.dumps(out, ensure_ascii=False) + "\n")
                    elif item in ("function_call_output", "custom_tool_call_output"):
                        output = p.get("output", "")
                        if isinstance(output, str):
                            job_ids = re.findall(r"Submitted batch job (\d+)", output)
                            if job_ids:
                                runtime.write(json.dumps({**base, "event": "observed_batch_submission", "job_ids": job_ids}) + "\n")
                    else:
                        omitted["response_item:" + str(item)] += 1
        summary = dict(source=str(args.source), snapshot_bytes=byte_limit,
                       captured_at_utc=captured_at,
                       first_timestamp=first_time, last_timestamp=last_time,
                       physical_lines_seen=line_no, record_types=dict(counts),
                       event_types=dict(events), unique_tool_calls=dict(calls),
                       argument_keyword_counts=dict(function_classes), daily=dict(daily),
                       configurations=[dict(approval_policy=a, sandbox_policy=s, model=m, contexts=n)
                                       for (a, s, m), n in configurations.items()],
                       last_reported_total_token_usage=last_usage,
                       malformed_complete_line_numbers=malformed, omitted=dict(omitted),
                       compute_job=os.environ["SLURM_JOB_ID"],
                       cautions=["Keyword counts are overlapping mentions, not executed commands or wasted-work judgments.",
                                 "User-role messages do not by themselves establish human authorship.",
                                 "Session duration does not establish continuous active or unattended execution.",
                                 "Messages are capped at 16000 characters; selected source lines may need a targeted reread.",
                                 "Internal reasoning, system/developer instructions, general tool outputs and raw arguments are not exported.",
                                 "Token usage is the final reported cumulative snapshot, not summed across events or priced."])
        (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k: summary[k] for k in ("snapshot_bytes", "first_timestamp", "last_timestamp", "record_types", "event_types", "unique_tool_calls", "malformed_complete_line_numbers")}, indent=2))


if __name__ == "__main__":
    main()
