#!/usr/bin/env python3
"""Resolve only pinned event-symlink metadata for EP17's fixed retained roster."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener

import reconcile_released_events as core


OUTPUT_PARENT = core.EPISODE / "outputs/runs/released_event_manifest"
COUNTS = {"sub-01": 213, "sub-02": 213, "sub-03": 213, "sub-06": 194}
ROSTER_COUNT = 833
INTERNAL_SECONDS = 1080
ROOT_TREE_CAP = 512 * 1024
SUBJECT_TREE_CAP = 8 * 1024 * 1024
POINTER_CAP = 512
EVENT_OBJECT_CAP = 128 * 1024
FULL_EVENT_CAP = 64 * 1024 * 1024
POINTER = re.compile(
    r"\.\./\.\./\.\./\.git/annex/objects/(?P<directory>[A-Za-z0-9]{2}/[A-Za-z0-9]{2})/"
    r"(?P<key>MD5E-s(?P<bytes>[0-9]+)--(?P<md5>[0-9a-f]{32})\.tsv)/(?P=key)"
)
SHA = re.compile(r"[0-9a-f]{40}")


def load_roster():
    with (core.METADATA / "metadata_report.json").open(encoding="utf-8") as stream:
        report = json.load(stream)
    core.require(report.get("structural_errors") == [] and
                 report.get("source_roles_complete") is False and
                 report.get("roles_assigned") is False, "invalid_completed_metadata_basis")
    core.require({row["participant"]: row["runs"] for row in report["participants"]} == COUNTS,
                 "retained_roster_scope_changed")
    with (core.METADATA / "run_metadata.csv").open(encoding="utf-8", newline="") as stream:
        raw = list(csv.DictReader(stream))
    roster, seen = [], set()
    for row in raw:
        participant = row["participant"]
        core.require(participant in COUNTS, "unexpected_retained_participant")
        item = {"participant": participant}
        for field in ("session", "run", "run_order", "presentations",
                      "first_glmsingle_trial_index_zero_based", "TR_count"):
            item[field] = core.integer(row[field], field)
        core.require(item["session"] >= 1 and 1 <= item["run"] <= 6 and
                     item["presentations"] == 60 and item["TR_count"] == 188 and
                     core.finite_number(row["TR_seconds"], "TR_seconds") == 1.49,
                     "unexpected_retained_run_structure")
        key = run_key(item)
        core.require(key not in seen, "duplicate_retained_run")
        seen.add(key)
        roster.append(item)
    core.require(len(roster) == ROSTER_COUNT, "retained_roster_count_changed")
    for participant, count in COUNTS.items():
        selected = [row for row in roster if row["participant"] == participant]
        core.require(len(selected) == count and
                     [row["run_order"] for row in selected] == list(range(count)),
                     "retained_run_order_changed")
    return roster


def run_key(row):
    return row["participant"], int(row["session"]), int(row["run"])


def event_path(row):
    participant, session, run = run_key(row)
    return (f"{participant}/ses-{session:02d}/func/"
            f"{participant}_ses-{session:02d}_task-things_run-{run:02d}_events.tsv")


def tree_members(payload):
    try:
        tree = json.loads(payload)
    except (ValueError, UnicodeError):
        raise core.Refusal("invalid_github_tree_json") from None
    core.require(tree.get("truncated") is False and isinstance(tree.get("tree"), list),
                 "incomplete_github_tree")
    members = {}
    for entry in tree["tree"]:
        core.require(entry["path"] not in members, "duplicate_github_tree_path")
        members[entry["path"]] = entry
    return members


def retained_leaf(members, row):
    relative = event_path(row).split("/", 1)[1]
    core.require(relative in members, "retained_event_missing_from_pinned_tree")
    entry = members[relative]
    core.require(entry.get("mode") == "120000" and entry.get("type") == "blob",
                 "retained_event_is_not_annex_symlink")
    core.require(isinstance(entry.get("size"), int) and 0 < entry["size"] <= POINTER_CAP and
                 isinstance(entry.get("sha"), str) and SHA.fullmatch(entry["sha"]),
                 "invalid_pinned_symlink_metadata")
    return entry


def pointer_spec(payload, leaf):
    core.require(len(payload) == leaf["size"], "symlink_length_mismatch")
    blob = b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
    core.require(hashlib.sha1(blob).hexdigest() == leaf["sha"], "pinned_symlink_blob_mismatch")
    try:
        pointer = payload.decode("ascii")
    except UnicodeError:
        raise core.Refusal("nonascii_annex_pointer") from None
    match = POINTER.fullmatch(pointer)
    core.require(match is not None, "unexpected_annex_pointer")
    size = int(match["bytes"])
    core.require(size > 0, "invalid_annex_size")
    return {"directory": match["directory"], "bytes": size, "md5": match["md5"]}


def pilot_pointer(spec):
    key = f"MD5E-s{spec['bytes']}--{spec['md5']}.tsv"
    return f"../../../.git/annex/objects/{spec['directory']}/{key}/{key}".encode("ascii")


def fetch_metadata(url, cap, deadline, counters):
    parts = urlsplit(url)
    core.require(parts.scheme == "https" and parts.hostname in
                 ("api.github.com", "raw.githubusercontent.com") and
                 parts.username is None and parts.password is None,
                 "unexpected_metadata_endpoint")
    core.require(time.monotonic() < deadline, "attempt_deadline")
    counters["requests"] += 1
    request = Request(url, headers={"User-Agent": "EP17-metadata-steward/1", "Accept-Encoding": "identity"})
    try:
        with build_opener(core.NoRedirect()).open(
                request, timeout=min(core.REQUEST_SECONDS, deadline - time.monotonic())) as response:
            core.require(response.status == 200, "unexpected_metadata_http_status")
            declared = response.headers.get("Content-Length")
            if declared is not None:
                core.require(declared.isdigit() and int(declared) <= cap, "metadata_response_too_large")
            core.require(response.headers.get("Content-Encoding", "identity").lower() == "identity",
                         "unexpected_metadata_content_encoding")
            payload = response.read(cap + 1)
            counters["received_bytes"] += len(payload)
    except HTTPError as error:
        error.close()
        raise core.Refusal(f"metadata_http_status_{error.code}") from None
    except (URLError, OSError, TimeoutError):
        raise core.Refusal("bounded_metadata_transfer_failed") from None
    core.require(len(payload) <= cap, "metadata_response_too_large")
    core.require(time.monotonic() < deadline, "attempt_deadline")
    return payload


def resolve_manifest(roster, fetch, deadline, counters, entries=None):
    root_url = f"https://api.github.com/repos/courtois-neuromod/things/git/trees/{core.EVENT_COMMIT}"
    root = tree_members(fetch(root_url, ROOT_TREE_CAP, deadline, counters))
    if entries is None:
        entries = []
    for participant in COUNTS:
        entry = root.get(participant, {})
        core.require(entry.get("type") == "tree" and entry.get("mode") == "040000" and
                     isinstance(entry.get("sha"), str) and SHA.fullmatch(entry["sha"]),
                     "missing_pinned_participant_tree")
        url = f"https://api.github.com/repos/courtois-neuromod/things/git/trees/{entry['sha']}?recursive=1"
        members = tree_members(fetch(url, SUBJECT_TREE_CAP, deadline, counters))
        for row in roster:
            if row["participant"] != participant:
                continue
            leaf = retained_leaf(members, row)
            key = run_key(row)
            reused = key[0] == "sub-01" and key[1] == 1 and key[2] in (1, 2, 3)
            if reused:
                payload = pilot_pointer(core.PILOT[key[2] - 1])
            else:
                url = f"https://raw.githubusercontent.com/courtois-neuromod/things/{core.EVENT_COMMIT}/{event_path(row)}"
                payload = fetch(url, POINTER_CAP, deadline, counters)
            spec = pointer_spec(payload, leaf)
            entries.append({**row, "source_path": event_path(row), "pointer_blob_sha": leaf["sha"],
                            **spec, "reuse_pilot": reused})
    core.require(len(entries) == len(roster), "manifest_scope_incomplete")
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    core.require(bool(os.environ.get("SLURM_JOB_ID")) and "-ln" not in socket.gethostname(),
                 "requires_coordinator_owned_compute")
    out = args.out.resolve()
    core.require(out.parent == OUTPUT_PARENT.resolve() and not out.exists(), "fresh_manifest_attempt_required")
    out.mkdir(parents=True)
    start = time.monotonic()
    report = {"analysis_id": "released_event_reconciliation", "stage": "retained_annex_manifest",
              "events_commit": core.EVENT_COMMIT, "release_commit": core.RELEASE_COMMIT,
              "fmriprep_commit": core.FMRIPREP_COMMIT,
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
              "internal_seconds": INTERNAL_SECONDS, "request_seconds": core.REQUEST_SECONDS,
              "maximum_requests": 835, "expected_retained_runs": ROSTER_COUNT,
              "maximum_response_body_bytes": ROOT_TREE_CAP + 1 + 4 * (SUBJECT_TREE_CAP + 1) + 830 * (POINTER_CAP + 1),
              "requests": 0, "received_bytes": 0, "automatic_retries": 0,
              "event_bodies_requested": False, "roles_assigned": False,
              "candidate_scoring_run": False, "audit_opened": False,
              "complete": False, "entries": [], "errors": []}
    signal.signal(signal.SIGALRM, core.deadline_alarm)
    signal.alarm(INTERNAL_SECONDS)
    try:
        roster = load_roster()
        resolve_manifest(roster, fetch_metadata, start + INTERNAL_SECONDS, report, report["entries"])
        report["complete"] = True
        report["total_event_bytes"] = sum(row["bytes"] for row in report["entries"])
        report["new_event_bytes"] = sum(row["bytes"] for row in report["entries"] if not row["reuse_pilot"])
        report["maximum_event_object_bytes"] = max(row["bytes"] for row in report["entries"])
        report["within_provisional_full_bounds"] = (
            report["total_event_bytes"] <= FULL_EVENT_CAP and
            report["maximum_event_object_bytes"] <= EVENT_OBJECT_CAP)
    except core.Refusal as error:
        report["errors"].append(error.detail)
    except (OSError, ValueError, KeyError, TypeError):
        report["errors"].append({"code": "manifest_metadata_processing_failure"})
    finally:
        signal.alarm(0)
    report["elapsed_seconds"] = time.monotonic() - start
    with (out / "annex_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"stage": report["stage"], "complete": report["complete"],
                      "resolved_runs": len(report["entries"]), "requests": report["requests"]}))
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
