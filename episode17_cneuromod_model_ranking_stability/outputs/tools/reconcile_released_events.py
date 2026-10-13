#!/usr/bin/env python3
"""Bounded trusted-builder pilot: three pinned released event files, metadata only.

Run only in a coordinator-owned Slurm allocation. Mixed event bytes stay in
memory; only the explicit stimulus/schedule projection reaches outputs.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import signal
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener


EPISODE = Path(__file__).resolve().parents[2]
METADATA = EPISODE / "outputs/runs/source_metadata/attempt003"
OUTPUT_PARENT = EPISODE / "outputs/runs/released_event_reconciliation"
RELEASE_COMMIT = "0f22b6c5eca4001e86fa936f3b6c2569a0179970"
FMRIPREP_COMMIT = "5155ba0d43836a4451141b90b5a70507f41205cf"
EVENT_COMMIT = "b17d3697171824c198ebc6f15fe7ceee9a1f526f"
CONP = ("https://sftp.conp.ca/users/cneuromod/ria-conp/515/"
        "1ad3c-321e-4111-abf0-95fee0dcbb90/annex/objects/")
INTERNAL_SECONDS = 540
REQUEST_SECONDS = 20
# Exact annex symlinks from the pinned official GitHub tree, not payloads.
PILOT = (
    {"run": 1, "directory": "W5/2J", "bytes": 41689,
     "md5": "6bd1c67c8fb1c35739090756ca5827a3"},
    {"run": 2, "directory": "08/W9", "bytes": 42672,
     "md5": "b89843ef07db10d7e6c5f30f94249834"},
    {"run": 3, "directory": "66/24", "bytes": 43414,
     "md5": "32b057f64d9622449095c6f147166736"},
)
CORE_FIELDS = ("subject_id", "session_id", "run_id", "TrialNumber", "order",
               "image_path", "things_category_nr", "category_nr", "exemplar_nr",
               "onset", "duration")
OPTIONAL_FIELDS = ("repetition", "atypical", "not_for_memory", "delay_days",
                   "delay_seconds", "trials_since_lastrep")
SOURCE_FIELDS = CORE_FIELDS + OPTIONAL_FIELDS
OUTPUT_FIELDS = (
    "participant", "session", "run", "run_order", "trial_in_run_zero_based",
    "glmsingle_trial_index_zero_based", "expected_provider_image_path",
    "source_image_basename", "image_path_suffix_matches_expected",
    "things_category_nr", "provider_category_nr", "provider_exemplar_nr",
    "onset_seconds", "duration_seconds", "onset_TR_after_two_volume_removal",
    "source_trial_number", "source_order", "source_repetition",
    "retained_image_presentation_number", "retained_previous_same_image_index_gap",
    "atypical", "not_for_memory", "delay_days", "delay_seconds",
    "trials_since_lastrep", "identity_and_onset_match",
)


class Refusal(Exception):
    """Only fixed codes and selected row/field coordinates, never source values."""

    def __init__(self, code, *, field=None, row=None):
        super().__init__(code)
        self.detail = {"code": code}
        if field is not None:
            self.detail["field"] = field
        if row is not None:
            self.detail["row_zero_based"] = row


def require(condition, code, **coordinates):
    if not condition:
        raise Refusal(code, **coordinates)


def integer(value, field, row=None):
    try:
        number = Decimal(value)
        require(number.is_finite() and number == number.to_integral_value(),
                "noninteger_metadata", field=field, row=row)
        return int(number)
    except (InvalidOperation, ValueError, TypeError, OverflowError):
        raise Refusal("invalid_integer_metadata", field=field, row=row) from None


def finite_number(value, field, row=None):
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        raise Refusal("invalid_numeric_metadata", field=field, row=row) from None
    require(math.isfinite(number), "nonfinite_metadata", field=field, row=row)
    return number


def provider_onset_tr(value):
    """Verbatim numerical rule from pinned GLMsingle_makedesign.py.

    Python int truncates toward zero; do not round, floor, or fit an offset.
    """
    return int((finite_number(value, "onset") - 3.0) / 1.49)


def source_identity(value, prefix, field, row):
    token = value.strip()
    if token.startswith(prefix):
        token = token[len(prefix):]
    return integer(token, field, row)


def optional_value(raw, field, row, unknowns):
    token = raw.get(field)
    if token is None or token.strip().lower() in ("", "n/a", "na", "nan", "none"):
        unknowns.append({"row_zero_based": row, "field": field,
                         "reason": "missing_or_declared_unknown"})
        return None
    if field in ("atypical", "not_for_memory"):
        values = {"true": True, "false": False, "1": True, "0": False}
        require(token.strip().lower() in values, "invalid_schedule_flag", field=field, row=row)
        return values[token.strip().lower()]
    if field in ("repetition", "trials_since_lastrep"):
        value = integer(token, field, row)
        require(value >= (1 if field == "repetition" else 0),
                "invalid_schedule_count", field=field, row=row)
        return value
    value = finite_number(token, field, row)
    require(value >= 0, "negative_repetition_delay", field=field, row=row)
    return value


def parse_projection(payload):
    """Project immediately; behavioral values are never returned or logged.

    TSV decoding necessarily receives mixed rows. This is a trusted-builder
    operation, not a claim that the fetched bytes lack behavioral columns.
    """
    try:
        text = payload.decode("utf-8-sig")
        reader = csv.reader(io.StringIO(text, newline=""), delimiter="\t", strict=True)
        header = next(reader)
        require(len(header) == len(set(header)), "duplicate_event_column")
        require(all(field in header for field in CORE_FIELDS), "missing_core_event_column")
        positions = {field: header.index(field) for field in SOURCE_FIELDS if field in header}
        rows = []
        for i, fields in enumerate(reader):
            require(len(fields) == len(header), "event_row_width_mismatch", row=i)
            rows.append({field: fields[position] for field, position in positions.items()})
        return rows
    except (UnicodeError, csv.Error, StopIteration):
        raise Refusal("invalid_event_tsv") from None


def reconcile(payload, expected, run, participant="sub-01", session=1):
    rows = parse_projection(payload)
    require(len(rows) == len(expected), "event_presentation_count_mismatch")
    projected, mismatches, unknowns = [], [], []
    for index, (raw, previous) in enumerate(zip(rows, expected)):
        require(source_identity(raw["subject_id"], "sub-", "subject_id", index) ==
                int(participant.removeprefix("sub-")),
                "wrong_event_participant", row=index)
        require(source_identity(raw["session_id"], "ses-", "session_id", index) == session,
                "wrong_event_session", row=index)
        require(source_identity(raw["run_id"], "run-", "run_id", index) == run,
                "wrong_event_run", row=index)
        trial_number = integer(raw["TrialNumber"], "TrialNumber", index)
        source_order = integer(raw["order"], "order", index)
        onset = finite_number(raw["onset"], "onset", index)
        duration = finite_number(raw["duration"], "duration", index)
        require(duration > 0, "nonpositive_stimulus_duration", row=index)
        image = PurePosixPath(raw["image_path"])
        require(".." not in image.parts and bool(image.name), "invalid_image_path", row=index)
        target = PurePosixPath(previous["provider_image_path"])
        source_ids = {
            "things_category_nr": integer(raw["things_category_nr"], "things_category_nr", index),
            "provider_category_nr": integer(raw["category_nr"], "category_nr", index),
            "provider_exemplar_nr": integer(raw["exemplar_nr"], "exemplar_nr", index),
        }
        suffix_match = image.parts[-len(target.parts):] == target.parts
        checks = {
            "image_basename": image.name == previous["image_name"],
            "image_path_suffix": suffix_match,
            "TrialNumber": trial_number == index + 1,
            "order": source_order == index,
            "onset_TR": provider_onset_tr(raw["onset"]) ==
                        int(previous["onset_TR_after_two_volume_removal"]),
            **{key: value == int(previous[key]) for key, value in source_ids.items()},
        }
        for field, passed in checks.items():
            if not passed:
                mismatches.append({"row_zero_based": index, "field": field})
        optional = {field: optional_value(raw, field, index, unknowns)
                    for field in OPTIONAL_FIELDS}
        projected.append({
            "participant": participant, "session": session, "run": run,
            "run_order": int(previous["run_order"]),
            "trial_in_run_zero_based": index,
            "glmsingle_trial_index_zero_based": int(previous["glmsingle_trial_index_zero_based"]),
            "expected_provider_image_path": previous["provider_image_path"],
            "source_image_basename": image.name,
            "image_path_suffix_matches_expected": suffix_match,
            **source_ids,
            "onset_seconds": onset, "duration_seconds": duration,
            "onset_TR_after_two_volume_removal": provider_onset_tr(raw["onset"]),
            "source_trial_number": trial_number, "source_order": source_order,
            "source_repetition": optional.pop("repetition"),
            "retained_image_presentation_number": int(previous["image_presentation_number"]),
            "retained_previous_same_image_index_gap": previous["previous_same_image_index_gap"],
            **optional, "identity_and_onset_match": all(checks.values()),
        })
    return projected, mismatches, unknowns


def load_expected():
    """Use only the completed, already public-to-builder metadata attempt."""
    with (METADATA / "metadata_report.json").open(encoding="utf-8") as stream:
        basis = json.load(stream)
    require(basis.get("analysis_id") == "source_metadata" and
            len(basis.get("participants", [])) == 4 and
            basis.get("structural_errors") == [] and
            basis.get("source_roles_complete") is False and
            basis.get("roles_assigned") is False, "unexpected_metadata_basis")
    selected = {run: [] for run in (1, 2, 3)}
    with (METADATA / "run_metadata.csv").open(encoding="utf-8", newline="") as stream:
        roster = [row for row in csv.DictReader(stream)
                  if row["participant"] == "sub-01" and row["session"] == "1"]
    require([int(row["run"]) for row in roster] == [1, 2, 3], "pilot_roster_mismatch")
    with (METADATA / "presentation_identity.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["participant"] == "sub-01" and row["session"] == "1":
                run = int(row["run"])
                require(run in selected, "unexpected_pilot_run")
                selected[run].append(row)
    for run, rows in selected.items():
        record = roster[run - 1]
        require(len(rows) == int(record["presentations"]) == 60, "pilot_roster_row_count")
        require([int(row["trial_in_run_zero_based"]) for row in rows] == list(range(60)),
                "pilot_roster_trial_order")
        require([int(row["glmsingle_trial_index_zero_based"]) for row in rows] ==
                list(range(60 * (run - 1), 60 * run)), "pilot_roster_global_order")
    return selected


class NoRedirect(HTTPRedirectHandler):
    def http_error_302(self, request, fp, code, message, headers):
        # HTTPRedirectHandler's normal redirect path can consume fp.read().
        # Close the stream at the error callback before any body read or URL hop.
        fp.close()
        raise Refusal(f"redirect_status_{code}")

    http_error_301 = http_error_302
    http_error_303 = http_error_302
    http_error_307 = http_error_302
    http_error_308 = http_error_302


def download(spec, deadline, transfer=None):
    """One GET, no redirect/retry/authentication; bounded by the pinned size."""
    key = f"MD5E-s{spec['bytes']}--{spec['md5']}.tsv"
    url = f"{CONP}{spec['directory']}/{key}/{key}"
    remaining = deadline - time.monotonic()
    require(remaining > 0, "attempt_deadline")
    request = Request(url, headers={"User-Agent": "EP17-metadata-steward/1", "Accept-Encoding": "identity"})
    try:
        with build_opener(NoRedirect()).open(request, timeout=min(REQUEST_SECONDS, remaining)) as response:
            require(response.status == 200, "unexpected_http_status")
            declared = response.headers.get("Content-Length")
            require(declared is not None and declared.isdigit() and int(declared) == spec["bytes"],
                    "content_length_mismatch")
            require(response.headers.get("Content-Encoding", "identity").lower() == "identity",
                    "unexpected_content_encoding")
            if transfer is not None:
                transfer["mixed_event_body_read_attempted"] = True
            payload = response.read(spec["bytes"] + 1)
            if transfer is not None:
                transfer["received_bytes"] = len(payload)
    except HTTPError as error:
        error.close()
        raise Refusal(f"http_status_{error.code}") from None
    except (URLError, TimeoutError, OSError):
        raise Refusal("bounded_transfer_failed") from None
    require(time.monotonic() < deadline, "attempt_deadline")
    require(len(payload) == spec["bytes"], "payload_length_mismatch")
    require(hashlib.md5(payload).hexdigest() == spec["md5"], "annex_md5_mismatch")
    if transfer is not None:
        transfer["annex_content_identity_verified"] = True
    return payload


def deadline_alarm(signum, frame):
    raise Refusal("attempt_deadline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    require(bool(os.environ.get("SLURM_JOB_ID")) and "-ln" not in socket.gethostname(),
            "requires_coordinator_owned_compute")
    out = args.out.resolve()
    require(out.parent == OUTPUT_PARENT.resolve() and not out.exists(), "fresh_episode_attempt_required")
    out.mkdir(parents=True)
    start = time.monotonic()
    report = {
        "analysis_id": "source_metadata", "stage": "released_event_reconciliation_pilot",
        "release_commit": RELEASE_COMMIT, "fmriprep_commit": FMRIPREP_COMMIT,
        "events_commit": EVENT_COMMIT, "source_url": f"https://github.com/courtois-neuromod/things/tree/{EVENT_COMMIT}",
        "source_license": "CC0; provider attribution retained",
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
        "selected_participant": "sub-01", "selected_session": 1, "selected_runs": [1, 2, 3],
        "internal_seconds": INTERNAL_SECONDS, "request_seconds": REQUEST_SECONDS,
        "maximum_event_bytes": sum(spec["bytes"] + 1 for spec in PILOT),
        "automatic_retries": 0, "redirects_allowed": False,
        "mixed_event_body_read_attempted": False,
        "behavioral_columns_selected": False, "behavioral_values_persisted_or_reported": False,
        "raw_event_files_written": False, "neural_values_read": False,
        "image_values_read": False, "model_features_read": False,
        "roles_assigned": False, "folds_created": False, "audit_opened": False,
        "candidate_scoring_run": False, "source_roles_complete": False,
        "raw_event_reconciliation_complete": False,
        "selected_pilot_reconciliation_complete": False,
        "repetition_note": "Released repetition and retained-run presentation counts are distinct; no equality/exclusion rule imposed.",
        "runs": [], "errors": [], "status": "running",
    }
    all_rows = []
    signal.signal(signal.SIGALRM, deadline_alarm)
    signal.alarm(INTERNAL_SECONDS)
    try:
        expected = load_expected()
        for spec in PILOT:
            started = time.monotonic()
            run_report = {"run": spec["run"], "annex_bytes": spec["bytes"],
                          "annex_md5": spec["md5"], "status": "started"}
            report["runs"].append(run_report)
            try:
                payload = download(spec, start + INTERNAL_SECONDS, run_report)
                rows, mismatches, unknowns = reconcile(payload, expected[spec["run"]], spec["run"])
                del payload
                all_rows.extend(rows)
                run_report.update({"status": "matched" if not mismatches else "mismatch",
                                   "projected_rows": len(rows), "mismatches": mismatches,
                                   "unknown_metadata": unknowns,
                                   "elapsed_seconds": time.monotonic() - started})
                require(not mismatches, "identity_or_onset_mismatch")
            except Refusal as error:
                run_report.update({"status": "failed", "error": error.detail,
                                   "elapsed_seconds": time.monotonic() - started})
                raise
        report["selected_pilot_reconciliation_complete"] = True
        report["status"] = "completed"
    except Refusal as error:
        report["status"] = "stopped"
        report["errors"].append(error.detail)
    except (OSError, ValueError, TypeError, KeyError):
        # Do not expose source data or filesystem contents through exception text.
        report["status"] = "stopped"
        report["errors"].append({"code": "metadata_or_output_processing_failure"})
    finally:
        signal.alarm(0)
    report["elapsed_seconds"] = time.monotonic() - start
    report["mixed_event_body_read_attempted"] = any(
        run.get("mixed_event_body_read_attempted", False) for run in report["runs"])
    report["projected_rows"] = len(all_rows)
    report["remaining_work"] = [
        "Measured full-roster proposal; no further files selected by this pilot",
        "Prospective repeat-completeness and taxonomy/balance specification",
        "Qualified image/concept universe and role-separated handoffs",
    ]
    with (out / "event_metadata.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    with (out / "reconciliation_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"analysis_id": report["analysis_id"], "status": report["status"],
                      "projected_rows": len(all_rows),
                      "selected_pilot_reconciliation_complete": report["selected_pilot_reconciliation_complete"]}))
    return 0 if report["selected_pilot_reconciliation_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
