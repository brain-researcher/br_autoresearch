#!/usr/bin/env python3
"""Prospective trusted EP17 full-development feature CLI; no automatic launch.

The coordinator must first inspect the completed fixed-32 pilot, then select
this new full attempt, actual Slurm memory/time limits and an external timeout.
All caps below are required caller arguments. The pilot's serial-model peak
RSS is not a measurement of this producer's four-resident-model memory use.
No neural/scoring/audit access or automatic reader release follows completion.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import socket
import sys
import time
import zipfile

EPISODE = Path(__file__).resolve().parents[2]
PILOT_REPORT = EPISODE / "outputs/runs/stimulus_features/pilot001/pilot_report.json"
PRIVATE_ROOT = EPISODE / "outputs/cache/private/stimulus_features"
REPORT_ROOT = EPISODE / "outputs/runs/stimulus_features"


class FullFeatureRefusal(ValueError):
    pass


def require(condition):
    if not condition:
        raise FullFeatureRefusal()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def pilot_projection(report):
    """Operational estimate only: same forward counts, different memory order."""
    models = ["DINO_R50", "SWAV_R50_800", "SUP_R50_V1", "RANDOM_R50_S17"]
    require(report.get("status") == "completed_private_staging"
            and report.get("pilot_feature_extraction_complete") is True
            and report.get("archive_content_identity_verified") is True
            and report.get("images_preprocessed") == 32
            and report.get("eligible_development_images") == 2423
            and report.get("completed_models") == models
            and report.get("audit_pixels_or_features_opened") is False
            and report.get("neural_values_read") is False)
    coverage = report["cost_projection_coverage"]
    fixed, variable = (coverage[name] for name in
                       ("fixed_seconds", "variable_seconds_including_residual"))
    require(all(isinstance(value, (float, int)) and not isinstance(value, bool)
                and math.isfinite(value) and value >= 0 for value in (fixed, variable)))
    seconds = 2 * (fixed + 2423 / 32 * variable) + 60
    return {"buffered_seconds_estimate_not_admission": seconds,
            "pilot_peak_rss_kib_not_full_memory_measurement": report["peak_rss_kib"],
            "decode_and_preprocess_count": 2423, "batch1_forward_count": 9692,
            "feature_payload_bytes": 148869120, "backbone_parameter_bytes_before_buffers": 376128512,
            "largest_additional_freeze_copy_bytes": 19849216,
            "pilot_models_resident_together": 1, "full_models_resident_together": 4,
            "full_route_adds_bank_freezing_and_complete_container_persistence": True,
            "both_routes_decode_and_preprocess_once_per_image": True,
            "memory_and_walltime_choices_require_coordinator_admission": True}


class Budget:
    """Observed peak-RSS/time refusals; Slurm and outer timeout bound native work."""
    def __init__(self, deadline_epoch, rss_limit_kib):
        require(math.isfinite(deadline_epoch) and type(rss_limit_kib) is int and rss_limit_kib > 0)
        self.deadline_epoch, self.rss_limit_kib = deadline_epoch, rss_limit_kib
        self.last_operation = "startup"
        self.completed = Counter()

    def __call__(self, event=None):
        self.last_operation = getattr(event, "operation", event or "boundary")
        require(time.time() < self.deadline_epoch)
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= self.rss_limit_kib)
        if getattr(event, "when", None) == "after":
            self.completed[self.last_operation] += 1


def _selected_pilot_report(path):
    path = Path(path)
    require(path.name == "pilot_report.json" and path.parent.parent == REPORT_ROOT
            and re.fullmatch(r"pilot[0-9]{3,6}", path.parent.name) is not None)
    return path


def _read_projection(path=PILOT_REPORT):
    path = _selected_pilot_report(path)
    with path.open("rb") as source:
        content = source.read(256 * 1024 + 1)
    require(len(content) <= 256 * 1024)
    report = json.loads(content)
    require(report.get("analysis_id") == "stimulus_features"
            and report.get("attempt") == path.parent.name)
    return pilot_projection(report)


def _validate_args(args):
    require(bool(os.environ.get("SLURM_JOB_ID")) and "-ln" not in socket.gethostname())
    require(re.fullmatch(r"full[0-9]{3,6}", args.attempt) is not None)
    require(math.isfinite(args.deadline_epoch) and args.deadline_epoch > time.time()
            and args.rss_limit_kib > 0 and args.private_byte_cap > 148869120
            and args.report_byte_cap > 0)
    _selected_pilot_report(args.pilot_report)


def _deliver_and_exec(args, original_args):
    import private_development_feature_io as feature_io
    import run_private_stimulus_feature_pilot as notice_reader

    _read_projection(args.pilot_report)
    feature_io.load_development_metadata()  # Fixed roles before private notice.
    require(not (PRIVATE_ROOT / args.attempt).exists()
            and not (REPORT_ROOT / args.attempt).exists())
    notice = feature_io.archive.ACQUISITION / "terms/password_images.txt"
    with notice.open("rb") as source:
        password = notice_reader.password_from_notice(source.read(notice_reader.NOTICE_BYTES + 1))
    read_fd, write_fd = os.pipe()
    try:
        try:
            require(os.write(write_fd, password) == len(password))
        finally:
            os.close(write_fd)
        del password
        os.set_inheritable(read_fd, True)
        os.execv(sys.executable, [sys.executable, "-B", str(Path(__file__).resolve()),
                                 *original_args, "--private-password-fd", str(read_fd)])
    finally:
        os.close(read_fd)


def _write_report(stream, report, cap):
    content = (json.dumps(report, sort_keys=True, allow_nan=False, indent=2) + "\n").encode("utf-8")
    require(len(content) <= cap)
    stream.seek(0)
    stream.write(content)
    stream.truncate()
    stream.flush()
    os.fsync(stream.fileno())


def _run(args):
    import development_features as producer
    import private_development_feature_io as feature_io
    import private_stimulus_feature_pilot as pilot

    budget = Budget(args.deadline_epoch, args.rss_limit_kib)
    budget()
    projection = _read_projection(args.pilot_report)
    design, manifest = feature_io.load_development_metadata()
    private = PRIVATE_ROOT / args.attempt / "development"
    report_dir = REPORT_ROOT / args.attempt
    require(not private.parent.exists() and not report_dir.exists())
    # Exclusive attempt ownership prevents a duplicate launch mutating history.
    pilot._mkdir_private(report_dir, exclusive=True)
    report_path = report_dir / "feature_report.json"
    started, cpu_started = time.monotonic(), time.process_time()
    report = {"analysis_id": "stimulus_features", "attempt": args.attempt, "status": "started",
        "started_utc": stamp(), "job_id": os.environ["SLURM_JOB_ID"],
        "step_id": os.environ.get("SLURM_STEP_ID"), "host": socket.gethostname(),
        "full_development_features_complete": False, "private_artifacts_releasable": False,
        "reader_handoffs_activated": False, "physical_role_isolation_verified": False,
        "candidate_scoring_authorized": False, "audit_pixels_or_features_opened": False,
        "neural_values_read": False, "selected_member_read_attempted": False,
        "image_pixels_read": False, "model_features_computed": False,
        "development_images": 2423, "development_concepts": 480,
        "model_ids": list(pilot.MODELS), "layers": list(pilot.LAYERS),
        "selected_pilot_report": str(args.pilot_report),
        "pilot_projection": projection,
        "caps": {"deadline_epoch": args.deadline_epoch, "observed_peak_rss_kib": args.rss_limit_kib,
                 "private_bytes": args.private_byte_cap, "report_bytes": args.report_byte_cap},
        "separation": "procedural_same_account_not_enforced_or_blinded"}
    phase = "private_output"
    saved_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}

    def stop(_sig, _frame):
        raise FullFeatureRefusal()

    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0))
    with pilot._exclusive_file(report_path) as report_stream:
        _write_report(report_stream, report, args.report_byte_cap)
        try:
            for sig in saved_handlers:
                signal.signal(sig, stop)
            signal.setitimer(signal.ITIMER_REAL, max(0.001, args.deadline_epoch - time.time()))
            pilot._mkdir_private(private.parent, exclusive=True)
            pilot._mkdir_private(private, exclusive=True)
            budget()
            phase = "archive_identity"
            with feature_io.archive.ARCHIVE.open("rb") as source:
                identity, _directory = pilot._verify_archive(source, feature_io.member_rows(design), budget)
                report["archive_content_identity_verified"] = True
                phase = "private_delivery"
                password = pilot._read_password(args.private_password_fd)
                os.close(args.private_password_fd)
                args.private_password_fd = None
                phase = "full_feature_extraction"
                with zipfile.ZipFile(source, "r", allowZip64=False) as zipped:
                    allowed = frozenset(design.image_ids)

                    def image_check(operation):
                        if operation == "before_member_read" and not report["selected_member_read_attempted"]:
                            report["selected_member_read_attempted"] = True
                            report["image_pixels_read"] = True  # Conservative: set before first possible decode.
                            report["first_image_access_utc"] = stamp()
                            _write_report(report_stream, report, args.report_byte_cap)
                        budget(operation)

                    def check(event):
                        budget(event)
                        if event.operation == "infer_image" and not report["model_features_computed"]:
                            report["model_features_computed"] = True
                            report["first_feature_computation_utc"] = stamp()
                            _write_report(report_stream, report, args.report_byte_cap)

                    def load(image_id):
                        return feature_io.exact_image_loader(zipped, allowed, image_id, password, image_check)

                    try:
                        panels = producer.build_development_features(design, manifest, image_loader=load,
                            random_checkpoint_path=pilot.RANDOM_CHECKPOINT, resource_check=check)
                    finally:
                        del password
                require(pilot._identity(source) == identity)
            require(budget.completed["preprocess_image"] == 2423
                    and budget.completed["infer_image"] == 9692
                    and budget.completed["load_backbone"] == 4)
            phase = "complete_feature_persistence"
            result = feature_io.save_complete_features(private / "features.h5", design, manifest, panels,
                byte_cap=args.private_byte_cap, check=budget)
            report["feature_file_bytes"] = result["bytes"]
            report["partial_cleanup_complete"] = result["partial_cleanup_complete"]
            report["full_development_features_complete"] = True
            report["status"] = "completed_private_staging"
            del panels
        except BaseException:
            report["status"], report["error_code"] = "stopped", phase + "_failed"
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in saved_handlers.items():
                signal.signal(sig, handler)
            if args.private_password_fd is not None:
                os.close(args.private_password_fd)
                args.private_password_fd = None
            report.update(finished_utc=stamp(), elapsed_seconds=time.monotonic() - started,
                process_cpu_seconds=time.process_time() - cpu_started,
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                last_operation=budget.last_operation, completed_operations=dict(budget.completed))
            _write_report(report_stream, report, args.report_byte_cap)
    return 0 if (report["status"] == "completed_private_staging"
                 and report["full_development_features_complete"] is True) else 1


def main(argv=None):
    original_args = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--pilot-report", type=Path, default=PILOT_REPORT,
                        help="Explicit successful pilot report; default remains pilot001, never auto-selected.")
    parser.add_argument("--deadline-epoch", required=True, type=float)
    parser.add_argument("--rss-limit-kib", required=True, type=int)
    parser.add_argument("--private-byte-cap", required=True, type=int)
    parser.add_argument("--report-byte-cap", required=True, type=int)
    parser.add_argument("--private-password-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(original_args)
    try:
        _validate_args(args)
        if args.private_password_fd is None:
            _deliver_and_exec(args, original_args)
            raise FullFeatureRefusal()
        require(args.private_password_fd >= 3)
        code = _run(args)
    except BaseException:
        code = 1
    finally:
        if args.private_password_fd is not None:
            try:
                os.close(args.private_password_fd)
            except OSError:
                pass
    print(json.dumps({"status": "completed" if code == 0 else "stopped",
                      "analysis_id": "stimulus_features", "attempt": args.attempt}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
