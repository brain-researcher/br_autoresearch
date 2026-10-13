#!/usr/bin/env python3
"""Diagnose the already released cost001 mapping, without fitting or scoring.

This one selected diagnostic reuses the exact released twelve-development-person
manifest and unchanged complete A/B-primary mapper. It may read those authorized
streams to locate the existing failure, then discards all mapped arrays. It
does not fit a model, compute a scientific summary, publish a partial cohort,
change support or access audit/private-reliability data. No metadata/source
discovery or fallback. The fresh directory contains operational metadata only.

The coordinator runs this in parent47565811, one CPU / 7 GiB, with all numerical
threads one and an outer timeout. Internal caps are at most300s and6GiB RSS;
native work remains bounded by the outer allocation/timeout. Exceptions retain
only their classes and traceback basename/function/line, never response values,
messages, source text or locals. Historical cost001 remains untouched.
"""
import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import resource
import signal
import socket
import time

import run_development_point_panel as panel


MANIFEST = panel.OUTPUTS / "runs/development_release/attempt001/view/point_panel_manifest.json"
DESTINATION = panel.OUTPUTS / "runs/observed_beta_development/cost001_mapping_diagnostic001"


def execute(*, manifest, output_dir, max_seconds, rss_limit_kib):
    panel._compute()
    panel._require(panel._path(manifest) == MANIFEST and panel._path(output_dir) == DESTINATION)
    panel._require(math.isfinite(max_seconds) and 0 < max_seconds <= 300
                   and type(rss_limit_kib) is int and 0 < rss_limit_kib <= 6 * 1024**2)
    panel._require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0))
    # Refuse a duplicate before any diagnostic, input access or prior-file edit.
    output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    output_dir.chmod(0o700)
    started, cpu_started = time.monotonic(), time.process_time()
    deadline = time.time() + max_seconds
    report = dict(status="started", phase="startup", job_id=os.environ["SLURM_JOB_ID"],
        host=socket.gethostname(), started_utc=datetime.now(timezone.utc).isoformat(),
        manifest_path=str(manifest), max_seconds=max_seconds, rss_limit_kib=rss_limit_kib,
        mapping_completed=False, scientific_summary_created=False, model_fitting_started=False,
        audit_values_read=False, private_reliability_read=False,
        diagnostic_scope="existing_released_complete_development_mapping_only",
        stage_elapsed_seconds={})
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    phase_start = started

    def check():
        panel._require(time.time() < deadline
                       and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= rss_limit_kib)

    def stage(name):
        nonlocal phase_start
        now = time.monotonic()
        report["stage_elapsed_seconds"][report["phase"]] = now - phase_start
        report["phase"], phase_start = name, now
        check()

    def stop(_signal, _frame):
        raise panel.ExecutionRefusal()

    try:
        panel._write_new(output_dir / "settings.json", panel._encoded(report, panel.REPORT_BYTES // 2))
        for sig in previous:
            signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, max_seconds)
        stage("released_metadata")
        inputs = panel.load_released_inputs(manifest, check=check)
        from development_handoff_reader import read_development_handoffs
        stage("development_mapping")
        development = read_development_handoffs(**inputs, check=check)
        panel._require(development.qualified_forward_header is not None)
        check()
        del development, inputs
        report.update(status="completed_mapping_diagnostic", mapping_completed=True)
    except BaseException as error:
        report.update(status="stopped", error_code=report["phase"] + "_failed",
                      exception_diagnostic=panel._exception_diagnostic(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        report["stage_elapsed_seconds"][report["phase"]] = time.monotonic() - phase_start
        report.update(elapsed_seconds=time.monotonic() - started,
            process_cpu_seconds=time.process_time() - cpu_started,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        panel._write_new(output_dir / "operational.json", panel._encoded(report, panel.REPORT_BYTES // 2))
    return 0 if report["status"] == "completed_mapping_diagnostic" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-seconds", required=True, type=float)
    parser.add_argument("--rss-limit-kib", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        code = execute(manifest=args.manifest, output_dir=args.output_dir,
                       max_seconds=args.max_seconds, rss_limit_kib=args.rss_limit_kib)
    except BaseException:
        code = 1
    print('{"status":"' + ("completed" if code == 0 else "stopped")
          + '","analysis_id":"cost001_mapping_diagnostic"}', flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
