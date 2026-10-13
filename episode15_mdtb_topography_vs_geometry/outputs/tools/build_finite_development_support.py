#!/usr/bin/env python3
"""Construct the adopted complete9024-map availability intersection once.

The released reader retains its complete identity/header/role checks. Only the
process-local sampler and assembly are replaced: strict validity is intersected
at original208 locations, and no numeric cohort or fitted score is returned.
Every selected map is visited even if the intersection becomes empty.
"""
import argparse
from datetime import datetime, timezone
import io
import os
from pathlib import Path
import resource
import signal
import socket
import time

import run_development_point_panel as panel

MANIFEST = panel.OUTPUTS / "runs/development_release/attempt001/view/point_panel_manifest.json"
DESTINATION = panel.OUTPUTS / "runs/observed_beta_development/finite_support001"


class IntersectionComplete(Exception):
    pass


def execute(*, manifest, output_dir, max_seconds=595, rss_limit_kib=6 * 1024**2):
    panel._compute()
    panel._require(panel._path(manifest) == MANIFEST and panel._path(output_dir) == DESTINATION)
    panel._require(0 < max_seconds <= 595 and 0 < rss_limit_kib <= 6 * 1024**2)
    panel._require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0))
    import numpy as np
    import development_handoff_reader as reader
    from build_anatomical_support import read_image, strict_trilinear, world_to_voxel
    from pilot_task_a_mapping import validate_beta_reference

    output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    output_dir.chmod(0o700)
    started, cpu_started = time.monotonic(), time.process_time()
    deadline = time.time() + max_seconds
    report = dict(status="started", phase="released_metadata", job_id=os.environ["SLURM_JOB_ID"],
        host=socket.gethostname(), started_utc=datetime.now(timezone.utc).isoformat(),
        mask_complete=False, completed_maps=0, expected_maps=9024,
        original_support_locations=208, retained_locations=None,
        all_participants_conditions_runs_and_regressors_retained=True,
        beta_magnitudes_or_scores_used_for_selection=False, original_support_modified=False,
        audit_values_read=False, private_reliability_read=False, numeric_cohort_returned=False,
        max_seconds=max_seconds, rss_limit_kib=rss_limit_kib)
    keep = np.ones(208, dtype=bool)
    original_ids = None
    old_sample, old_assemble = reader.sample_beta, reader._assemble
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}

    def check():
        panel._require(time.time() < deadline and
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= rss_limit_kib)

    def stop(_signal, _frame):
        raise panel.ExecutionRefusal()

    def validity_sample(archive, member, header, mask_header, native_world):
        check()
        validate_beta_reference(header, mask_header)
        image = read_image(archive, member, header)
        values, valid = strict_trilinear(image, world_to_voxel(image["affine"], native_world))
        panel._require(valid.shape == (208,) and values.shape == (208,))
        result = valid & np.isfinite(values)
        check()
        return result, image["payload_bytes"]

    def intersect(plan, voxel_ids, sample, resource_check):
        nonlocal original_ids
        panel._require(len(voxel_ids) == 208 and
            sum(len(plan[pid, session]) for pid in reader.DEVELOPMENT_IDS for session in reader.SESSIONS) == 9024)
        original_ids = np.asarray(voxel_ids, dtype=np.int64)
        report["phase"] = "all_selected_maps_intersection"
        for pid in reader.DEVELOPMENT_IDS:
            for session in reader.SESSIONS:
                for row in plan[pid, session]:
                    resource_check()
                    valid = sample(pid, row)
                    panel._require(valid.dtype == np.dtype(bool) and valid.shape == (208,))
                    keep[:] &= valid
                    report["completed_maps"] += 1
        resource_check()
        panel._require(report["completed_maps"] == 9024)
        raise IntersectionComplete()

    try:
        panel._write_new(output_dir / "settings.json", panel._encoded(report, panel.REPORT_BYTES))
        for sig in saved:
            signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, max_seconds)
        inputs = panel.load_released_inputs(manifest, check=check)
        reader.sample_beta, reader._assemble = validity_sample, intersect
        report["phase"] = "complete_reader_qualification"
        reader.read_development_handoffs(**inputs, check=check)
        raise panel.ExecutionRefusal()
    except IntersectionComplete:
        try:
            check()
            payload = io.BytesIO()
            np.savez(payload, keep=keep, original_voxel_ids=original_ids,
                selected_voxel_ids=original_ids[keep], completed_maps=np.array(9024, dtype=np.int64))
            panel._require(len(payload.getvalue()) <= 16384)
            panel._write_new(output_dir / "selection.npz", payload.getvalue())
            report.update(status="completed_common_availability_mask", mask_complete=True,
                retained_locations=int(keep.sum()), removed_locations=int((~keep).sum()),
                empty_support_stop=not bool(keep.any()))
        except BaseException as error:
            report.update(status="stopped", exception_diagnostic=panel._exception_diagnostic(error))
    except BaseException as error:
        report.update(status="stopped", exception_diagnostic=panel._exception_diagnostic(error))
    finally:
        reader.sample_beta, reader._assemble = old_sample, old_assemble
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in saved.items():
            signal.signal(sig, handler)
        report.update(elapsed_seconds=time.monotonic() - started,
            process_cpu_seconds=time.process_time() - cpu_started,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        panel._write_new(output_dir / "operational.json", panel._encoded(report, panel.REPORT_BYTES))
    return 0 if report["mask_complete"] and not report["empty_support_stop"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    return execute(manifest=args.manifest, output_dir=args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
