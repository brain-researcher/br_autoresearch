#!/usr/bin/env python3
"""One complete fixed-support mapping-quality census, never a fitting retry.

Use only the already released twelve development containers and the exact
cost001 manifest. The unchanged reader qualifies the complete plan, every ZIP
directory and allowed live header, units, fixed support and forward coordinates.
Process-local replacements of sample_beta and _assemble then measure the same
strict interpolation flags for every A/noninstruction and B-only primary map.
No numeric cohort is assembled or returned. Only metadata and invalid-location
counts are retained; no beta values, coordinates, scores or new support.

All parse, authentication, identity, header, forward-field and resource errors
remain fatal. Only the already identified strict beta-stencil invalidity is a
census observation. A complete census requires every original selected member.
Incomplete counts, if any, are explicitly labelled incomplete and are never a
complete-panel result. Prior cost001/diagnostic001 remain untouched.

Coordinator admission: one CPU / 7 GiB, <=600 seconds, 6 GiB soft observed RSS,
all numerical threads one, and an enclosing TERM/KILL timeout. Report + settings
are bounded by 16 MiB; native work is bounded by the allocation/outer timeout.
This executable cannot submit jobs, retry, remove people/locations, or impute.
"""
import argparse
from datetime import datetime, timezone
import itertools
import math
import os
from pathlib import Path
import resource
import signal
import socket
import time

import run_development_point_panel as panel


MANIFEST = panel.OUTPUTS / "runs/development_release/attempt001/view/point_panel_manifest.json"
DESTINATION = panel.OUTPUTS / "runs/observed_beta_development/cost001_mapping_diagnostic002"
BYTE_CAP = 16 * 1024**2
RESERVE = 2 * 1024**2  # Settings, categorical errors and aggregate metadata.
COUNT_FIELDS = (
    "invalid_locations", "nonfinite_interpolated_locations",
    "nonfinite_coordinate_locations", "outside_grid_locations",
    "positive_weight_corner_outside_locations",
    "nonfinite_raw_corner_locations", "nonfinite_scaled_corner_locations",
    "nonfinite_sum_with_finite_corners_locations",
)


class CensusCompleted(Exception):
    """Internal terminal marker: bypass the reader's numeric cohort return."""


def _quality(image, native_world):
    """Same positive-weight stencil; counts overlap and disclose no values."""
    import numpy as np
    from build_anatomical_support import scale_values, strict_trilinear, world_to_voxel

    coordinates = world_to_voxel(image["affine"], native_world)
    values, valid = strict_trilinear(image, coordinates)
    panel._require(coordinates.shape == (208, 3) and valid.shape == (208,))
    counts = dict.fromkeys(COUNT_FIELDS, 0)
    counts["invalid_locations"] = int((~valid | ~np.isfinite(values)).sum())
    counts["nonfinite_interpolated_locations"] = int((~np.isfinite(values)).sum())
    if not counts["invalid_locations"]:
        return counts

    volume = image["values"]
    shape = np.asarray(volume.shape, dtype=np.int64)
    finite_coordinates = np.isfinite(coordinates).all(axis=1)
    inside = ((coordinates >= 0) & (coordinates <= shape - 1)).all(axis=1)
    eligible = finite_coordinates & inside
    base = np.zeros(coordinates.shape, dtype=np.int64)
    fraction = np.zeros(coordinates.shape, dtype=np.float64)
    base[eligible] = np.floor(coordinates[eligible]).astype(np.int64)
    fraction[eligible] = coordinates[eligible] - base[eligible]
    outside_corner = np.zeros(208, dtype=bool)
    raw_nonfinite = np.zeros(208, dtype=bool)
    scaled_nonfinite = np.zeros(208, dtype=bool)
    for corner in itertools.product((0, 1), repeat=3):
        corner = np.asarray(corner, dtype=np.int64)
        weight = np.prod(np.where(corner, fraction, 1. - fraction), axis=1)
        active = eligible & (weight > 0)
        indices = base + corner
        in_bounds = ((indices >= 0) & (indices < shape)).all(axis=1)
        outside_corner |= active & ~in_bounds
        selected = np.flatnonzero(active & in_bounds)
        positions = indices[selected]
        raw = volume[positions[:, 0], positions[:, 1], positions[:, 2]]
        scaled = scale_values(image, raw)
        raw_nonfinite[selected[~np.isfinite(raw)]] = True
        scaled_nonfinite[selected[~np.isfinite(scaled)]] = True
    # Bind the classification to the unchanged sampler, not an alternative
    # validity rule. Finite-corner accumulation overflow remains invalid too.
    predicted_valid = eligible & ~outside_corner & ~scaled_nonfinite
    panel._require(np.array_equal(valid, predicted_valid))
    counts.update(
        nonfinite_coordinate_locations=int((~finite_coordinates).sum()),
        outside_grid_locations=int((finite_coordinates & ~inside).sum()),
        positive_weight_corner_outside_locations=int(outside_corner.sum()),
        nonfinite_raw_corner_locations=int(raw_nonfinite.sum()),
        nonfinite_scaled_corner_locations=int(scaled_nonfinite.sum()),
        nonfinite_sum_with_finite_corners_locations=int((valid & ~np.isfinite(values)).sum()))
    panel._require(counts["invalid_locations"] == int((~predicted_valid | ~np.isfinite(values)).sum()))
    return counts


def execute(*, manifest, output_dir, max_seconds, rss_limit_kib):
    panel._compute()
    panel._require(panel._path(manifest) == MANIFEST and panel._path(output_dir) == DESTINATION)
    panel._require(math.isfinite(max_seconds) and 0 < max_seconds <= 600
                   and type(rss_limit_kib) is int and 0 < rss_limit_kib <= 6 * 1024**2)
    panel._require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0))
    output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    output_dir.chmod(0o700)
    started, cpu_started = time.monotonic(), time.process_time()
    deadline = time.time() + max_seconds
    report = dict(status="started", phase="startup", job_id=os.environ["SLURM_JOB_ID"],
        host=socket.gethostname(), started_utc=datetime.now(timezone.utc).isoformat(),
        manifest_path=str(manifest), max_seconds=max_seconds, rss_limit_kib=rss_limit_kib,
        quality_census_complete=False, expected_maps=None, completed_maps=0,
        support_locations_per_map=208, invalid_maps=0,
        invalid_location_incidents=0, count_categories_may_overlap=True,
        across_map_location_counts_are_incidents_not_unique_locations=True,
        scope="all_original_released_A_noninstruction_and_B_only_primary_members",
        beta_values_coordinates_or_scores_recorded=False, numeric_cohort_returned=False,
        model_fitting_started=False, audit_values_read=False, private_reliability_read=False,
        support_or_participants_changed=False, invalid_map_metadata=[],
        participant_condition_totals=[], count_totals=dict.fromkeys(COUNT_FIELDS, 0))
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    reader = None
    old_sample = old_assemble = None
    summaries = {}
    detail_bytes = 0

    def check():
        panel._require(time.time() < deadline
                       and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= rss_limit_kib)

    def stop(_signal, _frame):
        raise panel.ExecutionRefusal()

    def quality_sample(archive, member, header, mask_header, native_world):
        from pilot_task_a_mapping import validate_beta_reference
        from build_anatomical_support import read_image
        check()
        validate_beta_reference(header, mask_header)
        image = read_image(archive, member, header)
        quality = _quality(image, native_world)
        check()
        return quality, image["payload_bytes"]

    def census(plan, voxel_ids, sample, resource_check):
        nonlocal detail_bytes
        panel._require(len(voxel_ids) == 208)
        report["phase"] = "complete_fixed_panel_quality_census"
        report["expected_maps"] = sum(len(plan[pid, session]) for pid in reader.DEVELOPMENT_IDS
                                       for session in reader.SESSIONS)
        for pid in reader.DEVELOPMENT_IDS:
            for session in reader.SESSIONS:
                for row in plan[pid, session]:
                    resource_check()
                    identity = {key: row[key] for key in (
                        "participant_id", "session", "task_set", "run", "reg_id",
                        "cond_name", "beta_member", "half", "geometry_partition")}
                    report["current_member_metadata"] = identity
                    quality = sample(pid, row)
                    panel._require(set(quality) == set(COUNT_FIELDS)
                                   and all(type(value) is int and 0 <= value <= 208
                                           for value in quality.values()))
                    key = pid, session, row["cond_name"]
                    if key not in summaries:
                        summaries[key] = dict(participant_id=pid, session=session,
                            condition=row["cond_name"], completed_maps=0, invalid_maps=0,
                            count_totals=dict.fromkeys(COUNT_FIELDS, 0))
                    summary = summaries[key]
                    summary["completed_maps"] += 1
                    report["completed_maps"] += 1
                    for name, count in quality.items():
                        summary["count_totals"][name] += count
                        report["count_totals"][name] += count
                    if quality["invalid_locations"]:
                        detail = dict(identity, invalid_location_counts=quality)
                        detail_bytes += len(panel._encoded(detail, BYTE_CAP)) + 8
                        panel._require(detail_bytes <= BYTE_CAP - RESERVE)
                        report["invalid_map_metadata"].append(detail)
                        report["invalid_maps"] += 1
                        summary["invalid_maps"] += 1
                    report["invalid_location_incidents"] += quality["invalid_locations"]
        resource_check()
        panel._require(report["completed_maps"] == report["expected_maps"])
        raise CensusCompleted()

    try:
        panel._write_new(output_dir / "settings.json", panel._encoded(report, panel.REPORT_BYTES))
        for sig in saved:
            signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, max_seconds)
        report["phase"] = "released_metadata"
        inputs = panel.load_released_inputs(manifest, check=check)
        import development_handoff_reader as reader
        old_sample, old_assemble = reader.sample_beta, reader._assemble
        reader.sample_beta, reader._assemble = quality_sample, census
        report["phase"] = "unchanged_complete_reader_qualification"
        reader.read_development_handoffs(**inputs, check=check)
        raise panel.ExecutionRefusal()  # A numeric cohort return is forbidden.
    except CensusCompleted:
        report.update(status="completed_quality_census", quality_census_complete=True)
    except BaseException as error:
        report.update(status="stopped", error_code=report["phase"] + "_failed",
                      exception_diagnostic=panel._exception_diagnostic(error))
    finally:
        if reader is not None and old_sample is not None:
            reader.sample_beta, reader._assemble = old_sample, old_assemble
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in saved.items():
            signal.signal(sig, handler)
        report["participant_condition_totals"] = [summaries[key] for key in sorted(summaries)]
        report.update(elapsed_seconds=time.monotonic() - started,
            process_cpu_seconds=time.process_time() - cpu_started,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        settings_bytes = (output_dir / "settings.json").stat().st_size
        panel._write_new(output_dir / "operational.json", panel._encoded(report, BYTE_CAP - settings_bytes))
    return 0 if report["quality_census_complete"] else 1


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
    print('{"status":"' + ("completed_quality_census" if code == 0 else "stopped")
          + '","analysis_id":"cost001_mapping_diagnostic002"}', flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
