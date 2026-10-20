#!/usr/bin/env python3
"""First failed audit-B stencil only; retain predictions, support and failed state.

Run on compute after the recorded failed B opening. This diagnoses source
availability/reader consistency; it never evaluates predictions or changes the
scientific panel. Outputs contain finite/nonfinite categories, not beta values.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import io
import itertools
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import struct
import time


EXPECTED_ERROR = "Task-A beta has an invalid or nonfinite fixed-support stencil; no support shrink"


class FirstFailedMember(Exception):
    def __init__(self, result):
        self.result = result
        super().__init__("first_failed_member_diagnosed_no_further_B_sampling")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def limits(deadline, rss_kib):
    require(math.isfinite(deadline) and deadline > time.time() and rss_kib > 0,
            "finite_future_deadline_and_positive_RSS_limit_required")
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "existing_timer_not_replaced")
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    def check():
        if time.time() >= deadline:
            raise TimeoutError("diagnostic_deadline")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > rss_kib:
            raise MemoryError("diagnostic_RSS_limit")
    def stop(_sig, _frame):
        raise TimeoutError("diagnostic_interrupted")
    try:
        for sig in previous:
            signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, deadline - time.time())
        check()
        yield check
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def scalar_class(value):
    if math.isnan(float(value)):
        return "nan"
    if math.isinf(float(value)):
        return "positive_infinity" if value > 0 else "negative_infinity"
    return "finite"


def direct_corner(archive, member, image, ijk, check):
    """Decode one exact scalar independently of NumPy reshape; export classes only."""
    check()
    with archive.open(archive.getinfo(member)) as stream:
        header = stream.read(348)
        require(len(header) == 348, "complete_NIfTI_header_required")
        endian = "<" if struct.unpack_from("<i", header)[0] == 348 else ">"
        require(struct.unpack_from(endian + "i", header)[0] == 348, "NIfTI_endian_invalid")
        datatype = struct.unpack_from(endian + "h", header, 70)[0]
        fmt, itemsize = {2: ("B", 1), 4: ("h", 2), 16: ("f", 4)}[datatype]
        dims = struct.unpack_from(endian + "8h", header, 40)
        shape = tuple(dims[1:4])
        require(dims[0] == 3 and shape == image["values"].shape, "direct_shape_mismatch")
        offset = struct.unpack_from(endian + "f", header, 108)[0]
        require(math.isfinite(offset) and offset == int(offset) and offset >= 352,
                "direct_offset_invalid")
        i, j, k = map(int, ijk)
        byte_offset = int(offset) + itemsize * (i + shape[0] * (j + shape[1] * k))
        require(byte_offset + itemsize <= archive.getinfo(member).file_size, "scalar_outside_member")
        stream.seek(byte_offset)
        payload = stream.read(itemsize)
        require(len(payload) == itemsize, "complete_direct_scalar_required")
        direct = struct.unpack(endian + fmt, payload)[0]
    array_value = image["values"][i, j, k]
    same = ((math.isnan(float(direct)) and math.isnan(float(array_value)))
            or float(direct) == float(array_value))
    check()
    return {"native_corner_ijk": [i, j, k], "NIfTI_member_byte_offset": byte_offset,
            "offset_rule": "voxel_offset + itemsize * (i + Nx * (j + Ny * k))",
            "datatype": datatype, "endian": endian, "raw_scalar_class": scalar_class(direct),
            "array_scalar_class": scalar_class(array_value), "exact_scalar_match": bool(same),
            "finite_magnitude_exported": False}


def classify(archive, member, header, mask, world, voxel_ids, check):
    import numpy as np
    from build_anatomical_support import read_image, strict_trilinear, world_to_voxel, scale_values
    from pilot_task_a_mapping import validate_beta_reference
    check()
    validate_beta_reference(header, mask)
    image = read_image(archive, member, header)  # Only the already failed member.
    coordinates = world_to_voxel(image["affine"], world)
    values, valid = strict_trilinear(image, coordinates)
    finite = np.isfinite(values)
    bad = ~(valid & finite)
    require(len(bad) == len(voxel_ids) == 188 and np.any(bad), "same_member_failure_must_reproduce")
    shape = np.asarray(image["values"].shape)
    point_details, witness, fallback = [], None, None
    for point in np.flatnonzero(bad):
        check()
        coordinate = coordinates[point]
        causes, corners = [], []
        if not np.isfinite(coordinate).all():
            causes.append("nonfinite_sampling_coordinate")
        elif not ((coordinate >= 0) & (coordinate <= shape - 1)).all():
            causes.append("sampling_coordinate_outside_grid")
        else:
            base = np.floor(coordinate).astype(int)
            fraction = coordinate - base
            for corner in itertools.product((0, 1), repeat=3):
                corner = np.asarray(corner)
                weight = float(np.prod(np.where(corner, fraction, 1. - fraction)))
                if weight <= 0:
                    continue  # The scientific sampler does not require zero-weight corners.
                ijk = base + corner
                if not ((ijk >= 0) & (ijk < shape)).all():
                    causes.append("positive_weight_corner_outside_grid")
                    continue
                if fallback is None:
                    fallback = ijk.copy()
                raw = image["values"][tuple(ijk)]
                scaled = scale_values(image, raw)
                if not np.isfinite(raw) or not np.isfinite(scaled):
                    cause = ("positive_weight_raw_nonfinite" if not np.isfinite(raw)
                             else "positive_weight_scaling_nonfinite")
                    causes.append(cause)
                    corners.append({"ijk": ijk.tolist(), "raw_class": scalar_class(raw),
                                    "scaled_class": scalar_class(scaled), "cause": cause})
                    if witness is None:
                        witness = ijk.copy()
        if not causes:
            causes.append("nonfinite_interpolant_without_classified_corner_failure")
        point_details.append({"support_ordinal": int(point), "frozen_voxel_id": int(voxel_ids[point]),
                              "sampler_valid": bool(valid[point]), "interpolant_finite": bool(finite[point]),
                              "causes": sorted(set(causes)), "nonfinite_positive_weight_corners": corners})
    counts = {}
    for detail in point_details:
        for cause in detail["causes"]:
            counts[cause] = counts.get(cause, 0) + 1
    witness = witness if witness is not None else fallback
    direct = direct_corner(archive, member, image, witness, check) if witness is not None else None
    return {"fixed_support_count": 188, "valid_finite_count": int((~bad).sum()),
            "invalid_count": int(bad.sum()), "invalid_frozen_voxel_ids": [int(v) for v in np.asarray(voxel_ids)[bad]],
            "invalid_points": point_details, "cause_point_counts_nonexclusive": counts,
            "finite_valid_by_support_ordinal": (~bad).tolist(), "direct_scalar_check": direct,
            "direct_scalar_unavailable_reason": None if direct else "all_invalid_points_have_no_eligible_in_grid_corner"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--rss-limit-kib", type=int, default=491520)
    parser.add_argument("--durable-byte-cap", type=int, default=4 * 1024 * 1024)
    args = parser.parse_args()
    require(os.environ.get("SLURM_JOB_ID") and "-ln" not in socket.gethostname(), "verified_compute_shell_required")
    require(args.durable_byte_cap > 0, "positive_durable_cap_required")
    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    cap = min(args.durable_byte_cap, soft) if soft != resource.RLIM_INFINITY else args.durable_byte_cap
    resource.setrlimit(resource.RLIMIT_FSIZE, (cap, hard))
    with limits(args.deadline_epoch, args.rss_limit_kib) as check:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        import run_observed_map_audit as runner
        import observed_map_audit_readers as readers
        check()
        policy = runner.actual_policy()
        output = args.output_dir.absolute()
        require(output.parent == runner.RUN_ROOT and not output.exists()
                and output.name.startswith("support_diagnostic"), "new_episode_local_support_diagnostic_directory_required")
        expected_manifest = runner.RUN_ROOT / "manifests001/view/evaluation_manifest.json"
        output.mkdir(mode=0o700)
        written = 0
        def publish(name, payload):
            nonlocal written
            check()
            require(written + len(payload) <= args.durable_byte_cap, "diagnostic_durable_cap")
            with (output / name).open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            written += len(payload)
        def publish_json(name, value):
            publish(name, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())
        with runner.EXECUTION_PATH.open("r", encoding="utf-8") as state_stream:
            fcntl.flock(state_stream.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            raw = state_stream.read(runner.STATE_BYTES + 1)
            require(len(raw.encode()) <= runner.STATE_BYTES, "bounded_prior_state_required")
            state = json.loads(raw)
            binding = {"prediction_dir": str(runner.PREDICTION_DIR), "evaluation_dir": str(runner.EVALUATION_DIR),
                       "evaluation_manifest": str(expected_manifest)}
            require(state.get("status") == "failed" and state.get("binding") == binding
                    and bool(state.get("attempts")) and state.get("first_B_opening_utc")
                    and not runner.EVALUATION_DIR.exists(), "exact_failed_prior_B_evaluation_required")
            predictions = runner.load_frozen_predictions(policy=policy, check=check)
            access = {"kind": "diagnostic_after_recorded_B_opening", "started_utc": now(),
                      "job_id": os.environ["SLURM_JOB_ID"], "step_id": os.environ.get("SLURM_STEP_ID"),
                      "prior_first_B_opening_utc": state["first_B_opening_utc"], "binding": binding,
                      "prior_B_access_disclosed": True, "audit_execution_state_modified": False,
                      "prediction_files_modified": False, "new_independent_evaluation": False,
                      "scope": "first failed selected B member; original12/18/188; no scores or support changes"}
            publish_json("access.json", access)  # Before any renewed B manifest/container read.
            inputs = runner.load_audit_inputs(expected_manifest, policy=policy, calibration=False, check=check)
            plan, _ = readers._compile_plan(inputs["member_rows"], session="ses-s2",
                summary=inputs["condition_multiplicity_by_session"]["ses-s2"], check=check)
            rows = {row["beta_member"]: row for person in readers.AUDIT_IDS for row in plan[person]}
            original = readers.sample_beta
            sampled = 0
            def diagnostic_sample(archive, member, header, mask, world):
                nonlocal sampled
                check()
                require(member in rows, "only_original_selected_B_members_allowed")
                try:
                    result = original(archive, member, header, mask, world)
                except ValueError as error:
                    if str(error) != EXPECTED_ERROR:
                        raise
                    detail = classify(archive, member, header, mask, world, predictions["voxel_ids"], check)
                    row = rows[member]
                    detail.update({"participant_id": row["participant_id"], "condition": row["cond_name"],
                        "run": row["run"], "regressor_id": row["reg_id"], "member": member,
                        "successful_member_samples_before_failure": sampled, "original_exception": str(error)})
                    raise FirstFailedMember(detail) from error
                sampled += 1
                return result
            readers.sample_beta = diagnostic_sample
            try:
                try:
                    readers.read_audit_observations(policy=policy, frozen_predictions=predictions,
                                                   **inputs, check=check)
                except FirstFailedMember as error:
                    result = error.result
                else:
                    raise RuntimeError("prior_sampler_failure_not_reproduced_no_scoring_performed")
            finally:
                readers.sample_beta = original
        result.update(kind="first_failed_audit_B_support_diagnostic", status="diagnostic_complete",
            completed_utc=now(), prior_B_access_disclosed=True, no_further_B_sampling_after_failed_member=True,
            separation="procedural_same_account_not_technically_enforced", predictive_scores_computed=False,
            whole_cohort_missingness_estimated=False, support_changed=False, prediction_files_modified=False,
            audit_execution_state_modified=False, peak_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        publish_json("diagnostic.json", result)
        fig, ax = plt.subplots(figsize=(11, 2.5))
        try:
            ax.imshow([result["finite_valid_by_support_ordinal"]], aspect="auto", interpolation="nearest",
                      cmap=ListedColormap(["#d65f2e", "#286b8b"]), vmin=0, vmax=1)
            ax.set_yticks([])
            ax.set_xlabel("Frozen support ordinal (0–187); blue = valid/finite, orange = invalid")
            ax.set_title(f"Diagnostic only: {result['participant_id']}, run {result['run']}, regressor {result['regressor_id']}\n"
                         f"First failed B member: {result['invalid_count']}/188 invalid; not a cohort estimate")
            fig.tight_layout()
            for suffix in ("png", "pdf"):
                check()
                stream = io.BytesIO()
                fig.savefig(stream, format=suffix, dpi=150)
                publish("support_diagnostic." + suffix, stream.getvalue())
        finally:
            plt.close(fig)
        direct = result["direct_scalar_check"]
        witness = (f"Direct byte-offset scalar: {direct['raw_scalar_class']}; array decoding agrees={direct['exact_scalar_match']}."
                   if direct else "No eligible in-grid corner was available for a direct scalar check.")
        lines = [f"1. Diagnostic after prior B opening: first failed member {result['member']} has {result['invalid_count']} invalid locations on the unchanged frozen188 support.",
                 f"2. Positive-weight/coordinate cause counts (overlap allowed): {json.dumps(result['cause_point_counts_nonexclusive'], sort_keys=True)}. {witness}",
                 "3. Predictions and failed evaluation state are unchanged; this is neither cohort missingness prevalence nor a predictive/absence result. No losses, shared14B values or later B members were sampled after this failure."]
        publish("3lines.md", ("\n".join(lines) + "\n").encode())
        check()
        print(json.dumps({"status": "diagnostic_complete", "output_dir": str(output),
                          "invalid_points_in_first_failed_member": result["invalid_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
