"""Nearest-only calibration, explicit freeze, then conditional distant prediction.

All scientific subcommands run on compute. Preparing a manifest fixes the
sampled parameter family before fitting; a finite grid is an explored envelope,
not a proof about the complete continuous rectangle. No device CI is computed.
"""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import signal
import time


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def prepare(path):
    from dataclasses import asdict
    from reference_s5_family import specification, initial_grid
    from reference_s5_operator import operator_specification
    value = dict(family=specification(), operator=operator_specification(),
        points=[asdict(p) for p in initial_grid()],
        search="fixed_initial12point_grid_for_each_of_three_effective_structures",
        grid_scope="numerically_explored_only; no complete_domain_coverage_claim",
        contact_alternatives="both_E_area_and_E_sublinear_retained_as_equal_dry_air_predictions",
        calibration=dict(source_target_db=-12.6, gate_target_db=-37.2,
            absolute_error_max_db=3., source_must_exceed_gate=True),
        held_aside="not_read_until_every_grid_record_is_complete_and_compatible_set_is_frozen")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)
    return value


def validate_manifest(value):
    from dataclasses import asdict
    from reference_s5_family import specification, initial_grid
    from reference_s5_operator import operator_specification
    if (value.get("family") != specification() or value.get("operator") != operator_specification()
            or value.get("points") != [asdict(p) for p in initial_grid()]
            or value.get("calibration") != dict(source_target_db=-12.6, gate_target_db=-37.2,
                absolute_error_max_db=3., source_must_exceed_gate=True)):
        raise ValueError("manifest differs from this selected prospective family/operator/grid")


def run(args, context):
    import numpy as np
    from fast_solver_continuous import make_clock
    from reference_s5_family import AirFamily, Parameters
    from reference_s5_operator import AirRecord, S5Operator, FRAMES, nearest_summary
    manifest = load_json(args.manifest); validate_manifest(manifest)
    if not 0 <= args.point < len(manifest["points"]) or not 1 <= args.stop_after_frame <= FRAMES:
        raise ValueError("point/frame outside selected finite experiment")
    if not (math.isfinite(args.deadline_seconds) and args.deadline_seconds > 0
            and math.isfinite(args.memory_mib) and args.memory_mib > 0):
        raise ValueError("finite positive time and memory limits required")
    directory = Path(args.run_dir); directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("operation report directory must be fresh and empty")
    started = time.perf_counter()
    clock = make_clock(args.deadline_seconds, args.memory_mib)
    previous = {}
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        previous[sig] = signal.signal(sig, lambda number, frame: setattr(clock, "interrupted", number))
    report = dict(context=context, structure=args.structure, point=args.point,
        target_frames=FRAMES, record_dir=str(Path(args.record_dir).resolve()),
        status="starting", outcome_access="nearest_only_after_complete_record",
        frame_wall_seconds=[], completed_frames_before=None, completed_frames_after=None)
    record = family = None
    try:
        family = AirFamily(args.structure, Parameters(**manifest["points"][args.point]), check=clock.check)
        operator = S5Operator(family)
        record = AirRecord(args.record_dir, operator, manifest, resume=args.resume)
        report["setup_seconds"] = time.perf_counter() - started
        report["completed_frames_before"] = record.next_frame
        while record.next_frame < args.stop_after_frame:
            clock.check()
            frame_started = time.perf_counter()
            state, readout, energy = operator.frame(record.state, record.next_frame)
            record.commit(state, readout, energy)
            report["frame_wall_seconds"].append(time.perf_counter() - frame_started)
            print(json.dumps(dict(committed_frames=record.next_frame,
                physical_seconds=record.next_frame * .004,
                frame_wall_seconds=report["frame_wall_seconds"][-1])), flush=True)
        report["status"] = "complete_record" if record.next_frame == FRAMES else "clean_prefix"
        if record.next_frame == FRAMES:
            report["nearest_calibration"] = nearest_summary(record.summary)
    except (TimeoutError, MemoryError) as exc:
        report["status"] = "bounded_stop"
        report["reason"] = str(exc)
    except BaseException as exc:
        report["status"] = "failed"
        report["reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if record is not None:
            report["completed_frames_after"] = record.next_frame
            report["committed_snapshot_time_seconds"] = record.next_frame * .004
            report["raw_clipped_count"] = record.clipped
            record.close()
        if family is not None:
            report["solver_counts"] = family.stats
            report["cache"] = family.cache.report()
        report["total_wall_seconds"] = time.perf_counter() - started
        report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        write_json(directory / "run_report.json", report)
    return report


def saved_summary(directory, manifest, structure, point):
    """Validate a full record before estimating; never inspect distant amplitudes."""
    import h5py
    import numpy as np
    from reference_s5_operator import FRAMES, SineSummary
    directory = Path(directory)
    with np.load(directory / "frame_checkpoint.npz", allow_pickle=False) as saved:
        binding_text = str(saved["binding"]); binding = json.loads(binding_text)
        if (int(saved["next_frame"]) != FRAMES or float(saved["time"]) != 16.
                or binding["manifest"] != manifest or binding["structure"] != structure
                or binding["parameters"] != manifest["points"][point]):
            raise ValueError(f"incomplete or different frozen record: {directory}")
        summary = SineSummary(); summary.count = FRAMES
        summary.xtx, summary.xty = saved["xtx"].copy(), saved["xty"].copy()
        clips = int(saved["clipped"])
    with h5py.File(directory / "record.h5", "r") as record:
        if record.attrs["binding"] != binding_text or not np.all(record["committed"][:]):
            raise ValueError("full published record not durably committed")
    return summary, clips, binding_text


def freeze(args):
    from reference_s5_family import STRUCTURES, CONTACT_NAMES
    from reference_s5_operator import nearest_summary
    manifest = load_json(args.manifest); validate_manifest(manifest)
    rows, accepted = [], []
    # Require the entire prefit sampled grid, including rejected points/models.
    # This prevents a convenient successful prefix from silently defining the set.
    for structure in STRUCTURES:
        for point in range(len(manifest["points"])):
            directory = Path(args.records_root) / structure / f"point_{point:02d}"
            summary, clips, binding = saved_summary(directory, manifest, structure, point)
            nearest = nearest_summary(summary)
            source, gate = nearest["nearest_source"], nearest["nearest_gate"]
            compatible = (source["status"] == gate["status"] == "finite"
                and abs(source["db"] + 12.6) <= 3. and abs(gate["db"] + 37.2) <= 3.
                and source["db"] > gate["db"])
            row = dict(structure=structure, point=point, record_dir=str(directory.resolve()),
                parameters=manifest["points"][point], nearest=nearest, compatible=compatible,
                raw_clipped_count=clips, binding=binding,
                named_templates=[f"{contact}/{structure}" for contact in CONTACT_NAMES])
            rows.append(row)
            if compatible:
                accepted.append(dict(structure=structure, point=point, record_dir=row["record_dir"]))
    value = dict(manifest=manifest, calibration=rows, compatible=accepted,
        status="fit_frozen_before_distant_evaluation", uncertainty="sampled_compatible_parameter_envelope",
        not_claimed=["continuous_domain_identified_set", "device_CI", "accepted_full_reference", "candidate_R2_ranking"])
    write_json(args.output, value)
    return value


def predict(args):
    import numpy as np
    from reference_s5_family import STRUCTURES
    from reference_s5_operator import ratio_db
    frozen = load_json(args.freeze)
    if frozen.get("status") != "fit_frozen_before_distant_evaluation":
        raise ValueError("nearest-only compatible fit must be frozen first")
    validate_manifest(frozen["manifest"])
    if not frozen["compatible"]:
        raise ValueError("no compatible sampled model; report calibration failure without opening distant check")
    # The public distant summary enters only this post-freeze stage.
    if args.paper_far_db != -45. or args.check_max_db != -42.:
        raise ValueError("retain the assigned paper summary and existing check threshold")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("prediction output directory must be fresh and empty")
    rr, cc = np.indices((64, 64)); distance = (abs(rr - 31) + abs(cc - 31)).ravel()
    far = distance >= 2; entries = []; amplitudes_all = []
    for candidate in frozen["compatible"]:
        match = [r for r in frozen["calibration"] if r["structure"] == candidate["structure"]
                 and r["point"] == candidate["point"] and r["compatible"]]
        if len(match) != 1 or match[0]["record_dir"] != candidate["record_dir"]:
            raise ValueError("frozen candidate is not an accepted calibration record")
        summary, clips, binding = saved_summary(candidate["record_dir"], frozen["manifest"],
                                                  candidate["structure"], candidate["point"])
        if binding != match[0]["binding"]:
            raise ValueError("prediction record differs from the frozen calibration record")
        amplitudes = summary.amplitudes(np.arange(4096)); driven = float(amplitudes[31 * 64 + 31])
        far_result = ratio_db(float(np.max(amplitudes[far])), driven)
        source_curve, gate_curve = [], []
        for d in range(33):
            source = ((cc == 31) & (abs(rr - 31) == d)).ravel()
            gate = ((rr == 31) & (abs(cc - 31) == d)).ravel()
            source_curve.append(ratio_db(float(np.max(amplitudes[source])), driven))
            gate_curve.append(ratio_db(float(np.max(amplitudes[gate])), driven))
        passes = (far_result["status"] == "negative_infinity_zero_numerator"
                  or (far_result["status"] == "finite" and far_result["db"] <= args.check_max_db))
        entries.append(dict(**candidate, named_templates=match[0]["named_templates"],
            far_maximum=far_result, distant_check_passes=passes,
            source_curve=source_curve, gate_curve=gate_curve, raw_clipped_count=clips))
        amplitudes_all.append(amplitudes)
    np.savez(out / "per_pad_amplitudes.npz", amplitudes_volts=np.stack(amplitudes_all),
             rows=rr.ravel(), columns=cc.ravel(), manhattan_distance=distance,
             structures=np.asarray([e["structure"] for e in entries]), points=np.asarray([e["point"] for e in entries]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    colors = dict(R1="tab:blue", R2="tab:orange", R3="tab:green")
    for axis, field, title in zip(axes, ("source_curve", "gate_curve"), ("Source direction", "Gate direction")):
        seen = set()
        for entry in entries:
            values = [v["db"] if v["status"] == "finite" else np.nan for v in entry[field]]
            label = entry["structure"] + " (both E contacts)" if entry["structure"] not in seen else None
            seen.add(entry["structure"])
            axis.plot(range(33), values, color=colors[entry["structure"]], alpha=.6, label=label)
        axis.hlines(args.paper_far_db, 2, 32, color="black", linestyles="--", label="Held-aside paper ≈−45 dB")
        axis.hlines(args.check_max_db, 2, 32, color="grey", linestyles=":", label="Prespecified ceiling")
        axis.set(xlabel="Pad separation", title=title, xlim=(0, 32))
    axes[0].set_ylabel("Amplitude relative to driven output (dB)")
    axes[1].legend(fontsize=7)
    fig.suptitle("Conditional dry-air S5 prediction; explored compatible parameters, no device CI")
    fig.tight_layout(); fig.savefig(out / "conditional_s5_prediction.png", dpi=180); plt.close(fig)
    result = dict(entries=entries, paper_far_db=args.paper_far_db, check_max_db=args.check_max_db,
        calibration=frozen["calibration"], uncertainty=frozen["uncertainty"],
        exact_zero_plot="zero/undefined dB are omitted from curves and retained explicitly in tables",
        interpretation=[
            "A prospective dry-air/probe/DAQ interpretation was fitted only to nearest source/gate anchors.",
            f"{sum(e['distant_check_passes'] for e in entries)} of {len(entries)} compatible sampled parameter points pass the distant ceiling; the three structural fits are alternatives, not independent devices.",
            "This conditional prediction neither bounds the unsampled domain nor accepts the full reference or changes the primary R² question."])
    write_json(out / "prediction.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare"); prep.add_argument("--output", required=True)
    work = subs.add_parser("run")
    work.add_argument("--manifest", required=True); work.add_argument("--structure", choices=("R1", "R2", "R3"), required=True)
    work.add_argument("--point", type=int, required=True); work.add_argument("--record-dir", required=True)
    work.add_argument("--run-dir", required=True); work.add_argument("--resume", action="store_true")
    work.add_argument("--stop-after-frame", type=int, default=4000)
    work.add_argument("--deadline-seconds", type=float, required=True); work.add_argument("--memory-mib", type=float, default=6656.)
    fit = subs.add_parser("freeze"); fit.add_argument("--manifest", required=True)
    fit.add_argument("--records-root", required=True); fit.add_argument("--output", required=True)
    pred = subs.add_parser("predict"); pred.add_argument("--freeze", required=True)
    pred.add_argument("--paper-far-db", type=float, required=True); pred.add_argument("--check-max-db", type=float, required=True)
    pred.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    from fast_solver_continuous import compute_context
    context = compute_context()
    if args.command == "prepare":
        prepare(args.output)
    elif args.command == "run":
        run(args, context)
    elif args.command == "freeze":
        freeze(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
