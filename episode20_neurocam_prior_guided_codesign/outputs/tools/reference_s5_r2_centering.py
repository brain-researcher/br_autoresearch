#!/usr/bin/env python3
"""Final common center002 point selected from nearest-only interior R2 results.

The interpolation selects a calibration input, not a measured response or a
guarantee of compatibility. All initial36, interior27, center0013 and center0023
full records precede the compatible-set freeze and any distant evaluation.
No automatic follow-up or parameter expansion is selected by this runner.
The center001 delegation pattern reuses the unchanged qualified physics.
"""
import argparse
from dataclasses import asdict
from pathlib import Path


ROUND_ID = "center002"
INTERIOR_BASE = "outputs/runs/reference_s5/refinement_interior001"
CENTER001_BASE = "outputs/runs/reference_s5/refinement_center001"


def centering_manifest():
    import reference_s5_refinement as refinement
    from reference_s5_family import Parameters
    load = 1.6817928305074292e-6
    cap_lo, cap_hi = 2.449489742783178e-12, 8.572321289096399e-12
    gate_lo, gate_hi = -41.03157761714907, -28.96425485911152
    fraction = (-37.2 - gate_lo) / (gate_hi - gate_lo)
    cap = cap_lo * (cap_hi / cap_lo) ** fraction
    manifest = refinement.refinement_manifest()
    manifest.update(
        round_id=ROUND_ID,
        points=[asdict(Parameters(load, cap))],
        prior_round_manifest=f"{CENTER001_BASE}/manifest.json",
        search="one_common_R2_gate_centering_point_for_each_of_R1_R2_R3",
        search_decision="adaptive_selection_from_completed_interior001_R2_nearest_only_results; no distant outcomes accessed",
        selection=dict(
            role="nearest_only_log_capacitance_interpolation_not_measured_prediction",
            fixed_load_siemens=load,
            capacitance_bracket_farad=[cap_lo, cap_hi],
            measured_gate_bracket_db=[gate_lo, gate_hi],
            target_gate_db=-37.2,
            rule="t=(target-gate_lo)/(gate_hi-gate_lo); C=C_lo*(C_hi/C_lo)**t; G unchanged",
            source_reports=[
                f"{INTERIOR_BASE}/operations_array_48037544/task_13_job_48037847/record_operation/run_report.json",
                f"{INTERIOR_BASE}/operations_array_48037544/task_14_job_48037848/record_operation/run_report.json"],
            source_structure="R2", source_points=[4, 5],
            predicted_interpolation_is_measured_result=False),
        search_history_required=dict(initial001=36, interior001=27, center001=3, center002=3),
        automatic_successor=False,
        held_aside="not_read_until_all69_history_records_are_complete_and_all_compatible_points_are_frozen")
    return manifest


def validate_manifest(value):
    if not isinstance(value, dict) or value != centering_manifest():
        raise ValueError("manifest differs from the selected center002 nearest-only triplet")


def prepare(path):
    from reference_s5_prediction import write_json
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = centering_manifest()
    write_json(target, manifest)
    return manifest


def run(args, context):
    """Reuse qualified full-record physics with temporary round admission."""
    if args.point != 0:
        raise ValueError("center002 contains exactly one selected point, index 0")
    import reference_s5_refinement as refinement
    from reference_s5_prediction import load_json
    validate_manifest(load_json(args.manifest))
    original_validator, original_round = refinement.validate_manifest, refinement.ROUND_ID
    refinement.validate_manifest = validate_manifest
    refinement.ROUND_ID = ROUND_ID
    try:
        return refinement.run(args, context)
    finally:
        refinement.validate_manifest = original_validator
        refinement.ROUND_ID = original_round


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--output", required=True)
    work = commands.add_parser("run")
    work.add_argument("--manifest", required=True)
    work.add_argument("--structure", choices=("R1", "R2", "R3"), required=True)
    work.add_argument("--point", type=int, choices=(0,), required=True)
    work.add_argument("--record-dir", required=True)
    work.add_argument("--run-dir", required=True)
    work.add_argument("--resume", action="store_true")
    work.add_argument("--stop-after-frame", type=int, choices=(4000,), default=4000)
    work.add_argument("--deadline-seconds", type=float, required=True)
    work.add_argument("--memory-mib", type=float, default=1536.)
    args = parser.parse_args()
    from fast_solver_continuous import compute_context
    context = compute_context()
    if args.command == "prepare":
        prepare(args.output)
        return 0
    report = run(args, context)
    return 2 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
