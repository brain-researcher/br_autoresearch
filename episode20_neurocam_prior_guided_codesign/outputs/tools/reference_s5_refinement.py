#!/usr/bin/env python3
"""Nearest-only interior001 follow-up; unchanged circuit and full-record reader.

This prospective wave follows the completed initial36 nearest-only results.
It changes only sampled parameter points within the already selected domain.
Original source files, manifests, records and the unopened distant check remain
unchanged. Run scientific commands only inside a coordinated Slurm allocation.
"""
import argparse
from dataclasses import asdict
import math
from pathlib import Path


ROUND_ID = "interior001"
PRIOR_FREEZE = "outputs/runs/reference_s5/result_job_47382257_step_1/nearest_only_freeze.json"


def refinement_manifest():
    from reference_s5_family import Parameters, specification
    from reference_s5_operator import operator_specification
    load_lo, load_hi = math.sqrt(1e-9 * 20e-6), 20e-6
    cap_lo, cap_hi = 0.2e-12, 30e-12
    positions = (0.25, 0.5, 0.75)
    loads = [load_lo * (load_hi / load_lo) ** t for t in positions]
    caps = [cap_lo * (cap_hi / cap_lo) ** t for t in positions]
    return dict(
        round_id=ROUND_ID, prior_freeze=PRIOR_FREEZE,
        family=specification(), operator=operator_specification(),
        points=[asdict(Parameters(g, c)) for g in loads for c in caps],
        calibration=dict(source_target_db=-12.6, gate_target_db=-37.2,
                         absolute_error_max_db=3., source_must_exceed_gate=True),
        search="fixed_common3x3_logquartile_interior_wave_for_each_of_R1_R2_R3",
        search_decision="initial36_completed_nearest_only_calibration; no distant outcomes accessed",
        grid_scope="numerically_explored_only; no complete_domain_coverage_claim",
        contact_alternatives="both_E_area_and_E_sublinear_retained_as_equal_dry_air_predictions",
        held_aside="not_read_until_all27_followup_records_and_original36_are_complete_and_combined_compatible_set_is_frozen")


def validate_manifest(value):
    """Reject any changed point, operator, family, threshold or read boundary."""
    if not isinstance(value, dict) or value != refinement_manifest():
        raise ValueError("manifest differs from the selected interior001 nearest-only wave")


def prepare(path):
    from reference_s5_prediction import write_json
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = refinement_manifest()
    write_json(target, manifest)  # Exclusive creation preserves an existing wave.
    return manifest


def run(args, context):
    """Delegate the exact full-record operation with process-local admission."""
    if (context["allocated_cpus"] != 1 or context["numerical_workers"] != 1
            or args.stop_after_frame != 4000):
        raise ValueError("one CPU/worker and the complete original 4000-frame target are required")
    if args.structure not in ("R1", "R2", "R3") or not 0 <= args.point < 9:
        raise ValueError("structure or point outside the selected interior001 wave")
    import reference_s5_family as original_family
    import reference_s5_prediction as prediction
    validate_manifest(prediction.load_json(args.manifest))
    # Load the subclass before any factory replacement, retaining its original
    # base class and the existing AirRecord physical-state restart comparison.
    from reference_s5_cram16_prototype import Cram16PrototypeAirFamily
    original_factory = original_family.AirFamily
    original_validator = prediction.validate_manifest
    instances = []

    def accelerated(structure, parameters, **kwargs):
        if structure not in ("R2", "R3"):
            raise ValueError("CRAM16 factory is restricted to full-state R2/R3")
        family = Cram16PrototypeAirFamily(structure, parameters, **kwargs)
        instances.append(family)
        return family

    prediction.validate_manifest = validate_manifest
    if args.structure in ("R2", "R3"):
        original_family.AirFamily = accelerated
    try:
        return prediction.run(args, context)
    finally:
        prediction.validate_manifest = original_validator
        original_family.AirFamily = original_factory
        if instances:
            family = instances[-1]
            prediction.write_json(Path(args.run_dir) / "numerical_backend.json", dict(
                backend="full_augmentation_IPF_CRAM16_exact_structured_shifted_inverse",
                round_id=ROUND_ID, family_structure=family.structure,
                physical_state_count=family.size, cram=family.cram_report(),
                resumed_physical_state_and_summary_validation="unchanged original AirRecord",
                full_target_frames=4000, model_reduction=False,
                held_aside_access=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--output", required=True)
    work = commands.add_parser("run")
    work.add_argument("--manifest", required=True)
    work.add_argument("--structure", choices=("R1", "R2", "R3"), required=True)
    work.add_argument("--point", type=int, required=True)
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
