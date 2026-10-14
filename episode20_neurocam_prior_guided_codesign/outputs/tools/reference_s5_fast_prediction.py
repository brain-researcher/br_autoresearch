#!/usr/bin/env python3
"""Run the unchanged full S5 record with equivalent numerical acceleration.

This entrypoint retains the original manifest, acquisition, record writer,
checkpoint validation, nearest-only fit and complete 4,000-frame target.
Only the R2/R3 family constructor is substituted for this one process, after
importing its original base classes. No live original source is modified.
"""
import argparse
from pathlib import Path


def run(args):
    from fast_solver_continuous import compute_context
    context = compute_context()
    from reference_s5_structured_solve import StructuredCachedAirFamily
    import reference_s5_family as original_family
    import reference_s5_prediction as prediction

    original = original_family.AirFamily
    instances = []

    def accelerated(structure, parameters, **kwargs):
        if structure not in ("R2", "R3"):
            raise ValueError("this acceleration entrypoint is scoped to full-state R2/R3")
        family = StructuredCachedAirFamily(structure, parameters, **kwargs)
        instances.append(family)
        return family

    original_family.AirFamily = accelerated
    try:
        report = prediction.run(args, context)
    finally:
        original_family.AirFamily = original
        if instances:
            family = instances[-1]
            prediction.write_json(Path(args.run_dir) / "numerical_backend.json", dict(
                backend="exact_separable_capacitance_and_cached_Scipy_1_12_Taylor_plan",
                family_structure=family.structure,
                physical_state_count=family.size,
                plan_cache=family.plan_report(),
                original_manifest_operator_record_and_checkpoints=True,
                model_reduction=False, held_aside_access=False))
    return 2 if report["status"] == "failed" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--structure", choices=("R2", "R3"), required=True)
    parser.add_argument("--point", type=int, required=True)
    parser.add_argument("--record-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-frame", type=int, default=4000)
    parser.add_argument("--deadline-seconds", type=float, required=True)
    parser.add_argument("--memory-mib", type=float, default=6656.)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
