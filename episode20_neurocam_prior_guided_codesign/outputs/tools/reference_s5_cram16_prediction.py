#!/usr/bin/env python3
"""Prepared CRAM16 entrypoint for the unchanged full S5 records.

The coordinator may select this only after numerical qualification and after
earlier writers for the selected records are terminal. This file does not
activate, qualify, cancel or submit anything. The original manifest, physical
state ordering, acquisition, record/checkpoint reader, 4,000-frame target and
nearest-only summary are retained. Only the R2/R3 numerical family constructor
is replaced within this process; no shared source or record binding changes.
"""
import argparse
from pathlib import Path


def run(args):
    from fast_solver_continuous import compute_context
    context = compute_context()
    if context["allocated_cpus"] != 1 or args.stop_after_frame != 4000:
        raise ValueError("one CPU and the complete original 4000-frame target are required")
    # Import the subclass before replacing the original factory. Its base
    # therefore stays the unchanged AirFamily, including reconstruct() used
    # by the existing exact checkpoint-state comparison.
    from reference_s5_cram16_prototype import Cram16PrototypeAirFamily
    import reference_s5_family as original_family
    import reference_s5_prediction as prediction

    original = original_family.AirFamily
    instances = []

    def accelerated(structure, parameters, **kwargs):
        if structure not in ("R2", "R3"):
            raise ValueError("this numerical entrypoint retains full-state R2/R3 only")
        family = Cram16PrototypeAirFamily(structure, parameters, **kwargs)
        instances.append(family)
        return family

    original_family.AirFamily = accelerated
    try:
        report = prediction.run(args, context)
    finally:
        original_family.AirFamily = original
        if instances:
            family = instances[-1]
            cram = family.cram_report()
            prediction.write_json(Path(args.run_dir) / "numerical_backend.json", dict(
                backend="full_augmentation_IPF_CRAM16_exact_structured_shifted_inverse",
                family_structure=family.structure,
                physical_state_count=family.size,
                cram=cram,
                work_accounting=dict(
                    shifts_per_completed_augmented_interval=8,
                    completed_augmented_interval_recurrences=cram["actions"],
                    inverse_requests=cram["hits"]+cram["misses"],
                    final_interval_may_be_interrupted=True,
                    legacy_operator_products="inherited Taylor counter; does not count CRAM work",
                    legacy_sparse_exponentials="completed CRAM16 interval actions in this entrypoint"),
                exact_duration_cache_keys=True,
                original_manifest_operator_record_and_checkpoints=True,
                resumed_physical_state_and_summary_validation="unchanged original AirRecord",
                full_target_frames=4000,
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
    parser.add_argument("--stop-after-frame", type=int, choices=(4000,), default=4000)
    parser.add_argument("--deadline-seconds", type=float, required=True)
    parser.add_argument("--memory-mib", type=float, default=1536.)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
