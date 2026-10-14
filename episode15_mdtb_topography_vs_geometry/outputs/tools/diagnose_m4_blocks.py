#!/usr/bin/env python3
"""Task-A-cache-only diagnosis of all three unchanged block-local M4 families."""
import argparse
import json
from pathlib import Path
import resource
import time

from block_m4_global import fit_block_global
from diagnose_m4_global import _load_cache
from paired_stretch_variants import PairedM4TaskAObjective, stretch_specifications, M4CertificateUnverified
import run_development_point_panel as runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds-per-family", type=float, default=300.)
    parser.add_argument("--max-cells", type=int, default=4096)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    runner._compute()
    cache, output = args.cache.resolve(), args.output.resolve()
    scope = runner.OUTPUTS / "runs/observed_beta_development"
    if scope not in cache.parents or scope not in output.parents or output.exists() or not output.parent.is_dir():
        parser.error("explicit existing private Task-A cache and fresh aggregate output in the observed-beta route required")
    original, _ = _load_cache(cache)
    started, rows = time.monotonic(), []
    def check():
        if time.time() >= args.deadline_epoch or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 900000:
            raise RuntimeError("diagnostic allocation bound reached")
    for spec in stretch_specifications()[1:]:
        check()
        objective = PairedM4TaskAObjective(original.source, original.target, original.B, original.O,
            global_gain=original.gain, condition_weights=original.weights, mode_matrices=spec.modes,
            rank_bound=16, dof_bound=spec.parameter_count,
            rank_rtol=original.rank_rtol, numerical_atol=original.atol)
        row = dict(variant=spec.name, operator_locality_check_exercised=False)
        try:
            numerical, certificate = fit_block_global(objective, spec, check=check,
                seconds=args.seconds_per_family, max_cells=args.max_cells)
            row.update(status="global_uniqueness_and_gap_passed", certificate=certificate)
        except M4CertificateUnverified as error:
            row.update(status="global_certificate_unresolved", reason=str(error), diagnostics=error.diagnostics)
        except (ArithmeticError, ValueError) as error:
            row.update(status="numerical_error", reason=str(error))
        rows.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
    report = dict(scope="remaining_families_on_cached_first_Task_A_inputs", rows=rows,
        elapsed_seconds=time.monotonic()-started, Task_B_scores_or_predictions_published=False,
        audit_values_read=False, production_acceptance_changed=False)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
