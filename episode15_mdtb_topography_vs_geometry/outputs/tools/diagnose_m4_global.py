#!/usr/bin/env python3
"""First Task-A-only global search, with a private reusable objective cache.

Mapping mode always preserves the original complete-panel refusal and writes
no Task-B score/prediction. Cache mode reads only the explicitly supplied cache
made by mapping mode. Neither mode activates production or opens audit data.
"""
import argparse
import os
from pathlib import Path
import resource
import signal
import time

import run_development_point_panel as runner


CACHE_FIELDS = {
    "source_task_a", "target_task_a", "B", "O", "global_gain", "condition_weights",
    "mode_matrices", "rank_bound", "dof_bound", "rank_rtol", "numerical_atol",
    "spectral_norm_bound", "variant",
}


def _save_cache(directory, positional, keywords, specification):
    import numpy as np
    directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    directory.chmod(0o700)
    values = dict(zip(("source_task_a", "target_task_a", "B", "O"), positional))
    values.update(keywords, spectral_norm_bound=specification.spectral_norm_bound,
                  variant=specification.name)
    if set(values) != CACHE_FIELDS:
        raise ValueError("cache must contain exactly the first Task-A objective inputs")
    path = directory / "task_a_objective.npz"
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        np.savez(stream, **values)
    return path


def _load_cache(path):
    import numpy as np
    from paired_stretch_variants import PairedM4TaskAObjective, stretch_specifications
    if (path.name != "task_a_objective.npz" or path.stat().st_size > 2*1024**2
            or path.stat().st_mode & 0o077 or path.parent.stat().st_mode & 0o077):
        raise ValueError("explicit small private Task-A cache with 0700/0600 protection required")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != CACHE_FIELDS:
            raise ValueError("unrecognized Task-A cache contents")
        data = {key: np.array(archive[key], copy=True) for key in archive.files}
    specification = stretch_specifications()[0]
    if (str(data.pop("variant")) != specification.name
            or float(data.pop("spectral_norm_bound")) != specification.spectral_norm_bound
            or not np.array_equal(data["mode_matrices"], specification.modes)):
        raise ValueError("cache is not the canonical diagonal model")
    positional = [data.pop(key) for key in ("source_task_a", "target_task_a", "B", "O")]
    for key in ("rank_bound", "dof_bound"):
        data[key] = int(data[key])
    for key in ("global_gain", "rank_rtol", "numerical_atol"):
        data[key] = float(data[key])
    return PairedM4TaskAObjective(*positional, **data), specification


def _attempt(objective, specification, args, check, operator=None):
    from diagonal_m4_global import fit_diagonal_global
    from paired_stretch_variants import M4CertificateUnverified
    result = dict(status="started_first_Task_A_global_search",
                  fit_method="normalized_linear_primal_and_global_profile_or_certified_balanced_vertex",
                  full_domain_positive_curvature_passed=False,
                  operator_locality_check_exercised=False)
    try:
        numerical, certificate = fit_diagonal_global(objective, specification, check=check,
            seconds=args.search_seconds, max_cells=args.max_cells)
        if operator is not None:
            operator(objective.O, numerical["C"], specification)
            result["operator_locality_check_exercised"] = True
        # Coefficients and target identity remain in worker memory only.
        result.update(status="global_uniqueness_and_gap_passed", certificate=certificate)
    except M4CertificateUnverified as error:
        result.update(status="global_certificate_unresolved", reason=str(error),
                      diagnostics=error.diagnostics, nonuniqueness_established=False)
    except (ArithmeticError, ValueError) as error:
        result.update(status="numerical_or_resource_error", reason=str(error))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-json", required=True, type=Path)
    cache = parser.add_mutually_exclusive_group(required=True)
    cache.add_argument("--cache-dir", type=Path)
    cache.add_argument("--reuse-cache", type=Path)
    parser.add_argument("--search-seconds", type=float, default=300.)
    parser.add_argument("--max-cells", type=int, default=4096)
    args, remaining = parser.parse_known_args(argv)
    runner._compute()
    output = runner._path(args.diagnostic_json)
    scope = runner.OUTPUTS / "runs/observed_beta_development"
    if output.exists() or not output.parent.is_dir() or scope not in output.parents:
        parser.error("a fresh aggregate diagnostic under observed_beta_development is required")
    selected = runner._path(args.cache_dir if args.cache_dir is not None else args.reuse_cache)
    if scope not in selected.parents:
        parser.error("the private cache must remain inside this episode's observed-beta route")
    if args.cache_dir is not None and (selected.exists() or not selected.parent.is_dir()):
        parser.error("cache directory must be fresh under an existing episode directory")
    if args.search_seconds <= 0 or args.max_cells <= 0:
        parser.error("positive finite search limits required")
    captured, panel_exit = {}, None
    if args.reuse_cache is not None:
        small = argparse.ArgumentParser(add_help=False)
        small.add_argument("--deadline-epoch", required=True, type=float)
        small.add_argument("--rss-limit-kib", required=True, type=int)
        limits = small.parse_args(remaining)
        if not limits.deadline_epoch > time.time() or limits.rss_limit_kib <= 0:
            parser.error("an unexpired finite deadline and positive RSS bound are required")
        def check():
            runner._require(time.time() < limits.deadline_epoch and
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= limits.rss_limit_kib)
        def stop(_signal, _frame):
            raise runner.ExecutionRefusal()
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
        try:
            for sig in previous:
                signal.signal(sig, stop)
            signal.setitimer(signal.ITIMER_REAL, limits.deadline_epoch-time.time())
            check()
            objective, specification = _load_cache(selected)
            check()
            captured = _attempt(objective, specification, args, check)
        except (ArithmeticError, ValueError) as error:
            captured = dict(status="numerical_or_resource_error", reason=str(error))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    else:
        import proposed_development_point_panel as adapter
        from paired_stretch_variants import M4CertificateUnverified
        original_bound = adapter.proposed_curvature_certificate
        original_objective = adapter.PairedM4TaskAObjective
        original_init = adapter.ProposedPairedStretchVariants.__init__
        operator = None

        def objective_with_arguments(*positional, **keywords):
            value = original_objective(*positional, **keywords)
            value._global_diagnostic_arguments = (positional, keywords)
            return value

        def initialize(instance, *positional, **keywords):
            nonlocal operator
            original_init(instance, *positional, **keywords)
            operator = instance.operator_certificate

        def inspect(objective, specification, *, check):
            try:
                return original_bound(objective, specification, check=check)
            except M4CertificateUnverified:
                if not captured:
                    check()
                    positional, keywords = objective._global_diagnostic_arguments
                    path = _save_cache(selected, positional, keywords, specification)
                    captured.update(_attempt(objective, specification, args, check, operator))
                    captured["private_Task_A_cache"] = str(path)
                raise

        adapter.PairedM4TaskAObjective = objective_with_arguments
        adapter.ProposedPairedStretchVariants.__init__ = initialize
        adapter.proposed_curvature_certificate = inspect
        try:
            panel_exit = runner.main(remaining)
        finally:
            adapter.PairedM4TaskAObjective = original_objective
            adapter.ProposedPairedStretchVariants.__init__ = original_init
            adapter.proposed_curvature_certificate = original_bound
    report = dict(status="first_Task_A_global_diagnostic", original_panel_exit=panel_exit,
        production_acceptance_changed=False, Task_B_scores_or_predictions_published=False,
        audit_values_read=False, cache_contains_only_Task_A_objective_inputs=True,
        diagnostic=captured)
    runner._write_new(output, runner._encoded(report, runner.REPORT_BYTES))
    return 0 if captured else 2


if __name__ == "__main__":
    raise SystemExit(main())
