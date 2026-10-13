"""Four fixed EP07 finalist recoveries, separate from all empirical readers.

These cases identify a single shared population signal. Independent native
contrast noise supplies rank16; its orientation is neither identified nor
claimed recovered. Drift has a declared day-mass pooling oracle. Scale means
a physical population-amplitude change on every day, not target-only scaling.
The actual selected recipes, member weights and optimizer seeds stay fixed.

Scientific imports occur inside functions so CLI allocation/path admission
precedes them. The CLI opens only the selected recipe JSON and its exclusive
report. Generated count truth never enters feature-only prediction packs.
"""
import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import socket
import time


CASES = ("stable", "drift", "null", "scale")
STREAMS = ("history_assisted", "today_only")
SEED = 7007400
NRMSE_LIMIT = .20
DIAGNOSTIC_TOLERANCE = 1e-8


class UnsupportedKnownAnswerRecipe(ValueError):
    """The prospective construction/bias bound does not qualify this recipe."""


class KnownAnswerConstructionError(ValueError):
    """A fixed construction or its assumed fitted coordinate binding failed."""


@dataclass(frozen=True)
class CaseCell:
    region: str
    stream: str
    fit_days: tuple
    development: object
    evaluation_counts: object
    h: object
    x: object


@dataclass(frozen=True)
class KnownAnswerCase:
    name: str
    cells: tuple[CaseCell, ...]


@dataclass(frozen=True)
class RecoveryResult:
    case: str
    region: str
    stream: str
    finalist_id: str
    oracle_h_amplitude: float
    oracle_beta: float
    prediction_beta: float
    nrmse: float
    slope_error: float
    slope_limit: float
    passed: bool


@dataclass(frozen=True)
class KnownAnswerReport:
    state: str
    results: tuple[RecoveryResult, ...]
    attempted_member_fits: int
    completed_member_predictions: int
    feasibility: tuple


def _check(check):
    if not callable(check):
        raise TypeError("caller-owned cancellation/resource check is required")
    check()


def member_oracle(member, case, *, stream):
    """Return (cell-mean amplitude, trial slope) without fitting another model."""
    if case not in CASES or stream not in STREAMS:
        raise ValueError("one fixed case and stream are required")
    if case == "scale":
        return 8., 16.
    if case == "null":
        return 4., 0.
    if case == "stable":
        return 4., 8.
    if stream == "today_only":
        return 4., -8.
    if member.weight_recipe is None:
        source = math.fsum(member.recipe.source_masses)
        target = member.recipe.calibration_mass
    else:
        source = member.weight_recipe.source_total_mass
        target = member.weight_recipe.target_mass
    return 4., 8. * (source - target) / (source + target)


def _representer_penalty(member, amplitude, beta):
    """Penalty of a known exact conditional-mean representer, not a refit.

    Five x bands and four h bands each give two centered informative columns
    +/-sign/2 after robust scaling. The target PCA signal loading is ones/4.
    Residualization restores the known cell mean separately, requiring no h
    coefficient. All remaining response coordinates have conditional mean0.
    """
    recipe = member.recipe
    feature, regression = recipe.features, recipe.regression
    valid_feature = (feature.scaling == "robust" and (
        (feature.family == "moments_quantiles" and feature.quantiles == (.25, .5, .75))
        or (feature.family == "robust_histogram" and feature.histogram_bins in (4, 8)
            and feature.histogram_quantiles == (.01, .99))))
    if (not valid_feature or recipe.adjustment not in ("explicit", "residualize", "both")
            or not regression.fit_intercept):
        raise UnsupportedKnownAnswerRecipe("fixed feature/mean representer is unsupported")
    h_coefficient = 0. if recipe.adjustment in ("residualize", "both") else amplitude
    qx, qh = 4. * beta, 4. * h_coefficient
    squared = 4. * qx * qx / 10. + 4. * qh * qh / 8.
    if regression.family in ("ridge", "reduced_rank_ridge"):
        return .5 * regression.penalty * squared
    if regression.family == "elastic_net":
        absolute = 2. * (abs(qx) + abs(qh))
        return regression.penalty * (regression.l1_ratio * absolute
            + .5 * (1. - regression.l1_ratio) * squared)
    if (regression.family == "shallow_network" and regression.activation == "relu"
            and regression.width >= 4):
        # A positive/negative ReLU pair represents each of the two signs;
        # balance input/output scales to minimize this representer's L2 cost.
        return 2. * regression.penalty * (
            abs(qx) / (math.sqrt(10.) / 2.) + abs(qh) / (math.sqrt(8.) / 2.))
    raise UnsupportedKnownAnswerRecipe("only the selected linear/ReLU grammar has a fixed bound")


def qualify_finalists(finalists):
    """Check exact declarations and the fixed approximation bound pre-generation.

    At a global penalized optimum, excess conditional prediction MSE is at
    most twice the feasible representer's penalty. This is a feasibility
    bound, never a convergence claim for fixed-epoch Adam. For an ensemble,
    use the triangle inequality on its members' native RMS bounds.
    """
    from finalist_controls import FinalistSpec, resolve_finalist
    if (not isinstance(finalists, tuple) or len(finalists) != 2
            or any(not isinstance(value, FinalistSpec) for value in finalists)
            or tuple(value.stream for value in finalists) != STREAMS):
        raise ValueError("the exact history/today finalist pair is required")
    records = []
    for finalist in finalists:
        if finalist != resolve_finalist(finalist.candidate_id, stream=finalist.stream):
            raise ValueError("finalist recipes/member weights differ from their fixed declaration")
        for case in CASES:
            oracle = tuple(member_oracle(member, case, stream=finalist.stream)
                           for member in finalist.members)
            amplitude = math.fsum(member.weight * value[0]
                                  for member, value in zip(finalist.members, oracle))
            beta = math.fsum(member.weight * value[1]
                            for member, value in zip(finalist.members, oracle))
            rms_bound = math.fsum(member.weight * math.sqrt(
                2. * _representer_penalty(member, *value) / 16.)
                for member, value in zip(finalist.members, oracle))
            fraction = rms_bound * rms_bound / (amplitude * amplitude + beta * beta)
            if not math.isfinite(fraction) or fraction > NRMSE_LIMIT**2:
                raise UnsupportedKnownAnswerRecipe("prospective penalty bound exceeds the fixed recovery gate")
            records.append((finalist.stream, case, fraction))
    return tuple(records)


def _arrays(region, repeats, beta, scale):
    import numpy as np
    offsets = np.arange(5, 15) if region == "M1" else np.arange(-15, 0)
    directions = np.repeat(np.arange(8), repeats)
    t = np.tile(np.arange(repeats) % 16, 8)
    x = np.where(t % 2 == 0, 1, -1)
    h = np.where((directions[:, None] + np.arange(len(offsets))[None, :]) % 2 == 0, 1, -1)
    walsh = np.array([[1 if (int(value) & mask).bit_count() % 2 == 0 else -1
                       for mask in range(2, 16)] for value in t], dtype=np.int64)
    q = np.empty((len(t), len(offsets), 15), dtype=np.int64)
    q[:, :, :14] = walsh[:, None, :]
    q[:, :, 14] = h * walsh[:, 0, None]
    noise = np.concatenate((q, -np.sum(q, axis=2, keepdims=True)), axis=2)
    mean = 4 * h + beta * x[:, None]
    counts = 64 + scale * (mean[:, :, None] + noise)
    bands = []
    for band in range(9):
        sign = np.broadcast_to(x[:, None], h.shape) if band % 2 == 0 else h
        bands.append(np.where(np.arange(16)[None, None, :] < (8 + 2 * sign[:, :, None]),
                              1., -1.).astype(np.float32))
    lfp = np.concatenate(bands, axis=2)
    return directions, offsets, x, h, counts, noise, lfp


def _construction_check(directions, x, h, counts, noise, *, beta, scale, check):
    import numpy as np
    _check(check)
    if counts.dtype.kind not in "iu" or np.any(counts < 0):
        raise KnownAnswerConstructionError("counts are not native nonnegative integers")
    if not np.array_equal(counts, 64 + scale * (4 * h[:, :, None]
            + beta * x[:, None, None] + noise)) or np.any(np.sum(noise, axis=2)):
        raise KnownAnswerConstructionError("conditional signal/contrast-noise construction changed")
    for direction in range(8):
        _check(check)
        for sign in (-1, 1):
            selected = (directions == direction) & (x == sign)
            if not np.any(selected) or np.any(np.sum(noise[selected], axis=0)):
                raise KnownAnswerConstructionError("noise conditional means must be exactly zero")
    # The declared native covariance has a unique leading common direction,
    # plus15 supported contrast directions. Check the small native Gram only.
    centered = counts.reshape(-1, 16).astype(np.float64) - 64.
    covariance = centered.T @ centered / len(centered)
    eigenvalues = np.linalg.eigvalsh(covariance)
    signal_variance = 16. * scale * scale * (16. + beta * beta)
    if (eigenvalues[0] <= .5 * scale * scale
            or abs(eigenvalues[-1] - signal_variance) > DIAGNOSTIC_TOLERANCE
            or not np.allclose(covariance @ np.ones(16), signal_variance * np.ones(16),
                               rtol=0, atol=DIAGNOSTIC_TOLERANCE)):
        raise KnownAnswerConstructionError("full native rank/leading signal is not the declared construction")


def generate_case(name, *, check):
    """Generate one immutable fixed case; no empirical data or fitted state."""
    import numpy as np
    from native_trials import BANDS, DayTrials, EARLIER, _snapshot
    _check(check)
    if name not in CASES:
        raise ValueError("only the four fixed case names are allowed")
    dates = EARLIER["Mihili"] + ("20140304",)
    rng = np.random.Generator(np.random.PCG64(SEED))
    permutations = tuple((rng.permutation(128), rng.permutation(16), rng.permutation(16))
                         for _ in dates)
    evaluation_order = rng.permutation(256)
    scale = 2 if name == "scale" else 1
    cells = []
    for region in ("M1", "PMd"):
        history = []
        for index, day in enumerate(dates):
            _check(check)
            beta = 0 if name == "null" else -8 if name == "drift" and index == 3 else 8
            directions, offsets, x, h, counts, noise, lfp = _arrays(region, 16, beta, scale)
            _construction_check(directions, x, h, counts, noise, beta=beta, scale=scale, check=check)
            rows, neurons, electrodes = permutations[index]
            columns = np.concatenate([16 * band + electrodes for band in range(9)])
            guide = np.array([(1000 * (index + 1) + electrode, low, high)
                              for low, high in BANDS for electrode in range(16)])
            units = np.column_stack((1000 * (index + 1) + np.arange(16), np.ones(16, dtype=int)))
            ids = tuple(f"trial_{int(row) + 1:06d}" for row in rows)
            history.append(DayTrials("Mihili", day, "20140304", region, "history_assisted",
                "fit", "calibration" if index == 3 else "training", ids,
                _snapshot(directions[rows]), _snapshot(offsets),
                _snapshot(lfp[rows][:, :, columns]), _snapshot(guide[columns]),
                _snapshot(counts[rows][:, :, neurons]), _snapshot(units[neurons])))
        beta = 0 if name == "null" else -8 if name == "drift" else 8
        directions, offsets, x, h, counts, noise, lfp = _arrays(region, 32, beta, scale)
        _construction_check(directions, x, h, counts, noise, beta=beta, scale=scale, check=check)
        _, neurons, electrodes = permutations[-1]
        columns = np.concatenate([16 * band + electrodes for band in range(9)])
        rows = evaluation_order
        development = DayTrials("Mihili", dates[-1], dates[-1], region, "history_assisted",
            "development_features", "development",
            tuple(f"trial_{int(row) + 1001:06d}" for row in rows),
            _snapshot(directions[rows]), _snapshot(offsets),
            _snapshot(lfp[rows][:, :, columns]), history[-1].lfp_guide, None, None)
        truths = (_snapshot(counts[rows][:, :, neurons]), _snapshot(h[rows]), _snapshot(x[rows]))
        for stream in STREAMS:
            fitting = tuple(history) if stream == STREAMS[0] else (replace(history[-1], stream=stream),)
            cells.append(CaseCell(region, stream, fitting, replace(development, stream=stream), *truths))
    return KnownAnswerCase(name, tuple(cells))


def _prediction_values(cell, prediction):
    import numpy as np
    from native_trials import NativePredictions
    target, development = cell.fit_days[-1], cell.development
    if (not isinstance(prediction, NativePredictions)
            or (prediction.animal, prediction.target_day, prediction.region, prediction.trial_ids,
                prediction.calibration_trial_ids) !=
               (target.animal, target.day, cell.region, development.trial_ids, target.trial_ids)
            or any(not np.array_equal(getattr(prediction, key), expected) for key, expected in (
                ("directions", development.directions), ("bin_offsets", development.bin_offsets),
                ("unit_guide", target.unit_guide)))):
        raise ValueError("prediction identities/native axes differ from this generated case")
    values = np.asarray(prediction.values)
    if values.shape != cell.evaluation_counts.shape or not np.isfinite(values).all():
        raise ValueError("finite complete native predictions are required")
    return values


def evaluate_prediction(cell, prediction, *, case, finalist):
    """Compare every native output with the fixed, centered scientific oracle."""
    import numpy as np
    if finalist.stream != cell.stream:
        raise ValueError("the finalist and generated stream must agree")
    values = _prediction_values(cell, prediction)
    oracles = tuple(member_oracle(member, case, stream=cell.stream) for member in finalist.members)
    amplitude = math.fsum(member.weight * value[0] for member, value in zip(finalist.members, oracles))
    beta = math.fsum(member.weight * value[1] for member, value in zip(finalist.members, oracles))
    centered = amplitude * cell.h[:, :, None] + beta * cell.x[:, None, None]
    mse = float(np.mean((values - (64. + centered))**2))
    denominator = float(np.mean(centered**2))
    nrmse = math.sqrt(mse / denominator)
    prediction_beta = float(np.mean(values * cell.x[:, None, None]))
    slope_error, slope_limit = abs(prediction_beta - beta), .20 * max(abs(beta), 1.)
    if not all(math.isfinite(value) for value in (nrmse, prediction_beta, slope_error)):
        raise FloatingPointError("nonfinite recovery statistic")
    return RecoveryResult(case, cell.region, cell.stream, finalist.candidate_id, amplitude, beta,
        prediction_beta, nrmse, slope_error, slope_limit,
        bool(nrmse <= NRMSE_LIMIT and slope_error <= slope_limit))


def _fitted_diagnostics(fitted, member, cell):
    import numpy as np
    primary = fitted if member.weight_recipe is None else fitted.primary
    for basis in primary.population.bases:
        if not np.allclose(basis.components[:, 0], np.full(16, .25), rtol=0, atol=DIAGNOSTIC_TOLERANCE):
            raise KnownAnswerConstructionError("fitted first coordinate is not the identified common signal")
    for mapping in primary.population.source_maps:
        expected = np.zeros(len(mapping))
        expected[0] = 1.
        if (not np.allclose(mapping[0], expected, rtol=0, atol=DIAGNOSTIC_TOLERANCE)
                or not np.allclose(mapping[:, 0], expected, rtol=0, atol=DIAGNOSTIC_TOLERANCE)):
            raise KnownAnswerConstructionError("fitted source map changes the identified common signal")
    if member.weight_recipe is not None:
        expected = np.full(3, member.weight_recipe.source_total_mass / 3.)
        if (not np.allclose(fitted.effective_recipe.source_masses, expected,
                            rtol=0, atol=DIAGNOSTIC_TOLERANCE)
                or fitted.effective_recipe.calibration_mass != member.weight_recipe.target_mass):
            raise KnownAnswerConstructionError("generated equal-reliability/similarity source weights changed")
        weights = fitted.source_weights
        if (np.asarray(weights.relative_weights).shape != (3,)
                or not np.allclose(weights.relative_weights, np.full(3, 1. / 3.),
                                   rtol=0, atol=DIAGNOSTIC_TOLERANCE)):
            raise KnownAnswerConstructionError("generated relative source weights are not equal")
        required = {"equal": (), "reliability": ("reliability",),
            "hierarchical_shrinkage": ("reliability", "shrunk_reliability"),
            "calibration_similarity": ("prototype_distances",)}[member.weight_recipe.family]
        for field, target in (("reliability", 1.), ("shrunk_reliability", 1.),
                              ("prototype_distances", 0.)):
            value = getattr(weights, field)
            if value is None:
                if field in required:
                    raise KnownAnswerConstructionError("required generated source-weight diagnostic is missing")
            elif (np.asarray(value).shape != (3,)
                    or not np.allclose(value, np.full(3, target), rtol=0, atol=DIAGNOSTIC_TOLERANCE)):
                raise KnownAnswerConstructionError("generated source-weight diagnostic differs from its known answer")


def run_known_answer_cases(finalists, *, check, progress=None):
    """Run each fixed case once, stopping on a failed recovery or exception."""
    from native_trials import _snapshot
    from primary_model import fit_primary_model, predict_primary_development
    from weighted_primary_model import fit_weighted_primary_model, predict_weighted_primary_development
    _check(check)
    feasibility = qualify_finalists(finalists)
    if progress is not None:
        progress({"stage": "recipe_bounds_qualified", "feasibility": feasibility})
    by_stream = {value.stream: value for value in finalists}
    results, attempted, completed = [], 0, 0
    for case in CASES:
        _check(check)
        generated = generate_case(case, check=check)
        for cell in generated.cells:
            finalist, combined, reference = by_stream[cell.stream], None, None
            for member in finalist.members:
                _check(check)
                attempted += 1
                if progress is not None:
                    progress({"stage": "member_fit", "case": case, "region": cell.region,
                        "stream": cell.stream, "member_id": member.candidate_id,
                        "attempted_member_fits": attempted, "completed_member_predictions": completed})
                fitted = prediction = None
                try:
                    if member.weight_recipe is None:
                        fitted = fit_primary_model(cell.fit_days, recipe=member.recipe, check=check)
                        prediction = predict_primary_development(fitted, cell.development, check=check)
                    else:
                        fitted = fit_weighted_primary_model(cell.fit_days, recipe=member.recipe,
                            weight_recipe=member.weight_recipe, check=check)
                        prediction = predict_weighted_primary_development(fitted, cell.development, check=check)
                    _fitted_diagnostics(fitted, member, cell)
                    # Validate each member against the same generated native
                    # identities before fixed-weight native averaging.
                    _prediction_values(cell, prediction)
                    reference = prediction
                    weighted = member.weight * prediction.values
                    combined = weighted.copy() if combined is None else combined + weighted
                    completed += 1
                    if progress is not None:
                        progress({"stage": "member_prediction_completed", "case": case,
                            "region": cell.region, "stream": cell.stream,
                            "member_id": member.candidate_id, "attempted_member_fits": attempted,
                            "completed_member_predictions": completed})
                finally:
                    fitted = prediction = None
            _check(check)
            result = evaluate_prediction(cell, replace(reference, values=_snapshot(combined)),
                                         case=case, finalist=finalist)
            results.append(result)
            if progress is not None:
                progress({"stage": "cell_completed", "result": asdict(result),
                    "attempted_member_fits": attempted, "completed_member_predictions": completed})
            if not result.passed:
                return KnownAnswerReport("scientific_recovery_failed", tuple(results), attempted, completed, feasibility)
        del generated
    return KnownAnswerReport("all_four_cases_passed", tuple(results), attempted, completed, feasibility)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    started_wall, started_cpu = time.time(), time.process_time()
    destination = Path(__file__).resolve().parents[2] / "outputs/runs/alternative_explanations/attempt001"
    if (not os.environ.get("SLURM_JOB_ID", "").isdigit() or "-ln" in socket.gethostname()
            or os.environ.get("SLURM_CPUS_PER_TASK") not in ("1", "2")
            or os.environ.get("SLURM_MEM_PER_NODE") != "7168"
            or any(os.environ.get(name) != "1" for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"))
            or not math.isfinite(args.deadline_epoch) or args.deadline_epoch <= started_wall):
        raise SystemExit("known-answer execution requires the selected bounded compute allocation")
    if (args.selection.resolve() != destination / "selected_pair.json"
            or args.out.resolve() != destination / "known_answer_report.json"
            or not destination.is_dir() or args.selection.stat().st_size > 1024**2):
        raise SystemExit("selection/output differ from the fixed finalist attempt")
    selection = json.loads(args.selection.read_text())
    if (selection.get("evidence_stage") != "development_only"
            or selection.get("held_out_opened") is not False
            or selection.get("scientific_checks_complete") is not False
            or set(selection.get("finalists", {})) != set(STREAMS)):
        raise SystemExit("selection is not the unopened development-only finalist pair")
    deadline = min(args.deadline_epoch, started_wall + 600.)
    def check():
        if time.time() >= deadline:
            raise TimeoutError("fixed known-answer deadline reached")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > int(6.5 * 2**30):
            raise MemoryError("fixed known-answer RSS limit exceeded")
    status = {"state": "running", "stage": "selected_recipe_admission", "started_utc": _utc(),
        "job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
        "selection": str(args.selection), "animal": "Mihili", "target_day": "20140304",
        "allocated_cpus": int(os.environ["SLURM_CPUS_PER_TASK"]), "numerical_threads": 1,
        "internal_limit_seconds": 600, "deadline_epoch": deadline,
        "cases": list(CASES), "seed": SEED, "evidence_stage": "generated_known_answer_only",
        "held_out_opened": False, "scientific_checks_complete": False,
        "attempted_member_fits": 0, "completed_member_predictions": 0, "results": []}
    with args.out.open("x") as handle:
        def save():
            status["updated_utc"] = _utc()
            status["wall_seconds"] = time.time() - started_wall
            status["process_cpu_seconds"] = time.process_time() - started_cpu
            status["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            # Keep the last complete snapshot if a later write is interrupted.
            # The original exclusive handle identifies our claimed path after
            # replacement; a failed temporary file is retained, never retried.
            target = Path(handle.name)
            temporary = target.with_name(target.name + ".tmp")
            with temporary.open("x") as output:
                json.dump(status, output, indent=2, allow_nan=False)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, target)
        save()
        try:
            check()
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            from finalist_controls import resolve_finalist
            finalists = tuple(resolve_finalist(selection["finalists"][stream]["candidate_id"], stream=stream)
                               for stream in STREAMS)
            for finalist in finalists:
                expected = json.loads(json.dumps(asdict(finalist), allow_nan=False))
                if selection["finalists"][finalist.stream].get("spec") != expected:
                    raise ValueError("selected serialized spec differs from its unchanged declared recipe")
            status["finalists"] = {value.stream: value.candidate_id for value in finalists}
            def progress(event):
                event = dict(event)
                result = event.pop("result", None)
                status.update(event)
                if result is not None:
                    status["results"].append(result)
                save()
            result = run_known_answer_cases(finalists, check=check, progress=progress)
            status.update(state=result.state, stage="finished", feasibility=result.feasibility,
                attempted_member_fits=result.attempted_member_fits,
                completed_member_predictions=result.completed_member_predictions)
            status["known_answer_recovery_complete"] = result.state == "all_four_cases_passed"
            save()
        except Exception as error:
            status.update(state="unresolved_qualification", error_type=type(error).__name__,
                          error=str(error), known_answer_recovery_complete=False)
            save()
            return 1
    return 0 if status["state"] == "all_four_cases_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
