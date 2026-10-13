"""Fixed EP07 primary arithmetic over already aligned frozen predictions.

There is no source/held-out reader, prediction transport, fitting or model
selection here. Explicit held_out metadata is a procedural binding, not proof
of source provenance or permission to open outcomes. In particular, this
module cannot relabel development-only predictor inputs as held-out inputs.
Final-opening authority and scientific/finalist checks remain external.

Joint R2 pools SSE and neuron-centered SST over all native bins and trials,
retaining constant neurons and negative scores. Each bootstrap resamples whole
trials within day/direction; predictions stay frozen and SST is recomputed.
The local PCG64(7007999) schedule has exactly 9999 draws, ordered draw -> day
(Mihili then Chewie, chronological) -> direction0..7. Within each stratum,
trial IDs are sorted by numeric suffix before Generator.integers int64 draws.
The same trial multiplicities serve both regions and all three predictors.

The one-sided centered bootstrap p is (1 + count[Delta* - Delta_hat >=
Delta_hat - .01])/10000, including ties. Regional max(p_today,p_mean) is the
intersection-union test; Holm uses the fixed two-region family. An unavailable
region has an internal non-rejectable p=1 placeholder, never a reported valid
p-value. These are approximate bootstrap tests conditional on fixed fitted
models, five days and two animals, not new-day/animal population inference.

Undefined original scores make a region unestimable; ANY undefined bootstrap
contrast makes its inference unavailable. Draws are never deleted or replaced,
and no denominator is clipped or epsilon-stabilized. Sign/frequency/omission
requirements use observed estimates only and do not add confidence claims.
"""
from dataclasses import dataclass
import re

import numpy as np

from native_trials import NativePredictions


DAY_ORDER = (("Mihili", "20140304"), ("Mihili", "20140307"),
             ("Chewie-L", "20161007"), ("Chewie-L", "20161014"), ("Chewie-L", "20161021"))
REGIONS = ("M1", "PMd")
COMPARATORS = ("today_only", "calibration_mean")
MODELS = ("history_assisted",) + COMPARATORS
BOOTSTRAP_SAMPLES = 9999
BOOTSTRAP_SEED = 7007999
MARGIN = .01
ALPHA = .05
_BATCH_SIZE = 256


@dataclass(frozen=True)
class HeldOutCell:
    animal: str
    day: str
    region: str
    role: str
    trial_ids: tuple[str, ...]
    directions: object
    bin_offsets: object
    unit_guide: object
    observed: object
    history: NativePredictions
    today: NativePredictions
    calibration_mean: NativePredictions


@dataclass(frozen=True)
class ObservedGates:
    """Array axes: animal=(Mihili,Chewie), region=REGIONS, comparator=COMPARATORS."""
    animal_differences: np.ndarray             # [2,2,2]
    aggregate_differences: np.ndarray          # [2,2]
    margin_pass: np.ndarray                    # [2]
    animal_sign_pass: np.ndarray               # [2]
    positive_day_counts: np.ndarray            # [2,2], same-day BOTH comparators
    day_frequency_pass: np.ndarray            # [2]
    omission_animal_differences: np.ndarray    # [5,2,2,2]
    omission_aggregate_differences: np.ndarray # [5,2,2]
    omission_pass: np.ndarray                  # [5,2]


@dataclass(frozen=True)
class MarginInference:
    available: np.ndarray                     # [2]
    valid_draws: np.ndarray                   # [2]
    comparator_p: np.ndarray                  # [2,2], NaN when unavailable
    regional_p: np.ndarray                    # [2], NaN when unavailable
    holm_input_p: np.ndarray                  # [2], unavailable placeholder=1
    holm_adjusted_p: np.ndarray               # [2], NaN when unavailable
    holm_rejected: np.ndarray                 # [2], unavailable always False


@dataclass(frozen=True)
class PrimaryInference:
    day_order: tuple
    regions: tuple
    comparators: tuple
    trial_ids: tuple                          # one numerically ordered tuple per day
    day_scores: np.ndarray                    # [5,2,3], MODELS order
    day_differences: np.ndarray               # [5,2,2]
    bootstrap_differences: np.ndarray         # [9999,2,2]
    observed: ObservedGates
    margins: MarginInference
    unavailable_reasons: tuple                # one tuple per region
    arithmetic_pass: np.ndarray               # [2]; external checks NOT included
    bootstrap_seed: int = BOOTSTRAP_SEED
    bootstrap_samples: int = BOOTSTRAP_SAMPLES
    rng_algorithm: str = "PCG64"
    scientific_checks_certified: bool = False
    authorizes_opening: bool = False


@dataclass(frozen=True)
class _BoundCell:
    cell: HeldOutCell
    order: np.ndarray
    trial_ids: tuple
    directions: np.ndarray
    shape: tuple
    calibration_ids: tuple


@dataclass(frozen=True)
class _TrialStatistics:
    sse: np.ndarray                # [trial,model]
    centered_sum: np.ndarray       # [trial,neuron]
    centered_square_sum: np.ndarray # [trial], summed across native bins/neurons
    bins: int


def _check(check):
    if not callable(check):
        raise TypeError("caller-owned cancellation/resource check required")
    check()


def _readonly(value):
    value = np.ascontiguousarray(value)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _ids(values):
    if (not isinstance(values, tuple) or not values
            or any(not isinstance(value, str) or re.fullmatch(r"trial_[0-9]{6,}", value) is None
                   for value in values)):
        raise ValueError("explicit native whole-trial identities are required")
    numbers = tuple(int(value[6:]) for value in values)
    if min(numbers) < 1 or len(set(numbers)) != len(numbers):
        raise ValueError("whole-trial IDs must have unique positive numeric identities")
    order = np.asarray(sorted(range(len(values)), key=numbers.__getitem__), dtype=np.int64)
    return numbers, order, tuple(values[i] for i in order)


def _bind_cell(cell):
    if not isinstance(cell, HeldOutCell):
        raise TypeError("explicit HeldOutCell objects required")
    if cell.role != "held_out":
        raise PermissionError("primary inference requires explicitly bound held-out cells")
    if (cell.animal, cell.day) not in DAY_ORDER or cell.region not in REGIONS:
        raise ValueError("cell is outside the fixed five-day/two-region panel")
    numbers, order, ids = _ids(cell.trial_ids)
    directions = np.asarray(cell.directions)
    if (directions.dtype.kind not in "iu" or directions.shape != (len(ids),)
            or np.any((directions < 0) | (directions > 7))
            or np.any(np.bincount(directions.astype(np.int64), minlength=8) < 6)):
        raise ValueError("all eight directions need at least six complete held-out trials")
    offsets = np.asarray(cell.bin_offsets)
    expected = np.arange(5, 15) if cell.region == "M1" else np.arange(-15, 0)
    if offsets.dtype.kind not in "iu" or not np.array_equal(offsets, expected):
        raise ValueError("the exact native region window is required")
    units = np.asarray(cell.unit_guide)
    if (units.dtype.kind not in "fiu" or units.ndim != 2 or units.shape[1] != 2
            or len(units) == 0 or not np.isfinite(units).all()
            or len({tuple(row) for row in units.tolist()}) != len(units)):
        raise ValueError("complete unique native neuron identities required")
    calibration = None
    for prediction in (cell.history, cell.today, cell.calibration_mean):
        if not isinstance(prediction, NativePredictions):
            raise TypeError("each frozen predictor must retain its NativePredictions identity binding")
        if ((prediction.animal, prediction.target_day, prediction.region) !=
                (cell.animal, cell.day, cell.region) or prediction.trial_ids != cell.trial_ids
                or not np.array_equal(prediction.directions, directions)
                or not np.array_equal(prediction.bin_offsets, offsets)
                or not np.array_equal(prediction.unit_guide, units)):
            raise ValueError("observed and predicted trial/direction/native axes differ")
        cal_numbers, _, cal_ids = _ids(prediction.calibration_trial_ids)
        if len(cal_ids) < 24 or set(numbers).intersection(cal_numbers):
            raise ValueError("calibration IDs must be complete and disjoint from held-out trials")
        if calibration is not None and calibration != cal_ids:
            raise ValueError("all frozen predictors must use the same target calibration support")
        calibration = cal_ids
    return _BoundCell(cell, order, ids, _readonly(directions[order]),
                      (len(ids), len(offsets), len(units)), calibration)


def _bind_panel(cells, check):
    cells = tuple(cells)
    if len(cells) != 10:
        raise ValueError("all five days and both regions are required without omission")
    bound = {}
    for cell in cells:
        _check(check)
        value = _bind_cell(cell)
        key = cell.animal, cell.day, cell.region
        if key in bound:
            raise ValueError("duplicate day/region cell")
        bound[key] = value
    if set(bound) != {(animal, day, region) for animal, day in DAY_ORDER for region in REGIONS}:
        raise ValueError("incomplete fixed panel")
    result = []
    for animal, day in DAY_ORDER:
        pair = tuple(bound[animal, day, region] for region in REGIONS)
        if (pair[0].trial_ids != pair[1].trial_ids
                or not np.array_equal(pair[0].directions, pair[1].directions)
                or pair[0].calibration_ids != pair[1].calibration_ids):
            raise ValueError("M1 and PMd must retain the same whole-trial roles and directions")
        result.append(pair)
    return tuple(result)


def _values(value, shape, *, counts):
    array = np.asarray(value)
    if (array.dtype.kind not in "fiu" or array.shape != shape or not np.isfinite(array).all()
            or (counts and (np.any(array < 0) or np.any(array != np.floor(array))))):
        raise ValueError("finite full native arrays required; observations must be nonnegative integer counts")
    array = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError("native arrays must be finite in float64")
    return array


def _trial_statistics(bound, check):
    """Compress whole trials, preserving the bootstrap mean correction in SST."""
    _check(check)
    cell = bound.cell
    y = _values(cell.observed, bound.shape, counts=True)[bound.order]
    with np.errstate(over="ignore", invalid="ignore"):
        z = y - np.mean(y, axis=(0, 1), keepdims=True)
        sums = np.sum(z, axis=1)
        squares = np.sum(z * z, axis=(1, 2))
    errors = []
    for prediction in (cell.history, cell.today, cell.calibration_mean):
        _check(check)
        p = _values(prediction.values, bound.shape, counts=False)[bound.order]
        with np.errstate(over="ignore", invalid="ignore"):
            errors.append(np.sum((y - p)**2, axis=(1, 2)))
    return _TrialStatistics(_readonly(np.column_stack(errors)), _readonly(sums),
                            _readonly(squares), bound.shape[1])


def _score_multiplicities(statistics, counts):
    """Each row of counts represents complete duplicated whole-trial occurrences."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        sse = counts @ statistics.sse
        sums = counts @ statistics.centered_sum
        sst = counts @ statistics.centered_square_sum - np.sum(sums * sums, axis=1) / (
            statistics.bins * np.sum(counts, axis=1))
        scores = 1 - sse / sst[:, None]
        differences = (sse[:, 1:] - sse[:, :1]) / sst[:, None]
    valid = ((sst > 0) & np.isfinite(sst) & np.isfinite(sse).all(axis=1)
             & np.isfinite(scores).all(axis=1) & np.isfinite(differences).all(axis=1))
    scores[~valid] = np.nan
    differences[~valid] = np.nan
    return scores, differences, valid


def _animal_aggregate(differences):
    with np.errstate(over="ignore", invalid="ignore"):
        animal = np.stack((np.mean(differences[:2], axis=0), np.mean(differences[2:], axis=0)))
        aggregate = np.mean(animal, axis=0)
    animal[~np.isfinite(animal)] = np.nan
    aggregate[~np.isfinite(aggregate)] = np.nan
    return animal, aggregate


def _observed_gates(differences):
    animal, aggregate = _animal_aggregate(differences)
    same_day = np.all(differences > 0, axis=2)
    positive_counts = np.stack((np.sum(same_day[:2], axis=0), np.sum(same_day[2:], axis=0)))
    omission_animal, omission_aggregate = [], []
    for omitted in range(5):
        with np.errstate(over="ignore", invalid="ignore"):
            means = np.stack([np.mean(differences[[day for day in group if day != omitted]], axis=0)
                              for group in ((0, 1), (2, 3, 4))])
            mean = np.mean(means, axis=0)
        means[~np.isfinite(means)] = np.nan
        mean[~np.isfinite(mean)] = np.nan
        omission_animal.append(means)
        omission_aggregate.append(mean)
    omission_animal, omission_aggregate = np.stack(omission_animal), np.stack(omission_aggregate)
    omission_pass = (np.all(omission_animal > 0, axis=(1, 3))
                     & np.all(omission_aggregate > 0, axis=2))
    return ObservedGates(*map(_readonly, (
        animal, aggregate, np.all(aggregate > MARGIN, axis=1), np.all(animal > 0, axis=(0, 2)),
        positive_counts, (positive_counts[0] == 2) & (positive_counts[1] >= 2),
        omission_animal, omission_aggregate, omission_pass)))


def _margin_inference(point, draws):
    """Fixed 9999-draw centered upper tail; unavailable regions retain Holm slots."""
    if point.shape != (2, 2) or draws.shape != (BOOTSTRAP_SAMPLES, 2, 2):
        raise ValueError("fixed region/comparator/bootstrap axes required")
    finite = np.isfinite(draws).all(axis=2)
    valid_draws = np.sum(finite, axis=0)
    available = np.isfinite(point).all(axis=1) & (valid_draws == BOOTSTRAP_SAMPLES)
    comparator_p = np.full((2, 2), np.nan)
    for region in range(2):
        if available[region]:
            with np.errstate(over="ignore", invalid="ignore"):
                error = draws[:, region] - point[region]
                observed = point[region] - MARGIN
            if not np.isfinite(error).all() or not np.isfinite(observed).all():
                available[region] = False
                continue
            comparator_p[region] = (1 + np.sum(error >= observed, axis=0)) / (BOOTSTRAP_SAMPLES + 1)
    regional = np.max(comparator_p, axis=1)
    holm_input = np.where(available, regional, 1.)
    order = np.argsort(holm_input, kind="stable")  # M1 first in a tie, same adjusted result.
    ordered_adjusted = np.minimum(1., np.maximum.accumulate(np.asarray([2., 1.]) * holm_input[order]))
    adjusted = np.empty(2)
    adjusted[order] = ordered_adjusted
    rejected = available & (adjusted <= ALPHA)
    adjusted[~available] = np.nan
    return MarginInference(*map(_readonly, (available, valid_draws, comparator_p,
        regional, holm_input, adjusted, rejected)))


def infer_primary(cells, *, check):
    """Evaluate the fixed complete panel; returned pass covers arithmetic only.

    All cell/predictor metadata and cross-region identities bind before any
    response/prediction value array is interpreted. Calibration IDs authenticate
    only identity/count/disjointness; they cannot verify per-direction support
    or prove the supplied template was fitted from calibration alone.
    """
    _check(check)
    panel = _bind_panel(cells, check)
    statistics = tuple(tuple(_trial_statistics(cell, check) for cell in pair) for pair in panel)
    point_scores = np.full((5, 2, 3), np.nan)
    point_differences = np.full((5, 2, 2), np.nan)
    strata = []
    for day, pair in enumerate(panel):
        strata.append(tuple(np.flatnonzero(pair[0].directions == direction) for direction in range(8)))
        counts = np.ones((1, len(pair[0].trial_ids)), dtype=np.int64)
        for region in range(2):
            scores, differences, _ = _score_multiplicities(statistics[day][region], counts)
            point_scores[day, region], point_differences[day, region] = scores[0], differences[0]
    observed = _observed_gates(point_differences)
    draws = np.full((BOOTSTRAP_SAMPLES, 2, 2), np.nan)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    for start in range(0, BOOTSTRAP_SAMPLES, _BATCH_SIZE):
        _check(check)
        size = min(_BATCH_SIZE, BOOTSTRAP_SAMPLES - start)
        counts = [np.zeros((size, len(pair[0].trial_ids)), dtype=np.int64) for pair in panel]
        # Preserve this RNG loop order even though arithmetic is batched below.
        for draw in range(size):
            _check(check)
            for day, groups in enumerate(strata):
                for indices in groups:
                    positions = rng.integers(0, len(indices), size=len(indices), dtype=np.int64)
                    counts[day][draw, indices] = np.bincount(positions, minlength=len(indices))
        day_draws = np.full((5, size, 2, 2), np.nan)
        for day in range(5):
            for region in range(2):
                _check(check)
                _, day_draws[day, :, region], _ = _score_multiplicities(statistics[day][region], counts[day])
        _, aggregate = _animal_aggregate(day_draws)
        draws[start:start + size] = aggregate
    margins = _margin_inference(observed.aggregate_differences, draws)
    reasons = []
    for region in range(2):
        missing = tuple(f"original_joint_score_unavailable:{animal}/{day}"
                        for index, (animal, day) in enumerate(DAY_ORDER)
                        if not np.isfinite(point_differences[index, region]).all())
        if margins.valid_draws[region] != BOOTSTRAP_SAMPLES:
            missing += ("undefined_bootstrap_contrast_no_redraw_or_deletion",)
        if not margins.available[region] and not missing:
            missing = ("nonfinite_aggregate_or_centered_bootstrap_arithmetic",)
        reasons.append(missing)
    passed = (margins.holm_rejected & observed.margin_pass & observed.animal_sign_pass
              & observed.day_frequency_pass & np.all(observed.omission_pass, axis=0))
    _check(check)
    return PrimaryInference(DAY_ORDER, REGIONS, COMPARATORS,
        tuple(pair[0].trial_ids for pair in panel), _readonly(point_scores),
        _readonly(point_differences), _readonly(draws), observed, margins,
        tuple(reasons), _readonly(passed))
