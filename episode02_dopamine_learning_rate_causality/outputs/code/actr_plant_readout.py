"""Conditional Monte Carlo readout of already frozen ACTR learning paths.

For each of the five existing seeds, the caller supplies pre-update plant
activities from the complete learned trajectory. This module calls the source
behavioral plant on independent readout RNG streams only. It never constructs
task inputs, evolves a network, consumes training RNG, opens empirical data or
fits a model. Noneligible trials may affect the supplied learning trajectory
upstream but do not enter this endpoint. The short synthetic_fixture is not
used as a production driver.

Finite, prespecified draw stages use a union-bounded Hoeffding error allowance
across every seed/session/stage. These are numerical Monte Carlo bounds
conditional on fixed learning paths, not biological uncertainty or scientific
effect margins. No result is handed to the observation fitter unless all five
paths and every session meet the explicit finite-logit precision criterion.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np

from actr_port import behavioral_plant
from development_score_core import SEED_COUNT
from mechanism_session_readout import SessionReadout, aggregate_trial_probabilities


class PlantReadoutError(ValueError):
    """Incomplete frozen inputs or invalid numerical settings, not a model result."""


@dataclass(frozen=True)
class FrozenPlantTrial:
    session_id: str
    activity: np.ndarray
    reward_tick: int
    reward: np.ndarray | None
    scale: float
    eligible: bool


@dataclass(frozen=True)
class FrozenSeedPath:
    seed_id: int
    trials: tuple[FrozenPlantTrial, ...]


@dataclass(frozen=True)
class MonteCarloSettings:
    draws_per_trial: tuple[int, ...]
    failure_probability: float
    maximum_logit_width: float
    maximum_total_plant_draws: int


@dataclass(frozen=True)
class SessionEstimate:
    seed_id: int
    session_id: str
    probability: float | None
    probability_lower: float
    probability_upper: float
    logit_width: float
    draws_per_trial: int
    total_plant_draws: int
    resolved: bool


@dataclass(frozen=True)
class PlantReadoutResult:
    status: str
    reason: str
    seed_ids: tuple[int, ...]
    estimates: tuple[SessionEstimate, ...]
    # Same order as seed_ids. None prevents using a favorable completed subset.
    readouts: tuple[SessionReadout, ...] | None
    total_plant_draws: int
    per_interval_failure_probability: float


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise PlantReadoutError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise PlantReadoutError(f"{name} must be a positive finite real number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise PlantReadoutError(f"{name} must be a positive finite real number")
    return value


def _settings(settings):
    if not isinstance(settings, MonteCarloSettings):
        raise PlantReadoutError("explicit MonteCarloSettings required")
    stages = tuple(_integer(n, "draw stage", 1) for n in settings.draws_per_trial)
    if not stages or stages != tuple(sorted(set(stages))):
        raise PlantReadoutError("finite increasing positive draw stages required")
    failure = _positive(settings.failure_probability, "numerical failure probability")
    if failure >= 1:
        raise PlantReadoutError("numerical failure probability must be below one")
    return MonteCarloSettings(stages, failure,
        _positive(settings.maximum_logit_width, "maximum logit width"),
        _integer(settings.maximum_total_plant_draws, "total plant-draw limit", 1))


def _frozen_trial(trial):
    activity = np.array(trial.activity, dtype=float, copy=True)
    if activity.ndim != 1 or len(activity) < 2 or not np.all(np.isfinite(activity)):
        raise PlantReadoutError("finite one-dimensional pre-update activity required")
    reward_tick = _integer(trial.reward_tick, "one-based reward tick", 751)
    if reward_tick > len(activity):
        raise PlantReadoutError("reward tick lies outside the supplied 1-ms trial")
    reward = None
    if trial.reward is None:
        if len(activity) != 3000 or reward_tick != 1600:
            raise PlantReadoutError("the source default reward convention requires 3000 ticks and reward tick 1600")
    else:
        reward = np.array(trial.reward, dtype=float, copy=True)
        if (reward.shape != activity.shape or not np.all(np.isfinite(reward))
                or np.any(reward < 0)):
            raise PlantReadoutError("explicit plant reward must be finite, nonnegative and aligned")
        reward.setflags(write=False)
    activity.setflags(write=False)
    return FrozenPlantTrial(trial.session_id, activity, reward_tick, reward,
                            _positive(trial.scale, "source plant scale"), True)


def _logit(p):
    return math.log(p) - math.log1p(-p)


def conditional_plant_readouts(*, mouse_id, session_ids, seed_paths,
                               readout_seed, settings, check):
    """Estimate the exact 750-ms event on all five supplied frozen seed paths.

The caller owns the qualified activity/task semantics and empirical authority.
``reward=None`` preserves the source default reward/background convention;
an explicit reward vector preserves its different background-off convention.
No default scale, reward tick, draw schedule, precision or resource limit is
chosen here. Draw streams use an explicit numerical namespace and never accept
or advance a training generator. The same root readout seed can intentionally
couple mechanisms/arms through common random numbers.

At each declared stage, every pending session receives the same additional
number of independent draws for each eligible trial. The whole pending stage
must fit the draw limit; otherwise return the preceding complete estimates as
unresolved. Any resource callback exception propagates without a fit-ready
partial packet. All five paths must be supplied even if some have converged.
"""
    if not callable(check):
        raise PlantReadoutError("explicit resource/cancellation callback required")
    check()
    mouse = _integer(mouse_id, "mouse ID", 1)
    root_seed = _integer(readout_seed, "readout RNG seed")
    settings = _settings(settings)
    names = tuple(session_ids)
    if (not names or any(not isinstance(s, str) or not s for s in names)
            or len(set(names)) != len(names)):
        raise PlantReadoutError("nonempty unique session roster required")
    paths = tuple(seed_paths)
    if len(paths) != SEED_COUNT or any(not isinstance(p, FrozenSeedPath) for p in paths):
        raise PlantReadoutError("exactly the five frozen seed paths are required")
    seeds = tuple(_integer(p.seed_id, "path seed ID") for p in paths)
    if len(set(seeds)) != SEED_COUNT:
        raise PlantReadoutError("the five path seed IDs must be distinct")

    # Copy fixed plant inputs once. Each work item is one seed/session and owns
    # nested RNG prefixes and cumulative event counts for its eligible trials.
    work = {}
    for path in paths:
        for name in names:
            work[(path.seed_id, name)] = []
        for trial_index, trial in enumerate(path.trials):
            check()
            if (not isinstance(trial, FrozenPlantTrial) or not isinstance(trial.session_id, str)
                    or trial.session_id not in names):
                raise PlantReadoutError("each supplied trial must belong to the explicit session roster")
            if not isinstance(trial.eligible, (bool, np.bool_)):
                raise PlantReadoutError("explicit Boolean model-trial eligibility required")
            if not trial.eligible:
                continue
            frozen = _frozen_trial(trial)
            rng = np.random.default_rng(np.random.SeedSequence(
                [0x45503032, root_seed, path.seed_id, trial_index]))
            work[(path.seed_id, trial.session_id)].append([frozen, rng, 0])
    if any(not trials for trials in work.values()):
        raise PlantReadoutError("every seed/session requires eligible frozen model trials")

    # Union bound covers all permitted looks, even looks skipped by early stop.
    # Independence is conditional on each supplied, fixed learning trajectory.
    interval_alpha = settings.failure_probability / (SEED_COUNT * len(names) * len(settings.draws_per_trial))
    if interval_alpha <= 0:
        raise PlantReadoutError("numerical error allowance is not representable")
    estimates = {key: SessionEstimate(key[0], key[1], None, 0.0, 1.0, math.inf, 0, 0, False)
                 for key in work}
    total_draws = 0
    reason = "draw_schedule_exhausted"
    for target_draws in settings.draws_per_trial:
        check()
        pending = [key for key in work if not estimates[key].resolved]
        required = sum(len(work[key]) * (target_draws - estimates[key].draws_per_trial) for key in pending)
        if total_draws + required > settings.maximum_total_plant_draws:
            reason = "draw_limit_before_next_complete_stage"
            break
        for key in pending:
            previous = estimates[key].draws_per_trial
            for entry in work[key]:
                trial, rng, _ = entry
                count = 0
                for _draw in range(target_draws - previous):
                    check()
                    licks, _state = behavioral_plant(trial.activity, rng, reward=trial.reward,
                                                     scale=trial.scale)
                    licks = np.asarray(licks)
                    if licks.ndim != 1 or not np.all(np.isfinite(licks)):
                        raise PlantReadoutError("source plant returned invalid lick-event times")
                    count += int(np.any((licks >= trial.reward_tick - 750) & (licks < trial.reward_tick)))
                entry[2] += count
            draws = len(work[key]) * target_draws
            probability = math.fsum(entry[2] / target_draws for entry in work[key]) / len(work[key])
            radius = math.sqrt((math.log(2) - math.log(interval_alpha)) / (2 * draws))
            lower = max(0.0, probability - radius)
            upper = min(1.0, probability + radius)
            width = _logit(upper) - _logit(lower) if 0 < lower < upper < 1 else math.inf
            resolved = width <= settings.maximum_logit_width
            total_draws += len(work[key]) * (target_draws - previous)
            estimates[key] = SessionEstimate(key[0], key[1], probability, lower, upper,
                                             width, target_draws, draws, resolved)
        if all(estimate.resolved for estimate in estimates.values()):
            reason = "all_seed_sessions_meet_frozen_precision"
            break
    check()
    complete = all(estimate.resolved for estimate in estimates.values())
    readouts = None
    if complete:
        rows = []
        for seed in seeds:
            trial_names, probabilities = [], []
            for name in names:
                count = estimates[(seed, name)].draws_per_trial
                for entry in work[(seed, name)]:
                    trial_names.append(name)
                    probabilities.append(entry[2] / count)
            rows.append(aggregate_trial_probabilities(mouse_id=mouse, session_ids=names,
                trial_session_ids=trial_names, trial_probabilities=probabilities,
                eligible=(True,) * len(probabilities)))
        readouts = tuple(rows)
    return PlantReadoutResult("resolved" if complete else "unresolved", reason, seeds,
        tuple(estimates[(seed, name)] for seed in seeds for name in names), readouts,
        total_draws, interval_alpha)
