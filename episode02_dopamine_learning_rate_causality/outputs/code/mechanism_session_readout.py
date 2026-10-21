"""Array-only bridge from mechanism trial probabilities to EP02 observation fits.

Each input probability must describe the declared preparatory-lick event on
one model trial. Average probabilities over explicitly eligible model trials,
then take the session logit: averaging logits defines a different readout.
This is the expected-rate parameter of the already selected session-binomial
observation model, not an assertion that heterogeneous trial probabilities
have an exact binomial sum distribution.

The caller supplies the qualified mechanism probabilities, explicit session
roster and eligibility mask, separately for each frozen mechanism/seed. This
module neither generates them from ACTR output nor treats a sampled lick as
its probability. Model-trial counts are not empirical binomial denominators.
No clipping, pseudocount, free readout intercept, missing-session interpolation,
filesystem reader, source access or scientific margin is introduced. Fitting
uses the existing shared-mouse logistic-normal observation recipe unchanged.
Array IDs cannot establish provenance or release the closed data boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np

from development_score_core import DEVELOPMENT_RECORDS
from logistic_normal_observation import (
    ObservationFit,
    TrainingMouse,
    fit_observation_model,
    predict_new_mouse,
    refit_final_observation_model,
)


class ReadoutError(ValueError):
    """The supplied readout cannot enter the existing finite-logit model."""


@dataclass(frozen=True)
class SessionReadout:
    mouse_id: int
    session_ids: tuple[str, ...]
    probabilities: tuple[float, ...]
    model_trial_counts: tuple[int, ...]


@dataclass(frozen=True)
class SessionCounts:
    """Verified empirical session counts; zero-information sessions stay present."""

    mouse_id: int
    session_ids: tuple[str, ...]
    successes: tuple[int, ...]
    trials: tuple[int, ...]


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ReadoutError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _sessions(values):
    names = tuple(values)
    if (not names or any(not isinstance(s, str) or not s for s in names)
            or len(set(names)) != len(names)):
        raise ReadoutError("nonempty, unique, explicit session IDs required")
    return names


def _probability(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ReadoutError("trial/readout probabilities must be real numbers, not lick-event Booleans")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ReadoutError("finite probabilities in [0, 1] required")
    return value


def aggregate_trial_probabilities(*, mouse_id, session_ids, trial_session_ids,
                                  trial_probabilities, eligible):
    """Expected eligible-trial probability in every explicitly named session.

The source-semantic or simulation layer owns eligibility. Excluded trials do
not enter the mean; their probability values are not inspected. Empty model
support fails instead of dropping a session or borrowing another session's
value. Exact 0/1 means remain visible here, but cannot pass the finite-logit
observation handoff below. No random-seed averaging occurs in this function.
"""
    mouse = _integer(mouse_id, "mouse ID")
    names = _sessions(session_ids)
    trial_names = tuple(trial_session_ids)
    probabilities = tuple(trial_probabilities)
    included = tuple(eligible)
    if len(trial_names) != len(probabilities) or len(trial_names) != len(included):
        raise ReadoutError("one session ID, probability and eligibility flag per model trial required")
    buckets = {name: [] for name in names}
    for name, probability, include in zip(trial_names, probabilities, included):
        if not isinstance(name, str) or name not in buckets:
            raise ReadoutError("model trial references a session outside the explicit roster")
        if not isinstance(include, (bool, np.bool_)):
            raise ReadoutError("eligibility must be an explicit Boolean mask")
        if include:
            buckets[name].append(_probability(probability))
    if any(not buckets[name] for name in names):
        raise ReadoutError("every session requires eligible model-trial probability support")
    return SessionReadout(mouse, names,
                          tuple(math.fsum(buckets[s]) / len(buckets[s]) for s in names),
                          tuple(len(buckets[s]) for s in names))


def _validated_readout(readout):
    if not isinstance(readout, SessionReadout):
        raise ReadoutError("SessionReadout required")
    mouse = _integer(readout.mouse_id, "mouse ID")
    names = _sessions(readout.session_ids)
    probabilities = tuple(_probability(p) for p in readout.probabilities)
    counts = tuple(_integer(n, "eligible model-trial count") for n in readout.model_trial_counts)
    if len(names) != len(probabilities) or len(names) != len(counts):
        raise ReadoutError("readout session arrays must have identical lengths")
    return SessionReadout(mouse, names, probabilities, counts)


def finite_session_logits(readout):
    """Convert without changing the supplied probabilities or smoothing endpoints.

The existing logistic-normal fitter requires finite mechanism logits. A 0/1
Monte Carlo frequency is not evidence of an impossible/certain event; the
probability-generation layer must resolve it under its declared recipe. This
handoff refuses exact endpoints rather than inventing a floor or pseudocount.
"""
    row = _validated_readout(readout)
    if any(p == 0 or p == 1 for p in row.probabilities):
        raise ReadoutError("boundary probability has no finite logit; probability-generation recipe must resolve it")
    return tuple(math.log(p) - math.log1p(-p) for p in row.probabilities)


def _index_rows(rows, row_type, expected, label):
    result = {}
    for row in rows:
        if not isinstance(row, row_type):
            raise ReadoutError(f"{label} rows require {row_type.__name__}")
        mouse = _integer(row.mouse_id, "mouse ID")
        if mouse not in expected or mouse in result:
            raise ReadoutError(f"{label} must contain each expected training mouse exactly once")
        result[mouse] = row
    if set(result) != set(expected):
        raise ReadoutError(f"{label} must contain the complete fixed training roster")
    return result


def _training_rows(training_counts, training_readouts, expected):
    observations = _index_rows(training_counts, SessionCounts, expected, "counts")
    readouts = _index_rows(training_readouts, SessionReadout, expected, "readouts")
    rows = []
    for mouse in expected:
        count = observations[mouse]
        names = _sessions(count.session_ids)
        readout = _validated_readout(readouts[mouse])
        if set(names) != set(readout.session_ids):
            raise ReadoutError("count and readout session rosters must match exactly, including zero-count sessions")
        logits = dict(zip(readout.session_ids, finite_session_logits(readout)))
        # Count validity and positive total support are checked by the unchanged
        # observation fitter. Model-trial counts never replace these denominators.
        rows.append(TrainingMouse(mouse, names, tuple(logits[name] for name in names),
                                  tuple(count.successes), tuple(count.trials)))
    return tuple(rows)


def fit_observation_from_readouts(*, training_counts, training_readouts,
                                  held_mouse_id, recipe):
    """One seed/configuration's exact eight-mouse observation fit; no held counts."""
    held = _integer(held_mouse_id, "held mouse ID")
    if held not in DEVELOPMENT_RECORDS:
        raise ReadoutError("held mouse must be one of the nine fixed development controls")
    expected = tuple(mouse for mouse in DEVELOPMENT_RECORDS if mouse != held)
    rows = _training_rows(training_counts, training_readouts, expected)
    return fit_observation_model(training_mice=rows, held_mouse_id=held, recipe=recipe)


def refit_final_observation_from_readouts(*, training_counts, training_readouts,
                                        recipe, check):
    """All-nine component after legitimate recipe lock and reader release only."""
    if not callable(check):
        raise ReadoutError("an explicit final-refit resource callback is required")
    check()
    rows = _training_rows(training_counts, training_readouts, DEVELOPMENT_RECORDS)
    return refit_final_observation_model(training_mice=rows, recipe=recipe, check=check)


def predict_development_from_readout(fit, readout):
    """Construct a held-mouse distribution using no held-mouse outcomes."""
    if not isinstance(fit, ObservationFit):
        raise ReadoutError("a development ObservationFit is required, not a final refit")
    row = _validated_readout(readout)
    return predict_new_mouse(fit, held_mouse_id=row.mouse_id, session_ids=row.session_ids,
                             mechanism_logits=finite_session_logits(row))
