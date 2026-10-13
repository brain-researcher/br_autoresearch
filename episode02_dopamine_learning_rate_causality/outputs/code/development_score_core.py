"""Array-only EP02 session-binomial and complete-panel development scoring.

The caller supplies verified eligible-session counts and population-predictive
logits. This module does not interpret trial/session codes, select observations,
fit a model, open files, or establish physical role or fitting isolation. Empty
or missing sessions require resolution by the eventual source adapter; they are
never silently removed here. The primary endpoint's session-1/8 support rule and
the four unbound scientific thresholds are outside this scoring component.

Scores follow SEARCH_POLICY.json: session-binomial log probability per eligible
trial, arithmetic mean of five seed scores within mouse, then equally weighted
nine-mouse candidate-minus-constant-null gains. Selection uses exact objective
coordinates. The separate incumbent-replacement tolerance is not implemented.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Sequence


DEVELOPMENT_RECORDS = (1, 2, 4, 6, 9, 11, 15, 16, 19)
SEED_COUNT = 5


class ScoreError(ValueError):
    """The supplied arrays cannot produce a complete finite scoring panel.

    This is an API/numerical refusal, not a scientific conclusion or a decision
    about whether a future fitted candidate counts as a valid adaptive trial.
    """


@dataclass(frozen=True)
class MouseFold:
    """One held mouse; both models share observations and the explicit seed bank.

    Session IDs are unique opaque keys in the caller's verified session order.
    Logit matrices have shape (5, number_of_sessions), with rows in seed_ids
    order. Training IDs describe lineage only; arrays cannot prove it occurred.
    """

    held_mouse_id: int
    training_mouse_ids: Sequence[int]
    session_ids: Sequence[str]
    successes: Sequence[int]
    trials: Sequence[int]
    seed_ids: Sequence[int]
    candidate_logits: Sequence[Sequence[float]]
    constant_null_logits: Sequence[Sequence[float]]


@dataclass(frozen=True)
class DevelopmentScore:
    """Aggregate-only complete-panel objective; contains no per-mouse values."""

    minimum_mouse_gain: float
    median_mouse_gain: float
    mean_mouse_gain: float
    frozen_complexity: float
    configuration_id: str


def _integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ScoreError(f"{name} must be an integer at least {minimum}")
    return int(value)


def _real(value, name: str, *, allow_infinity: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ScoreError(f"{name} must be a real scalar")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ScoreError(f"{name} is not representable") from error
    if math.isnan(result) or (not allow_infinity and not math.isfinite(result)):
        raise ScoreError(f"{name} must be {'non-NaN' if allow_infinity else 'finite'}")
    return result


def _finite_sum(values, name: str) -> float:
    try:
        result = math.fsum(values)
    except (ValueError, OverflowError) as error:
        raise ScoreError(f"{name} is not a finite representable sum") from error
    if not math.isfinite(result):
        raise ScoreError(f"{name} is not finite; no observations were dropped")
    return result


def _softplus(value: float) -> float:
    # The caller handles infinite logits before invoking this finite branch.
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def session_binomial_logpmf(successes: int, trials: int, logit: float) -> float:
    """Log P(K=k | n, sigmoid(logit)), including the binomial coefficient.

    The elementary distribution permits n=0 and exact probabilities 0/1.
    Impossible boundary observations return -inf. The complete-panel API
    separately requires positive supplied denominators and finite model scores;
    it never clips probabilities or converts an impossible event to a pseudocount.
    """
    n = _integer(trials, "trials")
    k = _integer(successes, "successes")
    if k > n:
        raise ScoreError("successes exceed trials")
    eta = _real(logit, "logit", allow_infinity=True)
    if n == 0:
        return 0.0
    if eta == math.inf:
        return 0.0 if k == n else -math.inf
    if eta == -math.inf:
        return 0.0 if k == 0 else -math.inf
    try:
        log_choose = (0.0 if k in (0, n) else
                      math.lgamma(n + 1) - math.fsum((math.lgamma(k + 1),
                                                    math.lgamma(n - k + 1))))
        # Skip zero-count terms explicitly, including the mathematical 0*log(0).
        terms = [log_choose]
        if k:
            terms.append(-k * _softplus(-eta))
        if n - k:
            terms.append(-(n - k) * _softplus(eta))
        return _finite_sum(terms, "binomial logPMF")
    except (ValueError, OverflowError) as error:
        if isinstance(error, ScoreError):
            raise
        raise ScoreError("binomial logPMF is not representable") from error


def _model_seed_scores(logits, successes, trials, seed_order, session_count):
    if len(logits) != SEED_COUNT or any(len(row) != session_count for row in logits):
        raise ScoreError("logit matrix must have exactly five rows and one column per session")
    total_trials = sum(trials)
    scores = []
    for index in seed_order:
        values = (session_binomial_logpmf(k, n, eta)
                  for k, n, eta in zip(successes, trials, logits[index]))
        score = _finite_sum(values, "held-mouse seed log probability") / total_trials
        if not math.isfinite(score):
            raise ScoreError("held-mouse seed score is not finite")
        scores.append(score)
    return scores


def score_loso_panel(folds: Sequence[MouseFold], *, frozen_complexity: float,
                     configuration_id: str) -> DevelopmentScore:
    """Score exactly nine complete LOSO folds without fitting or dropping rows.

    Numerical sums use Python float (binary64) and math.fsum. Fold order is the
    fixed source-record order; seed order is ascending supplied seed ID. Session
    order is supplied explicitly and shared by both models. For each model the
    five seed scores are averaged first; their mouse-level means are subtracted.
    """
    complexity = _real(frozen_complexity, "frozen_complexity")
    if complexity < 0:
        raise ScoreError("frozen_complexity must be nonnegative")
    if not isinstance(configuration_id, str) or not configuration_id:
        raise ScoreError("configuration_id must be a nonempty string")
    if len(folds) != len(DEVELOPMENT_RECORDS) or any(not isinstance(fold, MouseFold) for fold in folds):
        raise ScoreError("exactly nine MouseFold records are required")
    held_ids = [_integer(fold.held_mouse_id, "held_mouse_id", 1) for fold in folds]
    if len(set(held_ids)) != len(DEVELOPMENT_RECORDS) or set(held_ids) != set(DEVELOPMENT_RECORDS):
        raise ScoreError("held mice must equal the fixed nine development records")
    by_mouse = dict(zip(held_ids, folds))
    shared_seeds = None
    gains = []
    for mouse in DEVELOPMENT_RECORDS:
        fold = by_mouse[mouse]
        training = tuple(_integer(value, "training_mouse_id", 1) for value in fold.training_mouse_ids)
        if (len(training) != 8 or len(set(training)) != 8 or
                set(training) != set(DEVELOPMENT_RECORDS) - {mouse}):
            raise ScoreError("declared training roster must be exactly the other eight mice")
        sessions = tuple(fold.session_ids)
        if (not sessions or any(not isinstance(value, str) or not value for value in sessions) or
                len(set(sessions)) != len(sessions)):
            raise ScoreError("a nonempty unique verified session-key roster is required")
        if len(fold.successes) != len(sessions) or len(fold.trials) != len(sessions):
            raise ScoreError("count arrays must match every supplied session")
        successes = tuple(_integer(value, "successes") for value in fold.successes)
        trials = tuple(_integer(value, "trials", 1) for value in fold.trials)
        if any(k > n for k, n in zip(successes, trials)):
            raise ScoreError("successes exceed trials")
        seeds = tuple(_integer(value, "seed_id") for value in fold.seed_ids)
        if len(seeds) != SEED_COUNT or len(set(seeds)) != SEED_COUNT:
            raise ScoreError("exactly five distinct explicit seed IDs are required")
        ordered_seeds = tuple(sorted(seeds))
        if shared_seeds is None:
            shared_seeds = ordered_seeds
        elif ordered_seeds != shared_seeds:
            raise ScoreError("every held mouse must use the same five-seed bank")
        seed_order = sorted(range(SEED_COUNT), key=seeds.__getitem__)
        candidate = _model_seed_scores(fold.candidate_logits, successes, trials, seed_order, len(sessions))
        null = _model_seed_scores(fold.constant_null_logits, successes, trials, seed_order, len(sessions))
        candidate_mean = _finite_sum(candidate, "candidate seed-score sum") / SEED_COUNT
        null_mean = _finite_sum(null, "constant-null seed-score sum") / SEED_COUNT
        gain = candidate_mean - null_mean
        if not math.isfinite(gain):
            raise ScoreError("mouse gain is not finite; the whole panel is refused")
        gains.append(gain)
    ordered_gains = sorted(gains)
    return DevelopmentScore(minimum_mouse_gain=ordered_gains[0], median_mouse_gain=ordered_gains[4],
                            mean_mouse_gain=_finite_sum(gains, "mouse gain sum") / len(DEVELOPMENT_RECORDS),
                            frozen_complexity=complexity, configuration_id=configuration_id)


def selection_key(score: DevelopmentScore) -> tuple[float, float, float, float, str]:
    """Ascending key for the policy's exact total order; no tolerance or refit.

    Complexity is supplied from the caller's prospectively frozen definition.
    This helper does not define that scientific complexity measure, implement
    incumbent replacement, authorize a lock, or count an adaptive trial.
    """
    if not isinstance(score, DevelopmentScore):
        raise ScoreError("a DevelopmentScore is required")
    coordinates = tuple(_real(value, "score coordinate") for value in
                        (score.minimum_mouse_gain, score.median_mouse_gain, score.mean_mouse_gain))
    complexity = _real(score.frozen_complexity, "frozen_complexity")
    if complexity < 0 or not isinstance(score.configuration_id, str) or not score.configuration_id:
        raise ScoreError("invalid complexity or configuration ID")
    return (-coordinates[0], -coordinates[1], -coordinates[2], complexity, score.configuration_id)
