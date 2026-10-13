"""Project qualified EP19 risk/onset facts into exact forecast labels.

This scalar helper does not detect or qualify onsets, choose risk decisions,
read sensors, reconcile clocks, infer cue phases or construct a calibration
prefix. Inputs must already use one admitted integer timebase and the existing
recording/reset identity. A tick need not be a source sample; the caller must
supply an exact reconciled unit that represents the fixed 50-ms grid. No float
rounding, interpolation or timing correction occurs here.

The follow-up endpoint is a hard admitted evidence/source boundary, not merely
raw recording length or the duration of the at-risk prediction state. An earlier
unknown bound separately truncates continuous onset-status coverage. It must
include every unresolved possible onset as of the allowed knowledge cutoff. A
later accepted crossing cannot certify absence of an earlier unresolved one.
Every different prefix cutoff needs its corresponding as-of facts; full-record
adjudication cannot be copied back into a closed prefix.

The accepted onset's known_tick includes all required confirmation and earlier
competing-path resolution. It is never substituted for crossing_tick. Pending,
unarmed and unknown decisions remain unavailable, regardless of later evidence.
These labels do not establish detector, source-role or timing qualification.
"""
from dataclasses import dataclass
from numbers import Integral

from classical_forecast_backend import GroupIdentity


class LabelConstructionError(ValueError):
    """Supplied facts cannot support the requested label projection."""


def _require(condition, message):
    if not condition:
        raise LabelConstructionError(message)


def _integer(value, name):
    _require(isinstance(value, Integral) and not isinstance(value, bool),
             name + " must be an exact non-boolean integer")
    return int(value)


@dataclass(frozen=True)
class RiskDecision:
    identity: GroupIdentity
    decision_tick: int
    state: str  # at_risk, pending, unarmed, or unknown at this decision.


@dataclass(frozen=True)
class AcceptedOnset:
    """Caller-qualified earliest target onset; never first confirmation wins.

    known_tick is when this FIRST-onset identity is established, including the
    resolution of earlier competing candidates. It is at least crossing_tick.
    Both values use the decision's recording/reset boundary and timebase.
    """
    identity: GroupIdentity
    crossing_tick: int
    known_tick: int


@dataclass(frozen=True)
class Followup:
    """Admitted evidence and onset-status scope, with exclusive upper bounds.

    observed_stop_tick is the first tick outside admitted evidence: a HARD
    source/acquisition/reset/target or administrative boundary, not just an
    earlier loss of outcome coverage. Adjudication at/after this stop cannot be
    used. Merely suspending predictions does not truncate otherwise known
    follow-up after an accepted onset. None means the admitted observation
    scope is not established.

    unknown_from_tick, when present, is the earliest unresolved possible onset
    or other loss of onset-status coverage AS OF the admitted knowledge cutoff.
    Include earlier competing candidates even if they are eventually rejected.
    It cannot be reconstructed from AcceptedOnset alone. None denotes no such
    earlier unknown bound supplied; it cannot support a within-scope onset that
    is still unadjudicated at the cutoff. This bound limits COVERAGE, not when
    already admitted evidence can establish an earlier onset. For example,
    stop400, unknown390, crossing300 and known395 preserve an observed event
    at300 and only390 ticks of follow-up (with a decision at0).
    """
    identity: GroupIdentity
    observed_stop_tick: int | None
    unknown_from_tick: int | None = None


@dataclass(frozen=True)
class ForecastLabel:
    decision: RiskDecision
    observed_category: int | None
    complete_bins: int | None
    observed_through_bins: int | None
    reason: str


def label_for_decision(decision, followup, onset=None, *, ticks_per_second,
                       grid_origin_tick, label_stop_tick=None):
    """Return one immutable label without adding/removing any risk decision.

    Categories 0..17 denote [0,50),...,[850,900) ms after the decision; category
    18 denotes no onset in [0,900), including onset exactly at 900 ms. Censoring
    uses category -1 and only complete 50-ms intervals. For an observed event,
    complete_bins is its category index (complete event-free bins before onset)
    and is ignored by the existing event likelihood. observed_through_bins is
    always separate: an early event never supplies missing later follow-up.

    label_stop_tick is an optional exclusive label/knowledge cutoff, including
    a separately qualified calibration-prefix endpoint. A decision at or after
    that cutoff is refused, not added as a zero-loss fitting row. Confirmation
    or adjudication exactly at any exclusive cutoff is not yet available.

    ticks_per_second must be a positive multiple of 20 and grid_origin_tick is
    supplied explicitly. All times are integer ticks on this same admitted
    clock; this helper neither selects the clock nor rounds source timestamps.
    """
    _require(isinstance(decision, RiskDecision)
             and isinstance(decision.identity, GroupIdentity),
             "a qualified risk decision with its recording identity is required")
    tick = _integer(decision.decision_tick, "decision_tick")
    rate = _integer(ticks_per_second, "ticks_per_second")
    origin = _integer(grid_origin_tick, "grid_origin_tick")
    _require(rate > 0 and rate % 20 == 0, "timebase must represent 50 ms exactly")
    width = rate // 20
    _require((tick - origin) % width == 0, "decision is off the declared 50-ms grid")
    _require(decision.state in ("at_risk", "pending", "unarmed", "unknown"),
             "original qualified risk state is required")
    if label_stop_tick is not None:
        label_stop_tick = _integer(label_stop_tick, "label_stop_tick")
        _require(label_stop_tick > tick, "decision is outside the exclusive label prefix")
    if decision.state != "at_risk":
        # Do not inspect later onset/follow-up facts to restore this decision.
        return ForecastLabel(decision, None, None, None, "decision_not_at_risk")

    _require(isinstance(followup, Followup) and followup.identity == decision.identity,
             "follow-up must belong to this recording/reset identity")
    unknown = followup.unknown_from_tick
    if unknown is not None:
        unknown = _integer(unknown, "unknown_from_tick")
        _require(unknown >= tick, "unknown onset status precedes the admitted decision")
    crossing = known = None
    if onset is not None:
        _require(isinstance(onset, AcceptedOnset) and onset.identity == decision.identity,
                 "accepted onset must belong to this recording/reset identity")
        crossing = _integer(onset.crossing_tick, "crossing_tick")
        known = _integer(onset.known_tick, "known_tick")
        _require(crossing >= tick and known >= crossing,
                 "onset crossing/adjudication order contradicts the admitted decision")
    if followup.observed_stop_tick is None:
        return ForecastLabel(decision, -1, 0, 0, "observation_coverage_unknown")
    observed_stop = _integer(followup.observed_stop_tick, "observed_stop_tick")
    _require(observed_stop >= tick, "follow-up stops before the admitted decision")
    knowledge_cut = observed_stop if label_stop_tick is None else min(observed_stop, label_stop_tick)
    coverage_stop = knowledge_cut if unknown is None else min(knowledge_cut, unknown)

    if crossing is not None and crossing < knowledge_cut and known >= knowledge_cut:
        # The onset's eventual adjudication is outside allowed knowledge.
        # Requiring the explicit as-of bound prevents inventing absence between
        # an earlier still-pending path and this eventually selected crossing.
        _require(unknown is not None and unknown <= crossing,
                 "unavailable onset adjudication requires its earliest as-of unknown bound")

    coverage = min(18, (coverage_stop - tick) // width)
    if crossing is not None and crossing < coverage_stop and known < knowledge_cut:
        category = min(18, (crossing - tick) // width)
        reason = "observed_onset" if category < 18 else "complete_900ms_without_onset"
        return ForecastLabel(decision, category, category, coverage, reason)
    if coverage == 18:
        return ForecastLabel(decision, 18, 18, 18, "complete_900ms_without_onset")
    reason = "unknown_onset_status" if unknown is not None and unknown <= knowledge_cut else "administratively_censored"
    return ForecastLabel(decision, -1, coverage, coverage, reason)
