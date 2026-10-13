"""WAY risk/onset state from already qualified causal summaries and cue events.

This is the selected starting recipe, not source or timing qualification. The
caller supplies all five EMG and three hand/wrist summaries in the existing
path order, their admitted histories, frozen low/high thresholds and cue events
known at the current native 500-Hz tick. No clock alignment, filtering, cue
parsing, threshold calibration, physical-unit conversion or data access occurs.
ReturnSupport must come from the admitted completed-return predicate using
these same low thresholds. Its scalar result cannot prove that upstream fact.
The caller remains responsible for admitting pre-stream history after a reset;
the reducer only checks temporal consistency with history it observed itself.

Each stream covers one uninterrupted recording/reset identity. A cue end closes
only its cue; a gap closes the stream and requires a new reset identity upstream.
Saved snapshots never change. Later adjudication may label an earlier at-risk
decision, but cannot restore a pending row or revise an earlier evidence cutoff.
"""
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from classical_forecast_backend import GroupIdentity
from forecast_labels import AcceptedOnset, Followup, RiskDecision, label_for_decision
from way_completed_return import ReturnSupport


NATIVE_RATE_HZ = 500
DECISION_TICKS = 25
CONFIRMATION_TICKS = 20
QUIET_TICKS = 250


class RiskStreamError(ValueError):
    """Input contradicts the selected stream or its already observed history."""


def _require(condition, message):
    if not condition:
        raise RiskStreamError(message)


def _tick(value, name):
    _require(isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)),
             name + " must be an exact non-boolean native tick")
    return int(value)


def _eight(value, name):
    array = np.asarray(value, dtype=float)
    _require(array.shape == (8,), name + " must contain all eight primary paths")
    return tuple(float(x) for x in array)


@dataclass(frozen=True)
class CueScope:
    """Target cue scope, separate from the model's recording/reset identity."""

    identity: GroupIdentity
    start_tick: int


@dataclass(frozen=True)
class PendingCrossing:
    path_index: int
    crossing_tick: int


@dataclass(frozen=True)
class WayRiskSnapshot:
    cue: CueScope | None
    state: str
    decision: RiskDecision | None
    pending: tuple[PendingCrossing, ...]
    onset: AcceptedOnset | None
    followup: Followup | None
    grid_origin_tick: int
    closed: bool
    close_reason: str | None
    # Only binds real snapshot lineage for the cue/label join, not authority.
    _stream_token: object = field(repr=False, compare=False)
    _sequence: int = field(repr=False, compare=False)


class WayRiskStream:
    """Consume exactly one native tick at a time, without retrospective arming.

    ``start`` and ``end`` mean authenticated events available at this tick, not
    raw cue levels or later-corrected provider times. Supported return must bind
    exactly to [start-250, start). Missing/unknown support leaves this entire cue
    unknown; failed support leaves it unarmed. Neither can arm later in the cue.

    A high crossing (>= high) precedes a same-tick decision. A path rejects only
    below low; otherwise it confirms at crossing+20. Quiet means every path is
    <= low. After reject-all at r, discard all earlier quiet history; r itself
    may be the first fresh quiet sample. Re-arm using 250 PRECEDING quiet ticks,
    earliest r+250 for [r,r+250), and still let a current high crossing suspend
    that decision. All paths remain under surveillance during requalification.

    End/gap is an exclusive cut before this tick's sensor adjudication. For a
    valid sample t, the snapshot's evidence stop is t+1. An accepted onset alone
    never ends observation coverage. Keep the snapshot matching each desired
    prefix; the label join does not accept an arbitrary retrospective cutoff.
    """

    def __init__(self, identity, low, high, *, grid_origin_tick):
        _require(isinstance(identity, GroupIdentity) and identity.source == "WAY",
                 "a WAY recording/reset GroupIdentity is required")
        self.identity = identity
        self.grid_origin_tick = _tick(grid_origin_tick, "grid_origin_tick")
        self.low, self.high = _eight(low, "low"), _eight(high, "high")
        _require(all(np.isfinite(x) for x in self.low + self.high)
                 and all(h > l for l, h in zip(self.low, self.high)),
                 "finite separated low/high thresholds are required")
        self._token = object()
        self._sequence = 0
        self._last_tick = None
        self._terminal = False
        self._reach_seen = False
        self._last_end = None
        self._clear_cue()

    def _clear_cue(self):
        self._cue = None
        self._start_state = "unarmed"
        self._surveillance = False
        self._armed = False
        self._requalifying = False
        self._quiet_count = 0
        self._pending = {}
        self._accepted = []
        self._onset = None

    def _next_tick(self, tick):
        _require(not self._terminal, "closed reset segment requires a new stream identity")
        tick = _tick(tick, "tick")
        _require(self._last_tick is None or tick == self._last_tick + 1,
                 "ticks must be consecutive; close at the first gap instead of bridging it")
        return tick

    def _state(self):
        if self._cue is None:
            return "unarmed"
        if not self._surveillance:
            return self._start_state
        if self._accepted:
            return "unarmed" if self._onset is not None else "pending"
        return "at_risk" if self._armed else "pending"

    def _snapshot(self, stop, *, decision_tick=None, reason=None):
        state = "unknown" if reason == "gap" else self._state()
        if reason is not None and reason != "gap":
            state = "unarmed"
        decision = None
        if (self._cue is not None and decision_tick is not None
                and (decision_tick - self.grid_origin_tick) % DECISION_TICKS == 0):
            decision = RiskDecision(self.identity, decision_tick, state)
        pending = tuple(PendingCrossing(path, crossing)
                        for path, crossing in sorted(self._pending.items()))
        followup = None
        if self._cue is not None:
            unknown = None
            if not self._surveillance:
                unknown = self._cue.start_tick
            elif self._onset is None and self._pending:
                unknown = min(self._pending.values())
                if self._accepted:
                    unknown = min(unknown, min(crossing for crossing, _ in self._accepted))
            followup = Followup(self.identity, stop, unknown)
        return WayRiskSnapshot(self._cue, state, decision, pending, self._onset,
                               followup, self.grid_origin_tick, reason is not None,
                               reason, self._token, self._sequence)

    def _cut(self, tick, reason, *, terminal):
        self._sequence += 1
        self._last_tick = tick
        snapshot = self._snapshot(tick, reason=reason)
        if reason == "end":
            self._last_end = tick
        self._clear_cue()
        self._terminal = terminal
        return snapshot

    def step(self, tick, summaries, *, start=False, end=False,
             return_support=None, history_valid=True):
        tick = self._next_tick(tick)
        _require(all(isinstance(flag, (bool, np.bool_))
                     for flag in (start, end, history_valid)),
                 "start, end and history_valid must be boolean declarations")
        _require(not (start and end), "simultaneous start/end events are ambiguous")
        _require(not start or self._cue is None,
                 "a new start requires the preceding cue's observed end")
        _require(start or return_support is None, "return support belongs only to its start")
        status = "unknown"
        if start and return_support is not None:
            _require(isinstance(return_support, ReturnSupport)
                     and return_support.status in ("supported", "not_supported", "unknown"),
                     "a completed-return result is required")
            status = return_support.status
            if status == "supported":
                first = _tick(return_support.interval_start_tick, "return interval start")
                stop = _tick(return_support.interval_stop_tick, "return interval stop")
                _require(first == tick - QUIET_TICKS and stop == tick,
                         "return support must bind to this start's exact preceding interval")
                _require(not self._reach_seen or
                         (self._last_end is not None and self._last_end < first),
                         "an observed post-reach end must precede the whole return interval")
        if not history_valid:
            # A declared gap also terminates the reset when an end coincides.
            return self._cut(tick, "gap", terminal=True)
        if end:
            # Do not inspect a sample outside the old cue's evidence interval.
            return self._cut(tick, "end", terminal=False)
        if summaries is None:
            return self._cut(tick, "gap", terminal=True)
        values = _eight(summaries, "summaries")
        if not all(np.isfinite(x) for x in values):
            return self._cut(tick, "gap", terminal=True)

        self._sequence += 1
        self._last_tick = tick
        if start:
            self._cue = CueScope(self.identity, tick)
            self._start_state = "unknown" if status == "unknown" else "unarmed"
            self._surveillance = status == "supported"
            self._armed = self._surveillance
        if self._surveillance:
            if (self._requalifying and not self._pending and not self._accepted
                    and self._quiet_count >= QUIET_TICKS):
                self._armed, self._requalifying = True, False
            if not self._accepted:
                # Surveillance continues even while no predictions are emitted.
                for path, value in enumerate(values):
                    if path not in self._pending and value >= self.high[path]:
                        self._pending[path] = tick
                        self._armed = False
                        self._requalifying = True
                        self._quiet_count = 0
            had_pending = bool(self._pending)
            for path, crossing in tuple(self._pending.items()):
                if values[path] < self.low[path]:
                    del self._pending[path]
                elif tick - crossing >= CONFIRMATION_TICKS:
                    self._accepted.append((crossing, path))
                    self._reach_seen = True
                    del self._pending[path]
            if self._accepted:
                crossing, _ = min(self._accepted)
                if (self._onset is None
                        and not any(other <= crossing for other in self._pending.values())):
                    self._onset = AcceptedOnset(self.identity, crossing, tick)
                self._armed = False
                self._quiet_count = 0
            elif self._pending:
                self._quiet_count = 0
            elif self._requalifying:
                if had_pending:
                    # The rejection tick may begin a fresh [r,r+250) interval.
                    self._quiet_count = 0
                quiet = all(value <= low for value, low in zip(values, self.low))
                self._quiet_count = self._quiet_count + 1 if quiet else 0
        return self._snapshot(tick + 1, decision_tick=tick)

    def close(self, stop_tick, *, reason="administrative"):
        """Close before the next sample; never adjudicate across this boundary."""
        _require(reason in ("administrative", "gap"), "unknown closing boundary")
        stop_tick = self._next_tick(stop_tick)
        return self._cut(stop_tick, reason, terminal=True)


def label_from_snapshots(decision_snapshot, evidence_snapshot):
    """Join only a saved decision and later as-of evidence for its exact cue.

    There is deliberately no caller-selected prefix argument: a shorter prefix
    needs its own saved snapshot, including candidates unresolved at that time.
    The stream token prevents joining independently replayed divergent futures;
    it is neither source authentication nor a data-access permission mechanism.
    """
    _require(isinstance(decision_snapshot, WayRiskSnapshot)
             and isinstance(evidence_snapshot, WayRiskSnapshot),
             "two stream snapshots are required")
    _require(decision_snapshot.decision is not None, "snapshot has no grid decision")
    _require(decision_snapshot.cue is not None
             and decision_snapshot.cue == evidence_snapshot.cue,
             "decision and evidence must belong to the same nonempty cue scope")
    _require(decision_snapshot._stream_token is evidence_snapshot._stream_token
             and evidence_snapshot._sequence >= decision_snapshot._sequence,
             "evidence must be a same-stream snapshot at or after the decision")
    return label_for_decision(decision_snapshot.decision, evidence_snapshot.followup,
                              evidence_snapshot.onset, ticks_per_second=NATIVE_RATE_HZ,
                              grid_origin_tick=decision_snapshot.grid_origin_tick)
