"""Dormant EP07 adaptive-stage accounting from admitted development records.

This module performs no I/O, model fitting, scoring, finalist selection or
outcome opening. The caller supplies the per-candidate ``pairs`` entries from
allowed feedback reports, in attempt order. Record validation does not establish
their provenance. Failed unscored starting placeholders may precede repaired
records or another unscored failure; these retain one starting opportunity.
A second contacted record for one model is refused, never double-counted.
All sixteen attempted starting IDs remain required. Under the adopted
amendment, starting_08 is excluded from required evaluations and must remain
unscored: any declared outcome contact for it is refused. The caller owns the
provenance that its two numerical attempts were exhausted; this helper does
not infer convergence or retry exhaustion from a feedback placeholder.
Starting completion requires complete valid pairs for the other fifteen IDs,
not availability of all sixteen attempted configurations. Unscored IDs and the
adopted exclusion remain separately visible. Other unscored starting failures
do not prohibit adaptive accounting, but remain explicit required deficits.
Incumbents use complete valid starting pairs only. At least one such pair is
needed, and partial/unavailable starting contacts must be resolved before
successor records can be admitted.

At least three valid adaptive pairs are required. Each distinct successor ID
is one attempted paired opportunity. Unscored failures consume the
fourteen-attempt cap, but neither valid-pair count nor
patience. A partial outcome contact or unavailable scored result remains an
explicit technical incompleteness, even if other pairs have valid results.
Recorded stream starts/completions are counted separately; a pair-level start
with no stream entry is retained as an unattributed declared pair contact.

One global incumbent is maintained per stream using the recorded equal mean
of M1/PMd animal-balanced development R2. Strict improvements update that
incumbent, including improvements below .002. Patience resets only when either
stream improves by at least .002 against its immediately preceding incumbent;
small improvements do not accumulate toward a later reset. Ties retain the
earlier incumbent. Completion here concerns only the adaptive stage, never the
ten finalist checks, resource accounting, a full search, or held-out authority.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real
import re


STREAMS = ("history_assisted", "today_only")
STARTING_IDS = tuple(f"starting_{index:02d}" for index in range(16))
EXCLUDED_STARTING_IDS = ("starting_08",)
REQUIRED_STARTING_IDS = tuple(name for name in STARTING_IDS if name not in EXCLUDED_STARTING_IDS)
MAX_ATTEMPTS = 14
MIN_VALID = 3
PATIENCE = 7
IMPROVEMENT = .002


@dataclass(frozen=True)
class Incumbent:
    candidate_id: str
    global_selection_r2: float


@dataclass(frozen=True)
class PairAttempt:
    candidate_id: str
    status: str
    pair_contact_declared: bool
    stream_states: tuple[str, str]
    scores: tuple[float | None, float | None]


@dataclass(frozen=True)
class AdaptiveProgress:
    status: str
    stop_reason: str | None
    adaptive_stage_complete: bool
    starting_complete: bool
    all_attempted_starts_available: bool
    unscored_starting_ids: tuple[str, ...]
    excluded_starting_ids: tuple[str, ...]
    starting_deficit_ids: tuple[str, ...]
    starting_records: tuple[PairAttempt, ...]
    successor_records: tuple[PairAttempt, ...]
    attempted_successor_pairs: int
    valid_successor_pairs: int
    consecutive_valid_without_qualifying_improvement: int
    incumbents: tuple[Incumbent | None, Incumbent | None]
    streams: tuple[str, str]
    paired_opportunities_per_stream: tuple[int, int]
    recorded_started_evaluations: tuple[int, int]
    recorded_completed_evaluations: tuple[int, int]
    available_evaluations: tuple[int, int]
    declared_pair_contacts: int
    unattributed_declared_pair_contacts: int
    incomplete_contact_ids: tuple[str, ...]
    model_space_exhausted: bool
    whole_search_complete: bool = False
    authorizes_held_out: bool = False


def _finite(value):
    return (not isinstance(value, bool) and isinstance(value, Real)
            and math.isfinite(float(value)))


def _stream(record):
    if not isinstance(record, Mapping):
        raise ValueError("a stream record must be a mapping")
    if record.get("state") == "development_contact_started":
        if "complete" in record or "regions" in record:
            raise ValueError("a stream cannot be both pending and completed")
        return "started", None
    if type(record.get("complete")) is not bool:
        raise ValueError("completed stream feedback requires an explicit availability flag")
    regions = record.get("regions")
    if (not isinstance(regions, (tuple, list)) or len(regions) != 2
            or any(not isinstance(region, Mapping) for region in regions)
            or {region.get("region") for region in regions} != {"M1", "PMd"}):
        raise ValueError("feedback requires exactly M1 and PMd aggregate records")
    for region in regions:
        if (type(region.get("available")) is not bool
                or type(region.get("complete_new_days")) is not int
                or region["complete_new_days"] != 5
                or type(region.get("independent_animals")) is not int
                or region["independent_animals"] != 2):
            raise ValueError("feedback must retain the complete five-day/two-animal panel")
        values = tuple(region.get(name) for name in
                       ("model_r2", "calibration_mean_r2", "improvement_over_mean"))
        if region["available"]:
            if region.get("unavailable_reason") is not None or not all(map(_finite, values)):
                raise ValueError("available regional feedback must be finite")
        elif (not isinstance(region.get("unavailable_reason"), str)
              or not region["unavailable_reason"] or any(value is not None for value in values)):
            raise ValueError("unavailable regional feedback must retain its reason and null scores")
    available = all(region["available"] for region in regions)
    if record["complete"] != available:
        raise ValueError("stream availability disagrees with its regional feedback")
    score = record.get("global_selection_r2")
    if not available:
        if score is not None:
            raise ValueError("an unavailable stream has no selection score")
        return "unavailable", None
    expected = sum(float(region["model_r2"]) for region in regions) / 2
    if not _finite(score) or not math.isfinite(expected) or float(score) != expected:
        raise ValueError("selection score must equal the fixed mean of the two regional R2 values")
    return "available", float(score)


def _pair(record):
    if not isinstance(record, Mapping):
        raise ValueError("a candidate feedback record must be a mapping")
    candidate_id = record.get("candidate_id")
    if (not isinstance(candidate_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", candidate_id) is None):
        raise ValueError("a simple nonempty candidate ID is required")
    contacted, state, streams = (record.get(name) for name in
                                ("outcome_contact", "state", "streams"))
    if type(contacted) is not bool or not isinstance(streams, Mapping) or set(streams) - set(STREAMS):
        raise ValueError("feedback requires explicit contact state and only the two model streams")
    if not contacted:
        if state != "paired_prediction_incomplete_unscored" or streams:
            raise ValueError("an unscored failure cannot contain outcome feedback")
        return PairAttempt(candidate_id, "unscored_failure", False,
                           ("absent", "absent"), (None, None))
    if state not in ("paired_development_contact_started", "paired_development_scored"):
        raise ValueError("contacted candidate has an unknown completion state")
    values = tuple(_stream(streams[name]) if name in streams else ("absent", None)
                   for name in STREAMS)
    stream_states, scores = tuple(value[0] for value in values), tuple(value[1] for value in values)
    finished = all(value in ("available", "unavailable") for value in stream_states)
    if state == "paired_development_scored" and not finished:
        raise ValueError("paired completion cannot conceal a missing or unfinished stream")
    status = ("valid_pair" if all(value == "available" for value in stream_states)
              else "unavailable_pair") if state == "paired_development_scored" else "partial_contact"
    return PairAttempt(candidate_id, status, True, stream_states, scores)


def _starting(records):
    groups = {}
    for raw in records:
        record = _pair(raw)
        if record.candidate_id not in STARTING_IDS:
            raise ValueError("starting feedback must use the exact sixteen starting IDs")
        if record.candidate_id in EXCLUDED_STARTING_IDS and record.pair_contact_declared:
            raise ValueError("the adopted excluded starting_08 must remain unscored")
        previous = groups.get(record.candidate_id)
        if previous is not None:
            if previous.pair_contact_declared:
                raise ValueError("duplicate starting feedback or second outcome contact is not mergeable")
        groups[record.candidate_id] = record
    if set(groups) != set(STARTING_IDS):
        raise ValueError("all sixteen starting IDs are required before adaptive accounting")
    return tuple(groups[name] for name in STARTING_IDS)


def assess_adaptive_progress(starting_records, successor_records=(), *, model_space_exhausted=False):
    """Replay a bounded, ordered feedback history without selecting a finalist.

    ``starting_records`` can concatenate first-attempt and repair report entries.
    All sixteen attempted IDs are retained, including unscored starting_08;
    only the other fifteen are required for ``starting_complete``. Its retry
    exhaustion is caller-admitted provenance, not inferred from these records.
    ``successor_records`` has one distinct ID per attempted paired opportunity;
    it is not a list of retries or repeated snapshots. All limits are fixed by
    the episode contract. Exhaustion must be explicitly declared by the caller.
    """
    if type(model_space_exhausted) is not bool:
        raise ValueError("model-space exhaustion must be an explicit boolean")
    starting = _starting(starting_records)
    successors = tuple(_pair(record) for record in successor_records)
    if len(successors) > MAX_ATTEMPTS:
        raise ValueError("more than fourteen attempted adaptive pairs exceeds the fixed cap")
    ids = tuple(record.candidate_id for record in successors)
    if len(set(ids)) != len(ids) or set(ids).intersection(STARTING_IDS):
        raise ValueError("each successor must be a distinct new paired opportunity")
    valid_starting = tuple(record for record in starting if record.status == "valid_pair")
    starting_deficits = tuple(record.candidate_id for record in starting
                             if record.candidate_id in REQUIRED_STARTING_IDS
                             and record.status != "valid_pair")
    starting_complete = not starting_deficits
    all_attempted_available = all(record.status == "valid_pair" for record in starting)
    unscored_starting = tuple(record.candidate_id for record in starting
                             if record.status == "unscored_failure")
    unresolved_starting_contacts = any(record.status in ("partial_contact", "unavailable_pair")
                                      for record in starting)
    if successors and (not valid_starting or unresolved_starting_contacts):
        raise ValueError("adaptive accounting needs a complete starting pair and no unresolved starting contacts")
    leaders = [None, None]
    for record in valid_starting:
        for index, score in enumerate(record.scores):
            if leaders[index] is None or score > leaders[index].global_selection_r2:
                leaders[index] = Incumbent(record.candidate_id, score)
    valid, patience = 0, 0
    for record in successors:
        if patience >= PATIENCE and valid >= MIN_VALID:
            raise ValueError("adaptive feedback continues after the seven-valid-pair stopping rule")
        if record.status != "valid_pair":
            continue
        improvements = tuple(score - leader.global_selection_r2
                             for score, leader in zip(record.scores, leaders))
        valid += 1
        patience = 0 if any(value >= IMPROVEMENT for value in improvements) else patience + 1
        for index, score in enumerate(record.scores):
            if score > leaders[index].global_selection_r2:
                leaders[index] = Incumbent(record.candidate_id, score)
    records = starting + successors
    incomplete = tuple(record.candidate_id for record in records
                       if record.status in ("partial_contact", "unavailable_pair"))
    stop = ("maximum_attempted_pairs" if len(successors) == MAX_ATTEMPTS else
            "patience_exhausted" if patience >= PATIENCE else
            "model_space_exhausted" if model_space_exhausted else None)
    if incomplete:
        status, stop = "incomplete_technical", "incomplete_outcome_contact"
    elif not valid_starting:
        status, stop = "incomplete_starting", "no_complete_available_starting_pair"
    elif stop is not None:
        status = (("adaptive_complete" if starting_complete else "adaptive_complete_with_incomplete_starting")
                  if valid >= MIN_VALID else "incomplete_minimum_valid")
    else:
        status = "continue" if starting_complete else "continue_with_incomplete_starting"
    count = lambda states, index: sum(record.stream_states[index] in states for record in records)
    opportunities = len(STARTING_IDS) + len(successors)
    adaptive_complete = status in ("adaptive_complete", "adaptive_complete_with_incomplete_starting")
    return AdaptiveProgress(status, stop, adaptive_complete, starting_complete,
        all_attempted_available, unscored_starting, EXCLUDED_STARTING_IDS, starting_deficits,
        starting, successors,
        len(successors), valid, patience, tuple(leaders), STREAMS, (opportunities, opportunities),
        tuple(count({"started", "unavailable", "available"}, index) for index in range(2)),
        tuple(count({"unavailable", "available"}, index) for index in range(2)),
        tuple(count({"available"}, index) for index in range(2)),
        sum(record.pair_contact_declared for record in records),
        sum(record.pair_contact_declared and record.stream_states == ("absent", "absent")
            for record in records), incomplete, model_space_exhausted)
