"""Metadata-only fitting rosters for ordinary EP07 candidates and named controls.

The default always requires all three history sources (or zero for today-only).
The private control context is used by primary_control, never inferred from a
short input roster. No arbitrary subset, changed trial role or fitted state is
accepted. These bindings do not authenticate caller-supplied data or authorize
an outcome opening.
"""
from dataclasses import dataclass
from numbers import Integral

from native_trials import DayTrials, EARLIER, NEW


@dataclass(frozen=True)
class PrimaryControl:
    kind: str
    source_position: int | None = None


def validate_control(control):
    if not isinstance(control, PrimaryControl):
        raise TypeError("an explicit PrimaryControl is required")
    if control.kind not in ("remove_history", "omit_source", "remove_nuisance"):
        raise ValueError("unknown primary fitting control")
    if control.kind == "omit_source":
        if (isinstance(control.source_position, bool)
                or not isinstance(control.source_position, Integral)
                or control.source_position not in (1, 2, 3)):
            raise ValueError("source_position must be chronological position 1, 2 or 3")
    elif control.source_position is not None:
        raise ValueError("source_position is used only by omit_source")
    return control


def source_days(animal, stream, *, _control=None):
    if animal not in EARLIER or stream not in ("history_assisted", "today_only"):
        raise ValueError("one registered animal and primary stream are required")
    sources = EARLIER[animal] if stream == "history_assisted" else ()
    if _control is None:
        return sources
    control = validate_control(_control)
    if control.kind in ("remove_history", "omit_source"):
        if stream != "history_assisted":
            raise PermissionError("source removal is a history-only control")
        return (() if control.kind == "remove_history" else
                tuple(day for index, day in enumerate(sources, 1)
                      if index != control.source_position))
    return sources


def bind_fit_days(fit_days, *, _control=None):
    """Authenticate exact scope/roles/roster before interpreting value arrays.

    Return retained days chronologically, followed by target calibration. A
    control call here receives the already selected roster; primary_control
    first authenticates the complete original roster before selecting it.
    """
    days = tuple(fit_days)
    if not days or any(not isinstance(day, DayTrials) for day in days):
        raise ValueError("complete DayTrials fitting packs are required")
    first = days[0]
    if (first.animal not in NEW or first.target_day not in NEW[first.animal]
            or first.region not in ("M1", "PMd")
            or first.stream not in ("history_assisted", "today_only")):
        raise ValueError("unknown animal, target, region or stream")
    scope = first.animal, first.target_day, first.region, first.stream
    for day in days:
        if ((day.animal, day.target_day, day.region, day.stream) != scope
                or day.view_role != "fit"
                or day.role != ("calibration" if day.day == day.target_day else "training")):
            raise PermissionError("fitting accepts one training/calibration scope only")
    expected = source_days(first.animal, first.stream, _control=_control) + (first.target_day,)
    by_day = {day.day: day for day in days}
    if len(by_day) != len(days) or set(by_day) != set(expected):
        raise ValueError("the exact declared source/calibration roster is required")
    return tuple(by_day[day] for day in expected)
