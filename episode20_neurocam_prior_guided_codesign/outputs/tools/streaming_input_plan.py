"""Replay storage windows over ONE already compiled EP20 midpoint hold.

Window/aperture splitting does not choose new midpoint keys, resample the raw
world or reset any physical state. An existing compact hold tuple or a bounded
replayable hold factory supplies the same original intervals. Whole-plan and field replay validation is
owned by StreamingArrayAcquisition, before any physical advancement.
"""

from dataclasses import replace
import math
from numbers import Real

from array_acquisition import ProviderInterval
from raw_input_hold import MidpointHoldPlan
from streaming_midpoint_hold import StreamingMidpointHoldPlan
from streaming_acquisition_readout import ReadoutWindow
from streaming_array_acquisition import ArrayWindow, StreamPhysicalInterval


def bind_hold_windows(hold_plan, readout_window_factory, *, max_windows,
                      max_window_fragments, check):
    """Return a bounded replayable zero-argument ArrayWindow iterator factory.

    ``readout_window_factory`` must yield the same finite ReadoutWindow stream
    on each call. Its windows must exactly partition the hold's time support.
    Original hold interval ordinal is the physical event ID; a seam continues
    that same event with exactly the same field key/row/DAQ/reference and no
    repeated boundary charge. The original error bound continues to cover the
    identical held waveform. No numerical field is evaluated by this adapter.
    """
    if (not isinstance(hold_plan, (MidpointHoldPlan, StreamingMidpointHoldPlan))
            or not callable(hold_plan.field_provider)
            or not callable(readout_window_factory) or not callable(check)):
        raise ValueError("An existing finite hold plan, replayable readout factory and check are required")
    for value in (max_windows, max_window_fragments):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("Explicit positive window/fragment bounds are required")
    def validate(interval, previous_end):
        check()
        if not isinstance(interval, ProviderInterval):
            raise ValueError("Hold must supply compact ProviderInterval records")
        if any(isinstance(t, bool) or not isinstance(t, Real) or not math.isfinite(t)
               for t in (interval.start, interval.end)) or not interval.start < interval.end:
            raise ValueError("Each original hold interval must have finite positive duration")
        if previous_end is not None and interval.start != previous_end:
            raise ValueError("Original hold intervals must be exactly contiguous")
    if isinstance(hold_plan, MidpointHoldPlan):
        intervals = hold_plan.intervals
        if not isinstance(intervals, tuple) or not intervals:
            raise ValueError("Retained hold must contain a nonempty compact tuple")
        previous_end = None
        for interval in intervals:
            validate(interval, previous_end)
            previous_end = interval.end
        support_start, support_end = intervals[0].start, intervals[-1].end
        interval_factory = lambda: iter(intervals)
    else:
        support_start, support_end = hold_plan.start, hold_plan.end
        interval_factory = hold_plan.interval_factory
        if not callable(interval_factory):
            raise ValueError("Streaming hold must supply a replayable interval factory")

    def factory():
        next_start, event_index, count = support_start, 0, 0
        original_intervals = iter(interval_factory())
        original = next(original_intervals, None)
        if original is None:
            raise ValueError("Hold factory cannot be empty")
        validate(original, support_start)
        for readout in readout_window_factory():
            check()
            if (not isinstance(readout, ReadoutWindow) or count >= max_windows
                    or readout.index != count or readout.start != next_start
                    or readout.end > support_end):
                raise ValueError("Readout windows must exactly and sequentially cover the original hold support")
            fragments = []
            left = readout.start
            while left < readout.end:
                check()
                if original is None or len(fragments) >= max_window_fragments:
                    raise ValueError("Hold coverage or per-window physical fragment bound exceeded")
                if not original.start <= left < original.end:
                    raise ValueError("Readout/hold partition lost exact event coverage")
                right = min(original.end, readout.end)
                charge = original.injected_charge_coulomb if left == original.start else ()
                fragment = replace(original, start=left, end=right, injected_charge_coulomb=charge)
                fragments.append(StreamPhysicalInterval(event_index, fragment))
                left = right
                if right == original.end:
                    event_index += 1
                    previous_end = original.end
                    original = next(original_intervals, None)
                    if original is not None:
                        validate(original, previous_end)
            yield ArrayWindow(count, readout.start, readout.end, tuple(fragments), readout)
            next_start = readout.end
            count += 1
        if not count or next_start != support_end or original is not None:
            raise ValueError("Readout windows ended before exactly covering the complete hold plan")
    return factory
