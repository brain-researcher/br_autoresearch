"""Prepare all 72 EP18 physical sequences from an explicit trusted raw reader.

No files are opened here and no empirical recipe is selected. The coordinated
caller supplies the original main-event rows, frozen main join, E1 markers,
recording-segment boundaries and physical signal spans. A reader returns only
the requested raw 1000-Hz interval, in volts and its declared channel order.
Header scaling, participant access and authentic boundary derivation remain
the trusted caller's responsibilities. Grouping 309 events is not a boundary
derivation. Repeated-session data cannot enter this primary preparation.

The required recipe selects finite FIR taps, their origin, a recording-grid
decimation phase, and a fixed optional time-basis projection. Its only supported
order is explicitly declared: detrend_then_fir_then_decimate. A selected fixed
projection depends on sample times, not labels, effects or model residuals.
No estimated transform is transferred between sequences. The bridge does not
select anti-alias coefficients, response-context trimming or a score interval.

For FIR length K and origin o, y[n] = sum_k h[k] x[n+o-k]. Zero extension is
confined to the same physical sequence; the first K-1-o and last o source
samples are always excluded from the shared fit/score mask. The caller's
explicit score interval can only further restrict that mask. All 309 events
survive regardless of trimming, with unrounded onsets on the original clock.

The result contains 62 ordered channels. continuous_fold_runner applies the
existing fixed CommonSensorSpace projection to 61 dimensions exactly once.
This module neither whitens nor performs baseline correction or channel repair.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.signal import convolve

from continuous_fold_runner import PhysicalSequence


@dataclass(frozen=True)
class RecordingSegment:
    segment_id: str
    start_sample: int
    stop_sample: int


@dataclass(frozen=True)
class StimulusMarker:
    marker_id: str
    sample_one_based: int


@dataclass(frozen=True)
class SequenceSpan:
    """Explicit half-open raw and score intervals in recording sample indices."""
    sequence_id: str
    block: int
    within_block: int
    physical_sequence: int
    recording_segment_id: str
    start_sample: int
    stop_sample: int
    score_start_sample: int
    score_stop_sample: int


@dataclass(frozen=True)
class SequencePreparationRecipe:
    fir_coefficients: Sequence[float]
    fir_origin_sample: int
    decimation_phase_mod4: int
    detrend: str
    operation_order: str
    source_sample_rate_hz: int
    target_sample_rate_hz: int
    input_sample_unit: str


@dataclass(frozen=True)
class PreservedEvent:
    """Original and frozen rows, marker identity and explicit segment join."""
    source_row: Mapping[str, str]
    frozen_row: Mapping[str, str]
    marker_id: str
    recording_segment_id: str
    sample_zero_based: int
    onset_sequence_seconds: float


@dataclass(frozen=True)
class SequenceEventInputs:
    participant_id: str
    span: SequenceSpan
    events: tuple[PreservedEvent, ...]
    onsets_seconds: np.ndarray
    source_sample_indices: np.ndarray
    filter_left_support_samples: int
    filter_right_support_samples: int


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            number = np.nan
        if np.isfinite(number) and number.is_integer():
            return int(number)
    raise ValueError(f"{name} must be an integer")


def _number(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and real")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite and real") from error
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite and real")
    return number


def _name(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be an explicit nonempty string")
    return value


def _readonly(values, dtype=None):
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _copy_row(row):
    if not isinstance(row, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in row.items()):
        raise ValueError("Supply original text metadata rows without lossy conversions")
    return MappingProxyType(dict(row))


def _recipe(recipe):
    if not isinstance(recipe, SequencePreparationRecipe):
        raise ValueError("Supply the complete explicit sequence preparation recipe")
    if (recipe.source_sample_rate_hz != 1000 or recipe.target_sample_rate_hz != 250
            or isinstance(recipe.source_sample_rate_hz, bool)
            or isinstance(recipe.target_sample_rate_hz, bool)
            or recipe.input_sample_unit != "V"
            or recipe.operation_order != "detrend_then_fir_then_decimate"
            or recipe.detrend not in {"none", "fixed_constant_projection", "fixed_linear_projection"}):
        raise ValueError("Explicit 1000-to-250 Hz, volts and supported fixed operation choices required")
    raw = np.asarray(recipe.fir_coefficients)
    if raw.dtype.kind not in "iuf":
        raise ValueError("FIR taps must be real numeric values")
    taps = np.array(raw, dtype=np.float64, copy=True)
    if taps.ndim != 1 or not len(taps) or not np.isfinite(taps).all() or not np.any(taps):
        raise ValueError("Supply a nonzero finite one-dimensional FIR")
    origin = _integer(recipe.fir_origin_sample, "FIR origin")
    phase = _integer(recipe.decimation_phase_mod4, "decimation phase")
    if not 0 <= origin < len(taps) or not 0 <= phase < 4:
        raise ValueError("FIR origin or recording-grid decimation phase is out of range")
    return taps, origin, phase


def _spans(spans, segments, recording_samples):
    count = _integer(recording_samples, "recording sample count")
    segments = tuple(segments)
    if count < 1 or not segments or any(not isinstance(s, RecordingSegment) for s in segments):
        raise ValueError("Supply complete original recording-segment intervals")
    segment_by_id, end = {}, 0
    for segment in segments:
        _name(segment.segment_id, "recording segment ID")
        if any(isinstance(value, bool) or not isinstance(value, Integral)
               for value in (segment.start_sample, segment.stop_sample)):
            raise ValueError("Segment bounds must be integer recording indices")
        start = _integer(segment.start_sample, "segment start")
        stop = _integer(segment.stop_sample, "segment stop")
        if segment.segment_id in segment_by_id or start != end or not start < stop <= count:
            raise ValueError("Segments must partition the recording in their original order")
        segment_by_id[segment.segment_id] = segment
        end = stop
    if end != count:
        raise ValueError("Recording segments do not cover the original recording")
    spans = tuple(spans)
    if len(spans) != 72 or any(not isinstance(s, SequenceSpan) for s in spans):
        raise ValueError("All 72 physical sequence spans are required")
    locations, physical, identities = set(), set(), set()
    for span in spans:
        _name(span.sequence_id, "physical sequence ID")
        integer_fields = (
            "block", "within_block", "physical_sequence", "start_sample", "stop_sample",
            "score_start_sample", "score_stop_sample")
        if any(isinstance(getattr(span, field), bool) or not isinstance(getattr(span, field), Integral)
               for field in integer_fields):
            raise ValueError("Sequence identities and bounds must be integer recording indices")
        integers = [int(getattr(span, field)) for field in integer_fields]
        block, within, sequence, start, stop, score_start, score_stop = integers
        if (not 0 <= block < 12 or not 0 <= within < 6 or not 0 <= sequence < 72
                or (block, within) in locations or sequence in physical or span.sequence_id in identities):
            raise ValueError("The complete 12-by-6 roster must have unique original identities")
        if span.recording_segment_id not in segment_by_id:
            raise ValueError("Sequence refers to an unknown original recording segment")
        segment = segment_by_id[span.recording_segment_id]
        if not segment.start_sample <= start <= score_start < score_stop <= stop <= segment.stop_sample:
            raise ValueError("Raw and score intervals must stay inside one physical sequence and segment")
        locations.add((block, within))
        physical.add(sequence)
        identities.add(span.sequence_id)
    chronological = sorted(spans, key=lambda s: s.start_sample)
    if any(a.stop_sample > b.start_sample for a, b in zip(chronological, chronological[1:])):
        raise ValueError("Physical sequence raw intervals overlap; fit/score support would be shared")
    return tuple(sorted(spans, key=lambda s: (s.block, s.within_block)))


def _events(participant_id, source_rows, frozen_rows, markers, spans, recording_samples):
    source_rows, frozen_rows, markers = tuple(source_rows), tuple(frozen_rows), tuple(markers)
    if len(source_rows) != 22248 or len(frozen_rows) != 22248:
        raise ValueError("Keep all 22248 original main events; repeated events are a separate analysis")
    marker_by_sample, marker_ids = {}, set()
    for marker in markers:
        if not isinstance(marker, StimulusMarker):
            raise ValueError("Supply original E1 stimulus markers with explicit one-based positions")
        _name(marker.marker_id, "stimulus marker ID")
        sample = _integer(marker.sample_one_based, "marker sample")
        if (not 1 <= sample <= recording_samples or sample in marker_by_sample
                or marker.marker_id in marker_ids):
            raise ValueError("Stimulus marker positions and identities must be unique and in recording")
        marker_by_sample[sample] = marker
        marker_ids.add(marker.marker_id)
    by_sequence = {span.physical_sequence: [] for span in spans}
    span_by_sequence = {span.physical_sequence: span for span in spans}
    previous_sample = -1
    try:
        for source, frozen in zip(source_rows, frozen_rows):
            source, frozen = _copy_row(source), _copy_row(frozen)
            sample = _integer(source["sample"], "source sample")
            zero = sample - 1
            sequence = _integer(source["sequencenumber"], "source physical sequence")
            if sequence not in span_by_sequence:
                raise ValueError("Source event has an unknown physical sequence")
            span = span_by_sequence[sequence]
            source_onset = _number(source["onset"], "source onset")
            source_display = _number(source["time_stimon"], "display onset")
            presentation = _integer(source["presentationnumber"], "presentation number")
            key = "/".join(source["stim"].replace("\\", "/").split("/")[-2:])
            if (_integer(source["isteststim"], "source event class") != 0
                    or frozen["is_repeated"] != "0" or frozen["participant"] != participant_id
                    or _integer(source["blocksequencenumber"], "source block") != span.block
                    or _integer(source["withinsequencenumber"], "within-block sequence") != span.within_block
                    or _integer(frozen["block"], "frozen block") != span.block
                    or _integer(frozen["physical_sequence"], "frozen physical sequence") != sequence
                    or _integer(frozen["presentation"], "frozen presentation") != presentation
                    or presentation != len(by_sequence[sequence])
                    or _integer(frozen["sample"], "frozen source sample") != sample
                    or _integer(frozen["sample_zero_based"], "normalized sample") != zero
                    or sample <= previous_sample or sample not in marker_by_sample
                    or not span.start_sample <= zero < span.stop_sample
                    or round(source_onset * 1000) != sample
                    or _number(frozen["source_onset_seconds"], "retained source onset") != source_onset
                    or _number(frozen["source_time_stimon_seconds"], "retained display onset") != source_display
                    or abs(_number(frozen["onset_recording_seconds"], "recording onset") - zero / 1000) > 1e-9
                    or _number(source["duration"], "source duration") != 50
                    or _number(frozen["duration_seconds"], "frozen duration") != .05
                    or frozen["canonical_image"] != key
                    or _integer(source["objectnumber"], "source concept") != _integer(frozen["concept_number"], "frozen concept")
                    or _integer(source["stimnumber"], "source exemplar") != _integer(frozen["exemplar_number"], "frozen exemplar")):
                raise ValueError("Original event, frozen join, marker, duration or physical membership disagree")
            _name(frozen["archive_member"], "exact frozen archive member")
            by_sequence[sequence].append(PreservedEvent(source, frozen, marker_by_sample[sample].marker_id,
                span.recording_segment_id, zero, (zero - span.start_sample) / 1000))
            previous_sample = sample
    except KeyError as error:
        raise ValueError("Original or frozen main-event metadata lacks required fields") from error
    marker_positions = np.array(sorted(sample - 1 for sample in marker_by_sample), dtype=np.int64)
    for span in spans:
        events = by_sequence[span.physical_sequence]
        if len(events) != 309:
            raise ValueError("Every physical sequence must preserve all 309 events")
        inside = marker_positions[(marker_positions >= span.start_sample) & (marker_positions < span.stop_sample)]
        if not np.array_equal(inside, [event.sample_zero_based for event in events]):
            raise ValueError("A physical signal span contains an unmodeled stimulus marker")
    return by_sequence


def prepare_participant_sequences(*, participant_id: str, main_source_events, main_frozen_join,
                                  stimulus_markers, recording_segments, recording_sample_count,
                                  sequence_spans, original_channel_order, frozen_channel_order,
                                  recipe: SequencePreparationRecipe,
                                  read_samples: Callable[[int, int], np.ndarray],
                                  budget_check: Callable[[], None]):
    """Return all 72 prepared sequences or raise; never return a partial roster.

    read_samples(start, stop) returns [stop-start, original channels] raw samples
    in volts. It may be a bounded memmap slice reader owned by the trusted caller.
    Metadata is checked before the first reader call. No source signal interval
    overlaps another sequence. Only the frozen 62 channels enter preparation;
    the caller's per-channel scale conversion must precede this function.

    The FIR must include whatever anti-alias/filter recipe the caller selected;
    supplying taps is not evidence that its frequency response is scientifically
    adequate. No cutoff or sample-retention choice is inferred from EEG values.
    """
    _name(participant_id, "participant ID")
    if not callable(read_samples) or not callable(budget_check):
        raise ValueError("Explicit bounded source reader and budget callback required")
    budget_check()
    taps, origin, phase = _recipe(recipe)
    spans = _spans(sequence_spans, recording_segments, recording_sample_count)
    original, frozen = tuple(original_channel_order), tuple(frozen_channel_order)
    if (any(not isinstance(name, str) or not name for name in original + frozen)
            or len(frozen) != 62 or len(set(frozen)) != 62 or len(set(original)) != len(original)
            or not set(frozen).issubset(original)):
        raise ValueError("All 62 exact frozen channels must be present once; no repair or replacement")
    selected_columns = [original.index(name) for name in frozen]
    events = _events(participant_id, main_source_events, main_frozen_join, stimulus_markers,
                     spans, _integer(recording_sample_count, "recording sample count"))
    left, right = len(taps) - 1 - origin, origin
    grids = []
    for span in spans:
        count = span.stop_sample - span.start_sample
        local = np.arange((phase - span.start_sample) % 4, count, 4, dtype=np.int64)
        absolute = local + span.start_sample
        mask = ((local >= left) & (local < count - right)
                & (absolute >= span.score_start_sample) & (absolute < span.score_stop_sample))
        if not mask.any():
            raise ValueError("Explicit recipe leaves a physical sequence with no retained samples")
        grids.append((local, mask))
    prepared = []
    for span, (local, mask) in zip(spans, grids):
        budget_check()
        count = span.stop_sample - span.start_sample
        raw = np.asarray(read_samples(span.start_sample, span.stop_sample))
        if raw.dtype.kind not in "iuf" or raw.shape != (count, len(original)):
            raise ValueError("Raw reader returned a different sample axis or channel inventory")
        values = np.array(raw[:, selected_columns], dtype=np.float64, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite primary-channel EEG; no data-dependent repair or sample deletion")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            if recipe.detrend != "none":
                values -= values.mean(axis=0, keepdims=True)
                if recipe.detrend == "fixed_linear_projection":
                    axis = np.arange(count, dtype=np.float64) - (count - 1) / 2
                    denominator = axis @ axis
                    if denominator <= 0:
                        raise ValueError("Fixed linear projection requires at least two source samples")
                    values -= axis[:, None] * ((axis @ values) / denominator)[None, :]
            filtered = convolve(values, taps[:, None], mode="full", method="auto")[origin:origin + count]
            output = filtered[local]
        if not np.isfinite(output).all():
            raise ValueError("Prepared EEG is nonfinite; no numerical clipping is permitted")
        sequence_events = tuple(events[span.physical_sequence])
        predictors = SequenceEventInputs(participant_id, span, sequence_events,
            _readonly([event.onset_sequence_seconds for event in sequence_events]),
            _readonly([event.sample_zero_based for event in sequence_events], np.int64), left, right)
        prepared.append(PhysicalSequence(span.sequence_id, span.block, span.within_block,
            _readonly(local / 1000), _readonly(output), frozen, _readonly(mask), predictors))
        budget_check()
    return tuple(prepared)
