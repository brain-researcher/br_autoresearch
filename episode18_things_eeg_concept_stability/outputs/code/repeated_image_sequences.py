"""Development repeated-image metadata binding and sequence-confined EEG reads.

This separate adapter preserves the main-session reader's admission rules.
The 12-by-200 repeated-sequence schema is checked against E3/E1 metadata before
any EEG amplitude is read; missing main-session labels are not invented.
No concept score, false group, image feature, audit reader or learned transform
is part of this module. Execute entrypoints only in coordinated Slurm compute.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter
import csv
from dataclasses import dataclass
import gzip
import json
import os
from pathlib import Path
import re

import numpy as np
from scipy.signal import convolve

from brainvision_sequence_reader import (
    BrainVisionMetadata, bind_development_paths, read_recording_metadata,
    _compute_only)
from prepare_continuous_eeg import _integer
from qualify_structural_inputs import image_key


@dataclass(frozen=True)
class RepeatedSequence:
    sequence_id: str
    ordinal: int  # Chronological, zero-based; human odd/even uses ordinal + 1.
    sample_times_seconds: np.ndarray
    sensor_samples: np.ndarray
    channel_order: tuple[str, ...]
    retained_mask: np.ndarray
    event_onsets_seconds: np.ndarray
    image_indices: np.ndarray
    target_flags: np.ndarray
    response_flags: np.ndarray
    cue_seconds: float
    final_offset_seconds: float


@dataclass(frozen=True)
class RepeatedParticipant:
    participant_id: str
    image_ids: tuple[str, ...]
    sequences: tuple[RepeatedSequence, ...]
    metadata_summary: dict


@dataclass(frozen=True)
class BoundRepeatedSequence:
    sequence_id: str
    ordinal: int
    start_sample: int
    stop_sample: int
    score_start_sample: int
    score_stop_sample: int
    segment_id: str
    event_samples: tuple[int, ...]
    image_indices: tuple[int, ...]
    target_flags: tuple[int, ...]
    response_flags: tuple[int, ...]


@dataclass(frozen=True)
class BoundRepeatedParticipant:
    participant_id: str
    image_ids: tuple[str, ...]
    sequences: tuple[BoundRepeatedSequence, ...]
    recording: BrainVisionMetadata
    channel_order: tuple[str, ...]
    fir_coefficients: tuple[float, ...]
    metadata_summary: dict


SOURCE_FIELDS = ("isteststim", "sample", "onset", "time_stimon", "duration",
                 "stim", "istarget", "response")
FROZEN_FIELDS = ("participant", "is_repeated", "sample", "sample_zero_based",
                 "canonical_image", "source_onset_seconds", "source_time_stimon_seconds",
                 "onset_recording_seconds", "duration_seconds")
OPTIONAL_SOURCE_FIELDS = ("teststimnumber", "sequencenumber", "presentationnumber",
                          "blocksequencenumber", "withinsequencenumber", "stimnumber")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _readonly(values, dtype=None):
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _json(path, cap):
    path = Path(path)
    require(path.stat().st_size <= cap, "metadata file exceeds its bounded read")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _source_repeated(paths, check):
    require(paths.events.stat().st_size <= 32 * 1024**2, "source event table exceeds read bound")
    rows, counts = [], Counter()
    with paths.events.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = reader.fieldnames or ()
        require(len(fields) == len(set(fields)) and set(SOURCE_FIELDS).issubset(fields),
                "repeated source schema lacks required timing, identity or binary task fields")
        selected = SOURCE_FIELDS + tuple(name for name in OPTIONAL_SOURCE_FIELDS if name in fields)
        for index, row in enumerate(reader):
            if index % 1024 == 0:
                check()
            require(index < 24648 and None not in row and all(row[name] is not None for name in selected),
                    "source event table is malformed or exceeds original event count")
            kind = _integer(row["isteststim"], "source event class")
            require(kind in (0, 1), "source event class is not binary")
            counts[kind] += 1
            if kind == 1:
                rows.append({name: row[name] for name in selected})
    require(counts == {0: 22248, 1: 2400}, "source must retain all original main and repeated events")
    return tuple(rows)


def _frozen_repeated(paths, participants, check):
    require(paths.frozen_development_join.stat().st_size <= 512 * 1024**2,
            "frozen development join exceeds bounded read")
    rows, counts, seen_person = [], Counter(), False
    with gzip.open(paths.frozen_development_join, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        require(set(FROZEN_FIELDS).issubset(reader.fieldnames or ()), "frozen repeated join lacks required fields")
        for index, row in enumerate(reader):
            if index % 4096 == 0:
                check()
            require(index < 739440 and row["participant"] in participants,
                    "frozen development join contains an undeclared participant or excess rows")
            # The frozen join is participant-contiguous. Early termination is
            # safe only after complete coverage of the selected participant.
            if row["participant"] != paths.participant_id:
                if seen_person:
                    require(counts == {"0": 22248, "1": 2400}, "selected frozen participant is incomplete")
                    break
                continue
            seen_person = True
            require(row["is_repeated"] in ("0", "1"), "frozen event class is not binary")
            counts[row["is_repeated"]] += 1
            if row["is_repeated"] == "1":
                rows.append({name: row[name] for name in FROZEN_FIELDS})
    require(counts == {"0": 22248, "1": 2400}, "frozen development repeated roster is incomplete")
    return tuple(rows)


def _bind_rows(metadata, source_rows, frozen_rows, channels, taps, check):
    """Pure metadata binding; exact source/identity/marker joins precede reads."""
    require(isinstance(metadata, BrainVisionMetadata), "bound recording metadata required")
    person = metadata.paths.participant_id
    channels, taps = tuple(channels), tuple(taps)
    require(len(channels) == len(set(channels)) == 62 and set(channels).issubset(metadata.channel_order),
            "all 62 frozen channels are required without replacement")
    require(len(taps) == 129 and np.isfinite(taps).all() and np.any(np.asarray(taps) != 0),
            "the selected finite 129-tap FIR is required")
    require(len(source_rows) == len(frozen_rows) == 2400, "all 2400 repeated events are required")
    frozen = {}
    for row in frozen_rows:
        sample = _integer(row["sample"], "frozen source sample")
        require(sample not in frozen and row["participant"] == person and row["is_repeated"] == "1",
                "frozen repeated rows are duplicated or outside the selected participant")
        frozen[sample] = row
    events, previous = [], -1
    for row in source_rows:
        check()
        sample = _integer(row["sample"], "source sample")
        zero = sample - 1
        require(_integer(row["isteststim"], "source event class") == 1 and sample in frozen
                and previous < zero < metadata.recording_sample_count,
                "repeated event order, class or recording sample differs")
        previous = zero
        reference = frozen[sample]
        identity = image_key(row["stim"])
        require(identity is not None and identity == reference["canonical_image"]
                and _integer(reference["sample_zero_based"], "normalized source sample") == zero
                and row["onset"] == reference["source_onset_seconds"]
                and row["time_stimon"] == reference["source_time_stimon_seconds"]
                and float(reference["onset_recording_seconds"]) == zero / 1000.
                and round(float(row["onset"]) * 1000) == sample
                and float(row["duration"]) == 50. and float(reference["duration_seconds"]) == .05,
                "repeated source/frozen identity, timing or duration join differs")
        target, response = (_integer(row[name], name) for name in ("istarget", "response"))
        require(target in (0, 1) and response in (0, 1),
                "repeated target/response flags are missing or nonbinary; no zero default is selected")
        events.append((zero, identity, target, response))
    inventory = Counter(row[1] for row in events)
    require(len(inventory) == 200 and set(inventory.values()) == {12},
            "repeated identity coverage must be exactly 200 images shown 12 times")
    image_ids = tuple(sorted(inventory))
    image_index = {identity: index for index, identity in enumerate(image_ids)}
    cues = tuple(row for row in metadata.markers if row.marker_type == "Event"
                 and re.fullmatch(r"E\s*3", row.description))
    starts = [row.sample_zero_based for row in cues]
    require(starts and len(set(starts)) == len(starts) and starts == sorted(starts),
            "recorded E3 sequence starts must be unique and ordered")
    stimuli = tuple(row.sample_one_based - 1 for row in metadata.stimulus_markers)
    require(all(a < b for a, b in zip(stimuli, stimuli[1:])), "recorded E1 positions must be unique and ordered")
    grouped = {}
    for event in events:
        cue = bisect_right(starts, event[0]) - 1
        require(cue >= 0, "repeated event has no preceding E3 marker")
        grouped.setdefault(cue, []).append(event)
    observed_counts = [len(grouped[key]) for key in sorted(grouped)]
    require(len(grouped) == 12 and observed_counts == [200] * 12,
            f"repeated E3 schema unresolved: expected 12 physical sequences of 200 images; observed {observed_counts}")
    segments = metadata.recording_segments
    require(segments and segments[0].start_sample == 0 and segments[-1].stop_sample == metadata.recording_sample_count
            and all(a.stop_sample == b.start_sample for a, b in zip(segments, segments[1:])),
            "original recording segments must partition the recording")
    segment_starts = [row.start_sample for row in segments]
    spans = []
    for ordinal, cue in enumerate(sorted(grouped)):
        check()
        rows = grouped[cue]
        samples = tuple(row[0] for row in rows)
        require(set(row[1] for row in rows) == set(image_ids),
                "each repeated physical sequence must contain all 200 identities once")
        start = starts[cue]
        next_cue = starts[cue + 1] if cue + 1 < len(starts) else metadata.recording_sample_count
        actual = stimuli[bisect_left(stimuli, start):bisect_left(stimuli, next_cue)]
        require(samples == actual, "repeated source samples do not exactly match their E3/E1 interval")
        stop, score_start = samples[-1] + 801, samples[0] - 800
        segment_index = bisect_right(segment_starts, start) - 1
        require(segment_index >= 0, "repeated E3 precedes recording segment support")
        segment = segments[segment_index]
        require(segment.start_sample <= start and start + 128 <= score_start < stop
                and stop <= min(next_cue, segment.stop_sample),
                "complete repeated score/filter support must stay inside one E3 sequence and recording segment")
        require(not spans or spans[-1].stop_sample <= start,
                "repeated physical intervals overlap; fitting/scoring would share raw samples")
        spans.append(BoundRepeatedSequence(f"{person}:repeated:{ordinal}", ordinal, start, stop,
            score_start, stop, segment.segment_id, samples,
            tuple(image_index[row[1]] for row in rows), tuple(row[2] for row in rows), tuple(row[3] for row in rows)))
    optional = {}
    for field in OPTIONAL_SOURCE_FIELDS:
        values = [row.get(field) for row in source_rows]
        optional[field] = {"present": all(value is not None for value in values),
            "missing_value_count": sum(value in (None, "", "n/a", "nan", "None") for value in values),
            "unique_value_count": len(set(values))}
    summary = dict(participant_id=person, repeated_events=2400, unique_images=200, repetitions_per_image=12,
        physical_sequences=12, events_per_sequence=observed_counts, chronological_ordinal_origin=0,
        sequence_identity_source="actual_E3_intervals_with_exact_repeated_E1_join",
        optional_source_fields=optional, source_sample_rate_hz=1000, target_sample_rate_hz=250,
        raw_samples_per_sequence=[span.stop_sample-span.start_sample for span in spans],
        e3_to_first_e1_samples=[span.event_samples[0]-span.start_sample for span in spans],
        common_channels=list(channels), fir_taps=129, filter_past_support_samples=128,
        decimation_recording_phase_mod4=0, delay_compensation=False, learned_preprocessing=False,
        source_role="development_repeated_signal_recovery_only", audit_access=False,
        EEG_amplitudes_read=False, concept_or_false_scores_computed=False,
        evaluation_separation="procedural_same_account_not_technically_enforced")
    return BoundRepeatedParticipant(person, image_ids, tuple(spans), metadata, channels, taps, summary)


def bind_participant(*, development_manifest, bids_root, frozen_channel_order,
                     coefficient_record, participant_id, budget_check):
    """Bind development metadata only. The EEG binary is statted, not opened."""
    _compute_only()
    budget_check()
    manifest = _json(development_manifest, 8 * 1024**2)
    people = manifest.get("participants", ())
    require(manifest.get("role") == "development" and len(people) == len(set(people)) == 30
            and participant_id in people and manifest.get("event_count") == 739440,
            "exact frozen 30-participant development role is required")
    paths = bind_development_paths(development_manifest=development_manifest, bids_root=bids_root,
                                  participant_id=participant_id)
    channels = _json(frozen_channel_order, 65536)["common_recorded_channel_order"]
    coefficients = _json(coefficient_record, 8 * 1024**2)["coefficients"]
    expected = dict(numtaps=129, source_sample_rate_hz=1000, target_sample_rate_hz=250,
        fir_origin_sample=0, detrend="none", decimation_phase_mod4=0,
        operation_order="detrend_then_fir_then_decimate", input_sample_unit="V",
        past_support_samples=128, future_support_samples=0)
    require(all(coefficients.get(name) == value for name, value in expected.items()),
            "coefficient record differs from the adopted fixed causal FIR recipe")
    metadata = read_recording_metadata(paths, header_byte_limit=8 * 1024**2,
        marker_byte_limit=128 * 1024**2, budget_check=budget_check)
    source = _source_repeated(paths, budget_check)
    frozen = _frozen_repeated(paths, set(people), budget_check)
    return _bind_rows(metadata, source, frozen, channels, coefficients["coefficients_float64"], budget_check)


def _read_interval(bound, span, budget_check):
    _compute_only()
    require(span in bound.sequences, "only exactly bound repeated intervals may be read")
    metadata = bound.recording
    count, channels = span.stop_sample-span.start_sample, len(metadata.channel_order)
    # Independent cap for one source interval plus converted voltage data.
    require(count * channels * 12 <= 64 * 1024**2, "one repeated voltage interval exceeds 64 MiB admission")
    budget_check()
    with metadata.paths.binary.open("rb") as stream:
        require(os.fstat(stream.fileno()).st_size == metadata.binary_bytes,
                "source EEG size changed after metadata binding")
        stream.seek(span.start_sample * channels * 4)
        raw = np.fromfile(stream, dtype="<f4", count=count * channels)
    require(raw.size == count * channels, "short repeated EEG interval; no padding is allowed")
    values = raw.reshape(count, channels).astype(np.float64)
    values *= np.asarray(metadata.volts_per_stored_unit)[None, :]
    budget_check()
    return values


def load_bound_participant(bound, *, budget_check):
    """Read only bound repeated spans; preserve fixed filtering and clock."""
    _compute_only()
    require(isinstance(bound, BoundRepeatedParticipant), "a bound repeated participant is required")
    columns = [bound.recording.channel_order.index(name) for name in bound.channel_order]
    taps = np.asarray(bound.fir_coefficients, dtype=np.float64)
    sequences = []
    for span in bound.sequences:
        budget_check()
        values = _read_interval(bound, span, budget_check)[:, columns]
        require(np.isfinite(values).all(), "nonfinite repeated-channel EEG; no outcome-dependent removal")
        count = span.stop_sample-span.start_sample
        local = np.arange((-span.start_sample) % 4, count, 4, dtype=np.int64)
        absolute = local + span.start_sample
        mask = (local >= 128) & (absolute >= span.score_start_sample) & (absolute < span.score_stop_sample)
        require(mask.any(), "fixed repeated scoring interval has no complete-support samples")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            filtered = convolve(values, taps[:, None], mode="full", method="auto")[:count]
        output = filtered[local]
        require(np.isfinite(output).all(), "nonfinite repeated filtered EEG; no clipping is allowed")
        onsets = (np.asarray(span.event_samples)-span.start_sample)/1000.
        sequences.append(RepeatedSequence(span.sequence_id, span.ordinal,
            _readonly(local/1000.), _readonly(output), bound.channel_order, _readonly(mask),
            _readonly(onsets), _readonly(span.image_indices, np.int64),
            _readonly(span.target_flags, np.int64), _readonly(span.response_flags, np.int64),
            0., float(onsets[-1]+.05)))
    summary = dict(bound.metadata_summary, EEG_amplitudes_read=True,
                   retained_samples_per_sequence=[int(row.retained_mask.sum()) for row in sequences])
    return RepeatedParticipant(bound.participant_id, bound.image_ids, tuple(sequences), summary)
