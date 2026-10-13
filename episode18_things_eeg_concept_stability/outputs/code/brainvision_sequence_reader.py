"""Development-only BrainVision voltage reader and recorded-sequence adapter.

The existing frozen development manifest binds exact header/marker/binary
leaves. Header text is parsed with the already used structural-input parser;
the binary is only statted during metadata loading. Actual interval reads are
compute-only, explicitly bounded, and restricted to the exact 72 intervals
accepted by prepare_continuous_eeg. No complete recording is loaded or mapped.
The manifest is a procedural role list, not a neural-access grant or OS seal.

Voltage = stored little-endian float32 * per-channel resolution * unit-to-V.
Only the qualified 1000-Hz multiplexed float32 route is supported. Missing
resolution/unit fields are refused rather than guessed. Header conversion is
consistent with MNE's BrainVision reader and pybv's inverse writing operation:
https://raw.githubusercontent.com/mne-tools/mne-python/maint/1.8/mne/io/brainvision/brainvision.py
https://pybv.readthedocs.io/en/stable/_modules/pybv/io.html

The original paper identifies E3 as sequence start, E1 as image onset, E2 as
image offset (page 3), independently of the current EEG outcomes:
https://tijl.github.io/tijl-grootswagers-pdf/Grootswagers_et_al_-_2022_-_Human_EEG_recordings_for_1,854_concepts_presented_.pdf
Recorded E3 positions therefore anchor physical-sequence starts. Each complete
source-labelled sequence must match the E1 markers in its preceding E3 interval;
the actual cue-to-first-onset gap is preserved, never replaced by a nominal
constant. Post-last-onset padding and scoring intervals remain explicit caller
recipe fields. Construction refuses a crossing or inconsistency; it never
clips, drops or repairs a stream.
FIR/detrend/decimation choices remain the separate explicit bridge recipe.

No CLI, automatic execution, scientific recipe default, source mutation,
payload logging, reference-channel synthesis or audit reader is provided.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
import csv
from dataclasses import dataclass
import gzip
import json
from numbers import Integral
import os
from pathlib import Path
import re
import socket
from typing import Callable

import numpy as np

from qualify_structural_inputs import FIELDS, ini
from prepare_continuous_eeg import (RecordingSegment, SequenceSpan, StimulusMarker,
                                    SequencePreparationRecipe, _integer, _number,
                                    _spans, prepare_participant_sequences)


UNIT_TO_VOLTS = {"V": 1., "mV": 1e-3, "µV": 1e-6, "μV": 1e-6, "uV": 1e-6, "nV": 1e-9}


@dataclass(frozen=True)
class DevelopmentRecordingPaths:
    participant_id: str
    header: Path
    marker: Path
    binary: Path
    events: Path
    frozen_development_join: Path


@dataclass(frozen=True)
class RecordedMarker:
    marker_id: str
    marker_type: str
    description: str
    sample_zero_based: int
    points: int
    channel: int


@dataclass(frozen=True)
class BrainVisionMetadata:
    paths: DevelopmentRecordingPaths
    channel_order: tuple[str, ...]
    resolutions: tuple[float, ...]
    source_units: tuple[str, ...]
    volts_per_stored_unit: tuple[float, ...]
    sampling_hz: float
    recording_sample_count: int
    binary_bytes: int
    recording_segments: tuple[RecordingSegment, ...]
    stimulus_markers: tuple[StimulusMarker, ...]
    markers: tuple[RecordedMarker, ...]


@dataclass(frozen=True)
class RecordedSpanRecipe:
    cue_anchor: str
    post_last_inclusive_samples: int
    score_first_offset_samples: int
    score_last_inclusive_samples: int


def _positive_integer(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{label} must be an explicit positive integer")
    return int(value)


def _compute_only():
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Raw EEG and full event preparation require the coordinator's compute allocation")


def bind_development_paths(*, development_manifest, bids_root, participant_id):
    """Reuse the existing role manifest; never inspect audit or a mixed roster."""
    manifest_path, root = Path(development_manifest).resolve(), Path(bids_root).resolve()
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if (manifest.get("role") != "development" or participant_id not in manifest.get("participants", ())
            or not re.fullmatch(r"sub-[0-9]{2}", participant_id)):
        raise ValueError("This adapter accepts only a participant bound by the frozen development manifest")
    prefix = f"{participant_id}/eeg/{participant_id}_task-rsvp_eeg"
    declared = manifest.get("EEG_relative_paths_for_controlled_mount", ())
    relative = tuple(prefix + suffix for suffix in (".vhdr", ".vmrk", ".eeg"))
    if any(declared.count(path) != 1 for path in relative):
        raise ValueError("The development manifest does not bind each exact recording leaf once")
    header, marker, binary = (root / path for path in relative)
    if manifest.get("frozen_event_join") != "development_event_image_join.tsv.gz":
        raise ValueError("The manifest does not bind the frozen development event join")
    # This explicit standard leaf is the same source-event naming used in the
    # completed structural qualification, not a directory/file-name search.
    events = header.parent / f"{participant_id}_task-rsvp_events.tsv"
    return DevelopmentRecordingPaths(participant_id, header, marker, binary, events,
                                     manifest_path.parent / manifest["frozen_event_join"])


def _bounded_ini(path, limit):
    if path.stat().st_size > _positive_integer(limit, "metadata byte limit"):
        raise ValueError("Explicit BrainVision metadata leaf exceeds its admitted byte limit")
    return ini(path)


def read_recording_metadata(paths, *, header_byte_limit, marker_byte_limit, budget_check):
    """Read header/marker text and stat the binary; never open EEG values."""
    if not isinstance(paths, DevelopmentRecordingPaths) or not callable(budget_check):
        raise ValueError("Supply bound development paths and a budget callback")
    budget_check()
    header = _bounded_ini(paths.header, header_byte_limit)
    common, binary_info = header["Common Infos"], header["Binary Infos"]
    if (common.get("Codepage") != "UTF-8" or common.get("DataFormat") != "BINARY"
            or common.get("DataOrientation") != "MULTIPLEXED"
            or binary_info.get("BinaryFormat") != "IEEE_FLOAT_32"
            or common.get("DataFile") != paths.binary.name or common.get("MarkerFile") != paths.marker.name
            or paths.binary.parent != paths.header.parent or paths.marker.parent != paths.header.parent):
        raise ValueError("Recording differs from the explicit qualified UTF-8 multiplexed float32 route")
    if any(_integer(binary_info.get(key, "0"), key) != 0
           for key in ("DataOffset", "ChannelOffset", "SegmentHeaderSize", "TrailerSize")):
        raise ValueError("Unexpected binary offset/trailer layout; no format fallback is selected")
    if binary_info.get("UseBigEndianOrder", "NO") != "NO":
        raise ValueError("This source reader supports the qualified little-endian route only")
    sampling_hz = 1e6 / _number(common["SamplingInterval"], "sampling interval")
    count = _integer(common["NumberOfChannels"], "channel count")
    if sampling_hz != 1000 or count < 62:
        raise ValueError("Expected the qualified 1000-Hz source and full recorded channel inventory")
    channel_info = header["Channel Infos"]
    if set(channel_info) != {f"Ch{index}" for index in range(1, count + 1)}:
        raise ValueError("Recorded channel entries must preserve all original one-based positions")
    names, resolutions, units, scales = [], [], [], []
    for index in range(1, count + 1):
        fields = channel_info[f"Ch{index}"].split(",")
        if len(fields) < 4 or not fields[0] or not fields[2] or fields[3] not in UNIT_TO_VOLTS:
            raise ValueError("Every recorded channel requires its explicit name, resolution and voltage unit")
        resolution = _number(fields[2], "channel resolution")
        if resolution <= 0:
            raise ValueError("Channel resolution must be positive")
        names.append(fields[0].replace(r"\1", ","))
        resolutions.append(resolution)
        units.append(fields[3])
        scales.append(resolution * UNIT_TO_VOLTS[fields[3]])
    if len(set(names)) != count or not np.isfinite(scales).all() or not np.all(np.asarray(scales) > 0):
        raise ValueError("Channel labels or voltage scaling are not uniquely representable")
    binary_bytes = paths.binary.stat().st_size
    samples, remainder = divmod(binary_bytes, count * 4)
    if not samples or remainder:
        raise ValueError("Binary stat does not contain complete multiplexed float32 samples")
    if "DataPoints" in common and _integer(common["DataPoints"], "header DataPoints") not in (0, samples):
        raise ValueError("Header recording length disagrees with the immutable binary length")
    marker_file = _bounded_ini(paths.marker, marker_byte_limit)
    if (marker_file["Common Infos"].get("DataFile") != paths.binary.name
            or marker_file["Common Infos"].get("Codepage") != "UTF-8"):
        raise ValueError("Marker header does not refer to this same UTF-8 recording")
    records = []
    for marker_id, value in marker_file["Marker Infos"].items():
        if not re.fullmatch(r"Mk[1-9][0-9]*", marker_id):
            raise ValueError("Unexpected original marker identity")
        fields = value.split(",")
        if len(fields) < 5:
            raise ValueError("Original marker lacks required type/position/extent/channel fields")
        zero, points, channel = (_integer(fields[2], "marker sample") - 1,
                                 _integer(fields[3], "marker points"), _integer(fields[4], "marker channel"))
        if not 0 <= zero < samples or points < 1 or not 0 <= channel <= count:
            raise ValueError("Original marker is outside recording or channel support")
        records.append(RecordedMarker(marker_id, fields[0].replace(r"\1", ","),
            fields[1].replace(r"\1", ","), zero, points, channel))
    records.sort(key=lambda row: int(row.marker_id[2:]))
    if any(a.sample_zero_based > b.sample_zero_based for a, b in zip(records, records[1:])):
        raise ValueError("Original marker order does not follow recording sample order")
    starts = [row for row in records if row.marker_type == "New Segment"]
    if not starts or starts[0].sample_zero_based != 0 or any(
            a.sample_zero_based >= b.sample_zero_based for a, b in zip(starts, starts[1:])):
        raise ValueError("Recording segments must begin at zero and have distinct increasing starts")
    segments = tuple(RecordingSegment(row.marker_id, row.sample_zero_based,
        starts[index + 1].sample_zero_based if index + 1 < len(starts) else samples)
        for index, row in enumerate(starts))
    stimuli = tuple(StimulusMarker(row.marker_id, row.sample_zero_based + 1) for row in records
                    if row.marker_type == "Event" and re.fullmatch(r"E\s*1", row.description))
    budget_check()
    return BrainVisionMetadata(paths, tuple(names), tuple(resolutions), tuple(units), tuple(scales),
        sampling_hz, samples, binary_bytes, segments, stimuli, tuple(records))


def read_main_event_rows(paths, *, budget_check):
    """Compute-only structural fields; no response/RT/correctness columns kept."""
    _compute_only()
    source, frozen = [], []
    with paths.events.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not set(FIELDS).issubset(reader.fieldnames or ()):
            raise ValueError("Original events lack qualified structural/timing columns")
        for index, row in enumerate(reader):
            if index % 4096 == 0:
                budget_check()
            if _integer(row["isteststim"], "event class") == 0:
                source.append({name: row[name] for name in FIELDS})
    with gzip.open(paths.frozen_development_join, "rt", encoding="utf-8", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream, delimiter="\t")):
            if index % 4096 == 0:
                budget_check()
            if row["participant"] == paths.participant_id and row["is_repeated"] == "0":
                frozen.append(row)
    if len(source) != 22248 or len(frozen) != 22248:
        raise ValueError("All 22248 main events must survive source/development-join selection")
    return tuple(source), tuple(frozen)


def build_recorded_sequence_spans(metadata, main_source_events, *, recipe):
    """Join source-labelled streams to every E1 in the preceding E3 interval.

    The full source/frozen event identity join remains in the existing bridge
    before EEG reads (or the already joined partition-input builder). A count
    of309 does not select main membership. Cue gaps are measured coordinates;
    only padding/score offsets are analyst choices, with no timing tolerance.
    """
    if not isinstance(metadata, BrainVisionMetadata) or not isinstance(recipe, RecordedSpanRecipe):
        raise ValueError("Authentic recording metadata and an explicit recorded-span recipe required")
    if recipe.cue_anchor != "preceding_recorded_E3":
        raise ValueError("Select the actual preceding_recorded_E3 cue anchor explicitly")
    for value in (recipe.post_last_inclusive_samples, recipe.score_first_offset_samples,
                  recipe.score_last_inclusive_samples):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError("Span recipe choices must be integer source-sample offsets")
    if recipe.post_last_inclusive_samples < 0:
        raise ValueError("Post-last padding must be nonnegative")
    cues = tuple(row for row in metadata.markers if row.marker_type == "Event"
                 and re.fullmatch(r"E\s*3", row.description))
    positions = [row.sample_zero_based for row in cues]
    if (not cues or len({row.marker_id for row in cues}) != len(cues)
            or any(a >= b for a, b in zip(positions, positions[1:]))
            or any(not 0 <= value < metadata.recording_sample_count for value in positions)):
        raise ValueError("Physical sequence start markers must have unique ordered recorded positions")
    if any(not isinstance(row, StimulusMarker) for row in metadata.stimulus_markers):
        raise ValueError("Supply the original E1 marker positions")
    stimulus_positions = tuple(_integer(row.sample_one_based, "E1 sample") - 1
                               for row in metadata.stimulus_markers)
    if (any(a >= b for a, b in zip(stimulus_positions, stimulus_positions[1:]))
            or any(not 0 <= value < metadata.recording_sample_count for value in stimulus_positions)):
        raise ValueError("E1 markers must have unique ordered recorded positions")
    grouped = {index: [] for index in range(72)}
    for row in main_source_events:
        sequence = _integer(row["sequencenumber"], "source physical sequence")
        if _integer(row["isteststim"], "source event class") != 0 or sequence not in grouped:
            raise ValueError("Recorded primary spans cannot include repeated or unknown sequences")
        grouped[sequence].append(row)
    segment_starts = [segment.start_sample for segment in metadata.recording_segments]
    spans = []
    for sequence, rows in grouped.items():
        if len(rows) != 309 or [_integer(row["presentationnumber"], "presentation") for row in rows] != list(range(309)):
            raise ValueError("Each original physical stream requires its 309 ordered events")
        samples = tuple(_integer(row["sample"], "source sample") - 1 for row in rows)
        block = _integer(rows[0]["blocksequencenumber"], "source block")
        within = _integer(rows[0]["withinsequencenumber"], "within-block sequence")
        if (any(a >= b for a, b in zip(samples, samples[1:]))
                or any(_integer(row["blocksequencenumber"], "source block") != block
                       or _integer(row["withinsequencenumber"], "within-block sequence") != within
                       for row in rows)):
            raise ValueError("A source-labelled physical stream must retain its ordered samples and location")
        first, last = samples[0], samples[-1]
        cue_index = bisect_right(positions, first) - 1
        if cue_index < 0:
            raise ValueError("A source-labelled stream has no preceding recorded E3")
        next_cue = positions[cue_index + 1] if cue_index + 1 < len(cues) else metadata.recording_sample_count
        interval_e1 = stimulus_positions[bisect_left(stimulus_positions, positions[cue_index]):
                                        bisect_left(stimulus_positions, next_cue)]
        if samples != interval_e1:
            raise ValueError("Source-labelled sequence samples do not exactly match its recorded E3/E1 interval")
        start, stop = positions[cue_index], last + recipe.post_last_inclusive_samples + 1
        if stop > next_cue:
            raise ValueError("Selected padding crosses the next recorded physical sequence start")
        segment_index = bisect_right(segment_starts, start) - 1
        if segment_index < 0:
            raise ValueError("Sequence starts before the first recording segment")
        spans.append(SequenceSpan(f"{metadata.paths.participant_id}:main:{sequence}",
            block, within, sequence,
            metadata.recording_segments[segment_index].segment_id, start, stop,
            first + recipe.score_first_offset_samples, last + recipe.score_last_inclusive_samples + 1))
    return _spans(spans, metadata.recording_segments, metadata.recording_sample_count)


class SequenceVoltageReader:
    """Lazy exact-interval reader; constructor does not open the EEG binary."""
    def __init__(self, metadata, spans, *, interval_byte_limit, budget_check):
        if not isinstance(metadata, BrainVisionMetadata) or not callable(budget_check):
            raise ValueError("Bound recording metadata and a resource callback are required")
        self.metadata = metadata
        checked = _spans(spans, metadata.recording_segments, metadata.recording_sample_count)
        self.allowed = frozenset((span.start_sample, span.stop_sample) for span in checked)
        self.byte_limit = _positive_integer(interval_byte_limit, "interval byte limit")
        self.budget_check = budget_check
        self.scales = np.asarray(metadata.volts_per_stored_unit, dtype=np.float64)
        self.scales.setflags(write=False)

    def __call__(self, start, stop):
        _compute_only()
        if (isinstance(start, (bool, np.bool_)) or isinstance(stop, (bool, np.bool_))
                or not isinstance(start, Integral) or not isinstance(stop, Integral)
                or (start, stop) not in self.allowed):
            raise ValueError("Raw EEG reads are restricted to the prevalidated complete sequence intervals")
        channels = len(self.metadata.channel_order)
        values = (stop - start) * channels
        # Four-byte input plus eight-byte converted output; the bridge's own
        # copies/filter buffers and complete output roster are separate costs.
        if 12 * values > self.byte_limit:
            raise MemoryError("One source interval plus voltage output exceeds its admitted byte limit")
        self.budget_check()
        with self.metadata.paths.binary.open("rb") as stream:
            if os.fstat(stream.fileno()).st_size != self.metadata.binary_bytes:
                raise ValueError("Source recording size changed after metadata binding")
            stream.seek(start * channels * 4)
            raw = np.fromfile(stream, dtype="<f4", count=values)
        if raw.size != values:
            raise ValueError("Short EEG interval read; no padding or partial stream is allowed")
        volts = raw.reshape(stop - start, channels).astype(np.float64)
        volts *= self.scales[None, :]
        self.budget_check()
        return volts


def prepare_development_recording(*, paths, frozen_channel_order,
                                  span_recipe: RecordedSpanRecipe,
                                  preparation_recipe: SequencePreparationRecipe,
                                  header_byte_limit, marker_byte_limit, interval_byte_limit,
                                  budget_check: Callable[[], None]):
    """Connect authentic source metadata/voltage conversion to the reviewed bridge.

    The caller must prospectively authorize this participant and both scientific
    recipes within a bounded allocation. The existing development manifest is
    only a role binding. Bridge metadata/mask checks run before the lazy reader
    opens its first EEG interval. No fitting, tuning, feature read or export
    occurs in this integration function.
    """
    _compute_only()
    metadata = read_recording_metadata(paths, header_byte_limit=header_byte_limit,
        marker_byte_limit=marker_byte_limit, budget_check=budget_check)
    source, frozen = read_main_event_rows(paths, budget_check=budget_check)
    spans = build_recorded_sequence_spans(metadata, source, recipe=span_recipe)
    reader = SequenceVoltageReader(metadata, spans, interval_byte_limit=interval_byte_limit,
                                   budget_check=budget_check)
    return prepare_participant_sequences(participant_id=paths.participant_id,
        main_source_events=source, main_frozen_join=frozen, stimulus_markers=metadata.stimulus_markers,
        recording_segments=metadata.recording_segments, recording_sample_count=metadata.recording_sample_count,
        sequence_spans=spans, original_channel_order=metadata.channel_order,
        frozen_channel_order=frozen_channel_order, recipe=preparation_recipe,
        read_samples=reader, budget_check=budget_check)
