"""Private, lossless participant caches for already prepared EP18 sequences.

No filtering, model fit or recipe selection occurs here. Preserve the full
250-Hz arrays, one common mask, exact event text and original sequence clock.
The numerical arrays are uncompressed float64/bool/int64 NPZ members, never
pickles. A participant is published only by its final complete.json; a partial
directory cannot be loaded. A complete participant can remain useful when the
separate all-thirty production attempt stops. The fold runner retains ownership
of its single62-to-61 projection and every training-learned transform.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
import zipfile

import numpy as np

from continuous_fold_runner import PhysicalSequence
from prepare_continuous_eeg import (RecordingSegment, SequenceSpan, StimulusMarker,
    SequenceEventInputs, SequencePreparationRecipe, _events, _integer, _recipe, _spans)


SCHEMA = "ep18_continuous_sequence_cache_v1"
METADATA_CAP = 64 * 1024**2
PARTICIPANT_CAP = 512 * 1024**2


def require(value):
    if not value:
        raise ValueError("Prepared sequence cache contract differs")


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def read_json(path, cap):
    with Path(path).open("rb") as stream:
        data = stream.read(cap + 1)
    require(len(data) <= cap)
    return json.loads(data)


def publish_json(path, value, cap):
    """Exclusive final name for a fully written, fsynced record."""
    path = Path(path)
    data = encode(value)
    require(len(data) <= cap)
    pending = path.with_name(path.name + ".pending")
    with pending.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(pending, path)
    pending.unlink()


def recipe_record(recipe):
    taps, origin, phase = _recipe(recipe)
    record = asdict(recipe)
    record["fir_coefficients"] = taps.tolist()
    record["coefficient_sha256"] = hashlib.sha256(taps.astype("<f8").tobytes()).hexdigest()
    return record


def _readonly(value):
    value = np.array(value, copy=True)
    value.setflags(write=False)
    return value


def _metadata_and_arrays(sequences, participant_id, segments, recording_samples, recipe):
    require(len(sequences) == 72 and all(isinstance(item, PhysicalSequence) for item in sequences))
    records, arrays = [], {}
    for index, item in enumerate(sequences):
        event_input = item.preparation_input
        require(isinstance(event_input, SequenceEventInputs) and event_input.participant_id == participant_id)
        prefix = f"sequence_{index:02d}"
        arrays.update({prefix + "_volts": item.sensor_samples,
            prefix + "_times": item.sample_times_seconds, prefix + "_mask": item.retained_mask,
            prefix + "_onsets": event_input.onsets_seconds,
            prefix + "_samples": event_input.source_sample_indices})
        records.append({"sequence_id": item.sequence_id, "block": item.block,
            "within_block": item.within_block, "span": asdict(event_input.span),
            "filter_left_support_samples": event_input.filter_left_support_samples,
            "filter_right_support_samples": event_input.filter_right_support_samples,
            "events": [{"source_row": dict(event.source_row), "frozen_row": dict(event.frozen_row),
                "marker_id": event.marker_id, "recording_segment_id": event.recording_segment_id,
                "sample_zero_based": event.sample_zero_based,
                "onset_sequence_seconds": event.onset_sequence_seconds} for event in event_input.events]})
        require(tuple(item.channel_order) == tuple(sequences[0].channel_order))
    return {"schema": SCHEMA, "participant_id": participant_id,
        "channel_order": list(sequences[0].channel_order), "input_sample_unit": "V",
        "recording_sample_count": recording_samples,
        "recording_segments": [asdict(segment) for segment in segments],
        "preparation_recipe": recipe_record(recipe), "sequences": records}, arrays


def _restore(metadata, arrays, *, participant_id, channel_order, recipe, check):
    """Reuse the bridge's full event/span join and restore immutable inputs."""
    require(metadata["schema"] == SCHEMA and metadata["participant_id"] == participant_id
            and metadata["input_sample_unit"] == "V"
            and metadata["preparation_recipe"] == recipe_record(recipe)
            and metadata["channel_order"] == list(channel_order)
            and len(channel_order) == len(set(channel_order)) == 62)
    records = metadata["sequences"]
    require(len(records) == 72)
    segments = tuple(RecordingSegment(**item) for item in metadata["recording_segments"])
    original_spans = tuple(SequenceSpan(**item["span"]) for item in records)
    spans = _spans(original_spans, segments, metadata["recording_sample_count"])
    require(original_spans == spans)
    sources, frozen, markers = [], [], []
    for record in records:
        require(len(record["events"]) == 309)
        for event in record["events"]:
            sources.append(event["source_row"])
            frozen.append(event["frozen_row"])
            markers.append(StimulusMarker(event["marker_id"], _integer(event["source_row"]["sample"], "source sample")))
    joined = _events(participant_id, sources, frozen, markers, spans, metadata["recording_sample_count"])
    taps, origin, phase = _recipe(recipe)
    left, right = len(taps) - 1 - origin, origin
    expected_keys = {f"sequence_{index:02d}_{suffix}" for index in range(72)
                     for suffix in ("volts", "times", "mask", "onsets", "samples")}
    require(set(arrays) == expected_keys)
    result = []
    for index, (record, span) in enumerate(zip(records, spans)):
        check()
        require(record["sequence_id"] == span.sequence_id and record["block"] == span.block
                and record["within_block"] == span.within_block
                and record["filter_left_support_samples"] == left
                and record["filter_right_support_samples"] == right)
        prefix = f"sequence_{index:02d}"
        volts, times, mask, onsets, samples = (np.asarray(arrays[prefix + "_" + suffix])
            for suffix in ("volts", "times", "mask", "onsets", "samples"))
        count = span.stop_sample - span.start_sample
        local = np.arange((phase - span.start_sample) % 4, count, 4, dtype=np.int64)
        absolute = local + span.start_sample
        expected_mask = ((local >= left) & (local < count - right)
            & (absolute >= span.score_start_sample) & (absolute < span.score_stop_sample))
        require(volts.dtype == np.float64 and volts.shape == (len(local), 62)
                and np.isfinite(volts).all() and times.dtype == np.float64
                and np.array_equal(times, local / 1000.)
                and mask.dtype == np.bool_ and np.array_equal(mask, expected_mask) and mask.any()
                and onsets.dtype == np.float64 and samples.dtype == np.int64)
        events = tuple(joined[span.physical_sequence])
        require(np.array_equal(onsets, [event.onset_sequence_seconds for event in events])
                and np.array_equal(samples, [event.sample_zero_based for event in events]))
        for saved, event in zip(record["events"], events):
            require(saved["marker_id"] == event.marker_id
                    and saved["recording_segment_id"] == event.recording_segment_id
                    and saved["sample_zero_based"] == event.sample_zero_based
                    and saved["onset_sequence_seconds"] == event.onset_sequence_seconds)
        predictors = SequenceEventInputs(participant_id, span, events, _readonly(onsets),
                                         _readonly(samples), left, right)
        result.append(PhysicalSequence(span.sequence_id, span.block, span.within_block,
            _readonly(times), _readonly(volts), tuple(channel_order), _readonly(mask), predictors))
    return tuple(result)


def publish_prepared_participant(*, sequences, participant_id, recording_segments,
                                recording_sample_count, recipe, destination, local_stage,
                                remaining_durable_bytes, check):
    """Stage locally and publish one complete participant without overwriting.

    Caller owns the fresh local/durable roots. This function creates only the
    supplied new participant directories and leaves any incomplete files for
    diagnosis. Caller removes its own local stage after recording completion.
    """
    check()
    metadata, arrays = _metadata_and_arrays(tuple(sequences), participant_id,
        recording_segments, recording_sample_count, recipe)
    # Internal cache validation is the same routine used when loading. It
    # checks lossless event/grid/mask semantics, not a new scientific gate.
    restored = _restore(metadata, arrays, participant_id=participant_id,
        channel_order=sequences[0].channel_order, recipe=recipe, check=check)
    del restored
    metadata_bytes = encode(metadata)
    require(len(metadata_bytes) <= METADATA_CAP)
    estimated = sum(value.nbytes + 2048 for value in arrays.values()) + len(metadata_bytes) + 65536
    require(estimated <= PARTICIPANT_CAP and estimated <= remaining_durable_bytes)
    destination, local_stage = Path(destination), Path(local_stage)
    local_stage.mkdir(mode=0o700, exist_ok=False)
    with (local_stage / "arrays.npz").open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    with (local_stage / "metadata.json").open("xb") as stream:
        stream.write(metadata_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    sizes = {name: (local_stage / name).stat().st_size for name in ("arrays.npz", "metadata.json")}
    require(sum(sizes.values()) + 65536 <= min(PARTICIPANT_CAP, remaining_durable_bytes))
    check()
    destination.mkdir(mode=0o700, exist_ok=False)
    for name, size in sizes.items():
        with (local_stage / name).open("rb") as source, (destination / name).open("xb") as target:
            copied = 0
            while True:
                check()
                chunk = source.read(1024**2)
                if not chunk:
                    break
                target.write(chunk)
                copied += len(chunk)
            target.flush()
            os.fsync(target.fileno())
        require(copied == size)
    report = {"schema": SCHEMA, "status": "complete_prepared_participant", "participant_id": participant_id,
        "sequences": 72, "main_events": 22248, "channel_order": list(sequences[0].channel_order),
        "full_grid_samples": sum(len(item.sample_times_seconds) for item in sequences),
        "retained_samples": sum(int(item.retained_mask.sum()) for item in sequences),
        "files": sizes, "preparation_recipe": recipe_record(recipe), "sample_unit": "V",
        "projection_or_training_transform_applied": False, "model_scores_computed": False}
    check()
    publish_json(destination / "complete.json", report, 65536)
    report["artifact_bytes"] = sum(sizes.values()) + (destination / "complete.json").stat().st_size
    return report


def load_prepared_participant(directory, *, participant_id, channel_order, recipe, budget_check):
    """Load one complete cache; return existing fold-runner sequence objects."""
    directory = Path(directory)
    report = read_json(directory / "complete.json", 65536)
    require(report["status"] == "complete_prepared_participant" and report["schema"] == SCHEMA
            and report["participant_id"] == participant_id and report["sequences"] == 72
            and report["main_events"] == 22248 and report["sample_unit"] == "V"
            and report["projection_or_training_transform_applied"] is False
            and report["model_scores_computed"] is False
            and report["preparation_recipe"] == recipe_record(recipe)
            and report["channel_order"] == list(channel_order)
            and set(report["files"]) == {"arrays.npz", "metadata.json"})
    require(not (directory / "complete.json.pending").exists())
    require(all((directory / name).stat().st_size == size for name, size in report["files"].items())
            and sum(report["files"].values()) + 65536 <= PARTICIPANT_CAP)
    metadata = read_json(directory / "metadata.json", METADATA_CAP)
    with zipfile.ZipFile(directory / "arrays.npz") as archive:
        entries = archive.infolist()
        require(len(entries) == 360 and len({item.filename for item in entries}) == 360
                and all(item.compress_type == zipfile.ZIP_STORED for item in entries)
                and sum(item.file_size for item in entries) <= PARTICIPANT_CAP)
    with np.load(directory / "arrays.npz", allow_pickle=False) as source:
        arrays = {}
        for name in source.files:
            budget_check()
            arrays[name] = source[name]
    sequences = _restore(metadata, arrays, participant_id=participant_id,
        channel_order=channel_order, recipe=recipe, check=budget_check)
    require(report["full_grid_samples"] == sum(len(item.sample_times_seconds) for item in sequences)
            and report["retained_samples"] == sum(int(item.retained_mask.sum()) for item in sequences))
    return sequences
