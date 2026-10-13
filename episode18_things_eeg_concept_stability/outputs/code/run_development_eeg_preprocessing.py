#!/usr/bin/env python3
"""Complete fixed EP18 development EEG preparation without models or scores.

The coordinator separately selects actual EEG access and finite resources.
This callable/CLI is not a release by itself. Its explicit five-field release
contains development_manifest, bids_root, frozen_channel_order, geometry_report
and coefficient_record. Only the already bound30 development people are read.
Stream their frozen join once; reuse the authentic lazy voltage reader and
reviewed sequence bridge, processing all72 sequences of one person at a time.

Retain the full250-Hz grid in float64 volts, all62 frozen channels, common mask,
original event text and sequence-relative acquisition clock. No delay correction,
high-pass, detrending, average reference, PCA, whitening or model fitting occurs.
The fold runner applies its single62-to-61 projection and all learned transforms.
Target/button nuisance inputs later join the separately prepared event arrays;
their absence from this structural-only cache does not imply zero flags.

Temporary participant artifacts live only in a fresh L_SCRATCH_JOB child.
Completed private participant caches survive a resource stop, but the complete
all30 manifest is published only after exact coverage. No automatic continuation,
participant exclusion, repair, outcome release or scientific acceptance.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import os
from pathlib import Path
import resource
import time

import numpy as np

from bind_development_sequence_geometry import source_main_rows
from brainvision_sequence_reader import (RecordedSpanRecipe, SequenceVoltageReader,
    bind_development_paths, build_recorded_sequence_spans, read_recording_metadata)
from continuous_eeg_cache import (METADATA_CAP, PARTICIPANT_CAP, publish_json,
    publish_prepared_participant, read_json, recipe_record, require)
from false_partition_search import BudgetStop, THREAD_ENV, budget
from feature_extractors import require_compute
from prepare_continuous_eeg import SequencePreparationRecipe, prepare_participant_sequences


OUTPUTS = Path(__file__).resolve().parents[1]
NOT_AFTER = 1791927000
MAX_DURABLE = 12 * 1024**3
RECORD_RESERVE = 8 * 1024**2
INTERVAL_CAP = 64 * 1024**2
COEFFICIENT_SHA = "a03aa85ced6d54471dcbfff36525cbca7085c1d5860c23f5ed4b127ca9abd251"
SPAN = RecordedSpanRecipe("preceding_recorded_E3", 800, -800, 800)


def load_selected_release(filename):
    release = read_json(filename, 65536)
    require(set(release) == {"development_manifest", "bids_root", "frozen_channel_order",
                             "geometry_report", "coefficient_record"})
    paths = {name: Path(value) for name, value in release.items()}
    require(all(path.is_absolute() and ".." not in path.parts for path in paths.values()))
    expected = {
        "development_manifest": OUTPUTS / "runs/input_role_preparation/20261003-freeze_metadata01/development_manifest.json",
        "bids_root": Path(os.environ["OAK"]) / "data/br_autoresearch_data/things_eeg1/openneuro-ds003825-v1.2.0/source",
        "frozen_channel_order": OUTPUTS / "runs/feature_preparation/recorded_channel_order_release001.json",
        "geometry_report": OUTPUTS / "runs/feature_preparation/development_main_geometry_20261012_001/geometry_binding.json",
        "coefficient_record": OUTPUTS / "runs/feature_preparation/sequence_geometry_fir_20261012_001/metadata_and_fir.json"}
    require(paths == expected)
    manifest = read_json(paths["development_manifest"], 8 * 1024**2)
    people = manifest["participants"]
    require(manifest["role"] == "development" and len(people) == len(set(people)) == 30
            and people == sorted(people) and manifest["event_count"] == 30 * 24648)
    released = read_json(paths["frozen_channel_order"], 65536)
    require(set(released) == {"common_recorded_channel_order"})
    channels = tuple(released["common_recorded_channel_order"])
    require(len(channels) == len(set(channels)) == 62 and all(isinstance(name, str) for name in channels))
    geometry = read_json(paths["geometry_report"], 4 * 1024**2)
    require(geometry["status"] == "complete_development_main_geometry_binding"
            and geometry["participant_count"] == 30 and geometry["physical_sequences"] == 2160
            and geometry["main_events"] == 667440 and geometry["source_role_event_rows"] == 739440
            and geometry["common_recorded_channel_order"] == list(channels)
            and geometry["coefficient_sha256"] == COEFFICIENT_SHA
            and geometry["span_recipe"] == asdict(SPAN)
            and geometry["source_sample_rate_hz"] == 1000 and geometry["target_sample_rate_hz"] == 250
            and geometry["filter_past_support_samples"] == 128
            and geometry["filter_future_support_samples"] == 0
            and geometry["recording_grid_decimation_phase_mod4"] == 0)
    by_person = {item["participant_id"]: item for item in geometry["participants"]}
    require(len(geometry["participants"]) == len(by_person) == 30 and set(by_person) == set(people))
    for item in by_person.values():
        require(item["all_source_frozen_E1_E3_segment_joins_exact"] is True
                and item["frozen_channel_order_preserved"] is True
                and item["full_negative_and_positive_window_has_complete_filter_support"] is True
                and len(item["raw_sample_counts_by_block_then_sequence"]) == 72
                and len(item["retained_sample_counts_by_block_then_sequence"]) == 72)
    coefficient_record = read_json(paths["coefficient_record"], 8 * 1024**2)["coefficients"]
    taps = np.asarray(coefficient_record["coefficients_float64"], dtype=np.float64)
    require(taps.shape == (129,) and np.isfinite(taps).all()
            and hashlib.sha256(taps.astype("<f8").tobytes()).hexdigest() == COEFFICIENT_SHA
            and coefficient_record["coefficients_little_endian_float64_sha256"] == COEFFICIENT_SHA)
    # Use the exact stored coefficients. The historical coefficient record's
    # prospective label predates the subsequently selected policy recipe.
    recipe = SequencePreparationRecipe(tuple(taps), 0, 0, "none",
        "detrend_then_fir_then_decimate", 1000, 250, "V")
    return release, paths, people, channels, by_person, recipe


def prepare_one(person, paths, frozen, channels, geometry, recipe, check, operational):
    metadata = read_recording_metadata(paths, header_byte_limit=8 * 1024**2,
        marker_byte_limit=128 * 1024**2, budget_check=check)
    source = source_main_rows(paths.events, check)
    spans = build_recorded_sequence_spans(metadata, source, recipe=SPAN)
    require(len(metadata.channel_order) == geometry["original_channel_count"]
            and [span.stop_sample - span.start_sample for span in spans]
                == geometry["raw_sample_counts_by_block_then_sequence"])
    reader = SequenceVoltageReader(metadata, spans, interval_byte_limit=INTERVAL_CAP, budget_check=check)

    def read_interval(start, stop):
        operational["EEG_read_attempted"] = True
        values = reader(start, stop)
        operational["completed_voltage_intervals"] += 1
        return values

    sequences = prepare_participant_sequences(participant_id=person,
        main_source_events=source, main_frozen_join=frozen,
        stimulus_markers=metadata.stimulus_markers, recording_segments=metadata.recording_segments,
        recording_sample_count=metadata.recording_sample_count, sequence_spans=spans,
        original_channel_order=metadata.channel_order, frozen_channel_order=channels,
        recipe=recipe, read_samples=read_interval, budget_check=check)
    require([int(item.retained_mask.sum()) for item in sequences]
            == geometry["retained_sample_counts_by_block_then_sequence"]
            and sum(int(item.retained_mask.sum()) for item in sequences) == geometry["retained_samples"])
    return sequences, metadata


def run_all(release_path, output, local_root, durable_cap, check, operational):
    release, released_paths, people, channels, geometry, recipe = load_selected_release(release_path)
    # Include every full grid row, even unscored filter/score-edge samples.
    grid_upper = sum((count + 3) // 4 for person in people
        for count in geometry[person]["raw_sample_counts_by_block_then_sequence"])
    expected_upper = grid_upper * (62 * 8 + 8 + 1) + 30 * (METADATA_CAP + 2 * 1024**2) + RECORD_RESERVE
    require(expected_upper <= durable_cap)
    publish_json(output / "settings.json", {"release_manifest": str(release_path), "release": release,
        "participant_order": people, "preparation_recipe": recipe_record(recipe), "span_recipe": asdict(SPAN),
        "source_role": "development_only", "durable_byte_cap": durable_cap,
        "admitted_full_grid_and_metadata_upper_bytes": expected_upper,
        "participant_byte_cap": PARTICIPANT_CAP, "participant_metadata_byte_cap": METADATA_CAP,
        "local_staging_byte_cap": PARTICIPANT_CAP, "interval_buffer_byte_cap": INTERVAL_CAP,
        "trained_transforms_or_projection": "none; retained62channels; existing fold runner owns them",
        "target_button_inputs": "later exact join to separately prepared event arrays; absence is not zero",
        "model_scores": False, "audit_access": False}, 256 * 1024)
    used = (output / "settings.json").stat().st_size
    paths = {person: bind_development_paths(development_manifest=released_paths["development_manifest"],
        bids_root=released_paths["bids_root"], participant_id=person) for person in people}
    joins = {path.frozen_development_join for path in paths.values()}
    require(len(joins) == 1)
    join = next(iter(joins))
    require(join.stat().st_size <= 512 * 1024**2)
    completed, total, seen = {}, 0, set()
    with gzip.open(join, "rt", encoding="utf-8", newline="") as stream:
        table = csv.DictReader(stream, delimiter="\t")
        require(table.fieldnames is not None and len(table.fieldnames) == len(set(table.fieldnames))
                and {"participant", "is_repeated"}.issubset(table.fieldnames))
        for person, rows in itertools.groupby(table, key=lambda row: row["participant"]):
            check()
            require(person in paths and person not in seen)
            main, count, repeated = [], 0, 0
            for row in rows:
                if count % 4096 == 0:
                    check()
                require(count < 24648 and total < 739440 and None not in row
                        and all(value is not None for value in row.values()) and row["is_repeated"] in ("0", "1"))
                count += 1
                total += 1
                if row["is_repeated"] == "1":
                    repeated += 1
                else:
                    main.append(row)
            require(count == 24648 and repeated == 2400 and len(main) == 22248)
            started = time.monotonic()
            sequences, metadata = prepare_one(person, paths[person], main, channels,
                                               geometry[person], recipe, check, operational)
            report = publish_prepared_participant(sequences=sequences, participant_id=person,
                recording_segments=metadata.recording_segments,
                recording_sample_count=metadata.recording_sample_count, recipe=recipe,
                destination=output / person, local_stage=local_root / person,
                remaining_durable_bytes=durable_cap - used - RECORD_RESERVE, check=check)
            used += report["artifact_bytes"]
            completed[person] = {"directory": str(output / person), "artifact_bytes": report["artifact_bytes"],
                "full_grid_samples": report["full_grid_samples"], "retained_samples": report["retained_samples"],
                "elapsed_seconds": time.monotonic() - started}
            operational.update(participants_completed=len(completed), completed_participants=sorted(completed),
                               durable_bytes=used)
            # Only this run's generated local copies are removed. Private
            # durable participants and all source/historical data are retained.
            for name in ("arrays.npz", "metadata.json"):
                (local_root / person / name).unlink()
            (local_root / person).rmdir()
            del sequences, metadata, main
            seen.add(person)
            check()
    require(seen == set(people) and total == 739440 and len(completed) == 30)
    report = {"status": "complete_development_EEG_preprocessing", "participants": completed,
        "participant_count": 30, "physical_sequences": 2160, "main_events": 667440,
        "full_grid_samples": sum(item["full_grid_samples"] for item in completed.values()),
        "retained_samples": sum(item["retained_samples"] for item in completed.values()),
        "channel_order": list(channels), "preparation_recipe": recipe_record(recipe), "span_recipe": asdict(SPAN),
        "sample_unit": "V", "source_sample_rate_hz": 1000, "target_sample_rate_hz": 250,
        "source_role": "development_only", "durable_bytes_before_manifest": used,
        "all_source_events_preserved": True, "all_full_grid_samples_saved": True,
        "projection_or_training_transform_applied": False, "model_scores_computed": False,
        "audit_access": False, "automatic_successor": False, "scientific_result": False}
    check()
    publish_json(output / "complete.json", report, RECORD_RESERVE // 2)
    operational["all_development_sequences_complete"] = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--rss-byte-cap", type=int, required=True)
    parser.add_argument("--durable-byte-cap", type=int, required=True)
    args = parser.parse_args(argv)
    owned, started = False, time.time()
    operational = {"status": "started", "participants_completed": 0, "completed_participants": [],
        "EEG_read_attempted": False, "completed_voltage_intervals": 0,
        "all_development_sequences_complete": False, "model_scores_computed": False, "audit_access": False}
    try:
        require_compute()
        require(all(os.environ.get(name) == "1" for name in THREAD_ENV)
                and time.time() < args.deadline_unix <= min(time.time() + 7200, NOT_AFTER)
                and 0 < args.rss_byte_cap <= 6 * 1024**3
                and RECORD_RESERVE < args.durable_byte_cap <= MAX_DURABLE)
        require(all(path.is_absolute() and ".." not in path.parts for path in
                    (args.release_manifest, args.output_dir, args.local_dir)))
        require(OUTPUTS.resolve() in args.output_dir.resolve().parents
                and Path(os.environ["L_SCRATCH_JOB"]).resolve() in args.local_dir.resolve().parents)
        os.umask(0o077)
        args.output_dir.mkdir(mode=0o700, exist_ok=False)
        owned = True
        args.local_dir.mkdir(mode=0o700, exist_ok=False)
        operational.update(job_id=os.environ["SLURM_JOB_ID"], deadline_unix=args.deadline_unix,
                           rss_byte_cap=args.rss_byte_cap, durable_byte_cap=args.durable_byte_cap)
        with budget(args.deadline_unix, args.rss_byte_cap) as check:
            run_all(args.release_manifest, args.output_dir, args.local_dir,
                    args.durable_byte_cap, check, operational)
        operational["status"] = "complete_development_EEG_preprocessing"
        return 0
    except BudgetStop:
        operational["status"] = "resource_incomplete"
        return 75
    except BaseException as error:
        operational.update(status="stopped_error", error_type=type(error).__name__)
        return 1
    finally:
        if owned:
            operational.update(elapsed_seconds=time.time() - started,
                process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                finished_utc=datetime.now(timezone.utc).isoformat())
            publish_json(args.output_dir / "operational.json", operational, RECORD_RESERVE // 2)
        print('{"status":"' + operational["status"] + '"}', flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
