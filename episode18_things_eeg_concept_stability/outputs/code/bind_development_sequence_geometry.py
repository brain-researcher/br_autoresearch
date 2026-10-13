#!/usr/bin/env python3
"""Bind all development main-sequence geometry before any EEG amplitude read.

Use the existing literal thirty-person development manifest, its development
event join, exact source event/header/marker paths and the coordinator's narrow
release of the original frozen62-channel order. Never open the mixed-role
freeze record, audit inputs, image assets or EEG binaries (binary stat only).
CSV text necessarily contains unused columns; only the existing structural
FIELDS are selected from source events. Response/RT/correctness values are
never selected, parsed, retained, used or reported.

The coordinator prospectively selected actual preceding E3 through last E1
+800ms inclusive, scoring first E1-800ms through last E1+800ms inclusive, with
the existing causal129-tap recipe (128 past,0 future, original-grid phase0).
Primary model lags remain0..800ms; the identical mask also supports mandatory
negative-lag controls. No event is shifted and no negative control is run here.

Reuse build_recorded_sequence_spans and the bridge's complete _events join.
Stream the existing participant-contiguous frozen development join once,
keeping only one person's main rows in memory. Unexpected grouping/order or
coverage stops the complete binding; no person, event or sequence is dropped.
The final report contains outcome-free identity/support counts, not event
rows, image names, coordinates, EEG or scientific comparison results.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import itertools
import os
from pathlib import Path
import resource
import time
import traceback

from brainvision_sequence_reader import (RecordedSpanRecipe, bind_development_paths,
    build_recorded_sequence_spans, read_recording_metadata)
from false_partition_search import BudgetStop, THREAD_ENV, atomic_json, budget
from feature_extractors import require_compute
from prepare_continuous_eeg import _events, _integer
from prepare_false_partition_inputs import read_json, require
from qualify_structural_inputs import FIELDS


OUTPUTS = Path(__file__).resolve().parents[1]
MANIFEST = OUTPUTS / "runs/input_role_preparation/20261003-freeze_metadata01/development_manifest.json"
CHANNEL_RELEASE = OUTPUTS / "runs/feature_preparation/recorded_channel_order_release001.json"
SPAN = RecordedSpanRecipe(cue_anchor="preceding_recorded_E3", post_last_inclusive_samples=800,
    score_first_offset_samples=-800, score_last_inclusive_samples=800)
LEFT, RIGHT, PHASE = 128, 0, 0
COEFFICIENT_SHA256 = "a03aa85ced6d54471dcbfff36525cbca7085c1d5860c23f5ed4b127ca9abd251"
BYTE_CAP = 8 * 1024**2


def source_main_rows(path, check):
    """Select the reader's existing structural fields without behavioral columns."""
    require(path.stat().st_size <= 32 * 1024**2)
    main, total, repeated = [], 0, 0
    with path.open(encoding="utf-8-sig", newline="") as source:
        table = csv.reader(source, delimiter="\t")
        names = next(table)
        require(len(names) == len(set(names)) and set(FIELDS).issubset(names))
        indices = tuple(names.index(name) for name in FIELDS)
        for values in table:
            if total % 4096 == 0:
                check()
            require(total < 24648 and len(values) == len(names))
            total += 1
            row = {name: values[index] for name, index in zip(FIELDS, indices)}
            kind = _integer(row["isteststim"], "source event class")
            require(kind in (0, 1))
            if kind:
                repeated += 1
            else:
                main.append(row)
    require(total == 24648 and repeated == 2400 and len(main) == 22248)
    return tuple(main)


def retained_count(span):
    """The bridge's same absolute phase0 grid and full FIR-support intersection."""
    require(span.score_start_sample >= span.start_sample + LEFT
            and span.score_stop_sample <= span.stop_sample - RIGHT)
    first = span.score_start_sample + (PHASE - span.score_start_sample) % 4
    stop = span.score_stop_sample
    require(first < stop)
    return (stop - 1 - first) // 4 + 1


def bind_person(person, paths, frozen, channels, check):
    metadata = read_recording_metadata(paths, header_byte_limit=8 * 1024**2,
        marker_byte_limit=128 * 1024**2, budget_check=check)
    # Exactly the same original-column selection used by the EEG bridge.
    original = metadata.channel_order
    require(len(set(original)) == len(original) and set(channels).issubset(original))
    selected_columns = tuple(original.index(name) for name in channels)
    require(tuple(original[index] for index in selected_columns) == channels)
    source = source_main_rows(paths.events, check)
    spans = build_recorded_sequence_spans(metadata, source, recipe=SPAN)
    joined = _events(person, source, frozen, metadata.stimulus_markers,
                     spans, metadata.recording_sample_count)
    check()
    retained = [retained_count(span) for span in spans]
    gaps = Counter(joined[span.physical_sequence][0].sample_zero_based - span.start_sample
                   for span in spans)
    require(len(spans) == len(joined) == 72 and sum(map(len, joined.values())) == 22248)
    return dict(participant_id=person, physical_sequences=72, main_events=22248,
        source_event_rows=24648, source_repeated_events_excluded_from_main_binding=2400,
        original_channel_count=len(original), frozen_channel_count=62,
        frozen_channel_order_preserved=True, recording_segments=len(metadata.recording_segments),
        all_source_frozen_E1_E3_segment_joins_exact=True,
        raw_sample_counts_by_block_then_sequence=[span.stop_sample - span.start_sample for span in spans],
        retained_sample_counts_by_block_then_sequence=retained,
        retained_samples=sum(retained), actual_cue_gap_sample_counts=dict(sorted(gaps.items())),
        full_negative_and_positive_window_has_complete_filter_support=True,
        EEG_amplitudes_read=False, behavioral_values_selected_or_used=False)


def bind_all(manifest_path, bids_root, channel_release, check, operational):
    require(manifest_path == MANIFEST and channel_release == CHANNEL_RELEASE
            and bids_root == Path(os.environ["OAK"]) /
                "data/br_autoresearch_data/things_eeg1/openneuro-ds003825-v1.2.0/source")
    manifest, released = read_json(manifest_path), read_json(channel_release)
    people = manifest["participants"]
    require(manifest["role"] == "development" and len(people) == len(set(people)) == 30
            and people == sorted(people) and manifest["event_count"] == 30 * 24648)
    require(set(released) == {"common_recorded_channel_order"})
    channels = tuple(released["common_recorded_channel_order"])
    require(len(channels) == len(set(channels)) == 62
            and all(isinstance(name, str) and name for name in channels))
    paths = {person: bind_development_paths(development_manifest=manifest_path,
             bids_root=bids_root, participant_id=person) for person in people}
    joins = {value.frozen_development_join for value in paths.values()}
    require(len(joins) == 1)
    join_path = next(iter(joins))
    require(join_path.name == "development_event_image_join.tsv.gz"
            and join_path.stat().st_size <= 512 * 1024**2)
    results, seen, total = [], set(), 0
    with gzip.open(join_path, "rt", encoding="utf-8", newline="") as stream:
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
                require(count < 24648 and total < manifest["event_count"]
                        and None not in row and all(value is not None for value in row.values())
                        and row["is_repeated"] in ("0", "1"))
                count += 1
                total += 1
                if row["is_repeated"] == "1":
                    repeated += 1
                else:
                    main.append(row)
            require(count == 24648 and repeated == 2400 and len(main) == 22248)
            results.append(bind_person(person, paths[person], main, channels, check))
            del main
            seen.add(person)
            operational["participants_completed"] = len(results)
    require(seen == set(people) and total == manifest["event_count"] and len(results) == 30)
    check()
    return dict(status="complete_development_main_geometry_binding", participants=sorted(results,
        key=lambda result: result["participant_id"]), participant_count=30, physical_sequences=30 * 72,
        main_events=30 * 22248, source_role_event_rows=total,
        common_recorded_channel_order=list(channels), span_recipe=vars(SPAN),
        source_sample_rate_hz=1000, target_sample_rate_hz=250,
        filter_past_support_samples=LEFT, filter_future_support_samples=RIGHT,
        recording_grid_decimation_phase_mod4=PHASE, coefficient_sha256=COEFFICIENT_SHA256,
        primary_event_design_lags_ms=[0, 800], mandatory_negative_control_lags_ms=[-800, 0],
        event_and_sample_clock_shifted=False, EEG_amplitudes_read=False,
        behavioral_values_selected_or_used=False, image_values_read=False, audit_inputs_read=False,
        EEG_execution_or_comparison_released=False, empirical_scientific_result=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--frozen-channel-order", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--rss-byte-cap", type=int, required=True)
    args = parser.parse_args(argv)
    owned, started = False, time.time()
    operational = dict(status="started", participants_completed=0, geometry_binding_complete=False,
                       EEG_amplitudes_read=False)
    try:
        require_compute()
        require(all(os.environ.get(name) == "1" for name in THREAD_ENV))
        require(0 < args.deadline - time.time() <= 300 and 0 < args.rss_byte_cap <= 6 * 1024**3)
        require(all(path.is_absolute() and ".." not in path.parts for path in
                    (args.development_manifest, args.bids_root, args.frozen_channel_order, args.output)))
        os.umask(0o077)
        args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
        owned = True
        atomic_json(args.output / "settings.json", dict(development_manifest=str(args.development_manifest),
            bids_root=str(args.bids_root), frozen_channel_order=str(args.frozen_channel_order),
            deadline=args.deadline, rss_byte_cap=args.rss_byte_cap, durable_byte_cap=BYTE_CAP,
            span_recipe=vars(SPAN), filter_past_support_samples=LEFT, filter_future_support_samples=RIGHT,
            recording_grid_decimation_phase_mod4=PHASE, coefficient_sha256=COEFFICIENT_SHA256,
            source_event_path_layout="<literal-development-id>/eeg/<literal-development-id>_task-rsvp_events.tsv",
            EEG_access="binary_stat_only_no_open", behavioral_fields_selected=False), cap=64 * 1024)
        with budget(args.deadline, args.rss_byte_cap) as check:
            report = bind_all(args.development_manifest, args.bids_root, args.frozen_channel_order,
                              check, operational)
            check()
            atomic_json(args.output / "geometry_binding.json", report, cap=4 * 1024**2)
        operational.update(status="complete_development_main_geometry_binding", geometry_binding_complete=True)
        return 0
    except BudgetStop as error:
        operational.update(status="resource_incomplete", reason=str(error))
        return 75
    except BaseException as error:
        operational.update(status="stopped_error", error_type=type(error).__name__)
        # Preserve a diagnosable source location without messages, locals or
        # source/event values; an opaque failure must not force a repeat read.
        operational["failure_frames"] = [dict(file=Path(frame.f_code.co_filename).name,
            function=frame.f_code.co_name, line=line)
            for frame, line in traceback.walk_tb(error.__traceback__)][-8:]
        return 1
    finally:
        if owned:
            operational.update(elapsed_seconds=time.time() - started,
                process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                finished_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(args.output / "operational.json", operational, cap=64 * 1024)


if __name__ == "__main__":
    raise SystemExit(main())
