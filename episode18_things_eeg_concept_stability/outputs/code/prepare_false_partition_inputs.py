#!/usr/bin/env python3
"""Bind the full EP18 image bank to development-only partition-search metadata.

This is the trusted metadata builder, never an image-feature worker. It reads
the existing private UUID crosswalk and exact frozen development event join;
canonical names stay in memory and are not exported. Every original UUID and
all six complete image families are retained, including the 200 repeated image
rows in the feature-bank binding. Only the 22,248 main images enter the search
index. No EEG binary, pixel archive, caption detail, audit view, RT or correctness
value is opened as an input or retained/used. Original TSV lines necessarily
contain unused columns; only the explicitly listed structural fields and the
two target/button flags are selected from those lines.

The four-field release manifest contains absolute paths named feature_bank,
private_routing, development_manifest and event_paths. The latter maps exactly
the 30 development IDs to their original task-rsvp_events.tsv paths; paths are
supplied by the coordinator, never discovered. This manifest is a procedural
release view, not an access-control seal. The existing development manifest
binds the event join. Its original sample/onset normalization is reused.

For each participant, rows in the output NPZ follow sorted original main UUIDs.
The arrays retain original one-based samples, their exact frozen zero-based
counterparts, concept/block/sequence/presentation identities, and source binary
target/button flags. Response timing is NOT inferred here: the eventual frozen
recipe may explicitly place a coarse button event at its flagged stimulus onset.
The concept_by_block_image_row array gives the existing true grouping and a
complete image lookup for legal block-preserving false-group swaps.

This prepares executable inputs only. It generates no false partition, selects
no metric/filter/seed, grants no EEG access and reports no bank qualification.
Run on compute with caller-selected absolute deadline, RSS and durable caps;
the coordinator supplies hard native limits. An existing attempt is untouched.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import itertools
import json
import math
import os
from pathlib import Path
import resource
import socket
import time

from feature_extractors import DIMENSIONS, require_compute, validate_ids
from image_pixel_worker import bounded_process
from qualify_structural_inputs import image_key, integer


IMAGE_COUNT, MAIN_COUNT, CONCEPT_COUNT = 22448, 22248, 1854
MANIFEST_CAP, EVENT_CAP, JOIN_CAP = 8 * 1024**2, 32 * 1024**2, 512 * 1024**2
REPORT_RESERVE = 192 * 1024  # settings, completed report, and late failure marker
OUTPUTS = Path(__file__).resolve().parents[1]
SOURCE_FIELDS = ("isteststim", "sample", "onset", "time_stimon", "duration",
                 "stim", "objectnumber", "stimnumber", "blocksequencenumber",
                 "sequencenumber", "withinsequencenumber", "presentationnumber",
                 "istarget", "response")
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


class InputRefusal(ValueError):
    """Fixed categorical diagnostic only; never include private source values."""


def require(condition):
    if not condition:
        raise InputRefusal()


def path(value):
    result = Path(value)
    require(result.is_absolute() and ".." not in result.parts)
    return result


def read_json(filename):
    with path(filename).open("rb") as source:
        payload = source.read(MANIFEST_CAP + 1)
    require(len(payload) <= MANIFEST_CAP)
    return json.loads(payload)


def write_new(filename, value, cap):
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    require(len(payload) <= cap)
    with filename.open("xb") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
    return len(payload)


def feature_binding(bank):
    """Reuse completed assembly coverage; do not repeat feature inference/scans."""
    import numpy as np
    bank = path(bank)
    report, ids = read_json(bank / "report.json"), read_json(bank / "opaque_ids.json")
    require(report["status"] == "all_image_features_completed"
            and report["images"] == IMAGE_COUNT
            and report["all_images_each_family_exactly_once"] is True
            and report["EEG_read"] is False and set(report["families"]) == set(DIMENSIONS)
            and not (bank / "failure.json").exists())
    validate_ids(ids, IMAGE_COUNT)
    require(ids == sorted(ids))
    bound = {}
    for family, width in DIMENSIONS.items():
        filename = bank / (family + ".npy")
        prior = report["families"][family]
        require(prior["images"] == IMAGE_COUNT and prior["dimension"] == width
                and filename.stat().st_size == prior["feature_bytes"])
        values = np.load(filename, mmap_mode="r", allow_pickle=False)
        require(values.shape == (IMAGE_COUNT, width) and values.dtype == np.float32)
        del values
        bound[family] = {"path": str(filename), "shape": [IMAGE_COUNT, width],
                         "dtype": "float32", "feature_bytes": prior["feature_bytes"]}
    return ids, bound


def load_release(filename):
    release = read_json(filename)
    require(set(release) == {"feature_bank", "private_routing", "development_manifest", "event_paths"})
    manifest_path = path(release["development_manifest"])
    manifest = read_json(manifest_path)
    people = manifest["participants"]
    require(manifest["role"] == "development" and len(people) == len(set(people)) == 30
            and people == sorted(people) and manifest["event_count"] == 30 * 24648
            and manifest["frozen_event_join"] == "development_event_image_join.tsv.gz"
            and isinstance(release["event_paths"], dict) and set(release["event_paths"]) == set(people))
    paths = {}
    for person in people:
        require(isinstance(person, str) and len(person) == 6 and person.startswith("sub-")
                and person[4:].isdigit())
        prefix = f"{person}/eeg/{person}_task-rsvp_eeg"
        require(prefix + ".vhdr" in manifest["EEG_relative_paths_for_controlled_mount"])
        filename = path(release["event_paths"][person])
        require(filename.name == f"{person}_task-rsvp_events.tsv"
                and filename.parent.name == "eeg" and filename.parent.parent.name == person)
        paths[person] = filename
    return release, people, paths, manifest_path.parent / manifest["frozen_event_join"]


def load_routing(filename, feature_ids):
    require(path(filename).stat().st_size <= MANIFEST_CAP)
    routing, main, repeated = {}, set(), set()
    with path(filename).open(encoding="utf-8", newline="") as source:
        table = csv.DictReader(source, delimiter="\t")
        require(table.fieldnames == ["opaque_image_id", "canonical_image", "main", "repeated"])
        for row in table:
            identity, opaque = row["canonical_image"], row["opaque_image_id"]
            require(identity not in routing and row["main"] in {"True", "False"}
                    and row["repeated"] in {"True", "False"}
                    and (row["main"] == "True") != (row["repeated"] == "True"))
            routing[identity] = opaque
            (main if row["main"] == "True" else repeated).add(opaque)
    require(len(routing) == len(set(routing.values())) == IMAGE_COUNT
            and set(routing.values()) == set(feature_ids)
            and len(main) == MAIN_COUNT and len(repeated) == 200 and not main & repeated)
    return routing, sorted(main)


def source_main_rows(filename, check):
    """Retain structural columns plus binary flags, never response accuracy/RT."""
    require(filename.stat().st_size <= EVENT_CAP)
    rows, count, repeated = {}, 0, 0
    with filename.open(encoding="utf-8-sig", newline="") as source:
        table = csv.reader(source, delimiter="\t")
        fields = next(table)
        require(len(fields) == len(set(fields)) and set(SOURCE_FIELDS).issubset(fields))
        selected = [fields.index(name) for name in SOURCE_FIELDS]
        for values in table:
            if count % 4096 == 0:
                check()
            require(len(values) == len(fields) and count < 24648)
            count += 1
            row = dict(zip(SOURCE_FIELDS, (values[index] for index in selected)))
            kind = integer(row["isteststim"])
            require(kind in (0, 1))
            if kind:
                repeated += 1
                continue
            sample = integer(row["sample"])
            require(sample >= 1 and sample not in rows and float(row["duration"]) == 50.
                    and round(float(row["onset"]) * 1000) == sample
                    and integer(row["istarget"]) in (0, 1) and integer(row["response"]) in (0, 1))
            rows[sample] = row
    require(count == 24648 and repeated == 2400 and len(rows) == MAIN_COUNT)
    return rows


def participant_arrays(person, frozen, original, routing, main_ids, check):
    import numpy as np
    index = {opaque: row for row, opaque in enumerate(main_ids)}
    names = ("concept", "exemplar", "block", "physical_sequence", "within_block_sequence",
             "presentation", "sample_one_based", "sample_zero_based", "target_flag", "response_flag")
    arrays = {name: np.full(MAIN_COUNT, -1, dtype=np.int64) for name in names}
    by_concept = np.full((CONCEPT_COUNT, 12), -1, dtype=np.int32)
    seen, total, repeat = set(), 0, 0
    for row in frozen:
        if total % 4096 == 0:
            check()
        require(total < 24648)
        total += 1
        require(row["participant"] == person and row["is_repeated"] in {"0", "1"})
        if row["is_repeated"] == "1":
            repeat += 1
            continue
        sample, zero = integer(row["sample"]), integer(row["sample_zero_based"])
        require(sample in original and zero == sample - 1 and zero >= 0
                and float(row["onset_recording_seconds"]) == zero / 1000.)
        source = original[sample]
        require(source["onset"] == row["source_onset_seconds"]
                and source["time_stimon"] == row["source_time_stimon_seconds"]
                and image_key(source["stim"]) == row["canonical_image"]
                and row["canonical_image"] in routing)
        opaque = routing[row["canonical_image"]]
        require(opaque in index and opaque not in seen)
        seen.add(opaque)
        values = [integer(source[name]) for name in ("objectnumber", "stimnumber", "blocksequencenumber",
                  "sequencenumber", "withinsequencenumber", "presentationnumber")]
        concept, exemplar, block, sequence, within, presentation = values
        require(0 <= concept < CONCEPT_COUNT and 0 <= exemplar < 12 and 0 <= block < 12
                and 0 <= sequence < 72 and 0 <= within < 6 and 0 <= presentation < 309
                and concept == integer(row["concept_number"]) and exemplar == integer(row["exemplar_number"])
                and block == integer(row["block"]) and sequence == integer(row["physical_sequence"])
                and presentation == integer(row["presentation"]) and by_concept[concept, block] == -1)
        position = index[opaque]
        by_concept[concept, block] = position
        values += [sample, zero, integer(source["istarget"]), integer(source["response"])]
        for name, value in zip(names, values):
            arrays[name][position] = value
    require(total == 24648 and repeat == 2400 and len(seen) == MAIN_COUNT
            and np.all(by_concept >= 0) and len(set(arrays["sample_one_based"].tolist())) == MAIN_COUNT)
    positions = set()
    for sequence in range(72):
        selected = np.flatnonzero(arrays["physical_sequence"] == sequence)
        require(len(selected) == 309)
        selected = selected[np.argsort(arrays["presentation"][selected])]
        require(np.array_equal(arrays["presentation"][selected], np.arange(309))
                and np.all(np.diff(arrays["sample_zero_based"][selected]) > 0)
                and len(set(arrays["block"][selected].tolist())) == 1
                and len(set(arrays["within_block_sequence"][selected].tolist())) == 1)
        positions.add((int(arrays["block"][selected[0]]), int(arrays["within_block_sequence"][selected[0]])))
    require(positions == set(itertools.product(range(12), range(6))))
    for concept in range(CONCEPT_COUNT):
        require(set(arrays["exemplar"][by_concept[concept]].tolist()) == set(range(12)))
    arrays["concept_by_block_image_row"] = by_concept
    return arrays


def prepare(release_manifest, output, check, durable_cap, report):
    require_compute()
    import numpy as np
    release, people, event_paths, join_path = load_release(release_manifest)
    ids, families = feature_binding(release["feature_bank"])
    routing, main_ids = load_routing(release["private_routing"], ids)
    feature_index = {opaque: index for index, opaque in enumerate(ids)}
    used = (output / "settings.json").stat().st_size
    used += write_new(output / "image_index.json", {"all_feature_opaque_ids": ids,
        "main_opaque_ids": main_ids, "main_to_feature_row": [feature_index[opaque] for opaque in main_ids],
        "families": families}, MANIFEST_CAP)
    require(used + REPORT_RESERVE <= durable_cap and join_path.stat().st_size <= JOIN_CAP)
    participant_files, consumed = {}, []
    # Stream the already role-filtered join once; never load a mixed/audit index.
    with gzip.open(join_path, "rt", encoding="utf-8", newline="") as source:
        table = csv.DictReader(source, delimiter="\t")
        for person, frozen in itertools.groupby(table, key=lambda row: row["participant"]):
            require(len(consumed) < len(people) and person == people[len(consumed)])
            original = source_main_rows(event_paths[person], check)
            arrays = participant_arrays(person, frozen, original, routing, main_ids, check)
            del original
            expected = sum(value.nbytes + 2048 for value in arrays.values())
            require(used + expected + REPORT_RESERVE <= durable_cap)
            filename = output / (person + ".npz")
            with filename.open("xb") as target:
                np.savez(target, **arrays)
                target.flush()
                os.fsync(target.fileno())
            size = filename.stat().st_size
            require(size <= expected)
            used += size
            participant_files[person] = {"path": str(filename), "bytes": size, "main_images": MAIN_COUNT,
                                        "blocks": 12, "physical_sequences": 72}
            consumed.append(person)
            del arrays
            check()
    require(consumed == people)
    report.update(status="false_partition_inputs_complete", participants=participant_files,
        feature_images=IMAGE_COUNT, main_images=MAIN_COUNT, families=list(DIMENSIONS),
        actual_artifact_bytes_before_report=used, source_sample_rate_hz=1000,
        target_and_response_representation="unchanged binary source flags; event timing not selected here",
        all_main_images_once_per_participant=True, all_original_UUIDs_preserved=True,
        original_true_grouping_index="concept_by_block_image_row",
        source_columns_selected_in_memory=list(SOURCE_FIELDS),
        canonical_image_names_exported=False, caption_details_read=False,
        raw_EEG_or_pixels_read=False, audit_view_read=False, false_partitions_generated=0,
        false_bank_qualified=False, algorithm_or_seed_selected=False)
    check()
    return used


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--deadline-unix", required=True, type=float)
    parser.add_argument("--rss-bytes", required=True, type=int)
    parser.add_argument("--durable-bytes", required=True, type=int)
    args = parser.parse_args(argv)
    owned, report, started = False, {}, time.monotonic()
    try:
        require_compute()
        require(all(os.environ.get(name) == "1" for name in THREAD_ENV)
                and math.isfinite(args.deadline_unix) and args.deadline_unix > time.time()
                and args.rss_bytes > 0 and args.durable_bytes > MANIFEST_CAP + REPORT_RESERVE)
        output = path(args.output_dir)
        require(OUTPUTS / "runs" in output.resolve().parents)
        os.umask(0o077)
        output.mkdir(mode=0o700, parents=False, exist_ok=False)
        owned = True
        report = dict(status="started", job_id=os.environ["SLURM_JOB_ID"], hostname=socket.gethostname(),
            started_utc=datetime.now(timezone.utc).isoformat(),
            release_manifest=str(path(args.release_manifest)),
            separation="procedural_same_account_not_OS_isolation",
            caps={"deadline_unix": args.deadline_unix, "rss_bytes": args.rss_bytes,
                  "durable_bytes": args.durable_bytes})
        write_new(output / "settings.json", report, REPORT_RESERVE // 3)
        with bounded_process(args.deadline_unix, args.rss_bytes) as check:
            used = prepare(args.release_manifest, output, check, args.durable_bytes, report)
        report.update(elapsed_seconds=time.monotonic() - started,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        write_new(output / "report.json", report, REPORT_RESERVE // 3)
        print(json.dumps({"status": report["status"], "participants": 30, "main_images": MAIN_COUNT}), flush=True)
        return 0
    except BaseException as error:
        if owned:
            write_new(output / "failure.json", {"status": "false_partition_input_preparation_stopped",
                "error_type": type(error).__name__, "elapsed_seconds": time.monotonic() - started},
                REPORT_RESERVE // 3)
        print(json.dumps({"status": "false_partition_input_preparation_stopped"}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
