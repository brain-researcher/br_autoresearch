#!/usr/bin/env python3
"""EP05 original-source HDF5 adapter and bounded source qualification.

Reads only the selected provider source. No EP07/EP08 artifact is an input.
Qualification emits identity, shape, eligibility and support metadata, never
neural or behavioral magnitude summaries, fitted models or scores.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import re
import socket

import h5py
import numpy as np

DOI = "10.5061/dryad.xd2547dkt"
SOURCE_VERSION = "dryad-xd2547dkt-v5-payload"
M1_SESSIONS = (
    ("Mihili", "Mihili", "20140217"), ("Mihili", "Mihili", "20140218"),
    ("Mihili", "Mihili", "20140303"), ("Mihili", "Mihili", "20140304"),
    ("Mihili", "Mihili", "20140306"), ("Mihili", "Mihili", "20140307"),
    ("Chewie", "Chewie-L", "20160929"), ("Chewie", "Chewie-L", "20161005"),
    ("Chewie", "Chewie-L", "20161006"), ("Chewie", "Chewie-L", "20161007"),
    ("Chewie", "Chewie-L", "20161014"), ("Chewie", "Chewie-L", "20161021"),
    ("Chewie", "Chewie-R", "20150313"), ("Chewie", "Chewie-R", "20150319"),
    ("Chewie", "Chewie-R", "20150629"), ("Chewie", "Chewie-R", "20150630"),
)
OTHER_SESSIONS = (("Han", "20171127"), ("Han", "20171128"),
                  ("Han", "20171201"), ("Han", "20171204"), ("Han", "20171207"),
                  ("Lando", "20170731"), ("Lando", "20170803"), ("Lando", "20170917"))
REQUIRED = ("monkey", "date_time", "bin_size", "tgtDir", "idx_startTime",
            "idx_tgtOnTime", "idx_goCueTime", "idx_movement_on", "idx_endTime",
            "vel", "M1_lfp", "M1_lfp_guide", "M1_spikes", "M1_unit_guide")
EVENTS = ("idx_startTime", "idx_tgtOnTime", "idx_goCueTime", "idx_movement_on", "idx_endTime")


def compute_only():
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("source adapter must run in the coordinator's Slurm allocation")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


class TrialUnavailable(ValueError):
    pass


class M1Session:
    """Read-only MATLAB-v7.3 trial source; all public-source access is explicit."""
    def __init__(self, source, animal, implant, day):
        self.animal, self.implant, self.day = animal, implant, day
        self.session_id = f"{animal}_{day}"
        self.path = Path(source) / f"{animal}_CO_{day}.mat"

    def __enter__(self):
        self.handle = h5py.File(self.path, "r")
        self.group = self.handle["trial_data"]
        self.count = int(self.group["monkey"].size)
        if any(name not in self.group for name in REQUIRED):
            self.handle.close()
            raise ValueError("a required source field is absent")
        self.result_field = "result" if "result" in self.group else "Result"
        for name in (*REQUIRED, self.result_field):
            if self.group[name].size != self.count:
                self.handle.close()
                raise ValueError("trial-reference counts disagree between source fields")
        return self

    def __exit__(self, *args):
        self.handle.close()

    def field(self, trial, name):
        refs = self.group[name]
        reference = refs[np.unravel_index(trial, refs.shape, order="F")]
        if not reference:
            raise TrialUnavailable("empty_field_reference")
        return self.handle[reference]

    def text(self, trial, name):
        return "".join(chr(int(x)) for x in np.asarray(self.field(trial, name)).ravel(order="F"))

    def scalar(self, trial, name):
        node = self.field(trial, name)
        if node.size != 1:
            raise TrialUnavailable("nonscalar_identity_or_event")
        value = float(np.asarray(node).ravel()[0])
        if not math.isfinite(value):
            raise TrialUnavailable("nonfinite_identity_or_event")
        return value

    def metadata(self, trial):
        if self.text(trial, "monkey") != self.animal:
            raise ValueError("animal identity disagrees with selected source filename")
        date = self.text(trial, "date_time")
        match = re.match(r"^(\d{4})/(\d{1,2})/(\d{1,2})\s", date)
        if not match or "".join(f"{int(x):02d}" for x in match.groups()) != self.day:
            raise ValueError("day identity disagrees with selected source filename")
        seconds = self.scalar(trial, "bin_size")
        if seconds != 0.03:
            raise ValueError("source native bin is not the publication's 30ms")
        events = {}
        for name in EVENTS:
            value = self.scalar(trial, name)
            if value != math.floor(value) or value < 1:
                raise TrialUnavailable("invalid_one_based_event_index")
            events[name] = int(value) - 1
        if events["idx_startTime"] != 0:
            raise ValueError("unverified source trial-start convention")
        angle = self.scalar(trial, "tgtDir")
        direction = int(math.floor(angle / (math.pi / 4) + 0.5)) % 8
        deviation = abs(math.atan2(math.sin(angle - direction * math.pi / 4),
                                   math.cos(angle - direction * math.pi / 4)))
        return {"trial_id": f"{self.session_id}_trial{trial + 1:06d}",
                "source_trial_index_zero_based": trial, "animal": self.animal,
                "implant": self.implant, "session_id": self.session_id,
                "day": self.day, "result": self.text(trial, self.result_field),
                "direction": direction, "direction_radians": angle,
                "direction_circular_error": deviation,
                "native_bin_seconds": seconds, **events}

    def guides(self, trial):
        band = np.asarray(self.field(trial, "M1_lfp_guide")).T
        unit = np.asarray(self.field(trial, "M1_unit_guide")).T
        x, y = self.field(trial, "M1_lfp"), self.field(trial, "M1_spikes")
        if band.shape != (x.shape[0], 3) or unit.shape != (y.shape[0], 2):
            raise TrialUnavailable("guide_shape_mismatch")
        if not np.isfinite(band).all() or not np.isfinite(unit).all():
            raise TrialUnavailable("nonfinite_identity_guide")
        if len(set(map(tuple, band))) != len(band) or len(set(map(tuple, unit))) != len(unit):
            raise TrialUnavailable("duplicate_channel_or_unit_identity")
        return band, unit

    def window(self, trial, first_offset, last_offset, *, spikes=False):
        """Inclusive offsets around movement onset; no smoothing or transforms."""
        meta = self.metadata(trial)
        left = meta["idx_movement_on"] + first_offset
        right = meta["idx_movement_on"] + last_offset + 1
        x, vel = self.field(trial, "M1_lfp"), self.field(trial, "vel")
        if left < 0 or right > x.shape[1] or vel.shape != (2, x.shape[1]):
            raise TrialUnavailable("missing_common_window")
        band, units = self.guides(trial)
        result = {"metadata": meta, "lfp": x[:, left:right].T,
                  "velocity": vel[:, left:right].T, "lfp_guide": band,
                  "unit_guide": units,
                  "bin_offsets": np.arange(first_offset, last_offset + 1)}
        if spikes:
            node = self.field(trial, "M1_spikes")
            if node.shape[1] != x.shape[1]:
                raise TrialUnavailable("spike_time_axis_mismatch")
            # Preserve the released representation exactly. Integer-count
            # validity is not a kinematic endpoint eligibility condition.
            result["spikes"] = node[:, left:right].T
        return result


def inspect(source, out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "source_support.json").exists():
        raise FileExistsError("preserve prior source qualification; select a new diagnosed attempt")
    expected = {f"{a}_CO_{d}.mat" for a, _, d in M1_SESSIONS}
    expected.update(f"{a}_CO_{d}.mat" for a, d in OTHER_SESSIONS)
    expected.add("README.txt")
    objects = {entry.name: entry.stat().st_size for entry in source.iterdir() if entry.is_file()}
    report = {"analysis_id": "source_support", "source": str(source), "doi": DOI,
              "source_version": SOURCE_VERSION, "exposure": "entire_historical_release_exposed_development",
              "expected_object_count": 25, "observed_object_count": len(objects),
              "observed_total_bytes": sum(objects.values()), "objects": objects,
              "missing_expected_objects": sorted(expected - objects.keys()),
              "unexpected_objects": sorted(objects.keys() - expected),
              "scores_computed": False, "neural_values_or_moments_emitted": False,
              "trial_groups_or_session_folds_frozen": False,
              "kinematic_eligibility_requires_integer_spikes": False,
              "implant_assignment": "contract date-to-implant grouping; no hemisphere field assumed",
              "publication_window": "movement onset offsets -4 through +14 inclusive,19samples; paper Methods and Generate_all_figures trim_data",
              "main_example_window": "movement onset offsets0 through +15 inclusive,16samples; Main.m trim_data_v2; different from paper figure window",
              "sessions": []}
    rows = []
    for animal, implant, day in M1_SESSIONS:
        summary = {"animal": animal, "implant": implant, "day": day,
                   "session_id": f"{animal}_{day}"}
        rejected, supported = Counter(), Counter()
        main_example_supported = 0
        shape_pairs, guide_bands, matched_counts = set(), set(), set()
        with M1Session(source, animal, implant, day) as session:
            summary["source_trials"] = session.count
            for trial in range(session.count):
                try:
                    meta = session.metadata(trial)
                    if meta["result"] != "R":
                        raise TrialUnavailable("unsuccessful_reach")
                    if meta["direction_circular_error"] > 0.01:
                        raise TrialUnavailable("not_one_of_eight_nominal_directions_within_0p01rad")
                    record = session.window(trial, -4, 14, spikes=False)
                    if not np.isfinite(record["velocity"]).all():
                        raise TrialUnavailable("nonfinite_publication_velocity_window")
                    if not np.isfinite(record["lfp"]).all():
                        raise TrialUnavailable("nonfinite_publication_lfp_window")
                    bands, units = record["lfp_guide"], record["unit_guide"]
                    guide_bands.update(tuple(map(float, pair)) for pair in bands[:, 1:])
                    matched_counts.add(len(set(bands[:, 0]) & set(units[:, 0])))
                    y = session.field(trial, "M1_spikes")
                    x = session.field(trial, "M1_lfp")
                    shape_pairs.add((int(x.shape[0]), int(y.shape[0]), str(y.dtype)))
                    meta["spike_time_axis_matches_lfp"] = y.shape[1] == x.shape[1]
                    meta["publication_execution_support"] = True
                    try:
                        main_example = session.window(trial, 0, 15, spikes=False)
                        meta["main_example_execution_support"] = bool(
                            np.isfinite(main_example["lfp"]).all() and
                            np.isfinite(main_example["velocity"]).all())
                    except TrialUnavailable:
                        meta["main_example_execution_support"] = False
                    main_example_supported += int(meta["main_example_execution_support"])
                    supported[str(meta["direction"])] += 1
                    rows.append(meta)
                except TrialUnavailable as error:
                    rejected[str(error)] += 1
        summary.update(publication_supported_trials=sum(supported.values()),
                       main_example_supported_trials=main_example_supported,
                       publication_direction_counts={str(d): supported[str(d)] for d in range(8)},
                       support_failures=dict(rejected), band_pairs_hz=[list(p) for p in sorted(guide_bands)],
                       matched_spike_lfp_electrode_counts=sorted(matched_counts),
                       lfp_columns_spike_columns_spike_dtype=[list(x) for x in sorted(shape_pairs)])
        report["sessions"].append(summary)
        print(f"Source qualified: {animal} {day}", flush=True)
    # Metadata-only outputs preserve original trial numbering and biological
    # grouping; this is not a declaration that primary splits are now frozen.
    if rows:
        with (out / "trial_identity.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (out / "session_table.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["animal", "implant", "session_id", "day",
                                                   "source_trials", "publication_supported_trials"])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in report["sessions"])
    write_json(out / "source_support.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    compute_only()
    inspect(args.source, args.out)


if __name__ == "__main__":
    main()
