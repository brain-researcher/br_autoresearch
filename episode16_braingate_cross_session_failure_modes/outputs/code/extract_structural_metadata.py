#!/usr/bin/env python3
"""Extract only EP16-authorized structural fields from explicitly named archives.

MAT bytes are copied opaquely into an unlinked private scratch file, one session
at a time. Neural, cursor and impedance variables are never deserialized. The
retained view contains one target coordinate per trial, not target trajectories.
This preparation neither assigns roles nor declares neural packet eligibility.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import socket
import tarfile
import tempfile
import time

import numpy as np

from mat5_structural import field_vector, read_structural


PARTICIPANTS = {"T2", "T3", "T5", "T6", "T7", "T8", "T9", "T10", "T11"}


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [plain(item) for item in value]
    return value


def integers(value, field):
    vector = field_vector(value, field)
    if not np.all(np.isfinite(vector)) or np.any(vector != np.floor(vector)):
        raise ValueError(f"Invalid integer structural field: {field}")
    return vector.astype(np.int64)


def summarize(values, participant, day):
    if values["participant_id"] != participant:
        raise ValueError("Internal participant ID differs from archive identity")
    recorded_day = np.asarray(values["post_implant_day"]).reshape(-1)
    if len(recorded_day) != 1 or recorded_day[0] != day:
        raise ValueError("Internal day differs from session filename")
    block_ids = integers(values["blocks"], "block_id")
    block_starts = integers(values["blocks"], "sample_start_index")
    block_ends = integers(values["blocks"], "sample_end_index")
    starts = integers(values["trials"], "trial_start_index")
    ends = integers(values["trials"], "trial_end_index")
    trial_blocks = integers(values["trials"], "block_id")
    if not len(block_ids) == len(block_starts) == len(block_ends):
        raise ValueError("Unequal block metadata lengths")
    if not len(starts) == len(ends) == len(trial_blocks):
        raise ValueError("Unequal trial metadata lengths")
    if len(set(block_ids.tolist())) != len(block_ids):
        raise ValueError("Block IDs are not unique within session")
    lookup = {int(key): (int(start), int(end)) for key, start, end in zip(block_ids, block_starts, block_ends)}
    if any(key not in lookup for key in trial_blocks):
        raise ValueError("Trial references an unknown block")
    whole = np.array([lookup[int(key)][0] <= start < end <= lookup[int(key)][1]
                      for key, start, end in zip(trial_blocks, starts, ends)], dtype=bool)
    per_block = {str(int(key)): int(np.sum(whole & (trial_blocks == key))) for key in block_ids}
    # Necessary structural support only: later onset, cursor-distance and octant
    # checks can invalidate trials. No success or neural measurement enters here.
    possible = {0}
    for count in per_block.values():
        possible |= {value+count for value in possible}
    n_whole = int(whole.sum())
    two_sides = any(value >= 64 and n_whole-value >= 64 for value in possible)
    return {"participant": participant, "post_implant_day": day,
            "blocks": len(block_ids), "trials": len(starts), "whole_trials": n_whole,
            "whole_trial_counts_by_block": per_block,
            "whole_blocks_can_supply_two_64_trial_sides": two_sides,
            "final_trial_packet_eligibility_evaluated": False,
            "unique_trial_target_coordinates": len(np.unique(values["trial_targets"], axis=0))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--participants", nargs="+", required=True, choices=sorted(PARTICIPANTS))
    parser.add_argument("--max-sessions", type=int, help="Representative reader qualification only; omit for full census")
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Use the coordinator's verified compute allocation")
    if len(set(args.participants)) != len(args.participants):
        raise ValueError("Duplicate participants")
    if args.max_sessions is not None and args.max_sessions < 1:
        raise ValueError("Representative session count must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.scratch_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    started = time.monotonic()
    summaries, traces, archives = [], [], []
    decoded_bytes = 0
    with tempfile.TemporaryDirectory(prefix="ep16-structural-", dir=args.scratch_dir) as private:
        with (args.output_dir/"structural_sessions.jsonl").open("x") as output:
            for participant in args.participants:
                archive = args.source_root/"source"/f"decoding_{participant}.tar.gz"
                archives.append(str(archive))
                count = 0
                pattern = re.compile(rf"decoding/{participant}/{participant}_day_([0-9]+)_decoding\.mat")
                with tarfile.open(archive, "r|gz") as stream:
                    for member in stream:
                        match = pattern.fullmatch(member.name.removeprefix("./"))
                        if not member.isfile() or match is None:
                            continue
                        with tempfile.TemporaryFile(dir=private) as opaque:
                            source = stream.extractfile(member)
                            if source is None:
                                raise ValueError("Unreadable session member")
                            shutil.copyfileobj(source, opaque, length=1024*1024)
                            values, trace = read_structural(opaque)
                        summary = summarize(values, participant, int(match.group(1)))
                        record = {"archive": str(archive), "member": member.name,
                                  "fields": plain(values), "summary": summary, "field_access": trace}
                        output.write(json.dumps(record, allow_nan=False, separators=(",", ":"))+"\n")
                        output.flush()
                        summaries.append(summary)
                        traces.append(trace)
                        decoded_bytes += member.size
                        count += 1
                        if args.max_sessions is not None and count >= args.max_sessions:
                            break
                if count == 0:
                    raise ValueError(f"No exact decoding session members found for {participant}")
                print(json.dumps({"participant": participant, "structural_sessions_written": count,
                                  "elapsed_seconds": time.monotonic()-started}), flush=True)
    report = {"analysis_id": "structural_support", "phase": "field_selective_metadata",
              "status": "representative_reader_qualified" if args.max_sessions else "structural_fields_extracted",
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
              "observed_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic()-started,
              "archives": archives, "sessions": len(summaries),
              "opaque_member_bytes_read": decoded_bytes,
              "decoded_paths": sorted({path for trace in traces for path in trace["decoded_paths"]}),
              "skipped_paths": sorted({path for trace in traces for path in trace["skipped_paths"]}),
              "target_rows_decoded": sum(trace["target_rows_decoded"] for trace in traces),
              "neural_cursor_impedance_values_decoded": False, "role_assignment_complete": False,
              "schedule_equivalence_adjudicated": False, "session_summaries": summaries}
    (args.output_dir/"structural_metadata_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: value for key, value in report.items() if key != "session_summaries"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
