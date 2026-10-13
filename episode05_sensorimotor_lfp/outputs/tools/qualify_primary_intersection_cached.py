#!/usr/bin/env python3
"""Bounded guide-only retry with cached cell references; original checks intact."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import socket
import time

import h5py
import numpy as np

import qualify_primary_intersection as original
from source_adapter import EVENTS, M1Session, TrialUnavailable, compute_only


class CachedMetadataSession(M1Session):
    """Cache metadata pointers and at most one trial's fifteen dataset handles."""
    def _cache_reference_tables(self):
        names = ("monkey", "date_time", "bin_size", "tgtDir", *EVENTS, self.result_field,
                 "M1_lfp_guide", "M1_unit_guide", "M1_lfp", "M1_spikes", "vel")
        if len(set(names)) != 15:
            raise ValueError("Exactly the declared fifteen metadata/shape reference fields required")
        self._references = {}
        for name in names:
            # These are MATLAB cell/object-reference tables. Numerical target
            # datasets are not converted here or anywhere in this subclass.
            table = self.group[name]
            if h5py.check_dtype(ref=table.dtype) is not h5py.Reference:
                raise ValueError("Only HDF5 object-reference tables may be cached")
            references = np.asarray(table).ravel(order="F")
            if references.size != self.count:
                raise ValueError("Cached source reference count changed")
            self._references[name] = references
        self._current_trial = None
        self._trial_nodes = {}
        self.resolved_dataset_handles = 0
        self.peak_current_trial_handles = 0

    def __enter__(self):
        super().__enter__()
        try:
            self._cache_reference_tables()
        except Exception:
            self.handle.close()
            raise
        return self

    def field(self, trial, name):
        if (type(trial) is not int or not 0 <= trial < self.count
                or name not in self._references):
            raise ValueError("Only declared fields of valid original source trials are accessible")
        if trial != self._current_trial:
            self._trial_nodes.clear()
            self._current_trial = trial
        if name not in self._trial_nodes:
            reference = self._references[name][trial]
            if not reference:
                raise TrialUnavailable("empty_field_reference")
            self._trial_nodes[name] = self.handle[reference]
            self.resolved_dataset_handles += 1
            self.peak_current_trial_handles = max(self.peak_current_trial_handles, len(self._trial_nodes))
        return self._trial_nodes[name]

    def __exit__(self, *args):
        self._trial_nodes.clear()
        self._references.clear()
        return super().__exit__(*args)


def qualify_all_cached(source, records, expected, report, check=lambda: None,
                       session_factory=None, on_complete=lambda row: None):
    if set(records) != set(original.SESSIONS) or set(expected) != set(original.SESSIONS):
        raise ValueError("All original records must reconcile before any source guides")
    factory = CachedMetadataSession if session_factory is None else session_factory
    report.update(sessions=[], all_sixteen_qualified_before_values=False)
    for session_id in original.SESSIONS:
        check()
        started = time.monotonic()
        specification = next(row for row in original.M1_SESSIONS if f"{row[0]}_{row[2]}" == session_id)
        with factory(source, *specification) as session:
            # Reuse the adopted qualification verbatim. Repeated field calls
            # resolve to the same current-trial node; no checks are removed.
            description = original.qualify_session(session, records[session_id], expected[session_id], check)
            description["metadata_access_resources"] = {
                "elapsed_seconds": time.monotonic()-started,
                "cached_reference_table_count": len(session._references),
                "resolved_dataset_handles": session.resolved_dataset_handles,
                "peak_current_trial_handles": session.peak_current_trial_handles}
        report["sessions"].append(description)
        on_complete({"qualified_session": session_id, "completed_sessions": len(report["sessions"]),
                     **description["metadata_access_resources"]})
    report.update(pilot_session=original.choose_pilot(report["sessions"]),
        pilot_selection_rule="maximum u*(105*N*(4*k)^2+8*(4*k)^3); original session-order ties",
        global_full_HW_duplicate=all(row["unique_new_fit_supports"] == 0 for row in report["sessions"]),
        all_sixteen_qualified_before_values=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "trial-table", "scalar-bank-root", "hw-bank-root", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=300)
    args = parser.parse_args()
    compute_only()
    if not 0 < args.max_seconds <= 300:
        raise ValueError("The adopted fresh metadata retry bound is at most300seconds")
    if any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise RuntimeError("Explicit one-thread settings required")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "started", "scientific_trial_count_increment": 0,
        "job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
        "started_utc": datetime.now(timezone.utc).isoformat(), "source": str(args.source),
        "trial_table": str(args.trial_table), "scalar_bank_root": str(args.scalar_bank_root),
        "HW_bank_root": str(args.hw_bank_root), "spike_values_read": False,
        "LFP_XY_values_read": False, "scores_computed": False,
        "engineering_revision": "cached_cell_reference_tables_and_current_trial_nodes_v1",
        "internal_bound_seconds": args.max_seconds, "all_sixteen_qualified_before_values": False,
        "preserved_failed_attempt": "outputs/runs/source_support/primary_intersection_metadata_attempt001"}
    def deadline(*_):
        raise TimeoutError("Selected cached guide-only qualification deadline reached")
    def check():
        if time.monotonic()-started >= args.max_seconds:
            deadline()
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    try:
        records = {s: original.trial_records(args.trial_table, s) for s in original.SESSIONS}
        expected = {}
        for session in original.SESSIONS:
            check()
            scalar = json.loads((args.scalar_bank_root/session/"score_bank.json").read_text())
            history = json.loads((args.hw_bank_root/session/"score_banks.json").read_text())
            expected[session] = original.original_metadata(scalar, history, session, records[session])
        qualify_all_cached(args.source, records, expected, report, check,
            on_complete=lambda row: print(json.dumps(row, allow_nan=False), flush=True))
        check()
        report["status"] = "complete_primary_intersection_metadata"
    except Exception as error:
        report.update(status="technical_failure", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started,
            process_CPU_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,
            maximum_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        (args.out/"primary_intersection_metadata.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ("status", "elapsed_seconds", "pilot_session", "scientific_trial_count_increment")}), flush=True)


if __name__ == "__main__":
    main()
