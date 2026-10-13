"""Source-free pointer-cache equivalence and unchanged qualification checks."""
import copy
from pathlib import Path
import unittest

import h5py
import numpy as np

import qualify_primary_intersection as original
import qualify_primary_intersection_cached as cached
from source_adapter import M1Session, TrialUnavailable
from test_primary_intersection_metadata import FakeSession, ShapeOnly, guides, records_for


class ReferenceTable:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=object)
        self.shape, self.size = self.values.shape, self.values.size
        self.dtype = h5py.ref_dtype
        self.array_reads, self.element_reads = 0, 0

    def __array__(self, dtype=None, copy=None):
        self.array_reads += 1
        return np.asarray(self.values, dtype=dtype)

    def __getitem__(self, index):
        self.element_reads += 1
        return self.values[index]


class Handle:
    def __init__(self, targets):
        self.targets, self.reads, self.closed = targets, [], False

    def __getitem__(self, reference):
        if str(reference).startswith("unselected:"):
            raise AssertionError("Unretained source trial was dereferenced")
        self.reads.append(reference)
        return self.targets[reference]

    def close(self):
        self.closed = True


def field_names(result="result"):
    return ("monkey", "date_time", "bin_size", "tgtDir", *cached.EVENTS, result,
            "M1_lfp_guide", "M1_unit_guide", "M1_lfp", "M1_spikes", "vel")


class SyntheticCached(cached.CachedMetadataSession):
    def __enter__(self):
        self.count, self.result_field = 20, "result"
        self.rows = records_for(self.session_id)
        lfp, units = guides()
        tables, targets = {}, {}
        def chars(text):
            return np.asarray([ord(char) for char in text], dtype=np.uint16)
        for field_index, name in enumerate(field_names()):
            references = []
            for trial in range(self.count):
                if trial >= len(self.rows):
                    references.append(f"unselected:{name}:{trial}")
                    continue
                reference = f"{name}:{trial}"
                references.append(reference)
                row = self.rows[trial]
                if name == "monkey":
                    value = chars(self.animal)
                elif name == "date_time":
                    value = chars(f"{self.day[:4]}/{self.day[4:6]}/{self.day[6:]} 00:00:00")
                elif name == "bin_size":
                    value = np.array([[.03]])
                elif name == "tgtDir":
                    value = np.array([[row["direction"]*np.pi/4]])
                elif name in cached.EVENTS:
                    value = np.array([[row[name]+1.]])
                elif name == "result":
                    value = chars("R")
                elif name == "M1_lfp_guide":
                    value = lfp.T.copy()
                elif name == "M1_unit_guide":
                    value = units.T.copy()
                else:
                    shape = {"M1_lfp": (len(lfp), 20), "M1_spikes": (len(units), 20), "vel": (2, 20)}[name]
                    value = ShapeOnly(shape, name, [])
                targets[reference] = value
            shape = (1, self.count) if field_index % 2 else (self.count, 1)
            tables[name] = ReferenceTable(np.asarray(references, dtype=object).reshape(shape, order="F"))
        self.group, self.handle = tables, Handle(targets)
        self._cache_reference_tables()
        return self


class CachedMetadataTest(unittest.TestCase):
    def test_non_object_reference_dtype_is_rejected_before_table_conversion(self):
        for dtype in (np.dtype("float64"), np.dtype("object"), h5py.regionref_dtype):
            session = cached.CachedMetadataSession.__new__(cached.CachedMetadataSession)
            session.count, session.result_field = 2, "result"
            session.group = {name: ReferenceTable(["one", "two"]) for name in field_names()}
            forbidden = session.group["monkey"]
            forbidden.dtype = dtype
            with self.assertRaisesRegex(ValueError, "object-reference"):
                session._cache_reference_tables()
            self.assertEqual(forbidden.array_reads, 0)
            self.assertTrue(all(table.array_reads == 0 for table in session.group.values()))
            self.assertEqual(session._references, {})

    def test_cached_lookup_matches_original_Fortran_semantics_for_nonconsecutive_trials(self):
        for shape in ((1, 6), (6, 1), (2, 3)):
            session = cached.CachedMetadataSession.__new__(cached.CachedMetadataSession)
            session.count, session.result_field = 6, "result"
            session.group = {name: ReferenceTable(np.asarray([f"{name}:{i}" for i in range(6)], dtype=object)
                                                   .reshape(shape, order="F")) for name in field_names()}
            targets = {f"{name}:{i}": object() for name in field_names() for i in range(6)}
            session.handle = Handle(targets)
            expected = {(name, i): M1Session.field(session, i, name) for i in (0, 2, 5) for name in field_names()}
            session.handle.reads.clear()
            session._cache_reference_tables()
            before_element_reads = {name: table.element_reads for name, table in session.group.items()}
            for trial in (0, 2, 5):
                for name in field_names():
                    self.assertIs(session.field(trial, name), expected[name, trial])
                    self.assertIs(session.field(trial, name), expected[name, trial])
                self.assertEqual(len(session._trial_nodes), 15)
            self.assertEqual(session.resolved_dataset_handles, 45)
            self.assertEqual(session.peak_current_trial_handles, 15)
            self.assertEqual(len(session.handle.reads), 45)
            self.assertEqual({name: table.element_reads for name, table in session.group.items()}, before_element_reads)
            self.assertTrue(all(table.array_reads == 1 for table in session.group.values()))

    def test_null_bad_count_and_out_of_range_fail_without_fallback_access(self):
        with SyntheticCached(Path("unopened"), "Chewie", "Chewie-L", "20150313") as session:
            for trial, name in ((-1, "M1_lfp"), (20, "M1_lfp"), (True, "M1_lfp"), (0, "undeclared")):
                with self.assertRaises(ValueError):
                    session.field(trial, name)
            session._references["M1_lfp"][0] = None
            with self.assertRaises(TrialUnavailable):
                session.field(0, "M1_lfp")
        self.assertTrue(session.handle.closed)
        self.assertEqual(session._references, {})
        self.assertEqual(session._trial_nodes, {})
        session = cached.CachedMetadataSession.__new__(cached.CachedMetadataSession)
        session.count, session.result_field = 2, "result"
        session.group = {name: ReferenceTable(["one", "two"]) for name in field_names()}
        session.group["vel"] = ReferenceTable(["one"])
        with self.assertRaises(ValueError):
            session._cache_reference_tables()

    def test_original_qualification_output_is_exact_and_neural_values_never_open(self):
        session_id = "Chewie_20150313"
        expected = {"session_id": session_id, "complete_trials": 16,
                    "direction_counts": [2]*8, "original_electrode_ids": [2, 4, 7]}
        uncached = FakeSession(Path("unopened"), "Chewie", "Chewie-L", "20150313")
        uncached.count = 20
        before = original.qualify_session(uncached, uncached.rows, expected)
        with SyntheticCached(Path("unopened"), "Chewie", "Chewie-L", "20150313") as session:
            after = original.qualify_session(session, records_for(session_id), expected)
            self.assertEqual(after, before)
            self.assertEqual(session.resolved_dataset_handles, 15*16)
            self.assertEqual(session.peak_current_trial_handles, 15)
            self.assertEqual(len(session.handle.reads), 15*16)
            self.assertTrue(all(int(reference.rsplit(":", 1)[1]) < 16 for reference in session.handle.reads))
            self.assertTrue(all(table.array_reads == 1 and table.element_reads == 0 for table in session.group.values()))

    def test_cached_path_still_rejects_changed_last_guide(self):
        session_id = "Chewie_20150313"
        expected = {"session_id": session_id, "complete_trials": 16,
                    "direction_counts": [2]*8, "original_electrode_ids": [2, 4, 7]}
        with SyntheticCached(Path("unopened"), "Chewie", "Chewie-L", "20150313") as session:
            session.handle.targets["M1_unit_guide:15"][1, 0] = 99.
            with self.assertRaisesRegex(ValueError, "invariance"):
                original.qualify_session(session, records_for(session_id), expected)

    def test_complete_and_failure_progress_never_promotes_partial_metadata(self):
        records = {s: records_for(s) for s in original.SESSIONS}
        expected = {s: {"session_id": s, "complete_trials": 16, "direction_counts": [2]*8,
                       "original_electrode_ids": [2, 4, 7]} for s in original.SESSIONS}
        report, progress = {}, []
        cached.qualify_all_cached(Path("unopened"), records, expected, report,
                                 session_factory=SyntheticCached, on_complete=progress.append)
        self.assertTrue(report["all_sixteen_qualified_before_values"])
        self.assertEqual(len(report["sessions"]), 16)
        self.assertEqual([row["completed_sessions"] for row in progress], list(range(1, 17)))
        self.assertTrue(all(row["resolved_dataset_handles"] == 240 for row in progress))
        self.assertEqual(report["pilot_session"], original.choose_pilot(report["sessions"]))
        def fail_last(*args):
            if f"{args[1]}_{args[3]}" == original.SESSIONS[-1]:
                raise TimeoutError("synthetic bounded metadata stop")
            return SyntheticCached(*args)
        failed, progress = {}, []
        with self.assertRaises(TimeoutError):
            cached.qualify_all_cached(Path("unopened"), records, expected, failed,
                                     session_factory=fail_last, on_complete=progress.append)
        self.assertFalse(failed["all_sixteen_qualified_before_values"])
        self.assertEqual(len(failed["sessions"]), 15)
        self.assertEqual(len(progress), 15)
        self.assertNotIn("pilot_session", failed)


if __name__ == "__main__":
    unittest.main()
