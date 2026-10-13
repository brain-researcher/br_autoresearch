"""Focused reader-boundary checks; run only inside the substantive allocation."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import inspect_session_metadata as reader

reader.compute_only()
reader.load_runtime()
h5py, np = reader.h5py, reader.np


def fixture(path, *, second_animal="Mihili", short_reference=False):
    forbidden = set()
    with h5py.File(path, "w") as handle:
        group = handle.create_group("trial_data")
        group.attrs["MATLAB_class"] = b"struct"
        objects = handle.create_group("#refs#")
        fields = {key: [] for key in reader.META + reader.GUIDES + reader.ARRAYS + ("vel",)}
        for trial in range(2):
            metadata = {
                "monkey": "Mihili" if trial == 0 else second_animal,
                "date_time": "2014/2/17 12:00:00", "bin_size": .03,
                "result": "R", "tgtDir": 0. if trial == 0 else np.pi / 4,
                "trial_id": trial + 11, "idx_startTime": 1, "idx_tgtOnTime": 2,
                "idx_goCueTime": 2, "idx_movement_on": 3, "idx_endTime": 4,
            }
            for name, value in metadata.items():
                data = ([[ord(v)] for v in value] if isinstance(value, str) else [[value]])
                node = objects.create_dataset(f"{name}_{trial}", data=data)
                if isinstance(value, str):
                    node.attrs["MATLAB_class"] = b"char"
                fields[name].append(node.ref)
            for region in reader.REGIONS:
                lfp_guide = [[channel, *band] for channel in (1, 2) for band in reader.BANDS]
                unit_guide = [[1, 1], [2, 1]]
                for kind, value in (("lfp_guide", np.array(lfp_guide).T),
                                    ("unit_guide", np.array(unit_guide).T)):
                    name = f"{region}_{kind}"
                    fields[name].append(objects.create_dataset(f"{name}_{trial}", data=value).ref)
                for kind, count in (("lfp", len(lfp_guide)), ("spikes", 2)):
                    name = f"{region}_{kind}"
                    # Deliberately nonfinite neural payload: structural inventory
                    # must not examine it or turn it into a failed trial.
                    node = objects.create_dataset(f"{name}_{trial}", shape=(count, 4), fillvalue=np.nan)
                    fields[name].append(node.ref)
                    forbidden.add(node.name)
            node = objects.create_dataset(f"vel_{trial}", shape=(2, 4), fillvalue=np.nan)
            fields["vel"].append(node.ref)
            forbidden.add(node.name)
        for name, refs in fields.items():
            if refs:
                if short_reference and name == "M1_spikes":
                    refs = refs[:1]
                node = group.create_dataset(name, shape=(1, len(refs)), dtype=h5py.ref_dtype)
                node[0, :] = refs
    return forbidden


class MetadataReaderTests(unittest.TestCase):
    def test_neural_value_access_denied_and_all_trials_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "Mihili_CO_20140217.mat"
            forbidden = fixture(path, second_animal="unexpected")
            original_get = h5py.Dataset.__getitem__
            original_array = h5py.Dataset.__array__

            def get(node, *args, **kwargs):
                self.assertNotIn(node.name, forbidden, "read a neural/velocity value")
                return original_get(node, *args, **kwargs)

            def array(node, *args, **kwargs):
                self.assertNotIn(node.name, forbidden, "coerced a neural/velocity array")
                return original_array(node, *args, **kwargs)

            with patch.object(h5py.Dataset, "__getitem__", get), patch.object(h5py.Dataset, "__array__", array):
                result = reader.inspect_session(path, "Mihili", "20140217")
            self.assertEqual(result["source_trial_count"], 2)
            self.assertEqual(len(result["trials"]), 2)
            self.assertEqual(result["trials"][0]["problems"], [])
            self.assertIn("animal_identity_mismatch", result["trials"][1]["problems"])
            self.assertEqual(result["status"], "metadata_inventory_with_problems")
            self.assertEqual(result["trials"][0]["array_headers"]["M1_spikes"]["shape"], [4, 2])
            self.assertEqual(result["trials"][1]["source_trial_index_zero_based"], 1)
            self.assertEqual(len(result["guide_catalog"]["M1_lfp_guide"]), 1)
            self.assertEqual(result["trials"][0]["guide_catalog_indices"], result["trials"][1]["guide_catalog_indices"])
            reader.atomic_json(Path(temp) / "metadata.json", result)

    def test_neural_field_cannot_enter_metadata_value_reader(self):
        for name in reader.ARRAYS + ("vel", "unknown_field"):
            with self.assertRaises(PermissionError):
                reader.metadata_value(object(), name)

    def test_unequal_reference_counts_fail_without_truncating_to_minimum(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "Mihili_CO_20140217.mat"
            fixture(path, short_reference=True)
            with self.assertRaisesRegex(reader.MetadataError, "trial_reference_count_or_type_mismatch:M1_spikes"):
                reader.inspect_session(path, "Mihili", "20140217")

    def test_unsupported_mat_format_has_no_full_deserialization_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "Mihili_CO_20140217.mat"
            path.write_bytes(b"not an HDF5 file")
            result = reader.inspect_session(path, "Mihili", "20140217")
            self.assertEqual(result["status"], "unsupported_format")
            self.assertEqual(result["trials"], [])

    def test_failed_serialization_preserves_completed_report(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "report.json"
            reader.atomic_json(path, {"status": "completed_session"})
            prior = path.read_bytes()
            with self.assertRaises(ValueError):
                reader.atomic_json(path, {"bad": float("nan")})
            self.assertEqual(path.read_bytes(), prior)

    def test_compute_guard_rejects_login_and_missing_thread_caps(self):
        good = {"SLURM_JOB_ID": "1", "SLURM_CPUS_PER_TASK": "1",
                **{key: "1" for key in reader.THREAD_VARS}}
        with patch.dict(reader.os.environ, good, clear=True), patch.object(reader.socket, "gethostname", return_value="sh04-ln04"):
            with self.assertRaisesRegex(RuntimeError, "compute shell"):
                reader.compute_only()
        with patch.dict(reader.os.environ, good, clear=True), patch.object(reader.socket, "gethostname", return_value="compute"):
            reader.os.environ.pop("BLIS_NUM_THREADS")
            with self.assertRaisesRegex(RuntimeError, "thread caps"):
                reader.compute_only()


if __name__ == "__main__":
    unittest.main()
