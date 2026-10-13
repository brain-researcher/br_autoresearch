"""Four metadata-only fixtures; coordinator runs these in its Slurm session."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import development_stream_layout as layout


def cohort(view_root):
    headers, files = [], []
    people = sorted(layout.DEVELOPMENT_PEOPLE)
    for index in range(155):
        person = people[index % 15]
        stem = f"{person}/ses-01/eeg/{person}_ses-01_task-reachingandgrasping_run-{index + 1:04d}_eeg"
        header = {"member": stem + ".vhdr", "participant": person,
                  "data_file": Path(stem).name + ".eeg",
                  "marker_file": Path(stem).name + ".vmrk",
                  "data_member_present": True, "marker_member_present": True,
                  "data_format": "BINARY", "data_orientation": "MULTIPLEXED",
                  "binary_format": "IEEE_FLOAT_32", "channel_count": 3,
                  "sampling_interval_microseconds": 4000.0}
        headers.append(header)
        for suffix in (".vhdr", ".eeg", ".vmrk"):
            member = stem + suffix
            files.append({"source": "self_paced", "role": "development", "member": member,
                          "destination": str(view_root / "self_paced/development" / member),
                          "bytes": 120 if suffix == ".eeg" else 500})
    return ({"status": "development_header_inspection_complete", "header_count": 155,
             "headers": headers}, {"status": "development_view_materialized", "files": files})


class DevelopmentStreamLayoutTests(unittest.TestCase):
    def test_exact_complete_triplets_are_admitted_before_file_access(self):
        view_root = Path("/test/development_view")
        prior, view = cohort(view_root)
        # Admission is a metadata operation; no stat/open/resolve is allowed.
        with patch.object(Path, "stat", side_effect=AssertionError("premature stat")), \
                patch.object(Path, "open", side_effect=AssertionError("premature open")):
            admitted = layout.admit_runs(prior, view, view_root)
            self.assertEqual(len(admitted), 155)
            self.assertEqual({row["participant"] for row in admitted}, layout.DEVELOPMENT_PEOPLE)
            self.assertEqual(len({str(row["data_path"]) for row in admitted}), 155)
            self.assertTrue(all(not row["triplet_binding"]["unresolved"] for row in admitted))
            # Preserve three original-reference defects without inferring that
            # the materialized same-stem binary or marker is their replacement.
            changed = copy.deepcopy(prior)
            for index, kind in ((0, "marker"), (1, "data"), (2, "marker")):
                suffix = ".eeg" if kind == "data" else ".vmrk"
                changed["headers"][index][kind + "_file"] = "old_recorder_name" + suffix
                changed["headers"][index][kind + "_member_present"] = False
            retained = layout.admit_runs(changed, view, view_root)
            self.assertEqual(len(retained), 155)
            unresolved = [row for row in retained if row["triplet_binding"]["unresolved"]]
            self.assertEqual(len(unresolved), 3)
            self.assertEqual({row["header_member"] for row in unresolved},
                             {row["member"] for row in changed["headers"][:3]})
            with patch.object(layout, "inspect_binary", side_effect=AssertionError("binary access")), \
                    patch.object(layout, "inspect_markers", side_effect=AssertionError("marker access")):
                for item in unresolved:
                    row = layout.inspect_run(item)
                    self.assertEqual(len(row["triplet_binding"]["unresolved"]), 2)
                    self.assertEqual(row["triplet_binding"]["status"], "unresolved_no_source_access")
                    self.assertFalse(row["source_files_checked"])
                    self.assertIsNone(row["binary_layout"]["file_bytes"])
                    self.assertIsNone(row["binary_layout"]["sample_count"])
                    self.assertIsNone(row["segment_metadata"]["new_segments"])
            changed = copy.deepcopy(view)
            changed["files"].pop()
            with self.assertRaisesRegex(ValueError, "465"):
                layout.admit_runs(prior, changed, view_root)
            changed = copy.deepcopy(prior)
            changed["headers"][0]["data_file"] = "../../sub-01/closed.eeg"
            with self.assertRaisesRegex(ValueError, "safe_basename"):
                layout.admit_runs(changed, view, view_root)
            changed = copy.deepcopy(view)
            changed["files"][1]["role"] = "held_out"
            with self.assertRaisesRegex(ValueError, "identity"):
                layout.admit_runs(prior, changed, view_root)
            changed = copy.deepcopy(view)
            changed["files"][2]["destination"] = str(view_root / "self_paced/development/../closed.vmrk")
            with self.assertRaisesRegex(ValueError, "identity"):
                layout.admit_runs(prior, changed, view_root)
            changed = copy.deepcopy(view)
            changed["files"][2]["destination"] = "/test/sub-01/closed.vmrk"
            with self.assertRaisesRegex(ValueError, "identity"):
                layout.admit_runs(prior, changed, view_root)

    def test_binary_sample_count_divisibility_without_binary_open(self):
        header = {"data_format": "BINARY", "data_orientation": "MULTIPLEXED",
                  "binary_format": "IEEE_FLOAT_32", "channel_count": 3,
                  "sampling_interval_microseconds": 4000.0}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.eeg"
            path.write_bytes(bytes(120))
            with patch.object(Path, "open", side_effect=AssertionError("binary payload opened")):
                result = layout.inspect_binary(path, 120, header)
                self.assertEqual(result["sample_count"], 10)
                self.assertEqual(result["bytes_per_sample_frame"], 12)
                self.assertEqual(result["binary_prefix_bytes"], 0)
                self.assertEqual(result["remainder_bytes"], 0)
                self.assertAlmostEqual(result["nominal_stored_duration_seconds"], .04)
                with self.assertRaisesRegex(ValueError, "size"):
                    layout.inspect_binary(path, 119, header)
        integer = dict(header, binary_format="INT_16")
        self.assertEqual(layout.binary_layout(integer, 120)["sample_count"], 20)
        truncated = layout.binary_layout(header, 121)
        self.assertIsNone(truncated["sample_count"])
        self.assertEqual(truncated["remainder_bytes"], 1)
        for change in ({"binary_format": None}, {"binary_format": "UINT_16"},
                       {"data_orientation": "VECTORIZED"}, {"data_format": "ASCII"}):
            unresolved = layout.binary_layout(dict(header, **change), 120)
            self.assertIsNone(unresolved["sample_count"])
            self.assertIsNone(unresolved["bytes_per_sample_frame"])
            self.assertTrue(unresolved["unresolved"])

    def test_documentary_mapping_keeps_defects_and_blocks_competing_identified_runs(self):
        view_root = Path("/test/development_view")
        prior, view = cohort(view_root)
        basis = layout.PUBLISHER_MAPPING
        with patch.object(Path, "stat", side_effect=AssertionError("premature stat")), \
                patch.object(Path, "open", side_effect=AssertionError("premature open")):
            matched = layout.admit_runs(prior, view, view_root, mapping_basis=basis)
            self.assertEqual(len(matched), 155)
            self.assertTrue(all(row["filename_mapping"]["accepted"] for row in matched))
            self.assertTrue(all(row["triplet_binding"]["reference_verified"] for row in matched))
            changed = copy.deepcopy(prior)
            changed["headers"][0].update(data_file="legacy_recording.eeg", data_member_present=False)
            admitted = layout.admit_runs(changed, view, view_root, mapping_basis=basis)
            first = next(row for row in admitted if row["header_member"] == prior["headers"][0]["member"])
            self.assertTrue(first["filename_mapping"]["accepted"])
            self.assertFalse(first["triplet_binding"]["reference_verified"])
            self.assertEqual(first["triplet_binding"]["original_references"]["data"],
                             {"filename": "legacy_recording.eeg", "member_present": False,
                              "matches_manifest": False})
            self.assertEqual(len(first["triplet_binding"]["unresolved"]), 2)
            binary_path = first["data_path"]
            marker_path = first["marker_path"]
            binary = layout.binary_layout(first["header"], first["data_bytes"])
            def marker_reader(path, size, header, count, **options):
                self.assertEqual(path, marker_path)
                self.assertEqual(options["mapping_basis"], basis)
                self.assertEqual(options["known_data_files"], first["known_data_files"])
                return layout.parse_new_segments(io.BytesIO(
                    b"BrainVision Data Exchange Marker File Version 1.0\n"
                    b"[Common Infos]\nDataFile=legacy_recording.eeg\n[Marker Infos]\n"
                    b"Mk1=New Segment,,1,1,0\n"), binary_path.name, count, 4000.0, **options)
            with patch.object(layout, "inspect_binary", return_value=binary) as binary_reader, \
                    patch.object(layout, "inspect_markers", side_effect=marker_reader):
                observed = layout.inspect_run(first)
                binary_reader.assert_called_once_with(binary_path, first["data_bytes"], first["header"])
            self.assertTrue(observed["source_files_checked"])
            self.assertTrue(observed["metadata_file_association_accepted"])
            self.assertFalse(observed["triplet_binding"]["reference_verified"])
            binding = observed["segment_metadata"]["data_file_binding"]
            self.assertFalse(binding["reference_verified"])
            self.assertEqual(binding["original_references"], ["legacy_recording.eeg"])
            self.assertTrue(binding["documentary_association_accepted"])
            self.assertIn("marker_data_file_binding_unverified", observed["segment_metadata"]["unresolved"])
            # An embedded reference to another existing identified run is a
            # genuine competing association, not an absent legacy basename.
            changed["headers"][0].update(data_file=prior["headers"][1]["data_file"], data_member_present=True)
            conflicting = layout.admit_runs(changed, view, view_root, mapping_basis=basis)
            blocked = next(row for row in conflicting if row["header_member"] == first["header_member"])
            self.assertEqual(blocked["filename_mapping"]["status"], "blocked_competing_frozen_run_reference")
            with patch.object(layout, "inspect_binary", side_effect=AssertionError("competing binary read")), \
                    patch.object(layout, "inspect_markers", side_effect=AssertionError("competing marker read")):
                refused = layout.inspect_run(blocked)
                self.assertFalse(refused["source_files_checked"])
                self.assertFalse(refused["metadata_file_association_accepted"])
                self.assertIsNone(refused["binary_layout"]["sample_count"])
            duplicate = copy.deepcopy(view)
            duplicate["files"][-1] = copy.deepcopy(duplicate["files"][-2])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                layout.admit_runs(prior, duplicate, view_root, mapping_basis=basis)

        expected = prior["headers"][0]["data_file"]
        other = prior["headers"][1]["data_file"]
        prefix = b"BrainVision Data Exchange Marker File Version 1.0\n[Common Infos]\n"
        suffix = b"[Marker Infos]\nMk1=New Segment,,1,1,0\n"
        known = (expected, other)
        for declarations, count in ((b"", 0),
                                     ((f"DataFile={expected}\n" * 2).encode(), 2)):
            parsed = layout.parse_new_segments(io.BytesIO(prefix + declarations + suffix),
                                               expected, 10, 4000.0,
                                               mapping_basis=basis, known_data_files=known)
            binding = parsed["data_file_binding"]
            self.assertEqual(binding["reference_count"], count)
            self.assertFalse(binding["reference_verified"])
            self.assertTrue(binding["documentary_association_accepted"])
            self.assertEqual(len(parsed["new_segments"]), 1)
            self.assertIn("marker_data_file_binding_unverified", parsed["unresolved"])
        for declarations in (f"DataFile={other}\n", f"DataFile={expected}\nDataFile={other}\n"):
            parsed = layout.parse_new_segments(io.BytesIO(prefix + declarations.encode() + suffix),
                                               expected, 10, 4000.0,
                                               mapping_basis=basis, known_data_files=known)
            binding = parsed["data_file_binding"]
            self.assertEqual(binding["status"], "blocked_competing_frozen_run_reference")
            self.assertFalse(binding["reference_verified"])
            self.assertFalse(binding["documentary_association_accepted"])
            self.assertEqual(binding["competing_references"], [other])

    def test_only_new_segments_survive_with_one_based_positions_and_unknown_dates(self):
        content = (b"Brain Vision Data Exchange Marker File, Version 1.0\r\n"
                   b"[Common Infos]\r\nDataFile=run.eeg\r\n[Marker Infos]\r\n"
                   b"Mk1=New Segment,SECRET_DESCRIPTION,1,1,0,20240229123456123456\r\n"
                   b"Mk2=Stimulus,SECRET_STIMULUS\xff,no,valid,coordinates\r\n"
                   b"Mk3=Response,SECRET_RESPONSE,7,1,0,not_a_timestamp\r\n"
                   b"Mk4=New Segment,SECRET_DESCRIPTION,11,1,0\r\n"
                   b"Mk5=New Segment,SECRET_DESCRIPTION,10,1,0,20230229123456000000\r\n"
                   b"Mk6=New Segment,,22,1,0,\r\n")
        result = layout.parse_new_segments(io.BytesIO(content), "run.eeg", 20, 4000.0)
        segments = result["new_segments"]
        self.assertEqual(len(segments), 4)
        self.assertEqual([entry["position_one_based"] for entry in segments], [1, 11, 10, 22])
        self.assertEqual([entry["sample_offset_zero_based"] for entry in segments], [0, 10, 9, 21])
        self.assertEqual(segments[0]["timestamp"], "2024-02-29T12:34:56.123456")
        self.assertEqual(segments[1]["timestamp_status"], "absent")
        self.assertAlmostEqual(segments[1]["nominal_offset_seconds"], .04)
        self.assertIn("invalid_new_segment_timestamp", segments[2]["unresolved"])
        self.assertIn("new_segment_positions_not_strictly_increasing", segments[2]["unresolved"])
        self.assertIn("position_exceeds_sample_count", segments[3]["unresolved"])
        self.assertFalse(result["absence_proves_gap_free"])
        self.assertEqual(result["timestamp_timezone"], "unspecified")
        observed_spelling = content.replace(
            b"Brain Vision Data Exchange Marker File, Version 1.0",
            b"BrainVision Data Exchange Marker File Version 1.0", 1)
        self.assertEqual(layout.parse_new_segments(
            io.BytesIO(observed_spelling), "run.eeg", 20, 4000.0), result)
        for version in (b"1.1", b"2.0", b"3.0"):
            unsupported = observed_spelling.replace(b"Version 1.0", b"Version " + version, 1)
            refused = layout.parse_new_segments(io.BytesIO(unsupported), "run.eeg", 20, 4000.0)
            self.assertEqual(refused["unresolved"], ["unsupported_marker_signature"])
            self.assertEqual(refused["new_segments"], [])
        rendered = json.dumps(result)
        for forbidden in ("SECRET", "Stimulus", "Response", "not_a_timestamp"):
            self.assertNotIn(forbidden, rendered)
        mismatched = layout.parse_new_segments(io.BytesIO(content), "different.eeg", None, 4000.0)
        self.assertIn("marker_data_file_binding_unverified", mismatched["unresolved"])
        self.assertIn("position_sample_bound_unverified", mismatched["new_segments"][0]["unresolved"])
        absent = layout.parse_new_segments(io.BytesIO(
            b"Brain Vision Data Exchange Marker File, Version 1.0\n"
            b"[Common Infos]\nDataFile=run.eeg\n[Marker Infos]\n"
            b"Mk1=Stimulus,SECRET\n"), "run.eeg", 20, 4000.0)
        self.assertIn("no_new_segment_metadata_available", absent["unresolved"])
        self.assertEqual(absent["new_segments"], [])
        self.assertFalse(absent["absence_proves_gap_free"])


if __name__ == "__main__":
    unittest.main()
