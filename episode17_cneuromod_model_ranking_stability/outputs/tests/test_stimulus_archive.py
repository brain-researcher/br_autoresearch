"""Small source-free fixtures; execute only in the coordinator's Slurm shell."""
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import inspect_stimulus_archive as adapter


def tiny_zip(names=("images_fmri/acorn/acorn_01b.jpg",), extra=b"", comment=b""):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.comment = comment
        for name in names:
            info = zipfile.ZipInfo(name)
            info.extra = extra
            archive.writestr(info, b"fixture bytes, never a decoded image")
    return stream.getvalue()


def eocd_position(data):
    return data.rfind(b"PK\x05\x06")


def change_eocd(data, field_offset, fmt, value):
    data = bytearray(data)
    struct.pack_into(fmt, data, eocd_position(data) + field_offset, value)
    return bytes(data)


def inspect(data):
    return adapter.inspect_stream(io.BytesIO(data), len(data))


def identities():
    concepts, images, provider = [], [], []
    for cid, role in ((9, "development"), (2, "development"), (3, "sealed_audit")):
        name = f"concept{cid}"
        concept = {"things_category_nr": str(cid), "uniqueID": name,
                   "provider_category_nr": str(cid + 10), "qualified_image_count": "2",
                   "role": role, "outer_fold": "fold01" if role == "development" else "",
                   "development_block": "block01" if role == "development" else "",
                   "audit_block": "audit01" if role == "sealed_audit" else ""}
        concepts.append(concept)
        for exemplar in (2, 1):
            filename = f"{name}_{exemplar:02d}b.jpg"
            path = f"images_fmri/{name}/{filename}"
            images.append({"provider_image_path": path, "image_name": filename,
                           "concept_name": name, "things_category_nr": str(cid),
                           "provider_category_nr": str(cid + 10), "provider_exemplar_nr": str(exemplar),
                           "condition": "exp", "eligible_image": "True", "common_to_four": "True",
                           "failure_reasons": "", **{key: concept[key] for key in
                               ("role", "outer_fold", "development_block", "audit_block")}})
            provider.append({"image_path": path, "condition": "exp", "things_category_nr": str(cid),
                             "category_nr": str(cid + 10), "exemplar_nr": str(exemplar)})
    return concepts, images, provider


class TrackedStream(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.reads = []

    def read(self, size=-1):
        self.reads.append((self.tell(), size))
        if size < 0:
            raise AssertionError("unbounded source read")
        return super().read(size)


class DirectoryTests(unittest.TestCase):
    def test_only_two_bounded_source_reads_and_no_member_api(self):
        data = tiny_zip(comment=b"metadata-only comment")
        stream = TrackedStream(data)
        with mock.patch.object(zipfile.ZipFile, "open", side_effect=AssertionError("member opened")), \
             mock.patch.object(zipfile.ZipFile, "read", side_effect=AssertionError("member read")), \
             mock.patch.object(zipfile.ZipFile, "setpassword", side_effect=AssertionError("password supplied")):
            facts, entries = adapter.inspect_stream(stream, len(data))
        self.assertEqual(stream.reads, [(0, len(data)), (facts["central_directory_offset"], facts["central_directory_bytes"])])
        self.assertEqual(len(entries), 1)
        self.assertTrue(facts["zipfile_received_directory_only"])
        self.assertFalse(entries[0]["encrypted"])

    def test_full_tail_bound_and_original_member_offset(self):
        # An actual member body larger than the tail is never passed to ZipFile.
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("images_fmri/a/a.jpg", b"x" * 100000)
        data = stream.getvalue()
        tracked = TrackedStream(data)
        facts, _ = adapter.inspect_stream(tracked, len(data))
        self.assertEqual(tracked.reads[0], (len(data) - 65557, 65557))
        self.assertLessEqual(facts["total_requested_source_bytes"], adapter.TAIL_CAP + adapter.CD_CAP)

    def test_cap_refusal_before_zipfile_or_directory_read(self):
        data = change_eocd(tiny_zip(), 12, "<I", adapter.CD_CAP + 1)
        tracked = TrackedStream(data)
        with mock.patch.object(zipfile, "ZipFile", side_effect=AssertionError("parser reached")):
            with self.assertRaisesRegex(adapter.Refusal, "central_directory_cap"):
                adapter.inspect_stream(tracked, len(data))
        self.assertEqual(len(tracked.reads), 1)

    def test_entry_cap_and_multidisk_refuse(self):
        data = tiny_zip()
        over = change_eocd(change_eocd(data, 8, "<H", adapter.ENTRY_CAP + 1), 10, "<H", adapter.ENTRY_CAP + 1)
        with self.assertRaisesRegex(adapter.Refusal, "central_directory_cap"):
            inspect(over)
        with self.assertRaisesRegex(adapter.Refusal, "multidisk_zip"):
            inspect(change_eocd(data, 4, "<H", 1))

    def test_zip64_eocd_and_extra_refuse(self):
        with self.assertRaisesRegex(adapter.Refusal, "zip64_eocd"):
            inspect(change_eocd(tiny_zip(), 16, "<I", 0xFFFFFFFF))
        with self.assertRaisesRegex(adapter.Refusal, "zip64_entry"):
            inspect(tiny_zip(extra=struct.pack("<HH", 1, 0)))

    def test_zip64_locator_refuses(self):
        data = tiny_zip()
        pos = eocd_position(data)
        data = data[:pos] + b"PK\x06\x07" + b"\0" * 16 + data[pos:]
        with self.assertRaisesRegex(adapter.Refusal, "zip64_locator"):
            inspect(data)

    def test_trailing_bytes_and_unknown_directory_gap_refuse(self):
        with self.assertRaisesRegex(adapter.Refusal, "missing_classic_eocd"):
            inspect(tiny_zip() + b"trailer")
        data = tiny_zip()
        pos = eocd_position(data)
        with self.assertRaisesRegex(adapter.Refusal, "unknown_directory_layout"):
            inspect(data[:pos] + b"gap" + data[pos:])

    def test_encryption_aes_and_compression_are_metadata_only(self):
        extra = struct.pack("<HHH2sBH", 0x9901, 7, 2, b"AE", 3, 8)
        data = bytearray(tiny_zip(extra=extra))
        central = data.index(b"PK\x01\x02")
        struct.pack_into("<H", data, central + 8, 1)
        struct.pack_into("<H", data, central + 10, 99)
        _, entries = inspect(bytes(data))
        entry = entries[0]
        self.assertTrue(entry["encrypted"])
        self.assertEqual(entry["compression_method"], 99)
        self.assertEqual(entry["extra_field_ids"], [0x9901])
        self.assertEqual(entry["aes"], {"vendor_version": 2, "vendor_is_AE": True, "strength_code": 3, "actual_compression_method": 8})

    def test_malformed_extra_and_unsafe_names_refuse(self):
        with self.assertRaisesRegex(adapter.Refusal, "truncated_extra_value"):
            adapter.extra_facts(struct.pack("<HH", 0x1234, 10) + b"x")
        for name in ("../a.jpg", "/a.jpg", "C:/a.jpg", "images\\a.jpg", "./images/a.jpg"):
            with self.subTest(name=name), self.assertRaisesRegex(adapter.Refusal, "unsafe_member_name"):
                inspect(tiny_zip((name,)))

    def test_exact_duplicate_missing_and_unadopted_prefix(self):
        selected = [{"feature_row_zero_based": 0, "provider_image_path": "images_fmri/acorn/acorn_01b.jpg"}]
        _, entries = inspect(tiny_zip())
        self.assertTrue(adapter.match_selection(selected, entries)[0])
        self.assertFalse(adapter.match_selection(selected, entries + entries)[0])
        self.assertEqual(selected[0]["member_match_count"], 2)
        entries[0]["name"] = "archive/" + entries[0]["name"]
        self.assertFalse(adapter.match_selection(selected, entries)[0])
        self.assertEqual(selected[0]["member_match_count"], 0)


class SelectionTests(unittest.TestCase):
    def test_fixed_numeric_concepts_exact_utf8_image_order(self):
        concepts, images, provider = identities()
        expected = adapter.select_development_images(concepts, images, provider, count=2)
        self.assertEqual([row["things_category_nr"] for row in expected], [2, 9])
        self.assertTrue(all(row["image_name"].endswith("_01b.jpg") for row in expected))
        self.assertEqual(expected, adapter.select_development_images(concepts[::-1], images[::-1], provider[::-1], count=2))

    def test_role_inheritance_conflict_refuses(self):
        concepts, images, provider = identities()
        images[0]["role"] = "sealed_audit"
        with self.assertRaisesRegex(adapter.Refusal, "inherited_role_mismatch"):
            adapter.select_development_images(concepts, images, provider, count=2)

    def test_exact_provider_identifier_mismatch_refuses(self):
        concepts, images, provider = identities()
        provider[0]["exemplar_nr"] = "7"
        with self.assertRaisesRegex(adapter.Refusal, "provider_numeric_identity"):
            adapter.select_development_images(concepts, images, provider, count=2)

    def test_eligible_selection_does_not_replace_first_concept(self):
        concepts, images, provider = identities()
        for row in images:
            if row["things_category_nr"] == "2":
                row["eligible_image"] = "False"
        with self.assertRaisesRegex(adapter.Refusal, "qualified_count_changed"):
            adapter.select_development_images(concepts, images, provider, count=2)

    def test_duplicate_provider_path_refuses(self):
        concepts, images, provider = identities()
        with self.assertRaisesRegex(adapter.Refusal, "duplicate_provider_path"):
            adapter.select_development_images(concepts, images, provider + provider[:1], count=2)


class PublicationTests(unittest.TestCase):
    def attempt(self, *, late_failure=False, missing=False):
        selected = adapter.select_development_images(*identities(), count=2)
        names = tuple(row["provider_image_path"] for row in selected)
        facts, entries = inspect(tiny_zip(names))
        if missing:
            entries = entries[:1]
        original = adapter.mark_complete

        def completion(report, complete):
            original(report, complete)
            if late_failure:
                raise adapter.Refusal("fixture_deadline_after_completion")

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "new_attempt"
            with mock.patch.object(adapter, "validate_manifest"), \
                 mock.patch.object(adapter, "load_selection", return_value=selected), \
                 mock.patch.object(adapter, "inspect_archive", return_value=(facts, entries)), \
                 mock.patch.object(adapter, "mark_complete", side_effect=completion):
                status = adapter.run_attempt(out, arm_deadline=False)
            report = json.loads((out / "archive_metadata_report.json").read_text())
            sizes = sum((out / name).stat().st_size for name in ("archive_metadata_report.json", "selected_images.csv"))
            return status, report, sizes

    def test_success_keeps_pixels_decryption_features_and_roles_closed(self):
        status, report, size = self.attempt()
        self.assertEqual(status, 0)
        self.assertEqual(report["status"], "completed")
        for field in ("member_streams_opened", "password_supplied", "decryption_selected", "image_pixels_decoded",
                      "model_features_computed", "audit_pixels_or_features_opened", "source_roles_complete"):
            self.assertFalse(report[field])
        self.assertLessEqual(size, adapter.OUTPUT_CAP)

    def test_late_failure_clears_both_completion_flags(self):
        status, report, _ = self.attempt(late_failure=True)
        self.assertEqual(status, 2)
        self.assertEqual(report["status"], "stopped")
        self.assertFalse(report["archive_inspection_complete"])
        self.assertFalse(report["selected_member_mapping_complete"])

    def test_missing_member_reports_unresolved_without_false_absence_claim(self):
        status, report, _ = self.attempt(missing=True)
        self.assertEqual(status, 2)
        self.assertEqual(report["status"], "unresolved")
        self.assertTrue(report["archive_inspection_complete"])
        self.assertFalse(report["selected_member_mapping_complete"])
        self.assertEqual(report["selected_members"][1]["match_count"], 0)
        self.assertIsNone(report["selected_members"][1]["member"])

    def test_combined_output_cap_is_applied_before_writing(self):
        report = adapter.base_report()
        report["too_large"] = "x" * adapter.OUTPUT_CAP
        with self.assertRaisesRegex(adapter.Refusal, "output_size_cap"):
            adapter.render_outputs(report, [])


if __name__ == "__main__":
    unittest.main()
