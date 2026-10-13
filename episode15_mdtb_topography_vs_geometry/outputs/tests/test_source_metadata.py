"""Focused fixtures for design joins and the metadata-only source boundary."""
from collections import Counter
import csv
import io
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import join_source_metadata as metadata
from inspect_registration_headers import nifti_header


def image_header(slope=2.0, intercept=-3.0):
    raw = bytearray(348)
    struct.pack_into("<i", raw, 0, 348)
    struct.pack_into("<8h", raw, 40, 3, 2, 3, 4, 1, 1, 1, 1)
    struct.pack_into("<h", raw, 70, 16)
    struct.pack_into("<8f", raw, 76, 1, 2, 2, 3, 1, 1, 1, 1)
    struct.pack_into("<f", raw, 108, 352)
    struct.pack_into("<2f", raw, 112, slope, intercept)
    raw[123] = 2
    struct.pack_into("<h", raw, 254, 2)
    for offset, row in zip((280, 296, 312), ((2, 0, 0, 0), (0, 2, 0, 0), (0, 0, 3, 0))):
        struct.pack_into("<4f", raw, offset, *row)
    raw[344:348] = b"n+1\0"
    return bytes(raw)


def design_fixture(session):
    shared = ["rest"] + [f"Shared{i:02d}" for i in range(13)]
    only_a = [f"AOnly{i:02d}" for i in range(15)]
    only_b = [condition for conditions in metadata.DOMAINS.values() for condition in conditions]
    conditions = shared + (only_a if session == "ses-s1" else only_b)
    rows = []
    for run in range(1, 17):
        # Instruction reg_id 0 is a real member and must not be lost or shifted.
        for reg_id in range(17):
            rows.append(dict(run=run, task_name="Instruct", cond_name="Instruct",
                             reg_id=reg_id, instruction=1, common=0, task_code="instrct",
                             cond_code="", half=1 if run <= 8 else 2))
        for reg_id, condition in enumerate(conditions, 17):
            rows.append(dict(run=run, task_name=condition, cond_name=condition,
                             reg_id=reg_id, instruction=0, common=int(condition in shared),
                             task_code=condition.lower(), cond_code="", half=1 if run <= 8 else 2))
    return rows


def encode_table(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=["run", "task_name", "cond_name", "reg_id",
                                              "instruction", "common", "task_code", "cond_code", "half"],
                            delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


class HeaderOnlyStream(io.BytesIO):
    def read(self, count=-1):
        if count != 348 or self.tell() != 0:
            raise AssertionError("Attempted to consume anything beyond the first NIfTI header")
        return super().read(count)


class MetadataArchive:
    """All image bodies are inaccessible; only one exact A header can be opened."""
    def __init__(self):
        self.payloads, self.opened = {}, []
        for session in metadata.SESSIONS:
            rows = design_fixture(session)
            name = f"sub-02/func/{session}/sub-02_{session}_reginfo.tsv"
            self.payloads[name] = encode_table(rows)
            for row in rows:
                self.payloads[metadata.beta_member("sub-02", session, row["run"], row["reg_id"])] = None
        self.header_member = metadata.beta_member("sub-02", "ses-s1", 1, 17)

    def infolist(self):
        return [SimpleNamespace(filename=name) for name in self.payloads]

    def getinfo(self, name):
        return SimpleNamespace(file_size=len(self.payloads[name]))

    def open(self, name):
        self.opened.append(name)
        if name == self.header_member:
            return HeaderOnlyStream(image_header() + b"CLOSED_VOXELS")
        if self.payloads[name] is None:
            raise AssertionError(f"Forbidden image stream opened: {name}")
        return io.BytesIO(self.payloads[name])


class SourceMetadataTests(unittest.TestCase):
    def test_all_design_members_partitions_and_no_task_b_images(self):
        archive = MetadataArchive()
        records, summaries, design, selected = metadata.inspect_participant(archive, "sub-02")
        self.assertEqual(len(records["ses-s1"]), 736)
        self.assertEqual(len(records["ses-s2"]), 784)
        self.assertEqual(len(summaries["ses-s1"]), 29)
        self.assertEqual(len(design["Task_B_only_conditions"]), 18)
        self.assertEqual(len(design["shared_conditions_postdecision_Task_B_diagnostic_only"]), 14)
        self.assertEqual(len(archive.opened), 3)
        self.assertFalse(any("ses-s2" in name and name.endswith(".nii") for name in archive.opened))
        self.assertEqual(selected["member"], archive.header_member)
        self.assertEqual(selected["header"]["scl_slope"], 2.0)
        self.assertTrue(records["ses-s1"][0]["beta_member"].endswith("run-01_reg-00_beta.nii"))
        for row in records["ses-s1"] + records["ses-s2"]:
            self.assertEqual(row["half"], 1 if row["run"] <= 8 else 2)
            self.assertEqual(row["geometry_partition"], (row["run"] - 1) // 4 + 1)
        self.assertAlmostEqual(sum(item["numerator"] / item["denominator"]
                                   for item in design["condition_weights"].values()), 1.0)

    def test_missing_task_b_member_detected_without_opening_an_image(self):
        archive = MetadataArchive()
        del archive.payloads[metadata.beta_member("sub-02", "ses-s2", 16, 48)]
        with self.assertRaisesRegex(ValueError, "missing, extra or duplicate"):
            metadata.inspect_participant(archive, "sub-02")
        self.assertTrue(all(name.endswith("_reginfo.tsv") for name in archive.opened))

    def test_duplicate_zero_regressor_is_not_silently_deduplicated(self):
        rows = design_fixture("ses-s1")
        rows.append(dict(rows[0]))
        with self.assertRaisesRegex(ValueError, "duplicate run/reg_id"):
            metadata.parse_reginfo(encode_table(rows), "sub-02", "ses-s1")

    def test_instruction_semantics_and_source_half_must_agree(self):
        rows = design_fixture("ses-s1")
        rows[0]["instruction"] = 0
        with self.assertRaisesRegex(ValueError, "instruction flag/code"):
            metadata.parse_reginfo(encode_table(rows), "sub-02", "ses-s1")
        rows = design_fixture("ses-s1")
        rows[-1]["half"] = 1
        with self.assertRaisesRegex(ValueError, "source half disagrees"):
            metadata.parse_reginfo(encode_table(rows), "sub-02", "ses-s1")

    def test_no_successful_subset_for_missing_condition_run(self):
        records = metadata.parse_reginfo(encode_table(design_fixture("ses-s1")), "sub-02", "ses-s1")
        records = [row for row in records if not (row["cond_name"] == "AOnly00" and row["run"] == 16)]
        counts = Counter(row["beta_member"] for row in records)
        with self.assertRaisesRegex(ValueError, "condition lacks"):
            metadata.validate_session(records, counts, len(records))

    def test_spatial_condition_typo_is_not_aliased(self):
        archive = MetadataArchive()
        _, summaries, _, _ = metadata.inspect_participant(archive, "sub-02")
        b = summaries["ses-s2"]
        b["SpatialMapDiff"] = b.pop("SpatialMedDiff")
        with self.assertRaisesRegex(ValueError, "B-only condition labels differ"):
            metadata.validate_task_sets(summaries["ses-s1"], b)

    def test_duplicate_zip_member_rejected(self):
        records = metadata.parse_reginfo(encode_table(design_fixture("ses-s1")), "sub-02", "ses-s1")
        counts = Counter(row["beta_member"] for row in records)
        counts[records[0]["beta_member"]] = 2
        with self.assertRaisesRegex(ValueError, "missing, extra or duplicate"):
            metadata.validate_session(records, counts, 736)

    def test_demographic_missingness_is_aggregate_not_imputed_or_exported(self):
        payload = ("participant_id\tsex\tage\tnative_english\tses-rest\n"
                   "sub-02\tF\t24\t1\t1\nsub-04\tn/a\tn/a\t0\t0\n").encode()
        ids, summary = metadata.participant_ids_and_completeness(payload, expected_count=2)
        self.assertEqual(ids, ["sub-02", "sub-04"])
        self.assertFalse(summary["age_available_for_future_role_assignment"])
        self.assertEqual(summary["missing_values_by_field"]["age"], 1)
        self.assertEqual(summary["missing_values_by_field"]["sex"], 1)
        self.assertFalse(summary["person_level_demographic_values_exported"])
        self.assertNotIn("sub-02", repr(summary))

    def test_header_reads_348_bytes_and_preserves_scaling_semantics(self):
        stream = HeaderOnlyStream(image_header(slope=0, intercept=float("nan")) + b"CLOSED_VOXELS")
        header = nifti_header(stream)
        self.assertEqual(stream.tell(), 348)
        self.assertEqual(header["shape"], [2, 3, 4])
        self.assertTrue(header["scaling_disabled"])
        self.assertIsNone(header["scl_inter"])
        self.assertFalse(header["voxel_elements_read"])


if __name__ == "__main__":
    unittest.main()
