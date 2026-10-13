"""Source-free tests for the four fixed EP17 anatomical-unit witnesses."""

import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import qualify_anatomical_units as stage

nib, np = stage.nib, stage.np


def make_header(units="mm", qcode=0, scode=2, delta=0):
    header = nib.Nifti1Header()
    header.set_data_shape((2, 3, 4))
    header.set_data_dtype(np.uint8)
    affine = np.diag([2., 2., 2., 1.])
    affine[:3, 3] = [-10., -20., -30.]
    header.set_qform(affine, code=qcode)
    affine[0, 3] += delta
    header.set_sform(affine, code=scode)
    header.set_xyzt_units(units)
    header["vox_offset"] = 352
    return header


def recorded(header):
    return {"shape": list(header.get_data_shape()), "datatype_code": int(header["datatype"]),
            "pixdim": header["pixdim"].tolist(), "qform_code": int(header["qform_code"]),
            "sform_code": int(header["sform_code"]),
            "quaternion_and_offset": [float(header[x]) for x in
                ("quatern_b", "quatern_c", "quatern_d", "qoffset_x", "qoffset_y", "qoffset_z")],
            "srow_xyz": [header[x].tolist() for x in ("srow_x", "srow_y", "srow_z")],
            "voxel_values_read": False}


def compressed(header, tail=b"opaque voxel bytes" * 100000):
    return gzip.compress(header.binaryblock + b"\0" * 4 + tail, mtime=0)


def transport_report():
    return {"network_requests": 0, "transport_bytes_read": 0, "accesses": []}


class Response(io.BytesIO):
    status = 200

    def __init__(self, body, url):
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body))}
        self.url = url
        self.read_calls = []

    def geturl(self):
        return self.url

    def read(self, size=-1):
        self.read_calls.append(size)
        return super().read(size)


class UnitWitnessTests(unittest.TestCase):
    def test_fixed_pinned_paths_pointer_identities_and_total(self):
        identities = [stage.identity(s) for s in stage.SUBJECTS]
        self.assertEqual(sum(x["bytes"] for x in identities), 28069)
        self.assertEqual([x["bytes"] for x in identities], [7602, 7127, 6862, 6478])
        for item in identities:
            self.assertIn("/ses-01/func/", item["source_path"])
            self.assertIn("_run-1_part-mag_space-T1w_desc-brain_mask.nii.gz", item["source_path"])
            self.assertEqual(item["url"], stage.BASE + item["directory"] + "/" + item["annex_key"] + "/" + item["annex_key"])
        with self.assertRaisesRegex(stage.Refusal, "unselected_witness"):
            stage.identity("sub-04")

    def test_header_only_decode_both_endiannesses_with_large_opaque_tail(self):
        real_factory = stage.zlib.decompressobj
        calls = []

        class LimitedDecoder:
            def __init__(self, *args, **kwargs):
                self.decoder = real_factory(*args, **kwargs)

            def decompress(self, body, max_length):
                out = self.decoder.decompress(body, max_length=max_length)
                calls.append((max_length, len(out)))
                return out

        header = make_header()
        for endian in ("<", ">"):
            selected = header if header.endianness == endian else header.as_byteswapped()
            with patch.object(stage.zlib, "decompressobj", LimitedDecoder):
                result = stage.decode_header(compressed(selected))
            self.assertEqual(result.get_data_shape(), (2, 3, 4))
            self.assertEqual(result.endianness, endian)
        self.assertEqual(calls, [(352, 352), (352, 352)])

    def test_explicit_mm_required_without_voxel_size_inference(self):
        for units in ("unknown", "meter", "micron"):
            with self.subTest(units=units), self.assertRaisesRegex(stage.Refusal, "witness_explicit_mm_required"):
                stage.decode_header(compressed(make_header(units=units)))
        header = make_header()
        header["vox_offset"] = 348
        with self.assertRaisesRegex(stage.Refusal, "witness_payload_overlaps_header"):
            stage.decode_header(compressed(header))
        with self.assertRaisesRegex(stage.Refusal, "short_witness_header"):
            stage.decode_header(gzip.compress(b"short"))

    def test_union_numeric_geometry_matches_without_reopening_source(self):
        header = make_header()
        saved = recorded(header)
        before = json.dumps(saved, sort_keys=True)
        result = stage.compare_geometry(header, saved)
        self.assertEqual(result["maximum_corner_difference_mm"], 0)
        self.assertEqual(result["inherited_spatial_units"], "mm")
        self.assertFalse(result["union_header_rewritten"])
        np.testing.assert_array_equal(result["witness_geometry"]["selected_affine_mm"], header.get_sform())
        self.assertEqual(json.dumps(saved, sort_keys=True), before)
        saved = recorded(make_header(delta=.01))
        with self.assertRaisesRegex(stage.Refusal, "witness_union_affine_mismatch"):
            stage.compare_geometry(header, saved)

    def test_uncoded_standard_frame_disagreeing_forms_and_shape_refuse(self):
        for header, reason in ((make_header(scode=0), "native_T1w_form_unresolved"),
                               (make_header(scode=4), "native_T1w_form_unresolved"),
                               (make_header(qcode=1, scode=2), "different_form_frames"),
                               (make_header(qcode=2, scode=2, delta=.01), "authenticated_affines_disagree")):
            with self.subTest(reason=reason), self.assertRaisesRegex(stage.Refusal, reason):
                stage.compare_geometry(header, recorded(header))
        saved = recorded(make_header())
        saved["shape"] = [3, 3, 4]
        with self.assertRaisesRegex(stage.Refusal, "witness_union_shape_mismatch"):
            stage.compare_geometry(make_header(), saved)

    def test_annex_authentication_requires_whole_exact_object(self):
        body = compressed(make_header(), b"payload")
        expected = {"bytes": len(body), "md5": hashlib.md5(body).hexdigest()}
        stage.verify_object(body, expected)
        with self.assertRaisesRegex(stage.Refusal, "witness_annex_length_mismatch"):
            stage.verify_object(body[:-1], expected)
        corrupt = body[:-1] + bytes([body[-1] ^ 1])
        with self.assertRaisesRegex(stage.Refusal, "witness_annex_digest_mismatch"):
            stage.verify_object(corrupt, expected)

    def test_single_exact_transport_and_truthful_byte_accounting(self):
        body = compressed(make_header(), b"payload")
        expected = {"url": "https://example.invalid/fixed", "bytes": len(body),
                    "md5": hashlib.md5(body).hexdigest()}
        response = Response(body, expected["url"])
        opener = Mock()
        opener.open.return_value = response
        report = transport_report()
        with patch.object(stage, "identity", return_value=expected), patch.object(stage, "build_opener", return_value=opener):
            actual, got = stage.fetch_object("sub-01", report, time.monotonic()+60)
        self.assertEqual(actual, body)
        self.assertEqual(got, expected)
        self.assertEqual(report["transport_bytes_read"], len(body))
        self.assertEqual(report["network_requests"], 1)
        self.assertTrue(report["accesses"][0]["verified"])
        self.assertIn("compressed_bytes_first_read_attempted_utc", report["accesses"][0])
        self.assertEqual(response.read_calls, [len(body)])
        self.assertEqual(opener.open.call_count, 1)
        self.assertLessEqual(opener.open.call_args.kwargs["timeout"], 15)

    def test_redirect_length_encoding_and_total_bound_refuse_before_body(self):
        with self.assertRaisesRegex(stage.Refusal, "redirect_refused"):
            stage.NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.invalid/other")
        body = b"object"
        expected = {"url": "https://example.invalid/fixed", "bytes": len(body), "md5": hashlib.md5(body).hexdigest()}
        for field, value in (("Content-Length", None), ("Content-Length", "9000"),
                             ("Content-Encoding", "gzip"), ("Transfer-Encoding", "chunked")):
            response = Response(body, expected["url"])
            if value is None:
                response.headers.pop(field)
            else:
                response.headers[field] = value
            opener = Mock()
            opener.open.return_value = response
            with patch.object(stage, "identity", return_value=expected), patch.object(stage, "build_opener", return_value=opener):
                with self.subTest(field=field, value=value), self.assertRaises(stage.Refusal):
                    stage.fetch_object("sub-01", transport_report(), time.monotonic()+60)
            self.assertEqual(response.read_calls, [])
        report = transport_report()
        report["transport_bytes_read"] = stage.TOTAL_CAP
        with patch.object(stage, "build_opener") as builder, self.assertRaisesRegex(stage.Refusal, "witness_transport_bound"):
            stage.fetch_object("sub-01", report, time.monotonic()+60)
        builder.assert_not_called()

    def test_terminal_success_and_late_deadline_clear_success(self):
        header = make_header()
        body = compressed(header)
        prerequisites = {s: {"mask_header": recorded(header)} for s in stage.SUBJECTS}

        def fake_fetch(subject, report, deadline):
            item = stage.identity(subject)
            report["network_requests"] += 1
            report["transport_bytes_read"] += item["bytes"]
            return body, item

        def late_finish(report, deadline):
            report.update(status="completed", anatomical_spatial_units_witness_complete=True)
            raise stage.Refusal("attempt_deadline")

        for late in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                with patch.object(stage.anatomy, "load_metadata", return_value=prerequisites), patch.object(stage, "fetch_object", side_effect=fake_fetch):
                    if late:
                        with patch.object(stage, "finish", side_effect=late_finish):
                            code = stage.run_attempt(out)
                    else:
                        code = stage.run_attempt(out)
                report = json.loads((out / "unit_witness_report.json").read_text())
                self.assertEqual(code, 2 if late else 0)
                self.assertEqual(report["anatomical_spatial_units_witness_complete"], not late)
                self.assertEqual(report["transport_bytes_read"], 28069)
                self.assertFalse(report["structural_voxel_arrays_decoded"])
                self.assertFalse(report["response_units_verified"])
                self.assertFalse(report["audit_opened"])

    def test_incomplete_roster_and_network_failure_cannot_complete(self):
        with self.assertRaisesRegex(stage.Refusal, "incomplete_witness_roster"):
            stage.finish({"participants": [], "network_requests": 0, "transport_bytes_read": 0}, time.monotonic()+5)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with patch.object(stage.anatomy, "load_metadata", return_value={}), patch.object(stage, "fetch_object", side_effect=stage.URLError("fixture")) as fetch:
                code = stage.run_attempt(out)
            report = json.loads((out / "unit_witness_report.json").read_text())
            self.assertEqual(code, 2)
            self.assertFalse(report["anatomical_spatial_units_witness_complete"])
            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(report["automatic_retries"], 0)
            self.assertEqual(report["errors"][0]["exception_type"], "URLError")


if __name__ == "__main__":
    unittest.main()
