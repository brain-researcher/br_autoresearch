"""Synthetic fixtures only; coordinator runs these on compute before lookup."""

import io
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inspect_rat_map_nested_directories as lookup


def zip_bytes(entries):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, value in entries:
            archive.writestr(name, value)
    return stream.getvalue()


def directory(raw):
    fields = lookup.END.unpack(raw[-lookup.END.size:])
    return lookup.directory_entries(raw[fields[6]:fields[6] + fields[5]], fields[4])


class FakeRanges:
    def __init__(self, raw):
        self.raw = raw
        self.requests = []

    def get(self, first, last, purpose):
        self.requests.append((first, last, purpose))
        return self.raw[first:last + 1]


class FakeResponse:
    def __init__(self, status=206, content_range=None):
        self.status = status
        self.code = status
        self.msg = "Fixture response"
        self.headers = {"Content-Range": content_range or f"bytes 10-12/{lookup.ARCHIVE_BYTES}",
                        "Content-Length": "3"}
        self.reads = 0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False

    def info(self):
        return self.headers

    def close(self):
        self.closed = True

    def read(self, count):
        self.reads += 1
        return b"abc"[:count]


class NestedDirectoryTests(unittest.TestCase):
    def test_stored_nested_ranges_never_overlap_inner_payload(self):
        inner = zip_bytes([("maps/elecCoords_ratHD.mat", b"FORBIDDEN_MAP_VALUES"),
                           ("Rat5_conversion.py", b"FORBIDDEN_CODE_BODY")])
        outer = zip_bytes([("code/inside.zip", inner)])
        entry = directory(outer)[0]
        reader = FakeRanges(outer)
        result = lookup.inspect_nested(entry, reader)
        self.assertTrue(result["directory_inspected"])
        self.assertEqual(result["exact_rat_map_members"][0]["member_name"],
                         "maps/elecCoords_ratHD.mat")
        self.assertEqual(len(result["documentary_filename_candidates"]), 1)
        with zipfile.ZipFile(io.BytesIO(inner)) as archive:
            for item in archive.infolist():
                local = lookup.LOCAL.unpack_from(inner, item.header_offset)
                first = result["outer_data_start"] + item.header_offset + lookup.LOCAL.size + local[9] + local[10]
                last = first + item.compress_size - 1
                self.assertTrue(all(b < first or a > last for a, b, _ in reader.requests))
        self.assertEqual([x[2] for x in reader.requests], [
            "nested_container_local_header", "nested_container_local_name_extra",
            "nested_EOCD_only", "nested_central_directory_only"])

    def test_deflated_or_encrypted_container_reads_no_member_bytes(self):
        raw = zip_bytes([("inside.zip", zip_bytes([("elecCoords_ratHD.mat", b"values")]))])
        entry = directory(raw)[0]
        for altered in (dict(entry, method=8), dict(entry, flags=1)):
            reader = FakeRanges(raw)
            result = lookup.inspect_nested(altered, reader)
            self.assertFalse(result["directory_inspected"])
            self.assertIn("unresolved_reason", result)
            self.assertEqual(reader.requests, [])

    def test_mismatched_local_name_stops_before_nested_directory(self):
        raw = zip_bytes([("inside.zip", zip_bytes([("a.txt", b"body")]))])
        entry = dict(directory(raw)[0], member_name="renamed.zip")
        reader = FakeRanges(raw)
        result = lookup.inspect_nested(entry, reader)
        self.assertFalse(result["directory_inspected"])
        self.assertIn("name mismatch", result["unresolved_reason"])
        self.assertEqual(len(reader.requests), 2)

    def test_comment_zip64_and_wrong_extent_fail_without_wider_tail(self):
        good = (b"PK\x05\x06", 0, 0, 1, 1, 50, 100, 0)
        for index, value in ((7, 1), (4, 0xFFFF), (5, 0xFFFFFFFF), (6, 101)):
            fields = list(good)
            fields[index] = value
            with self.assertRaises(lookup.LookupStopped):
                lookup.directory_from_eocd(lookup.END.pack(*fields), 172)
        with self.assertRaises(lookup.LookupStopped):
            lookup.directory_entries(b"PK", 1)

    def test_nonexact_http_response_is_rejected_before_body_read(self):
        for response in (FakeResponse(status=200), FakeResponse(content_range="bytes 10-12/999")):
            reader = lookup.ExactRanges([], 3, time.monotonic() + 10)
            with patch.object(reader.opener, "open", return_value=response):
                with self.assertRaises(lookup.LookupStopped):
                    reader.get(10, 12, "fixture")
            self.assertEqual(response.reads, 0)

    def test_redirects_are_closed_without_reading_or_following(self):
        for status in (301, 302, 303, 307, 308):
            with self.subTest(status=status):
                requests = []
                reader = lookup.ExactRanges(requests, 3, time.monotonic() + 10)
                response = FakeResponse(status=status)
                response.headers["Location"] = lookup.ARCHIVE_URL
                with patch.object(response, "read", side_effect=AssertionError(
                        "redirect body must never be read")) as body_read:
                    with patch.object(lookup.urllib.request.HTTPSHandler, "https_open",
                                      return_value=response) as opened:
                        with self.assertRaisesRegex(lookup.LookupStopped, f"HTTP {status}"):
                            reader.get(10, 12, "redirect_fixture")
                        self.assertEqual(opened.call_count, 1)
                        self.assertEqual(opened.call_args.args[0].full_url, lookup.ARCHIVE_URL)
                    body_read.assert_not_called()
                self.assertTrue(response.closed)
                self.assertEqual(requests[0]["response_status"], status)
                self.assertEqual(requests[0]["body_bytes_read"], 0)

    def test_exact_http_read_and_budget_stop_prevent_second_request(self):
        requests = []
        reader = lookup.ExactRanges(requests, 3, time.monotonic() + 10)
        response = FakeResponse()
        with patch.object(reader.opener, "open", return_value=response) as opened:
            self.assertEqual(reader.get(10, 12, "fixture"), b"abc")
            with self.assertRaises(lookup.LookupStopped):
                reader.get(13, 13, "over_budget")
            self.assertEqual(opened.call_count, 1)
        self.assertEqual(requests[0]["body_bytes_read"], 3)


if __name__ == "__main__":
    unittest.main()
