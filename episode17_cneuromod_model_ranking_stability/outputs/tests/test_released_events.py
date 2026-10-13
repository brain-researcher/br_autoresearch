"""Source-free behavioral-boundary and reconciliation fixtures; compute only."""

import csv
import hashlib
import importlib.util
import io
from pathlib import Path
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError


SPEC = importlib.util.spec_from_file_location(
    "released_events", Path(__file__).resolve().parents[1] / "tools/reconcile_released_events.py")
events = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(events)


def fixture():
    expected, released = [], []
    for index, (name, concept, onset) in enumerate((
            ("girl_02s.jpg", "girl", "3.0"),
            ("coffee_pot_02s.jpg", "coffee_pot", "7.50"))):
        expected.append({
            "provider_image_path": f"images_fmri/{concept}/{name}",
            "image_name": name, "things_category_nr": str(677 if index == 0 else 381),
            "provider_category_nr": str(272 if index == 0 else 158),
            "provider_exemplar_nr": "2", "run_order": "0",
            "glmsingle_trial_index_zero_based": str(index),
            "onset_TR_after_two_volume_removal": str(0 if index == 0 else 3),
            "image_presentation_number": "1", "previous_same_image_index_gap": "",
        })
        released.append({
            "subject_id": "01", "session_id": "01", "run_id": "1",
            "TrialNumber": str(index + 1), "order": str(index),
            "image_path": f"stimuli/images_fmri/{concept}/{name}",
            "things_category_nr": expected[-1]["things_category_nr"],
            "category_nr": expected[-1]["provider_category_nr"], "exemplar_nr": "2",
            "onset": onset, "duration": "2.98", "repetition": "1",
            "atypical": "False", "not_for_memory": "False", "delay_days": "0.0",
            "delay_seconds": "0.0", "trials_since_lastrep": "n/a",
            "response": "CLOSED_RESPONSE_SENTINEL", "response_time": "CLOSED_RT_SENTINEL",
            "error": "CLOSED_ACCURACY_SENTINEL", "audit_model_score": "CLOSED_SCORE_SENTINEL",
            "atypical_log": "CLOSED_FREETEXT_SENTINEL",
        })
    return expected, released


def tsv(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


class ReconciliationTests(unittest.TestCase):
    def test_exact_identity_and_closed_columns_projected_out(self):
        expected, rows = fixture()
        projected, mismatches, unknowns = events.reconcile(tsv(rows), expected, 1)
        self.assertEqual(mismatches, [])
        self.assertEqual(len(projected), 2)
        self.assertTrue(all(row["identity_and_onset_match"] for row in projected))
        self.assertTrue(all(set(row) == set(events.OUTPUT_FIELDS) for row in projected))
        self.assertNotIn("CLOSED_", repr((projected, mismatches, unknowns)))
        self.assertEqual(projected[1]["onset_TR_after_two_volume_removal"], 3)

    def test_wrong_run_is_refused_instead_of_filtered(self):
        expected, rows = fixture()
        rows[1]["run_id"] = "2"
        with self.assertRaises(events.Refusal) as result:
            events.reconcile(tsv(rows), expected, 1)
        self.assertEqual(result.exception.detail, {"code": "wrong_event_run", "row_zero_based": 1})
        self.assertNotIn("CLOSED_", str(result.exception))

    def test_missing_presentation_is_not_silently_dropped(self):
        expected, rows = fixture()
        with self.assertRaises(events.Refusal) as result:
            events.reconcile(tsv(rows[:1]), expected, 1)
        self.assertEqual(result.exception.detail["code"], "event_presentation_count_mismatch")

    def test_missing_core_column_is_refused(self):
        expected, rows = fixture()
        for row in rows:
            del row["onset"]
        with self.assertRaises(events.Refusal) as result:
            events.reconcile(tsv(rows), expected, 1)
        self.assertEqual(result.exception.detail["code"], "missing_core_event_column")

    def test_missing_optional_schedule_fields_stay_unknown(self):
        expected, rows = fixture()
        for row in rows:
            for field in events.OPTIONAL_FIELDS:
                del row[field]
        projected, mismatches, unknowns = events.reconcile(tsv(rows), expected, 1)
        self.assertEqual(mismatches, [])
        self.assertEqual(len(unknowns), 12)
        self.assertIsNone(projected[0]["source_repetition"])
        self.assertIsNone(projected[0]["atypical"])

    def test_released_and_retained_repetition_are_not_forced_equal(self):
        expected, rows = fixture()
        rows[1]["repetition"] = "3"
        projected, mismatches, _ = events.reconcile(tsv(rows), expected, 1)
        self.assertEqual(mismatches, [])
        self.assertEqual(projected[1]["source_repetition"], 3)
        self.assertEqual(projected[1]["retained_image_presentation_number"], 1)

    def test_exact_int_semantics_not_round_or_floor(self):
        self.assertEqual(events.provider_onset_tr("4.48"), 0)
        self.assertEqual(events.provider_onset_tr("1.50"), -1)
        self.assertEqual(events.provider_onset_tr("7.50"), 3)

    def test_trial_reordering_and_image_substitution_are_reported(self):
        expected, rows = fixture()
        rows[1]["TrialNumber"] = "1"
        rows[1]["order"] = "0"
        rows[1]["image_path"] = rows[0]["image_path"]
        _, mismatches, _ = events.reconcile(tsv(rows), expected, 1)
        fields = {item["field"] for item in mismatches}
        self.assertTrue({"TrialNumber", "order", "image_basename", "image_path_suffix"} <= fields)

    def test_no_offset_fitting_when_provider_tr_disagrees(self):
        expected, rows = fixture()
        rows[1]["onset"] = "8.97"
        _, mismatches, _ = events.reconcile(tsv(rows), expected, 1)
        self.assertIn({"row_zero_based": 1, "field": "onset_TR"}, mismatches)

    def test_invalid_allowed_field_does_not_echo_other_values(self):
        expected, rows = fixture()
        rows[0]["onset"] = "CLOSED_RESPONSE_SENTINEL"
        with self.assertRaises(events.Refusal) as result:
            events.reconcile(tsv(rows), expected, 1)
        self.assertEqual(result.exception.detail["code"], "invalid_numeric_metadata")
        self.assertNotIn("CLOSED_", str(result.exception))


class FakeResponse:
    status = 200

    def __init__(self, payload, declared):
        self.payload = payload
        self.headers = {"Content-Length": str(declared)}
        self.requested_bytes = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, count):
        self.requested_bytes = count
        return self.payload[:count]


class TransferTests(unittest.TestCase):
    def call(self, payload, expected_payload, declared=None):
        spec = {"directory": "W5/2J", "bytes": len(expected_payload),
                "md5": hashlib.md5(expected_payload).hexdigest()}
        response = FakeResponse(payload, len(expected_payload) if declared is None else declared)
        with patch.object(events, "build_opener") as opener:
            opener.return_value.open.return_value = response
            result = events.download(spec, time.monotonic() + 30)
            self.assertEqual(opener.return_value.open.call_count, 1)
        self.assertEqual(response.requested_bytes, len(expected_payload) + 1)
        return result

    def test_matching_annex_object(self):
        self.assertEqual(self.call(b"safe fixture", b"safe fixture"), b"safe fixture")

    def test_wrong_digest_same_length_refused(self):
        with self.assertRaises(events.Refusal) as result:
            self.call(b"BAD", b"OK!")
        self.assertEqual(result.exception.detail["code"], "annex_md5_mismatch")

    def test_oversize_and_short_objects_refused(self):
        for payload in (b"short", b"expected plus forbidden bytes"):
            with self.subTest(payload_length=len(payload)):
                with self.assertRaises(events.Refusal) as result:
                    self.call(payload, b"expected")
                self.assertEqual(result.exception.detail["code"], "payload_length_mismatch")

    def test_declared_size_refused_before_body(self):
        with self.assertRaises(events.Refusal) as result:
            self.call(b"expected", b"expected", declared=10000)
        self.assertEqual(result.exception.detail["code"], "content_length_mismatch")

    def test_every_redirect_rejected_without_body_read(self):
        class ForbiddenBody:
            closed = False

            def read(self, *args):
                raise AssertionError("redirect body must not be read")

            def close(self):
                self.closed = True

        for code in (301, 302, 303, 307, 308):
            with self.subTest(code=code):
                body = ForbiddenBody()
                with self.assertRaises(events.Refusal) as result:
                    getattr(events.NoRedirect(), f"http_error_{code}")(
                        None, body, code, "redirect", {"Location": "https://authentication.invalid/"})
                self.assertTrue(body.closed)
                self.assertEqual(result.exception.detail["code"], f"redirect_status_{code}")

    def test_http_denial_closed_without_read_or_retry(self):
        class ForbiddenBody:
            closed = False

            def read(self, *args):
                raise AssertionError("denial body must not be read")

            def close(self):
                self.closed = True

        body = ForbiddenBody()
        denied = HTTPError("https://sftp.conp.ca/fixture", 403, "Forbidden", {}, body)
        spec = {"directory": "W5/2J", "bytes": 3, "md5": hashlib.md5(b"OK!").hexdigest()}
        with patch.object(events, "build_opener") as opener:
            opener.return_value.open.side_effect = denied
            with self.assertRaises(events.Refusal) as result:
                events.download(spec, time.monotonic() + 30)
            self.assertEqual(opener.return_value.open.call_count, 1)
        self.assertTrue(body.closed)
        self.assertEqual(result.exception.detail["code"], "http_status_403")


if __name__ == "__main__":
    unittest.main()
