"""Source-free fixtures for exact retained scope, pilot reuse and partial stops."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import reconcile_released_events as core
import prepare_released_event_manifest as manifest
import reconcile_retained_events as full

fixture_spec = importlib.util.spec_from_file_location(
    "pilot_fixtures", Path(__file__).with_name("test_released_events.py"))
pilot_fixtures = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(pilot_fixtures)


def pointer_leaf(payload, path="unused"):
    digest = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
    return {"path": path, "mode": "120000", "type": "blob", "size": len(payload), "sha": digest}


def fixture_roster():
    return [{"participant": participant, "session": 1, "run": run, "run_order": run - 1,
             "presentations": 2, "first_glmsingle_trial_index_zero_based": 2 * (run - 1), "TR_count": 188}
            for participant, run in (("sub-01", 1), ("sub-01", 2), ("sub-01", 3), ("sub-02", 1))]


def fixture_manifest():
    roster = fixture_roster()
    entries = []
    for row in roster:
        reused = row["participant"] == "sub-01"
        spec = core.PILOT[row["run"] - 1] if reused else {
            "directory": "ab/CD", "bytes": 1000, "md5": "1" * 32}
        payload = manifest.pilot_pointer(spec)
        entries.append({**row, **spec, "source_path": manifest.event_path(row),
                        "pointer_blob_sha": pointer_leaf(payload)["sha"], "reuse_pilot": reused})
    return roster, {"complete": True, "errors": [], "events_commit": core.EVENT_COMMIT,
                    "release_commit": core.RELEASE_COMMIT, "fmriprep_commit": core.FMRIPREP_COMMIT,
                    "entries": entries}


def fake_fetch_factory(roster, fail_last=False):
    calls = []
    root = {"truncated": False, "tree": [
        {"path": subject, "type": "tree", "mode": "040000", "sha": str(i + 1) * 40}
        for i, subject in enumerate(manifest.COUNTS)]}
    contents = {}
    for i, subject in enumerate(manifest.COUNTS):
        leaves = []
        for row in roster:
            if row["participant"] != subject:
                continue
            spec = core.PILOT[row["run"] - 1] if subject == "sub-01" else {
                "directory": "ab/CD", "bytes": 1000, "md5": "1" * 32}
            payload = manifest.pilot_pointer(spec)
            path = manifest.event_path(row)
            leaves.append(pointer_leaf(payload, path.split("/", 1)[1]))
            contents[path] = payload
        contents[f"tree-{str(i + 1) * 40}"] = json.dumps({"truncated": False, "tree": leaves}).encode()

    def fetch(url, cap, deadline, counters):
        calls.append(url)
        if url.endswith(core.EVENT_COMMIT):
            return json.dumps(root).encode()
        if "?recursive=1" in url:
            return contents["tree-" + url.split("/trees/")[1].split("?")[0]]
        if fail_last:
            raise core.Refusal("fixture_metadata_denial")
        return contents[url.split(core.EVENT_COMMIT + "/")[1]]
    return fetch, calls


class ManifestTests(unittest.TestCase):
    def test_pinned_symlink_resolves_annex_identity(self):
        payload = manifest.pilot_pointer(core.PILOT[0])
        self.assertEqual(manifest.pointer_spec(payload, pointer_leaf(payload)),
                         {key: core.PILOT[0][key] for key in ("directory", "bytes", "md5")})

    def test_blob_change_and_nonpointer_refused(self):
        payload = manifest.pilot_pointer(core.PILOT[0])
        with self.assertRaises(core.Refusal) as result:
            manifest.pointer_spec(payload.replace(b"W5/2J", b"W5/2K"), pointer_leaf(payload))
        self.assertEqual(result.exception.detail["code"], "pinned_symlink_blob_mismatch")
        with self.assertRaises(core.Refusal) as result:
            manifest.pointer_spec(b"https://unexpected.invalid/event.tsv", pointer_leaf(b"https://unexpected.invalid/event.tsv"))
        self.assertEqual(result.exception.detail["code"], "unexpected_annex_pointer")

    def test_direct_event_blob_refused_before_fetch(self):
        row = fixture_roster()[0]
        path = manifest.event_path(row).split("/", 1)[1]
        leaf = pointer_leaf(manifest.pilot_pointer(core.PILOT[0]), path)
        leaf["mode"] = "100644"
        with self.assertRaises(core.Refusal) as result:
            manifest.retained_leaf({path: leaf}, row)
        self.assertEqual(result.exception.detail["code"], "retained_event_is_not_annex_symlink")

    def test_truncated_tree_and_missing_run_refused(self):
        with self.assertRaises(core.Refusal):
            manifest.tree_members(json.dumps({"truncated": True, "tree": []}).encode())
        with self.assertRaises(core.Refusal):
            manifest.retained_leaf({}, fixture_roster()[0])

    def test_exact_pilot_pointers_reused_without_gets(self):
        roster = fixture_roster()
        fetch, calls = fake_fetch_factory(roster)
        entries = manifest.resolve_manifest(roster, fetch, 999999, {})
        self.assertEqual(len(entries), 4)
        self.assertEqual(sum(row["reuse_pilot"] for row in entries), 3)
        raw_calls = [url for url in calls if "raw.githubusercontent.com" in url]
        self.assertEqual(len(raw_calls), 1)
        self.assertIn("/sub-02/", raw_calls[0])

    def test_partial_manifest_entries_survive_later_failure(self):
        roster = fixture_roster()
        fetch, calls = fake_fetch_factory(roster, fail_last=True)
        entries = []
        with self.assertRaises(core.Refusal):
            manifest.resolve_manifest(roster, fetch, 999999, {}, entries)
        self.assertEqual(len(entries), 3)
        self.assertTrue(all(row["reuse_pilot"] for row in entries))

    def test_manifest_requires_exact_scope_and_pilot_identity(self):
        roster, document = fixture_manifest()
        self.assertEqual(len(full.validate_manifest(document, roster)), 4)
        altered = copy.deepcopy(document)
        altered["entries"].pop()
        with self.assertRaises(core.Refusal) as result:
            full.validate_manifest(altered, roster)
        self.assertEqual(result.exception.detail["code"], "manifest_roster_mismatch")
        altered = copy.deepcopy(document)
        altered["entries"][0]["md5"] = "0" * 32
        with self.assertRaises(core.Refusal) as result:
            full.validate_manifest(altered, roster)
        self.assertEqual(result.exception.detail["code"], "pilot_annex_identity_changed")

    def test_manifest_fails_closed_on_partial_or_oversized_selection(self):
        roster, document = fixture_manifest()
        document["complete"] = False
        with self.assertRaises(core.Refusal):
            full.validate_manifest(document, roster)
        document["complete"] = True
        document["entries"][-1]["bytes"] = manifest.EVENT_OBJECT_CAP + 1
        with self.assertRaises(core.Refusal) as result:
            full.validate_manifest(document, roster)
        self.assertEqual(result.exception.detail["code"], "event_object_exceeds_selected_budget")


class FullReconciliationTests(unittest.TestCase):
    def test_generalized_context_checks_actual_participant_session(self):
        expected, raw = pilot_fixtures.fixture()
        for row in raw:
            row["subject_id"] = "02"
            row["session_id"] = "ses-04"
        rows, mismatches, _ = core.reconcile(pilot_fixtures.tsv(raw), expected, 1, "sub-02", 4)
        self.assertEqual(mismatches, [])
        self.assertTrue(all(row["participant"] == "sub-02" and row["session"] == 4 for row in rows))
        self.assertNotIn("CLOSED_", repr(rows))
        with self.assertRaises(core.Refusal) as result:
            core.reconcile(pilot_fixtures.tsv(raw), expected, 1)
        self.assertEqual(result.exception.detail["code"], "wrong_event_participant")

    def test_pilot_reuse_never_invokes_download(self):
        _, document = fixture_manifest()
        key = ("sub-01", 1, 1)
        original_rows = [{"fixture": "already projected"}]
        with patch.object(core, "download", side_effect=AssertionError("pilot must not download")):
            rows, report = full.process_run(document["entries"][0], {}, {key: original_rows},
                                            {1: {"unknown_metadata": []}}, 999999)
        self.assertIs(rows, original_rows)
        self.assertEqual(report["status"], "matched")
        self.assertEqual(report["new_event_bytes"], 0)

    def test_new_run_mismatch_keeps_projection_but_cannot_complete(self):
        _, document = fixture_manifest()
        expected, raw = pilot_fixtures.fixture()
        for row in raw:
            row["subject_id"] = "02"
        raw[1]["image_path"] = raw[0]["image_path"]
        payload = pilot_fixtures.tsv(raw)
        with patch.object(core, "download", return_value=payload) as download:
            rows, report = full.process_run(document["entries"][-1], {("sub-02", 1, 1): expected}, {}, {}, 999999)
        self.assertEqual(download.call_count, 1)
        self.assertEqual(report["status"], "mismatch")
        self.assertEqual(len(rows), 2)
        self.assertNotIn("CLOSED_", repr((rows, report)))

    def test_stopped_publication_preserves_projection_without_complete_claim(self):
        roster, document = fixture_manifest()
        expected, raw = pilot_fixtures.fixture()
        for row in raw:
            row["subject_id"] = "02"
        raw[1]["image_path"] = raw[0]["image_path"]
        pilot_rows, _, unknowns = core.reconcile(pilot_fixtures.tsv(pilot_fixtures.fixture()[1]), expected, 1)
        prior = {( "sub-01", 1, run): pilot_rows for run in (1, 2, 3)}
        prior_reports = {run: {"unknown_metadata": unknowns} for run in (1, 2, 3)}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_parent = root / "manifests"
            manifest_path = manifest_parent / "attempt" / "annex_manifest.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(json.dumps(document))
            output_parent = root / "reconciliation"
            out = output_parent / "attempt"
            with patch.object(manifest, "OUTPUT_PARENT", manifest_parent), \
                 patch.object(full, "OUTPUT_PARENT", output_parent), \
                 patch.object(manifest, "load_roster", return_value=roster), \
                 patch.object(full, "load_presentations", return_value={("sub-02", 1, 1): expected}), \
                 patch.object(full, "load_pilot", return_value=(prior, prior_reports)), \
                 patch.object(core, "download", return_value=pilot_fixtures.tsv(raw)), \
                 patch.object(full.socket, "gethostname", return_value="fixture-compute"), \
                 patch.dict(full.os.environ, {"SLURM_JOB_ID": "fixture"}), \
                 patch.object(full.signal, "signal"), patch.object(full.signal, "alarm"), \
                 patch.object(sys, "argv", ["fixture", "--manifest", str(manifest_path), "--out", str(out)]):
                self.assertEqual(full.main(), 2)
            report = json.loads((out / "reconciliation_report.json").read_text())
            projection = (out / "event_metadata.csv").read_text()
            self.assertEqual(report["status"], "stopped")
            self.assertFalse(report["retained_roster_event_reconciliation_complete"])
            self.assertFalse(report["source_roles_complete"])
            self.assertEqual(report["projected_rows"], 8)
            self.assertNotIn("CLOSED_", projection + json.dumps(report))


if __name__ == "__main__":
    unittest.main()
