import contextlib
import importlib.machinery
import importlib.util
import io
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bin" / "data-cleanup"
LOADER = importlib.machinery.SourceFileLoader("data_cleanup", str(SCRIPT))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
data_cleanup = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(data_cleanup)


class DataCleanupTests(unittest.TestCase):
    def test_completed_report(self):
        report = data_cleanup.load_report()
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["completed_on"], "2026-09-24")
        self.assertEqual(report["estimated_reclaimed_bytes"], 82_027_835_392)

    def test_fixed_record_list(self):
        records = data_cleanup.load_report()["cleanup_records"]
        self.assertEqual(
            [item["logical_id"] for item in records],
            [
                "scratch_things_eeg1_ingest",
                "scratch_ep05_payload",
                "personal_link_001201_full",
                "personal_link_001201_sentinels",
                "personal_epic_ptir_coating",
            ],
        )
        self.assertTrue(
            all(
                item["result"]
                == "redundant_source_removed_durable_copy_retained"
                for item in records
            )
        )

    def test_show_selects_fixed_record(self):
        records = data_cleanup.load_report()["cleanup_records"]
        record = data_cleanup.record_by_id(records, "scratch_ep05_payload")
        self.assertEqual(
            record["durable_path"],
            "/oak/stanford/groups/russpold/data/br_autoresearch_data/"
            "sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source",
        )
        with self.assertRaises(SystemExit):
            data_cleanup.record_by_id(records, "missing")

    def test_only_list_and_show_are_accepted(self):
        parser = data_cleanup.build_parser()
        self.assertEqual(parser.parse_args(["list"]).command, "list")
        self.assertEqual(
            parser.parse_args(["show", "scratch_things_eeg1_ingest"]).command,
            "show",
        )
        with contextlib.redirect_stderr(io.StringIO()):
            for command in ("plan", "prepare-receipt", "move", "delete"):
                with self.subTest(command=command), self.assertRaises(SystemExit):
                    parser.parse_args([command, "scratch_things_eeg1_ingest"])

    def test_user_supplied_paths_are_rejected(self):
        parser = data_cleanup.build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        "show",
                        "scratch_things_eeg1_ingest",
                        "--source",
                        "/tmp/other",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
