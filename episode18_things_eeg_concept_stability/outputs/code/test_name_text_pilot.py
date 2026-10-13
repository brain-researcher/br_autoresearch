"""Fixed selection and prerequisite regressions; no model/prompt payloads."""
import copy
import unittest

import run_name_text_pilot as pilot


class NameTextPilotTests(unittest.TestCase):
    def test_first64_selection_uses_ids_only_and_keeps_prepared_text_verbatim(self):
        rows = [{"uniqueID": f"fixture_{index:04d}", "name_prompt": f" verbatim fixture{index} ",
                 "provider_Word_fallback": "0"} for index in reversed(range(1854))]
        selected = pilot.select_first64(rows)
        self.assertEqual([row["uniqueID"] for row in selected], [f"fixture_{i:04d}" for i in range(64)])
        self.assertEqual(selected[0]["name_prompt"], " verbatim fixture0 ")
        self.assertEqual(len(rows), 1854)

    def test_missing_duplicate_or_empty_prepared_names_stop(self):
        rows = [{"uniqueID": f"fixture_{index:04d}", "name_prompt": "fixture", "provider_Word_fallback": "0"}
                for index in range(1854)]
        for case in (rows[:-1], rows + [rows[-1]]):
            with self.assertRaisesRegex(ValueError, "all1854"):
                pilot.select_first64(case)
        duplicate = copy.deepcopy(rows)
        duplicate[-1]["uniqueID"] = duplicate[0]["uniqueID"]
        with self.assertRaisesRegex(ValueError, "unique source"):
            pilot.select_first64(duplicate)
        rows[0]["name_prompt"] = " "
        with self.assertRaisesRegex(ValueError, "complete prepared"):
            pilot.select_first64(rows)

    def test_version_only_or_partial_asset_report_cannot_open_text_inference(self):
        with self.assertRaisesRegex(ValueError, "Completed offline"):
            pilot.qualified_clip({}, {}, {"status": "runtime_versions_qualified"}, "fixture")
        with self.assertRaisesRegex(ValueError, "whole selected"):
            pilot.qualified_clip({}, {}, {"status": "asset_runtime_qualification_completed",
                                         "asset_runtime_qualified": True, "load_results": {}, "files": {}}, "fixture")


if __name__ == "__main__":
    unittest.main()
