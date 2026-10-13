"""Small text/path fixtures only; execute in the coordinated Slurm allocation."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import inspect_development_acquisition as acquisition


def header_fixture(extra=False):
    names = ["Fp1", "Fz", "F3", "F7", "FT9", "FC5", "FC1", "C3", "T7", "TP9", "CP5", "CP1",
             "Pz", "P3", "P7", "O1", "Oz", "O2", "P4", "P8", "TP10", "CP6", "CP2", "C4", "T8",
             "FT10", "FC6", "FC2", "F4", "F8", "Fp2", "EOGU", "EOGD", "EOGL", "EOGR"]
    if extra:
        names += [str(index) for index in range(37, 65)]
    names += ["TRIG", "X", "Y", "Z"]
    channel_rows, setup_rows = [], []
    for index, name in enumerate(names, 1):
        auxiliary = name in {"TRIG", "X", "Y", "Z"}
        resolution = "0.298023" if auxiliary else "0.0488281"
        physical = 65 + ["TRIG", "X", "Y", "Z"].index(name) if auxiliary else index
        channel_rows.append(f"Ch{index}={name},,{resolution},µV")
        setup_rows.append(f"{index} {name} {physical} {resolution} µV DC {'280' if extra else '69'} Off")
    return (f"[Common Infos]\nNumberOfChannels={len(names)}\nSamplingInterval={'1000' if extra else '4000'}\n"
            "DataFormat=BINARY\nDataOrientation=MULTIPLEXED\nDataFile=never-open.eeg\nMarkerFile=never-open.vmrk\n"
            "[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n[Channel Infos]\n" + "\n".join(channel_rows)
            + "\n[Comment]\nBrainVision Recorder Professional - V. 1.23.0003\n"
            + acquisition.TABLE_COLUMNS + "\n" + "\n".join(setup_rows)
            + "\n\nAmplifier\nactiCHamp Base Unit (5002)\n\nS o f t w a r e  F i l t e r s\n===\nDisabled\n"
            + "\nImpedance [kOhm] at 00:00:00 :\nX: DO NOT REPORT IMPEDANCE\n")


def admission_fixture():
    root, headers, files = Path("/fixture-development-view"), [], []
    for index, person in enumerate(sorted(acquisition.DEVELOPMENT_PEOPLE)):
        for run in range(1, 12 if index < 5 else 11):
            member = f"{person}/ses-01/eeg/{person}_ses-01_task-reachingandgrasping_run-{run:04d}_eeg.vhdr"
            headers.append({"member": member, "participant": person})
            files.append({"source": "self_paced", "role": "development", "member": member,
                          "destination": str(root / "self_paced/development" / member), "bytes": 5000})
    return ({"status": "development_header_inspection_complete", "header_count": 155, "headers": headers},
            {"status": "development_view_materialized", "files": files}, root)


class DevelopmentAcquisitionTests(unittest.TestCase):
    def test_both_recorded_layouts_preserve_physical_channels_cutoffs_and_disabled_filters(self):
        for extra, count, cutoff in ((False, 39, "69"), (True, 67, "280")):
            with self.subTest(channel_count=count):
                text = header_fixture(extra)
                result = acquisition.parse_acquisition(text, acquisition.parse_header(text))
                self.assertEqual(result["unresolved"], [])
                self.assertEqual(len(result["channel_settings"]), count)
                self.assertEqual(result["software_filters"]["text"], "Disabled")
                auxiliary = result["channel_settings"][-4:]
                self.assertEqual([row["physical_channel"] for row in auxiliary], [65, 66, 67, 68])
                for row in auxiliary:
                    self.assertEqual((row["low_cutoff_seconds_token"], row["high_cutoff_hz_token"],
                                      row["notch_hz_token"], row["unit"]), ("DC", cutoff, "Off", "µV"))
                    self.assertIsNone(row["gradient_token"])
                    self.assertIsNone(row["offset_token"])
                self.assertNotIn("DO NOT REPORT IMPEDANCE", str(result))

    def test_missing_unsupported_or_inconsistent_settings_remain_unknown(self):
        original = header_fixture()
        old = acquisition.parse_header(original)
        for text in (original.split("[Comment]")[0], original.replace("Low Cutoff [s]", "Low Cutoff [Hz]"),
                     original.replace("1 Fp1 1 0.0488281", "1 Fp1 1 0.1")):
            with self.subTest(text=text[-40:]):
                result = acquisition.parse_acquisition(text, old)
                self.assertEqual(result["channel_settings"], [])
                self.assertTrue(result["unresolved"])
        enabled = acquisition.parse_acquisition(original.replace("\nDisabled\n", "\nEnabled\n"), old)
        self.assertIsNone(enabled["software_filters"])
        self.assertIn("software_filter_configuration_not_supported", enabled["unresolved"])

    def test_exact_scope_excludes_signal_files_and_refuses_path_substitution_or_duplicates(self):
        prior, view, root = admission_fixture()
        view["files"].append({"source": "self_paced", "member": "never-open.eeg", "role": "development"})
        admitted = acquisition.select_headers(prior, view, root)
        self.assertEqual(len(admitted), 155)
        self.assertTrue(all(path.suffix == ".vhdr" for path, _, _ in admitted))
        wrong = copy.deepcopy(view)
        wrong["files"][0]["destination"] = "/unapproved/other.vhdr"
        with self.assertRaisesRegex(ValueError, "fixed_materialized_header_path"):
            acquisition.select_headers(prior, wrong, root)
        wrong = copy.deepcopy(view)
        wrong["files"].append(wrong["files"][0])
        with self.assertRaisesRegex(ValueError, "exact_same_155"):
            acquisition.select_headers(prior, wrong, root)

    def test_matching_reports_cannot_admit_a_held_out_person_or_reduced_cohort(self):
        prior, view, root = admission_fixture()
        original = prior["headers"][0]["member"]
        replaced = original.replace("sub-02", "sub-01")
        prior["headers"][0].update(member=replaced, participant="sub-01")
        view["files"][0].update(member=replaced, destination=str(root / "self_paced/development" / replaced))
        with self.assertRaisesRegex(ValueError, "only_original_development"):
            acquisition.select_headers(prior, view, root)
        prior, view, root = admission_fixture()
        prior["headers"].pop()
        with self.assertRaisesRegex(ValueError, "exact_completed_155"):
            acquisition.select_headers(prior, view, root)


if __name__ == "__main__":
    unittest.main()
