"""Synthetic recovery check for the October 16 source-semantics checker.

The fixture is a file-format and known-answer test, not an observation: nine
synthetic controls with reward at sample 300, a 750-ms lickState window and a
causal 150-ms lick-rate trace like the release's continuous `lick` field.
Run only inside a Slurm allocation.
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import check_source_semantics as checker  # noqa: E402
import development_role_reader as reader  # noqa: E402

R = 300
KERNEL = (1 / 0.15) * np.exp(-np.arange(61) / 15.0)  # causal 150-ms rate kernel, like the release


def train(start, stop, rng):
    positions, t = [], start
    while t < stop:
        positions.append(t)
        t += 14 + int(rng.integers(-1, 2))
    return positions


def mouse_arrays(rng):
    trial, sesh = [], []
    for session in range(1, 9):
        codes = np.ones(100, dtype=np.int64) * 0
        codes[:10] = 1
        if session >= 4:
            codes[10:20] = 2
        rng.shuffle(codes)
        trial.extend(codes)
        sesh.extend([session] * 100)
    trial, sesh = np.array(trial), np.array(sesh)
    events = np.zeros((trial.size, checker.SAMPLES), dtype=bool)
    for row, (code, session) in enumerate(zip(trial, sesh)):
        candidates = list(np.nonzero(rng.random(checker.SAMPLES) < 0.002)[0])
        if code in (0, 2) and rng.random() < session / 10:
            candidates += train(R - 150 + int(rng.integers(20, 120)), R + 5, rng)
        if code in (0, 1):
            onset = R + int(rng.integers(5, 40))
            candidates += train(onset, min(onset + 200, checker.SAMPLES), rng)
        last = -100
        for t in sorted(candidates):
            if t - last >= 7 and t < checker.SAMPLES:
                events[row, t] = True
                last = t
    state = events[:, R - 75:R].any(axis=1).astype(np.float64)
    lick = np.stack([np.convolve(row.astype(float), KERNEL)[:checker.SAMPLES] for row in events]).astype(np.float32)
    return {"trialID": trial.reshape(1, -1).astype(np.int8), "seshID": sesh.reshape(1, -1).astype(np.uint8),
            "lickState": state.reshape(1, -1).astype(np.uint8), "lick": lick}


def fixture(root):
    root.mkdir()
    rng = np.random.default_rng(16)
    entries = []
    for index in reader.DEVELOPMENT_RECORDS:
        name = f"source_record_{index:02d}"
        (root / name).mkdir()
        fields = {}
        for field, value in mouse_arrays(rng).items():
            path = root / name / f"{field}.npy"
            np.save(path, value, allow_pickle=False)
            fields[field] = {"relative_path": f"{name}/{field}.npy", "dtype": str(value.dtype),
                             "shape": list(value.shape), "npy_bytes": path.stat().st_size}
        entries.append({"opaque_animal_id": name, "source_index_1_based": index,
                        "cohort_id": "development_control", "fields": fields})
    manifest = {"schema_version": "ep02.dudman.role_materialization.v1",
                "pack_format": "deterministic_npy_tree_v1", "role": "development_full",
                "visibility": "candidate_readable", "source_payload_sha256": reader.SOURCE_SHA256,
                "round_trip_verified": True, "outcome_summaries_emitted": False,
                "record_count": 9, "record_entries": entries}
    (root / "MANIFEST.json").write_text(json.dumps(manifest))


class SourceSemanticsCheckerTests(unittest.TestCase):
    def test_known_reward_index_window_codes_kernel_and_admission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "development_full"
            fixture(root)
            with self.assertRaises(PermissionError):
                reader.open_development_pack(root, admit_development_root=checker.admission(root.parent / "other"),
                                             maximum_mapped_bytes=64 << 20, check=checker.budget(120))
            with reader.open_development_pack(root, admit_development_root=checker.admission(root),
                                              maximum_mapped_bytes=64 << 20, check=checker.budget(120)) as pack:
                structure = checker.structure(pack)
                kernel = checker.format_stage(pack)
                result = checker.semantics(pack, theta=round(kernel["isolated_peak_height_mode"] / 2, 1))
        record = structure["records"]["1"]
        self.assertEqual(record["rows"], 800)
        self.assertEqual(record["seshID"]["distinct_values"], list(range(1, 9)))
        self.assertEqual(record["lickState"]["distinct_finite_values"], [0, 1])
        self.assertTrue(abs(kernel["isolated_peak_height_mode"] - 1 / 0.15) <= 0.5, kernel)
        self.assertTrue(9 <= kernel["half_maximum_width_samples"] <= 13, kernel)
        self.assertNotIn("stopped", result)
        self.assertEqual(result["code_labels"], {"cued_rewarded": 0, "uncued": 1, "omission": 2})
        peak = result["predicates"]["peak"]
        self.assertEqual(peak["R_star"], R)
        self.assertEqual(result["R_star_primary_onset"], R)
        self.assertEqual(result["predicates"]["onset"]["table_at_R_star"]["w75"]["agreement"], 1.0)
        self.assertEqual(result["cue_index_implied"], R - 150)
        self.assertEqual(peak["table_at_R_star"]["w75"]["agreement"], 1.0)
        self.assertGreater(peak["discordance_vs_w75"]["w100"]["discordant_trials"], 20)
        self.assertEqual(peak["discordance_vs_w75"]["w100"]["fraction_state_sides_with_w75"], 1.0)
        for marker in ("E1_uncued_lick_rise", "E2_rewarded_minus_omission_rise"):
            self.assertTrue(1 <= result[marker]["cross_10_minus_R"] <= 30, result[marker])
            self.assertTrue(1 <= result[marker]["cross_50_minus_R"] <= 80, result[marker])
        self.assertLess(result["post_reward_lick_presence_by_code"]["2"],
                        result["post_reward_lick_presence_by_code"]["0"])
        self.assertEqual(result["eligible_cued_trials_by_session"]["1"]["1"], 90)
        text = json.dumps([structure, kernel, result])
        for forbidden in ("gain", "centre", "center", "probability_by_session"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
