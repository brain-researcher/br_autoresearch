"""One full-size restart seam: two continuous frames plus one restored frame.

Run only in the coordinator's finite Slurm allocation. Child processes are
sequential, so the uninterrupted and restored solvers never coexist. The
observers below retain evidence; they do not alter the propagation equations.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import unittest
from unittest.mock import patch


TOOLS = Path(__file__).resolve().parent
EPISODE = TOOLS.parents[1]


def _fixture(kind, root, deadline_seconds, memory_mib):
    import fast_solver_continuous as driver
    driver.compute_context()  # Check the actual shell before numerical imports.
    import h5py
    import numpy as np
    import fast_solver_frame_checkpoint as checkpoints
    from fast_io_readout import VectorizedReadout, read_window, readouts_bitwise_equal
    from fast_solver_continuous_kernel import dwell_spans

    run_dir, record_dir = root / (kind + "_run"), root / (kind + "_record")
    run_dir.mkdir()
    if kind == "uninterrupted":
        record_dir.mkdir()
    original_run = driver.run_frames
    original_commit = checkpoints.commit_frame
    original_resume = checkpoints.FrameRecordWriter.resume

    def observe_commit(path, checkpoint, writer):
        original_commit(path, checkpoint, writer)
        if checkpoint.next_frame == 1:
            checkpoints._atomic_save(run_dir / "frame1.npz", checkpoint)

    def observe_resume(cls, path, **kwargs):
        writer = original_resume(path, **kwargs)
        canonical = checkpoints.load_frame(kwargs["rolling_checkpoint_path"], state_count=69848)
        accepted = kwargs["checkpoint"]
        assert canonical.next_frame == canonical.committed_output_rows == accepted.next_frame == 1
        assert canonical.time.hex() == accepted.time.hex()
        assert canonical.last_row == accepted.last_row
        assert canonical.boundary_phase == accepted.boundary_phase
        assert canonical.cumulative_energy == accepted.cumulative_energy
        assert canonical.physical_energy == accepted.physical_energy
        assert np.array_equal(canonical.physical_state, accepted.physical_state)
        assert writer._next == 1
        assert bool(writer._file["windows_written"][0])
        assert not np.any(writer._file["windows_written"][1:])
        assert not bool(writer._file.attrs["record_complete"])
        return writer

    def observed_run(case, plan, fast, *args, **kwargs):
        base, template = fast.base, plan.template
        second = plan.window(1)
        active = False
        actual_charges, source_work, snapshots, observed = [], [], {}, {}
        expected_charges = []
        for first, last, _, _, _ in dwell_spans(second, template):
            source = {node: value for node, value in template.action_charge_objects[first]
                      if base.spec.pads <= node < base.spec.pads + 64}
            if any(source.values()):
                expected_charges.append(sorted((node, value) for node, value in source.items() if value))
            for action in range(first, last):
                down = {node: value for node, value in template.action_charge_objects[action]
                        if node >= base.n_up}
                if any(down.values()):
                    expected_charges.append(sorted((node, value) for node, value in down.items() if value))
        original_begin, original_charge = fast.begin_dwell, base._apply_charge
        original_read = VectorizedReadout.read_window

        def begin(row, start, charge):
            nonlocal active
            active = second.start <= start < second.end
            if start == second.start:
                snapshots["seam_before_charge"] = fast.full_state(start)
            result = original_begin(row, start, charge)
            if active:
                source_work.append(result[1])
            if start == second.start:
                snapshots["seam_after_source_charge"] = fast.full_state(start)
            return result

        def apply_charge(entry, charge):
            if active:
                ids = np.flatnonzero(charge)
                actual_charges.append([(int(i), float(charge[i])) for i in ids])
            return original_charge(entry, charge)

        def read(owner, window, means):
            result = original_read(owner, window, means)
            if window.index == 1:
                snapshots["hold_mean"] = means.copy()
                snapshots["split_bounds"] = window.split_bounds.copy()
                observed[1] = result
            return result

        with patch.object(fast, "begin_dwell", begin), patch.object(base, "_apply_charge", apply_charge), \
                patch.object(VectorizedReadout, "read_window", read):
            result = original_run(case, plan, fast, *args, **kwargs)
        assert actual_charges == expected_charges, "frame-2 source/DAQ charges were skipped or duplicated"
        assert actual_charges and source_work, "the boundary-charge check must be exercised"
        # The writer has closed before the existing reader is invoked.
        for frame in (0, 1):
            reconstructed = read_window(record_dir / "record.h5", plan, frame)
            if frame == 1:
                assert not readouts_bitwise_equal(observed[1], reconstructed)
            for name in reconstructed.__dataclass_fields__:
                snapshots[f"frame{frame}_{name}"] = np.asarray(getattr(reconstructed, name))
        snapshots["source_boundary_work"] = np.asarray(source_work)
        snapshots["pixel_observation"] = np.array([
            template.observation_roles[int(role)] == "pixel" for role in template.obs_role])
        snapshots["frame_start"] = np.array(second.start)
        snapshots["frame_end"] = np.array(second.end)
        np.savez(run_dir / "seam_evidence.npz", **snapshots)
        with (run_dir / "charge_events.json").open("x") as stream:
            json.dump(actual_charges, stream, allow_nan=False)
        with h5py.File(record_dir / "record.h5", "r") as f:
            assert f["raw/code"].shape[0] == 4000
            assert np.array_equal(f["windows_written"][:2], [True, True])
            assert not np.any(f["windows_written"][2:])
            assert f.attrs["simulation_status"] == "checkpointed"
            assert not bool(f.attrs["record_complete"])
        return result

    arguments = ["--run-dir", str(run_dir), "--record-dir", str(record_dir),
                 "--frames", "4000", "--stop-after-frame", "2",
                 "--deadline-seconds", str(deadline_seconds),
                 "--memory-limit-mib", str(memory_mib), "--cache-mib", "1024"]
    if kind == "restored":
        arguments += ["--resume-checkpoint", str(root / "uninterrupted_run/frame1.npz")]
    with patch.object(driver, "run_frames", observed_run), \
            patch.object(checkpoints, "commit_frame", observe_commit), \
            patch.object(checkpoints.FrameRecordWriter, "resume", classmethod(observe_resume)):
        return driver.main(arguments)


class RestartIntegrationTests(unittest.TestCase):
    def test_two_frame_restart_matches_uninterrupted(self):
        from fast_solver_continuous import compute_context
        compute_context()
        import h5py
        import numpy as np
        from fast_io_readout import WORK_FIELDS
        from fast_solver_compare import STATE_ATOL, STATE_RTOL, WORK_ATOL, WORK_RTOL, _quantizer
        from fast_solver_fastpath_compare import code_pixel_comparison, voltage_comparison
        from fast_solver_frame_checkpoint import BOUNDARY_PHASE, load_frame

        root = Path(os.environ["EP20_RESTART_TEST_DIR"]).resolve()
        self.assertTrue(root.is_relative_to(EPISODE / "outputs/runs"))
        self.assertTrue(root.is_dir() and not any(root.iterdir()), "test directory must be fresh and empty")
        budget = float(os.environ["EP20_RESTART_DEADLINE_SECONDS"])
        memory = float(os.environ.get("EP20_RESTART_MEMORY_MIB", "6656"))
        self.assertTrue(np.isfinite(budget) and budget > 0 and np.isfinite(memory) and memory > 1024)
        started = time.monotonic()

        def launch(kind):
            remaining = budget - (time.monotonic() - started)
            self.assertGreater(remaining, 60., "shared seam-check deadline exhausted")
            command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()),
                       "--fixture", kind, "--root", str(root),
                       "--deadline-seconds", str(remaining - 15.), "--memory-mib", str(memory)]
            with (root / (kind + ".stdout.txt")).open("x") as stream:
                result = subprocess.run(command, cwd=TOOLS, stdout=stream, stderr=subprocess.STDOUT,
                                        timeout=remaining, check=False)
            self.assertEqual(result.returncode, 0, f"inspect {kind}.stdout.txt and solver_report.json")

        launch("uninterrupted")
        # Copy only after clean process/file closure. The copied second row is
        # deliberately newer than frame1.npz and must be invalidated/replayed.
        shutil.copytree(root / "uninterrupted_record", root / "restored_record")
        launch("restored")

        reports = [json.loads((root / f"{kind}_run/solver_report.json").read_text())
                   for kind in ("uninterrupted", "restored")]
        for report, count in zip(reports, (2, 1)):
            self.assertEqual(report["status"], "checkpointed")
            self.assertEqual(report["frames_completed"], 2)
            self.assertEqual(report["frames_this_run"], count)
            self.assertEqual(len(report["per_frame_seconds"]), count)
        saved = load_frame(root / "uninterrupted_run/frame1.npz", state_count=69848)
        self.assertEqual(saved.next_frame, 1)
        self.assertEqual(saved.boundary_phase, BOUNDARY_PHASE)
        endpoints = [load_frame(root / f"{kind}_record/frame_checkpoint.npz", state_count=69848)
                     for kind in ("uninterrupted", "restored")]
        for endpoint in endpoints:
            self.assertEqual(endpoint.next_frame, 2)
            self.assertEqual(endpoint.committed_output_rows, 2)
            self.assertEqual(endpoint.boundary_phase, BOUNDARY_PHASE)
        self.assertEqual(endpoints[0].time.hex(), endpoints[1].time.hex())
        self.assertEqual(endpoints[0].last_row, endpoints[1].last_row)
        comparisons = {"endpoint_states": voltage_comparison(endpoints[0].physical_state,
                                                              endpoints[1].physical_state)}
        self.assertTrue(comparisons["endpoint_states"]["passed"], comparisons["endpoint_states"])

        with np.load(root / "uninterrupted_run/seam_evidence.npz", allow_pickle=False) as a, \
                np.load(root / "restored_run/seam_evidence.npz", allow_pickle=False) as b:
            self.assertEqual(float(a["frame_start"]).hex(), saved.time.hex())
            self.assertEqual(float(a["frame_end"]).hex(), endpoints[0].time.hex())
            for name in ("seam_before_charge", "seam_after_source_charge", "hold_mean",
                         "frame1_aperture_mean"):
                comparisons[name] = voltage_comparison(a[name], b[name])
                self.assertTrue(comparisons[name]["passed"], (name, comparisons[name]))
            structural = ("index", "raw_id", "raw_valid", "raw_clipped", "returned_id", "returned_valid",
                          "returned_clipped", "observation_id", "site", "role", "has_value", "valid",
                          "clipped", "assigned", "available", "support_ptr", "support_raw",
                          "support_start", "support_end")
            for name in ("split_bounds", "frame_start", "frame_end", *("frame1_" + n for n in structural)):
                np.testing.assert_array_equal(a[name], b[name], err_msg=name)
            # Row zero was already committed; reopening must preserve it exactly.
            for name in a.files:
                if name.startswith("frame0_"):
                    np.testing.assert_array_equal(a[name], b[name], err_msg=name)
            thresholds, levels = _quantizer()
            ptr = a["frame1_support_ptr"]
            pixel = a["pixel_observation"]
            np.testing.assert_array_equal(pixel, b["pixel_observation"])
            self.assertTrue(np.any(pixel) and np.all(np.diff(ptr)[pixel] == 3),
                            "fixture must retain its three-conversion pixels")
            pixel_rows = np.array([a["frame1_support_raw"][ptr[i]:ptr[i + 1]]
                                   for i in np.flatnonzero(pixel)]) - a["frame1_raw_id"][0]
            codes, pixels = code_pixel_comparison(a["frame1_aperture_mean"], b["frame1_aperture_mean"],
                                                  thresholds, levels, pixel_rows)
            self.assertTrue(codes["passed"], codes)
            self.assertTrue(pixels["passed"], pixels)
            comparisons.update(codes=codes, pixels=pixels)
            lsb = float(levels[1] - levels[0])
            np.testing.assert_allclose(b["frame1_returned_value"], a["frame1_returned_value"],
                                       rtol=0., atol=lsb * (1 + 1e-9))
            np.testing.assert_allclose(b["frame1_value"][pixel], a["frame1_value"][pixel],
                                       rtol=0., atol=lsb / 3 * (1 + 1e-9), equal_nan=True)
            np.testing.assert_allclose(b["frame1_value"][~pixel], a["frame1_value"][~pixel],
                                       rtol=0., atol=lsb * (1 + 1e-9), equal_nan=True)
            np.testing.assert_allclose(b["source_boundary_work"], a["source_boundary_work"],
                                       rtol=WORK_RTOL, atol=WORK_ATOL)

        self.assertEqual(json.loads((root / "uninterrupted_run/charge_events.json").read_text()),
                         json.loads((root / "restored_run/charge_events.json").read_text()))
        with h5py.File(root / "uninterrupted_record/record.h5", "r") as a, \
                h5py.File(root / "restored_record/record.h5", "r") as b:
            for name in ("energy/window_totals", "energy/cumulative_totals"):
                np.testing.assert_array_equal(a[name][0], b[name][0])
                np.testing.assert_allclose(b[name][:2], a[name][:2], rtol=WORK_RTOL, atol=WORK_ATOL)
            for endpoint, f in zip(endpoints, (a, b)):
                np.testing.assert_array_equal(f["energy/cumulative_totals"][1],
                    [endpoint.cumulative_energy[name] for name in WORK_FIELDS])
            columns = json.loads(a.attrs["accounting_columns_json"])
            clip_column = columns.index("clip_events")
            np.testing.assert_array_equal(a["accounting/window"][:2, clip_column],
                                          b["accounting/window"][:2, clip_column])
        np.testing.assert_allclose(endpoints[1].physical_energy, endpoints[0].physical_energy,
                                   rtol=WORK_RTOL, atol=WORK_ATOL)
        for name in ("stored_start", "stored_end", "input_work", "heat", "supplied_charge_work", "residual"):
            np.testing.assert_allclose(reports[1]["last_window_energy_audit"][name],
                                       reports[0]["last_window_energy_audit"][name],
                                       rtol=WORK_RTOL, atol=WORK_ATOL, err_msg=name)
        with (root / "seam_comparison.json").open("x") as stream:
            json.dump(dict(qualified=True, propagated_frames=3, comparisons=comparisons,
                state_atol=STATE_ATOL, state_rtol=STATE_RTOL, work_atol=WORK_ATOL, work_rtol=WORK_RTOL,
                independent_energy_residual="descriptive; no new threshold",
                elapsed_seconds=time.monotonic() - started), stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    if "--fixture" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--fixture", choices=("uninterrupted", "restored"), required=True)
        parser.add_argument("--root", type=Path, required=True)
        parser.add_argument("--deadline-seconds", type=float, required=True)
        parser.add_argument("--memory-mib", type=float, required=True)
        arguments = parser.parse_args()
        sys.exit(_fixture(arguments.fixture, arguments.root, arguments.deadline_seconds, arguments.memory_mib))
    unittest.main()
