from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path


SLURM_ROOT = Path(__file__).resolve().parents[1] / "slurm"


class SlurmWrapperTests(unittest.TestCase):
    @staticmethod
    def _procedure_lock_handoff_functions() -> str:
        text = (SLURM_ROOT / "run_post36_tail.sbatch").read_text(encoding="utf-8")
        start = "# BEGIN EP12_PROCEDURE_LOCK_HANDOFF_FUNCTIONS"
        end = "# END EP12_PROCEDURE_LOCK_HANDOFF_FUNCTIONS"
        return text.split(start, 1)[1].split(end, 1)[0]

    @staticmethod
    def _tail_continuation_functions() -> str:
        text = (SLURM_ROOT / "run_post36_tail.sbatch").read_text(encoding="utf-8")
        start = "# BEGIN EP12_TAIL_CONTINUATION_FUNCTIONS"
        end = "# END EP12_TAIL_CONTINUATION_FUNCTIONS"
        return text.split(start, 1)[1].split(end, 1)[0]

    @staticmethod
    def _interruptible_child_wait_function() -> str:
        text = (SLURM_ROOT / "run_full_search_null.sbatch").read_text(
            encoding="utf-8"
        )
        start = "# BEGIN EP12_INTERRUPTIBLE_CHILD_WAIT"
        end = "# END EP12_INTERRUPTIBLE_CHILD_WAIT"
        return text.split(start, 1)[1].split(end, 1)[0]

    def test_live_wrappers_do_not_rebuild_retired_provenance_layers(self) -> None:
        forbidden = (
            "source_freeze",
            "source_archive",
            "attest_observed_trials",
            "scientific_transition",
            "slurm_handoff",
            "secure_publish_lock",
            "full_null_launch_permit",
        )
        for path in SLURM_ROOT.glob("*.sbatch"):
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, f"{path.name} still invokes {token}")

    def test_scientific_execution_wrappers_remain(self) -> None:
        expected = {
            "check_production_readiness.sbatch",
            "qualify_methods.sbatch",
            "run_post36_tail.sbatch",
            "run_procedure_lock.sbatch",
            "run_full_search_null.sbatch",
            "run_full_search_null_aggregate.sbatch",
            "run_coverage_array.sbatch",
            "run_successor_array.sbatch",
        }
        self.assertTrue(expected.issubset({path.name for path in SLURM_ROOT.glob("*.sbatch")}))

    def test_obsolete_pre_null_shortcut_is_removed(self) -> None:
        self.assertFalse((SLURM_ROOT / "pre_null_gate.sbatch").exists())
        self.assertFalse(
            (SLURM_ROOT.parent / "ep12_executor" / "pre_null_gate.py").exists()
        )

    def test_no_final_data_wrapper_exists_before_scientific_completion(self) -> None:
        self.assertFalse((SLURM_ROOT / "run_final_evaluation.sbatch").exists())

    def test_tail_hands_normal_stop_to_procedure_lock(self) -> None:
        text = (SLURM_ROOT / "run_post36_tail.sbatch").read_text(encoding="utf-8")
        self.assertIn(
            "EP12_TAIL_SCRIPT=${EP12_ROOT}/outputs/executor/slurm/run_post36_tail.sbatch",
            text,
        )
        self.assertIn("ep12_submit_tail_continuation_once", text)
        self.assertIn("continuation_launches", text)
        self.assertIn("refusing controller replay", text)
        self.assertNotIn('sbatch --dependency="afterok:${SLURM_JOB_ID}" "$0"', text)
        self.assertIn('if [[ "${EP12_CONTROLLER_RC}" -eq 0 ]]; then', text)
        self.assertIn('status.get("action") == "stopped"', text)
        self.assertIn(
            'status.get("procedure_lock_submission_permitted") is True', text
        )
        self.assertIn('--dependency="afterok:${tail_job_id}"', text)
        self.assertIn('run_procedure_lock.sbatch', text)
        self.assertIn('procedure_lock_launch.json', text)
        self.assertIn('procedure_lock_launch.claim', text)
        self.assertIn('"phase": "scheduler_submission_started"', text)
        self.assertIn('if [[ "${EP12_CONTROLLER_RC}" -eq 12 ]]; then', text)

    def test_procedure_lock_handoff_replay_submits_only_once(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-procedure-handoff-") as name:
            root = Path(name)
            handoff = root / "procedure_lock_launch.json"
            claim = root / "procedure_lock_launch.claim"
            procedure_script = root / "run_procedure_lock.sbatch"
            procedure_script.write_text("#!/bin/bash\n", encoding="utf-8")
            counter = root / "sbatch-count"
            fake_sbatch = root / "sbatch"
            fake_sbatch.write_text(
                "\n".join(
                    (
                        "#!/bin/bash",
                        "set -euo pipefail",
                        f"counter={shlex.quote(str(counter))}",
                        'count=0',
                        'if [[ -f "${counter}" ]]; then read -r count < "${counter}"; fi',
                        'printf "%s\\n" "$((count + 1))" > "${counter}"',
                        'printf "%s\\n" "731245"',
                        "",
                    )
                ),
                encoding="utf-8",
            )
            fake_sbatch.chmod(0o700)
            script = f"""
set -euo pipefail
{self._procedure_lock_handoff_functions()}
ep12_submit_procedure_lock_once "$1" "$2" "$3" 910001 "$4"
ep12_submit_procedure_lock_once "$1" "$2" "$3" 910001 "$4"
"""
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    script,
                    "ep12-handoff-replay-test",
                    str(handoff),
                    str(claim),
                    str(procedure_script),
                    str(fake_sbatch),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(counter.read_text(encoding="utf-8"), "1\n")
            payload = json.loads(handoff.read_text(encoding="utf-8"))
            self.assertEqual(payload["procedure_lock_job_id"], "731245")
            self.assertEqual(payload["tail_terminal_job_id"], "910001")
            self.assertEqual(payload["dependency"], "afterok:910001")
            self.assertFalse(payload["final_connectivity_accessed"])
            self.assertFalse(payload["final_connectivity_access_authorized"])
            self.assertTrue((claim / "claimed.json").is_file())
            self.assertTrue((claim / "submission_started.json").is_file())

    def test_tail_continuation_replay_submits_only_once(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-tail-continuation-") as name:
            root = Path(name)
            handoff = root / "910100.json"
            claim = root / "910100.claim"
            tail_script = root / "run_post36_tail.sbatch"
            tail_script.write_text("#!/bin/bash\n", encoding="utf-8")
            counter = root / "sbatch-count"
            fake_sbatch = root / "sbatch"
            fake_sbatch.write_text(
                "\n".join(
                    (
                        "#!/bin/bash",
                        "set -euo pipefail",
                        f"counter={shlex.quote(str(counter))}",
                        "count=0",
                        'if [[ -f "${counter}" ]]; then read -r count < "${counter}"; fi',
                        'printf "%s\\n" "$((count + 1))" > "${counter}"',
                        'printf "%s\\n" "812300;sherlock"',
                        "",
                    )
                ),
                encoding="utf-8",
            )
            fake_sbatch.chmod(0o700)
            script = f"""
set -euo pipefail
{self._tail_continuation_functions()}
ep12_submit_tail_continuation_once "$1" "$2" "$3" 910100 post36_slot_037 "$4"
ep12_submit_tail_continuation_once "$1" "$2" "$3" 910100 post36_slot_037 "$4"
"""
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    script,
                    "ep12-tail-continuation-test",
                    str(handoff),
                    str(claim),
                    str(tail_script),
                    str(fake_sbatch),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(counter.read_text(encoding="utf-8"), "1\n")
            payload = json.loads(handoff.read_text(encoding="utf-8"))
            self.assertEqual(payload["tail_child_job_id"], "812300")
            self.assertEqual(payload["tail_parent_job_id"], "910100")
            self.assertEqual(payload["next_slot_id"], "post36_slot_037")
            self.assertTrue((claim / "claimed.json").is_file())
            self.assertTrue((claim / "submission_started.json").is_file())

    def test_orphan_procedure_lock_claim_fails_closed_without_submission(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-procedure-orphan-claim-") as name:
            root = Path(name)
            handoff = root / "procedure_lock_launch.json"
            claim = root / "procedure_lock_launch.claim"
            claim.mkdir()
            procedure_script = root / "run_procedure_lock.sbatch"
            procedure_script.write_text("#!/bin/bash\n", encoding="utf-8")
            counter = root / "sbatch-count"
            fake_sbatch = root / "sbatch"
            fake_sbatch.write_text(
                "\n".join(
                    (
                        "#!/bin/bash",
                        f"touch {shlex.quote(str(counter))}",
                        'printf "%s\\n" "731246"',
                        "",
                    )
                ),
                encoding="utf-8",
            )
            fake_sbatch.chmod(0o700)
            script = f"""
set -euo pipefail
{self._procedure_lock_handoff_functions()}
ep12_submit_procedure_lock_once "$1" "$2" "$3" 910002 "$4"
"""
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    script,
                    "ep12-handoff-orphan-test",
                    str(handoff),
                    str(claim),
                    str(procedure_script),
                    str(fake_sbatch),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("refusing duplicate submission", result.stderr)
            self.assertFalse(counter.exists())
            self.assertFalse(handoff.exists())

    def test_qualification_requests_serial_resources(self) -> None:
        text = (SLURM_ROOT / "qualify_methods.sbatch").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --ntasks=1", text)
        self.assertIn("#SBATCH --cpus-per-task=1", text)
        self.assertIn("#SBATCH --mem=6G", text)
        self.assertNotIn("#SBATCH --cpus-per-task=32", text)
        self.assertNotIn("#SBATCH --mem=192G", text)
        self.assertIn("python3 -m ep12_executor qualify-methods", text)
        self.assertNotIn("python3 -m ep12_executor.cli", text)

    def test_module_wrappers_bootstrap_lmod_before_nounset(self) -> None:
        for path in SLURM_ROOT.glob("*.sbatch"):
            text = path.read_text(encoding="utf-8")
            if "module purge" not in text:
                continue
            profile_offset = text.index(". /etc/profile >/dev/null")
            nounset_offset = text.index("set -euo pipefail")
            module_offset = text.index("module purge")
            self.assertLess(profile_offset, nounset_offset, path.name)
            self.assertLess(nounset_offset, module_offset, path.name)

    def test_null_restart_signal_writes_authenticated_payload(self) -> None:
        text = (SLURM_ROOT / "run_full_search_null.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn('printf "%s\\n" "ep12-safe-requeue-request-v1"', text)
        self.assertIn('mv -f -- "${EP12_RESTART_SENTINEL_TMP}"', text)
        self.assertNotIn("trap 'touch", text)
        self.assertIn('--restart-sentinel "${EP12_RESTART_SENTINEL}" &', text)
        self.assertIn('ep12_wait_for_child_after_usr1 "${EP12_CHILD_PID}"', text)
        self.assertIn('wait "${child_pid}"', text)
        self.assertNotIn('kill -0 "${EP12_CHILD_PID}"', text)

    def test_wrapper_signal_race_returns_child_status_after_trapped_usr1(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-wrapper-signal-") as name:
            root = Path(name)
            script = (
                r'''
set -uo pipefail
sentinel="$1/restart.request"
temporary="${sentinel}.tmp"
handle_usr1() {
  printf "%s\n" "ep12-safe-requeue-request-v1" > "${temporary}"
  mv -f -- "${temporary}" "${sentinel}"
  EP12_USR1_WAIT_INTERRUPTED=1
  sleep 0.1
}
trap handle_usr1 SIGUSR1
python3 - "${sentinel}" <<'PY' &
import pathlib
import sys
import time

sentinel = pathlib.Path(sys.argv[1])
while not sentinel.is_file():
    time.sleep(0.001)
raise SystemExit(75)
PY
child=$!
(sleep 0.1; kill -USR1 $$) &
'''
                + self._interruptible_child_wait_function()
                + r'''
set +e
ep12_wait_for_child_after_usr1 "${child}"
rc=$?
test "$(cat "${sentinel}")" = "ep12-safe-requeue-request-v1"
exit "${rc}"
'''
            )
            result = subprocess.run(
                ["bash", "-c", script, "ep12-signal-test", str(root)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 75, result.stderr)
            self.assertEqual(
                (root / "restart.request").read_bytes(),
                b"ep12-safe-requeue-request-v1\n",
            )

    def test_wrapper_does_not_retry_genuine_child_status_138(self) -> None:
        script = (
            "set -uo pipefail\n"
            + self._interruptible_child_wait_function()
            + r'''
set +e
(exit 138) &
child=$!
ep12_wait_for_child_after_usr1 "${child}"
exit $?
'''
        )
        result = subprocess.run(
            ["bash", "-c", script, "ep12-child-138-test"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        self.assertEqual(result.returncode, 138, result.stderr)

    def test_null_aggregate_authenticates_scratch_then_stages_durable(self) -> None:
        text = (SLURM_ROOT / "run_full_search_null_aggregate.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn('--output-dir "${EP12_FULL_NULL_WORK_ROOT}"', text)
        self.assertIn('--publish-stage "${EP12_FULL_NULL_DURABLE_ROOT}"', text)
        self.assertNotIn("--aggregate-only", text)


if __name__ == "__main__":
    unittest.main()
