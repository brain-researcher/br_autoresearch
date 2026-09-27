from __future__ import annotations

import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ep12_executor.policy import EpisodePolicy, digest_object
from ep12_executor.runtime import EPISODE_ROOT

try:
    from ep12_executor.contract_roles import ACTIVE_EPOCH, PREFIX_EPOCH, RANK_RULE
    from ep12_executor.procedure_lock import (
        ProcedureLockError,
        verify_procedure_lock,
        write_procedure_lock,
    )

    SCIENTIFIC_STACK_AVAILABLE = True
except ImportError:
    SCIENTIFIC_STACK_AVAILABLE = False


@unittest.skipUnless(SCIENTIFIC_STACK_AVAILABLE, "needs the scientific Python stack")
class ProcedureLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = EpisodePolicy.load(EPISODE_ROOT / "SEARCH_POLICY.yaml")

    def _lock(self) -> dict:
        core = {
            "policy_id": self.policy.policy_id,
            "policy_sha256": self.policy.policy_hash,
            "prefix_contract_id": "ep12_outgoing_tum_v1",
            "active_contract_id": "ep12_outgoing_tum_post36_v2",
            "prefix_contract_epoch": PREFIX_EPOCH,
            "active_contract_epoch": ACTIVE_EPOCH,
            "prevalence_parameter_rank_definition": RANK_RULE,
            "pre_null_input_binding": {
                "bundle_sha256": "1" * 64,
                "runtime_source_binding_sha256": "2" * 64,
                "callback_data_sha256": "3" * 64,
                "model_state_sha256": "4" * 64,
                "snapshot_binding_sha256": "5" * 64,
            },
            "procedure_locked": False,
            "null_eligible": False,
            "full_search_null_contract": {
                "required_replicates": self.policy.null_replicates,
                "pass_threshold": self.policy.null_pass_threshold,
                "rerun_complete_frozen_controller": True,
                "observed_realized_schedule_may_be_replayed": False,
            },
            "final_connectivity_accessed": False,
            "final_connectivity_access_authorized": False,
        }
        return {**core, "procedure_lock_sha256": digest_object(core)}

    def test_scientific_lock_verifies_without_generic_receipts(self) -> None:
        lock = self._lock()
        verify_procedure_lock(lock, policy=self.policy)
        self.assertNotIn("manifest_file_sha256", lock["pre_null_input_binding"])
        self.assertNotIn("attestation", lock)

    def test_final_access_or_mutation_is_rejected(self) -> None:
        lock = self._lock()
        lock["final_connectivity_accessed"] = True
        with self.assertRaises(ProcedureLockError):
            verify_procedure_lock(lock, policy=self.policy)

    def test_write_is_exclusive_and_identical_retry_is_semantic(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-procedure-lock-") as directory:
            output = Path(directory)
            lock = self._lock()
            first = write_procedure_lock(output, lock)
            second = write_procedure_lock(output, lock)
            self.assertEqual(first, second)
            self.assertEqual(
                json.loads((output / "procedure_lock.json").read_text(encoding="utf-8")),
                lock,
            )

    def test_concurrent_different_locks_cannot_overwrite_destination(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-procedure-race-") as directory:
            output = Path(directory)
            first = self._lock()
            second_core = {
                key: value
                for key, value in first.items()
                if key != "procedure_lock_sha256"
            }
            second_core["test_variant"] = "different"
            second = {
                **second_core,
                "procedure_lock_sha256": digest_object(second_core),
            }
            barrier = threading.Barrier(2)

            def write(value: dict) -> str:
                barrier.wait()
                try:
                    write_procedure_lock(output, value)
                except ProcedureLockError:
                    return "rejected"
                return "written"

            with ThreadPoolExecutor(max_workers=2) as executor:
                outcomes = list(executor.map(write, (first, second)))
            self.assertEqual(sorted(outcomes), ["rejected", "written"])
            persisted = json.loads(
                (output / "procedure_lock.json").read_text(encoding="utf-8")
            )
            self.assertIn(persisted, (first, second))

    def test_symlink_destination_is_never_accepted_as_the_lock(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-procedure-link-") as directory:
            output = Path(directory) / "output"
            output.mkdir()
            victim = Path(directory) / "victim.json"
            victim.write_text(json.dumps(self._lock()), encoding="utf-8")
            (output / "procedure_lock.json").symlink_to(victim)
            with self.assertRaisesRegex(ProcedureLockError, "unsafe"):
                write_procedure_lock(output, self._lock())


if __name__ == "__main__":
    unittest.main()
