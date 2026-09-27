from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

import ep12_executor.final_evaluation as final_module
from ep12_executor.final_evaluation import (
    DIRECTIONS,
    FinalAccessError,
    publish_final_result,
    run_final_transaction,
)


class HardInterruption(BaseException):
    pass


def bounded_result(
    value: Mapping[str, Any], final_lock: Mapping[str, Any]
) -> Mapping[str, Any]:
    result = dict(value)
    expected = {
        "operation_id",
        "procedure_lock_id",
        "role_rows_id",
        "state",
    }
    if set(result) != expected:
        raise FinalAccessError("future result is outside its bounded allowlist")
    if (
        result["operation_id"] != final_lock["operation_id"]
        or result["procedure_lock_id"] != final_lock["procedure_lock_id"]
        or result["role_rows_id"] != final_lock["role_rows_id"]
        or result["state"] != "result_committed"
    ):
        raise FinalAccessError("future result belongs to another operation")
    return result


class FakeHooks:
    def __init__(self, mode: str = "complete", durable: Mapping[str, Any] | None = None):
        self.mode = mode
        self.durable = dict(durable) if durable is not None else None
        self.prepare_count = 0
        self.access_count = 0
        self.reconcile_count = 0

    def prepare(self, final_lock: Mapping[str, Any]) -> None:
        self.prepare_count += 1

    def _result(self, final_lock: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "operation_id": final_lock["operation_id"],
            "procedure_lock_id": final_lock["procedure_lock_id"],
            "role_rows_id": final_lock["role_rows_id"],
            "state": "result_committed",
        }

    def evaluate_and_commit(
        self, *, final_lock: Mapping[str, Any], result_path: Path
    ) -> Mapping[str, Any]:
        self.access_count += 1
        if not (result_path.parent / "final_open.json").is_file():
            raise AssertionError("final outcome hook ran before the durable opening")
        result = self._result(final_lock)
        if self.mode == "hard_crash_without_result":
            raise HardInterruption("worker disappeared after opening")
        if self.mode == "lost_ack_after_commit":
            publish_final_result(result_path, result)
            raise HardInterruption("result committed before acknowledgement")
        return result

    def reconcile_committed(
        self, *, final_lock: Mapping[str, Any]
    ) -> Mapping[str, Any] | None:
        self.reconcile_count += 1
        return self.durable


class FinalTransactionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="ep12-final-core-")
        self.output = Path(self.temporary.name) / "final"
        self.lock = {
            "operation_id": "ep12-final-operation-1",
            "procedure_lock_id": "procedure-1",
            "role_rows_id": "roles-1",
            "selected_trial_id": "selected-42",
            "final_types": ["f1", "f2"],
            "directions": list(DIRECTIONS),
            "maximum_open_count": 1,
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run(self, hooks: FakeHooks, lock: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return run_final_transaction(
            final_lock=self.lock if lock is None else lock,
            output_dir=self.output,
            hooks=hooks,
            validate_result=bounded_result,
        )

    def test_core_has_no_data_reader_or_command_line_entrypoint(self) -> None:
        self.assertFalse(hasattr(final_module, "MaleCNSFinalEvaluator"))
        self.assertFalse(hasattr(final_module, "main"))

    def test_same_operation_reconciles_without_second_access(self) -> None:
        hooks = FakeHooks()
        result = self._run(hooks)
        self.assertEqual(result["state"], "result_committed")
        self.assertEqual(hooks.prepare_count, 1)
        self.assertEqual(hooks.access_count, 1)
        self.assertEqual(self._run(hooks), result)
        self.assertEqual(hooks.prepare_count, 1)
        self.assertEqual(hooks.access_count, 1)

    def test_lost_ack_after_commit_never_reopens(self) -> None:
        hooks = FakeHooks("lost_ack_after_commit")
        with self.assertRaises(HardInterruption):
            self._run(hooks)
        self.assertTrue((self.output / "final_open.json").is_file())
        self.assertTrue((self.output / "final_result.json").is_file())
        result = self._run(hooks)
        self.assertEqual(result["state"], "result_committed")
        self.assertEqual(hooks.access_count, 1)

    def test_retrieval_only_reconcile_after_interruption(self) -> None:
        first = FakeHooks("hard_crash_without_result")
        with self.assertRaises(HardInterruption):
            self._run(first)
        durable = first._result(self.lock)
        retry = FakeHooks(durable=durable)
        result = self._run(retry)
        self.assertEqual(result, durable)
        self.assertEqual(first.access_count, 1)
        self.assertEqual(retry.access_count, 0)
        self.assertEqual(retry.reconcile_count, 1)

    def test_unreconciled_open_fails_closed(self) -> None:
        first = FakeHooks("hard_crash_without_result")
        with self.assertRaises(HardInterruption):
            self._run(first)
        retry = FakeHooks()
        failure = self._run(retry)
        self.assertEqual(failure["state"], "technical_failure")
        self.assertEqual(failure["final_open_count"], 1)
        self.assertEqual(retry.access_count, 0)
        self.assertEqual(self._run(retry), failure)

    def test_different_operation_and_multiple_openings_are_rejected(self) -> None:
        hooks = FakeHooks()
        self._run(hooks)
        changed = dict(self.lock)
        changed["operation_id"] = "ep12-final-operation-2"
        with self.assertRaises(FinalAccessError):
            self._run(hooks, changed)
        unsafe = dict(self.lock)
        unsafe["maximum_open_count"] = 2
        with tempfile.TemporaryDirectory(prefix="ep12-final-unsafe-") as temporary:
            with self.assertRaisesRegex(FinalAccessError, "exactly one"):
                run_final_transaction(
                    final_lock=unsafe,
                    output_dir=Path(temporary) / "final",
                    hooks=FakeHooks(),
                    validate_result=bounded_result,
                )

    def test_symlinked_lock_and_mutex_are_rejected(self) -> None:
        self.output.mkdir()
        victim = self.output.parent / "victim.json"
        victim.write_text("{}\n", encoding="utf-8")
        (self.output / "final_lock.json").symlink_to(victim)
        with self.assertRaises(FinalAccessError):
            self._run(FakeHooks())
        (self.output / "final_lock.json").unlink()
        (self.output / ".final-operation.lock").unlink(missing_ok=True)
        (self.output / ".final-operation.lock").symlink_to(victim)
        with self.assertRaises(FinalAccessError):
            self._run(FakeHooks())


if __name__ == "__main__":
    unittest.main()
