from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from ep12_executor.policy import digest_object
from ep12_executor.post36_orchestration import (
    Post36OrchestrationError,
    _submit_stage,
    _submit_stage_held,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    root = tmp_path / "episode"
    transaction = root / "outputs/prelaunch/txn"
    transaction.mkdir(parents=True, mode=0o700)
    wrapper = root / "wrapper.sbatch"
    wrapper.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    core: dict[str, object] = {
        "schema_version": "ep12.post36_orchestration_stage_intent.v1",
        "orchestration_intent_sha256": "a" * 64,
        "stage": "synthetic",
        "submission_user": "zijiao",
        "predecessor_job_id": "11",
        "dependency": "afterok:11",
        "kill_on_invalid_dependency": True,
        "submission_barrier": "user_hold_receipt_before_release_v1",
        "wrapper": str(wrapper),
        "wrapper_sha256": _sha(wrapper),
        "arguments": ["bound-argument"],
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    named = {**core, "job_name": f"synthetic-{digest_object(core)[:24]}"}
    intent = {**named, "stage_intent_sha256": digest_object(named)}
    return root, transaction, intent


def _held(job_id: str, job_name: str, user: str) -> dict[str, str]:
    return {
        "job_id": job_id,
        "job_name": job_name,
        "user": user,
        "state": "PENDING",
        "reason": "JobHeldUser",
    }


def _case_receipt_exists_before_exact_release_and_retry_is_inert(
    case: unittest.TestCase, tmp_path: Path
) -> None:
    root, transaction, intent = _fixture(tmp_path)
    commands: list[list[str]] = []
    releases: list[list[str]] = []

    def submit(command: list[str]) -> str:
        commands.append(command)
        return "123"

    def resolve_name(_name: str, _user: str):
        return None

    def resolve_state(job_id: str, job_name: str, user: str):
        return _held(job_id, job_name, user)

    def release(command: list[str]) -> str:
        receipt = transaction / "stages/synthetic/receipt.json"
        assert receipt.is_file()
        releases.append(command)
        return ""

    receipt = _submit_stage(
        transaction_root=transaction,
        root=root,
        stage_intent=intent,
        submitter=submit,
        job_resolver=resolve_name,
        job_state_resolver=resolve_state,
        releaser=release,
    )
    assert commands[0][0:3] == ["/usr/bin/sbatch", "--parsable", "--hold"]
    assert "--export=NONE" in commands[0]
    assert releases == [["/usr/bin/scontrol", "release", "123"]]
    assert receipt["slurm_job_id"] == "123"
    release_path = transaction / "stages/synthetic/release_receipt.json"
    assert release_path.is_file()
    assert receipt["release_receipt_path"] == str(release_path)
    assert receipt["release_receipt_file_sha256"] == _sha(release_path)
    assert receipt["release_receipt_sha256"] == json.loads(
        release_path.read_text(encoding="utf-8")
    )["release_receipt_sha256"]

    def unexpected(*_args, **_kwargs):
        raise AssertionError("identical retry performed scheduler I/O")

    assert _submit_stage(
        transaction_root=transaction,
        root=root,
        stage_intent=intent,
        submitter=unexpected,
        job_resolver=unexpected,
        job_state_resolver=unexpected,
        releaser=unexpected,
    ) == receipt


def _case_lost_sbatch_ack_recovers_one_held_job_without_duplicate(
    case: unittest.TestCase, tmp_path: Path
) -> None:
    root, transaction, intent = _fixture(tmp_path)
    submits = 0
    recovery_visible = False

    def submit(_command: list[str]) -> str:
        nonlocal submits, recovery_visible
        submits += 1
        recovery_visible = True
        raise RuntimeError("lost sbatch acknowledgement")

    with case.assertRaisesRegex(RuntimeError, "lost sbatch"):
        _submit_stage(
            transaction_root=transaction,
            root=root,
            stage_intent=intent,
            submitter=submit,
            job_resolver=lambda _name, _user: None,
            job_state_resolver=lambda job, name, user: _held(job, name, user),
            releaser=lambda _command: "",
        )
    assert recovery_visible
    receipt = _submit_stage(
        transaction_root=transaction,
        root=root,
        stage_intent=intent,
        submitter=lambda _command: (_ for _ in ()).throw(
            AssertionError("duplicate sbatch")
        ),
        job_resolver=lambda name, user: {
            "job_id": "124", "job_name": name, "user": user
        },
        job_state_resolver=lambda job, name, user: _held(job, name, user),
        releaser=lambda _command: "",
    )
    assert submits == 1
    assert receipt["slurm_job_id"] == "124"


def _case_crash_after_accept_before_receipt_recovers_known_id(
    case: unittest.TestCase, tmp_path: Path
) -> None:
    root, transaction, intent = _fixture(tmp_path)
    submits = 0
    state_calls = 0

    def submit(_command: list[str]) -> str:
        nonlocal submits
        submits += 1
        return "125"

    def first_state(job: str, name: str, user: str):
        nonlocal state_calls
        state_calls += 1
        if state_calls == 1:
            raise RuntimeError("crash before receipt")
        return _held(job, name, user)

    with case.assertRaisesRegex(RuntimeError, "crash before receipt"):
        _submit_stage(
            transaction_root=transaction,
            root=root,
            stage_intent=intent,
            submitter=submit,
            job_resolver=lambda _name, _user: None,
            job_state_resolver=first_state,
            releaser=lambda _command: "",
        )
    receipt = _submit_stage(
        transaction_root=transaction,
        root=root,
        stage_intent=intent,
        submitter=lambda _command: (_ for _ in ()).throw(
            AssertionError("duplicate sbatch")
        ),
        job_resolver=lambda _name, _user: None,
        job_state_resolver=first_state,
        releaser=lambda _command: "",
    )
    assert submits == 1
    assert receipt["slurm_job_id"] == "125"


def _case_lost_release_ack_observes_nonheld_without_duplicate_release(
    case: unittest.TestCase,
    tmp_path: Path,
) -> None:
    root, transaction, intent = _fixture(tmp_path)
    scheduler_reason = "JobHeldUser"
    release_calls = 0

    def state(job: str, name: str, user: str):
        row = _held(job, name, user)
        row["reason"] = scheduler_reason
        return row

    def release(_command: list[str]) -> str:
        nonlocal scheduler_reason, release_calls
        release_calls += 1
        scheduler_reason = "Dependency"
        raise RuntimeError("lost release acknowledgement")

    with case.assertRaisesRegex(RuntimeError, "lost release"):
        _submit_stage(
            transaction_root=transaction,
            root=root,
            stage_intent=intent,
            submitter=lambda _command: "126",
            job_resolver=lambda _name, _user: None,
            job_state_resolver=state,
            releaser=release,
        )
    receipt = _submit_stage(
        transaction_root=transaction,
        root=root,
        stage_intent=intent,
        submitter=lambda _command: (_ for _ in ()).throw(
            AssertionError("duplicate sbatch")
        ),
        job_resolver=lambda _name, _user: None,
        job_state_resolver=state,
        releaser=lambda _command: (_ for _ in ()).throw(
            AssertionError("duplicate release")
        ),
    )
    assert release_calls == 1
    assert receipt["slurm_job_id"] == "126"
    release_receipt = json.loads(
        (transaction / "stages/synthetic/release_receipt.json").read_text()
    )
    assert release_receipt["receipt_was_durable_before_release"] is True


def _case_stage_arguments_reject_actual_newline_and_nul(
    case: unittest.TestCase, tmp_path: Path
) -> None:
    root, transaction, intent = _fixture(tmp_path)
    for bad in ("line\nbreak", "nul\x00byte"):
        changed = dict(intent)
        changed["arguments"] = [bad]
        core = {key: value for key, value in changed.items()
                if key != "stage_intent_sha256"}
        changed["stage_intent_sha256"] = digest_object(core)
        with case.assertRaisesRegex(
            Post36OrchestrationError, "positional arguments"
        ):
            _submit_stage(
                transaction_root=transaction / digest_object(changed)[:8],
                root=root,
                stage_intent=changed,
                submitter=lambda _command: "127",
                job_resolver=lambda _name, _user: None,
                job_state_resolver=lambda job, name, user: _held(job, name, user),
                releaser=lambda _command: "",
            )


def _case_exact_parenthesized_user_hold_is_canonicalized_and_near_misses_fail(
    case: unittest.TestCase, tmp_path: Path
) -> None:
    root, transaction, intent = _fixture(tmp_path / "safe")

    def state_with_reason(reason: str):
        def resolve(job_id: str, job_name: str, user: str):
            return {
                **_held(job_id, job_name, user),
                "reason": reason,
            }

        return resolve

    receipt = _submit_stage_held(
        transaction_root=transaction,
        root=root,
        stage_intent=intent,
        submitter=lambda _command: "130",
        job_resolver=lambda _name, _user: None,
        job_state_resolver=state_with_reason("(JobHeldUser)"),
    )
    assert receipt["slurm_job_id"] == "130"

    for index, unsafe in enumerate(
        (
            "(JobHeldUser )",
            "JobHeldUser ",
            "(JobHeldAdmin)",
            "jobhelduser",
            "Dependency",
        )
    ):
        with case.subTest(reason=unsafe):
            unsafe_root, unsafe_transaction, unsafe_intent = _fixture(
                tmp_path / f"unsafe-{index}"
            )
            with case.assertRaisesRegex(
                Post36OrchestrationError,
                "was not held through durable receipt publication",
            ):
                _submit_stage_held(
                    transaction_root=unsafe_transaction,
                    root=unsafe_root,
                    stage_intent=unsafe_intent,
                    submitter=lambda _command: str(131 + index),
                    job_resolver=lambda _name, _user: None,
                    job_state_resolver=state_with_reason(unsafe),
                )
            assert not (
                unsafe_transaction / "stages/synthetic/receipt.json"
            ).exists()


class Post36ExactOnceBarrierTests(unittest.TestCase):
    def _run_with_temporary_path(self, function) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-exact-once-") as directory:
            function(self, Path(directory))

    def test_receipt_exists_before_exact_release_and_retry_is_inert(self) -> None:
        self._run_with_temporary_path(
            _case_receipt_exists_before_exact_release_and_retry_is_inert
        )

    def test_lost_sbatch_ack_recovers_one_held_job_without_duplicate(self) -> None:
        self._run_with_temporary_path(
            _case_lost_sbatch_ack_recovers_one_held_job_without_duplicate
        )

    def test_crash_after_accept_before_receipt_recovers_known_id(self) -> None:
        self._run_with_temporary_path(
            _case_crash_after_accept_before_receipt_recovers_known_id
        )

    def test_lost_release_ack_observes_nonheld_without_duplicate_release(
        self,
    ) -> None:
        self._run_with_temporary_path(
            _case_lost_release_ack_observes_nonheld_without_duplicate_release
        )

    def test_stage_arguments_reject_actual_newline_and_nul(self) -> None:
        self._run_with_temporary_path(
            _case_stage_arguments_reject_actual_newline_and_nul
        )

    def test_exact_parenthesized_user_hold_is_canonicalized_and_near_misses_fail(
        self,
    ) -> None:
        self._run_with_temporary_path(
            _case_exact_parenthesized_user_hold_is_canonicalized_and_near_misses_fail
        )


if __name__ == "__main__":
    unittest.main()
