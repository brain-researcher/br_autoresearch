from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


SLURM_ROOT = Path(__file__).resolve().parents[1] / "slurm"


class SlurmHandoffTests(unittest.TestCase):
    def _script(self, name: str) -> str:
        return (SLURM_ROOT / name).read_text(encoding="utf-8")

    def test_small_wrappers_invoke_fixed_python_directly(self) -> None:
        archive_payloads = {
            "test_executor.sbatch": '"${EP12_PYTHON}" -P -B -m unittest discover',
            "qualify_methods.sbatch": '"${EP12_PYTHON}" -P -B -m ep12_executor qualify-methods',
            "check_production_readiness.sbatch": (
                '"${EP12_PYTHON}" -P -B -m ep12_executor.production_readiness'
            ),
            "run_post36_tail_from_archive.sbatch": (
                '"${EP12_PYTHON}" -P -B -m ep12_executor.post36_tail run-one'
            ),
            "run_procedure_lock_from_archive.sbatch": (
                '"${EP12_PYTHON}" -P -B -m ep12_executor.source_archive_freeze_bridge'
            ),
        }
        for name, payload in archive_payloads.items():
            with self.subTest(name=name):
                script = self._script(name)
                self.assertNotIn("\nsrun ", script)
                self.assertIn(
                    "EP12_PYTHON=/share/software/user/open/python/3.12.1/bin/python3.12",
                    script,
                )
                self.assertIn(payload, script)

        for name in ("attest_observed_trials.sbatch", "run_procedure_lock.sbatch"):
            with self.subTest(name=name):
                script = self._script(name)
                self.assertNotIn("srun --ntasks=1 python3", script)
                self.assertIn("python3 -m ep12_executor", script)

    def test_staged_wrappers_do_not_reenter_srun_with_sbatch_export_env(self) -> None:
        for name in (
            "test_executor.sbatch",
            "qualify_methods.sbatch",
            "check_production_readiness.sbatch",
            "run_post36_tail_from_archive.sbatch",
            "run_procedure_lock_from_archive.sbatch",
            "run_post36_tail.sbatch",
            "run_procedure_lock.sbatch",
            "run_full_search_null.sbatch",
            "run_full_search_null_aggregate.sbatch",
        ):
            with self.subTest(name=name):
                self.assertNotIn("\nsrun ", self._script(name))

    def test_every_wrapper_bootstraps_validated_lmod_before_module_use(self) -> None:
        archive_names = (
            "test_executor.sbatch",
            "qualify_methods.sbatch",
            "check_production_readiness.sbatch",
            "run_post36_tail_from_archive.sbatch",
            "run_procedure_lock_from_archive.sbatch",
        )
        legacy_names = (
            "run_post36_tail.sbatch",
            "run_procedure_lock.sbatch",
            "run_full_search_null.sbatch",
            "run_full_search_null_aggregate.sbatch",
        )
        initializer = "/share/software/user/open/lmod/lmod/init/bash"
        module_path = (
            "/share/software/modules/system:/home/groups/russpold/modules:"
            "/home/groups/russpold/neurodesk/local/containers/modules:"
            "/share/software/modules/math:/share/software/modules/devel:"
            "/share/software/modules/categories"
        )
        for name in archive_names:
            with self.subTest(name=name):
                script = self._script(name)
                self.assertIn(f"EP12_LMOD_INIT={initializer}", script)
                self.assertIn(f"export MODULEPATH={module_path}", script)
                self.assertIn("export LMOD_SYSTEM_DEFAULT_MODULES=devel,math", script)
                self.assertIn('source "${EP12_LMOD_INIT}"', script)
                self.assertNotIn("/etc/profile.d/modules.sh", script)
                self.assertIn(
                    "EP12_PYTHON=/share/software/user/open/python/3.12.1/bin/python3.12",
                    script,
                )
                self.assertIn('"${EP12_PYTHON}" --version', script)
                self.assertLess(
                    script.index('source "${EP12_LMOD_INIT}"'),
                    script.index("module purge"),
                )
                self.assertLess(
                    script.index('"${EP12_BOOTSTRAP}" verify-and-stage'),
                    script.index('source "${EP12_LMOD_INIT}"'),
                )

        for name in legacy_names:
            with self.subTest(name=name):
                script = self._script(name)
                self.assertIn(f"LMOD_INIT={initializer}", script)
                self.assertIn(f"export MODULEPATH={module_path}", script)
                self.assertIn("export LMOD_SYSTEM_DEFAULT_MODULES=devel,math", script)
                self.assertIn('source "${LMOD_INIT}"', script)
                self.assertNotIn("/etc/profile.d/modules.sh", script)
                self.assertIn(
                    'EP12_PYTHON_BIN="$(command -v python3 || true)"', script
                )
                self.assertIn(
                    '"${EP12_PYTHON_BIN}" != "/share/software/user/open/python/3.12.1/bin/python3"',
                    script,
                )
                self.assertIn('sys.version.split()[0]', script)
                self.assertLess(script.index('source "${LMOD_INIT}"'), script.index("module purge"))

        clean = subprocess.run(
            [
                "env",
                "-i",
                f"PATH={os.environ.get('PATH', '/usr/bin:/bin')}",
                f"HOME={os.environ.get('HOME', '/tmp')}",
                "bash",
                "--noprofile",
                "--norc",
                "-c",
                (
                    f"export MODULEPATH={module_path}; "
                    "export LMOD_SYSTEM_DEFAULT_MODULES=devel,math; "
                    f"source {initializer}; module purge; module load python/3.12.1; "
                    "test \"$(command -v python3)\" = "
                    "\"/share/software/user/open/python/3.12.1/bin/python3\"; "
                    "test \"$(python3 -c 'import sys; print(sys.version.split()[0])')\" "
                    "= \"3.12.1\""
                ),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        )
        self.assertEqual(clean.returncode, 0, clean.stderr)

    def test_archive_wrappers_stage_pinned_source_before_episode_work(self) -> None:
        wrappers = {
            "test_executor.sbatch": (
                11,
                '"${EP12_PYTHON}" -P -B -m unittest discover',
            ),
            "qualify_methods.sbatch": (
                17,
                '"${EP12_PYTHON}" -P -B -m ep12_executor qualify-methods',
            ),
            "check_production_readiness.sbatch": (
                25,
                '"${EP12_PYTHON}" -P -B -m ep12_executor.production_readiness',
            ),
            "run_post36_tail_from_archive.sbatch": (
                16,
                '"${EP12_PYTHON}" -P -B -m ep12_executor.post36_tail run-one',
            ),
            "run_procedure_lock_from_archive.sbatch": (
                16,
                '"${EP12_PYTHON}" -P -B -m ep12_executor.source_archive_freeze_bridge',
            ),
        }
        stager = '"${EP12_PYTHON}" -I -S "${EP12_BOOTSTRAP}" verify-and-stage'
        for name, (argument_count, payload) in wrappers.items():
            with self.subTest(name=name):
                script = self._script(name)
                self.assertIn("#SBATCH --export=NONE", script)
                self.assertIn(
                    f'[[ $# -eq {argument_count} ]] || ep12_fail "expected exactly {argument_count} positional arguments"',
                    script,
                )
                self.assertIn(
                    "EP12_BOOTSTRAP_SHA256=65f84b2879325e2549cdff7f138aebd3df40db64811327d7f4bd7d50c1bc3aaa",
                    script,
                )
                self.assertIn(stager, script)
                self.assertIn("PYTHONDONTWRITEBYTECODE=1", script)
                self.assertIn("PYTHONPYCACHEPREFIX", script)
                self.assertIn(
                    "export PYTHONPATH=${EP12_STAGED_EXECUTOR}${EP12_MODULE_PYTHONPATH:+:${EP12_MODULE_PYTHONPATH}}",
                    script,
                )
                self.assertNotIn('${EP12_SOURCE_FREEZE_MANIFEST:?', script)
                self.assertNotIn('${EP12_SOURCE_FREEZE_MANIFEST_SHA256:?', script)
                self.assertLess(script.index(stager), script.index(payload))
                self.assertLess(
                    script.index(stager),
                    script.index("export PYTHONPATH=${EP12_STAGED_EXECUTOR}"),
                )

    def test_legacy_wrappers_retain_source_freeze_where_still_supported(self) -> None:
        payload_markers = {
            "run_post36_tail.sbatch": "python3 -m ep12_executor.post36_tail",
            "run_procedure_lock.sbatch": "python3 -m ep12_executor.resource_release_launch",
            "run_full_search_null.sbatch": "python3 -m ep12_executor.full_null_launch_permit",
            "run_full_search_null_aggregate.sbatch": "python3 -m ep12_executor.full_null_launch_permit",
        }
        for name, payload in payload_markers.items():
            with self.subTest(name=name):
                script = self._script(name)
                stager = "python3 -B -m ep12_executor.source_freeze stage"
                self.assertIn('${EP12_SOURCE_FREEZE_MANIFEST:?', script)
                self.assertIn('${EP12_SOURCE_FREEZE_MANIFEST_SHA256:?', script)
                self.assertIn(stager, script)
                self.assertIn(
                    'export PYTHONPATH="${EP12_FROZEN_SOURCE_ROOT}/outputs/executor${EP12_MODULE_PYTHONPATH:+:${EP12_MODULE_PYTHONPATH}}"',
                    script,
                )
                self.assertLess(script.index(stager), script.index(payload))

    def test_staged_wrappers_and_configs_are_used_when_paths_are_explicit(self) -> None:
        readiness = self._script("check_production_readiness.sbatch")
        self.assertIn(
            '--falsifier-plan "${EP12_STAGED_EXECUTOR}/OBSERVED_POST36_TAIL_PLAN.json"',
            readiness,
        )
        self.assertIn(
            '--budget-override "${EP12_STAGED_EXECUTOR}/DEVELOPMENT_BUDGET_OVERRIDE.yaml"',
            readiness,
        )
        self.assertIn('--policy "${EP12_STAGED_ROOT}/SEARCH_POLICY.yaml"', readiness)
        qualification = self._script("qualify_methods.sbatch")
        self.assertIn(
            '--contract "${EP12_STAGED_EXECUTOR}/SCIENTIFIC_CONTRACT_POST36.yaml"',
            qualification,
        )
        self.assertIn(
            '--qualification-spec "${EP12_STAGED_EXECUTOR}/SYNTHETIC_QUALIFICATION_SPEC_POST36.yaml"',
            qualification,
        )
        tail = self._script("run_post36_tail_from_archive.sbatch")
        self.assertIn(
            '--plan "${EP12_STAGED_EXECUTOR}/OBSERVED_POST36_TAIL_PLAN.json"', tail
        )
        self.assertIn(
            '--active-contract "${EP12_STAGED_EXECUTOR}/SCIENTIFIC_CONTRACT_POST36.yaml"',
            tail,
        )
        procedure = self._script("run_procedure_lock_from_archive.sbatch")
        self.assertIn("ep12_executor.source_archive_freeze_bridge", procedure)
        self.assertIn(
            '/bin/bash "${EP12_STAGED_EXECUTOR}/slurm/run_procedure_lock.sbatch"',
            procedure,
        )
        aggregate = self._script("run_full_search_null_aggregate.sbatch")
        self.assertIn(
            'EP12_AGGREGATE_ENTRYPOINT="${EP12_FROZEN_SOURCE_ROOT}/outputs/executor/slurm/run_full_search_null.sbatch"',
            aggregate,
        )

    def test_readiness_uses_v2_evidence_and_archive_handoff_only_after_success(self) -> None:
        script = self._script("check_production_readiness.sbatch")
        self.assertIn("scientific_qualification_qualification003", script)
        readiness = script.index(
            '"${EP12_PYTHON}" -P -B -m ep12_executor.production_readiness'
        )
        tail_submit = script.index(
            '"${EP12_PYTHON}" -P -B -m ep12_executor.archive_slurm_handoff submit'
        )
        self.assertLess(readiness, tail_submit)
        self.assertIn("--kind readiness_to_archive_tail", script)
        self.assertIn('--parent-job-id "${SLURM_JOB_ID}"', script)
        self.assertIn(
            '--context-root "${EP12_RUN_ROOT}/production_readiness"', script
        )
        self.assertIn(
            '"${EP12_ROOT}/outputs/prelaunch/post36_orchestration/start_authorization.json"',
            script,
        )
        self.assertIn("verify_post36_start_authorization", script)
        self.assertIn('str(replacement.get("slurm_job_id")) == replacement_id', script)
        self.assertIn('archive.get("tree_sha256") == tree', script)
        self.assertNotIn("ep12_executor.slurm_handoff submit", script)
        self.assertNotIn("--export-env", script)
        self.assertNotIn("EP12_AUTO_LAUNCH_FULL_NULL", script)

        handoff = (
            Path(__file__).resolve().parents[1] / "ep12_executor" / "slurm_handoff.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'READINESS_SCHEMA = "ep12.production_readiness_gate.v2"', handoff
        )
        self.assertIn(
            'READINESS_MANIFEST_SCHEMA = "ep12.production_readiness_gate_manifest.v2"',
            handoff,
        )
        for field in (
            "observed_prefix_contract_sha256",
            "active_contract_sha256",
            "transition_manifest_file_sha256",
            "transition_receipt_file_sha256",
            "transition_receipt_sha256",
        ):
            self.assertIn(f'"{field}"', handoff)

    def test_archive_tail_and_procedure_bind_exact_authority_and_hold(self) -> None:
        tail = self._script("run_post36_tail_from_archive.sbatch")
        self.assertIn(
            '"${EP12_ROOT}/outputs/prelaunch/post36_orchestration/start_authorization.json"',
            tail,
        )
        self.assertIn(
            '"${EP12_ROOT}/outputs/prelaunch/post36_downstream_launch/receipt.json"',
            tail,
        )
        self.assertIn("readiness_to_archive_tail|archive_tail_to_next_tail", tail)
        self.assertIn("ep12_executor.archive_slurm_handoff verify-running-child", tail)
        terminal = tail.split("  0)", 1)[1].split("  10)", 1)[0]
        continuation = tail.split("  10)", 1)[1].split("  3|4)", 1)[0]
        deadline_stop = tail.split("  12)", 1)[1].split("  3|4)", 1)[0]
        self.assertIn("--kind archive_tail_to_procedure_lock", terminal)
        self.assertIn("--kind archive_tail_to_next_tail", continuation)
        self.assertIn("wall_deadline_stop.json", deadline_stop)
        self.assertNotIn("archive_slurm_handoff submit", deadline_stop)

        procedure = self._script("run_procedure_lock_from_archive.sbatch")
        self.assertIn(
            '"${EP12_ROOT}/outputs/prelaunch/post36_orchestration/receipt.json"',
            procedure,
        )
        self.assertIn(
            '"${EP12_ROOT}/outputs/prelaunch/post36_downstream_launch/receipt.json"',
            procedure,
        )
        self.assertIn(
            '[[ "${EP12_HANDOFF_KIND}" == archive_tail_to_procedure_lock ]]',
            procedure,
        )
        self.assertIn("ep12_executor.archive_slurm_handoff verify-running-child", procedure)
        self.assertIn(
            "EP12_PROCEDURE_LOCK_LAUNCH_MODE=hold EP12_AUTO_LAUNCH_FULL_NULL=0",
            procedure,
        )
        self.assertNotIn("EP12_AUTO_LAUNCH_FULL_NULL=1", procedure)
        bridge = procedure.index("ep12_executor.source_archive_freeze_bridge")
        hold = procedure.index("EP12_PROCEDURE_LOCK_LAUNCH_MODE=hold")
        self.assertLess(bridge, hold)
    def test_tail_terminal_dispatches_lock_but_failure_cases_do_not_submit(self) -> None:
        script = self._script("run_post36_tail.sbatch")
        self.assertIn("#SBATCH --requeue", script)
        self.assertIn("#SBATCH --open-mode=append", script)
        terminal_case = script.split("  0)", 1)[1].split("  10)", 1)[0]
        retry_case = script.split("  3)", 1)[1].split("  4)", 1)[0]
        unsupported_case = script.split("  4)", 1)[1].split("  *)", 1)[0]
        self.assertIn("EP12_PROCEDURE_LOCK_DISPATCHER", terminal_case)
        self.assertIn("python3 -m ep12_executor.slurm_handoff submit", terminal_case)
        self.assertIn("--kind post36_tail_to_procedure_lock", terminal_case)
        self.assertIn("--dependency=afterany:<readiness-job-id>", script)
        continuation_case = script.split("  10)", 1)[1].split("  3)", 1)[0]
        deadline_case = script.split("  12)", 1)[1].split("  3)", 1)[0]
        self.assertIn("python3 -m ep12_executor.slurm_handoff submit", continuation_case)
        self.assertIn("--kind post36_tail_to_next_tail", continuation_case)
        self.assertIn(
            "EP12_PROCEDURE_LOCK_LAUNCH_MODE=hold",
            terminal_case,
        )
        self.assertIn("EP12_AUTO_LAUNCH_FULL_NULL=0", terminal_case)
        self.assertIn("EP12_SOURCE_FREEZE_MANIFEST=", terminal_case)
        self.assertIn("EP12_SOURCE_FREEZE_MANIFEST=", continuation_case)
        self.assertNotIn("ALL,", script)
        self.assertIn("wall-deadline stop/pending receipt", deadline_case)
        self.assertNotIn("slurm_handoff submit", deadline_case)
        self.assertNotIn("sbatch", retry_case)
        self.assertNotIn("sbatch", unsupported_case)

    def test_tail_requeue_uses_a_fresh_transient_attempt_incarnation(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "ep12_executor" / "post36_tail.py"
        ).read_text(encoding="utf-8")
        self.assertIn('os.environ.get("SLURM_RESTART_COUNT", "0")', source)
        self.assertIn('f"{slurm_job_id}.restart{restart_count}"', source)
        self.assertIn("controller lease", source)

    def test_tail_rejects_ambient_ep12_poisoning_before_work(self) -> None:
        environment = dict(os.environ)
        environment["EP12_UNAUTHORIZED_OVERRIDE"] = "/tmp/poison"
        completed = subprocess.run(
            ["bash", str(SLURM_ROOT / "run_post36_tail.sbatch")],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("refusing ambient EP12 override", completed.stderr)

    def test_procedure_lock_has_full_materialization_and_benchmark_envelope(self) -> None:
        script = self._script("run_procedure_lock.sbatch")
        self.assertIn("#SBATCH --partition=normal", script)
        self.assertIn("#SBATCH --cpus-per-task=32", script)
        self.assertIn("#SBATCH --mem=192G", script)
        self.assertIn("#SBATCH --time=12:00:00", script)
        self.assertIn("#SBATCH --requeue", script)
        self.assertIn("#SBATCH --open-mode=append", script)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            self.assertIn(f"export {name}=1", script)

    def test_procedure_lock_selects_authenticated_true_maximum_timing(self) -> None:
        script = self._script("run_procedure_lock.sbatch")
        self.assertNotIn("coverage_016/trial_report.json", script)
        self.assertIn("verify_attestation_root", script)
        self.assertIn('row["source_files_sha256"]["trial_report.json"]', script)
        self.assertIn('report.get("wall_seconds")', script)
        self.assertIn(
            'selected = min(candidates, key=lambda item: (-item[0], item[1]))',
            script,
        )
        self.assertIn(
            'len(candidates) != policy.minimum_valid_trials',
            script,
        )

    def test_procedure_lock_dispatches_only_authenticated_scratch_bundle(self) -> None:
        script = self._script("run_procedure_lock.sbatch")
        self.assertIn(
            'EP12_FULL_NULL_SCRATCH_BUNDLE="${EP12_SCRATCH}/development_run002/full_search_runtime_bundle"',
            script,
        )
        self.assertIn(
            '--output-dir "${EP12_FULL_NULL_DURABLE_BUNDLE}"',
            script,
        )
        self.assertIn(
            '--scratch-bundle-dir "${EP12_FULL_NULL_SCRATCH_BUNDLE}"',
            script,
        )
        self.assertIn("verify_scratch_stage_receipt", script)
        self.assertIn(
            'EP12_FULL_NULL_STAGE_RECEIPT="${EP12_FULL_NULL_DURABLE_BUNDLE}_scratch_stage.json"',
            script,
        )
        self.assertIn(
            "export EP12_FULL_NULL_BUNDLE EP12_FULL_NULL_BUNDLE_SHA256",
            script,
        )
        self.assertIn('--runtime-bundle "${EP12_FULL_NULL_BUNDLE}"', script)
        self.assertNotIn(
            '--runtime-bundle "${EP12_FULL_NULL_DURABLE_BUNDLE}"',
            script,
        )

    def test_procedure_lock_default_hold_persists_then_requires_exact_authorization(self) -> None:
        script = self._script("run_procedure_lock.sbatch")
        self.assertIn('case "${EP12_PROCEDURE_LOCK_LAUNCH_MODE}" in', script)
        self.assertIn("hold)", script)
        self.assertIn("release)", script)
        self.assertIn("EP12_AUTO_LAUNCH_FULL_NULL=0", script)
        self.assertIn("EP12_AUTO_LAUNCH_FULL_NULL=1", script)
        self.assertIn("resource_release_launch verify-runtime", script)
        self.assertIn("--source-freeze-manifest", script)
        materialize = script.index(
            "python3 -m ep12_executor.post_lock_runtime_materializer"
        )
        stage_authentication = script.index("verify_scratch_stage_receipt")
        pending = script.index('"schema_version": "ep12.resource_authorization_pending.v1"')
        hold = script.rindex('if [[ "${EP12_AUTO_LAUNCH_FULL_NULL}" == "0" ]]')
        dispatch = script.rindex(
            "python3 -m ep12_executor.procedure_lock_dispatch"
        )
        self.assertLess(materialize, stage_authentication)
        self.assertLess(stage_authentication, pending)
        self.assertLess(pending, hold)
        self.assertLess(hold, dispatch)
        pending_block = script[pending:dispatch]
        self.assertIn("resource_authorization_pending.json", pending_block)
        self.assertIn('"runtime_bundle_sha256"', pending_block)
        self.assertIn('"resource_plan_sha256"', pending_block)
        for name in (
            "projected_minimum_program_busy_core_hours",
            "projected_maximum_program_busy_core_hours",
            "projected_minimum_program_allocated_core_hours",
            "projected_maximum_program_allocated_core_hours",
        ):
            self.assertIn(name, script)
        self.assertIn("verify_resource_plan", script)
        self.assertIn("full_search_null_submitted\": False", pending_block)
        self.assertIn("exit 0", pending_block)
        self.assertIn("--resource-authorization", pending_block)
        self.assertIn("--resource-authorization-file-sha256", pending_block)
        self.assertIn("--resource-authorization-pending", pending_block)
        self.assertIn("--full-search-null-aggregate-script", script)

    def test_full_null_publication_authenticates_and_reconciles_every_tree(self) -> None:
        script = self._script("run_full_search_null.sbatch")
        self.assertIn("EP12_FULL_NULL_LAUNCH_PERMIT", script)
        self.assertIn("ep12_executor.full_null_launch_permit verify", script)
        self.assertLess(
            script.index("ep12_executor.full_null_launch_permit verify"),
            script.index('case "${EP12_MODE}" in'),
        )
        aggregate = script.split('  aggregate)', 1)[1].split('  *)', 1)[0]
        self.assertGreaterEqual(aggregate.count("--aggregate-only"), 2)
        self.assertIn('--publish-stage "${EP12_STAGE_ROOT}"', aggregate)
        self.assertNotIn('cp -a "${EP12_WORK_ROOT}/."', aggregate)
        self.assertIn('cmp -s "${EP12_WORK_MANIFEST}"', aggregate)
        self.assertIn("existing durable aggregate differs", aggregate)
        self.assertIn("staged aggregate differs", aggregate)
        self.assertIn("published aggregate differs", aggregate)
        self.assertIn("ep12_executor.secure_publish_lock exec", aggregate)
        self.assertIn("ep12_executor.secure_publish_lock verify", aggregate)
        self.assertNotIn('exec 9>"${EP12_DURABLE_ROOT}.publish.lock"', aggregate)
        self.assertLess(
            aggregate.index('--publish-stage "${EP12_STAGE_ROOT}"'),
            aggregate.index('mv -T "${EP12_STAGE_ROOT}"'),
        )

    def test_full_null_requeues_only_a_checkpointed_array_incarnation(self) -> None:
        script = self._script("run_full_search_null.sbatch")
        self.assertIn("#SBATCH --partition=normal", script)
        self.assertIn("#SBATCH --cpus-per-task=32", script)
        self.assertIn("#SBATCH --mem=192G", script)
        self.assertIn("#SBATCH --time=2-00:00:00", script)
        self.assertIn("#SBATCH --requeue", script)
        self.assertIn("#SBATCH --open-mode=append", script)
        self.assertIn("#SBATCH --signal=B:SIGUSR1@46800", script)
        for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            self.assertIn(f"export {name}=1", script)
        replicate = script.split("  replicate)", 1)[1].split("  aggregate)", 1)[0]
        self.assertIn("trap ep12_request_restart SIGUSR1", replicate)
        self.assertIn('--restart-sentinel "${EP12_RESTART_SENTINEL}" &', replicate)
        self.assertIn("if (( EP12_REPLICATE_RC == 75 )); then", replicate)
        self.assertIn(
            'scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"',
            replicate,
        )
        self.assertIn('exit "${EP12_REPLICATE_RC}"', replicate)
        self.assertNotIn("scontrol requeue", script.split("  aggregate)", 1)[1])

    def test_full_null_aggregate_has_one_explicit_right_sized_allocation(self) -> None:
        script = self._script("run_full_search_null_aggregate.sbatch")
        self.assertIn("ep12_executor.full_null_launch_permit verify", script)
        self.assertLess(
            script.index("ep12_executor.full_null_launch_permit verify"),
            script.index('exec /bin/bash "${EP12_AGGREGATE_ENTRYPOINT}" aggregate'),
        )
        self.assertIn("#SBATCH --partition=normal", script)
        self.assertIn("#SBATCH --cpus-per-task=1", script)
        self.assertIn("#SBATCH --mem=32G", script)
        self.assertIn("#SBATCH --time=12:00:00", script)
        self.assertIn("#SBATCH --no-requeue", script)
        self.assertNotIn("#SBATCH --array", script)
        self.assertNotIn("#SBATCH --signal", script)
        self.assertIn(
            'exec /bin/bash "${EP12_AGGREGATE_ENTRYPOINT}" aggregate', script
        )
        self.assertIn("EP12_FULL_NULL_MAIN_WRAPPER_SHA256", script)
        self.assertIn("sha256sum --", script)
        self.assertIn("differs from the dispatcher-frozen", script)

    def test_full_null_aggregate_checks_staged_main_wrapper_identity(self) -> None:
        script = self._script("run_full_search_null_aggregate.sbatch")
        stager = "python3 -B -m ep12_executor.source_freeze stage"
        staged_entrypoint = (
            'EP12_AGGREGATE_ENTRYPOINT="${EP12_FROZEN_SOURCE_ROOT}/outputs/'
            'executor/slurm/run_full_search_null.sbatch"'
        )
        digest = 'sha256sum -- "${EP12_AGGREGATE_ENTRYPOINT}"'
        self.assertIn(staged_entrypoint, script)
        self.assertIn(digest, script)
        self.assertLess(script.index(stager), script.index(staged_entrypoint))
        self.assertLess(script.index(staged_entrypoint), script.index(digest))
        self.assertLess(
            script.index(digest),
            script.index("ep12_executor.full_null_launch_permit verify"),
        )
        self.assertNotIn(
            "EP12_AGGREGATE_ENTRYPOINT=/oak/stanford", script
        )


if __name__ == "__main__":
    unittest.main()
