#!/bin/bash
#SBATCH --job-name=ep12-procedure-lock
#SBATCH --partition=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/procedure-lock-%j.out

# Build the authenticated pre-null lock, then mechanically dispatch either the
# complete 99-replicate null array plus its afterok aggregate or an authenticated
# `null_not_eligible` status.  This script itself is never an array job.
set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  echo "usage: sbatch $0" >&2
  exit 2
fi
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "procedure-lock entrypoint must not run as an array" >&2
  exit 2
fi

for INHERITED_EP12_NAME in $(compgen -A variable EP12_ || true); do
  case "${INHERITED_EP12_NAME}" in
    EP12_SOURCE_FREEZE_MANIFEST|EP12_SOURCE_FREEZE_MANIFEST_SHA256|EP12_SOURCE_STAGE_ROOT|EP12_PROCEDURE_LOCK_LAUNCH_MODE|EP12_AUTO_LAUNCH_FULL_NULL|EP12_FULL_NULL_RESOURCE_AUTHORIZATION|EP12_FULL_NULL_RESOURCE_AUTHORIZATION_SHA256|EP12_RESOURCE_AUTHORIZATION_PENDING|EP12_RESOURCE_AUTHORIZATION_PENDING_FILE_SHA256|EP12_RESOURCE_RELEASE_INTENT) ;;
    *)
      echo "refusing ambient EP12 override: ${INHERITED_EP12_NAME}" >&2
      exit 2
      ;;
  esac
done
: "${EP12_SOURCE_FREEZE_MANIFEST:?source-freeze manifest is required}"
: "${EP12_SOURCE_FREEZE_MANIFEST_SHA256:?source-freeze file SHA-256 is required}"
: "${EP12_SOURCE_STAGE_ROOT:?immutable source-stage root is required}"
: "${EP12_PROCEDURE_LOCK_LAUNCH_MODE:?procedure-lock launch mode is required}"
if [[ ! "${EP12_SOURCE_FREEZE_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "source-freeze file SHA-256 is malformed" >&2
  exit 2
fi
EP12_RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"
if [[ ! "${EP12_RESTART_COUNT}" =~ ^[0-9]+$ ]]; then
  echo "Slurm restart count must be a nonnegative integer" >&2
  exit 2
fi

EP12_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_SCRATCH=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_RUN_ROOT="${EP12_ROOT}/outputs/development_run002"
EP12_TRANSITION_ROOT="${EP12_ROOT}/outputs/prelaunch/scientific_contract_transition"
EP12_TAIL_ROOT="${EP12_RUN_ROOT}/post36_tail"
EP12_PRE_NULL_INPUT_ROOT="${EP12_RUN_ROOT}/pre_null_inputs"
EP12_PROCEDURE_LOCK_ROOT="${EP12_RUN_ROOT}/procedure_lock"
EP12_PRE_NULL_DISPATCH_ROOT="${EP12_RUN_ROOT}/pre_null_dispatch"
EP12_PRE_NULL_DISPATCH_PENDING_ROOT="${EP12_RUN_ROOT}/pre_null_dispatch_pending"
EP12_PRE_NULL_DISPATCH_PENDING_RECEIPT="${EP12_PRE_NULL_DISPATCH_PENDING_ROOT}/resource_authorization_pending.json"
EP12_OBSERVED_FINALIZATION_ROOT="${EP12_RUN_ROOT}/observed_finalization"
EP12_OBSERVED_ATTESTATION_ROOT="${EP12_RUN_ROOT}/observed_attestation"
EP12_DEVELOPMENT_SNAPSHOT="${EP12_SCRATCH}/development_snapshot_v1"
EP12_DEVELOPMENT_MATERIALIZATION="${EP12_ROOT}/outputs/development_run001/materialization"
EP12_HIERARCHY_PREPARATION="${EP12_RUN_ROOT}/preparation"
EP12_FULL_NULL_DURABLE_BUNDLE="${EP12_RUN_ROOT}/full_search_runtime_bundle"
EP12_FULL_NULL_SCRATCH_BUNDLE="${EP12_SCRATCH}/development_run002/full_search_runtime_bundle"
EP12_FULL_NULL_WORK_ROOT="${EP12_SCRATCH}/development_run002/full_search_null"
EP12_FULL_NULL_DURABLE_ROOT="${EP12_RUN_ROOT}/full_search_null"
EP12_COMPARISON_ARCHIVE="${EP12_PRE_NULL_INPUT_ROOT}/complete_comparison_archive.json"
EP12_MODE_SUPPORT_RESULT="${EP12_PRE_NULL_INPUT_ROOT}/mode_support.json"
EP12_CONTROL_RESULTS="${EP12_PRE_NULL_INPUT_ROOT}/required_control_results.json"
EP12_NULL_GENERATOR_SELECTION="${EP12_PRE_NULL_INPUT_ROOT}/null_generator_selection.json"
EP12_NULL_CONTROLLER_PROGRAM="${EP12_PRE_NULL_INPUT_ROOT}/null_controller_program.json"
EP12_PRE_NULL_MANIFEST="${EP12_PRE_NULL_INPUT_ROOT}/manifest.json"
EP12_RUNTIME_SOURCE_BINDING="${EP12_PRE_NULL_INPUT_ROOT}/runtime_source_binding.json"
EP12_OBSERVED_ATTESTATION_MANIFEST="${EP12_OBSERVED_ATTESTATION_ROOT}/manifest.json"
EP12_DEVELOPMENT_ROLE_MANIFEST="${EP12_ROOT}/outputs/prelaunch/source_preflight/annotation_role_manifest.json"
EP12_FULL_NULL_RUNTIME_PROVIDER="ep12_executor.full_search_runtime:load_runtime_from_environment"
EP12_COST_PROJECTION="${EP12_ROOT}/outputs/development_run001/real_cost_projection.json"

case "${EP12_PROCEDURE_LOCK_LAUNCH_MODE}" in
  hold)
    if [[ "${EP12_AUTO_LAUNCH_FULL_NULL:-}" != "0" ]]; then
      echo "hold launch requires EP12_AUTO_LAUNCH_FULL_NULL=0" >&2
      exit 2
    fi
    EP12_AUTO_LAUNCH_FULL_NULL=0
    EP12_FULL_NULL_RESOURCE_AUTHORIZATION=""
    EP12_FULL_NULL_RESOURCE_AUTHORIZATION_SHA256=""
    EP12_RESOURCE_AUTHORIZATION_PENDING=""
    EP12_RESOURCE_AUTHORIZATION_PENDING_FILE_SHA256=""
    EP12_RESOURCE_RELEASE_INTENT=""
    ;;
  release)
    if [[ "${EP12_AUTO_LAUNCH_FULL_NULL:-}" != "1" ]]; then
      echo "release launch requires EP12_AUTO_LAUNCH_FULL_NULL=1" >&2
      exit 2
    fi
    EP12_AUTO_LAUNCH_FULL_NULL=1
    : "${EP12_FULL_NULL_RESOURCE_AUTHORIZATION:?release authorization path is required}"
    : "${EP12_FULL_NULL_RESOURCE_AUTHORIZATION_SHA256:?release authorization SHA-256 is required}"
    : "${EP12_RESOURCE_AUTHORIZATION_PENDING:?release pending path is required}"
    : "${EP12_RESOURCE_AUTHORIZATION_PENDING_FILE_SHA256:?release pending SHA-256 is required}"
    : "${EP12_RESOURCE_RELEASE_INTENT:?release intent path is required}"
    ;;
  *)
    echo "procedure-lock launch mode must be hold or release" >&2
    exit 2
    ;;
esac
if [[ "${EP12_AUTO_LAUNCH_FULL_NULL}" == "1" ]]; then
  if [[ ! "${EP12_FULL_NULL_RESOURCE_AUTHORIZATION_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "AUTO=1 requires an exact resource-authorization file SHA-256" >&2
    exit 2
  fi
  if [[ ! -f "${EP12_FULL_NULL_RESOURCE_AUTHORIZATION}" || -L "${EP12_FULL_NULL_RESOURCE_AUTHORIZATION}" ]]; then
    echo "AUTO=1 resource authorization is missing or unsafe" >&2
    exit 2
  fi
  if [[ "$(sha256sum "${EP12_FULL_NULL_RESOURCE_AUTHORIZATION}" | awk '{print $1}')" != "${EP12_FULL_NULL_RESOURCE_AUTHORIZATION_SHA256}" ]]; then
    echo "AUTO=1 resource authorization bytes differ from the supplied digest" >&2
    exit 2
  fi
  if [[ "${EP12_RESOURCE_AUTHORIZATION_PENDING}" != "${EP12_PRE_NULL_DISPATCH_PENDING_RECEIPT}" || \
        ! "${EP12_RESOURCE_AUTHORIZATION_PENDING_FILE_SHA256}" =~ ^[0-9a-f]{64}$ || \
        ! -f "${EP12_RESOURCE_AUTHORIZATION_PENDING}" || \
        -L "${EP12_RESOURCE_AUTHORIZATION_PENDING}" || \
        "$(sha256sum "${EP12_RESOURCE_AUTHORIZATION_PENDING}" | awk '{print $1}')" != "${EP12_RESOURCE_AUTHORIZATION_PENDING_FILE_SHA256}" ]]; then
    echo "release pending receipt path or bytes differ from the exact launch" >&2
    exit 2
  fi
fi

cd "${EP12_ROOT}"
export MODULEPATH=/share/software/modules/system:/home/groups/russpold/modules:/home/groups/russpold/neurodesk/local/containers/modules:/share/software/modules/math:/share/software/modules/devel:/share/software/modules/categories
export LMOD_SYSTEM_DEFAULT_MODULES=devel,math
if ! type module >/dev/null 2>&1; then
  LMOD_INIT=/share/software/user/open/lmod/lmod/init/bash
  if [[ ! -r "${LMOD_INIT}" || ! -f "${LMOD_INIT}" || -L "${LMOD_INIT}" ]]; then
    echo "validated Lmod initialization is unavailable: ${LMOD_INIT}" >&2
    exit 2
  fi
  source "${LMOD_INIT}"
fi
type module >/dev/null 2>&1 || { echo "Lmod initialization failed" >&2; exit 2; }
module purge
unset PYTHONPATH
module load python/3.12.1 \
  py-numpy/1.26.3_py312 \
  py-scipy/1.12.0_py312 \
  py-scikit-learn/1.5.1_py312 \
  py-pandas/2.2.1_py312 \
  py-pyarrow/18.1.0_py312
EP12_MODULE_PYTHONPATH="${PYTHONPATH:-}"
EP12_PYTHON_BIN="$(command -v python3 || true)"
if [[ "${EP12_PYTHON_BIN}" != "/share/software/user/open/python/3.12.1/bin/python3" ]] || \
   [[ "$(python3 -c 'import sys; print(sys.version.split()[0])')" != "3.12.1" ]]; then
  echo "validated Python 3.12.1 is unavailable" >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TMPDIR="${EP12_SCRATCH}"
EP12_SOURCE_STAGE_PARENT="${EP12_SCRATCH}/source_stages"
EP12_PYCACHE_PARENT="${EP12_SCRATCH}/source_pycache"
mkdir -p "${EP12_SOURCE_STAGE_PARENT}" "${EP12_PYCACHE_PARENT}"
for EP12_PRIVATE_PARENT in "${EP12_SOURCE_STAGE_PARENT}" "${EP12_PYCACHE_PARENT}"; do
  if [[ ! -d "${EP12_PRIVATE_PARENT}" || -L "${EP12_PRIVATE_PARENT}" ]]; then
    echo "unsafe private Scratch directory: ${EP12_PRIVATE_PARENT}" >&2
    exit 2
  fi
done
EP12_PYCACHE_ROOT="$(mktemp -d "${EP12_PYCACHE_PARENT}/job-${SLURM_JOB_ID:-nojob}-XXXXXX")"
export PYTHONPYCACHEPREFIX="${EP12_PYCACHE_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
EP12_EXPECTED_SOURCE_STAGE_ROOT="${EP12_SOURCE_STAGE_PARENT}/source-${EP12_SOURCE_FREEZE_MANIFEST_SHA256}"
if [[ "${EP12_SOURCE_STAGE_ROOT}" != "${EP12_EXPECTED_SOURCE_STAGE_ROOT}" || \
      ! -d "${EP12_SOURCE_STAGE_ROOT}/outputs/executor" || \
      -L "${EP12_SOURCE_STAGE_ROOT}" ]]; then
  echo "immutable source-stage root does not match the freeze content address" >&2
  exit 2
fi
export PYTHONPATH="${EP12_SOURCE_STAGE_ROOT}/outputs/executor${EP12_MODULE_PYTHONPATH:+:${EP12_MODULE_PYTHONPATH}}"
EP12_FROZEN_SOURCE_ROOT="$(python3 -B -m ep12_executor.source_freeze stage \
  --manifest "${EP12_SOURCE_FREEZE_MANIFEST}" \
  --expected-file-sha256 "${EP12_SOURCE_FREEZE_MANIFEST_SHA256}" \
  --stage-parent "${EP12_SOURCE_STAGE_PARENT}")"
case "${EP12_FROZEN_SOURCE_ROOT}" in
  "${EP12_SOURCE_STAGE_PARENT}"/source-*) ;;
  *) echo "source stage escaped its private parent" >&2; exit 2 ;;
esac
if [[ "${EP12_FROZEN_SOURCE_ROOT}" != "${EP12_SOURCE_STAGE_ROOT}" ]]; then
  echo "verified source stage differs from the submitted immutable stage" >&2
  exit 2
fi
if [[ ! -d "${EP12_FROZEN_SOURCE_ROOT}/outputs/executor" || -L "${EP12_FROZEN_SOURCE_ROOT}" ]]; then
  echo "source stage is missing or unsafe" >&2
  exit 2
fi
export PYTHONPATH="${EP12_FROZEN_SOURCE_ROOT}/outputs/executor${EP12_MODULE_PYTHONPATH:+:${EP12_MODULE_PYTHONPATH}}"
EP12_POLICY="${EP12_FROZEN_SOURCE_ROOT}/SEARCH_POLICY.yaml"
EP12_PREFIX_CONTRACT="${EP12_FROZEN_SOURCE_ROOT}/outputs/executor/SCIENTIFIC_CONTRACT.yaml"
EP12_ACTIVE_CONTRACT="${EP12_FROZEN_SOURCE_ROOT}/outputs/executor/SCIENTIFIC_CONTRACT_POST36.yaml"
EP12_CONTRACT_TRANSITION_RECEIPT="${EP12_TRANSITION_ROOT}/receipt.json"
EP12_CONTRACT_TRANSITION_SPEC="${EP12_FROZEN_SOURCE_ROOT}/outputs/executor/CONTRACT_TRANSITION_SPEC.json"
EP12_POST36_PLAN="${EP12_FROZEN_SOURCE_ROOT}/outputs/executor/OBSERVED_POST36_TAIL_PLAN.json"
EP12_FULL_NULL_SCRIPT="${EP12_FROZEN_SOURCE_ROOT}/outputs/executor/slurm/run_full_search_null.sbatch"
EP12_FULL_NULL_AGGREGATE_SCRIPT="${EP12_FROZEN_SOURCE_ROOT}/outputs/executor/slurm/run_full_search_null_aggregate.sbatch"
if [[ "${EP12_PROCEDURE_LOCK_LAUNCH_MODE}" == "release" ]]; then
  python3 -m ep12_executor.resource_release_launch verify-runtime \
    --intent "${EP12_RESOURCE_RELEASE_INTENT}" \
    --resource-authorization "${EP12_FULL_NULL_RESOURCE_AUTHORIZATION}" \
    --resource-authorization-file-sha256 "${EP12_FULL_NULL_RESOURCE_AUTHORIZATION_SHA256}" \
    --resource-authorization-pending "${EP12_RESOURCE_AUTHORIZATION_PENDING}" \
    --resource-authorization-pending-file-sha256 "${EP12_RESOURCE_AUTHORIZATION_PENDING_FILE_SHA256}" \
    --source-freeze-manifest "${EP12_SOURCE_FREEZE_MANIFEST}" \
    --source-freeze-manifest-sha256 "${EP12_SOURCE_FREEZE_MANIFEST_SHA256}"
fi

# Select the true maximum only across the exact 36 reports authenticated by the
# separate observed-trial attestation.  The selector reads timing and identity
# metadata only, rechecks the report bytes after attestation verification, and
# resolves equal timings by lexical trial ID.  An override is intentionally not
# accepted because a smaller hand-picked report would weaken the resource guard.
EP12_MAX_TIMING_REPORT=$(python3 -c '
import hashlib
import json
import math
import sys
from pathlib import Path

from ep12_executor.attest_observed_trials import verify_attestation_root
from ep12_executor.policy import EpisodePolicy

run_root = Path(sys.argv[1]).resolve(strict=True)
attestation_dir = Path(sys.argv[2]).resolve(strict=True)
policy = EpisodePolicy.load(Path(sys.argv[3]).resolve(strict=True))
verified = verify_attestation_root(
    run_root=run_root,
    attestation_dir=attestation_dir,
    policy=policy,
)
candidates = []
for trial_id, row in sorted(verified["_trial_rows_by_id"].items()):
    report_path = (
        run_root / row["source_trial_directory"] / "trial_report.json"
    ).resolve(strict=True)
    if not report_path.is_relative_to(run_root):
        raise SystemExit("attested timing report escaped the observed run root")
    report_bytes = report_path.read_bytes()
    if hashlib.sha256(report_bytes).hexdigest() != row["source_files_sha256"]["trial_report.json"]:
        raise SystemExit(f"attested timing report changed after verification: {trial_id}")
    report = json.loads(report_bytes)
    wall_seconds = report.get("wall_seconds")
    if (
        not isinstance(wall_seconds, (int, float))
        or isinstance(wall_seconds, bool)
        or not math.isfinite(float(wall_seconds))
        or float(wall_seconds) <= 0.0
        or report.get("trial_id") != trial_id
        or report.get("status") != "completed"
        or report.get("data_role") != "development_only"
        or report.get("final_connectivity_accessed") is not False
    ):
        raise SystemExit(f"ineligible authenticated timing metadata: {trial_id}")
    candidates.append((float(wall_seconds), trial_id, report_path))
if len(candidates) != policy.minimum_valid_trials:
    raise SystemExit("authenticated timing candidate count differs from policy")
selected = min(candidates, key=lambda item: (-item[0], item[1]))
print(selected[2])
' "${EP12_RUN_ROOT}" "${EP12_OBSERVED_ATTESTATION_ROOT}" "${EP12_POLICY}")

# Materialize the four scientific inputs plus authenticated selected fitted
# state.  An identical completed bundle is verified on retry; a partial or
# changed bundle fails closed.  This step never opens final-role connectivity.
python3 -m ep12_executor.pre_null_materializer \
  --output-dir "${EP12_PRE_NULL_INPUT_ROOT}" \
  --run-root "${EP12_RUN_ROOT}" \
  --observed-finalization-dir "${EP12_OBSERVED_FINALIZATION_ROOT}" \
  --observed-attestation-dir "${EP12_OBSERVED_ATTESTATION_ROOT}" \
  --tail-root "${EP12_TAIL_ROOT}" \
  --snapshot-dir "${EP12_DEVELOPMENT_SNAPSHOT}" \
  --materialization-dir "${EP12_DEVELOPMENT_MATERIALIZATION}" \
  --preparation-dir "${EP12_HIERARCHY_PREPARATION}" \
  --policy "${EP12_POLICY}" \
  --contract "${EP12_ACTIVE_CONTRACT}" \
  --prefix-contract "${EP12_PREFIX_CONTRACT}" \
  --contract-transition-receipt "${EP12_CONTRACT_TRANSITION_RECEIPT}" \
  --contract-transition-spec "${EP12_CONTRACT_TRANSITION_SPEC}" \
  --post36-plan "${EP12_POST36_PLAN}"

for EP12_REQUIRED_FILE in \
  "${EP12_COMPARISON_ARCHIVE}" \
  "${EP12_MODE_SUPPORT_RESULT}" \
  "${EP12_CONTROL_RESULTS}" \
  "${EP12_NULL_GENERATOR_SELECTION}" \
  "${EP12_NULL_CONTROLLER_PROGRAM}" \
  "${EP12_PRE_NULL_MANIFEST}" \
  "${EP12_RUNTIME_SOURCE_BINDING}" \
  "${EP12_OBSERVED_ATTESTATION_MANIFEST}" \
  "${EP12_DEVELOPMENT_ROLE_MANIFEST}" \
  "${EP12_COST_PROJECTION}" \
  "${EP12_PREFIX_CONTRACT}" \
  "${EP12_ACTIVE_CONTRACT}" \
  "${EP12_CONTRACT_TRANSITION_RECEIPT}" \
  "${EP12_CONTRACT_TRANSITION_SPEC}" \
  "${EP12_POST36_PLAN}" \
  "${EP12_MAX_TIMING_REPORT}" \
  "${EP12_TAIL_ROOT}/realized_schedule.json" \
  "${EP12_TAIL_ROOT}/realized_schedule_manifest.json" \
  "${EP12_TAIL_ROOT}/ledger.jsonl" \
  "${EP12_FULL_NULL_SCRIPT}" \
  "${EP12_FULL_NULL_AGGREGATE_SCRIPT}"; do
  if [[ ! -f "${EP12_REQUIRED_FILE}" ]]; then
    echo "required pre-null artifact is missing: ${EP12_REQUIRED_FILE}" >&2
    exit 3
  fi
done

EP12_PREFIX_CONTRACT_SHA256="$(sha256sum "${EP12_PREFIX_CONTRACT}" | awk '{print $1}')"
EP12_ACTIVE_CONTRACT_SHA256="$(sha256sum "${EP12_ACTIVE_CONTRACT}" | awk '{print $1}')"
EP12_CONTRACT_TRANSITION_FILE_SHA256="$(sha256sum "${EP12_CONTRACT_TRANSITION_RECEIPT}" | awk '{print $1}')"
EP12_CONTRACT_TRANSITION_SEMANTIC_SHA256="$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["transition_receipt_sha256"])' \
  "${EP12_CONTRACT_TRANSITION_RECEIPT}")"
EP12_OBSERVED_ATTESTATION_SHA256="$(sha256sum "${EP12_OBSERVED_ATTESTATION_MANIFEST}" | awk '{print $1}')"
EP12_DEVELOPMENT_ROLE_MANIFEST_SHA256="$(sha256sum "${EP12_DEVELOPMENT_ROLE_MANIFEST}" | awk '{print $1}')"

# Direct Python is intentional: this is a one-task batch step and must not
# inherit conflicting step-level Slurm variables through a nested srun.
python3 -m ep12_executor.procedure_lock \
  --policy "${EP12_POLICY}" \
  --prefix-contract-sha256 "${EP12_PREFIX_CONTRACT_SHA256}" \
  --active-contract-sha256 "${EP12_ACTIVE_CONTRACT_SHA256}" \
  --contract-transition-semantic-sha256 "${EP12_CONTRACT_TRANSITION_SEMANTIC_SHA256}" \
  --contract-transition-file-sha256 "${EP12_CONTRACT_TRANSITION_FILE_SHA256}" \
  --observed-attestation-sha256 "${EP12_OBSERVED_ATTESTATION_SHA256}" \
  --development-role-manifest-sha256 "${EP12_DEVELOPMENT_ROLE_MANIFEST_SHA256}" \
  --comparison-archive "${EP12_COMPARISON_ARCHIVE}" \
  --mode-support "${EP12_MODE_SUPPORT_RESULT}" \
  --realized-schedule "${EP12_TAIL_ROOT}/realized_schedule.json" \
  --realized-schedule-manifest "${EP12_TAIL_ROOT}/realized_schedule_manifest.json" \
  --realized-ledger "${EP12_TAIL_ROOT}/ledger.jsonl" \
  --control-results "${EP12_CONTROL_RESULTS}" \
  --null-generator-selection "${EP12_NULL_GENERATOR_SELECTION}" \
  --null-controller-program "${EP12_NULL_CONTROLLER_PROGRAM}" \
  --pre-null-manifest "${EP12_PRE_NULL_MANIFEST}" \
  --runtime-source-binding "${EP12_RUNTIME_SOURCE_BINDING}" \
  --output-dir "${EP12_PROCEDURE_LOCK_ROOT}"

EP12_NULL_ELIGIBLE=$(python3 -c \
  'import json,sys; print("true" if json.load(open(sys.argv[1]))["null_eligible"] else "false")' \
  "${EP12_PROCEDURE_LOCK_ROOT}/procedure_lock.json")

EP12_DISPATCH_ARGS=(
  --policy "${EP12_POLICY}"
  --procedure-lock "${EP12_PROCEDURE_LOCK_ROOT}/procedure_lock.json"
  --output-dir "${EP12_PRE_NULL_DISPATCH_ROOT}"
  --full-search-null-script "${EP12_FULL_NULL_SCRIPT}"
  --full-search-null-aggregate-script "${EP12_FULL_NULL_AGGREGATE_SCRIPT}"
  --source-freeze-manifest "${EP12_SOURCE_FREEZE_MANIFEST}"
  --source-freeze-manifest-sha256 "${EP12_SOURCE_FREEZE_MANIFEST_SHA256}"
)
if [[ "${EP12_NULL_ELIGIBLE}" == "true" ]]; then
  python3 -m ep12_executor.post_lock_runtime_materializer \
    --pre-null-dir "${EP12_PRE_NULL_INPUT_ROOT}" \
    --procedure-lock "${EP12_PROCEDURE_LOCK_ROOT}/procedure_lock.json" \
    --output-dir "${EP12_FULL_NULL_DURABLE_BUNDLE}" \
    --full-null-work-root "${EP12_FULL_NULL_WORK_ROOT}" \
    --policy "${EP12_POLICY}" \
    --development-role-manifest "${EP12_DEVELOPMENT_ROLE_MANIFEST}" \
    --cost-projection "${EP12_COST_PROJECTION}" \
    --max-timing-report "${EP12_MAX_TIMING_REPORT}" \
    --scratch-bundle-dir "${EP12_FULL_NULL_SCRATCH_BUNDLE}"

  EP12_FULL_NULL_STAGE_RECEIPT="${EP12_FULL_NULL_DURABLE_BUNDLE}_scratch_stage.json"
  EP12_FULL_NULL_STAGE_IDENTITY=$(python3 -c '
import hashlib
import json
import sys
from pathlib import Path

from ep12_executor.full_search_runtime import verify_scratch_stage_receipt

receipt_path = Path(sys.argv[1]).resolve(strict=True)
expected_durable = Path(sys.argv[2]).resolve(strict=True)
expected_scratch = Path(sys.argv[3]).resolve(strict=True)
receipt = verify_scratch_stage_receipt(json.loads(receipt_path.read_bytes()))
durable = Path(receipt["durable_bundle_dir"]).resolve(strict=True)
scratch = Path(receipt["scratch_bundle_dir"]).resolve(strict=True)
manifest_path = (scratch / "runtime_bundle.json").resolve(strict=True)
manifest_bytes = manifest_path.read_bytes()
manifest = json.loads(manifest_bytes)
if durable != expected_durable or scratch != expected_scratch:
    raise SystemExit("scratch-stage receipt paths differ from this launch")
if hashlib.sha256(manifest_bytes).hexdigest() != receipt["scratch_manifest_file_sha256"]:
    raise SystemExit("staged runtime manifest changed after receipt creation")
if manifest.get("bundle_sha256") != receipt["bundle_sha256"]:
    raise SystemExit("staged runtime bundle identity differs from receipt")
print(str(scratch) + "\t" + receipt["bundle_sha256"])
' "${EP12_FULL_NULL_STAGE_RECEIPT}" "${EP12_FULL_NULL_DURABLE_BUNDLE}" "${EP12_FULL_NULL_SCRATCH_BUNDLE}")
  IFS=$'\t' read -r EP12_FULL_NULL_BUNDLE EP12_FULL_NULL_BUNDLE_SHA256 EP12_UNEXPECTED_STAGE_FIELD \
    <<< "${EP12_FULL_NULL_STAGE_IDENTITY}"
  if [[ -z "${EP12_FULL_NULL_BUNDLE}" || -z "${EP12_FULL_NULL_BUNDLE_SHA256}" || -n "${EP12_UNEXPECTED_STAGE_FIELD}" ]]; then
    echo "authenticated Scratch runtime identity is malformed" >&2
    exit 3
  fi
  export EP12_FULL_NULL_BUNDLE EP12_FULL_NULL_BUNDLE_SHA256
  EP12_DISPATCH_ARGS+=(
    --runtime-provider "${EP12_FULL_NULL_RUNTIME_PROVIDER}"
    --runtime-bundle "${EP12_FULL_NULL_BUNDLE}"
    --runtime-bundle-sha256 "${EP12_FULL_NULL_BUNDLE_SHA256}"
    --full-null-work-root "${EP12_FULL_NULL_WORK_ROOT}"
    --full-null-durable-root "${EP12_FULL_NULL_DURABLE_ROOT}"
  )
  # Always retain the historical hold after the production benchmark and
  # authenticated Scratch stage exist.  Any later authorization must bind this
  # exact pending receipt and cannot erase or reinterpret it.
  python3 -c '
import hashlib
import json
import math
import sys
from pathlib import Path

from ep12_executor.full_search_runtime import (
    verify_resource_plan,
    verify_scratch_stage_receipt,
)
from ep12_executor.model_artifacts import atomic_json
from ep12_executor.policy import EpisodePolicy, digest_object
from ep12_executor.procedure_lock import verify_procedure_lock

durable_bundle = Path(sys.argv[1]).resolve(strict=True)
stage_receipt_path = Path(sys.argv[2]).resolve(strict=True)
procedure_lock_path = Path(sys.argv[3]).resolve(strict=True)
policy = EpisodePolicy.load(Path(sys.argv[4]).resolve(strict=True))
run_root = Path(sys.argv[5]).resolve(strict=True)
pending_dir = Path(sys.argv[6])
if pending_dir != run_root / "pre_null_dispatch_pending":
    raise SystemExit("resource hold receipt path escaped the observed run root")
if pending_dir.exists() and (pending_dir.is_symlink() or not pending_dir.is_dir()):
    raise SystemExit("resource hold receipt directory is unsafe")
pending_dir.mkdir(parents=False, exist_ok=True)

manifest_path = durable_bundle / "runtime_bundle.json"
resource_plan_path = durable_bundle / "resource_plan.json"
manifest_bytes = manifest_path.read_bytes()
resource_plan_bytes = resource_plan_path.read_bytes()
manifest = json.loads(manifest_bytes)
resource_plan = verify_resource_plan(json.loads(resource_plan_bytes), policy=policy)
manifest_core = {
    key: value for key, value in manifest.items() if key != "bundle_sha256"
}
if (
    manifest.get("bundle_sha256") != digest_object(manifest_core)
    or manifest.get("resource_plan_sha256") != resource_plan["plan_sha256"]
    or manifest.get("files", {}).get("resource_plan.json")
    != hashlib.sha256(resource_plan_bytes).hexdigest()
):
    raise SystemExit("durable runtime bundle and resource plan identities differ")

stage_receipt = verify_scratch_stage_receipt(
    json.loads(stage_receipt_path.read_bytes())
)
if stage_receipt.get("bundle_sha256") != manifest["bundle_sha256"]:
    raise SystemExit("Scratch stage receipt differs from durable runtime bundle")
procedure_lock_bytes = procedure_lock_path.read_bytes()
procedure_lock = json.loads(procedure_lock_bytes)
verify_procedure_lock(procedure_lock, policy=policy)

projection_names = (
    "projected_minimum_program_busy_core_hours",
    "projected_maximum_program_busy_core_hours",
    "projected_minimum_program_allocated_core_hours",
    "projected_maximum_program_allocated_core_hours",
)
projections = {name: float(resource_plan[name]) for name in projection_names}
if any(not math.isfinite(value) or value <= 0.0 for value in projections.values()):
    raise SystemExit("resource hold receipt projection is invalid")
core = {
    "schema_version": "ep12.resource_authorization_pending.v1",
    "action": "resource_authorization_pending",
    "reason": "explicit_exact_resource_authorization_required_before_full_null_dispatch",
    "auto_launch_full_null": False,
    "runtime_bundle_sha256": manifest["bundle_sha256"],
    "runtime_bundle_manifest_file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    "resource_plan_sha256": resource_plan["plan_sha256"],
    "resource_plan_file_sha256": hashlib.sha256(resource_plan_bytes).hexdigest(),
    "scratch_stage_receipt_sha256": stage_receipt["receipt_sha256"],
    "procedure_lock_sha256": procedure_lock["procedure_lock_sha256"],
    "procedure_lock_file_sha256": hashlib.sha256(procedure_lock_bytes).hexdigest(),
    **projections,
    "program_search_count": int(resource_plan["program_search_count"]),
    "synthetic_null_search_count": int(resource_plan["synthetic_null_search_count"]),
    "full_search_null_submitted": False,
    "jobs_submitted": [],
    "development_only": True,
    "final_connectivity_accessed": False,
    "final_connectivity_access_authorized": False,
}
receipt = {**core, "receipt_sha256": digest_object(core)}
receipt_path = pending_dir / "resource_authorization_pending.json"
if receipt_path.exists():
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise SystemExit("existing resource authorization hold receipt is unsafe")
    if json.loads(receipt_path.read_bytes()) != receipt:
        raise SystemExit("existing resource authorization hold receipt changed")
else:
    atomic_json(receipt_path, receipt)
' "${EP12_FULL_NULL_DURABLE_BUNDLE}" "${EP12_FULL_NULL_STAGE_RECEIPT}" \
    "${EP12_PROCEDURE_LOCK_ROOT}/procedure_lock.json" "${EP12_POLICY}" \
    "${EP12_RUN_ROOT}" "${EP12_PRE_NULL_DISPATCH_PENDING_ROOT}"
  if [[ "${EP12_AUTO_LAUNCH_FULL_NULL}" == "0" ]]; then
    exit 0
  fi
  EP12_DISPATCH_ARGS+=(
    --resource-authorization "${EP12_FULL_NULL_RESOURCE_AUTHORIZATION}"
    --resource-authorization-file-sha256 "${EP12_FULL_NULL_RESOURCE_AUTHORIZATION_SHA256}"
    --resource-authorization-pending "${EP12_PRE_NULL_DISPATCH_PENDING_RECEIPT}"
  )
fi

python3 -m ep12_executor.procedure_lock_dispatch "${EP12_DISPATCH_ARGS[@]}"
