#!/bin/bash
#SBATCH --job-name=ep12-post36-tail-archive
#SBATCH --partition=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --export=NONE
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/outputs/development_run002/slurm-post36-tail-archive-%j.out

set -euo pipefail
umask 077

ep12_fail() { echo "archive-bound post36 tail: $*" >&2; exit 2; }
ep12_sha() { [[ "$2" =~ ^[0-9a-f]{64}$ ]] || ep12_fail "$1 is not a lowercase SHA-256"; }

[[ $# -eq 16 ]] || ep12_fail "expected exactly 16 positional arguments"
EP12_ROOT=$1
EP12_ARCHIVE_ROOT=$2
EP12_ARCHIVE_RECEIPT_FILE_SHA256=$3
EP12_ARCHIVE_RECEIPT_SHA256=$4
EP12_ARCHIVE_MANIFEST_FILE_SHA256=$5
EP12_ARCHIVE_MANIFEST_SHA256=$6
EP12_ARCHIVE_TREE_SHA256=$7
EP12_CONTROLLER_AUTHORITY=$8
EP12_CONTROLLER_AUTHORITY_FILE_SHA256=$9
EP12_CONTROLLER_AUTHORITY_SHA256=${10}
EP12_DOWNSTREAM_RECEIPT=${11}
EP12_DOWNSTREAM_RECEIPT_FILE_SHA256=${12}
EP12_DOWNSTREAM_RECEIPT_SHA256=${13}
EP12_REPLACEMENT_JOB_ID=${14}
EP12_HANDOFF_KIND=${15}
EP12_HANDOFF_RECEIPT=${16}
readonly EP12_ROOT EP12_ARCHIVE_ROOT EP12_ARCHIVE_RECEIPT_FILE_SHA256
readonly EP12_ARCHIVE_RECEIPT_SHA256 EP12_ARCHIVE_MANIFEST_FILE_SHA256
readonly EP12_ARCHIVE_MANIFEST_SHA256 EP12_ARCHIVE_TREE_SHA256
readonly EP12_CONTROLLER_AUTHORITY EP12_CONTROLLER_AUTHORITY_FILE_SHA256
readonly EP12_CONTROLLER_AUTHORITY_SHA256
readonly EP12_DOWNSTREAM_RECEIPT EP12_DOWNSTREAM_RECEIPT_FILE_SHA256
readonly EP12_DOWNSTREAM_RECEIPT_SHA256 EP12_REPLACEMENT_JOB_ID
readonly EP12_HANDOFF_KIND EP12_HANDOFF_RECEIPT

EP12_CANONICAL_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_SCRATCH=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_PYTHON=/share/software/user/open/python/3.12.1/bin/python3.12
EP12_BOOTSTRAP_SHA256=65f84b2879325e2549cdff7f138aebd3df40db64811327d7f4bd7d50c1bc3aaa
readonly EP12_CANONICAL_ROOT EP12_SCRATCH EP12_PYTHON EP12_BOOTSTRAP_SHA256

IFS=$' \t\n'
export PATH=/usr/bin:/bin
export LD_LIBRARY_PATH=/share/software/user/open/python/3.12.1/lib
export LANG=C LC_ALL=C TZ=UTC PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONNOUSERSITE=1
unset BASH_ENV CDPATH CONDA_PREFIX ENV GLOBIGNORE HOME
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONBREAKPOINT PYTHONUSERBASE VIRTUAL_ENV

[[ -f "${EP12_PYTHON}" && ! -L "${EP12_PYTHON}" && -x "${EP12_PYTHON}" ]] || ep12_fail "fixed Python is unavailable"
[[ "$(/usr/bin/readlink -f -- "${EP12_PYTHON}")" == "${EP12_PYTHON}" ]] || ep12_fail "fixed Python path changed"
[[ "$("${EP12_PYTHON}" --version 2>&1)" == "Python 3.12.1" ]] || ep12_fail "fixed Python version changed"
[[ "${EP12_ROOT}" == "${EP12_CANONICAL_ROOT}" ]] || ep12_fail "episode root is not canonical"
[[ "${EP12_ARCHIVE_ROOT}" == /* && "${EP12_ARCHIVE_ROOT}" != *$'\n'* ]] || ep12_fail "archive root is invalid"
for EP12_NAME in EP12_ARCHIVE_RECEIPT_FILE_SHA256 EP12_ARCHIVE_RECEIPT_SHA256 \
  EP12_ARCHIVE_MANIFEST_FILE_SHA256 EP12_ARCHIVE_MANIFEST_SHA256 EP12_ARCHIVE_TREE_SHA256 \
  EP12_CONTROLLER_AUTHORITY_FILE_SHA256 EP12_CONTROLLER_AUTHORITY_SHA256 \
  EP12_DOWNSTREAM_RECEIPT_FILE_SHA256 EP12_DOWNSTREAM_RECEIPT_SHA256; do
  ep12_sha "${EP12_NAME}" "${!EP12_NAME}"
done
[[ "${EP12_REPLACEMENT_JOB_ID}" =~ ^[1-9][0-9]*$ ]] || ep12_fail "replacement job ID is invalid"
[[ "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ ]] || ep12_fail "SLURM_JOB_ID is invalid"
[[ -n "${SLURM_JOB_NAME:-}" && "${SLURM_JOB_NAME}" != *$'\n'* ]] || ep12_fail "SLURM_JOB_NAME is invalid"
case "${EP12_HANDOFF_KIND}" in
  readiness_to_archive_tail|archive_tail_to_next_tail) ;;
  *) ep12_fail "tail handoff kind is invalid" ;;
esac
[[ "${EP12_CONTROLLER_AUTHORITY}" == "${EP12_ROOT}/outputs/prelaunch/post36_orchestration/start_authorization.json" ]] || ep12_fail "controller start-authorization path is not canonical"
[[ "${EP12_DOWNSTREAM_RECEIPT}" == "${EP12_ROOT}/outputs/prelaunch/post36_downstream_launch/receipt.json" ]] || ep12_fail "downstream receipt path is not canonical"
EP12_HANDOFF_PREFIX="${EP12_ROOT}/outputs/development_run002/archive_slurm_handoffs/${EP12_HANDOFF_KIND}/parent-"
[[ "${EP12_HANDOFF_RECEIPT}" == "${EP12_HANDOFF_PREFIX}"*"/receipt.json" && "${EP12_HANDOFF_RECEIPT}" != *$'\n'* ]] || ep12_fail "handoff receipt path is not canonical"
EP12_HANDOFF_PARENT=${EP12_HANDOFF_RECEIPT#"${EP12_HANDOFF_PREFIX}"}
EP12_HANDOFF_PARENT=${EP12_HANDOFF_PARENT%/receipt.json}
[[ "${EP12_HANDOFF_PARENT}" =~ ^[1-9][0-9]*$ ]] || ep12_fail "handoff parent component is invalid"
for EP12_CONTROL in "${EP12_CONTROLLER_AUTHORITY}" "${EP12_DOWNSTREAM_RECEIPT}" "${EP12_HANDOFF_RECEIPT}"; do
  [[ -f "${EP12_CONTROL}" && ! -L "${EP12_CONTROL}" ]] || ep12_fail "required control receipt is missing or unsafe"
done
[[ "$(/usr/bin/sha256sum -- "${EP12_CONTROLLER_AUTHORITY}" | /usr/bin/cut -d' ' -f1)" == "${EP12_CONTROLLER_AUTHORITY_FILE_SHA256}" ]] || ep12_fail "controller start-authorization file SHA-256 mismatch"
[[ "$(/usr/bin/sha256sum -- "${EP12_DOWNSTREAM_RECEIPT}" | /usr/bin/cut -d' ' -f1)" == "${EP12_DOWNSTREAM_RECEIPT_FILE_SHA256}" ]] || ep12_fail "downstream receipt file SHA-256 mismatch"

EP12_BOOTSTRAP=${EP12_ARCHIVE_ROOT}/outputs/executor/ep12_executor/source_archive_bootstrap.py
[[ -f "${EP12_BOOTSTRAP}" && ! -L "${EP12_BOOTSTRAP}" ]] || ep12_fail "archive bootstrap is missing or unsafe"
[[ "$(/usr/bin/sha256sum -- "${EP12_BOOTSTRAP}" | /usr/bin/cut -d' ' -f1)" == "${EP12_BOOTSTRAP_SHA256}" ]] || ep12_fail "archive bootstrap SHA-256 mismatch"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
EP12_STAGE_PARENT=${EP12_SCRATCH}/source_archive_stage/${EP12_ARCHIVE_TREE_SHA256}
EP12_TMP_PARENT=${EP12_SCRATCH}/archive_wrapper_tmp
/usr/bin/mkdir -p -m 0700 "${EP12_STAGE_PARENT}" "${EP12_TMP_PARENT}"
for EP12_PRIVATE in "${EP12_STAGE_PARENT}" "${EP12_TMP_PARENT}"; do
  [[ -d "${EP12_PRIVATE}" && ! -L "${EP12_PRIVATE}" ]] || ep12_fail "unsafe private Scratch directory"
  [[ "$(/usr/bin/stat -c %u -- "${EP12_PRIVATE}")" == "$(/usr/bin/id -u)" ]] || ep12_fail "private Scratch owner changed"
  EP12_MODE=$(/usr/bin/stat -c %a -- "${EP12_PRIVATE}")
  (( (8#${EP12_MODE} & 8#022) == 0 )) || ep12_fail "private Scratch directory is writable by group or world"
done
EP12_JOB_TMP=$(/usr/bin/mktemp -d "${EP12_TMP_PARENT}/tail-${SLURM_JOB_ID}.XXXXXX")
export TMPDIR=${EP12_JOB_TMP} PYTHONPYCACHEPREFIX=${EP12_JOB_TMP}/pycache
/usr/bin/mkdir -m 0700 "${PYTHONPYCACHEPREFIX}"
EP12_STAGE_RESULT=${EP12_JOB_TMP}/source_archive_stage.json
"${EP12_PYTHON}" -I -S "${EP12_BOOTSTRAP}" verify-and-stage \
  --archive-root "${EP12_ARCHIVE_ROOT}" --stage-parent "${EP12_STAGE_PARENT}" \
  --expected-receipt-file-sha256 "${EP12_ARCHIVE_RECEIPT_FILE_SHA256}" \
  --expected-receipt-sha256 "${EP12_ARCHIVE_RECEIPT_SHA256}" \
  --expected-manifest-file-sha256 "${EP12_ARCHIVE_MANIFEST_FILE_SHA256}" \
  --expected-manifest-sha256 "${EP12_ARCHIVE_MANIFEST_SHA256}" \
  --expected-tree-sha256 "${EP12_ARCHIVE_TREE_SHA256}" > "${EP12_STAGE_RESULT}"
EP12_STAGED_ROOT=$("${EP12_PYTHON}" -I -S - \
  "${EP12_STAGE_RESULT}" "${EP12_STAGE_PARENT}" "${EP12_ARCHIVE_ROOT}" \
  "${EP12_ARCHIVE_RECEIPT_FILE_SHA256}" "${EP12_ARCHIVE_RECEIPT_SHA256}" \
  "${EP12_ARCHIVE_MANIFEST_FILE_SHA256}" "${EP12_ARCHIVE_MANIFEST_SHA256}" \
  "${EP12_ARCHIVE_TREE_SHA256}" <<'PY'
import json,os,sys
from pathlib import Path
path,parent,archive,receipt_file,receipt,manifest_file,manifest,tree=sys.argv[1:]
encoded=Path(path).read_bytes(); value=json.loads(encoded)
keys={"schema_version","stage_status","staged_root","upstream_archived_root","archive_receipt_file_sha256","archive_receipt_sha256","manifest_file_sha256","manifest_sha256","tree_sha256","complete"}
expected=os.path.abspath(os.path.join(parent,"source-"+tree))
canonical=(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True)+"\n").encode("ascii")
if not (set(value)==keys and encoded==canonical and value["schema_version"]=="ep12.source_archive_bootstrap_stage.v1" and value["stage_status"] in {"created","recovered_identical","reused_identical"} and value["staged_root"]==expected and value["upstream_archived_root"]==os.path.abspath(archive) and value["archive_receipt_file_sha256"]==receipt_file and value["archive_receipt_sha256"]==receipt and value["manifest_file_sha256"]==manifest_file and value["manifest_sha256"]==manifest and value["tree_sha256"]==tree and value["complete"] is True):
    raise SystemExit("bootstrap stage-result identity mismatch")
print(expected)
PY
)
EP12_STAGED_EXECUTOR=${EP12_STAGED_ROOT}/outputs/executor
for EP12_REQUIRED in ep12_executor/archive_slurm_handoff.py ep12_executor/post36_tail.py \
  OBSERVED_POST36_TAIL_PLAN.json SCIENTIFIC_CONTRACT.yaml SCIENTIFIC_CONTRACT_POST36.yaml \
  CONTRACT_TRANSITION_SPEC.json slurm/run_post36_tail_from_archive.sbatch \
  slurm/run_procedure_lock_from_archive.sbatch; do
  [[ -f "${EP12_STAGED_EXECUTOR}/${EP12_REQUIRED}" && ! -L "${EP12_STAGED_EXECUTOR}/${EP12_REQUIRED}" ]] || ep12_fail "required staged tail source is missing or unsafe"
done
[[ -f "${EP12_STAGED_ROOT}/SEARCH_POLICY.yaml" && ! -L "${EP12_STAGED_ROOT}/SEARCH_POLICY.yaml" ]] || ep12_fail "staged search policy is missing or unsafe"

EP12_LMOD_INIT=/share/software/user/open/lmod/lmod/init/bash
[[ -f "${EP12_LMOD_INIT}" && ! -L "${EP12_LMOD_INIT}" && -r "${EP12_LMOD_INIT}" ]] || ep12_fail "Lmod initialization is unavailable"
export MODULEPATH=/share/software/modules/system:/home/groups/russpold/modules:/home/groups/russpold/neurodesk/local/containers/modules:/share/software/modules/math:/share/software/modules/devel:/share/software/modules/categories
export LMOD_SYSTEM_DEFAULT_MODULES=devel,math
source "${EP12_LMOD_INIT}"
module purge
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONBREAKPOINT PYTHONUSERBASE
module load python/3.12.1 py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-scikit-learn/1.5.1_py312 py-pandas/2.2.1_py312 py-pyarrow/18.1.0_py312
EP12_MODULE_PYTHONPATH=${PYTHONPATH:-}
EP12_MODULE_LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export PATH=/share/software/user/open/python/3.12.1/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/share/software/user/open/python/3.12.1/lib${EP12_MODULE_LD_LIBRARY_PATH:+:${EP12_MODULE_LD_LIBRARY_PATH}}
export PYTHONPATH=${EP12_STAGED_EXECUTOR}${EP12_MODULE_PYTHONPATH:+:${EP12_MODULE_PYTHONPATH}}
unset HOME PYTHONHOME PYTHONSTARTUP PYTHONBREAKPOINT PYTHONUSERBASE

"${EP12_PYTHON}" -P -B -m ep12_executor.archive_slurm_handoff verify-running-child \
  --episode-root "${EP12_ROOT}" --executor-root "${EP12_STAGED_EXECUTOR}" \
  --kind "${EP12_HANDOFF_KIND}" --handoff-receipt-path "${EP12_HANDOFF_RECEIPT}" \
  --executed-wrapper "$0" --slurm-job-id "${SLURM_JOB_ID}" --slurm-job-name "${SLURM_JOB_NAME}" \
  --bootstrap-stage-result-path "${EP12_STAGE_RESULT}" --staged-root "${EP12_STAGED_ROOT}" \
  --wrapper-arguments "$@" > "${EP12_JOB_TMP}/handoff_runtime.json"

EP12_RUN_ROOT=${EP12_ROOT}/outputs/development_run002
EP12_TAIL_ROOT=${EP12_RUN_ROOT}/post36_tail
EP12_TAIL_SCRATCH=${EP12_SCRATCH}/development_run002/post36_tail
EP12_TRANSITION_ROOT=${EP12_ROOT}/outputs/prelaunch/scientific_contract_transition
/usr/bin/mkdir -p "${EP12_TAIL_SCRATCH}"
set +e
"${EP12_PYTHON}" -P -B -m ep12_executor.post36_tail run-one \
  --plan "${EP12_STAGED_EXECUTOR}/OBSERVED_POST36_TAIL_PLAN.json" \
  --policy "${EP12_STAGED_ROOT}/SEARCH_POLICY.yaml" \
  --prefix-contract "${EP12_STAGED_EXECUTOR}/SCIENTIFIC_CONTRACT.yaml" \
  --active-contract "${EP12_STAGED_EXECUTOR}/SCIENTIFIC_CONTRACT_POST36.yaml" \
  --contract-transition-receipt "${EP12_TRANSITION_ROOT}/receipt.json" \
  --contract-transition-spec "${EP12_STAGED_EXECUTOR}/CONTRACT_TRANSITION_SPEC.json" \
  --contract-transition-manifest "${EP12_TRANSITION_ROOT}/manifest.json" \
  --readiness-dir "${EP12_RUN_ROOT}/production_readiness" \
  --run-root "${EP12_RUN_ROOT}" --observed-finalization-dir "${EP12_RUN_ROOT}/observed_finalization" \
  --tail-root "${EP12_TAIL_ROOT}" --snapshot-dir "${EP12_SCRATCH}/development_snapshot_v1" \
  --scratch-root "${EP12_TAIL_SCRATCH}"
EP12_RC=$?
set -e
case "${EP12_RC}" in
  0)
    "${EP12_PYTHON}" -P -B -m ep12_executor.archive_slurm_handoff submit \
      --episode-root "${EP12_ROOT}" --executor-root "${EP12_STAGED_EXECUTOR}" \
      --kind archive_tail_to_procedure_lock --parent-job-id "${SLURM_JOB_ID}" \
      --context-root "${EP12_TAIL_ROOT}"
    ;;
  10)
    "${EP12_PYTHON}" -P -B -m ep12_executor.archive_slurm_handoff submit \
      --episode-root "${EP12_ROOT}" --executor-root "${EP12_STAGED_EXECUTOR}" \
      --kind archive_tail_to_next_tail --parent-job-id "${SLURM_JOB_ID}" \
      --context-root "${EP12_TAIL_ROOT}"
    ;;
  12)
    # The controller has already committed a hash-chained stop plus
    # wall_deadline_stop.json.  A descendant tail or procedure-lock job would
    # violate that durable authority, so terminate the chain successfully.
    exit 0
    ;;
  3|4) exit "${EP12_RC}" ;;
  *) exit "${EP12_RC}" ;;
esac
