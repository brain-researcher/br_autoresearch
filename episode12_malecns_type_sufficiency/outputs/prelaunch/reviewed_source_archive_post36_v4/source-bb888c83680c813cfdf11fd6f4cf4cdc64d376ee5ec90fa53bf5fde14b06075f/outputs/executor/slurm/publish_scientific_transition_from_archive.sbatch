#!/bin/bash
#SBATCH --job-name=ep12-transition-publish-archive
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --export=NONE
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/slurm-transition-publish-archive-%j.out

set -euo pipefail
umask 077

ep12_fail() {
  echo "post36 archive publisher wrapper: $*" >&2
  exit 2
}

ep12_require_sha256() {
  [[ "$2" =~ ^[0-9a-f]{64}$ ]] || ep12_fail "$1 is not a lowercase SHA-256"
}

[[ $# -eq 15 ]] || ep12_fail "expected exactly 15 positional arguments"

EP12_ROOT=$1
EP12_ARCHIVE_ROOT=$2
EP12_ARCHIVE_RECEIPT_FILE_SHA256=$3
EP12_ARCHIVE_RECEIPT_SHA256=$4
EP12_ARCHIVE_MANIFEST_FILE_SHA256=$5
EP12_ARCHIVE_MANIFEST_SHA256=$6
EP12_ARCHIVE_TREE_SHA256=$7
EP12_ORCHESTRATION_INTENT=$8
EP12_ORCHESTRATION_INTENT_FILE_SHA256=$9
EP12_ORCHESTRATION_INTENT_SHA256=${10}
EP12_STAGE_RECEIPT=${11}
EP12_COMPARISON=${12}
EP12_REPLACEMENT_TERMINAL_IDENTITY=${13}
EP12_POST36_START_AUTHORIZATION=${14}
EP12_OBSOLETE_CANCELLATION_RECEIPT=${15}
readonly EP12_ROOT EP12_ARCHIVE_ROOT EP12_ARCHIVE_RECEIPT_FILE_SHA256
readonly EP12_ARCHIVE_RECEIPT_SHA256 EP12_ARCHIVE_MANIFEST_FILE_SHA256
readonly EP12_ARCHIVE_MANIFEST_SHA256 EP12_ARCHIVE_TREE_SHA256
readonly EP12_ORCHESTRATION_INTENT EP12_ORCHESTRATION_INTENT_FILE_SHA256
readonly EP12_ORCHESTRATION_INTENT_SHA256 EP12_STAGE_RECEIPT EP12_COMPARISON
readonly EP12_REPLACEMENT_TERMINAL_IDENTITY
readonly EP12_POST36_START_AUTHORIZATION EP12_OBSOLETE_CANCELLATION_RECEIPT

EP12_CANONICAL_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_SCRATCH=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_BOOTSTRAP_SHA256=65f84b2879325e2549cdff7f138aebd3df40db64811327d7f4bd7d50c1bc3aaa
EP12_PYTHON=/share/software/user/open/python/3.12.1/bin/python3.12
readonly EP12_CANONICAL_ROOT EP12_SCRATCH EP12_BOOTSTRAP_SHA256 EP12_PYTHON

IFS=$' \t\n'
export PATH=/usr/bin:/bin
export LD_LIBRARY_PATH=/share/software/user/open/python/3.12.1/lib
export LANG=C
export LC_ALL=C
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export TZ=UTC
unset BASH_ENV CDPATH CONDA_PREFIX ENV GLOBIGNORE HOME
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONBREAKPOINT PYTHONUSERBASE VIRTUAL_ENV
[[ -f "${EP12_PYTHON}" && ! -L "${EP12_PYTHON}" && -x "${EP12_PYTHON}" ]] || \
  ep12_fail "fixed Python 3.12.1 is missing, unsafe, or not executable"
[[ "$(/usr/bin/readlink -f -- "${EP12_PYTHON}")" == "${EP12_PYTHON}" ]] || \
  ep12_fail "fixed Python executable does not resolve to itself"
[[ "$("${EP12_PYTHON}" --version 2>&1)" == "Python 3.12.1" ]] || \
  ep12_fail "fixed Python version is not 3.12.1"

[[ "${EP12_ROOT}" == "${EP12_CANONICAL_ROOT}" ]] || ep12_fail "episode root is not canonical"
[[ "${EP12_ARCHIVE_ROOT}" == /* && "${EP12_ARCHIVE_ROOT}" != *$'\n'* ]] || \
  ep12_fail "archive root must be an absolute single-line path"
for EP12_HASH_NAME in \
  EP12_ARCHIVE_RECEIPT_FILE_SHA256 \
  EP12_ARCHIVE_RECEIPT_SHA256 \
  EP12_ARCHIVE_MANIFEST_FILE_SHA256 \
  EP12_ARCHIVE_MANIFEST_SHA256 \
  EP12_ARCHIVE_TREE_SHA256 \
  EP12_ORCHESTRATION_INTENT_FILE_SHA256 \
  EP12_ORCHESTRATION_INTENT_SHA256; do
  ep12_require_sha256 "${EP12_HASH_NAME}" "${!EP12_HASH_NAME}"
done
[[ "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ ]] || ep12_fail "SLURM_JOB_ID is invalid"
[[ -n "${SLURM_JOB_NAME:-}" && "${SLURM_JOB_NAME}" != *$'\n'* ]] || \
  ep12_fail "SLURM_JOB_NAME is invalid"
[[ "${EP12_ORCHESTRATION_INTENT}" == \
  "${EP12_ROOT}/outputs/prelaunch/post36_orchestration/intent.json" ]] || \
  ep12_fail "orchestration intent path is not canonical"
[[ "${EP12_STAGE_RECEIPT}" == \
  "${EP12_ROOT}/outputs/prelaunch/post36_orchestration/stages/publisher/receipt.json" ]] || \
  ep12_fail "publisher stage receipt path is not canonical"
[[ "${EP12_COMPARISON}" == \
  "${EP12_ROOT}/outputs/prelaunch/contract_transition_probe/comparison_v1/comparison.json" ]] || \
  ep12_fail "comparison path is not canonical"
[[ "${EP12_REPLACEMENT_TERMINAL_IDENTITY}" == \
  "${EP12_ROOT}/outputs/prelaunch/replacement_attestation_launch/terminal_identity.json" ]] || \
  ep12_fail "replacement terminal-identity path is not canonical"
[[ "${EP12_POST36_START_AUTHORIZATION}" == \
  "${EP12_ROOT}/outputs/prelaunch/post36_orchestration/start_authorization.json" ]] || \
  ep12_fail "post36 start-authorization path is not canonical"
[[ "${EP12_OBSOLETE_CANCELLATION_RECEIPT}" == \
  "${EP12_ROOT}/outputs/prelaunch/replacement_attestation_launch/obsolete_cancellation/receipt.json" ]] || \
  ep12_fail "obsolete cancellation-receipt path is not canonical"
for EP12_INPUT in \
  "${EP12_ORCHESTRATION_INTENT}" "${EP12_STAGE_RECEIPT}" \
  "${EP12_COMPARISON}" "${EP12_REPLACEMENT_TERMINAL_IDENTITY}" \
  "${EP12_POST36_START_AUTHORIZATION}" "${EP12_OBSOLETE_CANCELLATION_RECEIPT}"; do
  [[ -f "${EP12_INPUT}" && ! -L "${EP12_INPUT}" ]] || \
    ep12_fail "required publisher input is missing, nonregular, or a symlink"
done
EP12_OBSERVED_INTENT_FILE_SHA256=$(/usr/bin/sha256sum -- "${EP12_ORCHESTRATION_INTENT}")
EP12_OBSERVED_INTENT_FILE_SHA256=${EP12_OBSERVED_INTENT_FILE_SHA256%% *}
[[ "${EP12_OBSERVED_INTENT_FILE_SHA256}" == "${EP12_ORCHESTRATION_INTENT_FILE_SHA256}" ]] || \
  ep12_fail "orchestration intent file SHA-256 mismatch"

EP12_BOOTSTRAP=${EP12_ARCHIVE_ROOT}/outputs/executor/ep12_executor/source_archive_bootstrap.py
[[ -f "${EP12_BOOTSTRAP}" && ! -L "${EP12_BOOTSTRAP}" ]] || \
  ep12_fail "pinned archive bootstrap is missing, nonregular, or a symlink"
EP12_OBSERVED_BOOTSTRAP_SHA256=$(/usr/bin/sha256sum -- "${EP12_BOOTSTRAP}")
EP12_OBSERVED_BOOTSTRAP_SHA256=${EP12_OBSERVED_BOOTSTRAP_SHA256%% *}
[[ "${EP12_OBSERVED_BOOTSTRAP_SHA256}" == "${EP12_BOOTSTRAP_SHA256}" ]] || \
  ep12_fail "archive bootstrap SHA-256 mismatch"

export BLIS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
EP12_STAGE_PARENT=${EP12_SCRATCH}/source_archive_stage/${EP12_ARCHIVE_TREE_SHA256}
EP12_TMP_PARENT=${EP12_SCRATCH}/archive_wrapper_tmp
/usr/bin/mkdir -p -m 0700 "${EP12_STAGE_PARENT}" "${EP12_TMP_PARENT}"
[[ -d "${EP12_STAGE_PARENT}" && ! -L "${EP12_STAGE_PARENT}" ]] || \
  ep12_fail "archive stage parent is unsafe"
EP12_JOB_TMP=$(/usr/bin/mktemp -d "${EP12_TMP_PARENT}/publish-${SLURM_JOB_ID}.XXXXXX")
export TMPDIR=${EP12_JOB_TMP}
export PYTHONPYCACHEPREFIX=${EP12_JOB_TMP}/pycache
/usr/bin/mkdir -m 0700 "${PYTHONPYCACHEPREFIX}"
EP12_STAGE_RESULT=${EP12_JOB_TMP}/source_archive_stage.json

"${EP12_PYTHON}" -I -S "${EP12_BOOTSTRAP}" verify-and-stage \
  --archive-root "${EP12_ARCHIVE_ROOT}" \
  --stage-parent "${EP12_STAGE_PARENT}" \
  --expected-receipt-file-sha256 "${EP12_ARCHIVE_RECEIPT_FILE_SHA256}" \
  --expected-receipt-sha256 "${EP12_ARCHIVE_RECEIPT_SHA256}" \
  --expected-manifest-file-sha256 "${EP12_ARCHIVE_MANIFEST_FILE_SHA256}" \
  --expected-manifest-sha256 "${EP12_ARCHIVE_MANIFEST_SHA256}" \
  --expected-tree-sha256 "${EP12_ARCHIVE_TREE_SHA256}" \
  > "${EP12_STAGE_RESULT}"

EP12_STAGED_ROOT=$("${EP12_PYTHON}" -I -S - \
  "${EP12_STAGE_RESULT}" "${EP12_STAGE_PARENT}" "${EP12_ARCHIVE_ROOT}" \
  "${EP12_ARCHIVE_RECEIPT_FILE_SHA256}" "${EP12_ARCHIVE_RECEIPT_SHA256}" \
  "${EP12_ARCHIVE_MANIFEST_FILE_SHA256}" "${EP12_ARCHIVE_MANIFEST_SHA256}" \
  "${EP12_ARCHIVE_TREE_SHA256}" <<'PY'
import json
import os
from pathlib import Path
import sys

path, parent, archive, receipt_file, receipt, manifest_file, manifest, tree = sys.argv[1:]
encoded = Path(path).read_bytes()
value = json.loads(encoded.decode("utf-8"))
keys = {
    "schema_version", "stage_status", "staged_root", "upstream_archived_root",
    "archive_receipt_file_sha256", "archive_receipt_sha256",
    "manifest_file_sha256", "manifest_sha256", "tree_sha256", "complete",
}
expected_root = os.path.abspath(os.path.join(parent, "source-" + tree))
if not (
    set(value) == keys
    and encoded == (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
    and value["schema_version"] == "ep12.source_archive_bootstrap_stage.v1"
    and value["stage_status"] in {"created", "recovered_identical", "reused_identical"}
    and value["staged_root"] == expected_root
    and value["upstream_archived_root"] == os.path.abspath(archive)
    and value["archive_receipt_file_sha256"] == receipt_file
    and value["archive_receipt_sha256"] == receipt
    and value["manifest_file_sha256"] == manifest_file
    and value["manifest_sha256"] == manifest
    and value["tree_sha256"] == tree
    and value["complete"] is True
):
    raise SystemExit("bootstrap stage-result identity mismatch")
print(expected_root)
PY
)

EP12_STAGED_EXECUTOR=${EP12_STAGED_ROOT}/outputs/executor
for EP12_MODULE_FILE in \
  ep12_executor/post36_receipt_bound.py \
  ep12_executor/scientific_transition_publish.py \
  ep12_executor/scientific_transition_binding.py; do
  [[ -f "${EP12_STAGED_EXECUTOR}/${EP12_MODULE_FILE}" && \
     ! -L "${EP12_STAGED_EXECUTOR}/${EP12_MODULE_FILE}" ]] || \
    ep12_fail "required staged publisher module is missing or unsafe"
done

EP12_LMOD_INIT=/share/software/user/open/lmod/lmod/init/bash
[[ -r "${EP12_LMOD_INIT}" && -f "${EP12_LMOD_INIT}" && ! -L "${EP12_LMOD_INIT}" ]] || \
  ep12_fail "validated Lmod initialization is unavailable"
export MODULEPATH=/share/software/modules/system:/home/groups/russpold/modules:/home/groups/russpold/neurodesk/local/containers/modules:/share/software/modules/math:/share/software/modules/devel:/share/software/modules/categories
export LMOD_SYSTEM_DEFAULT_MODULES=devel,math
source "${EP12_LMOD_INIT}"
type module >/dev/null 2>&1 || ep12_fail "Lmod initialization failed"
module purge
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONBREAKPOINT PYTHONUSERBASE
module load python/3.12.1 \
  py-numpy/1.26.3_py312 \
  py-scipy/1.12.0_py312 \
  py-scikit-learn/1.5.1_py312
EP12_MODULE_PYTHONPATH=${PYTHONPATH:-}
EP12_MODULE_LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
readonly EP12_MODULE_PYTHONPATH EP12_MODULE_LD_LIBRARY_PATH
export PATH=/share/software/user/open/python/3.12.1/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/share/software/user/open/python/3.12.1/lib${EP12_MODULE_LD_LIBRARY_PATH:+:${EP12_MODULE_LD_LIBRARY_PATH}}
export PYTHONPATH=${EP12_STAGED_EXECUTOR}${EP12_MODULE_PYTHONPATH:+:${EP12_MODULE_PYTHONPATH}}
unset HOME PYTHONHOME PYTHONSTARTUP PYTHONBREAKPOINT PYTHONUSERBASE
EP12_RUNTIME_RESULT=${EP12_JOB_TMP}/post36_stage_runtime.json
"${EP12_PYTHON}" -P -B -m ep12_executor.post36_receipt_bound verify-stage-runtime \
  --episode-root "${EP12_ROOT}" \
  --orchestration-intent-path "${EP12_ORCHESTRATION_INTENT}" \
  --expected-intent-file-sha256 "${EP12_ORCHESTRATION_INTENT_FILE_SHA256}" \
  --expected-intent-sha256 "${EP12_ORCHESTRATION_INTENT_SHA256}" \
  --stage-receipt-path "${EP12_STAGE_RECEIPT}" \
  --expected-stage publisher \
  --executed-wrapper "$0" \
  --slurm-job-id "${SLURM_JOB_ID}" \
  --slurm-job-name "${SLURM_JOB_NAME}" \
  --bootstrap-stage-result-path "${EP12_STAGE_RESULT}" \
  --staged-root "${EP12_STAGED_ROOT}" \
  --wrapper-arguments "$@" \
  > "${EP12_RUNTIME_RESULT}"
[[ -s "${EP12_RUNTIME_RESULT}" && ! -L "${EP12_RUNTIME_RESULT}" ]] || \
  ep12_fail "stage-runtime verifier produced no regular result"

"${EP12_PYTHON}" -P -B - "${EP12_STAGED_EXECUTOR}" <<'PY'
from pathlib import Path
import sys
from ep12_executor import post36_receipt_bound
from ep12_executor import scientific_transition_binding
from ep12_executor import scientific_transition_publish

root = Path(sys.argv[1])
for module, relative in {
    post36_receipt_bound: "ep12_executor/post36_receipt_bound.py",
    scientific_transition_binding: "ep12_executor/scientific_transition_binding.py",
    scientific_transition_publish: "ep12_executor/scientific_transition_publish.py",
}.items():
    expected = (root / relative).resolve(strict=True)
    if Path(module.__file__).resolve(strict=True) != expected:
        raise SystemExit(f"live-source import rejected: {module.__name__}")
PY

"${EP12_PYTHON}" -P -B -m ep12_executor.scientific_transition_publish \
  --episode-root "${EP12_ROOT}" \
  --executor-root "${EP12_STAGED_EXECUTOR}" \
  --synthetic-invariance "${EP12_COMPARISON}" \
  --replacement-terminal-identity "${EP12_REPLACEMENT_TERMINAL_IDENTITY}" \
  --post36-start-authorization "${EP12_POST36_START_AUTHORIZATION}" \
  --obsolete-cancellation-receipt "${EP12_OBSOLETE_CANCELLATION_RECEIPT}"
