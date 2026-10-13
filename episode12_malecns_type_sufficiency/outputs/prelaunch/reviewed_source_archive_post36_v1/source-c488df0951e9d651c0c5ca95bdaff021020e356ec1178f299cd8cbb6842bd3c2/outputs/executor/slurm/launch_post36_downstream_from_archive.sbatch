#!/bin/bash
#SBATCH --job-name=ep12-post36-downstream-archive
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --export=NONE
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/slurm-post36-downstream-archive-%j.out

set -euo pipefail
umask 077

ep12_fail() {
  echo "post36 archive downstream wrapper: $*" >&2
  exit 2
}

ep12_require_sha256() {
  [[ "$2" =~ ^[0-9a-f]{64}$ ]] || ep12_fail "$1 is not a lowercase SHA-256"
}

[[ $# -eq 12 ]] || ep12_fail "expected exactly 12 positional arguments"

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
EP12_PUBLISHER_JOB_ID=${12}
readonly EP12_ROOT EP12_ARCHIVE_ROOT EP12_ARCHIVE_RECEIPT_FILE_SHA256
readonly EP12_ARCHIVE_RECEIPT_SHA256 EP12_ARCHIVE_MANIFEST_FILE_SHA256
readonly EP12_ARCHIVE_MANIFEST_SHA256 EP12_ARCHIVE_TREE_SHA256
readonly EP12_ORCHESTRATION_INTENT EP12_ORCHESTRATION_INTENT_FILE_SHA256
readonly EP12_ORCHESTRATION_INTENT_SHA256 EP12_STAGE_RECEIPT EP12_PUBLISHER_JOB_ID

EP12_CANONICAL_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_SCRATCH=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_BOOTSTRAP_SHA256=65f84b2879325e2549cdff7f138aebd3df40db64811327d7f4bd7d50c1bc3aaa
EP12_PYTHON=/share/software/user/open/python/3.12.1/bin/python3.12
readonly EP12_CANONICAL_ROOT EP12_SCRATCH
readonly EP12_BOOTSTRAP_SHA256 EP12_PYTHON

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
[[ "${EP12_PUBLISHER_JOB_ID}" =~ ^[1-9][0-9]*$ ]] || ep12_fail "publisher job ID is invalid"
[[ "${EP12_ORCHESTRATION_INTENT}" == \
  "${EP12_ROOT}/outputs/prelaunch/post36_orchestration/intent.json" ]] || \
  ep12_fail "orchestration intent path is not canonical"
[[ "${EP12_STAGE_RECEIPT}" == \
  "${EP12_ROOT}/outputs/prelaunch/post36_orchestration/stages/downstream/receipt.json" ]] || \
  ep12_fail "downstream stage receipt path is not canonical"
for EP12_INPUT in "${EP12_ORCHESTRATION_INTENT}" "${EP12_STAGE_RECEIPT}"; do
  [[ -f "${EP12_INPUT}" && ! -L "${EP12_INPUT}" ]] || \
    ep12_fail "required downstream input is missing, nonregular, or a symlink"
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
EP12_STAGE_PARENT=${EP12_SCRATCH}/source_archive_stage/${EP12_ARCHIVE_TREE_SHA256}
EP12_TMP_PARENT=${EP12_SCRATCH}/archive_wrapper_tmp
/usr/bin/mkdir -p -m 0700 "${EP12_STAGE_PARENT}" "${EP12_TMP_PARENT}"
[[ -d "${EP12_STAGE_PARENT}" && ! -L "${EP12_STAGE_PARENT}" ]] || \
  ep12_fail "archive stage parent is unsafe"
EP12_JOB_TMP=$(/usr/bin/mktemp -d "${EP12_TMP_PARENT}/downstream-${SLURM_JOB_ID}.XXXXXX")
export TMPDIR=${EP12_JOB_TMP}
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
  ep12_executor/source_archive_freeze_bridge.py \
  ep12_executor/post36_downstream_launch.py; do
  [[ -f "${EP12_STAGED_EXECUTOR}/${EP12_MODULE_FILE}" && \
     ! -L "${EP12_STAGED_EXECUTOR}/${EP12_MODULE_FILE}" ]] || \
    ep12_fail "required staged downstream module is missing or unsafe"
done
export PYTHONPATH=${EP12_STAGED_EXECUTOR}
EP12_RUNTIME_RESULT=${EP12_JOB_TMP}/post36_stage_runtime.json
"${EP12_PYTHON}" -P -B -m ep12_executor.post36_receipt_bound verify-stage-runtime \
  --episode-root "${EP12_ROOT}" \
  --orchestration-intent-path "${EP12_ORCHESTRATION_INTENT}" \
  --expected-intent-file-sha256 "${EP12_ORCHESTRATION_INTENT_FILE_SHA256}" \
  --expected-intent-sha256 "${EP12_ORCHESTRATION_INTENT_SHA256}" \
  --stage-receipt-path "${EP12_STAGE_RECEIPT}" \
  --expected-stage downstream \
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
from ep12_executor import post36_downstream_launch
from ep12_executor import post36_receipt_bound
from ep12_executor import source_archive_freeze_bridge

root = Path(sys.argv[1])
for module, relative in {
    post36_downstream_launch: "ep12_executor/post36_downstream_launch.py",
    post36_receipt_bound: "ep12_executor/post36_receipt_bound.py",
    source_archive_freeze_bridge: "ep12_executor/source_archive_freeze_bridge.py",
}.items():
    expected = (root / relative).resolve(strict=True)
    if Path(module.__file__).resolve(strict=True) != expected:
        raise SystemExit(f"live-source import rejected: {module.__name__}")
PY

EP12_BRIDGE_RESULT=${EP12_JOB_TMP}/source_archive_freeze_bridge.json
"${EP12_PYTHON}" -P -B -m ep12_executor.source_archive_freeze_bridge \
  --archive-root "${EP12_ARCHIVE_ROOT}" \
  --archive-receipt-file-sha256 "${EP12_ARCHIVE_RECEIPT_FILE_SHA256}" \
  --archive-receipt-sha256 "${EP12_ARCHIVE_RECEIPT_SHA256}" \
  --archive-manifest-file-sha256 "${EP12_ARCHIVE_MANIFEST_FILE_SHA256}" \
  --archive-manifest-sha256 "${EP12_ARCHIVE_MANIFEST_SHA256}" \
  --archive-tree-sha256 "${EP12_ARCHIVE_TREE_SHA256}" \
  --live-episode-root "${EP12_ROOT}" \
  --require-canonical-live-root \
  > "${EP12_BRIDGE_RESULT}"

"${EP12_PYTHON}" -I -S - \
  "${EP12_BRIDGE_RESULT}" "${EP12_ROOT}" "${EP12_ARCHIVE_ROOT}" \
  "${EP12_ARCHIVE_RECEIPT_FILE_SHA256}" "${EP12_ARCHIVE_RECEIPT_SHA256}" \
  "${EP12_ARCHIVE_MANIFEST_FILE_SHA256}" "${EP12_ARCHIVE_MANIFEST_SHA256}" \
  "${EP12_ARCHIVE_TREE_SHA256}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

result_path, root, archive, receipt_file, receipt, manifest_file, manifest, tree = sys.argv[1:]
encoded = Path(result_path).read_bytes()
value = json.loads(encoded.decode("utf-8"))
if encoded != (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii"):
    raise SystemExit("source-freeze bridge output is not canonical JSON")
archive_identity = value.get("archive_identity")
freeze = value.get("source_freeze_identity")
receipt_path = Path(root) / "outputs/prelaunch/source_archive_freeze_bridge/receipt.json"
freeze_path = Path(root) / "outputs/prelaunch/executable_source_freeze_manifest.json"
if not (
    value.get("publication_status") in {"created", "reused_identical"}
    and value.get("receipt_path") == str(receipt_path)
    and value.get("live_episode_root") == root
    and isinstance(archive_identity, dict)
    and archive_identity.get("archived_root") == archive
    and archive_identity.get("archive_receipt_file_sha256") == receipt_file
    and archive_identity.get("archive_receipt_sha256") == receipt
    and archive_identity.get("manifest_file_sha256") == manifest_file
    and archive_identity.get("manifest_sha256") == manifest
    and archive_identity.get("tree_sha256") == tree
    and isinstance(freeze, dict)
    and freeze.get("schema_version") == "ep12.executable_source_freeze_manifest.v1"
    and freeze.get("manifest_path") == str(freeze_path)
    and value.get("archive_and_live_file_maps_identical") is True
    and value.get("archive_reverified_after_freeze") is True
    and value.get("live_surface_reverified_after_freeze") is True
    and value.get("real_connectivity_accessed") is False
    and value.get("final_connectivity_access_authorized") is False
    and receipt_path.is_file()
    and not receipt_path.is_symlink()
    and hashlib.sha256(receipt_path.read_bytes()).hexdigest() == value.get("receipt_file_sha256")
):
    raise SystemExit("source-freeze bridge identity mismatch")
PY

"${EP12_PYTHON}" -P -B -m ep12_executor.post36_downstream_launch \
  --episode-root "${EP12_ROOT}" \
  --executor-root "${EP12_STAGED_EXECUTOR}" \
  --orchestration-intent "${EP12_ORCHESTRATION_INTENT}" \
  --orchestration-intent-sha256 "${EP12_ORCHESTRATION_INTENT_SHA256}" \
  --publisher-job-id "${EP12_PUBLISHER_JOB_ID}" \
  --coordinator-job-id "${SLURM_JOB_ID}"
