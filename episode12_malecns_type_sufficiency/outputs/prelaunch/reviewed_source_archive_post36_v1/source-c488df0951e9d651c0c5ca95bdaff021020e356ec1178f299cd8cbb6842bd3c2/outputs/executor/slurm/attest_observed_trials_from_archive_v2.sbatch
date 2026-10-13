#!/bin/bash
#SBATCH --job-name=ep12-attest36-archive-v2
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --export=NONE
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/slurm-attest36-archive-v2-%j.out

set -euo pipefail
umask 077

if [[ $# -ne 15 ]]; then
  echo "expected exactly 15 receipt-bound arguments" >&2
  exit 2
fi
EP12_ROOT=$1
ARCHIVE_ROOT=$2
RECEIPT_FILE_SHA=$3
RECEIPT_SHA=$4
MANIFEST_FILE_SHA=$5
MANIFEST_SHA=$6
TREE_SHA=$7
LAUNCH_INTENT=$8
LAUNCH_INTENT_FILE_SHA=$9
LAUNCH_INTENT_SHA=${10}
ATTESTATION_OUTPUT=${11}
STAGE_RESULT=${12}
STAGED_ROOT=${13}
STAGE_RECEIPT=${14}
TERMINAL_OUTPUT=${15}

CANONICAL_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
SCRATCH_ROOT=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
PYTHON=/share/software/user/open/python/3.12.1/bin/python3.12
PYTHON_LIBRARY=/share/software/user/open/python/3.12.1/lib
BOOTSTRAP_SHA=65f84b2879325e2549cdff7f138aebd3df40db64811327d7f4bd7d50c1bc3aaa
BOOTSTRAP=$ARCHIVE_ROOT/outputs/executor/ep12_executor/source_archive_bootstrap.py

if [[ $EP12_ROOT != "$CANONICAL_ROOT" || ! ${SLURM_JOB_ID:-} =~ ^[1-9][0-9]*$ || -z ${SLURM_JOB_NAME:-} ]]; then
  echo "noncanonical episode root or invalid Slurm job ID" >&2
  exit 2
fi
for VALUE in "$RECEIPT_FILE_SHA" "$RECEIPT_SHA" "$MANIFEST_FILE_SHA" "$MANIFEST_SHA" "$TREE_SHA" "$LAUNCH_INTENT_FILE_SHA" "$LAUNCH_INTENT_SHA"; do
  if [[ ! $VALUE =~ ^[0-9a-f]{64}$ ]]; then
    echo "invalid lowercase SHA-256 argument" >&2
    exit 2
  fi
done
if [[ ! -f $PYTHON || -L $PYTHON || ! -x $PYTHON || ! -d $PYTHON_LIBRARY || -L $PYTHON_LIBRARY ]]; then
  echo "trusted Python 3.12.1 is unavailable" >&2
  exit 2
fi

unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONBREAKPOINT VIRTUAL_ENV CONDA_PREFIX HOME LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PYTHON_LIBRARY
export LANG=C
export LC_ALL=C
export PATH=/usr/bin:/bin
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export TZ=UTC

if [[ $("$PYTHON" --version 2>&1) != "Python 3.12.1" ]]; then
  echo "trusted Python version changed" >&2
  exit 2
fi

TMP_PARENT=$SCRATCH_ROOT/archive_wrapper_tmp
JOB_TMP=$("$PYTHON" -I -S -c '
import os, pathlib, stat, sys, tempfile
p = pathlib.Path(sys.argv[1])
p.mkdir(mode=0o700, parents=True, exist_ok=True)
m = os.lstat(p)
if stat.S_ISLNK(m.st_mode) or not stat.S_ISDIR(m.st_mode):
    raise SystemExit("temporary parent is unsafe")
if m.st_uid != os.geteuid() or stat.S_IMODE(m.st_mode) != 0o700:
    raise SystemExit("temporary parent owner/mode is unsafe")
print(tempfile.mkdtemp(prefix="attest-", dir=p))
' "$TMP_PARENT")
BOOTSTRAP_COPY=$JOB_TMP/source_archive_bootstrap.py
STAGE_TMP=$JOB_TMP/source_archive_stage.json

"$PYTHON" -I -S - "$BOOTSTRAP" "$BOOTSTRAP_SHA" "$LAUNCH_INTENT" "$LAUNCH_INTENT_FILE_SHA" "$LAUNCH_INTENT_SHA" "$STAGE_RECEIPT" "$0" "$SLURM_JOB_ID" "$SLURM_JOB_NAME" "$BOOTSTRAP_COPY" "$@" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

def fail(message):
    raise SystemExit(message)

def read_regular(raw, label):
    path = Path(os.path.abspath(raw))
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        value = os.lstat(current)
        if stat.S_ISLNK(value.st_mode):
            fail(f"{label} traverses a symlink")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            fail(f"{label} is not regular")
        chunks = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                return b"".join(chunks)
            chunks.append(block)
    finally:
        os.close(fd)

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

bootstrap, bootstrap_sha, intent_path, intent_file_sha, intent_sha = sys.argv[1:6]
receipt_path, wrapper_path, job_id, job_name, bootstrap_copy = sys.argv[6:11]
wrapper_args = sys.argv[11:]
if len(wrapper_args) != 15:
    fail("wrapper argument count changed")
episode_root = Path(wrapper_args[0])
canonical_receipt = episode_root / "outputs/prelaunch/replacement_attestation_launch/stages/replacement_attestation/receipt.json"
if Path(os.path.abspath(receipt_path)) != canonical_receipt:
    fail("replacement stage receipt path is not canonical")

bootstrap_bytes = read_regular(bootstrap, "archive bootstrap")
if hashlib.sha256(bootstrap_bytes).hexdigest() != bootstrap_sha:
    fail("archive bootstrap hash mismatch")

intent_bytes = read_regular(intent_path, "replacement launch intent")
if hashlib.sha256(intent_bytes).hexdigest() != intent_file_sha:
    fail("replacement launch intent file hash mismatch")
try:
    intent = json.loads(intent_bytes)
except Exception as exc:
    fail(f"replacement launch intent is malformed: {exc}")
if intent_bytes != (canonical(intent) + "\n").encode("ascii"):
    fail("replacement launch intent is not canonical")
intent_core = {key: value for key, value in intent.items() if key != "intent_sha256"}
if intent.get("intent_sha256") != intent_sha:
    fail("replacement launch intent semantic argument changed")
if hashlib.sha256(canonical(intent_core).encode("ascii")).hexdigest() != intent_sha:
    fail("replacement launch intent semantic hash mismatch")

receipt_bytes = read_regular(receipt_path, "replacement stage receipt")
receipt_file_sha = hashlib.sha256(receipt_bytes).hexdigest()
try:
    receipt = json.loads(receipt_bytes)
except Exception as exc:
    fail(f"replacement stage receipt is malformed: {exc}")
if receipt_bytes != (canonical(receipt) + "\n").encode("ascii"):
    fail("replacement stage receipt is not canonical")
receipt_core = {key: value for key, value in receipt.items() if key != "stage_receipt_sha256"}
if hashlib.sha256(canonical(receipt_core).encode("ascii")).hexdigest() != receipt.get("stage_receipt_sha256"):
    fail("replacement stage receipt semantic hash mismatch")
if (receipt.get("slurm_job_id") != job_id
        or receipt.get("slurm_job_name") != job_name):
    fail("replacement stage receipt Slurm identity mismatch")
if receipt.get("submission_barrier") != "user_hold_receipt_before_release_v1":
    fail("replacement stage receipt lacks user-hold barrier")
if receipt.get("arguments") != wrapper_args:
    fail("replacement stage receipt arguments changed")
wrapper_bytes = read_regular(wrapper_path, "spooled batch script")
if hashlib.sha256(wrapper_bytes).hexdigest() != receipt.get("wrapper_sha256"):
    fail("spooled batch script hash mismatch")
command = receipt.get("sbatch_command")
if not isinstance(command, list) or "--hold" not in command or "--export=NONE" not in command:
    fail("replacement sbatch command lacks hold or sanitized export")
if command[-16] != receipt.get("wrapper") or command[-15:] != wrapper_args:
    fail("replacement sbatch command suffix changed")

stage_intent_path = str(Path(receipt_path).with_name("intent.json"))
stage_intent_bytes = read_regular(stage_intent_path, "replacement stage intent")
try:
    stage_intent = json.loads(stage_intent_bytes)
except Exception as exc:
    fail(f"replacement stage intent is malformed: {exc}")
if stage_intent_bytes != (canonical(stage_intent) + "\n").encode("ascii"):
    fail("replacement stage intent is not canonical")
stage_core = {key: value for key, value in stage_intent.items() if key != "stage_intent_sha256"}
if hashlib.sha256(canonical(stage_core).encode("ascii")).hexdigest() != stage_intent.get("stage_intent_sha256"):
    fail("replacement stage intent semantic hash mismatch")
if stage_intent.get("stage_intent_sha256") != receipt.get("stage_intent_sha256"):
    fail("replacement stage intent/receipt mismatch")
if stage_intent.get("orchestration_intent_sha256") != intent_sha:
    fail("replacement stage intent/launch intent mismatch")
if stage_intent.get("arguments") != wrapper_args:
    fail("replacement stage intent arguments changed")

def read_canonical_artifact(path, label, hash_field):
    encoded = read_regular(str(path), label)
    try:
        value = json.loads(encoded)
    except Exception as exc:
        fail(f"{label} is malformed: {exc}")
    if encoded != (canonical(value) + "\n").encode("ascii"):
        fail(f"{label} is not canonical")
    core = {key: item for key, item in value.items() if key != hash_field}
    if hashlib.sha256(canonical(core).encode("ascii")).hexdigest() != value.get(hash_field):
        fail(f"{label} semantic hash mismatch")
    return value

release_intent_path = canonical_receipt.with_name("release_intent.json")
release_authorization_path = canonical_receipt.with_name("release_authorization.json")
release_intent = read_canonical_artifact(
    release_intent_path, "replacement release intent", "release_intent_sha256"
)
release_authorization = read_canonical_artifact(
    release_authorization_path,
    "replacement release authorization",
    "release_authorization_sha256",
)
if (set(release_intent) != {
        "schema_version", "stage", "submission_user", "slurm_job_id",
        "slurm_job_name", "stage_receipt_file_sha256", "stage_receipt_sha256",
        "release_command", "authorization_context",
        "receipt_was_durable_before_release",
        "final_connectivity_accessed", "release_intent_sha256",
    }
        or release_intent.get("schema_version") != "ep12.post36_stage_release_intent.v1"
        or release_intent.get("stage") != "replacement_attestation"
        or release_intent.get("submission_user") != receipt.get("submission_user")
        or release_intent.get("slurm_job_id") != job_id
        or release_intent.get("slurm_job_name") != job_name
        or release_intent.get("stage_receipt_file_sha256") != receipt_file_sha
        or release_intent.get("stage_receipt_sha256") != receipt.get("stage_receipt_sha256")
        or release_intent.get("release_command") != ["/usr/bin/scontrol", "release", job_id]
        or release_intent.get("receipt_was_durable_before_release") is not True
        or release_intent.get("final_connectivity_accessed") is not False):
    fail("replacement release intent lineage changed")
release_context = release_intent.get("authorization_context")
if (not isinstance(release_context, dict)
        or set(release_context) != {
            "schema_version", "start_authorization_path",
            "start_authorization_file_sha256", "start_authorization_sha256",
        }
        or release_context.get("schema_version") != "ep12.post36_replacement_release_context.v1"):
    fail("replacement release context schema changed")
start_path = episode_root / "outputs/prelaunch/post36_orchestration/start_authorization.json"
if Path(os.path.abspath(release_context.get("start_authorization_path", ""))) != start_path:
    fail("post36 start authorization path is not canonical")
start_bytes = read_regular(str(start_path), "post36 start authorization")
if hashlib.sha256(start_bytes).hexdigest() != release_context.get("start_authorization_file_sha256"):
    fail("post36 start authorization file hash changed")
try:
    start_authorization = json.loads(start_bytes)
except Exception as exc:
    fail(f"post36 start authorization is malformed: {exc}")
if start_bytes != (canonical(start_authorization) + "\n").encode("ascii"):
    fail("post36 start authorization is not canonical")
start_core = {
    key: value for key, value in start_authorization.items()
    if key != "start_authorization_sha256"
}
if (start_authorization.get("schema_version") != "ep12.post36_start_authorization.v1"
        or hashlib.sha256(canonical(start_core).encode("ascii")).hexdigest()
        != start_authorization.get("start_authorization_sha256")
        or start_authorization.get("start_authorization_sha256")
        != release_context.get("start_authorization_sha256")
        or start_authorization.get("replacement_job_id") != job_id
        or start_authorization.get("replacement_stage_receipt_path") != receipt_path
        or start_authorization.get("replacement_stage_receipt_file_sha256") != receipt_file_sha
        or start_authorization.get("replacement_stage_receipt_sha256") != receipt.get("stage_receipt_sha256")
        or start_authorization.get("all_child_stage_and_release_receipts_durable") is not True
        or start_authorization.get("obsolete_job_terminally_cancelled") is not True
        or start_authorization.get("replacement_may_be_released") is not True
        or start_authorization.get("obsolete_job_is_execution_dependency") is not False
        or start_authorization.get("values_used_for_transition_decisions") is not False
        or start_authorization.get("final_connectivity_accessed") is not False
        or start_authorization.get("final_connectivity_access_authorized") is not False):
    fail("post36 start authorization identity changed")
if (set(release_authorization) != {
        "schema_version", "stage", "submission_user", "slurm_job_id",
        "slurm_job_name", "stage_receipt_file_sha256", "stage_receipt_sha256",
        "release_intent_sha256", "release_command",
        "authorization_persisted_before_first_scontrol_byte",
        "duplicate_release_after_uncertain_ack_allowed",
        "final_connectivity_accessed", "release_authorization_sha256",
    }
        or release_authorization.get("schema_version") != "ep12.post36_stage_release_authorization.v1"
        or release_authorization.get("stage") != "replacement_attestation"
        or release_authorization.get("submission_user") != receipt.get("submission_user")
        or release_authorization.get("slurm_job_id") != job_id
        or release_authorization.get("slurm_job_name") != job_name
        or release_authorization.get("stage_receipt_file_sha256") != receipt_file_sha
        or release_authorization.get("stage_receipt_sha256") != receipt.get("stage_receipt_sha256")
        or release_authorization.get("release_intent_sha256") != release_intent.get("release_intent_sha256")
        or release_authorization.get("release_command") != ["/usr/bin/scontrol", "release", job_id]
        or release_authorization.get("authorization_persisted_before_first_scontrol_byte") is not True
        or release_authorization.get("duplicate_release_after_uncertain_ack_allowed") is not False
        or release_authorization.get("final_connectivity_accessed") is not False):
    fail("replacement release authorization lineage changed")

tree_sha = wrapper_args[6]
if wrapper_args[7] != intent_path or wrapper_args[8] != intent_file_sha or wrapper_args[9] != intent_sha:
    fail("replacement wrapper launch-intent arguments changed")
if wrapper_args[10] != str(episode_root / "outputs/development_run002/observed_attestation"):
    fail("replacement attestation output changed")
if wrapper_args[11] != str(episode_root / "outputs/prelaunch/replacement_attestation_launch/bootstrap_stage_result.json"):
    fail("replacement bootstrap result path changed")
expected_stage_parent = Path("/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/source_archive_stage") / tree_sha
if wrapper_args[12] != str(expected_stage_parent / ("source-" + tree_sha)):
    fail("replacement staged root changed")
if wrapper_args[13] != receipt_path:
    fail("replacement stage receipt argument changed")
if wrapper_args[14] != str(episode_root / "outputs/prelaunch/replacement_attestation_launch/terminal_identity.json"):
    fail("replacement terminal output changed")
expected_stage_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
stage_parent_stat = os.lstat(expected_stage_parent)
if (stat.S_ISLNK(stage_parent_stat.st_mode)
        or not stat.S_ISDIR(stage_parent_stat.st_mode)
        or stage_parent_stat.st_uid != os.geteuid()
        or stat.S_IMODE(stage_parent_stat.st_mode) != 0o700):
    fail("replacement stage parent owner/mode is unsafe")

target = Path(bootstrap_copy)
fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o400)
try:
    view = memoryview(bootstrap_bytes)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            fail("short bootstrap copy write")
        view = view[written:]
    os.fsync(fd)
finally:
    os.close(fd)
PY

"$PYTHON" -I -S "$BOOTSTRAP_COPY" verify-and-stage   --archive-root "$ARCHIVE_ROOT"   --stage-parent "$(dirname "$STAGED_ROOT")"   --expected-receipt-file-sha256 "$RECEIPT_FILE_SHA"   --expected-receipt-sha256 "$RECEIPT_SHA"   --expected-manifest-file-sha256 "$MANIFEST_FILE_SHA"   --expected-manifest-sha256 "$MANIFEST_SHA"   --expected-tree-sha256 "$TREE_SHA" > "$STAGE_TMP"

STAGE_RESULT_FILE_SHA=$("$PYTHON" -I -S - "$STAGE_TMP" "$STAGE_RESULT" "$STAGED_ROOT" "$ARCHIVE_ROOT" "$RECEIPT_FILE_SHA" "$RECEIPT_SHA" "$MANIFEST_FILE_SHA" "$MANIFEST_SHA" "$TREE_SHA" <<'PY'
import errno
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import sys

source, target, staged_root, archive_root, receipt_file, receipt, manifest_file, manifest, tree = sys.argv[1:]
encoded = Path(source).read_bytes()
value = json.loads(encoded)
canonical = (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
if encoded != canonical:
    raise SystemExit("bootstrap stage result is not canonical")
expected = {
    "schema_version": "ep12.source_archive_bootstrap_stage.v1",
    "staged_root": staged_root,
    "upstream_archived_root": archive_root,
    "archive_receipt_file_sha256": receipt_file,
    "archive_receipt_sha256": receipt,
    "manifest_file_sha256": manifest_file,
    "manifest_sha256": manifest,
    "tree_sha256": tree,
    "complete": True,
}
if set(value) != set(expected) | {"stage_status"} or any(value.get(k) != v for k, v in expected.items()):
    raise SystemExit("bootstrap stage result identity mismatch")
if value.get("stage_status") not in {"created", "recovered_identical", "reused_identical"}:
    raise SystemExit("bootstrap stage result status changed")
path = Path(os.path.abspath(target))
current = Path(path.anchor)
for part in path.parts[1:-1]:
    current /= part
    metadata = os.lstat(current)
    if stat.S_ISLNK(metadata.st_mode):
        raise SystemExit("bootstrap stage-result target traverses a symlink")
temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o640)
try:
    view = memoryview(encoded)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise SystemExit("short bootstrap stage-result write")
        view = view[written:]
    os.fsync(fd)
finally:
    os.close(fd)
try:
    os.link(temporary, path, follow_symlinks=False)
except FileExistsError:
    existing_fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if not stat.S_ISREG(os.fstat(existing_fd).st_mode) or os.read(existing_fd, len(encoded) + 1) != encoded:
            raise SystemExit("bootstrap stage-result target conflicts")
    finally:
        os.close(existing_fd)
finally:
    os.unlink(temporary)
directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
print(hashlib.sha256(encoded).hexdigest())
PY
)

STAGED_EXECUTOR=$STAGED_ROOT/outputs/executor
STAGED_POLICY=$STAGED_ROOT/SEARCH_POLICY.yaml
if [[ ! -d $STAGED_EXECUTOR || -L $STAGED_EXECUTOR || ! -f $STAGED_POLICY || -L $STAGED_POLICY ]]; then
  echo "verified staged source layout is incomplete" >&2
  exit 2
fi
export PYTHONPATH=$STAGED_EXECUTOR

"$PYTHON" -P -B -c 'from ep12_executor.replacement_attestation_terminal import main; raise SystemExit(main())' attest-and-publish   --episode-root "$EP12_ROOT"   --launch-intent "$LAUNCH_INTENT"   --launch-intent-file-sha256 "$LAUNCH_INTENT_FILE_SHA"   --launch-intent-sha256 "$LAUNCH_INTENT_SHA"   --attestation-output "$ATTESTATION_OUTPUT"   --terminal-identity-output "$TERMINAL_OUTPUT"   --bootstrap-stage-result "$STAGE_RESULT"   --bootstrap-stage-result-file-sha256 "$STAGE_RESULT_FILE_SHA"   --staged-root "$STAGED_ROOT"   --stage-receipt "$STAGE_RECEIPT"   --executed-wrapper "$0"   --slurm-job-id "$SLURM_JOB_ID"   --slurm-job-name "$SLURM_JOB_NAME"
