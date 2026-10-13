#!/bin/bash
#SBATCH --job-name=ep12-post36-tail
#SBATCH --partition=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/outputs/development_run002/slurm-post36-tail-%j.out

# One invocation realizes at most one valid development-only slot. A successful
# continuation submits the same bounded controller as an afterok child; stops,
# failures, and wall-deadline decisions never submit another slot.
# Exact-once launchers use a sanitized environment; initialize Sherlock Lmod
# explicitly before enabling nounset.
# shellcheck source=/etc/profile
. /etc/profile >/dev/null
set -euo pipefail

EP12_ROOT=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_SCRATCH=/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency
EP12_RUN_ROOT=${EP12_ROOT}/outputs/development_run002
EP12_TAIL_SCRIPT=${EP12_ROOT}/outputs/executor/slurm/run_post36_tail.sbatch

cd "${EP12_ROOT}"
module purge
module load python/3.12.1 \
  py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 \
  py-scikit-learn/1.5.1_py312 py-pandas/2.2.1_py312 \
  py-pyarrow/18.1.0_py312
export PYTHONPATH="${EP12_ROOT}/outputs/executor${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "${EP12_SCRATCH}/development_run002/post36_tail"

# BEGIN EP12_PROCEDURE_LOCK_HANDOFF_FUNCTIONS
ep12_validate_procedure_lock_handoff() {
  local handoff_path=$1
  python3 - "${handoff_path}" <<'PY'
import json
import os
import re
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    metadata = os.lstat(path)
except OSError as error:
    raise SystemExit(f"cannot inspect existing procedure-lock handoff: {error}")
if not stat.S_ISREG(metadata.st_mode):
    raise SystemExit("existing procedure-lock handoff is not a regular file")
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as error:
    raise SystemExit(f"cannot read existing procedure-lock handoff: {error}")
expected_keys = {
    "dependency",
    "final_connectivity_access_authorized",
    "final_connectivity_accessed",
    "procedure_lock_job_id",
    "tail_terminal_job_id",
}
if not isinstance(payload, dict) or set(payload) != expected_keys:
    raise SystemExit("existing procedure-lock handoff has an invalid shape")
lock_job_id = payload.get("procedure_lock_job_id")
tail_job_id = payload.get("tail_terminal_job_id")
if not isinstance(lock_job_id, str) or re.fullmatch(r"[1-9][0-9]*", lock_job_id) is None:
    raise SystemExit("existing procedure-lock handoff has an invalid child job ID")
if not isinstance(tail_job_id, str) or re.fullmatch(r"[1-9][0-9]*", tail_job_id) is None:
    raise SystemExit("existing procedure-lock handoff has an invalid parent job ID")
if payload.get("dependency") != f"afterok:{tail_job_id}":
    raise SystemExit("existing procedure-lock handoff has an invalid dependency")
if not (
    payload.get("final_connectivity_accessed") is False
    and payload.get("final_connectivity_access_authorized") is False
):
    raise SystemExit("existing procedure-lock handoff crosses the final-data boundary")
PY
}

ep12_submit_procedure_lock_once() {
  local handoff_path=$1
  local claim_dir=$2
  local procedure_lock_script=$3
  local tail_job_id=$4
  local sbatch_binary=$5

  [[ "${tail_job_id}" =~ ^[1-9][0-9]*$ ]] || {
    echo "procedure-lock handoff requires a numeric Slurm job ID" >&2
    return 2
  }

  # A completed handoff is the idempotent replay result. Validate it before
  # acquiring a claim or invoking sbatch, including if it is a dangling link.
  if [[ -e "${handoff_path}" || -L "${handoff_path}" ]]; then
    ep12_validate_procedure_lock_handoff "${handoff_path}" || return 2
    return 0
  fi

  # mkdir is the single-winner claim. An orphan claim means submission may
  # already have crossed the scheduler boundary, so replay fails closed rather
  # than risking a second procedure-lock job.
  if ! mkdir -- "${claim_dir}"; then
    if [[ -e "${handoff_path}" || -L "${handoff_path}" ]]; then
      ep12_validate_procedure_lock_handoff "${handoff_path}" || return 2
      return 0
    fi
    echo "procedure-lock handoff claim exists without a completed handoff; refusing duplicate submission" >&2
    return 2
  fi

  python3 - "${claim_dir}/claimed.json" "${tail_job_id}" <<'PY'
import json
from pathlib import Path
import sys
from ep12_executor.runtime import publish_once

value = {
    "dependency": f"afterok:{sys.argv[2]}",
    "final_connectivity_access_authorized": False,
    "final_connectivity_accessed": False,
    "phase": "claimed_before_scheduler_submission",
    "tail_terminal_job_id": sys.argv[2],
}
content = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
publish_once(Path(sys.argv[1]), content)
PY

  # Close the migration race with an older invocation that may have published
  # the handoff between our first check and claim acquisition.
  if [[ -e "${handoff_path}" || -L "${handoff_path}" ]]; then
    ep12_validate_procedure_lock_handoff "${handoff_path}" || return 2
    return 0
  fi

  python3 - "${claim_dir}/submission_started.json" "${tail_job_id}" <<'PY'
import json
from pathlib import Path
import sys
from ep12_executor.runtime import publish_once

value = {
    "dependency": f"afterok:{sys.argv[2]}",
    "final_connectivity_access_authorized": False,
    "final_connectivity_accessed": False,
    "phase": "scheduler_submission_started",
    "tail_terminal_job_id": sys.argv[2],
}
content = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
publish_once(Path(sys.argv[1]), content)
PY

  local lock_job_output
  if ! lock_job_output=$(
    /usr/bin/env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C \
      "${sbatch_binary}" --parsable --kill-on-invalid-dep=yes \
      --dependency="afterok:${tail_job_id}" \
      "${procedure_lock_script}"
  ); then
    echo "procedure-lock scheduler submission failed; claim retained to prevent replay" >&2
    return 2
  fi
  [[ "${lock_job_output}" =~ ^([1-9][0-9]*)(\;[A-Za-z0-9_.-]+)?$ ]] || {
    echo "procedure-lock handoff returned an invalid Slurm job ID; claim retained" >&2
    return 2
  }
  local lock_job_id=${BASH_REMATCH[1]}

  python3 - "${handoff_path}" "${lock_job_id}" "${tail_job_id}" <<'PY'
import json
from pathlib import Path
import sys
from ep12_executor.runtime import publish_once

path = Path(sys.argv[1])
value = {
    "dependency": f"afterok:{sys.argv[3]}",
    "final_connectivity_access_authorized": False,
    "final_connectivity_accessed": False,
    "procedure_lock_job_id": sys.argv[2],
    "tail_terminal_job_id": sys.argv[3],
}
content = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
publish_once(path, content)
PY
}
# END EP12_PROCEDURE_LOCK_HANDOFF_FUNCTIONS

# BEGIN EP12_TAIL_CONTINUATION_FUNCTIONS
ep12_validate_tail_continuation() {
  local handoff_path=$1
  local expected_parent_job_id=$2
  local expected_next_slot_id=$3
  python3 - "${handoff_path}" "${expected_parent_job_id}" "${expected_next_slot_id}" <<'PY'
import json
import os
import re
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_parent = sys.argv[2]
expected_next_slot = sys.argv[3]
try:
    metadata = os.lstat(path)
except OSError as error:
    raise SystemExit(f"cannot inspect existing tail-continuation handoff: {error}")
if not stat.S_ISREG(metadata.st_mode):
    raise SystemExit("existing tail-continuation handoff is not a regular file")
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as error:
    raise SystemExit(f"cannot read existing tail-continuation handoff: {error}")
expected_keys = {
    "dependency",
    "final_connectivity_access_authorized",
    "final_connectivity_accessed",
    "next_slot_id",
    "tail_child_job_id",
    "tail_parent_job_id",
}
if not isinstance(payload, dict) or set(payload) != expected_keys:
    raise SystemExit("existing tail-continuation handoff has an invalid shape")
parent = payload.get("tail_parent_job_id")
child = payload.get("tail_child_job_id")
next_slot = payload.get("next_slot_id")
if parent != expected_parent or re.fullmatch(r"[1-9][0-9]*", str(parent)) is None:
    raise SystemExit("existing tail-continuation handoff has the wrong parent")
if not isinstance(child, str) or re.fullmatch(r"[1-9][0-9]*", child) is None:
    raise SystemExit("existing tail-continuation handoff has an invalid child")
if not isinstance(next_slot, str) or not next_slot:
    raise SystemExit("existing tail-continuation handoff has no next slot")
if expected_next_slot != "-" and next_slot != expected_next_slot:
    raise SystemExit("existing tail-continuation handoff has the wrong next slot")
if payload.get("dependency") != f"afterok:{parent}":
    raise SystemExit("existing tail-continuation handoff has an invalid dependency")
if not (
    payload.get("final_connectivity_accessed") is False
    and payload.get("final_connectivity_access_authorized") is False
):
    raise SystemExit("existing tail-continuation handoff crosses the final-data boundary")
PY
}

ep12_submit_tail_continuation_once() {
  local handoff_path=$1
  local claim_dir=$2
  local tail_script=$3
  local parent_job_id=$4
  local next_slot_id=$5
  local sbatch_binary=$6

  [[ "${parent_job_id}" =~ ^[1-9][0-9]*$ ]] || return 2
  [[ -n "${next_slot_id}" ]] || return 2
  if [[ -e "${handoff_path}" || -L "${handoff_path}" ]]; then
    ep12_validate_tail_continuation \
      "${handoff_path}" "${parent_job_id}" "${next_slot_id}" || return 2
    return 0
  fi
  if ! mkdir -- "${claim_dir}"; then
    if [[ -e "${handoff_path}" || -L "${handoff_path}" ]]; then
      ep12_validate_tail_continuation \
        "${handoff_path}" "${parent_job_id}" "${next_slot_id}" || return 2
      return 0
    fi
    echo "tail-continuation claim exists without a completed handoff; refusing duplicate submission" >&2
    return 2
  fi

  python3 - "${claim_dir}/claimed.json" "${parent_job_id}" "${next_slot_id}" <<'PY'
import json
from pathlib import Path
import sys
from ep12_executor.runtime import publish_once

value = {
    "dependency": f"afterok:{sys.argv[2]}",
    "final_connectivity_access_authorized": False,
    "final_connectivity_accessed": False,
    "next_slot_id": sys.argv[3],
    "phase": "claimed_before_scheduler_submission",
    "tail_parent_job_id": sys.argv[2],
}
publish_once(
    Path(sys.argv[1]),
    (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
)
PY
  if [[ -e "${handoff_path}" || -L "${handoff_path}" ]]; then
    ep12_validate_tail_continuation \
      "${handoff_path}" "${parent_job_id}" "${next_slot_id}" || return 2
    return 0
  fi
  python3 - "${claim_dir}/submission_started.json" "${parent_job_id}" "${next_slot_id}" <<'PY'
import json
from pathlib import Path
import sys
from ep12_executor.runtime import publish_once

value = {
    "dependency": f"afterok:{sys.argv[2]}",
    "final_connectivity_access_authorized": False,
    "final_connectivity_accessed": False,
    "next_slot_id": sys.argv[3],
    "phase": "scheduler_submission_started",
    "tail_parent_job_id": sys.argv[2],
}
publish_once(
    Path(sys.argv[1]),
    (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
)
PY

  local child_output
  if ! child_output=$(
    /usr/bin/env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C \
      "${sbatch_binary}" --parsable --kill-on-invalid-dep=yes \
      --dependency="afterok:${parent_job_id}" "${tail_script}"
  ); then
    echo "tail-continuation submission failed; claim retained to prevent replay" >&2
    return 2
  fi
  [[ "${child_output}" =~ ^([1-9][0-9]*)(\;[A-Za-z0-9_.-]+)?$ ]] || {
    echo "tail-continuation submission returned an invalid job ID; claim retained" >&2
    return 2
  }
  local child_job_id=${BASH_REMATCH[1]}
  python3 - "${handoff_path}" "${child_job_id}" "${parent_job_id}" "${next_slot_id}" <<'PY'
import json
from pathlib import Path
import sys
from ep12_executor.runtime import publish_once

value = {
    "dependency": f"afterok:{sys.argv[3]}",
    "final_connectivity_access_authorized": False,
    "final_connectivity_accessed": False,
    "next_slot_id": sys.argv[4],
    "tail_child_job_id": sys.argv[2],
    "tail_parent_job_id": sys.argv[3],
}
publish_once(
    Path(sys.argv[1]),
    (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
)
PY
}
# END EP12_TAIL_CONTINUATION_FUNCTIONS

: "${SLURM_JOB_ID:?post-36 execution requires a Slurm job ID}"
[[ "${SLURM_JOB_ID}" =~ ^[1-9][0-9]*$ ]] || {
  echo "post-36 execution requires a numeric Slurm job ID" >&2
  exit 2
}
EP12_CONTINUATION_ROOT=${EP12_RUN_ROOT}/post36_tail/continuation_launches
EP12_CURRENT_CONTINUATION=${EP12_CONTINUATION_ROOT}/${SLURM_JOB_ID}.json
EP12_CURRENT_CONTINUATION_CLAIM=${EP12_CONTINUATION_ROOT}/${SLURM_JOB_ID}.claim
mkdir -p "${EP12_CONTINUATION_ROOT}"
if [[ -e "${EP12_CURRENT_CONTINUATION}" || -L "${EP12_CURRENT_CONTINUATION}" ]]; then
  ep12_validate_tail_continuation \
    "${EP12_CURRENT_CONTINUATION}" "${SLURM_JOB_ID}" -
  exit 0
fi
if [[ -e "${EP12_CURRENT_CONTINUATION_CLAIM}" || -L "${EP12_CURRENT_CONTINUATION_CLAIM}" ]]; then
  echo "tail-continuation submission is unresolved for this Slurm job; refusing controller replay" >&2
  exit 2
fi

set +e
python3 -m ep12_executor.post36_tail run-one \
  --plan "${EP12_ROOT}/outputs/executor/OBSERVED_POST36_TAIL_PLAN.json" \
  --policy "${EP12_ROOT}/SEARCH_POLICY.yaml" \
  --prefix-contract "${EP12_ROOT}/outputs/executor/SCIENTIFIC_CONTRACT.yaml" \
  --active-contract "${EP12_ROOT}/outputs/executor/SCIENTIFIC_CONTRACT_POST36.yaml" \
  --readiness-dir "${EP12_RUN_ROOT}/production_readiness" \
  --run-root "${EP12_RUN_ROOT}" \
  --observed-finalization-dir "${EP12_RUN_ROOT}/observed_finalization" \
  --tail-root "${EP12_RUN_ROOT}/post36_tail" \
  --snapshot-dir "${EP12_SCRATCH}/development_snapshot_v1" \
  --scratch-root "${EP12_SCRATCH}/development_run002/post36_tail"
EP12_CONTROLLER_RC=$?
set -e

if [[ "${EP12_CONTROLLER_RC}" -eq 10 ]]; then
  EP12_STATUS=${EP12_RUN_ROOT}/post36_tail/controller_status.json
  EP12_NEXT_SLOT=$(python3 - "${EP12_STATUS}" <<'PY'
import json
from pathlib import Path
import sys

status = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
next_slot = status.get("next_slot_id")
if not (
    status.get("action") == "continue"
    and status.get("next_tail_slot_submission_permitted") is True
    and isinstance(next_slot, str)
    and next_slot
    and status.get("final_connectivity_accessed") is False
    and status.get("final_connectivity_access_authorized") is False
):
    raise SystemExit("post-36 status does not authorize a tail continuation")
print(next_slot)
PY
  )
  ep12_submit_tail_continuation_once \
    "${EP12_CURRENT_CONTINUATION}" \
    "${EP12_CURRENT_CONTINUATION_CLAIM}" \
    "${EP12_TAIL_SCRIPT}" \
    "${SLURM_JOB_ID}" \
    "${EP12_NEXT_SLOT}" \
    /usr/bin/sbatch
  exit 0
fi
if [[ "${EP12_CONTROLLER_RC}" -eq 0 ]]; then
  : "${SLURM_JOB_ID:?procedure-lock handoff requires a Slurm job ID}"
  EP12_STATUS=${EP12_RUN_ROOT}/post36_tail/controller_status.json
  EP12_PROCEDURE_LOCK_SCRIPT=${EP12_ROOT}/outputs/executor/slurm/run_procedure_lock.sbatch
  EP12_HANDOFF=${EP12_RUN_ROOT}/post36_tail/procedure_lock_launch.json
  EP12_HANDOFF_CLAIM=${EP12_RUN_ROOT}/post36_tail/procedure_lock_launch.claim

  # The controller's sanitized status is the only authority for this handoff.
  # It contains execution state and gate counts, but no connectivity outcomes.
  python3 - "${EP12_STATUS}" <<'PY'
import json
from pathlib import Path
import sys

status = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not (
    status.get("action") == "stopped"
    and status.get("procedure_lock_submission_permitted") is True
    and status.get("final_connectivity_accessed") is False
    and status.get("final_connectivity_access_authorized") is False
):
    raise SystemExit("post-36 stop does not authorize procedure-lock submission")
PY
  ep12_submit_procedure_lock_once \
    "${EP12_HANDOFF}" \
    "${EP12_HANDOFF_CLAIM}" \
    "${EP12_PROCEDURE_LOCK_SCRIPT}" \
    "${SLURM_JOB_ID}" \
    /usr/bin/sbatch
  exit 0
fi
if [[ "${EP12_CONTROLLER_RC}" -eq 12 ]]; then
  exit 0
fi
exit "${EP12_CONTROLLER_RC}"
