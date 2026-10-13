#!/bin/bash
# Continue only in the same attended task allocation; never submit or renew.
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep06_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s\n' "$ep06_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}"
[[ "$ep06_host" != *-ln* && "$SLURM_JOB_ID" == 47354289 \
   && "$SLURM_CPUS_PER_TASK" == 1 && "$SLURM_MEM_PER_NODE" == 7168 ]] || exit 2
cd "${OAK:?}/users/zijiao/br_autoresearch/episode06_lfp_regional_fingerprints"
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
umask 077
export TMPDIR="${L_SCRATCH_JOB:?}/ep06_robustness"
mkdir "$TMPDIR"
ep06_launch=outputs/runs/residual_fingerprint_development/launch003
mkdir "$ep06_launch/robustness_claim"
ep06_status="$ep06_launch/status_${SLURM_JOB_ID}.txt"
ep06_allocation="$ep06_launch/allocation_${SLURM_JOB_ID}.txt"
printf 'robustness_started_epoch=%s\n' "$(date +%s)" >> "$ep06_status"
ulimit -S -f 1048576

ep06_deadline=$(python3.12 -B - "$ep06_allocation" <<'PY'
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
fields = dict(t.split('=', 1) for t in Path(sys.argv[1]).read_text().split() if '=' in t)
if fields['JobId'] != os.environ['SLURM_JOB_ID'] or not 1 <= int(fields['NumCPUs']) <= 2:
    raise SystemExit('original task allocation required')
end = datetime.fromisoformat(fields['EndTime']).replace(tzinfo=timezone.utc).timestamp()
clock = json.loads(Path('outputs/runs/residual_fingerprint_development/search_clock.json').read_text())
if clock['wall_clock_limit_seconds'] != 96 * 3600:
    raise SystemExit('original search clock limit differs')
deadline = min(end - 45, clock['started_epoch'] + clock['wall_clock_limit_seconds'], time.time() + 600)
if deadline - time.time() < 180:
    raise SystemExit('insufficient attended allocation/search time')
print(int(deadline))
PY
)
ep06_rc=0
if (cd outputs/tests && timeout --signal=TERM --kill-after=5s 90s \
    python3.12 -B -m unittest -v test_run_robustness_controls) \
    > "$ep06_launch/robustness_checks_${SLURM_JOB_ID}.txt" 2>&1; then :; else ep06_rc=$?; fi
printf 'robustness_checks_exit=%s\n' "$ep06_rc" >> "$ep06_status"
(( ep06_rc == 0 )) || exit "$ep06_rc"

ep06_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep06_astra_deps="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
ep06_rc=0
if PYTHONPATH="$ep06_astra/lib/python3.12/site-packages:$ep06_astra_deps" \
  timeout --signal=TERM --kill-after=5s 30s "$ep06_astra/bin/astra-tools" \
  validate outputs/astra/v0.0.14/astra.yaml --json \
  > "$ep06_launch/robustness_astra_${SLURM_JOB_ID}.json" 2>&1; then :; else ep06_rc=$?; fi
printf 'robustness_astra_exit=%s\n' "$ep06_rc" >> "$ep06_status"

ep06_limit=$((ep06_deadline - $(date +%s)))
(( ep06_limit >= 30 )) || exit 1
ep06_rc=0
if timeout --signal=TERM --kill-after=5s "${ep06_limit}s" python3.12 -u -B \
  outputs/tools/run_robustness_controls.py --deadline-epoch "$ep06_deadline" \
  > "$ep06_launch/robustness_${SLURM_JOB_ID}.txt" 2>&1; then :; else ep06_rc=$?; fi
printf 'robustness_exit=%s\nrobustness_ended_epoch=%s\n' "$ep06_rc" "$(date +%s)" >> "$ep06_status"
(( ep06_rc == 0 )) || exit "$ep06_rc"
printf 'status=robustness_batch_complete\n' >> "$ep06_status"
