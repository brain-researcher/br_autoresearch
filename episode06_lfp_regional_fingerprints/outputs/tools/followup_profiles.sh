#!/bin/bash
# Run inside the coordinator's bounded attended Slurm shell; never on login.
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep06_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s\n' "$ep06_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}"
[[ "$ep06_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 1 \
   && "$SLURM_MEM_PER_NODE" == 7168 ]] || exit 2
ep06_project="${OAK:?}/users/zijiao/br_autoresearch"
cd "$ep06_project/episode06_lfp_regional_fingerprints"
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
umask 077
export TMPDIR="${L_SCRATCH_JOB:?}/ep06_followups"
mkdir "$TMPDIR"
ep06_launch=outputs/runs/residual_fingerprint_development/launch002
[[ -d "$ep06_launch" ]] || exit 2
mkdir "$ep06_launch/claim"
ep06_status="$ep06_launch/status_${SLURM_JOB_ID}.txt"
ep06_allocation="$ep06_launch/allocation_${SLURM_JOB_ID}.txt"
TZ=UTC scontrol show job -o "$SLURM_JOB_ID" > "$ep06_allocation"
printf 'episode=EP06\njob=%s\nstatus=started\n' "$SLURM_JOB_ID" > "$ep06_status"
ulimit -S -f 1048576

ep06_rc=0
if (cd outputs/tests && timeout --signal=TERM --kill-after=5s 240s python3.12 -B -m unittest -v \
  test_pca_ridge test_bandwise_families test_primary_profile test_metadata_classifier \
  test_development_interventions test_profile_controls test_run_followup_profiles \
  test_run_development_controls) > "$ep06_launch/checks_${SLURM_JOB_ID}.txt" 2>&1; then :; else ep06_rc=$?; fi
printf 'checks_exit=%s\n' "$ep06_rc" >> "$ep06_status"
(( ep06_rc == 0 )) || exit "$ep06_rc"

ep06_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep06_astra_deps="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
ep06_rc=0
if PYTHONPATH="$ep06_astra/lib/python3.12/site-packages:$ep06_astra_deps" \
  timeout --signal=TERM --kill-after=5s 30s "$ep06_astra/bin/astra-tools" \
  validate outputs/astra/v0.0.14/astra.yaml --json \
  > "$ep06_launch/astra_${SLURM_JOB_ID}.json" 2>&1; then :; else ep06_rc=$?; fi
printf 'astra_exit=%s\n' "$ep06_rc" >> "$ep06_status"

ep06_deadline=$(python3.12 -B - "$ep06_allocation" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
fields = dict(t.split('=', 1) for t in Path(sys.argv[1]).read_text().split() if '=' in t)
if not 1 <= int(fields['NumCPUs']) <= 2:
    raise SystemExit('actual CPU allocation exceeds reservation')
end = datetime.fromisoformat(fields['EndTime']).replace(tzinfo=timezone.utc).timestamp()
clock = json.loads(Path('outputs/runs/residual_fingerprint_development/search_clock.json').read_text())
if clock['wall_clock_limit_seconds'] != 96 * 3600:
    raise SystemExit('original search clock limit differs')
deadline = min(end - 45, clock['started_epoch'] + clock['wall_clock_limit_seconds'])
if deadline - time.time() < 240:
    raise SystemExit('insufficient allocation/search time')
print(int(deadline))
PY
)

# Freeze both successors and the mismatch before any new evaluation score.
for ep06_mode in fit_predict score; do
  for ep06_stage in followup_profiles development_controls; do
    ep06_args=(--mode "$ep06_mode" --deadline-epoch "$ep06_deadline")
    if [[ "$ep06_mode" == fit_predict ]]; then
      ep06_args+=(--views "$SCRATCH/br_autoresearch/episode06_lfp_regional_fingerprints/development001/development_views.h5")
    fi
    ep06_limit=$((ep06_deadline - $(date +%s)))
    (( ep06_limit >= 30 )) || exit 1
    ep06_rc=0
    if timeout --signal=TERM --kill-after=5s "${ep06_limit}s" python3.12 -u -B \
      "outputs/tools/run_${ep06_stage}.py" "${ep06_args[@]}" \
      > "$ep06_launch/${ep06_stage}_${ep06_mode}_${SLURM_JOB_ID}.txt" 2>&1; then :; else ep06_rc=$?; fi
    printf '%s_%s_exit=%s\n' "$ep06_stage" "$ep06_mode" "$ep06_rc" >> "$ep06_status"
    (( ep06_rc == 0 )) || exit "$ep06_rc"
  done
done
printf 'status=followup_batch_complete\n' >> "$ep06_status"
