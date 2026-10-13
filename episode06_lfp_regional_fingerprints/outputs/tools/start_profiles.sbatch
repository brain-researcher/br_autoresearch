#!/bin/bash
# First twelve EP06 development comparisons; one separately scoped EP07 figure.
#SBATCH --job-name=ep06-starts
#SBATCH --partition=russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/residual_fingerprint_development/launch001/slurm-%j.out
#SBATCH --error=outputs/runs/residual_fingerprint_development/launch001/slurm-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep06_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s\n' "$ep06_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}"
[[ "$ep06_host" != *-ln* && "$SLURM_CPUS_PER_TASK" -ge 1 \
   && "$SLURM_CPUS_PER_TASK" -le 2 && "$SLURM_MEM_PER_NODE" == 7168 ]] || exit 2
ep06_project="${OAK:?}/users/zijiao/br_autoresearch"
cd "$ep06_project/episode06_lfp_regional_fingerprints"
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312 viz py-matplotlib/3.8.3_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
umask 077
export TMPDIR="${L_SCRATCH_JOB:?}/ep06_starts"
mkdir "$TMPDIR"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
mkdir "$MPLCONFIGDIR"
ep06_launch="outputs/runs/residual_fingerprint_development/launch001"
ep06_attempt="$ep06_project/episode06_lfp_regional_fingerprints/outputs/runs/residual_fingerprint_development/starting001"
[[ -d "$ep06_launch" && ! -e "$ep06_attempt" ]] || exit 2
mkdir "$ep06_launch/claim"
ep06_status="$ep06_launch/status_${SLURM_JOB_ID}.txt"
ep06_allocation="$ep06_launch/allocation_${SLURM_JOB_ID}.txt"
TZ=UTC scontrol show job -o "$SLURM_JOB_ID" > "$ep06_allocation"
ep06_cpus=$(awk '{for (i=1;i<=NF;i++) if ($i ~ /^NumCPUs=/) {split($i,a,"="); print a[2]; exit}}' "$ep06_allocation")
[[ "$ep06_cpus" -ge 1 && "$ep06_cpus" -le 2 ]] || exit 2
printf 'episode=EP06\njob=%s\nallocated_cpus=%s\nstatus=started\n' \
  "$SLURM_JOB_ID" "$ep06_cpus" > "$ep06_status"
ulimit -S -f 1048576

if (cd outputs/tests && timeout --signal=TERM --kill-after=5s 240s \
  python3.12 -B -m unittest -v test_bandwise_prediction test_primary_profile \
  test_cca_mapping test_bandwise_families test_profile_references test_run_initial_profiles) \
  > "$ep06_launch/component_checks_${SLURM_JOB_ID}.txt" 2>&1; then
  printf 'component_checks_exit=0\n' >> "$ep06_status"
else
  ep06_rc=$?
  printf 'component_checks_exit=%s\nstatus=stopped_before_neural_fitting\n' "$ep06_rc" >> "$ep06_status"
  exit "$ep06_rc"
fi

ep06_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep06_astra_deps="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
for ep06_scope in episode06_lfp_regional_fingerprints episode07_lfp_session_transfer; do
  ep06_rc=0
  if (cd "$ep06_project/$ep06_scope" && \
      PYTHONPATH="$ep06_astra/lib/python3.12/site-packages:$ep06_astra_deps" \
      timeout --signal=TERM --kill-after=5s 30s "$ep06_astra/bin/astra-tools" \
      validate outputs/astra/v0.0.14/astra.yaml --json) \
      > "$ep06_launch/astra_${ep06_scope}_${SLURM_JOB_ID}.json" 2>&1; then :; else ep06_rc=$?; fi
  printf 'astra_%s_exit=%s\n' "$ep06_scope" "$ep06_rc" >> "$ep06_status"
done

# EP07 consumes only its completed report in this explicitly separate scope.
printf 'ep07_figure_started_epoch=%s\n' "$(date +%s)" >> "$ep06_status"
ep07_rc=0
if (cd "$ep06_project/episode07_lfp_session_transfer" && \
    timeout --signal=TERM --kill-after=5s 90s python3.12 -B outputs/tools/plot_primary_result.py) \
    > "$ep06_launch/ep07_figure_${SLURM_JOB_ID}.txt" 2>&1; then :; else ep07_rc=$?; fi
printf 'ep07_figure_exit=%s\nep07_figure_ended_epoch=%s\n' "$ep07_rc" "$(date +%s)" >> "$ep06_status"

# Start the unchanged 96-hour search clock immediately before the first fit.
# This clock persists across later allocations and never resets on a retry.
ep06_deadline=$(python3.12 -B - "$ep06_allocation" <<'PY'
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
fields = dict(t.split('=', 1) for t in Path(sys.argv[1]).read_text().split() if '=' in t)
end = datetime.fromisoformat(fields['EndTime']).replace(tzinfo=timezone.utc).timestamp()
clock_path = Path('outputs/runs/residual_fingerprint_development/search_clock.json')
if clock_path.exists():
    clock = json.loads(clock_path.read_text())
    if clock['wall_clock_limit_seconds'] != 96 * 3600:
        raise SystemExit('original search clock limit differs')
else:
    now = time.time()
    if end - now < 300:
        raise SystemExit('insufficient allocation time before first fit')
    clock = {'episode': 'EP06', 'started_epoch': now,
             'started_utc': datetime.fromtimestamp(now, timezone.utc).isoformat(),
             'wall_clock_limit_seconds': 96 * 3600,
             'first_job_id': os.environ['SLURM_JOB_ID']}
    with clock_path.open('x') as handle:
        json.dump(clock, handle, indent=2)
        handle.write('\n')
deadline = min(end - 45, clock['started_epoch'] + clock['wall_clock_limit_seconds'])
if deadline - time.time() < 240:
    raise SystemExit('insufficient remaining allocation/search time')
print(int(deadline))
PY
)
for ep06_mode in fit_predict score; do
  ep06_args=(--mode "$ep06_mode" --attempt-dir "$ep06_attempt" --deadline-epoch "$ep06_deadline")
  if [[ "$ep06_mode" == fit_predict ]]; then
    ep06_args+=(--views "$SCRATCH/br_autoresearch/episode06_lfp_regional_fingerprints/development001/development_views.h5")
  fi
  ep06_limit=$((ep06_deadline - $(date +%s)))
  (( ep06_limit >= 30 )) || exit 1
  ep06_rc=0
  if timeout --signal=TERM --kill-after=5s "${ep06_limit}s" python3.12 -u -B \
      outputs/tools/run_initial_profiles.py "${ep06_args[@]}" \
      > "$ep06_launch/${ep06_mode}_${SLURM_JOB_ID}.txt" 2>&1; then :; else ep06_rc=$?; fi
  printf '%s_exit=%s\n' "$ep06_mode" "$ep06_rc" >> "$ep06_status"
  (( ep06_rc == 0 )) || exit "$ep06_rc"
done
printf 'status=initial_comparisons_complete\n' >> "$ep06_status"
