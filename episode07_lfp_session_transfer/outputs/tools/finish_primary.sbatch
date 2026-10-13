#!/bin/bash
# Resume only the unstarted primary and conditional EP06 preparation.
#SBATCH --job-name=ep07-final
#SBATCH --partition=russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/primary_evaluation/attempt001/slurm-%j.out
#SBATCH --error=outputs/runs/primary_evaluation/attempt001/slurm-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep07_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s\n' "$ep07_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}"
[[ "$ep07_host" != *-ln* && "$SLURM_CPUS_PER_TASK" -ge 1 \
   && "$SLURM_CPUS_PER_TASK" -le 2 && "$SLURM_MEM_PER_NODE" == 7168 ]] || exit 2
ep07_project="${OAK:?}/users/zijiao/br_autoresearch"
cd "$ep07_project/episode07_lfp_session_transfer"
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312 py-pytorch/2.4.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
umask 077
export TMPDIR="${L_SCRATCH_JOB:?}/ep07_final"
mkdir "$TMPDIR"
ep07_primary="outputs/runs/primary_evaluation/attempt001"
ep06_episode="$ep07_project/episode06_lfp_regional_fingerprints"
ep06_attempt="$ep06_episode/outputs/runs/source_session_support/development001"
[[ -d "$ep07_primary" && -d "$ep06_attempt" ]] || exit 2
mkdir "$ep07_primary/completion_claim"
ep07_status="$ep07_primary/status_${SLURM_JOB_ID}.txt"
ep07_allocation="$ep07_primary/allocation_${SLURM_JOB_ID}.txt"
TZ=UTC scontrol show job -o "$SLURM_JOB_ID" > "$ep07_allocation"
ep07_cpus=$(awk '{for (i=1;i<=NF;i++) if ($i ~ /^NumCPUs=/) {split($i,a,"="); print a[2]; exit}}' "$ep07_allocation")
[[ "$ep07_cpus" -ge 1 && "$ep07_cpus" -le 2 ]] || exit 2
printf 'parent_episode=EP07\nparent_job=%s\nallocated_cpus=%s\nstatus=started\n' \
  "$SLURM_JOB_ID" "$ep07_cpus" > "$ep07_status"
ulimit -S -f 524288

# Reuse actual passing prerequisite checks from the completed predecessor.
# The primary runner independently binds every scientific eligibility record.
ep07_deadline=$(python3.12 -B - "$ep07_allocation" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
prior = Path('outputs/runs/alternative_explanations/qualification_repair001/status_47350382.txt')
status = dict(line.split('=', 1) for line in prior.read_text().splitlines())
for name in ('continuation_checks_exit', 'ep06_component_checks_exit',
             'astra_episode07_lfp_session_transfer_exit',
             'astra_episode06_lfp_regional_fingerprints_exit', 'known_answer_exit'):
    if status.get(name) != '0':
        raise SystemExit('completed prerequisite absent: ' + name)
fields = dict(t.split('=', 1) for t in Path(sys.argv[1]).read_text().split() if '=' in t)
end = datetime.fromisoformat(fields['EndTime']).replace(tzinfo=timezone.utc).timestamp()
clock = json.loads(Path('outputs/runs/matched_search/search_clock.json').read_text())
if clock['wall_clock_limit_seconds'] != 96 * 3600:
    raise SystemExit('original search clock changed')
deadline = min(end - 45, clock['started_epoch'] + clock['wall_clock_limit_seconds'])
if deadline - time.time() < 720:
    raise SystemExit('insufficient allocation/search time for primary and companion')
print(int(deadline))
PY
)
for ep07_mode in freeze fit predict score; do
  # Always nonempty: older Bash with nounset rejects expansion of an empty array.
  ep07_args=(--mode "$ep07_mode" --out "$ep07_primary" --deadline-epoch "$ep07_deadline")
  case "$ep07_mode" in
    fit) ep07_args+=(--candidate-directory "$SCRATCH/br_autoresearch/episode07_lfp_session_transfer/source_support_attempt002/candidate") ;;
    predict|score) ep07_args+=(--heldout "$SCRATCH/br_autoresearch/episode07_lfp_session_transfer/source_support_attempt002/sealed/heldout.h5") ;;
  esac
  ep07_limit=$((ep07_deadline - $(date +%s)))
  if (( ep07_limit < 120 )); then
    printf 'status=incomplete_primary_deadline\n' >> "$ep07_status"
    exit 1
  fi
  timeout --signal=TERM --kill-after=5s "${ep07_limit}s" python3.12 -u -B \
    outputs/tools/run_primary_evaluation.py "${ep07_args[@]}" \
    > "$ep07_primary/${ep07_mode}.txt" 2>&1
  printf 'primary_%s_exit=0\n' "$ep07_mode" >> "$ep07_status"
done
printf 'status=primary_evaluation_complete\n' >> "$ep07_status"

ep06_now=$(date +%s)
ep06_limit=$((ep07_deadline - ep06_now))
if (( ep06_limit >= 180 )); then
  if (( ep06_limit > 900 )); then ep06_limit=900; fi
  ep06_deadline=$((ep06_now + ep06_limit))
  mkdir -p "$SCRATCH/br_autoresearch/episode06_lfp_regional_fingerprints"
  printf 'ep06_preparation_started_epoch=%s\n' "$ep06_now" >> "$ep07_status"
  ep06_rc=0
  if (cd "$ep06_episode" && timeout --signal=TERM --kill-after=5s "$((ep06_limit - 5))s" \
      python3.12 -u -B outputs/tools/prepare_after_ep07.py --deadline-epoch "$ep06_deadline") \
      > "$ep06_attempt/preparation_${SLURM_JOB_ID}.txt" 2>&1; then :; else ep06_rc=$?; fi
  printf 'ep06_preparation_exit=%s\nep06_preparation_ended_epoch=%s\n' "$ep06_rc" "$(date +%s)" >> "$ep07_status"
else
  printf 'ep06_preparation=not_started_remaining_time\n' >> "$ep07_status"
fi
