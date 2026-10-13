#!/bin/bash
# Remaining matched development, then fixed finalist controls and four recoveries.
# Every stage is bounded; this job never opens empirical held-out data.
#SBATCH --job-name=ep07-final-dev
#SBATCH --partition=russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/matched_search/adaptive002/slurm-%j.out
#SBATCH --error=outputs/runs/matched_search/adaptive002/slurm-%j.err
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
export TMPDIR="${L_SCRATCH_JOB:?}/ep07_adaptive002"
mkdir "$TMPDIR"
ep07_run="outputs/runs/matched_search/adaptive002"
ep07_controls="outputs/runs/alternative_explanations/attempt001"
[[ -d "$ep07_run" && -d "$ep07_controls" ]] || exit 2
mkdir "$ep07_run/batch_claim" "$ep07_controls/batch_claim"
TZ=UTC scontrol show job -o "$SLURM_JOB_ID" > "$ep07_run/allocation.txt"
ep07_alloc_cpus=$(awk '{for (i=1;i<=NF;i++) if ($i ~ /^NumCPUs=/) {split($i,a,"="); print a[2]; exit}}' "$ep07_run/allocation.txt")
[[ "$ep07_alloc_cpus" -ge 1 && "$ep07_alloc_cpus" -le 2 ]] || exit 2
printf 'parent_episode=EP07\nparent_job=%s\nallocated_cpus=%s\nstatus=started\n' \
  "$SLURM_JOB_ID" "$ep07_alloc_cpus" > "$ep07_run/status.txt"
ulimit -S -f 524288

# Shared numerical changes must pass before any new empirical fitting.
ep07_shared_rc=0
if (cd outputs/tools && timeout --signal=TERM --kill-after=5s 180s \
    python3.12 -B -m unittest -v test_native_trials test_lfp_features \
    test_primary_population test_primary_design test_primary_model \
    test_source_weighting test_weighted_primary_model test_adaptive_progress) \
    > "$ep07_run/shared_checks.txt" 2>&1; then :; else ep07_shared_rc=$?; fi
printf 'shared_checks_exit=%s\n' "$ep07_shared_rc" >> "$ep07_run/status.txt"
ep07_control_rc=0
if (cd outputs/tools && timeout --signal=TERM --kill-after=5s 180s \
    python3.12 -B -m unittest -v test_primary_control test_fit_mismatch \
    test_finalist_controls test_known_answer_cases) \
    > "$ep07_run/control_checks.txt" 2>&1; then :; else ep07_control_rc=$?; fi
printf 'control_checks_exit=%s\n' "$ep07_control_rc" >> "$ep07_run/status.txt"
# Dormant final transport has no empirical outcome access in this job.
ep07_final_rc=0
if (cd outputs/tools && timeout --signal=TERM --kill-after=5s 180s \
    python3.12 -B -m unittest -v test_final_features test_final_predictions) \
    > "$ep07_run/final_transport_checks.txt" 2>&1; then :; else ep07_final_rc=$?; fi
printf 'final_transport_checks_exit=%s\n' "$ep07_final_rc" >> "$ep07_run/status.txt"

# Separately scoped EP06 companion uses fabricated arrays, never neural data.
ep06_run="$ep07_project/episode06_lfp_regional_fingerprints/outputs/runs/bandwise_checks_job_${SLURM_JOB_ID}"
mkdir "$ep06_run"
printf 'parent_episode=EP07\nparent_job=%s\nstatus=started\n' "$SLURM_JOB_ID" > "$ep06_run/status.txt"
ep06_rc=0
if (cd "$ep07_project/episode06_lfp_regional_fingerprints" && \
    timeout --signal=TERM --kill-after=5s 120s python3.12 -B -m unittest \
    discover -s outputs/tests -p test_bandwise_prediction.py -v) \
    > "$ep06_run/checks.txt" 2>&1; then :; else ep06_rc=$?; fi
printf 'checks_exit=%s\n' "$ep06_rc" >> "$ep06_run/status.txt"
ep07_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep07_astra_deps="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
ep07_astra_rc=0
if PYTHONPATH="$ep07_astra/lib/python3.12/site-packages:$ep07_astra_deps" \
    timeout --signal=TERM --kill-after=5s 30s "$ep07_astra/bin/astra-tools" \
    validate outputs/astra/v0.0.14/astra.yaml --json \
    > "$ep07_run/astra_validation.json" 2>&1; then :; else ep07_astra_rc=$?; fi
printf 'astra_exit=%s\n' "$ep07_astra_rc" >> "$ep07_run/status.txt"
ep06_astra_rc=0
if (cd "$ep07_project/episode06_lfp_regional_fingerprints" && \
    PYTHONPATH="$ep07_astra/lib/python3.12/site-packages:$ep07_astra_deps" \
    timeout --signal=TERM --kill-after=5s 30s "$ep07_astra/bin/astra-tools" \
    validate outputs/astra/v0.0.14/astra.yaml --json) \
    > "$ep06_run/astra_validation.json" 2>&1; then :; else ep06_astra_rc=$?; fi
printf 'astra_exit=%s\nstatus=closed\n' "$ep06_astra_rc" >> "$ep06_run/status.txt"
if [[ "$ep07_shared_rc" -ne 0 ]]; then
  printf 'status=stopped_shared_checks\n' >> "$ep07_run/status.txt"
  exit "$ep07_shared_rc"
fi

ep07_deadlines=$(python3.12 -B - <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import time
run = Path('outputs/runs/matched_search/adaptive002')
fields = dict(t.split('=', 1) for t in (run / 'allocation.txt').read_text().split() if '=' in t)
allocation_end = datetime.fromisoformat(fields['EndTime']).replace(tzinfo=timezone.utc).timestamp()
clock = json.loads((run.parent / 'search_clock.json').read_text())
if clock['wall_clock_limit_seconds'] != 96 * 3600:
    raise SystemExit('search clock limit changed')
end = min(allocation_end, clock['started_epoch'] + clock['wall_clock_limit_seconds'])
if end - time.time() < 600:
    raise SystemExit('insufficient remaining allocation/search time')
print(int(end - 180), int(end - 45))
PY
)
read -r ep07_fit_deadline ep07_score_deadline <<< "$ep07_deadlines"
ep07_stop=continue
for ep07_index in 7 8 9 10 11 12 13; do
  ep07_now=$(date +%s)
  if (( ep07_fit_deadline - ep07_now < 180 )); then
    printf 'status=incomplete_allocation_deadline\n' >> "$ep07_run/status.txt"
    exit 1
  fi
  ep07_fit_rc=0
  if timeout --signal=TERM --kill-after=10s "$((ep07_fit_deadline - ep07_now + 5))s" \
      python3.12 -u -B outputs/tools/run_adaptive_models.py --index "$ep07_index" \
      --candidate-directory "$SCRATCH/br_autoresearch/episode07_lfp_session_transfer/source_support_attempt002/candidate" \
      --out "$ep07_run" --deadline-epoch "$ep07_fit_deadline" \
      > "$ep07_run/worker_${ep07_index}.txt" 2>&1; then :; else ep07_fit_rc=$?; fi
  printf 'pair_%s_worker_exit=%s\n' "$ep07_index" "$ep07_fit_rc" >> "$ep07_run/status.txt"
  ep07_score_rc=0
  if timeout --signal=TERM --kill-after=5s 120s python3.12 -u -B \
      outputs/tools/score_adaptive_models.py --index "$ep07_index" --out "$ep07_run" \
      --scorer-directory "$SCRATCH/br_autoresearch/episode07_lfp_session_transfer/source_support_attempt002/scorer" \
      --deadline-epoch "$ep07_score_deadline" > "$ep07_run/scorer_${ep07_index}.txt" 2>&1; \
      then :; else ep07_score_rc=$?; fi
  printf 'pair_%s_scorer_exit=%s\n' "$ep07_index" "$ep07_score_rc" >> "$ep07_run/status.txt"
  if [[ "$ep07_score_rc" -ne 0 ]]; then
    printf 'status=incomplete_scorer\n' >> "$ep07_run/status.txt"
    exit "$ep07_score_rc"
  fi
  ep07_stop=$(python3.12 -B - <<'PY'
import json
from pathlib import Path
value = json.loads(Path('outputs/runs/matched_search/adaptive002/progress.json').read_text())
print('stop' if value['stop_reason'] is not None else 'continue')
PY
)
  if [[ "$ep07_stop" == stop ]]; then break; fi
done
if [[ "$ep07_stop" != stop || "$ep07_control_rc" -ne 0 ]]; then
  printf 'status=development_closed_controls_unavailable\n' >> "$ep07_run/status.txt"
  exit 1
fi
python3.12 -u -B outputs/tools/run_finalist_controls.py --mode select \
  --last-index "$ep07_index" --out "$ep07_controls" --deadline-epoch "$ep07_fit_deadline" \
  > "$ep07_controls/selection.txt" 2>&1

for ep07_control in 1 2 3 4 5 6 7 8 9 10; do
  ep07_now=$(date +%s)
  if (( ep07_fit_deadline - ep07_now < 180 )); then
    printf 'status=incomplete_controls_deadline\n' >> "$ep07_run/status.txt"
    exit 1
  fi
  ep07_fit_rc=0
  if timeout --signal=TERM --kill-after=10s "$((ep07_fit_deadline - ep07_now + 5))s" \
      python3.12 -u -B outputs/tools/run_finalist_controls.py --mode fit \
      --control "$ep07_control" --out "$ep07_controls" \
      --candidate-directory "$SCRATCH/br_autoresearch/episode07_lfp_session_transfer/source_support_attempt002/candidate" \
      --deadline-epoch "$ep07_fit_deadline" > "$ep07_controls/fit_${ep07_control}.txt" 2>&1; \
      then :; else ep07_fit_rc=$?; fi
  printf 'control_%s_fit_exit=%s\n' "$ep07_control" "$ep07_fit_rc" >> "$ep07_run/status.txt"
  if [[ "$ep07_fit_rc" -ne 0 ]]; then
    printf 'status=incomplete_control_fit\n' >> "$ep07_run/status.txt"
    exit "$ep07_fit_rc"
  fi
  python3.12 -u -B outputs/tools/run_finalist_controls.py --mode score \
    --control "$ep07_control" --out "$ep07_controls" \
    --scorer-directory "$SCRATCH/br_autoresearch/episode07_lfp_session_transfer/source_support_attempt002/scorer" \
    --deadline-epoch "$ep07_score_deadline" > "$ep07_controls/score_${ep07_control}.txt" 2>&1
done
ep07_known_deadline=$(( $(date +%s) + 600 ))
if (( ep07_known_deadline > ep07_fit_deadline )); then ep07_known_deadline=$ep07_fit_deadline; fi
ep07_known_rc=0
if timeout --signal=TERM --kill-after=10s 605s python3.12 -u -B \
    outputs/tools/known_answer_cases.py \
    --selection "$ep07_controls/selected_pair.json" \
    --out "$ep07_controls/known_answer_report.json" --deadline-epoch "$ep07_known_deadline" \
    > "$ep07_controls/known_answer_cases.txt" 2>&1; then :; else ep07_known_rc=$?; fi
printf 'known_answer_exit=%s\n' "$ep07_known_rc" >> "$ep07_run/status.txt"
if [[ "$ep07_known_rc" -ne 0 ]]; then
  printf 'status=known_answer_checks_not_passed\n' >> "$ep07_run/status.txt"
  exit "$ep07_known_rc"
fi
printf 'status=development_controls_and_recoveries_closed_no_held_out_opening\n' >> "$ep07_run/status.txt"
