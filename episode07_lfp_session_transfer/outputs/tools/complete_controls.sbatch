#!/bin/bash
# First execution of unstarted controls; primary opening only after full eligibility.
#SBATCH --job-name=ep07-controls
#SBATCH --partition=russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/alternative_explanations/attempt001/slurm-%j.out
#SBATCH --error=outputs/runs/alternative_explanations/attempt001/slurm-%j.err
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
export TMPDIR="${L_SCRATCH_JOB:?}/ep07_controls"
mkdir "$TMPDIR"
ep07_controls="outputs/runs/alternative_explanations/attempt001"
ep07_primary="outputs/runs/primary_evaluation/attempt001"
[[ -d "$ep07_controls" && -d "$ep07_primary" \
   && -d outputs/runs/final_fit/attempt001 ]] || exit 2
mkdir "$ep07_controls/completion_claim"
ep07_status="$ep07_controls/completion_status_${SLURM_JOB_ID}.txt"
ep07_allocation="$ep07_controls/allocation_${SLURM_JOB_ID}.txt"
TZ=UTC scontrol show job -o "$SLURM_JOB_ID" > "$ep07_allocation"
ep07_cpus=$(awk '{for (i=1;i<=NF;i++) if ($i ~ /^NumCPUs=/) {split($i,a,"="); print a[2]; exit}}' "$ep07_allocation")
[[ "$ep07_cpus" -ge 1 && "$ep07_cpus" -le 2 ]] || exit 2
printf 'parent_episode=EP07\nparent_job=%s\nallocated_cpus=%s\nstatus=started\n' \
  "$SLURM_JOB_ID" "$ep07_cpus" > "$ep07_status"
ulimit -S -f 524288

# Only the corrected test fixture is rechecked; passing unaffected suites stay reused.
ep07_control_rc=0
if (cd outputs/tools && timeout --signal=TERM --kill-after=5s 120s \
    python3.12 -B -m unittest -v test_primary_control) \
    > "$ep07_controls/repaired_checks_${SLURM_JOB_ID}.txt" 2>&1; then :; else ep07_control_rc=$?; fi
printf 'repaired_checks_exit=%s\n' "$ep07_control_rc" >> "$ep07_status"
# New frozen-fit/scorer transports use fabricated files, never actual held data.
ep07_final_rc=0
if (cd outputs/tools && timeout --signal=TERM --kill-after=5s 180s \
    python3.12 -B -m unittest -v test_final_fit test_final_score) \
    > "$ep07_primary/transport_checks_${SLURM_JOB_ID}.txt" 2>&1; then :; else ep07_final_rc=$?; fi
printf 'final_transport_checks_exit=%s\n' "$ep07_final_rc" >> "$ep07_status"
ep07_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep07_astra_deps="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
ep07_astra_rc=0
if PYTHONPATH="$ep07_astra/lib/python3.12/site-packages:$ep07_astra_deps" \
    timeout --signal=TERM --kill-after=5s 30s "$ep07_astra/bin/astra-tools" \
    validate outputs/astra/v0.0.14/astra.yaml --json \
    > "$ep07_controls/astra_validation_${SLURM_JOB_ID}.json" 2>&1; then :; else ep07_astra_rc=$?; fi
printf 'astra_exit=%s\n' "$ep07_astra_rc" >> "$ep07_status"
[[ "$ep07_control_rc" -eq 0 ]] || exit "$ep07_control_rc"

ep07_deadlines=$(python3.12 -B - "$ep07_allocation" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
fields = dict(t.split('=', 1) for t in Path(sys.argv[1]).read_text().split() if '=' in t)
allocation_end = datetime.fromisoformat(fields['EndTime']).replace(tzinfo=timezone.utc).timestamp()
base = Path('outputs/runs/matched_search')
clock = json.loads((base / 'search_clock.json').read_text())
if clock['wall_clock_limit_seconds'] != 96 * 3600:
    raise SystemExit('search clock changed')
progress = json.loads((base / 'adaptive002/progress.json').read_text())
if not progress['starting_complete'] or not progress['adaptive_stage_complete'] or not progress['stop_reason']:
    raise SystemExit('original matched search must be complete before controls')
last_index = int(progress['successor_records'][-1]['candidate_id'].split('_')[1])
end = min(allocation_end, clock['started_epoch'] + clock['wall_clock_limit_seconds'])
if end - time.time() < 600:
    raise SystemExit('insufficient remaining allocation/search time')
print(int(end - 180), int(end - 45), last_index)
PY
)
read -r ep07_fit_deadline ep07_score_deadline ep07_last_index <<< "$ep07_deadlines"
python3.12 -u -B outputs/tools/run_finalist_controls.py --mode select \
  --last-index "$ep07_last_index" --out "$ep07_controls" --deadline-epoch "$ep07_fit_deadline" \
  > "$ep07_controls/selection.txt" 2>&1
for ep07_control in 1 2 3 4 5 6 7 8 9 10; do
  ep07_now=$(date +%s)
  if (( ep07_fit_deadline - ep07_now < 180 )); then
    printf 'status=incomplete_controls_deadline\n' >> "$ep07_status"
    exit 1
  fi
  ep07_fit_rc=0
  if timeout --signal=TERM --kill-after=10s "$((ep07_fit_deadline - ep07_now + 5))s" \
      python3.12 -u -B outputs/tools/run_finalist_controls.py --mode fit \
      --control "$ep07_control" --out "$ep07_controls" \
      --candidate-directory "$SCRATCH/br_autoresearch/episode07_lfp_session_transfer/source_support_attempt002/candidate" \
      --deadline-epoch "$ep07_fit_deadline" > "$ep07_controls/fit_${ep07_control}.txt" 2>&1; \
      then :; else ep07_fit_rc=$?; fi
  printf 'control_%s_fit_exit=%s\n' "$ep07_control" "$ep07_fit_rc" >> "$ep07_status"
  [[ "$ep07_fit_rc" -eq 0 ]] || exit "$ep07_fit_rc"
  python3.12 -u -B outputs/tools/run_finalist_controls.py --mode score \
    --control "$ep07_control" --out "$ep07_controls" \
    --scorer-directory "$SCRATCH/br_autoresearch/episode07_lfp_session_transfer/source_support_attempt002/scorer" \
    --deadline-epoch "$ep07_score_deadline" > "$ep07_controls/score_${ep07_control}.txt" 2>&1
done
ep07_known_deadline=$(( $(date +%s) + 600 ))
if (( ep07_known_deadline > ep07_fit_deadline )); then ep07_known_deadline=$ep07_fit_deadline; fi
ep07_known_rc=0
if timeout --signal=TERM --kill-after=10s 605s python3.12 -u -B \
    outputs/tools/known_answer_cases.py --selection "$ep07_controls/selected_pair.json" \
    --out "$ep07_controls/known_answer_report.json" --deadline-epoch "$ep07_known_deadline" \
    > "$ep07_controls/known_answer_cases.txt" 2>&1; then :; else ep07_known_rc=$?; fi
printf 'known_answer_exit=%s\n' "$ep07_known_rc" >> "$ep07_status"
if [[ "$ep07_known_rc" -ne 0 || "$ep07_final_rc" -ne 0 ]]; then
  printf 'status=primary_unavailable_no_held_opening\n' >> "$ep07_status"
  exit 1
fi

# Eligible primary only: four separate processes preserve phase boundaries.
for ep07_mode in freeze fit predict score; do
  ep07_args=()
  case "$ep07_mode" in
    fit) ep07_args=(--candidate-directory "$SCRATCH/br_autoresearch/episode07_lfp_session_transfer/source_support_attempt002/candidate") ;;
    predict|score) ep07_args=(--heldout "$SCRATCH/br_autoresearch/episode07_lfp_session_transfer/source_support_attempt002/sealed/heldout.h5") ;;
  esac
  ep07_now=$(date +%s)
  ep07_limit=$((ep07_score_deadline - ep07_now))
  if (( ep07_limit < 120 )); then
    printf 'status=incomplete_primary_deadline\n' >> "$ep07_status"
    exit 1
  fi
  timeout --signal=TERM --kill-after=10s "${ep07_limit}s" python3.12 -u -B \
    outputs/tools/run_primary_evaluation.py --mode "$ep07_mode" "${ep07_args[@]}" \
    --out "$ep07_primary" --deadline-epoch "$ep07_score_deadline" \
    > "$ep07_primary/${ep07_mode}.txt" 2>&1
  printf 'primary_%s_exit=0\n' "$ep07_mode" >> "$ep07_status"
done
printf 'status=primary_evaluation_complete\n' >> "$ep07_status"
