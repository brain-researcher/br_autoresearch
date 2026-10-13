#!/bin/bash
# First sixteen development pairs; no adaptive/final/held-out launch or retry.
# Four numerical workers, one thread each. Charge the parent allocation once.
#SBATCH --job-name=ep07-starting16
#SBATCH --partition=russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/matched_search/attempt001/slurm-%j.out
#SBATCH --error=outputs/runs/matched_search/attempt001/slurm-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep07_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s\n' "$ep07_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}"
[[ "$ep07_host" != *-ln* && "$SLURM_CPUS_PER_TASK" -ge 4 \
   && "$SLURM_CPUS_PER_TASK" -le 6 && "$SLURM_MEM_PER_NODE" == 28672 ]] || exit 2
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
export TMPDIR="${L_SCRATCH_JOB:?}/ep07_starting_attempt001"
mkdir "$TMPDIR"
ep07_run="outputs/runs/matched_search/attempt001"
[[ -d "$ep07_run" ]] || exit 2
mkdir "$ep07_run/batch_claim" || exit 2
TZ=UTC scontrol show job -o "$SLURM_JOB_ID" > "$ep07_run/allocation.txt"
ep07_alloc_cpus=$(awk '{for (i=1;i<=NF;i++) if ($i ~ /^NumCPUs=/) {split($i,a,"="); print a[2]; exit}}' "$ep07_run/allocation.txt")
[[ "$ep07_alloc_cpus" -ge 4 && "$ep07_alloc_cpus" -le 6 ]] || exit 2
printf 'parent_episode=EP07\nparent_job=%s\nallocated_cpus=%s\nstatus=started\n' \
  "$SLURM_JOB_ID" "$ep07_alloc_cpus" > "$ep07_run/status.txt"
# Bound individual output files; the full prediction collection is estimated
# below 2 GiB. No raw source is copied or changed.
ulimit -S -f 262144
ep07_checks_rc=0
if (cd outputs/tools && timeout --signal=TERM --kill-after=5s 180s python3.12 -B - <<'PY'
import sys
import torch
import unittest
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
print("torch=" + str(torch.__version__), flush=True)
suite = unittest.defaultTestLoader.loadTestsFromNames([
    "test_native_trials", "test_lfp_features", "test_population_bridge",
    "test_primary_population", "test_primary_design", "test_regression",
    "test_primary_model", "test_candidate_predictions", "test_development_score"])
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
PY
) > "$ep07_run/component_checks.txt" 2>&1; then :; else ep07_checks_rc=$?; fi
printf 'component_checks_exit=%s\n' "$ep07_checks_rc" >> "$ep07_run/status.txt"

# Changed ASTRA is validated once here; it cannot open outcomes or approve a fit.
ep07_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep07_astra_deps="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
ep07_astra_rc=0
if PYTHONPATH="$ep07_astra/lib/python3.12/site-packages:$ep07_astra_deps" \
    timeout --signal=TERM --kill-after=5s 30s "$ep07_astra/bin/astra-tools" \
    validate outputs/astra/v0.0.14/astra.yaml --json \
    > "$ep07_run/astra_validation.json" 2>&1; then :; else ep07_astra_rc=$?; fi
printf 'astra_exit=%s\n' "$ep07_astra_rc" >> "$ep07_run/status.txt"
if [[ "$ep07_checks_rc" -ne 0 ]]; then
  printf 'status=stopped_before_real_data_fitting\n' >> "$ep07_run/status.txt"
  exit "$ep07_checks_rc"
fi

# Establish the first actual matched-search clock. A later attempt must read
# and preserve this record, never manufacture a new 96-hour window.
ep07_deadlines=$(python3.12 -B - <<'PY'
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
run = Path('outputs/runs/matched_search/attempt001')
fields = dict(t.split('=', 1) for t in (run / 'allocation.txt').read_text().split() if '=' in t)
end = datetime.fromisoformat(fields['EndTime']).replace(tzinfo=timezone.utc).timestamp()
now = time.time()
if end - now < 600:
    raise SystemExit('insufficient remaining allocation for substantive starting fits')
clock = run.parent / 'search_clock.json'
with clock.open('x') as handle:
    json.dump({'started_epoch': now, 'started_utc': datetime.now(timezone.utc).isoformat(),
        'job_id': os.environ['SLURM_JOB_ID'], 'wall_clock_limit_seconds': 96 * 3600,
        'clock_not_reset_by_later_attempts': True}, handle, indent=2)
# Leave time for scoring and clean closeout. No renewals or automatic successor.
print(int(min(end - 300, now + 6500)), int(end - 45))
PY
)
read -r ep07_fit_deadline ep07_score_deadline <<< "$ep07_deadlines"
ep07_pids=()
for ep07_worker in 0 1 2 3; do
  timeout --signal=TERM --kill-after=10s 6520s python3.12 -u -B \
    outputs/tools/run_starting_models.py --worker "$ep07_worker" \
    --candidate-directory "$SCRATCH/br_autoresearch/episode07_lfp_session_transfer/source_support_attempt002/candidate" \
    --out "$ep07_run" --deadline-epoch "$ep07_fit_deadline" \
    > "$ep07_run/worker_${ep07_worker}.txt" 2>&1 &
  ep07_pids+=("$!")
done
ep07_worker_rc=0
for ep07_pid in "${ep07_pids[@]}"; do
  if wait "$ep07_pid"; then :; else ep07_worker_rc=1; fi
done
printf 'workers_exit=%s\n' "$ep07_worker_rc" >> "$ep07_run/status.txt"
ep07_score_rc=0
if timeout --signal=TERM --kill-after=5s 240s python3.12 -u -B \
    outputs/tools/score_starting_models.py --out "$ep07_run" \
    --scorer-directory "$SCRATCH/br_autoresearch/episode07_lfp_session_transfer/source_support_attempt002/scorer" \
    --deadline-epoch "$ep07_score_deadline" > "$ep07_run/scorer.txt" 2>&1; then
  :
else
  ep07_score_rc=$?
fi
printf 'scorer_exit=%s\nstatus=closed\n' "$ep07_score_rc" >> "$ep07_run/status.txt"
if [[ "$ep07_worker_rc" -ne 0 || "$ep07_score_rc" -ne 0 ]]; then exit 1; fi
