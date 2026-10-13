#!/bin/bash
# Final numerical attempts for two unscored pairs; no automatic further retry.
# Two workers, one thread each; all allocation cost charged once to EP07.
#SBATCH --job-name=ep07-en-repair
#SBATCH --partition=russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=14G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/matched_search/attempt002/slurm-%j.out
#SBATCH --error=outputs/runs/matched_search/attempt002/slurm-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep07_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s\n' "$ep07_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}"
[[ "$ep07_host" != *-ln* && "$SLURM_CPUS_PER_TASK" -ge 2 \
   && "$SLURM_CPUS_PER_TASK" -le 3 && "$SLURM_MEM_PER_NODE" == 14336 ]] || exit 2
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
export TMPDIR="${L_SCRATCH_JOB:?}/ep07_starting_attempt002"
mkdir "$TMPDIR"
ep07_run="outputs/runs/matched_search/attempt002"
[[ -d "$ep07_run" ]] || exit 2
mkdir "$ep07_run/batch_claim" || exit 2
TZ=UTC scontrol show job -o "$SLURM_JOB_ID" > "$ep07_run/allocation.txt"
ep07_alloc_cpus=$(awk '{for (i=1;i<=NF;i++) if ($i ~ /^NumCPUs=/) {split($i,a,"="); print a[2]; exit}}' "$ep07_run/allocation.txt")
[[ "$ep07_alloc_cpus" -ge 2 && "$ep07_alloc_cpus" -le 3 ]] || exit 2
printf 'parent_episode=EP07\nparent_job=%s\nallocated_cpus=%s\nstatus=started\n' \
  "$SLURM_JOB_ID" "$ep07_alloc_cpus" > "$ep07_run/status.txt"
ulimit -S -f 262144
ep07_checks_rc=0
if (cd outputs/tools && timeout --signal=TERM --kill-after=5s 180s \
    python3.12 -B -m unittest -v \
    test_regression.RegressionTests.test_correlated_unequal_scale_elastic_net_matches_scalar_solution_and_residual_kkt \
    test_regression.RegressionTests.test_elementwise_elastic_net_has_analytic_soft_threshold_and_refuses_nonconvergence) \
    > "$ep07_run/regression_checks.txt" 2>&1; then :; else ep07_checks_rc=$?; fi
printf 'regression_checks_exit=%s\n' "$ep07_checks_rc" >> "$ep07_run/status.txt"
# Dormant source weights are not a dependency or new gate for this repair.
ep07_weights_rc=0
if (cd outputs/tools && timeout --signal=TERM --kill-after=5s 60s \
    python3.12 -B -m unittest -v test_source_weighting) \
    > "$ep07_run/source_weighting_checks.txt" 2>&1; then :; else ep07_weights_rc=$?; fi
printf 'source_weighting_checks_exit=%s\n' "$ep07_weights_rc" >> "$ep07_run/status.txt"

# EP06 scalar arithmetic only; no neural readers or sibling fit state.
ep06_run="$ep07_project/episode06_lfp_regional_fingerprints/outputs/runs/nomination_migration_checks_job_${SLURM_JOB_ID}"
mkdir "$ep06_run"
printf 'parent_episode=EP07\nparent_job=%s\nstatus=started\n' "$SLURM_JOB_ID" > "$ep06_run/status.txt"
ep06_checks_rc=0
if (cd "$ep07_project/episode06_lfp_regional_fingerprints/outputs/tests" && \
    timeout --signal=TERM --kill-after=5s 90s python3.12 -B -m unittest -v \
    test_component_nomination test_component_outer_stability test_component_support_h) \
    > "$ep06_run/checks.txt" 2>&1; then :; else ep06_checks_rc=$?; fi
printf 'checks_exit=%s\n' "$ep06_checks_rc" >> "$ep06_run/status.txt"

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
if [[ "$ep07_checks_rc" -ne 0 ]]; then
  printf 'status=stopped_before_real_data_fitting\n' >> "$ep07_run/status.txt"
  exit "$ep07_checks_rc"
fi

# Reuse the actual first-fit clock. No time or resource budget reset.
ep07_deadlines=$(python3.12 -B - <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import time
run = Path('outputs/runs/matched_search/attempt002')
fields = dict(t.split('=', 1) for t in (run / 'allocation.txt').read_text().split() if '=' in t)
allocation_end = datetime.fromisoformat(fields['EndTime']).replace(tzinfo=timezone.utc).timestamp()
clock = json.loads((run.parent / 'search_clock.json').read_text())
if clock['wall_clock_limit_seconds'] != 96 * 3600:
    raise SystemExit('search clock limit changed')
end = min(allocation_end, clock['started_epoch'] + clock['wall_clock_limit_seconds'])
now = time.time()
if end - now < 600:
    raise SystemExit('insufficient remaining allocation/search time for substantive repair')
print(int(min(end - 300, now + 6500)), int(end - 45))
PY
)
read -r ep07_fit_deadline ep07_score_deadline <<< "$ep07_deadlines"
ep07_pids=()
for ep07_worker in 0 1; do
  timeout --signal=TERM --kill-after=10s 6520s python3.12 -u -B \
    outputs/tools/run_starting_models.py --repair --worker "$ep07_worker" \
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
    outputs/tools/score_starting_models.py --repair --out "$ep07_run" \
    --scorer-directory "$SCRATCH/br_autoresearch/episode07_lfp_session_transfer/source_support_attempt002/scorer" \
    --deadline-epoch "$ep07_score_deadline" > "$ep07_run/scorer.txt" 2>&1; then :; else ep07_score_rc=$?; fi
printf 'scorer_exit=%s\nstatus=closed\n' "$ep07_score_rc" >> "$ep07_run/status.txt"
if [[ "$ep07_worker_rc" -ne 0 || "$ep07_score_rc" -ne 0 ]]; then exit 1; fi
