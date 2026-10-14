#!/bin/bash
# Complete-cohort G/T reporting only; no fitting, control search or audit access.
# Prepared wrapper only; no standalone reporting job is selected or submitted.
# Coordinator preferentially runs the reporter in a task-owned companion
# allocation after all 30 participants complete; never pad runtime. Both
# production arrays must close. Old array elements 27/28 were cancelled and
# replaced by 47720344_27/28, so afterany preserves lifecycle ordering while
# the reporter's original all-30-complete guard controls scientific access.
# Create core_direct_report001 before any separately authorized sbatch use.
#SBATCH --job-name=ep18-core-report
#SBATCH --partition=russpold,normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --dependency=afterany:47716305:47717611:47720344
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/baseline_models/core_direct_report001/slurm-%j.out
#SBATCH --error=outputs/runs/baseline_models/core_direct_report001/slurm-%j.err
set -euo pipefail
umask 077
[[ $# -eq 0 ]] || exit 2
ep18_report_host=$(hostname)
[[ -n "${SLURM_JOB_ID:-}" && "$ep18_report_host" != *-ln* ]] || exit 2
[[ "${SLURM_CPUS_PER_TASK:-}" == 1 && "${SLURM_MEM_PER_NODE:-}" == 2048 ]] || exit 2
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]] || exit 2
[[ -z "${SLURM_ARRAY_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || exit 2
: "${OAK:?}" "${L_SCRATCH_JOB:?}"
ep18_report_episode="$OAK/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability"
ep18_report_code="$ep18_report_episode/outputs/code"
ep18_report_output="$ep18_report_episode/outputs/runs/baseline_models/core_direct_report001"
[[ -d "$ep18_report_output" ]] || exit 2
ep18_report_local="$L_SCRATCH_JOB/ep18-core-report"
mkdir -m 700 "$ep18_report_local"
mkdir -m 700 "$ep18_report_local/tmp" "$ep18_report_local/cache" "$ep18_report_local/matplotlib"
export TMPDIR="$ep18_report_local/tmp" XDG_CACHE_HOME="$ep18_report_local/cache"
export MPLCONFIGDIR="$ep18_report_local/matplotlib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
printf 'hostname=%s\njob_id=%s\nprerequisites=47716305,47717611_all_elements,47720344_all_elements\noriginal27_28_replaced=true\nrelease_guard=all30_complete_records\nstatus=started\n' \
  "$ep18_report_host" "$SLURM_JOB_ID" > "$ep18_report_output/status.txt"
ep18_report_finish() {
  ep18_report_exit=$?
  trap - EXIT
  printf 'exit_code=%s\n' "$ep18_report_exit" >> "$ep18_report_output/status.txt"
  if (( ep18_report_exit == 0 )); then
    printf 'status=complete_report\n' >> "$ep18_report_output/status.txt"
  else
    printf 'status=report_failed_preserve_completed_fits\n' >> "$ep18_report_output/status.txt"
  fi
  exit "$ep18_report_exit"
}
trap ep18_report_finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 viz py-matplotlib/3.8.3_py312
cd "$ep18_report_local"
ep18_report_timeout=$((${SLURM_JOB_END_TIME:?}-$(date +%s)-30))
(( ep18_report_timeout > 0 )) || exit 124
if (( ep18_report_timeout > 540 )); then ep18_report_timeout=540; fi
# The unchanged reporter checks all 30 complete records before reading losses,
# then validates every participant's original settings and six outer folds.
timeout --signal=TERM --kill-after=5s "${ep18_report_timeout}s" python3.12 -B \
  "$ep18_report_code/report_core_direct.py" \
  --initialized-root "$ep18_report_episode/outputs/cache/private/core_reference_20261013_001" \
  --run-root "$ep18_report_episode/outputs/cache/private/core_direct001" \
  --output-root "$ep18_report_output" \
  --figure-root "$ep18_report_episode/outputs/figures/core_direct001" \
  > "$ep18_report_output/execution.txt" 2>&1
