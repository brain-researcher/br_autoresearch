#!/bin/bash
# Submit with afterok dependency on the selected complete cohort array.
#SBATCH --job-name=ep01-fmri-summary
#SBATCH --partition=russpold,normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=15G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --output=outputs/runs/exploratory_fmri_specification_sensitivity/cohort_20261013_001/summary-%j.out
#SBATCH --error=outputs/runs/exploratory_fmri_specification_sensitivity/cohort_20261013_001/summary-%j.err
set -euo pipefail
umask 077
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* ]]
[[ "${SLURM_CPUS_PER_TASK:-}" == 2 && "${SLURM_MEM_PER_NODE:-}" == 15360 ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 && -z "${SLURM_JOB_GPUS:-}" ]]
: "${OAK:?}" "${GROUP_HOME:?}" "${L_SCRATCH_JOB:?}"
ep01_base="$OAK/users/zijiao/br_autoresearch/episode01_narps_deep_search"
ep01_root="$ep01_base/outputs/runs/exploratory_fmri_specification_sensitivity"
ep01_ops="$ep01_root/cohort_20261013_001/summary_ops_${SLURM_JOB_ID}"
mkdir -m 700 "$ep01_ops"
ep01_begin=$SECONDS
finish() {
  ep01_exit=$?
  trap - EXIT
  printf '{"job_id":"%s","exit_code":%s,"elapsed_seconds":%s,"allocated_cpus":2,"memory_mib":15360,"automatic_retry":false}\n' \
    "$SLURM_JOB_ID" "$ep01_exit" "$((SECONDS-ep01_begin))" > "$ep01_ops/exit.json"
  exit "$ep01_exit"
}
trap finish EXIT
ulimit -c 0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME
export TMPDIR="$L_SCRATCH_JOB/ep01-summary-tmp"
export MPLCONFIGDIR="$L_SCRATCH_JOB/ep01-matplotlib"
mkdir -m 700 "$TMPDIR" "$MPLCONFIGDIR"
ep01_python="$GROUP_HOME/zijiao/miniforge3/envs/encoding_model/bin/python"
[[ -x "$ep01_python" ]]
cd "$ep01_base/outputs/code"
timeout --signal=TERM --kill-after=10s 1750s "$ep01_python" -B \
  summarize_exploratory_fmri.py \
  --prepared-dir "$ep01_root/prepared_20261013_002" \
  --run-root "$ep01_root/pilot_20261013_47670430/models02" \
  --run-root "$ep01_root/cohort_20261013_001/models" \
  --output-dir "$ep01_root/summary_20261013_001" \
  --max-seconds 1740 --max-rss-gib 12 > "$ep01_ops/summary.log" 2>&1
