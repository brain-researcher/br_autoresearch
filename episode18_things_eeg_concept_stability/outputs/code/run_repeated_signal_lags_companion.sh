#!/bin/bash
# Attended saved-record reporting in spare resources of the task-owned decoder.
# No EEG payload, live EP21 collection, scheduler loop or successor process.
set -euo pipefail
[[ $# == 0 && "${SLURM_JOB_ID:-}" == 47586225 && "$(hostname)" != *-ln* ]]
[[ "${SLURM_CPUS_PER_TASK:-}" == 1 && -n "${SLURM_STEP_ID:-}" ]]
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]]
(( $(date +%s) + 240 < ${SLURM_JOB_END_TIME:?} ))
ep18_root="${OAK:?}/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability"
ep18_run="$ep18_root/outputs/runs/repeated_signal_lags_job_${SLURM_JOB_ID}_step_${SLURM_STEP_ID}"
ep18_figure="$ep18_root/outputs/figures/repeated_signal_lags_job_${SLURM_JOB_ID}_step_${SLURM_STEP_ID}"
mkdir -m 700 "$ep18_run"
ep18_local="${L_SCRATCH_JOB:?}/repeated-lags-${SLURM_STEP_ID}"
mkdir -m 700 "$ep18_local"
export TMPDIR="$ep18_local" MPLCONFIGDIR="$ep18_local/matplotlib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 viz py-matplotlib/3.8.3_py312
printf 'hostname=%s job=%s step=%s cpus=1 threads=1 companion_memory_mib=512\n' \
  "$(hostname)" "$SLURM_JOB_ID" "$SLURM_STEP_ID" | tee "$ep18_run/allocation.txt"
cd "$ep18_root/outputs/code"
/usr/bin/time -v -o "$ep18_run/resources.txt" \
  timeout --signal=TERM --kill-after=5s 180s python3.12 -u -B report_repeated_signal_lags.py \
  --input-root "$ep18_root/outputs/runs/repeated_signal_detrend/detrend001" \
  --output-dir "$ep18_run/report" --figure-dir "$ep18_figure" \
  > "$ep18_run/stdout.txt" 2>&1
printf 'saved_record_lag_report_complete=%s\n' "$ep18_run/report"
