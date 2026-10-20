#!/bin/bash
# Finite attended refit of the sixteen prospectively frozen changed directions.
# Parent: three CPUs/16 GiB; decoder stays one thread with a15-GiB RSS guard.
# Companion: one CPU/768 MiB, worker RSS guard600 MiB; no other companion runs.
set -euo pipefail
[[ $# == 0 && "${SLURM_JOB_ID:-}" == 47586225 && "$(hostname)" != *-ln* ]]
[[ "${SLURM_CPUS_PER_TASK:-}" == 1 && -n "${SLURM_STEP_ID:-}" ]]
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]]
ep18_started=$(date +%s)
(( ep18_started + 900 < ${SLURM_JOB_END_TIME:?} ))
ep18_root="${OAK:?}/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability"
ep18_base="$ep18_root/outputs/runs/repeated_selector_refit"
ep18_selector="$ep18_root/outputs/runs/repeated_selector_job_47586225_step_2/report/repeated_selector_sensitivity.complete.json"
ep18_operation="$ep18_base/operations_job_${SLURM_JOB_ID}_step_${SLURM_STEP_ID}"
mkdir -p "$ep18_base"
mkdir -m 700 "$ep18_operation"
ep18_local="${L_SCRATCH_JOB:?}/selector-refit-${SLURM_STEP_ID}"
mkdir -m 700 "$ep18_local"
export TMPDIR="$ep18_local" MPLCONFIGDIR="$ep18_local/matplotlib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 viz py-matplotlib/3.8.3_py312
printf 'hostname=%s job=%s step=%s cpus=1 threads=1 companion_memory_mib=768 worker_rss_guard_mib=600\n' \
  "$(hostname)" "$SLURM_JOB_ID" "$SLURM_STEP_ID" | tee "$ep18_operation/allocation.txt"
cd "$ep18_root/outputs/code"
/usr/bin/time -v -o "$ep18_operation/refit_resources.txt" \
  timeout --signal=TERM --kill-after=5s 795s python3.12 -u -B run_repeated_selector_refit.py \
  --selector "$ep18_selector" --output-root "$ep18_base/pooled001" \
  --operations "$ep18_operation/refit" \
  --deadline-epoch "$((ep18_started+780))" --rss-limit-bytes 629145600 \
  > "$ep18_operation/refit_stdout.txt" 2>&1
/usr/bin/time -v -o "$ep18_operation/report_resources.txt" \
  timeout --signal=TERM --kill-after=5s 60s python3.12 -u -B report_repeated_selector_refit.py \
  --selector-report "$ep18_selector" \
  --original-root "$ep18_root/outputs/runs/repeated_image_signal_recovery/control001" \
  --refit-root "$ep18_base/pooled001" \
  --original-report "$ep18_root/outputs/runs/repeated_image_signal_recovery/report001/repeated_image_control.complete.json" \
  --output-root "$ep18_base/report001" --figure-root "$ep18_root/outputs/figures/repeated_selector_refit001" \
  > "$ep18_operation/report_stdout.txt" 2>&1
printf 'all30_selector_sensitivity_report_complete=%s\n' "$ep18_base/report001"
