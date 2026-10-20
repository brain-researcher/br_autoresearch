#!/bin/bash
# Four finite work groups, seven/eight people each, then one full-cohort report.
# First participant was measured separately; no duplicate index0 fitting.
#SBATCH --job-name=ep18-repeat-cohort
#SBATCH --partition=russpold,normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/repeated_image_signal_recovery/cohort-%j.out
#SBATCH --error=outputs/runs/repeated_image_signal_recovery/cohort-%j.err
set -euo pipefail
umask 077
[[ $# == 0 && -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* ]]
[[ "${SLURM_CPUS_PER_TASK:-}" == 4 && "${SLURM_MEM_PER_NODE:-}" == 8192 ]]
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]]
ep18_root="${OAK:?}/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability"
ep18_base="$ep18_root/outputs/runs/repeated_image_signal_recovery"
ep18_operations="$ep18_base/operations_job_$SLURM_JOB_ID"
mkdir -m 700 "$ep18_operations"
ep18_local="${L_SCRATCH_JOB:?}/ep18-repeated-cohort"
mkdir -m 700 "$ep18_local"
export TMPDIR="$ep18_local" MPLCONFIGDIR="$ep18_local/matplotlib"
mkdir -m 700 "$MPLCONFIGDIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1
unset PYTHONPATH PYTHONHOME
ep18_runtime="${GROUP_HOME:?}/users/zijiao/ep18_model_runtime_cpu241_py312_46842156"
ep18_python="$ep18_runtime/bin/python"
[[ -x "$ep18_python" && -f "$ep18_base/control001/sub-07/complete.json" ]]
ep18_deadline=$((${SLURM_JOB_END_TIME:?}-120))
cd "$ep18_root/outputs/code"
bash -n run_repeated_image_cohort001.sbatch
printf 'hostname=%s job_id=%s cpus=4 memory_mib=8192 workers=4 threads_per_worker=1\n' \
  "$(hostname)" "$SLURM_JOB_ID"
ep18_group() {
  local ep18_group_id=$1 ep18_index ep18_left
  for ((ep18_index=ep18_group_id+1; ep18_index<30; ep18_index+=4)); do
    ep18_left=$((ep18_deadline-$(date +%s)))
    (( ep18_left > 60 )) || return 75
    /usr/bin/time -v -o "$ep18_operations/resources_person_${ep18_index}.txt" \
      timeout --signal=TERM --kill-after=15s "$((ep18_left+15))s" \
      "$ep18_python" -u -B run_repeated_image_control.py \
      --config "$ep18_root/outputs/config/repeated_image_control001.json" \
      --participant-index "$ep18_index" --deadline-epoch "$ep18_deadline" \
      --rss-limit-bytes 1879048192 --operations "$ep18_operations/person_${ep18_index}"
  done
}
ep18_pids=()
for ep18_group_id in 0 1 2 3; do
  ep18_group "$ep18_group_id" > "$ep18_operations/group_${ep18_group_id}.out" \
    2> "$ep18_operations/group_${ep18_group_id}.err" &
  ep18_pids+=("$!")
done
ep18_exit=0
for ep18_pid in "${ep18_pids[@]}"; do
  if ! wait "$ep18_pid"; then ep18_exit=2; fi
done
printf 'worker_groups_exit=%s\n' "$ep18_exit"
(( ep18_exit == 0 )) || exit "$ep18_exit"
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 viz py-matplotlib/3.8.3_py312
ep18_left=$((SLURM_JOB_END_TIME-30-$(date +%s)))
(( ep18_left > 30 )) || exit 75
timeout --signal=TERM --kill-after=10s "${ep18_left}s" python3.12 -B report_repeated_image_control.py \
  --development-manifest "$ep18_root/outputs/runs/input_role_preparation/20261003-freeze_metadata01/development_manifest.json" \
  --repeated-root "$ep18_base/control001" \
  --core-report "$ep18_root/outputs/runs/baseline_models/core_direct_checkpoint_batch_report001/core_reference.complete.json" \
  --output-root "$ep18_base/report001" \
  --figure-root "$ep18_root/outputs/figures/repeated_image_control001"
printf 'all30_report_complete\n'
