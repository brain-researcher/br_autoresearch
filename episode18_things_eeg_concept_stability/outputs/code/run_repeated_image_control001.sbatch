#!/bin/bash
# First complete person measures cost; later arrays reuse the unchanged worker.
#SBATCH --job-name=ep18-repeated
#SBATCH --partition=russpold,normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/repeated_image_signal_recovery/slurm-%A_%a.out
#SBATCH --error=outputs/runs/repeated_image_signal_recovery/slurm-%A_%a.err
set -euo pipefail
umask 077
[[ $# == 0 && -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* ]]
[[ "${SLURM_CPUS_PER_TASK:-}" == 1 && "${SLURM_MEM_PER_NODE:-}" == 7168 ]]
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]]
ep18_root="${OAK:?}/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability"
ep18_base="$ep18_root/outputs/runs/repeated_image_signal_recovery"
ep18_index="${SLURM_ARRAY_TASK_ID:-0}"
[[ "$ep18_index" =~ ^[0-9]+$ && "$ep18_index" -lt 30 ]]
ep18_local="${L_SCRATCH_JOB:?}/ep18-repeated"
mkdir -m 700 "$ep18_local"
export TMPDIR="$ep18_local"
export MPLCONFIGDIR="$ep18_local/matplotlib"
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
[[ -x "$ep18_python" ]]
printf 'hostname=%s job_id=%s cpus=%s memory_mib=%s participant_index=%s\n' \
  "$(hostname)" "$SLURM_JOB_ID" "$SLURM_CPUS_PER_TASK" "$SLURM_MEM_PER_NODE" "$ep18_index"
cd "$ep18_root/outputs/code"
if [[ -z "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  module load math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 viz py-matplotlib/3.8.3_py312
  timeout --signal=TERM --kill-after=10s 300s python3.12 -B -m unittest -v \
    test_repeated_image_sequences test_repeated_image_control test_report_repeated_image_control \
    > "$ep18_base/checks_${SLURM_JOB_ID}.txt" 2>&1
  ep18_astra="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
  env PYTHONPATH="$ep18_astra/lib/python3.12/site-packages:$ep18_runtime/lib/python3.12/site-packages" \
    timeout --signal=TERM --kill-after=5s 60s "$ep18_astra/bin/astra-tools" \
    validate "$ep18_root/outputs/astra/v0.0.14/astra.yaml" --json \
    > "$ep18_base/astra_${SLURM_JOB_ID}.json" 2>&1 || \
    printf 'ASTRA validation incomplete; inspect the recorded diagnostics.\n'
  module purge
  unset PYTHONPATH PYTHONHOME
  module load devel python/3.12.1
  unset PYTHONPATH PYTHONHOME
fi
ep18_deadline=$((${SLURM_JOB_END_TIME:?}-90))
ep18_left=$((ep18_deadline-$(date +%s)))
(( ep18_left > 60 ))
/usr/bin/time -v -o "$ep18_base/resources_${SLURM_JOB_ID}.txt" \
  timeout --signal=TERM --kill-after=15s "$((ep18_left+15))s" \
  "$ep18_python" -u -B run_repeated_image_control.py \
  --config "$ep18_root/outputs/config/repeated_image_control001.json" \
  --participant-index "$ep18_index" --deadline-epoch "$ep18_deadline" \
  --operations "$ep18_base/operations_job_${SLURM_JOB_ID}"
