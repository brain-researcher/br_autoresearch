#!/bin/bash
# One finite full30-person preparation, including the two changed cache tests.
# The independently scoped EP21 hardware checks share this parent serially.
# Coordinator precreates only operations_root; all scientific output is fresh.
#SBATCH --job-name=ep18-eegprep
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/eeg_preprocessing/operations_20261012_001/%x-%j.out
#SBATCH --error=outputs/runs/eeg_preprocessing/operations_20261012_001/%x-%j.err
set -euo pipefail
umask 077
[[ $# == 0 ]]
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* ]]
[[ "${SLURM_CPUS_PER_TASK:-}" == 1 && "${SLURM_MEM_PER_NODE:-}" == 7168 ]]
[[ "${SLURM_JOB_NUM_NODES:-}" == 1 && "${SLURM_JOB_CPUS_PER_NODE:-}" == 1 ]]
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]]
: "${OAK:?}" "${GROUP_HOME:?}" "${L_SCRATCH_JOB:?}"
ep18_episode="$OAK/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability"
ep18_code="$ep18_episode/outputs/code"
ep18_output="$ep18_episode/outputs/runs/eeg_preprocessing/development_20261012_001"
ep18_release="$ep18_episode/outputs/config/development_eeg_release_20261012_001.json"
ep18_operations_root="$ep18_episode/outputs/runs/eeg_preprocessing/operations_20261012_001"
[[ -d "$ep18_operations_root" && ! -e "$ep18_output" ]]
ep18_operations="$ep18_operations_root/initial_${SLURM_JOB_ID}"
mkdir -m 700 "$ep18_operations"
ep18_stage=allocation_admission
ep18_started=$SECONDS
ep18_deadline=0
ep21_checks_exit=-1
ep18_checks_exit=-1
finish() {
  ep18_exit=$?
  trap - EXIT
  printf '{"stage":"%s","exit_code":%s,"job_id":"%s","elapsed_seconds":%s,"deadline_unix":%s,"allocated_cpus":1,"reserved_cpu_hours":2,"ep18_checks_exit":%s,"ep21_checks_exit":%s,"automatic_retry":false,"automatic_successor":false}\n' \
    "$ep18_stage" "$ep18_exit" "$SLURM_JOB_ID" "$((SECONDS-ep18_started))" "$ep18_deadline" \
    "$ep18_checks_exit" "$ep21_checks_exit" > "$ep18_operations/exit.json"
  exit "$ep18_exit"
}
trap finish EXIT
ep18_start=${SLURM_JOB_START_TIME:?}
ep18_end=${SLURM_JOB_END_TIME:?}
[[ "$ep18_start" =~ ^[0-9]+$ && "$ep18_end" =~ ^[0-9]+$ ]]
ep18_now=$(date +%s)
(( ep18_start > 0 && ep18_start <= ep18_now && ep18_end > ep18_start ))
ep18_deadline=$((ep18_start+7200-60))
if (( ep18_end-60 < ep18_deadline )); then ep18_deadline=$((ep18_end-60)); fi
if (( 1791927000 < ep18_deadline )); then ep18_deadline=1791927000; fi
phase_window() {
  local ep18_cap=$1 ep18_current ep18_outer_end
  ep18_current=$(date +%s)
  ep18_outer_end=$((ep18_current+ep18_cap))
  if (( ep18_deadline-5 < ep18_outer_end )); then ep18_outer_end=$((ep18_deadline-5)); fi
  ep18_phase_seconds=$((ep18_outer_end-ep18_current))
  (( ep18_phase_seconds > 20 )) || return 75
  ep18_phase_deadline=$((ep18_outer_end-10))
}
phase_window 60
printf 'hostname=%s job_id=%s start=%s end=%s deadline=%s cpus=1 memory_mib=7168\n' \
  "$(hostname)" "$SLURM_JOB_ID" "$ep18_start" "$ep18_end" "$ep18_deadline"
ep18_free_kib=$(df -Pk "$L_SCRATCH_JOB" | awk 'NR==2 {print $4}')
[[ "$ep18_free_kib" =~ ^[0-9]+$ && "$ep18_free_kib" -ge 1048576 ]]
ep18_local="$L_SCRATCH_JOB/ep18-eegprep-${SLURM_JOB_ID}"
mkdir -m 700 "$ep18_local"
mkdir -m 700 "$ep18_local/tmp" "$ep18_local/cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export TMPDIR="$ep18_local/tmp" XDG_CACHE_HOME="$ep18_local/cache"
ulimit -c 0
source /share/software/user/open/lmod/lmod/init/bash
module purge
module load devel python/3.12.1
unset PYTHONPATH PYTHONHOME
ep18_runtime="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156"
ep18_python="$ep18_runtime/bin/python"
[[ -x "$ep18_python" ]]

# EP21 receives no EP18 data. Its tests mock hardware and read no neural
# payload. Record independently; failure cannot authorize release of GPUs.
ep18_stage=independent_ep21_hardware_checks
phase_window 180
ep21_episode="$OAK/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
ep21_checks="$ep21_episode/outputs/runs/content_launcher_qualification/hardware001"
[[ -d "$ep21_checks" ]]
set +e
(
  set -euo pipefail
  mkdir -m 700 "$ep21_checks/parent_${SLURM_JOB_ID}"
  ep21_record="$ep21_checks/parent_${SLURM_JOB_ID}"
  ulimit -S -f 65536
  module purge
  unset PYTHONPATH PYTHONHOME
  module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312 py-pytorch/2.4.1_py312
  cd "$ep21_episode/outputs/converter_training/tests"
  timeout --signal=TERM --kill-after=5s "${ep18_phase_seconds}s" python3.12 -B -m unittest \
    test_content_cuda_remaining_hardware test_content_cuda_remaining_admission \
    test_content_cuda_remaining_training > "$ep21_record/checks.txt" 2>&1
  printf 'checks_exit=0\nparent_job=%s\nadditional_allocated_CPU_hours=0\n' "$SLURM_JOB_ID" > "$ep21_record/status.txt"
) 
ep21_checks_exit=$?
set -e
printf 'ep21_checks_exit=%s\n' "$ep21_checks_exit" > "$ep18_operations/ep21_companion.txt"

ep18_stage=changed_cache_checks
phase_window 180
cd "$ep18_code"
set +e
timeout --signal=TERM --kill-after=5s "${ep18_phase_seconds}s" "$ep18_python" -B -m unittest \
  test_continuous_eeg_cache > "$ep18_operations/cache_checks.txt" 2>&1
ep18_checks_exit=$?
set -e
(( ep18_checks_exit == 0 )) || exit "$ep18_checks_exit"

ep18_stage=astra_validation
ep18_astra_env="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
phase_window 60
set +e
env PYTHONPATH="$ep18_astra_env/lib/python3.12/site-packages:$ep18_runtime/lib/python3.12/site-packages" \
  timeout --signal=TERM --kill-after=5s "${ep18_phase_seconds}s" "$ep18_astra_env/bin/astra-tools" validate \
  "$ep18_episode/outputs/astra/v0.0.14/astra.yaml" --json > "$ep18_operations/astra_validation.json" 2>&1
ep18_astra_exit=$?
phase_window 60
env PYTHONPATH="$ep18_astra_env/lib/python3.12/site-packages:$ep18_runtime/lib/python3.12/site-packages" \
  timeout --signal=TERM --kill-after=5s "${ep18_phase_seconds}s" "$ep18_astra_env/bin/astra-tools" validate \
  "$ep21_episode/outputs/astra/v0.0.14/astra.yaml" --json > "$ep18_operations/ep21_astra_validation.json" 2>&1
ep21_astra_exit=$?
set -e
printf 'ep18_astra_exit=%s\nep21_astra_exit=%s\n' "$ep18_astra_exit" "$ep21_astra_exit" > "$ep18_operations/astra_exits.txt"

ep18_stage=full_development_EEG_preparation
phase_window 7200
cd "$ep18_episode"
timeout --signal=TERM --kill-after=5s "${ep18_phase_seconds}s" "$ep18_python" -B \
  "$ep18_code/run_development_eeg_preprocessing.py" --release-manifest "$ep18_release" \
  --output-dir "$ep18_output" --local-dir "$ep18_local/participants" \
  --deadline-unix "$ep18_phase_deadline" --rss-byte-cap 6442450944 \
  --durable-byte-cap 12884901888 > "$ep18_operations/preparation.txt" \
  2> "$ep18_operations/preparation_stderr.txt"
ep18_stage=complete_development_EEG_preparation
