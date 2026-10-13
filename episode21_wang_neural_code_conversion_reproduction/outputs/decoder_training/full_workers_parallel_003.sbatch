#!/bin/bash
# Unsubmitted candidate: work48 only, within the existing amended total caps.
# Optional component checks share this parent; no retries or successors.
#SBATCH --job-name=ep21-full-workers-003
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep21_parallel_host=$(hostname)
printf 'PARALLEL_HOST=%s JOB=%s ARRAY=%s INDEX=%s CPUS=%s MEM_MIB=%s\n' \
  "$ep21_parallel_host" "${SLURM_JOB_ID:?}" "${SLURM_ARRAY_JOB_ID:?}" \
  "${SLURM_ARRAY_TASK_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}"
[[ "$ep21_parallel_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 1 && "$SLURM_MEM_PER_NODE" == 7168 ]]
ep21_parallel_index=$SLURM_ARRAY_TASK_ID
[[ "$ep21_parallel_index" =~ ^[0-9]+$ ]]
[[ "$ep21_parallel_index" == 47 ]]
cd "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-h5py/3.10.0_py312 py-scipy/1.12.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
for ep21_parallel_thread_name in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS \
  NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS BLIS_NUM_THREADS
do
  [[ "${!ep21_parallel_thread_name}" == 1 ]]
done
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
[[ -n "${L_SCRATCH_JOB:?}" && -d "$L_SCRATCH_JOB" ]]
export TMPDIR="$L_SCRATCH_JOB"
ep21_parallel_checks="${PWD}/outputs/decoder_training/parallel003_checks_job_${SLURM_JOB_ID}"
mkdir "$ep21_parallel_checks"
printf 'job=%s\nparent_charged_once_to=EP21\n' "$SLURM_JOB_ID" > "$ep21_parallel_checks/status.txt"
set +e
(
  cd outputs/decoder_training/tests || exit 2
  timeout --signal=TERM --kill-after=5s 180s python3.12 -B -m unittest -v \
    test_submit_full_decoder_jobs test_reviewed_deadline_partials test_reviewed_io_quarantine \
    test_decoder_parallelism_amendment test_worker_scratch_policy \
    test_train_decoder_work_item test_train_layer_continuation \
    test_decoder_array_allocation test_decoder_remainder_bundle \
    test_reviewed_component_startup_replacement test_component_startup_integration
) > "$ep21_parallel_checks/checks.txt" 2>&1
ep21_parallel_checks_exit=$?
set -e
printf 'required_checks_exit=%s\n' "$ep21_parallel_checks_exit" >> "$ep21_parallel_checks/status.txt"
[[ "$ep21_parallel_checks_exit" == 0 ]]
printf 'Required_parallelism_checks=passed\n'
# Serial optional engineering checks; this file does not activate salvage.
# Every child inherits one CPU and the parent's unchanged memory envelope.
# The outer420s guard includes child setup; each operation has its own limit.
set +e
EP21_COMPONENT_CHECK_DIR="$ep21_parallel_checks" \
  timeout --signal=TERM --kill-after=5s 420s bash <<'EP21_OPTIONAL_COMPONENTS' 2>&1 | \
  tee "$ep21_parallel_checks/optional_wrapper.txt" > /dev/null
set -uo pipefail
ep21_component_checks=${EP21_COMPONENT_CHECK_DIR:?}
ep21_component_base="${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
cd "$ep21_component_base" || exit 2
# Assertion failures stop that unittest batch; other independent checks continue.
# A new storage error or any log-write/read failure stops this parent before fit.
ep21_check_storage_text() {
  LC_ALL=C grep -E -i -q -- '^[[:space:]]*(OSError|IOError|BrokenPipeError):[[:space:]]+\[Errno[[:space:]]+(5|108)\]|^[^{}]*:[[:space:]]*(Input/output error|Cannot send after transport endpoint shutdown)[[:space:]]*$' "$1"
  case $? in 1) return 0 ;; *) return 74 ;; esac
}
(
  cd outputs/decoder_training/tests || exit 2
  timeout --foreground --signal=TERM --kill-after=5s 180s python3.12 -B -m unittest -v -f \
    test_reviewed_io_recovery test_decoder_io_recovery_assembly \
    test_decoder_assembly test_decoder_remainder_assembly \
    test_io_recovery_accounting test_io_remainder_worker test_submit_io_remainders
) 2>&1 | tee "$ep21_component_checks/io_recovery_component_checks.txt" > /dev/null
ep21_recovery_pipe=("${PIPESTATUS[@]}")
[[ "${ep21_recovery_pipe[1]}" == 0 ]] || exit 74
ep21_recovery_component_exit=${ep21_recovery_pipe[0]}
ep21_check_storage_text "$ep21_component_checks/io_recovery_component_checks.txt" || exit 74
printf 'optional_io_recovery_component_exit=%s\n' "$ep21_recovery_component_exit" >> "$ep21_component_checks/status.txt" || exit 74

# Existing PyTorch module only. Its environment cannot leak into decoder fitting.
(
set -uo pipefail
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-scikit-learn/1.5.1_py312 py-pytorch/2.4.1_py312 || exit 2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
cd "${OAK:?}/users/zijiao/br_autoresearch/episode19_causal_motor_forecasting_benchmark" || exit 2
ep19_gru_checks="outputs/runs/gru_component_checks_job_${SLURM_JOB_ID}"
mkdir "$ep19_gru_checks" || exit 74
printf 'job=%s\nparent_charged_once_to=EP21\nempirical_execution=false\n' \
  "$SLURM_JOB_ID" > "$ep19_gru_checks/status.txt" || exit 74
(
  cd outputs/tests || exit 2
  timeout --foreground --signal=TERM --kill-after=5s 120s python3.12 -B -m unittest -v -f \
    test_gru_forecast_backend test_tcn_forecast_backend test_eegnet_forecast_backend
) 2>&1 | tee "$ep19_gru_checks/tests.txt" > /dev/null
ep19_gru_pipe=("${PIPESTATUS[@]}")
[[ "${ep19_gru_pipe[1]}" == 0 ]] || exit 74
ep19_gru_exit=${ep19_gru_pipe[0]}
ep21_check_storage_text "$ep19_gru_checks/tests.txt" || exit 74
printf 'tests_exit=%s\n' "$ep19_gru_exit" >> "$ep19_gru_checks/status.txt" || exit 74
exit "$ep19_gru_exit"
) 2>&1 | tee "$ep21_component_checks/ep19_gru_wrapper.txt" > /dev/null
ep21_gru_pipe=("${PIPESTATUS[@]}")
[[ "${ep21_gru_pipe[1]}" == 0 ]] || exit 74
ep21_gru_component_exit=${ep21_gru_pipe[0]}
[[ "$ep21_gru_component_exit" != 74 ]] || exit 74
ep21_check_storage_text "$ep21_component_checks/ep19_gru_wrapper.txt" || exit 74
printf 'optional_ep19_gru_exit=%s\n' "$ep21_gru_component_exit" >> "$ep21_component_checks/status.txt" || exit 74

ep21_component_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep21_component_dependencies="${GROUP_HOME}/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
(
  cd "${OAK}/users/zijiao/br_autoresearch/episode19_causal_motor_forecasting_benchmark" || exit 2
  PYTHONPATH="$ep21_component_astra/lib/python3.12/site-packages:$ep21_component_dependencies" \
    timeout --foreground --signal=TERM --kill-after=5s 30s "$ep21_component_astra/bin/astra-tools" validate \
    outputs/astra/v0.0.14/astra.yaml --json
) 2>&1 | tee "$ep21_component_checks/ep19_astra_validation.json" > /dev/null
ep21_ep19_astra_pipe=("${PIPESTATUS[@]}")
[[ "${ep21_ep19_astra_pipe[1]}" == 0 ]] || exit 74
ep21_ep19_astra_exit=${ep21_ep19_astra_pipe[0]}
ep21_check_storage_text "$ep21_component_checks/ep19_astra_validation.json" || exit 74
printf 'optional_ep19_astra_exit=%s\n' "$ep21_ep19_astra_exit" >> "$ep21_component_checks/status.txt" || exit 74
PYTHONPATH="$ep21_component_astra/lib/python3.12/site-packages:$ep21_component_dependencies" \
  timeout --foreground --signal=TERM --kill-after=5s 30s "$ep21_component_astra/bin/astra-tools" validate \
  outputs/astra/v0.0.14/astra.yaml --json 2>&1 | \
  tee "$ep21_component_checks/ep21_astra_validation.json" > /dev/null
ep21_ep21_astra_pipe=("${PIPESTATUS[@]}")
[[ "${ep21_ep21_astra_pipe[1]}" == 0 ]] || exit 74
ep21_ep21_astra_exit=${ep21_ep21_astra_pipe[0]}
ep21_check_storage_text "$ep21_component_checks/ep21_astra_validation.json" || exit 74
printf 'optional_ep21_astra_exit=%s\n' "$ep21_ep21_astra_exit" >> "$ep21_component_checks/status.txt" || exit 74
[[ "$ep21_recovery_component_exit" == 0 && "$ep21_gru_component_exit" == 0 \
   && "$ep21_ep19_astra_exit" == 0 && "$ep21_ep21_astra_exit" == 0 ]]
EP21_OPTIONAL_COMPONENTS
ep21_optional_pipe=("${PIPESTATUS[@]}")
ep21_optional_components_exit=${ep21_optional_pipe[0]}
ep21_optional_log_exit=${ep21_optional_pipe[1]}
set -e
printf 'optional_components_exit=%s\noptional_log_write_exit=%s\n' \
  "$ep21_optional_components_exit" "$ep21_optional_log_exit" >> "$ep21_parallel_checks/status.txt"
[[ "$ep21_optional_log_exit" == 0 && "$ep21_optional_components_exit" != 74 ]] || exit 74
# Inspect only this child's named logs, including partial logs after a timeout.
# Missing optional outputs mean that operation never reached log publication.
for ep21_optional_evidence in \
  "$ep21_parallel_checks/optional_wrapper.txt" \
  "$ep21_parallel_checks/io_recovery_component_checks.txt" \
  "$ep21_parallel_checks/ep19_gru_wrapper.txt" \
  "$ep21_parallel_checks/ep19_astra_validation.json" \
  "$ep21_parallel_checks/ep21_astra_validation.json" \
  "${OAK}/users/zijiao/br_autoresearch/episode19_causal_motor_forecasting_benchmark/outputs/runs/gru_component_checks_job_${SLURM_JOB_ID}/tests.txt"
do
  [[ -e "$ep21_optional_evidence" ]] || continue
  set +e
  LC_ALL=C grep -E -i -q -- '^[[:space:]]*(OSError|IOError|BrokenPipeError):[[:space:]]+\[Errno[[:space:]]+(5|108)\]|^[^{}]*:[[:space:]]*(Input/output error|Cannot send after transport endpoint shutdown)[[:space:]]*$' \
    "$ep21_optional_evidence"
  ep21_optional_storage_match=$?
  set -e
  [[ "$ep21_optional_storage_match" == 1 ]] || exit 74
done
# Independent optional check failure does not change the work48 fit admission.
printf -v ep21_parallel_work 'work_%04d' "$((ep21_parallel_index + 1))"
printf 'WORK_ITEM=%s PUBLICATION_BLOCKS=4 NUMERICAL_BLOCK_UNITS=64\n' "$ep21_parallel_work"
# Existing allocation EndTime minus300s bounds all fitting after startup checks.
exec timeout --signal=TERM --kill-after=10s 28530s python3.12 -u -B \
  outputs/decoder_training/tools/train_decoder_work_item.py \
  --authorization outputs/decoder_training_full_authorization.json \
  --manifest outputs/decoder_training/full_continuation_001/work_manifest.json \
  --work-item "$ep21_parallel_work" --max-seconds 28500 --publication-blocks 4
