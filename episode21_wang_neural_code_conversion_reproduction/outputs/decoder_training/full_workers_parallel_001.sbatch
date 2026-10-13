#!/bin/bash
# Explicit user-authorized concurrency expansion, preserving all total caps.
# Fresh immutable assignments only; no retries, submissions or successors.
#SBATCH --job-name=ep21-full-workers-001
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
[[ ( "$ep21_parallel_index" -ge 22 && "$ep21_parallel_index" -le 29 ) || \
   ( "$ep21_parallel_index" -ge 33 && "$ep21_parallel_index" -le 44 ) || \
   ( "$ep21_parallel_index" -ge 47 && "$ep21_parallel_index" -le 59 ) || \
   ( "$ep21_parallel_index" -ge 62 && "$ep21_parallel_index" -le 74 ) ]]
cd "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-h5py/3.10.0_py312 py-scipy/1.12.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
[[ -n "${L_SCRATCH_JOB:?}" && -d "$L_SCRATCH_JOB" ]]
export TMPDIR="$L_SCRATCH_JOB"
ep21_parallel_checks="${PWD}/outputs/decoder_training/parallel_checks_job_${SLURM_JOB_ID}"
mkdir "$ep21_parallel_checks"
printf 'job=%s\nparent_charged_once_to=EP21\n' "$SLURM_JOB_ID" > "$ep21_parallel_checks/status.txt"
set +e
(
  cd outputs/decoder_training/tests || exit 2
  timeout --signal=TERM --kill-after=5s 180s python3.12 -B -m unittest -v \
    test_submit_full_decoder_jobs test_reviewed_deadline_partials test_reviewed_io_quarantine \
    test_decoder_parallelism_amendment test_worker_scratch_policy \
    test_train_decoder_work_item test_train_layer_continuation \
    test_decoder_array_allocation test_decoder_remainder_bundle
) > "$ep21_parallel_checks/checks.txt" 2>&1
ep21_parallel_checks_exit=$?
set -e
printf 'required_checks_exit=%s\n' "$ep21_parallel_checks_exit" >> "$ep21_parallel_checks/status.txt"
[[ "$ep21_parallel_checks_exit" == 0 ]]
printf 'Required_parallelism_checks=passed\n'
# Independent pending component fixes share only the first substantive parent.
# They neither admit quarantined files nor change decoder data or fitting.
if [[ "$ep21_parallel_index" == 22 ]]; then
  set +e
  (
    cd outputs/decoder_training/tests || exit 2
    timeout --signal=TERM --kill-after=5s 60s python3.12 -B -m unittest -v test_inspect_closed_io_fragments
  ) > "$ep21_parallel_checks/inspection_component_checks.txt" 2>&1
  ep21_inspection_component_exit=$?
  printf 'optional_inspection_component_exit=%s\n' "$ep21_inspection_component_exit" >> "$ep21_parallel_checks/status.txt"
  timeout --signal=TERM --kill-after=5s 90s bash <<'EP15_SOURCE_ASSERTION' > "$ep21_parallel_checks/ep15_wrapper.txt" 2>&1
set -uo pipefail
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-scikit-learn/1.5.1_py312 || exit 2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
cd "${OAK:?}/users/zijiao/br_autoresearch/episode15_mdtb_topography_vs_geometry" || exit 2
ep15_source_checks="outputs/runs/source_assertion_checks_job_${SLURM_JOB_ID}"
mkdir "$ep15_source_checks" || exit 2
printf 'job=%s\nparent_charged_once_to=EP21\nempirical_execution=false\n' \
  "$SLURM_JOB_ID" > "$ep15_source_checks/status.txt" || exit 2
(
  cd outputs/tests || exit 2
  timeout --foreground --signal=TERM --kill-after=5s 60s python3.12 -B -m unittest -v test_local_functional_source
) > "$ep15_source_checks/tests.txt" 2>&1
ep15_source_exit=$?
printf 'tests_exit=%s\n' "$ep15_source_exit" >> "$ep15_source_checks/status.txt"
exit "$ep15_source_exit"
EP15_SOURCE_ASSERTION
  ep21_ep15_source_exit=$?
  printf 'optional_ep15_source_exit=%s\n' "$ep21_ep15_source_exit" >> "$ep21_parallel_checks/status.txt"
  ep21_parallel_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
  ep21_parallel_dependencies="${GROUP_HOME}/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
  PYTHONPATH="$ep21_parallel_astra/lib/python3.12/site-packages:$ep21_parallel_dependencies" \
    timeout --signal=TERM --kill-after=5s 30s "$ep21_parallel_astra/bin/astra-tools" validate \
    outputs/astra/v0.0.14/astra.yaml --json > "$ep21_parallel_checks/astra_validation.json" 2>&1
  ep21_parallel_astra_exit=$?
  printf 'astra_exit=%s\n' "$ep21_parallel_astra_exit" >> "$ep21_parallel_checks/status.txt"
  set -e
fi
printf -v ep21_parallel_work 'work_%04d' "$((ep21_parallel_index + 1))"
printf 'WORK_ITEM=%s PUBLICATION_BLOCKS=4 NUMERICAL_BLOCK_UNITS=64\n' "$ep21_parallel_work"
# Existing allocation EndTime minus300s bounds all fitting after startup checks.
exec timeout --signal=TERM --kill-after=10s 28530s python3.12 -u -B \
  outputs/decoder_training/tools/train_decoder_work_item.py \
  --authorization outputs/decoder_training_full_authorization.json \
  --manifest outputs/decoder_training/full_continuation_001/work_manifest.json \
  --work-item "$ep21_parallel_work" --max-seconds 28500 --publication-blocks 4
