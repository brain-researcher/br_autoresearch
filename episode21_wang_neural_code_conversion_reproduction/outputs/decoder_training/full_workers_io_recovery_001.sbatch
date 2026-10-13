#!/bin/bash
# Manual fresh assignments after the fixed five I/O failures were quarantined.
# Original failed fragments remain untouched; no retry/dispatch/successor.
#SBATCH --job-name=ep21-full-workers-001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep21_io_host=$(hostname)
printf 'RECOVERY_HOST=%s JOB=%s ARRAY=%s INDEX=%s CPUS=%s MEM_MIB=%s\n' \
  "$ep21_io_host" "${SLURM_JOB_ID:?}" "${SLURM_ARRAY_JOB_ID:?}" \
  "${SLURM_ARRAY_TASK_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}"
[[ "$ep21_io_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 1 && "$SLURM_MEM_PER_NODE" == 7168 ]]
case "$SLURM_ARRAY_TASK_ID" in 14|18|19|20|21) ;; *) exit 2 ;; esac
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
ep21_io_checks="${PWD}/outputs/decoder_training/io_recovery_checks_job_${SLURM_JOB_ID}"
mkdir "$ep21_io_checks"
printf 'job=%s\nparent_charged_once_to=EP21\n' "$SLURM_JOB_ID" > "$ep21_io_checks/status.txt"
set +e
(
  cd outputs/decoder_training/tests || exit 2
  timeout --signal=TERM --kill-after=5s 120s python3.12 -B -m unittest -v \
    test_submit_full_decoder_jobs test_reviewed_deadline_partials test_reviewed_io_quarantine
) > "$ep21_io_checks/checks.txt" 2>&1
ep21_io_checks_exit=$?
set -e
printf 'required_checks_exit=%s\n' "$ep21_io_checks_exit" >> "$ep21_io_checks/status.txt"
[[ "$ep21_io_checks_exit" == 0 ]]
# Four unactivated final-checkpoint fixtures only, on the first fresh item.
# No converter driver, scientific inputs or extra allocation. Child runtime
# and failure are isolated from the independent decoder fitting environment.
if [[ "$SLURM_ARRAY_TASK_ID" == 14 ]]; then
  set +e
  timeout --signal=TERM --kill-after=5s 100s bash <<'EP21_FINAL_ARTIFACT_CHECKS' > "$ep21_io_checks/final_artifact_checks.txt" 2>&1
set -uo pipefail
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312 py-pytorch/2.4.1_py312 || exit 2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
cd outputs/converter_training/tests || exit 2
timeout --foreground --signal=TERM --kill-after=5s 60s python3.12 -B -m unittest -v test_final_content_artifact
EP21_FINAL_ARTIFACT_CHECKS
  ep21_final_artifact_exit=$?
  set -e
  printf 'optional_final_artifact_exit=%s\n' "$ep21_final_artifact_exit" >> "$ep21_io_checks/status.txt"
fi

# The next fresh worker carries only the newly selected EP20 array-component
# fixtures. Explicit parameters, no reference fit or candidate simulation.
# This serial optional child shares the unchanged parent resource envelope.
if [[ "$SLURM_ARRAY_TASK_ID" == 18 ]]; then
  set +e
  timeout --signal=TERM --kill-after=5s 110s bash <<'EP20_FIELD_COMPONENT_CHECKS' > "$ep21_io_checks/ep20_field_wrapper.txt" 2>&1
set -uo pipefail
cd "${OAK:?}/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign" || exit 2
ep20_field_checks="outputs/runs/field_component_checks_job_${SLURM_JOB_ID}"
mkdir "$ep20_field_checks" || exit 2
printf 'job=%s\nparent_charged_once_to=EP21\nreference_fit=false\ncandidate_execution=false\n' \
  "$SLURM_JOB_ID" > "$ep20_field_checks/status.txt" || exit 2
(
  cd outputs/tools || exit 2
  timeout --foreground --signal=TERM --kill-after=5s 60s python3.12 -B -m unittest -v \
    test_development_fields test_field_power_scales
) > "$ep20_field_checks/tests.txt" 2>&1
ep20_field_test_exit=$?
printf 'tests_exit=%s\n' "$ep20_field_test_exit" >> "$ep20_field_checks/status.txt"
ep20_field_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep20_field_dependencies="${GROUP_HOME}/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
PYTHONPATH="$ep20_field_astra/lib/python3.12/site-packages:$ep20_field_dependencies" \
  timeout --foreground --signal=TERM --kill-after=5s 30s "$ep20_field_astra/bin/astra-tools" validate \
  outputs/astra/v0.0.14/astra.yaml --json > "$ep20_field_checks/astra_validation.json" 2>&1
ep20_field_astra_exit=$?
printf 'astra_exit=%s\n' "$ep20_field_astra_exit" >> "$ep20_field_checks/status.txt"
[[ "$ep20_field_test_exit" == 0 && "$ep20_field_astra_exit" == 0 ]]
EP20_FIELD_COMPONENT_CHECKS
  ep20_field_child_exit=$?
  set -e
  printf 'optional_ep20_field_exit=%s\n' "$ep20_field_child_exit" >> "$ep21_io_checks/status.txt"
fi

printf -v ep21_io_work 'work_%04d' "$((SLURM_ARRAY_TASK_ID + 1))"
printf 'WORK_ITEM=%s PUBLICATION_BLOCKS=4 NUMERICAL_BLOCK_UNITS=64\n' "$ep21_io_work"
# Existing empirical kernel clips its deadline to actual Slurm EndTime minus300s.
exec timeout --signal=TERM --kill-after=10s 28530s python3.12 -u -B \
  outputs/decoder_training/tools/train_decoder_work_item.py \
  --authorization outputs/decoder_training_full_authorization.json \
  --manifest outputs/decoder_training/full_continuation_001/work_manifest.json \
  --work-item "$ep21_io_work" --max-seconds 28500 --publication-blocks 4
