#!/bin/bash
# Only the explicitly reviewed, registered remainder of original work5/6/7.
# This script never submits, retries, extends, or reopens original fragments.
#SBATCH --job-name=ep21-full-workers-001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/decoder_training/full_remainders001-%A_%a.out
#SBATCH --error=outputs/decoder_training/full_remainders001-%A_%a.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep21_remainder_host=$(hostname)
printf 'REMAINDER_HOST=%s JOB=%s ARRAY=%s INDEX=%s CPUS=%s MEM_MIB=%s\n' \
  "$ep21_remainder_host" "${SLURM_JOB_ID:?}" "${SLURM_ARRAY_JOB_ID:?}" \
  "${SLURM_ARRAY_TASK_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}"
[[ "$ep21_remainder_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 1 \
   && "$SLURM_MEM_PER_NODE" == 7168 && "$SLURM_ARRAY_TASK_ID" =~ ^[456]$ ]]
printf -v ep21_remainder_work 'work_%04d' "$((SLURM_ARRAY_TASK_ID + 1))"
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
# Changed continuation and accounting paths must pass in this allocation
# before the new physical remainder receives empirical input. No job dispatch,
# scientific payload, or extra allocation occurs in these finite fixtures.
# Reuse the unchanged five helper checks passed in job46881800; this gate has72.
ep21_remainder_checks="${PWD}/outputs/decoder_training/remainder_admission_checks_job_${SLURM_JOB_ID}"
mkdir "$ep21_remainder_checks"
printf 'job=%s\nparent_charged_once_to=EP21\n' "$SLURM_JOB_ID" > "$ep21_remainder_checks/status.txt"
set +e
(
  cd outputs/decoder_training/tests || exit 2
  timeout --signal=TERM --kill-after=5s 180s python3.12 -B -m unittest -v \
    test_decoder_array_allocation test_submit_full_decoder_jobs \
    test_reviewed_pre_admission_replacement test_reviewed_deadline_partials \
    test_reviewed_startup_replacement \
    test_submit_deadline_remainders test_decoder_remainder_assembly test_decoder_assembly
) > "$ep21_remainder_checks/checks.txt" 2>&1
ep21_remainder_checks_exit=$?
set -e
printf 'required_checks_exit=%s\n' "$ep21_remainder_checks_exit" >> "$ep21_remainder_checks/status.txt"
[[ "$ep21_remainder_checks_exit" == 0 ]]
printf 'Required_exact_remainder_admission_checks=passed\n'

# Future fresh workers may opt into grouped durable writes after coordinator
# review of these six results. This optional qualification does not activate
# buffering in this remainder, and a failed check cannot block its fitting.
if [[ "$SLURM_ARRAY_TASK_ID" == 6 ]]; then
  set +e
  (
    cd outputs/decoder_training/tests || exit 2
    timeout --signal=TERM --kill-after=5s 60s python3.12 -B -m unittest -v \
      test_buffered_publication
  ) > "$ep21_remainder_checks/buffered_publication_checks.txt" 2>&1
  ep21_publication_checks_exit=$?
  printf 'Optional_buffered_publication_checks_exit=%s\n' "$ep21_publication_checks_exit"
  printf 'optional_buffered_publication_checks_exit=%s\n' "$ep21_publication_checks_exit" >> "$ep21_remainder_checks/status.txt"

  # Unactivated comparator arithmetic only; no tau responses or stage grant.
  # Reuse the same NumPy runtime and substantive parent without another job.
  (
    cd outputs/converter_training/tests || exit 2
    timeout --signal=TERM --kill-after=5s 60s python3.12 -B -m unittest -v \
      test_brain_loss
  ) > "$ep21_remainder_checks/brain_loss_checks.txt" 2>&1
  ep21_brain_loss_checks_exit=$?
  printf 'Optional_brain_loss_checks_exit=%s\n' "$ep21_brain_loss_checks_exit"
  printf 'optional_brain_loss_checks_exit=%s\n' "$ep21_brain_loss_checks_exit" >> "$ep21_remainder_checks/status.txt"
  set -e
fi

# Validate the changed program document in the same allocation. Navigation
# validation is recorded independently and does not gate authorized fitting.
ep21_remainder_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep21_remainder_dependencies="${GROUP_HOME}/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
set +e
PYTHONPATH="$ep21_remainder_astra/lib/python3.12/site-packages:$ep21_remainder_dependencies" \
  timeout --signal=TERM --kill-after=5s 30s "$ep21_remainder_astra/bin/astra-tools" validate \
  outputs/astra/v0.0.14/astra.yaml --json > "$ep21_remainder_checks/astra_validation.json" 2>&1
ep21_remainder_astra_exit=$?
set -e
printf 'astra_exit=%s\n' "$ep21_remainder_astra_exit" >> "$ep21_remainder_checks/status.txt"

# Three separately authorized implementation scopes, selected only for the
# first remainder (work_0007). Serial checks share this substantive allocation;
# no source payload, reader release, separate job or scientific simulation.
# The child changes modules only in its own shell. Its failure is recorded
# without changing the decoder's required gate or allocation-end deadline.
if [[ "$SLURM_ARRAY_TASK_ID" == 6 ]]; then
  set +e
  timeout --signal=TERM --kill-after=5s 540s bash <<'EP02_EP15_EP19_COMPONENTS'
set -uo pipefail
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-scikit-learn/1.5.1_py312 py-pytorch/2.4.1_py312 || exit 2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
module list
component_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
component_dependencies="${GROUP_HOME}/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
component_failures=0
run_component_checks() (
  component_episode=$1
  shift
  cd "${OAK:?}/users/zijiao/br_autoresearch/$component_episode" || exit 2
  component_dir="outputs/runs/component_checks_job_${SLURM_JOB_ID}"
  mkdir "$component_dir" || exit 2
  printf 'job=%s\nparent_charged_once_to=EP21\nempirical_execution=false\n' \
    "$SLURM_JOB_ID" > "$component_dir/status.txt" || exit 2
  exec > "$component_dir/wrapper.txt" 2>&1 || exit 2
  hostname
  printf 'SLURM_JOB_ID=%s CPU_LIMIT=1 MEMORY_GIB=7\n' "$SLURM_JOB_ID"
  timeout --foreground --signal=TERM --kill-after=5s 10s python3.12 -B -c \
    'import sys, numpy, scipy, sklearn; print(sys.version); print("numpy", numpy.__version__, "scipy", scipy.__version__, "sklearn", sklearn.__version__)'
  component_runtime_exit=$?
  printf 'runtime_exit=%s\n' "$component_runtime_exit" >> "$component_dir/status.txt" || exit 2
  [[ "$component_runtime_exit" == 0 ]] || exit "$component_runtime_exit"
  (
    cd outputs/tests || exit 2
    timeout --foreground --signal=TERM --kill-after=5s 120s python3.12 -B -m unittest -v "$@"
  ) > "$component_dir/tests.txt" 2>&1
  component_tests_exit=$?
  printf 'tests_exit=%s\n' "$component_tests_exit" >> "$component_dir/status.txt" || exit 2
  PYTHONPATH="$component_astra/lib/python3.12/site-packages:$component_dependencies" \
    timeout --foreground --signal=TERM --kill-after=5s 30s "$component_astra/bin/astra-tools" validate \
    outputs/astra/v0.0.14/astra.yaml --json > "$component_dir/astra_validation.json" 2>&1
  component_astra_exit=$?
  printf 'astra_exit=%s\n' "$component_astra_exit" >> "$component_dir/status.txt" || exit 2
  [[ "$component_tests_exit" == 0 && "$component_astra_exit" == 0 ]]
)
run_component_checks episode02_dopamine_learning_rate_causality test_logistic_normal_observation || component_failures=1
run_component_checks episode15_mdtb_topography_vs_geometry \
  test_local_parcel_prediction test_development_lopo test_smooth_relocation \
  test_matched_random_warp test_local_isometry_variants \
  test_paired_stretch_variants test_anatomical_group_baseline \
  test_matched_random_basis || component_failures=1
run_component_checks episode19_causal_motor_forecasting_benchmark test_boosted_hazard_backend test_temporal_hazard_backend || component_failures=1
exit "$component_failures"
EP02_EP15_EP19_COMPONENTS
  ep21_component_child_exit=$?
  printf 'EP02_EP15_EP19_component_child_exit=%s\n' "$ep21_component_child_exit"
  printf 'optional_component_child_exit=%s\n' "$ep21_component_child_exit" >> "$ep21_remainder_checks/status.txt"
  set -e
fi

# Worker independently verifies actual scheduler resources, exact accepted
# registration, fresh paths and the allocation end minus 300 seconds.
exec timeout --signal=TERM --kill-after=10s 28530s python3.12 -u -B \
  outputs/decoder_training/tools/train_decoder_remainder.py \
  --authorization outputs/decoder_training_full_authorization.json \
  --manifest outputs/decoder_training/full_continuation_001/work_manifest.json \
  --remainder-plan outputs/decoder_training/full_continuation_001/deadline_remainders_001.json \
  --work-item "$ep21_remainder_work" --max-seconds 28500
