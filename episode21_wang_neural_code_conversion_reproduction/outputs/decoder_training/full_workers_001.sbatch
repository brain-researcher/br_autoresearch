#!/bin/bash
# Coordinator submits a finite authorized array after complete state preparation.
# Submission validates immutable manifest order work_0001 through work_NNNN.
#SBATCH --job-name=ep21-full-fit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/decoder_training/full_workers001-%A_%a.out
#SBATCH --error=outputs/decoder_training/full_workers001-%A_%a.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep21_worker_host=$(hostname)
printf 'HOST=%s JOB=%s ARRAY=%s INDEX=%s CPUS=%s MEM_MIB=%s\n' "$ep21_worker_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_ARRAY_JOB_ID:?}" "${SLURM_ARRAY_TASK_ID:?}" \
  "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}"
[[ "$ep21_worker_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 1 && "$SLURM_MEM_PER_NODE" == 7168 ]]
[[ "$SLURM_ARRAY_TASK_ID" =~ ^(0|[1-9][0-9]*)$ ]]
printf -v ep21_work_item 'work_%04d' "$((SLURM_ARRAY_TASK_ID + 1))"
cd "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
[[ -f outputs/decoder_training_full_authorization.json \
   && -f outputs/decoder_training/full_continuation_001/work_manifest.json ]]
source /share/software/user/open/lmod/lmod/init/bash
module purge
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-h5py/3.10.0_py312 py-scipy/1.12.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
[[ -n "${L_SCRATCH_JOB:?}" && -d "$L_SCRATCH_JOB" ]]
export TMPDIR="$L_SCRATCH_JOB"
# Only the first, never-replayed work item hosts these three source-free
# component checks. They share the substantive allocation's fixed ceiling;
# an unrelated episode's failure is recorded without stopping EP21 fitting.
if [[ "$SLURM_ARRAY_TASK_ID" == 0 ]]; then
  set +e
  ep21_shared_astra="${GROUP_HOME:-}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
  for ep21_check_episode in \
    episode02_dopamine_learning_rate_causality \
    episode15_mdtb_topography_vs_geometry \
    episode20_neurocam_prior_guided_codesign
  do
    (
      cd "${OAK}/users/zijiao/br_autoresearch/${ep21_check_episode}" || exit 2
      ep21_shared_dir="outputs/runs/component_checks_job_${SLURM_JOB_ID}"
      mkdir -p outputs/runs || exit 2
      mkdir "$ep21_shared_dir" || exit 2
      case "$ep21_check_episode" in
        episode02_dopamine_learning_rate_causality)
          timeout --signal=TERM --kill-after=5s 60s python3.12 -B -m unittest \
            discover -s outputs/tests -p test_actr_port.py -v \
            > "$ep21_shared_dir/tests.txt" 2>&1 ;;
        episode15_mdtb_topography_vs_geometry)
          timeout --signal=TERM --kill-after=5s 120s python3.12 -B -m unittest \
            discover -s outputs/tests -p test_fit_m3_task_a.py -v \
            > "$ep21_shared_dir/tests.txt" 2>&1 ;;
        episode20_neurocam_prior_guided_codesign)
          timeout --signal=TERM --kill-after=5s 60s python3.12 -B \
            outputs/tools/test_loaded_node.py -v \
            > "$ep21_shared_dir/tests.txt" 2>&1 ;;
      esac
      ep21_shared_test_exit=$?
      if [[ -n "${GROUP_HOME:-}" && -x "$ep21_shared_astra" ]]; then
        timeout --signal=TERM --kill-after=5s 30s "$ep21_shared_astra" validate \
          outputs/astra/v0.0.14/astra.yaml --json \
          > "$ep21_shared_dir/astra_validation.json" 2>&1
        ep21_shared_astra_exit=$?
      else
        printf 'Optional ASTRA runtime unavailable\n' > "$ep21_shared_dir/astra_validation.json"
        ep21_shared_astra_exit=127
      fi
      printf 'job=%s\nchecks_exit=%s\nastra_exit=%s\nparent_charged_once_to=EP21\n' \
        "$SLURM_JOB_ID" "$ep21_shared_test_exit" "$ep21_shared_astra_exit" \
        > "$ep21_shared_dir/status.txt"
      printf 'Shared_component_episode=%s checks_exit=%s astra_exit=%s\n' \
        "$ep21_check_episode" "$ep21_shared_test_exit" "$ep21_shared_astra_exit"
    )
    printf 'Shared_component_wrapper_episode=%s exit=%s\n' "$ep21_check_episode" "$?"
  done
  set -e
fi
# WorkBudget clips to the verified allocation EndTime minus 300 seconds,
# including all startup/shared-check overhead. The outer timeout is a ceiling.
# Completed/partial work exits through the worker; this array never resubmits.
exec timeout --signal=TERM --kill-after=10s 28530s python3.12 -u -B \
  outputs/decoder_training/tools/train_decoder_work_item.py \
  --authorization outputs/decoder_training_full_authorization.json \
  --manifest outputs/decoder_training/full_continuation_001/work_manifest.json \
  --work-item "$ep21_work_item" --max-seconds 28500
