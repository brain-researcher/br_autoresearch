#!/bin/bash
# Explicit operator-selected fresh work only; never submits or retries jobs.
# The first fresh global item hosts changed-code checks in this allocation.
#SBATCH --job-name=ep21-full-workers-001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep21_refill_host=$(hostname)
printf 'REFILL_HOST=%s JOB=%s ARRAY=%s INDEX=%s CPUS=%s MEM_MIB=%s\n' \
  "$ep21_refill_host" "${SLURM_JOB_ID:?}" "${SLURM_ARRAY_JOB_ID:?}" \
  "${SLURM_ARRAY_TASK_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}"
[[ "$ep21_refill_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 1 \
   && "$SLURM_MEM_PER_NODE" == 7168 ]]
[[ "$SLURM_ARRAY_TASK_ID" =~ ^(0|[1-9][0-9]*)$ && "$SLURM_ARRAY_TASK_ID" -ge 8 ]]
cd "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
[[ -n "${L_SCRATCH_JOB:?}" && -d "$L_SCRATCH_JOB" ]]
export TMPDIR="$L_SCRATCH_JOB"

if [[ "$SLURM_ARRAY_TASK_ID" == 8 ]]; then
  ep21_refill_checks="${PWD}/outputs/decoder_training/refill_checks_job_${SLURM_JOB_ID}"
  mkdir "$ep21_refill_checks"
  (
    source /share/software/user/open/lmod/lmod/init/bash
    module purge
    module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-h5py/3.10.0_py312 py-scipy/1.12.0_py312
    cd outputs/decoder_training/tests
    timeout --signal=TERM --kill-after=5s 180s python3.12 -B -m unittest -v \
      test_decoder_array_allocation test_submit_full_decoder_jobs \
      test_reviewed_pre_admission_replacement
  ) > "$ep21_refill_checks/checks.txt" 2>&1
  printf 'Required_manual_refill_and_accounting_checks=passed\n'

  # Separate episode scopes: only named fixture checks and ASTRA documents.
  # No scientific sibling output is an EP21 input. Their failures are logged
  # independently, and the parent allocation is charged once to EP21.
  # Child module/path changes cannot alter the empirical worker environment.
  set +e
  timeout --signal=TERM --kill-after=5s 1020s bash <<'EP21_SHARED_CHECKS'
set -uo pipefail
source /share/software/user/open/lmod/lmod/init/bash
module purge
module load devel python/3.12.1 || exit 2
unset PYTHONPATH PYTHONHOME
shared_python="${GROUP_HOME:?}/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/bin/python"
shared_dependencies="${GROUP_HOME}/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
shared_astra="${GROUP_HOME}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
[[ -x "$shared_python" ]] || exit 2
for shared_episode in \
  episode18_things_eeg_concept_stability \
  episode19_causal_motor_forecasting_benchmark \
  episode20_neurocam_prior_guided_codesign \
  episode21_wang_neural_code_conversion_reproduction
do
  (
    cd "${OAK:?}/users/zijiao/br_autoresearch/${shared_episode}" || exit 2
    shared_dir="outputs/runs/component_checks_job_${SLURM_JOB_ID}"
    mkdir -p outputs/runs || exit 2
    mkdir "$shared_dir" || exit 2
    printf 'job=%s\nparent_charged_once_to=EP21\n' "$SLURM_JOB_ID" > "$shared_dir/status.txt"
    case "$shared_episode" in
      episode18_things_eeg_concept_stability)
        shared_test_dir=outputs/code
        shared_tests=(test_training_transforms.py test_continuous_sequence_model.py test_slot_capacity.py test_label_rbf.py) ;;
      episode19_causal_motor_forecasting_benchmark)
        shared_test_dir=outputs/tests
        shared_tests=(test_development_acquisition.py) ;;
      episode20_neurocam_prior_guided_codesign)
        shared_test_dir=outputs/tools
        shared_tests=(test_sparse_array_rc.py test_causal_cpe.py test_cpe_array.py test_acquisition_readout.py test_daq_mux_hold.py test_array_acquisition.py) ;;
      episode21_wang_neural_code_conversion_reproduction)
        shared_test_dir=outputs/decoder_training/tests
        shared_tests=() ;;  # Required changed-code suite already ran above.
    esac
    # Older Bash treats an empty array expansion as unbound under nounset.
    # EP21 has only ASTRA here; its required tests already ran above.
    if [[ "$shared_episode" != episode21_wang_neural_code_conversion_reproduction ]]; then
    for shared_test in "${shared_tests[@]}"; do
      shared_seconds=60
      case "$shared_test" in test_cpe_array.py|test_daq_mux_hold.py|test_array_acquisition.py) shared_seconds=120 ;; esac
      [[ "$shared_test" != test_acquisition_readout.py ]] || shared_seconds=30
      # Keep nested checks in the outer timeout's process group so its
      # aggregate deadline cannot leave a check running beside the worker.
      timeout --foreground --signal=TERM --kill-after=5s "${shared_seconds}s" "$shared_python" -B -m unittest \
        discover -s "$shared_test_dir" -p "$shared_test" -v \
        > "$shared_dir/${shared_test%.py}.txt" 2>&1
      shared_test_exit=$?
      printf '%s_exit=%s\n' "${shared_test%.py}" "$shared_test_exit" >> "$shared_dir/status.txt"
      printf 'Shared_component_episode=%s test=%s exit=%s\n' "$shared_episode" "$shared_test" "$shared_test_exit"
    done
    fi
    PYTHONPATH="$shared_astra/lib/python3.12/site-packages:$shared_dependencies" \
      timeout --foreground --signal=TERM --kill-after=5s 30s "$shared_astra/bin/astra-tools" validate \
      outputs/astra/v0.0.14/astra.yaml --json > "$shared_dir/astra_validation.json" 2>&1
    printf 'astra_exit=%s\n' "$?" >> "$shared_dir/status.txt"
  )
  printf 'Shared_component_wrapper_episode=%s exit=%s\n' "$shared_episode" "$?"
done
EP21_SHARED_CHECKS
  ep21_refill_shared_exit=$?
  set -e
  printf 'shared_wrapper_exit=%s\nparent_charged_once_to=EP21\n' \
    "$ep21_refill_shared_exit" > "$ep21_refill_checks/shared_status.txt"
fi

if [[ "$SLURM_ARRAY_TASK_ID" == 9 ]]; then
  # One explicit repair of the skipped index-8 EP21 ASTRA check. No repeat of
  # passing dispatch/component tests, and no change to the worker environment.
  ep21_astra_repair_dir="${PWD}/outputs/runs/component_checks_job_${SLURM_JOB_ID}"
  mkdir -p outputs/runs
  mkdir "$ep21_astra_repair_dir"
  printf 'job=%s\nparent_charged_once_to=EP21\n' "$SLURM_JOB_ID" > "$ep21_astra_repair_dir/status.txt"
  set +e
  timeout --signal=TERM --kill-after=5s 60s bash <<'EP21_ASTRA_REPAIR' > "$ep21_astra_repair_dir/wrapper.txt" 2>&1
set -uo pipefail
source /share/software/user/open/lmod/lmod/init/bash
module purge
module load devel python/3.12.1 || exit 2
unset PYTHONPATH PYTHONHOME
repair_dependencies="${GROUP_HOME:?}/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
repair_astra="${GROUP_HOME}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
repair_dir="outputs/runs/component_checks_job_${SLURM_JOB_ID}"
PYTHONPATH="$repair_astra/lib/python3.12/site-packages:$repair_dependencies" \
  timeout --foreground --signal=TERM --kill-after=5s 30s "$repair_astra/bin/astra-tools" validate \
  outputs/astra/v0.0.14/astra.yaml --json > "$repair_dir/astra_validation.json" 2>&1
repair_exit=$?
printf 'astra_exit=%s\n' "$repair_exit" >> "$repair_dir/status.txt"
exit "$repair_exit"
EP21_ASTRA_REPAIR
  ep21_astra_repair_exit=$?
  set -e
  printf 'astra_repair_wrapper_exit=%s\n' "$ep21_astra_repair_exit" >> "$ep21_astra_repair_dir/status.txt"
  printf 'EP21_ASTRA_repair_exit=%s\n' "$ep21_astra_repair_exit"
fi

if [[ "$SLURM_ARRAY_TASK_ID" == 10 ]]; then
  # Correct the diagnosed source-caller loss reduction for the later converter
  # stage. Public source/analytic fixture only; historical lock stays immutable.
  set +e
  # Setup and logging are optional too. The 235s deadline plus 5s kill grace
  # keep the whole child inside 240s, even if it cannot create its log folder.
  timeout --signal=TERM --kill-after=5s 235s bash <<'EP21_LOSS_CORRECTION'
set -uo pipefail
loss_dir="outputs/pipeline_conformance/converter_loss_checks_job_${SLURM_JOB_ID}"
mkdir "$loss_dir" || exit 2
printf 'job=%s\nparent_charged_once_to=EP21\n' "$SLURM_JOB_ID" > "$loss_dir/status.txt" || exit 2
exec > "$loss_dir/wrapper.txt" 2>&1 || exit 2
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-pytorch/2.4.1_py312 || exit 2
(
  cd outputs/pipeline_conformance/tools || exit 2
  timeout --foreground --signal=TERM --kill-after=5s 60s python3.12 -B -m unittest \
    -v test_feature_protocol.TaggedContentLossCallerTests
) > "$loss_dir/tests.txt" 2>&1
loss_tests_exit=$?
printf 'tests_exit=%s\n' "$loss_tests_exit" >> "$loss_dir/status.txt" || exit 2
loss_fixture_exit=125
if [[ "$loss_tests_exit" == 0 ]]; then
  timeout --foreground --signal=TERM --kill-after=5s 120s python3.12 -B \
    outputs/pipeline_conformance/tools/run_converter_loss_correction.py \
    > "$loss_dir/fixture.txt" 2>&1
  loss_fixture_exit=$?
fi
printf 'fixture_exit=%s\n' "$loss_fixture_exit" >> "$loss_dir/status.txt" || exit 2
loss_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
loss_dependencies="${GROUP_HOME}/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
PYTHONPATH="$loss_astra/lib/python3.12/site-packages:$loss_dependencies" \
  timeout --foreground --signal=TERM --kill-after=5s 30s "$loss_astra/bin/astra-tools" validate \
  outputs/astra/v0.0.14/astra.yaml --json > "$loss_dir/astra_validation.json" 2>&1
loss_astra_exit=$?
printf 'astra_exit=%s\n' "$loss_astra_exit" >> "$loss_dir/status.txt" || exit 2
[[ "$loss_tests_exit" == 0 && "$loss_fixture_exit" == 0 && "$loss_astra_exit" == 0 ]]
loss_child_exit=$?
printf 'loss_correction_child_exit=%s\n' "$loss_child_exit" >> "$loss_dir/status.txt" || exit 2
exit "$loss_child_exit"
EP21_LOSS_CORRECTION
  ep21_loss_check_exit=$?
  printf 'EP21_loss_correction_exit=%s\n' "$ep21_loss_check_exit"
  set -e

  # Separately scoped EP19 acquisition-header qualification. Reuse its four
  # passing reader tests; no EEG/marker/held-out payload is opened. Keep all
  # setup and logging optional, within 70s TERM + 5s kill grace and 1GiB VM.
  set +e
  timeout --signal=TERM --kill-after=5s 70s bash <<'EP19_HEADER_METADATA'
set -uo pipefail
ulimit -v 1048576 || exit 2
cd "${OAK:?}/users/zijiao/br_autoresearch/episode19_causal_motor_forecasting_benchmark" || exit 2
header_log="outputs/runs/source_views_and_timing/job_${SLURM_JOB_ID}_acquisition_metadata.txt"
set -C
exec > "$header_log" 2>&1 || exit 2
printf 'job=%s\nparent_charged_once_to=EP21\n' "$SLURM_JOB_ID" || exit 2
source /share/software/user/open/lmod/lmod/init/bash
module purge
module load devel python/3.12.1 || exit 2
unset PYTHONPATH PYTHONHOME
python3.12 -u -B outputs/code/inspect_development_acquisition.py \
  --output-dir "outputs/runs/source_views_and_timing/job_${SLURM_JOB_ID}_acquisition_metadata" \
  --max-seconds 60
header_exit=$?
printf 'acquisition_metadata_exit=%s\n' "$header_exit"
exit "$header_exit"
EP19_HEADER_METADATA
  ep19_header_exit=$?
  printf 'EP19_acquisition_metadata_child_exit=%s\n' "$ep19_header_exit"
  set -e
fi

if [[ "$SLURM_ARRAY_TASK_ID" == 11 ]]; then
  # Focused checks of the new reader/channel filter and array-only converter
  # implementation. No empirical converter invocation or real payload input.
  # This optional child is bounded including setup; the parent remains charged
  # once to the existing decoder grant and always proceeds to its fit worker.
  set +e
  timeout --signal=TERM --kill-after=5s 175s bash <<'EP21_CONVERTER_COMPONENTS'
set -uo pipefail
component_dir="outputs/converter_training/component_checks_job_${SLURM_JOB_ID}"
mkdir "$component_dir" || exit 2
printf 'job=%s\nparent_charged_once_to=EP21\nempirical_converter_execution=false\n' \
  "$SLURM_JOB_ID" > "$component_dir/status.txt" || exit 2
exec > "$component_dir/wrapper.txt" 2>&1 || exit 2
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-h5py/3.10.0_py312 py-pytorch/2.4.1_py312 || exit 2
timeout --foreground --signal=TERM --kill-after=5s 120s python3.12 -B - \
  > "$component_dir/converter_components.txt" 2>&1 <<'EP21_CONVERTER_TESTS_PY'
import sys
import torch
import unittest
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
print("torch=" + str(torch.__version__), flush=True)
suite = unittest.defaultTestLoader.discover("outputs/converter_training/tests", pattern="test_*.py")
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
EP21_CONVERTER_TESTS_PY
component_tests_exit=$?
printf 'converter_components_exit=%s\n' "$component_tests_exit" >> "$component_dir/status.txt" || exit 2
timeout --foreground --signal=TERM --kill-after=5s 30s python3.12 -B -m unittest \
  discover -s outputs/decoder_training/tests -p test_decoder_assembly.py -v \
  > "$component_dir/decoder_assembly.txt" 2>&1
assembly_tests_exit=$?
printf 'decoder_assembly_exit=%s\n' "$assembly_tests_exit" >> "$component_dir/status.txt" || exit 2
[[ "$component_tests_exit" == 0 && "$assembly_tests_exit" == 0 ]]
EP21_CONVERTER_COMPONENTS
  ep21_converter_component_exit=$?
  printf 'EP21_converter_components_child_exit=%s\n' "$ep21_converter_component_exit"
  set -e
fi

if [[ "$SLURM_ARRAY_TASK_ID" == 12 ]]; then
  # Required checks for the observed deadline-partial dispatch repair run
  # before this fresh assignment or any optional sibling task. This allocation
  # remains substantive decoder work within the original eight-hour ceiling.
  ep21_deadline_checks="${PWD}/outputs/decoder_training/deadline_partial_checks_job_${SLURM_JOB_ID}"
  mkdir "$ep21_deadline_checks"
  (
    source /share/software/user/open/lmod/lmod/init/bash
    module purge
    unset PYTHONPATH PYTHONHOME
    module load devel python/3.12.1
    cd outputs/decoder_training/tests
    timeout --signal=TERM --kill-after=5s 180s python3.12 -B -m unittest -v \
      test_decoder_array_allocation test_submit_full_decoder_jobs \
      test_reviewed_pre_admission_replacement test_reviewed_deadline_partials \
      test_reviewed_startup_replacement
  ) > "$ep21_deadline_checks/checks.txt" 2>&1
  printf 'Required_reviewed_deadline_partial_checks=passed\n'

  # Explicit repair of the diagnosed module-path removal in indices10/11.
  # Reuse their passing AST/caller and ASTRA checks. Only the missed fixture,
  # converter component tests and assembly tests run here, before EP18 work.
  set +e
  timeout --signal=TERM --kill-after=5s 295s bash <<'EP21_MODULE_PATH_REPAIR'
set -uo pipefail
repair_dir="outputs/converter_training/module_path_repair_job_${SLURM_JOB_ID}"
mkdir "$repair_dir" || exit 2
printf 'job=%s\nparent_charged_once_to=EP21\nempirical_converter_execution=false\n' \
  "$SLURM_JOB_ID" > "$repair_dir/status.txt" || exit 2
exec > "$repair_dir/wrapper.txt" 2>&1 || exit 2
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-h5py/3.10.0_py312 py-pytorch/2.4.1_py312 || exit 2
timeout --foreground --signal=TERM --kill-after=5s 120s python3.12 -B \
  outputs/pipeline_conformance/tools/run_converter_loss_correction.py \
  > "$repair_dir/loss_fixture.txt" 2>&1
repair_loss_exit=$?
printf 'loss_fixture_exit=%s\n' "$repair_loss_exit" >> "$repair_dir/status.txt" || exit 2
timeout --foreground --signal=TERM --kill-after=5s 120s python3.12 -B - \
  > "$repair_dir/converter_components.txt" 2>&1 <<'EP21_REPAIRED_COMPONENTS_PY'
import sys
import torch
import unittest
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
print("torch=" + str(torch.__version__), flush=True)
suite = unittest.defaultTestLoader.discover("outputs/converter_training/tests", pattern="test_*.py")
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
EP21_REPAIRED_COMPONENTS_PY
repair_components_exit=$?
printf 'converter_components_exit=%s\n' "$repair_components_exit" >> "$repair_dir/status.txt" || exit 2
timeout --foreground --signal=TERM --kill-after=5s 30s python3.12 -B -m unittest \
  discover -s outputs/decoder_training/tests -p test_decoder_assembly.py -v \
  > "$repair_dir/decoder_assembly.txt" 2>&1
repair_assembly_exit=$?
printf 'decoder_assembly_exit=%s\n' "$repair_assembly_exit" >> "$repair_dir/status.txt" || exit 2
[[ "$repair_loss_exit" == 0 && "$repair_components_exit" == 0 && "$repair_assembly_exit" == 0 ]]
EP21_MODULE_PATH_REPAIR
  ep21_module_repair_exit=$?
  printf 'EP21_module_path_repair_child_exit=%s\n' "$ep21_module_repair_exit"
  set -e

  # Separate EP18 scope under its standing public-label preparation authority.
  # Aggregate this potentially short task with substantive decoder work; never
  # pad runtime. Full parent cost stays charged once to EP21. No EP18 result
  # becomes an input to the decoder, and no image/EEG reader is released.
  set +e
  timeout --signal=TERM --kill-after=5s 1795s bash <<'EP18_LABEL_GEOMETRY'
set -uo pipefail
cd "${OAK:?}/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability" || exit 2
label_log="outputs/runs/feature_preparation/label_geometry_${SLURM_JOB_ID}.txt"
set -C
exec > "$label_log" 2>&1 || exit 2
printf 'job=%s\nparent_charged_once_to=EP21\nnested_EP18_CPU_seconds_limit=1800\n' \
  "$SLURM_JOB_ID" || exit 2
source /share/software/user/open/lmod/lmod/init/bash
module purge
module load devel python/3.12.1 || exit 2
unset PYTHONPATH PYTHONHOME
label_python="${GROUP_HOME:?}/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/bin/python"
[[ -x "$label_python" ]] || exit 2
"$label_python" -u -B outputs/code/prepare_label_geometry.py \
  --output-dir "outputs/runs/feature_preparation/label_geometry_${SLURM_JOB_ID}" \
  --max-seconds 1750
label_exit=$?
printf 'label_geometry_exit=%s\n' "$label_exit"
exit "$label_exit"
EP18_LABEL_GEOMETRY
  ep18_label_geometry_exit=$?
  printf 'EP18_label_geometry_child_exit=%s\n' "$ep18_label_geometry_exit"
  set -e

  # Separate EP15 implementation scope: test the new complete LOPO dependency
  # orchestration; no role containers, empirical fitting or reader release.
  # All setup/logging/checks fit within 115s TERM plus 5s kill grace.
  set +e
  timeout --signal=TERM --kill-after=5s 115s bash <<'EP15_LOPO_CHECKS'
set -uo pipefail
cd "${OAK:?}/users/zijiao/br_autoresearch/episode15_mdtb_topography_vs_geometry" || exit 2
lopo_dir="outputs/runs/lopo_checks_job_${SLURM_JOB_ID}"
mkdir "$lopo_dir" || exit 2
printf 'job=%s\nparent_charged_once_to=EP21\nempirical_execution=false\n' \
  "$SLURM_JOB_ID" > "$lopo_dir/status.txt" || exit 2
exec > "$lopo_dir/wrapper.txt" 2>&1 || exit 2
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 || exit 2
(
  cd outputs/tests || exit 2
  timeout --foreground --signal=TERM --kill-after=5s 60s python3.12 -B -m unittest -v \
    test_development_lopo test_local_anatomical_basis test_local_functional_source
) > "$lopo_dir/tests.txt" 2>&1
lopo_tests_exit=$?
printf 'tests_exit=%s\n' "$lopo_tests_exit" >> "$lopo_dir/status.txt" || exit 2
lopo_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
lopo_dependencies="${GROUP_HOME}/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
PYTHONPATH="$lopo_astra/lib/python3.12/site-packages:$lopo_dependencies" \
  timeout --foreground --signal=TERM --kill-after=5s 30s "$lopo_astra/bin/astra-tools" validate \
  outputs/astra/v0.0.14/astra.yaml --json > "$lopo_dir/astra_validation.json" 2>&1
lopo_astra_exit=$?
printf 'astra_exit=%s\n' "$lopo_astra_exit" >> "$lopo_dir/status.txt" || exit 2
[[ "$lopo_tests_exit" == 0 && "$lopo_astra_exit" == 0 ]]
EP15_LOPO_CHECKS
  ep15_lopo_child_exit=$?
  printf 'EP15_lopo_checks_child_exit=%s\n' "$ep15_lopo_child_exit"
  set -e
fi

if [[ "$SLURM_ARRAY_TASK_ID" == 15 ]]; then
  # Qualify only the newly prepared exact-remainder metadata helper. The
  # remainder execution path stays unactivated. These optional checks share
  # this substantive fresh participant fit's fixed allocation and budget.
  set +e
  timeout --signal=TERM --kill-after=5s 85s bash <<'EP21_REMAINDER_METADATA_CHECKS'
set -uo pipefail
remainder_check_dir="outputs/decoder_training/remainder_checks_job_${SLURM_JOB_ID}"
mkdir "$remainder_check_dir" || exit 2
printf 'job=%s\nremainder_execution=false\nparent_charged_once_to=EP21\n' \
  "$SLURM_JOB_ID" > "$remainder_check_dir/status.txt" || exit 2
exec > "$remainder_check_dir/wrapper.txt" 2>&1 || exit 2
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 || exit 2
(
  cd outputs/decoder_training/tests || exit 2
  timeout --foreground --signal=TERM --kill-after=5s 60s python3.12 -B -m unittest \
    -v test_decoder_deadline_remainders
) > "$remainder_check_dir/checks.txt" 2>&1
remainder_checks_exit=$?
printf 'checks_exit=%s\n' "$remainder_checks_exit" >> "$remainder_check_dir/status.txt" || exit 2
exit "$remainder_checks_exit"
EP21_REMAINDER_METADATA_CHECKS
  ep21_remainder_checks_exit=$?
  printf 'EP21_remainder_metadata_checks_child_exit=%s\n' "$ep21_remainder_checks_exit"
  set -e
fi

if [[ "$SLURM_ARRAY_TASK_ID" =~ ^(13|16|17|31|32|46|60|61)$ ]]; then
  # This explicit fresh batch uses the publication adapter qualified in the
  # closed six-check prelude of physical job 46891123. Numerical blocks stay
  # 64 units; the worker retains its normal admission and memory checks.
  ep21_publication_evidence=outputs/decoder_training/remainder_admission_checks_job_46891123
  [[ -f "$ep21_publication_evidence/status.txt" \
     && ! -L "$ep21_publication_evidence/status.txt" \
     && -f "$ep21_publication_evidence/buffered_publication_checks.txt" \
     && ! -L "$ep21_publication_evidence/buffered_publication_checks.txt" ]]
  [[ "$(grep -Fxc 'job=46891123' "$ep21_publication_evidence/status.txt")" == 1 \
     && "$(grep -Fxc 'required_checks_exit=0' "$ep21_publication_evidence/status.txt")" == 1 \
     && "$(grep -Fxc 'optional_buffered_publication_checks_exit=0' "$ep21_publication_evidence/status.txt")" == 1 ]]
  awk '
    /^test_.* \(test_buffered_publication\.BufferedPublicationTests\..*\) \.\.\. ok$/ { passed++ }
    /^Ran 6 tests in [0-9.]+s$/ { summaries++ }
    NF { last = $0 }
    END { exit !(passed == 6 && summaries == 1 && last == "OK") }
  ' "$ep21_publication_evidence/buffered_publication_checks.txt"
  [[ -f outputs/decoder_training_full_authorization.json \
     && -f outputs/decoder_training/full_continuation_001/work_manifest.json ]]
  source /share/software/user/open/lmod/lmod/init/bash
  module purge
  unset PYTHONPATH PYTHONHOME
  module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-h5py/3.10.0_py312 py-scipy/1.12.0_py312
    # Each explicitly selected worker checks changed append-only review
    # accounting before empirical input, including when its siblings remain
    # pending. This short prelude shares its substantive allocation.
    ep21_supplemental_checks="${PWD}/outputs/decoder_training/refill_supplemental_checks_job_${SLURM_JOB_ID}"
    mkdir "$ep21_supplemental_checks"
    printf 'job=%s\nparent_charged_once_to=EP21\n' "$SLURM_JOB_ID" \
      > "$ep21_supplemental_checks/status.txt"
    set +e
    (
      cd outputs/decoder_training/tests || exit 2
      timeout --signal=TERM --kill-after=5s 120s python3.12 -B -m unittest -v \
        test_decoder_array_allocation test_submit_full_decoder_jobs \
        test_reviewed_deadline_partials
    ) > "$ep21_supplemental_checks/checks.txt" 2>&1
    ep21_supplemental_checks_exit=$?
    set -e
    printf 'required_checks_exit=%s\n' "$ep21_supplemental_checks_exit" \
      >> "$ep21_supplemental_checks/status.txt"
    [[ "$ep21_supplemental_checks_exit" == 0 ]]
    printf 'Required_append_only_partial_review_checks=passed\n'
  printf -v ep21_buffered_work 'work_%04d' "$((SLURM_ARRAY_TASK_ID + 1))"
  printf 'WORK_ITEM=%s PUBLICATION_BLOCKS=4 NUMERICAL_BLOCK_UNITS=64 QUALIFIED_BY_JOB=46891123\n' "$ep21_buffered_work"
  # Same outer ceiling and actual-allocation EndTime-minus-300s clipping as
  # the default worker route; this branch neither schedules nor retries.
  exec timeout --signal=TERM --kill-after=10s 28530s python3.12 -u -B \
    outputs/decoder_training/tools/train_decoder_work_item.py \
    --authorization outputs/decoder_training_full_authorization.json \
    --manifest outputs/decoder_training/full_continuation_001/work_manifest.json \
    --work-item "$ep21_buffered_work" --max-seconds 28500 --publication-blocks 4
fi

# This resets modules/runtime and clips fitting to actual EndTime minus 300s,
# including all startup/check overhead. No historical index-0 checks rerun.
exec bash outputs/decoder_training/full_workers_001.sbatch
