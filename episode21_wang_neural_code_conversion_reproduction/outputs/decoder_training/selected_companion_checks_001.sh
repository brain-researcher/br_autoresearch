#!/bin/bash
# Unsubmitted, finite post-success companion work inside the EP21 assembly job.
# The outer wrapper owns a 1150s timeout + 10s kill grace after confirming at
# least 1250s remain. Child envelopes total1050s. No retries, scheduler calls,
# parallel children or empirical fitting/pixel access.
# EP17's one private operator derivation is the explicitly selected exception
# to the component-only checks; it consumes its previously admitted calibration.
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep21_companion_host=$(hostname)
[[ "$ep21_companion_host" != *-ln* && "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ \
   && "${SLURM_CPUS_PER_TASK:-}" == 1 && "${SLURM_MEM_PER_NODE:-}" == 7168 ]] || exit 2
[[ -n "${L_SCRATCH_JOB:-}" && -d "$L_SCRATCH_JOB" && "${TMPDIR:-}" == "$L_SCRATCH_JOB" ]] || exit 2
[[ "${OMP_NUM_THREADS:-}" == 1 && "${OPENBLAS_NUM_THREADS:-}" == 1 \
   && "${MKL_NUM_THREADS:-}" == 1 && "${NUMEXPR_NUM_THREADS:-}" == 1 \
   && "${VECLIB_MAXIMUM_THREADS:-}" == 1 && "${BLIS_NUM_THREADS:-}" == 1 \
   && -z "${CUDA_VISIBLE_DEVICES:-}" ]] || exit 2
ep21_companion_project="${OAK:?}/users/zijiao/br_autoresearch"
ep21_companion_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep21_companion_dependencies="${GROUP_HOME}/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
ep21_companion_rc=0

# Each subshell keeps cwd/environment/failure separate. Foreground inner
# timeouts remain in the outer timeout's process group, so its kill bounds
# the entire finite tail rather than leaving a nested child behind.
if (
  set -euo pipefail
  cd "$ep21_companion_project/episode20_neurocam_prior_guided_codesign" || exit 2
  ep20_companion_run="outputs/runs/streaming_nuisance_checks_job_${SLURM_JOB_ID}"
  mkdir -p outputs/runs || exit 2
  mkdir "$ep20_companion_run" || exit 2
  printf 'parent_episode=EP21\nparent_job=%s\nparent_charged_once=true\nstatus=started\n' \
    "$SLURM_JOB_ID" > "$ep20_companion_run/status.txt" || exit 2
  ep20_companion_tests=0
  if ( cd outputs/tools && timeout --foreground --signal=TERM --kill-after=5s 300s \
      python3.12 -B -m unittest -v test_development_simulation test_acquisition_readout \
      test_array_acquisition test_raw_world_time test_observation_nuisance \
      test_streaming_acquisition_readout test_streaming_array_acquisition \
      test_streaming_input_plan test_streaming_simulation_output test_streaming_development_simulation \
      test_cpe_array test_daq_mux_hold test_raw_input_hold test_daq_external_inputs \
      test_cpe_external_inputs test_external_input_acquisition \
      test_streaming_midpoint_hold test_repeating_acquisition_schedule \
      ) > "$ep20_companion_run/tests.txt" 2>&1; then
    ep20_companion_tests=0
  else
    ep20_companion_tests=$?
  fi
  printf 'tests_exit=%s\n' "$ep20_companion_tests" >> "$ep20_companion_run/status.txt" || exit 2
  ep20_companion_astra=0
  if PYTHONPATH="$ep21_companion_astra/lib/python3.12/site-packages:$ep21_companion_dependencies" \
      timeout --foreground --signal=TERM --kill-after=5s 30s "$ep21_companion_astra/bin/astra-tools" \
      validate outputs/astra/v0.0.14/astra.yaml --json \
      > "$ep20_companion_run/astra_validation.json" 2>&1; then
    ep20_companion_astra=0
  else
    ep20_companion_astra=$?
  fi
  printf 'astra_exit=%s\nstatus=closed\n' "$ep20_companion_astra" >> "$ep20_companion_run/status.txt" || exit 2
  [[ "$ep20_companion_tests" -eq 0 && "$ep20_companion_astra" -eq 0 ]]
); then
  :
else
  ep21_companion_rc=2
fi

if (
  set -euo pipefail
  cd "$ep21_companion_project/episode15_mdtb_topography_vs_geometry" || exit 2
  ep15_companion_run="outputs/runs/development_reader_checks_job_${SLURM_JOB_ID}"
  mkdir -p outputs/runs || exit 2
  mkdir "$ep15_companion_run" || exit 2
  printf 'parent_episode=EP21\nparent_job=%s\nparent_charged_once=true\nstatus=started\n' \
    "$SLURM_JOB_ID" > "$ep15_companion_run/status.txt" || exit 2
  ep15_companion_tests=0
  if ( cd outputs/tests && timeout --foreground --signal=TERM --kill-after=5s 120s \
      python3.12 -B -m unittest -v test_development_handoff_reader test_development_point_panel \
      ) > "$ep15_companion_run/tests.txt" 2>&1; then
    ep15_companion_tests=0
  else
    ep15_companion_tests=$?
  fi
  printf 'tests_exit=%s\n' "$ep15_companion_tests" >> "$ep15_companion_run/status.txt" || exit 2
  ep15_companion_astra=0
  if PYTHONPATH="$ep21_companion_astra/lib/python3.12/site-packages:$ep21_companion_dependencies" \
      timeout --foreground --signal=TERM --kill-after=5s 30s "$ep21_companion_astra/bin/astra-tools" \
      validate outputs/astra/v0.0.14/astra.yaml --json \
      > "$ep15_companion_run/astra_validation.json" 2>&1; then
    ep15_companion_astra=0
  else
    ep15_companion_astra=$?
  fi
  printf 'astra_exit=%s\nstatus=closed\n' "$ep15_companion_astra" >> "$ep15_companion_run/status.txt" || exit 2
  [[ "$ep15_companion_tests" -eq 0 && "$ep15_companion_astra" -eq 0 ]]
); then
  :
else
  ep21_companion_rc=2
fi

if (
  set -euo pipefail
  cd "$ep21_companion_project/episode17_cneuromod_model_ranking_stability" || exit 2
  ep17_companion_run="outputs/runs/private_operator_checks_job_${SLURM_JOB_ID}"
  mkdir -p outputs/runs || exit 2
  mkdir "$ep17_companion_run" || exit 2
  printf 'parent_episode=EP21\nparent_job=%s\nparent_charged_once=true\nstatus=started\n' \
    "$SLURM_JOB_ID" > "$ep17_companion_run/status.txt" || exit 2
  ep17_companion_tests=0
  if timeout --foreground --signal=TERM --kill-after=10s 90s python3.12 -B -m unittest discover \
      -s outputs/tests -p test_derive_private_operators.py -v > "$ep17_companion_run/tests.txt" 2>&1; then
    ep17_companion_tests=0
  else
    ep17_companion_tests=$?
  fi
  printf 'tests_exit=%s\n' "$ep17_companion_tests" >> "$ep17_companion_run/status.txt" || exit 2
  ep17_companion_operator=skipped_test_failure
  if [[ "$ep17_companion_tests" -eq 0 ]]; then
    # The producer writes operator values only to its protected private path;
    # its stdout contains status/timing/RSS only. Do not echo private artifacts.
    if timeout --foreground --signal=TERM --kill-after=10s 180s python3.12 -B \
        outputs/tools/derive_private_operators.py > "$ep17_companion_run/operator.txt" 2>&1; then
      ep17_companion_operator=0
    else
      ep17_companion_operator=$?
    fi
  fi
  printf 'operator_exit=%s\n' "$ep17_companion_operator" >> "$ep17_companion_run/status.txt" || exit 2
  # Source-free adapter fixtures only: no password descriptor, archive access
  # or empirical feature-pilot dispatch. This exact runtime is already qualified.
  ep17_companion_stimulus=0
  if env -u PYTHONPATH PYTHONNOUSERSITE=1 \
      timeout --foreground --signal=TERM --kill-after=5s 120s \
      "${GROUP_HOME}/zijiao/miniforge3/envs/encoding_model/bin/python" -B \
      -m unittest discover -s outputs/tests -p test_private_stimulus_feature_pilot.py -v \
      > "$ep17_companion_run/stimulus_adapter_tests.txt" 2>&1; then
    ep17_companion_stimulus=0
  else
    ep17_companion_stimulus=$?
  fi
  printf 'stimulus_adapter_tests_exit=%s\n' "$ep17_companion_stimulus" >> "$ep17_companion_run/status.txt" || exit 2
  ep17_companion_astra=0
  if PYTHONPATH="$ep21_companion_astra/lib/python3.12/site-packages:$ep21_companion_dependencies" \
      timeout --foreground --signal=TERM --kill-after=5s 30s "$ep21_companion_astra/bin/astra-tools" \
      validate outputs/astra/v0.0.14/astra.yaml --json \
      > "$ep17_companion_run/astra_validation.json" 2>&1; then
    ep17_companion_astra=0
  else
    ep17_companion_astra=$?
  fi
  printf 'astra_exit=%s\nstatus=closed\n' "$ep17_companion_astra" >> "$ep17_companion_run/status.txt" || exit 2
  [[ "$ep17_companion_tests" -eq 0 && "$ep17_companion_operator" == 0 \
     && "$ep17_companion_stimulus" -eq 0 && "$ep17_companion_astra" -eq 0 ]]
); then
  :
else
  ep21_companion_rc=2
fi

if (
  set -euo pipefail
  cd "$ep21_companion_project/episode21_wang_neural_code_conversion_reproduction" || exit 2
  ep21_companion_run="outputs/runs/assembly_companion_checks_job_${SLURM_JOB_ID}"
  mkdir -p outputs/runs || exit 2
  mkdir "$ep21_companion_run" || exit 2
  printf 'parent_episode=EP21\nparent_job=%s\nparent_charged_once=true\nstatus=started\n' \
    "$SLURM_JOB_ID" > "$ep21_companion_run/status.txt" || exit 2
  ep21_companion_converter=0
  # The entire separate module setup and thirteen source-free reader/driver
  # checks share this 60s bound. This is the previously qualified CPU Torch
  # module binding; it cannot alter the primary assembly environment.
  if timeout --foreground --signal=TERM --kill-after=5s 60s bash \
      > "$ep21_companion_run/converter_checks.txt" 2>&1 <<'EP21_CONVERTER_CHECKS'
set -euo pipefail
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312 py-pytorch/2.4.1_py312
cd outputs/converter_training/tests
exec python3.12 -B - <<'EP21_CONVERTER_CHECKS_PY'
import sys
import torch
import unittest

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
print("torch=" + str(torch.__version__), flush=True)
suite = unittest.defaultTestLoader.loadTestsFromNames([
    "test_decoder_interface", "test_content_pair_driver"])
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
EP21_CONVERTER_CHECKS_PY
EP21_CONVERTER_CHECKS
  then
    ep21_companion_converter=0
  else
    ep21_companion_converter=$?
  fi
  printf 'converter_checks_exit=%s\n' "$ep21_companion_converter" >> "$ep21_companion_run/status.txt" || exit 2
  ep21_companion_astra_rc=0
  if PYTHONPATH="$ep21_companion_astra/lib/python3.12/site-packages:$ep21_companion_dependencies" \
      timeout --foreground --signal=TERM --kill-after=5s 30s "$ep21_companion_astra/bin/astra-tools" \
      validate outputs/astra/v0.0.14/astra.yaml --json \
      > "$ep21_companion_run/astra_validation.json" 2>&1; then
    ep21_companion_astra_rc=0
  else
    ep21_companion_astra_rc=$?
  fi
  printf 'astra_exit=%s\nstatus=closed\n' "$ep21_companion_astra_rc" >> "$ep21_companion_run/status.txt" || exit 2
  [[ "$ep21_companion_converter" -eq 0 && "$ep21_companion_astra_rc" -eq 0 ]]
); then
  :
else
  ep21_companion_rc=2
fi
exit "$ep21_companion_rc"
