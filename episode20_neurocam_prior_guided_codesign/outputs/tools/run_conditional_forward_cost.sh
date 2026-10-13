#!/bin/bash
# One substantive EP20 cost attempt; submit with explicit 1 CPU, 7 GiB, 2 h.
set -euo pipefail
host=$(hostname)
if [[ -z ${SLURM_JOB_ID:-} || $host == *-ln* || ${SLURM_CPUS_PER_TASK:-0} != 1 ]]; then
    printf '%s\n' 'Requires the selected one-CPU Slurm compute allocation.' >&2
    exit 2
fi
episode="$OAK/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign"
run_dir=${1:?Pass the selected existing attempt directory}
if [[ $run_dir != "$episode/outputs/runs/model_specification/conditional_forward_cost_attempt003" || ! -d $run_dir ]]; then
    printf '%s\n' 'The selected EP20 attempt directory is required.' >&2
    exit 2
fi
printf 'host=%s job=%s cpus=%s\n' "$host" "$SLURM_JOB_ID" "$SLURM_CPUS_PER_TASK"
date -u '+%Y-%m-%d %H:%M:%S UTC'
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export TMPDIR="${L_SCRATCH_JOB:?Slurm job-local temporary directory required}"
cd "$episode/outputs/tools"
python3.12 -B -c 'import sys, numpy, scipy, h5py; print(sys.version); print("numpy",numpy.__version__,"scipy",scipy.__version__,"h5py",h5py.__version__)'
set +e
/usr/bin/time -v -o "$run_dir/tests_resources.txt" timeout --signal=TERM --kill-after=5s 300s \
    python3.12 -B -m unittest -v \
    test_analog_conditioning test_conditioned_array_solver test_conditioned_cpe_array \
    test_spectral_noise test_physical_noise_ports test_cpe_noise \
    test_polynomial_noise test_cpe_polynomial_noise test_polynomial_acquisition test_conditional_noise_clock \
    test_external_daq_noise test_daq_external_inputs test_external_input_acquisition \
    test_switched_line test_coupled_rc.CoupledRCTests.test_scalar_limit_matches_existing_r1_and_input_work \
    test_circuit_noise_provider test_physical_noise_acquisition \
    test_cpe_array test_daq_mux_hold test_cpe_external_inputs \
    test_array_acquisition test_development_simulation \
    test_streaming_array_acquisition test_streaming_development_simulation \
    test_raw_input_hold test_streaming_midpoint_hold > "$run_dir/tests.txt" 2>&1
tests_exit=$?
printf 'affected_checks_exit=%s\n' "$tests_exit"
cd "$episode"
astra_env="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
astra_deps="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
PYTHONPATH="$astra_env/lib/python3.12/site-packages:$astra_deps" \
    timeout --signal=TERM --kill-after=5s 30s "$astra_env/bin/astra-tools" \
    validate outputs/astra/v0.0.14/astra.yaml --json > "$run_dir/astra_validation.json" 2>&1
astra_exit=$?
printf 'astra_validation_exit=%s\n' "$astra_exit"
set -e
if [[ $tests_exit != 0 ]]; then
    printf '%s\n' 'Affected checks failed; the full forward attempt did not start.'
    exit 20
fi
cd "$episode/outputs/tools"
set +e
/usr/bin/time -v -o "$run_dir/forward_resources.txt" timeout --signal=TERM --kill-after=5s 6100s \
    python3.12 -u -B run_conditional_forward_cost.py --run-dir "$run_dir" \
    --noise-representation cubic \
    --reuse-noise "$episode/outputs/runs/model_specification/conditional_forward_cost_attempt002/conditional_noise_draw.npz" \
    > "$run_dir/forward_stdout.txt" 2>&1
forward_exit=$?
set -e
printf 'forward_exit=%s\n' "$forward_exit"
date -u '+%Y-%m-%d %H:%M:%S UTC'
exit "$forward_exit"
