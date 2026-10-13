#!/bin/bash
# A separately selected EP17 child inside the substantive EP18 parent.
set -euo pipefail
umask 077
test -n "$SLURM_JOB_ID"
test -n "$SLURM_STEP_ID"
test "$SLURM_CPUS_PER_TASK" = 1
test "$SLURM_MEM_PER_NODE" = 4096
case "$(hostname)" in *-ln*) exit 2;; esac
ep17_episode="$OAK/users/zijiao/br_autoresearch/episode17_cneuromod_model_ranking_stability"
cd "$ep17_episode"
ep17_ops="$PWD/outputs/runs/stimulus_features/pilot_operations_20261012_001"
mkdir "$ep17_ops"
exec >"$ep17_ops/stdout.txt" 2>"$ep17_ops/stderr.txt"
trap 'ep17_exit=$?; printf "exit=%s\nfinished_utc=%s\n" "$ep17_exit" "$(date -u +%FT%TZ)" >>"$ep17_ops/status.txt"' EXIT
printf 'hostname=%s\njob_id=%s\nstep_id=%s\ncpus=%s\nmemory_mib=%s\nstarted_utc=%s\n' \
  "$(hostname)" "$SLURM_JOB_ID" "$SLURM_STEP_ID" "$SLURM_CPUS_PER_TASK" \
  "$SLURM_MEM_PER_NODE" "$(date -u +%FT%TZ)" >"$ep17_ops/allocation.txt"
# Conservative admission reference: Oct3 midnight UTC +240h.
# Refuse a queued launch that cannot finish the entire selected step by it.
test "$(( $(date -u +%s) + 900 ))" -le 1791849600
test "$(( $(date -u +%s) + 900 ))" -lt "$SLURM_JOB_END_TIME"
ep17_local="$L_SCRATCH_JOB/ep17_private_pilot_001"
mkdir "$ep17_local"
ep17_python="$GROUP_HOME/zijiao/miniforge3/envs/encoding_model/bin/python"
ep17_h5py="$GROUP_HOME/users/zijiao/ep17_h5py_3_10_0_cp311"
ep17_ld_path="$(printenv LD_LIBRARY_PATH || true)"
ep17_clean() {
  env -i "PATH=$PATH" "HOME=$HOME" "USER=$USER" \
    "GROUP_HOME=$GROUP_HOME" "OAK=$OAK" "LD_LIBRARY_PATH=$ep17_ld_path" \
    "SLURM_JOB_ID=$SLURM_JOB_ID" "SLURM_STEP_ID=$SLURM_STEP_ID" \
    SLURM_CPUS_PER_TASK=1 SLURM_MEM_PER_NODE=4096 "L_SCRATCH_JOB=$L_SCRATCH_JOB" \
    "TMPDIR=$ep17_local" "XDG_CACHE_HOME=$ep17_local/cache" "HF_HOME=$ep17_local/hf" \
    "TORCH_HOME=$ep17_local/torch" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= LANG=C LC_ALL=C TZ=UTC "$@"
}
ep17_clean "PYTHONPATH=$ep17_h5py" \
  timeout --signal=TERM --kill-after=5s 30s "$ep17_python" -B -m unittest discover \
  -s outputs/tests -p test_private_stimulus_pilot_launcher.py -v >"$ep17_ops/launcher_checks.txt" 2>&1
printf 'launcher_checks_exit=0\n' >>"$ep17_ops/status.txt"
ep17_astra="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep17_cpu312="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156"
ep17_clean "PYTHONPATH=$ep17_astra/lib/python3.12/site-packages:$ep17_cpu312/lib/python3.12/site-packages" \
  timeout --signal=TERM --kill-after=5s 60s "$ep17_astra/bin/astra-tools" \
  validate outputs/astra/v0.0.14/astra.yaml --json >"$ep17_ops/astra_validation.json" 2>"$ep17_ops/astra_stderr.txt"
printf 'astra_exit=0\n' >>"$ep17_ops/status.txt"
ep17_clean "PYTHONPATH=$ep17_h5py" \
  timeout --signal=TERM --kill-after=10s 660s "$ep17_python" -B \
  outputs/tools/run_private_stimulus_feature_pilot.py
printf 'real_image_pilot_exit=0\n' >>"$ep17_ops/status.txt"
