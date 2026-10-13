#!/bin/bash
# One public-model qualification; independent serialized companions share cost.
#SBATCH --job-name=ep21-recon-qual
#SBATCH --partition=russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/reconstruction_qualification/attempt001/slurm-%j.out
#SBATCH --error=outputs/runs/reconstruction_qualification/attempt001/slurm-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep21_recon_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s\n' "$ep21_recon_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}"
[[ "$ep21_recon_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 1 && "$SLURM_MEM_PER_NODE" == 7168 ]] || exit 2
cd "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312 py-pytorch/2.4.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=''
umask 077
[[ -d "${L_SCRATCH_JOB:?}" ]] || exit 2
ep21_recon_free_kib=$(df -Pk "$L_SCRATCH_JOB" | awk 'NR==2 {print $4}')
[[ "$ep21_recon_free_kib" =~ ^[0-9]+$ && "$ep21_recon_free_kib" -ge 4194304 ]] || exit 2
ep21_recon_scratch="$L_SCRATCH_JOB/ep21_reconstruction_qualification001"
mkdir "$ep21_recon_scratch"
export TMPDIR="$ep21_recon_scratch/tmp"
mkdir "$TMPDIR"
export XDG_CACHE_HOME="$ep21_recon_scratch/cache" TORCH_HOME="$ep21_recon_scratch/torch"
ep21_recon_run="outputs/runs/reconstruction_qualification/attempt001"
ep21_recon_diag="outputs/runs/brain_pair_prefix_diagnostic/attempt002"
[[ -d "$ep21_recon_run" && -d "$ep21_recon_diag" ]] || exit 2
mkdir "$ep21_recon_run/batch_claim" || exit 2
TZ=UTC scontrol show job -o "$SLURM_JOB_ID" > "$ep21_recon_run/allocation.txt"
python3.12 -B outputs/reconstruction_preparation/run_reconstruction_qualification.py --check-admission
printf 'parent_episode=EP21\nparent_job=%s\nseparate_qualification_grant=true\nstatus=started\n' \
  "$SLURM_JOB_ID" > "$ep21_recon_run/status.txt"
ulimit -c 0
ulimit -S -f 2097152

ep21_recon_admission_rc=0
if (
  ulimit -S -f 32768
  cd outputs/reconstruction_preparation
  timeout --signal=TERM --kill-after=5s 45s python3.12 -B -m unittest -v test_reconstruction_admission
) > "$ep21_recon_run/parent_admission_checks.txt" 2>&1; then :; else ep21_recon_admission_rc=$?; fi
printf 'parent_admission_checks_exit=%s\n' "$ep21_recon_admission_rc" >> "$ep21_recon_run/status.txt"

ep21_recon_checks_rc=0
if (
  ulimit -S -f 32768
  cd outputs/reconstruction_preparation
  timeout --signal=TERM --kill-after=5s 150s python3.12 -B - <<'PY'
import sys
import torch
import unittest
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
suite = unittest.defaultTestLoader.loadTestsFromNames([
    'test_generator_assets', 'test_reconstruction_core'])
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
PY
) > "$ep21_recon_run/qualification_checks.txt" 2>&1; then :; else ep21_recon_checks_rc=$?; fi
printf 'qualification_checks_exit=%s\n' "$ep21_recon_checks_rc" >> "$ep21_recon_run/status.txt"

# Keep the diagnosed HDF5 cleanup regression in its own child. Its result
# cannot be mistaken for the independent public-runtime or brain checks.
ep21_collection_rc=0
if (
  ulimit -S -f 32768
  cd outputs/converter_training/tests
  timeout --signal=TERM --kill-after=5s 115s python3.12 -B -m unittest -v test_finding_a_collection
) > "$ep21_recon_run/collection_checks.txt" 2>&1; then :; else ep21_collection_rc=$?; fi
printf 'collection_checks_exit=%s\n' "$ep21_collection_rc" >> "$ep21_recon_run/status.txt"

ep21_recon_brain_checks_rc=0
if (
  ulimit -S -f 32768
  cd outputs/converter_training/tests
  timeout --signal=TERM --kill-after=5s 90s python3.12 -B -m unittest -v \
    test_brain_prefix_diagnostic_admission test_brain_pair_admission
) > "$ep21_recon_run/brain_admission_checks.txt" 2>&1; then :; else ep21_recon_brain_checks_rc=$?; fi
printf 'brain_admission_checks_exit=%s\n' "$ep21_recon_brain_checks_rc" >> "$ep21_recon_run/status.txt"

ep21_recon_content_checks_rc=0
if (
  ulimit -S -f 32768
  cd outputs/converter_training/tests
  timeout --signal=TERM --kill-after=5s 45s python3.12 -B -m unittest -v test_content_pair_admission
) > "$ep21_recon_run/content_admission_checks.txt" 2>&1; then :; else ep21_recon_content_checks_rc=$?; fi
printf 'content_admission_checks_exit=%s\n' "$ep21_recon_content_checks_rc" >> "$ep21_recon_run/status.txt"

ep21_recon_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep21_recon_astra_deps="${GROUP_HOME}/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
ep21_recon_astra_rc=0
if PYTHONPATH="$ep21_recon_astra/lib/python3.12/site-packages:$ep21_recon_astra_deps" \
    timeout --signal=TERM --kill-after=5s 30s "$ep21_recon_astra/bin/astra-tools" \
    validate outputs/astra/v0.0.14/astra.yaml --json \
    > "$ep21_recon_run/astra_validation.json" 2>&1; then :; else ep21_recon_astra_rc=$?; fi
printf 'astra_exit=%s\n' "$ep21_recon_astra_rc" >> "$ep21_recon_run/status.txt"

ep21_recon_diag_rc=2
if [[ "$ep21_recon_admission_rc" -eq 0 && "$ep21_recon_brain_checks_rc" -eq 0 ]]; then
  if timeout --signal=TERM --kill-after=10s 310s python3.12 -u -B \
      outputs/converter_training/tools/run_brain_prefix_diagnostic.py --stage reconstruction-002 \
      > "$ep21_recon_diag/diagnostic.txt" 2>&1; then
    ep21_recon_diag_rc=0
  else ep21_recon_diag_rc=$?; fi
fi
printf 'parent_job=%s\ndiagnostic_exit=%s\nstatus=closed\n' \
  "$SLURM_JOB_ID" "$ep21_recon_diag_rc" > "$ep21_recon_diag/status.txt"

ep21_recon_rc=2
if [[ "$ep21_recon_admission_rc" -eq 0 && "$ep21_recon_checks_rc" -eq 0 ]]; then
  if ep21_recon_seconds=$(python3.12 -B outputs/reconstruction_preparation/run_reconstruction_qualification.py --remaining-seconds); then
    if timeout --signal=TERM --kill-after=10s "${ep21_recon_seconds}s" python3.12 -u -B \
        outputs/reconstruction_preparation/run_reconstruction_qualification.py \
        > "$ep21_recon_run/qualification.txt" 2>&1; then
      ep21_recon_rc=0
    else ep21_recon_rc=$?; fi
  else ep21_recon_rc=$?; fi
fi
printf 'qualification_exit=%s\nstatus=closed\n' "$ep21_recon_rc" >> "$ep21_recon_run/status.txt"
exit "$ep21_recon_rc"
