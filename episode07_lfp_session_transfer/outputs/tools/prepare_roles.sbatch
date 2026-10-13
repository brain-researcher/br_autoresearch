#!/bin/bash
# Adopted eleven-session preparation; no fitting, score or automatic retry.
# Aggregate new source-only EP21 component checks after the substantive export.
# Each episode owns its outputs; the parent allocation is charged once to EP07.
#SBATCH --job-name=ep07-11roles
#SBATCH --partition=russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/source_support/attempt002/slurm-%j.out
#SBATCH --error=outputs/runs/source_support/attempt002/slurm-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep07_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s\n' "$ep07_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}"
[[ "$ep07_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 1 \
   && "$SLURM_MEM_PER_NODE" == 7168 ]] || exit 2
ep07_project="${OAK:?}/users/zijiao/br_autoresearch"
cd "$ep07_project/episode07_lfp_session_transfer"
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312 py-pytorch/2.4.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
umask 077
export TMPDIR="${L_SCRATCH_JOB:?}/ep07_roles_attempt002"
mkdir "$TMPDIR"
ep07_run="outputs/runs/source_support/attempt002"
[[ -d "$ep07_run" ]] || exit 2
mkdir "$ep07_run/batch_claim" || exit 2
TZ=UTC scontrol show job -o "$SLURM_JOB_ID" > "$ep07_run/allocation.txt"
printf 'parent_episode=EP07\nparent_job=%s\nstatus=started\n' "$SLURM_JOB_ID" > "$ep07_run/status.txt"
ep07_check_rc=0
if (cd outputs/tools && timeout --signal=TERM --kill-after=5s 60s \
    python3.12 -B -m unittest -v test_eleven_session_roles) \
    > "$ep07_run/roster_checks.txt" 2>&1; then :; else ep07_check_rc=$?; fi
printf 'roster_checks_exit=%s\n' "$ep07_check_rc" >> "$ep07_run/status.txt"
ep07_prepare_rc=2
if [[ "$ep07_check_rc" -eq 0 ]]; then
  if timeout --signal=TERM --kill-after=10s 1800s python3.12 -u -B \
      outputs/tools/prepare_role_views.py \
      --source "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
      --views "$SCRATCH/br_autoresearch/episode07_lfp_session_transfer/source_support_attempt002" \
      --out "$ep07_run" > "$ep07_run/preparation.txt" 2>&1; then
    ep07_prepare_rc=0
  else
    ep07_prepare_rc=$?
  fi
fi
printf 'preparation_exit=%s\n' "$ep07_prepare_rc" >> "$ep07_run/status.txt"

# No empirical EP21 reader is invoked. These are previously prepared unit
# checks, separate from the unchanged live EP21 measurement and its tail.
ep07_child="$ep07_project/episode21_wang_neural_code_conversion_reproduction/outputs/runs/secondary_component_checks_job_${SLURM_JOB_ID}"
mkdir "$ep07_child"
printf 'parent_episode=EP07\nparent_job=%s\nparent_charged_once=true\n' \
  "$SLURM_JOB_ID" > "$ep07_child/status.txt"
ep07_companion_rc=0
for ep07_suite in brain prediction metrics inference secondary; do
  case "$ep07_suite" in
    brain) ep07_names=(test_final_brain_artifact test_empirical_brain_inputs test_brain_pair_driver test_brain_pair_admission); ep07_limit=120 ;;
    prediction) ep07_names=(test_finding_a_prediction); ep07_limit=120 ;;
    metrics) ep07_names=(test_finding_a_metrics); ep07_limit=120 ;;
    inference) ep07_names=(test_finding_a_inference); ep07_limit=90 ;;
    secondary) ep07_names=(test_finding_a_secondary); ep07_limit=90 ;;
  esac
  ep07_suite_rc=0
  if (cd "$ep07_project/episode21_wang_neural_code_conversion_reproduction/outputs/converter_training/tests" && \
      timeout --signal=TERM --kill-after=5s "$ep07_limit" python3.12 -B - "${ep07_names[@]}" <<'PY'
import sys
import torch
import unittest
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
suite = unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:])
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
PY
  ) > "$ep07_child/${ep07_suite}.txt" 2>&1; then :; else ep07_suite_rc=$?; fi
  printf '%s_exit=%s\n' "$ep07_suite" "$ep07_suite_rc" >> "$ep07_child/status.txt"
  if [[ "$ep07_suite_rc" -ne 0 ]]; then ep07_companion_rc=1; fi
done
printf 'status=closed\n' >> "$ep07_child/status.txt"

# The existing live EP21 parent validates its own program. Validate only the
# amended EP07 program here, without opening any data or treating it as a gate.
ep07_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep07_astra_deps="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
ep07_astra_rc=0
if PYTHONPATH="$ep07_astra/lib/python3.12/site-packages:$ep07_astra_deps" \
    timeout --signal=TERM --kill-after=5s 30s "$ep07_astra/bin/astra-tools" \
    validate outputs/astra/v0.0.14/astra.yaml --json \
    > "$ep07_run/astra_validation.json" 2>&1; then :; else ep07_astra_rc=$?; fi
printf 'companion_exit=%s\nastra_exit=%s\nstatus=closed\n' \
  "$ep07_companion_rc" "$ep07_astra_rc" >> "$ep07_run/status.txt"
exit "$ep07_prepare_rc"
