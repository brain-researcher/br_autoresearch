#!/bin/bash
# Original independent brain pair;3CPUs required for16GiB at8000MiB/core.
# One parent, shared absolute deadline, no retry or automatic successor.
#SBATCH --job-name=ep21-brain-p007
#SBATCH --partition=russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/brain_pair_training/sub03_to_sub07_001/slurm-%j.out
#SBATCH --error=outputs/runs/brain_pair_training/sub03_to_sub07_001/slurm-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep21_brain_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s\n' "$ep21_brain_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}"
[[ "$ep21_brain_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 3 \
   && "$SLURM_MEM_PER_NODE" == 16384 ]] || exit 2
cd "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1
python3.12 -B - <<'PY'
import json
from pathlib import Path
p = json.loads(Path("outputs/brain_pair_roster_007_authorization.json").read_text())
if p.get("authorized") is not True or not p.get("approval_source"):
    raise SystemExit("UNACTIVATED: separate next-brain-pair authority is required")
PY
module load math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312 py-pytorch/2.4.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
umask 077
[[ -d "${L_SCRATCH_JOB:?}" ]] || exit 2
ep21_brain_free_kib=$(df -Pk "$L_SCRATCH_JOB" | awk 'NR==2 {print $4}')
[[ "$ep21_brain_free_kib" =~ ^[0-9]+$ && "$ep21_brain_free_kib" -ge 262144 ]] || exit 2
export TMPDIR="$L_SCRATCH_JOB/ep21_brain_pair_sub03_to_sub07_001"
mkdir "$TMPDIR"
ep21_brain_run="outputs/runs/brain_pair_training/sub03_to_sub07_001"
[[ -d "$ep21_brain_run" ]] || exit 2
mkdir "$ep21_brain_run/batch_claim" || exit 2
TZ=UTC scontrol show job -o "$SLURM_JOB_ID" > "$ep21_brain_run/allocation.txt"
printf 'parent_episode=EP21\nparent_job=%s\nseparate_brain_pair_grant=true\nstatus=started\n' \
  "$SLURM_JOB_ID" > "$ep21_brain_run/status.txt"
ulimit -c 0
ulimit -S -f 2097152
ep21_brain_admission_rc=0
if python3.12 -B outputs/converter_training/tools/run_brain_pair_roster.py --stage roster-007 --check-admission \
    > "$ep21_brain_run/admission.txt" 2>&1; then :; else ep21_brain_admission_rc=$?; fi
printf 'admission_exit=%s\n' "$ep21_brain_admission_rc" >> "$ep21_brain_run/status.txt"
if [[ "$ep21_brain_admission_rc" -ne 0 ]]; then
  printf 'brain_pair_exit=2\nstatus=closed\n' >> "$ep21_brain_run/status.txt"
  exit "$ep21_brain_admission_rc"
fi
printf 'prerequisites=reused_13_passing_methods_job_47334643\n' >> "$ep21_brain_run/status.txt"

# Root submits this independent pair only after the common new checks pass.
ep21_brain_checks_rc=2
if grep -Fxq 'brain_admission_checks_exit=0' outputs/runs/brain_pair_training/sub03_to_sub06_001/status.txt; then
  ep21_brain_checks_rc=0
fi
printf 'brain_admission_checks_reused_from=sub03_to_sub06_001\nbrain_admission_checks_exit=%s\n' "$ep21_brain_checks_rc" >> "$ep21_brain_run/status.txt"

# Re-admission computes the same StartTime-based budget after the companions.
ep21_brain_rc=2
if [[ "$ep21_brain_checks_rc" -eq 0 ]]; then
  if ep21_brain_seconds=$(python3.12 -B outputs/converter_training/tools/run_brain_pair_roster.py --stage roster-007 --remaining-seconds); then
    printf 'brain_pair_timeout_seconds=%s\n' "$ep21_brain_seconds" >> "$ep21_brain_run/status.txt"
    if timeout --signal=TERM --kill-after=10s "${ep21_brain_seconds}s" python3.12 -u -B \
        outputs/converter_training/tools/run_brain_pair_roster.py --stage roster-007 \
        > "$ep21_brain_run/fit.txt" 2>&1; then
      ep21_brain_rc=0
    else ep21_brain_rc=$?; fi
  else ep21_brain_rc=$?; fi
fi
printf 'brain_pair_exit=%s\nstatus=closed\n' "$ep21_brain_rc" >> "$ep21_brain_run/status.txt"
exit "$ep21_brain_rc"
