#!/bin/bash
# First full brain-loss pair selected October11; see the separate grant/log.
# One exact tau-train pair, final-only model, no retry or automatic successor.
# Submit from EP21 only after adoption and creation of the fresh fixed log dir.
#SBATCH --job-name=ep21-brain-pair
#SBATCH --partition=russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/brain_pair_training/attempt001/slurm-%j.out
#SBATCH --error=outputs/runs/brain_pair_training/attempt001/slurm-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep21_brain_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s\n' "$ep21_brain_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}"
[[ "$ep21_brain_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 1 \
   && "$SLURM_MEM_PER_NODE" == 7168 ]] || exit 2
ep21_brain_root="${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
cd "$ep21_brain_root"
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1
# Refuse the inactive proposal before scientific imports, claims or payload I/O.
python3.12 -B - <<'PY'
import json
from pathlib import Path
p = json.loads(Path("outputs/brain_pair_proposal.json").read_text())
if p.get("authorized") is not True or not p.get("approval_source"):
    raise SystemExit("UNACTIVATED: separate brain-pair execution authority is required")
PY
module load math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312 py-pytorch/2.4.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
umask 077
[[ -d "${L_SCRATCH_JOB:?}" ]] || exit 2
export TMPDIR="$L_SCRATCH_JOB/ep21_brain_pair_attempt001"
mkdir "$TMPDIR"
ep21_brain_run="outputs/runs/brain_pair_training/attempt001"
[[ -d "$ep21_brain_run" ]] || exit 2
mkdir "$ep21_brain_run/batch_claim" || exit 2
TZ=UTC scontrol show job -o "$SLURM_JOB_ID" > "$ep21_brain_run/allocation.txt"
python3.12 -B outputs/converter_training/tools/run_brain_pair.py --check-admission
printf 'parent_episode=EP21\nparent_job=%s\nseparate_brain_pair_grant=true\nstatus=started\n' \
  "$SLURM_JOB_ID" > "$ep21_brain_run/status.txt"
# This cap permits the768MiB model; the writer/profile enforce tighter bounds.
ulimit -S -f 2097152
printf 'prerequisites=reused_13_passing_methods_job_47334643\n' >> "$ep21_brain_run/status.txt"
ep21_brain_rc=2
if timeout --signal=TERM --kill-after=10s 84620s python3.12 -u -B \
    outputs/converter_training/tools/run_brain_pair.py \
    > "$ep21_brain_run/fit.txt" 2>&1; then
  ep21_brain_rc=0
else
  ep21_brain_rc=$?
fi
printf 'brain_pair_exit=%s\n' "$ep21_brain_rc" >> "$ep21_brain_run/status.txt"

# Validate only the changed EP21 program, using the qualified existing runtime.
ep21_brain_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep21_brain_astra_deps="${GROUP_HOME}/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
ep21_brain_end_epoch=$(python3.12 -B - <<'PYEND'
from datetime import datetime, timezone
from pathlib import Path
fields = dict(t.split('=', 1) for t in Path('outputs/runs/brain_pair_training/attempt001/allocation.txt').read_text().split() if '=' in t)
print(int(datetime.fromisoformat(fields['EndTime']).replace(tzinfo=timezone.utc).timestamp()))
PYEND
)
ep21_brain_astra_rc=0
ep21_brain_now_epoch=$(date -u +%s)
if (( ep21_brain_end_epoch - ep21_brain_now_epoch < 45 )); then
  ep21_brain_astra_rc=2
  printf 'astra_validation=deferred_insufficient_remaining_time\n' >> "$ep21_brain_run/status.txt"
elif PYTHONPATH="$ep21_brain_astra/lib/python3.12/site-packages:$ep21_brain_astra_deps" \
    timeout --signal=TERM --kill-after=5s 30s "$ep21_brain_astra/bin/astra-tools" \
    validate outputs/astra/v0.0.14/astra.yaml --json \
    > "$ep21_brain_run/astra_validation.json" 2>&1; then :; else ep21_brain_astra_rc=$?; fi
printf 'astra_exit=%s\nstatus=closed\n' "$ep21_brain_astra_rc" >> "$ep21_brain_run/status.txt"
exit "$ep21_brain_rc"
