#!/bin/bash
# Unsubmitted: exact deadline and inspected I/O complements must all close first.
# The accepted assembly intent must carry the same complete recovery review.
#SBATCH --job-name=ep21-full-assembly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/decoder_training/full_assembly_io_recovery001-%j.out
#SBATCH --error=outputs/decoder_training/full_assembly_io_recovery001-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep21_io_assembly_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s\n' "$ep21_io_assembly_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}"
[[ "$ep21_io_assembly_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 1 && "$SLURM_MEM_PER_NODE" == 7168 ]]
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
ep21_primary_rc=0
if timeout --signal=TERM --kill-after=10s 14130s python3.12 -u -B \
  outputs/decoder_training/tools/decoder_assembly.py \
  --authorization outputs/decoder_training_full_authorization.json --block-units 256 \
  --remainder-manifest outputs/decoder_training/full_continuation_001/deadline_remainders_002.json \
  --io-recovery-review outputs/decoder_training/full_continuation_001/reviewed_io_salvage_001.json; then
  ep21_primary_rc=0
else
  ep21_primary_rc=$?
fi

# Companion work is optional only AFTER the unchanged primary succeeds. Read
# its already-verified allocation metadata, not a new scheduler observation.
# Any missing/invalid metadata, check failure or tail timeout leaves primaryrc
# unchanged. The parent allocation is charged once to EP21, with no successor.
if [[ "$ep21_primary_rc" -eq 0 ]]; then
  ep21_companion_remaining=0
  if ep21_companion_remaining=$(timeout --signal=TERM --kill-after=5s 5s python3.12 -B - <<'PY'
import json
import math
import os
from pathlib import Path
import socket
import time

path = Path("outputs/decoder_training/full_continuation_001/assembled_decoder_manifest.json")
if path.is_symlink():
    raise ValueError("symlinked assembly metadata")
with path.open("rb") as stream:
    raw = stream.read(64 * 1024 * 1024 + 1)
if len(raw) > 64 * 1024 * 1024:
    raise ValueError("assembly metadata bound")
record = json.loads(raw)
allocation = record["assembly_allocation"]
end = allocation["job_end_epoch"]
if (record.get("status") != "complete_fixed_decoder_collection"
        or record.get("complete_decoder") is not True
        or allocation.get("job_id") != os.environ["SLURM_JOB_ID"]
        or allocation.get("hostname") != socket.gethostname().split(".")[0]
        or allocation.get("allocated_cpus") != 1 or allocation.get("memory_gib") != 7
        or allocation.get("gpus") != 0 or allocation.get("time_limit_seconds") != 14400
        or isinstance(end, bool) or not isinstance(end, (int, float)) or not math.isfinite(end)):
    raise ValueError("completed primary allocation metadata mismatch")
print(max(0, math.floor(end - time.time())))
PY
  ); then
    if [[ "$ep21_companion_remaining" =~ ^[0-9]+$ && "$ep21_companion_remaining" -ge 1250 ]]; then
      if timeout --signal=TERM --kill-after=10s 1150s bash \
          outputs/decoder_training/selected_companion_checks_001.sh; then
        printf 'COMPANION_TAIL_EXIT=0 PRIMARY_ASSEMBLY_EXIT=0\n' || :
      else
        ep21_companion_rc=$?
        printf 'COMPANION_TAIL_EXIT=%s PRIMARY_ASSEMBLY_EXIT=0\n' "$ep21_companion_rc" || :
      fi
    else
      printf 'COMPANION_TAIL=skipped_insufficient_time REMAINING_SECONDS=%s PRIMARY_ASSEMBLY_EXIT=0\n' \
        "$ep21_companion_remaining" || :
    fi
  else
    printf 'COMPANION_TAIL=skipped_unverified_remaining_time PRIMARY_ASSEMBLY_EXIT=0\n' || :
  fi
fi
exit "$ep21_primary_rc"
