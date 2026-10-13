#!/bin/bash
# Fixed, reviewed column-copy attempt002; no requeue or automatic successor.
#SBATCH --job-name=ep17-role-full
#SBATCH --partition=russpold,normal
#SBATCH --constraint=CPU_GEN:SIE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/role_materialization/full_columns001-%j.out
#SBATCH --error=outputs/runs/role_materialization/full_columns001-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep17_column_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s\n' "$ep17_column_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}"
[[ "$ep17_column_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 1 && "$SLURM_MEM_PER_NODE" == 2048 ]]
cd "${OAK:?}/users/zijiao/br_autoresearch/episode17_cneuromod_model_ranking_stability"
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312
export LC_ALL=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export TMPDIR="${L_SCRATCH_JOB:?}"
[[ -d "$TMPDIR" ]]
umask 077
ep17_column_run="$PWD/outputs/runs/role_materialization/full_columns_job_${SLURM_JOB_ID}"
mkdir "$ep17_column_run"
printf 'job=%s\nparent_charged_once_to=EP17\nreader_handoffs_activated=false\n' \
  "$SLURM_JOB_ID" > "$ep17_column_run/status.txt"
ep17_column_capacity="$ep17_column_run/capacity_observation.json"
# Coordinator assessment at 07:51:25 UTC from fresh sh_quota and exact-parent df.
# Oak was 171.5/250 TB; the unused GROUP_SCRATCH access error does not supply
# capacity for this Oak-only operation. Preserve the 560 GB concurrent reserve.
(
  set -o noclobber
  cat > "$ep17_column_capacity" <<EOF
{
  "parent": "$PWD/outputs/cache",
  "slurm_job_id": "$SLURM_JOB_ID",
  "observed_at_utc": "2026-10-09T07:51:25Z",
  "quota_checked": true,
  "filesystem_available_bytes": 86276024791040,
  "quota_available_bytes": 77000000000000,
  "concurrent_reserved_bytes": 560000000000
}
EOF
)
set +e
(
  ulimit -v 2097152 || exit 2
  exec timeout --signal=TERM --kill-after=10s 9060s python3.12 -B \
    outputs/tools/materialize_neural_roles_columns.py --mode full \
    --capacity-observation "$ep17_column_capacity" --max-seconds 9000
) > "$ep17_column_run/full_copy.txt" 2>&1
ep17_column_exit=$?
set -e
printf 'full_copy_exit=%s\n' "$ep17_column_exit" >> "$ep17_column_run/status.txt"
exit "$ep17_column_exit"
