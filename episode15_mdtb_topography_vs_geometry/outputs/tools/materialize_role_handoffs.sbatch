#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=ep15-role-handoffs
#SBATCH --no-requeue
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode15_mdtb_topography_vs_geometry/outputs/runs/role_handoffs/full001-slurm-%j.out

set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* && "${SLURM_CPUS_PER_TASK:-}" == 4 ]]
cd "$OAK/users/zijiao/br_autoresearch/episode15_mdtb_topography_vs_geometry"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export PYTHONPATH="outputs/tools:outputs/tests${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
ep15_job_record="outputs/runs/role_handoffs/full001-job-${SLURM_JOB_ID}"
mkdir "$ep15_job_record"
printf 'hostname=%s\njob_id=%s\ncpus=%s\n' "$(hostname)" "$SLURM_JOB_ID" "$SLURM_CPUS_PER_TASK" > "$ep15_job_record/execution.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$ep15_job_record/started_utc.txt"
set +e
timeout --signal=TERM --kill-after=20s 3360s python3.12 -B outputs/tools/materialize_role_handoffs.py \
  --output-dir outputs/runs/role_handoffs/full001 \
  --private-dir outputs/cache/private/role_handoffs/full001 \
  --max-seconds 3300 > "$ep15_job_record/materialization.txt" 2>&1
ep15_result=$?
set -e
printf 'materialization=%s\n' "$ep15_result" > "$ep15_job_record/stage_exit_status.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$ep15_job_record/finished_utc.txt"
exit "$ep15_result"
