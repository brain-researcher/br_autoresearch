#!/bin/bash
#SBATCH --job-name=ep17-roles
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:35:00
#SBATCH --no-requeue
# Root supplies fresh submission001/slurm_%j.{out,err} paths via $OAK.

set -eo pipefail
hostname
printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-}"
if [[ -z "${SLURM_JOB_ID:-}" || "$(hostname -s)" == *-ln* ]]; then
    printf 'Refusing execution outside the owned compute allocation.\n' >&2
    exit 2
fi
source /etc/profile >/dev/null
set -u
module purge
module load devel python/3.12.1
EP17_PYTHON=$(command -v python3.12)
printf 'selected_python=%s\n' "$EP17_PYTHON"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
readonly EP17_ROOT="${OAK:?OAK must name the authorized group filesystem}/users/zijiao/br_autoresearch/episode17_cneuromod_model_ranking_stability"
cd "$EP17_ROOT"
if [[ -e outputs/runs/concept_role_assignment/full001 ]]; then
    printf 'Selected attempt already exists; preserving it and stopping.\n' >&2
    exit 2
fi
timeout --signal=TERM --kill-after=10s 1860s \
    "$EP17_PYTHON" -B outputs/tools/run_concept_role_assignment.py --mode full \
    --out outputs/runs/concept_role_assignment/full001
