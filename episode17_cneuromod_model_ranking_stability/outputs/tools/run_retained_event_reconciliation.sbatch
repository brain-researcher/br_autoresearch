#!/bin/bash
#SBATCH --job-name=ep17-events
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:40:00
#SBATCH --no-requeue
# The coordinator supplies --output/--error at submission, expanded from $OAK:
# outputs/runs/released_event_reconciliation/submission001/slurm_%j.out
# outputs/runs/released_event_reconciliation/submission001/slurm_%j.err

set -eo pipefail
hostname
printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-}"
if [[ -z "${SLURM_JOB_ID:-}" || "$(hostname -s)" == *-ln* ]]; then
    printf 'Refusing execution outside the coordinator-owned compute allocation.\n' >&2
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
if [[ -e outputs/runs/released_event_manifest/attempt001 ||
      -e outputs/runs/released_event_reconciliation/retained001 ]]; then
    printf 'Selected attempt already exists; preserve it and stop.\n' >&2
    exit 2
fi

timeout --signal=TERM --kill-after=10s 1140s \
    "$EP17_PYTHON" -B outputs/tools/prepare_released_event_manifest.py \
    --out outputs/runs/released_event_manifest/attempt001

# This command validates complete exact-roster coverage and fixed aggregate/
# per-object byte limits before the first event-body request. A failed manifest
# exits the shell above; no fallback, retry, or new allocation is created.
timeout --signal=TERM --kill-after=10s 1140s \
    "$EP17_PYTHON" -B outputs/tools/reconcile_retained_events.py \
    --manifest outputs/runs/released_event_manifest/attempt001/annex_manifest.json \
    --out outputs/runs/released_event_reconciliation/retained001
