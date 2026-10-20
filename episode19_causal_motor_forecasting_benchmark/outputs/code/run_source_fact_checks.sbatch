#!/bin/bash
#SBATCH --job-name=ep19-srcfacts
#SBATCH --partition=russpold
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
# EP19 source-fact checks (development data only). See source_timing_handoff.md, October 16.
set -euo pipefail
EP=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode19_causal_motor_forecasting_benchmark
OUT=$EP/outputs/runs/source_views_and_timing/source_facts_job_${SLURM_JOB_ID}
mkdir -p "$OUT"
cd "$OUT"
ml python/3.12.1 py-numpy/1.26.3_py312 py-scipy/1.12.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date -Is)"
python3 -P "$EP/outputs/code/way_source_fact_checks.py" --workers 8 --output way_source_facts.json
echo "way_done=$(date -Is)"
python3 -P "$EP/outputs/code/self_paced_source_fact_checks.py" --workers 8 --output self_paced_source_facts.json
echo "self_paced_done=$(date -Is)"
