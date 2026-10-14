#!/bin/bash
#SBATCH --job-name=ep23_theory_checks
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode23_spectral_operator_convergence/outputs/runs/p1_p5_theory/%x-%j.out
# EP23 P1-P5 numerical refutation attempts (passing is not proof).
set -euo pipefail
ml devel python/3.12.1 math py-numpy/1.26.3 py-scipy/1.12.0_py312
E=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode23_spectral_operator_convergence
EP04=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode04_shared_scene_geometry
M=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/steward_acquisition/things_metadata_osf_jum2f_20260922/payload
ROWS=$EP04/outputs/runs/source_readiness/20261003-development_embeddings01/development_rows.jsonl
cd "$SCRATCH/br_autoresearch/episode23_spectral_operator_convergence"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p "$E/outputs/runs/p1_p5_theory/${RUN:?set RUN}"
python3 -P -s "$E/outputs/code/theory_checks.py" --out "$E/outputs/runs/p1_p5_theory/$RUN/theory_checks.json"
