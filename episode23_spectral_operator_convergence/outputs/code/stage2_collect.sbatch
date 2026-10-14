#!/bin/bash
#SBATCH --job-name=ep23_stage2_collect
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode23_spectral_operator_convergence/outputs/runs/%x-%j.out
# EP23: collect A1/A2/A3 task outputs (WHICH=a1|a1r32|a2|a3) and validate ASTRA.
set -euo pipefail
ml devel python/3.12.1 math py-numpy/1.26.3 py-scipy/1.12.0_py312
E=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode23_spectral_operator_convergence
EP04=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode04_shared_scene_geometry
M=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/steward_acquisition/things_metadata_osf_jum2f_20260922/payload
ROWS=$EP04/outputs/runs/source_readiness/20261003-development_embeddings01/development_rows.jsonl
cd "$SCRATCH/br_autoresearch/episode23_spectral_operator_convergence"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
case "${WHICH:?}" in
  a1)    python3 -P -s "$E/outputs/code/a1_metric_family.py" --collect "$E/outputs/runs/a1_metric_family/$RUN/cells" --out "$E/outputs/runs/a1_metric_family/$RUN/a1_report.json" | tee "$E/outputs/runs/a1_metric_family/$RUN/a1_summary.txt" ;;
  a2)    python3 -P -s "$E/outputs/code/a2_coarse_fine.py" --mode collect --collect-dir "$E/outputs/runs/a2_coarse_fine/$RUN/tasks" --out "$E/outputs/runs/a2_coarse_fine/$RUN/a2_report.json" | tee "$E/outputs/runs/a2_coarse_fine/$RUN/a2_summary.txt" ;;
  a3)    python3 -P -s "$E/outputs/code/a3_pooling_simulation.py" --collect "$E/outputs/runs/a3_pooling_simulation/$RUN/tasks" --out "$E/outputs/runs/a3_pooling_simulation/$RUN/a3_report.json" | tee "$E/outputs/runs/a3_pooling_simulation/$RUN/a3_summary.txt" ;;
esac
