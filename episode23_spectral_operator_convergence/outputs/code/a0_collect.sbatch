#!/bin/bash
#SBATCH --job-name=ep23_a0_collect
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode23_spectral_operator_convergence/outputs/runs/a0_operator_alignment/%x-%j.out
# EP23 A0 stage 4: merge task reports, group sign tests, and validate the ASTRA program.
set -euo pipefail
ml devel python/3.12.1 math py-numpy/1.26.3 py-scipy/1.12.0_py312
E=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode23_spectral_operator_convergence
R=$E/outputs/runs/a0_operator_alignment/${RUN:?set RUN}
cd "$E"
python3 -P -s "$E/outputs/code/a0_alignment.py" --collect "$R/tasks" --out "$R/a0_report.json" | tee "$R/a0_summary.txt"
astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
timeout --signal=TERM --kill-after=10s 60s "$astra" validate outputs/astra/v0.0.14/astra.yaml --json > "$R/astra_validation.json" 2>&1 || echo "astra_exit=$?"
