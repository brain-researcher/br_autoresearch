#!/bin/bash
#SBATCH --job-name=ep19-rawonset
#SBATCH --partition=russpold
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode19_causal_motor_forecasting_benchmark/outputs/runs/source_views_and_timing/raw_onset_test_job_%j.out
# EP19 self-paced raw-onset lag test (development only; pre-declared in the log and timing handoff).
set -euo pipefail
EP=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode19_causal_motor_forecasting_benchmark
FREEZE=$EP/outputs/runs/source_views_and_timing/detector_freeze_job_48219799
OUT=$EP/outputs/runs/source_views_and_timing/raw_onset_test_job_${SLURM_JOB_ID}
cd "$EP"
ml python/3.12.1 py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 > /dev/null 2>&1
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date -Is)"
python3 -P outputs/code/self_paced_raw_onset_test.py --freeze-dir "$FREEZE" --output-dir "$OUT" --workers 8 ${PARTICIPANTS:+--participants "$PARTICIPANTS"}
echo "test_done=$(date -Is)"
module purge > /dev/null 2>&1
ml python/3.12.1 viz py-matplotlib/3.8.3_py312 > /dev/null 2>&1
python3 -P outputs/code/plot_raw_onset_test.py --input-dir "$OUT" --output "$OUT/raw_onset_test.png"
echo "plot_done=$(date -Is)"
