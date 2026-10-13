#!/usr/bin/env bash
#SBATCH --job-name=ep10-v3-m2-observed
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=russpold,normal
#SBATCH --output=/scratch/users/%u/br_autoresearch/episode10_single_cell_coprojection/%x-%j.out
#SBATCH --error=/scratch/users/%u/br_autoresearch/episode10_single_cell_coprojection/%x-%j.err

set -euo pipefail
umask 007

: "${SCRATCH:?SCRATCH is required}"
: "${OAK:?OAK is required}"
: "${GROUP_HOME:?GROUP_HOME is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
: "${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK is required}"

python_bin="$GROUP_HOME/users/zijiao/envs/ep10_targets_py312/bin/python"
episode="$OAK/users/zijiao/br_autoresearch/episode10_single_cell_coprojection"
launcher="$episode/outputs/stage0/tools/launch_v3_m2_observed_search_v1.py"
scratch_root="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection"

interface_root="$scratch_root/v3_model_interface_preflight_v1"
geometry_root="$scratch_root/v3_m1_geometry_v1"
target_root="$scratch_root/gao_mop_development_targets_v3_ep10_assets"
m1_checkpoint_root="$scratch_root/v3_m1_real_fit_single_thread_v3_job_45739355/checkpoints"

core_report="$scratch_root/v3_m2_core_contract_qualification_job_45832824/v3_m2_core_contract_qualification.json"
actual_k_report="$scratch_root/v3_m2_actual_k_gate_v1_job_45847880/v3_m2_actual_k_gate_summary.json"
m1_adequacy_report="$scratch_root/v3_m2_m1_adequacy_gate_job_45773980/v3_m2_m1_adequacy.json"
optional_report="$scratch_root/v3_m2_optional_family_qualification_job_45924290/v3_m2_optional_family_qualification.json"
scale_report="$scratch_root/v3_m2_search_scale_benchmark_v1_job_45924664/v3_m2_search_scale_benchmark_summary.json"
timing_report="$scratch_root/v3_m2_falsifier_timing_v1_job_45881554/v3_m2_falsifier_timing_supplement.json"

test -x "$python_bin"
test -f "$launcher"
test ! -L "$launcher"

ml purge
ml load devel python/3.12.1

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

cd "$episode"

"$python_bin" "$launcher" \
  --interface-root "$interface_root" \
  --geometry-root "$geometry_root" \
  --target-root "$target_root" \
  --m1-checkpoint-root "$m1_checkpoint_root" \
  --core-report "$core_report" \
  --actual-k-report "$actual_k_report" \
  --m1-adequacy-report "$m1_adequacy_report" \
  --optional-report "$optional_report" \
  --scale-report "$scale_report" \
  --falsifier-timing-report "$timing_report"

printf 'EP10 v3 M2 observed search including selected refit: PASS\n'
