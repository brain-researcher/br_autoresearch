#!/bin/bash
# All30 real-data post/pre diagnostic plus small independently scoped fixture checks.
#SBATCH --job-name=ep18-signal-detrend
#SBATCH --partition=russpold,normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/repeated_signal_detrend/slurm-%j.out
#SBATCH --error=outputs/runs/repeated_signal_detrend/slurm-%j.err
set -euo pipefail
umask 077
[[ $# == 0 && -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* ]]
[[ "${SLURM_CPUS_PER_TASK:-}" == 1 && "${SLURM_MEM_PER_NODE:-}" == 7168 ]]
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]]
ep18_root="${OAK:?}/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability"
ep18_base="$ep18_root/outputs/runs/repeated_signal_detrend"
ep15_tools="$OAK/users/zijiao/br_autoresearch/episode15_mdtb_topography_vs_geometry/outputs/tools"
ep20_wrapper="$OAK/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign/outputs/tools/run_reference_s5_refinement_result.sh"
ep18_local="${L_SCRATCH_JOB:?}/ep18-signal-detrend"
mkdir -m 700 "$ep18_local"
export TMPDIR="$ep18_local" MPLCONFIGDIR="$ep18_local/matplotlib"
mkdir -m 700 "$MPLCONFIGDIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 viz py-matplotlib/3.8.3_py312
ep18_runtime="${GROUP_HOME:?}/users/zijiao/ep18_model_runtime_cpu241_py312_46842156"
ep18_astra="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
printf 'hostname=%s job_id=%s cpus=1 memory_mib=7168 threads=1\n' "$(hostname)" "$SLURM_JOB_ID"
cd "$ep18_root/outputs/code"
bash -n run_repeated_signal_detrend001.sbatch
bash -n "$ep20_wrapper"
timeout --signal=TERM --kill-after=10s 300s python3.12 -B -m unittest -v \
  test_repeated_signal_detrend test_report_repeated_signal_detrend \
  > "$ep18_base/checks_${SLURM_JOB_ID}.txt" 2>&1
# EP15 fixtures use temporary arrays/mocks only; no active policy or audit reader execution.
cd "$ep15_tools"
if timeout --signal=TERM --kill-after=10s 300s python3.12 -B -m unittest -v \
  test_observed_map_audit_predictions test_observed_map_audit_readers_runner test_report_observed_map_audit \
  > "../runs/observed_map_audit_fixture_checks_${SLURM_JOB_ID}.txt" 2>&1; then
  printf 'EP15 fixture checks passed; no audit data accessed.\n'
else
  printf 'EP15 fixture checks failed; inspect its recorded diagnostics; independent EP18 work continues.\n'
fi
cd "$ep18_root/outputs/code"
env PYTHONPATH="$ep18_astra/lib/python3.12/site-packages:$ep18_runtime/lib/python3.12/site-packages" \
  timeout --signal=TERM --kill-after=5s 60s "$ep18_astra/bin/astra-tools" \
  validate "$ep18_root/outputs/astra/v0.0.14/astra.yaml" --json \
  > "$ep18_base/astra_${SLURM_JOB_ID}.json" 2>&1 || \
  printf 'ASTRA validation incomplete; inspect recorded diagnostics.\n'
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1
unset PYTHONPATH PYTHONHOME
ep18_deadline=$((${SLURM_JOB_END_TIME:?}-120))
ep18_left=$((ep18_deadline-$(date +%s)))
(( ep18_left > 60 ))
/usr/bin/time -v -o "$ep18_base/resources_${SLURM_JOB_ID}.txt" \
  timeout --signal=TERM --kill-after=15s "$((ep18_left+15))s" "$ep18_runtime/bin/python" -u -B \
  run_repeated_signal_detrend.py --config "$ep18_root/outputs/config/repeated_signal_detrend001.json" \
  --operations "$ep18_base/operations_job_$SLURM_JOB_ID" --deadline-epoch "$ep18_deadline"
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 viz py-matplotlib/3.8.3_py312
ep18_left=$((SLURM_JOB_END_TIME-30-$(date +%s)))
(( ep18_left > 30 ))
timeout --signal=TERM --kill-after=10s "${ep18_left}s" python3.12 -B report_repeated_signal_detrend.py \
  --development-manifest "$ep18_root/outputs/runs/input_role_preparation/20261003-freeze_metadata01/development_manifest.json" \
  --diagnostic-root "$ep18_base/detrend001" \
  --original-report "$ep18_root/outputs/runs/repeated_signal_diagnostic/report001/repeated_signal_diagnostic.complete.json" \
  --output-root "$ep18_base/report001" \
  --figure-root "$ep18_root/outputs/figures/repeated_signal_detrend001"
printf 'all30_repeated_signal_detrend_report_complete\n'
