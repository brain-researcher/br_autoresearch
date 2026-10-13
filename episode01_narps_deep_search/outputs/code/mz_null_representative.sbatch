#!/bin/bash -l
#SBATCH --job-name=ep01-null-pilot
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:35:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-representative-%j.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-representative-%j.err

# One required null representative, with short implementation checks grouped
# into its substantive allocation. No retry, follow-on job or null-array wave.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
EP_NULL_DIR="$PWD"
EP_CONTACT_DIR="$EP_NULL_DIR/../episode20_neurocam_prior_guided_codesign"
EP_NULL_CHECKS="outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-checks-${SLURM_JOB_ID}"
EP_CONTACT_CHECKS="outputs/runs/model_specification/passive_contact_checks_${SLURM_JOB_ID}"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
EP_NULL_PYTHON=$(command -v python3.12)
EP_NULL_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
hostname
printf 'SLURM_JOB_ID=%s CPUS=%s PURPOSE=null_representative\n' "$SLURM_JOB_ID" "$SLURM_CPUS_PER_TASK"
"$EP_NULL_PYTHON" --version
bash -n outputs/code/mz_null_representative.sbatch
mkdir "$EP_NULL_CHECKS"
timeout --signal=TERM --kill-after=10s 60s "$EP_NULL_PYTHON" -B \
  outputs/code/test_mz_null_representative.py -v > "$EP_NULL_CHECKS/tests.txt" 2>&1
printf 'EP01_NULL_TESTS_PASS\n'
if timeout --signal=TERM --kill-after=10s 30s "$EP_NULL_ASTRA" validate \
    outputs/astra/v0.0.14/astra.yaml --json \
    > "outputs/astra/v0.0.14/validation_20261005_null_${SLURM_JOB_ID}.json"; then
  printf 'EP01_NULL_ASTRA_PASS\n'
else
  printf 'EP01_NULL_ASTRA_OPEN: inspect validation output\n' >&2
fi

# EP20 remains a separate scope: no source data or results enter EP01.
if (
  cd "$EP_CONTACT_DIR" || exit 1
  mkdir "$EP_CONTACT_CHECKS" || exit 1
  timeout --signal=TERM --kill-after=10s 30s "$EP_NULL_PYTHON" -B \
    outputs/tools/test_passive_contact.py -v > "$EP_CONTACT_CHECKS/tests.txt" 2>&1 || exit 1
  printf 'EP20_CONTACT_TESTS_PASS\n'
  if timeout --signal=TERM --kill-after=10s 30s "$EP_NULL_ASTRA" validate \
      outputs/astra/v0.0.14/astra.yaml --json \
      > "outputs/astra/v0.0.14/validation_20261005_contact_${SLURM_JOB_ID}.json"; then
    printf 'EP20_CONTACT_ASTRA_PASS\n'
  else
    printf 'EP20_CONTACT_ASTRA_OPEN: inspect validation output\n' >&2
  fi
); then
  printf 'EP20_CONTACT_CHECKS_FINISHED\n'
else
  printf 'EP20_CONTACT_CHECKS_FAILED: preserve evidence for coordinator; independent EP01 pilot continues\n' >&2
fi

exec timeout --signal=TERM --kill-after=10s 5400s "$EP_NULL_PYTHON" -B \
  outputs/code/run_mz_null_representative.py \
  --design-events outputs/runs/behavioral_mechanism_kill_test/20261003-MC-SESOI-replicate000-attempt003/simulated_events.csv.gz \
  --implementation outputs/code/behavioral_implementation.json \
  --output-dir "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-representative-${SLURM_JOB_ID}" \
  --max-seconds 5400
