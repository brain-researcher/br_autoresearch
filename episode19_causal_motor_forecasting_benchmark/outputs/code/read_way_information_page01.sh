#!/usr/bin/env bash
# One document-only attempt; the coordinator supplies the 90-second outer cap.
set -euo pipefail

ep19_host=$(hostname)
printf 'hostname=%s SLURM_JOB_ID=%s SLURM_CPUS_PER_TASK=%s\n' \
  "$ep19_host" "${SLURM_JOB_ID-unset}" "${SLURM_CPUS_PER_TASK-unset}"
if [[ -z ${SLURM_JOB_ID:-} || $ep19_host == *-ln* || ${SLURM_CPUS_PER_TASK:-} != 1 ]]; then
  printf 'One CPU in an actual compute-node Slurm step is required.\n' >&2
  exit 2
fi
: "${OAK:?OAK is required}"
: "${L_SCRATCH_JOB:?Per-job node-local scratch is required}"
ep19_expected="$OAK/users/zijiao/br_autoresearch/episode19_causal_motor_forecasting_benchmark"
if [[ $(pwd -P) != "$ep19_expected" || ! -d $L_SCRATCH_JOB ]]; then
  printf 'Use the exact EP19 working directory and existing L_SCRATCH_JOB.\n' >&2
  exit 2
fi

source /share/software/user/open/lmod/lmod/init/bash
module purge
module load system poppler/0.47.0

ep19_source_url='https://pmc-oa-opendata.s3.amazonaws.com/PMC4365902.1/sdata201447-s1.pdf'
ep19_xml_url='https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4365902/fullTextXML'
ep19_output_parent="$ep19_expected/outputs/source_documents/way_information"
ep19_output="$ep19_output_parent/20261005-page01"
mkdir -p "$ep19_output_parent"
mkdir "$ep19_output"  # Refuse a previous attempt, including partial output.
set -o noclobber
printf '%s\n' "$ep19_source_url" > "$ep19_output/source_url.txt"
printf '%s\n' "$ep19_xml_url" > "$ep19_output/source_xml_url.txt"
printf 'job_id=%s\nhostname=%s\ncpus=1\nexpected_pdf_bytes=193612\nrequested_pages=1\n' \
  "$SLURM_JOB_ID" "$ep19_host" > "$ep19_output/execution.txt"

ep19_scratch=$(mktemp -d "$L_SCRATCH_JOB/ep19-information-page01.XXXXXX")
ep19_pdf="$ep19_scratch/Information.pdf"
# Exactly this public document: no redirects, retry, alternate URL or archive.
curl --fail --silent --show-error --max-time 20 --max-filesize 250000 \
  "$ep19_source_url" --output "$ep19_pdf"
if [[ $(stat -c '%s' "$ep19_pdf") != 193612 ]]; then
  printf 'PDF size differs from the official supplementary-material record.\n' >&2
  exit 2
fi
cp --no-clobber "$ep19_pdf" "$ep19_output/Information.pdf"
timeout --signal=TERM --kill-after=2s 15s pdfinfo "$ep19_pdf" \
  > "$ep19_output/pdfinfo.txt"
timeout --signal=TERM --kill-after=2s 20s pdftotext \
  -f 1 -l 1 -layout -enc UTF-8 "$ep19_pdf" "$ep19_scratch/page01.txt"
cp --no-clobber "$ep19_scratch/page01.txt" "$ep19_output/page01.txt"
printf 'complete_document_page01_only\n' > "$ep19_output/status.txt"
printf 'EP19 documentation page 1 complete: %s\n' "$ep19_output"
