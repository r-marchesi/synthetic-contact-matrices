#!/bin/bash

# We will generate for the adapters that are currently finished
SPLITS=("10_percent" "30_percent" "50_percent" "80_percent" "100_percent")

for SPLIT in "${SPLITS[@]}"; do
  JOB_NAME="gen_rq1_${SPLIT}"
  echo "Submitting Generation job for split: ${SPLIT}"
  sbatch --job-name="${JOB_NAME}" run_single_generation.sh "${SPLIT}"
done