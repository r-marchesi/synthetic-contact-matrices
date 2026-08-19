#!/bin/bash

# Define array of splits to launch
SPLITS=("10_percent" "30_percent" "50_percent" "80_percent" "100_percent")

for SPLIT in "${SPLITS[@]}"; do
  JOB_NAME="sc_rq1_gemma2_${SPLIT}_v2"
  echo "Submitting Slurm job for split: ${SPLIT} (Job Name: ${JOB_NAME})"
  sbatch --job-name="${JOB_NAME}" run_single_split.sh "${SPLIT}"
done