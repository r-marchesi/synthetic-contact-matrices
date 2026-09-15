#!/bin/bash

EXPERIMENTS=("exp_01_json" "exp_02_yaml" "exp_03_cot" "exp_04_nl")

for EXP in "${EXPERIMENTS[@]}"; do
  JOB_NAME="gen_sweep_${EXP}"
  echo "Submitting Generation job for experiment: ${EXP}"
  sbatch --job-name="${JOB_NAME}" run_single_generation.sh "${EXP}"
done