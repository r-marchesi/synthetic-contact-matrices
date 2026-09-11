#!/bin/bash

# Define array of experiments to launch
EXPERIMENTS=("exp_01_json" "exp_02_yaml" "exp_03_cot" "exp_04_nl")

for EXP in "${EXPERIMENTS[@]}"; do
  JOB_NAME="sc_${EXP}"
  echo "Submitting Slurm job for experiment: ${EXP} (Job Name: ${JOB_NAME})"
  sbatch --job-name="${JOB_NAME}" run_single_experiment.sh "${EXP}"
done