#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=h200
#SBATCH --nodelist=euler
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --mem=100G
#SBATCH --qos=normal
#SBATCH --container-mounts=/storage/DSH/projects/synthetic-contact-matrices/
#SBATCH --container-image=/storage/DSH/projects/synthetic-contact-matrices/image.sqsh
#SBATCH --output=/storage/DSH/projects/synthetic-contact-matrices/slurm_outputs/%x_%j.out
#SBATCH --error=/storage/DSH/projects/synthetic-contact-matrices/slurm_outputs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rmarchesi@fbk.eu

EXP_NAME=$1

if [ -z "$EXP_NAME" ]; then
  echo "Error: No experiment name provided."
  exit 1
fi

echo "Starting Gemma 2 (9B) Generation Job for Experiment: $EXP_NAME"

cd /storage/DSH/projects/synthetic-contact-matrices
mkdir -p data/results
mkdir -p slurm_outputs

source .env
export HF_HOME="/storage/DSH/projects/synthetic-contact-matrices/hf_cache"

CONFIG_FILE="configs/${EXP_NAME}.json"

# Define the target epochs to evaluate
EPOCHS=(2 4 6 8 10)

for EPOCH in "${EPOCHS[@]}"; do
    echo "=================================================="
    echo "Generating matrices for ${EXP_NAME} at Epoch ${EPOCH}"
    echo "=================================================="

    python scripts/004_generate_val.py \
        --config "${CONFIG_FILE}" \
        --target_epoch "${EPOCH}"

    echo "Completed Epoch ${EPOCH} for ${EXP_NAME}."
done

echo "All requested epochs for ${EXP_NAME} finished."