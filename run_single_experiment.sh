#!/bin/bash
#SBATCH --job-name=sc_sweep_gemma2
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

echo "Starting Gemma 2 (9B) Prompt Sweep Job..."
echo "Processing Experiment: $EXP_NAME"

cd /storage/DSH/projects/synthetic-contact-matrices
mkdir -p slurm_outputs
mkdir -p logs

# Load credentials and redirect HF to the local cached weights
source .env
export HF_HOME="/storage/DSH/projects/synthetic-contact-matrices/hf_cache"

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$((25000 + SLURM_JOB_ID % 10000))

CONFIG_FILE="configs/${EXP_NAME}.json"
SPLIT_FILE="data/splits/train_ids_100_percent.csv"

echo "=================================================="
echo "Config: ${CONFIG_FILE}"
echo "=================================================="

python scripts/003_fine_tune.py \
    --config "${CONFIG_FILE}" \
    --train_ids "${SPLIT_FILE}"

echo "Completed ${EXP_NAME}."