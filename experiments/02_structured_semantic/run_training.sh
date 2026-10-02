#!/bin/bash
#SBATCH --job-name=sc_02_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=h200
#SBATCH --nodelist=euler
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --mem=100G
#SBATCH --qos=normal
#SBATCH --container-mounts=/storage/DSH/projects/synthetic-contact-matrices/
#SBATCH --container-image=/storage/DSH/projects/synthetic-contact-matrices/image.sqsh
#SBATCH --output=/storage/DSH/projects/synthetic-contact-matrices/experiments/02_structured_semantic/logs/%x_%j.out
#SBATCH --error=/storage/DSH/projects/synthetic-contact-matrices/experiments/02_structured_semantic/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rmarchesi@fbk.eu

# Move to the project root for execution
cd /storage/DSH/projects/synthetic-contact-matrices

# Ensure the new local logs directory exists
mkdir -p experiments/02_structured_semantic/logs

# Load credentials and redirect HF to the local cached weights
source .env
export HF_HOME="/storage/DSH/projects/synthetic-contact-matrices/hf_cache"

# Prevent port collisions on the node
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$((25000 + SLURM_JOB_ID % 10000))

echo "=================================================="
echo "Starting Phase 2: Structured Semantic Training"
echo "Job ID: $SLURM_JOB_ID"
echo "=================================================="

# Execute the new training script with 15 epochs and a constant learning rate
python experiments/02_structured_semantic/scripts/02_train.py \
    --epochs 15 \
    --batch_size 4 \
    --lr 5e-5

echo "Phase 2 Training Completed."