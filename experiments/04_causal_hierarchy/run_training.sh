#!/bin/bash
#SBATCH --job-name=sc_04_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=h200
#SBATCH --nodelist=euler
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --mem=100G
#SBATCH --qos=normal
#SBATCH --container-mounts=/storage/DSH/projects/synthetic-contact-matrices/
#SBATCH --container-image=/storage/DSH/projects/synthetic-contact-matrices/image.sqsh
#SBATCH --output=/storage/DSH/projects/synthetic-contact-matrices/experiments/04_causal_hierarchy/logs/%x_%j.out
#SBATCH --error=/storage/DSH/projects/synthetic-contact-matrices/experiments/04_causal_hierarchy/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rmarchesi@fbk.eu

# Move to project root
cd /storage/DSH/projects/synthetic-contact-matrices

# Create logs directory
mkdir -p experiments/04_causal_hierarchy/logs

# Load environment and cache paths
source .env
export HF_HOME="/storage/DSH/projects/synthetic-contact-matrices/hf_cache"

# Prevent multi-node port collisions
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$((25000 + SLURM_JOB_ID % 10000))

echo "=================================================="
echo "Starting Exp 04: Causal Hierarchy Training"
echo "Job ID: $SLURM_JOB_ID"
echo "=================================================="

echo "Step 1: Preparing Causal Social-Circle DSL Dataset..."
python experiments/04_causal_hierarchy/scripts/01_prepare_data.py

echo "Step 2: Training Completion-Only LoRA on Gemma-2-9B-IT (Cosine LR)..."
python experiments/04_causal_hierarchy/scripts/02_train.py \
    --epochs 8 \
    --batch_size 2 \
    --grad_accum 8 \
    --lr 1e-4 \
    --lora_r 32 \
    --lora_alpha 64

echo "=================================================="
echo "Exp 04 Training Completed Successfully."
echo "=================================================="