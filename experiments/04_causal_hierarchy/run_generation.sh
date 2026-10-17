#!/bin/bash
#SBATCH --job-name=sc_04_gen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=h200
#SBATCH --nodelist=euler
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --mem=80G
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

echo "=================================================="
echo "Starting Exp 04: Generation & Unified Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "=================================================="

# Checkpoints / Epochs to sweep
CHECKPOINTS=("4" "6" "8")

for EPOCH in "${CHECKPOINTS[@]}"; do
    echo "--------------------------------------------------"
    echo "Generating Plan-Guided Diaries for Epoch: $EPOCH"
    echo "--------------------------------------------------"
    
    python experiments/04_causal_hierarchy/scripts/03_generate.py \
        --epoch "$EPOCH" \
        --mode plan_guided \
        --temperature 0.95 \
        --min_p 0.03 \
        --batch_size 16 \
        --samples -1
done

echo "--------------------------------------------------"
echo "Generating Full-Free Ablation for Epoch: 6"
echo "--------------------------------------------------"
python experiments/04_causal_hierarchy/scripts/03_generate.py \
    --epoch 6 \
    --mode full_free \
    --temperature 0.95 \
    --min_p 0.03 \
    --batch_size 16 \
    --samples -1

echo "--------------------------------------------------"
echo "Running Unified Evaluation Suite..."
echo "--------------------------------------------------"
python experiments/04_causal_hierarchy/scripts/04_evaluate.py

echo "=================================================="
echo "Exp 04 Generation & Evaluation Completed."
echo "Results saved to experiments/04_causal_hierarchy/results/"
echo "=================================================="