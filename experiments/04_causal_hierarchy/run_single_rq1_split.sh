#!/bin/bash
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

set -e
cd /storage/DSH/projects/synthetic-contact-matrices

SPLIT=$1
PG_EPOCH=${2:-6}
FF_EPOCH=${3:-6}
MAX_EPOCHS=${4:-6}

if [ -z "$SPLIT" ]; then
    echo "Error: SPLIT argument is required (e.g., 80_percent)."
    exit 1
fi

source .env
export HF_HOME="/storage/DSH/projects/synthetic-contact-matrices/hf_cache"
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$((25000 + SLURM_JOB_ID % 10000))

echo "=================================================="
echo "RQ1 Parallel Worker | Split: $SPLIT | Max Epochs: $MAX_EPOCHS"
echo "Best Plan-Guided Epoch: $PG_EPOCH | Best Full-Free Epoch: $FF_EPOCH"
echo "Job ID: $SLURM_JOB_ID | Master Port: $MASTER_PORT"
echo "=================================================="

# Clean any prior checkpoints/generations for this split
rm -rf "experiments/04_causal_hierarchy/checkpoints/adapter_${SPLIT}"
rm -f "experiments/04_causal_hierarchy/generated/generated_${SPLIT}_"*.jsonl

# 1. Train LoRA Adapter for this split
python experiments/04_causal_hierarchy/scripts/02_train.py \
    --split_tag "$SPLIT" \
    --epochs "$MAX_EPOCHS" \
    --batch_size 2 \
    --grad_accum 8 \
    --lr 1e-4 \
    --lora_r 32 \
    --lora_alpha 64

# 2. Generate with best plan_guided setup
echo "Generating plan_guided (Epoch $PG_EPOCH) for $SPLIT..."
python experiments/04_causal_hierarchy/scripts/03_generate.py \
    --split_tag "$SPLIT" \
    --epoch "$PG_EPOCH" \
    --mode plan_guided \
    --temperature 0.95 \
    --min_p 0.03 \
    --batch_size 16 \
    --samples -1

# 3. Generate with best full_free setup
echo "Generating full_free (Epoch $FF_EPOCH) for $SPLIT..."
python experiments/04_causal_hierarchy/scripts/03_generate.py \
    --split_tag "$SPLIT" \
    --epoch "$FF_EPOCH" \
    --mode full_free \
    --temperature 0.95 \
    --min_p 0.03 \
    --batch_size 16 \
    --samples -1

echo "=================================================="
echo "Completed RQ1 Worker for Split: $SPLIT"
echo "=================================================="