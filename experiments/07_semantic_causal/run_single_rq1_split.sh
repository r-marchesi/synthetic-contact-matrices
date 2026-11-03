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
#SBATCH --output=/storage/DSH/projects/synthetic-contact-matrices/experiments/07_semantic_causal/logs/%x_%j.out
#SBATCH --error=/storage/DSH/projects/synthetic-contact-matrices/experiments/07_semantic_causal/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rmarchesi@fbk.eu

set -e

SPLIT=$1
if [ -z "$SPLIT" ]; then
    echo "Error: SPLIT argument is required (e.g. 80_percent)"
    exit 1
fi

cd /storage/DSH/projects/synthetic-contact-matrices
source .env
export HF_HOME="/storage/DSH/projects/synthetic-contact-matrices/hf_cache"
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

echo "=================================================="
echo "Exp 07 RQ1 Parallel Worker | Split: $SPLIT | Job: $SLURM_JOB_ID"
echo "=================================================="

# 1. Clean old checkpoints/generated files for this specific split only
rm -rf "experiments/07_semantic_causal/checkpoints/adapter_${SPLIT}"
rm -f  "experiments/07_semantic_causal/generated/generated_${SPLIT}_"*.jsonl

# 2. Ensure JSONL split data exists
if [ ! -f "experiments/07_semantic_causal/data/train_causal_${SPLIT}.jsonl" ]; then
    python experiments/07_semantic_causal/scripts/01_prepare_data.py \
        --train_ids "data/splits/train_ids_${SPLIT}.csv"
fi

# 3. Train 16-Epoch LoRA Adapter for this split
echo "--------------------------------------------------"
echo "Training LoRA Adapter for $SPLIT (16 Epochs)..."
echo "--------------------------------------------------"
python experiments/07_semantic_causal/scripts/02_train.py \
    --split_tag "$SPLIT" \
    --epochs 16 \
    --batch_size 2 \
    --grad_accum 8 \
    --lr 8e-5 \
    --lora_r 32 \
    --lora_alpha 64

# 4. Generate plan_guided and full_free at Epochs 12, 14, and 16
EPOCHS=("12" "14" "16")
MODES=("plan_guided" "full_free")

for EP in "${EPOCHS[@]}"; do
    for MODE in "${MODES[@]}"; do
        echo "--------------------------------------------------"
        echo "Generating $SPLIT | Mode: $MODE | Epoch: $EP"
        echo "--------------------------------------------------"
        python experiments/07_semantic_causal/scripts/03_generate.py \
            --split_tag "$SPLIT" \
            --epoch "$EP" \
            --mode "$MODE" \
            --temperature 0.95 \
            --min_p 0.03 \
            --batch_size 16 \
            --samples -1
    done
done

echo "✅ Completed RQ1 Split: $SPLIT"