#!/bin/bash
#SBATCH --job-name=sc_07_16ep
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
cd /storage/DSH/projects/synthetic-contact-matrices

mkdir -p experiments/07_semantic_causal/logs
mkdir -p experiments/07_semantic_causal/data
mkdir -p experiments/07_semantic_causal/checkpoints
mkdir -p experiments/07_semantic_causal/generated
mkdir -p experiments/07_semantic_causal/results

source .env
export HF_HOME="/storage/DSH/projects/synthetic-contact-matrices/hf_cache"
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$((25000 + SLURM_JOB_ID % 10000))

echo "=================================================="
echo "Starting Exp 07: Clean Semantic Causal Hierarchy (16 Epochs)"
echo "Job ID: $SLURM_JOB_ID"
echo "=================================================="

# 1. Clean any prior Exp 07 artifacts
rm -rf experiments/07_semantic_causal/checkpoints/adapter_100_percent
rm -f  experiments/07_semantic_causal/generated/generated_100_percent_*.jsonl

# 2. Prepare all 5 split JSONLs with Clean Semantic encoding
ALL_SPLITS=("100_percent" "80_percent" "50_percent" "30_percent" "10_percent")
for SPLIT in "${ALL_SPLITS[@]}"; do
    echo "Preparing dataset for split: $SPLIT..."
    python experiments/07_semantic_causal/scripts/01_prepare_data.py \
        --train_ids "data/splits/train_ids_${SPLIT}.csv"
done

# 3. Train 100_percent LoRA model for 16 Epochs
echo "--------------------------------------------------"
echo "Training 100_percent LoRA Adapter (16 Epochs)..."
echo "--------------------------------------------------"
python experiments/07_semantic_causal/scripts/02_train.py \
    --split_tag "100_percent" \
    --epochs 16 \
    --batch_size 2 \
    --grad_accum 8 \
    --lr 8e-5 \
    --lora_r 32 \
    --lora_alpha 64

# 4. Generate BOTH plan_guided and full_free across Epochs 2, 4, 6, 8, 10, 12, 14, 16
CHECKPOINTS=("2" "4" "6" "8" "10" "12" "14" "16")
MODES=("plan_guided" "full_free")

for EPOCH in "${CHECKPOINTS[@]}"; do
    for MODE in "${MODES[@]}"; do
        echo "--------------------------------------------------"
        echo "Generating 100_percent | Mode: $MODE | Epoch: $EPOCH"
        echo "--------------------------------------------------"
        python experiments/07_semantic_causal/scripts/03_generate.py \
            --split_tag "100_percent" \
            --epoch "$EPOCH" \
            --mode "$MODE" \
            --temperature 0.95 \
            --min_p 0.03 \
            --batch_size 16 \
            --samples -1
    done
done

# 5. Evaluate Standard Metrics + Colleague WSMAD Epoch-Stopping Metric
echo "--------------------------------------------------"
echo "Running Unified Evaluation & WSMAD Epoch-Stopping Sweep..."
echo "--------------------------------------------------"
python experiments/07_semantic_causal/scripts/04_evaluate_and_wsmad.py

echo "=================================================="
echo "Exp 07 100% 16-Epoch Pipeline Completed!"
echo "=================================================="