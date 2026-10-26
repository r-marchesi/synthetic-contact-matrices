#!/bin/bash
#SBATCH --job-name=sc_04_100pct
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

mkdir -p experiments/04_causal_hierarchy/logs
mkdir -p experiments/04_causal_hierarchy/generated
mkdir -p experiments/04_causal_hierarchy/results

source .env
export HF_HOME="/storage/DSH/projects/synthetic-contact-matrices/hf_cache"
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$((25000 + SLURM_JOB_ID % 10000))

echo "=================================================="
echo "Starting Full 100% Pipeline (Updated 6-Location Schema)"
echo "Job ID: $SLURM_JOB_ID"
echo "=================================================="

# 1. Full cleanup of all splits (including partial artifacts from stopped rq1_scaling runs)
rm -rf experiments/04_causal_hierarchy/data/*.jsonl
rm -rf experiments/04_causal_hierarchy/checkpoints/adapter_*
rm -f  experiments/04_causal_hierarchy/generated/*.jsonl
rm -rf experiments/04_causal_hierarchy/results/*
mkdir -p experiments/04_causal_hierarchy/data
mkdir -p experiments/04_causal_hierarchy/checkpoints
mkdir -p experiments/04_causal_hierarchy/generated
mkdir -p experiments/04_causal_hierarchy/results

# 2. Prepare all 5 split JSONLs sequentially now so parallel jobs never collide on val_causal.jsonl
ALL_SPLITS=("100_percent" "80_percent" "50_percent" "30_percent" "10_percent")
for SPLIT in "${ALL_SPLITS[@]}"; do
    echo "Preparing dataset for split: $SPLIT..."
    python experiments/04_causal_hierarchy/scripts/01_prepare_data.py \
        --train_ids "data/splits/train_ids_${SPLIT}.csv"
done

# 3. Train 100_percent LoRA model (8 Epochs, Cosine LR)
echo "--------------------------------------------------"
echo "Training 100_percent LoRA Adapter..."
echo "--------------------------------------------------"
python experiments/04_causal_hierarchy/scripts/02_train.py \
    --split_tag "100_percent" \
    --epochs 8 \
    --batch_size 2 \
    --grad_accum 8 \
    --lr 1e-4 \
    --lora_r 32 \
    --lora_alpha 64

# 4. Generate BOTH plan_guided and full_free across Epochs 4, 6, 8
CHECKPOINTS=("4" "6" "8")
MODES=("plan_guided" "full_free")

for EPOCH in "${CHECKPOINTS[@]}"; do
    for MODE in "${MODES[@]}"; do
        echo "--------------------------------------------------"
        echo "Generating 100_percent | Mode: $MODE | Epoch: $EPOCH"
        echo "--------------------------------------------------"
        python experiments/04_causal_hierarchy/scripts/03_generate.py \
            --split_tag "100_percent" \
            --epoch "$EPOCH" \
            --mode "$MODE" \
            --temperature 0.95 \
            --min_p 0.03 \
            --batch_size 16 \
            --samples -1
    done
done

# 5. Evaluate all 100_percent runs & RQ2 subgroups
echo "--------------------------------------------------"
echo "Running Unified Evaluation & RQ2 Subgroup Breakdown..."
echo "--------------------------------------------------"
python experiments/04_causal_hierarchy/scripts/04_evaluate.py
python experiments/04_causal_hierarchy/scripts/05_evaluate_rq2_subgroups.py

echo "=================================================="
echo "100% Pipeline Completed! Check leaderboard_04_causal.csv"
echo "=================================================="