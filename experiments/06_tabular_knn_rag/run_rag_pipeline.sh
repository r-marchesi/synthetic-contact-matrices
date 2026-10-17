#!/bin/bash
#SBATCH --job-name=sc_06_rag
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=h200
#SBATCH --nodelist=euler
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --mem=80G
#SBATCH --qos=normal
#SBATCH --container-mounts=/storage/DSH/projects/synthetic-contact-matrices/
#SBATCH --container-image=/storage/DSH/projects/synthetic-contact-matrices/image.sqsh
#SBATCH --output=/storage/DSH/projects/synthetic-contact-matrices/experiments/06_tabular_knn_rag/logs/%x_%j.out
#SBATCH --error=/storage/DSH/projects/synthetic-contact-matrices/experiments/06_tabular_knn_rag/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rmarchesi@fbk.eu

set -e
cd /storage/DSH/projects/synthetic-contact-matrices
mkdir -p experiments/06_tabular_knn_rag/logs

source .env
export HF_HOME="/storage/DSH/projects/synthetic-contact-matrices/hf_cache"

echo "=================================================="
echo "Starting Exp 06: Tabular-kNN Quantile RAG Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "=================================================="

# Ensure Exp 04 data exists
if [ ! -f "experiments/04_causal_hierarchy/data/val_causal.jsonl" ]; then
    echo "Preparing Causal DSL data via Exp 04..."
    python experiments/04_causal_hierarchy/scripts/01_prepare_data.py
fi

echo "Step 1: Building Epidemiological Weighted Gower Index..."
python experiments/06_tabular_knn_rag/scripts/01_build_tabular_index.py \
    --split_tag 100_percent

echo "Step 2A: Generating Full Validation Set with Zero-Shot Base Model + Tabular RAG..."
python experiments/06_tabular_knn_rag/scripts/02_generate_rag.py \
    --split_tag 100_percent \
    --top_k 40 \
    --temperature 0.90 \
    --min_p 0.03 \
    --batch_size 8 \
    --samples -1

# If Exp 04 training has finished, also run Hybrid SFT + Tabular RAG
if [ -d "experiments/04_causal_hierarchy/checkpoints/adapter_100_percent" ]; then
    echo "Step 2B: Generating Full Validation Set with Hybrid SFT (Epoch 6) + Tabular RAG..."
    python experiments/06_tabular_knn_rag/scripts/02_generate_rag.py \
        --split_tag 100_percent \
        --use_sft_adapter \
        --sft_epoch 6 \
        --top_k 40 \
        --temperature 0.90 \
        --min_p 0.03 \
        --batch_size 8 \
        --samples -1
fi

echo "Step 3: Running Unified Evaluation Suite..."
python experiments/06_tabular_knn_rag/scripts/03_evaluate_rag.py

echo "=================================================="
echo "Exp 06 Tabular-kNN RAG Completed!"
echo "Results saved to experiments/06_tabular_knn_rag/results/"
echo "=================================================="