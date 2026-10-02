#!/bin/bash
#SBATCH --job-name=sc_03_rag
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=h200
#SBATCH --nodelist=euler
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --mem=80G
#SBATCH --qos=normal
#SBATCH --container-mounts=/storage/DSH/projects/synthetic-contact-matrices/
#SBATCH --container-image=/storage/DSH/projects/synthetic-contact-matrices/image.sqsh
#SBATCH --output=/storage/DSH/projects/synthetic-contact-matrices/experiments/03_rag_generation/logs/%x_%j.out
#SBATCH --error=/storage/DSH/projects/synthetic-contact-matrices/experiments/03_rag_generation/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rmarchesi@fbk.eu

# Move to the project root
cd /storage/DSH/projects/synthetic-contact-matrices

# Ensure directories exist
mkdir -p experiments/03_rag_generation/logs
mkdir -p experiments/03_rag_generation/generated

# Load credentials and redirect HF cache to avoid anonymous rate limits
source .env
export HF_HOME="/storage/DSH/projects/synthetic-contact-matrices/hf_cache"

echo "=================================================="
echo "Starting Exp 03: RAG Constrained Inference"
echo "Job ID: $SLURM_JOB_ID"
echo "=================================================="

# Dynamically count the exact number of rows in the validation set
VAL_FILE="experiments/02_structured_semantic/data/val_dense.jsonl"
TOTAL_SAMPLES=$(wc -l < "$VAL_FILE")

echo "Found $TOTAL_SAMPLES participants in validation set. Generating all..."
    
python experiments/03_rag_generation/scripts/02_generate_rag.py \
    --num_samples "$TOTAL_SAMPLES"

echo "=================================================="
echo "Generation Completed."
echo "Outputs saved to experiments/03_rag_generation/generated/rag_generated_contacts.jsonl"
echo "=================================================="