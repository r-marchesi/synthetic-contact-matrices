#!/bin/bash
#SBATCH --job-name=sc_02_gen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=h200
#SBATCH --nodelist=euler
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --mem=80G
#SBATCH --qos=normal
#SBATCH --container-mounts=/storage/DSH/projects/synthetic-contact-matrices/
#SBATCH --container-image=/storage/DSH/projects/synthetic-contact-matrices/image.sqsh
#SBATCH --output=/storage/DSH/projects/synthetic-contact-matrices/experiments/02_structured_semantic/logs/%x_%j.out
#SBATCH --error=/storage/DSH/projects/synthetic-contact-matrices/experiments/02_structured_semantic/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rmarchesi@fbk.eu

cd /storage/DSH/projects/synthetic-contact-matrices

mkdir -p experiments/02_structured_semantic/logs
mkdir -p experiments/02_structured_semantic/generated

source .env
export HF_HOME="/storage/DSH/projects/synthetic-contact-matrices/hf_cache"

echo "=================================================="
echo "Starting Phase 2: Constrained Inference (Full Set)"
echo "Job ID: $SLURM_JOB_ID"
echo "=================================================="

# -1 signals the script to process the entire validation JSONL
NUM_SAMPLES="-1"

CHECKPOINTS=("3" "6" "9" "12" "15")

for EPOCH in "${CHECKPOINTS[@]}"; do
    echo "--------------------------------------------------"
    echo "Generating constraints for Checkpoint: Epoch $EPOCH"
    echo "--------------------------------------------------"
    
    python experiments/02_structured_semantic/scripts/03_generate_constrained.py \
        --epoch "$EPOCH" \
        --samples "$NUM_SAMPLES"
done

echo "=================================================="
echo "Phase 2 Constrained Generation Completed."
echo "Outputs saved to experiments/02_structured_semantic/generated/"
echo "=================================================="