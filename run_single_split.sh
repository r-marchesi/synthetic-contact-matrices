#!/bin/bash
#SBATCH --job-name=sc_rq1_gemma2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=h200
#SBATCH --nodelist=euler
#SBATCH --gres=gpu:1g.35gb:1
#SBATCH --mem=100G
#SBATCH --qos=normal
#SBATCH --container-mounts=/storage/DSH/projects/synthetic-contact-matrices/
#SBATCH --container-image=/storage/DSH/projects/synthetic-contact-matrices/image.sqsh
#SBATCH --output=/storage/DSH/projects/synthetic-contact-matrices/slurm_outputs/%x_%j.out
#SBATCH --error=/storage/DSH/projects/synthetic-contact-matrices/slurm_outputs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rmarchesi@fbk.eu

# Capture split parameter passed from master script
SPLIT=$1

if [ -z "$SPLIT" ]; then
  echo "Error: No split parameter provided."
  exit 1
fi

echo "Starting Gemma 2 (9B) Fine-Tuning Job..."
echo "Processing Split: $SPLIT"

cd /storage/DSH/projects/synthetic-contact-matrices
mkdir -p slurm_outputs

source .env
export HF_HOME="/storage/DSH/projects/synthetic-contact-matrices/hf_cache"

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$((25000 + SLURM_JOB_ID % 10000))

python scripts/03_fine_tune.py \
    --train_file data/splits/train_${SPLIT}.jsonl \
    --model_name "google/gemma-2-9b-it" \
    --output_dir models/gemma2-9b-contact-${SPLIT}