#!/bin/bash
#SBATCH --job-name=gen_rq1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=h200
#SBATCH --nodelist=euler
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --mem=100G
#SBATCH --qos=normal
#SBATCH --container-mounts=/storage/DSH/projects/synthetic-contact-matrices/
#SBATCH --container-image=/storage/DSH/projects/synthetic-contact-matrices/image.sqsh
#SBATCH --output=/storage/DSH/projects/synthetic-contact-matrices/slurm_outputs/%x_%j.out
#SBATCH --error=/storage/DSH/projects/synthetic-contact-matrices/slurm_outputs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rmarchesi@fbk.eu

SPLIT=$1

if [ -z "$SPLIT" ]; then
  echo "Error: No split parameter provided."
  exit 1
fi

echo "Starting vLLM Generation for Split: $SPLIT"

cd /storage/DSH/projects/synthetic-contact-matrices
mkdir -p data/results

export HF_TOKEN="hf_your_new_token_here"
export HF_HOME="/storage/DSH/projects/synthetic-contact-matrices/hf_cache"

python scripts/04_generate.py \
    --adapter_path "models/gemma2-9b-contact-${SPLIT}" \
    --output_file "data/results/generated_contacts_${SPLIT}.jsonl"