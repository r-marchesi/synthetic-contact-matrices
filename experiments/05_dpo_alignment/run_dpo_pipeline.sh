#!/bin/bash
#SBATCH --job-name=sc_05_dpo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=h200
#SBATCH --nodelist=euler
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --mem=100G
#SBATCH --qos=normal
#SBATCH --container-mounts=/storage/DSH/projects/synthetic-contact-matrices/
#SBATCH --container-image=/storage/DSH/projects/synthetic-contact-matrices/image.sqsh
#SBATCH --output=/storage/DSH/projects/synthetic-contact-matrices/experiments/05_dpo_alignment/logs/%x_%j.out
#SBATCH --error=/storage/DSH/projects/synthetic-contact-matrices/experiments/05_dpo_alignment/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rmarchesi@fbk.eu

set -e
cd /storage/DSH/projects/synthetic-contact-matrices
mkdir -p experiments/05_dpo_alignment/logs

source .env
export HF_HOME="/storage/DSH/projects/synthetic-contact-matrices/hf_cache"
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$((25000 + SLURM_JOB_ID % 10000))

SFT_EPOCH="6"

echo "=================================================="
echo "Exp 05 - Step 1: Sampling K=4 Candidates per Training Participant"
echo "=================================================="
python experiments/05_dpo_alignment/scripts/01_sample_candidates.py \
    --sft_epoch "$SFT_EPOCH" \
    --mode plan_guided \
    --num_candidates 4 \
    --max_participants 1500 \
    --batch_size 32

echo "=================================================="
echo "Exp 05 - Step 2: Scoring via OOF Training C2ST + Physics Critic"
echo "=================================================="
python experiments/05_dpo_alignment/scripts/02_score_and_pair.py

echo "=================================================="
echo "Exp 05 - Step 3: Running Precomputed-Ref DPO + NLL Alignment"
echo "=================================================="
python experiments/05_dpo_alignment/scripts/03_train_dpo.py \
    --sft_epoch "$SFT_EPOCH" \
    --epochs 2 \
    --batch_size 1 \
    --grad_accum 8 \
    --lr 1e-5 \
    --beta 0.1 \
    --nll_alpha 0.1

echo "=================================================="
echo "Exp 05 - Step 4: Validation Generation & Unified Evaluation"
echo "=================================================="
python experiments/05_dpo_alignment/scripts/04_generate_and_eval.py \
    --dpo_epoch final \
    --mode plan_guided \
    --batch_size 16 \
    --samples -1

echo "=================================================="
echo "Exp 05 DPO Pipeline & Evaluation Completed!"
echo "=================================================="