#!/bin/bash
set -e

cd /storage/DSH/projects/synthetic-contact-matrices
mkdir -p experiments/04_causal_hierarchy/logs

# Usage: bash experiments/04_causal_hierarchy/submit_rq1_parallel.sh [PG_EPOCH] [FF_EPOCH]
PG_EPOCH=${1:-6}
FF_EPOCH=${2:-6}

# Train only up to the highest epoch needed between PG_EPOCH and FF_EPOCH
if [ "$PG_EPOCH" -gt "$FF_EPOCH" ]; then
    MAX_EPOCHS="$PG_EPOCH"
else
    MAX_EPOCHS="$FF_EPOCH"
fi

echo "=================================================="
echo "Launching 4 Parallel RQ1 Scaling Jobs on euler"
echo "  -> Plan-Guided Epoch: $PG_EPOCH"
echo "  -> Full-Free Epoch:   $FF_EPOCH"
echo "  -> Training Epochs:   $MAX_EPOCHS"
echo "=================================================="

SPLITS=("80_percent" "50_percent" "30_percent" "10_percent")
JOB_IDS=()

# Ensure split JSONLs exist before launching parallel jobs
for SPLIT in "${SPLITS[@]}"; do
    if [ ! -f "experiments/04_causal_hierarchy/data/train_causal_${SPLIT}.jsonl" ]; then
        echo "Preparing missing JSONL for $SPLIT..."
        python3 experiments/04_causal_hierarchy/scripts/01_prepare_data.py \
            --train_ids "data/splits/train_ids_${SPLIT}.csv"
    fi
done

# Submit all 4 splits in parallel
for SPLIT in "${SPLITS[@]}"; do
    JID=$(sbatch --parsable \
        --job-name="rq1_${SPLIT}" \
        experiments/04_causal_hierarchy/run_single_rq1_split.sh \
        "$SPLIT" "$PG_EPOCH" "$FF_EPOCH" "$MAX_EPOCHS")
    echo "Submitted $SPLIT -> Slurm Job ID: $JID"
    JOB_IDS+=("$JID")
done

# Build dependency string (afterok:JID1:JID2:JID3:JID4) for automatic final evaluation
DEP_STR=$(IFS=:; echo "${JOB_IDS[*]}")

EVAL_JID=$(sbatch --parsable \
    --dependency=afterok:${DEP_STR} \
    --job-name="rq1_eval_all" \
    --ntasks=1 \
    --cpus-per-task=8 \
    --partition=h200 \
    --nodelist=euler \
    --gres=gpu:3g.71gb:1 \
    --mem=64G \
    --qos=normal \
    --container-mounts=/storage/DSH/projects/synthetic-contact-matrices/ \
    --container-image=/storage/DSH/projects/synthetic-contact-matrices/image.sqsh \
    --output=/storage/DSH/projects/synthetic-contact-matrices/experiments/04_causal_hierarchy/logs/%x_%j.out \
    --error=/storage/DSH/projects/synthetic-contact-matrices/experiments/04_causal_hierarchy/logs/%x_%j.err \
    --wrap="cd /storage/DSH/projects/synthetic-contact-matrices && python experiments/04_causal_hierarchy/scripts/04_evaluate.py")

echo "--------------------------------------------------"
echo "✅ All 4 RQ1 split jobs submitted in parallel!"
echo "✅ Queued automatic final evaluation job -> Job ID: $EVAL_JID (runs after ${DEP_STR})"
echo "=================================================="