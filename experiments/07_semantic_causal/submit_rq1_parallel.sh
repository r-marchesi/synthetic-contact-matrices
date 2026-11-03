#!/bin/bash
set -e

PG_EPOCH=${1:-14}
FF_EPOCH=${2:-14}

SPLITS=("80_percent" "50_percent" "30_percent" "10_percent")
JOB_IDS=()

echo "Submitting 4 parallel Slurm jobs for Exp 07 RQ1 splits (Target Epochs: PG=$PG_EPOCH, FF=$FF_EPOCH)..."

for SPLIT in "${SPLITS[@]}"; do
    JID=$(sbatch --parsable \
        --job-name="sc07_${SPLIT}" \
        experiments/07_semantic_causal/run_single_rq1_split.sh "$SPLIT")
    echo "  -> Submitted $SPLIT : Job ID $JID"
    JOB_IDS+=("$JID")
done

DEP_STR=$(IFS=:; echo "${JOB_IDS[*]}")

EVAL_JID=$(sbatch --parsable \
    --dependency=afterok:${DEP_STR} \
    --job-name="sc07_eval_all" \
    --ntasks=1 \
    --cpus-per-task=8 \
    --partition=h200 \
    --nodelist=euler \
    --gres=gpu:3g.71gb:1 \
    --mem=64G \
    --qos=normal \
    --container-mounts=/storage/DSH/projects/synthetic-contact-matrices/ \
    --container-image=/storage/DSH/projects/synthetic-contact-matrices/image.sqsh \
    --output=/storage/DSH/projects/synthetic-contact-matrices/experiments/07_semantic_causal/logs/%x_%j.out \
    --error=/storage/DSH/projects/synthetic-contact-matrices/experiments/07_semantic_causal/logs/%x_%j.err \
    --mail-type=END,FAIL \
    --mail-user=rmarchesi@fbk.eu \
    --wrap="cd /storage/DSH/projects/synthetic-contact-matrices && source .env && \
            python experiments/07_semantic_causal/scripts/04_evaluate_and_wsmad.py && \
            python experiments/07_semantic_causal/scripts/07_plot_rq1_scaling.py --pg_epoch ${PG_EPOCH} --ff_epoch ${FF_EPOCH} && \
            python experiments/07_semantic_causal/scripts/05_deep_analysis_and_exports.py --pg_epoch ${PG_EPOCH} --ff_epoch ${FF_EPOCH} && \
            python experiments/07_semantic_causal/scripts/06_build_html_notebook.py --pg_epoch ${PG_EPOCH} --ff_epoch ${FF_EPOCH} --skip_refresh")

echo "=================================================================="
echo "✅ All 4 RQ1 split jobs submitted!"
echo "✅ Chained unified evaluation & notebook builder job: Job ID $EVAL_JID"
echo "   (Will run automatically after jobs ${DEP_STR} finish)"
echo "=================================================================="