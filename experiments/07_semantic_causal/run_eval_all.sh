#!/bin/bash
#SBATCH --job-name=sc07_eval_all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=h200
#SBATCH --nodelist=euler
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --mem=64G
#SBATCH --qos=normal
#SBATCH --container-mounts=/storage/DSH/projects/synthetic-contact-matrices/
#SBATCH --container-image=/storage/DSH/projects/synthetic-contact-matrices/image.sqsh
#SBATCH --output=/storage/DSH/projects/synthetic-contact-matrices/experiments/07_semantic_causal/logs/%x_%j.out
#SBATCH --error=/storage/DSH/projects/synthetic-contact-matrices/experiments/07_semantic_causal/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rmarchesi@fbk.eu

set -e

PG_EPOCH=${1:-14}
FF_EPOCH=${2:-14}

cd /storage/DSH/projects/synthetic-contact-matrices
mkdir -p experiments/07_semantic_causal/logs
mkdir -p experiments/07_semantic_causal/results

source .env
export HF_HOME="/storage/DSH/projects/synthetic-contact-matrices/hf_cache"

echo "=================================================================="
echo "Starting Exp 07 Unified Evaluation (RQ1 + RQ2 + WSMAD + Notebook)"
echo "Target Epochs -> Plan-Guided: Ep ${PG_EPOCH} | Full-Free: Ep ${FF_EPOCH}"
echo "Job ID: $SLURM_JOB_ID"
echo "=================================================================="

# 1. Evaluate all generated splits & checkpoints (Standard Metrics + WSMAD)
echo "------------------------------------------------------------------"
echo "Step 1/4: Running 04_evaluate_and_wsmad.py..."
echo "------------------------------------------------------------------"
python experiments/07_semantic_causal/scripts/04_evaluate_and_wsmad.py

# 2. Plot RQ1 Scaling Laws (10%, 30%, 50%, 80%, 100%)
echo "------------------------------------------------------------------"
echo "Step 2/4: Running 07_plot_rq1_scaling.py..."
echo "------------------------------------------------------------------"
python experiments/07_semantic_causal/scripts/07_plot_rq1_scaling.py \
    --pg_epoch "${PG_EPOCH}" \
    --ff_epoch "${FF_EPOCH}"

# 3. Run Deep Analysis (Volumes, Matrices, RQ2 Subgroups, C2ST, Overfitting, Shareable CSVs)
echo "------------------------------------------------------------------"
echo "Step 3/4: Running 05_deep_analysis_and_exports.py..."
echo "------------------------------------------------------------------"
python experiments/07_semantic_causal/scripts/05_deep_analysis_and_exports.py \
    --pg_epoch "${PG_EPOCH}" \
    --ff_epoch "${FF_EPOCH}"

# 4. Build Self-Contained HTML Computational Notebook & Markdown Report
echo "------------------------------------------------------------------"
echo "Step 4/4: Running 06_build_html_notebook.py..."
echo "------------------------------------------------------------------"
python experiments/07_semantic_causal/scripts/06_build_html_notebook.py \
    --pg_epoch "${PG_EPOCH}" \
    --ff_epoch "${FF_EPOCH}" \
    --skip_refresh

echo "=================================================================="
echo "✅ Exp 07 Full Evaluation Complete!"
echo "   -> Leaderboard:   experiments/07_semantic_causal/results/leaderboard_07_semantic.csv"
echo "   -> RQ1 Summary:   experiments/07_semantic_causal/results/rq1_scaling/rq1_scaling_summary.csv"
echo "   -> HTML Notebook: experiments/07_semantic_causal/results/notebook_exp07.html"
echo "   -> MD Report:     experiments/07_semantic_causal/results/DEEP_ANALYSIS_REPORT.md"
echo "   -> Shareable CSV: experiments/07_semantic_causal/results/shareable_csvs/"
echo "=================================================================="
