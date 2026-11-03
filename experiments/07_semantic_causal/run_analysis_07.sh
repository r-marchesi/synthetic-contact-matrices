#!/bin/bash
#SBATCH --job-name=sc_07_analysis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=h200
#SBATCH --nodelist=euler
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --mem=100G
#SBATCH --qos=normal
#SBATCH --container-mounts=/storage/DSH/projects/synthetic-contact-matrices/
#SBATCH --container-image=/storage/DSH/projects/synthetic-contact-matrices/image.sqsh
#SBATCH --output=/storage/DSH/projects/synthetic-contact-matrices/experiments/07_semantic_causal/logs/%x_%j.out
#SBATCH --error=/storage/DSH/projects/synthetic-contact-matrices/experiments/07_semantic_causal/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rmarchesi@fbk.eu

set -e
cd /storage/DSH/projects/synthetic-contact-matrices



source .env
export HF_HOME="/storage/DSH/projects/synthetic-contact-matrices/hf_cache"
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$((25000 + SLURM_JOB_ID % 10000))

echo "=================================================="
echo "Starting Exp 07: post training"
echo "Job ID: $SLURM_JOB_ID"
echo "=================================================="


# 3. Run Deep Analysis, Overfitting Audit & Export 19-Column Shareable CSVs
echo "--------------------------------------------------"
echo "Step 2/3: Running 05_deep_analysis_and_exports.py..."
echo "--------------------------------------------------"
python experiments/07_semantic_causal/scripts/05_deep_analysis_and_exports.py

# 4. Build Self-Contained HTML Notebook & Markdown Report
echo "--------------------------------------------------"
echo "Step 3/3: Running 06_build_html_notebook.py..."
echo "--------------------------------------------------"
python experiments/07_semantic_causal/scripts/06_build_html_notebook.py --skip_refresh

echo "=================================================="
echo "✅ Exp 07 Analysis Complete!"
echo "   -> HTML Notebook: experiments/07_semantic_causal/results/notebook_exp07.html"
echo "   -> Markdown Report: experiments/07_semantic_causal/results/DEEP_ANALYSIS_REPORT.md"
echo "   -> Shareable CSVs: experiments/07_semantic_causal/results/shareable_csvs/"
echo "=================================================="
