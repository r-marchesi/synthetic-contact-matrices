import os
import glob
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Standard epidemiological 5-year age bins
AGE_LABELS = [
    "0-4", "5-9", "10-14", "15-19", "20-24", "25-29", 
    "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", 
    "60-64", "65-69", "70+"
]

def get_age_bin(age_str):
    try:
        val = float(age_str)
        return min(14, int(val // 5))
    except (ValueError, TypeError):
        return 14

def build_contact_matrix(df, valid_part_ids, age_col):
    matrix = np.zeros((15, 15))
    if df.empty:
        return matrix
        
    df = df.copy()
    part_bin_counts = {i: 0 for i in range(15)}
    for _, part_age in valid_part_ids.items():
        part_bin_counts[part_age] += 1
        
    df = df[df[age_col].notna() & (df[age_col] != 'Unknown') & (df[age_col] != '')]
    
    for _, row in df.iterrows():
        p_id = str(row['part_id'])
        if p_id not in valid_part_ids:
            continue
            
        p_bin = valid_part_ids[p_id]
        c_bin = get_age_bin(row[age_col])
        matrix[p_bin, c_bin] += 1
        
    for p_bin in range(15):
        if part_bin_counts[p_bin] > 0:
            matrix[p_bin, :] /= part_bin_counts[p_bin]
            
    return matrix

def plot_matrices(gt_matrix, gen_matrix, exp_name, epoch, output_path):
    """Plots Ground Truth, Generated, and Difference matrices side-by-side."""
    diff_matrix = gen_matrix - gt_matrix
    
    # Establish a shared color scale for GT and Gen so they are visually comparable
    vmax = max(np.max(gt_matrix), np.max(gen_matrix))
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle(f"Contact Matrix Analysis: {exp_name} (Epoch {epoch})", fontsize=18, y=1.05)

    # 1. Ground Truth Heatmap
    sns.heatmap(gt_matrix, ax=axes[0], cmap="YlOrRd", vmin=0, vmax=vmax, 
                xticklabels=AGE_LABELS, yticklabels=AGE_LABELS)
    axes[0].set_title("Ground Truth", fontsize=14)
    axes[0].set_ylabel("Participant Age", fontsize=12)
    axes[0].set_xlabel("Contact Age", fontsize=12)

    # 2. Generated Heatmap
    sns.heatmap(gen_matrix, ax=axes[1], cmap="YlOrRd", vmin=0, vmax=vmax, 
                xticklabels=AGE_LABELS, yticklabels=AGE_LABELS)
    axes[1].set_title("Generated Output", fontsize=14)
    axes[1].set_ylabel("Participant Age", fontsize=12)
    axes[1].set_xlabel("Contact Age", fontsize=12)

    # 3. Difference Heatmap (Diverging Colormap)
    # Blue = Under-predicted, Red = Over-predicted, White = Perfect match
    max_diff = np.max(np.abs(diff_matrix))
    sns.heatmap(diff_matrix, ax=axes[2], cmap="coolwarm", center=0, 
                vmin=-max_diff, vmax=max_diff, 
                xticklabels=AGE_LABELS, yticklabels=AGE_LABELS)
    axes[2].set_title("Difference (Generated - GT)", fontsize=14)
    axes[2].set_ylabel("Participant Age", fontsize=12)
    axes[2].set_xlabel("Contact Age", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def generate_plots(master_csv, val_ids_csv, parsed_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading Ground Truth validation data...")
    val_ids_df = pd.read_csv(val_ids_csv)
    val_ids_set = set(val_ids_df['part_id'].astype(str))
    
    master_df = pd.read_csv(master_csv, low_memory=False)
    master_df['part_id'] = master_df['part_id'].astype(str)
    
    gt_df = master_df[master_df['part_id'].isin(val_ids_set)].copy()
    
    gt_part_ages = gt_df.drop_duplicates(subset=['part_id'])[['part_id', 'part_age_exact']]
    val_part_dict = {
        str(row['part_id']): get_age_bin(row['part_age_exact']) 
        for _, row in gt_part_ages.iterrows()
    }
    
    gt_matrix = build_contact_matrix(gt_df, val_part_dict, age_col='cnt_age_exact')
    
    parsed_files = glob.glob(os.path.join(parsed_dir, "*_results.csv"))
    file_pattern = re.compile(r"(.*)_epoch_(\d+)_results\.csv")
    
    for file_path in parsed_files:
        filename = os.path.basename(file_path)
        match = file_pattern.match(filename)
        if not match:
            continue
            
        prompt_style = match.group(1)
        epoch = int(match.group(2))
        
        gen_df = pd.read_csv(file_path, low_memory=False)
        gen_df['part_id'] = gen_df['part_id'].astype(str)
        
        gen_matrix = build_contact_matrix(gen_df, val_part_dict, age_col='Contact Age')
        
        output_filename = f"{prompt_style}_epoch_{epoch}_matrix.png"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"Generating plot: {output_filename}...")
        plot_matrices(gt_matrix, gen_matrix, prompt_style, epoch, output_path)
        
    print(f"\nAll plots saved successfully to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_csv", type=str, default="data/processed/master_data.csv")
    parser.add_argument("--val_ids", type=str, default="data/splits/val_ids.csv")
    parser.add_argument("--parsed_dir", type=str, default="data/parsed_results")
    parser.add_argument("--output_dir", type=str, default="data/results/plots")
    
    args = parser.parse_args()
    generate_plots(args.master_csv, args.val_ids, args.parsed_dir, args.output_dir)