import os
import glob
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- PATH RESOLUTION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "../.."))
DATA_DIR = os.path.join(EXP_DIR, "data")
GEN_DIR = os.path.join(EXP_DIR, "generated")
OUT_DIR = os.path.join(EXP_DIR, "evaluation_plots")

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

def parse_dense_semantic(df_jsonl):
    records = []
    line_parser = re.compile(
        r"- Setting:\s*(.*?)\s*\|\s*Age:\s*(.*?)\s*\|\s*Gender:\s*(.*?)\s*\|\s*Freq:\s*(.*?)\s*\|\s*Dist:\s*(.*?)(?:\n|$)"
    )
    
    for _, row in df_jsonl.iterrows():
        part_id = str(row['part_id'])
        completion = str(row.get('generated_completion', ''))
        
        matches = list(line_parser.finditer(completion))
        
        if not matches:
            records.append({
                "part_id": part_id,
                "Contact Setting": "Unknown",
                "Contact Age": "Unknown",
                "Contact Gender": "Unknown",
                "Contact Frequency": "Unknown",
                "Distance during Contact": "Unknown",
                "valid_contact": False
            })
            continue
            
        for match in matches:
            records.append({
                "part_id": part_id,
                "Contact Setting": match.group(1).strip(),
                "Contact Age": match.group(2).strip(),
                "Contact Gender": match.group(3).strip(),
                "Contact Frequency": match.group(4).strip(),
                "Distance during Contact": match.group(5).strip(),
                "valid_contact": True
            })
            
    return pd.DataFrame(records)

def load_ground_truth(master_csv, val_ids_csv):
    val_ids_df = pd.read_csv(val_ids_csv)
    val_ids_set = set(val_ids_df['part_id'].astype(str))
    
    master_df = pd.read_csv(master_csv, low_memory=False)
    master_df['part_id'] = master_df['part_id'].astype(str)
    gt_df = master_df[master_df['part_id'].isin(val_ids_set)].copy()
    
    demos_df = gt_df.drop_duplicates(subset=['part_id'])[['part_id', 'part_age_exact', 'hh_size', 'part_gender']]
    demos_df['part_age_exact'] = pd.to_numeric(demos_df['part_age_exact'], errors='coerce')
    demos_df['hh_size'] = pd.to_numeric(demos_df['hh_size'], errors='coerce').fillna(1)
    
    valid_contacts = gt_df[gt_df['cont_id'].notna()].copy()
    
    contact_map = {
        'cnt_age_exact': 'Contact Age',
        'cnt_gender': 'Contact Gender',
        'frequency_multi': 'Contact Frequency',
        'distance': 'Distance during Contact',
        'setting': 'Contact Setting'
    }
    
    keep_cols = ['part_id']
    for raw_col, clean_col in contact_map.items():
        if raw_col in valid_contacts.columns:
            valid_contacts[clean_col] = valid_contacts[raw_col]
            keep_cols.append(clean_col)
            
    valid_contacts = valid_contacts[keep_cols]
    valid_contacts['valid_contact'] = True
    
    return valid_contacts, demos_df, val_ids_set

def build_contact_matrix(df, val_part_dict, age_col):
    matrix = np.zeros((15, 15))
    if df.empty:
        return matrix
        
    df = df.copy()
    part_bin_counts = {i: 0 for i in range(15)}
    for _, part_age in val_part_dict.items():
        part_bin_counts[part_age] += 1
        
    df = df[df[age_col].notna() & (df[age_col] != 'Unknown') & (df[age_col] != '')]
    
    for _, row in df.iterrows():
        if not row.get('valid_contact', True):
            continue
            
        p_id = str(row['part_id'])
        if p_id not in val_part_dict:
            continue
            
        p_bin = val_part_dict[p_id]
        c_bin = get_age_bin(row[age_col])
        matrix[p_bin, c_bin] += 1
        
    for p_bin in range(15):
        if part_bin_counts[p_bin] > 0:
            matrix[p_bin, :] /= part_bin_counts[p_bin]
            
    return matrix

def plot_volumes(gt_contacts, gen_contacts, val_ids_set, output_path, epoch):
    """Plots a clean grouped bar chart for number of contacts generated per participant."""
    
    # Filter to only valid contacts (ignoring rows where valid_contact is False)
    gt_valid = gt_contacts[gt_contacts['valid_contact'] == True]
    gen_valid = gen_contacts[gen_contacts['valid_contact'] == True]

    gt_counts = gt_valid.groupby('part_id').size()
    gt_counts_dict = {pid: gt_counts.get(pid, 0) for pid in val_ids_set}
    
    gen_counts = gen_valid.groupby('part_id').size()
    gen_counts_dict = {pid: gen_counts.get(pid, 0) for pid in val_ids_set}
    
    vol_df = pd.DataFrame([
        {"Volume": vol, "Source": "Ground Truth"} for vol in gt_counts_dict.values()
    ] + [
        {"Volume": vol, "Source": "Generated"} for vol in gen_counts_dict.values()
    ])
    
    plt.figure(figsize=(14, 7))
    sns.histplot(
        data=vol_df, x='Volume', hue='Source', multiple='dodge', 
        discrete=True, shrink=0.8, palette=['#1f77b4', '#ff7f0e'], alpha=0.9
    )
    
    plt.title(f"Contact Volume Distribution (Epoch {epoch})", fontsize=16, pad=15)
    plt.xlabel("Number of Contacts per Participant", fontsize=13)
    plt.ylabel("Count of Participants", fontsize=13)
    
    # Cap the x-axis to clip severe outliers and keep the chart readable
    max_vol = int(vol_df['Volume'].quantile(0.99)) + 2 
    plt.xlim(-0.5, max_vol)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_matrices(gt_matrix, gen_matrix, epoch, output_path):
    """Plots Heatmaps with Numerical Annotations inside the cells."""
    diff_matrix = gen_matrix - gt_matrix
    vmax = max(np.max(gt_matrix), np.max(gen_matrix))
    
    # Custom formatter to hide tiny zeros for a cleaner matrix, but show real numbers
    def custom_annot(matrix):
        annot = np.empty_like(matrix, dtype=object)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                annot[i, j] = f"{val:.1f}" if val >= 0.1 else ""
        return annot

    fig, axes = plt.subplots(1, 3, figsize=(26, 8))
    fig.suptitle(f"Contact Matrix Analysis (Epoch {epoch})", fontsize=20, y=1.05)

    sns.heatmap(gt_matrix, ax=axes[0], cmap="YlOrRd", vmin=0, vmax=vmax, 
                xticklabels=AGE_LABELS, yticklabels=AGE_LABELS,
                annot=custom_annot(gt_matrix), fmt="", annot_kws={"size": 8, "alpha": 0.8})
    axes[0].set_title("Ground Truth", fontsize=15, pad=10)
    axes[0].set_ylabel("Participant Age", fontsize=12)
    axes[0].set_xlabel("Contact Age", fontsize=12)

    sns.heatmap(gen_matrix, ax=axes[1], cmap="YlOrRd", vmin=0, vmax=vmax, 
                xticklabels=AGE_LABELS, yticklabels=AGE_LABELS,
                annot=custom_annot(gen_matrix), fmt="", annot_kws={"size": 8, "alpha": 0.8})
    axes[1].set_title("Generated Output", fontsize=15, pad=10)
    axes[1].set_ylabel("Participant Age", fontsize=12)
    axes[1].set_xlabel("Contact Age", fontsize=12)

    max_diff = np.max(np.abs(diff_matrix))
    
    def diff_annot(matrix):
        annot = np.empty_like(matrix, dtype=object)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                annot[i, j] = f"{val:+.1f}" if abs(val) >= 0.1 else ""
        return annot

    sns.heatmap(diff_matrix, ax=axes[2], cmap="coolwarm", center=0, 
                vmin=-max_diff, vmax=max_diff, 
                xticklabels=AGE_LABELS, yticklabels=AGE_LABELS,
                annot=diff_annot(diff_matrix), fmt="", annot_kws={"size": 8, "alpha": 0.8})
    axes[2].set_title("Difference (Generated - GT)", fontsize=15, pad=10)
    axes[2].set_ylabel("Participant Age", fontsize=12)
    axes[2].set_xlabel("Contact Age", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def evaluate_plots(master_csv, val_ids_csv):
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("Loading Ground Truth data...")
    gt_contacts, demos_df, val_ids_set = load_ground_truth(master_csv, val_ids_csv)
    
    val_part_dict = {
        str(row['part_id']): get_age_bin(row['part_age_exact']) 
        for _, row in demos_df.iterrows()
    }
    
    gt_matrix = build_contact_matrix(gt_contacts, val_part_dict, age_col='Contact Age')
    
    gen_files = glob.glob(os.path.join(GEN_DIR, "generated_epoch_*.jsonl"))
    
    for file_path in gen_files:
        filename = os.path.basename(file_path)
        match = re.search(r"epoch_(\d+)", filename)
        if not match:
            continue
            
        epoch = match.group(1)
        print(f"\nProcessing Epoch {epoch}...")
        
        # Load and parse the dense semantic format
        gen_jsonl = pd.read_json(file_path, lines=True)
        gen_contacts = parse_dense_semantic(gen_jsonl)
        
        # Build the matrix
        gen_matrix = build_contact_matrix(gen_contacts, val_part_dict, age_col='Contact Age')
        
        # 1. Plot Matrix with Values
        matrix_out = os.path.join(OUT_DIR, f"matrix_epoch_{epoch}.png")
        plot_matrices(gt_matrix, gen_matrix, epoch, matrix_out)
        print(f"Saved Matrix Plot -> {matrix_out}")
        
        # 2. Plot Volume Distribution
        vol_out = os.path.join(OUT_DIR, f"volume_epoch_{epoch}.png")
        plot_volumes(gt_contacts, gen_contacts, val_ids_set, vol_out, epoch)
        print(f"Saved Volume Plot -> {vol_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_csv", type=str, default=os.path.join(PROJECT_ROOT, "data/processed/master_data.csv"))
    parser.add_argument("--val_ids", type=str, default=os.path.join(PROJECT_ROOT, "data/splits/val_ids.csv"))
    args = parser.parse_args()
    
    evaluate_plots(args.master_csv, args.val_ids)