import os
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
        # Handles both 'rag_completion' from Exp 03 and 'generated_completion' from Exp 02
        completion = str(row.get('rag_completion', row.get('generated_completion', '')))
        
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

def load_ground_truth(master_csv, eval_part_ids):
    master_df = pd.read_csv(master_csv, low_memory=False)
    master_df['part_id'] = master_df['part_id'].astype(str)
    
    # Restrict ground truth strictly to the participants present in the generated output
    gt_df = master_df[master_df['part_id'].isin(eval_part_ids)].copy()
    
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
    
    return valid_contacts, demos_df

def build_contact_matrix(df, val_part_dict, age_col):
    matrix = np.zeros((15, 15))
    if df.empty:
        return matrix
        
    df = df.copy()
    part_bin_counts = {i: 0 for i in range(15)}
    for _, part_bin in val_part_dict.items():
        part_bin_counts[part_bin] += 1
        
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

def plot_volumes(gt_contacts, gen_contacts, eval_ids_set, output_path, label="RAG"):
    gt_valid = gt_contacts[gt_contacts['valid_contact'] == True]
    gen_valid = gen_contacts[gen_contacts['valid_contact'] == True]

    gt_counts = gt_valid.groupby('part_id').size()
    gt_counts_dict = {pid: gt_counts.get(pid, 0) for pid in eval_ids_set}
    
    gen_counts = gen_valid.groupby('part_id').size()
    gen_counts_dict = {pid: gen_counts.get(pid, 0) for pid in eval_ids_set}
    
    vol_df = pd.DataFrame([
        {"Volume": vol, "Source": "Ground Truth"} for vol in gt_counts_dict.values()
    ] + [
        {"Volume": vol, "Source": f"Generated ({label})"} for vol in gen_counts_dict.values()
    ])
    
    plt.figure(figsize=(14, 7))
    sns.histplot(
        data=vol_df, x='Volume', hue='Source', multiple='dodge', 
        discrete=True, shrink=0.8, palette=['#1f77b4', '#ff7f0e'], alpha=0.9
    )
    
    plt.title(f"Contact Volume Distribution ({label})", fontsize=16, pad=15)
    plt.xlabel("Number of Contacts per Participant", fontsize=13)
    plt.ylabel("Count of Participants", fontsize=13)
    
    max_vol = int(vol_df['Volume'].quantile(0.99)) + 2 
    plt.xlim(-0.5, max_vol)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_matrices(gt_matrix, gen_matrix, label, output_path):
    diff_matrix = gen_matrix - gt_matrix
    vmax = max(np.max(gt_matrix), np.max(gen_matrix), 0.1)
    
    def custom_annot(matrix):
        annot = np.empty_like(matrix, dtype=object)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                annot[i, j] = f"{val:.1f}" if val >= 0.1 else ""
        return annot

    fig, axes = plt.subplots(1, 3, figsize=(26, 8))
    fig.suptitle(f"Contact Matrix Analysis ({label})", fontsize=20, y=1.05)

    sns.heatmap(gt_matrix, ax=axes[0], cmap="YlOrRd", vmin=0, vmax=vmax, 
                xticklabels=AGE_LABELS, yticklabels=AGE_LABELS,
                annot=custom_annot(gt_matrix), fmt="", annot_kws={"size": 8, "alpha": 0.8})
    axes[0].set_title("Ground Truth", fontsize=15, pad=10)
    axes[0].set_ylabel("Participant Age", fontsize=12)
    axes[0].set_xlabel("Contact Age", fontsize=12)

    sns.heatmap(gen_matrix, ax=axes[1], cmap="YlOrRd", vmin=0, vmax=vmax, 
                xticklabels=AGE_LABELS, yticklabels=AGE_LABELS,
                annot=custom_annot(gen_matrix), fmt="", annot_kws={"size": 8, "alpha": 0.8})
    axes[1].set_title(f"Generated ({label})", fontsize=15, pad=10)
    axes[1].set_ylabel("Participant Age", fontsize=12)
    axes[1].set_xlabel("Contact Age", fontsize=12)

    max_diff = max(np.max(np.abs(diff_matrix)), 0.1)
    
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

def evaluate_rag_plots(gen_file, master_csv):
    os.makedirs(OUT_DIR, exist_ok=True)
    
    if not os.path.exists(gen_file):
        raise FileNotFoundError(f"Generated output not found at: {gen_file}")
        
    print(f"Loading generated data from: {gen_file}")
    gen_jsonl = pd.read_json(gen_file, lines=True)
    eval_ids_set = set(gen_jsonl['part_id'].astype(str))
    print(f"Found {len(eval_ids_set)} participants in generated data.")
    
    print("Loading Ground Truth matching the evaluated participants...")
    gt_contacts, demos_df = load_ground_truth(master_csv, eval_ids_set)
    
    val_part_dict = {
        str(row['part_id']): get_age_bin(row['part_age_exact']) 
        for _, row in demos_df.iterrows()
    }
    
    gt_matrix = build_contact_matrix(gt_contacts, val_part_dict, age_col='Contact Age')
    
    print("Parsing generated contacts...")
    gen_contacts = parse_dense_semantic(gen_jsonl)
    gen_matrix = build_contact_matrix(gen_contacts, val_part_dict, age_col='Contact Age')
    
    # 1. Contact Matrix Comparison
    matrix_out = os.path.join(OUT_DIR, "matrix_rag.png")
    plot_matrices(gt_matrix, gen_matrix, label="Exp 03 RAG", output_path=matrix_out)
    print(f"Saved Matrix Plot -> {matrix_out}")
    
    # 2. Volume Distribution Comparison
    vol_out = os.path.join(OUT_DIR, "volume_rag.png")
    plot_volumes(gt_contacts, gen_contacts, eval_ids_set, output_path=vol_out, label="Exp 03 RAG")
    print(f"Saved Volume Plot -> {vol_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gen_file", 
        type=str, 
        default=os.path.join(GEN_DIR, "rag_generated_contacts.jsonl"),
        help="Path to generated JSONL file"
    )
    parser.add_argument(
        "--master_csv", 
        type=str, 
        default=os.path.join(PROJECT_ROOT, "data/processed/master_data.csv"),
        help="Path to processed master_data.csv"
    )
    args = parser.parse_args()
    
    evaluate_rag_plots(args.gen_file, args.master_csv)