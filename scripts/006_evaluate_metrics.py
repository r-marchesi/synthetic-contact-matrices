import os
import glob
import re
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

def get_age_bin(age_str):
    """Maps raw age strings to the 15 standard demographic 5-year age bins."""
    try:
        val = float(age_str)
        return min(14, int(val // 5))
    except (ValueError, TypeError):
        return 14  # Map 'Unknown' or invalid to the oldest bin

def build_contact_matrix(df, valid_part_ids, age_col):
    """
    Constructs a 15x15 contact matrix C_ij representing the average number of 
    contacts in age bin j for a participant in age bin i.
    """
    matrix = np.zeros((15, 15))
    
    if df.empty:
        return matrix
        
    df = df.copy()
    
    # Calculate total participants per age bin for normalization
    part_bin_counts = {i: 0 for i in range(15)}
    for _, part_age in valid_part_ids.items():
        part_bin_counts[part_age] += 1
        
    # Drop rows where contact is unknown or missing
    df = df[df[age_col].notna() & (df[age_col] != 'Unknown') & (df[age_col] != '')]
    
    for _, row in df.iterrows():
        p_id = str(row['part_id'])
        if p_id not in valid_part_ids:
            continue
            
        p_bin = valid_part_ids[p_id]
        c_bin = get_age_bin(row[age_col])
        matrix[p_bin, c_bin] += 1
        
    # Normalize by the number of participants in each bin to get the average
    for p_bin in range(15):
        if part_bin_counts[p_bin] > 0:
            matrix[p_bin, :] /= part_bin_counts[p_bin]
            
    return matrix

def get_contact_volumes(df, valid_part_ids, filter_col):
    """Returns an array of total contacts per participant to compute the Wasserstein distance."""
    if df.empty:
        return np.zeros(len(valid_part_ids))
        
    # Filter out placeholder rows for participants with 0 contacts
    valid_contacts = df[df[filter_col].notna() & (df[filter_col] != 'Unknown') & (df[filter_col] != '')]
    counts = valid_contacts.groupby('part_id').size().to_dict()
    
    volumes = [counts.get(p_id, 0) for p_id in valid_part_ids.keys()]
    return np.array(volumes)

def evaluate_sweep(master_csv, val_ids_csv, parsed_dir, output_csv):
    print("Loading Ground Truth validation data...")
    val_ids_df = pd.read_csv(val_ids_csv)
    val_ids_set = set(val_ids_df['part_id'].astype(str))
    
    master_df = pd.read_csv(master_csv, low_memory=False)
    master_df['part_id'] = master_df['part_id'].astype(str)
    
    gt_df = master_df[master_df['part_id'].isin(val_ids_set)].copy()
    
    # Map each validation participant to their age bin using the RAW MixIT column name
    gt_part_ages = gt_df.drop_duplicates(subset=['part_id'])[['part_id', 'part_age_exact']]
    val_part_dict = {
        str(row['part_id']): get_age_bin(row['part_age_exact']) 
        for _, row in gt_part_ages.iterrows()
    }
    
    # 1. Compute Ground Truth Targets using RAW MixIT column names
    gt_matrix = build_contact_matrix(gt_df, val_part_dict, age_col='cnt_age_exact')
    gt_volumes = get_contact_volumes(gt_df, val_part_dict, filter_col='cont_id')
    
    print(f"Ground truth established for {len(val_part_dict)} participants.")
    
    # 2. Iterate and evaluate all sweep files
    results = []
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
        
        # Calculate Generated Matrix and Volumes using NATURAL LANGUAGE column names
        gen_matrix = build_contact_matrix(gen_df, val_part_dict, age_col='Contact Age')
        gen_volumes = get_contact_volumes(gen_df, val_part_dict, filter_col='Contact Age')
        
        # Metric 1: L_matrix (Mean Squared Error)
        l_matrix = np.mean((gt_matrix - gen_matrix) ** 2)
        
        # Metric 2: L_contacts (Wasserstein Distance)
        l_contacts = wasserstein_distance(gt_volumes, gen_volumes)
        
        results.append({
            "Experiment": prompt_style,
            "Epoch": epoch,
            "L_matrix (MSE)": l_matrix,
            "L_contacts (Wasserstein)": l_contacts,
            "Total Generated Contacts": len(gen_df)
        })
        print(f"Evaluated {prompt_style} (Epoch {epoch})")

    if not results:
        print("No valid parsed sweep files found.")
        return

    # 3. Create Leaderboard and save
    leaderboard = pd.DataFrame(results)
    leaderboard = leaderboard.sort_values(by=["L_matrix (MSE)", "L_contacts (Wasserstein)"], ascending=True)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    leaderboard.to_csv(output_csv, index=False)
    
    print("\n==================================================")
    print("🏆 SWEEP LEADERBOARD (Ranked by Matrix Accuracy)")
    print("==================================================")
    print(leaderboard.to_string(index=False))
    print(f"\nLeaderboard saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_csv", type=str, default="data/processed/master_data.csv")
    parser.add_argument("--val_ids", type=str, default="data/splits/val_ids.csv")
    parser.add_argument("--parsed_dir", type=str, default="data/parsed_results")
    parser.add_argument("--output_csv", type=str, default="data/results/evaluation_leaderboard.csv")
    
    args = parser.parse_args()
    evaluate_sweep(args.master_csv, args.val_ids, args.parsed_dir, args.output_csv)