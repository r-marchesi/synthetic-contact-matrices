import os
import glob
import re
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon

CONTACT_MAP = {
    'cnt_age_exact': 'Contact Age',
    'setting': 'Contact Setting',
}

def get_age_bin(age_str):
    try:
        return min(14, int(float(age_str) // 5))
    except (ValueError, TypeError):
        return 14

def prep_ground_truth(master_csv, val_ids_csv):
    """Loads GT, maps demographics, and extracts required contact columns."""
    val_ids_df = pd.read_csv(val_ids_csv)
    val_ids_set = set(val_ids_df['part_id'].astype(str))
    
    master_df = pd.read_csv(master_csv, low_memory=False)
    master_df['part_id'] = master_df['part_id'].astype(str)
    gt_df = master_df[master_df['part_id'].isin(val_ids_set)].copy()
    
    # Participant Dictionary: part_id -> age_bin
    demos = gt_df.drop_duplicates(subset=['part_id'])
    val_part_dict = {
        str(row['part_id']): get_age_bin(row['part_age_exact']) 
        for _, row in demos.iterrows()
    }
    
    # Format GT contacts
    valid_contacts = gt_df[gt_df['cont_id'].notna()].copy()
    for raw, clean in CONTACT_MAP.items():
        if raw in valid_contacts.columns:
            valid_contacts[clean] = valid_contacts[raw]
            
    return valid_contacts, val_part_dict

def calculate_matrix_and_volumes(contacts_df, val_part_dict):
    """Calculates the 15x15 contact matrix and participant volumes."""
    matrix = np.zeros((15, 15))
    part_bin_counts = {i: 0 for i in range(15)}
    for p_bin in val_part_dict.values():
        part_bin_counts[p_bin] += 1
        
    volumes = {pid: 0 for pid in val_part_dict.keys()}
    
    if not contacts_df.empty:
        df = contacts_df[contacts_df['Contact Age'].notna() & (contacts_df['Contact Age'] != 'Unknown')]
        for _, row in df.iterrows():
            pid = str(row['part_id'])
            if pid in val_part_dict:
                p_bin = val_part_dict[pid]
                c_bin = get_age_bin(row['Contact Age'])
                matrix[p_bin, c_bin] += 1
                volumes[pid] += 1
                
    # Normalize matrix
    for p_bin in range(15):
        if part_bin_counts[p_bin] > 0:
            matrix[p_bin, :] /= part_bin_counts[p_bin]
            
    return matrix, volumes

def bootstrap_se_matrix(gt_contacts, val_part_dict, n_bootstrap=200):
    """
    Bootstraps the ground truth data at the *participant level* to calculate 
    the Standard Error (SE) for each cell in the 15x15 matrix.
    """
    print(f"Bootstrapping GT matrix (N={n_bootstrap} iterations) to compute Standard Errors...")
    
    # Precompute each participant's contact vector for fast resampling
    part_vectors = {pid: np.zeros(15) for pid in val_part_dict.keys()}
    df = gt_contacts[gt_contacts['Contact Age'].notna() & (gt_contacts['Contact Age'] != 'Unknown')]
    
    for _, row in df.iterrows():
        pid = str(row['part_id'])
        if pid in part_vectors:
            c_bin = get_age_bin(row['Contact Age'])
            part_vectors[pid][c_bin] += 1

    part_ids = list(val_part_dict.keys())
    n_parts = len(part_ids)
    
    boot_matrices = np.zeros((n_bootstrap, 15, 15))
    
    for b in range(n_bootstrap):
        sampled_ids = np.random.choice(part_ids, size=n_parts, replace=True)
        
        # Count sampled participants per age bin
        bin_counts = np.zeros(15)
        for pid in sampled_ids:
            bin_counts[val_part_dict[pid]] += 1
            
        # Sum contact vectors
        for pid in sampled_ids:
            p_bin = val_part_dict[pid]
            boot_matrices[b, p_bin, :] += part_vectors[pid]
            
        # Normalize
        for p_bin in range(15):
            if bin_counts[p_bin] > 0:
                boot_matrices[b, p_bin, :] /= bin_counts[p_bin]
                
    # Calculate Standard Error (standard deviation across bootstrap samples)
    se_matrix = np.std(boot_matrices, axis=0)
    return se_matrix

def calc_l_matrix(gt_matrix, gen_matrix, se_matrix, epsilon=0.01):
    """Primary Metric: Uncertainty-Weighted MSE"""
    squared_diff = (gen_matrix - gt_matrix) ** 2
    variance_penalty = (se_matrix ** 2) + epsilon
    return np.mean(squared_diff / variance_penalty)

def calc_l_contacts(gt_volumes, gen_volumes, val_part_dict):
    """Secondary Metric: Age-conditioned Wasserstein Distance of contact distributions."""
    gt_vols_by_age = {i: [] for i in range(15)}
    gen_vols_by_age = {i: [] for i in range(15)}
    
    for pid, p_bin in val_part_dict.items():
        gt_vols_by_age[p_bin].append(gt_volumes[pid])
        gen_vols_by_age[p_bin].append(gen_volumes[pid])
        
    total_parts = len(val_part_dict)
    l_contacts = 0
    
    for i in range(15):
        w_i = len(gt_vols_by_age[i]) / total_parts
        if w_i > 0:
            w1 = wasserstein_distance(gt_vols_by_age[i], gen_vols_by_age[i])
            l_contacts += w_i * w1
            
    return l_contacts

def calc_l_setting(gt_contacts, gen_contacts, val_part_dict):
    """Tertiary Metric: Age-conditioned Jensen-Shannon Divergence for Settings."""
    # Standardize settings list to align probability vectors
    all_settings = set(gt_contacts['Contact Setting'].dropna().unique())
    all_settings.update(gen_contacts['Contact Setting'].dropna().unique())
    all_settings = sorted([str(s) for s in all_settings if str(s).strip() not in ['', 'Unknown']])
    
    if not all_settings:
        return 0
        
    gt_settings = gt_contacts.merge(pd.Series(val_part_dict, name='p_bin'), left_on='part_id', right_index=True)
    gen_settings = gen_contacts.merge(pd.Series(val_part_dict, name='p_bin'), left_on='part_id', right_index=True)
    
    total_parts = len(val_part_dict)
    l_setting = 0
    
    for i in range(15):
        # Calculate weights based on participant population, NOT contact volume
        w_i = sum(1 for p in val_part_dict.values() if p == i) / total_parts
        if w_i == 0:
            continue
            
        gt_s = gt_settings[gt_settings['p_bin'] == i]['Contact Setting'].astype(str)
        gen_s = gen_settings[gen_settings['p_bin'] == i]['Contact Setting'].astype(str)
        
        gt_counts = gt_s[gt_s.isin(all_settings)].value_counts()
        gen_counts = gen_s[gen_s.isin(all_settings)].value_counts()
        
        p_real = np.array([gt_counts.get(s, 0) for s in all_settings], dtype=float)
        p_syn = np.array([gen_counts.get(s, 0) for s in all_settings], dtype=float)
        
        sum_real = p_real.sum()
        sum_syn = p_syn.sum()
        
        # If both are empty, divergence is 0. If only one is empty, it's a severe mismatch (JSD=1)
        if sum_real == 0 and sum_syn == 0:
            jsd = 0
        elif sum_real == 0 or sum_syn == 0:
            jsd = 1.0  
        else:
            p_real /= sum_real
            p_syn /= sum_syn
            # scipy jensenshannon returns the distance; square it for divergence
            jsd = jensenshannon(p_real, p_syn) ** 2 
            
        l_setting += w_i * jsd
        
    return l_setting

def evaluate_sweep(master_csv, val_ids_csv, parsed_dir, output_csv):
    print("Initializing Formal Metrics Pipeline...")
    gt_contacts, val_part_dict = prep_ground_truth(master_csv, val_ids_csv)
    
    gt_matrix, gt_volumes = calculate_matrix_and_volumes(gt_contacts, val_part_dict)
    se_matrix = bootstrap_se_matrix(gt_contacts, val_part_dict, n_bootstrap=200)
    
    parsed_files = glob.glob(os.path.join(parsed_dir, "*_results.csv"))
    file_pattern = re.compile(r"(.*)_epoch_(\d+)_results\.csv")
    
    results = []
    
    for file_path in parsed_files:
        filename = os.path.basename(file_path)
        match = file_pattern.match(filename)
        if not match:
            continue
            
        prompt_style = match.group(1)
        epoch = int(match.group(2))
        
        gen_contacts = pd.read_csv(file_path, low_memory=False)
        gen_contacts['part_id'] = gen_contacts['part_id'].astype(str)
        
        gen_matrix, gen_volumes = calculate_matrix_and_volumes(gen_contacts, val_part_dict)
        
        l_mat = calc_l_matrix(gt_matrix, gen_matrix, se_matrix)
        l_con = calc_l_contacts(gt_volumes, gen_volumes, val_part_dict)
        l_set = calc_l_setting(gt_contacts, gen_contacts, val_part_dict)
        
        results.append({
            "Experiment": prompt_style,
            "Epoch": epoch,
            "L_matrix": l_mat,
            "L_contacts": l_con,
            "L_setting": l_set
        })
        print(f"Calculated Metrics for {prompt_style} (Epoch {epoch})")

    df_res = pd.DataFrame(results)
    
    # ---------------------------------------------------------
    # COMPOSITE SCORE CALCULATION
    # Standardize (Z-score) metrics so they share the same scale
    # ---------------------------------------------------------
    for col in ['L_matrix', 'L_contacts', 'L_setting']:
        z_col = f"{col}_z"
        # Standard Z-score normalization
        df_res[z_col] = (df_res[col] - df_res[col].mean()) / (df_res[col].std() + 1e-8)
        
    # Apply formal weighting: L = 0.6 * L_mat + 0.25 * L_con + 0.15 * L_set
    df_res['Combined_L_Score'] = (
        0.60 * df_res['L_matrix_z'] + 
        0.25 * df_res['L_contacts_z'] + 
        0.15 * df_res['L_setting_z']
    )
    
    # Sort by the final composite score (Lower is better)
    df_res = df_res.sort_values('Combined_L_Score', ascending=True)
    
    # Clean up columns for the final leaderboard
    cols = ['Experiment', 'Epoch', 'Combined_L_Score', 'L_matrix', 'L_contacts', 'L_setting']
    df_res = df_res[cols]
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_res.to_csv(output_csv, index=False)
    
    print("\n=========================================================================")
    print("🏆 FORMAL EVALUATION LEADERBOARD (Ranked by Combined Score)")
    print("=========================================================================")
    print(df_res.to_string(index=False, float_format="%.4f"))
    print(f"\nLeaderboard saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_csv", type=str, default="data/processed/master_data.csv")
    parser.add_argument("--val_ids", type=str, default="data/splits/val_ids.csv")
    parser.add_argument("--parsed_dir", type=str, default="data/parsed_results")
    parser.add_argument("--output_csv", type=str, default="data/results/formal_evaluation_leaderboard.csv")
    
    args = parser.parse_args()
    evaluate_sweep(args.master_csv, args.val_ids, args.parsed_dir, args.output_csv)