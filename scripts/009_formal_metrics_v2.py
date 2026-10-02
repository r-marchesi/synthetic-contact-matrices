import os
import glob
import re
import argparse
import numpy as np
import pandas as pd

def get_age_bin(age_str):
    try:
        return min(14, int(float(age_str) // 5))
    except (ValueError, TypeError):
        return 14

def prep_ground_truth(master_csv, val_ids_csv):
    val_ids_df = pd.read_csv(val_ids_csv)
    val_ids_set = set(val_ids_df['part_id'].astype(str))
    
    master_df = pd.read_csv(master_csv, low_memory=False)
    master_df['part_id'] = master_df['part_id'].astype(str)
    gt_df = master_df[master_df['part_id'].isin(val_ids_set)].copy()
    
    demos = gt_df.drop_duplicates(subset=['part_id'])
    val_part_dict = {
        str(row['part_id']): get_age_bin(row['part_age_exact']) 
        for _, row in demos.iterrows()
    }
    
    # Get population vector N_i for reciprocity calculations
    N_vector = np.zeros(15)
    for p_bin in val_part_dict.values():
        N_vector[p_bin] += 1
        
    valid_contacts = gt_df[gt_df['cont_id'].notna()].copy()
    if 'cnt_age_exact' in valid_contacts.columns:
        valid_contacts['Contact Age'] = valid_contacts['cnt_age_exact']
    if 'setting' in valid_contacts.columns:
        valid_contacts['Contact Setting'] = valid_contacts['setting']
            
    return valid_contacts, val_part_dict, N_vector

def build_matrix(df, val_part_dict, setting_filter=None):
    """Builds a 15x15 contact matrix, optionally filtered by setting."""
    matrix = np.zeros((15, 15))
    part_bin_counts = {i: 0 for i in range(15)}
    for p_bin in val_part_dict.values():
        part_bin_counts[p_bin] += 1
        
    if df.empty:
        return matrix
        
    filtered_df = df[df['Contact Age'].notna() & (df['Contact Age'] != 'Unknown')].copy()
    
    if setting_filter:
        filtered_df = filtered_df[filtered_df['Contact Setting'].astype(str).str.contains(setting_filter, case=False, na=False)]
        
    for _, row in filtered_df.iterrows():
        pid = str(row['part_id'])
        if pid in val_part_dict:
            p_bin = val_part_dict[pid]
            c_bin = get_age_bin(row['Contact Age'])
            matrix[p_bin, c_bin] += 1
            
    for p_bin in range(15):
        if part_bin_counts[p_bin] > 0:
            matrix[p_bin, :] /= part_bin_counts[p_bin]
            
    return matrix

def calc_tvd_volume(gt_contacts, gen_contacts, val_part_dict):
    """Calculates Total Variation Distance (TVD) for contact volumes."""
    def get_distribution(df):
        counts = [0] * len(val_part_dict)
        if not df.empty:
            vol_series = df.groupby('part_id').size()
            counts = [vol_series.get(pid, 0) for pid in val_part_dict.keys()]
        
        # Cap outliers at 50 to prevent unbounded arrays
        capped_counts = np.clip(counts, 0, 50)
        hist, _ = np.histogram(capped_counts, bins=np.arange(52))
        return hist / np.sum(hist) # Return probability distribution
        
    p_gt = get_distribution(gt_contacts)
    p_gen = get_distribution(gen_contacts)
    
    # TVD formula: 0.5 * sum(|P - Q|)
    return 0.5 * np.sum(np.abs(p_gt - p_gen))

def calc_reciprocity_error(matrix, N_vector):
    """Calculates physical symmetry: C_ij * N_i ≈ C_ji * N_j"""
    pop_matrix = np.zeros((15, 15))
    for i in range(15):
        for j in range(15):
            pop_matrix[i, j] = matrix[i, j] * N_vector[i]
            
    asymmetry = np.abs(pop_matrix - pop_matrix.T)
    # Average absolute difference in total reciprocal contacts
    return np.mean(asymmetry)

def evaluate_sweep(master_csv, val_ids_csv, parsed_dir, output_csv):
    print("Initializing Interpretable Metrics Pipeline...")
    gt_contacts, val_part_dict, N_vector = prep_ground_truth(master_csv, val_ids_csv)
    
    # Pre-calculate Ground Truth matrices
    gt_mat_total = build_matrix(gt_contacts, val_part_dict)
    gt_mat_home = build_matrix(gt_contacts, val_part_dict, setting_filter="Home")
    gt_mat_work = build_matrix(gt_contacts, val_part_dict, setting_filter="Work")
    
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
        
        gen_mat_total = build_matrix(gen_contacts, val_part_dict)
        gen_mat_home = build_matrix(gen_contacts, val_part_dict, setting_filter="Home")
        gen_mat_work = build_matrix(gen_contacts, val_part_dict, setting_filter="Work")
        
        # 1. Matrix MAE (Absolute errors)
        mae_total = np.mean(np.abs(gt_mat_total - gen_mat_total))
        mae_home = np.mean(np.abs(gt_mat_home - gen_mat_home))
        mae_work = np.mean(np.abs(gt_mat_work - gen_mat_work))
        
        # 2. Volume TVD (Bounded 0 to 1)
        tvd_vol = calc_tvd_volume(gt_contacts, gen_contacts, val_part_dict)
        
        # 3. Reciprocity (Symmetry Check)
        reciprocity = calc_reciprocity_error(gen_mat_total, N_vector)
        
        results.append({
            "Experiment": prompt_style,
            "Epoch": epoch,
            "MAE_Total": mae_total,
            "MAE_Home": mae_home,
            "MAE_Work": mae_work,
            "Volume_TVD": tvd_vol,
            "Reciprocity_Error": reciprocity
        })
        print(f"Calculated Metrics for {prompt_style} (Epoch {epoch})")

    df_res = pd.DataFrame(results)
    
    # Sort strictly by MAE Total (the most important epidemiological metric)
    df_res = df_res.sort_values('MAE_Total', ascending=True)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_res.to_csv(output_csv, index=False)
    
    print("\n=========================================================================================")
    print("🏆 FORMAL EVALUATION (Ranked by Total Matrix Mean Absolute Error)")
    print("=========================================================================================")
    print(df_res.to_string(index=False, float_format="%.4f"))
    print(f"\nLeaderboard saved to {output_csv}")
    print("\nHow to interpret:")
    print("- MAE_Total: Average error in contacts/day for any cell. Closer to 0 is better.")
    print("- Volume_TVD: % of participants whose contact count is misaligned. 0.15 = 15% error.")
    print("- Reciprocity: Absolute difference in C_ij*N_i vs C_ji*N_j. Closer to 0 means physically logical.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_csv", type=str, default="data/processed/master_data.csv")
    parser.add_argument("--val_ids", type=str, default="data/splits/val_ids.csv")
    parser.add_argument("--parsed_dir", type=str, default="data/parsed_results")
    parser.add_argument("--output_csv", type=str, default="data/results/interpretable_leaderboard.csv")
    
    args = parser.parse_args()
    evaluate_sweep(args.master_csv, args.val_ids, args.parsed_dir, args.output_csv)