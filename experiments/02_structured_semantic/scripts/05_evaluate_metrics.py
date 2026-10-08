import os
import glob
import re
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# --- PATH RESOLUTION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "../.."))
DATA_DIR = os.path.join(EXP_DIR, "data")
GEN_DIR = os.path.join(EXP_DIR, "generated")
OUT_DIR = os.path.join(EXP_DIR, "results")

# --- SHARED PARSERS (Duplicated to keep script standalone) ---
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

# --- CORE METRICS CALCULATION ---
def calculate_reciprocity_error(matrix):
    """
    Calculates the Mean Absolute Error between the upper and lower triangles 
    of the contact matrix to measure symmetry (a key law of epidemiology).
    """
    if matrix.sum() == 0:
        return float('nan')
        
    upper = np.triu(matrix, k=1)
    lower = np.tril(matrix, k=-1).T
    
    # We only calculate error on non-zero entries to avoid rewarding sparse matrices
    mask = (upper > 0) | (lower > 0)
    if not np.any(mask):
        return 0.0
        
    mae = np.mean(np.abs(upper[mask] - lower[mask]))
    return mae

def calculate_c2st(gt_contacts, gen_contacts):
    """
    Trains an XGBoost classifier to distinguish GT from Generated data.
    Returns the ROC-AUC score.
    """
    features = [
        'Contact Setting', 'Contact Age', 'Contact Gender', 
        'Contact Frequency', 'Distance during Contact'
    ]
    
    gt = gt_contacts[features].copy()
    gen = gen_contacts[features].copy()
    
    gt['is_synthetic'] = 0
    gen['is_synthetic'] = 1
    
    combined = pd.concat([gt, gen], ignore_index=True)
    
    # FIX 1: Force Age to be purely numeric so '34.0' and '34' evaluate identically
    combined['Contact Age'] = pd.to_numeric(combined['Contact Age'], errors='coerce')
    
    # FIX 2: Normalize strings (lowercase, strip) to prevent trivial whitespace splits
    cat_cols = ['Contact Setting', 'Contact Gender', 'Contact Frequency', 'Distance during Contact']
    for col in cat_cols:
        combined[col] = combined[col].astype(str).str.strip().str.lower()
        # Unify all missing value representations
        combined[col] = combined[col].replace(['nan', '<na>', 'none', ''], 'unknown')
        
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col])
        
    X = combined[features]
    y = combined['is_synthetic']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    clf = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=4,
        learning_rate=0.1,
        eval_metric='auc',
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    
    return auc

def run_evaluation(master_csv, val_ids_csv):
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("Loading Ground Truth data...")
    gt_contacts, demos_df, val_ids_set = load_ground_truth(master_csv, val_ids_csv)
    
    val_part_dict = {
        str(row['part_id']): get_age_bin(row['part_age_exact']) 
        for _, row in demos_df.iterrows()
    }
    
    gt_valid = gt_contacts[gt_contacts['valid_contact'] == True]
    gt_matrix = build_contact_matrix(gt_valid, val_part_dict, age_col='Contact Age')
    gt_reciprocity = calculate_reciprocity_error(gt_matrix)
    
    print(f"Ground Truth Reciprocity Error: {gt_reciprocity:.4f}")
    
    gen_files = glob.glob(os.path.join(GEN_DIR, "generated_epoch_*.jsonl"))
    
    results = []
    
    for file_path in gen_files:
        filename = os.path.basename(file_path)
        match = re.search(r"epoch_(\d+)", filename)
        if not match:
            continue
            
        epoch = int(match.group(1))
        print(f"\nEvaluating Epoch {epoch}...")
        
        gen_jsonl = pd.read_json(file_path, lines=True)
        gen_contacts = parse_dense_semantic(gen_jsonl)
        gen_valid = gen_contacts[gen_contacts['valid_contact'] == True]
        
        # 1. Physics: Reciprocity
        gen_matrix = build_contact_matrix(gen_valid, val_part_dict, age_col='Contact Age')
        reciprocity = calculate_reciprocity_error(gen_matrix)
        
        # 2. Realism: C2ST AUC
        c2st_auc = calculate_c2st(gt_valid, gen_valid)
        
        # 3. Validity: Syntax success rate
        valid_pct = (len(gen_valid) / max(1, len(gen_contacts))) * 100
        
        print(f"  -> Reciprocity Error: {reciprocity:.4f}")
        print(f"  -> C2ST AUC: {c2st_auc:.4f} (Closer to 0.5 is better)")
        print(f"  -> Valid Syntax: {valid_pct:.1f}%")
        
        results.append({
            "Epoch": epoch,
            "C2ST_AUC": c2st_auc,
            "Reciprocity_Error": reciprocity,
            "Valid_Syntax_Pct": valid_pct
        })

    results_df = pd.DataFrame(results).sort_values("Epoch")
    out_csv = os.path.join(OUT_DIR, "evaluation_metrics.csv")
    results_df.to_csv(out_csv, index=False)
    
    print("\n========================================")
    print("FINAL EVALUATION SUMMARY")
    print("========================================")
    print(results_df.to_string(index=False))
    print(f"\nMetrics saved to: {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_csv", type=str, default=os.path.join(PROJECT_ROOT, "data/processed/master_data.csv"))
    parser.add_argument("--val_ids", type=str, default=os.path.join(PROJECT_ROOT, "data/splits/val_ids.csv"))
    args = parser.parse_args()
    
    run_evaluation(args.master_csv, args.val_ids)