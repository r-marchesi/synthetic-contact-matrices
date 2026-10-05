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

# Share the exact same parsers from the plotting script
from evaluate_physics import (
    load_ground_truth, 
    parse_dense_semantic, 
    build_contact_matrix, 
    get_age_bin
)

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
    # 1. Standardize columns
    features = [
        'Contact Setting', 'Contact Age', 'Contact Gender', 
        'Contact Frequency', 'Distance during Contact'
    ]
    
    gt = gt_contacts[features].copy()
    gen = gen_contacts[features].copy()
    
    # 2. Add Target Label (1 for Generated, 0 for Real)
    gt['is_synthetic'] = 0
    gen['is_synthetic'] = 1
    
    combined = pd.concat([gt, gen], ignore_index=True)
    combined = combined.replace('Unknown', np.nan)
    
    # 3. Encode Categorical Variables
    le = LabelEncoder()
    for col in features:
        combined[col] = combined[col].astype(str)
        combined[col] = le.fit_transform(combined[col])
        
    X = combined[features]
    y = combined['is_synthetic']
    
    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # 5. Train XGBoost
    clf = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=4,
        learning_rate=0.1,
        eval_metric='auc',
        use_label_encoder=False,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    # 6. Evaluate
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

    # Save and summarize
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