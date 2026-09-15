import json
import re
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

AGE_LABELS = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", 
              "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70+"]
SETTINGS = ["Home", "Work", "School", "Leisure", "Transport", "Other"]

def get_age_bin(age):
    try:
        age = float(age)
    except (ValueError, TypeError):
        return None
    if age >= 70: return 14
    return int(age // 5)

def extract_demographics(messages):
    demos = {"Age": None, "Gender": "Unknown", "HH_Size": "Unknown"}
    for msg in messages:
        if msg["role"] == "user":
            content = msg["content"]
            age_match = re.search(r'- Participant Age:\s*(\d+)', content)
            if age_match: demos["Age"] = int(age_match.group(1))
            
            gender_match = re.search(r'- Participant Gender:\s*([A-Za-z]+)', content)
            if gender_match: demos["Gender"] = gender_match.group(1).strip()
                
            hh_match = re.search(r'- Household Size:\s*(\d+)', content)
            if hh_match: demos["HH_Size"] = hh_match.group(1).strip()
    return demos

def standardize_location(loc):
    if not loc: return "Other"
    loc = str(loc).lower()
    if "home" in loc: return "Home"
    if "work" in loc: return "Work"
    if "school" in loc or "education" in loc: return "School"
    if "leisure" in loc: return "Leisure"
    if "transport" in loc: return "Transport"
    return "Other"

def parse_dataset(filepath):
    """Parses a dataset (train or val) and returns participant records."""
    parsed_data = []
    with open(filepath, "r") as f:
        for line in f:
            row = json.loads(line)
            part_id = row.get("part_id", "unknown")
            demos = extract_demographics(row["messages"])
            p_bin = get_age_bin(demos["Age"])
            if p_bin is None: continue
            
            try:
                # We pull the ground truth contacts from the assistant message
                contacts = json.loads(next(msg for msg in row["messages"] if msg["role"] == "assistant")["content"])
            except Exception:
                contacts = []
                
            parsed_data.append({
                "part_id": part_id,
                "p_bin": p_bin,
                "demos": demos,
                "contacts": contacts
            })
    return parsed_data

def aggregate_metrics(parsed_data):
    """Calculates matrices, degrees, and group stats for a given dataset."""
    M_sum = np.zeros((15, 15))
    P_counts = np.zeros(15)
    M_strat = {s: np.zeros((15, 15)) for s in SETTINGS}
    degrees = []
    groups = {"Gender": {}, "HH_Size": {}}
    
    for p in parsed_data:
        p_bin, demos, contacts = p["p_bin"], p["demos"], p["contacts"]
        P_counts[p_bin] += 1
        degrees.append(len(contacts))
        
        for g_type in ["Gender", "HH_Size"]:
            val = demos[g_type]
            if val not in groups[g_type]:
                groups[g_type][val] = []
            groups[g_type][val].append(len(contacts))
            
        for c in contacts:
            c_bin = get_age_bin(c.get("Contact Age"))
            if c_bin is not None:
                M_sum[p_bin, c_bin] += 1
                loc = standardize_location(c.get("Location of Contact"))
                M_strat[loc][p_bin, c_bin] += 1
                
    with np.errstate(divide='ignore', invalid='ignore'):
        M_avg = np.nan_to_num(np.divide(M_sum, P_counts[:, np.newaxis]))
        for s in SETTINGS:
            M_strat[s] = np.nan_to_num(np.divide(M_strat[s], P_counts[:, np.newaxis]))
            
    return M_avg, M_strat, degrees, groups, np.mean(degrees), len(parsed_data)

def evaluate_train_vs_val(val_metrics, train_file, split_name, plots_dir):
    M_val, M_strat_val, deg_val, grp_val, avg_val, count_val = val_metrics
    
    train_data = parse_dataset(train_file)
    M_train, M_strat_train, deg_train, grp_train, avg_train, count_train = aggregate_metrics(train_data)

    mae = np.mean(np.abs(M_val - M_train))
    frobenius = np.linalg.norm(M_val - M_train, 'fro')

    # --- PLOTTING ---
    sns.set_theme(style="whitegrid")
    
    # 1. Overall Matrices
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    vmax = max(np.max(M_val), np.max(M_train))
    
    sns.heatmap(M_val, ax=axes[0], cmap="rocket_r", vmin=0, vmax=vmax, xticklabels=AGE_LABELS, yticklabels=AGE_LABELS)
    axes[0].set_title(f"Validation Set Matrix (N={count_val})")
    axes[0].set_xlabel("Contact Age")
    axes[0].set_ylabel("Participant Age")
    
    sns.heatmap(M_train, ax=axes[1], cmap="rocket_r", vmin=0, vmax=vmax, xticklabels=AGE_LABELS, yticklabels=AGE_LABELS)
    axes[1].set_title(f"Train Set Matrix ({split_name}, N={count_train})")
    axes[1].set_xlabel("Contact Age")
    axes[1].set_ylabel("Participant Age")
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"sanity_overall_matrices_{split_name}.png"), dpi=300)
    plt.close()

    # 2. Degree Distribution
    plt.figure(figsize=(10, 6))
    sns.kdeplot(deg_val, label=f'Validation (Avg: {avg_val:.2f})', color='#2b83ba', fill=True, alpha=0.3, bw_adjust=1.5)
    sns.kdeplot(deg_train, label=f'Train Split (Avg: {avg_train:.2f})', color='#4daf4a', fill=True, alpha=0.3, bw_adjust=1.5)
    plt.xlim(0, max(np.percentile(deg_val, 99), np.percentile(deg_train, 99)))
    plt.xlabel("Total Contacts per Participant")
    plt.ylabel("Density")
    plt.title(f"Sanity Check: Degree Distribution ({split_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"sanity_degree_dist_{split_name}.png"), dpi=300)
    plt.close()

    # 3. Stratified Matrices (Home, Work, School)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    strat_settings = ["Home", "Work", "School"]
    
    for i, setting in enumerate(strat_settings):
        vmax_s = max(np.max(M_strat_val[setting]), np.max(M_strat_train[setting]))
        
        sns.heatmap(M_strat_val[setting], ax=axes[0, i], cmap="rocket_r", vmin=0, vmax=vmax_s, 
                    xticklabels=AGE_LABELS if i==0 else False, yticklabels=AGE_LABELS if i==0 else False)
        axes[0, i].set_title(f"VAL: {setting}")
        
        sns.heatmap(M_strat_train[setting], ax=axes[1, i], cmap="rocket_r", vmin=0, vmax=vmax_s, 
                    xticklabels=AGE_LABELS if i==0 else False, yticklabels=AGE_LABELS if i==0 else False)
        axes[1, i].set_title(f"TRAIN: {setting}")

    axes[1, 1].set_xlabel("Contact Age")
    axes[0, 0].set_ylabel("Participant Age")
    axes[1, 0].set_ylabel("Participant Age")
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"sanity_stratified_{split_name}.png"), dpi=300)
    plt.close()

    return count_train, avg_train, mae, frobenius

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity Check: Compare Train Splits vs Validation Set")
    parser.add_argument("--validation_file", type=str, default="data/splits/validation_set.jsonl")
    parser.add_argument("--train_dir", type=str, default="data/splits")
    parser.add_argument("--plots_dir", type=str, default="data/results/plots_sanity")
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    print("Loading Baseline Validation Data...")
    val_data = parse_dataset(args.validation_file)
    val_metrics = aggregate_metrics(val_data)
    _, _, _, _, avg_val, count_val = val_metrics
    
    print(f"Validation Set Baseline: N={count_val}, Avg Contacts={avg_val:.2f}\n")

    splits = ["10_percent", "30_percent", "50_percent", "80_percent", "100_percent"]
    
    print(f"{'Split':<12} | {'Train N':<8} | {'Avg Train':<10} | {'MAE vs Val':<12} | {'F-Norm vs Val':<15}")
    print("-" * 65)

    for split in splits:
        # Assuming training files are named like 'train_10_percent.jsonl'
        train_file = os.path.join(args.train_dir, f"train_{split}.jsonl")
        
        if os.path.exists(train_file):
            count_train, avg_train, mae, frob = evaluate_train_vs_val(val_metrics, train_file, split, args.plots_dir)
            print(f"{split:<12} | {count_train:<8} | {avg_train:<10.2f} | {mae:<12.4f} | {frob:<15.4f}")
        else:
            print(f"{split:<12} | {'Missing':<8} | {'-':<10} | {'-':<12} | {'-':<15}")
            
    print(f"\nSanity check plots saved to {args.plots_dir}/")