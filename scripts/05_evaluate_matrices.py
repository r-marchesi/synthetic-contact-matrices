import json
import re
import argparse
import os
import numpy as np
import pandas as pd
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

def aggregate_population_metrics(parsed_data):
    """Calculates population-level matrices, degrees, and group stats."""
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
            if not isinstance(c, dict): continue # Guard against LLM hallucinating flat strings
            c_age = c.get("Contact Age")
            if c_age is None: continue
            c_bin = get_age_bin(c_age)
            
            if c_bin is not None:
                M_sum[p_bin, c_bin] += 1
                loc = standardize_location(c.get("Location of Contact"))
                M_strat[loc][p_bin, c_bin] += 1
                
    with np.errstate(divide='ignore', invalid='ignore'):
        M_avg = np.nan_to_num(np.divide(M_sum, P_counts[:, np.newaxis]))
        for s in SETTINGS:
            M_strat[s] = np.nan_to_num(np.divide(M_strat[s], P_counts[:, np.newaxis]))
            
    avg_degree = np.mean(degrees) if degrees else 0
    return M_avg, M_strat, degrees, groups, avg_degree, len(parsed_data)

def evaluate_split(val_metrics, synthetic_file, split_name, val_lookup, plots_dir):
    M_val, M_strat_val, deg_val, grp_val, avg_val, count_val = val_metrics
    
    synth_data = []
    with open(synthetic_file, "r") as f:
        for line in f:
            row = json.loads(line)
            part_id = row.get("part_id")
            
            # Look up the participant demographics from the validation baseline
            if part_id in val_lookup:
                synth_data.append({
                    "part_id": part_id,
                    "p_bin": val_lookup[part_id]["p_bin"],
                    "demos": val_lookup[part_id]["demos"],
                    "contacts": row.get("generated_contacts", [])
                })
                
    synth_metrics = aggregate_population_metrics(synth_data)
    M_synth, M_strat_synth, deg_synth, grp_synth, avg_synth, count_synth = synth_metrics

    mae = np.mean(np.abs(M_val - M_synth))
    frobenius = np.linalg.norm(M_val - M_synth, 'fro')

    # --- PLOTTING ---
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Overall Non-Stratified Matrices
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    vmax_overall = max(np.max(M_val), np.max(M_synth))
    
    sns.heatmap(M_val, ax=axes[0], cmap="rocket_r", vmin=0, vmax=vmax_overall, xticklabels=AGE_LABELS, yticklabels=AGE_LABELS)
    axes[0].set_title(f"Real Contacts Matrix (Population N={count_val})")
    axes[0].set_xlabel("Contact Age")
    axes[0].set_ylabel("Participant Age")
    
    sns.heatmap(M_synth, ax=axes[1], cmap="rocket_r", vmin=0, vmax=vmax_overall, xticklabels=AGE_LABELS, yticklabels=AGE_LABELS)
    axes[1].set_title(f"Synthetic Contacts Matrix ({split_name}, N={count_synth})")
    axes[1].set_xlabel("Contact Age")
    axes[1].set_ylabel("Participant Age")
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"overall_matrices_{split_name}.png"), dpi=300)
    plt.close()

    # Plot 2: Degree Distribution
    plt.figure(figsize=(10, 6))
    sns.kdeplot(deg_val, label=f'Real (Avg: {avg_val:.2f})', color='#2b83ba', fill=True, alpha=0.3, bw_adjust=1.5)
    sns.kdeplot(deg_synth, label=f'Synthetic (Avg: {avg_synth:.2f})', color='#d7191c', fill=True, alpha=0.3, bw_adjust=1.5)
    
    # Safe x-limit calculation to handle empty datasets during runtime
    max_real = np.percentile(deg_val, 99) if deg_val else 0
    max_synth = np.percentile(deg_synth, 99) if deg_synth else 0
    plt.xlim(0, max(max_real, max_synth, 10))
    
    plt.xlabel("Total Contacts per Participant")
    plt.ylabel("Density")
    plt.title(f"Degree Distribution: Real vs. Synthetic ({split_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"degree_dist_{split_name}.png"), dpi=300)
    plt.close()

    # Plot 3: Average Contacts by Group (Gender & HH Size)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, g_type in enumerate(["Gender", "HH_Size"]):
        labels = sorted([k for k in grp_val[g_type].keys() if k != "Unknown"])
        real_means = [np.mean(grp_val[g_type][l]) if l in grp_val[g_type] and grp_val[g_type][l] else 0 for l in labels]
        synth_means = [np.mean(grp_synth[g_type][l]) if l in grp_synth[g_type] and grp_synth[g_type][l] else 0 for l in labels]
        
        x = np.arange(len(labels))
        width = 0.35
        axes[i].bar(x - width/2, real_means, width, label='Real Population', color='#2b83ba')
        axes[i].bar(x + width/2, synth_means, width, label='Synthetic Population', color='#d7191c')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(labels)
        axes[i].set_ylabel("Average Contacts")
        axes[i].set_title(f"By {g_type.replace('_', ' ')}")
        axes[i].legend()
    plt.suptitle(f"Demographic Contact Averages ({split_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"grouped_averages_{split_name}.png"), dpi=300)
    plt.close()

    # Plot 4: Stratified Matrices (Home, Work, School)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    strat_settings = ["Home", "Work", "School"]
    
    for i, setting in enumerate(strat_settings):
        vmax = max(np.max(M_strat_val[setting]), np.max(M_strat_synth[setting]))
        
        # Real Row
        sns.heatmap(M_strat_val[setting], ax=axes[0, i], cmap="rocket_r", vmin=0, vmax=vmax, 
                    xticklabels=AGE_LABELS if i==0 else False, yticklabels=AGE_LABELS if i==0 else False)
        axes[0, i].set_title(f"REAL: {setting}")
        
        # Synth Row
        sns.heatmap(M_strat_synth[setting], ax=axes[1, i], cmap="rocket_r", vmin=0, vmax=vmax, 
                    xticklabels=AGE_LABELS if i==0 else False, yticklabels=AGE_LABELS if i==0 else False)
        axes[1, i].set_title(f"SYNTH: {setting}")

    axes[1, 1].set_xlabel("Contact Age")
    axes[0, 0].set_ylabel("Participant Age")
    axes[1, 0].set_ylabel("Participant Age")
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"stratified_matrices_{split_name}.png"), dpi=300)
    plt.close()

    return count_synth, avg_synth, mae, frobenius

def plot_learning_curve(splits, maes, frobs, plots_dir):
    x = [int(s.split('_')[0]) for s in splits]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Training Data Split (%)')
    ax1.set_ylabel('Mean Absolute Error (MAE)', color='#2b83ba')
    ax1.plot(x, maes, marker='o', color='#2b83ba', linewidth=2, label="MAE")
    ax1.tick_params(axis='y', labelcolor='#2b83ba')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Frobenius Norm', color='#d7191c')
    ax2.plot(x, frobs, marker='s', color='#d7191c', linewidth=2, linestyle='--', label="Frobenius Norm")
    ax2.tick_params(axis='y', labelcolor='#d7191c')
    
    plt.title("RQ1: Error Metrics vs. Training Data Size")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "rq1_learning_curve.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Synthetic Contact Matrices (Population Level)")
    parser.add_argument("--validation_file", type=str, default="data/splits/validation_set.jsonl")
    parser.add_argument("--results_dir", type=str, default="data/results")
    parser.add_argument("--plots_dir", type=str, default="data/results/plots")
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    # 1. Build the true population baseline from the validation set
    print("Loading Baseline Validation Data...")
    val_data = []
    val_lookup = {}
    with open(args.validation_file, "r") as f:
        for line in f:
            row = json.loads(line)
            part_id = row.get("part_id")
            demos = extract_demographics(row["messages"])
            p_bin = get_age_bin(demos["Age"])
            if p_bin is None: continue
            
            try:
                contacts = json.loads(next(msg for msg in row["messages"] if msg["role"] == "assistant")["content"])
            except Exception:
                contacts = []
                
            record = {"part_id": part_id, "p_bin": p_bin, "demos": demos, "contacts": contacts}
            val_data.append(record)
            val_lookup[part_id] = record
            
    val_metrics = aggregate_population_metrics(val_data)
    _, _, _, _, avg_val, count_val = val_metrics
    
    print(f"Validation Population Baseline: N={count_val}, Avg Contacts={avg_val:.2f}\n")

    splits = ["10_percent", "30_percent", "50_percent", "80_percent", "100_percent"]
    valid_splits, maes, frobs = [], [], []
    
    print(f"{'Split':<12} | {'Synth N':<8} | {'Avg Synth':<10} | {'MAE vs Pop':<10} | {'F-Norm vs Pop':<12}")
    print("-" * 65)

    for split in splits:
        synth_file = os.path.join(args.results_dir, f"generated_contacts_{split}.jsonl")
        
        if os.path.exists(synth_file):
            count_synth, avg_s, mae, frob = evaluate_split(val_metrics, synth_file, split, val_lookup, args.plots_dir)
            print(f"{split:<12} | {count_synth:<8} | {avg_s:<10.2f} | {mae:<10.4f} | {frob:<12.4f}")
            valid_splits.append(split)
            maes.append(mae)
            frobs.append(frob)
        else:
            print(f"{split:<12} | {'Missing':<8} | {'-':<10} | {'-':<10} | {'-':<12}")
            
    if len(valid_splits) > 1:
        plot_learning_curve(valid_splits, maes, frobs, args.plots_dir)
        
    print(f"\nAll population-level plots saved to {args.plots_dir}/")