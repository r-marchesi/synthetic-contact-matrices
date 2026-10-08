import os
import glob
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CONTACT_MAP = {
    'cnt_age_exact': 'Contact Age',
    'cnt_gender': 'Contact Gender',
    'frequency_multi': 'Contact Frequency',
    'phys_contact': 'Physical Contact',
    'distance': 'Distance during Contact',
    'relationship': 'Relationship to Participant',
    'setting': 'Contact Setting',
    'location_multi': 'Location of Contact'
}

CATEGORICAL_VARS = [
    'Contact Gender', 'Contact Frequency', 'Physical Contact', 
    'Distance during Contact', 'Relationship to Participant', 
    'Contact Setting', 'Location of Contact'
]

def load_ground_truth(master_csv, val_ids_csv):
    """Loads and aligns ground truth data with natural language keys, including extended demographics."""
    val_ids_df = pd.read_csv(val_ids_csv)
    val_ids_set = set(val_ids_df['part_id'].astype(str))
    
    master_df = pd.read_csv(master_csv, low_memory=False)
    master_df['part_id'] = master_df['part_id'].astype(str)
    gt_df = master_df[master_df['part_id'].isin(val_ids_set)].copy()
    
    # Extract Demographics & cast numeric
    demo_cols = ['part_id', 'part_age_exact', 'hh_size', 'part_gender', 
                 'hh_income', 'work_status', 'school_status']
    
    # Ensure columns exist, fill with 'Unknown' if missing from the raw data
    for col in demo_cols:
        if col not in gt_df.columns:
            gt_df[col] = 'Unknown'
            
    demos_df = gt_df.drop_duplicates(subset=['part_id'])[demo_cols].copy()
    
    # Process Age
    demos_df['part_age_exact'] = pd.to_numeric(demos_df['part_age_exact'], errors='coerce')
    demos_df['part_age_bracket'] = pd.cut(
        demos_df['part_age_exact'], 
        bins=[0, 18, 35, 55, 120], 
        labels=['0-18', '19-35', '36-55', '56+']
    )
    
    # Process Household Size
    demos_df['hh_size_num'] = pd.to_numeric(demos_df['hh_size'], errors='coerce').fillna(1)
    demos_df['hh_size_bracket'] = pd.cut(
        demos_df['hh_size_num'],
        bins=[0, 1, 2, 4, 100],
        labels=['1', '2', '3-4', '5+']
    )
    
    # Clean up categorical demographic variables
    demos_df['part_gender'] = demos_df['part_gender'].fillna('Unknown')
    demos_df['hh_income'] = demos_df['hh_income'].fillna('Unknown')
    
    # Simplify Work/School status to booleans/presence if they are complex strings
    # Adjust this logic based on how your raw data codes these variables
    demos_df['present_at_work'] = demos_df['work_status'].astype(str).apply(
        lambda x: 'Yes' if 'work' in x.lower() or 'employed' in x.lower() or x in ['1', '1.0', 'True'] else 'No'
    )
    demos_df['present_at_school'] = demos_df['school_status'].astype(str).apply(
        lambda x: 'Yes' if 'school' in x.lower() or 'student' in x.lower() or x in ['1', '1.0', 'True'] else 'No'
    )
    
    # Extract Contacts and map columns safely
    valid_contacts = gt_df[gt_df['cont_id'].notna()].copy()
    keep_cols = ['part_id']
    for raw_col, clean_col in CONTACT_MAP.items():
        if raw_col in valid_contacts.columns:
            valid_contacts[clean_col] = valid_contacts[raw_col]
            keep_cols.append(clean_col)
            
    valid_contacts = valid_contacts[keep_cols]
    
    return valid_contacts, demos_df, val_ids_set

def plot_volumes(gt_contacts, gen_contacts, val_ids_set, output_path, title):
    """Plots a side-by-side discrete histogram of the number of contacts per participant."""
    gt_counts = gt_contacts.groupby('part_id').size()
    gt_counts_dict = {pid: gt_counts.get(pid, 0) for pid in val_ids_set}
    
    gen_counts = gen_contacts.groupby('part_id').size()
    gen_counts_dict = {pid: gen_counts.get(pid, 0) for pid in val_ids_set}
    
    # Package into a single DataFrame for Seaborn dodging
    vol_df = pd.DataFrame([
        {"Volume": vol, "Source": "Ground Truth"} for vol in gt_counts_dict.values()
    ] + [
        {"Volume": vol, "Source": "Generated"} for vol in gen_counts_dict.values()
    ])
    
    plt.figure(figsize=(12, 6))
    sns.histplot(data=vol_df, x='Volume', hue='Source', multiple='dodge', 
                 discrete=True, shrink=0.8, palette=['#1f77b4', '#ff7f0e'])
    
    plt.title(f"Contact Volume Histogram: {title}", fontsize=15)
    plt.xlabel("Number of Contacts per Participant", fontsize=12)
    plt.ylabel("Count of Participants", fontsize=12)
    
    # Cap x-axis for readability
    max_vol = int(vol_df['Volume'].quantile(0.99)) + 2 
    plt.xlim(-0.5, max_vol)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_categorical_distributions(gt_contacts, gen_contacts, output_path, title):
    """Plots normalized bar charts for all categorical variables."""
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle(f"Categorical Distributions: {title}", fontsize=18, y=1.02)
    axes = axes.flatten()
    
    for i, var in enumerate(CATEGORICAL_VARS):
        ax = axes[i]
        gt_dist = gt_contacts[var].value_counts(normalize=True).rename('Ground Truth')
        gen_dist = gen_contacts[var].value_counts(normalize=True).rename('Generated')
        
        comp_df = pd.concat([gt_dist, gen_dist], axis=1).fillna(0)
        comp_df.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'], alpha=0.85)
        
        ax.set_title(var, fontsize=14)
        ax.set_ylabel("Proportion")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
        
    for j in range(len(CATEGORICAL_VARS), len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_stratified_categorical_distributions(gt_contacts, gen_contacts, demos_df, stratify_col, output_path, title):
    """Plots normalized bar charts stratified by a conditioning demographic variable."""
    # Merge demographics to align participant traits with their contacts
    gt_merged = gt_contacts.merge(demos_df[['part_id', stratify_col]], on='part_id', how='left')
    gen_merged = gen_contacts.merge(demos_df[['part_id', stratify_col]], on='part_id', how='left')
    
    # Cast to string and drop missing demographics for clean plotting
    gt_merged[stratify_col] = gt_merged[stratify_col].astype(str)
    gen_merged[stratify_col] = gen_merged[stratify_col].astype(str)
    
    # Filter out common missing value representations
    strata = sorted([s for s in gt_merged[stratify_col].unique() if s.lower() not in ["nan", "unknown", "none"]])
    
    if not strata:
        print(f"  -> No valid strata found for {stratify_col}. Skipping plot.")
        return

    # Create a dynamic grid: Rows = Variables, Columns = Strata
    fig, axes = plt.subplots(len(CATEGORICAL_VARS), len(strata), 
                             figsize=(6 * len(strata), 5 * len(CATEGORICAL_VARS)))
    fig.suptitle(f"Categorical Distributions Stratified by {stratify_col}: {title}", fontsize=20, y=1.01)
    
    # Ensure axes is a 2D numpy array even if we only have 1 stratum
    if len(CATEGORICAL_VARS) == 1 and len(strata) == 1:
        axes = np.array([[axes]])
    elif len(CATEGORICAL_VARS) == 1:
        axes = np.expand_dims(axes, axis=0)
    elif len(strata) == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, var in enumerate(CATEGORICAL_VARS):
        for j, stratum in enumerate(strata):
            ax = axes[i, j]
            
            gt_subset = gt_merged[gt_merged[stratify_col] == stratum]
            gen_subset = gen_merged[gen_merged[stratify_col] == stratum]
            
            gt_dist = gt_subset[var].value_counts(normalize=True).rename('Ground Truth')
            gen_dist = gen_subset[var].value_counts(normalize=True).rename('Generated')
            
            comp_df = pd.concat([gt_dist, gen_dist], axis=1).fillna(0)
            
            if not comp_df.empty:
                comp_df.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'], alpha=0.85, legend=(i==0 and j==0))
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
                
            ax.set_title(f"{var}\n({stratify_col}: {stratum})", fontsize=14)
            ax.set_ylabel("Proportion")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_conditioning(gt_contacts, gen_contacts, demos_df, output_path, title):
    """Checks if the LLM respected explicit demographic conditioning."""
    pp_df = demos_df.copy()
    
    # Metric 1: Number of Home Contacts
    gt_home = gt_contacts[gt_contacts['Contact Setting'].astype(str).str.contains('Home', case=False, na=False)]
    gen_home = gen_contacts[gen_contacts['Contact Setting'].astype(str).str.contains('Home', case=False, na=False)]
    
    pp_df = pp_df.merge(gt_home.groupby('part_id').size().rename('GT_Home'), on='part_id', how='left').fillna({'GT_Home': 0})
    pp_df = pp_df.merge(gen_home.groupby('part_id').size().rename('Gen_Home'), on='part_id', how='left').fillna({'Gen_Home': 0})
    
    # Metric 2: Mean Contact Age
    gt_contacts_c = gt_contacts.copy()
    gen_contacts_c = gen_contacts.copy()
    gt_contacts_c['Contact Age Num'] = pd.to_numeric(gt_contacts_c['Contact Age'], errors='coerce')
    gen_contacts_c['Contact Age Num'] = pd.to_numeric(gen_contacts_c['Contact Age'], errors='coerce')
    
    pp_df = pp_df.merge(gt_contacts_c.groupby('part_id')['Contact Age Num'].mean().rename('GT_Mean_Age'), on='part_id', how='left')
    pp_df = pp_df.merge(gen_contacts_c.groupby('part_id')['Contact Age Num'].mean().rename('Gen_Mean_Age'), on='part_id', how='left')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Conditioning Variable Fidelity: {title}", fontsize=16, y=1.05)
    
    # Panel A: Household Size -> Home Contacts
    pp_df['hh_size_clip'] = pp_df['hh_size_num'].clip(upper=6)
    home_by_hh = pp_df.groupby('hh_size_clip')[['GT_Home', 'Gen_Home']].mean()
    
    home_by_hh.plot(kind='bar', ax=axes[0], color=['#1f77b4', '#ff7f0e'], alpha=0.85)
    axes[0].set_title("Does Household Size influence 'Home' contacts?", fontsize=13)
    axes[0].set_xlabel("Participant Household Size", fontsize=11)
    axes[0].set_ylabel("Average 'Home' Contacts", fontsize=11)
    axes[0].set_xticklabels([f"{int(x)}" if x < 6 else "6+" for x in home_by_hh.index], rotation=0)
    axes[0].legend(['Ground Truth', 'Generated'])
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    
    # Panel B: Participant Age -> Contact Age (Assortativity)
    pp_df['part_age_bracket_10'] = (pp_df['part_age_exact'] // 10) * 10
    age_by_part = pp_df.groupby('part_age_bracket_10')[['GT_Mean_Age', 'Gen_Mean_Age']].mean()
    
    age_by_part.plot(kind='line', marker='o', ax=axes[1], color=['#1f77b4', '#ff7f0e'], linewidth=2.5, markersize=8)
    axes[1].set_title("Age Assortativity (Do older people interact with older people?)", fontsize=13)
    axes[1].set_xlabel("Participant Age Bracket", fontsize=11)
    axes[1].set_ylabel("Average Contact Age", fontsize=11)
    axes[1].legend(['Ground Truth', 'Generated'])
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def evaluate_distributions(master_csv, val_ids_csv, parsed_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading Ground Truth validation data and demographics...")
    gt_contacts, demos_df, val_ids_set = load_ground_truth(master_csv, val_ids_csv)
    
    parsed_files = glob.glob(os.path.join(parsed_dir, "*_results.csv"))
    file_pattern = re.compile(r"(.*)_epoch_(\d+)_results\.csv")
    
    # Define which columns from demos_df we want to stratify by
    stratification_targets = {
        'part_gender': 'gender',
        'part_age_bracket': 'age',
        'hh_income': 'income',
        'hh_size_bracket': 'hh_size',
        'present_at_work': 'work',
        'present_at_school': 'school'
    }
    
    for file_path in parsed_files:
        filename = os.path.basename(file_path)
        match = file_pattern.match(filename)
        if not match:
            continue
            
        prompt_style = match.group(1)
        epoch = match.group(2)
        title = f"{prompt_style} (Epoch {epoch})"
        
        exp_dir = os.path.join(output_dir, f"{prompt_style}_epoch_{epoch}")
        os.makedirs(exp_dir, exist_ok=True)
        
        print(f"Generating distribution plots for {title}...")
        gen_contacts = pd.read_csv(file_path, low_memory=False)
        gen_contacts['part_id'] = gen_contacts['part_id'].astype(str)
        
        # 1. Base Volume Plots
        plot_volumes(gt_contacts, gen_contacts, val_ids_set, 
                     os.path.join(exp_dir, "volume_histogram.png"), title)
        
        # 2. Base Categorical Plots
        plot_categorical_distributions(gt_contacts, gen_contacts, 
                                       os.path.join(exp_dir, "categorical_distributions.png"), title)
        
        # 3. Stratified Categorical Plots
        print("  Generating stratified plots...")
        for col_name, file_suffix in stratification_targets.items():
            plot_stratified_categorical_distributions(
                gt_contacts, gen_contacts, demos_df, col_name,
                os.path.join(exp_dir, f"stratified_by_{file_suffix}.png"), title
            )
        
        # 4. Correlation Plots
        plot_conditioning(gt_contacts, gen_contacts, demos_df, 
                          os.path.join(exp_dir, "conditioning_correlations.png"), title)

    print(f"\nAll distribution analysis complete. Plots saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_csv", type=str, default="data/processed/master_data.csv")
    parser.add_argument("--val_ids", type=str, default="data/splits/val_ids.csv")
    parser.add_argument("--parsed_dir", type=str, default="data/parsed_results")
    parser.add_argument("--output_dir", type=str, default="data/results/distributions")
    
    args = parser.parse_args()
    evaluate_distributions(args.master_csv, args.val_ids, args.parsed_dir, args.output_dir)