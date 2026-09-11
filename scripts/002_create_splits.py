import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def get_age_bin(age):
    try:
        age = float(age)
    except (ValueError, TypeError):
        return 14  
    if age >= 70:
        return 14
    return int(age // 5)

def build_strata(df):
    # Updated to use raw column names
    age_bins = df['part_age_exact'].apply(get_age_bin)
    genders = df['part_gender'].astype(str).str.strip().str.upper()
    
    strata = age_bins.astype(str) + "_" + genders
    strata_counts = strata.value_counts()
    rare_strata = strata_counts[strata_counts < 2].index
    
    strata = strata.apply(lambda x: "RARE_STRATUM" if x in rare_strata else x)
    return strata

def generate_splits(master_csv, output_dir, seed=42, val_ratio=0.20):
    print(f"Reading master data from: {master_csv}")
    df = pd.read_csv(master_csv, low_memory=False)

    participant_df = df.drop_duplicates(subset=['part_id']).copy()
    participant_df['stratum'] = build_strata(participant_df)

    train_df, val_df = train_test_split(
        participant_df, test_size=val_ratio, stratify=participant_df['stratum'], random_state=seed
    )

    os.makedirs(output_dir, exist_ok=True)
    val_output = os.path.join(output_dir, "val_ids.csv")
    val_df[['part_id']].to_csv(val_output, index=False)
    print(f"\nSaved Validation IDs (N={len(val_df)}) to: {val_output}")

    split_fractions = [
        ("100_percent", 1.0),
        ("80_percent", 0.80),
        ("50_percent", 0.50),
        ("30_percent", 0.30),
        ("10_percent", 0.10)
    ]

    current_pool = train_df
    for i, (name, target_frac) in enumerate(split_fractions):
        if target_frac == 1.0:
            subset = current_pool
        else:
            prev_frac = split_fractions[i - 1][1]
            relative_frac = target_frac / prev_frac
            
            s_counts = current_pool['stratum'].value_counts()
            valid_strata = s_counts[s_counts >= 2].index
            safe_stratum = current_pool['stratum'].apply(
                lambda x: x if x in valid_strata else "RARE_STRATUM"
            )

            subset, _ = train_test_split(
                current_pool, train_size=relative_frac, stratify=safe_stratum, random_state=seed
            )
            
        current_pool = subset
        output_path = os.path.join(output_dir, f"train_ids_{name}.csv")
        subset[['part_id']].to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_csv", type=str, default="data/processed/master_data.csv")
    parser.add_argument("--output_dir", type=str, default="data/splits")
    args = parser.parse_args()
    generate_splits(args.master_csv, args.output_dir)