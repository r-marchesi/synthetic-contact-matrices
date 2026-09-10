import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def get_age_bin(age):
    try:
        age = float(age)
    except (ValueError, TypeError):
        return 14  # Default to oldest bucket if missing
    if age >= 70:
        return 14
    return int(age // 5)

def build_strata(df):
    """
    Constructs a stratification key using the 15 age bins and gender.
    Groups rare strata (<2 samples) to ensure train_test_split succeeds.
    """
    age_bins = df['Participant Age'].apply(get_age_bin)
    genders = df['Participant Gender'].astype(str).str.strip().str.upper()
    
    strata = age_bins.astype(str) + "_" + genders
    strata_counts = strata.value_counts()
    rare_strata = strata_counts[strata_counts < 2].index
    
    # Reassign rare combinations to fallback to allow stratified splitting
    strata = strata.apply(lambda x: "RARE_STRATUM" if x in rare_strata else x)
    return strata

def generate_splits(master_csv, output_dir, seed=42, val_ratio=0.20):
    print(f"Reading master data from: {master_csv}")
    df = pd.read_csv(master_csv, low_memory=False)

    # 1. Deduplicate to participant level
    participant_df = df.drop_duplicates(subset=['part_id']).copy()
    total_participants = len(participant_df)
    print(f"Loaded {total_participants} unique participants.")

    # 2. Build stratification column
    participant_df['stratum'] = build_strata(participant_df)

    # 3. Stratified Train / Validation Split (80% / 20%)
    train_df, val_df = train_test_split(
        participant_df,
        test_size=val_ratio,
        stratify=participant_df['stratum'],
        random_state=seed
    )

    os.makedirs(output_dir, exist_ok=True)

    # Save Validation IDs
    val_output = os.path.join(output_dir, "val_ids.csv")
    val_df[['part_id']].to_csv(val_output, index=False)
    print(f"\nSaved Validation IDs (N={len(val_df)}) to: {val_output}")

    # 4. Nested Stratified Training Subsets
    # Using nested slicing guarantees train_10 is a strict subset of train_30, etc.
    split_fractions = [
        ("100_percent", 1.0),
        ("80_percent", 0.80),
        ("50_percent", 0.50),
        ("30_percent", 0.30),
        ("10_percent", 0.10)
    ]

    current_pool = train_df
    saved_subsets = {}

    for i, (name, target_frac) in enumerate(split_fractions):
        if target_frac == 1.0:
            subset = current_pool
        else:
            prev_frac = split_fractions[i - 1][1]
            # Slicing fraction relative to current pool size
            relative_frac = target_frac / prev_frac
            
            # Recalculate stratum counts in the current pool to prevent singletons
            s_counts = current_pool['stratum'].value_counts()
            valid_strata = s_counts[s_counts >= 2].index
            safe_stratum = current_pool['stratum'].apply(
                lambda x: x if x in valid_strata else "RARE_STRATUM"
            )

            subset, _ = train_test_split(
                current_pool,
                train_size=relative_frac,
                stratify=safe_stratum,
                random_state=seed
            )
            
        current_pool = subset
        saved_subsets[name] = subset

        output_path = os.path.join(output_dir, f"train_ids_{name}.csv")
        subset[['part_id']].to_csv(output_path, index=False)

    print("\nTraining Split Summary:")
    print(f"{'Split Name':<15} | {'N Participants':<15} | {'Relative % of Train':<20}")
    print("-" * 55)
    train_n = len(train_df)
    for name, _ in reversed(split_fractions):
        n_count = len(saved_subsets[name])
        pct = (n_count / train_n) * 100
        print(f"{name:<15} | {n_count:<15} | {pct:<20.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create stratified participant ID splits.")
    parser.add_argument(
        "--master_csv",
        type=str,
        default="data/processed/master_data.csv",
        help="Path to master aggregated data CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/splits",
        help="Output directory to save ID CSV files"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.20,
        help="Fraction of participants reserved for validation"
    )

    args = parser.parse_args()
    generate_splits(args.master_csv, args.output_dir, args.seed, args.val_ratio)