import json
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_jsonl(filepath):
    """Loads JSONL records into a list."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    return records

def save_jsonl(records, filepath):
    """Saves a list of records to a JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f"Saved {len(records)} records to {filepath}")

def extract_demographics_for_stratification(records):
    """
    Parses the user prompt strings back into a flat DataFrame 
    to facilitate multi-variable stratification.
    """
    rows = []
    for idx, rec in enumerate(records):
        user_content = rec['messages'][0]['content']
        # Parse lines like "- Participant Age: 34"
        demographics = {'original_index': idx}
        for line in user_content.split('\n'):
            if line.startswith('- '):
                parts = line[2:].split(': ', 1)
                if len(parts) == 2:
                    demographics[parts[0]] = parts[1]
        rows.append(demographics)
    
    df = pd.DataFrame(rows)
    return df

def create_composite_stratification_key(df):
    """
    Creates a composite key combining key demographic variables 
    to enable stratified sampling across multiple dimensions.
    """
    def bin_age(val):
        try:
            age = float(val)
            if age < 18: return '0-17'
            elif age < 30: return '18-29'
            elif age < 50: return '30-49'
            elif age < 70: return '50-69'
            else: return '70+'
        except ValueError:
            return 'Unknown'

    df['age_group'] = df.get('Participant Age', pd.Series(['Unknown']*len(df))).apply(bin_age)
    gender = df.get('Participant Gender', pd.Series(['Unknown']*len(df)))
    region = df.get('Region', pd.Series(['Unknown']*len(df)))
    education = df.get('Educational Attainment', pd.Series(['Unknown']*len(df)))

    # Combine into a single stratification string
    composite_key = (
        df['age_group'].astype(str) + "_" +
        gender.astype(str) + "_" +
        region.astype(str) + "_" +
        education.astype(str)
    )
    
    # Handle rare strata by grouping ultra-small bins into an 'Other' category
    value_counts = composite_key.value_counts()
    rare_strata = value_counts[value_counts < 2].index
    composite_key = composite_key.apply(lambda x: 'Other' if x in rare_strata else x)
    
    return composite_key

def generate_rq1_splits(input_jsonl, output_dir):
    print(f"Loading data from {input_jsonl}...")
    records = load_jsonl(input_jsonl)
    
    df_demo = extract_demographics_for_stratification(records)
    df_demo['strat_key'] = create_composite_stratification_key(df_demo)
    
    indices = np.arange(len(records))
    
    # Dictionary to hold the lists of part_ids for every split
    split_ids_mapping = {}
    
    # 1. Separate a fixed 20% validation set (stratified)
    train_indices, val_indices, _, _ = train_test_split(
        indices, 
        indices, 
        test_size=0.20, 
        stratify=df_demo['strat_key'], 
        random_state=42
    )
    
    val_records = [records[i] for i in val_indices]
    save_jsonl(val_records, os.path.join(output_dir, "validation_set.jsonl"))
    split_ids_mapping["validation_set"] = [str(rec.get('part_id', 'Unknown')) for rec in val_records]
    
    # Train pool dataframe for subsetting
    df_train = df_demo.iloc[train_indices].copy()
    train_pool_indices = train_indices
    
    # 2. Generate RQ1 Subsets (10%, 30%, 50%, 80% of the training pool)
    fractions = {
        '10_percent': 0.10, 
        '30_percent': 0.30, 
        '50_percent': 0.50, 
        '80_percent': 0.80,
        '100_percent': 1.0
    }   
    
    for name, frac in fractions.items():
        if frac == 1.0:
            subset_indices = train_pool_indices
        else:
            sub_indices, _, _, _ = train_test_split(
                train_pool_indices,
                train_pool_indices,
                test_size=(1.0 - frac),
                stratify=df_train['strat_key'],
                random_state=42
            )
            subset_indices = sub_indices
            
        subset_records = [records[i] for i in subset_indices]
        split_name = f"train_{name}"
        
        save_jsonl(subset_records, os.path.join(output_dir, f"{split_name}.jsonl"))
        split_ids_mapping[split_name] = [str(rec.get('part_id', 'Unknown')) for rec in subset_records]

    # 3. Save the consolidated ID lists to a single JSON file
    mapping_filepath = os.path.join(output_dir, "all_splits_ids.json")
    with open(mapping_filepath, 'w', encoding='utf-8') as f:
        json.dump(split_ids_mapping, f, indent=4)
        
    print(f"Saved master ID mapping list to {mapping_filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create stratified data splits for RQ1.")
    parser.add_argument("--input", type=str, default="data/processed/llm_training_data_full.jsonl", help="Path to processed JSONL")
    parser.add_argument("--output_dir", type=str, default="data/splits", help="Directory to save split JSONLs and the ID lists")
    
    args = parser.parse_args()
    generate_rq1_splits(args.input, args.output_dir)