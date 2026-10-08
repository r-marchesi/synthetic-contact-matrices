import os
import shutil
import pandas as pd

def prepare_share_data():
    # Define primary paths
    data_dir = "data"
    out_dir = os.path.join(data_dir, "to_share")
    
    # Corrected path to point to the 'processed' subfolder
    master_csv = os.path.join(data_dir, "processed", "master_data.csv") 
    
    val_ids_csv = os.path.join(data_dir, "splits", "val_ids.csv") 
    parsed_csv = os.path.join(data_dir, "parsed_results", "prompt_cot_epoch_2_results.csv")
    
    # 1. Create the target directory
    os.makedirs(out_dir, exist_ok=True)
    print(f"Created directory: {out_dir}")

    # 2. Copy master_data.csv
    out_master = os.path.join(out_dir, "master_data.csv")
    if os.path.exists(master_csv):
        shutil.copy(master_csv, out_master)
        print("Copied master_data.csv")
    else:
        print(f"Error: {master_csv} not found.")
        return

    # Load master data for subsequent extraction and formatting
    master_df = pd.read_csv(master_csv, low_memory=False)
    master_df['part_id'] = master_df['part_id'].astype(str)

    # 3. Extract and save validation_set.csv
    if os.path.exists(val_ids_csv):
        val_ids_df = pd.read_csv(val_ids_csv)
        val_ids_set = set(val_ids_df['part_id'].astype(str))
        
        # Filter master_df using the validation participants logic
        val_df = master_df[master_df['part_id'].isin(val_ids_set)].copy()
        
        out_val = os.path.join(out_dir, "validation_set.csv")
        val_df.to_csv(out_val, index=False)
        print(f"Saved validation_set.csv ({len(val_df)} rows)")
    else:
        print(f"Warning: {val_ids_csv} not found. Cannot extract validation_set.csv.")

    # 4. Format and save prompt_cot_epoch_2_results.csv
    if os.path.exists(parsed_csv):
        parsed_df = pd.read_csv(parsed_csv, low_memory=False)
        parsed_df['part_id'] = parsed_df['part_id'].astype(str)
        
        # Isolate unique participant-level information from the master data
        part_info_df = master_df.drop_duplicates(subset=['part_id'])
        
        # Identify columns present in master_data but missing from the parsed results
        missing_cols = [col for col in master_df.columns if col not in parsed_df.columns and col != 'part_id']
        
        # Slice the participant info to only the columns that need to be merged
        cols_to_merge = ['part_id'] + missing_cols
        part_info_to_merge = part_info_df[cols_to_merge]
        
        # Merge the parsed results with the missing variables
        merged_df = pd.merge(parsed_df, part_info_to_merge, on='part_id', how='left')
        
        # Reorder columns to align exactly with the master_data format where possible
        final_cols = [col for col in master_df.columns if col in merged_df.columns]
        # Append any newly generated columns that were strictly in parsed_results
        final_cols += [col for col in merged_df.columns if col not in final_cols]
        
        merged_df = merged_df[final_cols]
        
        out_parsed = os.path.join(out_dir, "prompt_cot_epoch_2_results.csv")
        merged_df.to_csv(out_parsed, index=False)
        print(f"Saved formatted prompt_cot_epoch_2_results.csv ({len(merged_df)} rows)")
    else:
        print(f"Warning: {parsed_csv} not found. Cannot process results.")

if __name__ == "__main__":
    prepare_share_data()