import pandas as pd
import json
import os
import argparse

# 1. Map raw column names to natural language concepts for the LLM
RESPONDENT_MAP = {
    'part_age_exact': 'Participant Age',
    'part_gender': 'Participant Gender',
    'hh_size': 'Household Size',
    'presence_work': 'Attends Work',
    'presence_school': 'Attends School',
    'educational_attainment': 'Educational Attainment',
    'region_nuts1': 'Region',
    'occupation': 'Occupation',
    'household_monthly_net_income': 'Household Monthly Net Income'
}

CONTACT_MAP = {
    'cnt_age_exact': 'Contact Age',
    'cnt_gender': 'Contact Gender',
    'frequency_multi': 'Contact Frequency',
    'phys_contact': 'Physical Contact',
    'distance': 'Distance from Home',
    'relationship': 'Relationship to Participant',
    'setting': 'Contact Setting',
    'location_multi': 'Location of Contact'
}

def load_and_merge_mixit(data_path):
    """
    Loads raw MixIT relational CSVs and performs left-merges 
    to construct complete participant and contact dataframes.
    """
    print(f"Loading raw MixIT relational tables from: {data_path}")
    
    p_com = pd.read_csv(os.path.join(data_path, "MixIT_it_participant_common.csv"))
    p_ext = pd.read_csv(os.path.join(data_path, "MixIT_it_participant_extra.csv"))
    h_com = pd.read_csv(os.path.join(data_path, "MixIT_it_household_common.csv"))
    
    h_ext_path = os.path.join(data_path, "MixIT_it_household_extra.csv")
    h_ext = pd.read_csv(h_ext_path) if os.path.exists(h_ext_path) else None

    # Merge participant hierarchy
    parent_df = p_com.merge(p_ext, on='part_id', how='left')
    parent_df = parent_df.merge(h_com, on='hh_id', how='left')
    if h_ext is not None:
        parent_df = parent_df.merge(h_ext, on='hh_id', how='left')

    # Load and merge contact hierarchy
    c_com = pd.read_csv(os.path.join(data_path, "MixIT_it_contact_common.csv"))
    c_ext = pd.read_csv(os.path.join(data_path, "MixIT_it_contact_extra.csv"))
    child_df = c_com.merge(c_ext, on='cont_id', how='left')

    # Standardize missing values as strings
    parent_df = parent_df.fillna("Unknown")
    child_df = child_df.fillna("Unknown")

    return parent_df, child_df

def format_user_prompt(parent_row):
    """
    Formats respondent characteristics into a natural language markdown list.
    """
    prompt = "Generate the social contacts for a participant with the following demographics:\n"
    for raw_col, natural_name in RESPONDENT_MAP.items():
        val = str(parent_row[raw_col]) if raw_col in parent_row else "Unknown"
        prompt += f"- {natural_name}: {val}\n"
    return prompt.strip()

def format_assistant_completion(contacts_df):
    """
    Formats the list of contacts into a strict, human-readable JSON array.
    """
    if contacts_df is None or len(contacts_df) == 0:
        return "[]" # Empty JSON array for zero contacts

    contact_list = []
    for _, c_row in contacts_df.iterrows():
        contact_dict = {}
        for raw_col, natural_name in CONTACT_MAP.items():
            # Keep numeric types where possible, cast others to string
            val = c_row[raw_col]
            if pd.isna(val) or val == "Unknown":
                contact_dict[natural_name] = "Unknown"
            else:
                contact_dict[natural_name] = val
        contact_list.append(contact_dict)

    # Convert the python list of dicts to a beautifully indented JSON string
    return json.dumps(contact_list, indent=2, ensure_ascii=False)

def prepare_dataset(data_path, output_jsonl):
    """
    Executes table merging, structures data into ChatML/JSONL format, and saves output.
    """
    parent_df, child_df = load_and_merge_mixit(data_path)
    
    grouped_contacts = child_df.groupby('part_id')
    formatted_records = []

    print("Formatting participant contact networks into chat structure...")
    for _, parent_row in parent_df.iterrows():
        p_id = parent_row['part_id']
        
        user_msg = format_user_prompt(parent_row)
        
        if p_id in grouped_contacts.groups:
            contacts_for_part = grouped_contacts.get_group(p_id)
        else:
            contacts_for_part = None
            
        assistant_msg = format_assistant_completion(contacts_for_part)

        record = {
            "part_id": str(p_id),
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ]
        }
        formatted_records.append(record)

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    print(f"Saving {len(formatted_records)} records to {output_jsonl}...")
    
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for rec in formatted_records:
            f.write(json.dumps(rec) + '\n')

    print("Data preparation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MixIT data into LLM chat training format.")
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="data/data_mixit_postpandemic_contacts",
        help="Path to the directory containing raw MixIT CSV files"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/processed/llm_training_data_full.jsonl",
        help="Output path for processed JSONL file"
    )

    args = parser.parse_args()
    prepare_dataset(args.data_path, args.output)