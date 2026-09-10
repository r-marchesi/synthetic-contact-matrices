import pandas as pd
import os
import argparse

# The exact variables we are carrying forward
RESPONDENT_MAP = {
    'part_id': 'part_id',  # Keep ID for grouping later
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
    'part_id': 'part_id',  # Join key
    'cont_id': 'cont_id',  # Helps track unique contacts
    'cnt_age_exact': 'Contact Age',
    'cnt_gender': 'Contact Gender',
    'frequency_multi': 'Contact Frequency',
    'phys_contact': 'Physical Contact',
    'distance': 'Distance from Home',
    'relationship': 'Relationship to Participant',
    'setting': 'Contact Setting',
    'location_multi': 'Location of Contact'
}

def create_master_csv(data_path, output_csv):
    print(f"Loading raw MixIT relational tables from: {data_path}")
    
    # 1. Load Participant & Household Tables
    p_com = pd.read_csv(os.path.join(data_path, "MixIT_it_participant_common.csv"))
    p_ext = pd.read_csv(os.path.join(data_path, "MixIT_it_participant_extra.csv"))
    h_com = pd.read_csv(os.path.join(data_path, "MixIT_it_household_common.csv"))
    
    h_ext_path = os.path.join(data_path, "MixIT_it_household_extra.csv")
    h_ext = pd.read_csv(h_ext_path) if os.path.exists(h_ext_path) else None

    # Merge participant hierarchy
    participants = p_com.merge(p_ext, on='part_id', how='left')
    participants = participants.merge(h_com, on='hh_id', how='left')
    if h_ext is not None:
        participants = participants.merge(h_ext, on='hh_id', how='left')

    # Filter and rename participant columns
    missing_p_cols = [col for col in RESPONDENT_MAP.keys() if col not in participants.columns]
    if missing_p_cols:
        print(f"Warning: Missing expected participant columns: {missing_p_cols}")
    
    p_cols_to_keep = [c for c in RESPONDENT_MAP.keys() if c in participants.columns]
    participants = participants[p_cols_to_keep].rename(columns=RESPONDENT_MAP)
    participants = participants.fillna("Unknown")

    # 2. Load Contact Tables
    c_com_path = os.path.join(data_path, "MixIT_it_contact_common.csv")
    c_ext_path = os.path.join(data_path, "MixIT_it_contact_extra.csv")
    
    if os.path.exists(c_com_path) and os.path.exists(c_ext_path):
        c_com = pd.read_csv(c_com_path)
        c_ext = pd.read_csv(c_ext_path)
        contacts = c_com.merge(c_ext, on='cont_id', how='left')
        
        # Filter and rename contact columns
        missing_c_cols = [col for col in CONTACT_MAP.keys() if col not in contacts.columns]
        if missing_c_cols:
            print(f"Warning: Missing expected contact columns: {missing_c_cols}")
            
        c_cols_to_keep = [c for c in CONTACT_MAP.keys() if c in contacts.columns]
        contacts = contacts[c_cols_to_keep].rename(columns=CONTACT_MAP)
    else:
        print("Warning: Contact files not found. Creating empty contact dataframe.")
        contacts = pd.DataFrame(columns=list(CONTACT_MAP.values()))

    # 3. Merge into a single Master Flattened Dataframe
    print("Merging participants and contacts...")
    # LEFT JOIN ensures participants with 0 contacts remain in the dataset (contacts will be NaN)
    master_df = participants.merge(contacts, on='part_id', how='left')

    # 4. Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    master_df.to_csv(output_csv, index=False)
    
    total_participants = master_df['part_id'].nunique()
    total_contacts = master_df['cont_id'].nunique()
    
    print(f"\nSuccessfully generated master CSV: {output_csv}")
    print(f"Total Unique Participants: {total_participants}")
    print(f"Total Unique Contacts: {total_contacts}")
    print(f"Total Rows (Participant-Contact pairs): {len(master_df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate MixIT raw data into a master flat CSV.")
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="data/data_mixit_postpandemic_contacts",
        help="Path to the directory containing raw MixIT CSV files"
    )
    parser.add_argument(
        "--output_csv", 
        type=str, 
        default="data/processed/master_data.csv",
        help="Output path for the aggregated master CSV"
    )

    args = parser.parse_args()
    create_master_csv(args.data_path, args.output_csv)