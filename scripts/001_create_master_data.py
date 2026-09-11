import pandas as pd
import os
import argparse

PARTICIPANT_COLS = [
    'part_id',
    'part_age_exact',
    'part_gender',
    'hh_size',
    'presence_work',
    'presence_school',
    'educational_attainment',
    'region_nuts1',
    'occupation',
    'household_monthly_net_income'
]

CONTACT_COLS = [
    'part_id',  
    'cont_id',  
    'cnt_age_exact',
    'cnt_gender',
    'frequency_multi',
    'phys_contact',
    'distance',
    'relationship',
    'setting',
    'location_multi'
]

def create_master_csv(data_path, output_csv):
    print(f"Loading raw MixIT relational tables from: {data_path}")
    
    p_com = pd.read_csv(os.path.join(data_path, "MixIT_it_participant_common.csv"))
    p_ext = pd.read_csv(os.path.join(data_path, "MixIT_it_participant_extra.csv"))
    h_com = pd.read_csv(os.path.join(data_path, "MixIT_it_household_common.csv"))
    
    h_ext_path = os.path.join(data_path, "MixIT_it_household_extra.csv")
    h_ext = pd.read_csv(h_ext_path) if os.path.exists(h_ext_path) else None

    participants = p_com.merge(p_ext, on='part_id', how='left')
    participants = participants.merge(h_com, on='hh_id', how='left')
    if h_ext is not None:
        participants = participants.merge(h_ext, on='hh_id', how='left')

    p_cols_to_keep = [c for c in PARTICIPANT_COLS if c in participants.columns]
    participants = participants[p_cols_to_keep]
    participants = participants.fillna("Unknown")

    c_com_path = os.path.join(data_path, "MixIT_it_contact_common.csv")
    c_ext_path = os.path.join(data_path, "MixIT_it_contact_extra.csv")
    
    if os.path.exists(c_com_path) and os.path.exists(c_ext_path):
        c_com = pd.read_csv(c_com_path)
        c_ext = pd.read_csv(c_ext_path)
        contacts = c_com.merge(c_ext, on='cont_id', how='left')
            
        c_cols_to_keep = [c for c in CONTACT_COLS if c in contacts.columns]
        contacts = contacts[c_cols_to_keep]
    else:
        print("Warning: Contact files not found. Creating empty contact dataframe.")
        contacts = pd.DataFrame(columns=CONTACT_COLS)

    print("Merging participants and contacts...")
    master_df = participants.merge(contacts, on='part_id', how='left')

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    master_df.to_csv(output_csv, index=False)
    
    print(f"\nSuccessfully generated master CSV: {output_csv}")
    print(f"Total Unique Participants: {master_df['part_id'].nunique()}")
    print(f"Total Rows (Participant-Contact pairs): {len(master_df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate MixIT raw data into a master flat CSV.")
    parser.add_argument("--data_path", type=str, default="data/data_mixit_postpandemic_contacts")
    parser.add_argument("--output_csv", type=str, default="data/processed/master_data.csv")
    args = parser.parse_args()
    create_master_csv(args.data_path, args.output_csv)