import os
import glob
import argparse
import pandas as pd
from collections import Counter

# Resolve paths dynamically relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "../.."))
SPLITS_DIR = os.path.join(PROJECT_ROOT, "data", "splits")

# Deterministic hierarchy to prevent FSM combinatorial explosion
SETTING_ORDER = ["Home", "Work", "School", "Other"]

def find_default_train_split():
    # It will find train_ids_100_percent.csv right here
    candidates = [
        "train_ids_100_percent.csv",
        "train_ids_100.csv",
        "train_ids.csv",
        "train_100.csv",
    ]
    for name in candidates:
        path = os.path.join(SPLITS_DIR, name)
        if os.path.isfile(path):
            return path
    
    # Fallback: look for any train*.csv file in data/splits/
    matches = glob.glob(os.path.join(SPLITS_DIR, "*train*.csv"))
    if matches:
        return sorted(matches)[-1]
    
    available = os.listdir(SPLITS_DIR) if os.path.isdir(SPLITS_DIR) else []
    raise FileNotFoundError(
        f"Could not locate a training split in '{SPLITS_DIR}'. Available files: {available}"
    )

def normalize_setting(val):
    val = str(val).strip().title()
    for s in SETTING_ORDER:
        if s.lower() in val.lower():
            return s
    return "Other"

def format_participant_record(part_df):
    first = part_df.iloc[0]
    p_id = str(first["part_id"])
    p_age = first.get("part_age_exact", "Unknown")
    p_gender = first.get("part_gender", "Unknown")
    p_occ = first.get("occupation", "Unknown")
    hh_size = first.get("hh_size", 1)
    
    contacts = []
    contact_rows = part_df[part_df["cont_id"].notna()]
    
    for _, row in contact_rows.iterrows():
        contacts.append({
            "setting": normalize_setting(row.get("setting", "Other")),
            "age": row.get("cnt_age_exact", "Unknown"),
            "gender": row.get("cnt_gender", "Unknown"),
            "freq": row.get("frequency_multi", "Unknown"),
            "dist": row.get("distance", "Unknown")
        })
    
    # Macro-Ordering: sort deterministically by setting
    contacts.sort(key=lambda x: SETTING_ORDER.index(x["setting"]) if x["setting"] in SETTING_ORDER else 99)
    
    # Macro-Header counts
    counts = Counter(c["setting"] for c in contacts)
    macro_str = f"Total: {len(contacts)} | Home: {counts['Home']} | Work: {counts['Work']} | School: {counts['School']} | Other: {counts['Other']}"
    
    demo_str = f"[DEMO] Age: {p_age} | Gender: {p_gender} | Occupation: {p_occ} | HH_Size: {hh_size}"
    macro_header = f"[MACRO] {macro_str}"
    
    contact_lines = []
    for c in contacts:
        line = f"- Setting: {c['setting']} | Age: {c['age']} | Gender: {c['gender']} | Freq: {c['freq']} | Dist: {c['dist']}"
        contact_lines.append(line)
        
    contacts_block = "[CONTACTS]\n" + ("\n".join(contact_lines) if contact_lines else "- None")
    
    prompt = f"{demo_str}\n"
    target = f"{macro_header}\n{contacts_block}"
    
    return {"part_id": p_id, "prompt": prompt, "completion": target}

def process_dataset(master_csv, train_ids_csv, val_ids_csv, exp_dir):
    data_out_dir = os.path.join(exp_dir, "data")
    os.makedirs(data_out_dir, exist_ok=True)
    
    print(f"Loading master CSV: {master_csv}")
    print(f"Loading train split: {train_ids_csv}")
    print(f"Loading val split:   {val_ids_csv}")
    
    df = pd.read_csv(master_csv, low_memory=False)
    train_ids = set(pd.read_csv(train_ids_csv)["part_id"].astype(str))
    val_ids = set(pd.read_csv(val_ids_csv)["part_id"].astype(str))
    
    train_records, val_records = [], []
    
    for part_id, group in df.groupby("part_id"):
        rec = format_participant_record(group)
        pid_str = str(part_id)
        if pid_str in train_ids:
            train_records.append(rec)
        elif pid_str in val_ids:
            val_records.append(rec)
            
    train_path = os.path.join(data_out_dir, "train_dense.jsonl")
    val_path = os.path.join(data_out_dir, "val_dense.jsonl")
    
    pd.DataFrame(train_records).to_json(train_path, orient="records", lines=True)
    pd.DataFrame(val_records).to_json(val_path, orient="records", lines=True)
    
    print(f"✅ Generated {len(train_records)} train records -> {train_path}")
    print(f"✅ Generated {len(val_records)} validation records -> {val_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--master_csv", 
        default=os.path.join(PROJECT_ROOT, "data", "processed", "master_data.csv")
    )
    parser.add_argument(
        "--train_ids", 
        default=None,
        help="Path to training IDs CSV. If omitted, automatically resolves from data/splits/."
    )
    parser.add_argument(
        "--val_ids", 
        default=os.path.join(SPLITS_DIR, "val_ids.csv")
    )
    parser.add_argument("--exp_dir", default=EXP_DIR)
    args = parser.parse_args()
    
    # Automatically find the 100% split file
    resolved_train_ids = args.train_ids or find_default_train_split()
    process_dataset(args.master_csv, resolved_train_ids, args.val_ids, args.exp_dir)