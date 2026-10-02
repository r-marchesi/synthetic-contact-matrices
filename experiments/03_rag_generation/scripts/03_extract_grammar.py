import os
import json
import argparse
import pandas as pd

# --- PATH RESOLUTION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
VECTOR_DIR = os.path.join(EXP_DIR, "data", "vector_store")

def extract_empirical_bounds(master_csv_path):
    os.makedirs(VECTOR_DIR, exist_ok=True)
    out_file = os.path.join(VECTOR_DIR, "empirical_grammar.json")
    
    print(f"Loading raw data from: {master_csv_path}")
    df = pd.read_csv(master_csv_path, low_memory=False)
    
    print("Extracting empirical categorical sets...")
    # Extract unique values and drop NaNs. Convert everything to strings.
    settings = df['setting'].dropna().unique().tolist()
    freqs = df['frequency_multi'].dropna().unique().tolist()
    dists = df['distance'].dropna().unique().tolist()
    
    # We ensure "Unknown" is in every list since your processing script outputs it for NaNs
    if "Unknown" not in settings: settings.append("Unknown")
    if "Unknown" not in freqs: freqs.append("Unknown")
    if "Unknown" not in dists: dists.append("Unknown")

    print("Extracting age boundaries...")
    # We extract the max participant and contact ages to cap the regex
    max_part_age = int(df['part_age_exact'].max())
    max_cont_age = int(df['cnt_age_exact'].max())
    global_max_age = max(max_part_age, max_cont_age)
    
    # Build the rules dictionary
    rules = {
        "valid_settings": [str(s) for s in settings],
        "valid_frequencies": [str(f) for f in freqs],
        "valid_distances": [str(d) for d in dists],
        "max_age_digits": len(str(global_max_age))
    }
    
    with open(out_file, "w") as f:
        json.dump(rules, f, indent=2)
        
    print(f"✅ Extracted Rules Saved to: {out_file}")
    print(f"   Settings: {rules['valid_settings']}")
    print(f"   Frequencies: {rules['valid_frequencies']}")
    print(f"   Distances: {rules['valid_distances']}")
    print(f"   Max Age Digits: {rules['max_age_digits']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Point this to your original master CSV
    parser.add_argument("--master_csv", type=str, default="data/processed/master_data.csv")
    args = parser.parse_args()
    
    extract_empirical_bounds(args.master_csv)