import json
import argparse
import os
from collections import defaultdict

def extract_ground_truth_schema(validation_file):
    """Dynamically learns the exact schema and allowed categorical domains from real data."""
    required_keys = set()
    categorical_domains = defaultdict(set)
    numeric_ranges = defaultdict(lambda: {"min": float("inf"), "max": float("-inf")})
    expected_types = {}

    with open(validation_file, "r") as f:
        for line in f:
            row = json.loads(line)
            assistant_msg = next((m for m in row["messages"] if m["role"] == "assistant"), None)
            if not assistant_msg:
                continue
            try:
                contacts = json.loads(assistant_msg["content"])
            except Exception:
                continue

            for c in contacts:
                if not isinstance(c, dict):
                    continue
                for k, v in c.items():
                    required_keys.add(k)
                    if isinstance(v, bool):
                        expected_types[k] = bool
                    elif isinstance(v, (int, float)):
                        expected_types[k] = (int, float)
                        numeric_ranges[k]["min"] = min(numeric_ranges[k]["min"], v)
                        numeric_ranges[k]["max"] = max(numeric_ranges[k]["max"], v)
                    elif isinstance(v, str):
                        expected_types[k] = str
                        categorical_domains[k].add(v.strip())

    return required_keys, expected_types, categorical_domains, numeric_ranges

def audit_split(synthetic_file, required_keys, expected_types, categorical_domains, numeric_ranges):
    total_participants = 0
    valid_json_participants = 0
    total_contacts = 0
    
    contacts_with_exact_keys = 0
    contacts_with_valid_types = 0
    contacts_with_valid_categories = 0
    
    hallucinated_keys = set()
    missing_keys = set()
    category_mismatches = defaultdict(set)
    high_degree_outliers = []

    with open(synthetic_file, "r") as f:
        for line in f:
            total_participants += 1
            try:
                row = json.loads(line)
            except Exception:
                continue

            contacts = row.get("generated_contacts")
            if not isinstance(contacts, list):
                continue
            valid_json_participants += 1
            
            # Identify participants causing the degree distribution tail (> 15 contacts)
            if len(contacts) > 15:
                high_degree_outliers.append((row.get("part_id"), len(contacts)))

            for c in contacts:
                total_contacts += 1
                if not isinstance(c, dict):
                    continue

                curr_keys = set(c.keys())
                
                # Check keys
                diff_extra = curr_keys - required_keys
                diff_missing = required_keys - curr_keys
                if diff_extra:
                    hallucinated_keys.update(diff_extra)
                if diff_missing:
                    missing_keys.update(diff_missing)
                if curr_keys == required_keys:
                    contacts_with_exact_keys += 1

                # Check data types
                types_valid = True
                for k, exp_type in expected_types.items():
                    if k in c and not isinstance(c[k], exp_type):
                        types_valid = False
                        break
                if types_valid:
                    contacts_with_valid_types += 1

                # Check categorical domain adherence
                cats_valid = True
                for k, allowed_vals in categorical_domains.items():
                    if k in c and isinstance(c[k], str):
                        clean_val = c[k].strip()
                        if clean_val not in allowed_vals:
                            category_mismatches[k].add(clean_val)
                            cats_valid = False
                if cats_valid:
                    contacts_with_valid_categories += 1

    return {
        "participants": total_participants,
        "valid_json_pct": (valid_json_participants / total_participants * 100) if total_participants else 0,
        "total_contacts": total_contacts,
        "exact_keys_pct": (contacts_with_exact_keys / total_contacts * 100) if total_contacts else 0,
        "valid_types_pct": (contacts_with_valid_types / total_contacts * 100) if total_contacts else 0,
        "valid_cats_pct": (contacts_with_valid_categories / total_contacts * 100) if total_contacts else 0,
        "hallucinated_keys": list(hallucinated_keys)[:5],
        "missing_keys": list(missing_keys)[:5],
        "category_mismatches": {k: list(v)[:3] for k, v in category_mismatches.items()},
        "top_outliers": sorted(high_degree_outliers, key=lambda x: x[1], reverse=True)[:5]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit Synthetic Data Structure and Schema")
    parser.add_argument("--validation_file", type=str, default="data/splits/validation_set.jsonl")
    parser.add_argument("--results_dir", type=str, default="data/results")
    args = parser.parse_args()

    print("Extracting Canonical Schema from Validation Set...")
    req_keys, exp_types, cat_domains, num_ranges = extract_ground_truth_schema(args.validation_file)
    print(f"Target Schema expects {len(req_keys)} keys per contact: {sorted(list(req_keys))}\n")

    splits = ["10_percent", "30_percent", "50_percent", "80_percent", "100_percent"]

    print(f"{'Split':<12} | {'JSON OK %':<10} | {'Exact Keys %':<13} | {'Valid Types %':<14} | {'Valid Cats %':<13}")
    print("-" * 75)

    reports = {}
    for split in splits:
        path = os.path.join(args.results_dir, f"generated_contacts_{split}.jsonl")
        if os.path.exists(path):
            rep = audit_split(path, req_keys, exp_types, cat_domains, num_ranges)
            reports[split] = rep
            print(f"{split:<12} | {rep['valid_json_pct']:<10.1f} | {rep['exact_keys_pct']:<13.1f} | {rep['valid_types_pct']:<14.1f} | {rep['valid_cats_pct']:<13.1f}")

    print("\n" + "=" * 50)
    print("DETAILED SCHEMA DRIFT & OUTLIER REPORT")
    print("=" * 50)
    for split, rep in reports.items():
        print(f"\n--- Split: {split} ---")
        print(f"Total contacts generated: {rep['total_contacts']}")
        if rep["hallucinated_keys"]:
            print(f"  [!] Hallucinated / Typo Keys: {rep['hallucinated_keys']}")
        if rep["missing_keys"]:
            print(f"  [!] Missing Keys: {rep['missing_keys']}")
        if rep["category_mismatches"]:
            print(f"  [!] Off-Domain Categories: {rep['category_mismatches']}")
        if rep["top_outliers"]:
            print(f"  [!] Outliers (>15 contacts): {rep['top_outliers']}")
        if not (rep["hallucinated_keys"] or rep["missing_keys"] or rep["category_mismatches"]):
            print("  [OK] 100% compliant with canonical schema.")