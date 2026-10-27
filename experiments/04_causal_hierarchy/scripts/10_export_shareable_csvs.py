import os
import sys
import shutil
import argparse
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "../.."))
GEN_DIR = os.path.join(EXP_DIR, "generated")
SHARE_DIR = os.path.join(EXP_DIR, "results", "shareable_csvs")

sys.path.append(SCRIPT_DIR)
from core import parse_causal_completion

PARTICIPANT_COLS = [
    "part_id",
    "part_age_exact",
    "part_gender",
    "hh_size",
    "presence_work",
    "presence_school",
    "educational_attainment",
    "region_nuts1",
    "occupation",
    "household_monthly_net_income",
]

CONTACT_COLS = [
    "cont_id",
    "cnt_age_exact",
    "cnt_gender",
    "frequency_multi",
    "phys_contact",
    "distance",
    "relationship",
    "setting",
    "location_multi",
]

MASTER_SCHEMA_COLS = PARTICIPANT_COLS + CONTACT_COLS


def to_master_na(val, is_numeric=False, is_bool_like=False):
    """Maps DSL 'Unknown'/'None' tokens back to native NaN/typed values matching master_data.csv."""
    if val is None or pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s.lower() in ["", "unknown", "none", "nan", "<na>"]:
        return np.nan
    if is_numeric:
        try:
            return int(round(float(s)))
        except (ValueError, TypeError):
            return np.nan
    if is_bool_like:
        if s.lower() in ["true", "1", "yes"]:
            return True
        if s.lower() in ["false", "0", "no"]:
            return False
    return s


def convert_generated_jsonl_to_master_df(gen_jsonl_path, part_lookup_df, id_prefix="syn"):
    """
    Converts a generated JSONL file into a flat DataFrame with the exact 19-column
    relational structure of master_data.csv.
    """
    gen_df = pd.read_json(gen_jsonl_path, lines=True)
    gen_df["part_id"] = gen_df["part_id"].astype(str).str.strip()

    part_dict = part_lookup_df.set_index("part_id").to_dict(orient="index")
    out_rows = []
    cont_counter = 1

    for _, row in gen_df.iterrows():
        pid = str(row["part_id"]).strip()
        p_info = part_dict.get(pid, {c: "Unknown" for c in PARTICIPANT_COLS if c != "part_id"})
        base_part_row = {"part_id": pid, **p_info}

        parsed_contacts = parse_causal_completion(pid, row.get("generated_completion", ""))

        # Zero-contact participant: emit 1 row with NaN contact columns (exact left-join behavior)
        if not parsed_contacts or parsed_contacts[0]["is_zero_contact"] or not parsed_contacts[0]["valid_syntax"]:
            empty_contact = {c: np.nan for c in CONTACT_COLS}
            out_rows.append({**base_part_row, **empty_contact})
            continue

        for c in parsed_contacts:
            c_row = {
                "cont_id": f"{id_prefix}_{cont_counter}",
                "cnt_age_exact": to_master_na(c["Contact Age"], is_numeric=True),
                "cnt_gender": to_master_na(c["Contact Gender"]),
                "frequency_multi": to_master_na(c["Contact Frequency"], is_numeric=True),
                "phys_contact": to_master_na(c["Physical Contact"], is_bool_like=True),
                "distance": to_master_na(c["Distance during Contact"]),
                "relationship": to_master_na(c["Relationship to Participant"]),
                "setting": to_master_na(c["Contact Setting"]),
                "location_multi": to_master_na(c["Location of Contact"]),
            }
            cont_counter += 1
            out_rows.append({**base_part_row, **c_row})

    result_df = pd.DataFrame(out_rows)
    return result_df.reindex(columns=MASTER_SCHEMA_COLS)


def export_shareable_datasets(args):
    os.makedirs(SHARE_DIR, exist_ok=True)

    print(f"Loading master dataset: {args.master_csv}")
    master_df = pd.read_csv(args.master_csv, low_memory=False)
    master_df["part_id"] = master_df["part_id"].astype(str).str.strip()
    master_df = master_df.reindex(columns=MASTER_SCHEMA_COLS)

    # 1. Copy master_data.csv
    master_out = os.path.join(SHARE_DIR, "master_data.csv")
    master_df.to_csv(master_out, index=False)

    # 2. Extract and save validation_data.csv in the exact same structure
    val_ids_df = pd.read_csv(args.val_ids)
    val_ids_set = set(val_ids_df["part_id"].astype(str).str.strip())
    val_gt_df = master_df[master_df["part_id"].isin(val_ids_set)].copy()
    val_out = os.path.join(SHARE_DIR, "validation_data.csv")
    val_gt_df.to_csv(val_out, index=False)

    # Participant lookup table for generated datasets
    part_lookup_df = master_df.drop_duplicates(subset=["part_id"])[PARTICIPANT_COLS].copy()

    # 3. Export Generated Plan-Guided and Full-Free CSVs
    pg_jsonl = os.path.join(GEN_DIR, f"generated_100_percent_plan_guided_epoch_{args.pg_epoch}.jsonl")
    ff_jsonl = os.path.join(GEN_DIR, f"generated_100_percent_full_free_epoch_{args.ff_epoch}.jsonl")

    exported_summaries = [
        ("master_data.csv", master_df),
        ("validation_data.csv", val_gt_df),
    ]

    if os.path.exists(pg_jsonl):
        pg_df = convert_generated_jsonl_to_master_df(pg_jsonl, part_lookup_df, id_prefix="syn_pg")
        pg_out = os.path.join(SHARE_DIR, f"generated_plan_guided_epoch_{args.pg_epoch}.csv")
        pg_df.to_csv(pg_out, index=False)
        exported_summaries.append((os.path.basename(pg_out), pg_df))

    if os.path.exists(ff_jsonl):
        ff_df = convert_generated_jsonl_to_master_df(ff_jsonl, part_lookup_df, id_prefix="syn_ff")
        ff_out = os.path.join(SHARE_DIR, f"generated_full_free_epoch_{args.ff_epoch}.csv")
        ff_df.to_csv(ff_out, index=False)
        exported_summaries.append((os.path.basename(ff_out), ff_df))

    print("\n==========================================================================================")
    print("📦 SHAREABLE CSV DATASETS EXPORTED (19-COLUMN MASTER SCHEMA)")
    print("==========================================================================================")
    summary_rows = []
    for fname, df_item in exported_summaries:
        n_parts = df_item["part_id"].nunique()
        n_contacts = int(df_item["cont_id"].notna().sum())
        n_zero = int(df_item.groupby("part_id")["cont_id"].apply(lambda s: s.isna().all()).sum())
        summary_rows.append({
            "File_Name": fname,
            "Columns": len(df_item.columns),
            "Total_Rows": len(df_item),
            "Unique_Participants": n_parts,
            "Total_Valid_Contacts": n_contacts,
            "Zero_Contact_Participants": n_zero,
            "Mean_Contacts_Per_Part": n_contacts / max(1, n_parts),
        })
    print(pd.DataFrame(summary_rows).to_string(index=False, float_format="%.4f"))
    print(f"\n✅ All shareable CSV files saved to: {SHARE_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--master_csv",
        default=os.path.join(PROJECT_ROOT, "data", "processed", "master_data.csv"),
    )
    parser.add_argument(
        "--val_ids",
        default=os.path.join(PROJECT_ROOT, "data", "splits", "val_ids.csv"),
    )
    parser.add_argument("--pg_epoch", type=int, default=8)
    parser.add_argument("--ff_epoch", type=int, default=8)
    args = parser.parse_args()

    export_shareable_datasets(args)