import os
import sys
import json
import argparse
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "../.."))
OUT_DIR = os.path.join(EXP_DIR, "data")

sys.path.append(SCRIPT_DIR)
from core import extract_participant_demos, build_prompt, format_causal_completion


def build_split_records(master_df, target_part_ids):
    sub_df = master_df[master_df["part_id"].isin(target_part_ids)].copy()
    records = []

    for part_id, group_df in sub_df.groupby("part_id", sort=False):
        first_row = group_df.iloc[0]
        demos = extract_participant_demos(first_row)
        prompt_str = build_prompt(demos)
        completion_str = format_causal_completion(demos, group_df)

        records.append({
            "part_id": str(part_id),
            "demos": demos,
            "prompt": prompt_str,
            "completion": completion_str,
        })
    return records


def save_jsonl(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--master_csv",
        default=os.path.join(PROJECT_ROOT, "data/processed/master_data.csv"),
    )
    parser.add_argument(
        "--train_ids",
        default=os.path.join(PROJECT_ROOT, "data/splits/train_ids_100_percent.csv"),
    )
    parser.add_argument(
        "--val_ids",
        default=os.path.join(PROJECT_ROOT, "data/splits/val_ids.csv"),
    )
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    master_df = pd.read_csv(args.master_csv, low_memory=False)
    master_df["part_id"] = master_df["part_id"].astype(str).str.strip()

    train_ids = set(pd.read_csv(args.train_ids)["part_id"].astype(str).str.strip())
    val_ids = set(pd.read_csv(args.val_ids)["part_id"].astype(str).str.strip())

    split_tag = (
        os.path.basename(args.train_ids)
        .replace("train_ids_", "")
        .replace(".csv", "")
    )
    train_out = os.path.join(OUT_DIR, f"train_causal_{split_tag}.jsonl")
    val_out = os.path.join(OUT_DIR, "val_causal.jsonl")

    train_records = build_split_records(master_df, train_ids)
    save_jsonl(train_records, train_out)
    print(f"Saved {len(train_records)} semantic training records -> {train_out}")

    if not os.path.exists(val_out):
        val_records = build_split_records(master_df, val_ids)
        save_jsonl(val_records, val_out)
        print(f"Saved {len(val_records)} semantic validation records -> {val_out}")