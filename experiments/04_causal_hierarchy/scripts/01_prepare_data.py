import os
import sys
import glob
import argparse
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "../.."))
SPLITS_DIR = os.path.join(PROJECT_ROOT, "data", "splits")

sys.path.append(SCRIPT_DIR)
from core import extract_participant_demos, build_prompt, format_causal_completion


def find_default_train_split():
    candidates = [
        "train_ids_100_percent.csv",
        "train_ids_100.csv",
        "train_ids.csv",
    ]
    for name in candidates:
        path = os.path.join(SPLITS_DIR, name)
        if os.path.isfile(path):
            return path
    matches = glob.glob(os.path.join(SPLITS_DIR, "*train*.csv"))
    if matches:
        return sorted(matches)[-1]
    raise FileNotFoundError(f"Could not locate a training split in '{SPLITS_DIR}'.")


def prepare_causal_datasets(master_csv, train_ids_csv, val_ids_csv, exp_dir, split_tag="100_percent"):
    data_out_dir = os.path.join(exp_dir, "data")
    os.makedirs(data_out_dir, exist_ok=True)

    print(f"Loading master CSV: {master_csv}")
    df = pd.read_csv(master_csv, low_memory=False)
    df["part_id"] = df["part_id"].astype(str).str.strip()

    train_ids = set(pd.read_csv(train_ids_csv)["part_id"].astype(str).str.strip())
    val_ids = set(pd.read_csv(val_ids_csv)["part_id"].astype(str).str.strip())

    train_records, val_records = [], []

    for part_id, group in df.groupby("part_id", sort=False):
        demos = extract_participant_demos(group.iloc[0])
        prompt = build_prompt(demos)
        completion = format_causal_completion(demos, group)

        record = {
            "part_id": str(part_id),
            "demos": demos,
            "prompt": prompt,
            "completion": completion,
        }

        if part_id in train_ids:
            train_records.append(record)
        elif part_id in val_ids:
            val_records.append(record)

    train_path = os.path.join(data_out_dir, f"train_causal_{split_tag}.jsonl")
    val_path = os.path.join(data_out_dir, "val_causal.jsonl")

    pd.DataFrame(train_records).to_json(train_path, orient="records", lines=True)
    pd.DataFrame(val_records).to_json(val_path, orient="records", lines=True)

    print(f"✅ Saved {len(train_records)} training records -> {train_path}")
    print(f"✅ Saved {len(val_records)} validation records -> {val_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--master_csv",
        default=os.path.join(PROJECT_ROOT, "data", "processed", "master_data.csv"),
    )
    parser.add_argument("--train_ids", default=None)
    parser.add_argument("--val_ids", default=os.path.join(SPLITS_DIR, "val_ids.csv"))
    parser.add_argument("--exp_dir", default=EXP_DIR)
    args = parser.parse_args()

    resolved_train = args.train_ids or find_default_train_split()
    split_tag = os.path.splitext(os.path.basename(resolved_train))[0].replace("train_ids_", "")
    prepare_causal_datasets(args.master_csv, resolved_train, args.val_ids, args.exp_dir, split_tag)