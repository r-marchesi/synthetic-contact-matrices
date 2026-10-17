import os
import sys
import json
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
EXP04_DIR = os.path.abspath(os.path.join(EXP_DIR, "../04_causal_hierarchy"))
EXP04_SCRIPTS = os.path.join(EXP04_DIR, "scripts")
OUT_DIR = os.path.join(EXP_DIR, "data")

sys.path.append(EXP04_SCRIPTS)
from core import parse_causal_completion, PLAN_REGEX


def safe_float(val, default=-1.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def age_life_stage(age_val):
    """Buckets numeric age into epidemiological life stages for exact cohort matching."""
    if age_val < 0:
        return "unknown"
    if age_val < 6:
        return "infant_toddler"
    if age_val < 19:
        return "school_age"
    if age_val < 30:
        return "young_adult"
    if age_val < 65:
        return "working_adult"
    return "senior"


def build_tabular_index(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    train_file = os.path.join(EXP04_DIR, "data", f"train_causal_{args.split_tag}.jsonl")

    if not os.path.exists(train_file):
        raise FileNotFoundError(
            f"Training file not found at {train_file}. "
            f"Run experiments/04_causal_hierarchy/scripts/01_prepare_data.py first."
        )

    print(f"Loading structured training records from: {train_file}")
    index_records = []

    with open(train_file, "r") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            rec = json.loads(line)
            demos = rec["demos"]
            comp = rec["completion"]
            parsed = parse_causal_completion(rec["part_id"], comp)

            is_zero = parsed[0]["is_zero_contact"]
            degree = 0 if is_zero else len(parsed)

            m = PLAN_REGEX.search(comp)
            plan_line = m.group(0) if m else f"Total: {degree} | Home: 0 | Work: 0 | School: 0 | Other: 0"

            p_age = safe_float(demos.get("age"), default=-1.0)
            hh_size = safe_float(demos.get("hh_size"), default=1.0)

            index_records.append({
                "idx": idx,
                "part_id": str(rec["part_id"]),
                "demos": demos,
                "age_num": p_age,
                "life_stage": age_life_stage(p_age),
                "hh_size_num": hh_size,
                "prompt": rec["prompt"],
                "completion": comp,
                "plan_line": plan_line,
                "degree": degree,
            })

    out_path = os.path.join(OUT_DIR, f"tabular_index_{args.split_tag}.json")
    with open(out_path, "w") as f:
        json.dump(index_records, f, indent=2)

    degrees = [r["degree"] for r in index_records]
    print(f"✅ Saved {len(index_records)} indexed training profiles -> {out_path}")
    print(
        f"   Degree summary: Mean={np.mean(degrees):.2f} | "
        f"Median={np.median(degrees):.1f} | P90={np.percentile(degrees, 90):.1f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_tag", type=str, default="100_percent")
    args = parser.parse_args()
    build_tabular_index(args)