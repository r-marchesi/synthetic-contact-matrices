import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OrdinalEncoder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
EXP04_SCRIPTS = os.path.abspath(os.path.join(EXP_DIR, "../04_causal_hierarchy/scripts"))
DATA_DIR = os.path.join(EXP_DIR, "data")

sys.path.append(EXP04_SCRIPTS)
from core import parse_causal_completion, compute_age_gap, PLAN_REGEX


def is_absent_flag(val_str):
    """Returns True if work_today / school_today explicitly indicates non-attendance."""
    s = str(val_str).strip().lower()
    return s in ["no", "0", "false", "not present", "absent", "none"]


def compute_physics_penalty(demos, completion_text, parsed_rows):
    """
    Evaluates domain & cross-variable consistency of a single diary.
    Returns a non-negative penalty (0.0 = perfect physical consistency).
    """
    penalty = 0.0

    # 1. Syntax validity check
    if not all(r["valid_syntax"] for r in parsed_rows):
        return 10.0

    is_zero = parsed_rows[0]["is_zero_contact"]
    actual_contacts = [] if is_zero else parsed_rows
    actual_total = len(actual_contacts)

    # 2. Check [PLAN] header vs actual [CONTACTS] counts
    m = PLAN_REGEX.search(str(completion_text))
    if not m:
        penalty += 3.0
    else:
        p_tot, p_home, p_work, p_school, p_other = [int(m.group(i)) for i in range(1, 6)]
        if p_tot != (p_home + p_work + p_school + p_other):
            penalty += 2.0
        if p_tot != actual_total:
            penalty += 2.0

    # Count settings
    setting_counts = {"Home": 0, "Work": 0, "School": 0, "Other": 0}
    for r in actual_contacts:
        s = r["Contact Setting"]
        if s in setting_counts:
            setting_counts[s] += 1

    # 3. Work / School attendance gating
    if is_absent_flag(demos.get("work_today")) and setting_counts["Work"] > 0:
        penalty += 0.8 * setting_counts["Work"]
    if is_absent_flag(demos.get("school_today")) and setting_counts["School"] > 0:
        penalty += 0.8 * setting_counts["School"]

    # 4. Household size vs Home contacts consistency
    try:
        hh_size = int(float(demos.get("hh_size", 1)))
        if hh_size == 1 and setting_counts["Home"] > 2:
            penalty += 0.5 * (setting_counts["Home"] - 2)
        elif hh_size > 1 and setting_counts["Home"] > (hh_size + 3):
            penalty += 0.4 * (setting_counts["Home"] - (hh_size + 3))
    except (ValueError, TypeError):
        pass

    # 5. Within-contact Age_Gap & Role-Age consistency
    p_age_str = demos.get("age", "Unknown")
    for r in actual_contacts:
        c_age_str = r["Contact Age"]
        declared_gap = r["Age_Gap"]
        expected_gap = compute_age_gap(p_age_str, c_age_str)
        if expected_gap != "Unknown" and declared_gap != expected_gap:
            penalty += 0.4

        # Check obvious biological role-age inversions if relationship is explicit
        rel_lower = str(r["Relationship to Participant"]).lower()
        try:
            p_age = float(p_age_str)
            c_age = float(c_age_str)
            if "child" in rel_lower and c_age >= p_age - 10:
                penalty += 0.6
            elif "parent" in rel_lower and c_age <= p_age + 10:
                penalty += 0.6
        except (ValueError, TypeError):
            pass

    return penalty


def build_diary_feature_table(records):
    """
    Builds a flat table of (participant, diary_id, contact_rows) with engineered
    cross-variable interaction features so XGBoost can score diary realism.
    """
    rows = []
    for rec in records:
        pid = rec["part_id"]
        demos = rec["demos"]

        # Diary 0 = Ground Truth, Diary 1..K = Candidates
        all_diaries = [("gt", rec["gt_completion"])] + [
            (f"cand_{i}", c) for i, c in enumerate(rec["candidates"])
        ]

        for diary_tag, comp_text in all_diaries:
            parsed = parse_causal_completion(pid, comp_text)
            valid = all(r["valid_syntax"] for r in parsed)
            is_zero = parsed[0]["is_zero_contact"] if valid else False
            vol = 0 if (not valid or is_zero) else len(parsed)

            try:
                p_age = float(demos.get("age", -1))
            except (ValueError, TypeError):
                p_age = -1.0
            try:
                hh_size = float(demos.get("hh_size", 1))
            except (ValueError, TypeError):
                hh_size = 1.0

            home_cnt = sum(1 for r in parsed if not is_zero and r["Contact Setting"] == "Home")

            for r in parsed:
                try:
                    c_age = float(r["Contact Age"]) if not is_zero else -1.0
                except (ValueError, TypeError):
                    c_age = -1.0

                age_diff = (c_age - p_age) if (c_age >= 0 and p_age >= 0) else -999.0

                rows.append({
                    "part_id": pid,
                    "diary_key": f"{pid}__{diary_tag}",
                    "is_synthetic": 0 if diary_tag == "gt" else 1,
                    "p_age": p_age,
                    "p_gender": str(demos.get("gender")),
                    "hh_size": hh_size,
                    "work_today": str(demos.get("work_today")),
                    "school_today": str(demos.get("school_today")),
                    "occupation": str(demos.get("occupation")),
                    "education": str(demos.get("education")),
                    "income": str(demos.get("income")),
                    "region": str(demos.get("region")),
                    "part_volume": vol,
                    "home_to_hh_ratio": home_cnt / max(1.0, hh_size),
                    "setting": r["Contact Setting"],
                    "location": r["Location of Contact"],
                    "relation": r["Relationship to Participant"],
                    "age_gap": r["Age_Gap"],
                    "c_age": c_age,
                    "age_diff": age_diff,
                    "c_gender": r["Contact Gender"],
                    "freq": r["Contact Frequency"],
                    "phys": r["Physical Contact"],
                    "dist": r["Distance during Contact"],
                })
    return pd.DataFrame(rows)


def compute_oof_c2st_realism(flat_df):
    """
    Fits a 5-fold StratifiedGroupKFold XGBoost classifier on the training diaries
    and returns a dictionary mapping diary_key -> realism_score in [0, 1].
    """
    num_cols = ["p_age", "hh_size", "part_volume", "home_to_hh_ratio", "c_age", "age_diff"]
    cat_cols = [
        "p_gender", "work_today", "school_today", "occupation", "education",
        "income", "region", "setting", "location", "relation", "age_gap",
        "c_gender", "freq", "phys", "dist",
    ]
    feature_cols = num_cols + cat_cols

    X = flat_df[feature_cols].copy()
    X[cat_cols] = X[cat_cols].astype(str).apply(lambda s: s.str.strip().str.lower())
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[cat_cols] = enc.fit_transform(X[cat_cols])

    y = flat_df["is_synthetic"].values
    groups = flat_df["part_id"].astype(str).values

    # Weight positive/negative classes equally since there are K candidates per 1 GT
    n_pos = np.sum(y == 1)
    n_neg = max(1, np.sum(y == 0))
    scale_pos_weight = float(n_neg) / float(max(1, n_pos))

    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_synth_prob = np.zeros(len(flat_df))

    for tr_idx, te_idx in sgkf.split(X, y, groups):
        clf = xgb.XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
        )
        clf.fit(X.iloc[tr_idx], y[tr_idx])
        oof_synth_prob[te_idx] = clf.predict_proba(X.iloc[te_idx])[:, 1]

    flat_df = flat_df.copy()
    flat_df["realism_score"] = 1.0 - oof_synth_prob
    return flat_df.groupby("diary_key")["realism_score"].mean().to_dict()


def build_preference_pairs(args):
    cand_file = os.path.join(DATA_DIR, f"candidates_{args.split_tag}.jsonl")
    records = []
    with open(cand_file, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} participant candidate pools. Training OOF C2ST Critic...")
    flat_df = build_diary_feature_table(records)
    realism_map = compute_oof_c2st_realism(flat_df)

    dpo_pairs = []
    stats_chosen_cand = 0
    stats_chosen_gt = 0

    for rec in records:
        pid = rec["part_id"]
        demos = rec["demos"]
        prompt = rec["prompt"]
        gt_comp = rec["gt_completion"]
        gt_parsed = parse_causal_completion(pid, gt_comp)
        gt_vol = 0 if gt_parsed[0]["is_zero_contact"] else len(gt_parsed)

        scored_cands = []
        for i, cand_text in enumerate(rec["candidates"]):
            d_key = f"{pid}__cand_{i}"
            parsed = parse_causal_completion(pid, cand_text)
            phys_pen = compute_physics_penalty(demos, cand_text, parsed)
            c2st_realism = realism_map.get(d_key, 0.5)

            cand_vol = 0 if (not parsed[0]["valid_syntax"] or parsed[0]["is_zero_contact"]) else len(parsed)
            vol_dev_pen = 0.05 * abs(cand_vol - gt_vol)

            # Composite utility: higher is better
            total_reward = (2.0 * c2st_realism) - phys_pen - vol_dev_pen
            scored_cands.append({
                "text": cand_text,
                "reward": total_reward,
                "phys_pen": phys_pen,
                "realism": c2st_realism,
            })

        scored_cands.sort(key=lambda x: x["reward"], reverse=True)
        best_cand = scored_cands[0]
        worst_cand = scored_cands[-1]

        # Skip if all candidates are identical
        if best_cand["text"].strip() == worst_cand["text"].strip():
            continue

        # Pair 1: Best clean candidate (if physics-valid) vs Worst candidate;
        # otherwise use Ground Truth as the chosen anchor vs Worst candidate
        if best_cand["phys_pen"] == 0.0 and (best_cand["reward"] - worst_cand["reward"]) >= 0.15:
            chosen_text = best_cand["text"]
            stats_chosen_cand += 1
        else:
            chosen_text = gt_comp
            stats_chosen_gt += 1

        if chosen_text.strip() != worst_cand["text"].strip():
            dpo_pairs.append({
                "part_id": pid,
                "prompt": prompt,
                "chosen": chosen_text,
                "rejected": worst_cand["text"],
                "reward_margin": float(best_cand["reward"] - worst_cand["reward"]),
                "worst_phys_pen": float(worst_cand["phys_pen"]),
                "worst_realism": float(worst_cand["realism"]),
            })

    out_file = os.path.join(DATA_DIR, f"dpo_pairs_{args.split_tag}.jsonl")
    with open(out_file, "w") as f:
        for p in dpo_pairs:
            f.write(json.dumps(p) + "\n")

    print(f"\n✅ Built {len(dpo_pairs)} preference pairs -> {out_file}")
    print(f"   - Chosen from best synthetic candidate: {stats_chosen_cand}")
    print(f"   - Chosen from ground-truth anchor:      {stats_chosen_gt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_tag", type=str, default="100_percent")
    args = parser.parse_args()

    build_preference_pairs(args)