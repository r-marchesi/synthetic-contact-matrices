import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(EXP_DIR, "data")
GEN_DIR = os.path.join(EXP_DIR, "generated")
DEEP_DIR = os.path.join(EXP_DIR, "results", "deep_analysis")

sys.path.append(SCRIPT_DIR)
from core import parse_causal_completion

CONTACT_TUPLE_KEYS = [
    "Location of Contact",
    "Contact Setting",
    "Relationship to Participant",
    "Contact Age",
    "Contact Gender",
    "Contact Frequency",
    "Physical Contact",
    "Distance during Contact",
]

DEMO_TUPLE_KEYS = [
    "age", "gender", "hh_size", "work_today", "school_today",
    "occupation", "education", "income", "region",
]


def contact_to_tuple(c_row):
    return tuple(str(c_row.get(k, "Unknown")).strip().lower() for k in CONTACT_TUPLE_KEYS)


def demos_to_tuple(demos_dict):
    return tuple(str(demos_dict.get(k, "Unknown")).strip().lower() for k in DEMO_TUPLE_KEYS)


def extract_diaries_and_contacts(df_jsonl, comp_col, demos_map=None):
    """
    Returns:
      - contact_tuples: list of all valid non-zero 8-attribute contact tuples
      - diaries: dict mapping part_id -> canonical sorted tuple of contact_tuples
      - demo_diaries: list of (demo_tuple, diary_tuple) for non-zero diaries
    """
    contact_tuples = []
    diaries = {}
    demo_diaries = []

    for _, row in df_jsonl.iterrows():
        pid = str(row["part_id"]).strip()
        parsed = parse_causal_completion(pid, row.get(comp_col, ""))
        if not parsed or not parsed[0]["valid_syntax"] or parsed[0]["is_zero_contact"]:
            diaries[pid] = tuple()
            continue

        c_list = [contact_to_tuple(r) for r in parsed]
        contact_tuples.extend(c_list)
        canonical_diary = tuple(sorted(c_list))
        diaries[pid] = canonical_diary

        if demos_map and pid in demos_map:
            demo_diaries.append((demos_map[pid], canonical_diary))

    return contact_tuples, diaries, demo_diaries


def compute_dcr_distribution(query_diaries, train_diaries_by_len, min_k=2):
    """
    Computes the minimum normalized Hamming distance (0.0 = exact copy, 1.0 = completely distinct)
    between each query diary (with K >= min_k) and all training diaries of the same degree K.
    """
    min_distances = []

    for pid, q_diary in query_diaries.items():
        k = len(q_diary)
        if k < min_k:
            continue
        candidates = train_diaries_by_len.get(k, [])
        if not candidates:
            min_distances.append(1.0)
            continue

        q_arr = np.array(q_diary)  # shape (k, 8)
        best_dist = 1.0
        for t_arr in candidates:
            # Fraction of attributes that differ across the k sorted contacts
            diff_frac = float(np.mean(q_arr != t_arr))
            if diff_frac < best_dist:
                best_dist = diff_frac
                if best_dist == 0.0:
                    break
        min_distances.append(best_dist)

    return np.array(min_distances, dtype=float)


def run_overfitting_audit(args):
    os.makedirs(DEEP_DIR, exist_ok=True)

    train_df = pd.read_json(args.train_file, lines=True)
    val_df = pd.read_json(args.val_file, lines=True)

    train_demos_map = {str(r["part_id"]).strip(): demos_to_tuple(r["demos"]) for _, r in train_df.iterrows()}
    val_demos_map = {str(r["part_id"]).strip(): demos_to_tuple(r["demos"]) for _, r in val_df.iterrows()}

    train_contacts, train_diaries, train_demo_diaries = extract_diaries_and_contacts(
        train_df, "completion", train_demos_map
    )
    val_gt_contacts, val_gt_diaries, val_gt_demo_diaries = extract_diaries_and_contacts(
        val_df, "completion", val_demos_map
    )

    train_contact_set = set(train_contacts)
    train_diary_set = set(d for d in train_diaries.values() if len(d) > 0)
    train_demo_diary_set = set(train_demo_diaries)

    # Pre-index training diaries by length for fast DCR calculation
    train_diaries_by_len = {}
    for d in set(train_diaries.values()):
        if len(d) >= 2:
            train_diaries_by_len.setdefault(len(d), []).append(np.array(d))

    datasets_to_audit = [("Validation Ground Truth (Baseline)", val_gt_contacts, val_gt_diaries, val_gt_demo_diaries)]

    pg_file = os.path.join(GEN_DIR, f"generated_100_percent_plan_guided_epoch_{args.pg_epoch}.jsonl")
    ff_file = os.path.join(GEN_DIR, f"generated_100_percent_full_free_epoch_{args.ff_epoch}.jsonl")

    for label, fpath in [
        (f"Plan-Guided (Ep {args.pg_epoch})", pg_file),
        (f"Full-Free (Ep {args.ff_epoch})", ff_file),
    ]:
        if os.path.exists(fpath):
            gdf = pd.read_json(fpath, lines=True)
            g_contacts, g_diaries, g_demo_diaries = extract_diaries_and_contacts(
                gdf, "generated_completion", val_demos_map
            )
            datasets_to_audit.append((label, g_contacts, g_diaries, g_demo_diaries))

    summary_rows = []
    degree_rows = []
    dcr_dict = {}

    for label, c_list, d_dict, dd_list in datasets_to_audit:
        # 1. Single-contact level
        total_c = len(c_list)
        unique_c = len(set(c_list))
        in_train_c = sum(1 for c in c_list if c in train_contact_set)
        novel_c = total_c - in_train_c

        # 2. Full diary level (non-zero diaries)
        non_zero_diaries = [d for d in d_dict.values() if len(d) > 0]
        multi_diaries = [d for d in d_dict.values() if len(d) >= 2]
        complex_diaries = [d for d in d_dict.values() if len(d) >= 3]

        nz_matches = sum(1 for d in non_zero_diaries if d in train_diary_set)
        multi_matches = sum(1 for d in multi_diaries if d in train_diary_set)
        complex_matches = sum(1 for d in complex_diaries if d in train_diary_set)

        # 3. Conditioned (Participant 9-Demo + Full Diary) exact matches with Train
        cond_matches = sum(1 for dd in dd_list if dd in train_demo_diary_set)

        # 4. Self-match with held-out Val_GT for the same participant
        if "Ground Truth" not in label:
            self_matches = sum(
                1 for pid, d in d_dict.items()
                if len(d) > 0 and d == val_gt_diaries.get(pid, None)
            )
            self_match_pct = (self_matches / max(1, len(non_zero_diaries))) * 100.0
        else:
            self_match_pct = np.nan

        # 5. Distance to Closest Training Record (DCR) for K >= 2
        dcr_vals = compute_dcr_distribution(d_dict, train_diaries_by_len, min_k=2)
        dcr_dict[label] = dcr_vals

        summary_rows.append({
            "Dataset": label,
            "Total_Contacts": total_c,
            "Unique_8Tuple_Contacts": unique_c,
            "Single_Contact_In_Train_%": (in_train_c / max(1, total_c)) * 100.0,
            "Single_Contact_Novel_%": (novel_c / max(1, total_c)) * 100.0,
            "Diary_Match_Train_% (K>=1)": (nz_matches / max(1, len(non_zero_diaries))) * 100.0,
            "Diary_Match_Train_% (K>=2)": (multi_matches / max(1, len(multi_diaries))) * 100.0,
            "Diary_Match_Train_% (K>=3)": (complex_matches / max(1, len(complex_diaries))) * 100.0,
            "Conditioned_Demo+Diary_Copy_%": (cond_matches / max(1, len(dd_list))) * 100.0,
            "Exact_Val_Self_Match_% (K>=1)": self_match_pct,
            "Mean_DCR_to_Train (K>=2)": float(np.mean(dcr_vals)) if len(dcr_vals) > 0 else np.nan,
        })

        # Breakdown by degree K
        for deg_label, deg_filter in [
            ("K = 1 (Single Contact)", lambda k: k == 1),
            ("K = 2 (Two Contacts)", lambda k: k == 2),
            ("K = 3-4 (Medium Diary)", lambda k: 3 <= k <= 4),
            ("K >= 5 (Large Diary)", lambda k: k >= 5),
        ]:
            sub_d = [d for d in d_dict.values() if deg_filter(len(d))]
            m_cnt = sum(1 for d in sub_d if d in train_diary_set)
            degree_rows.append({
                "Dataset": label,
                "Diary_Degree_Bin": deg_label,
                "N_Diaries": len(sub_d),
                "Exact_Matches_in_Train": m_cnt,
                "Exact_Match_%": (m_cnt / max(1, len(sub_d))) * 100.0,
            })

    df_ovf_sum = pd.DataFrame(summary_rows)
    df_ovf_deg = pd.DataFrame(degree_rows)

    df_ovf_sum.to_csv(os.path.join(DEEP_DIR, "table7_overfitting_summary.csv"), index=False)
    df_ovf_deg.to_csv(os.path.join(DEEP_DIR, "table7b_diary_copy_by_degree.csv"), index=False)

    # Plot 3-panel Overfitting & Privacy Diagnostic Figure
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle("Deep Analysis 7: Overfitting, Memorization & Novelty Audit (Train vs. Val GT vs. Generated)", fontsize=16, y=1.03)

    # Panel A: Single-Contact Overlap vs. Novelty
    plot_a = df_ovf_sum.set_index("Dataset")[["Single_Contact_In_Train_%", "Single_Contact_Novel_%"]]
    plot_a.plot(kind="bar", stacked=True, ax=axes[0], color=["#64748b", "#16a34a"], alpha=0.9)
    axes[0].set_title("A. Single-Contact 8-Tuple Overlap vs. Novelty (%)")
    axes[0].set_ylabel("Percentage of Contacts (%)")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=20, ha="right")
    axes[0].legend(["Seen in Train (Common Archetypes)", "Novel 8-Tuple (Never in Train)"])
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Panel B: Full Diary Exact Match Rate by Degree K
    sns.barplot(
        data=df_ovf_deg, x="Diary_Degree_Bin", y="Exact_Match_%", hue="Dataset", ax=axes[1]
    )
    axes[1].set_title("B. Full Diary Exact Match Rate with Train Set by Degree K")
    axes[1].set_xlabel("Participant Diary Size (K)")
    axes[1].set_ylabel("Exact Diary Match in Train (%)")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=20, ha="right")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    # Panel C: Distance to Closest Training Record (DCR) for K >= 2
    for label, dcr_arr in dcr_dict.items():
        if len(dcr_arr) > 0:
            sns.kdeplot(dcr_arr, ax=axes[2], fill=True, alpha=0.25, linewidth=2, label=f"{label} (Mean={np.mean(dcr_arr):.3f})")
    axes[2].set_title("C. Distance to Closest Training Diary (DCR, K >= 2)")
    axes[2].set_xlabel("Normalized Attribute Hamming Distance to Closest Train Diary")
    axes[2].set_ylabel("Density")
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(DEEP_DIR, "fig7_overfitting_dcr.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n==========================================================================================")
    print("🔍 TABLE 7A: OVERFITTING, MEMORIZATION & NOVELTY SUMMARY")
    print("==========================================================================================")
    print(df_ovf_sum.to_string(index=False, float_format="%.2f"))

    print("\n==========================================================================================")
    print("🔍 TABLE 7B: FULL DIARY EXACT MATCH RATE WITH TRAINING SET BY DEGREE (K)")
    print("==========================================================================================")
    print(df_ovf_deg.to_string(index=False, float_format="%.2f"))
    print(f"\n✅ Saved Overfitting Audit tables & figure to: {DEEP_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_file",
        default=os.path.join(DATA_DIR, "train_causal_100_percent.jsonl"),
    )
    parser.add_argument(
        "--val_file",
        default=os.path.join(DATA_DIR, "val_causal.jsonl"),
    )
    parser.add_argument("--pg_epoch", type=int, default=8)
    parser.add_argument("--ff_epoch", type=int, default=8)
    args = parser.parse_args()

    run_overfitting_audit(args)