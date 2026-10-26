import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(EXP_DIR, "data")
GEN_DIR = os.path.join(EXP_DIR, "generated")
OUT_DIR = os.path.join(EXP_DIR, "results", "rq2_subgroups")

sys.path.append(SCRIPT_DIR)
from importlib import import_module
eval04 = import_module("04_evaluate")


def assign_demographic_strata(demos_df):
    """Creates clean interpretable subgroup buckets across all 9 participant variables."""
    df = demos_df.copy()

    # 1. Age brackets
    age_num = pd.to_numeric(df["age"], errors="coerce")
    df["Age_Bracket"] = pd.cut(
        age_num,
        bins=[-1, 18, 34, 49, 64, 120],
        labels=["0-18", "19-34", "35-49", "50-64", "65+"],
    ).astype(str)

    # 2. Household Size brackets
    hh_num = pd.to_numeric(df["hh_size"], errors="coerce").fillna(1)
    df["HH_Size_Bracket"] = pd.cut(
        hh_num,
        bins=[0, 1, 2, 4, 100],
        labels=["1", "2", "3-4", "5+"],
    ).astype(str)

    # 3. Clean categorical strata
    df["Gender"] = df["gender"].astype(str)
    df["Work_Today"] = df["work_today"].astype(str)
    df["School_Today"] = df["school_today"].astype(str)
    df["Education"] = df["education"].astype(str)
    df["Income"] = df["income"].astype(str)
    df["Occupation"] = df["occupation"].astype(str)
    df["Region_NUTS1"] = df["region"].astype(str)

    return df


def compute_subpopulation_vector_mae(gt_sub, gen_sub, sub_pids):
    """
    Computes the 1x15 average contact rate vector across the 15 contact age bins
    for a specific participant subgroup and returns the MAE between GT and Generated.
    """
    n_parts = len(sub_pids)
    if n_parts == 0:
        return np.nan

    def get_rate_vec(df):
        vec = np.zeros(15)
        valid = df[(df["valid_syntax"] == True) & (df["is_zero_contact"] == False)]
        for _, row in valid.iterrows():
            c_bin = eval04.get_age_bin(row["Contact Age"])
            if c_bin is not None:
                vec[c_bin] += 1.0
        return vec / float(n_parts)

    v_gt = get_rate_vec(gt_sub)
    v_gen = get_rate_vec(gen_sub)
    return float(np.mean(np.abs(v_gt - v_gen)))


def plot_rq2_diagnostics(demos_strata, gt_parsed, gen_parsed, out_dir):
    """Generates a 4-panel epidemiological conditioning & subgroup fidelity figure."""
    valid_gt = gt_parsed[(gt_parsed["valid_syntax"] == True) & (gt_parsed["is_zero_contact"] == False)].copy()
    valid_gen = gen_parsed[(gen_parsed["valid_syntax"] == True) & (gen_parsed["is_zero_contact"] == False)].copy()

    valid_gt["c_age_num"] = pd.to_numeric(valid_gt["Contact Age"], errors="coerce")
    valid_gen["c_age_num"] = pd.to_numeric(valid_gen["Contact Age"], errors="coerce")

    pp = demos_strata.copy()
    pp["p_age_num"] = pd.to_numeric(pp["age"], errors="coerce")
    pp["hh_num"] = pd.to_numeric(pp["hh_size"], errors="coerce").fillna(1).clip(upper=6)

    # Home contacts per participant
    gt_home = valid_gt[valid_gt["Location of Contact"].astype(str).str.contains("Home", case=False, na=False)]
    gen_home = valid_gen[valid_gen["Location of Contact"].astype(str).str.contains("Home", case=False, na=False)]
    pp["GT_Home"] = pp["part_id"].map(gt_home.groupby("part_id").size()).fillna(0)
    pp["Gen_Home"] = pp["part_id"].map(gen_home.groupby("part_id").size()).fillna(0)

    # Work contacts per participant
    gt_work = valid_gt[valid_gt["Location of Contact"].astype(str).str.contains("Work", case=False, na=False)]
    gen_work = valid_gen[valid_gen["Location of Contact"].astype(str).str.contains("Work", case=False, na=False)]
    pp["GT_Work"] = pp["part_id"].map(gt_work.groupby("part_id").size()).fillna(0)
    pp["Gen_Work"] = pp["part_id"].map(gen_work.groupby("part_id").size()).fillna(0)

    # Total contacts per participant
    pp["GT_Total"] = pp["part_id"].map(valid_gt.groupby("part_id").size()).fillna(0)
    pp["Gen_Total"] = pp["part_id"].map(valid_gen.groupby("part_id").size()).fillna(0)

    # Mean contact age per participant
    pp["GT_Mean_CAge"] = pp["part_id"].map(valid_gt.groupby("part_id")["c_age_num"].mean())
    pp["Gen_Mean_CAge"] = pp["part_id"].map(valid_gen.groupby("part_id")["c_age_num"].mean())

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.suptitle("RQ2: Conditioning Fidelity & Subgroup Performance (Exp 04 Epoch 6)", fontsize=18, y=1.02)

    # Panel A: Household Size -> Home Contacts
    hh_grp = pp.groupby("hh_num")[["GT_Home", "Gen_Home"]].mean()
    hh_grp.plot(kind="bar", ax=axes[0, 0], color=["#1f77b4", "#ff7f0e"], alpha=0.88)
    axes[0, 0].set_title("A. Home Contacts by Household Size", fontsize=14)
    axes[0, 0].set_xlabel("Household Size (clipped at 6+)")
    axes[0, 0].set_ylabel("Mean Daily Home Contacts")
    axes[0, 0].set_xticklabels([f"{int(x)}" if x < 6 else "6+" for x in hh_grp.index], rotation=0)
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.5)

    # Panel B: Participant Age -> Mean Contact Age (Assortativity)
    pp["age_decade"] = (pp["p_age_num"] // 10) * 10
    age_grp = pp.groupby("age_decade")[["GT_Mean_CAge", "Gen_Mean_CAge"]].mean()
    age_grp.plot(kind="line", marker="o", linewidth=2.5, ax=axes[0, 1], color=["#1f77b4", "#ff7f0e"])
    axes[0, 1].set_title("B. Age Assortativity (Participant Age vs. Mean Contact Age)", fontsize=14)
    axes[0, 1].set_xlabel("Participant Age Decade")
    axes[0, 1].set_ylabel("Mean Contact Age (Years)")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # Panel C: Work_Today -> Work Contacts
    work_grp = pp.groupby("Work_Today")[["GT_Work", "Gen_Work"]].mean()
    work_grp.plot(kind="bar", ax=axes[1, 0], color=["#1f77b4", "#ff7f0e"], alpha=0.88)
    axes[1, 0].set_title("C. Work Contacts by Daily Work Presence (Work_Today)", fontsize=14)
    axes[1, 0].set_xlabel("Present at Work Today")
    axes[1, 0].set_ylabel("Mean Daily Work Contacts")
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=20, ha="right")
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.5)

    # Panel D: Total Contact Volume by Age Bracket
    age_vol = pp.reindex(columns=["Age_Bracket", "GT_Total", "Gen_Total"]).groupby("Age_Bracket")[["GT_Total", "Gen_Total"]].mean()
    age_order = [b for b in ["0-18", "19-34", "35-49", "50-64", "65+"] if b in age_vol.index]
    age_vol.loc[age_order].plot(kind="bar", ax=axes[1, 1], color=["#1f77b4", "#ff7f0e"], alpha=0.88)
    axes[1, 1].set_title("D. Mean Total Contacts by Participant Age Bracket", fontsize=14)
    axes[1, 1].set_xlabel("Participant Age Bracket")
    axes[1, 1].set_ylabel("Mean Daily Contacts")
    axes[1, 1].set_xticklabels(age_order, rotation=0)
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "rq2_conditioning_and_subgroups.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved RQ2 4-panel diagnostic figure -> {plot_path}")


def run_rq2_evaluation(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    val_df = pd.read_json(args.val_file, lines=True)
    val_df["part_id"] = val_df["part_id"].astype(str).str.strip()

    demos_df = pd.DataFrame(val_df["demos"].tolist())
    demos_df["part_id"] = demos_df["part_id"].astype(str).str.strip()
    demos_strata = assign_demographic_strata(demos_df)

    gt_parsed = eval04.parse_jsonl_records(val_df, col_name="completion")
    gen_jsonl = pd.read_json(args.gen_file, lines=True)
    gen_jsonl["part_id"] = gen_jsonl["part_id"].astype(str).str.strip()
    gen_parsed = eval04.parse_jsonl_records(gen_jsonl, col_name="generated_completion")

    strat_Variables = [
        "Age_Bracket",
        "Gender",
        "HH_Size_Bracket",
        "Work_Today",
        "School_Today",
        "Education",
        "Income",
        "Occupation",
        "Region_NUTS1",
    ]

    rows = []
    for var in strat_Variables:
        for level, grp in demos_strata.groupby(var):
            if str(level).lower() in ["nan", "none"] or len(grp) < 15:
                continue
            sub_pids = sorted(grp["part_id"].unique())
            gt_sub = gt_parsed[gt_parsed["part_id"].isin(sub_pids)]
            gen_sub = gen_parsed[gen_parsed["part_id"].isin(sub_pids)]

            gt_vols = eval04.get_volume_array(gt_sub, sub_pids)
            gen_vols = eval04.get_volume_array(gen_sub, sub_pids)

            vec_mae = compute_subpopulation_vector_mae(gt_sub, gen_sub, sub_pids)
            vol_tvd = eval04.calc_tvd(gt_vols, gen_vols)
            vol_wass = float(wasserstein_distance(gt_vols, gen_vols))

            rows.append({
                "Dimension": var,
                "Subgroup": str(level),
                "N_Participants": len(sub_pids),
                "Contact_Rate_MAE": vec_mae,
                "Mean_Vol_GT": float(gt_vols.mean()),
                "Mean_Vol_Gen": float(gen_vols.mean()),
                "Vol_Diff": float(gen_vols.mean() - gt_vols.mean()),
                "Volume_TVD": vol_tvd,
                "Volume_Wass": vol_wass,
            })

    rq2_df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "rq2_subgroup_leaderboard.csv")
    rq2_df.to_csv(out_csv, index=False)

    plot_rq2_diagnostics(demos_strata, gt_parsed, gen_parsed, OUT_DIR)

    print("\n==========================================================================================")
    print("📊 RQ2 SUBGROUP PERFORMANCE BREAKDOWN (Exp 04: 100% Plan-Guided Epoch 6)")
    print("==========================================================================================")
    print(rq2_df.to_string(index=False, float_format="%.4f"))
    print(f"\nSaved full RQ2 table to: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_file", type=str, default=os.path.join(DATA_DIR, "val_causal.jsonl"))
    parser.add_argument(
        "--gen_file",
        type=str,
        default=os.path.join(GEN_DIR, "generated_100_percent_plan_guided_epoch_6.jsonl"),
    )
    args = parser.parse_args()
    run_rq2_evaluation(args)