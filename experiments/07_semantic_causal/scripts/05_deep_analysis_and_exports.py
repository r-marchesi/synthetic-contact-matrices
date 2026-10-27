import os
import re
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance, pearsonr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "../.."))
DATA_DIR = os.path.join(EXP_DIR, "data")
GEN_DIR = os.path.join(EXP_DIR, "generated")
RES_DIR = os.path.join(EXP_DIR, "results")
DEEP_DIR = os.path.join(RES_DIR, "deep_analysis")
RQ2_DIR = os.path.join(RES_DIR, "rq2_subgroups")
RQ1_DIR = os.path.join(RES_DIR, "rq1_scaling")
SHARE_DIR = os.path.join(RES_DIR, "shareable_csvs")

sys.path.append(SCRIPT_DIR)
from importlib import import_module
eval07 = import_module("04_evaluate_and_wsmad")
from core import (
    parse_causal_completion,
    INCOME_CODE_TO_LABEL,
    PARTICIPANT_COLS,
    CONTACT_COLS,
    MASTER_SCHEMA_COLS,
)

AGE_LABELS = eval07.AGE_LABELS
DEMO_FEATURES = eval07.DEMO_FEATURES
CONTACT_FEATURES = eval07.CONTACT_FEATURES
MARGINAL_FEATURES = eval07.MARGINAL_C2ST_FEATURES
LOCATIONS = ["Home", "Work", "School", "Leisure", "Transport", "Other"]

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


def select_best_epochs_by_wsmad(leaderboard_csv, pg_override=0, ff_override=0):
    """Automatically selects the epoch with minimum WSMAD_Norm_[0,1] on 100_percent split."""
    pg_ep = pg_override if pg_override > 0 else 8
    ff_ep = ff_override if ff_override > 0 else 8

    if os.path.exists(leaderboard_csv):
        df = pd.read_csv(leaderboard_csv)
        if pg_override <= 0:
            pg_rows = df[df["Run"].astype(str).str.match(r"^100_percent_plan_guided_epoch_\d+$")].sort_values("WSMAD_Norm_[0,1]")
            if not pg_rows.empty:
                pg_ep = int(pg_rows.iloc[0]["Run"].split("_")[-1])
        if ff_override <= 0:
            ff_rows = df[df["Run"].astype(str).str.match(r"^100_percent_full_free_epoch_\d+$")].sort_values("WSMAD_Norm_[0,1]")
            if not ff_rows.empty:
                ff_ep = int(ff_rows.iloc[0]["Run"].split("_")[-1])

    return pg_ep, ff_ep


# ==============================================================================
# 1. VOLUME, LOCATION & COHABITANT / NON-COHABITANT ANALYSIS
# ==============================================================================
def compute_volume_stats(vols, ref_vols=None):
    vols = np.asarray(vols, dtype=float)
    mean_v = float(np.mean(vols))
    var_v = float(np.var(vols))
    return {
        "N_Participants": len(vols),
        "Zero_Contact_%": float(np.mean(vols == 0) * 100.0),
        "Mean": mean_v,
        "SD": float(np.std(vols)),
        "Dispersion_Index (Var/Mean)": float(var_v / mean_v) if mean_v > 0 else np.nan,
        "Median": float(np.median(vols)),
        "IQR": float(np.percentile(vols, 75) - np.percentile(vols, 25)),
        "P90": float(np.percentile(vols, 90)),
        "P99": float(np.percentile(vols, 99)),
        "Max": float(np.max(vols)),
        "Volume_TVD": eval07.calc_tvd(ref_vols, vols) if ref_vols is not None else 0.0,
        "Volume_Wass": float(wasserstein_distance(ref_vols, vols)) if ref_vols is not None else 0.0,
    }


def analyze_volumes(gt_sub, gen_runs_dict, eval_pids, out_dir):
    gt_vols = eval07.get_volume_array(gt_sub, eval_pids)
    rows = [{"Source": "Ground Truth", **compute_volume_stats(gt_vols, gt_vols)}]

    vol_arrays = {"Ground Truth": gt_vols}
    for label, gen_df in gen_runs_dict.items():
        g_vols = eval07.get_volume_array(gen_df, eval_pids)
        vol_arrays[label] = g_vols
        rows.append({"Source": label, **compute_volume_stats(g_vols, gt_vols)})

    df_vol_stats = pd.DataFrame(rows)
    df_vol_stats.to_csv(os.path.join(out_dir, "table1_volume_statistics.csv"), index=False)

    # Location & Cohabitant/Non-Cohabitant volume breakdown
    loc_rows = []
    n_parts = max(1, len(eval_pids))
    for source_name, df_parsed in [("Ground Truth", gt_sub)] + list(gen_runs_dict.items()):
        valid = df_parsed[(df_parsed["valid_syntax"] == True) & (df_parsed["is_zero_contact"] == False)]
        loc_counts = valid.groupby("Location of Contact").size().to_dict()
        is_cohab = valid["Relationship to Participant"].astype(str).str.lower() == "cohabitant"
        n_cohab = int(is_cohab.sum())
        n_non_cohab = int((~is_cohab).sum())

        rec = {
            "Source": source_name,
            "Mean_Cohabitants": float(n_cohab) / n_parts,
            "Mean_Non_Cohabitants": float(n_non_cohab) / n_parts,
        }
        for loc in LOCATIONS:
            cnt = sum(v for k, v in loc_counts.items() if loc.lower() in str(k).lower())
            rec[f"Mean_{loc}"] = float(cnt) / n_parts
        loc_rows.append(rec)

    df_loc_vols = pd.DataFrame(loc_rows)
    df_loc_vols.to_csv(os.path.join(out_dir, "table1b_location_volumes.csv"), index=False)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Deep Analysis 1: Contact Volume, Right Tail & Arena Decomposition (100% Split)", fontsize=18, y=1.02)

    hist_records = [{"Source": src, "Volume": v} for src, arr in vol_arrays.items() for v in arr]
    hist_df = pd.DataFrame(hist_records)
    sns.histplot(
        data=hist_df, x="Volume", hue="Source", multiple="dodge",
        discrete=True, shrink=0.8, ax=axes[0, 0], stat="probability", common_norm=False,
    )
    max_x = int(np.percentile(gt_vols, 99)) + 2
    axes[0, 0].set_xlim(-0.5, max(12, max_x))
    axes[0, 0].set_title("A. Discrete Contact Volume PMF (Clipped at P99)")
    axes[0, 0].set_xlabel("Number of Daily Contacts (k)")
    axes[0, 0].set_ylabel("Probability P(K = k)")
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.5)

    for src, arr in vol_arrays.items():
        x_sorted = np.sort(arr)
        y_ecdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
        axes[0, 1].step(x_sorted, y_ecdf, where="post", linewidth=2.2, label=src)
    axes[0, 1].set_xlim(0, max(15, max_x))
    axes[0, 1].set_title("B. Empirical Cumulative Distribution Function (ECDF)")
    axes[0, 1].set_xlabel("Number of Daily Contacts (k)")
    axes[0, 1].set_ylabel("F(k) = P(K <= k)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    k_grid = np.arange(0, max(20, int(np.max(gt_vols)) + 1))
    for src, arr in vol_arrays.items():
        surv = [np.mean(arr >= k) for k in k_grid]
        axes[1, 0].plot(k_grid, surv, marker="o", markersize=4, linewidth=2.0, label=src)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("C. Right-Tail Complementary CDF P(K >= k) [Log Scale]")
    axes[1, 0].set_xlabel("Degree Threshold (k)")
    axes[1, 0].set_ylabel("P(K >= k) [Log Scale]")
    axes[1, 0].legend()
    axes[1, 0].grid(True, which="both", linestyle="--", alpha=0.5)

    cols_to_plot = ["Mean_Cohabitants", "Mean_Non_Cohabitants"] + [f"Mean_{l}" for l in LOCATIONS]
    clean_labels = ["Cohabitants", "Non-Cohab"] + LOCATIONS
    loc_plot_df = df_loc_vols.set_index("Source")[cols_to_plot].T
    loc_plot_df.index = clean_labels
    loc_plot_df.plot(kind="bar", ax=axes[1, 1], alpha=0.88)
    axes[1, 1].set_title("D. Mean Daily Contacts by Social Role & Location")
    axes[1, 1].set_xlabel("Role / Social Arena")
    axes[1, 1].set_ylabel("Mean Contacts per Participant")
    axes[1, 1].tick_params(axis="x", rotation=20)
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig1_volume_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return df_vol_stats, df_loc_vols


# ==============================================================================
# 2. 15x15 CONTACT MATRICES BY LOCATION & SUBGROUP
# ==============================================================================
def matrix_epidemiological_metrics(gt_mat, gen_mat, N_vec):
    diff = gen_mat - gt_mat
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    flat_gt, flat_gen = gt_mat.flatten(), gen_mat.flatten()
    if np.std(flat_gt) > 0 and np.std(flat_gen) > 0:
        r_val = float(pearsonr(flat_gt, flat_gen)[0])
    else:
        r_val = np.nan

    norm_prod = np.linalg.norm(flat_gt) * np.linalg.norm(flat_gen)
    cos_sim = float(np.dot(flat_gt, flat_gen) / norm_prod) if norm_prod > 0 else np.nan

    rho_gt = float(np.max(np.abs(np.linalg.eigvals(gt_mat))))
    rho_gen = float(np.max(np.abs(np.linalg.eigvals(gen_mat))))
    r0_ratio = float(rho_gen / rho_gt) if rho_gt > 0 else np.nan

    assort_gt = float(np.trace(gt_mat) / np.sum(gt_mat)) if np.sum(gt_mat) > 0 else 0.0
    assort_gen = float(np.trace(gen_mat) / np.sum(gen_mat)) if np.sum(gen_mat) > 0 else 0.0

    recip_gt = eval07.calc_weighted_reciprocity(gt_mat, N_vec)
    recip_gen = eval07.calc_weighted_reciprocity(gen_mat, N_vec)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "Pearson_r": r_val,
        "Cosine_Sim": cos_sim,
        "R0_GT": rho_gt,
        "R0_Gen": rho_gen,
        "R0_Ratio": r0_ratio,
        "Diag_Assort_GT": assort_gt,
        "Diag_Assort_Gen": assort_gen,
        "Reciprocity_GT": recip_gt,
        "Reciprocity_Gen": recip_gen,
    }


def analyze_location_matrices(gt_sub, gen_df, val_part_bins, N_vec, run_label, out_dir):
    settings_to_eval = [("Total", None)] + [(loc, loc) for loc in LOCATIONS]
    rows = []

    fig, axes = plt.subplots(len(settings_to_eval), 3, figsize=(22, 4.5 * len(settings_to_eval)))
    fig.suptitle(f"Deep Analysis 2: 15x15 Contact Matrices by Location ({run_label})", fontsize=20, y=1.01)

    for idx, (loc_name, loc_filter) in enumerate(settings_to_eval):
        gt_m = eval07.build_contact_matrix(gt_sub, val_part_bins, location_filter=loc_filter)
        gen_m = eval07.build_contact_matrix(gen_df, val_part_bins, location_filter=loc_filter)
        m_stats = matrix_epidemiological_metrics(gt_m, gen_m, N_vec)
        rows.append({"Run": run_label, "Matrix_Setting": loc_name, **m_stats})

        vmax = max(np.max(gt_m), np.max(gen_m), 0.05)
        diff_m = gen_m - gt_m
        max_d = max(np.max(np.abs(diff_m)), 0.05)

        sns.heatmap(gt_m, ax=axes[idx, 0], cmap="YlOrRd", vmin=0, vmax=vmax, xticklabels=AGE_LABELS, yticklabels=AGE_LABELS)
        axes[idx, 0].set_title(f"{loc_name}: Ground Truth (R0={m_stats['R0_GT']:.2f}, Diag={m_stats['Diag_Assort_GT']*100:.1f}%)")
        sns.heatmap(gen_m, ax=axes[idx, 1], cmap="YlOrRd", vmin=0, vmax=vmax, xticklabels=AGE_LABELS, yticklabels=AGE_LABELS)
        axes[idx, 1].set_title(f"{loc_name}: Generated (R0={m_stats['R0_Gen']:.2f}, Diag={m_stats['Diag_Assort_Gen']*100:.1f}%)")
        sns.heatmap(diff_m, ax=axes[idx, 2], cmap="coolwarm", center=0, vmin=-max_d, vmax=max_d, xticklabels=AGE_LABELS, yticklabels=AGE_LABELS)
        axes[idx, 2].set_title(f"{loc_name}: Diff (MAE={m_stats['MAE']:.4f}, r={m_stats['Pearson_r']:.3f})")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"fig2_matrices_by_location_{run_label}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    df_loc_mat = pd.DataFrame(rows)
    df_loc_mat.to_csv(os.path.join(out_dir, f"table2_location_matrices_{run_label}.csv"), index=False)
    return df_loc_mat


def analyze_rq2_subgroups_full(gt_sub, gen_df, demos_df, run_label, out_dir, rq2_dir):
    os.makedirs(rq2_dir, exist_ok=True)

    # 1. Major subgroup 15x15 matrices
    sub_defs = [
        ("Gender: Female", demos_df[demos_df["gender"] == "F"]),
        ("Gender: Male", demos_df[demos_df["gender"] == "M"]),
        ("Work: In-person", demos_df[demos_df["work_today"].astype(str).str.contains("In-person", case=False, na=False)]),
        ("Work: Remote/No work", demos_df[demos_df["work_today"].astype(str).isin(["Remote work", "No work"])]),
        ("Work: Not applicable", demos_df[demos_df["work_attendance_sem"].astype(str) == "Not applicable"]),
        ("HH Size: 1-2", demos_df[pd.to_numeric(demos_df["hh_size"], errors="coerce").fillna(1) <= 2]),
        ("HH Size: 3+", demos_df[pd.to_numeric(demos_df["hh_size"], errors="coerce").fillna(1) >= 3]),
        ("Region: North", demos_df[demos_df["region"].astype(str).str.contains("North", case=False, na=False)]),
        ("Region: Centre/South/Islands", demos_df[~demos_df["region"].astype(str).str.contains("North", case=False, na=False)]),
    ]

    mat_rows = []
    fig, axes = plt.subplots(len(sub_defs), 3, figsize=(22, 4.1 * len(sub_defs)))
    fig.suptitle(f"Deep Analysis 3: 15x15 Contact Matrices by Population Subgroup ({run_label})", fontsize=20, y=1.01)

    for idx, (sub_name, sub_d) in enumerate(sub_defs):
        pids = sorted(sub_d["part_id"].unique())
        if len(pids) < 10:
            continue
        p_bins = {r["part_id"]: eval07.get_age_bin(r["age"]) for _, r in sub_d.iterrows()}
        N_vec = np.zeros(15)
        for b in p_bins.values():
            if b is not None:
                N_vec[b] += 1

        gt_s = gt_sub[gt_sub["part_id"].isin(pids)]
        gen_s = gen_df[gen_df["part_id"].isin(pids)]

        gt_m = eval07.build_contact_matrix(gt_s, p_bins)
        gen_m = eval07.build_contact_matrix(gen_s, p_bins)
        m_stats = matrix_epidemiological_metrics(gt_m, gen_m, N_vec)

        gt_v = eval07.get_volume_array(gt_s, pids)
        gen_v = eval07.get_volume_array(gen_s, pids)

        mat_rows.append({
            "Subgroup": sub_name,
            "N_Participants": len(pids),
            "Mean_Vol_GT": float(gt_v.mean()),
            "Mean_Vol_Gen": float(gen_v.mean()),
            "Volume_TVD": eval07.calc_tvd(gt_v, gen_v),
            **m_stats,
        })

        vmax = max(np.max(gt_m), np.max(gen_m), 0.1)
        diff_m = gen_m - gt_m
        max_d = max(np.max(np.abs(diff_m)), 0.1)

        sns.heatmap(gt_m, ax=axes[idx, 0], cmap="YlOrRd", vmin=0, vmax=vmax, xticklabels=AGE_LABELS, yticklabels=AGE_LABELS)
        axes[idx, 0].set_title(f"{sub_name} (N={len(pids)}): GT [Mean Vol={gt_v.mean():.2f}]")
        sns.heatmap(gen_m, ax=axes[idx, 1], cmap="YlOrRd", vmin=0, vmax=vmax, xticklabels=AGE_LABELS, yticklabels=AGE_LABELS)
        axes[idx, 1].set_title(f"{sub_name}: Generated [Mean Vol={gen_v.mean():.2f}]")
        sns.heatmap(diff_m, ax=axes[idx, 2], cmap="coolwarm", center=0, vmin=-max_d, vmax=max_d, xticklabels=AGE_LABELS, yticklabels=AGE_LABELS)
        axes[idx, 2].set_title(f"{sub_name}: Diff [MAE={m_stats['MAE']:.4f}, r={m_stats['Pearson_r']:.3f}]")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"fig3_matrices_by_subgroup_{run_label}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    df_sub_mat = pd.DataFrame(mat_rows)
    df_sub_mat.to_csv(os.path.join(out_dir, f"table3_subgroup_matrices_{run_label}.csv"), index=False)

    # 2. Full 9-dimension RQ2 subgroup table (using explicit attendance & semantic income labels)
    demos_strata = eval07.assign_demographic_strata(demos_df)
    demos_strata["Work_Attendance"] = demos_df["work_attendance_sem"].astype(str)
    demos_strata["School_Attendance"] = demos_df["school_attendance_sem"].astype(str)
    demos_strata["Household_Income"] = demos_df["income"].apply(
        lambda c: f"{c}: {INCOME_CODE_TO_LABEL.get(str(c), 'Unknown')}" if str(c) != "Unknown" else "Unknown"
    )

    strat_vars = [
        "Age_Bracket", "Gender", "HH_Size_Bracket", "Work_Attendance",
        "School_Attendance", "Education", "Household_Income", "Occupation", "Region_NUTS1",
    ]

    rq2_rows = []
    for var in strat_vars:
        for level, grp in demos_strata.groupby(var):
            if str(level).lower() in ["nan", "none"] or len(grp) < 10:
                continue
            sub_pids = sorted(grp["part_id"].unique())
            gt_s = gt_sub[gt_sub["part_id"].isin(sub_pids)]
            gen_s = gen_df[gen_df["part_id"].isin(sub_pids)]

            gt_vols = eval07.get_volume_array(gt_s, sub_pids)
            gen_vols = eval07.get_volume_array(gen_s, sub_pids)

            part_bins_sub = {str(r["part_id"]): eval07.get_age_bin(r["age"]) for _, r in grp.iterrows()}
            M_gt, N_vec = eval07.build_col_respondent_matrix(gt_s, part_bins_sub)
            M_gen, _ = eval07.build_col_respondent_matrix(gen_s, part_bins_sub)

            n_s = max(1.0, float(np.sum(N_vec)))
            col_w = N_vec / n_s
            w_norm = 0.0
            for r_bin in range(15):
                if N_vec[r_bin] > 0:
                    c_gt, c_gen = M_gt[:, r_bin], M_gen[:, r_bin]
                    denom = np.sum(c_gt) + np.sum(c_gen)
                    w_norm += col_w[r_bin] * (np.sum(np.abs(c_gt - c_gen)) / denom if denom > 0 else 0.0)

            rq2_rows.append({
                "Dimension": var,
                "Subgroup": str(level),
                "N_Participants": len(sub_pids),
                "WSMAD_Subgroup_[0,1]": float(w_norm),
                "Mean_Vol_GT": float(gt_vols.mean()),
                "Mean_Vol_Gen": float(gen_vols.mean()),
                "Vol_Diff": float(gen_vols.mean() - gt_vols.mean()),
                "Volume_TVD": eval07.calc_tvd(gt_vols, gen_vols),
                "Volume_Wass": float(wasserstein_distance(gt_vols, gen_vols)),
            })

    df_rq2 = pd.DataFrame(rq2_rows)
    df_rq2.to_csv(os.path.join(rq2_dir, "rq2_subgroup_leaderboard.csv"), index=False)

    # 3. 4-Panel RQ2 Conditioning Diagnostic Plot
    valid_gt = gt_sub[(gt_sub["valid_syntax"] == True) & (gt_sub["is_zero_contact"] == False)].copy()
    valid_gen = gen_df[(gen_df["valid_syntax"] == True) & (gen_df["is_zero_contact"] == False)].copy()
    valid_gt["c_age_num"] = pd.to_numeric(valid_gt["Contact Age"], errors="coerce")
    valid_gen["c_age_num"] = pd.to_numeric(valid_gen["Contact Age"], errors="coerce")

    pp = demos_strata.copy()
    pp["p_age_num"] = pd.to_numeric(pp["age"], errors="coerce")
    pp["hh_num"] = pd.to_numeric(pp["hh_size"], errors="coerce").fillna(1).clip(upper=6)

    gt_home = valid_gt[valid_gt["Location of Contact"] == "Home"]
    gen_home = valid_gen[valid_gen["Location of Contact"] == "Home"]
    pp["GT_Home"] = pp["part_id"].map(gt_home.groupby("part_id").size()).fillna(0)
    pp["Gen_Home"] = pp["part_id"].map(gen_home.groupby("part_id").size()).fillna(0)

    pp["GT_Total"] = pp["part_id"].map(valid_gt.groupby("part_id").size()).fillna(0)
    pp["Gen_Total"] = pp["part_id"].map(valid_gen.groupby("part_id").size()).fillna(0)
    pp["GT_Mean_CAge"] = pp["part_id"].map(valid_gt.groupby("part_id")["c_age_num"].mean())
    pp["Gen_Mean_CAge"] = pp["part_id"].map(valid_gen.groupby("part_id")["c_age_num"].mean())

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.suptitle(f"RQ2: Conditioning Fidelity & Explicit Attendance Breakdown ({run_label})", fontsize=18, y=1.02)

    hh_grp = pp.groupby("hh_num")[["GT_Home", "Gen_Home"]].mean()
    hh_grp.plot(kind="bar", ax=axes[0, 0], color=["#1f77b4", "#ff7f0e"], alpha=0.88)
    axes[0, 0].set_title("A. Mean Home Contacts by Household Size")
    axes[0, 0].set_xlabel("Household Size (clipped at 6+)")
    axes[0, 0].set_ylabel("Mean Daily Home Contacts")
    axes[0, 0].tick_params(axis="x", rotation=0)
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.5)

    pp["age_decade"] = (pp["p_age_num"] // 10) * 10
    age_grp = pp.groupby("age_decade")[["GT_Mean_CAge", "Gen_Mean_CAge"]].mean()
    age_grp.plot(kind="line", marker="o", linewidth=2.5, ax=axes[0, 1], color=["#1f77b4", "#ff7f0e"])
    axes[0, 1].set_title("B. Age Assortativity (Participant Age vs. Mean Contact Age)")
    axes[0, 1].set_xlabel("Participant Age Decade")
    axes[0, 1].set_ylabel("Mean Contact Age (Years)")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    work_grp = pp.groupby("Work_Attendance")[["GT_Total", "Gen_Total"]].mean()
    work_grp.plot(kind="bar", ax=axes[1, 0], color=["#1f77b4", "#ff7f0e"], alpha=0.88)
    axes[1, 0].set_title("C. Mean Total Contacts by Explicit Work Attendance Status")
    axes[1, 0].set_xlabel("Work Attendance Status")
    axes[1, 0].set_ylabel("Mean Daily Total Contacts")
    axes[1, 0].tick_params(axis="x", rotation=20)
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.5)

    age_vol = pp.groupby("Age_Bracket")[["GT_Total", "Gen_Total"]].mean()
    age_order = [b for b in ["0-18", "19-34", "35-49", "50-64", "65+"] if b in age_vol.index]
    age_vol.loc[age_order].plot(kind="bar", ax=axes[1, 1], color=["#1f77b4", "#ff7f0e"], alpha=0.88)
    axes[1, 1].set_title("D. Mean Total Contacts by Participant Age Bracket")
    axes[1, 1].set_xlabel("Participant Age Bracket")
    axes[1, 1].set_ylabel("Mean Daily Contacts")
    axes[1, 1].tick_params(axis="x", rotation=0)
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(rq2_dir, "rq2_conditioning_and_subgroups.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return df_sub_mat, df_rq2


# ==============================================================================
# 3. CORRELATIONS, MARGINALS & C2ST DIAGNOSTICS
# ==============================================================================
def analyze_correlations_and_marginals(gt_joint, gen_joint, run_label, out_dir):
    corr_cols = DEMO_FEATURES + CONTACT_FEATURES

    def prep_df(df):
        d = df[corr_cols].copy()
        for col in ["age", "Contact Age"]:
            num = pd.to_numeric(d[col], errors="coerce").fillna(-5)
            d[col] = (num // 10 * 10).astype(int).astype(str)
        return d.astype(str)

    gt_c = prep_df(gt_joint)
    gen_c = prep_df(gen_joint)

    k = len(corr_cols)
    mat_gt = np.eye(k)
    mat_gen = np.eye(k)
    pair_rows = []

    for i in range(k):
        for j in range(i + 1, k):
            c1, c2 = corr_cols[i], corr_cols[j]
            v_gt = eval07.cramers_v(gt_c[c1], gt_c[c2])
            v_gen = eval07.cramers_v(gen_c[c1], gen_c[c2])
            mat_gt[i, j] = mat_gt[j, i] = v_gt
            mat_gen[i, j] = mat_gen[j, i] = v_gen
            pair_rows.append({
                "Variable_1": c1,
                "Variable_2": c2,
                "Cramers_V_GT": v_gt,
                "Cramers_V_Gen": v_gen,
                "Abs_Diff": abs(v_gt - v_gen),
            })

    diff_mat = np.abs(mat_gt - mat_gen)
    corr_mae = float(np.mean(diff_mat[~np.eye(k, dtype=bool)]))

    fig_c, axes_c = plt.subplots(1, 3, figsize=(24, 7))
    sns.heatmap(mat_gt, ax=axes_c[0], vmin=0, vmax=1, cmap="Blues", xticklabels=corr_cols, yticklabels=corr_cols)
    axes_c[0].set_title("Ground Truth Pairwise Cramér's V")
    sns.heatmap(mat_gen, ax=axes_c[1], vmin=0, vmax=1, cmap="Blues", xticklabels=corr_cols, yticklabels=corr_cols)
    axes_c[1].set_title(f"Generated Pairwise Cramér's V ({run_label})")
    sns.heatmap(diff_mat, ax=axes_c[2], vmin=0, vmax=0.4, cmap="Reds", xticklabels=corr_cols, yticklabels=corr_cols, annot=True, fmt=".2f", annot_kws={"size": 6})
    axes_c[2].set_title(f"|Delta| Cramér's V (MAE = {corr_mae:.4f})")
    for ax in axes_c:
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"fig4a_correlation_cramers_v_{run_label}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_c)

    df_pairs = pd.DataFrame(pair_rows).sort_values("Cramers_V_GT", ascending=False)
    df_pairs.to_csv(os.path.join(out_dir, f"table4_pairwise_correlations_{run_label}.csv"), index=False)

    cat_contact_vars = [c for c in CONTACT_FEATURES if c != "Contact Age"]
    marg_rows = []
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(f"Deep Analysis 4: Marginal Contact Attribute Distributions ({run_label})", fontsize=18, y=1.02)
    axes = axes.flatten()

    for idx, var in enumerate(cat_contact_vars):
        p_gt = gt_joint[var].astype(str).value_counts(normalize=True).rename("Ground Truth")
        p_gen = gen_joint[var].astype(str).value_counts(normalize=True).rename("Generated")
        comp = pd.concat([p_gt, p_gen], axis=1).fillna(0.0)
        tvd_var = float(0.5 * np.sum(np.abs(comp["Ground Truth"] - comp["Generated"])))

        for cat_label, r_val in comp.iterrows():
            marg_rows.append({
                "Variable": var,
                "Category": cat_label,
                "Prop_GT": float(r_val["Ground Truth"]),
                "Prop_Gen": float(r_val["Generated"]),
                "Abs_Diff": float(abs(r_val["Ground Truth"] - r_val["Generated"])),
                "Variable_TVD": tvd_var,
            })

        comp.head(10).plot(kind="bar", ax=axes[idx], color=["#1f77b4", "#ff7f0e"], alpha=0.88)
        axes[idx].set_title(f"{var} (TVD = {tvd_var:.4f})", fontsize=12)
        axes[idx].tick_params(axis="x", rotation=30)
        axes[idx].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"fig4_marginal_distributions_{run_label}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    df_marg = pd.DataFrame(marg_rows)
    df_marg.to_csv(os.path.join(out_dir, f"table4b_marginal_distributions_{run_label}.csv"), index=False)
    return df_pairs, df_marg


def analyze_c2st_suite(gt_joint, gen_joints_dict, out_dir):
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import OrdinalEncoder
    import xgboost as xgb

    def run_detailed_c2st(gt_j, gen_j, feature_cols):
        gt = gt_j[feature_cols + ["part_id"]].copy()
        gen = gen_j[feature_cols + ["part_id"]].copy()
        gt["is_synthetic"] = 0
        gen["is_synthetic"] = 1

        combined = pd.concat([gt, gen], ignore_index=True)
        groups = combined["part_id"].astype(str)
        X = combined[feature_cols].copy()
        y = combined["is_synthetic"].values

        num_cols = [c for c in ["age", "hh_size", "Contact Age", "part_volume"] if c in feature_cols]
        cat_cols = [c for c in feature_cols if c not in num_cols]

        for nc in num_cols:
            X[nc] = pd.to_numeric(X[nc], errors="coerce").fillna(-1.0)
        if cat_cols:
            X[cat_cols] = X[cat_cols].astype(str).apply(lambda s: s.str.strip().str.lower())
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X[cat_cols] = enc.fit_transform(X[cat_cols])

        sgkf = StratifiedGroupKFold(n_splits=5)
        oof_preds = np.zeros(len(combined))
        importances = np.zeros(len(feature_cols))

        for tr_idx, te_idx in sgkf.split(X, y, groups):
            clf = xgb.XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.08,
                subsample=0.8, colsample_bytree=0.8, eval_metric="auc", random_state=42,
            )
            clf.fit(X.iloc[tr_idx], y[tr_idx])
            oof_preds[te_idx] = clf.predict_proba(X.iloc[te_idx])[:, 1]
            importances += clf.feature_importances_ / 5.0

        auc = float(roc_auc_score(y, oof_preds))
        imp_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance_Pct": importances * 100.0,
            "Uniform_Baseline_Pct": 100.0 / len(feature_cols),
        }).sort_values("Importance_Pct", ascending=False)
        return auc, imp_df, y, oof_preds

    joint_18_cols = DEMO_FEATURES + CONTACT_FEATURES
    joint_19_cols = DEMO_FEATURES + CONTACT_FEATURES + ["part_volume"]

    ablation_rows = []
    all_imp_dfs = []

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Deep Analysis 5: Classifier Two-Sample Test (C2ST) Diagnostics & Feature Importance", fontsize=17, y=1.02)

    for label, gen_j in gen_joints_dict.items():
        auc_5, _, _, _ = run_detailed_c2st(gt_joint, gen_j, MARGINAL_FEATURES)
        auc_18, _, _, _ = run_detailed_c2st(gt_joint, gen_j, joint_18_cols)
        auc_19, imp_19, y_true, oof_p = run_detailed_c2st(gt_joint, gen_j, joint_19_cols)

        ablation_rows.append({
            "Run": label,
            "Marginal_5Col_AUC": auc_5,
            "Conditioned_18Col_AUC (No Vol)": auc_18,
            "Full_Joint_19Col_AUC (+Vol)": auc_19,
            "Top_1_Driver": f"{imp_19.iloc[0]['Feature']} ({imp_19.iloc[0]['Importance_Pct']:.1f}%)",
            "Top_2_Driver": f"{imp_19.iloc[1]['Feature']} ({imp_19.iloc[1]['Importance_Pct']:.1f}%)",
            "Top_3_Driver": f"{imp_19.iloc[2]['Feature']} ({imp_19.iloc[2]['Importance_Pct']:.1f}%)",
        })

        imp_19_copy = imp_19.copy()
        imp_19_copy.insert(0, "Run", label)
        all_imp_dfs.append(imp_19_copy)

        if label == list(gen_joints_dict.keys())[0]:
            sns.kdeplot(oof_p[y_true == 0], ax=axes[1], fill=True, label="Ground Truth Rows (y=0)", color="#1f77b4")
            sns.kdeplot(oof_p[y_true == 1], ax=axes[1], fill=True, label=f"Generated Rows ({label}, y=1)", color="#ff7f0e")
            axes[1].axvline(0.50, color="black", linestyle="--", label="Indistinguishable (0.50)")
            axes[1].set_title(f"B. Out-of-Fold Predicted P(Synthetic | x) [{label}, AUC={auc_19:.4f}]")
            axes[1].set_xlabel("XGBoost Out-of-Fold Probability P(is_synthetic = 1)")
            axes[1].set_ylabel("Density")
            axes[1].legend()
            axes[1].grid(True, linestyle="--", alpha=0.5)

    df_c2st_ablation = pd.DataFrame(ablation_rows)
    df_c2st_ablation.to_csv(os.path.join(out_dir, "table5_c2st_ablations.csv"), index=False)

    df_imp_all = pd.concat(all_imp_dfs, ignore_index=True)
    df_imp_all.to_csv(os.path.join(out_dir, "table5b_c2st_feature_importances.csv"), index=False)

    sns.barplot(data=df_imp_all, y="Feature", x="Importance_Pct", hue="Run", ax=axes[0], orient="h")
    axes[0].axvline(100.0 / 19.0, color="red", linestyle="--", label="Uniform Noise Baseline (5.26%)")
    axes[0].set_title("A. Full 19-Feature Joint C2ST Importance Breakdown")
    axes[0].set_xlabel("Relative Gain Importance (%)")
    axes[0].set_ylabel("Participant / Contact Feature")
    axes[0].legend()
    axes[0].grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig5_c2st_diagnostics.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return df_c2st_ablation, df_imp_all


# ==============================================================================
# 4. OVERFITTING, MEMORIZATION & SHAREABLE CSV EXPORT
# ==============================================================================
def run_overfitting_and_csv_exports(train_file, val_file, master_csv, val_ids_csv, pg_ep, ff_ep, deep_dir, share_dir):
    os.makedirs(deep_dir, exist_ok=True)
    os.makedirs(share_dir, exist_ok=True)

    train_df = pd.read_json(train_file, lines=True)
    val_df = pd.read_json(val_file, lines=True)

    def c_tuple(c_row):
        return tuple(str(c_row.get(k, "Unknown")).strip().lower() for k in CONTACT_TUPLE_KEYS)

    def d_tuple(d_dict):
        return tuple(str(d_dict.get(k, "Unknown")).strip().lower() for k in DEMO_FEATURES)

    train_demos_map = {str(r["part_id"]).strip(): d_tuple(r["demos"]) for _, r in train_df.iterrows()}
    val_demos_map = {str(r["part_id"]).strip(): d_tuple(r["demos"]) for _, r in val_df.iterrows()}

    def extract_sets(df_j, col, d_map):
        contacts, diaries, demo_diaries = [], {}, []
        for _, r in df_j.iterrows():
            pid = str(r["part_id"]).strip()
            parsed = parse_causal_completion(pid, r.get(col, ""))
            if not parsed or not parsed[0]["valid_syntax"] or parsed[0]["is_zero_contact"]:
                diaries[pid] = tuple()
                continue
            c_list = [c_tuple(x) for x in parsed]
            contacts.extend(c_list)
            can_d = tuple(sorted(c_list))
            diaries[pid] = can_d
            if d_map and pid in d_map:
                demo_diaries.append((d_map[pid], can_d))
        return contacts, diaries, demo_diaries

    tr_c, tr_d, tr_dd = extract_sets(train_df, "completion", train_demos_map)
    va_c, va_d, va_dd = extract_sets(val_df, "completion", val_demos_map)

    tr_c_set = set(tr_c)
    tr_d_set = set(d for d in tr_d.values() if len(d) > 0)
    tr_dd_set = set(tr_dd)

    tr_by_len = {}
    for d in set(tr_d.values()):
        if len(d) >= 2:
            tr_by_len.setdefault(len(d), []).append(np.array(d))

    datasets = [("Validation Ground Truth (Baseline)", va_c, va_d, va_dd)]
    pg_file = os.path.join(GEN_DIR, f"generated_100_percent_plan_guided_epoch_{pg_ep}.jsonl")
    ff_file = os.path.join(GEN_DIR, f"generated_100_percent_full_free_epoch_{ff_ep}.jsonl")

    for lbl, fp in [(f"Plan-Guided (Ep {pg_ep})", pg_file), (f"Full-Free (Ep {ff_ep})", ff_file)]:
        if os.path.exists(fp):
            gdf = pd.read_json(fp, lines=True)
            gc, gd, gdd = extract_sets(gdf, "generated_completion", val_demos_map)
            datasets.append((lbl, gc, gd, gdd))

    ovf_rows, deg_rows, dcr_dict = [], [], {}
    for lbl, clist, ddict, ddlist in datasets:
        tot_c = len(clist)
        in_tr_c = sum(1 for c in clist if c in tr_c_set)
        nz_d = [d for d in ddict.values() if len(d) > 0]
        m2_d = [d for d in ddict.values() if len(d) >= 2]
        m3_d = [d for d in ddict.values() if len(d) >= 3]

        dcr_vals = []
        for d in m2_d:
            cands = tr_by_len.get(len(d), [])
            if not cands:
                dcr_vals.append(1.0)
                continue
            q_arr = np.array(d)
            best = min(float(np.mean(q_arr != t_arr)) for t_arr in cands)
            dcr_vals.append(best)
        dcr_arr = np.array(dcr_vals, dtype=float)
        dcr_dict[lbl] = dcr_arr

        ovf_rows.append({
            "Dataset": lbl,
            "Total_Contacts": tot_c,
            "Unique_8Tuple_Contacts": len(set(clist)),
            "Single_Contact_In_Train_%": (in_tr_c / max(1, tot_c)) * 100.0,
            "Single_Contact_Novel_%": ((tot_c - in_tr_c) / max(1, tot_c)) * 100.0,
            "Diary_Match_Train_% (K>=1)": (sum(1 for d in nz_d if d in tr_d_set) / max(1, len(nz_d))) * 100.0,
            "Diary_Match_Train_% (K>=2)": (sum(1 for d in m2_d if d in tr_d_set) / max(1, len(m2_d))) * 100.0,
            "Diary_Match_Train_% (K>=3)": (sum(1 for d in m3_d if d in tr_d_set) / max(1, len(m3_d))) * 100.0,
            "Conditioned_Demo+Diary_Copy_%": (sum(1 for dd in ddlist if dd in tr_dd_set) / max(1, len(ddlist))) * 100.0,
            "Mean_DCR_to_Train (K>=2)": float(np.mean(dcr_arr)) if len(dcr_arr) > 0 else np.nan,
        })

        for deg_lbl, fn in [
            ("K = 1 (Single Contact)", lambda k: k == 1),
            ("K = 2 (Two Contacts)", lambda k: k == 2),
            ("K = 3-4 (Medium Diary)", lambda k: 3 <= k <= 4),
            ("K >= 5 (Large Diary)", lambda k: k >= 5),
        ]:
            sub_d = [d for d in ddict.values() if fn(len(d))]
            mc = sum(1 for d in sub_d if d in tr_d_set)
            deg_rows.append({
                "Dataset": lbl,
                "Diary_Degree_Bin": deg_lbl,
                "N_Diaries": len(sub_d),
                "Exact_Matches_in_Train": mc,
                "Exact_Match_%": (mc / max(1, len(sub_d))) * 100.0,
            })

    df_ovf_sum = pd.DataFrame(ovf_rows)
    df_ovf_deg = pd.DataFrame(deg_rows)
    df_ovf_sum.to_csv(os.path.join(deep_dir, "table7_overfitting_summary.csv"), index=False)
    df_ovf_deg.to_csv(os.path.join(deep_dir, "table7b_diary_copy_by_degree.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle("Deep Analysis 7: Overfitting, Memorization & Novelty Audit", fontsize=16, y=1.03)

    plot_a = df_ovf_sum.set_index("Dataset")[["Single_Contact_In_Train_%", "Single_Contact_Novel_%"]]
    plot_a.plot(kind="bar", stacked=True, ax=axes[0], color=["#64748b", "#16a34a"], alpha=0.9)
    axes[0].set_title("A. Single-Contact 8-Tuple Overlap vs. Novelty (%)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    sns.barplot(data=df_ovf_deg, x="Diary_Degree_Bin", y="Exact_Match_%", hue="Dataset", ax=axes[1])
    axes[1].set_title("B. Full Diary Exact Match Rate with Train Set by Degree K")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    for lbl, dcr_arr in dcr_dict.items():
        if len(dcr_arr) > 0:
            sns.kdeplot(dcr_arr, ax=axes[2], fill=True, alpha=0.25, linewidth=2, label=f"{lbl} (Mean={np.mean(dcr_arr):.3f})")
    axes[2].set_title("C. Distance to Closest Training Diary (DCR, K >= 2)")
    axes[2].set_xlabel("Normalized Attribute Hamming Distance to Closest Train Diary")
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(deep_dir, "fig7_overfitting_dcr.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Export Shareable 19-Column CSVs
    if os.path.exists(master_csv):
        m_df = pd.read_csv(master_csv, low_memory=False)
        m_df["part_id"] = m_df["part_id"].astype(str).str.strip()
        m_df = m_df.reindex(columns=MASTER_SCHEMA_COLS)
        m_df.to_csv(os.path.join(share_dir, "master_data.csv"), index=False)

        v_ids = set(pd.read_csv(val_ids_csv)["part_id"].astype(str).str.strip())
        v_df = m_df[m_df["part_id"].isin(v_ids)].copy()
        v_df.to_csv(os.path.join(share_dir, "validation_data.csv"), index=False)

        part_lookup = m_df.drop_duplicates("part_id")[PARTICIPANT_COLS].set_index("part_id").to_dict(orient="index")

        def export_gen_csv(jsonl_p, out_name, pref):
            if not os.path.exists(jsonl_p):
                return
            gdf = pd.read_json(jsonl_p, lines=True)
            rows_out = []
            cnt = 1
            for _, r in gdf.iterrows():
                pid = str(r["part_id"]).strip()
                p_base = {"part_id": pid, **part_lookup.get(pid, {})}
                parsed = parse_causal_completion(pid, r.get("generated_completion", ""))
                if not parsed or parsed[0]["is_zero_contact"] or not parsed[0]["valid_syntax"]:
                    rows_out.append({**p_base, **{c: np.nan for c in CONTACT_COLS}})
                    continue
                for c in parsed:
                    rows_out.append({
                        **p_base,
                        "cont_id": f"{pref}_{cnt}",
                        "cnt_age_exact": int(c["Contact Age"]) if str(c["Contact Age"]).isdigit() else np.nan,
                        "cnt_gender": c["Contact Gender"] if c["Contact Gender"] != "Unknown" else np.nan,
                        "frequency_multi": int(c["Contact Frequency"]) if str(c["Contact Frequency"]).isdigit() else np.nan,
                        "phys_contact": True if c["Physical Contact"] == "True" else (False if c["Physical Contact"] == "False" else np.nan),
                        "distance": c["Distance during Contact"] if c["Distance during Contact"] != "Unknown" else np.nan,
                        "relationship": c["Relationship to Participant"] if c["Relationship to Participant"] != "Unknown" else np.nan,
                        "setting": c["Contact Setting"] if c["Contact Setting"] != "Unknown" else np.nan,
                        "location_multi": c["Location of Contact"] if c["Location of Contact"] != "Unknown" else np.nan,
                    })
                    cnt += 1
            pd.DataFrame(rows_out).reindex(columns=MASTER_SCHEMA_COLS).to_csv(os.path.join(share_dir, out_name), index=False)

        export_gen_csv(pg_file, f"generated_plan_guided_epoch_{pg_ep}.csv", "syn_pg")
        export_gen_csv(ff_file, f"generated_full_free_epoch_{ff_ep}.csv", "syn_ff")

    return df_ovf_sum, df_ovf_deg


def run_all_deep_analyses(args):
    os.makedirs(DEEP_DIR, exist_ok=True)
    os.makedirs(RQ2_DIR, exist_ok=True)
    os.makedirs(RQ1_DIR, exist_ok=True)

    lead_csv = os.path.join(RES_DIR, "leaderboard_07_semantic.csv")
    pg_ep, ff_ep = select_best_epochs_by_wsmad(lead_csv, args.pg_epoch, args.ff_epoch)
    print(f"🎯 Selected Flagship Epochs by WSMAD Stopping: Plan-Guided = Ep {pg_ep} | Full-Free = Ep {ff_ep}")

    val_df = pd.read_json(args.val_file, lines=True)
    val_df["part_id"] = val_df["part_id"].astype(str).str.strip()
    demos_df = pd.DataFrame(val_df["demos"].tolist())
    demos_df["part_id"] = demos_df["part_id"].astype(str).str.strip()
    eval_pids = sorted(val_df["part_id"].unique())

    val_part_bins = {r["part_id"]: eval07.get_age_bin(r["age"]) for _, r in demos_df.iterrows()}
    N_vec = np.zeros(15)
    for b in val_part_bins.values():
        if b is not None:
            N_vec[b] += 1

    gt_sub = eval07.parse_jsonl_records(val_df, col_name="completion")
    gt_vols = eval07.get_volume_array(gt_sub, eval_pids)
    gt_vol_map = dict(zip(eval_pids, gt_vols))
    gt_joint = gt_sub[(gt_sub["valid_syntax"] == True) & (gt_sub["is_zero_contact"] == False)].merge(
        demos_df, on="part_id", how="left"
    )
    gt_joint["part_volume"] = gt_joint["part_id"].map(gt_vol_map)

    pg_file = os.path.join(GEN_DIR, f"generated_100_percent_plan_guided_epoch_{pg_ep}.jsonl")
    ff_file = os.path.join(GEN_DIR, f"generated_100_percent_full_free_epoch_{ff_ep}.jsonl")

    gen_runs_dict, gen_joints_dict = {}, {}
    for label, f_path in [(f"Plan-Guided (Ep {pg_ep})", pg_file), (f"Full-Free (Ep {ff_ep})", ff_file)]:
        if os.path.exists(f_path):
            g_jsonl = pd.read_json(f_path, lines=True)
            g_jsonl["part_id"] = g_jsonl["part_id"].astype(str).str.strip()
            g_parsed = eval07.parse_jsonl_records(g_jsonl, col_name="generated_completion")
            gen_runs_dict[label] = g_parsed

            g_vols = eval07.get_volume_array(g_parsed, eval_pids)
            g_vol_map = dict(zip(eval_pids, g_vols))
            g_joint = g_parsed[(g_parsed["valid_syntax"] == True) & (g_parsed["is_zero_contact"] == False)].merge(
                demos_df, on="part_id", how="left"
            )
            g_joint["part_volume"] = g_joint["part_id"].map(g_vol_map)
            gen_joints_dict[label] = g_joint

    if not gen_runs_dict:
        print("No generated files found yet.")
        return

    flagship_label = list(gen_runs_dict.keys())[0]
    flagship_tag = "plan_guided" if "Plan-Guided" in flagship_label else "full_free"

    print("1/5 Analyzing Contact Volumes, Right Tails & Cohabitant/Non-Cohabitant Split...")
    analyze_volumes(gt_sub, gen_runs_dict, eval_pids, DEEP_DIR)

    print("2/5 Analyzing 15x15 Contact Matrices by Location...")
    analyze_location_matrices(gt_sub, gen_runs_dict[flagship_label], val_part_bins, N_vec, flagship_tag, DEEP_DIR)

    print("3/5 Analyzing RQ2 Subgroups (with Explicit Attendance & Euro Income Brackets)...")
    analyze_rq2_subgroups_full(gt_sub, gen_runs_dict[flagship_label], demos_df, flagship_tag, DEEP_DIR, RQ2_DIR)

    print("4/5 Analyzing Correlations, Marginals & C2ST Feature Importances...")
    analyze_correlations_and_marginals(gt_joint, gen_joints_dict[flagship_label], flagship_tag, DEEP_DIR)
    analyze_c2st_suite(gt_joint, gen_joints_dict, DEEP_DIR)

    print("5/5 Running Overfitting Audit & Exporting 19-Column Shareable CSVs...")
    run_overfitting_and_csv_exports(
        os.path.join(DATA_DIR, "train_causal_100_percent.jsonl"),
        args.val_file,
        os.path.join(PROJECT_ROOT, "data/processed/master_data.csv"),
        os.path.join(PROJECT_ROOT, "data/splits/val_ids.csv"),
        pg_ep,
        ff_ep,
        DEEP_DIR,
        SHARE_DIR,
    )
    print(f"\n✅ Exp 07 Deep Analysis & CSV Exports Complete! Saved to {RES_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_file", type=str, default=os.path.join(DATA_DIR, "val_causal.jsonl"))
    parser.add_argument("--pg_epoch", type=int, default=0, help="0 = auto-select best epoch by WSMAD")
    parser.add_argument("--ff_epoch", type=int, default=0, help="0 = auto-select best epoch by WSMAD")
    args = parser.parse_args()

    run_all_deep_analyses(args)