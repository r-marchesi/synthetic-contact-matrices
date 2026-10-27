import os
import re
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance, chi2_contingency
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(EXP_DIR, "data")
GEN_DIR = os.path.join(EXP_DIR, "generated")
OUT_DIR = os.path.join(EXP_DIR, "results")

sys.path.append(SCRIPT_DIR)
from core import parse_causal_completion

AGE_LABELS = [
    "0-4", "5-9", "10-14", "15-19", "20-24", "25-29",
    "30-34", "35-39", "40-44", "45-49", "50-54", "55-59",
    "60-64", "65-69", "70+",
]

CONTACT_FEATURES = [
    "Contact Setting",
    "Location of Contact",
    "Relationship to Participant",
    "Age_Gap",
    "Contact Age",
    "Contact Gender",
    "Contact Frequency",
    "Physical Contact",
    "Distance during Contact",
]

MARGINAL_C2ST_FEATURES = [
    "Location of Contact",
    "Contact Age",
    "Contact Gender",
    "Contact Frequency",
    "Distance during Contact",
]

DEMO_FEATURES = [
    "age", "gender", "hh_size", "work_today", "school_today",
    "occupation", "education", "income", "region",
]

WSMAD_DIMENSIONS = [
    "Age_Bracket", "Gender", "HH_Size_Bracket", "Work_Today",
    "School_Today", "Education", "Income", "Occupation", "Region_NUTS1",
]


def get_age_bin(age_str):
    try:
        val = float(age_str)
        if np.isnan(val) or val < 0:
            return None
        return min(14, int(val // 5))
    except (ValueError, TypeError):
        return None


def parse_jsonl_records(df_jsonl, col_name):
    all_rows = []
    for _, row in df_jsonl.iterrows():
        pid = str(row["part_id"]).strip()
        comp = row.get(col_name, "")
        all_rows.extend(parse_causal_completion(pid, comp))
    return pd.DataFrame(all_rows)


def assign_demographic_strata(demos_df):
    df = demos_df.copy()
    age_num = pd.to_numeric(df["age"], errors="coerce")
    df["Age_Bracket"] = pd.cut(
        age_num, bins=[-1, 18, 34, 49, 64, 120],
        labels=["0-18", "19-34", "35-49", "50-64", "65+"],
    ).astype(str)

    hh_num = pd.to_numeric(df["hh_size"], errors="coerce").fillna(1)
    df["HH_Size_Bracket"] = pd.cut(
        hh_num, bins=[0, 1, 2, 4, 100], labels=["1", "2", "3-4", "5+"]
    ).astype(str)

    df["Gender"] = df["gender"].astype(str)
    df["Work_Today"] = df["work_today"].astype(str)
    df["School_Today"] = df["school_today"].astype(str)
    df["Education"] = df["education"].astype(str)
    df["Income"] = df["income"].astype(str)
    df["Occupation"] = df["occupation"].astype(str)
    df["Region_NUTS1"] = df["region"].astype(str)
    return df


def build_contact_matrix(parsed_df, val_part_bins, location_filter=None):
    matrix = np.zeros((15, 15))
    part_bin_counts = {i: 0 for i in range(15)}
    for p_bin in val_part_bins.values():
        if p_bin is not None:
            part_bin_counts[p_bin] += 1

    contacts_only = parsed_df[
        (parsed_df["valid_syntax"] == True) & (parsed_df["is_zero_contact"] == False)
    ].copy()

    if location_filter:
        loc_mask = contacts_only["Location of Contact"].astype(str).str.contains(
            location_filter, case=False, na=False
        )
        contacts_only = contacts_only[loc_mask]

    for _, row in contacts_only.iterrows():
        pid = str(row["part_id"])
        p_bin = val_part_bins.get(pid)
        c_bin = get_age_bin(row["Contact Age"])
        if p_bin is not None and c_bin is not None:
            matrix[p_bin, c_bin] += 1

    for p_bin in range(15):
        if part_bin_counts[p_bin] > 0:
            matrix[p_bin, :] /= part_bin_counts[p_bin]

    return matrix


def build_col_respondent_matrix(parsed_df, part_bins_sub):
    """Matrix M[a, r] with Contact Age in rows (a) and Respondent Age in columns (r)."""
    M = np.zeros((15, 15), dtype=float)
    N_vec = np.zeros(15, dtype=float)
    for pid, r_bin in part_bins_sub.items():
        if r_bin is not None:
            N_vec[r_bin] += 1.0

    contacts_only = parsed_df[
        (parsed_df["valid_syntax"] == True) & (parsed_df["is_zero_contact"] == False)
    ]
    for _, row in contacts_only.iterrows():
        pid = str(row["part_id"])
        r_bin = part_bins_sub.get(pid)
        a_bin = get_age_bin(row["Contact Age"])
        if r_bin is not None and a_bin is not None:
            M[a_bin, r_bin] += 1.0

    for r_bin in range(15):
        if N_vec[r_bin] > 0:
            M[:, r_bin] /= N_vec[r_bin]
    return M, N_vec


def compute_wsmad_metrics(gt_parsed, gen_parsed, demos_strata):
    """
    Computes the Colleague's Weighted Subgroup Mixing Absolute Difference (WSMAD):
      - Weighted average across respondent age columns r (weighted by N_{s,r}/N_s)
      - Weighted across subgroups s in dimension d (weighted by N_s/N)
      - Averaged across all 9 respondent dimensions d in {1..9}
    Returns (wsmad_norm_01, wsmad_cell_mae, dim_breakdown_dict).
    """
    dim_norm_scores = {}
    dim_raw_scores = {}

    for dim in WSMAD_DIMENSIONS:
        total_dim_n = 0.0
        accum_norm = 0.0
        accum_raw = 0.0

        for level, grp in demos_strata.groupby(dim):
            if str(level).lower() in ["nan", "none"] or len(grp) < 5:
                continue
            part_bins_sub = {
                str(r["part_id"]): get_age_bin(r["age"]) for _, r in grp.iterrows()
            }
            sub_pids = set(part_bins_sub.keys())
            gt_s = gt_parsed[gt_parsed["part_id"].isin(sub_pids)]
            gen_s = gen_parsed[gen_parsed["part_id"].isin(sub_pids)]

            M_gt, N_vec = build_col_respondent_matrix(gt_s, part_bins_sub)
            M_gen, _ = build_col_respondent_matrix(gen_s, part_bins_sub)

            n_s = np.sum(N_vec)
            if n_s == 0:
                continue

            col_weights = N_vec / n_s
            norm_diffs = np.zeros(15)
            raw_maes = np.zeros(15)

            for r_bin in range(15):
                if N_vec[r_bin] == 0:
                    continue
                c_gt, c_gen = M_gt[:, r_bin], M_gen[:, r_bin]
                l1 = np.sum(np.abs(c_gt - c_gen))
                denom = np.sum(c_gt) + np.sum(c_gen)
                norm_diffs[r_bin] = (l1 / denom) if denom > 0 else 0.0
                raw_maes[r_bin] = np.mean(np.abs(c_gt - c_gen))

            accum_norm += n_s * float(np.sum(col_weights * norm_diffs))
            accum_raw += n_s * float(np.sum(col_weights * raw_maes))
            total_dim_n += n_s

        dim_norm_scores[dim] = (accum_norm / total_dim_n) if total_dim_n > 0 else np.nan
        dim_raw_scores[dim] = (accum_raw / total_dim_n) if total_dim_n > 0 else np.nan

    wsmad_norm_01 = float(np.nanmean(list(dim_norm_scores.values())))
    wsmad_cell_mae = float(np.nanmean(list(dim_raw_scores.values())))
    return wsmad_norm_01, wsmad_cell_mae, dim_norm_scores


def calc_spectral_radius_ratio(gt_mat, gen_mat):
    rho_gt = np.max(np.abs(np.linalg.eigvals(gt_mat)))
    rho_gen = np.max(np.abs(np.linalg.eigvals(gen_mat)))
    return float(rho_gen / rho_gt) if rho_gt > 0 else np.nan


def calc_weighted_reciprocity(matrix, N_vector):
    pop_matrix = matrix * N_vector[:, None]
    return float(np.mean(np.abs(pop_matrix - pop_matrix.T)))


def get_volume_array(parsed_df, eval_part_ids):
    valid_contacts = parsed_df[
        (parsed_df["valid_syntax"] == True) & (parsed_df["is_zero_contact"] == False)
    ]
    counts = valid_contacts.groupby("part_id").size().to_dict()
    return np.array([counts.get(pid, 0) for pid in eval_part_ids])


def calc_tvd(gt_vols, gen_vols, max_cap=50):
    bins = np.arange(max_cap + 2)
    p_gt, _ = np.histogram(np.clip(gt_vols, 0, max_cap), bins=bins, density=True)
    p_gen, _ = np.histogram(np.clip(gen_vols, 0, max_cap), bins=bins, density=True)
    return float(0.5 * np.sum(np.abs(p_gt - p_gen)))


def cramers_v(x, y):
    confusion = pd.crosstab(x, y)
    if confusion.shape[0] < 2 or confusion.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(confusion)[0]
    n = confusion.sum().sum()
    if n <= 1:
        return 0.0
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((kcorr - 1), (rcorr - 1))
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0


def compute_correlation_delta(gt_joint, gen_joint, cols):
    def prep_for_corr(df):
        d = df[cols].copy()
        for col in ["age", "Contact Age"]:
            if col in d.columns:
                num = pd.to_numeric(d[col], errors="coerce").fillna(-5)
                d[col] = (num // 10 * 10).astype(str)
        return d.astype(str)

    gt_c = prep_for_corr(gt_joint)
    gen_c = prep_for_corr(gen_joint)
    k = len(cols)
    diffs = []
    for i in range(k):
        for j in range(i + 1, k):
            diffs.append(abs(cramers_v(gt_c[cols[i]], gt_c[cols[j]]) - cramers_v(gen_c[cols[i]], gen_c[cols[j]])))
    return float(np.mean(diffs))


def evaluate_c2st(gt_df, gen_df, feature_cols):
    gt = gt_df[feature_cols + ["part_id"]].copy()
    gen = gen_df[feature_cols + ["part_id"]].copy()
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

    for train_idx, test_idx in sgkf.split(X, y, groups):
        clf = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, eval_metric="auc", random_state=42,
        )
        clf.fit(X.iloc[train_idx], y[train_idx])
        oof_preds[test_idx] = clf.predict_proba(X.iloc[test_idx])[:, 1]
        importances += clf.feature_importances_ / 5.0

    auc = float(roc_auc_score(y, oof_preds))
    top_indices = np.argsort(importances)[::-1][:5]
    top_drivers = [f"{feature_cols[idx]} ({importances[idx]*100:.1f}%)" for idx in top_indices]
    return auc, " | ".join(top_drivers)


def plot_epoch_stopping_curves(leaderboard_df, out_dir):
    """Plots WSMAD_Norm_[0,1], WSMAD_Cell_MAE, Volume_TVD, and Joint_C2ST_AUC vs Epoch."""
    pat = re.compile(r"^100_percent_(plan_guided|full_free)_epoch_(\d+)$")
    rows = []
    for _, r in leaderboard_df.iterrows():
        m = pat.match(str(r["Run"]))
        if m:
            rows.append({
                "Mode": "Plan-Guided" if m.group(1) == "plan_guided" else "Full-Free",
                "Epoch": int(m.group(2)),
                "WSMAD_Norm_[0,1]": r["WSMAD_Norm_[0,1]"],
                "WSMAD_Cell_MAE": r["WSMAD_Cell_MAE"],
                "Volume_TVD": r["Volume_TVD"],
                "Joint_C2ST_AUC": r["Joint_C2ST_AUC"],
            })
    if not rows:
        return

    df_ep = pd.DataFrame(rows).sort_values(["Mode", "Epoch"])
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Experiment 07: 16-Epoch Stopping Trajectory (WSMAD, Volume TVD, & Joint C2ST)", fontsize=16, y=1.02)

    panels = [
        ("WSMAD_Norm_[0,1]", "A. Colleague WSMAD Normalized [0, 1] (Lower = Better)", axes[0, 0]),
        ("WSMAD_Cell_MAE", "B. Colleague WSMAD Cell MAE (Lower = Better)", axes[0, 1]),
        ("Volume_TVD", "C. Contact Volume TVD (Lower = Better)", axes[1, 0]),
        ("Joint_C2ST_AUC", "D. 19-Feature Joint C2ST ROC-AUC (0.50 = Ideal)", axes[1, 1]),
    ]

    for col, title, ax in panels:
        for mode_name, grp in df_ep.groupby("Mode"):
            ax.plot(grp["Epoch"], grp[col], marker="o", linewidth=2.4, markersize=7, label=mode_name)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Training Epoch")
        ax.set_ylabel(col)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "wsmad_epoch_stopping_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_evaluation(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    val_df = pd.read_json(args.val_file, lines=True)
    val_df["part_id"] = val_df["part_id"].astype(str).str.strip()

    demos_df = pd.DataFrame(val_df["demos"].tolist())
    demos_df["part_id"] = demos_df["part_id"].astype(str).str.strip()
    demos_strata = assign_demographic_strata(demos_df)

    gt_parsed_all = parse_jsonl_records(val_df, col_name="completion")
    gen_files = sorted(glob.glob(os.path.join(GEN_DIR, "generated_*.jsonl")))
    if not gen_files:
        print(f"No generated files found in {GEN_DIR}")
        return

    summary_rows = []
    wsmad_dim_rows = []

    for g_path in gen_files:
        tag = os.path.splitext(os.path.basename(g_path))[0].replace("generated_", "")
        print(f"\nEvaluating Exp 07: {tag}...")

        gen_jsonl = pd.read_json(g_path, lines=True)
        gen_jsonl["part_id"] = gen_jsonl["part_id"].astype(str).str.strip()
        eval_pids = sorted(gen_jsonl["part_id"].unique())

        gt_sub = gt_parsed_all[gt_parsed_all["part_id"].isin(eval_pids)].copy()
        gen_sub = parse_jsonl_records(gen_jsonl, col_name="generated_completion")

        sub_demos = demos_df[demos_df["part_id"].isin(eval_pids)].copy()
        val_part_bins = {row["part_id"]: get_age_bin(row["age"]) for _, row in sub_demos.iterrows()}
        N_vec = np.zeros(15)
        for b in val_part_bins.values():
            if b is not None:
                N_vec[b] += 1

        valid_syntax_pct = float(gen_sub.groupby("part_id")["valid_syntax"].all().mean() * 100.0)

        gt_mat_tot = build_contact_matrix(gt_sub, val_part_bins)
        gen_mat_tot = build_contact_matrix(gen_sub, val_part_bins)
        gt_mat_home = build_contact_matrix(gt_sub, val_part_bins, "Home")
        gen_mat_home = build_contact_matrix(gen_sub, val_part_bins, "Home")
        gt_mat_work = build_contact_matrix(gt_sub, val_part_bins, "Work")
        gen_mat_work = build_contact_matrix(gen_sub, val_part_bins, "Work")
        gt_mat_school = build_contact_matrix(gt_sub, val_part_bins, "School")
        gen_mat_school = build_contact_matrix(gen_sub, val_part_bins, "School")

        mae_tot = float(np.mean(np.abs(gt_mat_tot - gen_mat_tot)))
        mae_home = float(np.mean(np.abs(gt_mat_home - gen_mat_home)))
        mae_work = float(np.mean(np.abs(gt_mat_work - gen_mat_work)))
        mae_school = float(np.mean(np.abs(gt_mat_school - gen_mat_school)))
        r0_ratio = calc_spectral_radius_ratio(gt_mat_tot, gen_mat_tot)
        recip_err = calc_weighted_reciprocity(gen_mat_tot, N_vec)

        # Colleague WSMAD Epoch-Stopping Metric
        wsmad_norm_01, wsmad_cell_mae, dim_scores = compute_wsmad_metrics(gt_sub, gen_sub, demos_strata)
        wsmad_dim_rows.append({"Run": tag, "WSMAD_Norm_[0,1]": wsmad_norm_01, "WSMAD_Cell_MAE": wsmad_cell_mae, **dim_scores})

        gt_vols = get_volume_array(gt_sub, eval_pids)
        gen_vols = get_volume_array(gen_sub, eval_pids)
        vol_tvd = calc_tvd(gt_vols, gen_vols)
        vol_wass = float(wasserstein_distance(gt_vols, gen_vols))

        gt_vol_map = dict(zip(eval_pids, gt_vols))
        gen_vol_map = dict(zip(eval_pids, gen_vols))

        gt_joint = gt_sub[(gt_sub["valid_syntax"] == True) & (gt_sub["is_zero_contact"] == False)].merge(
            sub_demos, on="part_id", how="left"
        )
        gt_joint["part_volume"] = gt_joint["part_id"].map(gt_vol_map)

        gen_joint = gen_sub[(gen_sub["valid_syntax"] == True) & (gen_sub["is_zero_contact"] == False)].merge(
            sub_demos, on="part_id", how="left"
        )
        gen_joint["part_volume"] = gen_joint["part_id"].map(gen_vol_map)

        joint_cols = DEMO_FEATURES + CONTACT_FEATURES + ["part_volume"]
        corr_cols = ["age", "gender", "hh_size", "work_today", "school_today"] + CONTACT_FEATURES

        if len(gen_joint) > 20 and len(gt_joint) > 20:
            corr_mae = compute_correlation_delta(gt_joint, gen_joint, corr_cols)
            marginal_c2st_auc, _ = evaluate_c2st(gt_joint, gen_joint, MARGINAL_C2ST_FEATURES)
            joint_c2st_auc, top_leaks = evaluate_c2st(gt_joint, gen_joint, joint_cols)
        else:
            corr_mae, marginal_c2st_auc, joint_c2st_auc, top_leaks = np.nan, np.nan, np.nan, "Insufficient contacts"

        summary_rows.append({
            "Run": tag,
            "Syntax_Valid_%": valid_syntax_pct,
            "WSMAD_Norm_[0,1]": wsmad_norm_01,
            "WSMAD_Cell_MAE": wsmad_cell_mae,
            "MAE_Total": mae_tot,
            "MAE_Home": mae_home,
            "MAE_Work": mae_work,
            "MAE_School": mae_school,
            "R0_Ratio": r0_ratio,
            "Reciprocity_Err": recip_err,
            "Volume_TVD": vol_tvd,
            "Volume_Wass": vol_wass,
            "Mean_Vol_GT": float(gt_vols.mean()),
            "Mean_Vol_Gen": float(gen_vols.mean()),
            "Corr_MAE": corr_mae,
            "Marginal_C2ST_AUC": marginal_c2st_auc,
            "Joint_C2ST_AUC": joint_c2st_auc,
            "Top_5_C2ST_Drivers": top_leaks,
        })

    # Sort leaderboard by WSMAD_Norm_[0,1] (Colleague's Epoch Stopping Criterion)
    leaderboard = pd.DataFrame(summary_rows).sort_values("WSMAD_Norm_[0,1]", ascending=True)
    out_csv = os.path.join(OUT_DIR, "leaderboard_07_semantic.csv")
    leaderboard.to_csv(out_csv, index=False)

    df_wsmad_dims = pd.DataFrame(wsmad_dim_rows).sort_values("WSMAD_Norm_[0,1]", ascending=True)
    df_wsmad_dims.to_csv(os.path.join(OUT_DIR, "wsmad_dimension_breakdown.csv"), index=False)

    plot_epoch_stopping_curves(leaderboard, OUT_DIR)

    print("\n=========================================================================================================")
    print("🏆 EXPERIMENT 07 LEADERBOARD (SORTED BY COLLEAGUE WSMAD EPOCH-STOPPING METRIC)")
    print("=========================================================================================================")
    display_cols = [c for c in leaderboard.columns if c != "Top_5_C2ST_Drivers"]
    print(leaderboard[display_cols].to_string(index=False, float_format="%.4f"))
    print(f"\nSaved to: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_file", type=str, default=os.path.join(DATA_DIR, "val_causal.jsonl"))
    args = parser.parse_args()

    run_evaluation(args)