import os
import sys
import glob
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance, pearsonr
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(EXP_DIR, "data")
GEN_DIR = os.path.join(EXP_DIR, "generated")
OUT_DIR = os.path.join(EXP_DIR, "results", "deep_analysis")

sys.path.append(SCRIPT_DIR)
from importlib import import_module
eval04 = import_module("04_evaluate")
from core import parse_causal_completion

AGE_LABELS = eval04.AGE_LABELS
DEMO_FEATURES = eval04.DEMO_FEATURES
CONTACT_FEATURES = eval04.CONTACT_FEATURES
MARGINAL_FEATURES = eval04.MARGINAL_C2ST_FEATURES

LOCATIONS = ["Home", "Work", "School", "Leisure", "Transport", "Other"]


# ==============================================================================
# 1. VOLUME & DEGREE DISTRIBUTION ANALYSIS
# ==============================================================================
def compute_volume_stats(vols, ref_vols=None):
    vols = np.asarray(vols, dtype=float)
    mean_v = float(np.mean(vols))
    var_v = float(np.var(vols))
    stats = {
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
        "Volume_TVD": eval04.calc_tvd(ref_vols, vols) if ref_vols is not None else 0.0,
        "Volume_Wass": float(wasserstein_distance(ref_vols, vols)) if ref_vols is not None else 0.0,
    }
    return stats


def analyze_volumes(gt_sub, gen_runs_dict, eval_pids, out_dir):
    gt_vols = eval04.get_volume_array(gt_sub, eval_pids)
    rows = [{"Source": "Ground Truth", **compute_volume_stats(gt_vols, gt_vols)}]

    vol_arrays = {"Ground Truth": gt_vols}
    for label, gen_df in gen_runs_dict.items():
        g_vols = eval04.get_volume_array(gen_df, eval_pids)
        vol_arrays[label] = g_vols
        rows.append({"Source": label, **compute_volume_stats(g_vols, gt_vols)})

    df_vol_stats = pd.DataFrame(rows)
    df_vol_stats.to_csv(os.path.join(out_dir, "table1_volume_statistics.csv"), index=False)

    # Location-specific mean volumes per participant
    loc_rows = []
    for source_name, df_parsed in [("Ground Truth", gt_sub)] + list(gen_runs_dict.items()):
        valid = df_parsed[(df_parsed["valid_syntax"] == True) & (df_parsed["is_zero_contact"] == False)]
        loc_counts = valid.groupby("Location of Contact").size().to_dict()
        rec = {"Source": source_name}
        for loc in LOCATIONS:
            # Fallback substring match if old Location strings contain 'Home (indoor)'
            cnt = sum(v for k, v in loc_counts.items() if loc.lower() in str(k).lower())
            rec[f"Mean_{loc}"] = float(cnt) / max(1, len(eval_pids))
        loc_rows.append(rec)
    df_loc_vols = pd.DataFrame(loc_rows)
    df_loc_vols.to_csv(os.path.join(out_dir, "table1b_location_volumes.csv"), index=False)

    # Plot 4-Panel Volume Figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Deep Analysis 1: Contact Volume & Degree Distributions (100% Split)", fontsize=18, y=1.02)

    # Panel A: Discrete Histogram
    hist_records = []
    for src, arr in vol_arrays.items():
        for v in arr:
            hist_records.append({"Source": src, "Volume": v})
    hist_df = pd.DataFrame(hist_records)
    sns.histplot(
        data=hist_df, x="Volume", hue="Source", multiple="dodge",
        discrete=True, shrink=0.8, ax=axes[0, 0], stat="probability", common_norm=False
    )
    max_x = int(np.percentile(gt_vols, 99)) + 2
    axes[0, 0].set_xlim(-0.5, max(12, max_x))
    axes[0, 0].set_title("A. Discrete Contact Volume PMF (Clipped at P99)")
    axes[0, 0].set_xlabel("Number of Daily Contacts (k)")
    axes[0, 0].set_ylabel("Probability P(K = k)")
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.5)

    # Panel B: Empirical CDF
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

    # Panel C: Log-Survival Right-Tail Plot P(K >= k)
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

    # Panel D: Mean Contacts by Social Location
    loc_plot_df = df_loc_vols.set_index("Source")[[f"Mean_{l}" for l in LOCATIONS]].T
    loc_plot_df.index = LOCATIONS
    loc_plot_df.plot(kind="bar", ax=axes[1, 1], alpha=0.88)
    axes[1, 1].set_title("D. Mean Daily Contacts Decomposed by Social Location")
    axes[1, 1].set_xlabel("Social Location")
    axes[1, 1].set_ylabel("Mean Contacts per Participant")
    axes[1, 1].set_xticklabels(LOCATIONS, rotation=0)
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig1_volume_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return df_vol_stats, df_loc_vols


# ==============================================================================
# 2. 15x15 CONTACT MATRICES BY LOCATION & BY SUBGROUP
# ==============================================================================
def matrix_epidemiological_metrics(gt_mat, gen_mat, N_vec):
    diff = gen_mat - gt_mat
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    flat_gt, flat_gen = gt_mat.flatten(), gen_mat.flatten()
    if np.std(flat_gt) > 0 and np.std(flat_gen) > 0:
        r_val, _ = pearsonr(flat_gt, flat_gen)
        r_val = float(r_val)
    else:
        r_val = np.nan

    norm_prod = np.linalg.norm(flat_gt) * np.linalg.norm(flat_gen)
    cos_sim = float(np.dot(flat_gt, flat_gen) / norm_prod) if norm_prod > 0 else np.nan

    rho_gt = float(np.max(np.abs(np.linalg.eigvals(gt_mat))))
    rho_gen = float(np.max(np.abs(np.linalg.eigvals(gen_mat))))
    r0_ratio = float(rho_gen / rho_gt) if rho_gt > 0 else np.nan

    # Diagonal Assortativity Fraction: trace(C) / sum(C)
    assort_gt = float(np.trace(gt_mat) / np.sum(gt_mat)) if np.sum(gt_mat) > 0 else 0.0
    assort_gen = float(np.trace(gen_mat) / np.sum(gen_mat)) if np.sum(gen_mat) > 0 else 0.0

    recip_gt = eval04.calc_weighted_reciprocity(gt_mat, N_vec)
    recip_gen = eval04.calc_weighted_reciprocity(gen_mat, N_vec)

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
        gt_m = eval04.build_contact_matrix(gt_sub, val_part_bins, location_filter=loc_filter)
        gen_m = eval04.build_contact_matrix(gen_df, val_part_bins, location_filter=loc_filter)
        m_stats = matrix_epidemiological_metrics(gt_m, gen_m, N_vec)
        rows.append({"Run": run_label, "Matrix_Setting": loc_name, **m_stats})

        vmax = max(np.max(gt_m), np.max(gen_m), 0.05)
        diff_m = gen_m - gt_m
        max_d = max(np.max(np.abs(diff_m)), 0.05)

        sns.heatmap(gt_m, ax=axes[idx, 0], cmap="YlOrRd", vmin=0, vmax=vmax, xticklabels=AGE_LABELS, yticklabels=AGE_LABELS, cbar=True)
        axes[idx, 0].set_title(f"{loc_name}: Ground Truth (R0={m_stats['R0_GT']:.2f}, Diag={m_stats['Diag_Assort_GT']*100:.1f}%)")

        sns.heatmap(gen_m, ax=axes[idx, 1], cmap="YlOrRd", vmin=0, vmax=vmax, xticklabels=AGE_LABELS, yticklabels=AGE_LABELS, cbar=True)
        axes[idx, 1].set_title(f"{loc_name}: Generated (R0={m_stats['R0_Gen']:.2f}, Diag={m_stats['Diag_Assort_Gen']*100:.1f}%)")

        sns.heatmap(diff_m, ax=axes[idx, 2], cmap="coolwarm", center=0, vmin=-max_d, vmax=max_d, xticklabels=AGE_LABELS, yticklabels=AGE_LABELS, cbar=True)
        axes[idx, 2].set_title(f"{loc_name}: Diff (MAE={m_stats['MAE']:.4f}, r={m_stats['Pearson_r']:.3f})")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"fig2_matrices_by_location_{run_label}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    df_loc_mat = pd.DataFrame(rows)
    df_loc_mat.to_csv(os.path.join(out_dir, f"table2_location_matrices_{run_label}.csv"), index=False)
    return df_loc_mat


def analyze_subgroup_matrices(gt_sub, gen_df, demos_df, run_label, out_dir):
    """Computes full 15x15 matrices & metrics divided by key demographic subgroups."""
    sub_defs = [
        ("Gender: Female", demos_df[demos_df["gender"] == "F"]),
        ("Gender: Male", demos_df[demos_df["gender"] == "M"]),
        ("Work: In-person", demos_df[demos_df["work_today"].astype(str).str.contains("In-person", case=False, na=False)]),
        ("Work: No/Remote", demos_df[~demos_df["work_today"].astype(str).str.contains("In-person", case=False, na=False)]),
        ("HH Size: 1-2", demos_df[pd.to_numeric(demos_df["hh_size"], errors="coerce").fillna(1) <= 2]),
        ("HH Size: 3+", demos_df[pd.to_numeric(demos_df["hh_size"], errors="coerce").fillna(1) >= 3]),
        ("Region: North", demos_df[demos_df["region"].astype(str).str.contains("North", case=False, na=False)]),
        ("Region: Centre/South/Islands", demos_df[~demos_df["region"].astype(str).str.contains("North", case=False, na=False)]),
    ]

    rows = []
    fig, axes = plt.subplots(len(sub_defs), 3, figsize=(22, 4.2 * len(sub_defs)))
    fig.suptitle(f"Deep Analysis 3: 15x15 Contact Matrices by Population Subgroup ({run_label})", fontsize=20, y=1.01)

    for idx, (sub_name, sub_d) in enumerate(sub_defs):
        pids = sorted(sub_d["part_id"].unique())
        if len(pids) < 10:
            continue
        p_bins = {r["part_id"]: eval04.get_age_bin(r["age"]) for _, r in sub_d.iterrows()}
        N_vec = np.zeros(15)
        for b in p_bins.values():
            if b is not None:
                N_vec[b] += 1

        gt_s = gt_sub[gt_sub["part_id"].isin(pids)]
        gen_s = gen_df[gen_df["part_id"].isin(pids)]

        gt_m = eval04.build_contact_matrix(gt_s, p_bins)
        gen_m = eval04.build_contact_matrix(gen_s, p_bins)
        m_stats = matrix_epidemiological_metrics(gt_m, gen_m, N_vec)

        gt_v = eval04.get_volume_array(gt_s, pids)
        gen_v = eval04.get_volume_array(gen_s, pids)

        rows.append({
            "Subgroup": sub_name,
            "N_Participants": len(pids),
            "Mean_Vol_GT": float(gt_v.mean()),
            "Mean_Vol_Gen": float(gen_v.mean()),
            "Volume_TVD": eval04.calc_tvd(gt_v, gen_v),
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
        axes[idx, 2].set_title(f"{sub_name}: Diff [Matrix MAE={m_stats['MAE']:.4f}, r={m_stats['Pearson_r']:.3f}]")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"fig3_matrices_by_subgroup_{run_label}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    df_sub_mat = pd.DataFrame(rows)
    df_sub_mat.to_csv(os.path.join(out_dir, f"table3_subgroup_matrices_{run_label}.csv"), index=False)
    return df_sub_mat


# ==============================================================================
# 3. PAIRWISE CORRELATIONS & MARGINAL DISTRIBUTIONS
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

    pair_rows = []
    for i in range(len(corr_cols)):
        for j in range(i + 1, len(corr_cols)):
            c1, c2 = corr_cols[i], corr_cols[j]
            v_gt = eval04.cramers_v(gt_c[c1], gt_c[c2])
            v_gen = eval04.cramers_v(gen_c[c1], gen_c[c2])
            pair_rows.append({
                "Variable_1": c1,
                "Variable_2": c2,
                "Cramers_V_GT": v_gt,
                "Cramers_V_Gen": v_gen,
                "Abs_Diff": abs(v_gt - v_gen),
            })

    df_pairs = pd.DataFrame(pair_rows).sort_values("Cramers_V_GT", ascending=False)
    df_pairs.to_csv(os.path.join(out_dir, f"table4_pairwise_correlations_{run_label}.csv"), index=False)

    # Marginal categorical TVD across all categorical contact variables
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

        comp.head(8).plot(kind="bar", ax=axes[idx], color=["#1f77b4", "#ff7f0e"], alpha=0.88)
        axes[idx].set_title(f"{var} (TVD = {tvd_var:.4f})", fontsize=12)
        axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=35, ha="right")
        axes[idx].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"fig4_marginal_distributions_{run_label}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    df_marg = pd.DataFrame(marg_rows)
    df_marg.to_csv(os.path.join(out_dir, f"table4b_marginal_distributions_{run_label}.csv"), index=False)
    return df_pairs, df_marg


# ==============================================================================
# 4. C2ST DISCRIMINABILITY & FULL 19-FEATURE IMPORTANCE DIAGNOSTICS
# ==============================================================================
def run_detailed_c2st(gt_joint, gen_joint, feature_cols):
    gt = gt_joint[feature_cols + ["part_id"]].copy()
    gen = gen_joint[feature_cols + ["part_id"]].copy()
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
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            random_state=42,
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


def analyze_c2st_suite(gt_joint, gen_joints_dict, out_dir):
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

        # Plot OOF probability density on Panel B for the first/flagship model
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

    # Panel A: Feature Importance Bar Chart across all 19 features
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
# 5. MAIN ORCHESTRATOR & AUTOMATED MARKDOWN REPORT GENERATOR
# ==============================================================================
def run_deep_analysis(args):
    os.makedirs(OUT_DIR, exist_ok=True)

    val_df = pd.read_json(args.val_file, lines=True)
    val_df["part_id"] = val_df["part_id"].astype(str).str.strip()
    demos_df = pd.DataFrame(val_df["demos"].tolist())
    demos_df["part_id"] = demos_df["part_id"].astype(str).str.strip()
    eval_pids = sorted(val_df["part_id"].unique())

    val_part_bins = {r["part_id"]: eval04.get_age_bin(r["age"]) for _, r in demos_df.iterrows()}
    N_vec = np.zeros(15)
    for b in val_part_bins.values():
        if b is not None:
            N_vec[b] += 1

    gt_sub = eval04.parse_jsonl_records(val_df, col_name="completion")
    gt_vols = eval04.get_volume_array(gt_sub, eval_pids)
    gt_vol_map = dict(zip(eval_pids, gt_vols))
    gt_joint = gt_sub[(gt_sub["valid_syntax"] == True) & (gt_sub["is_zero_contact"] == False)].merge(
        demos_df, on="part_id", how="left"
    )
    gt_joint["part_volume"] = gt_joint["part_id"].map(gt_vol_map)

    # Locate flagship 100% plan_guided and full_free files
    pg_file = os.path.join(GEN_DIR, f"generated_100_percent_plan_guided_epoch_{args.pg_epoch}.jsonl")
    ff_file = os.path.join(GEN_DIR, f"generated_100_percent_full_free_epoch_{args.ff_epoch}.jsonl")

    gen_runs_dict = {}
    gen_joints_dict = {}

    for label, f_path in [(f"Plan-Guided (Ep {args.pg_epoch})", pg_file), (f"Full-Free (Ep {args.ff_epoch})", ff_file)]:
        if os.path.exists(f_path):
            g_jsonl = pd.read_json(f_path, lines=True)
            g_jsonl["part_id"] = g_jsonl["part_id"].astype(str).str.strip()
            g_parsed = eval04.parse_jsonl_records(g_jsonl, col_name="generated_completion")
            gen_runs_dict[label] = g_parsed

            g_vols = eval04.get_volume_array(g_parsed, eval_pids)
            g_vol_map = dict(zip(eval_pids, g_vols))
            g_joint = g_parsed[(g_parsed["valid_syntax"] == True) & (g_parsed["is_zero_contact"] == False)].merge(
                demos_df, on="part_id", how="left"
            )
            g_joint["part_volume"] = g_joint["part_id"].map(g_vol_map)
            gen_joints_dict[label] = g_joint

    if not gen_runs_dict:
        print(f"No 100_percent generated files found in {GEN_DIR} yet.")
        return

    flagship_label = list(gen_runs_dict.keys())[0]
    flagship_tag = "plan_guided" if "Plan-Guided" in flagship_label else "full_free"

    print("1/4 Analyzing Contact Volumes & Degree Distributions...")
    df_vol_stats, df_loc_vols = analyze_volumes(gt_sub, gen_runs_dict, eval_pids, OUT_DIR)

    print("2/4 Analyzing 15x15 Contact Matrices (Total, 6 Locations, and RQ2 Subgroups)...")
    df_loc_mat = analyze_location_matrices(gt_sub, gen_runs_dict[flagship_label], val_part_bins, N_vec, flagship_tag, OUT_DIR)
    df_sub_mat = analyze_subgroup_matrices(gt_sub, gen_runs_dict[flagship_label], demos_df, flagship_tag, OUT_DIR)

    print("3/4 Analyzing Pairwise Cramér's V Associations & Marginal Distributions...")
    df_pairs, df_marg = analyze_correlations_and_marginals(gt_joint, gen_joints_dict[flagship_label], flagship_tag, OUT_DIR)

    print("4/4 Running C2ST Feature Ablations & Full 19-Feature Importance Diagnostics...")
    df_c2st_ablation, df_imp_all = analyze_c2st_suite(gt_joint, gen_joints_dict, OUT_DIR)

    # Print comprehensive tables to console
    print("\n==========================================================================================")
    print("TABLE 1A: CONTACT VOLUME & DEGREE DISTRIBUTION STATISTICS (100% SPLIT)")
    print("==========================================================================================")
    print(df_vol_stats.to_string(index=False, float_format="%.4f"))

    print("\n==========================================================================================")
    print("TABLE 1B: MEAN DAILY CONTACTS PER PARTICIPANT BY LOCATION")
    print("==========================================================================================")
    print(df_loc_vols.to_string(index=False, float_format="%.4f"))

    print("\n==========================================================================================")
    print(f"TABLE 2: 15x15 CONTACT MATRIX METRICS BY LOCATION ({flagship_label})")
    print("==========================================================================================")
    print(df_loc_mat.to_string(index=False, float_format="%.4f"))

    print("\n==========================================================================================")
    print(f"TABLE 3: RQ2 SUBGROUP 15x15 MATRIX & VOLUME FIDELITY ({flagship_label})")
    print("==========================================================================================")
    print(df_sub_mat.to_string(index=False, float_format="%.4f"))

    print("\n==========================================================================================")
    print(f"TABLE 4: TOP 15 STRONGEST PAIRWISE ASSOCIATIONS (CRAMÉR'S V: GT vs. {flagship_label})")
    print("==========================================================================================")
    print(df_pairs.head(15).to_string(index=False, float_format="%.4f"))

    print("\n==========================================================================================")
    print("TABLE 5A: C2ST DISCRIMINABILITY ACROSS FEATURE SETS (5-COL vs 18-COL vs 19-COL)")
    print("==========================================================================================")
    print(df_c2st_ablation.to_string(index=False, float_format="%.4f"))

    print(f"\n✅ All deep analysis tables and figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_file", type=str, default=os.path.join(DATA_DIR, "val_causal.jsonl"))
    parser.add_argument("--pg_epoch", type=int, default=8)
    parser.add_argument("--ff_epoch", type=int, default=8)
    args = parser.parse_args()

    run_deep_analysis(args)