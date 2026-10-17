import os
import sys
import glob
import re
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

DEMO_FEATURES = [
    "age", "gender", "hh_size", "work_today", "school_today",
    "occupation", "education", "income", "region",
]


def get_age_bin(age_str):
    """Maps valid numeric ages to 0..14; returns None for unparseable/Unknown ages."""
    try:
        val = float(age_str)
        if np.isnan(val) or val < 0:
            return None
        return min(14, int(val // 5))
    except (ValueError, TypeError):
        return None


def parse_jsonl_records(df_jsonl, col_name):
    """Parses completion strings into a flat DataFrame of contacts + participant flags."""
    all_rows = []
    for _, row in df_jsonl.iterrows():
        pid = str(row["part_id"]).strip()
        comp = row.get(col_name, "")
        parsed = parse_causal_completion(pid, comp)
        all_rows.extend(parsed)
    return pd.DataFrame(all_rows)


def build_contact_matrix(parsed_df, val_part_bins, setting_filter=None):
    """Builds a 15x15 contact matrix C_ij normalized by participant bin counts."""
    matrix = np.zeros((15, 15))
    part_bin_counts = {i: 0 for i in range(15)}
    for p_bin in val_part_bins.values():
        if p_bin is not None:
            part_bin_counts[p_bin] += 1

    contacts_only = parsed_df[
        (parsed_df["valid_syntax"] == True) & (parsed_df["is_zero_contact"] == False)
    ].copy()

    if setting_filter:
        contacts_only = contacts_only[contacts_only["Contact Setting"] == setting_filter]

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


def calc_spectral_radius_ratio(gt_mat, gen_mat):
    """Ratio of dominant eigenvalues rho(C_gen) / rho(C_gt), a proxy for R0 fidelity."""
    rho_gt = np.max(np.abs(np.linalg.eigvals(gt_mat)))
    rho_gen = np.max(np.abs(np.linalg.eigvals(gen_mat)))
    if rho_gt == 0:
        return np.nan
    return float(rho_gen / rho_gt)


def calc_weighted_reciprocity(matrix, N_vector):
    """Population-weighted reciprocity error: mean(|C_ij * N_i - C_ji * N_j|)."""
    pop_matrix = matrix * N_vector[:, None]
    return float(np.mean(np.abs(pop_matrix - pop_matrix.T)))


def get_volume_array(parsed_df, eval_part_ids):
    """Returns array of valid contact counts per participant."""
    valid_contacts = parsed_df[
        (parsed_df["valid_syntax"] == True) & (parsed_df["is_zero_contact"] == False)
    ]
    counts = valid_contacts.groupby("part_id").size().to_dict()
    return np.array([counts.get(pid, 0) for pid in eval_part_ids])


def calc_tvd(gt_vols, gen_vols, max_cap=50):
    """Total Variation Distance between discrete contact volume distributions."""
    bins = np.arange(max_cap + 2)
    p_gt, _ = np.histogram(np.clip(gt_vols, 0, max_cap), bins=bins, density=True)
    p_gen, _ = np.histogram(np.clip(gen_vols, 0, max_cap), bins=bins, density=True)
    return float(0.5 * np.sum(np.abs(p_gt - p_gen)))


def cramers_v(x, y):
    """Bias-corrected Cramér's V between two categorical/binned series."""
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
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2corr / denom))


def compute_correlation_delta(gt_joint, gen_joint, cols, plot_path=None):
    """Computes pairwise Cramér's V matrices for GT vs Gen and returns Mean Absolute Error."""
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
    mat_gt = np.eye(k)
    mat_gen = np.eye(k)

    for i in range(k):
        for j in range(i + 1, k):
            v_gt = cramers_v(gt_c[cols[i]], gt_c[cols[j]])
            v_gen = cramers_v(gen_c[cols[i]], gen_c[cols[j]])
            mat_gt[i, j] = mat_gt[j, i] = v_gt
            mat_gen[i, j] = mat_gen[j, i] = v_gen

    diff = np.abs(mat_gt - mat_gen)
    off_diag_mask = ~np.eye(k, dtype=bool)
    corr_mae = float(np.mean(diff[off_diag_mask]))

    if plot_path:
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        sns.heatmap(mat_gt, ax=axes[0], vmin=0, vmax=1, cmap="Blues", xticklabels=cols, yticklabels=cols)
        axes[0].set_title("Ground Truth Pairwise Cramér's V")
        sns.heatmap(mat_gen, ax=axes[1], vmin=0, vmax=1, cmap="Blues", xticklabels=cols, yticklabels=cols)
        axes[1].set_title("Generated Pairwise Cramér's V")
        sns.heatmap(diff, ax=axes[2], vmin=0, vmax=0.5, cmap="Reds", xticklabels=cols, yticklabels=cols, annot=True, fmt=".2f", annot_kws={"size": 7})
        axes[2].set_title(f"|Delta| Cramér's V (MAE = {corr_mae:.4f})")
        for ax in axes:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return corr_mae


def evaluate_joint_c2st(gt_joint, gen_joint, feature_cols):
    """
    Trains an out-of-fold XGBoost classifier on Joint (Participant + Contact + Volume) features
    grouped by part_id. Returns ROC-AUC and Top-5 most discriminative features.
    """
    gt = gt_joint[feature_cols + ["part_id"]].copy()
    gen = gen_joint[feature_cols + ["part_id"]].copy()
    gt["is_synthetic"] = 0
    gen["is_synthetic"] = 1

    # Prefix group IDs so GT and Gen for the same part_id can be grouped together
    combined = pd.concat([gt, gen], ignore_index=True)
    groups = combined["part_id"].astype(str)

    X = combined[feature_cols].copy()
    y = combined["is_synthetic"].values

    num_cols = ["age", "hh_size", "Contact Age", "part_volume"]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    for nc in num_cols:
        if nc in X.columns:
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
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            random_state=42,
        )
        clf.fit(X.iloc[train_idx], y[train_idx])
        oof_preds[test_idx] = clf.predict_proba(X.iloc[test_idx])[:, 1]
        importances += clf.feature_importances_ / 5.0

    auc = float(roc_auc_score(y, oof_preds))
    top_indices = np.argsort(importances)[::-1][:5]
    top_drivers = [f"{feature_cols[idx]} ({importances[idx]*100:.1f}%)" for idx in top_indices]
    return auc, " | ".join(top_drivers)


def plot_matrix_comparison(gt_mat, gen_mat, title, output_path):
    diff_mat = gen_mat - gt_mat
    vmax = max(np.max(gt_mat), np.max(gen_mat), 0.1)
    max_diff = max(np.max(np.abs(diff_mat)), 0.1)

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle(f"15x15 Contact Matrix Comparison: {title}", fontsize=18, y=1.03)

    sns.heatmap(gt_mat, ax=axes[0], cmap="YlOrRd", vmin=0, vmax=vmax, xticklabels=AGE_LABELS, yticklabels=AGE_LABELS, annot=True, fmt=".1f", annot_kws={"size": 7})
    axes[0].set_title("Ground Truth")
    sns.heatmap(gen_mat, ax=axes[1], cmap="YlOrRd", vmin=0, vmax=vmax, xticklabels=AGE_LABELS, yticklabels=AGE_LABELS, annot=True, fmt=".1f", annot_kws={"size": 7})
    axes[1].set_title("Generated")
    sns.heatmap(diff_mat, ax=axes[2], cmap="coolwarm", center=0, vmin=-max_diff, vmax=max_diff, xticklabels=AGE_LABELS, yticklabels=AGE_LABELS, annot=True, fmt="+.1f", annot_kws={"size": 7})
    axes[2].set_title("Difference (Generated - GT)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_volume_comparison(gt_vols, gen_vols, title, output_path):
    vol_df = pd.DataFrame({
        "Volume": np.concatenate([gt_vols, gen_vols]),
        "Source": ["Ground Truth"] * len(gt_vols) + ["Generated"] * len(gen_vols),
    })
    plt.figure(figsize=(13, 6))
    sns.histplot(data=vol_df, x="Volume", hue="Source", multiple="dodge", discrete=True, shrink=0.8)
    max_x = int(np.percentile(vol_df["Volume"], 99)) + 2
    plt.xlim(-0.5, max(10, max_x))
    plt.title(f"Contact Volume Distribution: {title}", fontsize=15)
    plt.xlabel("Contacts per Participant")
    plt.ylabel("Participant Count")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_evaluation(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    val_df = pd.read_json(args.val_file, lines=True)
    val_df["part_id"] = val_df["part_id"].astype(str).str.strip()

    demos_df = pd.DataFrame(val_df["demos"].tolist())
    demos_df["part_id"] = demos_df["part_id"].astype(str).str.strip()

    gt_parsed_all = parse_jsonl_records(val_df, col_name="completion")
    gen_files = sorted(glob.glob(os.path.join(GEN_DIR, "generated_*.jsonl")))
    if not gen_files:
        print(f"No generated files found in {GEN_DIR}")
        return

    summary_rows = []

    for g_path in gen_files:
        tag = os.path.splitext(os.path.basename(g_path))[0].replace("generated_", "")
        print(f"\nEvaluating: {tag}...")
        plot_dir = os.path.join(OUT_DIR, tag)
        os.makedirs(plot_dir, exist_ok=True)

        gen_jsonl = pd.read_json(g_path, lines=True)
        gen_jsonl["part_id"] = gen_jsonl["part_id"].astype(str).str.strip()
        eval_pids = sorted(gen_jsonl["part_id"].unique())

        gt_sub = gt_parsed_all[gt_parsed_all["part_id"].isin(eval_pids)].copy()
        gen_sub = parse_jsonl_records(gen_jsonl, col_name="generated_completion")

        # Participant age bins & population vector
        sub_demos = demos_df[demos_df["part_id"].isin(eval_pids)].copy()
        val_part_bins = {
            row["part_id"]: get_age_bin(row["age"]) for _, row in sub_demos.iterrows()
        }
        N_vec = np.zeros(15)
        for b in val_part_bins.values():
            if b is not None:
                N_vec[b] += 1

        # 1. Syntax & Plan Validity (at participant level)
        part_syntax = gen_sub.groupby("part_id")["valid_syntax"].all()
        valid_syntax_pct = float(part_syntax.mean() * 100.0)

        # 2. Matrices & Reciprocity
        gt_mat_tot = build_contact_matrix(gt_sub, val_part_bins)
        gen_mat_tot = build_contact_matrix(gen_sub, val_part_bins)
        gt_mat_home = build_contact_matrix(gt_sub, val_part_bins, "Home")
        gen_mat_home = build_contact_matrix(gen_sub, val_part_bins, "Home")
        gt_mat_work = build_contact_matrix(gt_sub, val_part_bins, "Work")
        gen_mat_work = build_contact_matrix(gen_sub, val_part_bins, "Work")

        mae_tot = float(np.mean(np.abs(gt_mat_tot - gen_mat_tot)))
        mae_home = float(np.mean(np.abs(gt_mat_home - gen_mat_home)))
        mae_work = float(np.mean(np.abs(gt_mat_work - gen_mat_work)))
        r0_ratio = calc_spectral_radius_ratio(gt_mat_tot, gen_mat_tot)
        recip_err = calc_weighted_reciprocity(gen_mat_tot, N_vec)

        # 3. Volumes
        gt_vols = get_volume_array(gt_sub, eval_pids)
        gen_vols = get_volume_array(gen_sub, eval_pids)
        vol_tvd = calc_tvd(gt_vols, gen_vols)
        vol_wass = float(wasserstein_distance(gt_vols, gen_vols))

        # 4. Joint DataFrames for Correlation & C2ST
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
            corr_mae = compute_correlation_delta(
                gt_joint, gen_joint, corr_cols,
                plot_path=os.path.join(plot_dir, "correlation_cramers_v.png"),
            )
            c2st_auc, top_leaks = evaluate_joint_c2st(gt_joint, gen_joint, joint_cols)
        else:
            corr_mae, c2st_auc, top_leaks = np.nan, np.nan, "Insufficient contacts"

        plot_matrix_comparison(gt_mat_tot, gen_mat_tot, tag, os.path.join(plot_dir, "matrix_total.png"))
        plot_volume_comparison(gt_vols, gen_vols, tag, os.path.join(plot_dir, "volume_distribution.png"))

        print(f"  -> Valid Syntax: {valid_syntax_pct:.1f}%")
        print(f"  -> Matrix MAE (Total): {mae_tot:.4f} | R0 Ratio: {r0_ratio:.3f}")
        print(f"  -> Volume TVD: {vol_tvd:.4f} | Mean Vol (GT vs Gen): {gt_vols.mean():.2f} vs {gen_vols.mean():.2f}")
        print(f"  -> Pairwise Correlation MAE (Cramér's V): {corr_mae:.4f}")
        print(f"  -> Joint C2ST AUC: {c2st_auc:.4f} (0.50 = indistinguishable)")
        print(f"  -> Top C2ST Discriminators: {top_leaks}")

        summary_rows.append({
            "Run": tag,
            "Syntax_Valid_%": valid_syntax_pct,
            "MAE_Total": mae_tot,
            "MAE_Home": mae_home,
            "MAE_Work": mae_work,
            "R0_Ratio": r0_ratio,
            "Reciprocity_Err": recip_err,
            "Volume_TVD": vol_tvd,
            "Volume_Wass": vol_wass,
            "Mean_Vol_GT": float(gt_vols.mean()),
            "Mean_Vol_Gen": float(gen_vols.mean()),
            "Corr_MAE": corr_mae,
            "Joint_C2ST_AUC": c2st_auc,
            "Top_5_C2ST_Drivers": top_leaks,
        })

    leaderboard = pd.DataFrame(summary_rows).sort_values("MAE_Total", ascending=True)
    out_csv = os.path.join(OUT_DIR, "leaderboard_04_causal.csv")
    leaderboard.to_csv(out_csv, index=False)

    print("\n================================================================================")
    print("🏆 EXPERIMENT 04 LEADERBOARD")
    print("================================================================================")
    display_cols = [c for c in leaderboard.columns if c != "Top_5_C2ST_Drivers"]
    print(leaderboard[display_cols].to_string(index=False, float_format="%.4f"))
    print(f"\nFull leaderboard (including Top-5 C2ST feature drivers) saved to: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_file", type=str, default=os.path.join(DATA_DIR, "val_causal.jsonl"))
    args = parser.parse_args()

    run_evaluation(args)