import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(EXP_DIR, "data")
GEN_DIR = os.path.join(EXP_DIR, "generated")
OUT_DIR = os.path.join(EXP_DIR, "results")

sys.path.append(SCRIPT_DIR)
from importlib import import_module
eval04 = import_module("04_evaluate")
rq2_mod = import_module("05_evaluate_rq2_subgroups")

DIMENSIONS = [
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


def build_col_respondent_matrix(parsed_df, part_bins_sub):
    """
    Builds a 15x15 contact matrix M[a, r] where:
      - rows a in {0..14} are CONTACT 5-year age bins
      - columns r in {0..14} are RESPONDENT 5-year age bins
    Returns:
      - M: (15, 15) matrix of mean daily contacts per respondent in column r
      - N_vec: (15,) array of respondent counts in each age bin r
    """
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
        a_bin = eval04.get_age_bin(row["Contact Age"])
        if r_bin is not None and a_bin is not None:
            M[a_bin, r_bin] += 1.0

    for r_bin in range(15):
        if N_vec[r_bin] > 0:
            M[:, r_bin] /= N_vec[r_bin]

    return M, N_vec


def compute_subgroup_wsmad(gt_parsed, gen_parsed, grp_demos):
    """
    Computes the respondent-age-weighted mixing absolute difference for a single subgroup:
      1. norm_diff in [0, 1]: column-wise L1 difference normalized by (col_sum_GT + col_sum_Gen),
         weighted by N_{s,r} / N_s.
      2. cell_mae: column-wise mean cell absolute difference (1/15 * sum_a |M^GT_{a,r} - M^Gen_{a,r}|),
         weighted by N_{s,r} / N_s.
    """
    part_bins_sub = {
        str(r["part_id"]): eval04.get_age_bin(r["age"])
        for _, r in grp_demos.iterrows()
    }
    sub_pids = set(part_bins_sub.keys())
    gt_sub = gt_parsed[gt_parsed["part_id"].isin(sub_pids)]
    gen_sub = gen_parsed[gen_parsed["part_id"].isin(sub_pids)]

    M_gt, N_vec = build_col_respondent_matrix(gt_sub, part_bins_sub)
    M_gen, _ = build_col_respondent_matrix(gen_sub, part_bins_sub)

    total_n = np.sum(N_vec)
    if total_n == 0:
        return 0.0, 0.0

    col_weights = N_vec / total_n  # w_{s,r}, sums to 1 across occupied respondent age bins
    norm_col_diffs = np.zeros(15, dtype=float)
    raw_col_maes = np.zeros(15, dtype=float)

    for r_bin in range(15):
        if N_vec[r_bin] == 0:
            continue
        col_gt = M_gt[:, r_bin]
        col_gen = M_gen[:, r_bin]
        l1_diff = np.sum(np.abs(col_gt - col_gen))
        denom = np.sum(col_gt) + np.sum(col_gen)

        norm_col_diffs[r_bin] = (l1_diff / denom) if denom > 0 else 0.0
        raw_col_maes[r_bin] = np.mean(np.abs(col_gt - col_gen))

    weighted_norm_diff = float(np.sum(col_weights * norm_col_diffs))
    weighted_cell_mae = float(np.sum(col_weights * raw_col_maes))
    return weighted_norm_diff, weighted_cell_mae


def evaluate_all_runs_wsmad(args):
    val_df = pd.read_json(args.val_file, lines=True)
    val_df["part_id"] = val_df["part_id"].astype(str).str.strip()

    demos_df = pd.DataFrame(val_df["demos"].tolist())
    demos_df["part_id"] = demos_df["part_id"].astype(str).str.strip()
    demos_strata = rq2_mod.assign_demographic_strata(demos_df)

    gt_parsed = eval04.parse_jsonl_records(val_df, col_name="completion")
    gen_files = sorted(glob.glob(os.path.join(GEN_DIR, "generated_*.jsonl")))

    rows = []
    for g_path in gen_files:
        tag = os.path.splitext(os.path.basename(g_path))[0].replace("generated_", "")
        gen_jsonl = pd.read_json(g_path, lines=True)
        gen_jsonl["part_id"] = gen_jsonl["part_id"].astype(str).str.strip()
        gen_parsed = eval04.parse_jsonl_records(gen_jsonl, col_name="generated_completion")

        dim_norm_scores = {}
        dim_raw_scores = {}

        for dim in DIMENSIONS:
            total_dim_n = 0.0
            accum_norm = 0.0
            accum_raw = 0.0

            for level, grp in demos_strata.groupby(dim):
                if str(level).lower() in ["nan", "none"] or len(grp) < 5:
                    continue
                n_s = float(len(grp))
                w_norm, w_raw = compute_subgroup_wsmad(gt_parsed, gen_parsed, grp)
                accum_norm += n_s * w_norm
                accum_raw += n_s * w_raw
                total_dim_n += n_s

            dim_norm_scores[dim] = (accum_norm / total_dim_n) if total_dim_n > 0 else np.nan
            dim_raw_scores[dim] = (accum_raw / total_dim_n) if total_dim_n > 0 else np.nan

        wsmad_norm_01 = float(np.nanmean(list(dim_norm_scores.values())))
        wsmad_cell_mae = float(np.nanmean(list(dim_raw_scores.values())))

        rows.append({
            "Run": tag,
            "WSMAD_Norm_[0,1]": wsmad_norm_01,
            "WSMAD_Cell_MAE": wsmad_cell_mae,
            "Dim_Work_Today": dim_norm_scores["Work_Today"],
            "Dim_School_Today": dim_norm_scores["School_Today"],
            "Dim_HH_Size": dim_norm_scores["HH_Size_Bracket"],
            "Dim_Income": dim_norm_scores["Income"],
            "Dim_Occupation": dim_norm_scores["Occupation"],
            "Dim_Age": dim_norm_scores["Age_Bracket"],
            "Dim_Gender": dim_norm_scores["Gender"],
            "Dim_Education": dim_norm_scores["Education"],
            "Dim_Region": dim_norm_scores["Region_NUTS1"],
        })

    df_wsmad = pd.DataFrame(rows).sort_values("WSMAD_Norm_[0,1]", ascending=True)
    out_csv = os.path.join(OUT_DIR, "wsmad_epoch_stopping_leaderboard.csv")
    df_wsmad.to_csv(out_csv, index=False)

    print("\n=========================================================================================================")
    print("🎯 METRIC: WEIGHTED SUBGROUP MIXING ABSOLUTE DIFFERENCE (WSMAD) FOR EPOCH STOPPING")
    print("   (Bounded in [0, 1], where 0.0000 = Exact Subgroup & Respondent-Age-Group Mixing Match)")
    print("=========================================================================================================")
    print(df_wsmad.to_string(index=False, float_format="%.4f"))
    print(f"\n✅ Saved WSMAD Epoch-Stopping Leaderboard -> {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_file", default=os.path.join(DATA_DIR, "val_causal.jsonl"))
    args = parser.parse_args()
    evaluate_all_runs_wsmad(args)