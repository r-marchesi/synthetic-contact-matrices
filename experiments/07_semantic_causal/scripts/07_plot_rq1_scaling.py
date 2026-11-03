import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RES_DIR = os.path.join(EXP_DIR, "results")
RQ1_DIR = os.path.join(RES_DIR, "rq1_scaling")

RUN_PATTERN = re.compile(r"^(\d+)_percent_(plan_guided|full_free)_epoch_(\d+)$")


def plot_rq1_curves(leaderboard_csv, out_dir, pg_epoch=14, ff_epoch=14):
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(leaderboard_csv):
        print(f"Leaderboard not found: {leaderboard_csv}")
        return

    df = pd.read_csv(leaderboard_csv)

    # Parse all runs
    parsed_rows = []
    for _, row in df.iterrows():
        m = RUN_PATTERN.match(str(row["Run"]))
        if not m:
            continue
        parsed_rows.append({
            **row.to_dict(),
            "Split_Pct": int(m.group(1)),
            "Mode_Raw": m.group(2),
            "Epoch": int(m.group(3)),
        })

    if not parsed_rows:
        print("No matching RQ1 runs found in leaderboard.")
        return

    pdf = pd.DataFrame(parsed_rows)

    # For each (Split_Pct, Mode_Raw), pick target epoch if available, else best WSMAD epoch for that split/mode
    selected_records = []
    for (pct, mode_raw), grp in pdf.groupby(["Split_Pct", "Mode_Raw"]):
        target_ep = pg_epoch if mode_raw == "plan_guided" else ff_epoch
        ep_match = grp[grp["Epoch"] == target_ep]
        chosen = ep_match.iloc[0] if not ep_match.empty else grp.sort_values("WSMAD_Norm_[0,1]").iloc[0]
        ep_used = int(chosen["Epoch"])

        selected_records.append({
            "Split_Pct": int(pct),
            "Mode": "Plan-Guided" if mode_raw == "plan_guided" else "Full-Free",
            "Epoch": ep_used,
            "WSMAD_Norm_[0,1]": chosen["WSMAD_Norm_[0,1]"],
            "WSMAD_Cell_MAE": chosen["WSMAD_Cell_MAE"],
            "MAE_Total": chosen["MAE_Total"],
            "MAE_Home": chosen["MAE_Home"],
            "MAE_Work": chosen["MAE_Work"],
            "MAE_School": chosen["MAE_School"],
            "Volume_TVD": chosen["Volume_TVD"],
            "Volume_Wass": chosen["Volume_Wass"],
            "Corr_MAE": chosen["Corr_MAE"],
            "Marginal_C2ST_AUC": chosen["Marginal_C2ST_AUC"],
            "Joint_C2ST_AUC": chosen["Joint_C2ST_AUC"],
            "R0_Ratio": chosen["R0_Ratio"],
        })

    rq1_df = pd.DataFrame(selected_records).sort_values(["Mode", "Split_Pct"])
    out_csv = os.path.join(out_dir, "rq1_scaling_summary.csv")
    rq1_df.to_csv(out_csv, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("RQ1: Impact of Fine-Tuning Dataset Size on Synthetic Contact Quality (Exp 07)", fontsize=17, y=1.02)

    metrics = [
        ("WSMAD_Norm_[0,1]", "A. Colleague WSMAD Normalized [0, 1] (Lower is Better)", axes[0, 0]),
        ("MAE_Total", "B. 15x15 Contact Matrix MAE (Lower is Better)", axes[0, 1]),
        ("Volume_TVD", "C. Contact Volume TVD (Lower is Better)", axes[1, 0]),
        ("Joint_C2ST_AUC", "D. 19-Feature Joint C2ST ROC-AUC (0.50 = Ideal)", axes[1, 1]),
    ]

    colors = {"Plan-Guided": "#1f77b4", "Full-Free": "#ff7f0e"}

    for col, title, ax in metrics:
        for mode_label, grp in rq1_df.groupby("Mode"):
            ax.plot(
                grp["Split_Pct"], grp[col],
                marker="o", linewidth=2.5, markersize=8,
                label=mode_label, color=colors.get(mode_label, "#2ca02c"),
            )
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Training Pool Size (%)")
        ax.set_ylabel(col)
        ax.set_xticks([10, 30, 50, 80, 100])
        if "C2ST" in col:
            ax.axhline(0.50, color="gray", linestyle="--", alpha=0.7, label="Indistinguishable (0.50)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "rq1_scaling_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n==========================================================================================")
    print("📈 EXP 07 RQ1 SCALING SUMMARY TABLE")
    print("==========================================================================================")
    print(rq1_df.to_string(index=False, float_format="%.4f"))
    print(f"\n✅ Saved RQ1 summary CSV -> {out_csv}")
    print(f"✅ Saved RQ1 scaling plot -> {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--leaderboard_csv",
        type=str,
        default=os.path.join(RES_DIR, "leaderboard_07_semantic.csv"),
    )
    parser.add_argument("--out_dir", type=str, default=RQ1_DIR)
    parser.add_argument("--pg_epoch", type=int, default=14)
    parser.add_argument("--ff_epoch", type=int, default=14)
    args = parser.parse_args()

    plot_rq1_curves(args.leaderboard_csv, args.out_dir, args.pg_epoch, args.ff_epoch)