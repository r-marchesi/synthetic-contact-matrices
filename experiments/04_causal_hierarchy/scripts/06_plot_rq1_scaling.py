import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RES_DIR = os.path.join(EXP_DIR, "results")

RUN_PATTERN = re.compile(r"^(\d+)_percent_(plan_guided|full_free)_epoch_(\d+)$")


def plot_rq1_curves(leaderboard_csv, out_dir, pg_epoch=6, ff_epoch=6):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(leaderboard_csv)

    records = []
    for _, row in df.iterrows():
        m = RUN_PATTERN.match(str(row["Run"]))
        if not m:
            continue
        pct = int(m.group(1))
        mode = m.group(2)
        ep = int(m.group(3))

        if (mode == "plan_guided" and ep == pg_epoch) or (mode == "full_free" and ep == ff_epoch):
            records.append({
                "Split_Pct": pct,
                "Mode": f"Plan-Guided (Ep {ep})" if mode == "plan_guided" else f"Full-Free (Ep {ep})",
                "MAE_Total": row["MAE_Total"],
                "MAE_Home": row["MAE_Home"],
                "MAE_Work": row["MAE_Work"],
                "MAE_School": row["MAE_School"],
                "Volume_TVD": row["Volume_TVD"],
                "Corr_MAE": row["Corr_MAE"],
                "Marginal_C2ST_AUC": row["Marginal_C2ST_AUC"],
                "Joint_C2ST_AUC": row["Joint_C2ST_AUC"],
                "R0_Ratio": row["R0_Ratio"],
            })

    rq1_df = pd.DataFrame(records).sort_values(["Mode", "Split_Pct"])
    if rq1_df.empty:
        print("No matching RQ1 runs found. Check --pg_epoch and --ff_epoch arguments.")
        return

    out_csv = os.path.join(out_dir, "rq1_scaling_summary.csv")
    rq1_df.to_csv(out_csv, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("RQ1: Impact of Fine-Tuning Dataset Size on Synthetic Contact Quality", fontsize=17, y=1.02)

    metrics = [
        ("MAE_Total", "A. 15x15 Contact Matrix MAE (Lower is Better)", axes[0, 0]),
        ("Volume_TVD", "B. Contact Volume TVD (Lower is Better)", axes[0, 1]),
        ("Corr_MAE", "C. Pairwise Cramér's V Correlation MAE (Lower is Better)", axes[1, 0]),
        ("Joint_C2ST_AUC", "D. Joint 19-Feature C2ST ROC-AUC (0.50 = Ideal)", axes[1, 1]),
    ]

    colors = {"Plan-Guided": "#1f77b4", "Full-Free": "#ff7f0e"}

    for col, title, ax in metrics:
        for mode_label, grp in rq1_df.groupby("Mode"):
            c = colors["Plan-Guided"] if "Plan-Guided" in mode_label else colors["Full-Free"]
            ax.plot(
                grp["Split_Pct"], grp[col],
                marker="o", linewidth=2.5, markersize=8, label=mode_label, color=c
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

    print("\n================================================================================")
    print("📈 RQ1 SCALING SUMMARY TABLE")
    print("================================================================================")
    print(rq1_df.to_string(index=False, float_format="%.4f"))
    print(f"\nSaved RQ1 summary CSV -> {out_csv}")
    print(f"Saved RQ1 scaling plot -> {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--leaderboard_csv",
        type=str,
        default=os.path.join(RES_DIR, "leaderboard_04_causal.csv"),
    )
    parser.add_argument("--out_dir", type=str, default=os.path.join(RES_DIR, "rq1_scaling"))
    parser.add_argument("--pg_epoch", type=int, default=8)
    parser.add_argument("--ff_epoch", type=int, default=8)
    args = parser.parse_args()

    plot_rq1_curves(args.leaderboard_csv, args.out_dir, args.pg_epoch, args.ff_epoch)