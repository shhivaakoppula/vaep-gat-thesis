# ==============================================================================
# CHART FOR STEP 21 - VAEP AUC-ROC BY LABEL HORIZON N
# ==============================================================================
# Reads 21_label_horizon_ablation_summary.csv (no new model runs) and draws two panels:
# validation AUC-ROC vs N with the frozen N=10 choice marked, and validation positive rate vs N.
# ==============================================================================

import sys  # for sys.exit
import pandas as pd  # reading the summary table
import matplotlib  # plotting library
matplotlib.use("Agg")  # write PNG files directly, no GUI window
import matplotlib.pyplot as plt  # plotting interface

from common import OUTPUT_DIR  # shared output folder

SUMMARY_FILE = OUTPUT_DIR / "21_label_horizon_ablation_summary.csv"  # Step 21 result table
CHART_FILE = OUTPUT_DIR / "21_chart_label_horizon.png"  # figure written here
DOCUMENTED_HORIZON = 10  # the pre-committed RQ1 choice

if not SUMMARY_FILE.exists():  # Step 21 has not been run yet
    print(f"CANNOT FIND STEP 21's SUMMARY FILE\nExpected it at:\n  {SUMMARY_FILE}\nRun the label-horizon ablation first.")  # explain
    sys.exit(1)  # stop
print(f"[PASS] Step 21 summary file: {SUMMARY_FILE}")  # confirm input

results_df = pd.read_csv(SUMMARY_FILE).sort_values("N").reset_index(drop=True)  # horizons in increasing order
print(f"Horizons found in summary: {results_df['N'].tolist()}")  # list them

fig, (ax_auc, ax_rate) = plt.subplots(2, 1, figsize=(8, 9), sharex=True)  # two stacked panels sharing the x axis

ax_auc.plot(results_df["N"], results_df["validation_auc_roc"], marker="o", linewidth=2, color="#1f6feb")  # AUC line
documented_row = results_df[results_df["N"] == DOCUMENTED_HORIZON]  # the N=10 point
if not documented_row.empty:  # highlight it when present
    ax_auc.scatter(documented_row["N"], documented_row["validation_auc_roc"], color="#d1242f", zorder=5, s=110,  # big red dot
                   label=f"N={DOCUMENTED_HORIZON} (documented RQ1 choice)")  # legend text
    ax_auc.legend(loc="upper right")  # show the legend
for _, row in results_df.iterrows():  # label every point
    ax_auc.annotate(f"{row['validation_auc_roc']:.4f}", (row["N"], row["validation_auc_roc"]),  # AUC value
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)  # above the point
ax_auc.set_ylabel("Validation AUC-ROC")  # y label
ax_auc.set_title("VAEP validation AUC-ROC by label horizon N\n(shorter horizon = mechanically easier prediction task)")  # title
ax_auc.grid(True, alpha=0.3)  # light grid

ax_rate.plot(results_df["N"], results_df["validation_positive_rate_pct"], marker="s", linewidth=2, color="#8250df")  # positive-rate line
for _, row in results_df.iterrows():  # label every point
    ax_rate.annotate(f"{row['validation_positive_rate_pct']:.2f}%", (row["N"], row["validation_positive_rate_pct"]),  # rate value
                     textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)  # above the point
ax_rate.set_xlabel("Label horizon N (actions)")  # x label
ax_rate.set_ylabel("Validation positive rate (%)")  # y label
ax_rate.set_title("Validation positive rate by label horizon N")  # title
ax_rate.grid(True, alpha=0.3)  # light grid
ax_rate.set_xticks(results_df["N"])  # one tick per horizon

fig.tight_layout()  # tidy spacing
fig.savefig(CHART_FILE, dpi=200)  # write the PNG
print(f"\nChart saved to:\n{CHART_FILE}")  # where
