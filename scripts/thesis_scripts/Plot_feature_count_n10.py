# ==============================================================================
# CHART FOR STEP 22 - GAT FEATURE-COUNT LADDER AT N=10
# ==============================================================================
# Reads 22_gat_feature_count_ablation_summary.csv and draws a bar chart of validation AUC-ROC by
# feature-count config, ordered from 8 features down to 1. No numbers are invented; if the summary
# file is missing the script says so and stops.
# ==============================================================================

import sys  # for sys.exit
import pandas as pd  # reading the summary table
import matplotlib  # plotting library
matplotlib.use("Agg")  # draw straight to a PNG file, no window needed
import matplotlib.pyplot as plt  # plotting interface

from common import OUTPUT_DIR  # shared output folder

SUMMARY_FILE = OUTPUT_DIR / "22_gat_feature_count_ablation_summary.csv"  # Step 22 result table
CHART_FILE = OUTPUT_DIR / "22_chart_feature_count_n10.png"  # figure written here

if not SUMMARY_FILE.exists():  # Step 22 has not been run yet
    print(f"CANNOT FIND STEP 22's SUMMARY FILE\nExpected it at:\n  {SUMMARY_FILE}\nRun Step 22 first.")  # explain
    sys.exit(1)  # stop
print(f"[PASS] Step 22 summary file: {SUMMARY_FILE}")  # confirm input

results_df = pd.read_csv(SUMMARY_FILE).sort_values("n_features", ascending=False).reset_index(drop=True)  # 8 features first
print(f"Feature-count configs found: {results_df['feature_set'].tolist()}")  # list them

fig, ax = plt.subplots(figsize=(9, 6))  # one panel
best_idx = results_df["validation_auc_roc"].idxmax()  # row of the best AUC
colors = ["#2da44e" if i == best_idx else "#1f6feb" for i in range(len(results_df))]  # green for the best bar, blue for the rest
bars = ax.bar(results_df["feature_set"], results_df["validation_auc_roc"], color=colors)  # one bar per config
for bar, (_, row) in zip(bars, results_df.iterrows()):  # label every bar
    ax.annotate(f"{row['validation_auc_roc']:.4f}\n({int(row['n_features'])} feat)",  # AUC and feature count
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),  # anchor at the top of the bar
                textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)  # nudge up a little
ax.set_ylabel("Validation AUC-ROC")  # y label
ax.set_title("GAT validation AUC-ROC by feature-count config, at N=10\n(green bar = best in this ladder)")  # title
ax.set_ylim(max(0, results_df["validation_auc_roc"].min() - 0.03), results_df["validation_auc_roc"].max() + 0.03)  # zoom on the AUC range
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")  # tilt the long labels
ax.grid(True, axis="y", alpha=0.3)  # light horizontal grid

new_row = results_df[results_df["feature_set"] == "count8_add_dist_to_actor"]  # the 8-feature config
baseline_row = results_df[results_df["feature_set"] == "count7_full"]  # the original 7-feature config
if not new_row.empty and not baseline_row.empty:  # only when both exist
    new_auc, baseline_auc = float(new_row.iloc[0]["validation_auc_roc"]), float(baseline_row.iloc[0]["validation_auc_roc"])  # the two AUCs
    fig.text(0.5, 0.01, f"dist_to_actor check: count8={new_auc:.4f} vs. count7_full={baseline_auc:.4f}  (difference: {new_auc - baseline_auc:+.4f})",  # caption
             ha="center", fontsize=9, style="italic")  # centred italic note

fig.tight_layout(rect=(0, 0.04, 1, 1))  # leave room for the caption
fig.savefig(CHART_FILE, dpi=200)  # write the PNG
print(f"\nChart saved to:\n{CHART_FILE}")  # where
