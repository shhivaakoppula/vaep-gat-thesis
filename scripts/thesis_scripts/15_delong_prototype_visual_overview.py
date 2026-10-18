# ==============================================================================
# 15_delong_prototype_visual_overview.py - DELONG TEST (PROTOTYPE SCALE), COMPLETE VISUAL OVERVIEW
# ==============================================================================
# Builds the five pictures that show the prototype-scale comparison between VAEP and GAT: the
# paired DeLong result, not only the two AUC numbers. It does NOT retrain anything and does NOT
# recompute the DeLong test. Step 11 already saved the result and the paired predictions, and this
# script only reads those files and draws them.
# No outside literature is cited here, in line with the other prototype visual scripts.
# ==============================================================================

import matplotlib  # plotting library
matplotlib.use("Agg")  # draw to files only, so no window is needed
import matplotlib.pyplot as plt  # pyplot interface
import pandas as pd  # tables
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score  # curves and AUC

import common  # shared helpers
from common import OUTPUT_DIR, print_header, require_file  # shared paths and helpers

VISUALS_DIR = OUTPUT_DIR / "delong_prototype_visuals"  # where the pictures go
DELONG_RESULT_FILE = OUTPUT_DIR / "11_delong_test_result.csv"  # saved DeLong result (one row)
COMMON_PREDICTIONS_FILE = OUTPUT_DIR / "11_delong_common_predictions.csv"  # saved paired predictions
ALPHA = 0.05  # significance level used in Step 11
VAEP_COLOUR, GAT_COLOUR = "#3b6ea5", "#c0392b"  # blue for VAEP, red for GAT


def load_everything():  # reads the DeLong result and the paired predictions
    print_header("LOAD MY SAVED DELONG RESULT AND PAIRED PREDICTIONS")  # title
    require_file(DELONG_RESULT_FILE, "DeLong test result")  # input 1
    require_file(COMMON_PREDICTIONS_FILE, "Paired common predictions")  # input 2
    result = pd.read_csv(DELONG_RESULT_FILE)  # one-row result table
    paired = pd.read_csv(COMMON_PREDICTIONS_FILE)  # paired predictions
    print("My saved DeLong result:")  # label
    print(result.to_string(index=False))  # show it
    print()  # blank line
    print(f"Paired observations: {len(paired):,}")  # size
    return result, paired  # both tables


def chart_overlaid_roc_curves(paired):  # draws the two ROC curves on one plot
    print_header("CHART 1 - MY REAL ROC CURVES, GAT vs VAEP, OVERLAID")  # title
    y_true = paired["true_target"].to_numpy()  # labels
    vaep_prob = paired["vaep_probability"].to_numpy()  # VAEP scores
    gat_prob = paired["gat_probability"].to_numpy()  # GAT scores
    vaep_fpr, vaep_tpr, _ = roc_curve(y_true, vaep_prob)  # VAEP curve
    gat_fpr, gat_tpr, _ = roc_curve(y_true, gat_prob)  # GAT curve
    vaep_auc = roc_auc_score(y_true, vaep_prob)  # VAEP AUC
    gat_auc = roc_auc_score(y_true, gat_prob)  # GAT AUC
    print(f"VAEP AUC-ROC on this paired set: {vaep_auc:.4f}")  # report
    print(f"GAT AUC-ROC on this paired set: {gat_auc:.4f}")  # report
    fig, ax = plt.subplots(figsize=(8, 8))  # square canvas
    ax.plot(vaep_fpr, vaep_tpr, color=VAEP_COLOUR, linewidth=2, label=f"VAEP (AUC = {vaep_auc:.4f})")  # VAEP line
    ax.plot(gat_fpr, gat_tpr, color=GAT_COLOUR, linewidth=2, label=f"GAT (AUC = {gat_auc:.4f})")  # GAT line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random guessing (AUC = 0.5000)")  # chance diagonal
    ax.set_xlabel("False Positive Rate")  # x label
    ax.set_ylabel("True Positive Rate")  # y label
    ax.set_title("My Prototype ROC Curves — VAEP vs GAT\n" f"Same {len(y_true):,} paired validation observations")  # title
    ax.legend(loc="lower right")  # legend
    plt.tight_layout()  # fit the layout
    plt.savefig(VISUALS_DIR / "01_overlaid_roc_curves.png", dpi=150)  # save
    plt.close()  # free the figure
    print(f"Saved: {VISUALS_DIR / '01_overlaid_roc_curves.png'}")  # confirm


def chart_overlaid_pr_curves(paired):  # draws the two precision-recall curves on one plot
    print_header("CHART 2 - MY REAL PRECISION-RECALL CURVES, OVERLAID")  # title
    y_true = paired["true_target"].to_numpy()  # labels
    vaep_precision, vaep_recall, _ = precision_recall_curve(y_true, paired["vaep_probability"].to_numpy())  # VAEP curve
    gat_precision, gat_recall, _ = precision_recall_curve(y_true, paired["gat_probability"].to_numpy())  # GAT curve
    baseline = y_true.mean()  # precision of random guessing = share of positives
    print(f"Random-guessing baseline precision at this class balance: {baseline*100:.2f}%")  # report
    fig, ax = plt.subplots(figsize=(8, 8))  # square canvas
    ax.plot(vaep_recall, vaep_precision, color=VAEP_COLOUR, linewidth=2, label="VAEP")  # VAEP line
    ax.plot(gat_recall, gat_precision, color=GAT_COLOUR, linewidth=2, label="GAT")  # GAT line
    ax.axhline(baseline, color="k", linestyle="--", alpha=0.6, label=f"Random guessing (precision = {baseline*100:.2f}%)")  # baseline
    ax.set_xlabel("Recall")  # x label
    ax.set_ylabel("Precision")  # y label
    ax.set_title("My Prototype Precision-Recall Curves — VAEP vs GAT")  # title
    ax.legend()  # legend
    plt.tight_layout()  # fit the layout
    plt.savefig(VISUALS_DIR / "02_overlaid_pr_curves.png", dpi=150)  # save
    plt.close()  # free the figure
    print(f"Saved: {VISUALS_DIR / '02_overlaid_pr_curves.png'}")  # confirm


def chart_auc_comparison_bar(result):  # draws the AUC bar chart with the DeLong verdict
    print_header("CHART 3 - AUC-ROC COMPARISON, WITH MY DELONG RESULT")  # title
    row = result.iloc[0]  # the single result row
    vaep_auc = float(row["vaep_auc_roc_paired_common"])  # VAEP AUC
    gat_auc = float(row["gat_auc_roc_paired_common"])  # GAT AUC
    p_value = float(row["delong_p_value"])  # p-value
    z_stat = float(row["delong_z_statistic"])  # z statistic
    significant = bool(row["significant_at_0.05"])  # verdict flag
    print(f"VAEP AUC-ROC: {vaep_auc:.4f}")  # report
    print(f"GAT AUC-ROC: {gat_auc:.4f}")
    print(f"DeLong z-statistic: {z_stat:.4f}")  # show the DeLong z-statistic
    print(f"DeLong p-value: {p_value:.6f}")  # show the p-value
    print(f"Significant at 0.05: {significant}")  # show whether the result is significant at 0.05
    fig, ax = plt.subplots(figsize=(7, 7))  # canvas
    bars = ax.bar(["VAEP", "GAT"], [vaep_auc, gat_auc], color=[VAEP_COLOUR, GAT_COLOUR], width=0.5)  # two bars
    for bar, value in zip(bars, [vaep_auc, gat_auc]):  # label each bar
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.4f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.0)  # AUC range
    ax.set_ylabel("AUC-ROC")  # y label
    verdict = f"p = {p_value:.4f} — statistically significant" if significant else f"p = {p_value:.4f} — NOT statistically significant"  # title text
    ax.set_title(f"My Prototype AUC-ROC: VAEP vs GAT\n" f"Paired DeLong test: {verdict}")  # title
    plt.tight_layout()  # fit the layout
    plt.savefig(VISUALS_DIR / "03_auc_comparison_bar.png", dpi=150)  # save
    plt.close()  # free the figure
    print(f"Saved: {VISUALS_DIR / '03_auc_comparison_bar.png'}")  # confirm


def chart_model_agreement_scatter(paired):  # draws the scatter of the two models' scores
    print_header("CHART 4 - WHERE VAEP AND GAT AGREE OR DISAGREE")  # title
    print("What this chart shows: every paired validation action, plotted by VAEP's predicted probability on one axis "
          "and GAT's predicted probability on the other. Points near the diagonal line are where my two models agree "
          "closely. Points far from the diagonal are where they disagree.")  # explanation
    negative = paired[paired["true_target"] == 0]  # actions that were not followed by a goal
    positive = paired[paired["true_target"] == 1]  # actions that were
    fig, ax = plt.subplots(figsize=(8, 8))  # canvas
    ax.scatter(negative["vaep_probability"], negative["gat_probability"], s=8, alpha=0.25, color="#8c8c8c", label=f"Negative (n={len(negative):,})")  # grey points
    ax.scatter(positive["vaep_probability"], positive["gat_probability"], s=30, alpha=0.85, color=GAT_COLOUR, label=f"Positive (n={len(positive):,})", edgecolors="black", linewidth=0.3)  # red points on top
    max_val = max(paired["vaep_probability"].max(), paired["gat_probability"].max())  # axis limit for the diagonal
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="Perfect agreement line")  # diagonal
    ax.set_xlabel("VAEP predicted probability")  # x label
    ax.set_ylabel("GAT predicted probability")  # y label
    ax.set_title("Where My Two Models Agree and Disagree")  # title
    ax.legend()  # legend
    plt.tight_layout()  # fit the layout
    plt.savefig(VISUALS_DIR / "04_model_agreement_scatter.png", dpi=150)  # save
    plt.close()  # free the figure
    print(f"Saved: {VISUALS_DIR / '04_model_agreement_scatter.png'}")  # confirm


def chart_delong_scorecard(result):  # draws the one-page scorecard of the DeLong result
    print_header("CHART 5 - MY DELONG TEST RESULT, AS ONE CLEAR PICTURE")  # title
    row = result.iloc[0]  # the single result row
    vaep_auc = float(row["vaep_auc_roc_paired_common"])
    gat_auc = float(row["gat_auc_roc_paired_common"])  # GAT AUC on the shared set
    diff = float(row["auc_difference_vaep_minus_gat"])  # AUC gap (VAEP minus GAT)
    z_stat = float(row["delong_z_statistic"])  # DeLong z-statistic
    p_value = float(row["delong_p_value"])  # DeLong p-value
    n_common = int(row["n_common_observations"])  # number of shared observations
    n_positive = int(row["n_positive_common"])  # number of positives among them
    significant = bool(row["significant_at_0.05"])  # significant at the 0.05 level
    fig, ax = plt.subplots(figsize=(9, 6))  # canvas
    ax.axis("off")  # text only
    lines = [  # label / value pairs
        ("Paired validation observations", f"{n_common:,}"),
        ("Positive examples in paired set", f"{n_positive:,}"),
        ("VAEP AUC-ROC", f"{vaep_auc:.4f}"),
        ("GAT AUC-ROC", f"{gat_auc:.4f}"),
        ("Difference (VAEP minus GAT)", f"{diff:.4f}"),
        ("DeLong z-statistic", f"{z_stat:.4f}"),
        ("DeLong p-value (two-sided)", f"{p_value:.6f}"),
        ("Significant at α = 0.05?", "YES" if significant else "NO — not significant at this scale"),
    ]
    y = 0.95  # start near the top
    for label, value in lines:  # one row per pair
        ax.text(0.05, y, label, fontsize=13, fontweight="bold", transform=ax.transAxes)  # left column
        ax.text(0.65, y, value, fontsize=13, transform=ax.transAxes)  # right column
        y -= 0.11  # move down
    ax.set_title("My Prototype-Scale DeLong Test — Full Result Summary", fontsize=14, fontweight="bold")  # title
    plt.tight_layout()  # fit the layout
    plt.savefig(VISUALS_DIR / "05_delong_scorecard.png", dpi=150)  # save
    plt.close()  # free the figure
    print(f"Saved: {VISUALS_DIR / '05_delong_scorecard.png'}")  # confirm


def main():  # runs the whole step from start to finish
    common.verify_environment()  # optional interpreter check
    print_header("DELONG TEST (PROTOTYPE SCALE) - VISUAL OVERVIEW")  # title
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)  # make the output folder
    result, paired = load_everything()  # read the saved files
    chart_overlaid_roc_curves(paired)  # picture 1
    chart_overlaid_pr_curves(paired)  # picture 2
    chart_auc_comparison_bar(result)  # picture 3
    chart_model_agreement_scatter(paired)  # picture 4
    chart_delong_scorecard(result)  # picture 5
    print_header("ALL CHARTS COMPLETE")  # done
    print(f"Every chart is saved in:\n{VISUALS_DIR}")  # where
    print()  # blank line
    print("What I have now, for my comparison document:")  # recap
    print("  01 - My real ROC curves, VAEP and GAT, overlaid")
    print("  02 - My real Precision-Recall curves, overlaid")  # list figure 02 for the user
    print("  03 - AUC-ROC bar comparison, with my DeLong verdict")  # list figure 03 for the user
    print("  04 - Where my two models agree and disagree")  # list figure 04 for the user
    print("  05 - My full DeLong test result, as one clear scorecard")  # list figure 05 for the user
    return 0  # success


if __name__ == "__main__":  # only when run directly
    common.run_main(main, "DELONG VISUAL OVERVIEW")  # shared error handling
