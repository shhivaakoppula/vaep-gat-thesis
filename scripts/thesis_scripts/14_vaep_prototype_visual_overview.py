# ==============================================================================
# 14_vaep_prototype_visual_overview.py - VAEP PROTOTYPE, COMPLETE VISUAL OVERVIEW
# ==============================================================================
# The VAEP equivalent of the GAT prototype visual overview. It builds the pictures that show how the 34-match
# prototype data looks from VAEP's point of view, plus pictures of what the already trained prototype VAEP model
# produced. It does NOT retrain anything: Step 09 already trained the model and saved its results. This script
# reads those files, and rebuilds only a small amount of raw data (action types and one feature) that was not
# saved at prototype scale.
# Charts 1 to 4 describe the raw data shared with the GAT overview. Charts 5 to 9 are VAEP specific: feature
# importance, how the model separates goal-leading actions from the rest, and its ROC and precision-recall curves.
# Speed: the end-of-action feature only needs the current action (a0), so the game state is built with one action
# instead of three, and each match's events are read once. The feature values are identical.
# ==============================================================================

import time  # progress timing

import numpy as np  # arrays
import pandas as pd  # tables
import matplotlib  # plotting library
matplotlib.use("Agg")  # draw to files only
import matplotlib.pyplot as plt  # pyplot interface
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score  # curves and AUC
from socceraction.spadl import statsbomb as statsbomb_spadl  # events -> SPADL actions
import socceraction.vaep.features as fs  # VAEP feature functions
from socceraction import spadl  # SPADL helpers

import common  # shared helpers
from common import OUTPUT_DIR, print_header, require_file  # shared paths and helpers
import common_spadl as cs  # loader and lenient metadata lookup

VISUALS_DIR = OUTPUT_DIR / "vaep_prototype_visuals"  # where the pictures go
TARGET_FILE = OUTPUT_DIR / "rq1_literal_next10_target_34_matches.csv"  # prototype target
SPLIT_FILE = OUTPUT_DIR / "final_common_match_split.csv"  # prototype split
FEATURE_IMPORTANCE_FILE = OUTPUT_DIR / "09_vaep_feature_importance.csv"  # saved by Step 09
VALIDATION_PREDICTIONS_FILE = OUTPUT_DIR / "09_vaep_validation_predictions.csv"  # saved by Step 09
NB_PREV_ACTIONS_FOR_A0 = 1  # only the current action (a0) is needed for end_x_a0 and end_y_a0, and a0 does not depend on this number


def safe_to_bool(series):
    """Convert a column to True/False by checking known spellings (the file loads as int64 0/1, not bool)."""
    return series.astype(str).str.strip().str.lower().map(  # hand back series.astype(str).str.strip().str.lower().map({'true': T...
        {"true": True, "1": True, "1.0": True, "false": False, "0": False, "0.0": False})  # unknown spellings become NaN


def load_target_and_split():  # reads the verified target and the frozen split
    print_header("LOAD MY TARGET AND SPLIT")  # title
    require_file(TARGET_FILE, "Prototype target")  # input 1
    require_file(SPLIT_FILE, "Prototype split")  # input 2
    target = pd.read_csv(TARGET_FILE)  # target table
    split = pd.read_csv(SPLIT_FILE)  # split table
    print("Actual columns in my target file:")
    print(list(target.columns))  # print the result of list(target.columns)
    target["target_bool"] = safe_to_bool(target["target_score_next10_literal"])  # explicit boolean label
    target = target.merge(split, on="game_id", how="left")  # attach the split columns
    print()
    print(f"Target rows loaded: {len(target):,}")  # tell the user: Target rows loaded (plus its value)
    print(f"Matches in split: {len(split):,}")  # tell the user: Matches in split (plus its value)
    return target, split  # hand back target, split


def build_action_types_and_sample_feature(loader, metadata, target, match_ids):
    """Count action types over all matches and compute distance-to-goal at the end of each eligible action."""
    print_header("REBUILD ACTION TYPES AND ONE KEY FEATURE FROM RAW MATCHES")  # title
    print("I do not need 360 data for this step, only raw events, so this runs much faster than my GAT overview's equivalent step.")
    print()  # print a blank line
    home = dict(zip(metadata["game_id"], metadata["home_team_id"]))  # game_id -> home team id
    type_name_counter = {}  # action type name -> count
    feature_rows = []  # one small table per match
    start_time = time.time()
    for number, game_id in enumerate(sorted(match_ids), start=1):  # one match at a time
        if game_id not in home:  # no metadata
            print(f"  [SKIP - no metadata] Match {game_id}")
            continue  # skip to the next item
        try:  # a broken match is skipped
            home_team_id = int(home[game_id])
            events = loader.events(game_id=int(game_id))  # raw events
            if events is None or events.empty:
                continue  # skip to the next item
            actions = statsbomb_spadl.convert_to_actions(events, home_team_id).reset_index(drop=True)  # SPADL actions
            actions["action_index"] = range(len(actions))  # position inside the match
            actions = spadl.add_names(actions)  # readable names
            for tname, count in actions["type_name"].value_counts().items():  # count every action type
                type_name_counter[tname] = type_name_counter.get(tname, 0) + count
            # Distance to goal at the end of the current action: the single most important feature in the full-scale
            # VAEP run. It was not saved at prototype scale, so it is rebuilt here for this one chart.
            gamestates = fs.gamestates(actions, NB_PREV_ACTIONS_FOR_A0)  # only a0 is needed
            gamestates = fs.play_left_to_right(gamestates, home_team_id)  # both teams attack left to right
            end_location = fs.endlocation(gamestates)  # end_x_a0 and end_y_a0
            end_x = end_location["end_x_a0"].to_numpy(dtype=float)
            end_y = end_location["end_y_a0"].to_numpy(dtype=float)  # end height of each action as floats
            dist_to_goal = np.sqrt((105.0 - end_x) ** 2 + (34.0 - end_y) ** 2)  # metres to the centre of the goal line
            target_for_match = target[target["game_id"] == game_id]  # rows of this match
            merged = pd.DataFrame({"action_index": actions["action_index"].values, "end_dist_to_goal": dist_to_goal}).merge(
                target_for_match, on="action_index", how="inner")  # keep eligible actions
            feature_rows.append(merged[["end_dist_to_goal", "target_bool"]])
        except Exception as ex:  # report and go on
            print(f"  [SKIP - match {game_id}] {type(ex).__name__}: {ex}")
            continue  # skip to the next item
        if number % 10 == 0 or number == len(match_ids):  # progress line
            print(f"  [{number}/{len(match_ids)}] processed. Elapsed: {(time.time() - start_time)/60:.1f} min.")
    feature_df = pd.concat(feature_rows, ignore_index=True) if feature_rows else pd.DataFrame()  # one table
    return type_name_counter, feature_df


def chart_action_type_distribution(type_name_counter):  # draws chart 1: how often each action type occurs
    print_header("CHART 1 - ACTION TYPE DISTRIBUTION (CATEGORICAL)")  # title
    if not type_name_counter:
        print("[SKIP] No action types collected.")  # tell the user: [SKIP] No action types collected
        return  # leave the function
    counts = pd.Series(type_name_counter).sort_values(ascending=True)  # smallest first
    print("What this chart shows: how many of each kind of action appear across all my 34 prototype matches - the same "
          "raw data both VAEP and GAT train on.")
    print()  # print a blank line
    print(counts.to_string())  # print the result of counts.to_string()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(counts.index, counts.values, color="#3b6ea5")  # draw horizontal bars
    ax.set_xlabel("Number of actions")  # x axis label
    ax.set_title("Action Type Distribution — My 34 Prototype Matches")  # chart title
    for i, v in enumerate(counts.values):  # loop over the bars
        ax.text(v, i, f" {v:,}", va="center", fontsize=8)  # write the count at the end of each bar
    plt.tight_layout()  # trim the margins
    plt.savefig(VISUALS_DIR / "01_action_type_distribution.png", dpi=150)  # save the picture to the figures folder
    plt.close()  # close the figure to free memory
    print(f"Saved: {VISUALS_DIR / '01_action_type_distribution.png'}")  # tell the user: Saved (plus its value)


def chart_target_distribution(target):  # draws chart 2: how unbalanced the target label is
    print_header("CHART 2 - TARGET LABEL DISTRIBUTION (CATEGORICAL)")  # title
    counts = target["target_bool"].value_counts()
    positive = int(counts.get(True, 0))  # number of positive labels
    negative = int(counts.get(False, 0))  # number of negative labels
    print("This is the exact same target both VAEP and GAT predict.")  # tell the user: This is the exact same target both VAEP and GAT predict
    print(f"Positive: {positive:,}")  # tell the user: Positive (plus its value)
    print(f"Negative: {negative:,}")  # tell the user: Negative (plus its value)
    total = positive + negative
    if total > 0:  # only compute the rate when there is data
        print(f"Positive rate: {positive / total * 100:.4f}%")  # tell the user: Positive rate: % (plus its value)
    fig, ax = plt.subplots(figsize=(6, 6))
    labels = [f"Negative\n{negative:,}", f"Positive\n{positive:,}"]  # text for the two bars
    values = [negative, positive]  # heights of the two bars
    ax.bar(labels, values, color=["#8c8c8c", "#c0392b"])  # draw the two bars
    ax.set_ylabel("Number of actions")  # y axis label
    ax.set_title("My Target Label — Extremely Imbalanced")  # chart title
    for i, v in enumerate(values):  # loop over the bars
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=10)  # write the count above each bar
    plt.tight_layout()  # trim the margins
    plt.savefig(VISUALS_DIR / "02_target_label_distribution.png", dpi=150)  # save the picture to the figures folder
    plt.close()  # close the figure to free memory
    print(f"Saved: {VISUALS_DIR / '02_target_label_distribution.png'}")  # tell the user: Saved (plus its value)


def chart_split_distribution(split):  # draws chart 3: how many matches are in each split
    print_header("CHART 3 - MATCH SPLIT DISTRIBUTION (CATEGORICAL)")  # title
    counts = split["split"].value_counts()
    print(counts.to_string())  # print the result of counts.to_string()
    fig, ax = plt.subplots(figsize=(6, 6))
    colours = {"train": "#3b6ea5", "validation": "#e0a029", "test": "#c0392b"}  # colour of each split
    labels = counts.index.tolist()  # split names for the x axis
    values = counts.values.tolist()  # number of matches per split
    ax.bar(labels, values, color=[colours.get(lbl, "#666666") for lbl in labels])  # draw one bar per split
    ax.set_ylabel("Number of matches")  # y axis label
    ax.set_title("My 34 Prototype Matches — Split Into Three Groups")  # chart title
    for i, v in enumerate(values):  # loop over the bars
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=11)  # write the count above each bar
    plt.tight_layout()  # trim the margins
    plt.savefig(VISUALS_DIR / "03_match_split_distribution.png", dpi=150)  # save the picture to the figures folder
    plt.close()  # close the figure to free memory
    print(f"Saved: {VISUALS_DIR / '03_match_split_distribution.png'}")  # tell the user: Saved (plus its value)


def chart_actions_per_match(target):  # draws chart 4: how many eligible actions each match has
    print_header("CHART 4 - ACTIONS PER MATCH (NUMERICAL)")  # title
    counts = target.groupby("game_id").size()
    print(f"Minimum: {counts.min():,}  Maximum: {counts.max():,}  Mean: {counts.mean():.1f}  Median: {counts.median():.1f}  Std dev: {counts.std():.1f}")  # tell the user: Minimum: Maximum: Mean: Median: Std dev (plus its value)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(counts.values, bins=15, color="#3b6ea5", edgecolor="black")  # draw the histogram of actions per match
    ax.set_xlabel("Eligible actions in one match")  # x axis label
    ax.set_ylabel("Number of matches")  # y axis label
    ax.set_title("How Many Eligible Actions Each of My 34 Matches Has")  # chart title
    plt.tight_layout()  # trim the margins
    plt.savefig(VISUALS_DIR / "04_actions_per_match.png", dpi=150)  # save the picture to the figures folder
    plt.close()  # close the figure to free memory
    print(f"Saved: {VISUALS_DIR / '04_actions_per_match.png'}")  # tell the user: Saved (plus its value)


def chart_feature_importance():
    """Top 15 features of the ALREADY trained prototype model (nothing is recomputed)."""
    print_header("CHART 5 - MY VAEP MODEL'S FEATURE IMPORTANCE")  # title
    if not FEATURE_IMPORTANCE_FILE.exists():
        print(f"[SKIP] {FEATURE_IMPORTANCE_FILE} not found - I need to run my Step 09 VAEP training first.")  # tell the user: [SKIP] not found - I need to run my Step 09 VAEP training first (plus its value)
        return  # leave the function
    importance = pd.read_csv(FEATURE_IMPORTANCE_FILE)
    top15 = importance.sort_values("importance", ascending=False).head(15).sort_values("importance", ascending=True)  # top 15, smallest first for the bar chart
    print("Top 15 features my trained VAEP model relies on most:")
    print(top15.to_string(index=False))  # print the result of top15.to_string(index=False)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top15["feature"], top15["importance"], color="#3b6ea5")  # draw the 15 largest importances as horizontal bars
    ax.set_xlabel("Importance (as reported by my trained model)")  # x axis label
    ax.set_title("My VAEP Model's Top 15 Most Important Features")  # chart title
    plt.tight_layout()  # trim the margins
    plt.savefig(VISUALS_DIR / "05_feature_importance.png", dpi=150)  # save the picture to the figures folder
    plt.close()  # close the figure to free memory
    print(f"Saved: {VISUALS_DIR / '05_feature_importance.png'}")  # tell the user: Saved (plus its value)


def chart_top_feature_by_target(feature_df):  # draws chart 6: the top feature for positive versus negative targets
    print_header("CHART 6 - DISTANCE TO GOAL (END OF ACTION), BY TARGET")  # title
    if feature_df.empty:
        print("[SKIP] No feature data collected.")  # tell the user: [SKIP] No feature data collected
        return  # leave the function
    positive = feature_df[feature_df["target_bool"] == True]["end_dist_to_goal"]
    negative = feature_df[feature_df["target_bool"] == False]["end_dist_to_goal"]  # values of the feature for negative targets
    print(f"Mean distance to goal, POSITIVE actions: {positive.mean():.2f} metres")  # tell the user: Mean distance to goal, POSITIVE actions: metres (plus its value)
    print(f"Mean distance to goal, NEGATIVE actions: {negative.mean():.2f} metres")  # tell the user: Mean distance to goal, NEGATIVE actions: metres (plus its value)
    print("This is my sanity check that my single most important feature carries real, sensible information - actions "  # tell the user: This is my sanity check that my single most important feature...
          "that lead to a goal should end closer to the opponent's goal than actions that do not.")
    fig, ax = plt.subplots(figsize=(8, 6))  # create the figure
    ax.boxplot([negative.dropna(), positive.dropna()], tick_labels=["Negative\n(no goal followed)", "Positive\n(goal followed)"])  # draw one box per group (continues on the next line)
    ax.set_ylabel("Distance to opponent's goal at end of action (metres)")  # y axis label
    ax.set_title("My Top VAEP Feature — Positive vs Negative Target")  # chart title
    plt.tight_layout()  # trim the margins
    plt.savefig(VISUALS_DIR / "06_top_feature_by_target.png", dpi=150)  # save the picture to the figures folder
    plt.close()  # close the figure to free memory
    print(f"Saved: {VISUALS_DIR / '06_top_feature_by_target.png'}")  # tell the user: Saved (plus its value)


def load_validation_predictions():  # reads the saved validation predictions
    print_header("LOAD MY SAVED VALIDATION PREDICTIONS")  # title
    require_file(VALIDATION_PREDICTIONS_FILE, "Validation predictions")
    predictions = pd.read_csv(VALIDATION_PREDICTIONS_FILE)  # the predictions table
    print(f"Rows loaded: {len(predictions):,}")  # tell the user: Rows loaded (plus its value)
    return predictions  # hand back predictions


def chart_predicted_probability_distribution(predictions):  # draws chart 7: the predicted probabilities of both classes
    print_header("CHART 7 - MY MODEL'S PREDICTED PROBABILITIES, BY TRUE LABEL")  # title
    positive = predictions[predictions["true_target"] == 1]["predicted_probability"]
    negative = predictions[predictions["true_target"] == 0]["predicted_probability"]  # predictions of the negative actions
    print("What this chart shows: the actual probability my trained model assigned to every validation action, split by "  # tell the user: What this chart shows: the actual probability my trained model...
          "whether a goal genuinely followed. If my model has learned something real, positive actions should cluster "
          "at higher predicted probabilities than negative ones.")
    fig, ax = plt.subplots(figsize=(9, 6))  # create the figure
    ax.hist(negative, bins=40, alpha=0.6, label=f"Negative (n={len(negative):,})", color="#8c8c8c", density=True)  # histogram of the negatives
    ax.hist(positive, bins=40, alpha=0.6, label=f"Positive (n={len(positive):,})", color="#c0392b", density=True)  # histogram of the positives
    ax.set_xlabel("Predicted probability")  # x axis label
    ax.set_ylabel("Density")  # y axis label
    ax.set_title("My Model's Predicted Probabilities — Positive vs Negative")  # chart title
    ax.legend()  # show the legend
    plt.tight_layout()  # trim the margins
    plt.savefig(VISUALS_DIR / "07_predicted_probability_distribution.png", dpi=150)  # save the picture to the figures folder
    plt.close()  # close the figure to free memory
    print(f"Saved: {VISUALS_DIR / '07_predicted_probability_distribution.png'}")  # tell the user: Saved (plus its value)


def chart_roc_curve(predictions):  # draws chart 8: the ROC curve
    print_header("CHART 8 - MY REAL ROC CURVE")  # title
    y_true = predictions["true_target"].to_numpy()
    y_pred = predictions["predicted_probability"].to_numpy()  # predicted probabilities as an array
    fpr, tpr, _ = roc_curve(y_true, y_pred)  # points of the ROC curve
    auc = roc_auc_score(y_true, y_pred)  # area under the ROC curve
    print(f"My prototype VAEP validation AUC-ROC: {auc:.4f}")  # tell the user: My prototype VAEP validation AUC-ROC (plus its value)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, color="#3b6ea5", linewidth=2, label=f"My VAEP model (AUC = {auc:.4f})")  # draw the model's ROC curve
    ax.plot([0, 1], [0, 1], "k--", label="Random guessing (AUC = 0.5000)")  # draw the diagonal for random guessing
    ax.set_xlabel("False Positive Rate")  # x axis label
    ax.set_ylabel("True Positive Rate")  # y axis label
    ax.set_title("My Prototype VAEP — Real ROC Curve")  # chart title
    ax.legend()  # show the legend
    plt.tight_layout()  # trim the margins
    plt.savefig(VISUALS_DIR / "08_roc_curve.png", dpi=150)  # save the picture to the figures folder
    plt.close()  # close the figure to free memory
    print(f"Saved: {VISUALS_DIR / '08_roc_curve.png'}")  # tell the user: Saved (plus its value)


def chart_pr_curve(predictions):  # draws chart 9: the precision-recall curve
    print_header("CHART 9 - MY REAL PRECISION-RECALL CURVE")  # title
    y_true = predictions["true_target"].to_numpy()
    y_pred = predictions["predicted_probability"].to_numpy()  # predicted probabilities as an array
    precision, recall, _ = precision_recall_curve(y_true, y_pred)  # points of the precision-recall curve
    baseline = y_true.mean()  # precision of random guessing
    print(f"My baseline (random) precision at this class balance: {baseline*100:.2f}%")
    print("This chart matters more than my ROC curve for judging how well my model finds rare positive examples, since "  # tell the user: This chart matters more than my ROC curve for judging how well my...
          "my positive rate is only about 1%.")
    fig, ax = plt.subplots(figsize=(7, 7))  # create the figure
    ax.plot(recall, precision, color="#3b6ea5", linewidth=2, label="My VAEP model")  # draw the model's curve
    ax.axhline(baseline, color="k", linestyle="--", label=f"Random guessing (precision = {baseline*100:.2f}%)")  # draw the baseline for random guessing
    ax.set_xlabel("Recall")  # x axis label
    ax.set_ylabel("Precision")  # y axis label
    ax.set_title("My Prototype VAEP — Real Precision-Recall Curve")  # chart title
    ax.legend()  # show the legend
    plt.tight_layout()  # trim the margins
    plt.savefig(VISUALS_DIR / "09_precision_recall_curve.png", dpi=150)  # save the picture to the figures folder
    plt.close()  # close the figure to free memory
    print(f"Saved: {VISUALS_DIR / '09_precision_recall_curve.png'}")  # tell the user: Saved (plus its value)


def main():  # runs the whole step from start to finish
    common.verify_environment()  # optional interpreter check
    print_header("VAEP PROTOTYPE - COMPLETE VISUAL OVERVIEW")  # title
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)  # output folder
    target, split = load_target_and_split()  # inputs
    chart_target_distribution(target)  # chart 2
    chart_split_distribution(split)  # chart 3
    chart_actions_per_match(target)  # chart 4
    loader = cs.create_loader()  # StatsBomb reader
    match_ids = split["game_id"].tolist()
    metadata = cs.find_match_metadata_lenient(loader, match_ids)  # home team ids
    type_name_counter, feature_df = build_action_types_and_sample_feature(loader, metadata, target, match_ids)  # rebuild raw data
    chart_action_type_distribution(type_name_counter)  # chart 1
    chart_feature_importance()  # chart 5
    chart_top_feature_by_target(feature_df)  # chart 6
    predictions = load_validation_predictions()  # saved Step 09 predictions
    chart_predicted_probability_distribution(predictions)  # chart 7
    chart_roc_curve(predictions)  # chart 8
    chart_pr_curve(predictions)  # chart 9
    print_header("ALL CHARTS COMPLETE")
    print(f"Every chart is saved in:\n{VISUALS_DIR}")  # tell the user: Every chart is saved in (plus its value)
    print()  # print a blank line
    print("What I have now, for my document:")  # tell the user: What I have now, for my document
    print("  01 - Action type distribution (categorical, shared raw data)")  # tell the user: 01 - Action type distribution (categorical, shared raw data)
    print("  02 - Target label distribution (categorical, shared)")  # tell the user: 02 - Target label distribution (categorical, shared)
    print("  03 - Train/validation/test split (categorical, shared)")  # tell the user: 03 - Train/validation/test split (categorical, shared)
    print("  04 - Actions per match (numerical, shared)")  # tell the user: 04 - Actions per match (numerical, shared)
    print("  05 - My VAEP model's top 15 feature importances")  # tell the user: 05 - My VAEP model's top 15 feature importances
    print("  06 - My top feature, positive vs negative target")  # tell the user: 06 - My top feature, positive vs negative target
    print("  07 - My model's predicted probabilities, by true label")  # tell the user: 07 - My model's predicted probabilities, by true label
    print("  08 - My real ROC curve")  # tell the user: 08 - My real ROC curve
    print("  09 - My real Precision-Recall curve")  # tell the user: 09 - My real Precision-Recall curve
    return 0  # success


if __name__ == "__main__":  # only when run directly
    common.run_main(main, "VAEP VISUAL OVERVIEW")  # shared error handling
