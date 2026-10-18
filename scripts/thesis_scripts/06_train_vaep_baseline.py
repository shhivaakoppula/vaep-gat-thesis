# ==============================================================================
# STEP 09 (file 06_train_vaep_baseline.py) - TRAIN THE VAEP BASELINE ON THE VERIFIED RQ1 TARGET
# ==============================================================================
# Builds VAEP's own feature table (the 14 socceraction feature functions on a 3-previous-action game state), joins it to
# the verified target (target_score_next10_literal, checked in Step 08C), trains an XGBoost classifier on the 24 training
# matches and reports AUC-ROC on the 5 validation matches.
# The 5 test matches are NOT touched here. They are reserved for the single final evaluation at the end of the thesis.
# Only VAEP's scoring-probability model is trained, because the RQ1 target is one binary outcome (a goal for the team in
# possession within the next 10 actions). The full offensive-plus-defensive action value formula is not computed.
# Speed and memory changes compared with the original: each match is joined to the target right after its features are
# built (the table of all actions is never kept next to the joined copy), the feature matrix is written in two pieces
# instead of concatenating a third copy, and matches are looked up in dictionaries. Features stay float64 on purpose so the
# saved feature CSV is byte for byte the same as before. Model, seed and parameters are unchanged.
# ==============================================================================

import numpy as np  # arrays
import pandas as pd  # tables
from xgboost import XGBClassifier  # the VAEP model
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, brier_score_loss  # metrics

import common  # paths and helpers
from common import OUTPUT_DIR, print_header, require_file  # shared pieces
import common_spadl as cs  # loader, metadata, SPADL actions
import socceraction.vaep.features as fs  # the VAEP feature functions

TARGET_FILE = OUTPUT_DIR / "rq1_literal_next10_target_34_matches.csv"  # verified target
SPLIT_FILE = OUTPUT_DIR / "final_common_match_split.csv"  # frozen split
FEATURES_FILE = OUTPUT_DIR / "09_vaep_feature_matrix_34_matches.csv"  # full feature matrix for my records
VALIDATION_PREDICTIONS_FILE = OUTPUT_DIR / "09_vaep_validation_predictions.csv"  # one probability per validation action
RESULTS_SUMMARY_FILE = OUTPUT_DIR / "09_vaep_baseline_results_summary.csv"  # metric/value table
FEATURE_IMPORTANCE_FILE = OUTPUT_DIR / "09_vaep_feature_importance.csv"  # importance per feature
NB_PREV_ACTIONS = 3  # current action plus the 3 before it
RANDOM_STATE = 42  # seed
TARGET_COLUMN = "target_score_next10_literal"  # label column
NON_FEATURE_COLUMNS = ["game_id", "action_index", TARGET_COLUMN]  # columns that are not model inputs


def load_verified_target():  # reads the verified target and checks its columns
    print_header("LOAD MY VERIFIED TARGET (from Step 07C, confirmed by Step 08C)")  # title
    target = pd.read_csv(TARGET_FILE)  # saved target
    missing = set(NON_FEATURE_COLUMNS) - set(target.columns)  # absent columns
    if missing:
        raise ValueError(f"Missing target columns: {sorted(missing)}")  # stop with the error: Missing target columns
    target = target[NON_FEATURE_COLUMNS].copy()  # keep the three columns
    target["game_id"] = pd.to_numeric(target["game_id"], errors="raise").astype(int)  # integer key
    target["action_index"] = pd.to_numeric(target["action_index"], errors="raise").astype(int)  # integer key
    target[TARGET_COLUMN] = target[TARGET_COLUMN].astype(bool)  # boolean label
    print(f"Verified rows: {len(target):,}")
    print(f"Verified positive targets: {int(target[TARGET_COLUMN].sum()):,}")  # tell the user: Verified positive targets (plus its value)
    print(f"Positive rate: {target[TARGET_COLUMN].mean() * 100:.4f}%")  # tell the user: Positive rate: % (plus its value)
    return target  # hand back target


def load_split():  # reads the frozen split and lists the match ids
    print_header("LOAD FROZEN MATCH SPLIT")  # title
    split = pd.read_csv(SPLIT_FILE)  # split table
    split["game_id"] = pd.to_numeric(split["game_id"], errors="raise").astype(int)  # integer ids
    split["split"] = split["split"].astype(str).str.lower().str.strip()  # normalised labels
    print(split["split"].value_counts().sort_index())
    train_ids = split[split["split"] == "train"]["game_id"].tolist()  # training matches
    validation_ids = split[split["split"] == "validation"]["game_id"].tolist()  # validation matches
    test_ids = split[split["split"] == "test"]["game_id"].tolist()  # test matches (only counted)
    print(f"Training matches: {len(train_ids)}")
    print(f"Validation matches: {len(validation_ids)}")  # tell the user: Validation matches (plus its value)
    print(f"Test matches (NOT used in this script): {len(test_ids)}")  # tell the user: Test matches (NOT used in this script) (plus its value)
    return train_ids, validation_ids, test_ids  # hand back train_ids, validation_ids, test_ids


def build_features_for_match(loader, game_id, home_team_id):
    """VAEP features for every action of one match (float64, as in the original) plus game_id and action_index."""
    actions = cs.load_actions(loader, game_id, home_team_id)  # SPADL actions with action_index and readable names
    gamestates = fs.gamestates(actions, NB_PREV_ACTIONS)  # list: current action plus the previous ones
    gamestates = fs.play_left_to_right(gamestates, int(home_team_id))  # every team attacks left to right
    features = pd.concat([fn(gamestates) for fn in cs.VAEP_FEATURE_FUNCTIONS], axis=1)  # the 14 feature blocks side by side
    features["game_id"] = int(game_id)  # key 1
    features["action_index"] = actions["action_index"].values  # key 2
    return features


def build_and_join(loader, home, match_ids, label, target_by_game):
    """Build features per match and keep only the rows that exist in the verified target (the eligibility filter).
    Returns the joined table plus the number of feature rows before joining."""
    print_header(f"BUILD VAEP FEATURES — {label} MATCHES")  # title
    joined_parts = []  # one joined table per match
    total_feature_rows = 0  # feature rows before joining
    match_ids_sorted = sorted(match_ids)  # fixed order
    for number, game_id in enumerate(match_ids_sorted, start=1):
        features = build_features_for_match(loader, game_id, int(home[game_id]))  # features of this match
        print(f"[{number}/{len(match_ids_sorted)}] Match {game_id}: {features.shape[0]:,} actions, {features.shape[1]:,} feature columns")
        total_feature_rows += len(features)  # add this match's feature rows to the running total
        # inner join on the two keys keeps the feature order; only eligible rows survive
        joined_parts.append(features.merge(target_by_game.get(game_id, target_by_game["__empty__"]), on=["game_id", "action_index"], how="inner"))
        del features  # free the full-match table straight away
    combined_rows = total_feature_rows
    print()  # print a blank line
    print(f"Total {label} feature rows: {combined_rows:,}")  # tell the user: Total feature rows (plus its value)
    return pd.concat(joined_parts, ignore_index=True), combined_rows  # hand back pd.concat(joined_parts, ignore_index=True), combined_rows


def print_join_summary(rows_before, rows_after):  # prints how many feature rows the join kept and dropped
    print(f"Feature rows before joining to verified target: {rows_before:,}")  # tell the user: Feature rows before joining to verified target (plus its value)
    print(f"Feature rows after keeping only eligible, verified target rows: {rows_after:,}")  # tell the user: Feature rows after keeping only eligible, verified target rows (plus its value)
    print(f"Rows dropped (ineligible, end-of-match actions): {rows_before - rows_after:,}")  # tell the user: Rows dropped (ineligible, end-of-match actions) (plus its value)


def train_xgboost(X_train, y_train):  # trains the XGBoost classifier with the fixed settings
    print_header("TRAIN XGBOOST — MY VAEP BASELINE MODEL")  # title
    n_positive = int(y_train.sum())  # positives in training
    n_negative = int(len(y_train) - n_positive)  # negatives in training
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0  # class-imbalance weight
    print(f"Training positive examples: {n_positive:,}")
    print(f"Training negative examples: {n_negative:,}")  # tell the user: Training negative examples (plus its value)
    print(f"scale_pos_weight I am applying: {scale_pos_weight:.4f}")  # tell the user: scale_pos_weight I am applying (plus its value)
    print("(I calculate this directly from my training data as negative count divided by positive count. It is not an arbitrary "  # tell the user: (I calculate this directly from my training data as negative count...
          "number — it tells XGBoost to treat each positive example as if it appeared this many times, which is the standard fix "
          "for class imbalance this severe.)")
    print()  # print a blank line
    print("XGBoost parameters — initial engineering choices, not yet systematically tuned:")  # tell the user: XGBoost parameters — initial engineering choices, not yet...
    print("  n_estimators = 100  (standard starting value)")  # tell the user: n_estimators = 100 (standard starting value)
    print("  max_depth = 3       (shallow trees, reduces overfitting risk on my current 24-match training set)")  # tell the user: max_depth = 3 (shallow trees, reduces overfitting risk on my...
    print("  learning_rate = 0.1 (standard default step size)")  # tell the user: learning_rate = 0.1 (standard default step size)
    print(f"  random_state = {RANDOM_STATE}  (fixed for exact reproducibility)")  # tell the user: random_state = (fixed for exact reproducibility) (plus its value)
    # use_label_encoder was removed in xgboost 2.0 and was ignored anyway, so it is not passed here
    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, scale_pos_weight=scale_pos_weight,
                          eval_metric="logloss", random_state=RANDOM_STATE)
    print()  # print a blank line
    print("Fitting model on training data...")  # tell the user: Fitting model on training data
    model.fit(X_train, y_train)
    print("Model fitting complete.")  # tell the user: Model fitting complete
    return model, scale_pos_weight  # hand back model, scale_pos_weight


def evaluate_on_validation(model, X_val, y_val):  # computes the validation metrics and prints them
    print_header("EVALUATE ON MY 5 VALIDATION MATCHES")  # title
    y_pred_proba = model.predict_proba(X_val)[:, 1]  # probability of the positive class
    auc_roc = roc_auc_score(y_val, y_pred_proba)  # main metric
    pr_auc = average_precision_score(y_val, y_pred_proba)  # precision-recall AUC
    brier = brier_score_loss(y_val, y_pred_proba)  # calibration
    print(f"Validation rows: {len(y_val):,}")
    print(f"Validation positive rate: {y_val.mean() * 100:.4f}%")  # tell the user: Validation positive rate: % (plus its value)
    print()  # print a blank line
    print("=" * 80)  # print a divider line
    print(f"  MY VAEP BASELINE VALIDATION AUC-ROC = {auc_roc:.4f}")  # tell the user: MY VAEP BASELINE VALIDATION AUC-ROC (plus its value)
    print("=" * 80)  # print a divider line
    print()  # print a blank line
    print(f"PR-AUC (Average Precision): {pr_auc:.4f}")  # tell the user: PR-AUC (Average Precision) (plus its value)
    print(f"Brier score: {brier:.4f}")  # tell the user: Brier score (plus its value)
    print()  # print a blank line
    print("WHAT THIS AUC-ROC MEANS: if I pick one random action that truly led to a goal within the next 10 actions, and one "  # tell the user: WHAT THIS AUC-ROC MEANS: if I pick one random action that truly... (plus its value)
          "random action that did not, my model correctly scores the true one higher "
          f"{auc_roc * 100:.1f}% of the time, measured on 5 matches it never saw during training.")
    print()  # print a blank line
    print("IMPORTANT: this is my VALIDATION result, not my final thesis result. My 5 test matches remain completely untouched. "  # tell the user: IMPORTANT: this is my VALIDATION result, not my final thesis...
          "I will only evaluate on them once, at the very end, after both VAEP and GAT are fully built.")
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)  # ROC points (computed as before, not saved)
    roc_curve_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
    return y_pred_proba, auc_roc, pr_auc, brier, roc_curve_df  # hand back y_pred_proba, auc_roc, pr_auc, brier, roc_curve_df


def main():  # runs the whole step from start to finish
    common.verify_environment()  # optional interpreter check
    print_header("STEP 09 — TRAIN VAEP BASELINE")  # title
    print("Purpose:")
    print("I am training VAEP's own feature representation against my Step 08C-verified target, on my 24 training matches, and "  # tell the user: I am training VAEP's own feature representation against my Step...
          "evaluating on my 5 validation matches. This produces my first real AUC-ROC number for the VAEP side of RQ1.")
    print()  # print a blank line
    print("IMPORTANT:")  # tell the user: IMPORTANT
    print("This script does not touch my 5 test matches.")  # tell the user: This script does not touch my 5 test matches
    print("This script does not compute VAEP's full action-value formula, only the scoring-probability model my target needs.")  # tell the user: This script does not compute VAEP's full action-value formula,...

    require_file(TARGET_FILE, "Verified target (Step 07C/08C)")  # input must exist
    require_file(SPLIT_FILE, "Frozen split")  # input must exist
    target = load_verified_target()
    train_ids, validation_ids, test_ids = load_split()  # the match ids of the three splits
    loader = cs.create_loader()  # the local StatsBomb reader
    metadata = cs.find_match_metadata(loader, train_ids + validation_ids)  # home team ids (test matches are never loaded)
    home = dict(zip(metadata["game_id"], metadata["home_team_id"]))  # game_id -> home team id

    # one small target table per match, so each join only looks at its own rows
    target_by_game = {int(g): part for g, part in target.groupby("game_id", sort=False)}
    target_by_game["__empty__"] = target.iloc[0:0]  # used when a match has no target rows
    train_merged, train_rows_before = build_and_join(loader, home, train_ids, "TRAINING", target_by_game)
    validation_merged, validation_rows_before = build_and_join(loader, home, validation_ids, "VALIDATION", target_by_game)  # validation features joined to the target (continues on the next line)

    print_header("JOIN TRAINING FEATURES TO VERIFIED TARGET")  # print the title of the training join
    print_join_summary(train_rows_before, len(train_merged))  # print the training join counts
    print_header("JOIN VALIDATION FEATURES TO VERIFIED TARGET")  # print the title of the validation join
    print_join_summary(validation_rows_before, len(validation_merged))  # print the validation join counts

    train_merged.to_csv(FEATURES_FILE, index=False)  # first half of the saved feature matrix, with header
    validation_merged.to_csv(FEATURES_FILE, index=False, header=False, mode="a")  # second half appended, no second header
    print()
    print(f"Full feature matrix saved to:\n{FEATURES_FILE}")  # tell the user: Full feature matrix saved to (plus its value)

    feature_columns = [c for c in train_merged.columns if c not in NON_FEATURE_COLUMNS]  # model inputs
    X_train, y_train = train_merged[feature_columns], train_merged[TARGET_COLUMN].astype(int)
    X_val, y_val = validation_merged[feature_columns], validation_merged[TARGET_COLUMN].astype(int)  # validation inputs and labels
    print_header("FINAL TRAINING / VALIDATION MATRICES")  # print the title of the matrix sizes
    print(f"Training matrix: {X_train.shape[0]:,} rows, {X_train.shape[1]:,} feature columns")  # tell the user: Training matrix: rows, feature columns (plus its value)
    print(f"Validation matrix: {X_val.shape[0]:,} rows, {X_val.shape[1]:,} feature columns")  # tell the user: Validation matrix: rows, feature columns (plus its value)

    model, scale_pos_weight = train_xgboost(X_train, y_train)  # train the model
    y_pred_proba, auc_roc, pr_auc, brier, roc_curve_df = evaluate_on_validation(model, X_val, y_val)  # evaluate on the validation matches

    # every individual prediction is saved so the DeLong test can be run later without retraining VAEP
    predictions_out = validation_merged[["game_id", "action_index"]].copy()
    predictions_out["true_target"] = y_val.values  # true label of every validation action
    predictions_out["predicted_probability"] = y_pred_proba  # predicted probability of every validation action
    predictions_out.to_csv(VALIDATION_PREDICTIONS_FILE, index=False)  # save the predictions
    print()  # print a blank line
    print(f"Validation predictions saved to:\n{VALIDATION_PREDICTIONS_FILE}")  # tell the user: Validation predictions saved to (plus its value)

    importance = pd.DataFrame({"feature": feature_columns, "importance": model.feature_importances_}).sort_values("importance", ascending=False)  # feature importance table, largest first (continues on the next line)
    importance.to_csv(FEATURE_IMPORTANCE_FILE, index=False)  # save the importance table
    print(f"Feature importance saved to:\n{FEATURE_IMPORTANCE_FILE}")  # tell the user: Feature importance saved to (plus its value)
    print()  # print a blank line
    print("Top 10 most important features:")  # tell the user: Top 10 most important features
    print(importance.head(10).to_string(index=False))  # print the result of importance.head(10).to_string(index=False)

    summary_rows = [  # metric/value rows for the summary (continues on the next lines)
        {"metric": "train_rows", "value": len(X_train)},
        {"metric": "train_positive", "value": int(y_train.sum())},
        {"metric": "validation_rows", "value": len(X_val)},
        {"metric": "validation_positive", "value": int(y_val.sum())},
        {"metric": "n_feature_columns", "value": len(feature_columns)},
        {"metric": "scale_pos_weight", "value": round(scale_pos_weight, 4)},
        {"metric": "validation_auc_roc", "value": round(auc_roc, 4)},
        {"metric": "validation_pr_auc", "value": round(pr_auc, 4)},
        {"metric": "validation_brier_score", "value": round(brier, 4)},
        {"metric": "n_estimators", "value": 100},
        {"metric": "max_depth", "value": 3},
        {"metric": "learning_rate", "value": 0.1},
        {"metric": "random_state", "value": RANDOM_STATE},
        {"metric": "test_matches_touched", "value": "NO — reserved"},
    ]  # metric/value table
    pd.DataFrame(summary_rows).to_csv(RESULTS_SUMMARY_FILE, index=False)
    print()  # print a blank line
    print(f"Results summary saved to:\n{RESULTS_SUMMARY_FILE}")  # tell the user: Results summary saved to (plus its value)

    print_header("STEP 09 COMPLETE")  # print the closing banner
    print(f"My VAEP baseline validation AUC-ROC is {auc_roc:.4f}, trained on {len(X_train):,} actions from {len(train_ids)} matches, "  # tell the user: My VAEP baseline validation AUC-ROC is , trained on actions from... (plus its value)
          f"evaluated on {len(X_val):,} actions from {len(validation_ids)} matches never seen during training.")
    print()  # print a blank line
    print("NEXT STEP: I train GAT on this exact same verified target and the exact same 24 training / 5 validation match split, "  # tell the user: NEXT STEP: I train GAT on this exact same verified target and the...
          "using my confirmed 7-feature graph representation and 2-layer, 4-head attention architecture. Only after GAT also has a "
          "validation AUC-ROC do I touch my 5 test matches, once, for my final reported comparison.")
    return 0  # success


if __name__ == "__main__":  # only when run directly
    common.run_main(main, "STEP 09")  # shared error handling
