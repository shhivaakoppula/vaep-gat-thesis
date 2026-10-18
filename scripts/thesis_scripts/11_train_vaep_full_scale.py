# ==============================================================================
# STEP 13 (file 11_train_vaep_full_scale.py) - TRAIN THE VAEP BASELINE AT FULL SCALE
# ==============================================================================
# Builds the 14-function VAEP feature table for every training and validation match, keeps only the
# rows that are eligible in the verified Step 12 target, trains XGBoost (100 trees, depth 3, lr 0.1,
# scale_pos_weight = negatives / positives, seed 42) and evaluates on the validation matches.
# Memory change compared with the original: each match is joined to the target right after its features are
# built, so the full table of ALL actions is never held next to the joined copy, and the key/label columns are
# popped out of the frame instead of copying a second feature-only table. Rows, row order and columns are the same.
# ==============================================================================

import time  # progress timing
import numpy as np  # numeric helpers
import pandas as pd  # tables
from xgboost import XGBClassifier  # the VAEP model
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss  # headline metrics

import common  # paths and helpers
from common import OUTPUT_DIR, print_header, require_file  # shared pieces
import common_spadl as cs  # loader, metadata, VAEP features

FULL_TARGET_FILE = OUTPUT_DIR / "rq1_literal_next10_target_full_dataset.csv"  # Step 12 target
FULL_SPLIT_FILE = OUTPUT_DIR / "full_scale_match_split.csv"  # Step 12 split
VALIDATION_PREDICTIONS_FILE = OUTPUT_DIR / "13_vaep_full_scale_validation_predictions.csv"  # predictions (used by Steps 15/18/24...)
RESULTS_SUMMARY_FILE = OUTPUT_DIR / "13_vaep_full_scale_results_summary.csv"  # metric/value table
FEATURE_IMPORTANCE_FILE = OUTPUT_DIR / "13_vaep_full_scale_feature_importance.csv"  # importance per feature
MODEL_SAVE_FILE = OUTPUT_DIR / "13_vaep_full_scale_model.json"  # saved XGBoost model
RANDOM_STATE = 42  # seed
TARGET_COLUMN = "target_score_next10_literal"  # label column
NON_FEATURE_COLUMNS = ["game_id", "action_index", TARGET_COLUMN]  # columns that are not model inputs


def load_verified_target():
    """Read the Step 12 target (only the three needed columns) with strict types."""
    print_header("LOAD MY VERIFIED FULL-SCALE TARGET (Step 12)")  # title
    target = pd.read_csv(FULL_TARGET_FILE)  # whole file, so a missing column is detected like before
    missing = set(NON_FEATURE_COLUMNS) - set(target.columns)  # absent columns
    if missing:  # file is wrong
        raise ValueError(f"Missing target columns: {sorted(missing)}")  # stop
    target = target[NON_FEATURE_COLUMNS].copy()  # keep the three columns
    target["game_id"] = pd.to_numeric(target["game_id"], errors="raise").astype(int)  # integer key
    target["action_index"] = pd.to_numeric(target["action_index"], errors="raise").astype(int)  # integer key
    target[TARGET_COLUMN] = target[TARGET_COLUMN].astype(bool)  # boolean label
    print(f"Verified rows: {len(target):,} | positives: {int(target[TARGET_COLUMN].sum()):,} | rate: {target[TARGET_COLUMN].mean() * 100:.4f}%")  # summary
    return target  # clean table


def load_split():
    """Match ids of the training and validation sets."""
    print_header("LOAD FULL-SCALE MATCH SPLIT")  # title
    split = pd.read_csv(FULL_SPLIT_FILE)  # split table
    split["game_id"] = pd.to_numeric(split["game_id"], errors="raise").astype(int)  # integer ids
    split["split"] = split["split"].astype(str).str.lower().str.strip()  # normalise labels
    print(split["split"].value_counts().sort_index())  # counts
    train_ids = split[split["split"] == "train"]["game_id"].tolist()  # training matches
    validation_ids = split[split["split"] == "validation"]["game_id"].tolist()  # validation matches
    print(f"Training matches: {len(train_ids)} | Validation matches: {len(validation_ids)}")  # counts
    return train_ids, validation_ids  # two lists


def build_joined_features(loader, metadata, match_ids, target_by_game, label):
    """Build features per match and keep only rows that exist in the verified target (inner join, left order kept)."""
    print_header(f"BUILD VAEP FEATURES - {label} MATCHES")  # title
    home = dict(zip(metadata["game_id"], metadata["home_team_id"]))  # game_id -> home team id (faster than filtering per match)
    ids_sorted = sorted(match_ids)  # fixed order, same as before
    parts, failed, n_rows_built, start = [], [], 0, time.time()  # collected frames, failures, row counter, timer
    for number, game_id in enumerate(ids_sorted, start=1):  # one match at a time
        if game_id not in home:  # no metadata
            print(f"  [SKIP - no metadata] Match {game_id}")  # report
            failed.append(int(game_id))  # remember
            continue
        try:  # a broken match must not stop the run
            features = cs.build_features_for_match(loader, game_id, int(home[game_id]))  # all actions of the match
            n_rows_built += len(features)  # count before filtering
            tgt = target_by_game.get(game_id)  # eligible rows of this match
            if tgt is None:  # match absent from the target
                tgt = pd.DataFrame({"game_id": [], "action_index": [], TARGET_COLUMN: []})  # empty join
            parts.append(features.merge(tgt, on=["game_id", "action_index"], how="inner"))  # keep eligible rows only
        except Exception as ex:  # report and move on
            print(f"  [FAILED] Match {game_id}: {ex}")  # report
            failed.append(int(game_id))  # remember
            continue
        if number % 20 == 0 or number == len(ids_sorted):  # progress line
            elapsed = time.time() - start  # seconds so far
            rate = number / elapsed if elapsed > 0 else 0  # matches per second
            print(f"  [{number}/{len(ids_sorted)}] processed. Elapsed {elapsed / 60:.1f} min. "
                  f"Estimated remaining {((len(ids_sorted) - number) / rate / 60) if rate > 0 else float('nan'):.1f} min.")  # show
    merged = pd.concat(parts, ignore_index=True)  # one table, same row order as the old global merge
    del parts  # free the pieces
    print(f"Feature rows built: {n_rows_built:,} | rows after join to verified eligible target: {len(merged):,} | dropped: {n_rows_built - len(merged):,}")  # report
    if failed:  # failures
        print(f"Matches that failed: {failed} ({len(failed)} of {len(ids_sorted)})")  # report
    return merged, failed  # joined table and failures


def main():  # runs the whole step from start to finish
    common.verify_environment()  # optional interpreter check
    print_header("STEP 13 - TRAIN VAEP AT FULL SCALE")  # title
    require_file(FULL_TARGET_FILE, "Verified full-scale target")  # input 1
    require_file(FULL_SPLIT_FILE, "Full-scale split")  # input 2
    target = load_verified_target()  # labels
    train_ids, validation_ids = load_split()  # match ids
    target_by_game = {gid: grp for gid, grp in target.groupby("game_id")}  # per-match slices for the per-match join
    loader = cs.create_loader()  # StatsBomb reader
    metadata = cs.find_match_metadata(loader, train_ids + validation_ids)  # home team ids

    train_merged, train_failed = build_joined_features(loader, metadata, train_ids, target_by_game, "TRAINING")  # training rows
    validation_merged, val_failed = build_joined_features(loader, metadata, validation_ids, target_by_game, "VALIDATION")  # validation rows

    y_train = train_merged.pop(TARGET_COLUMN).astype(int)  # label out of the frame, in place (no second copy)
    y_val = validation_merged.pop(TARGET_COLUMN).astype(int)  # same for validation
    val_keys = validation_merged[["game_id", "action_index"]].copy()  # keys needed for the predictions file
    train_merged = train_merged.drop(columns=["game_id", "action_index"])  # features only (training)
    X_val = validation_merged.drop(columns=["game_id", "action_index"])  # features only (validation)
    X_train = train_merged  # rename for clarity
    del validation_merged  # free memory
    feature_columns = list(X_train.columns)  # model input names in model order
    print_header("FINAL TRAINING / VALIDATION MATRICES")  # title
    print(f"Training matrix: {X_train.shape[0]:,} rows, {X_train.shape[1]:,} columns")  # size
    print(f"Validation matrix: {X_val.shape[0]:,} rows, {X_val.shape[1]:,} columns")  # size

    print_header("TRAIN XGBOOST - MY FULL-SCALE VAEP BASELINE")  # title
    n_positive = int(y_train.sum())  # positive training rows
    n_negative = int(len(y_train) - n_positive)  # negative training rows
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0  # class-imbalance weight
    print(f"Training positives: {n_positive:,} | negatives: {n_negative:,} | scale_pos_weight: {scale_pos_weight:.4f}")  # report
    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, scale_pos_weight=scale_pos_weight,  # same settings as before
                          eval_metric="logloss", random_state=RANDOM_STATE, tree_method="hist")  # histogram trees
    t0 = time.time()  # timer
    model.fit(X_train, y_train)  # train
    print(f"Model fitting complete in {(time.time() - t0) / 60:.1f} minutes.")  # time

    print_header("EVALUATE ON MY VALIDATION MATCHES")  # title
    y_pred_proba = model.predict_proba(X_val)[:, 1]  # probability of the positive class
    auc_roc = roc_auc_score(y_val, y_pred_proba)  # ranking quality
    pr_auc = average_precision_score(y_val, y_pred_proba)  # precision-recall area
    brier = brier_score_loss(y_val, y_pred_proba)  # calibration error
    print(f"Validation rows: {len(y_val):,} | positive rate: {y_val.mean() * 100:.4f}%")  # size
    print(f"MY FULL-SCALE VAEP VALIDATION AUC-ROC = {auc_roc:.4f} | PR-AUC {pr_auc:.4f} | Brier {brier:.4f}")  # headline numbers

    predictions_out = val_keys  # keys of the validation rows
    predictions_out["true_target"] = y_val.values  # label
    predictions_out["predicted_probability"] = y_pred_proba  # prediction
    predictions_out.to_csv(VALIDATION_PREDICTIONS_FILE, index=False)  # save
    print(f"Validation predictions saved to:\n{VALIDATION_PREDICTIONS_FILE}")  # where

    importance = pd.DataFrame({"feature": feature_columns, "importance": model.feature_importances_}).sort_values("importance", ascending=False)  # importance table
    importance.to_csv(FEATURE_IMPORTANCE_FILE, index=False)  # save
    print(f"Feature importance saved to:\n{FEATURE_IMPORTANCE_FILE}\nTop 10 most important features:\n{importance.head(10).to_string(index=False)}")  # show
    model.save_model(str(MODEL_SAVE_FILE))  # save the model
    print(f"Model saved to:\n{MODEL_SAVE_FILE}")  # where

    pd.DataFrame([  # metric/value summary read back by Step 18
        {"metric": "train_rows", "value": len(X_train)}, {"metric": "train_positive", "value": int(y_train.sum())},  # training size
        {"metric": "validation_rows", "value": len(X_val)}, {"metric": "validation_positive", "value": int(y_val.sum())},  # validation size
        {"metric": "n_feature_columns", "value": len(feature_columns)}, {"metric": "scale_pos_weight", "value": round(scale_pos_weight, 4)},  # model setup
        {"metric": "validation_auc_roc", "value": round(auc_roc, 4)}, {"metric": "validation_pr_auc", "value": round(pr_auc, 4)},  # results
        {"metric": "validation_brier_score", "value": round(brier, 4)},  # calibration
        {"metric": "n_train_matches", "value": len(train_ids)}, {"metric": "n_validation_matches", "value": len(validation_ids)},  # match counts
        {"metric": "n_train_matches_failed", "value": len(train_failed)}, {"metric": "n_validation_matches_failed", "value": len(val_failed)},  # failures
        {"metric": "random_state", "value": RANDOM_STATE},  # seed
    ]).to_csv(RESULTS_SUMMARY_FILE, index=False)  # save
    print(f"Results summary saved to:\n{RESULTS_SUMMARY_FILE}")  # where
    print_header("STEP 13 COMPLETE")  # done
    return 0  # success


if __name__ == "__main__":  # only when run directly
    common.run_main(main, "STEP 13")  # shared error handling
