# ==============================================================================
# STEP 21 (file 22_label_horizon_ablation.py) - LABEL HORIZON ABLATION, N = 5..13 AND 15, VAEP
# ==============================================================================
# Only the label horizon N changes. Every match is converted to SPADL actions and VAEP features ONCE
# (cached to two pickle files); for each N the target is rebuilt from the cached action arrays, joined to the
# cached features, and the same XGBoost model as Step 13 is trained and evaluated.
# Changes compared with the original: the window labelling uses cumulative sums instead of nested loops, and the
# train/validation matrices are taken from the cached feature table with a row mask instead of two full merges.
# Rows, row order, hyper-parameters and seeds are unchanged.
# ==============================================================================

import gc  # release memory between horizons
import pickle  # resumable checkpoint of the one-time pass
import time  # progress timing
import numpy as np  # arrays
import pandas as pd  # tables
from xgboost import XGBClassifier  # the VAEP model
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss  # headline metrics

import common  # paths and helpers
from common import OUTPUT_DIR, print_header, require_file  # shared pieces
import common_spadl as cs  # loader, metadata, SPADL ids, window labels, VAEP features

FULL_SPLIT_FILE = OUTPUT_DIR / "full_scale_match_split.csv"  # Step 12 split (reused, never rebuilt here)
RAW_CACHE_CHECKPOINT_FILE = OUTPUT_DIR / "21_horizon_raw_cache_checkpoint.pkl"  # progress file of the one-time pass
ACTIONS_CACHE_FILE = OUTPUT_DIR / "21_horizon_actions_cache.pkl"  # finished cache 1: small action columns
FEATURES_CACHE_FILE = OUTPUT_DIR / "21_horizon_features_cache.pkl"  # finished cache 2: VAEP features of every action
ABLATION_SUMMARY_FILE = OUTPUT_DIR / "21_label_horizon_ablation_summary.csv"  # final table
HORIZON_VALUES_TO_TEST = [5, 6, 7, 8, 9, 10, 11, 12, 13, 15]  # horizons compared
RANDOM_STATE = 42  # seed
CHECKPOINT_EVERY_N_MATCHES = 20  # how often the one-time pass saves progress
STEP13_ORIGINAL_N10_AUC = 0.7731  # Step 13 result for N=10
STEP13_MATCH_TOLERANCE = 0.01  # how close the rebuilt N=10 run should land
ACTION_CACHE_COLUMNS = ["game_id", "action_index", "period_id", "time_seconds", "team_id", "type_id", "result_id"]  # columns kept per action


def load_split():
    """Match ids of the training and validation sets."""
    print_header("LOAD FULL-SCALE MATCH SPLIT")  # title
    split = pd.read_csv(FULL_SPLIT_FILE)  # split table
    split["game_id"] = pd.to_numeric(split["game_id"], errors="raise").astype(int)  # integer ids
    split["split"] = split["split"].astype(str).str.lower().str.strip()  # normalise labels
    print(split["split"].value_counts().sort_index())  # counts
    return split[split["split"] == "train"]["game_id"].tolist(), split[split["split"] == "validation"]["game_id"].tolist()  # two lists


def build_actions_and_features_for_match(loader, game_id, home_team_id):
    """One conversion, two products: the small action table and the full VAEP feature table."""
    actions = cs.load_actions(loader, game_id, home_team_id)  # events -> SPADL actions
    action_cache = actions[["action_index", "period_id", "time_seconds", "team_id", "type_id", "result_id"]].copy()  # label-independent columns
    action_cache["game_id"] = int(game_id)  # add the match id
    action_cache = action_cache[ACTION_CACHE_COLUMNS]  # fixed column order
    features = cs.compute_vaep_features(actions, home_team_id)  # features for every action
    features["game_id"] = int(game_id)  # key 1
    features["action_index"] = actions["action_index"].values  # key 2
    return action_cache, features  # both products


def load_raw_cache_checkpoint():
    """Resume state of the one-time pass, or empty state when there is none."""
    if RAW_CACHE_CHECKPOINT_FILE.exists():  # an earlier run was interrupted
        with open(RAW_CACHE_CHECKPOINT_FILE, "rb") as fh:  # open the pickle
            data = pickle.load(fh)  # read it
        return data["action_frames"], data["feature_frames"], set(data["processed_match_ids"]), list(data["failed_matches"])  # restored state
    return [], [], set(), []  # fresh start


def save_raw_cache_checkpoint(action_frames, feature_frames, processed_match_ids, failed_matches):
    """Write the resume state to disk."""
    with open(RAW_CACHE_CHECKPOINT_FILE, "wb") as fh:  # open for writing
        pickle.dump({"action_frames": action_frames, "feature_frames": feature_frames,  # frames so far
                     "processed_match_ids": list(processed_match_ids), "failed_matches": failed_matches}, fh)  # bookkeeping


def build_raw_cache(loader, metadata, all_match_ids):
    """The one-time expensive pass over every match."""
    print_header("ONE-TIME PASS - CONVERT EVERY MATCH TO ACTIONS + VAEP FEATURES")  # title
    action_frames, feature_frames, already_processed, failed_matches = load_raw_cache_checkpoint()  # resume if possible
    if already_processed:  # resuming
        print(f"[RESUMING FROM CHECKPOINT] {len(already_processed)} matches already processed.")  # tell the user
    home = dict(zip(metadata["game_id"], metadata["home_team_id"]))  # game_id -> home team id
    ids_sorted = sorted(set(int(m) for m in all_match_ids))  # every match once, in order
    remaining_ids = [m for m in ids_sorted if m not in already_processed]  # still to do
    print(f"Matches remaining to process: {len(remaining_ids)} of {len(ids_sorted)} total.")  # report
    start = time.time()  # timer
    for number, game_id in enumerate(remaining_ids, start=1):  # one match at a time
        try:  # a broken match must not stop the pass
            if game_id not in home:  # no metadata
                print(f"  [SKIP - no metadata] Match {game_id}")  # report
                failed_matches.append((int(game_id), "no metadata"))  # remember
                already_processed.add(game_id)  # do not retry
                continue
            action_cache, features = build_actions_and_features_for_match(loader, game_id, int(home[game_id]))  # convert
            action_frames.append(action_cache)  # keep
            feature_frames.append(features)  # keep
        except Exception as ex:  # report and move on
            print(f"  [FAILED - match skipped, run continues] Match {game_id}: {type(ex).__name__}: {ex}")  # report
            failed_matches.append((int(game_id), f"{type(ex).__name__}: {ex}"))  # remember
        already_processed.add(game_id)  # done (successfully or not)
        if number % CHECKPOINT_EVERY_N_MATCHES == 0 or number == len(remaining_ids):  # time to save
            elapsed = time.time() - start  # seconds so far
            rate = number / elapsed if elapsed > 0 else 0  # matches per second
            print(f"  [{number}/{len(remaining_ids)} this session] ({len(already_processed)}/{len(ids_sorted)} total) "
                  f"Elapsed {elapsed / 60:.1f} min. Estimated remaining {((len(remaining_ids) - number) / rate / 60) if rate > 0 else float('nan'):.1f} min.")  # show
            save_raw_cache_checkpoint(action_frames, feature_frames, already_processed, failed_matches)  # save progress
            print(f"  [CHECKPOINT SAVED] {RAW_CACHE_CHECKPOINT_FILE}")  # confirm
            gc.collect()  # release memory
    save_raw_cache_checkpoint(action_frames, feature_frames, already_processed, failed_matches)  # final save
    if failed_matches:  # failures
        print(f"Matches that failed across the whole pass: {len(failed_matches)} of {len(ids_sorted)}.")  # count
        for gid, reason in failed_matches:  # list them
            print(f"  {gid}: {reason}")  # show
    return action_frames, feature_frames  # lists of per-match frames


def get_or_build_combined_cache(loader, metadata, all_match_ids):
    """Load the two cache files if they exist, otherwise run the one-time pass and write them."""
    if ACTIONS_CACHE_FILE.exists() and FEATURES_CACHE_FILE.exists():  # cache already built
        print_header("REUSING ALREADY-BUILT CACHE - NO STATSBOMB PARSING NEEDED THIS RUN")  # title
        actions_df, features_df = pd.read_pickle(ACTIONS_CACHE_FILE), pd.read_pickle(FEATURES_CACHE_FILE)  # load both
        print(f"Cached action rows: {len(actions_df):,} | cached feature rows: {len(features_df):,}")  # sizes
        return actions_df, features_df  # done
    action_frames, feature_frames = build_raw_cache(loader, metadata, all_match_ids)  # expensive pass
    print_header("COMBINE ALL MATCHES' CACHED ACTIONS AND FEATURES")  # title
    actions_df = pd.concat(action_frames, ignore_index=True)  # one action table
    features_df = pd.concat(feature_frames, ignore_index=True)  # one feature table
    del action_frames, feature_frames  # free the per-match pieces
    gc.collect()  # release memory
    print(f"Combined action rows: {len(actions_df):,} | feature table: {features_df.shape[0]:,} x {features_df.shape[1]:,} "
          f"(~{features_df.memory_usage(deep=True).sum() / 1e9:.2f} GB)")  # sizes
    actions_df.to_pickle(ACTIONS_CACHE_FILE)  # save cache 1
    features_df.to_pickle(FEATURES_CACHE_FILE)  # save cache 2
    print(f"Caches saved to:\n{ACTIONS_CACHE_FILE}\n{FEATURES_CACHE_FILE}")  # where
    return actions_df, features_df  # both tables


def build_target_for_horizon(actions_df, N, shot_type_id, success_result_id, owngoal_result_id):
    """Eligible actions and their N-action label for every match, from the cached action columns only."""
    target_column = f"target_score_next{N}_literal"  # label column name
    frames = []  # one small table per match
    for game_id, group in actions_df.groupby("game_id", sort=False):  # match by match, in first-appearance order
        group = group.sort_values("action_index").reset_index(drop=True)  # chronological order inside the match
        eligible, future_team_goal, future_opp_own = cs.window_labels(  # label the next-N window
            group["type_id"].values, group["result_id"].values, group["team_id"].values,
            N, shot_type_id, success_result_id, owngoal_result_id)
        result_df = pd.DataFrame({"game_id": int(game_id), "action_index": group["action_index"].values,  # keys
                                  target_column: future_team_goal | future_opp_own})  # label
        frames.append(result_df[eligible].reset_index(drop=True))  # keep eligible rows only
    return pd.concat(frames, ignore_index=True), target_column  # whole-dataset target and its column name


def rows_in_target(keys_df, target_part, target_column):
    """Boolean mask of features_df rows that appear in target_part, plus their labels, in features_df order."""
    probe = target_part.assign(_hit=True)  # marker column for matched rows
    merged = keys_df.merge(probe, on=["game_id", "action_index"], how="left")  # left join keeps features_df order
    mask = merged["_hit"].notna().to_numpy()  # True where the row has a label
    labels = merged.loc[mask, target_column].astype(int).to_numpy()  # labels for those rows
    return mask, labels  # mask over features_df plus labels


def main():  # runs the whole step from start to finish
    common.verify_environment()  # optional interpreter check
    print_header("STEP 21 - LABEL HORIZON ABLATION (N=5..13, 15)")  # title
    require_file(FULL_SPLIT_FILE, "Full-scale match split")  # input
    train_ids, validation_ids = load_split()  # match ids
    train_id_set, validation_id_set = set(train_ids), set(validation_ids)  # fast membership tests
    loader = cs.create_loader()  # StatsBomb reader
    all_ids = train_ids + validation_ids  # every match
    metadata = cs.find_match_metadata(loader, all_ids)  # home team ids
    actions_df, features_df = get_or_build_combined_cache(loader, metadata, all_ids)  # one-time pass or cache
    shot_type_id, success_result_id, owngoal_result_id = cs.get_spadl_ids()  # SPADL ids
    keys_df = features_df[["game_id", "action_index"]]  # the two join columns
    feature_columns = [c for c in features_df.columns if c not in ("game_id", "action_index")]  # model inputs in original order
    results = []  # one row per horizon

    for N in HORIZON_VALUES_TO_TEST:  # one run per horizon
        print_header(f"ABLATION RUN - LABEL HORIZON N={N}")  # title
        target_df, target_column = build_target_for_horizon(actions_df, N, shot_type_id, success_result_id, owngoal_result_id)  # labels
        print(f"Eligible actions at N={N}: {len(target_df):,} | positive rate: {target_df[target_column].mean() * 100:.4f}%")  # report
        target_df.to_csv(OUTPUT_DIR / f"21_horizon_N{N}_target.csv", index=False)  # used by Step 23

        train_mask, y_train_arr = rows_in_target(keys_df, target_df[target_df["game_id"].isin(train_id_set)], target_column)  # training rows
        val_mask, y_val_arr = rows_in_target(keys_df, target_df[target_df["game_id"].isin(validation_id_set)], target_column)  # validation rows
        X_train = features_df.loc[train_mask, feature_columns]  # training matrix
        X_val = features_df.loc[val_mask, feature_columns]  # validation matrix
        y_train, y_val = pd.Series(y_train_arr), pd.Series(y_val_arr)  # labels as int series
        val_keys = keys_df.loc[val_mask].copy()  # keys for the predictions file
        print(f"[N={N}] Training rows after join: {len(X_train):,} | Validation rows after join: {len(X_val):,}")  # sizes

        print_header(f"TRAIN XGBOOST - HORIZON N={N}")  # title
        n_positive = int(y_train.sum())  # positives
        n_negative = int(len(y_train) - n_positive)  # negatives
        scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0  # recomputed for every N
        print(f"Training positives: {n_positive:,} | negatives: {n_negative:,} | scale_pos_weight: {scale_pos_weight:.4f}")  # report
        model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, scale_pos_weight=scale_pos_weight,  # same as Step 13
                              eval_metric="logloss", random_state=RANDOM_STATE, tree_method="hist")  # histogram trees
        t0 = time.time()  # timer
        model.fit(X_train, y_train)  # train
        fit_elapsed = time.time() - t0  # seconds
        print(f"[N={N}] Model fitting complete in {fit_elapsed / 60:.1f} minutes.")  # time

        y_pred_proba = model.predict_proba(X_val)[:, 1]  # validation probabilities
        auc_roc = roc_auc_score(y_val, y_pred_proba)  # ranking quality
        pr_auc = average_precision_score(y_val, y_pred_proba)  # precision-recall area
        brier = brier_score_loss(y_val, y_pred_proba)  # calibration error
        print(f"[N={N}] Validation rows {len(y_val):,} | positive rate {y_val.mean() * 100:.4f}% | AUC-ROC {auc_roc:.4f} | PR-AUC {pr_auc:.4f} | Brier {brier:.4f}")  # report

        if N == 10:  # self-check against Step 13
            diff = abs(auc_roc - STEP13_ORIGINAL_N10_AUC)  # distance
            print(f"[SELF-CHECK] Step 13 N=10 AUC-ROC {STEP13_ORIGINAL_N10_AUC:.4f}; this run {auc_roc:.4f} (difference {diff:.4f}).")  # compare
            print("[WARNING] above tolerance; do not trust the other N values yet." if diff > STEP13_MATCH_TOLERANCE
                  else "[PASS] rebuilt N=10 matches Step 13; the sweep is trustworthy.")  # verdict

        predictions_out = val_keys  # keys of the validation rows
        predictions_out["true_target"] = y_val.values  # label
        predictions_out["predicted_probability"] = y_pred_proba  # prediction
        predictions_out.to_csv(OUTPUT_DIR / f"21_horizon_N{N}_validation_predictions.csv", index=False)  # save
        print(f"[N={N}] Validation predictions saved.")  # confirm

        results.append({"N": N, "n_train_rows": len(X_train), "n_train_positive": int(y_train.sum()),  # row for the table
                        "n_validation_rows": len(X_val), "n_validation_positive": int(y_val.sum()),
                        "validation_positive_rate_pct": round(y_val.mean() * 100, 4), "scale_pos_weight": round(scale_pos_weight, 4),
                        "validation_auc_roc": round(auc_roc, 4), "validation_pr_auc": round(pr_auc, 4),
                        "validation_brier_score": round(brier, 4), "training_minutes": round(fit_elapsed / 60, 2)})
        del X_train, X_val, model, target_df  # free memory before the next horizon
        gc.collect()  # make sure it is released

    results_df = pd.DataFrame(results)  # final table
    print_header("STEP 21 FINAL COMPARISON - VAEP AUC-ROC BY LABEL HORIZON N")  # title
    print(results_df.to_string(index=False))  # show
    best = results_df.loc[results_df["validation_auc_roc"].idxmax()]  # best horizon
    print(f"\nHighest validation AUC-ROC: N={int(best['N'])} with {best['validation_auc_roc']:.4f}")  # report
    results_df.to_csv(ABLATION_SUMMARY_FILE, index=False)  # save
    print(f"Ablation summary saved to:\n{ABLATION_SUMMARY_FILE}")  # where
    print_header("STEP 21 COMPLETE")  # done
    return 0  # success


if __name__ == "__main__":  # only when run directly
    common.run_main(main, "STEP 21")  # shared error handling
