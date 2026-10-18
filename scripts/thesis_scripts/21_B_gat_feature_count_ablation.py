# ==============================================================================
# STEP 22 (file 21_B_gat_feature_count_ablation.py) - FEATURE-COUNT LADDER (8 DOWN TO 1)
# PLUS ONE NEW 8TH FEATURE, dist_to_actor
# ==============================================================================
# Same graphs, same k=5 edges, same training as Steps 14/19/20. Only the node-feature columns change.
# A config whose predictions already exist (this script's own file, or an exactly matching Step 20 file)
# is not retrained; its metrics are recomputed from the saved file.
# ==============================================================================

import gc  # to release memory between configs

import common_gat as cg  # torch first, then numpy/pandas
import numpy as np  # NaN for resumed rows
import pandas as pd  # result table
import common  # paths and helpers
from common import OUTPUT_DIR, print_header, require_file  # shared pieces

TRAIN_CHECKPOINT_FILE = OUTPUT_DIR / "14_gat_checkpoint_training.pt"  # Step 14 training graphs (read only)
VALIDATION_CHECKPOINT_FILE = OUTPUT_DIR / "14_gat_checkpoint_validation.pt"  # Step 14 validation graphs (read only)
ABLATION_SUMMARY_FILE = OUTPUT_DIR / "22_gat_feature_count_ablation_summary.csv"  # final table
NODE_FEATURE_NAMES = ["x_norm", "y_norm", "teammate", "keeper", "actor", "dist_to_goal", "angle_to_goal"]  # column order of x

# The drop-one-at-a-time ladder, in the order it is printed. 'reuse_file' names a Step 20 predictions file that
# holds exactly this column set; if it exists it is loaded instead of retraining.
FEATURE_COUNT_CONFIGS = [
    {"name": "count8_add_dist_to_actor", "n_features": 8, "column_indices": [0, 1, 2, 3, 4, 5, 6], "augment_with_dist_to_actor": True, "reuse_file": None},  # 7 + dist_to_actor
    {"name": "count7_full", "n_features": 7, "column_indices": [0, 1, 2, 3, 4, 5, 6], "augment_with_dist_to_actor": False,  # original 7
     "reuse_file": "20_gat_features_full_7_features_validation_predictions.csv"},
    {"name": "count6_drop_angle_to_goal", "n_features": 6, "column_indices": [0, 1, 2, 3, 4, 5], "augment_with_dist_to_actor": False, "reuse_file": None},  # no angle
    {"name": "count5_drop_goal_geometry", "n_features": 5, "column_indices": [0, 1, 2, 3, 4], "augment_with_dist_to_actor": False,  # no goal geometry
     "reuse_file": "20_gat_features_no_goal_geometry_validation_predictions.csv"},
    {"name": "count4_drop_keeper_too", "n_features": 4, "column_indices": [0, 1, 2, 4], "augment_with_dist_to_actor": False, "reuse_file": None},  # no keeper
    {"name": "count3_drop_teammate_too", "n_features": 3, "column_indices": [0, 1, 4], "augment_with_dist_to_actor": False,  # position + actor
     "reuse_file": "20_gat_features_position_actor_only_validation_predictions.csv"},
    {"name": "count2_drop_actor_too", "n_features": 2, "column_indices": [0, 1], "augment_with_dist_to_actor": False, "reuse_file": None},  # position only
    {"name": "count1_x_norm_only", "n_features": 1, "column_indices": [0], "augment_with_dist_to_actor": False, "reuse_file": None},  # x only
]
STEP14_ORIGINAL_FULL_FEATURE_AUC = 0.7178  # Step 14 result for the full 7 features
STEP14_MATCH_TOLERANCE = 0.02  # how close the count7_full run should land


def build_variant(base_graphs, config):
    """Slice the requested columns, then append dist_to_actor when the config asks for it."""
    subset = cg.select_features(base_graphs, config["column_indices"])  # keep the chosen columns
    return cg.add_dist_to_actor_feature(subset) if config["augment_with_dist_to_actor"] else subset  # optional 8th feature


def describe_columns(config):
    """Human-readable list of the kept columns."""
    names = [NODE_FEATURE_NAMES[i] for i in config["column_indices"]]  # names of the sliced columns
    return names + ["dist_to_actor (NEW)"] if config["augment_with_dist_to_actor"] else names  # mark the new one


def check_full_feature_self_consistency(auc_roc):
    """Compare count7_full with Step 14's original number."""
    diff = abs(auc_roc - STEP14_ORIGINAL_FULL_FEATURE_AUC)  # distance
    print(f"[SELF-CHECK] Step 14 AUC-ROC {STEP14_ORIGINAL_FULL_FEATURE_AUC:.4f}; this run {auc_roc:.4f} (difference {diff:.4f}).")  # compare
    print("[WARNING] above tolerance; do not trust the ladder yet." if diff > STEP14_MATCH_TOLERANCE
          else "[PASS] close enough to Step 14; the ladder is a clean comparison.")  # verdict


def resolve_existing_predictions_file(config):
    """(file to load or None, file this script writes). Own earlier file first, then the Step 20 file."""
    own_file = OUTPUT_DIR / f"22_gat_feature_count_{config['name']}_validation_predictions.csv"  # this script's file
    if own_file.exists():  # an earlier run of this script finished the config
        return own_file, own_file  # load it
    if config["reuse_file"] is not None:  # a Step 20 file with the same columns may exist
        reuse_path = OUTPUT_DIR / config["reuse_file"]  # its path
        if reuse_path.exists():  # found
            return reuse_path, own_file  # load it, but still write to own_file if retraining were needed
    return None, own_file  # nothing to reuse: train


def main():  # runs the whole step from start to finish
    common.verify_environment()  # optional interpreter check
    print_header("STEP 22 - GAT FEATURE-COUNT LADDER + NEW 8TH FEATURE")  # title
    require_file(TRAIN_CHECKPOINT_FILE, "Step 14 training graphs checkpoint")  # input 1
    require_file(VALIDATION_CHECKPOINT_FILE, "Step 14 validation graphs checkpoint")  # input 2
    base_train, train_keys = cg.load_graph_checkpoint(TRAIN_CHECKPOINT_FILE, "TRAINING")  # training graphs
    base_val, val_keys = cg.load_graph_checkpoint(VALIDATION_CHECKPOINT_FILE, "VALIDATION")  # validation graphs
    if (base_train[0].x.shape[1] if base_train else 0) != len(NODE_FEATURE_NAMES):  # layout changed
        raise ValueError("Step 14's checkpoint no longer has the fixed 7-column layout.")  # stop
    pos_weight = cg.compute_pos_weight(train_keys)  # same in every run
    results = []  # one row per config

    for config in FEATURE_COUNT_CONFIGS:  # one run per config
        name, n_features = config["name"], config["n_features"]  # shortcuts
        print_header(f"ABLATION RUN - '{name}' ({n_features} features)")  # title
        kept = describe_columns(config)  # column names
        print(f"Columns kept: {kept}")  # show
        existing_file, own_file = resolve_existing_predictions_file(config)  # reuse or train?

        if existing_file is not None:  # result already on disk
            print(f"[RESUMING FROM DISK] {existing_file}")  # tell the user
            auc, pr_auc, brier = cg.metrics_from_predictions(existing_file)  # recompute metrics from the saved file
            print(f"[RESULT - FROM SAVED PREDICTIONS] '{name}': validation AUC-ROC = {auc:.4f}")  # headline number
            if name == "count7_full":  # baseline row gets the self-check
                check_full_feature_self_consistency(auc)  # compare
            results.append({"feature_set": name, "n_features": n_features, "features_kept": ", ".join(kept),  # row for the table
                            "n_train_graphs": len(base_train), "n_validation_graphs": len(base_val),
                            "validation_auc_roc": round(auc, 4), "validation_pr_auc": round(pr_auc, 4),
                            "validation_brier_score": round(brier, 4), "training_minutes": np.nan, "source": str(existing_file)})
            continue  # next config

        train_v = build_variant(base_train, config)  # training graphs for this config
        val_v = build_variant(base_val, config)  # validation graphs for this config
        model, seconds = cg.train_gat(train_v, val_v, pos_weight, in_channels=n_features, name=name)  # train
        _, y_pred, auc, pr_auc, brier = cg.evaluate(model, val_v)  # evaluate
        print(f"\n[RESULT] '{name}': validation AUC-ROC = {auc:.4f}")  # headline number
        if name == "count7_full":  # baseline row gets the self-check
            check_full_feature_self_consistency(auc)  # compare
        cg.save_predictions(val_keys, y_pred, own_file)  # predictions for later DeLong tests
        results.append({"feature_set": name, "n_features": n_features, "features_kept": ", ".join(kept),  # row for the table
                        "n_train_graphs": len(train_v), "n_validation_graphs": len(val_v),
                        "validation_auc_roc": round(auc, 4), "validation_pr_auc": round(pr_auc, 4),
                        "validation_brier_score": round(brier, 4), "training_minutes": round(seconds / 60, 2), "source": "freshly_trained"})
        del train_v, val_v, model  # free memory
        gc.collect()  # make sure it is released

    results_df = pd.DataFrame(results)  # final table
    print_header("STEP 22 FINAL COMPARISON - FEATURE-COUNT LADDER (8 DOWN TO 1)")  # title
    print(results_df.sort_values("n_features", ascending=False).to_string(index=False))  # show, 8 features first
    new_row = results_df[results_df["feature_set"] == "count8_add_dist_to_actor"]  # the 8-feature row
    base_row = results_df[results_df["feature_set"] == "count7_full"]  # the 7-feature row
    if not new_row.empty and not base_row.empty:  # both present
        new_auc, base_auc = float(new_row.iloc[0]["validation_auc_roc"]), float(base_row.iloc[0]["validation_auc_roc"])  # the two AUCs
        print(f"\nNEW 8TH FEATURE CHECK: count8={new_auc:.4f} vs count7_full={base_auc:.4f} (difference {new_auc - base_auc:+.4f})")  # report
    best = results_df.loc[results_df["validation_auc_roc"].idxmax()]  # best row
    print(f"Highest validation AUC-ROC: '{best['feature_set']}' with {best['validation_auc_roc']:.4f}")  # report
    results_df.to_csv(ABLATION_SUMMARY_FILE, index=False)  # save
    print(f"Ablation summary saved to:\n{ABLATION_SUMMARY_FILE}")  # where
    print_header("STEP 22 COMPLETE")  # done
    return 0  # success


if __name__ == "__main__":  # only when run directly
    common.run_main(main, "STEP 22")  # shared error handling
