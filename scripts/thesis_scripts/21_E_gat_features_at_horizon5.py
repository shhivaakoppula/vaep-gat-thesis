# ==============================================================================
# STEP 23 (file 21_E_gat_features_at_horizon5.py) - GAT FEATURE-COUNT LADDER AT LABEL HORIZON N=5
# ==============================================================================
# Step 14's N=10 graphs are relabelled with the N=5 target from Step 21 (only y changes; x, edges and the
# actor mask are reused). The same 8 feature-count configs as Step 22 are then trained from scratch.
# Every row is a fresh run because the label differs; only this script's own earlier predictions are reused.
# ==============================================================================

import gc  # to release memory between configs

import common_gat as cg  # torch first, then numpy/pandas
import torch  # building the relabelled y tensors
import numpy as np  # NaN for resumed rows
import pandas as pd  # tables
from torch_geometric.data import Data  # graph container
import common  # paths and helpers
from common import OUTPUT_DIR, print_header, require_file  # shared pieces

TRAIN_CHECKPOINT_FILE = OUTPUT_DIR / "14_gat_checkpoint_training.pt"  # Step 14 training graphs (read only)
VALIDATION_CHECKPOINT_FILE = OUTPUT_DIR / "14_gat_checkpoint_validation.pt"  # Step 14 validation graphs (read only)
HORIZON5_TARGET_FILE = OUTPUT_DIR / "21_horizon_N5_target.csv"  # N=5 labels built by the label-horizon script
HORIZON5_TARGET_COLUMN = "target_score_next5_literal"  # label column inside that file
ABLATION_SUMMARY_FILE = OUTPUT_DIR / "23_gat_features_at_horizon5_summary.csv"  # final table
NODE_FEATURE_NAMES = ["x_norm", "y_norm", "teammate", "keeper", "actor", "dist_to_goal", "angle_to_goal"]  # column order of x
FEATURE_COUNT_CONFIGS = [  # same ladder as Step 22, names end in _at_N5
    {"name": "count8_add_dist_to_actor_at_N5", "n_features": 8, "column_indices": [0, 1, 2, 3, 4, 5, 6], "augment_with_dist_to_actor": True},
    {"name": "count7_full_at_N5", "n_features": 7, "column_indices": [0, 1, 2, 3, 4, 5, 6], "augment_with_dist_to_actor": False},
    {"name": "count6_drop_angle_to_goal_at_N5", "n_features": 6, "column_indices": [0, 1, 2, 3, 4, 5], "augment_with_dist_to_actor": False},
    {"name": "count5_drop_goal_geometry_at_N5", "n_features": 5, "column_indices": [0, 1, 2, 3, 4], "augment_with_dist_to_actor": False},
    {"name": "count4_drop_keeper_too_at_N5", "n_features": 4, "column_indices": [0, 1, 2, 4], "augment_with_dist_to_actor": False},
    {"name": "count3_drop_teammate_too_at_N5", "n_features": 3, "column_indices": [0, 1, 4], "augment_with_dist_to_actor": False},
    {"name": "count2_drop_actor_too_at_N5", "n_features": 2, "column_indices": [0, 1], "augment_with_dist_to_actor": False},
    {"name": "count1_x_norm_only_at_N5", "n_features": 1, "column_indices": [0], "augment_with_dist_to_actor": False},
]
VAEP_N5_VALIDATION_AUC = 0.8820  # context only
VAEP_N10_VALIDATION_AUC = 0.7731  # context only
GAT_N10_FULL_FEATURE_AUC = 0.7178  # context only


def load_horizon5_targets():
    """Read the N=5 target file and return game_id, action_index and the label."""
    print_header("LOAD STEP 21's N=5 TARGET FILE")  # title
    require_file(HORIZON5_TARGET_FILE, "Step 21 N=5 target file")  # must exist
    targets = pd.read_csv(HORIZON5_TARGET_FILE, usecols=["game_id", "action_index", HORIZON5_TARGET_COLUMN])  # only the 3 needed columns
    targets["game_id"] = targets["game_id"].astype(int)  # integer key
    targets["action_index"] = targets["action_index"].astype(int)  # integer key
    targets[HORIZON5_TARGET_COLUMN] = targets[HORIZON5_TARGET_COLUMN].astype(bool)  # boolean label
    print(f"N=5-eligible rows available for relabelling: {len(targets):,}")  # size
    return targets[["game_id", "action_index", HORIZON5_TARGET_COLUMN]]  # same column order as before


def relabel_graphs_with_horizon5(graphs, keys_df, horizon5_targets, label):
    """Replace every graph's y with its N=5 label. Unmatched graphs would be dropped (and reported)."""
    print_header(f"RELABEL {label} GRAPHS WITH N=5 TARGETS")  # title
    keys_df = keys_df.reset_index(drop=True).copy()  # clean index
    keys_df["game_id"] = keys_df["game_id"].astype(int)  # integer key
    keys_df["action_index"] = keys_df["action_index"].astype(int)  # integer key
    keys_df["_original_order"] = range(len(keys_df))  # remember where each graph sat in the list
    merged = keys_df.merge(horizon5_targets, on=["game_id", "action_index"], how="left")  # attach N=5 labels
    n_total, n_missing = len(merged), int(merged[HORIZON5_TARGET_COLUMN].isna().sum())  # match statistics
    print(f"{label}: {n_total - n_missing:,} of {n_total:,} graphs matched an N=5 target row ({n_missing:,} missing).")  # report
    if n_missing > 0:  # should be impossible: N=5 eligibility contains N=10 eligibility
        print(f"[WARNING] {n_missing:,} graphs could not be matched and are dropped; investigate before trusting results.")  # warn
    else:  # all matched
        print("[PASS] Every graph found a matching N=5 target row.")  # reassurance

    matched = merged[merged[HORIZON5_TARGET_COLUMN].notna()].reset_index(drop=True)  # keep matched rows only
    order = matched["_original_order"].to_numpy()  # position of each matched graph in the old list
    labels = matched[HORIZON5_TARGET_COLUMN].astype(bool).to_numpy()  # new labels as booleans (replaces the slow row-by-row loop)
    new_graphs = [Data(x=graphs[int(o)].x, edge_index=graphs[int(o)].edge_index, edge_attr=graphs[int(o)].edge_attr,  # reuse tensors
                       y=torch.tensor([float(bool(l))], dtype=torch.float), actor_mask=graphs[int(o)].actor_mask)  # only y is new
                  for o, l in zip(order, labels)]  # one new graph per matched row
    new_keys_df = matched[["game_id", "action_index"]].copy()  # keys for the predictions files
    new_keys_df["true_target"] = matched[HORIZON5_TARGET_COLUMN].astype(int)  # integer label
    print(f"{label} positive rate AT N=5: {new_keys_df['true_target'].mean() * 100:.4f}% "
          f"({int(new_keys_df['true_target'].sum()):,} of {len(new_keys_df):,})")  # report
    return new_graphs, new_keys_df  # relabelled graphs and keys


def build_variant(base_graphs, config):
    """Slice the requested columns, then append dist_to_actor when the config asks for it."""
    subset = cg.select_features(base_graphs, config["column_indices"])  # keep the chosen columns
    return cg.add_dist_to_actor_feature(subset) if config["augment_with_dist_to_actor"] else subset  # optional 8th feature


def describe_columns(config):
    """Human-readable list of the kept columns."""
    names = [NODE_FEATURE_NAMES[i] for i in config["column_indices"]]  # sliced names
    return names + ["dist_to_actor (NEW)"] if config["augment_with_dist_to_actor"] else names  # mark the new one


def main():  # runs the whole step from start to finish
    common.verify_environment()  # optional interpreter check
    print_header("STEP 23 - GAT FEATURE-COUNT LADDER AT LABEL HORIZON N=5")  # title
    require_file(TRAIN_CHECKPOINT_FILE, "Step 14 training graphs checkpoint")  # input 1
    require_file(VALIDATION_CHECKPOINT_FILE, "Step 14 validation graphs checkpoint")  # input 2
    train_graphs_n10, train_keys_n10 = cg.load_graph_checkpoint(TRAIN_CHECKPOINT_FILE, "TRAINING")  # still N=10 labelled
    val_graphs_n10, val_keys_n10 = cg.load_graph_checkpoint(VALIDATION_CHECKPOINT_FILE, "VALIDATION")  # still N=10 labelled
    if (train_graphs_n10[0].x.shape[1] if train_graphs_n10 else 0) != len(NODE_FEATURE_NAMES):  # layout changed
        raise ValueError("Step 14's checkpoint no longer has the fixed 7-column layout.")  # stop
    horizon5_targets = load_horizon5_targets()  # N=5 labels
    base_train, train_keys = relabel_graphs_with_horizon5(train_graphs_n10, train_keys_n10, horizon5_targets, "TRAINING")  # relabel training
    base_val, val_keys = relabel_graphs_with_horizon5(val_graphs_n10, val_keys_n10, horizon5_targets, "VALIDATION")  # relabel validation
    del train_graphs_n10, val_graphs_n10, train_keys_n10, val_keys_n10, horizon5_targets  # the N=10 versions are no longer needed
    gc.collect()  # release memory
    pos_weight = cg.compute_pos_weight(train_keys)  # recomputed for the N=5 label
    print(f"pos_weight (N=5 label): {pos_weight:.4f}")  # report
    results = []  # one row per config

    for config in FEATURE_COUNT_CONFIGS:  # one run per config
        name, n_features = config["name"], config["n_features"]  # shortcuts
        print_header(f"ABLATION RUN - '{name}' ({n_features} features)")  # title
        kept = describe_columns(config)  # column names
        print(f"Columns kept: {kept}")  # show
        pred_file = OUTPUT_DIR / f"23_gat_features_at_horizon5_{name}_validation_predictions.csv"  # this config's predictions

        if pred_file.exists():  # an earlier run of this script finished it
            print(f"[RESUMING FROM DISK] {pred_file}")  # tell the user
            auc, pr_auc, brier = cg.metrics_from_predictions(pred_file)  # recompute metrics from the file
            print(f"[RESULT - FROM SAVED PREDICTIONS] '{name}': validation AUC-ROC = {auc:.4f}")  # headline number
            results.append({"feature_set": name, "n_features": n_features, "features_kept": ", ".join(kept),  # row for the table
                            "n_train_graphs": len(base_train), "n_validation_graphs": len(base_val),
                            "validation_auc_roc": round(auc, 4), "validation_pr_auc": round(pr_auc, 4),
                            "validation_brier_score": round(brier, 4), "training_minutes": np.nan, "source": "resumed_from_disk"})
            continue  # next config

        train_v, val_v = build_variant(base_train, config), build_variant(base_val, config)  # graphs for this config
        model, seconds = cg.train_gat(train_v, val_v, pos_weight, in_channels=n_features, name=name)  # train
        _, y_pred, auc, pr_auc, brier = cg.evaluate(model, val_v)  # evaluate
        print(f"\n[RESULT] '{name}': validation AUC-ROC = {auc:.4f}")  # headline number
        cg.save_predictions(val_keys, y_pred, pred_file)  # predictions for later DeLong tests
        results.append({"feature_set": name, "n_features": n_features, "features_kept": ", ".join(kept),  # row for the table
                        "n_train_graphs": len(train_v), "n_validation_graphs": len(val_v),
                        "validation_auc_roc": round(auc, 4), "validation_pr_auc": round(pr_auc, 4),
                        "validation_brier_score": round(brier, 4), "training_minutes": round(seconds / 60, 2), "source": "freshly_trained"})
        del train_v, val_v, model  # free memory
        gc.collect()  # make sure it is released

    results_df = pd.DataFrame(results)  # final table
    print_header("STEP 23 FINAL COMPARISON - GAT AUC-ROC AT N=5, BY FEATURE COUNT")  # title
    print(results_df.sort_values("n_features", ascending=False).to_string(index=False))  # show
    full_row = results_df[results_df["feature_set"] == "count7_full_at_N5"]  # the 7-feature row
    if not full_row.empty:  # headline comparison
        g5 = float(full_row.iloc[0]["validation_auc_roc"])  # GAT at N=5
        print(f"\nGAT N=10 {GAT_N10_FULL_FEATURE_AUC:.4f} | GAT N=5 {g5:.4f} (change {g5 - GAT_N10_FULL_FEATURE_AUC:+.4f}) | "
              f"VAEP N=10 {VAEP_N10_VALIDATION_AUC:.4f} | VAEP N=5 {VAEP_N5_VALIDATION_AUC:.4f} | "
              f"gap N=5 {VAEP_N5_VALIDATION_AUC - g5:+.4f} vs gap N=10 {VAEP_N10_VALIDATION_AUC - GAT_N10_FULL_FEATURE_AUC:+.4f}")  # report
    best = results_df.loc[results_df["validation_auc_roc"].idxmax()]  # best row
    print(f"Highest validation AUC-ROC in this ladder: '{best['feature_set']}' with {best['validation_auc_roc']:.4f}")  # report
    results_df.to_csv(ABLATION_SUMMARY_FILE, index=False)  # save
    print(f"Ablation summary saved to:\n{ABLATION_SUMMARY_FILE}")  # where
    print_header("STEP 23 COMPLETE")  # done
    return 0  # success


if __name__ == "__main__":  # only when run directly
    common.run_main(main, "STEP 23")  # shared error handling
