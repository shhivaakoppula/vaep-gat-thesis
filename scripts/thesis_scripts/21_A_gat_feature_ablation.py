# ==============================================================================
# STEP 20 (file 21_A_gat_feature_ablation.py) - GAT NODE-FEATURE ABLATION, FULL SCALE
# ==============================================================================
# k stays at 5 and the edges are reused untouched from Step 14's checkpoint. Only which columns of the
# 7-column node-feature matrix the GAT sees changes. A feature set whose predictions file already
# exists on disk is not retrained; its metrics are recomputed from that file (resume path).
# ==============================================================================

import gc  # to release memory between feature sets

import common_gat as cg  # torch first, then numpy/pandas
import numpy as np  # NaN for the resumed rows
import pandas as pd  # result table
import common  # paths and helpers
from common import OUTPUT_DIR, print_header, require_file  # shared pieces

TRAIN_CHECKPOINT_FILE = OUTPUT_DIR / "14_gat_checkpoint_training.pt"  # Step 14 training graphs (read only)
VALIDATION_CHECKPOINT_FILE = OUTPUT_DIR / "14_gat_checkpoint_validation.pt"  # Step 14 validation graphs (read only)
ABLATION_SUMMARY_FILE = OUTPUT_DIR / "20_gat_feature_ablation_summary.csv"  # final table
NODE_FEATURE_NAMES = ["x_norm", "y_norm", "teammate", "keeper", "actor", "dist_to_goal", "angle_to_goal"]  # column order of x
FEATURE_SETS_TO_TEST = {  # name -> column indices to keep
    "full_7_features": [0, 1, 2, 3, 4, 5, 6],  # baseline
    "no_goal_geometry": [0, 1, 2, 3, 4],  # drop dist_to_goal, angle_to_goal
    "no_team_role_flags": [0, 1, 4, 5, 6],  # drop teammate, keeper
    "no_actor_flag": [0, 1, 2, 3, 5, 6],  # drop actor
    "position_actor_only": [0, 1, 4],  # smallest useful graph
}
STEP14_ORIGINAL_FULL_FEATURE_AUC = 0.7178  # Step 14 result for the full 7 features
STEP14_MATCH_TOLERANCE = 0.02  # how close the full-feature run should land


def check_full_feature_self_consistency(auc_roc):
    """Compare the full-feature run with Step 14's original number."""
    diff = abs(auc_roc - STEP14_ORIGINAL_FULL_FEATURE_AUC)  # distance
    print(f"[SELF-CHECK] Step 14 full-feature AUC-ROC {STEP14_ORIGINAL_FULL_FEATURE_AUC:.4f}; this run {auc_roc:.4f} (difference {diff:.4f}).")  # compare
    print("[WARNING] above tolerance; do not trust the other rows yet." if diff > STEP14_MATCH_TOLERANCE
          else "[PASS] close enough to Step 14; the other rows are clean comparisons.")  # verdict


def main():  # runs the whole step from start to finish
    common.verify_environment()  # optional interpreter check
    print_header("STEP 20 - GAT NODE-FEATURE ABLATION")  # title
    require_file(TRAIN_CHECKPOINT_FILE, "Step 14 training graphs checkpoint")  # input 1
    require_file(VALIDATION_CHECKPOINT_FILE, "Step 14 validation graphs checkpoint")  # input 2
    base_train, train_keys = cg.load_graph_checkpoint(TRAIN_CHECKPOINT_FILE, "TRAINING")  # training graphs
    base_val, val_keys = cg.load_graph_checkpoint(VALIDATION_CHECKPOINT_FILE, "VALIDATION")  # validation graphs
    n_columns = base_train[0].x.shape[1] if base_train else 0  # width of the node-feature matrix
    if n_columns != len(NODE_FEATURE_NAMES):  # layout changed
        raise ValueError(f"Expected {len(NODE_FEATURE_NAMES)} node-feature columns, found {n_columns}.")  # stop
    pos_weight = cg.compute_pos_weight(train_keys)  # same in every run
    results = []  # one row per feature set

    for name, cols in FEATURE_SETS_TO_TEST.items():  # one run per feature set
        print_header(f"ABLATION RUN - FEATURE SET '{name}'")  # title
        kept = [NODE_FEATURE_NAMES[i] for i in cols]  # names of the kept columns
        print(f"Columns kept ({len(cols)}): {kept}")  # show
        pred_file = OUTPUT_DIR / f"20_gat_features_{name}_validation_predictions.csv"  # predictions of this run

        if pred_file.exists():  # a previous run already finished this set
            print(f"[RESUMING FROM DISK] using {pred_file.name}")  # tell the user
            auc, pr_auc, brier = cg.metrics_from_predictions(pred_file)  # recompute metrics from the saved file
            print(f"[RESULT - FROM SAVED PREDICTIONS] '{name}': validation AUC-ROC = {auc:.4f}")  # headline number
            if name == "full_7_features":  # self-check applies to the baseline row
                check_full_feature_self_consistency(auc)  # compare
            results.append({"feature_set": name, "n_features": len(cols), "features_kept": ", ".join(kept),  # row for the table
                            "n_train_graphs": len(base_train), "n_validation_graphs": len(base_val),
                            "validation_auc_roc": round(auc, 4), "validation_pr_auc": round(pr_auc, 4),
                            "validation_brier_score": round(brier, 4), "training_minutes": np.nan, "resumed_from_disk": True})
            continue  # next feature set

        train_sub = cg.select_features(base_train, cols)  # slice the columns, training
        val_sub = cg.select_features(base_val, cols)  # slice the columns, validation
        model, seconds = cg.train_gat(train_sub, val_sub, pos_weight, in_channels=len(cols), name=name)  # train
        _, y_pred, auc, pr_auc, brier = cg.evaluate(model, val_sub)  # evaluate
        print(f"\n[RESULT] '{name}': validation AUC-ROC = {auc:.4f}")  # headline number
        if name == "full_7_features":  # self-check applies to the baseline row
            check_full_feature_self_consistency(auc)  # compare
        cg.save_predictions(val_keys, y_pred, pred_file)  # keep predictions for later DeLong tests
        results.append({"feature_set": name, "n_features": len(cols), "features_kept": ", ".join(kept),  # row for the table
                        "n_train_graphs": len(train_sub), "n_validation_graphs": len(val_sub),
                        "validation_auc_roc": round(auc, 4), "validation_pr_auc": round(pr_auc, 4),
                        "validation_brier_score": round(brier, 4), "training_minutes": round(seconds / 60, 2), "resumed_from_disk": False})
        del train_sub, val_sub, model  # free memory
        gc.collect()  # make sure it is released

    results_df = pd.DataFrame(results)  # final table
    print_header("STEP 20 FINAL COMPARISON - GAT AUC-ROC BY FEATURE SET")  # title
    print(results_df.to_string(index=False))  # show
    best = results_df.loc[results_df["validation_auc_roc"].idxmax()]  # best feature set
    print(f"\nHighest validation AUC-ROC: '{best['feature_set']}' with {best['validation_auc_roc']:.4f}")  # report
    results_df.to_csv(ABLATION_SUMMARY_FILE, index=False)  # save
    print(f"Ablation summary saved to:\n{ABLATION_SUMMARY_FILE}")  # where
    print_header("STEP 20 COMPLETE")  # done
    return 0  # success


if __name__ == "__main__":  # only when run directly
    common.run_main(main, "STEP 20")  # shared error handling
