# ==============================================================================
# STEP 19 (file 21_C_gat_k_ablation.py) - GAT k-NEAREST-NEIGHBOURS ABLATION, FULL SCALE
# ==============================================================================
# Only k changes (3, 5, 8). Step 14's already-built graphs are loaded, their edges are thrown away
# and rebuilt at each k from the stored player positions. Everything else (7 node features, GAT
# architecture, 30 epochs, Adam 0.001, pos_weight, seed 42) is identical to Step 14.
# k=5 acts as a self-check against Step 14's original 0.7178.
# ==============================================================================

import gc  # to release memory between k values

import common_gat as cg  # imports torch first (Windows DLL order), then numpy/pandas
import pandas as pd  # result table
import common  # paths and helpers
from common import OUTPUT_DIR, print_header, require_file  # shared pieces

TRAIN_CHECKPOINT_FILE = OUTPUT_DIR / "14_gat_checkpoint_training.pt"  # Step 14 training graphs (read only)
VALIDATION_CHECKPOINT_FILE = OUTPUT_DIR / "14_gat_checkpoint_validation.pt"  # Step 14 validation graphs (read only)
ABLATION_SUMMARY_FILE = OUTPUT_DIR / "19_gat_k_ablation_summary.csv"  # final table
K_VALUES_TO_TEST = [3, 5, 8]  # the values of k compared
STEP14_ORIGINAL_K5_AUC = 0.7178  # what Step 14 reported for k=5
STEP14_MATCH_TOLERANCE = 0.02  # how close the rebuilt k=5 should land


def main():  # runs the whole step from start to finish
    common.verify_environment()  # optional interpreter check
    print_header("STEP 19 - GAT k-NEAREST-NEIGHBOURS ABLATION")  # title
    require_file(TRAIN_CHECKPOINT_FILE, "Step 14 training graphs checkpoint")  # input 1
    require_file(VALIDATION_CHECKPOINT_FILE, "Step 14 validation graphs checkpoint")  # input 2
    base_train, train_keys = cg.load_graph_checkpoint(TRAIN_CHECKPOINT_FILE, "TRAINING")  # training graphs and labels
    base_val, val_keys = cg.load_graph_checkpoint(VALIDATION_CHECKPOINT_FILE, "VALIDATION")  # validation graphs and labels
    pos_weight = cg.compute_pos_weight(train_keys)  # same at every k, depends only on label counts
    results = []  # one row per k

    for k in K_VALUES_TO_TEST:  # one full run per k
        print_header(f"ABLATION RUN - k = {k}")  # title
        train_k = cg.rebuild_graphs_at_k(base_train, k, "training")  # new edges for training graphs
        val_k = cg.rebuild_graphs_at_k(base_val, k, "validation")  # new edges for validation graphs
        print(f"pos_weight applied: {pos_weight:.4f}")  # same at every k
        model, seconds = cg.train_gat(train_k, val_k, pos_weight, in_channels=7, name=f"k={k}")  # train
        _, y_pred, auc, pr_auc, brier = cg.evaluate(model, val_k)  # final evaluation
        print(f"\n[RESULT] k={k}: validation AUC-ROC = {auc:.4f}")  # headline number

        if k == 5:  # self-check against Step 14
            diff = abs(auc - STEP14_ORIGINAL_K5_AUC)  # distance from the original number
            print(f"[SELF-CHECK] Step 14 k=5 AUC-ROC was {STEP14_ORIGINAL_K5_AUC:.4f}; this run gives {auc:.4f} (difference {diff:.4f}).")  # compare
            print("[WARNING] difference above tolerance; do not trust the other k values yet." if diff > STEP14_MATCH_TOLERANCE
                  else "[PASS] close enough to Step 14's number; edge rebuilding matches.")  # verdict

        cg.save_predictions(val_keys, y_pred, OUTPUT_DIR / f"19_gat_k{k}_validation_predictions.csv")  # predictions for later DeLong tests
        results.append({"k": k, "n_train_graphs": len(train_k), "n_validation_graphs": len(val_k),  # row for the table
                        "validation_auc_roc": round(auc, 4), "validation_pr_auc": round(pr_auc, 4),
                        "validation_brier_score": round(brier, 4), "training_minutes": round(seconds / 60, 2)})
        del train_k, val_k, model  # free memory before the next k
        gc.collect()  # make sure it is released

    results_df = pd.DataFrame(results)  # final table
    print_header("STEP 19 FINAL COMPARISON - GAT AUC-ROC BY k")  # title
    print(results_df.to_string(index=False))  # show
    best = results_df.loc[results_df["validation_auc_roc"].idxmax()]  # best k
    print(f"\nHighest validation AUC-ROC: k={int(best['k'])} with {best['validation_auc_roc']:.4f}")  # report
    results_df.to_csv(ABLATION_SUMMARY_FILE, index=False)  # save
    print(f"Ablation summary saved to:\n{ABLATION_SUMMARY_FILE}")  # where
    print_header("STEP 19 COMPLETE")  # done
    return 0  # success


if __name__ == "__main__":  # only when run directly
    common.run_main(main, "STEP 19")  # shared error handling
