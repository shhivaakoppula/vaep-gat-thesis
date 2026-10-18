# ==============================================================================
# STEP 19B (file 21_D__gat_k8_standalone.py) - k=8 ONLY, RESUME-SAFE
# ==============================================================================
# Trains the k=8 GAT with a checkpoint after every epoch, so an interrupted run continues where it
# stopped. k=3 and k=5 are NOT retrained; their saved prediction files are read to build the final
# three-row table. The model, the optimiser and the training steps are the same as Step 19.
# NOTE on the seed: this script, like the original, builds the model and optimiser FIRST and seeds
# torch only afterwards (and only on a fresh start). That order is kept on purpose, because it is
# what produced your saved k=8 numbers. Changing it would change the initial weights.
# ==============================================================================

import gc  # to release memory

import common_gat as cg  # torch first, then numpy/pandas
import torch  # checkpoint saving and loading
import pandas as pd  # result table
import common  # paths and helpers
from common import OUTPUT_DIR, print_header, require_file  # shared pieces

TRAIN_CHECKPOINT_FILE = OUTPUT_DIR / "14_gat_checkpoint_training.pt"  # Step 14 training graphs (read only)
VALIDATION_CHECKPOINT_FILE = OUTPUT_DIR / "14_gat_checkpoint_validation.pt"  # Step 14 validation graphs (read only)
K8_TRAINING_CHECKPOINT_FILE = OUTPUT_DIR / "19b_gat_k8_training_checkpoint.pt"  # written after every epoch
K8_PREDICTIONS_FILE = OUTPUT_DIR / "19_gat_k8_validation_predictions.csv"  # k=8 predictions
K3_PREDICTIONS_FILE = OUTPUT_DIR / "19_gat_k3_validation_predictions.csv"  # finished earlier, read only
K5_PREDICTIONS_FILE = OUTPUT_DIR / "19_gat_k5_validation_predictions.csv"  # finished earlier, read only
ABLATION_SUMMARY_FILE = OUTPUT_DIR / "19_gat_k_ablation_summary.csv"  # final table
K = 8  # the single k trained here
FULL_SCALE_VAEP_AUC = 0.7746  # VAEP headline from Step 13 / 18, for the closing comparison


def save_training_checkpoint(path, epoch, model, optimiser, pos_weight_value):
    """Write model weights, optimiser state and the epoch number to disk."""
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),  # weights
                "optimiser_state_dict": optimiser.state_dict(), "pos_weight_value": pos_weight_value}, path)  # optimiser and loss weight


def row_from_predictions(path, k, training_minutes=None):
    """One table row built from a saved predictions file (no retraining)."""
    require_file(path, f"k={k} saved validation predictions")  # must exist
    auc, pr_auc, brier = cg.metrics_from_predictions(path)  # recompute the three metrics
    n = len(pd.read_csv(path, usecols=["true_target"]))  # number of validation graphs
    return {"k": k, "n_validation_graphs": n, "validation_auc_roc": round(auc, 4), "validation_pr_auc": round(pr_auc, 4),
            "validation_brier_score": round(brier, 4), "training_minutes": training_minutes}  # same columns as before


def main():  # runs the whole step from start to finish
    common.verify_environment()  # optional interpreter check
    print_header("STEP 19B - GAT k-ABLATION - k=8 ONLY, RESUME-SAFE")  # title
    require_file(TRAIN_CHECKPOINT_FILE, "Step 14 training graphs checkpoint")  # input 1
    require_file(VALIDATION_CHECKPOINT_FILE, "Step 14 validation graphs checkpoint")  # input 2
    base_train, train_keys = cg.load_graph_checkpoint(TRAIN_CHECKPOINT_FILE, "TRAINING")  # training graphs
    base_val, val_keys = cg.load_graph_checkpoint(VALIDATION_CHECKPOINT_FILE, "VALIDATION")  # validation graphs
    train_k8 = cg.rebuild_graphs_at_k(base_train, K, "training")  # k=8 edges, training
    val_k8 = cg.rebuild_graphs_at_k(base_val, K, "validation")  # k=8 edges, validation

    pos_weight = cg.compute_pos_weight(train_keys)  # loss weight
    model = cg.GATBaseline()  # built BEFORE seeding, as in the original
    optimiser = torch.optim.Adam(model.parameters(), lr=cg.LEARNING_RATE)  # optimiser
    start_epoch = 1  # default: start from the beginning
    if K8_TRAINING_CHECKPOINT_FILE.exists():  # an earlier run was interrupted
        ckpt = torch.load(K8_TRAINING_CHECKPOINT_FILE, weights_only=False)  # load it
        model.load_state_dict(ckpt["model_state_dict"])  # restore weights
        optimiser.load_state_dict(ckpt["optimiser_state_dict"])  # restore optimiser state
        start_epoch = ckpt["epoch"] + 1  # continue with the next epoch
        print(f"[RESUME] continuing from epoch {start_epoch} of {cg.N_EPOCHS}")  # tell the user
    else:  # fresh start
        torch.manual_seed(cg.RANDOM_STATE)  # seed AFTER the model exists, exactly like the original
        print("No existing k=8 checkpoint found. Starting at epoch 1.")  # tell the user

    def after_epoch(epoch, model_, optimiser_):  # hook run at the end of every epoch
        save_training_checkpoint(K8_TRAINING_CHECKPOINT_FILE, epoch, model_, optimiser_, pos_weight)  # safety net

    model, seconds = cg.train_gat(train_k8, val_k8, pos_weight, name="k=8", start_epoch=start_epoch,  # train (or finish training)
                                  model=model, optimiser=optimiser, after_epoch=after_epoch)  # pass the prepared objects
    training_minutes = round(seconds / 60, 2)  # minutes spent in this run
    _, y_pred, auc, _, _ = cg.evaluate(model, val_k8)  # final evaluation
    print(f"\n[RESULT] k=8: validation AUC-ROC = {auc:.4f}")  # headline number
    cg.save_predictions(val_keys, y_pred, K8_PREDICTIONS_FILE)  # predictions
    del train_k8, val_k8, model  # free memory
    gc.collect()  # make sure it is released

    print_header("STEP 19 FINAL COMPARISON - GAT AUC-ROC BY k")  # title
    results_df = pd.DataFrame([row_from_predictions(K3_PREDICTIONS_FILE, 3), row_from_predictions(K5_PREDICTIONS_FILE, 5),  # k=3 and k=5 from disk
                               row_from_predictions(K8_PREDICTIONS_FILE, 8, training_minutes)])  # k=8 just trained
    print(results_df.to_string(index=False))  # show
    best = results_df.loc[results_df["validation_auc_roc"].idxmax()]  # best k
    print(f"\nHighest validation AUC-ROC: k={int(best['k'])} with {best['validation_auc_roc']:.4f} (VAEP full scale: {FULL_SCALE_VAEP_AUC:.4f})")  # report
    results_df.to_csv(ABLATION_SUMMARY_FILE, index=False)  # save
    print(f"Ablation summary saved to:\n{ABLATION_SUMMARY_FILE}")  # where
    print_header("STEP 19B COMPLETE")  # done
    return 0  # success


if __name__ == "__main__":  # only when run directly
    common.run_main(main, "STEP 19B")  # shared error handling
