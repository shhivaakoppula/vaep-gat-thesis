# ==============================================================================
# STEP 28 (file 17_parity_gat_plus_vaep_features.py) - PARITY RUN, CELL 4: GAT THAT ALSO RECEIVES VAEP'S FEATURES (A + B)
# ==============================================================================
# Purpose (thesis Section 3.5): give the GAT the action context that VAEP has. The graph, the two GATConv layers,
# the actor-node readout, the loss, the optimiser, the learning rate, the batch size, the number of epochs and the
# seed are EXACTLY those of Step 14. The only change is late fusion:
#   VAEP feature row (set A, standardised) -> Linear(F, 64) -> ReLU, joined to the 128-value actor vector,
#   then the classifier head (Linear(192, 32) - ReLU - Dropout(0.2) - Linear(32, 1)).
# Inputs (produced by earlier steps): output/14_gat_checkpoint_training.pt, 14_gat_checkpoint_validation.pt,
#   output/27_vaep_features_A_train_N{H}.pkl, ..._validation_N{H}.pkl, and for N = 5 output/21_horizon_N5_target.csv
# Usage:
#   python 17_parity_gat_plus_vaep_features.py                  (N=10, seed 42)
#   python 17_parity_gat_plus_vaep_features.py --horizon 5
#   python 17_parity_gat_plus_vaep_features.py --seed 43        (repeat run)
# Speed: the evaluation loop collects tensors and joins them once. The training loop must keep the same operation
# order to give the same weights, so it is unchanged in substance.
# ==============================================================================

import common_gat as cg  # imports torch FIRST (needed on this Windows setup), then numpy and pandas
import torch  # tensors
import torch.nn as nn  # layers and loss
import torch.nn.functional as F  # ELU
from torch_geometric.loader import DataLoader  # batches graphs
from torch_geometric.nn import GATConv  # attention layer

import argparse  # command-line options
import gc  # free memory
import time  # timing

import numpy as np  # arrays
import pandas as pd  # tables
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss  # metrics

import common  # shared helpers
from common import OUTPUT_DIR, print_header  # shared paths and header printer

TRAIN_CKPT = OUTPUT_DIR / "14_gat_checkpoint_training.pt"  # Step 14 training graphs
VAL_CKPT = OUTPUT_DIR / "14_gat_checkpoint_validation.pt"  # Step 14 validation graphs
N5_TARGET_FILE = OUTPUT_DIR / "21_horizon_N5_target.csv"  # N=5 labels
N5_COLUMN = "target_score_next5_literal"  # label column for N=5
N_EPOCHS = cg.N_EPOCHS  # 30, identical to Step 14
LEARNING_RATE = cg.LEARNING_RATE  # 0.001
TRAIN_BATCH_SIZE = cg.TRAIN_BATCH_SIZE  # 256
EVAL_BATCH_SIZE = cg.EVAL_BATCH_SIZE  # 512
VAEP_HIDDEN = 64  # width of the VAEP branch


class GATPlusVAEP(nn.Module):
    """Step 14's GAT plus a small MLP on the VAEP features, fused before the classifier."""

    def __init__(self, n_vaep):  # builds the layers (GAT part plus the VAEP feature branch)
        super().__init__()  # module set-up
        self.gat1 = GATConv(in_channels=7, out_channels=64, heads=4, edge_dim=1)  # same first layer as Step 14
        self.gat2 = GATConv(in_channels=64 * 4, out_channels=32, heads=4, edge_dim=1)  # same second layer
        self.vaep_mlp = nn.Sequential(nn.Linear(n_vaep, VAEP_HIDDEN), nn.ReLU())  # VAEP branch
        self.classifier = nn.Sequential(nn.Linear(32 * 4 + VAEP_HIDDEN, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))  # fused head (the layer creation order fixes the random init)

    def forward(self, x, edge_index, edge_attr, actor_mask, vaep):  # one forward pass: GAT on the graph, VAEP features joined at the actor node
        h = F.elu(self.gat1(x, edge_index, edge_attr))  # attention round 1
        h = F.elu(self.gat2(h, edge_index, edge_attr))  # attention round 2
        actor = h[actor_mask]  # ball-carrier embedding of every graph
        fused = torch.cat([actor, self.vaep_mlp(vaep)], dim=1)  # join graph and VAEP information
        return self.classifier(fused).squeeze(-1)  # one logit per graph


def prepare_matrix(pkl_path, y_col, mean=None, std=None):
    """Read one VAEP feature pickle and return a standardised float32 matrix (memory-light: columns are popped as copied)."""
    feats = pd.read_pickle(pkl_path)  # keys, label and A features
    feat_cols = [c for c in feats.columns if c not in ("game_id", "action_index", y_col)]  # feature columns only
    key_list = list(zip(feats["game_id"].astype(np.int64).tolist(), feats["action_index"].astype(np.int64).tolist()))  # (game, action) per row
    n_rows = len(feats)  # rows
    arr = np.empty((n_rows, len(feat_cols)), dtype=np.float32)  # destination matrix
    for j, c in enumerate(feat_cols):  # copy one column at a time
        arr[:, j] = feats.pop(c).to_numpy(dtype=np.float32)  # pop frees the source column straight away
    del feats
    gc.collect()  # release the DataFrame
    if mean is None:  # training matrix: compute mean and std
        mean = np.empty(len(feat_cols), dtype=np.float32)
        std = np.empty(len(feat_cols), dtype=np.float32)  # empty array for the per-column spread
        for j in range(len(feat_cols)):  # loop over the feature columns
            col = arr[:, j]  # the values of this column
            mean[j] = np.nanmean(col, dtype=np.float64)  # mean ignoring NaN
            sd = np.nanstd(col, dtype=np.float64)  # std ignoring NaN
            std[j] = sd if sd > 0 else 1.0  # avoid division by zero
    step = 50000  # rows per chunk
    for i in range(0, n_rows, step):  # standardise chunk by chunk
        blk = arr[i:i + step]  # a view, no copy
        blk -= mean
        blk /= std  # divide in place so the columns are scaled to unit spread
        blk[~np.isfinite(blk)] = 0.0  # NaN and inf become 0 (the mean after standardising)
    print(f"  {pkl_path.name}: {n_rows:,} rows x {len(feat_cols)} features, {arr.nbytes / 1e9:.2f} GB as float32")
    return arr, key_list, mean, std, feat_cols  # give back the matrix, keys, mean, spread and column names


def attach_features(graphs, keys, arr, key_list, relabel=None):
    """Attach the standardised VAEP row to every graph; optionally relabel (N=5)."""
    pos = {k: i for i, k in enumerate(key_list)}  # key -> row number (last duplicate wins)
    if len(pos) != len(key_list):
        print(f"  note: {len(key_list) - len(pos):,} duplicate feature keys (last row used)")  # warn about repeated keys (the last row is used)
    kept_graphs, kept_rows = [], []  # survivors
    for g, (gid, ai) in zip(graphs, zip(keys["game_id"].values, keys["action_index"].values)):  # graph with its key
        i = pos.get((int(gid), int(ai)))  # feature row
        if i is None:  # no features for this action
            continue
        if relabel is not None:  # N=5: replace the label
            y = relabel.get((int(gid), int(ai)))
            if y is None:  # no N=5 label
                continue
            g.y = torch.tensor([float(y)], dtype=torch.float)  # set the label of this graph
        g.vaep = torch.from_numpy(arr[i]).unsqueeze(0)  # (1, F) row stored on the graph
        kept_graphs.append(g)
        kept_rows.append((int(gid), int(ai), int(g.y.item())))  # remember the keys of the graphs that were kept
    kept_keys = pd.DataFrame(kept_rows, columns=["game_id", "action_index", "true_target"])  # keys of the survivors
    print(f"  graphs matched to a VAEP feature row: {len(kept_graphs):,} of {len(graphs):,}")
    return kept_graphs, kept_keys  # give back the kept graphs and keys


def evaluate(model, graphs):
    """(y_true, y_pred, auc, pr_auc, brier) with the VAEP branch."""
    model.eval()  # dropout off
    loader = DataLoader(graphs, batch_size=EVAL_BATCH_SIZE, shuffle=False)  # fixed order
    probs, ys = [], []  # tensors per batch
    with torch.no_grad():  # no gradients
        for b in loader:
            logits = model(b.x, b.edge_index, b.edge_attr, b.actor_mask, b.vaep)  # scores
            probs.append(torch.sigmoid(logits))  # probabilities
            ys.append(b.y)  # labels
    p = torch.cat(probs).double().numpy()  # float32 to float64 is exact (same as .tolist())
    y = torch.cat(ys).double().numpy()  # labels as float64
    return y, p, roc_auc_score(y, p), average_precision_score(y, p), brier_score_loss(y, p)


def main():  # runs the whole step from start to finish
    common.verify_environment()  # optional interpreter check
    ap = argparse.ArgumentParser()  # options
    ap.add_argument("--horizon", type=int, default=10, choices=[5, 10])  # label horizon
    ap.add_argument("--seed", type=int, default=42)  # torch seed
    args = ap.parse_args()  # read them
    H, SEED = args.horizon, args.seed  # short names
    tag = f"N{H}_seed{SEED}"  # file-name tag

    print_header(f"STEP 28 - PARITY CELL 4 (GAT + VAEP features), N = {H}, seed {SEED}")  # title
    train_pkl = OUTPUT_DIR / f"27_vaep_features_A_train_N{H}.pkl"  # A features, training
    val_pkl = OUTPUT_DIR / f"27_vaep_features_A_validation_N{H}.pkl"  # A features, validation
    for p in [TRAIN_CKPT, VAL_CKPT, train_pkl, val_pkl]:  # inputs must exist
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run Step 14 and script 16 first.")  # stop with a clear message if a needed file is missing

    y_col = "target_score_next10_literal" if H == 10 else N5_COLUMN  # label column in the pickles
    relabel_tr = relabel_va = None  # no relabelling at N=10
    if H == 5:  # N=5 labels come from their own file
        t5 = pd.read_csv(N5_TARGET_FILE)
        d = {(int(a), int(b)): bool(c) for a, b, c in zip(t5["game_id"], t5["action_index"], t5[N5_COLUMN])}  # (game, action) -> label
        relabel_tr = relabel_va = d

    tr_graphs, tr_keys = cg.load_graph_checkpoint(TRAIN_CKPT, "TRAINING")  # Step 14 training graphs
    va_graphs, va_keys = cg.load_graph_checkpoint(VAL_CKPT, "VALIDATION")  # Step 14 validation graphs
    print("Building standardised VAEP feature matrices (training mean and std):")
    tr_arr, tr_klist, mean, std, feat_cols = prepare_matrix(train_pkl, y_col)  # training matrix, sets mean and std
    k_g = set(zip(tr_keys["game_id"].astype(int), tr_keys["action_index"].astype(int)))  # keys on graphs
    k_f = set(tr_klist)  # keys on features
    print(f"training keys: graphs {len(k_g):,}  features {len(k_f):,}  in graphs only {len(k_g-k_f):,}  in features only {len(k_f-k_g):,}")
    tr_graphs, tr_keys = attach_features(tr_graphs, tr_keys, tr_arr, tr_klist, relabel=relabel_tr)  # attach to training graphs
    va_arr, va_klist, _, _, _ = prepare_matrix(val_pkl, y_col, mean=mean, std=std)  # validation uses the TRAINING mean and std
    va_graphs, va_keys = attach_features(va_graphs, va_keys, va_arr, va_klist, relabel=relabel_va)  # attach to validation graphs
    del tr_klist, va_klist, k_g, k_f
    gc.collect()  # release key lists
    n_vaep = len(feat_cols)  # width of the VAEP input
    print(f"VAEP feature columns used: {n_vaep}")

    torch.manual_seed(SEED)  # seed BEFORE the weights are created
    n_pos = int(tr_keys["true_target"].sum())  # positive training graphs
    n_neg = len(tr_keys) - n_pos  # negative training graphs
    pos_weight = n_neg / n_pos  # loss weight for the rare class (fails loudly when there are no positives, as before)
    print(f"train graphs {len(tr_graphs):,}  positives {n_pos:,}  pos_weight {pos_weight:.4f}")

    model = GATPlusVAEP(n_vaep)  # weights come from the seed above
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Adam
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float))  # weighted loss
    loader = DataLoader(tr_graphs, batch_size=TRAIN_BATCH_SIZE, shuffle=True)  # reshuffled each epoch from the seeded generator

    hist = []  # one row per epoch
    t0 = time.time()  # overall timer
    for epoch in range(1, N_EPOCHS + 1):  # fixed 30 epochs
        model.train()  # dropout on
        tot, nb, e0 = 0.0, 0, time.time()  # loss sum, batch count, epoch timer
        for b in loader:  # one training batch
            opt.zero_grad()  # clear gradients
            logits = model(b.x, b.edge_index, b.edge_attr, b.actor_mask, b.vaep)  # forward
            loss = loss_fn(logits, b.y)  # weighted BCE
            loss.backward()  # gradients
            opt.step()  # update weights
            tot += loss.item()
            nb += 1  # count the batches
        row = {"epoch": epoch, "average_loss": tot / nb, "epoch_seconds": time.time() - e0}  # history row
        msg = f"Epoch {epoch:02d}/{N_EPOCHS} | loss {row['average_loss']:.6f} | {row['epoch_seconds']:.0f}s"  # log line
        if epoch % 5 == 0 or epoch == N_EPOCHS:  # monitoring only, never used for selection
            _, _, a, _, _ = evaluate(model, va_graphs)
            row["validation_auc_monitoring_only"] = a  # keep the monitoring AUC for the history table
            msg += f" | [monitoring only] val AUC {a:.4f}"  # add the AUC to the log line
        print(msg)  # show the log line
        hist.append(row)  # store the history row
    print(f"training time {(time.time()-t0)/60:.1f} min")  # show how long training took
    pd.DataFrame(hist).to_csv(OUTPUT_DIR / f"28_gat_plus_vaep_{tag}_training_history.csv", index=False)  # history

    y, p, auc, pr, br = evaluate(model, va_graphs)  # final evaluation
    print_header("RESULT")
    print(f"GAT + VAEP features, validation AUC-ROC = {auc:.4f}  PR-AUC = {pr:.4f}  Brier = {br:.4f}")  # show the final validation numbers
    out = va_keys.copy()  # keys and labels
    out["predicted_probability"] = p  # scores
    out.to_csv(OUTPUT_DIR / f"28_gat_plus_vaep_{tag}_validation_predictions.csv", index=False)  # predictions
    pd.DataFrame([{"horizon": H, "seed": SEED, "validation_auc_roc": round(auc, 4),
                   "validation_pr_auc": round(pr, 4), "validation_brier": round(br, 4),
                   "train_graphs": len(tr_graphs), "validation_graphs": len(va_graphs),
                   "n_vaep_features": n_vaep, "epochs": N_EPOCHS, "lr": LEARNING_RATE,
                   "batch": TRAIN_BATCH_SIZE}]).to_csv(OUTPUT_DIR / f"28_gat_plus_vaep_{tag}_summary.csv", index=False)  # summary
    torch.save(model.state_dict(), OUTPUT_DIR / f"28_gat_plus_vaep_{tag}_model_state.pt")  # weights
    print("Next: run 18_parity_decomposition.py")
    return 0  # success


if __name__ == "__main__":  # only when run directly
    common.run_main(main, "STEP 28")  # shared error handling
