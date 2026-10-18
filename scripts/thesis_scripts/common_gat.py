# ==============================================================================
# common_gat.py  -  the GAT model, graph helpers, training loop and evaluation
# ==============================================================================
# Steps 12, 14 (steps 19-23 in your numbering of the ablations) and the parity scripts all repeated
# this code. It lives here once. Nothing about the model, the seeds, the batch order or the
# optimiser changed, so a run produces the same weights and the same predictions as before.
# IMPORT ORDER MATTERS: torch is imported first, before numpy and pandas, because that is the order
# that avoids the Windows DLL clash you hit earlier.
# ==============================================================================

import os  # for the OpenMP workaround below

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # must be set before torch loads its libraries

import torch  # tensors and autograd (first on purpose)
import torch.nn as nn  # layers and the loss
import torch.nn.functional as F  # the ELU activation
from torch_geometric.data import Data  # one graph = one Data object
from torch_geometric.loader import DataLoader  # batches many small graphs into one big graph
from torch_geometric.nn import GATConv  # the graph attention layer

import time  # for epoch timing in the log
import numpy as np  # arrays
import pandas as pd  # tables
from sklearn.neighbors import NearestNeighbors  # k-nearest-neighbour search for the edges
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss  # headline metrics

# Training settings that are identical in every GAT script (changing them would change the results).
N_EPOCHS = 30  # fixed number of epochs, no early stopping
LEARNING_RATE = 0.001  # Adam learning rate
TRAIN_BATCH_SIZE = 256  # graphs per training batch
EVAL_BATCH_SIZE = 512  # graphs per evaluation batch
RANDOM_STATE = 42  # the seed passed to torch.manual_seed


class GATBaseline(nn.Module):
    """Two GAT layers (4 heads each) followed by a small classifier on the actor node."""

    def __init__(self, in_channels=7):  # builds the two GAT layers and the classifier
        super().__init__()  # set up the nn.Module machinery
        self.gat1 = GATConv(in_channels=in_channels, out_channels=64, heads=4, edge_dim=1)  # first attention layer
        self.gat2 = GATConv(in_channels=64 * 4, out_channels=32, heads=4, edge_dim=1)  # second attention layer
        self.classifier = nn.Sequential(  # turns the actor embedding into one logit
            nn.Linear(32 * 4, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1),  # 128 -> 32 -> 1
        )

    def forward(self, x, edge_index, edge_attr, actor_mask):  # one forward pass over a batch of graphs
        h = self.gat1(x, edge_index, edge_attr)  # first round of attention over neighbours
        h = F.elu(h)  # non-linearity
        h = self.gat2(h, edge_index, edge_attr)  # second round of attention
        h = F.elu(h)  # non-linearity
        actor_embeddings = h[actor_mask]  # keep only the ball-carrier node of every graph
        logits = self.classifier(actor_embeddings)  # one score per graph
        return logits.squeeze(-1)  # shape (batch,)


def load_graph_checkpoint(path, label):
    """Read a Step 14 checkpoint and return (list of graphs, DataFrame of keys and true labels)."""
    data = torch.load(path, weights_only=False)  # the file holds pickled Data objects, so weights_only must be False
    graphs = data["graphs"]  # one Data object per eligible action
    keys_df = pd.DataFrame(data["keys_and_labels"])  # game_id, action_index, true_target for every graph
    print(f"{label} graphs loaded: {len(graphs):,} | positives: {int(keys_df['true_target'].sum()):,}")  # quick sanity line
    return graphs, keys_df  # hand both back


def rebuild_graph_edges(graph, k):
    """Same graph, new k-nearest-neighbour edges. Node features, target and actor mask are untouched."""
    x = graph.x.numpy()  # node feature matrix as numpy (shares memory, no copy)
    n_players = x.shape[0]  # number of visible players in this frame
    positions = x[:, 0:2]  # columns 0-1 are x_norm and y_norm
    teammate = x[:, 2]  # column 2 is the teammate flag

    effective_k = min(k, n_players - 1)  # cannot have more neighbours than other players
    if effective_k < 1:  # a single-player graph has no edges at all
        edge_index = torch.zeros((2, 0), dtype=torch.long)  # empty edge list
        edge_attr = torch.zeros((0, 1), dtype=torch.float)  # empty edge features
    else:
        nbrs = NearestNeighbors(n_neighbors=effective_k + 1).fit(positions)  # +1 because each point finds itself first
        _, indices = nbrs.kneighbors(positions)  # indices of the nearest points, row i = neighbours of node i
        src = np.repeat(np.arange(n_players), effective_k)  # node i repeated k times, in node order
        dst = indices[:, 1 : effective_k + 1].reshape(-1)  # columns 1..k of every row, row by row (same order as the old double loop)
        same_team = (teammate[src] == teammate[dst]).astype(np.float32)  # 1.0 when both players share the teammate flag
        edge_index = torch.from_numpy(np.stack([src, dst]).astype(np.int64))  # shape (2, E) as PyG expects
        edge_attr = torch.from_numpy(same_team).reshape(-1, 1)  # shape (E, 1)

    return Data(  # new graph object sharing the unchanged tensors
        x=graph.x, edge_index=edge_index, edge_attr=edge_attr, y=graph.y, actor_mask=graph.actor_mask,
    )


def rebuild_graphs_at_k(graphs, k, label):
    """Rebuild edges for a whole list of graphs and report the average edge count."""
    start = time.time()  # timer
    rebuilt = [rebuild_graph_edges(g, k) for g in graphs]  # one new graph per old graph
    total_edges = sum(g.edge_index.shape[1] for g in rebuilt)  # edges over all graphs
    average = total_edges / len(rebuilt) if rebuilt else 0  # mean edges per graph
    print(f"Rebuilt {label} edges at k={k} in {time.time() - start:.1f}s (avg {average:.1f} edges/graph)")  # progress line
    return rebuilt  # list of Data objects


def select_features(graphs, column_indices):
    """Keep only some columns of x; edges, target and actor mask are reused as they are."""
    return [  # one slimmer graph per input graph
        Data(x=g.x[:, column_indices], edge_index=g.edge_index, edge_attr=g.edge_attr, y=g.y, actor_mask=g.actor_mask)
        for g in graphs
    ]


def add_dist_to_actor_feature(graphs):
    """Append an 8th node feature: each player's Euclidean distance to the ball-carrier (the node with actor=1)."""
    augmented = []  # new graphs go here
    for g in graphs:  # one graph at a time (kept per graph so the arithmetic is identical to before)
        positions = g.x[:, 0:2]  # x_norm, y_norm of every player
        actor_rows = torch.nonzero(g.x[:, 4] == 1.0, as_tuple=False)  # rows flagged as the actor (column 4)
        if actor_rows.numel() == 0:  # Step 14 guarantees one actor per graph, so this should never happen
            raise ValueError("Found a graph with no actor-flagged node while computing dist_to_actor.")  # stop
        actor_position = positions[actor_rows[0, 0]]  # position of the first (only) actor
        dist_to_actor = torch.norm(positions - actor_position, dim=1, keepdim=True)  # distance of every node to the actor, shape (n, 1)
        augmented.append(Data(x=torch.cat([g.x, dist_to_actor], dim=1), edge_index=g.edge_index,  # x with the new last column
                              edge_attr=g.edge_attr, y=g.y, actor_mask=g.actor_mask))  # everything else reused
    return augmented  # list of 8-feature graphs


def compute_pos_weight(train_keys):
    """negatives / positives of the training set, or 1.0 when there are no positives."""
    n_positive = int(train_keys["true_target"].sum())  # number of positive graphs
    n_negative = int(len(train_keys) - n_positive)  # everything else
    return n_negative / n_positive if n_positive > 0 else 1.0  # same rule as the originals


def evaluate(model, graphs, batch_size=EVAL_BATCH_SIZE):
    """Run the model over graphs and return (y_true, y_pred, auc_roc, pr_auc, brier). batch_size only changes memory use."""
    model.eval()  # switch dropout off
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)  # fixed order
    probs_list, targets_list = [], []  # collect tensors, join once at the end (cheaper than extending Python lists)
    with torch.no_grad():  # no gradients needed
        for batch in loader:  # one batch of graphs at a time
            logits = model(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, actor_mask=batch.actor_mask)  # raw scores
            probs_list.append(torch.sigmoid(logits))  # probabilities, still float32
            targets_list.append(batch.y)  # true labels
    y_pred = torch.cat(probs_list).double().numpy()  # float32 -> float64 is exact, same numbers .tolist() gave
    y_true = torch.cat(targets_list).double().numpy()  # labels as float64
    auc_roc = roc_auc_score(y_true, y_pred)  # ranking quality
    pr_auc = average_precision_score(y_true, y_pred)  # precision-recall area
    brier = brier_score_loss(y_true, y_pred)  # calibration error
    return y_true, y_pred, auc_roc, pr_auc, brier  # same tuple the old evaluate returned


def train_gat(train_graphs, validation_graphs, pos_weight_value, in_channels=7, name="GAT",
              n_epochs=N_EPOCHS, start_epoch=1, model=None, optimiser=None, after_epoch=None, history=None):
    """Train the GAT exactly as before and return (model, seconds).

    The order of operations is the one that fixes the random numbers: seed -> build model -> build
    optimiser -> build DataLoader. Do not reorder it. If a model and optimiser are passed in (resume
    case) the seed is not reset, matching the resume branch of the old k=8 script.
    """
    fresh = model is None  # True when starting from scratch
    if fresh:  # fresh run
        torch.manual_seed(RANDOM_STATE)  # fix the seed BEFORE the weights are created
        model = GATBaseline(in_channels=in_channels)  # initial weights come from that seed
        optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Adam as before
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], dtype=torch.float))  # weighted loss for the rare positives
    loader = DataLoader(train_graphs, batch_size=TRAIN_BATCH_SIZE, shuffle=True)  # reshuffled every epoch from the seeded generator

    t0 = time.time()  # overall timer
    for epoch in range(start_epoch, n_epochs + 1):  # fixed 30 epochs
        e0 = time.time()  # epoch timer
        model.train()  # dropout on
        total_loss, n_batches = 0.0, 0  # running loss
        for batch in loader:  # one training batch
            optimiser.zero_grad()  # clear old gradients
            logits = model(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, actor_mask=batch.actor_mask)  # forward pass
            loss = loss_fn(logits, batch.y)  # weighted binary cross-entropy
            loss.backward()  # gradients
            optimiser.step()  # weight update
            total_loss += loss.item()  # accumulate for the log
            n_batches += 1  # count batches
        average_loss = total_loss / n_batches if n_batches > 0 else float("nan")  # mean loss this epoch
        epoch_seconds = time.time() - e0  # seconds this epoch took
        row = {"epoch": epoch, "average_loss": average_loss, "epoch_seconds": epoch_seconds}  # one history row
        line = f"  [{name}] Epoch {epoch:02d}/{n_epochs} | Loss = {average_loss:.6f} | {epoch_seconds:.1f}s"  # log line
        if epoch % 5 == 0 or epoch == n_epochs:  # monitoring only, never used to pick a model
            val_auc = evaluate(model, validation_graphs)[2]  # evaluate once and reuse the number
            row["validation_auc_monitoring_only"] = val_auc  # keep it for the history table
            line += f" | [monitoring only] Validation AUC = {val_auc:.4f}"  # extra evaluation
        if history is not None:  # caller wants the per-epoch table
            history.append(row)  # store it
        print(line)  # show progress
        if after_epoch is not None:  # optional hook (used by the resumable k=8 script)
            after_epoch(epoch, model, optimiser)  # e.g. save a checkpoint
    return model, time.time() - t0  # trained model and wall-clock seconds


def save_predictions(keys_df, y_pred, path):
    """Write keys + predicted probability to CSV in the same column layout as before."""
    out = keys_df.copy()  # do not modify the caller's table
    out["predicted_probability"] = y_pred  # new column
    out.to_csv(path, index=False)  # same CSV as the old scripts wrote
    print(f"Validation predictions saved to:\n{path}")  # tell the user where


def metrics_from_predictions(path):
    """Recompute AUC-ROC, PR-AUC and Brier from a saved predictions CSV (used when resuming from disk)."""
    df = pd.read_csv(path)  # saved earlier by save_predictions
    y_true = df["true_target"].to_numpy()  # labels
    y_pred = df["predicted_probability"].to_numpy()  # probabilities
    return roc_auc_score(y_true, y_pred), average_precision_score(y_true, y_pred), brier_score_loss(y_true, y_pred)  # three numbers
