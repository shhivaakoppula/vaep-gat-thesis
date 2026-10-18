# ==============================================================================
# STEP 10 (file 07_train_gat_baseline.py) - TRAIN THE GAT ON THE VERIFIED RQ1 TARGET (PROTOTYPE, 34 MATCHES)
# ==============================================================================
# The GAT side of the RQ1 comparison. It uses the same verified target and the same 24 train / 5 validation match split
# as the VAEP baseline in script 06, so the two models are compared fairly. The 5 test matches are not touched.
# Graph for each action: one node per visible player (7 features: x, y, teammate flag, goalkeeper flag, actor flag,
# distance to goal, angle to goal), each player joined to its 5 nearest players (k=5, Raabe et al. 2023), edge feature =
# 1 if both players are on the same team. Two GAT layers (4 heads each) make an embedding per player, and the prediction
# is read from the ACTOR node only, because the actor is the player doing the action being scored (like VAEP itself).
# Speed changes compared with the original: the freeze frames of a match are indexed by event id once (the original
# filtered the whole frame table for every single action), rows are walked as arrays instead of with iterrows, and the
# edge lists are made with numpy instead of a double Python loop. The neighbour search itself is still scikit-learn's
# NearestNeighbors, so ties between equally distant players are resolved exactly as before. The training loop is left
# alone (same seed, same batch size 64, same order of operations), so weights and predictions are the same.
# ==============================================================================

import json  # reading the 360 files
import gc  # release memory between matches

import common_gat as cg  # imports torch first (Windows DLL order), then numpy/pandas
import torch  # tensors
import torch.nn as nn  # the loss
import numpy as np  # arrays
import pandas as pd  # tables
from torch_geometric.data import Data  # one graph
from torch_geometric.loader import DataLoader  # batches graphs
from sklearn.neighbors import NearestNeighbors  # k-nearest-neighbour search

import common  # paths and helpers
from common import OUTPUT_DIR, DATA_DIR, print_header, require_file  # shared pieces
import common_spadl as cs  # loader, metadata, SPADL actions

THREE_SIXTY_DIR = DATA_DIR / "three-sixty"  # 360 freeze-frame json files, one per match, named by match id
TARGET_FILE = OUTPUT_DIR / "rq1_literal_next10_target_34_matches.csv"  # verified target
SPLIT_FILE = OUTPUT_DIR / "final_common_match_split.csv"  # frozen split
VALIDATION_PREDICTIONS_FILE = OUTPUT_DIR / "10_gat_validation_predictions.csv"  # predictions for the DeLong test
RESULTS_SUMMARY_FILE = OUTPUT_DIR / "10_gat_baseline_results_summary.csv"  # metric/value table
MODEL_SAVE_FILE = OUTPUT_DIR / "10_gat_model_state.pt"  # model weights
K_NEAREST_NEIGHBOURS = 5  # neighbours per node
N_EPOCHS = 30  # fixed number of epochs
LEARNING_RATE = 0.001  # Adam learning rate
RANDOM_STATE = 42  # seed
GOAL_X_NORM, GOAL_Y_NORM = 1.0, 0.5  # goal position on the normalised pitch
TARGET_COLUMN = "target_score_next10_literal"  # label column
EVENT_ID_CANDIDATES = ["original_event_id", "event_id"]  # SPADL columns that link back to the StatsBomb event uuid


def load_verified_target():  # reads the verified target and checks its columns
    print_header("LOAD MY VERIFIED TARGET (from Step 07C, confirmed by Step 08C)")  # title
    target = pd.read_csv(TARGET_FILE)  # saved target
    required_columns = ["game_id", "action_index", TARGET_COLUMN]  # needed columns
    missing = set(required_columns) - set(target.columns)
    if missing:  # stop if a needed column is missing
        raise ValueError(f"Missing target columns: {sorted(missing)}")  # stop with the error: Missing target columns
    target = target[required_columns].copy()  # keep three columns
    target["game_id"] = pd.to_numeric(target["game_id"], errors="raise").astype(int)  # integer key
    target["action_index"] = pd.to_numeric(target["action_index"], errors="raise").astype(int)  # integer key
    target[TARGET_COLUMN] = target[TARGET_COLUMN].astype(bool)  # boolean label
    print(f"Verified rows: {len(target):,}")
    print(f"Verified positive targets: {int(target[TARGET_COLUMN].sum()):,}")  # tell the user: Verified positive targets (plus its value)
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


def find_event_id_column(actions):
    """Name of the SPADL column that holds the StatsBomb event uuid."""
    for candidate in EVENT_ID_CANDIDATES:  # in order of preference
        if candidate in actions.columns:  # found
            return candidate
    raise KeyError("I could not find a column linking SPADL actions back to the original StatsBomb event UUID. I looked for "  # stop with the error: I could not find a column linking SPADL actions back to the...
                   f"these column names: {EVENT_ID_CANDIDATES}. Actual columns available: {list(actions.columns)}. I need to check "
                   "my installed socceraction version's SPADL output schema and update EVENT_ID_CANDIDATES above.")


def load_360_frames_for_match(game_id):
    """Freeze frames of one match as {event_uuid: (x, y, teammate, actor, keeper) numpy arrays}, players in file order."""
    frame_file = THREE_SIXTY_DIR / f"{int(game_id)}.json"  # path of the 360 file
    if not frame_file.exists():  # missing file
        raise FileNotFoundError(f"I could not find the 360 data file for match {game_id} at:\n{frame_file}\n"
                                f"I am assuming my 360 JSON files are stored one per match under {THREE_SIXTY_DIR}. "
                                "If this path is wrong, I need to update THREE_SIXTY_DIR at the top of this script.")
    with open(frame_file, "r", encoding="utf-8") as f:  # open it
        raw_frames = json.load(f)  # parse
    collected = {}  # event_uuid -> five parallel lists
    for entry in raw_frames:  # one entry per event
        uuid = entry.get("event_uuid")  # which event
        for p in entry.get("freeze_frame", []):  # every visible player
            loc = p.get("location", [None, None])  # [x, y]
            rec = collected.setdefault(uuid, ([], [], [], [], []))  # lists for this event
            rec[0].append(loc[0]); rec[1].append(loc[1])  # position (None becomes NaN, as in the DataFrame version)
            rec[2].append(bool(p.get("teammate", False))); rec[3].append(bool(p.get("actor", False))); rec[4].append(bool(p.get("keeper", False)))  # flags
    return {u: (np.array(r[0], dtype=float), np.array(r[1], dtype=float), np.array(r[2], dtype=float),  # x, y, teammate
               np.array(r[3], dtype=float), np.array(r[4], dtype=float)) for u, r in collected.items()}  # actor, keeper


def build_graph_for_action(frame, target_label):
    """One PyG graph from one freeze frame, or None if it has fewer than 2 players or not exactly one actor."""
    x_raw, y_raw, teammate, actor, keeper = frame  # unpack the five arrays
    n_players = len(x_raw)  # visible players
    if n_players < 2:  # a graph needs at least two nodes
        return None
    k = min(K_NEAREST_NEIGHBOURS, n_players - 1)  # cannot have more neighbours than other players
    x_norm = x_raw / 120.0  # pitch length in StatsBomb units
    y_norm = y_raw / 80.0  # pitch width in StatsBomb units
    if actor.sum() != 1:  # exactly one actor node is needed, it is where the prediction is read
        return None
    dx, dy = GOAL_X_NORM - x_norm, GOAL_Y_NORM - y_norm  # vector to the goal centre
    distance_goal = np.sqrt(dx ** 2 + dy ** 2)  # distance to goal
    angle_goal = np.arctan2(np.abs(dy), np.abs(dx)) / np.pi  # angle to goal
    node_features = np.stack([x_norm, y_norm, teammate, keeper, actor, distance_goal, angle_goal], axis=1)  # 7 features
    positions = np.stack([x_norm, y_norm], axis=1)  # coordinates for the neighbour search
    _, indices = NearestNeighbors(n_neighbors=k + 1).fit(positions).kneighbors(positions)  # each point finds itself first, then k neighbours
    src = np.repeat(np.arange(n_players), k)  # node i repeated k times
    dst = indices[:, 1:k + 1].reshape(-1)  # neighbour columns 1..k, row by row
    same_team = (teammate[src] == teammate[dst]).astype(np.float32)  # 1.0 when both share the teammate flag
    return Data(x=torch.tensor(node_features, dtype=torch.float),  # node features
                edge_index=torch.from_numpy(np.stack([src, dst]).astype(np.int64)),  # (2, E) edge list
                edge_attr=torch.from_numpy(same_team).reshape(-1, 1),  # (E, 1) same-team flag
                y=torch.tensor([float(target_label)], dtype=torch.float),  # label
                actor_mask=torch.tensor(actor == 1.0, dtype=torch.bool))  # which node is the actor


def build_graphs_for_matches(loader, home, match_ids, target_by_game, label):  # builds one graph per eligible action that has a usable 360 frame
    print_header(f"BUILD GAT GRAPHS — {label} MATCHES")  # title
    graphs, keys_and_labels = [], []  # graphs and their (game_id, action_index, label)
    match_ids_sorted = sorted(match_ids)  # fixed order
    total_actions, total_graphs_built = 0, 0  # running totals
    for number, game_id in enumerate(match_ids_sorted, start=1):
        actions = cs.load_actions(loader, game_id, int(home[game_id]))  # SPADL actions
        event_id_column = find_event_id_column(actions)  # link column
        frames = load_360_frames_for_match(game_id)  # freeze frames by event uuid
        tgt = target_by_game.get(game_id)  # eligible rows of this match
        if tgt is None:  # match absent from the target
            tgt = pd.DataFrame({"action_index": [], TARGET_COLUMN: []})  # nothing eligible
        eligible = actions.merge(tgt, on="action_index", how="inner")  # eligible actions in action order
        n_before_360 = len(eligible)  # actions that could get a graph
        graphs_this_match = 0
        for uuid, a_idx, label_value in zip(eligible[event_id_column].to_numpy(), eligible["action_index"].to_numpy(), eligible[TARGET_COLUMN].to_numpy()):  # go through the eligible actions as arrays, not row by row (continues on the next line)
            frame = frames.get(uuid)  # freeze frame of this event
            if frame is None:  # no 360 data for this event
                continue
            graph = build_graph_for_action(frame, label_value)  # graph or None
            if graph is None:  # unusable frame
                continue
            graphs.append(graph)  # keep the graph
            keys_and_labels.append({"game_id": int(game_id), "action_index": int(a_idx), "true_target": int(label_value)})  # keep its keys and label
            graphs_this_match += 1  # count the graphs of this match
        print(f"[{number}/{len(match_ids_sorted)}] Match {game_id}: {n_before_360:,} eligible actions, {graphs_this_match:,} graphs built "  # tell the user: [ / ] Match : eligible actions, graphs built (some actions have no... (plus its value)
              "(some actions have no matching 360 frame or fewer than 2 visible players)")
        total_actions += n_before_360  # add to the number of eligible actions
        total_graphs_built += graphs_this_match  # add to the number of graphs built
        del actions, frames, eligible  # free the match
        gc.collect()
    print()  # print a blank line
    print(f"Total eligible actions across {label.lower()} matches: {total_actions:,}")  # tell the user: Total eligible actions across matches (plus its value)
    print(f"Total graphs successfully built: {total_graphs_built:,}")  # tell the user: Total graphs successfully built (plus its value)
    coverage_pct = total_graphs_built / total_actions * 100 if total_actions > 0 else 0
    print(f"360 graph coverage: {coverage_pct:.2f}% of eligible actions")  # tell the user: 360 graph coverage: % of eligible actions (plus its value)
    return graphs, pd.DataFrame(keys_and_labels)  # hand back graphs, pd.DataFrame(keys_and_labels)


def train_gat(train_graphs, train_keys):  # trains the GAT with the fixed seed and settings
    print_header("TRAIN GAT — MY GRAPH ATTENTION NETWORK BASELINE")  # title
    torch.manual_seed(RANDOM_STATE)  # seed BEFORE the weights are created (do not reorder)
    n_positive = int(train_keys["true_target"].sum())  # positives
    n_negative = int(len(train_keys) - n_positive)  # negatives
    pos_weight_value = n_negative / n_positive if n_positive > 0 else 1.0  # class-imbalance weight
    print(f"Training graphs: {len(train_graphs):,}")
    print(f"Training positive: {n_positive:,}")  # tell the user: Training positive (plus its value)
    print(f"Training negative: {n_negative:,}")  # tell the user: Training negative (plus its value)
    print(f"pos_weight I am applying to my loss function: {pos_weight_value:.4f}")  # tell the user: pos_weight I am applying to my loss function (plus its value)
    print("(This is the direct GAT equivalent of the scale_pos_weight I used for XGBoost in Step 09 — calculated the same way, "  # tell the user: (This is the direct GAT equivalent of the scale_pos_weight I used...
          "from my own training data, so both models handle my severe class imbalance consistently.)")
    print()  # print a blank line
    print("GAT architecture — matches my already-documented design:")  # tell the user: GAT architecture — matches my already-documented design
    print("  Layer 1: GATConv(7 -> 64, 4 attention heads)")  # tell the user: Layer 1: GATConv(7 -> 64, 4 attention heads)
    print("  Layer 2: GATConv(256 -> 32, 4 attention heads)")  # tell the user: Layer 2: GATConv(256 -> 32, 4 attention heads)
    print("  Classifier: 128 -> 32 -> 1, dropout 0.2")  # tell the user: Classifier: 128 -> 32 -> 1, dropout 0.2
    print(f"  Epochs: {N_EPOCHS} (already configured)")  # tell the user: Epochs: (already configured) (plus its value)
    print(f"  Learning rate: {LEARNING_RATE} (standard Adam starting value)")  # tell the user: Learning rate: (standard Adam starting value) (plus its value)
    model = cg.GATBaseline(in_channels=7)  # same layers as the original, same order, so the same initial weights
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Adam
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], dtype=torch.float))  # weighted loss
    loader = DataLoader(train_graphs, batch_size=64, shuffle=True)  # batch size 64 in this step, reshuffled every epoch
    print()
    print("Training...")  # tell the user: Training
    print()  # print a blank line
    for epoch in range(1, N_EPOCHS + 1):
        model.train()  # dropout on
        total_loss, n_batches = 0.0, 0  # running loss
        for batch in loader:
            optimiser.zero_grad()  # clear old gradients
            logits = model(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, actor_mask=batch.actor_mask)  # forward pass
            loss = loss_fn(logits, batch.y)  # weighted binary cross-entropy
            loss.backward()  # gradients
            optimiser.step()  # weight update
            total_loss += loss.item()
            n_batches += 1  # count the batches
        average_loss = total_loss / n_batches if n_batches > 0 else float("nan")  # mean loss this epoch
        print(f"Epoch {epoch:02d}/{N_EPOCHS} | Average loss = {average_loss:.6f}")
    print()  # print a blank line
    print("Training complete — all epochs finished.")  # tell the user: Training complete — all epochs finished
    return model  # hand back model


def evaluate_on_validation(model, validation_graphs):  # computes the validation metrics and prints them
    print_header("EVALUATE ON MY 5 VALIDATION MATCHES")  # title
    y_true, y_pred_proba, auc_roc, pr_auc, brier = cg.evaluate(model, validation_graphs, batch_size=256)  # same batch size as the original
    print(f"Validation graphs: {len(y_true):,}")
    print(f"Validation positive rate: {y_true.mean() * 100:.4f}%")  # tell the user: Validation positive rate: % (plus its value)
    print()  # print a blank line
    print("=" * 80)  # print a divider line
    print(f"  MY GAT VALIDATION AUC-ROC = {auc_roc:.4f}")  # tell the user: MY GAT VALIDATION AUC-ROC (plus its value)
    print("=" * 80)  # print a divider line
    print()  # print a blank line
    print(f"PR-AUC (Average Precision): {pr_auc:.4f}")  # tell the user: PR-AUC (Average Precision) (plus its value)
    print(f"Brier score: {brier:.4f}")  # tell the user: Brier score (plus its value)
    print()  # print a blank line
    print("IMPORTANT: this is my VALIDATION result, not my final thesis result. My 5 test matches remain completely untouched.")  # tell the user: IMPORTANT: this is my VALIDATION result, not my final thesis...
    print()  # print a blank line
    print(f"FOR DIRECT COMPARISON: my Step 09 VAEP baseline validation AUC-ROC was 0.6873. My GAT validation AUC-ROC here is "  # tell the user: FOR DIRECT COMPARISON: my Step 09 VAEP baseline validation AUC-ROC... (plus its value)
          f"{auc_roc:.4f}. I am reporting both numbers honestly — I have not yet run the paired DeLong test to check whether any "
          "difference between them is statistically significant, and I will not claim one model is better until I do.")
    return y_pred_proba, y_true, auc_roc, pr_auc, brier  # hand back y_pred_proba, y_true, auc_roc, pr_auc, brier


def main():  # runs the whole step from start to finish
    common.verify_environment()  # optional interpreter check
    print_header("STEP 10 — TRAIN GAT BASELINE")  # title
    print("Purpose:")
    print("I am training my Graph Attention Network on the same Step 08C-verified target, the same 24 training / 5 validation "  # tell the user: I am training my Graph Attention Network on the same Step...
          "matches, and the same evaluation metrics I already used for VAEP in Step 09. This produces my first real AUC-ROC "
          "number for the GAT side of RQ1.")
    print()  # print a blank line
    print("IMPORTANT:")  # tell the user: IMPORTANT
    print("This script does not touch my 5 test matches.")  # tell the user: This script does not touch my 5 test matches

    require_file(TARGET_FILE, "Verified target (Step 07C/08C)")  # input must exist
    require_file(SPLIT_FILE, "Frozen split")  # input must exist
    target = load_verified_target()
    train_ids, validation_ids, test_ids = load_split()  # the match ids of the three splits
    loader = cs.create_loader()  # the local StatsBomb reader
    metadata = cs.find_match_metadata(loader, train_ids + validation_ids)  # home team ids (test matches are never loaded)
    home = dict(zip(metadata["game_id"], metadata["home_team_id"]))  # game_id -> home team id
    target_by_game = {gid: grp[["action_index", TARGET_COLUMN]] for gid, grp in target.groupby("game_id")}  # per-match slices

    train_graphs, train_keys = build_graphs_for_matches(loader, home, train_ids, target_by_game, "TRAINING")  # training graphs and their keys
    validation_graphs, validation_keys = build_graphs_for_matches(loader, home, validation_ids, target_by_game, "VALIDATION")  # validation graphs and their keys

    model = train_gat(train_graphs, train_keys)  # train the model
    y_pred_proba, y_true, auc_roc, pr_auc, brier = evaluate_on_validation(model, validation_graphs)  # evaluate on the validation graphs

    # same column layout as the VAEP predictions file (game_id, action_index, true_target, predicted_probability)
    predictions_out = validation_keys.copy()
    predictions_out["predicted_probability"] = y_pred_proba  # attach the predicted probability to each key
    predictions_out.to_csv(VALIDATION_PREDICTIONS_FILE, index=False)  # save the predictions
    print()  # print a blank line
    print(f"Validation predictions saved to:\n{VALIDATION_PREDICTIONS_FILE}")  # tell the user: Validation predictions saved to (plus its value)
    torch.save(model.state_dict(), MODEL_SAVE_FILE)  # model weights
    print(f"Model weights saved to:\n{MODEL_SAVE_FILE}")

    summary_rows = [  # metric/value rows for the summary (continues on the next lines)
        {"metric": "train_graphs", "value": len(train_graphs)},
        {"metric": "train_positive", "value": int(train_keys["true_target"].sum())},
        {"metric": "validation_graphs", "value": len(validation_graphs)},
        {"metric": "validation_positive", "value": int(validation_keys["true_target"].sum())},
        {"metric": "k_nearest_neighbours", "value": K_NEAREST_NEIGHBOURS},
        {"metric": "n_epochs", "value": N_EPOCHS},
        {"metric": "learning_rate", "value": LEARNING_RATE},
        {"metric": "validation_auc_roc", "value": round(auc_roc, 4)},
        {"metric": "validation_pr_auc", "value": round(pr_auc, 4)},
        {"metric": "validation_brier_score", "value": round(brier, 4)},
        {"metric": "random_state", "value": RANDOM_STATE},
        {"metric": "test_matches_touched", "value": "NO — reserved"},
    ]  # metric/value table
    pd.DataFrame(summary_rows).to_csv(RESULTS_SUMMARY_FILE, index=False)
    print(f"Results summary saved to:\n{RESULTS_SUMMARY_FILE}")  # tell the user: Results summary saved to (plus its value)

    print_header("STEP 10 COMPLETE")  # print the closing banner
    print(f"My GAT validation AUC-ROC is {auc_roc:.4f}, trained on {len(train_graphs):,} graphs from {len(train_ids)} matches, "  # tell the user: My GAT validation AUC-ROC is , trained on graphs from matches,... (plus its value)
          f"evaluated on {len(validation_graphs):,} graphs from {len(validation_ids)} matches never seen during training.")
    print()  # print a blank line
    print("NEXT STEP: I run a paired DeLong test comparing this GAT validation AUC-ROC against my Step 09 VAEP validation AUC-ROC, "  # tell the user: NEXT STEP: I run a paired DeLong test comparing this GAT...
          "using the two saved prediction files, on whichever observations are common to both (my Step 09 predictions and this "
          "step's predictions should both cover the same validation actions, minus any actions my GAT graph pipeline could not "
          "build a graph for due to missing 360 data). Only after that statistical test do I know whether any difference between "
          "the two models is real or just chance.")
    return 0  # success


if __name__ == "__main__":  # only when run directly
    common.run_main(main, "STEP 10")  # shared error handling
