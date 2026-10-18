# ==============================================================================
# STEP 14 (file 12_train_gat_full_scale.py) - BUILD THE 360 GRAPHS AND TRAIN THE GAT AT FULL SCALE
# ==============================================================================
# For every eligible action (verified Step 12 target) a graph is built from the 360 freeze frame:
#   nodes  = visible players, 7 features [x_norm, y_norm, teammate, keeper, actor, dist_to_goal, angle_to_goal]
#   edges  = each player to its k=5 nearest players (positions), edge_attr = same-team flag
# Then the GAT (2 GATConv layers x 4 heads + classifier on the actor node) is trained for 30 epochs
# (Adam 0.001, batch 256, BCE with pos_weight = negatives / positives, seed 42) and scored on the validation matches.
# Speed changes compared with the original: freeze frames are indexed by event id once per match (the original
# re-scanned the whole frame table for every action), rows are iterated as arrays instead of with iterrows, and the
# edge lists are built with numpy instead of a double Python loop. Graph tensors are identical.
# ==============================================================================

import json  # reading the 360 files
import gc  # release memory
import time  # progress timing
import os  # CHECKPOINT_EVERY override

import common_gat as cg  # imports torch first (Windows DLL order), then numpy/pandas
import torch  # tensors and checkpoints
import numpy as np  # arrays
import pandas as pd  # tables
from torch_geometric.data import Data  # graph container
from sklearn.neighbors import NearestNeighbors  # k-nearest-neighbour search

import common  # paths and helpers
from common import OUTPUT_DIR, DATA_DIR, print_header, require_file  # shared pieces
import common_spadl as cs  # loader, metadata, SPADL conversion

THREE_SIXTY_DIR = DATA_DIR / "three-sixty"  # 360 freeze-frame json files
FULL_TARGET_FILE = OUTPUT_DIR / "rq1_literal_next10_target_full_dataset.csv"  # Step 12 target
FULL_SPLIT_FILE = OUTPUT_DIR / "full_scale_match_split.csv"  # Step 12 split
VALIDATION_PREDICTIONS_FILE = OUTPUT_DIR / "14_gat_full_scale_validation_predictions.csv"  # predictions (used by later steps)
RESULTS_SUMMARY_FILE = OUTPUT_DIR / "14_gat_full_scale_results_summary.csv"  # metric/value table
MODEL_SAVE_FILE = OUTPUT_DIR / "14_gat_full_scale_model_state.pt"  # model weights
TRAINING_HISTORY_FILE = OUTPUT_DIR / "14_gat_full_scale_training_history.csv"  # per-epoch loss
K_NEAREST_NEIGHBOURS = 5  # neighbours per node
GOAL_X_NORM, GOAL_Y_NORM = 1.0, 0.5  # goal position on the normalised pitch
TARGET_COLUMN = "target_score_next10_literal"  # label column
EVENT_ID_CANDIDATES = ["original_event_id", "event_id"]  # SPADL columns that link back to the StatsBomb event uuid
CHECKPOINT_EVERY_N_MATCHES = int(os.environ.get("GRAPH_CHECKPOINT_EVERY", "20"))  # progress-save interval (raise it to save disk time)


def estimate_graphs_memory_gb(graphs):
    """Lower-bound size of all tensors in a list of graphs, in GB."""
    total = 0  # bytes
    for g in graphs:  # every graph
        for t in (g.x, g.edge_index, g.edge_attr, g.y, g.actor_mask):  # every tensor
            total += t.element_size() * t.nelement()  # bytes of this tensor
    return total / 1e9  # gigabytes


def load_verified_target():
    """Read the Step 12 target with strict types."""
    print_header("LOAD MY VERIFIED FULL-SCALE TARGET (Step 12)")  # title
    target = pd.read_csv(FULL_TARGET_FILE)  # whole file
    cols = ["game_id", "action_index", TARGET_COLUMN]  # needed columns
    missing = set(cols) - set(target.columns)  # absent columns
    if missing:  # wrong file
        raise ValueError(f"Missing target columns: {sorted(missing)}")  # stop
    target = target[cols].copy()  # keep three columns
    target["game_id"] = pd.to_numeric(target["game_id"], errors="raise").astype(int)  # integer key
    target["action_index"] = pd.to_numeric(target["action_index"], errors="raise").astype(int)  # integer key
    target[TARGET_COLUMN] = target[TARGET_COLUMN].astype(bool)  # boolean label
    print(f"Verified rows: {len(target):,} | positives: {int(target[TARGET_COLUMN].sum()):,}")  # summary
    return target  # clean table


def load_split():
    """Match ids of the training and validation sets."""
    print_header("LOAD FULL-SCALE MATCH SPLIT")  # title
    split = pd.read_csv(FULL_SPLIT_FILE)  # split table
    split["game_id"] = pd.to_numeric(split["game_id"], errors="raise").astype(int)  # integer ids
    split["split"] = split["split"].astype(str).str.lower().str.strip()  # normalise labels
    print(split["split"].value_counts().sort_index())  # counts
    return split[split["split"] == "train"]["game_id"].tolist(), split[split["split"] == "validation"]["game_id"].tolist()  # two lists


def find_event_id_column(actions):
    """Name of the SPADL column that holds the StatsBomb event uuid."""
    for candidate in EVENT_ID_CANDIDATES:  # in order of preference
        if candidate in actions.columns:  # found
            return candidate  # use it
    raise KeyError(f"No column linking SPADL actions to StatsBomb event uuids. Looked for {EVENT_ID_CANDIDATES}. Columns: {list(actions.columns)}")  # stop


def load_360_frames_for_match(game_id):
    """Freeze frames of one match as {event_uuid: (x, y, teammate, actor, keeper) numpy arrays}, rows in file order."""
    frame_file = THREE_SIXTY_DIR / f"{int(game_id)}.json"  # path of the 360 file
    if not frame_file.exists():  # missing file
        raise FileNotFoundError(f"I could not find the 360 data file for match {game_id} at:\n{frame_file}")  # stop
    with open(frame_file, "r", encoding="utf-8") as f:  # open it
        raw_frames = json.load(f)  # parse
    collected = {}  # event_uuid -> list of five lists
    for entry in raw_frames:  # one entry per event
        uuid = entry.get("event_uuid")  # which event
        for p in entry.get("freeze_frame", []):  # every visible player
            loc = p.get("location", [None, None])  # [x, y]
            rec = collected.setdefault(uuid, ([], [], [], [], []))  # five parallel lists for this event
            rec[0].append(loc[0]); rec[1].append(loc[1])  # position (None becomes NaN below, as in the DataFrame version)
            rec[2].append(bool(p.get("teammate", False))); rec[3].append(bool(p.get("actor", False))); rec[4].append(bool(p.get("keeper", False)))  # flags
    return {u: (np.array(r[0], dtype=float), np.array(r[1], dtype=float), np.array(r[2], dtype=float),  # x, y, teammate
               np.array(r[3], dtype=float), np.array(r[4], dtype=float)) for u, r in collected.items()}  # actor, keeper


def build_graph_for_action(frame, target_label):
    """One PyG graph from one freeze frame, or None if it has < 2 players or not exactly one actor."""
    x_raw, y_raw, teammate, actor, keeper = frame  # unpack the five arrays
    n_players = len(x_raw)  # visible players
    if n_players < 2:  # a graph needs at least two nodes
        return None
    k = min(K_NEAREST_NEIGHBOURS, n_players - 1)  # cannot have more neighbours than other players
    x_norm = x_raw / 120.0  # pitch length in StatsBomb units
    y_norm = y_raw / 80.0  # pitch width in StatsBomb units
    if actor.sum() != 1:  # exactly one ball carrier is required
        return None
    dx, dy = GOAL_X_NORM - x_norm, GOAL_Y_NORM - y_norm  # vector to the goal centre
    distance_goal = np.sqrt(dx ** 2 + dy ** 2)  # distance to goal
    angle_goal = np.arctan2(np.abs(dy), np.abs(dx)) / np.pi  # angle to goal, scaled to 0..0.5
    node_features = np.stack([x_norm, y_norm, teammate, keeper, actor, distance_goal, angle_goal], axis=1).astype(np.float32)  # 7 features, float32
    positions = np.stack([x_norm, y_norm], axis=1)  # coordinates used for the neighbour search
    _, indices = NearestNeighbors(n_neighbors=k + 1).fit(positions).kneighbors(positions)  # each point finds itself first, then k neighbours
    src = np.repeat(np.arange(n_players), k)  # node i repeated k times
    dst = indices[:, 1 : k + 1].reshape(-1)  # neighbour columns 1..k, row by row
    same_team = (teammate[src] == teammate[dst]).astype(np.float32)  # 1.0 when both share the teammate flag
    return Data(x=torch.tensor(node_features, dtype=torch.float),  # node features
                edge_index=torch.from_numpy(np.stack([src, dst]).astype(np.int64)),  # (2, E) edge list
                edge_attr=torch.from_numpy(same_team).reshape(-1, 1),  # (E, 1) same-team flag
                y=torch.tensor([float(target_label)], dtype=torch.float),  # label
                actor_mask=torch.tensor(actor == 1.0, dtype=torch.bool))  # which node is the actor


def checkpoint_path(label):  # file where progress for one split is saved
    return OUTPUT_DIR / f"14_gat_checkpoint_{label.lower()}.pt"  # one checkpoint per split


def save_checkpoint(label, graphs, keys_and_labels, processed_match_ids):  # writes the graphs built so far to disk
    torch.save({"graphs": graphs, "keys_and_labels": keys_and_labels, "processed_match_ids": list(processed_match_ids)}, checkpoint_path(label))  # same layout as before


def load_checkpoint(label):  # reads saved progress back so a stopped run can resume
    p = checkpoint_path(label)  # file for this split
    if p.exists():  # an earlier run saved progress
        data = torch.load(p, weights_only=False)  # read it
        return data["graphs"], data["keys_and_labels"], set(data["processed_match_ids"])  # restored state
    return [], [], set()  # fresh start


def build_graphs_for_matches(loader, home, match_ids, target_by_game, label):
    """Build all graphs of a split (resumable). Returns (graphs, keys DataFrame)."""
    print_header(f"BUILD GAT GRAPHS - {label} MATCHES")  # title
    graphs, keys_and_labels, already_processed = load_checkpoint(label)  # resume if possible
    if already_processed:  # resuming
        print(f"[RESUMING FROM CHECKPOINT] {len(already_processed)} matches done, {len(graphs):,} graphs built.")  # tell the user
    ids_sorted = sorted(match_ids)  # fixed order
    remaining_ids = [m for m in ids_sorted if m not in already_processed]  # still to do
    print(f"Matches remaining to process: {len(remaining_ids)} of {len(ids_sorted)} total.")  # report
    start, failed = time.time(), []  # timer, failures
    for number, game_id in enumerate(remaining_ids, start=1):  # one match at a time
        try:  # a broken match must not stop the run
            if game_id not in home:  # no metadata
                print(f"  [SKIP - no metadata] Match {game_id}")  # report
                failed.append((int(game_id), "no metadata"))  # remember
                already_processed.add(game_id)  # do not retry
                continue
            actions = cs.load_actions(loader, game_id, int(home[game_id]))  # SPADL actions
            uuid_col = find_event_id_column(actions)  # link column
            frames = load_360_frames_for_match(game_id)  # freeze frames by event uuid
            tgt = target_by_game.get(game_id)  # eligible rows of this match
            if tgt is None:  # match absent from the target
                tgt = pd.DataFrame({"action_index": [], TARGET_COLUMN: []})  # nothing eligible
            eligible = actions.merge(tgt, on="action_index", how="inner")  # eligible actions in action order
            for uuid, a_idx, label_value in zip(eligible[uuid_col].to_numpy(), eligible["action_index"].to_numpy(), eligible[TARGET_COLUMN].to_numpy()):  # arrays, no iterrows
                frame = frames.get(uuid)  # freeze frame of this event
                if frame is None:  # no 360 data for this event
                    continue
                graph = build_graph_for_action(frame, label_value)  # graph or None
                if graph is None:  # unusable frame
                    continue
                graphs.append(graph)  # keep
                keys_and_labels.append({"game_id": int(game_id), "action_index": int(a_idx), "true_target": int(label_value)})  # keys and label
            del actions, frames, eligible  # free the match
        except Exception as ex:  # report and move on
            print(f"  [FAILED - match skipped, run continues] Match {game_id}: {type(ex).__name__}: {ex}")  # report
            failed.append((int(game_id), f"{type(ex).__name__}: {ex}"))  # remember
        already_processed.add(game_id)  # done (successfully or not)
        gc.collect()  # release memory
        if number % CHECKPOINT_EVERY_N_MATCHES == 0 or number == len(remaining_ids):  # time to save
            elapsed = time.time() - start  # seconds so far
            rate = number / elapsed if elapsed > 0 else 0  # matches per second
            print(f"  [{number}/{len(remaining_ids)} this session] ({len(already_processed)}/{len(ids_sorted)} total) graphs so far: {len(graphs):,}. "
                  f"Elapsed {elapsed / 60:.1f} min. Estimated remaining {((len(remaining_ids) - number) / rate / 60) if rate > 0 else float('nan'):.1f} min.")  # show
            save_checkpoint(label, graphs, keys_and_labels, already_processed)  # save progress
            print(f"  [CHECKPOINT SAVED] {checkpoint_path(label)}")  # confirm
    save_checkpoint(label, graphs, keys_and_labels, already_processed)  # final save so a finished split is never rebuilt
    print(f"\nTotal {label} graphs successfully built: {len(graphs):,}")  # count
    for gid, reason in failed:  # documented failures
        print(f"    Match {gid}: {reason}")  # show
    print(f"Estimated tensor memory for {label} graphs: {estimate_graphs_memory_gb(graphs):.2f} GB (lower bound)")  # memory estimate
    return graphs, pd.DataFrame(keys_and_labels)  # graphs and keys


def main():  # runs the whole step from start to finish
    common.verify_environment()  # optional interpreter check
    print_header("STEP 14 - TRAIN GAT AT FULL SCALE")  # title
    require_file(FULL_TARGET_FILE, "Verified full-scale target")  # input 1
    require_file(FULL_SPLIT_FILE, "Full-scale split")  # input 2
    target = load_verified_target()  # labels
    train_ids, validation_ids = load_split()  # match ids
    target_by_game = {gid: grp[["game_id", "action_index", TARGET_COLUMN]] for gid, grp in target.groupby("game_id")}  # per-match slices
    loader = cs.create_loader()  # StatsBomb reader
    metadata = cs.find_match_metadata(loader, train_ids + validation_ids)  # home team ids
    home = dict(zip(metadata["game_id"], metadata["home_team_id"]))  # game_id -> home team id

    train_graphs, train_keys = build_graphs_for_matches(loader, home, train_ids, target_by_game, "TRAINING")  # training graphs
    gc.collect()  # release memory
    validation_graphs, validation_keys = build_graphs_for_matches(loader, home, validation_ids, target_by_game, "VALIDATION")  # validation graphs
    gc.collect()  # release memory

    print_header("TRAIN GAT - MY FULL-SCALE GRAPH ATTENTION NETWORK")  # title
    pos_weight = cg.compute_pos_weight(train_keys)  # loss weight
    print(f"Training graphs: {len(train_graphs):,} | pos_weight: {pos_weight:.4f}")  # report
    history = []  # filled with one row per epoch
    model, seconds = cg.train_gat(train_graphs, validation_graphs, pos_weight, in_channels=7, name="GAT", history=history)  # train (30 epochs)
    pd.DataFrame(history).to_csv(TRAINING_HISTORY_FILE, index=False)  # same columns as the original history file
    print(f"Training complete in {seconds / 60:.1f} minutes.")  # time
    print_header("FINAL EVALUATION ON THE VALIDATION MATCHES")  # title
    y_true, y_pred, auc_roc, pr_auc, brier = cg.evaluate(model, validation_graphs)  # final evaluation
    print(f"Validation graphs: {len(y_true):,} | positive rate: {y_true.mean() * 100:.4f}%")  # size
    print(f"MY FULL-SCALE GAT VALIDATION AUC-ROC = {auc_roc:.4f} | PR-AUC {pr_auc:.4f} | Brier {brier:.4f}")  # headline numbers
    cg.save_predictions(validation_keys, y_pred, VALIDATION_PREDICTIONS_FILE)  # predictions
    torch.save(model.state_dict(), MODEL_SAVE_FILE)  # model weights
    print(f"Model weights saved to:\n{MODEL_SAVE_FILE}")  # where

    pd.DataFrame([  # metric/value summary read back by Step 18
        {"metric": "train_graphs", "value": len(train_graphs)}, {"metric": "train_positive", "value": int(train_keys["true_target"].sum())},  # training size
        {"metric": "validation_graphs", "value": len(validation_graphs)}, {"metric": "validation_positive", "value": int(validation_keys["true_target"].sum())},  # validation size
        {"metric": "k_nearest_neighbours", "value": K_NEAREST_NEIGHBOURS}, {"metric": "n_epochs", "value": cg.N_EPOCHS},  # settings
        {"metric": "train_batch_size", "value": cg.TRAIN_BATCH_SIZE}, {"metric": "learning_rate", "value": cg.LEARNING_RATE},  # settings
        {"metric": "validation_auc_roc", "value": round(auc_roc, 4)}, {"metric": "validation_pr_auc", "value": round(pr_auc, 4)},  # results
        {"metric": "validation_brier_score", "value": round(brier, 4)},  # calibration
        {"metric": "n_train_matches", "value": len(train_ids)}, {"metric": "n_validation_matches", "value": len(validation_ids)},  # match counts
        {"metric": "random_state", "value": cg.RANDOM_STATE},  # seed
    ]).to_csv(RESULTS_SUMMARY_FILE, index=False)  # save
    print(f"Results summary saved to:\n{RESULTS_SUMMARY_FILE}")  # where
    print_header("STEP 14 COMPLETE")  # done
    return 0  # success


if __name__ == "__main__":  # only when run directly
    common.run_main(main, "STEP 14")  # shared error handling
