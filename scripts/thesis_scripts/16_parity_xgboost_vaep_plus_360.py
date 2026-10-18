# ==============================================================================
# STEP 27 (file 16_parity_xgboost_vaep_plus_360.py) - PARITY RUN, CELL 3: XGBOOST ON VAEP FEATURES (A) + 360 SUMMARY (B)
# ==============================================================================
# Purpose (thesis Section 3.5): give the tree model the SAME information the hybrid GAT will receive:
# VAEP's action-context features (set A) plus summary numbers computed from the 360 freeze frame (set B).
# It also trains a CONTROL: XGBoost on A only, restricted to exactly the actions that have a usable 360 frame.
# That separates "fewer training rows" from "extra 360 features".
# Same as Step 13: SPADL actions, 14 VAEP feature functions, 3-action game state, XGBoost n_estimators=100,
# max_depth=3, learning_rate=0.1, scale_pos_weight = negatives / positives, random_state=42, hist method.
# Same rule as Step 14 for a usable frame: at least 2 players, exactly one actor. No tuning on validation.
# Usage:
#   python 16_parity_xgboost_vaep_plus_360.py               (N = 10)
#   python 16_parity_xgboost_vaep_plus_360.py --horizon 5   (N = 5)
# Speed: freeze-frame summaries are computed only for events that became SPADL actions (the original summarised
# every frame in the file), and the per-match metadata lookup is a dictionary. The numbers are identical.
# ==============================================================================

import argparse  # --horizon option
import gc  # free memory between splits
import json  # read the 360 files
import time  # timing

import numpy as np  # arrays
import pandas as pd  # tables
from xgboost import XGBClassifier  # the tree model
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss  # metrics

import common  # shared helpers
from common import DATA_DIR, OUTPUT_DIR, print_header  # shared paths and header printer
import common_spadl as cs  # StatsBomb loader, SPADL actions, VAEP features

THREE_SIXTY_DIR = DATA_DIR / "three-sixty"  # 360 freeze-frame json files
FULL_SPLIT_FILE = OUTPUT_DIR / "full_scale_match_split.csv"  # train/validation match ids (Step 12)
TARGET_FILES = {  # horizon -> (target file, label column)
    10: (OUTPUT_DIR / "rq1_literal_next10_target_full_dataset.csv", "target_score_next10_literal"),
    5: (OUTPUT_DIR / "21_horizon_N5_target.csv", "target_score_next5_literal"),
}
RANDOM_STATE = 42  # XGBoost seed
EVENT_ID_CANDIDATES = ["original_event_id", "event_id"]  # SPADL column that links back to the StatsBomb event uuid

# The 12 summary features computed from the freeze frame. StatsBomb pitch is 120 x 80, goal centre (120, 40).
FRAME_FEATURE_NAMES = [
    "ff_n_visible", "ff_n_teammates_visible", "ff_n_opponents_visible",
    "ff_dist_nearest_opponent", "ff_n_opp_within_5", "ff_n_opp_within_10",
    "ff_n_tm_within_10", "ff_n_opp_ahead", "ff_n_opp_ahead_corridor",
    "ff_opp_keeper_visible", "ff_dist_actor_opp_keeper", "ff_mean_opp_x",
]


def frame_summary(players):
    """Summary features of one freeze frame, or None when the frame is unusable (< 2 players or not exactly one actor)."""
    if len(players) < 2:  # a frame needs at least two players
        return None
    xy = np.array([p.get("location", [np.nan, np.nan])[:2] for p in players], dtype=float)  # positions, one row per player
    tm = np.array([bool(p.get("teammate", False)) for p in players])  # True for the actor's team mates
    actor = np.array([bool(p.get("actor", False)) for p in players])  # True for the ball carrier
    keeper = np.array([bool(p.get("keeper", False)) for p in players])  # True for goalkeepers
    if actor.sum() != 1:  # exactly one actor is required
        return None
    a_xy = xy[actor][0]  # actor position
    opp = ~tm  # opponents
    tm_other = tm & ~actor  # team mates other than the actor
    d = np.sqrt(((xy - a_xy) ** 2).sum(axis=1))  # distance of every player to the actor
    n_opp = int(opp.sum())  # number of visible opponents
    out = {
        "ff_n_visible": float(len(players)),  # everyone visible
        "ff_n_teammates_visible": float(tm_other.sum()),  # team mates besides the actor
        "ff_n_opponents_visible": float(n_opp),  # opponents
        "ff_dist_nearest_opponent": float(d[opp].min()) if n_opp else np.nan,  # closest opponent
        "ff_n_opp_within_5": float((d[opp] <= 5).sum()),  # opponents within 5 units
        "ff_n_opp_within_10": float((d[opp] <= 10).sum()),  # opponents within 10 units
        "ff_n_tm_within_10": float((d[tm_other] <= 10).sum()),  # team mates within 10 units
        "ff_n_opp_ahead": float((xy[opp, 0] > a_xy[0]).sum()),  # opponents closer to the goal line than the actor
        "ff_n_opp_ahead_corridor": float(((xy[opp, 0] > a_xy[0]) & (np.abs(xy[opp, 1] - a_xy[1]) <= 10)).sum()),  # ... and within 10 units sideways
        "ff_mean_opp_x": float(xy[opp, 0].mean()) if n_opp else np.nan,  # average opponent depth
    }
    opp_keeper = opp & keeper  # opposing goalkeeper
    out["ff_opp_keeper_visible"] = float(opp_keeper.any())  # 1.0 when visible
    out["ff_dist_actor_opp_keeper"] = float(d[opp_keeper].min()) if opp_keeper.any() else np.nan  # distance to that keeper
    return out


def load_frame_features(game_id, wanted_uuids):
    """event_uuid -> summary dict for usable frames, only for the uuids that matter (later usable entries overwrite earlier ones, as before)."""
    with open(THREE_SIXTY_DIR / f"{int(game_id)}.json", "r", encoding="utf-8") as f:  # open the 360 file
        raw = json.load(f)  # parse it
    result = {}  # uuid -> summary
    for entry in raw:  # one entry per event
        uuid = entry.get("event_uuid")  # which event
        if uuid not in wanted_uuids:  # this event never became an action, so its summary would never be used
            continue
        s = frame_summary(entry.get("freeze_frame", []))  # summarise
        if s is not None:  # only usable frames are stored
            result[uuid] = s  # a later entry for the same uuid replaces an earlier one
    return result  # lookup table


def find_event_id_column(actions):  # name of the SPADL column that holds the event uuid
    for c in EVENT_ID_CANDIDATES:  # preferred order
        if c in actions.columns:
            return c  # use the first candidate that exists
    raise KeyError(f"No event id column among {EVENT_ID_CANDIDATES}: {list(actions.columns)}")  # stop


def build_match_table(loader, game_id, home_team_id):
    """A (VAEP) and B (360 summary) features for the actions of one match that have a usable frame."""
    actions = cs.load_actions(loader, game_id, home_team_id)  # SPADL actions with action_index
    A = cs.compute_vaep_features(actions, home_team_id)  # 14 feature blocks, float32
    A["game_id"] = int(game_id)  # key 1
    A["action_index"] = actions["action_index"].values  # key 2
    ev_col = find_event_id_column(actions)  # link column
    uuids = actions[ev_col].values  # event uuid of every action
    ff = load_frame_features(game_id, set(uuids.tolist()))  # summaries for those events
    rows = []  # one dict per usable action
    for ai, uuid in zip(actions["action_index"].values, uuids):  # in action order
        s = ff.get(uuid)  # summary or None
        if s is not None:
            r = dict(s)  # copy so the shared dict is not modified
            r["action_index"] = int(ai)  # join key
            rows.append(r)
    B = pd.DataFrame(rows, columns=FRAME_FEATURE_NAMES + ["action_index"])  # B table
    B[FRAME_FEATURE_NAMES] = B[FRAME_FEATURE_NAMES].astype(np.float32)  # half the memory
    return A.merge(B, on="action_index", how="inner")  # keep only actions with a usable frame (same rows that get a graph)


def build_split_table(loader, home, ids, target, y_col, label):
    """Feature table for all matches of a split, joined with the verified target."""
    print_header(f"BUILD A + B FEATURES - {label}")  # title
    parts, failed = [], []  # tables and failures
    t0 = time.time()  # timer
    for n, gid in enumerate(sorted(ids), start=1):  # one match at a time
        try:  # one bad match must not stop the run
            if gid not in home:  # no metadata
                failed.append((gid, "no metadata"))
                continue  # skip this match, it has no metadata
            parts.append(build_match_table(loader, gid, int(home[gid])))  # build and keep
        except Exception as ex:  # report and go on
            failed.append((gid, f"{type(ex).__name__}: {ex}"))
            print(f"  [FAILED - match skipped] {gid}: {ex}")  # tell the user which match failed and why
        if n % 20 == 0 or n == len(ids):  # progress line
            print(f"  [{n}/{len(ids)}] {(time.time()-t0)/60:.1f} min")
    table = pd.concat(parts, ignore_index=True)  # one big table
    del parts
    gc.collect()  # release the per-match tables
    merged = table.merge(target[["game_id", "action_index", y_col]], on=["game_id", "action_index"], how="inner")  # attach the label
    print(f"{label}: {len(table):,} rows with a usable frame; {len(merged):,} after joining the verified target "
          f"({len(table)-len(merged):,} end-of-match rows dropped).")  # report
    if failed:
        print(f"  Matches that failed: {failed}")  # list the matches that failed
    return merged, failed  # hand back the table and the failures


def fit_and_score(name, X_tr, y_tr, X_va, y_va, keys_va, out_pred_file, importance_file=None):
    """Train one XGBoost, score the validation set, save predictions (and importances)."""
    print_header(f"TRAIN XGBOOST - {name}")  # title
    n_pos = int(y_tr.sum())  # positives
    n_neg = int(len(y_tr) - n_pos)  # negatives
    spw = n_neg / n_pos  # class weight
    print(f"features: {X_tr.shape[1]}  train rows: {len(X_tr):,}  positives: {n_pos:,}  scale_pos_weight: {spw:.4f}")  # settings
    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, scale_pos_weight=spw,
                          eval_metric="logloss", random_state=RANDOM_STATE, tree_method="hist")  # same settings as Step 13
    t0 = time.time()  # timer
    model.fit(X_tr, y_tr)  # train
    print(f"fit time: {(time.time()-t0)/60:.1f} min")
    p = model.predict_proba(X_va)[:, 1]  # probability of a goal
    auc = roc_auc_score(y_va, p)  # ranking quality
    pr = average_precision_score(y_va, p)  # precision-recall area
    br = brier_score_loss(y_va, p)  # calibration
    print(f"VALIDATION AUC-ROC = {auc:.4f}   PR-AUC = {pr:.4f}   Brier = {br:.4f}")
    out = keys_va.copy()  # keys
    out["true_target"] = y_va.values  # labels
    out["predicted_probability"] = p  # scores
    out.to_csv(out_pred_file, index=False)  # save
    print(f"saved predictions: {out_pred_file}")
    if importance_file is not None:  # only the A+B model writes importances
        pd.DataFrame({"feature": X_tr.columns, "importance": model.feature_importances_}
                     ).sort_values("importance", ascending=False).to_csv(importance_file, index=False)
    return auc, pr, br, len(X_tr), n_pos  # hand back the four numbers and the training size


def main():  # runs the whole step from start to finish
    common.verify_environment()  # optional interpreter check
    ap = argparse.ArgumentParser()  # options
    ap.add_argument("--horizon", type=int, default=10, choices=[5, 10])  # label horizon
    args = ap.parse_args()  # read them
    H = args.horizon  # horizon
    target_file, y_col = TARGET_FILES[H]  # inputs for this horizon
    tag = f"N{H}"  # file-name tag

    print_header(f"STEP 27 - PARITY CELL 3 (XGBoost on A+B), horizon N = {H}")  # title
    for p in [target_file, FULL_SPLIT_FILE]:  # inputs must exist
        if not p.exists():
            raise FileNotFoundError(p)  # stop if a needed input file is missing

    target = pd.read_csv(target_file)  # verified target
    target["game_id"] = target["game_id"].astype(int)
    target["action_index"] = target["action_index"].astype(int)  # make the action index an integer
    target[y_col] = target[y_col].astype(bool)  # make the label a boolean

    split = pd.read_csv(FULL_SPLIT_FILE)  # split table
    split["split"] = split["split"].astype(str).str.lower().str.strip()  # normalise labels
    train_ids = split[split["split"] == "train"]["game_id"].astype(int).tolist()  # training matches
    val_ids = split[split["split"] == "validation"]["game_id"].astype(int).tolist()  # validation matches

    loader = cs.create_loader()  # StatsBomb reader
    metadata = cs.find_match_metadata(loader, train_ids + val_ids)  # home team ids
    home = dict(zip(metadata["game_id"], metadata["home_team_id"]))  # game_id -> home team id

    train_tab, train_failed = build_split_table(loader, home, train_ids, target, y_col, "TRAINING")  # training features
    val_tab, val_failed = build_split_table(loader, home, val_ids, target, y_col, "VALIDATION")  # validation features

    A_cols = [c for c in train_tab.columns if c not in FRAME_FEATURE_NAMES + ["game_id", "action_index", y_col]]  # VAEP columns only
    AB_cols = A_cols + FRAME_FEATURE_NAMES  # VAEP plus 360 summary
    y_tr = train_tab[y_col].astype(int)  # training labels
    y_va = val_tab[y_col].astype(int)  # validation labels
    keys_va = val_tab[["game_id", "action_index"]]  # validation keys

    for nm, tab in [("train", train_tab), ("validation", val_tab)]:  # feature files for the hybrid GAT (script 17)
        tab[["game_id", "action_index", y_col] + A_cols].to_pickle(OUTPUT_DIR / f"27_vaep_features_A_{nm}_{tag}.pkl")  # A columns with keys and label

    res = {}  # name -> metrics
    res["control_A_restricted"] = fit_and_score(
        "CONTROL: A only, rows with a usable frame", train_tab[A_cols], y_tr, val_tab[A_cols], y_va,
        keys_va, OUTPUT_DIR / f"27_xgb_A_restricted_{tag}_validation_predictions.csv")  # same rows, A only
    res["cell3_A_plus_B"] = fit_and_score(
        "CELL 3: A + B", train_tab[AB_cols], y_tr, val_tab[AB_cols], y_va, keys_va,
        OUTPUT_DIR / f"27_xgb_A_plus_B_{tag}_validation_predictions.csv",
        OUTPUT_DIR / f"27_xgb_A_plus_B_{tag}_feature_importance.csv")  # A plus B

    rows = []  # summary rows
    for k, (auc, pr, br, n_tr, n_pos) in res.items():
        rows.append({"model": k, "horizon": H, "validation_auc_roc": round(auc, 4),  # store one result row (continues on the next line)
                     "validation_pr_auc": round(pr, 4), "validation_brier": round(br, 4),
                     "train_rows": n_tr, "train_positive": n_pos,
                     "validation_rows": len(val_tab), "random_state": RANDOM_STATE,
                     "n_train_failed": len(train_failed), "n_val_failed": len(val_failed)})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / f"27_parity_xgboost_{tag}_summary.csv", index=False)  # save

    print_header("STEP 27 COMPLETE")  # done
    print("Next: run 17_parity_gat_plus_vaep_features.py (uses the 27_vaep_features_A_*.pkl files).")  # pointer
    return 0  # success


if __name__ == "__main__":  # only when run directly
    common.run_main(main, "STEP 27")  # shared error handling
