# ==============================================================================
# common_spadl.py  -  StatsBomb -> SPADL actions, the N-action target, and VAEP features
# ==============================================================================
# Steps 09, 11, 12-like scripts and the label-horizon sweep all converted events to SPADL
# actions and walked a window over the next N actions with nested Python loops.
# That walk is now done with cumulative sums, which is a few hundred times faster and gives
# exactly the same labels (the test in tests/ compares both on real StatsBomb matches).
# ==============================================================================

import numpy as np  # arrays
import pandas as pd  # tables
from socceraction.data.statsbomb import StatsBombLoader  # reads the local StatsBomb json files
from socceraction.spadl import statsbomb as statsbomb_spadl  # events -> SPADL actions
import socceraction.vaep.features as fs  # VAEP feature functions
from socceraction import spadl  # SPADL constants and helpers

from common import DATA_DIR, print_header  # shared paths and header printer

NB_PREV_ACTIONS = 3  # game state = the current action plus the two before it (what Step 13 used)

# The 14 feature functions, in the exact order Step 13 used. Order matters because it fixes the column order.
VAEP_FEATURE_FUNCTIONS = [
    fs.actiontype_onehot, fs.result_onehot, fs.actiontype_result_onehot, fs.bodypart_onehot,  # what happened and how
    fs.time, fs.startlocation, fs.endlocation, fs.startpolar, fs.endpolar,  # when and where
    fs.movement, fs.team, fs.time_delta, fs.space_delta, fs.goalscore,  # movement, possession and score context
]


def create_loader():
    """StatsBomb loader that reads the json files under <project>/data."""
    print_header("CREATE STATSBOMB LOADER")  # title
    loader = StatsBombLoader(getter="local", root=str(DATA_DIR))  # local files, no network
    print(f"[PASS] StatsBomb root: {DATA_DIR}")  # confirm
    return loader  # ready to use


def get_spadl_ids():
    """Numeric ids of the 'shot' action type and the 'success' / 'owngoal' results."""
    print_header("GET SPADL IDENTIFIERS")  # title
    actiontypes, results = spadl.actiontypes_df(), spadl.results_df()  # lookup tables
    shot_type_id = int(actiontypes[actiontypes["type_name"] == "shot"].iloc[0]["type_id"])  # id of 'shot'
    success_result_id = int(results[results["result_name"] == "success"].iloc[0]["result_id"])  # id of 'success'
    owngoal_result_id = int(results[results["result_name"] == "owngoal"].iloc[0]["result_id"])  # id of 'owngoal'
    print(f"Shot type ID: {shot_type_id} | Success result ID: {success_result_id} | Own-goal result ID: {owngoal_result_id}")  # report
    return shot_type_id, success_result_id, owngoal_result_id  # three ints


def find_match_metadata(loader, match_ids):
    """Home/away team ids for the requested matches, found by scanning every competition-season."""
    print_header("FIND MATCH METADATA")  # title
    remaining = set(int(x) for x in match_ids)  # matches still to be found
    parts = []  # collected tables
    for _, row in loader.competitions().iterrows():  # every competition-season on disk
        try:  # a season folder may be missing
            games = loader.games(competition_id=int(row["competition_id"]), season_id=int(row["season_id"]))  # its matches
        except Exception:  # skip unreadable seasons
            continue
        if games is None or games.empty:  # nothing there
            continue
        games = games.copy()  # avoid touching the loader's table
        games["game_id"] = pd.to_numeric(games["game_id"], errors="coerce")  # numeric ids
        games = games[games["game_id"].isin(remaining)]  # keep only requested matches
        if games.empty:  # none of ours in this season
            continue
        parts.append(games[["game_id", "home_team_id", "away_team_id"]])  # keep the three columns
        remaining -= set(games["game_id"].astype(int).tolist())  # tick them off
        if not remaining:  # found everything
            break
    if remaining:  # something is missing
        raise ValueError(f"Missing metadata for matches: {sorted(remaining)}")  # stop
    metadata = pd.concat(parts, ignore_index=True)  # one table
    metadata["game_id"] = metadata["game_id"].astype(int)  # integer ids
    metadata = metadata.drop_duplicates(subset=["game_id"])  # one row per match
    print(f"[PASS] Metadata found for {len(metadata):,} matches.")  # report
    return metadata  # game_id, home_team_id, away_team_id


def find_match_metadata_lenient(loader, match_ids):
    """Like find_match_metadata but quiet and forgiving: matches that cannot be found are simply absent from the result.
    Returns game_id and home_team_id only. Used by the visual-overview scripts, which skip missing matches."""
    remaining = set(int(x) for x in match_ids)  # matches still to be found
    parts = []  # collected tables
    for _, row in loader.competitions().iterrows():  # every competition-season on disk
        try:  # a season folder may be missing
            games = loader.games(competition_id=int(row["competition_id"]), season_id=int(row["season_id"]))  # its matches
        except Exception:  # skip unreadable seasons
            continue
        if games is None or games.empty:  # nothing there
            continue
        games = games.copy()  # avoid touching the loader's table
        games["game_id"] = pd.to_numeric(games["game_id"], errors="coerce")  # numeric ids
        games = games[games["game_id"].isin(remaining)]  # keep only requested matches
        if games.empty:  # none of ours in this season
            continue
        parts.append(games[["game_id", "home_team_id"]])  # keep the two columns
        remaining -= set(games["game_id"].astype(int).tolist())  # tick them off
        if not remaining:  # found everything
            break
    if not parts:  # nothing found at all
        return pd.DataFrame(columns=["game_id", "home_team_id"])  # empty table with the right columns
    metadata = pd.concat(parts, ignore_index=True)  # one table
    metadata["game_id"] = metadata["game_id"].astype(int)  # integer ids
    return metadata.drop_duplicates(subset=["game_id"])  # one row per match


def load_actions(loader, game_id, home_team_id):
    """Events -> SPADL actions with an action_index column and readable names."""
    events = loader.events(game_id=int(game_id))  # raw events of the match
    if events is None or events.empty:  # nothing to convert
        raise ValueError(f"No events for match {game_id}")  # stop
    actions = statsbomb_spadl.convert_to_actions(events, int(home_team_id)).reset_index(drop=True)  # SPADL actions in order
    actions["action_index"] = range(len(actions))  # 0..n-1 position inside the match
    return spadl.add_names(actions)  # adds type_name, result_name, bodypart_name


def window_labels(type_id, result_id, team_id, horizon, shot_type_id, success_result_id, owngoal_result_id):
    """Label every action with what happens in the NEXT `horizon` actions.

    Returns three boolean arrays of length n (all False where the action is not eligible) and the eligible mask:
        eligible                  - a full `horizon`-action future window exists (action i needs i + horizon <= n - 1)
        future_team_goal          - the same team scores a successful shot inside the window
        future_opponent_own_goal  - the opponent produces an own goal inside the window
    This is the old double loop rewritten with cumulative sums: counting events in a window is a difference of two
    running totals, so no inner loop is needed.
    """
    n = len(type_id)  # actions in the match
    eligible = np.zeros(n, dtype=bool)  # which actions have a full future window
    future_team_goal = np.zeros(n, dtype=bool)  # same-team goal in window
    future_opp_own = np.zeros(n, dtype=bool)  # opponent own goal in window
    if n <= horizon:  # not even the first action has a full window
        return eligible, future_team_goal, future_opp_own  # all False
    eligible[: n - horizon] = True  # actions 0 .. n-horizon-1 are eligible
    is_shot = type_id == shot_type_id  # shot actions
    is_goal = is_shot & (result_id == success_result_id)  # successful shots
    is_own = is_shot & (result_id == owngoal_result_id)  # own goals
    idx_all = np.arange(n - horizon)  # indices of eligible actions
    for team in np.unique(team_id[idx_all]):  # handle one team at a time (normally two)
        mine = team_id == team  # actions by this team
        cum_goal = np.concatenate(([0], np.cumsum(is_goal & mine)))  # running count of this team's goals
        cum_own = np.concatenate(([0], np.cumsum(is_own & ~mine)))  # running count of own goals by the other side
        idx = idx_all[mine[idx_all]]  # eligible actions taken by this team
        future_team_goal[idx] = (cum_goal[idx + horizon + 1] - cum_goal[idx + 1]) > 0  # goals among actions i+1 .. i+horizon
        future_opp_own[idx] = (cum_own[idx + horizon + 1] - cum_own[idx + 1]) > 0  # opponent own goals in the same window
    return eligible, future_team_goal, future_opp_own  # three boolean arrays plus the mask


def compute_vaep_features(actions, home_team_id):
    """VAEP feature table for every action of one match, numeric columns as float32 (as in Step 13)."""
    gamestates = fs.gamestates(actions, NB_PREV_ACTIONS)  # current action plus the previous ones
    gamestates = fs.play_left_to_right(gamestates, int(home_team_id))  # mirror so both teams attack left to right
    features = pd.concat([fn(gamestates) for fn in VAEP_FEATURE_FUNCTIONS], axis=1)  # glue the 14 blocks together
    numeric_cols = features.select_dtypes(include=[np.number]).columns  # numeric columns only
    features[numeric_cols] = features[numeric_cols].astype(np.float32)  # half the memory of float64
    return features  # one row per action


def build_features_for_match(loader, game_id, home_team_id):
    """VAEP features for one match plus the two join keys (game_id, action_index) as the last two columns."""
    actions = load_actions(loader, game_id, home_team_id)  # events -> SPADL actions
    features = compute_vaep_features(actions, home_team_id)  # 14 feature blocks, float32
    features["game_id"] = int(game_id)  # key 1
    features["action_index"] = actions["action_index"].values  # key 2
    return features  # one row per action
