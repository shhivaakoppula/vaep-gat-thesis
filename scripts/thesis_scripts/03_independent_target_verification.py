# ==============================================================================
# STEP 08A (file 03_independent_target_verification.py) - INDEPENDENT RQ1 TARGET VERIFICATION
# ==============================================================================
# Question behind the target: after the current action, does the team in possession score within the next 10
# standardized football actions? The current action is NOT included, the horizon is 10 actions.
# This script rebuilds that target a second time, straight from the ordered SPADL actions, and compares it with the
# target saved in Step 07C (rq1_literal_next10_target_34_matches.csv). It checks row counts, targets, teams, the two
# future flags, match-level totals and the frozen split, and writes a detailed audit CSV and a pass/fail summary.
# On purpose the window here is built with a different technique (a sliding window over padded arrays) than the
# cumulative-sum version the build scripts use, so the check stays independent. It is also much faster than the
# original double Python loop and gives the same flags.
# NOTE: this script scores EVERY action (also the last 10 of each match, which have an incomplete window), which is
# why the row-count check is expected to show 340 extra observations. Scripts 04 and 05 investigate exactly that.
# ==============================================================================

import sys  # python version text

import numpy as np  # arrays and sliding windows
import pandas as pd  # tables
from numpy.lib.stride_tricks import sliding_window_view  # windows over arrays without copying

import common  # shared helpers
from common import DATA_DIR, OUTPUT_DIR, print_header, require_file  # shared paths and helpers

try:  # the original reported import problems with a clear banner
    from socceraction.data.statsbomb import StatsBombLoader  # reads StatsBomb files
    from socceraction.spadl import statsbomb as statsbomb_spadl  # events -> SPADL actions
    from socceraction import spadl  # SPADL lookups
except Exception as exc:  # broken socceraction install
    print()
    print("=" * 80)  # print a divider line
    print("SOCCERACTION IMPORT FAILED")  # tell the user: SOCCERACTION IMPORT FAILED
    print("=" * 80)  # print a divider line
    print()  # print a blank line
    print(f"Error type: {type(exc).__name__}")  # tell the user: Error type (plus its value)
    print(f"Error message: {exc}")  # tell the user: Error message (plus its value)
    print()  # print a blank line
    print("This script must use the dedicated .vaep-venv environment.")  # tell the user: This script must use the dedicated .vaep-venv environment
    raise  # pass the error on to the shared handler

TARGET_FILE = OUTPUT_DIR / "rq1_literal_next10_target_34_matches.csv"  # Step 07C target
SPLIT_FILE = OUTPUT_DIR / "final_common_match_split.csv"  # frozen 24/5/5 split
AUDIT_FILE = OUTPUT_DIR / "08A_independent_target_verification.csv"  # detailed audit
SUMMARY_FILE = OUTPUT_DIR / "08A_verification_summary.csv"  # pass/fail table
NEXT_N_ACTIONS = 10  # future horizon in actions


def verify_environment_packages():  # prints the versions of the packages this step relies on
    print()  # print a blank line
    print("=" * 80)  # print a divider line
    print("ENVIRONMENT / PACKAGE VERIFICATION")  # tell the user: ENVIRONMENT / PACKAGE VERIFICATION
    print("=" * 80)  # print a divider line
    import numpy, scipy, pandas, pandera, socceraction, sklearn  # imported here, as in the original, so a missing one is reported at this point
    packages = [("Python", sys.version.split()[0]), ("NumPy", numpy.__version__), ("SciPy", scipy.__version__),
                ("pandas", pandas.__version__), ("Pandera", pandera.__version__), ("socceraction", socceraction.__version__),
                ("scikit-learn", sklearn.__version__)]  # name and version of each package
    for name, version in packages:
        print(f"{name:15s}: {version}")  # print the value
    print()  # print a blank line
    print("[PASS] Required package imports completed.")  # tell the user: [PASS] Required package imports completed


def get_spadl_ids():  # looks up the SPADL numbers for 'shot', 'success' and 'owngoal'
    print_header("SPADL IDENTIFIER VERIFICATION")  # title
    actiontypes, results = spadl.actiontypes_df(), spadl.results_df()  # lookup tables
    shot_rows = actiontypes[actiontypes["type_name"] == "shot"]
    if shot_rows.empty:  # stop if the library has no 'shot' action type
        raise ValueError("Could not find SPADL action type 'shot'.")  # stop with the error: Could not find SPADL action type 'shot'
    shot_type_id = int(shot_rows.iloc[0]["type_id"])
    success_rows = results[results["result_name"] == "success"]  # rows of the results table named 'success'
    if success_rows.empty:  # stop if the library has no 'success' result
        raise ValueError("Could not find SPADL result 'success'.")  # stop with the error: Could not find SPADL result 'success'
    success_result_id = int(success_rows.iloc[0]["result_id"])
    owngoal_rows = results[results["result_name"] == "owngoal"]  # rows of the results table named 'owngoal'
    if owngoal_rows.empty:  # stop if the library has no 'owngoal' result
        raise ValueError("Could not find SPADL result 'owngoal'.")  # stop with the error: Could not find SPADL result 'owngoal'
    owngoal_result_id = int(owngoal_rows.iloc[0]["result_id"])
    print(f"[PASS] Shot type ID: {shot_type_id}")  # tell the user: [PASS] Shot type ID (plus its value)
    print(f"[PASS] Success result ID: {success_result_id}")  # tell the user: [PASS] Success result ID (plus its value)
    print(f"[PASS] Own-goal result ID: {owngoal_result_id}")  # tell the user: [PASS] Own-goal result ID (plus its value)
    return shot_type_id, success_result_id, owngoal_result_id  # hand back shot_type_id, success_result_id, owngoal_result_id


def load_frozen_split():  # reads the frozen split and checks it
    print_header("LOAD FROZEN 24 / 5 / 5 MATCH SPLIT")  # title
    split_df = pd.read_csv(SPLIT_FILE)  # split table
    missing_columns = {"game_id", "split"} - set(split_df.columns)
    if missing_columns:  # stop if a needed column is missing
        raise ValueError(f"Frozen split is missing columns: {sorted(missing_columns)}")  # stop with the error: Frozen split is missing columns
    split_df["game_id"] = pd.to_numeric(split_df["game_id"], errors="raise").astype(int)  # integer ids
    split_df["split"] = split_df["split"].astype(str).str.lower().str.strip()  # normalised labels
    unexpected_splits = set(split_df["split"]) - {"train", "validation", "test"}
    if unexpected_splits:  # stop if the split has a label other than train, validation or test
        raise ValueError(f"Unexpected split values: {sorted(unexpected_splits)}")  # stop with the error: Unexpected split values
    duplicate_matches = split_df["game_id"].duplicated().sum()
    if duplicate_matches != 0:  # stop if a match appears twice
        raise ValueError(f"Frozen split contains duplicate game IDs: {duplicate_matches}")  # stop with the error: Frozen split contains duplicate game IDs
    counts = split_df["split"].value_counts().to_dict()
    print(f"Train matches: {counts.get('train', 0)}")  # tell the user: Train matches (plus its value)
    print(f"Validation matches: {counts.get('validation', 0)}")  # tell the user: Validation matches (plus its value)
    print(f"Test matches: {counts.get('test', 0)}")  # tell the user: Test matches (plus its value)
    if counts.get("train", 0) != 24 or counts.get("validation", 0) != 5 or counts.get("test", 0) != 5:
        raise ValueError("Frozen split is not exactly 24 train / 5 validation / 5 test.")  # stop with the error: Frozen split is not exactly 24 train / 5 validation / 5 test
    if len(split_df) != 34:
        raise ValueError("Frozen split does not contain exactly 34 matches.")  # stop with the error: Frozen split does not contain exactly 34 matches
    print("[PASS] Frozen split contains exactly 34 matches.")  # tell the user: [PASS] Frozen split contains exactly 34 matches
    return split_df  # hand back split_df


def load_saved_target():  # reads the saved Step 07C target and checks its structure
    print_header("LOAD STEP 07C LITERAL NEXT-10 TARGET")  # title
    target = pd.read_csv(TARGET_FILE)  # saved target
    expected_columns = ["game_id", "action_index", "period_id", "time_seconds", "team_id",
                        "target_score_next10_literal", "future_team_goal", "future_opponent_own_goal"]  # exact schema
    missing_columns = set(expected_columns) - set(target.columns)
    unexpected_columns = set(target.columns) - set(expected_columns)  # columns present that were not expected
    if missing_columns:  # stop if a needed column is missing
        raise ValueError(f"Step 07C target is missing columns: {sorted(missing_columns)}")  # stop with the error: Step 07C target is missing columns
    if unexpected_columns:
        raise ValueError(f"Step 07C target contains unexpected columns: {sorted(unexpected_columns)}")  # stop with the error: Step 07C target contains unexpected columns
    target = target[expected_columns].copy()  # fixed column order
    for column in ["game_id", "action_index", "period_id", "time_seconds", "team_id"]:  # numeric columns
        target[column] = pd.to_numeric(target[column], errors="raise")
    for column in ["target_score_next10_literal", "future_team_goal", "future_opponent_own_goal"]:  # boolean columns
        target[column] = target[column].astype(bool)
    missing_count = target.isna().sum().sum()  # number of missing values in the whole table
    if missing_count != 0:  # stop if any value is missing
        raise ValueError(f"Step 07C target contains {missing_count} missing values.")  # stop with the error: Step 07C target contains missing values
    duplicate_count = target[["game_id", "action_index"]].duplicated().sum()
    if duplicate_count != 0:  # stop if any (game_id, action_index) pair appears twice
        raise ValueError("Step 07C target contains duplicate (game_id, action_index) observations.")  # stop with the error: Step 07C target contains duplicate (game_id, action_index)...
    print(f"Rows: {len(target):,}")  # tell the user: Rows (plus its value)
    print(f"Matches: {target['game_id'].nunique():,}")  # tell the user: Matches (plus its value)
    print(f"Positive targets: {target['target_score_next10_literal'].sum():,}")  # tell the user: Positive targets (plus its value)
    print(f"Negative targets: {(~target['target_score_next10_literal']).sum():,}")  # tell the user: Negative targets (plus its value)
    print("[PASS] Step 07C target structure verified.")  # tell the user: [PASS] Step 07C target structure verified
    return target  # hand back target


def create_loader():  # creates the local StatsBomb reader after checking its folders
    print_header("CREATE LOCAL STATSBOMB LOADER")  # title
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"StatsBomb data root does not exist:\n{DATA_DIR}")  # stop with the error: StatsBomb data root does not exist
    for directory in [DATA_DIR / "events", DATA_DIR / "lineups", DATA_DIR / "matches"]:  # folders the loader needs
        if not directory.exists():
            raise FileNotFoundError(f"Required StatsBomb directory does not exist:\n{directory}")  # stop with the error: Required StatsBomb directory does not exist
    loader = StatsBombLoader(getter="local", root=str(DATA_DIR))  # local files only
    print(f"[PASS] Local StatsBomb root: {DATA_DIR}")
    return loader  # hand back loader


def build_game_metadata(loader, match_ids):  # finds the home team of every requested match
    print_header("FIND HOME TEAM IDs FOR 34 MATCHES")  # title
    remaining = set(int(game_id) for game_id in match_ids)  # still to find
    parts = []  # collected tables
    competitions = loader.competitions().copy()
    if competitions.empty:  # stop if the loader found no competitions
        raise ValueError("StatsBombLoader returned no competitions.")  # stop with the error: StatsBombLoader returned no competitions
    for _, row in competitions.iterrows():  # every competition-season
        competition_id, season_id = int(row["competition_id"]), int(row["season_id"])
        try:  # try to read the matches of one competition-season
            games = loader.games(competition_id=competition_id, season_id=season_id)  # matches of this competition-season
        except Exception:
            continue  # skip to the next item
        if games is None or games.empty:
            continue  # skip to the next item
        games = games.copy()
        games["game_id"] = pd.to_numeric(games["game_id"], errors="coerce")  # make the match id numeric
        games = games[games["game_id"].isin(remaining)]  # keep only the matches we still look for
        if games.empty:  # skip seasons that contain none of our matches
            continue  # skip to the next item
        parts.append(games[["game_id", "competition_id", "season_id", "home_team_id", "away_team_id"]])
        remaining -= set(games["game_id"].astype(int).tolist())  # tick off the matches found in this season
        if not remaining:  # stop searching once every match is found
            break  # leave the loop
    if remaining:
        raise ValueError(f"Could not find match metadata for game IDs: {sorted(remaining)}")  # stop with the error: Could not find match metadata for game IDs
    metadata = pd.concat(parts, ignore_index=True)
    metadata["game_id"] = metadata["game_id"].astype(int)  # make the match id an integer
    metadata = metadata.drop_duplicates(subset=["game_id"]).reset_index(drop=True)  # one row per match, with a clean index
    if len(metadata) != len(match_ids):  # stop if some match has no unique metadata
        raise ValueError("Could not construct unique metadata for all requested matches.")  # stop with the error: Could not construct unique metadata for all requested matches
    print(f"[PASS] Match metadata found for {len(metadata):,} matches.")  # tell the user: [PASS] Match metadata found for matches (plus its value)
    return metadata  # hand back metadata


def window_flags(team, is_goal_shot, is_own_goal_shot, horizon):
    """For every action i look at actions i+1 .. i+horizon (cut off at the end of the match).
    Returns (same-team successful shot in window, opponent own goal in window) as two boolean arrays.
    Built with a sliding window over arrays padded with False, so incomplete windows at the end of the match are
    simply shorter (padding can never count)."""
    n = len(team)  # actions in the match
    pad = np.zeros(horizon, dtype=bool)  # False padding that stands for "no action here"
    goal_pad = np.concatenate([is_goal_shot, pad])  # successful shots, padded
    own_pad = np.concatenate([is_own_goal_shot, pad])  # own goals, padded
    team_pad = np.concatenate([team, np.full(horizon, -1, dtype=team.dtype)])  # team ids, padded with an impossible id
    goal_win = sliding_window_view(goal_pad, horizon)[1:n + 1]  # row i = padded positions i+1 .. i+horizon
    own_win = sliding_window_view(own_pad, horizon)[1:n + 1]  # same for own goals
    team_win = sliding_window_view(team_pad, horizon)[1:n + 1]  # same for team ids
    future_team_goal = (goal_win & (team_win == team[:, None])).any(axis=1)  # a successful shot by the SAME team
    future_opponent_own_goal = (own_win & (team_win != team[:, None])).any(axis=1)  # an own goal by the OTHER side
    return future_team_goal, future_opponent_own_goal


def reconstruct_match_target(loader, game_id, home_team_id, shot_type_id, success_result_id, owngoal_result_id):  # rebuilds the target for one match from its SPADL actions
    print(f"Processing match {game_id}...")  # tell the user: Processing match (plus its value)
    events = loader.events(game_id=int(game_id))  # raw events
    if events is None or events.empty:
        raise ValueError(f"No events returned for match {game_id}.")  # stop with the error: No events returned for match
    actions = statsbomb_spadl.convert_to_actions(events, int(home_team_id))  # SPADL actions
    actions = actions.reset_index(drop=True).copy()  # clean positional index
    if actions.empty:
        raise ValueError(f"No SPADL actions generated for match {game_id}.")  # stop with the error: No SPADL actions generated for match
    actions["action_index"] = range(len(actions))  # action_index = position in the SPADL action order
    current_team = actions["team_id"].to_numpy()  # team of each action
    action_type = actions["type_id"].to_numpy()  # type of each action
    result = actions["result_id"].to_numpy()  # result of each action
    n_actions = len(actions)
    is_goal_shot = (action_type == shot_type_id) & (result == success_result_id)  # a successful shot
    is_own_goal_shot = (action_type == shot_type_id) & (result == owngoal_result_id)  # an own goal
    future_team_goal, future_opponent_own_goal = window_flags(current_team, is_goal_shot, is_own_goal_shot, NEXT_N_ACTIONS)  # the current action itself is never inspected
    independent_target = future_team_goal | future_opponent_own_goal  # positive when either flag is set
    result_df = pd.DataFrame({
        "game_id": int(game_id),
        "action_index": actions["action_index"].astype(int),
        "period_id": actions["period_id"].astype(int),
        "time_seconds": pd.to_numeric(actions["time_seconds"], errors="raise"),
        "team_id": actions["team_id"].astype(int),
        "independent_target": independent_target,
        "independent_future_team_goal": future_team_goal,
        "independent_future_opponent_own_goal": future_opponent_own_goal,
    })
    print(f"  SPADL actions: {len(actions):,}")  # tell the user: SPADL actions (plus its value)
    print(f"  Positive targets: {int(independent_target.sum()):,}")  # tell the user: Positive targets (plus its value)
    return result_df  # hand back result_df


def compare_targets(saved_target, independent_target):  # compares the saved and the independent target
    print_header("TARGET COMPARISON")  # title
    merged = saved_target.copy().merge(independent_target.copy(), on=["game_id", "action_index"], how="outer",
                                       indicator=True, suffixes=("_saved", "_independent"))  # outer join shows rows on one side only
    saved_only = int((merged["_merge"] == "left_only").sum())
    independent_only = int((merged["_merge"] == "right_only").sum())  # count rows that exist only on the independent side
    common_rows = merged[merged["_merge"] == "both"].copy()  # rows present on both sides
    target_mismatches = int((common_rows["target_score_next10_literal"].astype(bool) != common_rows["independent_target"].astype(bool)).sum())
    team_mismatches = int((common_rows["team_id_saved"].astype(int) != common_rows["team_id_independent"].astype(int)).sum())  # count rows where the two teams differ
    future_team_goal_mismatches = int((common_rows["future_team_goal"].astype(bool) != common_rows["independent_future_team_goal"].astype(bool)).sum())  # count rows where the future team-goal flag differs
    future_opponent_own_goal_mismatches = int((common_rows["future_opponent_own_goal"].astype(bool) != common_rows["independent_future_opponent_own_goal"].astype(bool)).sum())  # count rows where the future own-goal flag differs
    saved_rows, independent_rows = len(saved_target), len(independent_target)  # row counts of the two tables
    print(f"Saved rows: {saved_rows:,}")  # tell the user: Saved rows (plus its value)
    print(f"Independent rows: {independent_rows:,}")  # tell the user: Independent rows (plus its value)
    print(f"Saved-only observations: {saved_only:,}")  # tell the user: Saved-only observations (plus its value)
    print(f"Independent-only observations: {independent_only:,}")  # tell the user: Independent-only observations (plus its value)
    print(f"Target mismatches: {target_mismatches:,}")  # tell the user: Target mismatches (plus its value)
    print(f"Team mismatches: {team_mismatches:,}")  # tell the user: Team mismatches (plus its value)
    print(f"Future team-goal mismatches: {future_team_goal_mismatches:,}")  # tell the user: Future team-goal mismatches (plus its value)
    print(f"Future opponent-own-goal mismatches: {future_opponent_own_goal_mismatches:,}")  # tell the user: Future opponent-own-goal mismatches (plus its value)
    return pd.DataFrame([{"saved_rows": saved_rows, "independent_rows": independent_rows, "saved_only": saved_only,  # hand back pd.DataFrame([{'saved_rows': saved_rows, 'independent_row...
                          "independent_only": independent_only, "target_mismatches": target_mismatches,
                          "team_mismatches": team_mismatches, "future_team_goal_mismatches": future_team_goal_mismatches,
                          "future_opponent_own_goal_mismatches": future_opponent_own_goal_mismatches}])


def reconcile_matches(saved_target, independent_target):  # compares row and positive counts match by match
    print_header("MATCH-LEVEL RECONCILIATION")  # title
    saved_summary = saved_target.groupby("game_id").agg(saved_rows=("action_index", "size"),
                                                        saved_positives=("target_score_next10_literal", "sum")).reset_index()
    independent_summary = independent_target.groupby("game_id").agg(independent_rows=("action_index", "size"),  # per-match counts of the independent target
                                                                    independent_positives=("independent_target", "sum")).reset_index()
    reconciliation = saved_summary.merge(independent_summary, on="game_id", how="outer")  # one row per match
    count_columns = ["saved_rows", "saved_positives", "independent_rows", "independent_positives"]
    reconciliation[count_columns] = reconciliation[count_columns].fillna(0)  # absent side counts as 0
    reconciliation[count_columns] = reconciliation[count_columns].astype(int)
    reconciliation["row_count_match"] = reconciliation["saved_rows"] == reconciliation["independent_rows"]  # row counts agree on both sides
    reconciliation["positive_count_match"] = reconciliation["saved_positives"] == reconciliation["independent_positives"]  # positive counts agree on both sides
    row_count_failures = (~reconciliation["row_count_match"]).sum()  # matches whose row counts disagree
    positive_count_failures = (~reconciliation["positive_count_match"]).sum()  # matches whose positive counts disagree
    print(f"Matches checked: {len(reconciliation):,}")  # tell the user: Matches checked (plus its value)
    print(f"Row-count failures: {row_count_failures:,}")  # tell the user: Row-count failures (plus its value)
    print(f"Positive-count failures: {positive_count_failures:,}")  # tell the user: Positive-count failures (plus its value)
    return reconciliation  # hand back reconciliation


def verify_split(independent_target, split_df):  # checks that every match has a frozen split assignment
    print_header("FROZEN SPLIT VERIFICATION")  # title
    target_matches = independent_target[["game_id"]].drop_duplicates()
    merged = target_matches.merge(split_df[["game_id", "split"]], on="game_id", how="left", indicator=True)  # target matches joined to their split
    missing = merged["_merge"] != "both"  # True where a match has no split
    print(f"Target matches: {len(merged):,}")  # tell the user: Target matches (plus its value)
    print(f"Missing split assignments: {missing.sum():,}")  # tell the user: Missing split assignments (plus its value)
    if missing.any():
        raise ValueError("Some target matches do not have a frozen split assignment.")  # stop with the error: Some target matches do not have a frozen split assignment
    print()  # print a blank line
    print(merged["split"].value_counts().sort_index())  # print the result of merged['split'].value_counts().sort_index()
    return merged  # hand back merged


def main():  # runs the whole step from start to finish
    common.verify_environment()  # optional interpreter check
    print_header("STEP 08A — INDEPENDENT RQ1 TARGET VERIFICATION")  # title
    print("Target:")
    print("After the current action, does the team in possession score within the next 10 standardized football actions?")  # tell the user: After the current action, does the team in possession score within...
    print()  # print a blank line
    print("Current action included in target: NO")  # tell the user: Current action included in target: NO
    print(f"Future horizon: {NEXT_N_ACTIONS} actions")  # tell the user: Future horizon: actions (plus its value)
    verify_environment_packages()  # versions
    require_file(TARGET_FILE, "Step 07C target")
    require_file(SPLIT_FILE, "Frozen split")  # stop if the split file is missing
    shot_type_id, success_result_id, owngoal_result_id = get_spadl_ids()  # SPADL ids of 'shot', 'success' and 'owngoal'
    split_df = load_frozen_split()  # the frozen split
    saved_target = load_saved_target()  # the saved Step 07C target
    loader = create_loader()  # the local StatsBomb reader
    match_ids = sorted(split_df["game_id"].unique().tolist())  # matches to rebuild, in order
    print()  # print a blank line
    print(f"Matches to process: {len(match_ids):,}")  # tell the user: Matches to process (plus its value)
    metadata = build_game_metadata(loader, match_ids)
    home = dict(zip(metadata["game_id"], metadata["home_team_id"]))  # game_id -> home team id (same as the per-match filter)

    reconstructed = []  # one table per match
    for number, game_id in enumerate(match_ids, start=1):
        print()  # print a blank line
        print(f"[{number}/{len(match_ids)}]")  # tell the user: [ / ] (plus its value)
        if int(game_id) not in home:
            raise ValueError(f"No metadata found for match {game_id}.")  # stop with the error: No metadata found for match
        reconstructed.append(reconstruct_match_target(loader, game_id, int(home[int(game_id)]), shot_type_id, success_result_id, owngoal_result_id))
    independent_target = pd.concat(reconstructed, ignore_index=True)  # one table with all rebuilt matches

    print_header("INDEPENDENT RECONSTRUCTION COMPLETE")  # print the title of the next section
    print(f"Rows: {len(independent_target):,}")  # tell the user: Rows (plus its value)
    print(f"Matches: {independent_target['game_id'].nunique():,}")  # tell the user: Matches (plus its value)
    print(f"Positive targets: {int(independent_target['independent_target'].sum()):,}")  # tell the user: Positive targets (plus its value)
    print(f"Negative targets: {int((~independent_target['independent_target']).sum()):,}")  # tell the user: Negative targets (plus its value)

    comparison = compare_targets(saved_target, independent_target)  # compare saved and independent target
    reconciliation = reconcile_matches(saved_target, independent_target)  # compare row and positive counts per match
    split_membership = verify_split(independent_target, split_df)  # check the frozen split

    print_header("FINAL STEP 08A CHECKS")  # print the title of the checks
    c = comparison.iloc[0]  # the single comparison row
    checks = [
        ("34 matches reconstructed", independent_target["game_id"].nunique() == 34),
        ("Saved and independent row counts match", c["saved_rows"] == c["independent_rows"]),
        ("No saved-only observations", c["saved_only"] == 0),
        ("No independent-only observations", c["independent_only"] == 0),
        ("No target mismatches", c["target_mismatches"] == 0),
        ("No team mismatches", c["team_mismatches"] == 0),
        ("No future team-goal mismatches", c["future_team_goal_mismatches"] == 0),
        ("No future opponent-own-goal mismatches", c["future_opponent_own_goal_mismatches"] == 0),
        ("No match row-count failures", int((~reconciliation["row_count_match"]).sum()) == 0),
        ("No match positive-count failures", int((~reconciliation["positive_count_match"]).sum()) == 0),
        ("All target matches have split assignment", len(split_membership) == split_membership["game_id"].nunique()),
    ]
    for name, passed in checks:  # go through the checks one by one
        print(f"{'PASS' if passed else 'FAIL'}: {name}")  # print the value
    failed = [name for name, passed in checks if not passed]

    audit = independent_target.merge(saved_target, on=["game_id", "action_index"], how="outer", indicator=True,  # join every row from either side for the audit file (continues on the next line)
                                     suffixes=("_independent", "_saved"))  # every row from either side
    audit["target_match"] = audit["independent_target"].fillna(False).astype(bool) == audit["target_score_next10_literal"].fillna(False).astype(bool)
    audit["team_match"] = audit["team_id_independent"] == audit["team_id_saved"]  # True where the teams agree
    audit["future_team_goal_match"] = audit["independent_future_team_goal"].fillna(False).astype(bool) == audit["future_team_goal"].fillna(False).astype(bool)  # True where the future team-goal flags agree (missing counts as False)
    audit["future_opponent_own_goal_match"] = audit["independent_future_opponent_own_goal"].fillna(False).astype(bool) == audit["future_opponent_own_goal"].fillna(False).astype(bool)  # True where the future own-goal flags agree (missing counts as False)
    audit.to_csv(AUDIT_FILE, index=False)  # save the detailed audit
    print()  # print a blank line
    print(f"Detailed audit saved to:\n{AUDIT_FILE}")  # tell the user: Detailed audit saved to (plus its value)
    pd.DataFrame([{"check": name, "status": "PASS" if passed else "FAIL"} for name, passed in checks]).to_csv(SUMMARY_FILE, index=False)
    print()  # print a blank line
    print(f"Summary saved to:\n{SUMMARY_FILE}")  # tell the user: Summary saved to (plus its value)

    print_header("STEP 08A RESULT")  # print the title of the result
    if failed:  # at least one check failed
        print("RESULT: FAIL")  # tell the user: RESULT: FAIL
        print()  # print a blank line
        print("Failed checks:")  # tell the user: Failed checks
        for failure in failed:
            print(f"  - {failure}")  # print the value
        return 1  # hand back 1
    print("RESULT: PASS")  # tell the user: RESULT: PASS
    print()  # print a blank line
    print("Independent reconstruction agrees with the Step 07C target.")  # tell the user: Independent reconstruction agrees with the Step 07C target
    print()  # print a blank line
    print("The verified target uses:")  # tell the user: The verified target uses
    print("  Current action: EXCLUDED")  # tell the user: Current action: EXCLUDED
    print("  Future actions: i+1 through i+10")  # tell the user: Future actions: i+1 through i+10
    print("  Standardization: SPADL")  # tell the user: Standardization: SPADL
    print("  Same-team successful shot: GOAL")  # tell the user: Same-team successful shot: GOAL
    print("  Opponent own-goal: GOAL")  # tell the user: Opponent own-goal: GOAL
    return 0  # success


if __name__ == "__main__":  # only when run directly
    common.run_main(main, "STEP 08A")  # shared error handling
