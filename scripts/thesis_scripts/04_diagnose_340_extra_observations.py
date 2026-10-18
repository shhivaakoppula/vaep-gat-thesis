# ==============================================================================
# STEP 08B (file 04_diagnose_340_extra_observations.py) - DIAGNOSE THE EXTRA OBSERVATIONS
# ==============================================================================
# Step 08A found observations in the independent SPADL reconstruction that are absent from the Step 07C target
# (340 of them across 34 matches). This script identifies them and describes them: action type, result, period,
# split, whether they are the last action of the match. It only DESCRIBES them. It does NOT modify Step 07C, does NOT
# delete anything and does NOT force the datasets to match. It writes a detailed CSV and a small summary CSV.
# ==============================================================================

import numpy as np  # np.select for the category labels
import pandas as pd  # tables

import common  # shared helpers
from common import DATA_DIR, OUTPUT_DIR, print_header, require_file  # shared paths and helpers
from socceraction.data.statsbomb import StatsBombLoader  # reads StatsBomb files
from socceraction.spadl import statsbomb as statsbomb_spadl  # events -> SPADL actions
from socceraction import spadl  # SPADL lookups

TARGET_FILE = OUTPUT_DIR / "rq1_literal_next10_target_34_matches.csv"  # Step 07C target
SPLIT_FILE = OUTPUT_DIR / "final_common_match_split.csv"  # frozen split
OUTPUT_FILE = OUTPUT_DIR / "08B_extra_observations_diagnostic.csv"  # detailed diagnostic
SUMMARY_FILE = OUTPUT_DIR / "08B_extra_observations_summary.csv"  # summary counts
NEXT_N_ACTIONS = 10  # horizon of the target (kept for reference)


def load_saved_target():  # reads the saved Step 07C target and checks it
    print_header("LOAD STEP 07C TARGET")  # title
    target = pd.read_csv(TARGET_FILE)  # saved target
    required_columns = ["game_id", "action_index", "period_id", "time_seconds", "team_id",
                        "target_score_next10_literal", "future_team_goal", "future_opponent_own_goal"]
    missing = set(required_columns) - set(target.columns)  # columns that are required but absent
    if missing:  # stop if a column is missing
        raise ValueError(f"Missing target columns: {sorted(missing)}")  # stop with the error: Missing target columns
    target = target[required_columns].copy()  # fixed column order
    for column in ["game_id", "action_index", "period_id", "team_id"]:  # integer columns
        target[column] = pd.to_numeric(target[column], errors="raise").astype(int)
    target["time_seconds"] = pd.to_numeric(target["time_seconds"], errors="raise")  # float seconds
    for column in ["target_score_next10_literal", "future_team_goal", "future_opponent_own_goal"]:  # boolean columns
        target[column] = target[column].astype(bool)
    print(f"Step 07C rows: {len(target):,}")  # tell the user: Step 07C rows (plus its value)
    print(f"Step 07C matches: {target['game_id'].nunique():,}")  # tell the user: Step 07C matches (plus its value)
    return target  # hand back target


def load_split():  # reads the frozen split
    print_header("LOAD FROZEN MATCH SPLIT")  # title
    split = pd.read_csv(SPLIT_FILE)
    split["game_id"] = pd.to_numeric(split["game_id"], errors="raise").astype(int)  # make the match id an integer
    split["split"] = split["split"].astype(str).str.lower().str.strip()  # clean up the split label
    print(f"Matches: {len(split):,}")  # tell the user: Matches (plus its value)
    print(split["split"].value_counts().sort_index())  # print the result of split['split'].value_counts().sort_index()
    return split  # hand back split


def get_spadl_ids():  # looks up the SPADL numbers and names for shots and results
    print_header("GET SPADL IDENTIFIERS")  # title
    actiontypes, results = spadl.actiontypes_df(), spadl.results_df()  # lookup tables
    shot_type_id = int(actiontypes[actiontypes["type_name"] == "shot"].iloc[0]["type_id"])
    success_result_id = int(results[results["result_name"] == "success"].iloc[0]["result_id"])  # SPADL id of 'success'
    owngoal_result_id = int(results[results["result_name"] == "owngoal"].iloc[0]["result_id"])  # SPADL id of 'owngoal'
    actiontype_lookup = actiontypes[["type_id", "type_name"]].copy()  # id -> name
    result_lookup = results[["result_id", "result_name"]].copy()  # id -> name
    print(f"Shot type ID: {shot_type_id}")
    print(f"Success result ID: {success_result_id}")  # tell the user: Success result ID (plus its value)
    print(f"Own-goal result ID: {owngoal_result_id}")  # tell the user: Own-goal result ID (plus its value)
    return shot_type_id, success_result_id, owngoal_result_id, actiontype_lookup, result_lookup  # hand back shot_type_id, success_result_id, owngoal_result_id, actio...


def create_loader():  # creates the local StatsBomb reader
    print_header("CREATE STATSBOMB LOADER")  # title
    loader = StatsBombLoader(getter="local", root=str(DATA_DIR))  # local files
    print(f"[PASS] StatsBomb root: {DATA_DIR}")
    return loader  # hand back loader


def find_match_metadata(loader, match_ids):  # finds the home team of every requested match
    print_header("FIND MATCH METADATA")  # title
    remaining = set(int(x) for x in match_ids)  # still to find
    parts = []
    for _, row in loader.competitions().iterrows():  # every competition-season
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
        if games.empty:  # skip seasons without any of our matches
            continue  # skip to the next item
        parts.append(games[["game_id", "competition_id", "season_id", "home_team_id", "away_team_id"]])
        remaining -= set(games["game_id"].astype(int).tolist())  # tick off the matches found
        if not remaining:  # stop searching once every match is found
            break  # leave the loop
    if remaining:
        raise ValueError(f"Missing metadata for matches: {sorted(remaining)}")  # stop with the error: Missing metadata for matches
    metadata = pd.concat(parts, ignore_index=True)
    metadata["game_id"] = metadata["game_id"].astype(int)  # make the match id an integer
    metadata = metadata.drop_duplicates(subset=["game_id"])  # one row per match
    print(f"[PASS] Metadata found for {len(metadata):,} matches.")  # tell the user: [PASS] Metadata found for matches (plus its value)
    return metadata  # hand back metadata


def reconstruct_actions(loader, game_id, home_team_id):  # converts one match's events to SPADL actions
    events = loader.events(game_id=int(game_id))  # raw events
    if events is None or events.empty:
        raise ValueError(f"No events for match {game_id}")  # stop with the error: No events for match
    actions = statsbomb_spadl.convert_to_actions(events, int(home_team_id)).reset_index(drop=True).copy()  # SPADL actions
    actions["action_index"] = range(len(actions))  # position in the SPADL order
    return actions


def build_independent_table(loader, metadata, actiontype_lookup, result_lookup):  # rebuilds all actions of every match and adds readable names
    print_header("RECONSTRUCT ALL 34 MATCHES")  # title
    all_actions = []
    match_ids = sorted(metadata["game_id"].unique().tolist())  # matches in order
    home = dict(zip(metadata["game_id"], metadata["home_team_id"]))  # game_id -> home team id
    for number, game_id in enumerate(match_ids, start=1):
        print(f"[{number}/{len(match_ids)}] Match {game_id}")  # tell the user: [ / ] Match (plus its value)
        actions = reconstruct_actions(loader, game_id, int(home[game_id]))
        actions = actions.merge(actiontype_lookup, left_on="type_id", right_on="type_id", how="left")  # type_name
        actions = actions.merge(result_lookup, left_on="result_id", right_on="result_id", how="left")  # result_name
        actions["game_id"] = int(game_id)
        all_actions.append(actions)  # keep the actions of this match
        print(f"  SPADL actions: {len(actions):,}")  # tell the user: SPADL actions (plus its value)
    independent = pd.concat(all_actions, ignore_index=True)
    print()  # print a blank line
    print(f"Total independent SPADL actions: {len(independent):,}")  # tell the user: Total independent SPADL actions (plus its value)
    return independent  # hand back independent


def identify_extra_observations(saved_target, independent):  # finds the actions that exist only in the independent table
    print_header("IDENTIFY INDEPENDENT-ONLY OBSERVATIONS")  # title
    saved_keys = saved_target[["game_id", "action_index"]].drop_duplicates()
    independent_keys = independent[["game_id", "action_index"]].drop_duplicates()  # key pairs of the independent actions
    extra_keys = independent_keys.merge(saved_keys, on=["game_id", "action_index"], how="left", indicator=True)  # left join marks rows with no saved partner
    extra_keys = extra_keys[extra_keys["_merge"] == "left_only"][["game_id", "action_index"]]
    print(f"Independent observations: {len(independent_keys):,}")  # tell the user: Independent observations (plus its value)
    print(f"Saved observations: {len(saved_keys):,}")  # tell the user: Saved observations (plus its value)
    print(f"Independent-only observations: {len(extra_keys):,}")  # tell the user: Independent-only observations (plus its value)
    if len(extra_keys) == 0:
        print("[PASS] No extra observations.")  # tell the user: [PASS] No extra observations
        return pd.DataFrame()  # hand back pd.DataFrame()
    return extra_keys.merge(independent, on=["game_id", "action_index"], how="left")  # bring in the full action rows


def add_diagnostic_flags(extra, shot_type_id, success_result_id, owngoal_result_id):  # adds flags that describe each extra action
    if extra.empty:  # nothing to flag when there are no extra actions
        return extra  # hand back extra
    extra = extra.copy()
    extra["is_shot"] = extra["type_id"] == shot_type_id  # a shot
    extra["is_successful_shot"] = extra["is_shot"] & (extra["result_id"] == success_result_id)  # a goal
    extra["is_own_goal"] = extra["is_shot"] & (extra["result_id"] == owngoal_result_id)  # an own goal
    extra["period_id"] = pd.to_numeric(extra["period_id"], errors="coerce")
    extra["time_seconds"] = pd.to_numeric(extra["time_seconds"], errors="coerce")  # make the time numeric
    # Only descriptive: is this the last action of its match? (It is NOT assumed to be the reason yet.)
    extra["last_action_in_match"] = extra.groupby("game_id")["action_index"].transform("max") == extra["action_index"]
    extra["action_index_mod_2"] = extra["action_index"] % 2  # parity, basic structure check
    extra["diagnostic_action_category"] = np.select(  # same priority order as the old row-by-row function
        [extra["is_successful_shot"], extra["is_own_goal"], extra["is_shot"]],
        ["successful_shot", "own_goal", "other_shot"], default="non_shot")
    return extra  # hand back extra


def create_summary(extra):  # summarises the extra actions by type, result, period and match
    print_header("EXTRA OBSERVATION SUMMARY")  # title
    if extra.empty:
        print("No extra observations found.")  # tell the user: No extra observations found
        return pd.DataFrame()  # hand back pd.DataFrame()
    rows = [{"category": "overall", "group": "all_extra_observations", "count": len(extra)}]  # first row
    def add_counts(category, counts):
        for group, count in counts.items():  # one summary row per group
            rows.append({"category": category, "group": str(group), "count": int(count)})
    add_counts("action_type", extra["type_name"].fillna("MISSING").value_counts())  # count of extra actions per action type
    add_counts("result", extra["result_name"].fillna("MISSING").value_counts())  # count of extra actions per result
    add_counts("diagnostic_category", extra["diagnostic_action_category"].value_counts())  # count of extra actions per diagnostic category
    add_counts("period", extra["period_id"].value_counts().sort_index())  # count of extra actions per period
    add_counts("match", extra["game_id"].value_counts().sort_index())  # count of extra actions per match
    add_counts("last_action_in_match", extra["last_action_in_match"].value_counts())  # count of extra actions that are the last in their match
    return pd.DataFrame(rows)  # hand back pd.DataFrame(rows)


def main():  # runs the whole step from start to finish
    common.verify_environment()  # optional interpreter check
    print_header("STEP 08B — DIAGNOSE EXTRA OBSERVATIONS")
    print("Purpose:")  # tell the user: Purpose
    print("Identify why the independent SPADL reconstruction contains observations that are absent from Step 07C.")  # tell the user: Identify why the independent SPADL reconstruction contains...
    print()  # print a blank line
    print("IMPORTANT:")  # tell the user: IMPORTANT
    print("This script does NOT modify Step 07C.")  # tell the user: This script does NOT modify Step 07C
    print("This script does NOT delete observations.")  # tell the user: This script does NOT delete observations
    print("This script does NOT force the datasets to match.")  # tell the user: This script does NOT force the datasets to match
    require_file(TARGET_FILE, "Step 07C target")
    require_file(SPLIT_FILE, "Frozen split")  # stop if the split file is missing
    saved_target = load_saved_target()  # the saved Step 07C target
    split = load_split()  # the frozen split
    shot_type_id, success_result_id, owngoal_result_id, actiontype_lookup, result_lookup = get_spadl_ids()  # SPADL ids and lookup tables
    loader = create_loader()  # the local StatsBomb reader
    match_ids = sorted(split["game_id"].unique().tolist())  # matches to rebuild, in order
    metadata = find_match_metadata(loader, match_ids)  # home team and other details of each match
    independent = build_independent_table(loader, metadata, actiontype_lookup, result_lookup)  # all actions rebuilt independently
    extra = identify_extra_observations(saved_target, independent)  # the actions only the independent side has
    extra = add_diagnostic_flags(extra, shot_type_id, success_result_id, owngoal_result_id)  # add the diagnostic flags
    if not extra.empty:  # only when there are extra actions
        extra = extra.merge(split[["game_id", "split"]], on="game_id", how="left")  # attach the split name

    print_header("DIAGNOSTIC RESULTS")  # print the title of the results section
    if extra.empty:  # check whether there are any extra actions
        print("[PASS] No independent-only observations.")  # tell the user: [PASS] No independent-only observations
    else:
        print(f"Independent-only observations: {len(extra):,}")  # tell the user: Independent-only observations (plus its value)
        print()  # print a blank line
        print("By diagnostic action category:")  # tell the user: By diagnostic action category
        print(extra["diagnostic_action_category"].value_counts())  # print the result of extra['diagnostic_action_category'].value_counts()
        print()  # print a blank line
        print("By action type:")  # tell the user: By action type
        print(extra["type_name"].fillna("MISSING").value_counts())  # print the result of extra['type_name'].fillna('MISSING').value_coun...
        print()  # print a blank line
        print("By result:")  # tell the user: By result
        print(extra["result_name"].fillna("MISSING").value_counts())  # print the result of extra['result_name'].fillna('MISSING').value_co...
        print()  # print a blank line
        print("By split:")  # tell the user: By split
        print(extra["split"].value_counts())  # print the result of extra['split'].value_counts()
        print()  # print a blank line
        print("By period:")  # tell the user: By period
        print(extra["period_id"].value_counts().sort_index())  # print the result of extra['period_id'].value_counts().sort_index()
        print()  # print a blank line
        print("Successful shots among extras:")  # tell the user: Successful shots among extras
        print(int(extra["is_successful_shot"].sum()))  # print the result of int(extra['is_successful_shot'].sum())
        print()  # print a blank line
        print("Own goals among extras:")  # tell the user: Own goals among extras
        print(int(extra["is_own_goal"].sum()))  # print the result of int(extra['is_own_goal'].sum())
        print()  # print a blank line
        print("Last actions in match among extras:")  # tell the user: Last actions in match among extras
        print(int(extra["last_action_in_match"].sum()))  # print the result of int(extra['last_action_in_match'].sum())

    extra.to_csv(OUTPUT_FILE, index=False)  # detailed output
    print()
    print(f"Detailed diagnostic saved to:\n{OUTPUT_FILE}")  # tell the user: Detailed diagnostic saved to (plus its value)
    create_summary(extra).to_csv(SUMMARY_FILE, index=False)  # summary output
    print()
    print(f"Diagnostic summary saved to:\n{SUMMARY_FILE}")  # tell the user: Diagnostic summary saved to (plus its value)
    print_header("STEP 08B COMPLETE")
    print("No data was modified.")  # tell the user: No data was modified
    print("No observations were deleted.")  # tell the user: No observations were deleted
    print("The purpose of this step is only to identify the 340-observation discrepancy.")  # tell the user: The purpose of this step is only to identify the 340-observation...
    print()  # print a blank line
    print("NEXT DECISION:")  # tell the user: NEXT DECISION
    print("Inspect the diagnostic CSV before changing Step 07C or declaring the target frozen.")  # tell the user: Inspect the diagnostic CSV before changing Step 07C or declaring...
    return 0  # success


if __name__ == "__main__":  # only when run directly
    common.run_main(main, "STEP 08B")  # shared error handling
