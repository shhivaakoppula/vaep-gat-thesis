# ==============================================================================
# STEP 08C (file 05_independent_target_verification_with_eligibility.py)
# INDEPENDENT TARGET VERIFICATION WITH THE ELIGIBILITY RULE
# ==============================================================================
# Script 03 found 340 more rows than the saved Step 07C target. Script 04 explained them as the last 10 actions of
# each of the 34 matches, which have no full 10-action future window. This script tests that explanation: it rebuilds
# the target independently, keeps only actions that have at least 10 future actions, and compares the result with the
# saved Step 07C target. Expected outcome: 0 rows only on one side and 0 mismatches.
# The window is built with a sliding window over padded arrays (a different technique from the build scripts), and
# it is far faster than the original double Python loop while giving the same flags.
# Nothing is modified or forced to match; the script only reads, compares and writes two audit CSVs.
# ==============================================================================

import numpy as np  # arrays and sliding windows
import pandas as pd  # tables
from numpy.lib.stride_tricks import sliding_window_view  # windows over arrays without copying

import common  # shared helpers
from common import DATA_DIR, OUTPUT_DIR, print_header, require_file  # shared paths and helpers

from socceraction.data.statsbomb import StatsBombLoader  # reads StatsBomb files
from socceraction.spadl import statsbomb as statsbomb_spadl  # events -> SPADL actions
from socceraction import spadl  # SPADL lookups

TARGET_FILE = OUTPUT_DIR / "rq1_literal_next10_target_34_matches.csv"  # Step 07C target
SPLIT_FILE = OUTPUT_DIR / "final_common_match_split.csv"  # frozen split
OUTPUT_FILE = OUTPUT_DIR / "08C_eligible_verification_detail.csv"  # row-level comparison
SUMMARY_FILE = OUTPUT_DIR / "08C_eligible_verification_summary.csv"  # counts table
NEXT_N_ACTIONS = 10  # future horizon in actions


def load_saved_target():  # reads the saved Step 07C target and checks its columns
    print_header("LOAD STEP 07C TARGET")  # title
    target = pd.read_csv(TARGET_FILE)  # saved target
    required_columns = ["game_id", "action_index", "period_id", "time_seconds", "team_id",
                        "target_score_next10_literal", "future_team_goal", "future_opponent_own_goal"]  # needed schema
    missing = set(required_columns) - set(target.columns)
    if missing:  # stop if a needed column is missing
        raise ValueError(f"Missing target columns: {sorted(missing)}")  # stop with the error: Missing target columns
    target = target[required_columns].copy()  # fixed column order
    for column in ["game_id", "action_index", "period_id", "team_id"]:  # integer columns
        target[column] = pd.to_numeric(target[column], errors="raise").astype(int)
    target["time_seconds"] = pd.to_numeric(target["time_seconds"], errors="raise")  # seconds stay float
    for column in ["target_score_next10_literal", "future_team_goal", "future_opponent_own_goal"]:  # boolean columns
        target[column] = target[column].astype(bool)
    print(f"Step 07C rows: {len(target):,}")  # tell the user: Step 07C rows (plus its value)
    print(f"Step 07C matches: {target['game_id'].nunique():,}")  # tell the user: Step 07C matches (plus its value)
    print(f"Step 07C positive targets: {int(target['target_score_next10_literal'].sum()):,}")  # tell the user: Step 07C positive targets (plus its value)
    return target  # hand back target


def load_split():  # reads the frozen split
    print_header("LOAD FROZEN MATCH SPLIT")  # title
    split = pd.read_csv(SPLIT_FILE)  # split table
    split["game_id"] = pd.to_numeric(split["game_id"], errors="raise").astype(int)  # integer ids
    split["split"] = split["split"].astype(str).str.lower().str.strip()  # normalised labels
    print(f"Matches: {len(split):,}")
    print(split["split"].value_counts().sort_index())  # print the result of split['split'].value_counts().sort_index()
    return split  # hand back split


def get_spadl_ids():  # looks up the SPADL numbers for 'shot', 'success' and 'owngoal'
    print_header("GET SPADL IDENTIFIERS")  # title
    actiontypes, results = spadl.actiontypes_df(), spadl.results_df()  # lookup tables from the installed library
    shot_type_id = int(actiontypes[actiontypes["type_name"] == "shot"].iloc[0]["type_id"])
    success_result_id = int(results[results["result_name"] == "success"].iloc[0]["result_id"])  # SPADL id of 'success'
    owngoal_result_id = int(results[results["result_name"] == "owngoal"].iloc[0]["result_id"])  # SPADL id of 'owngoal'
    print(f"Shot type ID: {shot_type_id}")  # tell the user: Shot type ID (plus its value)
    print(f"Success result ID: {success_result_id}")  # tell the user: Success result ID (plus its value)
    print(f"Own-goal result ID: {owngoal_result_id}")  # tell the user: Own-goal result ID (plus its value)
    return shot_type_id, success_result_id, owngoal_result_id  # hand back shot_type_id, success_result_id, owngoal_result_id


def create_loader():  # creates the local StatsBomb reader
    print_header("CREATE STATSBOMB LOADER")  # title
    loader = StatsBombLoader(getter="local", root=str(DATA_DIR))  # local files only
    print(f"[PASS] StatsBomb root: {DATA_DIR}")
    return loader  # hand back loader


def find_match_metadata(loader, match_ids):  # finds the home team of every requested match
    print_header("FIND MATCH METADATA")  # title
    remaining = set(int(x) for x in match_ids)  # still to find
    parts = []  # collected tables
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
        if not remaining:  # stop early once everything is found
            break
    if remaining:  # stop if some match was not found
        raise ValueError(f"Missing metadata for matches: {sorted(remaining)}")  # stop with the error: Missing metadata for matches
    metadata = pd.concat(parts, ignore_index=True)
    metadata["game_id"] = metadata["game_id"].astype(int)  # make the match id an integer
    metadata = metadata.drop_duplicates(subset=["game_id"])  # one row per match
    print(f"[PASS] Metadata found for {len(metadata):,} matches.")  # tell the user: [PASS] Metadata found for matches (plus its value)
    return metadata  # hand back metadata


def window_flags(team, is_goal_shot, is_own_goal_shot, horizon):
    """For action i look at actions i+1 .. i+horizon only (never i itself). Padding is False so a window that runs
    past the end of the match is simply shorter. Returns (same-team goal flag, opponent own-goal flag)."""
    n = len(team)  # actions in the match
    pad = np.zeros(horizon, dtype=bool)  # False padding
    goal_pad = np.concatenate([is_goal_shot, pad])  # successful shots, padded
    own_pad = np.concatenate([is_own_goal_shot, pad])  # own goals, padded
    team_pad = np.concatenate([team, np.full(horizon, -1, dtype=team.dtype)])  # team ids, padded with an impossible id
    goal_win = sliding_window_view(goal_pad, horizon)[1:n + 1]  # row i = padded positions i+1 .. i+horizon
    own_win = sliding_window_view(own_pad, horizon)[1:n + 1]  # same for own goals
    team_win = sliding_window_view(team_pad, horizon)[1:n + 1]  # same for team ids
    future_team_goal = (goal_win & (team_win == team[:, None])).any(axis=1)  # a successful shot by the SAME team
    future_opponent_own_goal = (own_win & (team_win != team[:, None])).any(axis=1)  # an own goal by the OTHER side
    return future_team_goal, future_opponent_own_goal


def reconstruct_actions(loader, game_id, home_team_id):  # converts one match's events to SPADL actions
    events = loader.events(game_id=int(game_id))  # raw events
    if events is None or events.empty:
        raise ValueError(f"No events for match {game_id}")  # stop with the error: No events for match
    actions = statsbomb_spadl.convert_to_actions(events, int(home_team_id)).reset_index(drop=True).copy()  # SPADL actions
    actions["action_index"] = range(len(actions))  # position in the SPADL order
    return actions


def compute_literal_target_for_match(match_actions, shot_type_id, success_result_id, owngoal_result_id):
    """Add the independent flags and the number of future actions that really exist for each action."""
    match_actions = match_actions.reset_index(drop=True)  # clean positional index
    n = len(match_actions)  # actions in the match
    type_id = match_actions["type_id"].to_numpy()  # action types
    result_id = match_actions["result_id"].to_numpy()  # action results
    team_id = match_actions["team_id"].to_numpy()  # acting team
    is_goal_shot = (type_id == shot_type_id) & (result_id == success_result_id)  # successful shots
    is_own_goal_shot = (type_id == shot_type_id) & (result_id == owngoal_result_id)  # own goals
    future_team_goal, future_opponent_own_goal = window_flags(team_id, is_goal_shot, is_own_goal_shot, NEXT_N_ACTIONS)  # flags
    available = np.minimum(NEXT_N_ACTIONS, n - 1 - np.arange(n))  # how many future actions exist for each action
    match_actions = match_actions.copy()
    match_actions["future_team_goal_independent"] = future_team_goal  # flag: the same team scores within the next 10 actions
    match_actions["future_opponent_own_goal_independent"] = future_opponent_own_goal  # flag: the other side scores an own goal within the next 10 actions
    match_actions["future_actions_available"] = available  # how many future actions exist for each action (at most 10)
    match_actions["target_independent"] = future_team_goal | future_opponent_own_goal  # positive on either flag
    return match_actions


def apply_eligibility_rule(actions_with_target):
    """Eligible = a full 10-action future window exists. Ineligible = the window is cut short by the end of the match."""
    full_window = actions_with_target["future_actions_available"] >= NEXT_N_ACTIONS  # True where a full window exists
    return actions_with_target[full_window].copy(), actions_with_target[~full_window].copy()


def build_independent_eligible_table(loader, metadata, shot_type_id, success_result_id, owngoal_result_id):  # rebuilds every match, labels it and applies the eligibility rule
    print_header("RECONSTRUCT ALL 34 MATCHES, COMPUTE TARGET, APPLY ELIGIBILITY")  # title
    all_eligible, all_ineligible = [], []  # collected tables
    match_ids = sorted(metadata["game_id"].unique().tolist())  # matches in order
    home = dict(zip(metadata["game_id"], metadata["home_team_id"]))  # game_id -> home team id
    for number, game_id in enumerate(match_ids, start=1):
        actions = reconstruct_actions(loader, game_id, int(home[game_id]))  # SPADL actions
        actions["game_id"] = int(game_id)  # tag the match
        actions_with_target = compute_literal_target_for_match(actions, shot_type_id, success_result_id, owngoal_result_id)
        eligible, ineligible = apply_eligibility_rule(actions_with_target)  # split by window length
        print(f"[{number}/{len(match_ids)}] Match {game_id}: {len(actions):,} total, {len(eligible):,} eligible, {len(ineligible):,} ineligible")
        all_eligible.append(eligible)  # keep the eligible rows of this match
        all_ineligible.append(ineligible)  # keep the ineligible rows of this match
    independent_eligible = pd.concat(all_eligible, ignore_index=True)  # all eligible rows of all matches
    independent_ineligible = pd.concat(all_ineligible, ignore_index=True)  # all ineligible rows of all matches
    print()  # print a blank line
    print(f"Total independent eligible observations: {len(independent_eligible):,}")  # tell the user: Total independent eligible observations (plus its value)
    print(f"Total independent ineligible observations: {len(independent_ineligible):,}")  # tell the user: Total independent ineligible observations (plus its value)
    return independent_eligible, independent_ineligible  # hand back independent_eligible, independent_ineligible


def compare_against_saved_target(saved_target, independent_eligible):  # compares the independent eligible target with the saved one
    print_header("COMPARE INDEPENDENT ELIGIBLE TARGET vs STEP 07C")  # title
    saved = saved_target.rename(columns={"target_score_next10_literal": "target_saved", "future_team_goal": "future_team_goal_saved",
                                         "future_opponent_own_goal": "future_opponent_own_goal_saved", "team_id": "team_id_saved"})
    independent = independent_eligible.rename(columns={"team_id": "team_id_independent"})  # independent side with its team column renamed
    merged = saved.merge(independent, on=["game_id", "action_index"], how="outer", indicator=True)  # outer join shows one-sided rows
    only_saved = merged[merged["_merge"] == "left_only"]
    only_independent = merged[merged["_merge"] == "right_only"]  # rows that only the independent side has
    common_rows = merged[merged["_merge"] == "both"].copy()  # rows both sides have
    print(f"Saved (Step 07C) rows: {len(saved):,}")
    print(f"Independent eligible rows: {len(independent):,}")  # tell the user: Independent eligible rows (plus its value)
    print(f"Common rows: {len(common_rows):,}")  # tell the user: Common rows (plus its value)
    print(f"Saved-only rows: {len(only_saved):,}")  # tell the user: Saved-only rows (plus its value)
    print(f"Independent-only rows: {len(only_independent):,}")  # tell the user: Independent-only rows (plus its value)
    common_rows["team_mismatch"] = common_rows["team_id_saved"] != common_rows["team_id_independent"]
    common_rows["target_mismatch"] = common_rows["target_saved"] != common_rows["target_independent"]  # True where the targets disagree
    common_rows["future_team_goal_mismatch"] = common_rows["future_team_goal_saved"] != common_rows["future_team_goal_independent"]  # True where the future team-goal flags disagree
    common_rows["future_opponent_own_goal_mismatch"] = (common_rows["future_opponent_own_goal_saved"]  # True where the future own-goal flags disagree (continues on the next line)
                                                        != common_rows["future_opponent_own_goal_independent"])
    n_team_mismatch = int(common_rows["team_mismatch"].sum())  # number of rows with different teams
    n_target_mismatch = int(common_rows["target_mismatch"].sum())  # number of rows with different targets
    n_future_goal_mismatch = int((common_rows["future_team_goal_mismatch"] | common_rows["future_opponent_own_goal_mismatch"]).sum())  # number of rows where either future flag differs
    print()  # print a blank line
    print(f"Team ID mismatches: {n_team_mismatch:,}")  # tell the user: Team ID mismatches (plus its value)
    print(f"Target mismatches: {n_target_mismatch:,}")  # tell the user: Target mismatches (plus its value)
    print(f"Future-goal component mismatches: {n_future_goal_mismatch:,}")  # tell the user: Future-goal component mismatches (plus its value)
    print()  # print a blank line
    print(f"Independent eligible positive targets: {int(independent_eligible['target_independent'].sum()):,}")  # tell the user: Independent eligible positive targets (plus its value)
    print(f"Step 07C saved positive targets: {int(saved_target['target_score_next10_literal'].sum()):,}")  # tell the user: Step 07C saved positive targets (plus its value)
    return merged, only_saved, only_independent, n_team_mismatch, n_target_mismatch, n_future_goal_mismatch  # hand back merged, only_saved, only_independent, n_team_mismatch, n_...


def print_final_verdict(only_saved, only_independent, n_team_mismatch, n_target_mismatch, n_future_goal_mismatch):  # prints the verdict: confirmed or not yet confirmed
    print_header("STEP 08C FINAL VERDICT")  # title
    all_clear = (len(only_saved) == 0 and len(only_independent) == 0 and n_team_mismatch == 0
                 and n_target_mismatch == 0 and n_future_goal_mismatch == 0)  # everything must be zero
    if all_clear:
        print("[PASS] My independent, eligibility-filtered reconstruction exactly matches Step 07C.")  # tell the user: [PASS] My independent, eligibility-filtered reconstruction exactly...
        print("This confirms my 08B diagnosis: the 340 extra rows found in 08A were exactly the final 10 actions of each of my 34 "  # tell the user: This confirms my 08B diagnosis: the 340 extra rows found in 08A...
              "matches, correctly excluded because they lack a full 10-action future window.")
        print()  # print a blank line
        print("I now consider my RQ1 target, target_score_next10_literal, fully VERIFIED and ready to be frozen. "  # tell the user: I now consider my RQ1 target, target_score_next10_literal, fully...
              "I can proceed to train VAEP and GAT on this label.")
    else:
        print("[NOT YET CONFIRMED] My independent reconstruction does not fully match Step 07C.")  # tell the user: [NOT YET CONFIRMED] My independent reconstruction does not fully...
        print(f"Saved-only rows remaining: {len(only_saved):,}")  # tell the user: Saved-only rows remaining (plus its value)
        print(f"Independent-only rows remaining: {len(only_independent):,}")  # tell the user: Independent-only rows remaining (plus its value)
        print(f"Team mismatches: {n_team_mismatch:,}")  # tell the user: Team mismatches (plus its value)
        print(f"Target mismatches: {n_target_mismatch:,}")  # tell the user: Target mismatches (plus its value)
        print(f"Future-goal mismatches: {n_future_goal_mismatch:,}")  # tell the user: Future-goal mismatches (plus its value)
        print()  # print a blank line
        print("I do NOT consider my target frozen yet. I need to inspect the remaining discrepancy in the saved diagnostic CSV "  # tell the user: I do NOT consider my target frozen yet. I need to inspect the...
              "before training anything.")


def main():  # runs the whole step from start to finish
    common.verify_environment()  # optional interpreter check
    print_header("STEP 08C — INDEPENDENT TARGET VERIFICATION WITH ELIGIBILITY")  # title
    print("Purpose:")
    print("I am confirming whether the 340-row discrepancy found in 08A, and diagnosed in 08B as end-of-match actions lacking a "  # tell the user: I am confirming whether the 340-row discrepancy found in 08A, and...
          "full future window, fully disappears once I apply the same 10-action eligibility rule to my independent reconstruction.")
    print()  # print a blank line
    print("IMPORTANT:")  # tell the user: IMPORTANT
    print("This script does NOT modify Step 07C.")  # tell the user: This script does NOT modify Step 07C
    print("This script does NOT delete or force any observations.")  # tell the user: This script does NOT delete or force any observations
    print("I compute my own target independently and only compare it — I do not copy Step 07C's target values.")  # tell the user: I compute my own target independently and only compare it — I do...
    require_file(TARGET_FILE, "Step 07C target")  # input must exist
    require_file(SPLIT_FILE, "Frozen split")  # input must exist
    saved_target = load_saved_target()
    split = load_split()  # the frozen split
    shot_type_id, success_result_id, owngoal_result_id = get_spadl_ids()  # SPADL ids of 'shot', 'success' and 'owngoal'
    loader = create_loader()  # the local StatsBomb reader
    match_ids = sorted(split["game_id"].unique().tolist())  # matches to rebuild
    metadata = find_match_metadata(loader, match_ids)
    independent_eligible, independent_ineligible = build_independent_eligible_table(  # rebuild everything and apply the eligibility rule (continues on the next line)
        loader, metadata, shot_type_id, success_result_id, owngoal_result_id)
    merged, only_saved, only_independent, n_team_mismatch, n_target_mismatch, n_future_goal_mismatch = compare_against_saved_target(  # compare against the saved target (continues on the next line)
        saved_target, independent_eligible)

    print_header("CROSS-CHECK AGAINST 08B DIAGNOSIS")  # 08B found 34 x 10 = 340 extra rows
    print(f"My independently computed ineligible row count: {len(independent_ineligible):,}")
    print("08B's diagnosed extra-row count was: 340")  # tell the user: 08B's diagnosed extra-row count was: 340
    if len(independent_ineligible) == 340:
        print("[PASS] These counts match exactly, confirming 08B's diagnosis from a second, independent calculation.")  # tell the user: [PASS] These counts match exactly, confirming 08B's diagnosis from...
    else:
        print("[CHECK NEEDED] These counts do not match. I need to investigate this difference before concluding anything.")  # tell the user: [CHECK NEEDED] These counts do not match. I need to investigate...

    print_final_verdict(only_saved, only_independent, n_team_mismatch, n_target_mismatch, n_future_goal_mismatch)  # print the final verdict

    merged.to_csv(OUTPUT_FILE, index=False)  # row-level detail
    print()
    print(f"Detailed comparison saved to:\n{OUTPUT_FILE}")  # tell the user: Detailed comparison saved to (plus its value)
    summary_rows = [
        {"metric": "step07c_rows", "value": len(saved_target)},
        {"metric": "independent_eligible_rows", "value": len(independent_eligible)},
        {"metric": "independent_ineligible_rows", "value": len(independent_ineligible)},
        {"metric": "saved_only_rows", "value": len(only_saved)},
        {"metric": "independent_only_rows", "value": len(only_independent)},
        {"metric": "team_mismatches", "value": n_team_mismatch},
        {"metric": "target_mismatches", "value": n_target_mismatch},
        {"metric": "future_goal_mismatches", "value": n_future_goal_mismatch},
        {"metric": "step07c_positive_targets", "value": int(saved_target["target_score_next10_literal"].sum())},
        {"metric": "independent_eligible_positive_targets", "value": int(independent_eligible["target_independent"].sum())},
    ]  # counts table
    pd.DataFrame(summary_rows).to_csv(SUMMARY_FILE, index=False)
    print()  # print a blank line
    print(f"Summary saved to:\n{SUMMARY_FILE}")  # tell the user: Summary saved to (plus its value)
    print_header("STEP 08C COMPLETE")
    print("No data was modified.")  # tell the user: No data was modified
    print("No observations were deleted or forced to match.")  # tell the user: No observations were deleted or forced to match
    print("My target is either confirmed VERIFIED above, or I have an exact, documented discrepancy left to investigate before training anything.")  # tell the user: My target is either confirmed VERIFIED above, or I have an exact,...
    return 0  # success


if __name__ == "__main__":  # only when run directly
    common.run_main(main, "STEP 08C")  # shared error handling
