# ==============================================================================
# STEP 12B (file 10_diagnose_verification_failures_.py) - DIAGNOSE THE 115 GOAL-VERIFICATION FAILURES
# ==============================================================================
# The Step 12 full-scale run reported 931 goal-verification checks passed and 115 failed. Before trusting
# the full-scale target I want to know WHY those 115 failed.
# Hypothesis: the original check assumed the action right before ANY goal must be labelled positive. That is only
# right when the same team scores. If the action right before the goal was made by the DEFENDING team (a failed
# tackle, a blocked clearance, a failed save), a negative label is correct, because target_score_next10_literal
# tracks the acting team scoring, not conceding.
# For every real goal in the 426-match dataset this script checks: which team scored, which team made the action
# right before it, whether they match, what the saved target says for that action, and whether that makes sense.
# Every case is saved, not just the failures. The Step 12 target file is NOT modified.
# Speed: per-match dictionaries replace per-goal pandas MultiIndex lookups, type names come from a small dictionary
# instead of a merge for every action, and the metadata lookup stops once all matches are found.
# ==============================================================================

import numpy as np  # arrays
import pandas as pd  # tables
from socceraction.spadl import statsbomb as statsbomb_spadl  # events -> SPADL actions
from socceraction import spadl  # SPADL constants

import common  # shared helpers
from common import OUTPUT_DIR, print_header, require_file  # shared paths and helpers
import common_spadl as cs  # loader, metadata lookup, SPADL ids

FULL_TARGET_FILE = OUTPUT_DIR / "rq1_literal_next10_target_full_dataset.csv"  # Step 12 target
FULL_SPLIT_FILE = OUTPUT_DIR / "full_scale_match_split.csv"  # Step 12 split
DIAGNOSTIC_DETAIL_FILE = OUTPUT_DIR / "12B_verification_diagnostic_detail.csv"  # one row per goal
DIAGNOSTIC_SUMMARY_FILE = OUTPUT_DIR / "12B_verification_diagnostic_summary.csv"  # counts per category
NEXT_N_ACTIONS = 10  # window length of the target (kept for reference)
TARGET_COLUMN = "target_score_next10_literal"  # label column

# The six category texts (kept in one place so the summary and the verdict use the same strings).
CAT_NOT_ELIGIBLE = "prior action ineligible (near match end) - not applicable"
CAT_SAME_TRUE = "same team, correctly True - no problem"  # label for a goal by the same team that the target marks True (correct)
CAT_SAME_FALSE = "SAME TEAM BUT LABELLED FALSE - genuine problem to investigate"  # label for a goal by the same team that the target marks False (a real problem)
CAT_OPP_FALSE = "opposing team, correctly False - expected, not a bug"  # label for a goal by the other team that the target marks False (expected)
CAT_OPP_TRUE = "opposing team but labelled True - unexpected, investigate"  # label for a goal by the other team that the target marks True (unexpected)


def load_target_and_split():  # reads the saved target and the match split
    print_header("LOAD MY FULL-SCALE TARGET AND SPLIT")  # title
    require_file(FULL_TARGET_FILE, "Full-scale target")  # input 1
    require_file(FULL_SPLIT_FILE, "Full-scale split")  # input 2
    target = pd.read_csv(FULL_TARGET_FILE)  # target table
    split = pd.read_csv(FULL_SPLIT_FILE)  # split table
    print(f"Target rows: {len(target):,}")  # size
    print(f"Split matches: {len(split):,}")  # size
    return target, split  # both tables


def diagnose_match(loader, game_id, home_team_id, shot_type_id, success_result_id, type_names, target_for_match):
    """One dict per goal in the match. target_for_match maps action_index to the saved label (only eligible actions are present)."""
    events = loader.events(game_id=int(game_id))  # raw events
    if events is None or events.empty:  # nothing to examine
        return []
    actions = statsbomb_spadl.convert_to_actions(events, int(home_team_id)).reset_index(drop=True)  # SPADL actions in order
    type_id = actions["type_id"].values  # action type ids
    result_id = actions["result_id"].values  # result ids
    team_id = actions["team_id"].values  # team ids
    is_goal = (type_id == shot_type_id) & (result_id == success_result_id)  # successful shots = goals
    rows = []  # output rows
    for pos in np.where(is_goal)[0]:  # every goal
        if pos == 0:  # a goal cannot be the first action of a match, skip defensively
            continue
        scoring_team = int(team_id[pos])  # who scored
        prior_action_index = int(pos - 1)  # the action right before the goal
        prior_team = int(team_id[prior_action_index])  # who made it
        prior_type = str(type_names[int(type_id[prior_action_index])])  # its type name
        same_team = (scoring_team == prior_team)  # did the scorer also make the prior action?
        if prior_action_index in target_for_match:  # the prior action is in the target (eligible)
            my_target_value = bool(target_for_match[prior_action_index])  # saved label
            was_eligible = True
        else:  # too close to the end of the match
            my_target_value = None
            was_eligible = False  # the prior action is not in the target
        # Expectation: same team, label should be True. Opposing team, False is correct (they are about to concede).
        if not was_eligible:
            category = CAT_NOT_ELIGIBLE  # no label to judge, the prior action is not in the target
        elif same_team and my_target_value:  # same team and target True
            category = CAT_SAME_TRUE  # pick the matching category
        elif same_team and not my_target_value:  # same team and target False
            category = CAT_SAME_FALSE  # pick the matching category
        elif not same_team and not my_target_value:  # other team and target False
            category = CAT_OPP_FALSE  # pick the matching category
        elif not same_team and my_target_value:  # other team and target True
            category = CAT_OPP_TRUE  # pick the matching category
        else:
            category = "unclassified"  # fallback if none of the rules matched (should not happen)
        rows.append({"game_id": int(game_id), "goal_action_index": int(pos), "prior_action_index": prior_action_index,  # store one diagnostic row for this goal (continues on the next line)
                     "scoring_team": scoring_team, "prior_action_team": prior_team, "prior_action_type": prior_type,
                     "same_team": same_team, "prior_action_eligible": was_eligible,
                     "my_target_for_prior_action": my_target_value, "category": category})  # same columns as before
    return rows


def main():  # runs the whole step from start to finish
    common.verify_environment()  # optional interpreter check
    print_header("STEP 12B - DIAGNOSE GOAL-VERIFICATION FAILURES")  # title
    print("Purpose: for every real goal in my full dataset, I check whether the action immediately before it was "
          "performed by the scoring team or the defending team, and whether my target label makes sense given that.")  # purpose

    target, split = load_target_and_split()  # inputs
    game_col = target["game_id"].to_numpy()  # match id per target row
    index_col = target["action_index"].to_numpy()  # action index per target row
    label_col = target[TARGET_COLUMN].to_numpy()  # saved label per target row
    rows_by_game = target.groupby("game_id").indices  # match id -> row positions in the target (built once)

    loader = cs.create_loader()  # StatsBomb reader
    shot_type_id, success_result_id, _ = cs.get_spadl_ids()  # SPADL ids
    type_names = dict(zip(spadl.actiontypes_df()["type_id"], spadl.actiontypes_df()["type_name"]))  # type id -> name

    match_ids = split["game_id"].tolist()  # matches in split order
    metadata = cs.find_match_metadata(loader, match_ids)  # home team ids
    home_team_lookup = dict(zip(metadata["game_id"], metadata["home_team_id"]))  # game_id -> home team id
    print(f"Home team resolved for {len(home_team_lookup):,} of {len(set(match_ids)):,} matches.")  # same line as before

    print_header("DIAGNOSING EVERY GOAL IN EVERY MATCH")  # title
    all_rows = []  # all goal rows
    for number, game_id in enumerate(match_ids, start=1):  # one match at a time
        if game_id not in home_team_lookup:  # no metadata
            continue
        pos = rows_by_game.get(game_id)  # target rows of this match
        target_for_match = dict(zip(index_col[pos].tolist(), label_col[pos].tolist())) if pos is not None else {}  # action_index -> label for this match only
        all_rows.extend(diagnose_match(loader, game_id, int(home_team_lookup[game_id]), shot_type_id, success_result_id, type_names, target_for_match))  # diagnose
        if number % 50 == 0 or number == len(match_ids):  # progress line
            print(f"  [{number}/{len(match_ids)}] matches diagnosed")

    diagnostic_df = pd.DataFrame(all_rows)  # one row per goal
    diagnostic_df.to_csv(DIAGNOSTIC_DETAIL_FILE, index=False)  # save the detail
    print()  # blank line
    print(f"Full diagnostic detail saved to:\n{DIAGNOSTIC_DETAIL_FILE}")  # where

    print_header("CATEGORY BREAKDOWN - MY REAL EVIDENCE")  # title
    category_counts = diagnostic_df["category"].value_counts()  # counts per category
    for category, count in category_counts.items():  # print them
        print(f"  {count:5d}  {category}")
    summary = category_counts.reset_index()  # to a table
    summary.columns = ["category", "count"]  # name the columns
    summary.to_csv(DIAGNOSTIC_SUMMARY_FILE, index=False)  # save the summary
    print()  # blank line
    print(f"Category summary saved to:\n{DIAGNOSTIC_SUMMARY_FILE}")  # where

    print_header("MY VERDICT")  # title
    genuine_problems = category_counts.get(CAT_SAME_FALSE, 0)  # same team but labelled False
    unexpected_true = category_counts.get(CAT_OPP_TRUE, 0)  # opposing team but labelled True
    total_real_issues = genuine_problems + unexpected_true  # anything my hypothesis does not explain
    if total_real_issues == 0:  # hypothesis holds
        print("[HYPOTHESIS CONFIRMED] Every one of my 115 original failures is explained by the prior action belonging "
              "to the DEFENDING team, not the scoring team. My target-building logic is correct. My original "
              "verification check's assumption was too strict - it did not account for which team performed the prior "
              "action. I trust my full-scale target and proceed to training.")
    else:  # something remains
        print(f"[REAL ISSUE FOUND] {total_real_issues} cases remain unexplained by my hypothesis - {genuine_problems} "
              f"same-team cases labelled False, and {unexpected_true} opposing-team cases labelled True. I investigate "
              f"these specific rows in {DIAGNOSTIC_DETAIL_FILE} before trusting my target for training.")
    return 0  # success


if __name__ == "__main__":  # only when run directly
    common.run_main(main, "STEP 12B")  # shared error handling
