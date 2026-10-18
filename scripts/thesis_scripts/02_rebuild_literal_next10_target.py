# ==============================================================================
# 02_rebuild_literal_next10_target.py (printed as "07C") - REBUILD THE LITERAL NEXT-10 TARGET FROM SPADL ACTIONS
# ==============================================================================
# Rebuilds the next-10-actions target for the frozen 34-match split straight from the ordered SPADL actions:
# an action is positive when the SAME team scores a successful shot, or the opponent scores an own goal,
# within the following 10 actions. Actions without a full 10-action future are left out.
# Then it runs three audits (leakage from the current action, independence from the current goal, and a
# comparison with the old target) and saves the target, the audit table and the comparison.
# Speed: every match is converted to SPADL ONCE and reused (the original reloaded and reconverted the events of a
# match again for every positive row, which meant thousands of repeated conversions). The 10-action window is
# computed with cumulative sums instead of a Python loop. The saved files are identical.
# NOTE: like the original, the loader is created without getter="local", so socceraction fetches the open data
# through statsbombpy (this is why statsbombpy is in requirements.txt). Set STATSBOMB_GETTER=local to read the
# json files under <project>/data instead.
# ==============================================================================

import os  # environment variable for the getter

import common  # shared helpers (also holds the legacy numpy patch)
from common import DATA_DIR, OUTPUT_DIR  # shared paths

common.patch_numpy_legacy_types()  # must run BEFORE socceraction is imported (np.long etc. are gone in numpy 2.x)

import numpy as np  # arrays
import pandas as pd  # tables
from socceraction.data.statsbomb import StatsBombLoader  # reads StatsBomb events
from socceraction import spadl  # SPADL constants and helpers
import common_spadl as cs  # window_labels (cumulative-sum version of the 10-action window)

MATCH_SPLIT_FILE = OUTPUT_DIR / "final_common_match_split.csv"  # frozen 34-match split
CORRECTED_TARGET_FILE = OUTPUT_DIR / "rq1_literal_next10_target_34_matches.csv"  # the target this script writes
AUDIT_FILE = OUTPUT_DIR / "07C_target_audit.csv"  # per-match counts
COMPARISON_FILE = OUTPUT_DIR / "07C_old_vs_new_target_comparison.csv"  # old versus new target
OLD_TARGET_FILE = OUTPUT_DIR / "vaep_common_34_matches.csv"  # the earlier (non-literal) target
COMPETITION_ID, SEASON_ID = 11, 90  # competition and season that hold the 34 matches
HORIZON = 10  # look-ahead window in actions
EXPECTED_MATCHES = 34  # the thesis prototype uses exactly 34 matches
GETTER = os.environ.get("STATSBOMB_GETTER", "remote")  # "remote" is what the original did


def main():  # runs the whole step from start to finish
    common.verify_environment()  # optional interpreter check
    print("=" * 80)
    print("07C — REBUILD LITERAL NEXT-10 RQ1 TARGET")  # print the step title
    print("=" * 80)  # print a rule under the title

    # ---------------- STEP 1: frozen split ----------------
    print("\nSTEP 1 — Loading frozen 34-match split")
    split = pd.read_csv(MATCH_SPLIT_FILE)  # split table
    missing = [c for c in ["game_id", "split"] if c not in split.columns]  # required columns
    if missing:
        raise ValueError(f"Missing required columns in {MATCH_SPLIT_FILE}: {missing}")  # stop if the split file lacks a needed column
    match_ids = sorted(split["game_id"].astype(int).unique())  # sorted unique match ids
    print("Matches found:", len(match_ids))
    print("Match IDs:", match_ids)  # show the match ids
    if len(match_ids) != EXPECTED_MATCHES:  # check the number of matches against what is expected
        raise ValueError(f"Expected {EXPECTED_MATCHES} matches, found {len(match_ids)}.")  # stop if the match count is wrong

    # ---------------- STEP 2: StatsBomb data ----------------
    print("\nSTEP 2 — Loading StatsBomb data")
    loader = StatsBombLoader(getter=GETTER, root=str(DATA_DIR)) if GETTER == "local" else StatsBombLoader(root=str(DATA_DIR))  # same call as the original unless local is requested
    print("StatsBombLoader created.")
    games_metadata = loader.games(competition_id=COMPETITION_ID, season_id=SEASON_ID)  # team ids for every match
    home_by_game = {}  # game_id -> home team id
    for gid, home in zip(games_metadata["game_id"], games_metadata["home_team_id"]):
        home_by_game.setdefault(int(gid), int(home))  # first row wins, like iloc[0]

    success_id = spadl.config.results.index("success")  # SPADL id of a successful result
    owngoal_id = spadl.config.results.index("owngoal")  # SPADL id of an own goal
    shot_type_id = spadl.config.actiontypes.index("shot")  # SPADL id of a shot

    # ---------------- STEP 3 and 4: build the target for every match (events converted once) ----------------
    print("\nSTEP 3 — Processing 34 matches")
    all_targets, audit_rows = [], []  # results
    goal_flags = {}  # game_id -> boolean array "this action is a successful shot" (reused by the audits below)
    for counter, game_id in enumerate(match_ids, start=1):  # one match at a time
        print(f"\n[{counter:02d}/34] Processing match {game_id}")
        events = loader.events(game_id)  # raw events (read once)
        print("Raw events:", len(events))
        if game_id not in home_by_game:  # check that the match has metadata
            raise ValueError(f"Game data missing for ID {game_id}")  # stop if the metadata is missing
        home_team_id = home_by_game[game_id]  # home team of this match
        actions = spadl.statsbomb.convert_to_actions(events, home_team_id)  # SPADL actions
        actions = spadl.add_names(actions)  # readable names
        print("SPADL actions:", len(actions))
        type_arr = actions["type_id"].to_numpy()  # positional arrays, the same order as after reset_index
        result_arr = actions["result_id"].to_numpy()
        team_arr = actions["team_id"].to_numpy()  # team of each action as an array
        goal_flags[game_id] = (type_arr == shot_type_id) & (result_arr == success_id)  # successful shots
        if len(actions) <= HORIZON:  # no action has a full future window
            print("WARNING: fewer than or equal to 10 actions. No complete future-10 observations.")
            audit_rows.append({"game_id": game_id, "raw_events": len(events), "spadl_actions": len(actions),  # store a zero-count audit row (continues on the next line)
                               "eligible_observations": 0, "positive": 0, "negative": 0})
            continue  # go to the next match, nothing to label here
        eligible, team_goal, opp_own = cs.window_labels(type_arr, result_arr, team_arr, HORIZON, shot_type_id, success_id, owngoal_id)  # window labels for every action
        idx = np.flatnonzero(eligible)  # actions 0 .. n-11
        target = pd.DataFrame({  # same columns and dtypes as the loop version
            "action_index": idx,
            "team_id": team_arr[idx],
            "target_score_next10_literal": (team_goal[idx] | opp_own[idx]).astype(int),
            "future_team_goal": team_goal[idx].astype(int),
            "future_opponent_own_goal": opp_own[idx].astype(int),
        })
        target["game_id"] = game_id  # add the match id column
        target["period_id"] = actions.loc[target["action_index"], "period_id"].values  # label-based lookup, as before
        target["time_seconds"] = actions.loc[target["action_index"], "time_seconds"].values
        target = target[["game_id", "action_index", "period_id", "time_seconds", "team_id",  # keep the final columns in the saved order (continues on the next line)
                         "target_score_next10_literal", "future_team_goal", "future_opponent_own_goal"]]
        all_targets.append(target)  # keep this match's target table
        positive = int(target["target_score_next10_literal"].sum())  # positives in this match
        negative = int(len(target) - positive)  # negatives in this match
        print("Eligible observations:", len(target))  # show how many observations are eligible
        print("Positive:", positive)  # show the positive count
        print("Negative:", negative)  # show the negative count
        audit_rows.append({"game_id": game_id, "raw_events": len(events), "spadl_actions": len(actions),  # store the audit row of this match (continues on the next line)
                           "eligible_observations": len(target), "positive": positive, "negative": negative})

    print("\nSTEP 4 — Combining target data")  # announce the combining step
    if not all_targets:  # check that at least one match produced a target
        raise RuntimeError("No target data was generated.")  # stop if no target was produced
    target_df = pd.concat(all_targets, ignore_index=True)  # all matches
    audit_df = pd.DataFrame(audit_rows)  # per-match counts

    # ---------------- STEP 5: basic audit ----------------
    print("\n" + "=" * 80)
    print("STEP 5 — TARGET AUDIT")  # print a rule before the audit title
    print("=" * 80)  # print the audit title
    print("\nRows:", len(target_df))  # show the number of rows
    print("Matches:", target_df["game_id"].nunique())  # show the number of matches
    print("\nTarget distribution:")  # announce the label counts
    print(target_df["target_score_next10_literal"].value_counts().sort_index())  # show how often each label value occurs
    positive_total = int(target_df["target_score_next10_literal"].sum())  # total positives over all matches
    negative_total = int(len(target_df) - positive_total)  # total negatives over all matches
    positive_rate = positive_total / len(target_df) if len(target_df) > 0 else 0  # share of positives
    print("\nPositive:", positive_total)  # show the total positives
    print("Negative:", negative_total)  # show the total negatives
    print("Positive rate:", f"{positive_rate:.6%}")  # show the positive rate as a percentage

    # ---------------- STEP 6: current-action leakage test ----------------
    print("\n" + "=" * 80)
    print("STEP 6 — CURRENT-ACTION LEAKAGE TEST")  # print the leakage test title
    print("=" * 80)  # print a rule under the title
    target_by_game = {g: grp["target_score_next10_literal"].to_numpy() for g, grp in target_df.groupby("game_id")}  # per-match label arrays in action order
    n_goal_rows, n_goal_true = 0, 0  # counts for the two printed numbers
    for game_id in match_ids:
        flags = goal_flags[game_id]  # successful shots of this match
        n_eligible = len(flags) - HORIZON  # eligible actions are 0 .. n-11
        if n_eligible <= 0 or game_id not in target_by_game:  # nothing eligible in this match
            continue
        goal_idx = np.flatnonzero(flags[:n_eligible])  # eligible actions that are themselves goals
        n_goal_rows += len(goal_idx)
        n_goal_true += int((target_by_game[game_id][goal_idx] == 1).sum())  # ... and whose target is also 1
    print("Eligible current successful shots:", n_goal_rows)
    print("Current successful shots with literal target=True:", n_goal_true)  # show how many eligible goal actions have a True target
    print("IMPORTANT: This number can be greater than zero only when the same team scores AGAIN within the following 10 actions.")  # explain that the same team may score again within the window

    # ---------------- STEP 7: target independence check ----------------
    print("\n" + "=" * 80)
    print("STEP 7 — TARGET INDEPENDENCE CHECK")  # print the independence check title
    print("=" * 80)  # print a rule under the title
    positives = target_df[target_df["target_score_next10_literal"] == 1]  # positive observations
    non_current_goal_positive = 0  # positives where the action itself is not a goal
    for game_id, grp in positives.groupby("game_id"):  # one match at a time
        non_current_goal_positive += int((~goal_flags[int(game_id)][grp["action_index"].to_numpy()]).sum())
    print("Positive observations where current action is NOT a goal:", non_current_goal_positive)  # show positives whose own action is not a goal
    print("Total positive observations:", positive_total)  # show the total number of positives

    # ---------------- STEP 8: old versus new target ----------------
    print("\n" + "=" * 80)
    print("STEP 8 — OLD VS NEW TARGET COMPARISON")  # print the old-versus-new title
    print("=" * 80)  # print a rule under the title
    old_df = pd.read_csv(OLD_TARGET_FILE, usecols=["game_id", "action_index", "target_score_next10"])  # only the three needed columns
    comparison = target_df.merge(old_df, on=["game_id", "action_index"], how="left")  # attach the old label
    comparison["old_target"] = comparison["target_score_next10"].astype(bool)
    comparison["new_target"] = comparison["target_score_next10_literal"].astype(bool)  # new label as a boolean
    comparison["target_changed"] = comparison["old_target"] != comparison["new_target"]  # flag the rows where the label changed
    print("\nRows compared:", len(comparison))  # show how many rows were compared
    print("Rows where old and new target differ:", int(comparison["target_changed"].sum()))  # show how many labels changed
    print("\nOld target distribution:\n", comparison["old_target"].value_counts())  # show the old label counts
    print("\nNew literal target distribution:\n", comparison["new_target"].value_counts())  # show the new label counts

    # ---------------- STEP 9: save ----------------
    print("\n" + "=" * 80)
    print("STEP 9 — SAVING OUTPUTS")  # print the saving title
    print("=" * 80)  # print a rule under the title
    target_df.to_csv(CORRECTED_TARGET_FILE, index=False)  # save the corrected target
    audit_df.to_csv(AUDIT_FILE, index=False)  # save the per-match audit
    comparison.to_csv(COMPARISON_FILE, index=False)  # save the old-versus-new comparison
    print("\nSaved:", CORRECTED_TARGET_FILE)  # show the path of the target file
    print("Saved:", AUDIT_FILE)  # show the path of the audit file
    print("Saved:", COMPARISON_FILE)  # show the path of the comparison file

    print("\n" + "=" * 80)  # print a rule before the closing message
    print("07C COMPLETED")  # print that the step is complete
    print("=" * 80)  # print a rule under it
    print("\nIMPORTANT:\nThe new target was reconstructed directly from ordered SPADL actions.\n"  # print the reminder to audit before training (continues on the next lines)
          "Observations without a complete 10-action future horizon are excluded.\n"
          "DO NOT TRAIN VAEP OR GAT UNTIL THE 07C OUTPUT HAS BEEN AUDITED.")
    print("=" * 80)  # print a closing rule
    return 0  # success


if __name__ == "__main__":  # only when run directly
    common.run_main(main, "STEP 07C")  # shared error handling
