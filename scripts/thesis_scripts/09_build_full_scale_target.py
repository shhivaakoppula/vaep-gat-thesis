# ==============================================================================
# STEP 12 (file 09_build_full_scale_target.py) - BUILD THE FULL-SCALE FROZEN TARGET
# ==============================================================================
# 1. Builds and saves the train/validation split (292 training and 134 validation matches) from complete_matches.csv.
# 2. Converts every match to SPADL actions and labels each action: does the SAME team score a successful shot,
#    or does the opponent score an own goal, inside the next 10 actions? Only actions with a full 10-action
#    future window are kept (eligibility rule verified in the prototype).
# 3. Checks that the action just before every real goal is labelled positive.
# Speed/memory changes compared with the original: the window walk uses cumulative sums, and the goal check is
# done per match straight after the labels are built, so the raw actions of all 426 matches no longer have to be
# kept in memory. Counts and files are the same.
# ==============================================================================

import time  # progress timing
import pandas as pd  # tables

import common  # paths and helpers
from common import OUTPUT_DIR, print_header, require_file  # shared pieces
import common_spadl as cs  # loader, SPADL ids, window labelling

COMPLETE_MATCHES_FILE = OUTPUT_DIR / "complete_matches.csv"  # list of matches that have everything needed
FULL_TARGET_FILE = OUTPUT_DIR / "rq1_literal_next10_target_full_dataset.csv"  # the target table
FULL_SPLIT_FILE = OUTPUT_DIR / "full_scale_match_split.csv"  # train/validation assignment per match
GOAL_VERIFICATION_FILE = OUTPUT_DIR / "12_full_scale_goal_verification.csv"  # result of the goal check
SUMMARY_FILE = OUTPUT_DIR / "12_full_scale_target_summary.csv"  # one-row summary
NEXT_N_ACTIONS = 10  # label horizon
TRAIN_COMPETITIONS = ["UEFA Euro", "UEFA Women's Euro", "FIFA World Cup", "Women's World Cup"]  # international tournaments -> training
VALIDATION_COMPETITIONS = ["1. Bundesliga", "Ligue 1", "La Liga", "Major League Soccer", "African Cup of Nations"]  # domestic leagues -> validation
TARGET_COLUMN = "target_score_next10_literal"  # name of the label column


def build_full_split():
    """Split matches by competition type and save the split table."""
    print_header("BUILD AND DOCUMENT MY FULL-SCALE TRAIN/VALIDATION SPLIT")  # title
    require_file(COMPLETE_MATCHES_FILE, "Complete matches list")  # must exist
    matches = pd.read_csv(COMPLETE_MATCHES_FILE)  # all usable matches
    train_matches = matches[matches["competition_name"].isin(TRAIN_COMPETITIONS)].copy()  # tournaments
    validation_matches = matches[matches["competition_name"].isin(VALIDATION_COMPETITIONS)].copy()  # leagues
    for label, part in (("TRAINING", train_matches), ("VALIDATION", validation_matches)):  # same report for both
        print(f"{label} SET - every competition and season:")  # heading
        for (comp, season), group in part.groupby(["competition_name", "season_name"]):  # each competition-season
            print(f"  {comp} {season}: {len(group)} matches")  # count
        print(f"  TOTAL {label}: {len(part)} matches\n")  # total
    print("Note: UEFA Euro 2024 and Women's Euro 2025 are later than the Bundesliga 2023/24 validation matches; "
          "the split is by competition type, not by a calendar cut-off.")  # honest remark kept from the original
    split_df = pd.DataFrame([{"game_id": int(r["match_id"]), "split": "train"} for _, r in train_matches.iterrows()]  # train rows
                            + [{"game_id": int(r["match_id"]), "split": "validation"} for _, r in validation_matches.iterrows()])  # then validation rows
    split_df.to_csv(FULL_SPLIT_FILE, index=False)  # save
    print(f"Full-scale split saved to:\n{FULL_SPLIT_FILE}")  # where
    return train_matches, validation_matches, split_df  # tables


def build_target_for_match(loader, game_id, home_team_id, ids):
    """Label one match. Returns (target table, number of goal-check passes, failures, failed keys)."""
    shot_type_id, success_result_id, owngoal_result_id = ids  # SPADL ids
    actions = cs.load_actions(loader, game_id, home_team_id)  # events -> actions
    eligible, future_team_goal, future_opp_own = cs.window_labels(  # label the next-10 window of every action
        actions["type_id"].to_numpy(), actions["result_id"].to_numpy(), actions["team_id"].to_numpy(),
        NEXT_N_ACTIONS, shot_type_id, success_result_id, owngoal_result_id)
    target = future_team_goal | future_opp_own  # positive if either event happens
    result_df = pd.DataFrame({  # one row per action, then filtered to eligible ones
        "game_id": int(game_id), "action_index": actions["action_index"].to_numpy(),  # keys
        "period_id": actions["period_id"].to_numpy(), "time_seconds": actions["time_seconds"].to_numpy(),  # when
        "team_id": actions["team_id"].to_numpy(), "type_id": actions["type_id"].to_numpy(), "result_id": actions["result_id"].to_numpy(),  # who/what
        TARGET_COLUMN: target, "future_team_goal": future_team_goal, "future_opponent_own_goal": future_opp_own,  # labels
    })
    result_df = result_df[eligible].reset_index(drop=True)  # keep only actions with a full window

    # Goal check: the action right before every successful shot must be positive (if it is eligible at all).
    lookup = dict(zip(result_df["action_index"].to_numpy(), result_df[TARGET_COLUMN].to_numpy()))  # action_index -> label
    is_goal = ((actions["type_id"] == shot_type_id) & (actions["result_id"] == success_result_id)).to_numpy()  # real goals
    ai = actions["action_index"].to_numpy()  # action_index per row position
    passed = failed = 0  # counters for this match
    failed_keys = []  # details of any failure
    for pos in map(int, is_goal.nonzero()[0]):  # position of each goal
        if pos == 0:  # no action before it
            continue
        prior = int(ai[pos - 1])  # the action just before the goal
        if prior not in lookup:  # too close to the end to be eligible: expected, not a failure
            continue
        if lookup[prior]:  # correctly positive
            passed += 1
        else:  # wrongly negative
            failed += 1
            failed_keys.append((int(game_id), prior))  # remember which goal had no prior action to compare with
    return result_df, passed, failed, failed_keys  # everything the caller needs


def main():  # runs the whole step from start to finish
    common.verify_environment()  # optional interpreter check
    print_header("STEP 12 - BUILD MY FULL-SCALE FROZEN TARGET")  # title
    train_matches, validation_matches, _ = build_full_split()  # split
    matches_df = pd.concat([train_matches, validation_matches], ignore_index=True)  # all matches to process
    loader = cs.create_loader()  # StatsBomb reader
    ids = cs.get_spadl_ids()  # shot / success / owngoal ids

    print_header("BUILD MY FULL-SCALE TARGET - ALL MATCHES")  # title
    match_ids = matches_df["match_id"].tolist()  # matches in order
    home_team_lookup = {}  # game_id -> home_team_id
    for comp_id, season_id in matches_df[["competition_id", "season_id"]].drop_duplicates().values.tolist():  # each competition-season once
        try:  # a season may fail to load
            games = loader.games(competition_id=int(comp_id), season_id=int(season_id))  # its matches
        except Exception as ex:  # keep going
            print(f"  [WARNING] Could not load games for competition {comp_id}, season {season_id}: {ex}")  # report
            continue
        for gid, hid in zip(games["game_id"], games["home_team_id"]):  # every match in the season
            home_team_lookup[int(gid)] = int(hid)  # remember its home team
    print(f"Metadata resolved for {len(home_team_lookup)} of {len(match_ids)} matches.")  # coverage

    start, failed_matches, targets = time.time(), [], []  # timer, failures, per-match tables
    checks_passed = checks_failed = 0  # goal-check totals
    failed_details = []  # first few failing keys
    for number, game_id in enumerate(match_ids, start=1):  # one match at a time
        if game_id not in home_team_lookup:  # no metadata
            print(f"  [SKIP - no metadata] Match {game_id}")  # report
            failed_matches.append(int(game_id))  # remember
            continue
        try:  # a broken match must not stop the run
            result_df, p, f, keys = build_target_for_match(loader, game_id, home_team_lookup[game_id], ids)  # label it
            targets.append(result_df)  # keep the table (the raw actions are dropped here)
            checks_passed += p  # accumulate
            checks_failed += f  # accumulate
            failed_details += keys  # accumulate
        except Exception as ex:  # report and move on
            print(f"  [FAILED] Match {game_id}: {ex}")  # report
            failed_matches.append(int(game_id))  # remember
            continue
        if number % 10 == 0 or number == len(match_ids):  # progress line
            elapsed = time.time() - start  # seconds so far
            rate = number / elapsed if elapsed > 0 else 0  # matches per second
            print(f"  [{number}/{len(match_ids)}] processed. Elapsed {elapsed / 60:.1f} min. "
                  f"Estimated remaining {((len(match_ids) - number) / rate / 60) if rate > 0 else float('nan'):.1f} min.")  # show

    print(f"\nMatches successfully processed: {len(match_ids) - len(failed_matches)} of {len(match_ids)}")  # coverage
    if failed_matches:  # something failed
        print(f"Matches that failed or had no metadata: {failed_matches}")  # list them
    full_target = pd.concat(targets, ignore_index=True)  # one table for everything

    print_header("MY FULL-SCALE TARGET - SUMMARY")  # title
    n_total, n_positive = len(full_target), int(full_target[TARGET_COLUMN].sum())  # counts
    print(f"Total eligible observations: {n_total:,}\nPositive observations: {n_positive:,}")  # counts
    print(f"Positive rate: {n_positive / n_total * 100:.4f}%" if n_total > 0 else "No observations built.")  # rate
    full_target.to_csv(FULL_TARGET_FILE, index=False)  # save the target
    print(f"Full-scale target saved to:\n{FULL_TARGET_FILE}")  # where

    print_header("GOAL-VERIFICATION SANITY CHECK - FULL SCALE")  # title
    print(f"Checks passed: {checks_passed}\nChecks failed: {checks_failed}")  # counts
    print("[PASS] Every real goal is preceded by a positive label." if checks_failed == 0
          else f"[INVESTIGATE] {checks_failed} goals lack a positive prior action: {failed_details[:20]}")  # verdict
    pd.DataFrame({"checks_passed": [checks_passed], "checks_failed": [checks_failed]}).to_csv(GOAL_VERIFICATION_FILE, index=False)  # save

    pd.DataFrame([{  # one-row summary
        "total_matches_requested": len(matches_df), "matches_failed": len(failed_matches),  # coverage
        "total_eligible_observations": n_total, "positive_observations": n_positive,  # counts
        "positive_rate_pct": round(n_positive / n_total * 100, 4) if n_total > 0 else None,  # rate
        "goal_verification_checks_passed": checks_passed, "goal_verification_checks_failed": checks_failed,  # check results
    }]).to_csv(SUMMARY_FILE, index=False)  # save
    print(f"Summary saved to:\n{SUMMARY_FILE}")  # where
    print_header("STEP 12 COMPLETE")  # done
    return 0  # success


if __name__ == "__main__":  # only when run directly
    common.run_main(main, "STEP 12")  # shared error handling
