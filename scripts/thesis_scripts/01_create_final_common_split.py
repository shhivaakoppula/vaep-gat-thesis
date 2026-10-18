# ==============================================================================
# 01_create_final_common_split.py (printed as "06") - FINAL COMMON TRAIN / VALIDATION / TEST SPLIT
# ==============================================================================
# Takes the common 34-match dataset, looks up each match date, sorts the matches by date and cuts
# them into 24 train, 5 validation and 5 test matches (oldest to newest, so the test set is the
# latest). It writes the match-level manifest and one CSV per split.
# Speed: the match json files are scanned until all 34 matches are found (the original always
# read every file), and the split CSVs are written from one pass over the data.
# ==============================================================================

import json  # reading the match json files

import pandas as pd  # tables

import common  # shared helpers
from common import PROJECT_ROOT, OUTPUT_DIR, DATA_DIR  # shared paths

MATCHES_DIR = DATA_DIR / "matches"  # StatsBomb match lists, one json per competition and season
COMMON_FILE = OUTPUT_DIR / "vaep_common_34_matches.csv"  # the dataset to split
MANIFEST_FILE = OUTPUT_DIR / "final_common_match_split.csv"  # match-level split table
TRAIN_FILE = OUTPUT_DIR / "vaep_final_train.csv"  # rows of the training matches
VALIDATION_FILE = OUTPUT_DIR / "vaep_final_validation.csv"  # rows of the validation matches
TEST_FILE = OUTPUT_DIR / "vaep_final_test.csv"  # rows of the test matches


def read_match_records(match_ids):
    """Collect game_id, date and team names for the wanted matches from the json files."""
    wanted = set(match_ids)  # fast membership test
    records = []  # one dict per hit
    found = set()  # ids seen so far
    json_files = list(MATCHES_DIR.rglob("*.json"))  # rglob also searches subfolders like matches/11/90.json
    print("\nMatch JSON files found:", len(json_files))  # how many files exist
    for file in json_files:  # one competition-season file at a time
        if found >= wanted:  # every wanted match has been seen already
            break  # no need to open the remaining files
        try:  # a broken file must not stop the run
            with open(file, "r", encoding="utf-8") as f:  # open the file
                data = json.load(f)  # parse it
            for match in (data if isinstance(data, list) else [data]):  # a file holds a list of matches
                match_id = match.get("match_id")  # id of this match
                if match_id in wanted:  # only the 34 common matches matter
                    home_data, away_data = match.get("home_team"), match.get("away_team")  # StatsBomb nests the team data
                    records.append({"game_id": match_id, "match_date": match.get("match_date"),  # keep id and date
                                    "home_team": home_data.get("home_team_name") if isinstance(home_data, dict) else match.get("home_team"),  # home name
                                    "away_team": away_data.get("away_team_name") if isinstance(away_data, dict) else match.get("away_team")})  # away name
                    found.add(match_id)  # remember it
        except Exception as e:  # report and carry on
            print("Could not read:", file.name, "|", e)  # say which file failed
    return records  # may be empty if nothing was found


def main():  # runs the whole step from start to finish
    common.verify_environment()  # optional interpreter check
    print("=" * 80)  # top rule
    print("06 — CREATE FINAL COMMON TRAIN / VALIDATION / TEST SPLIT")  # title
    print("=" * 80)  # bottom rule

    # ---------------- 1. load the common dataset ----------------
    df = pd.read_csv(COMMON_FILE)  # the 34-match dataset
    print("\nCommon dataset:")  # label
    print("Rows       :", len(df))  # number of rows
    print("Columns    :", len(df.columns))  # number of columns
    print("Matches    :", df["game_id"].nunique())  # number of matches
    match_ids = sorted(df["game_id"].unique())  # sorted match ids
    print("\nCommon match IDs:")  # label
    print(match_ids)  # the ids
    assert len(match_ids) == 34, "Expected exactly 34 common matches."  # the thesis uses 34 matches

    # ---------------- 2. match dates ----------------
    match_records = read_match_records(match_ids)  # scan the json files
    if not match_records:  # json scan found nothing, so ask the loader instead
        print("\n[ℹ] JSON file extraction empty. Triggering StatsBomb API Fallback Loader...")  # tell the user
        from socceraction.data.statsbomb import StatsBombLoader  # imported only when needed
        loader = StatsBombLoader(root=str(DATA_DIR))  # reads the local open-data folder
        games_meta = loader.games(competition_id=11, season_id=90)  # the competition and season of the 34 matches
        filtered_meta = games_meta[games_meta["game_id"].isin(match_ids)]  # keep the 34 matches
        for _, row in filtered_meta.iterrows():  # few rows, so iterrows is fine here
            match_records.append({"game_id": int(row["game_id"]), "match_date": row["game_date"],
                                  "home_team": row.get("home_team_id"), "away_team": row.get("away_team_id")})  # same fields as the json path
    matches = pd.DataFrame(match_records)  # one row per hit
    if matches.empty:  # nothing found anywhere
        raise ValueError("Critical Error: Unable to resolve metadata or match paths for your common games.")  # stop
    if "match_date" not in matches.columns:  # guard against a missing column
        matches["match_date"] = pd.NA  # add an empty one so the checks below can report it

    # ---------------- 3. remove duplicates ----------------
    matches = matches.drop_duplicates(subset=["game_id"])  # keep the first record of each match
    matches["match_date"] = pd.to_datetime(matches["match_date"], errors="coerce")  # text to dates
    print("\nMatched metadata:")  # label
    print("Matches with dates:", matches["match_date"].notna().sum())  # how many have a date
    print("Matches without dates:", matches["match_date"].isna().sum())  # how many do not

    # ---------------- 4. all 34 need a date ----------------
    missing_dates = sorted(set(match_ids) - set(matches["game_id"]))  # matches absent from the metadata
    if missing_dates:  # at least one is absent
        print("\nERROR — These matches were not found:")  # report
        print(missing_dates)  # which ones
        raise ValueError("Not all common matches have match metadata.")  # stop
    if matches["match_date"].isna().any():  # a date could not be parsed
        print("\nERROR — Some matches have missing dates:")  # report
        print(matches[matches["match_date"].isna()])  # which ones
        raise ValueError("Missing match dates.")  # stop

    # ---------------- 5. chronological order ----------------
    matches = matches.sort_values(by=["match_date", "game_id"]).reset_index(drop=True)  # oldest first, ties by id
    show_cols = ["game_id", "match_date", "home_team", "away_team"]  # columns to print
    print("\n" + "=" * 80)  # top rule
    print("CHRONOLOGICAL MATCH ORDER")  # title
    print("=" * 80)  # bottom rule
    print(matches[show_cols].to_string(index=False))  # the ordered list

    # ---------------- 6. 24 / 5 / 5 split ----------------
    train_matches = matches.iloc[:24].copy()  # oldest 24
    validation_matches = matches.iloc[24:29].copy()  # next 5
    test_matches = matches.iloc[29:34].copy()  # newest 5
    train_matches["split"] = "train"  # label the groups
    validation_matches["split"] = "validation"
    test_matches["split"] = "test"  # the remaining 5 matches become the test set
    split_manifest = pd.concat([train_matches, validation_matches, test_matches], ignore_index=True)  # one table

    # ---------------- 7. print the split ----------------
    print("\n" + "=" * 80)  # top rule
    print("FINAL MATCH SPLIT")  # title
    print("=" * 80)  # bottom rule
    for title, part in (("TRAIN", train_matches), ("VALIDATION", validation_matches), ("TEST", test_matches)):  # three blocks
        print(f"\n{title} MATCHES:", len(part))  # block title with count
        print(part[show_cols].to_string(index=False))  # its matches

    # ---------------- 8. temporal order ----------------
    print("\n" + "=" * 80)  # top rule
    print("TEMPORAL ORDER CHECK")  # title
    print("=" * 80)  # bottom rule
    print("Latest train date      :", train_matches["match_date"].max())  # last training date
    print("Earliest validation date:", validation_matches["match_date"].min())  # first validation date
    print("Latest validation date :", validation_matches["match_date"].max())  # last validation date
    print("Earliest test date     :", test_matches["match_date"].min())  # first test date
    assert train_matches["match_date"].max() <= validation_matches["match_date"].min()  # train ends before validation starts
    assert validation_matches["match_date"].max() <= test_matches["match_date"].min()  # validation ends before test starts
    print("\nChronological ordering: PASS")  # confirmation

    # ---------------- 9. target distribution ----------------
    print("\n" + "=" * 80)  # top rule
    print("TARGET DISTRIBUTION BY SPLIT")  # title
    print("=" * 80)  # bottom rule
    split_lookup = dict(zip(split_manifest["game_id"], split_manifest["split"]))  # game_id -> split
    df["split"] = df["game_id"].map(split_lookup)  # tag every row
    assert df["split"].notna().all()  # no row may be left without a split
    summary = df.groupby("split")["target_score_next10"].agg(observations="size", positives="sum")  # counts per split
    summary["negatives"] = summary["observations"] - summary["positives"]  # negatives
    summary["positive_rate"] = summary["positives"] / summary["observations"]  # share of positives
    print(summary)  # the table

    # ---------------- 10. integrity ----------------
    print("\n" + "=" * 80)  # top rule
    print("SPLIT INTEGRITY CHECK")  # title
    print("=" * 80)  # bottom rule
    print("Train matches      :", train_matches["game_id"].nunique())  # 24
    print("Validation matches :", validation_matches["game_id"].nunique())  # 5
    print("Test matches       :", test_matches["game_id"].nunique())  # 5
    print("Total unique       :", split_manifest["game_id"].nunique())  # 34
    assert train_matches["game_id"].nunique() == 24
    assert validation_matches["game_id"].nunique() == 5  # stop if the validation set is not exactly 5 matches
    assert test_matches["game_id"].nunique() == 5  # stop if the test set is not exactly 5 matches
    assert split_manifest["game_id"].nunique() == 34  # stop if the manifest does not cover exactly 34 matches
    print("\nSplit integrity: PASS")  # confirmation

    # ---------------- 11 and 12. save ----------------
    split_manifest.to_csv(MANIFEST_FILE, index=False)  # match-level manifest
    for name, path in (("train", TRAIN_FILE), ("validation", VALIDATION_FILE), ("test", TEST_FILE)):  # one file per split
        df[df["split"] == name].to_csv(path, index=False)  # rows of that split, same column order as the input plus "split"

    # ---------------- 13. summary ----------------
    print("\n" + "=" * 80)  # top rule
    print("FILES SAVED")  # title
    print("=" * 80)  # bottom rule
    print(f"Manifest File  : {MANIFEST_FILE}")  # paths
    print(f"Train File     : {TRAIN_FILE}")
    print(f"Validation File: {VALIDATION_FILE}")  # show where the validation list was saved
    print(f"Test File      : {TEST_FILE}")  # show where the test list was saved
    print("\n" + "=" * 80)  # top rule
    print("06 COMPLETED")  # done
    print("=" * 80)  # bottom rule
    print("\nIMPORTANT:\nDo NOT train VAEP or GAT yet.\nFirst review the split dates and target distribution.")  # reminder
    print("=" * 80)  # bottom rule
    return 0  # success


if __name__ == "__main__":  # only when run directly
    common.run_main(main, "STEP 06")  # shared error handling
