# ==============================================================================
# 19_check_which_scripts_are_load_bearing.py - SAFETY CHECK BEFORE ARCHIVING PROTOTYPE SCRIPTS
# ==============================================================================
# Read-only. It does not delete or move anything. It lists the files the confirmed chain
# (06 to 11) depends on and then lists every script in the "VAEP Prototype" folder by
# modification time, so I can decide myself what to archive after the full-scale run works.
# ==============================================================================

import sys  # exit code
import datetime  # turn file timestamps into readable text

from common import PROJECT_ROOT, OUTPUT_DIR  # shared paths

SCRIPTS_DIR = PROJECT_ROOT / "VAEP Prototype"  # folder with the exploratory scripts

# Files that the confirmed, working scripts read. Never delete these.
CONFIRMED_LOAD_BEARING_OUTPUTS = [
    "rq1_literal_next10_target_34_matches.csv",
    "final_common_match_split.csv",
    "09_vaep_validation_predictions.csv",
    "09_vaep_feature_matrix_34_matches.csv",
    "10_gat_validation_predictions.csv",
    "10_gat_model_state.pt",
    "11_delong_test_result.csv",
    "11_delong_common_predictions.csv",
]


def main():  # runs the whole step from start to finish
    print("=" * 80)  # top rule
    print("MY CONFIRMED LOAD-BEARING FILES (never delete these)")  # title
    print("=" * 80)  # bottom rule
    for filename in CONFIRMED_LOAD_BEARING_OUTPUTS:  # check each file
        path = OUTPUT_DIR / filename  # where it should be
        if path.exists():  # present
            print(f"  [EXISTS] {filename}  ({path.stat().st_size / 1024:.1f} KB)")  # show the size in KB
        else:  # absent
            print(f"  [MISSING] {filename}  <- I should investigate why this is missing before I archive anything")  # warn

    print()  # blank line
    print("=" * 80)  # top rule
    print("EVERY SCRIPT IN MY VAEP PROTOTYPE FOLDER")  # title
    print("=" * 80)  # bottom rule
    print("I am listing every .py file, sorted by when I last modified it. Scripts near the TOP (oldest) are more "
          "likely to be early investigation dead-ends. Scripts near the BOTTOM (newest) are more likely to be my "
          "final, working chain.")  # explanation
    print()  # blank line

    if not SCRIPTS_DIR.exists():  # folder missing
        print(f"[ERROR] I cannot find {SCRIPTS_DIR}")  # say so
        print("I need to check this path is correct.")  # hint
        return 1  # non-zero exit code

    py_files = sorted(SCRIPTS_DIR.glob("*.py"), key=lambda p: p.stat().st_mtime)  # .py files, oldest first
    py_set = set(py_files)  # set lookup is faster than "in list"
    other_files = sorted([p for p in SCRIPTS_DIR.iterdir() if p.is_file() and p not in py_set and p.suffix.lower() not in [".md", ".txt"]],
                         key=lambda p: p.stat().st_mtime)  # oddly named files (spaces, no extension), oldest first
    for script in py_files + other_files:  # print them all
        mtime = datetime.datetime.fromtimestamp(script.stat().st_mtime)  # last modified time
        print(f"  {mtime.strftime('%Y-%m-%d %H:%M')}  {script.name}")  # one line per file

    print()  # blank line
    print("=" * 80)  # top rule
    print("WHAT I DO WITH THIS LIST")  # title
    print("=" * 80)  # bottom rule
    print("I look at the scripts furthest from the bottom five or six (00, 06, 07C, 08A, 08B, 08C, 09, 10, 11 - my "
          "confirmed working chain). Everything with a '05' prefix that is NOT in that confirmed list, I move into a "
          "new folder called _archive_investigation inside VAEP Prototype, rather than deleting it. I do this AFTER my "
          "full-scale run works, not before, so if anything unexpected breaks, I still have every intermediate file "
          "to check.")  # closing note
    return 0  # success


if __name__ == "__main__":  # only when run directly
    sys.exit(main())  # exit with the code
