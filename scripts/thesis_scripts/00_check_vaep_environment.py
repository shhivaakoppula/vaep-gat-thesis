# ==============================================================================
# STEP 01 (file 00_check_vaep_environment.py) - FINAL ENVIRONMENT CHECK FOR THE PROTOTYPE
# ==============================================================================
# Read-only check before any training starts. It confirms that the required packages import,
# that the prototype files and folders exist, and that the events file has the columns the
# later steps need. The only thing it creates is the (empty) VAEP_Prototype output folder.
# Nothing here is heavy: the events file is read for 5 rows only.
# ==============================================================================

import sys  # interpreter version and exit code
import importlib  # imports a package by its name (a string)

from common import PROJECT_ROOT, OUTPUT_DIR, PROTOTYPE_DIR  # shared paths (same default folder as before)

EVENTS = OUTPUT_DIR / "events_all.csv"  # the big events table
LABELS = PROTOTYPE_DIR / "Output" / "prototype_labels_with_split.csv"  # prototype labels with the split column
GRAPHS = PROTOTYPE_DIR / "prototype_final_graphs"  # folder with the prototype graphs
RESULTS = PROTOTYPE_DIR / "results"  # folder with the prototype GAT results
VAEP_DIR = PROTOTYPE_DIR / "VAEP_Prototype"  # output folder for the prototype VAEP

VAEP_DIR.mkdir(parents=True, exist_ok=True)  # create it if it is not there yet

LINE = "-" * 75  # divider used between sections


def main():  # runs the whole step from start to finish
    print("=" * 75)  # top rule
    print("RQ1 GAT vs VAEP PROTOTYPE")  # project title
    print("STEP 01 - FINAL ENVIRONMENT CHECK")  # step title
    print("=" * 75)  # bottom rule

    # ---------------- 1. python ----------------
    print("\n[1] PYTHON")  # section title
    print(LINE)  # divider
    print("Python version:")  # label
    print(sys.version)  # full version string

    # ---------------- 2. packages ----------------
    print("\n[2] REQUIRED PYTHON PACKAGES")  # section title
    print(LINE)  # divider
    packages = ["pandas", "numpy", "scipy", "sklearn", "torch", "torch_geometric", "socceraction", "xgboost"]  # every package the pipeline needs
    package_failed = False  # becomes True if any import fails
    for package_name in packages:  # check them one by one
        try:  # an import can fail for many reasons
            module = importlib.import_module(package_name)  # import it by name
            version = getattr(module, "__version__", "version unknown")  # most packages expose __version__
            print(f"[OK] {package_name:<20} {version}")  # report success
        except Exception as error:  # broken or missing package
            print(f"[FAIL] {package_name:<20} {error}")  # report the reason
            package_failed = True  # remember the failure

    # ---------------- 3. files and folders ----------------
    print("\n[3] REQUIRED FILES AND FOLDERS")  # section title
    print(LINE)  # divider
    required_paths = {"EVENTS CSV": EVENTS, "PROTOTYPE LABELS": LABELS, "GAT GRAPHS": GRAPHS, "GAT RESULTS": RESULTS, "VAEP DIRECTORY": VAEP_DIR}  # label -> path
    path_failed = False  # becomes True if anything is missing
    for name, path in required_paths.items():  # check each path
        if path.exists():  # present
            print(f"[OK] {name:<20} {'FILE' if path.is_file() else 'FOLDER'}")  # say whether it is a file or a folder
            print(f"     {path}")  # show where it is
        else:  # absent
            print(f"[FAIL] {name:<20}")  # report the failure
            print(f"       {path}")  # show where it was expected
            path_failed = True  # remember the failure

    # ---------------- 4. events file ----------------
    print("\n[4] EVENTS FILE CHECK")  # section title
    print(LINE)  # divider
    if EVENTS.exists():  # only inspect it when it exists
        print(f"File size: {EVENTS.stat().st_size / (1024 ** 3):.4f} GB")  # size on disk in GB
        try:  # reading can fail on a damaged file
            import pandas as pd  # imported here so a pandas problem is caught by this try block
            sample = pd.read_csv(EVENTS, nrows=5)  # five rows are enough to see the columns
            print(f"Columns detected: {len(sample.columns)}")  # how many columns
            print("First columns:")  # label
            for column in sample.columns[:20]:  # list the first twenty
                print(f"  - {column}")  # one per line
            print("\nRequired event columns:")  # label
            for column in ["id", "match_id", "type.name"]:  # the columns later steps rely on
                if column in sample.columns:  # present
                    print(f"[OK] {column}")  # fine
                else:  # absent
                    print(f"[FAIL] {column}")  # problem
                    path_failed = True  # remember the failure
        except Exception as error:  # the file could not be read
            print("[FAIL] Could not read events CSV")  # say so
            print(error)  # and why
            path_failed = True  # remember the failure
    else:  # file missing
        print("[FAIL] events_all.csv does not exist.")  # say so
        path_failed = True  # remember the failure

    # ---------------- 5. existing GAT results ----------------
    print("\n[5] EXISTING GAT PROTOTYPE RESULTS")  # section title
    print(LINE)  # divider
    gat_result_files = ["prototype_gat_test_predictions.csv", "prototype_gat_test_metrics.json", "prototype_gat_roc_curve.csv",
                        "prototype_gat_pr_curve.csv", "prototype_gat_confusion_matrix.csv", "prototype_gat_classification_report.txt"]  # expected result files
    for filename in gat_result_files:  # check each one
        if (RESULTS / filename).exists():  # present
            print(f"[OK] {filename}")  # fine
        else:  # absent (not fatal)
            print(f"[INFO] {filename} not found")  # just informational

    # ---------------- 6. final verdict ----------------
    print("\n" + "=" * 75)  # top rule
    if package_failed or path_failed:  # at least one problem
        print("RESULT: FAILED")  # verdict
        print("\nThere is still an environment/file problem.")  # explanation
        print("Do NOT continue to the VAEP training step.")  # instruction
        print("=" * 75)  # bottom rule
        return 1  # non-zero exit code
    print("RESULT: PASSED")  # verdict
    print("\nEnvironment is ready.")  # message 1
    print("Events file is accessible.")  # message 2
    print("Existing GAT prototype files are accessible.")  # message 3
    print("We can proceed to the next step.")  # message 4
    print("=" * 75)  # bottom rule
    return 0  # success


if __name__ == "__main__":  # only when run directly
    sys.exit(main())  # exit with the verdict code
