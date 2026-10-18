# ==============================================================================
# STEP 26 (file 25_error_analysis_vaep_vs_gat_n5.py)
# ERROR ANALYSIS AT N=5: WHERE DOES VAEP BEAT GAT, AND WHERE DO THEY DISAGREE?
# ==============================================================================
# VAEP = Step 21 N=5 validation predictions
# GAT  = Step 23 count8_add_dist_to_actor validation predictions (N=5)
# Same analysis as the N=10 script, only the input files and the output prefix (26) change.
# Usage:  python 25_error_analysis_vaep_vs_gat_n5.py [project_root]
# ==============================================================================

import sys  # command-line argument and sys.exit
from pathlib import Path  # path handling

import common  # shared paths
from common_error import run_error_analysis  # parts A to G

project_root = Path(sys.argv[1]) if len(sys.argv) > 1 else common.PROJECT_ROOT  # where output/ lives
output_dir = project_root / "output"  # all inputs and outputs are here


def find_first(names, desc):
    """Return the first existing file from a list of candidate names, or stop with a helpful list."""
    for n in names:  # in the given order
        if (output_dir / n).exists():  # found one
            return output_dir / n  # use it
    print(f"[MISSING] {desc}. Tried: {names}")  # nothing matched
    print("   Files with 'predictions' in output/:")  # help the user
    for f in sorted(output_dir.glob("*predictions*.csv")):  # list candidates
        print("     ", f.name)  # show
    sys.exit(1)  # stop


vaep_file = find_first(["21_label_horizon_N5_validation_predictions.csv",  # VAEP N=5, name 1
                        "21_vaep_label_horizon_N5_validation_predictions.csv",  # name 2
                        "21_horizon_N5_validation_predictions.csv"], "VAEP N=5 predictions")  # name 3
gat_file = find_first(["23_gat_features_at_horizon5_count8_add_dist_to_actor_at_N5_validation_predictions.csv",  # GAT N=5, name 1
                       "23_gat_features_at_horizon5_count8_add_dist_to_actor_validation_predictions.csv"], "GAT N=5 count8 predictions")  # name 2
print("VAEP file:", vaep_file.name)  # confirm choice
print("GAT file :", gat_file.name)  # confirm choice

run_error_analysis(vaep_file, gat_file, output_dir, prefix="26",  # output files keep the 26A..26G names
                   sanity_note="must match Step 24 at N=5: VAEP 0.8801, GAT 0.8289")  # expected headline numbers
