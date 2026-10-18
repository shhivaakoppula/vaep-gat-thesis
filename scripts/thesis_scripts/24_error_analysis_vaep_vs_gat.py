# ==============================================================================
# STEP 25 (file 24_error_analysis_vaep_vs_gat.py)
# ERROR ANALYSIS AT N=10: WHERE DOES VAEP BEAT GAT, AND WHERE DO THEY DISAGREE?
# ==============================================================================
# VAEP = 13_vaep_full_scale_validation_predictions.csv
# GAT  = 22_gat_feature_count_count8_add_dist_to_actor_validation_predictions.csv
# No retraining, only the saved predictions are read. The analysis itself lives in common_error.py.
# Usage:  python 24_error_analysis_vaep_vs_gat.py            (uses the default project folder)
#         python 24_error_analysis_vaep_vs_gat.py <project_root>
# The original file contained the whole script twice back to back; one copy is enough.
# ==============================================================================

import sys  # command-line argument and sys.exit
from pathlib import Path  # path handling

import common  # shared paths
from common_error import run_error_analysis  # parts A to G

# An optional first argument overrides the project folder, as in the original script.
project_root = Path(sys.argv[1]) if len(sys.argv) > 1 else common.PROJECT_ROOT  # where output/ lives
output_dir = project_root / "output"  # all inputs and outputs are here
vaep_file = output_dir / "13_vaep_full_scale_validation_predictions.csv"  # VAEP N=10 predictions
gat_file = output_dir / "22_gat_feature_count_count8_add_dist_to_actor_validation_predictions.csv"  # GAT N=10 best-feature predictions

for f in (vaep_file, gat_file):  # both inputs are required
    if not f.exists():  # one is missing
        sys.exit(f"[MISSING] {f}")  # stop with the path

run_error_analysis(vaep_file, gat_file, output_dir, prefix="25",  # output files keep the 25A..25G names
                   sanity_note="must match Step 24: VAEP 0.7731, GAT 0.7222")  # expected headline numbers
