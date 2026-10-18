# ==============================================================================
# STEP 24 (file 23_A_delong_best_features_both_horizons.py)
# DELONG TEST: BEST-FEATURE GAT (count8_add_dist_to_actor) vs VAEP, AT N=10 AND N=5
# ==============================================================================
# Nothing is retrained. The four saved validation-prediction files are located, paired by
# (game_id, action_index), checked for label agreement, and tested with the paired DeLong test.
# ==============================================================================

import sys  # for sys.exit
import pandas as pd  # reading and merging predictions

import common  # paths and helpers
from common import OUTPUT_DIR, print_header  # shared pieces
from common_stats import calculate_delong_p_value  # the DeLong test

SUMMARY_FILE = OUTPUT_DIR / "24_delong_best_features_both_horizons_summary.csv"  # where the two result rows go
REQUIRED_COLS = ["game_id", "action_index", "true_target", "predicted_probability"]  # every file needs these


def find_one(patterns, description):
    """Return the first file in OUTPUT_DIR matching any pattern (tried in order), or None with a clear message."""
    for pattern in patterns:  # try each pattern in the given order
        matches = sorted(OUTPUT_DIR.glob(pattern))  # alphabetical list of hits
        if matches:  # something matched
            if len(matches) > 1:  # more than one hit: say which one is used
                print(f"[INFO] {description}: several files matched '{pattern}', using {matches[0].name}")  # note
                for m in matches:  # list all hits
                    print(f"       also found: {m.name}")  # show them
            return matches[0]  # first hit wins
    print(f"[MISSING] Could not find {description}. Tried: {patterns}")  # nothing matched
    for f in sorted(OUTPUT_DIR.glob("*predictions*.csv")):  # help the user see what does exist
        print(f"            {f.name}")  # show candidates
    return None  # caller will stop


def run_comparison(horizon_label, vaep_file, gat_file):
    """Pair the two files, check labels, run DeLong, return one result dict (or None on failure)."""
    print_header(f"DELONG TEST - {horizon_label}")  # section title
    vaep_df, gat_df = pd.read_csv(vaep_file), pd.read_csv(gat_file)  # load both
    for name, df in (("VAEP", vaep_df), ("GAT", gat_df)):  # same column check for both
        missing_cols = set(REQUIRED_COLS) - set(df.columns)  # absent columns
        if missing_cols:  # cannot continue
            print(f"[FAIL] {name} file is missing columns {missing_cols}; has {list(df.columns)}")  # explain
            return None  # skip this horizon
    print(f"VAEP rows: {len(vaep_df):,} | GAT rows: {len(gat_df):,}")  # sizes

    merged = pd.merge(vaep_df[REQUIRED_COLS], gat_df[REQUIRED_COLS], on=["game_id", "action_index"], suffixes=("_vaep", "_gat"), how="inner")  # pair rows
    mismatches = int((merged["true_target_vaep"] != merged["true_target_gat"]).sum())  # label disagreements
    print(f"Paired observations: {len(merged):,} | label mismatches: {mismatches}")  # report
    if mismatches > 0:  # different targets were used
        print("[FAIL] The two files disagree on the true label for some paired rows. Stopping.")  # explain
        return None  # skip
    print("[PASS] Both files trace back to the same target on every paired row.")  # reassurance

    n_positive = int(merged["true_target_vaep"].sum())  # positives among paired rows
    print(f"Positive examples: {n_positive:,} ({100 * n_positive / len(merged):.4f}%)")  # rate
    y = merged["true_target_vaep"].to_numpy(dtype=float)  # labels
    auc_vaep, auc_gat, _, z, p = calculate_delong_p_value(  # the test, using the step-24 form of the AUC formula
        y, merged["predicted_probability_vaep"].to_numpy(dtype=float), merged["predicted_probability_gat"].to_numpy(dtype=float), auc_form="step24",
    )
    diff = auc_vaep - auc_gat  # gap
    if p < 0.05 and diff > 0:  # VAEP wins
        verdict = "SIGNIFICANT (p < 0.05) - VAEP significantly outperforms GAT"
    elif p < 0.05 and diff < 0:  # GAT wins
        verdict = "SIGNIFICANT (p < 0.05) - GAT significantly outperforms VAEP"
    else:  # no evidence of a difference
        verdict = "NOT SIGNIFICANT (p >= 0.05) - cannot conclude the models differ"
    print(f"VAEP AUC-ROC {auc_vaep:.4f} | GAT (count8) {auc_gat:.4f} | diff {diff:+.4f} | z {z:.4f} | p {p:.6f}")  # key numbers
    print(f"VERDICT: {verdict}")  # verdict
    return {"horizon": horizon_label, "n_paired": len(merged), "n_positive": n_positive, "auc_vaep": auc_vaep,  # one row of the summary
            "auc_gat_best": auc_gat, "difference": diff, "z_statistic": z, "p_value": p, "verdict": verdict}


def main():  # runs the whole step from start to finish
    common.verify_environment()  # optional interpreter check
    print_header("LOCATING THE FOUR PREDICTION FILES I NEED")  # title
    vaep_n10 = find_one(["13_vaep_full_scale_validation_predictions.csv"], "VAEP @ N=10 (Step 13)")  # N=10 VAEP
    gat_n10 = find_one(["22_gat_feature_count_count8_add_dist_to_actor_validation_predictions.csv"], "GAT @ N=10 count8 (Step 22)")  # N=10 GAT
    vaep_n5 = find_one(["21_label_horizon_N5_validation_predictions.csv", "21_vaep_label_horizon_N5_validation_predictions.csv",
                        "21*N5*predictions*.csv", "21*horizon*5*predictions*.csv"], "VAEP @ N=5 (Step 21)")  # N=5 VAEP
    gat_n5 = find_one(["23_gat_features_at_horizon5_count8_add_dist_to_actor_at_N5_validation_predictions.csv",
                       "23_gat_features_at_horizon5_count8_add_dist_to_actor_validation_predictions.csv",
                       "23*count8*predictions*.csv"], "GAT @ N=5 count8 (Step 23)")  # N=5 GAT
    if any(f is None for f in (vaep_n10, gat_n10, vaep_n5, gat_n5)):  # at least one input missing
        print("STOPPING - a prediction file is missing; DeLong needs per-action probabilities, not just the summary AUC.")  # explain
        return 1  # non-zero exit

    results = []  # collected result rows
    for label, vf, gf in (("N=10 (VAEP vs GAT count8_add_dist_to_actor)", vaep_n10, gat_n10),  # first horizon
                          ("N=5 (VAEP vs GAT count8_add_dist_to_actor)", vaep_n5, gat_n5)):  # second horizon
        r = run_comparison(label, vf, gf)  # run one comparison
        if r:  # keep successful ones only
            results.append(r)  # store
    if not results:  # nothing worked
        print("[FAIL] Neither comparison could be completed.")  # explain
        return 1  # non-zero exit
    pd.DataFrame(results).to_csv(SUMMARY_FILE, index=False)  # save the summary
    print_header("STEP 24 COMPLETE")  # done
    print(f"Summary saved to: {SUMMARY_FILE}")  # where
    return 0  # success


if __name__ == "__main__":  # only when run directly
    common.run_main(main, "STEP 24")  # shared error handling
