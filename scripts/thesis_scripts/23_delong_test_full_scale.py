# ==============================================================================
# STEP 18 (file 23_delong_test_full_scale.py) - PAIRED DELONG TEST: GAT vs VAEP, FULL SCALE (N=10, original 7-feature GAT)
# ==============================================================================
# Reads the saved validation predictions of Step 13 (VAEP) and Step 14 (GAT), keeps the actions that
# both models scored, checks that both files agree on the true label, and runs the paired DeLong test.
# Nothing is retrained. Outputs and numbers are the same as the original 23_delong_test_full_scale.py.
# ==============================================================================

import numpy as np  # arrays for the test
import pandas as pd  # reading and merging the prediction files
from sklearn.metrics import roc_auc_score  # independent AUC used as a safety check

import common  # paths and helpers
from common import OUTPUT_DIR, print_header, require_file  # the bits used below
from common_stats import calculate_delong_p_value  # the (now faster) DeLong test

ALPHA = 0.05  # significance threshold
VAEP_PREDICTIONS_FILE = OUTPUT_DIR / "13_vaep_full_scale_validation_predictions.csv"  # Step 13 output
GAT_PREDICTIONS_FILE = OUTPUT_DIR / "14_gat_full_scale_validation_predictions.csv"  # Step 14 output
VAEP_SUMMARY_FILE = OUTPUT_DIR / "13_vaep_full_scale_results_summary.csv"  # holds the VAEP headline AUC
GAT_SUMMARY_FILE = OUTPUT_DIR / "14_gat_full_scale_results_summary.csv"  # holds the GAT headline AUC
DELONG_RESULT_FILE = OUTPUT_DIR / "18_delong_test_full_scale_result.csv"  # the result row is saved here
COMMON_PREDICTIONS_FILE = OUTPUT_DIR / "18_delong_full_scale_common_predictions.csv"  # paired rows are saved here
FALLBACK_VAEP_HEADLINE_AUC = 0.7746  # used only if the summary file is missing
FALLBACK_GAT_HEADLINE_AUC = 0.7178  # used only if the summary file is missing
KEY_COLUMNS = ["game_id", "action_index", "true_target", "predicted_probability"]  # columns every file must have


def read_headline_auc(summary_file, fallback_value, model_name):
    """Read validation_auc_roc from a results summary; fall back to the recorded number if it is not there."""
    if not summary_file.exists():  # no summary file on disk
        print(f"[NOTE] {model_name} summary missing, using recorded {fallback_value}")  # say so
        return fallback_value, False  # value plus 'not read from file'
    summary = pd.read_csv(summary_file)  # metric/value table
    row = summary[summary["metric"] == "validation_auc_roc"]  # the one row we need
    if len(row) == 0:  # row not present
        print(f"[NOTE] {model_name} summary has no validation_auc_roc row, using recorded {fallback_value}")  # say so
        return fallback_value, False  # fall back
    value = float(row["value"].iloc[0])  # the stored headline AUC
    print(f"[PASS] {model_name} headline AUC-ROC read from {summary_file.name}: {value:.4f}")  # confirm
    return value, True  # value plus 'read from file'


def load_predictions(path, model_name):
    """Load one prediction file and force the four columns to clean numeric types."""
    print_header(f"LOAD {model_name} FULL-SCALE VALIDATION PREDICTIONS")  # section title
    predictions = pd.read_csv(path)  # read the CSV
    missing = set(KEY_COLUMNS) - set(predictions.columns)  # required columns that are absent
    if missing:  # something is wrong with the file
        raise ValueError(f"{model_name} predictions file is missing columns: {sorted(missing)}")  # stop
    predictions = predictions[KEY_COLUMNS].copy()  # keep only the four needed columns
    for column in ("game_id", "action_index", "true_target"):  # integer columns
        predictions[column] = pd.to_numeric(predictions[column], errors="raise").astype(int)  # strict conversion
    predictions["predicted_probability"] = pd.to_numeric(predictions["predicted_probability"], errors="raise")  # float column
    print(f"{model_name} rows: {len(predictions):,} | positive rate: {predictions['true_target'].mean() * 100:.4f}%")  # summary
    return predictions  # cleaned table


def find_common_observations(vaep_predictions, gat_predictions):
    """Inner-join on (game_id, action_index) and make sure both files carry the same true label."""
    print_header("FIND OBSERVATIONS COMMON TO BOTH MODELS")  # section title
    merged = vaep_predictions.rename(columns={"true_target": "true_target_vaep", "predicted_probability": "vaep_probability"}).merge(  # VAEP side
        gat_predictions.rename(columns={"true_target": "true_target_gat", "predicted_probability": "gat_probability"}),  # GAT side
        on=["game_id", "action_index"], how="inner",  # keep only actions both models scored
    )
    print(f"VAEP rows: {len(vaep_predictions):,} | GAT rows: {len(gat_predictions):,} | paired rows: {len(merged):,}")  # sizes
    print(f"Dropped from VAEP side: {len(vaep_predictions) - len(merged):,} | from GAT side: {len(gat_predictions) - len(merged):,}")  # losses
    n_mismatches = int((merged["true_target_vaep"] != merged["true_target_gat"]).sum())  # label disagreements
    print(f"True-label mismatches on common rows: {n_mismatches:,}")  # must be zero
    if n_mismatches > 0:  # the two files were built from different targets
        raise ValueError(f"{n_mismatches} rows disagree on the true target; refusing to test untrusted data.")  # stop
    print("[PASS] True labels agree on every common row.")  # reassurance
    merged["true_target"] = merged["true_target_vaep"]  # one label column for the test
    return merged  # paired table


def cross_check_against_sklearn(y_true, vaep_probs, gat_probs, delong_auc_vaep, delong_auc_gat):
    """Both DeLong AUCs must equal sklearn's AUC to within 1e-6, otherwise stop."""
    print_header("SELF-VERIFICATION - DELONG AUC vs SKLEARN AUC")  # section title
    sk_vaep = roc_auc_score(y_true, vaep_probs)  # independent VAEP AUC
    sk_gat = roc_auc_score(y_true, gat_probs)  # independent GAT AUC
    print(f"VAEP DeLong {delong_auc_vaep:.6f} | sklearn {sk_vaep:.6f}")  # compare
    print(f"GAT  DeLong {delong_auc_gat:.6f} | sklearn {sk_gat:.6f}")  # compare
    if abs(delong_auc_vaep - sk_vaep) > 1e-6 or abs(delong_auc_gat - sk_gat) > 1e-6:  # tolerance check
        raise ValueError("DeLong AUC and sklearn AUC disagree; stopping instead of reporting an unverified number.")  # stop
    print("[PASS] DeLong AUC values match sklearn within tolerance.")  # reassurance


def main():  # runs the whole step from start to finish
    common.verify_environment()  # optional interpreter check
    print_header("STEP 18 - PAIRED DELONG TEST: GAT vs VAEP - FULL SCALE")  # title
    require_file(VAEP_PREDICTIONS_FILE, "VAEP full-scale validation predictions (Step 13)")  # input 1
    require_file(GAT_PREDICTIONS_FILE, "GAT full-scale validation predictions (Step 14)")  # input 2
    vaep_headline_auc, vaep_from_file = read_headline_auc(VAEP_SUMMARY_FILE, FALLBACK_VAEP_HEADLINE_AUC, "VAEP")  # headline number
    gat_headline_auc, gat_from_file = read_headline_auc(GAT_SUMMARY_FILE, FALLBACK_GAT_HEADLINE_AUC, "GAT")  # headline number

    vaep_predictions = load_predictions(VAEP_PREDICTIONS_FILE, "VAEP")  # VAEP table
    gat_predictions = load_predictions(GAT_PREDICTIONS_FILE, "GAT")  # GAT table
    common_rows = find_common_observations(vaep_predictions, gat_predictions)  # paired table
    common_rows.to_csv(COMMON_PREDICTIONS_FILE, index=False)  # save it for later steps
    print(f"Common predictions saved to:\n{COMMON_PREDICTIONS_FILE}")  # where

    print_header("RUNNING PAIRED DELONG TEST")  # section title
    y_true = common_rows["true_target"].to_numpy(dtype=float)  # labels as floats
    vaep_probs = common_rows["vaep_probability"].to_numpy(dtype=float)  # VAEP scores
    gat_probs = common_rows["gat_probability"].to_numpy(dtype=float)  # GAT scores
    n_common, n_positive_common = len(common_rows), int(y_true.sum())  # sample sizes
    print(f"Paired observations: {n_common:,} | positives: {n_positive_common:,}")  # report

    auc_vaep, auc_gat, auc_diff, z_statistic, p_value = calculate_delong_p_value(y_true, vaep_probs, gat_probs)  # the test
    print(f"VAEP AUC-ROC on paired set: {auc_vaep:.4f}")  # VAEP AUC
    print(f"GAT AUC-ROC on paired set:  {auc_gat:.4f}")  # GAT AUC
    if z_statistic is not None:  # only verify when the test produced a statistic
        cross_check_against_sklearn(y_true, vaep_probs, gat_probs, auc_vaep, auc_gat)  # safety check
    print(f"AUC difference (VAEP minus GAT): {auc_diff:.4f}")  # gap
    if z_statistic is None:  # degenerate variance
        print("[COULD NOT COMPUTE] variance of the difference was not positive; no valid z or p.")  # report
        return 1  # non-zero exit

    print(f"DeLong z-statistic: {z_statistic:.4f}")  # z
    print(f"DeLong p-value (two-sided): {p_value:.6f}")  # p
    print_header("STEP 18 FINAL VERDICT - FULL SCALE")  # section title
    if p_value < ALPHA:  # significant
        who = "VAEP significantly outperforms GAT" if auc_diff > 0 else "GAT significantly outperforms VAEP"  # direction
        print(f"[SIGNIFICANT] p = {p_value:.6f} < {ALPHA}. {who} at full scale.")  # verdict
    else:  # not significant
        print(f"[NOT SIGNIFICANT] p = {p_value:.6f} >= {ALPHA}. The {auc_diff:.4f} gap could be chance.")  # verdict

    pd.DataFrame([{  # one-row result table
        "scale": "full_scale_292_train_134_validation",  # label of the data scale
        "n_common_observations": n_common,  # paired rows
        "n_positive_common": n_positive_common,  # paired positives
        "vaep_auc_roc_full_validation": round(vaep_headline_auc, 4),  # VAEP headline AUC
        "gat_auc_roc_full_validation": round(gat_headline_auc, 4),  # GAT headline AUC
        "vaep_headline_read_from_summary_file": vaep_from_file,  # where the VAEP number came from
        "gat_headline_read_from_summary_file": gat_from_file,  # where the GAT number came from
        "vaep_auc_roc_paired_common": round(auc_vaep, 4),  # VAEP AUC on paired rows
        "gat_auc_roc_paired_common": round(auc_gat, 4),  # GAT AUC on paired rows
        "auc_difference_vaep_minus_gat": round(auc_diff, 4),  # difference
        "delong_z_statistic": round(z_statistic, 4),  # z
        "delong_p_value": p_value,  # p, unrounded
        "significant_at_0.05": bool(p_value < ALPHA),  # yes/no
    }]).to_csv(DELONG_RESULT_FILE, index=False)  # write the result
    print(f"DeLong test result saved to:\n{DELONG_RESULT_FILE}")  # where
    print_header("STEP 18 COMPLETE")  # done
    return 0  # success


if __name__ == "__main__":  # run only when executed directly
    common.run_main(main, "STEP 18")  # shared error handling
