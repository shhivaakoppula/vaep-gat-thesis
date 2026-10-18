# ==============================================================================
# STEP 11 (file 08_delong_test_gat_vs_vaep.py) - PAIRED DELONG TEST: GAT vs VAEP (PROTOTYPE SCALE)
# ==============================================================================
# The statistical test that answers RQ1 at prototype scale. Two numbers exist:
#   Step 09 - VAEP validation AUC-ROC = 0.6873
#   Step 10 - GAT validation AUC-ROC  = 0.6554
# GAT is lower. The question is whether the 0.0319 gap is real or noise from the small prototype (24 training
# matches, 5 validation matches). DeLong's test (DeLong, DeLong & Clarke-Pearson, 1988) is built for two AUCs
# measured on the same observations, because a t-test would wrongly treat them as independent. The fast version
# of Sun and Xu (2014) is used: same covariance, computed from midranks.
# The GAT validation set (11,079 graphs) is smaller than the VAEP set (12,214 predictions) because about 9% of
# actions have no 360 data. A paired test needs the SAME observations on both sides, so the intersection is found
# first and the number of dropped rows is reported.
# The DeLong maths lives in common_stats.py (rankdata instead of a Python loop; identical numbers).
# ==============================================================================

import numpy as np  # arrays
import pandas as pd  # tables

import common  # shared helpers
from common import OUTPUT_DIR, print_header, require_file  # shared paths and helpers
from common_stats import calculate_delong_p_value  # paired DeLong test, step-11 form of the AUC formula

VAEP_PREDICTIONS_FILE = OUTPUT_DIR / "09_vaep_validation_predictions.csv"  # Step 09 output
GAT_PREDICTIONS_FILE = OUTPUT_DIR / "10_gat_validation_predictions.csv"  # Step 10 output
DELONG_RESULT_FILE = OUTPUT_DIR / "11_delong_test_result.csv"  # one-row result
COMMON_PREDICTIONS_FILE = OUTPUT_DIR / "11_delong_common_predictions.csv"  # paired predictions
ALPHA = 0.05  # significance level


def load_predictions(path, model_name):
    """Read one prediction file, check its columns and force clean types."""
    print_header(f"LOAD {model_name} VALIDATION PREDICTIONS")  # title
    predictions = pd.read_csv(path)  # the saved predictions
    required_columns = ["game_id", "action_index", "true_target", "predicted_probability"]  # what is needed
    missing = set(required_columns) - set(predictions.columns)  # absent columns
    if missing:
        raise ValueError(f"{model_name} predictions file is missing columns: {sorted(missing)}")  # stop
    predictions = predictions[required_columns].copy()  # keep four columns
    predictions["game_id"] = pd.to_numeric(predictions["game_id"], errors="raise").astype(int)  # integer match id
    predictions["action_index"] = pd.to_numeric(predictions["action_index"], errors="raise").astype(int)  # integer action index
    predictions["true_target"] = pd.to_numeric(predictions["true_target"], errors="raise").astype(int)  # 0/1 label
    predictions["predicted_probability"] = pd.to_numeric(predictions["predicted_probability"], errors="raise")  # float score
    print(f"{model_name} rows: {len(predictions):,}")
    print(f"{model_name} positive rate: {predictions['true_target'].mean() * 100:.4f}%")  # show the share of positive labels
    return predictions  # hand back the cleaned table


def find_common_observations(vaep_predictions, gat_predictions):
    """Inner-join the two prediction tables on (game_id, action_index) and check that the labels agree."""
    print_header("FIND OBSERVATIONS COMMON TO BOTH MODELS")  # title
    vaep_renamed = vaep_predictions.rename(columns={"true_target": "true_target_vaep", "predicted_probability": "vaep_probability"})  # VAEP side
    gat_renamed = gat_predictions.rename(columns={"true_target": "true_target_gat", "predicted_probability": "gat_probability"})  # GAT side
    merged = vaep_renamed.merge(gat_renamed, on=["game_id", "action_index"], how="inner")  # paired rows
    print(f"VAEP validation rows: {len(vaep_predictions):,}")
    print(f"GAT validation rows: {len(gat_predictions):,}")  # show how many GAT rows there are
    print(f"Common (paired) rows: {len(merged):,}")  # show how many rows both models share
    vaep_only = len(vaep_predictions) - len(merged)  # rows without a GAT graph
    gat_only = len(gat_predictions) - len(merged)  # should be 0
    print(f"Rows dropped from VAEP side (no matching GAT graph, e.g. missing 360 data): {vaep_only:,}")
    print(f"Rows dropped from GAT side (should be 0 — every GAT row is built from an action that has 360 data, "  # explain why no GAT row should be dropped (continues on the next line)
          f"and VAEP scores every eligible action regardless): {gat_only:,}")
    n_mismatches = int((merged["true_target_vaep"] != merged["true_target_gat"]).sum())  # label disagreements
    print()
    print(f"True-label mismatches between the two files on common rows: {n_mismatches:,}")  # show how many labels disagree
    if n_mismatches > 0:  # the targets differ, so no test may be run
        raise ValueError(f"I found {n_mismatches} rows where my VAEP file and my GAT file disagree on the true target for "
                         f"the same (game_id, action_index). Both files should trace back to the same verified target. "
                         f"I am stopping here rather than running a statistical test on data I cannot trust.")
    print("[PASS] True labels agree on every common row.")  # confirm that the labels agree on every shared row
    merged["true_target"] = merged["true_target_vaep"]  # one shared label column
    return merged


def main():  # runs the whole step from start to finish
    common.verify_environment()  # optional interpreter check
    print_header("STEP 11 — PAIRED DELONG TEST: GAT vs VAEP")  # title
    print("Purpose:")
    print("I am testing whether the difference between my VAEP validation AUC-ROC (0.6873, Step 09) and my GAT "  # print the purpose of this test (continues on the next lines)
          "validation AUC-ROC (0.6554, Step 10) is statistically significant, or could be explained by chance at my "
          "prototype's scale.")
    print()  # blank line
    print("IMPORTANT: this test runs on my 34-match PROTOTYPE only. It gives me an interim answer to RQ1, not my final "  # print the warning that this is the prototype only (continues on the next lines)
          "thesis result, which will come from the same test run on my full-scale test set once both models are "
          "trained at full scale.")
    require_file(VAEP_PREDICTIONS_FILE, "VAEP validation predictions (Step 09)")  # input 1
    require_file(GAT_PREDICTIONS_FILE, "GAT validation predictions (Step 10)")  # input 2

    vaep_predictions = load_predictions(VAEP_PREDICTIONS_FILE, "VAEP")  # VAEP side
    gat_predictions = load_predictions(GAT_PREDICTIONS_FILE, "GAT")  # GAT side
    paired = find_common_observations(vaep_predictions, gat_predictions)  # paired rows
    paired.to_csv(COMMON_PREDICTIONS_FILE, index=False)  # save them
    print()
    print(f"Common predictions saved to:\n{COMMON_PREDICTIONS_FILE}")  # show where the shared predictions were saved

    print_header("RUNNING PAIRED DELONG TEST")  # title
    y_true = paired["true_target"].to_numpy(dtype=float)  # labels
    vaep_probs = paired["vaep_probability"].to_numpy(dtype=float)  # VAEP scores
    gat_probs = paired["gat_probability"].to_numpy(dtype=float)  # GAT scores
    n_common = len(paired)
    n_positive_common = int(y_true.sum())  # count the positives in the shared set
    print(f"Paired observations used in this test: {n_common:,}")  # show how many pairs the test uses
    print(f"Positive examples in paired set: {n_positive_common:,}")  # show how many positives the pairs contain
    auc_vaep, auc_gat, auc_diff, z_statistic, p_value = calculate_delong_p_value(y_true, vaep_probs, gat_probs, auc_form="step18")  # the test
    print()
    print(f"VAEP AUC-ROC on paired common set: {auc_vaep:.4f}")  # show the VAEP AUC on the shared set
    print(f"GAT AUC-ROC on paired common set:  {auc_gat:.4f}")  # show the GAT AUC on the shared set
    print(f"(These may differ very slightly from my Step 09 and Step 10 headline numbers, because this test only uses "  # note that these can differ slightly from the headline numbers (continues on the next line)
          f"the {n_common:,} observations both models share, not each model's full validation set.)")
    print()  # blank line
    print(f"AUC difference (VAEP minus GAT): {auc_diff:.4f}")  # show the AUC gap (VAEP minus GAT)
    if z_statistic is None:  # degenerate variance
        print()
        print("[COULD NOT COMPUTE] My variance estimate came out non-positive, which can happen with very small or "  # say that the variance could not be computed (continues on the next lines)
              "degenerate samples. I cannot report a valid z-statistic or p-value from this run. I need to "
              "investigate before drawing any conclusion.")
        return 1  # stop with a non-zero exit code
    print(f"DeLong z-statistic: {z_statistic:.4f}")  # show the DeLong z-statistic
    print(f"DeLong p-value (two-sided): {p_value:.6f}")  # show the two-sided p-value
    print()  # blank line

    print_header("STEP 11 FINAL VERDICT")  # title
    if p_value < ALPHA:  # significant
        print(f"[SIGNIFICANT] p = {p_value:.6f} is below my significance threshold of {ALPHA}.")
        if auc_diff > 0:  # VAEP is better
            print("VAEP significantly outperforms GAT on my 34-match prototype. This is a genuine, honest finding: at this "
                  "scale, VAEP's tabular action-sequence features predict future scoring more accurately than my current "
                  "GAT graph representation. Possible reasons I will investigate before writing this up: my prototype's "
                  "small scale (only 24 training matches), my 91.96% graph coverage (missing 8% of actions GAT never sees "
                  "at all), or my current k=5 edge design and actor-node pooling choice needing refinement.")
        else:  # GAT is better
            print("GAT significantly outperforms VAEP on my 34-match prototype.")
    else:  # not significant
        print(f"[NOT SIGNIFICANT] p = {p_value:.6f} is at or above my significance threshold of {ALPHA}.")
        print("I cannot conclude that VAEP and GAT perform differently on my prototype. The 0.0319 gap between 0.6873 and "  # say that no difference can be concluded (continues on the next lines)
              "0.6554 could plausibly be due to chance at this scale (24 training matches, "
              f"{n_common:,} paired validation observations, "
              f"only {n_positive_common} positive examples). "
              "This is exactly why I planned to also run this same test at full scale (292 training matches, 134 "
              "validation matches) before drawing my final RQ1 conclusion — a small prototype like this one may simply "
              "not have enough statistical power to detect a real difference even if one exists.")

    pd.DataFrame([{  # one-row result file, read by the visual overview
        "scale": "prototype_34_matches",
        "n_common_observations": n_common,
        "n_positive_common": n_positive_common,
        "vaep_auc_roc_full_validation": 0.6873,
        "gat_auc_roc_full_validation": 0.6554,
        "vaep_auc_roc_paired_common": round(auc_vaep, 4),
        "gat_auc_roc_paired_common": round(auc_gat, 4),
        "auc_difference_vaep_minus_gat": round(auc_diff, 4),
        "delong_z_statistic": round(z_statistic, 4),
        "delong_p_value": p_value,
        "significant_at_0.05": bool(p_value < ALPHA),
    }]).to_csv(DELONG_RESULT_FILE, index=False)
    print()  # blank line
    print(f"DeLong test result saved to:\n{DELONG_RESULT_FILE}")  # show where the result file was saved
    print_header("STEP 11 COMPLETE")  # done
    print("I now have my first statistically tested RQ1 answer, at prototype scale. My next step, per my day-by-day "
          "timeline, is to scale this exact pipeline — target, VAEP, GAT, and this DeLong test — to my full training "
          "and validation sets before I report anything as my final thesis result.")
    return 0  # success


if __name__ == "__main__":  # only when run directly
    common.run_main(main, "STEP 11")  # shared error handling
