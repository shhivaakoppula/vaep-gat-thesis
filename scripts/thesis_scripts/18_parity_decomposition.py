# ==============================================================================
# STEP 29 (file 18_parity_decomposition.py) - SPLIT THE VAEP-GAT GAP INTO INPUT AND MODEL EFFECTS
# ==============================================================================
# Reads the saved validation predictions of four cells plus one control and writes the numbers
# for thesis Table 19 / Section 4.5:
#   cell 1  XGBoost on A      = VAEP as published        (Step 13 / Step 21)
#   cell 2  GAT on B          = the GAT of RQ1           (Step 14 / Step 23)
#   cell 3  XGBoost on A+B                               (Step 27)
#   cell 4  GAT on A+B                                   (Step 28)
#   control XGBoost on A, restricted to rows with a usable 360 frame (Step 27)
#   total gap            = AUC1 - AUC2
#   input effect, trees  = AUC_control - AUC3   (expected <= 0)
#   model effect at A+B  = AUC3 - AUC4
#   input effect, GAT    = AUC4 - AUC2          (expected >= 0)
#   (AUC1 - AUC_control is the effect of the smaller training set, reported separately)
# Tests: paired DeLong for the model effect (same input), and a paired bootstrap over MATCHES
# (whole matches resampled) for every bracket, because nested inputs can mislead DeLong
# (Demler et al., 2012) and because actions inside one match are correlated (Obuchowski, 1997).
# Usage:  python 18_parity_decomposition.py [--horizon 10] [--seed 42] [--n_boot 2000]
# Speed: the bootstrap AUC uses ranks (the Mann-Whitney form of the same AUC) instead of building a
# ROC curve for every draw. The resampling order and the random numbers are unchanged.
# ==============================================================================

import argparse  # command-line options

import numpy as np  # arrays and random numbers
import pandas as pd  # tables
from scipy.stats import rankdata  # fast ranks for the bootstrap AUC
from sklearn.metrics import roc_auc_score  # exact AUC for the point estimates

import common  # shared helpers
from common import OUTPUT_DIR  # shared paths
from common_stats import calculate_delong_p_value  # paired DeLong test (step-24 AUC form, as in the original)


def find_file(patterns, what):
    """First file in OUTPUT_DIR matching one of the patterns."""
    for pat in patterns:  # try the patterns in order
        hits = sorted(OUTPUT_DIR.glob(pat))  # alphabetical hits
        if hits:  # something matched
            return hits[0]  # first hit wins
    raise FileNotFoundError(f"Could not find {what}. Tried {patterns} in {OUTPUT_DIR}")  # nothing matched


def delong(y, pa, pb):
    """Paired DeLong test. Returns (auc_a, auc_b, z, p)."""
    auc_a, auc_b, _, z, p = calculate_delong_p_value(y.astype(float), pa, pb, auc_form="step24")  # shared implementation
    if z is None:  # degenerate variance: the original produced nan here
        z, p = float("nan"), float("nan")  # same printed result
    return auc_a, auc_b, z, p  # four numbers, as before


def load_pred(path, name):
    """Read one prediction file and rename its label and score columns with a suffix."""
    df = pd.read_csv(path)  # the saved predictions
    need = {"game_id", "action_index", "true_target", "predicted_probability"}  # required columns
    if not need <= set(df.columns):  # something is missing
        raise ValueError(f"{path.name} lacks columns {need - set(df.columns)}")  # stop
    return df[["game_id", "action_index", "true_target", "predicted_probability"]].rename(
        columns={"true_target": f"y_{name}", "predicted_probability": f"p_{name}"})  # y_<cell> and p_<cell>


def rank_auc(y, score):
    """AUC from ranks (Mann-Whitney). Equal to roc_auc_score up to floating-point rounding."""
    n_pos = int(y.sum())  # positives
    n_neg = len(y) - n_pos  # negatives
    ranks = rankdata(score)  # average ranks, ties share the mean rank
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)  # standard formula


def main():  # runs the whole step from start to finish
    common.verify_environment()  # optional interpreter check
    ap = argparse.ArgumentParser()  # command-line options
    ap.add_argument("--horizon", type=int, default=10, choices=[5, 10])  # label horizon N
    ap.add_argument("--seed", type=int, default=42)  # which GAT seed's predictions to read
    ap.add_argument("--n_boot", type=int, default=2000)  # bootstrap draws
    args = ap.parse_args()  # read them
    H, S = args.horizon, args.seed  # short names
    tag = f"N{H}"  # file-name tag

    if H == 10:  # horizon 10 files
        f1 = find_file(["13_vaep_full_scale_validation_predictions.csv"], "cell 1 (VAEP, N=10)")  # VAEP
        f2 = find_file(["14_gat_full_scale_validation_predictions.csv"], "cell 2 (GAT, N=10)")  # GAT
    else:  # horizon 5 files
        f1 = find_file(["21_label_horizon_N5_validation_predictions.csv", "21_vaep_label_horizon_N5_validation_predictions.csv",
                        "21*N5*predictions*.csv"], "cell 1 (VAEP, N=5)")  # VAEP
        f2 = find_file(["23_gat_features_at_horizon5_count7_full_at_N5_validation_predictions.csv",
                        "23*count7*predictions*.csv"], "cell 2 (GAT 7 features, N=5)")  # GAT
    fc = find_file([f"27_xgb_A_restricted_{tag}_validation_predictions.csv"], "control")  # control
    f3 = find_file([f"27_xgb_A_plus_B_{tag}_validation_predictions.csv"], "cell 3")  # XGBoost on A+B
    f4 = find_file([f"28_gat_plus_vaep_{tag}_seed{S}_validation_predictions.csv"], "cell 4")  # GAT on A+B
    names = {"c1": f1, "c2": f2, "ctl": fc, "c3": f3, "c4": f4}  # cell -> file
    for k, v in names.items():  # show what was found
        print(f"{k:4s} <- {v.name}")

    df = None  # merged table
    for k, path in names.items():  # merge the five files on the action key
        d = load_pred(path, k)  # one file
        df = d if df is None else df.merge(d, on=["game_id", "action_index"], how="inner")  # inner join keeps the paired rows
    ycols = [c for c in df.columns if c.startswith("y_")]  # the five label columns
    bad = sum((df[ycols[0]] != df[c]).sum() for c in ycols[1:])  # label disagreements against the first
    print(f"\npaired actions: {len(df):,}   positives: {int(df[ycols[0]].sum()):,}   label mismatches: {int(bad)}")  # summary
    if bad:  # the files were built on different targets
        raise SystemExit("Label mismatch between prediction files: they were built on different targets.")  # stop
    y = df[ycols[0]].values.astype(int)  # shared labels
    P = {k: df[f"p_{k}"].values for k in names}  # scores per cell

    auc = {k: roc_auc_score(y, P[k]) for k in P}  # exact AUC per cell
    print("\nAUC-ROC on the paired set:")  # label
    for k, label in [("c1", "cell 1 XGB on A (published)"), ("ctl", "control XGB on A, restricted rows"),
                     ("c3", "cell 3 XGB on A+B"), ("c4", "cell 4 GAT on A+B"), ("c2", "cell 2 GAT on B")]:  # print order
        print(f"  {label:38s} {auc[k]:.4f}")

    brackets = {  # name -> how to compute it from the AUC dictionary
        "total_gap (c1 - c2)": lambda a: a["c1"] - a["c2"],
        "training_set_size_effect (c1 - ctl)": lambda a: a["c1"] - a["ctl"],
        "input_effect_trees (ctl - c3)": lambda a: a["ctl"] - a["c3"],
        "model_effect_at_A+B (c3 - c4)": lambda a: a["c3"] - a["c4"],
        "input_effect_GAT (c4 - c2)": lambda a: a["c4"] - a["c2"],
    }
    point = {n: f(auc) for n, f in brackets.items()}  # point estimates

    # ---------------- bootstrap over matches ----------------
    rng = np.random.default_rng(12345)  # fixed generator, same stream as before
    games = df["game_id"].values  # match id per row
    uniq = np.unique(games)  # sorted unique matches
    order = np.argsort(games, kind="stable")  # rows grouped by match (one sort instead of one scan per match)
    bounds = np.searchsorted(games[order], uniq, side="left")  # start of each match inside that order
    ends = np.append(bounds[1:], len(order))  # end of each match
    idx_by_game = {g: np.sort(order[s:e]) for g, s, e in zip(uniq, bounds, ends)}  # row indices per match, ascending as before
    keys = list(P.keys())  # cells in a fixed order
    draws = {n: [] for n in brackets}  # one list of draws per bracket
    for b in range(args.n_boot):  # one bootstrap draw
        pick = rng.choice(uniq, size=len(uniq), replace=True)  # matches drawn with replacement
        idx = np.concatenate([idx_by_game[g] for g in pick])  # all rows of the drawn matches
        yb = y[idx]  # labels of the draw
        if yb.sum() == 0 or yb.sum() == len(yb):  # AUC needs both classes
            continue  # skip, as before
        a = {k: rank_auc(yb, P[k][idx]) for k in keys}  # AUC of every cell on this draw
        for n, f in brackets.items():  # every bracket
            draws[n].append(f(a))  # store its value
        if (b + 1) % 200 == 0:  # progress every 200 draws
            print(f"  bootstrap {b+1}/{args.n_boot}")

    rows = []  # result rows
    for n in brackets:  # one row per bracket
        d = np.array(draws[n])  # all draws
        lo, hi = np.percentile(d, [2.5, 97.5])  # 95 percent percentile interval
        rows.append({"quantity": n, "horizon": H, "estimate": round(point[n], 4),
                     "boot_ci_low": round(lo, 4), "boot_ci_high": round(hi, 4),
                     "ci_excludes_zero": bool(lo > 0 or hi < 0), "n_matches": len(uniq),
                     "n_boot_used": len(d)})  # same columns as before
    out = pd.DataFrame(rows)  # result table

    print("\nDeLong tests on the paired set (reference; the model effect compares equal inputs):")  # label
    for a, b, label in [("c3", "c4", "model effect at A+B: XGB(A+B) vs GAT(A+B)"),
                        ("c1", "c2", "total gap: VAEP vs GAT (RQ1)"),
                        ("ctl", "c3", "input effect, trees (nested inputs: use the bootstrap)"),
                        ("c4", "c2", "input effect, GAT (nested inputs: use the bootstrap)")]:  # four tests
        aa, bb, z, p = delong(y, P[a], P[b])  # run the test
        print(f"  {label:60s} AUC {aa:.4f} vs {bb:.4f}  diff {aa-bb:+.4f}  z {z:7.2f}  p {p:.3g}")  # one line each

    print("\n" + out.to_string(index=False))  # bracket table
    out.to_csv(OUTPUT_DIR / f"29_parity_decomposition_{tag}_seed{S}_summary.csv", index=False)  # save it

    print("\nPattern for thesis Table 20:")  # label
    m = point["model_effect_at_A+B (c3 - c4)"]  # model effect
    g = point["input_effect_GAT (c4 - c2)"]  # GAT gain from the extra input
    print(f"  model effect at A+B = {m:+.4f}; GAT gain from action context = {g:+.4f}; "
          f"total gap = {point['total_gap (c1 - c2)']:+.4f}")  # summary line
    print("  Compare with Table 20: patterns 1 to 4 are defined by the size of these two numbers "
          "relative to their bootstrap intervals and to the 0.05 total gap.")  # reading hint
    return 0  # success


if __name__ == "__main__":  # only when run directly
    common.run_main(main, "STEP 29")  # shared error handling
