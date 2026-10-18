# ==============================================================================
# common_error.py  -  the error analysis used by both 24_error_analysis_vaep_vs_gat.py (N=10)
# and 25_error_analysis_vaep_vs_gat_n5.py (N=5)
# ==============================================================================
# The two original scripts were identical except for which prediction files they read and the
# prefix of the CSV files they wrote. The analysis (parts A to G) now exists once. All statistics,
# the bootstrap seed (42), the 1000 resamples and the table layouts are unchanged.
# ==============================================================================

import sys  # for sys.exit on missing files
import numpy as np  # arrays and the random generator
import pandas as pd  # tables
from scipy import stats  # Spearman correlation and the Wilcoxon test
from sklearn.metrics import roc_auc_score, average_precision_score  # headline metrics

N_BOOT = 1000  # bootstrap resamples for the match-level confidence interval
MIN_POS_PER_GROUP = 30  # a group AUC with fewer positives is too noisy to report
NEED = ["game_id", "action_index", "true_target", "predicted_probability"]  # columns each file must contain


def safe_auc(y, p):
    """AUC, or NaN when a group has only one class (AUC is undefined there)."""
    if y.sum() == 0 or y.sum() == len(y):  # all negative or all positive
        return np.nan  # undefined
    return roc_auc_score(y, p)  # normal case


def load_and_pair(vaep_file, gat_file):
    """Read both prediction files (only the four needed columns) and pair them by game and action."""
    frames = {}  # name -> table
    for name, f in (("VAEP", vaep_file), ("GAT", gat_file)):  # same treatment for both
        header = pd.read_csv(f, nrows=0).columns  # read just the header first
        miss = set(NEED) - set(header)  # absent columns
        if miss:  # cannot continue
            sys.exit(f"[{name}] missing columns {miss}; found {list(header)}")  # stop with a clear message
        frames[name] = pd.read_csv(f, usecols=NEED)  # load only what is used (less memory than reading all columns)
    df = frames["VAEP"].merge(frames["GAT"], on=["game_id", "action_index"], suffixes=("_vaep", "_gat"))  # inner join
    if (df["true_target_vaep"] != df["true_target_gat"]).any():  # labels must agree
        sys.exit("[ERROR] true labels differ between the two files for the same action.")  # stop
    df = df.rename(columns={"true_target_vaep": "y", "predicted_probability_vaep": "p_vaep",  # friendlier names
                            "predicted_probability_gat": "p_gat"}).drop(columns="true_target_gat")  # one label column
    df["y"] = df["y"].astype(int)  # integer labels
    df["r_vaep"] = df["p_vaep"].rank(pct=True)  # rank percentile of the VAEP score (0 = lowest)
    df["r_gat"] = df["p_gat"].rank(pct=True)  # rank percentile of the GAT score
    df["pos_in_match"] = df.groupby("game_id")["action_index"].rank(pct=True)  # position of the action inside its match
    df["match_decile"] = np.minimum((df["pos_in_match"] * 10).apply(np.ceil), 10).astype(int)  # decile 1..10
    return df  # one paired table


def run_error_analysis(vaep_file, gat_file, output_dir, prefix, sanity_note):
    """Run parts A-G and save the CSV files as <prefix>A_..., <prefix>B_..., and so on."""

    def save(table, name):  # small helper used by every part
        table.to_csv(output_dir / name, index=False)  # write the CSV
        print(f"   saved -> {name}")  # tell the user

    rng = np.random.default_rng(42)  # seeded generator that draws the bootstrap seeds
    df = load_and_pair(vaep_file, gat_file)  # paired data

    # ---- A. sanity check ------------------------------------------------------
    print("=" * 80 + f"\nA. SANITY CHECK ({sanity_note})\n" + "=" * 80)  # section title
    print(f"   paired rows: {len(df):,}   positives: {df.y.sum():,} ({df.y.mean() * 100:.4f}%)")  # size
    rows = []  # one row per model
    for m in ("vaep", "gat"):  # both models
        auc, pr = roc_auc_score(df.y, df[f"p_{m}"]), average_precision_score(df.y, df[f"p_{m}"])  # metrics
        print(f"   {m.upper():5s} AUC-ROC {auc:.4f}   PR-AUC {pr:.4f}")  # show
        rows.append({"model": m.upper(), "auc_roc": auc, "pr_auc": pr})  # keep
    print(f"   Spearman correlation between the two models' scores: {stats.spearmanr(df.p_vaep, df.p_gat).correlation:.3f}")  # agreement
    save(pd.DataFrame(rows), f"{prefix}A_overall_sanity.csv")  # file A

    # ---- B. per-match AUC -------------------------------------------------------
    print("\n" + "=" * 80 + "\nB. PER-MATCH AUC\n" + "=" * 80)  # section title
    mrows = []  # one row per match
    for gid, d in df.groupby("game_id"):  # every match
        if d.y.sum() < 3:  # too few goal-leading actions to rank
            continue  # skip this match
        mrows.append({"game_id": gid, "n": len(d), "positives": int(d.y.sum()),  # match size
                      "auc_vaep": safe_auc(d.y, d.p_vaep), "auc_gat": safe_auc(d.y, d.p_gat)})  # AUC of each model
    mdf = pd.DataFrame(mrows).dropna()  # drop matches where an AUC was undefined
    mdf["diff"] = mdf.auc_vaep - mdf.auc_gat  # per-match gap
    print(f"   matches with >=3 positives: {len(mdf)} of {df.game_id.nunique()}")  # coverage
    print(f"   VAEP better in {int((mdf['diff'] > 0).sum())} matches, GAT better in {int((mdf['diff'] < 0).sum())}")  # wins
    print(f"   mean per-match AUC: VAEP {mdf.auc_vaep.mean():.4f}, GAT {mdf.auc_gat.mean():.4f}, mean gap {mdf['diff'].mean():+.4f}")  # means
    boots = [mdf["diff"].sample(len(mdf), replace=True, random_state=int(s)).mean()  # one resample = one mean gap
             for s in rng.integers(0, 1_000_000, N_BOOT)]  # same seed stream as the original script
    lo, hi = np.percentile(boots, [2.5, 97.5])  # 95% percentile interval
    print(f"   match-level bootstrap 95% CI of mean gap: [{lo:+.4f}, {hi:+.4f}]  ({N_BOOT} resamples)")  # interval
    print(f"   Wilcoxon signed-rank on per-match gaps: p = {stats.wilcoxon(mdf['diff']).pvalue:.2e}")  # paired test
    save(mdf.sort_values("diff"), f"{prefix}B_per_match_auc.csv")  # file B

    # ---- C. AUC by position in the match ----------------------------------------
    print("\n" + "=" * 80 + "\nC. AUC BY MATCH-TIME DECILE (1 = first tenth of the match's actions)\n" + "=" * 80)  # section title
    crow = [{"decile": k, "n": len(d), "positives": int(d.y.sum()),  # one row per decile
             "auc_vaep": safe_auc(d.y, d.p_vaep), "auc_gat": safe_auc(d.y, d.p_gat)} for k, d in df.groupby("match_decile")]  # AUCs
    cdf = pd.DataFrame(crow)  # table
    cdf["gap"] = cdf.auc_vaep - cdf.auc_gat  # VAEP minus GAT
    print(cdf.round(4).to_string(index=False))  # show
    save(cdf, f"{prefix}C_auc_by_match_decile.csv")  # file C

    # ---- D. capture rate --------------------------------------------------------
    print("\n" + "=" * 80 + "\nD. CAPTURE RATE: share of all positives inside each model's top X% of scores\n" + "=" * 80)  # section title
    drow = []  # one row per cut-off
    for top in (0.01, 0.05, 0.10, 0.25):  # top 1%, 5%, 10%, 25%
        row = {"top_fraction": top}  # start the row
        for m in ("vaep", "gat"):  # both models
            cut = df[f"p_{m}"].quantile(1 - top)  # score threshold for the top slice
            row[f"capture_{m}"] = df.loc[df[f"p_{m}"] >= cut, "y"].sum() / df.y.sum()  # share of positives caught
        row["random_expectation"] = top  # what random ranking would catch
        drow.append(row)  # keep
    ddf = pd.DataFrame(drow)  # table
    print(ddf.round(4).to_string(index=False))  # show
    save(ddf, f"{prefix}D_capture_rate.csv")  # file D

    # ---- E. calibration ---------------------------------------------------------
    print("\n" + "=" * 80 + "\nE. CALIBRATION (equal-count bins of each model's own predicted probability)\n" + "=" * 80)  # section title
    erow = []  # one table per model
    for m in ("vaep", "gat"):  # both models
        bins = pd.qcut(df[f"p_{m}"].rank(method="first"), 10, labels=False)  # ten equal-sized bins
        t = df.groupby(bins).agg(mean_pred=(f"p_{m}", "mean"), observed_rate=("y", "mean"), n=("y", "size"))  # per-bin averages
        t["model"] = m.upper()  # model name
        t["bin"] = t.index + 1  # bin number 1..10
        erow.append(t.reset_index(drop=True))  # keep
    edf = pd.concat(erow)[["model", "bin", "n", "mean_pred", "observed_rate"]]  # final layout
    print(edf.round(5).to_string(index=False))  # show
    save(edf, f"{prefix}E_calibration_bins.csv")  # file E

    # ---- F. disagreement --------------------------------------------------------
    print("\n" + "=" * 80 + "\nF. DISAGREEMENT BETWEEN THE MODELS (rank-percentile gap = r_vaep - r_gat)\n" + "=" * 80)  # section title
    df["rank_gap"] = df.r_vaep - df.r_gat  # positive when VAEP ranks the action higher
    pos, neg = df[df.y == 1], df[df.y == 0]  # split by true label
    print(f"   Goal-leading actions (positives): {len(pos):,}")  # count
    print(f"     VAEP ranks it higher than GAT : {(pos.rank_gap > 0).mean() * 100:.1f}%")  # share
    print(f"     both put it in the top 10%    : {((pos.r_vaep >= .9) & (pos.r_gat >= .9)).mean() * 100:.1f}%")  # share
    print(f"     VAEP top 10%, GAT not         : {((pos.r_vaep >= .9) & (pos.r_gat < .9)).mean() * 100:.1f}%")  # share
    print(f"     GAT top 10%, VAEP not         : {((pos.r_gat >= .9) & (pos.r_vaep < .9)).mean() * 100:.1f}%")  # share
    print(f"     both miss (both below median) : {((pos.r_vaep < .5) & (pos.r_gat < .5)).mean() * 100:.1f}%")  # share
    cols = ["game_id", "action_index", "y", "p_vaep", "p_gat", "r_vaep", "r_gat", "rank_gap"]  # columns in the lists
    topn = 25  # rows per list
    tables = {  # six action-level lists
        "false_negatives_vaep_worst": pos.nsmallest(topn, "r_vaep")[cols],  # goals VAEP ranked lowest
        "false_negatives_gat_worst": pos.nsmallest(topn, "r_gat")[cols],  # goals GAT ranked lowest
        "false_positives_vaep_worst": neg.nlargest(topn, "r_vaep")[cols],  # non-goals VAEP ranked highest
        "false_positives_gat_worst": neg.nlargest(topn, "r_gat")[cols],  # non-goals GAT ranked highest
        "positives_vaep_much_better": pos.nlargest(topn, "rank_gap")[cols],  # goals where VAEP beat GAT most
        "positives_gat_much_better": pos.nsmallest(topn, "rank_gap")[cols],  # goals where GAT beat VAEP most
    }
    for name, t in tables.items():  # write each list
        save(t, f"{prefix}F_{name}.csv")  # file F_*
    print("   (the six tables above are action-level lists to inspect in the raw event data)")  # hint

    # ---- G. optional: action type and pitch third --------------------------------
    print("\n" + "=" * 80 + "\nG. AUC BY ACTION TYPE AND PITCH THIRD (optional)\n" + "=" * 80)  # section title
    spadl = None  # path of a suitable actions file, if any
    for pat in ("*spadl*.csv", "*actions*.csv"):  # filename patterns to try
        for cand in sorted(output_dir.glob(pat)):  # alphabetical candidates
            if "prediction" in cand.name:  # never treat a predictions file as actions
                continue  # skip it
            try:  # peek at the header only
                head = pd.read_csv(cand, nrows=5)  # five rows are enough
            except Exception:  # unreadable file
                continue  # try the next one
            if {"game_id", "action_index"} <= set(head.columns):  # has the join keys
                spadl = cand  # use it
                break  # stop searching this pattern
        if spadl is not None:  # found one
            break  # stop searching patterns
    if spadl is None:  # nothing suitable
        print("   No SPADL/actions CSV with game_id + action_index found in output/. Skipping G.")  # explain
    else:  # a file exists
        print(f"   using {spadl.name}")  # tell which
        a = pd.read_csv(spadl)  # load it
        type_col = next((c for c in ("type_name", "action_type", "type") if c in a.columns), None)  # action-type column
        x_col = next((c for c in ("start_x", "x_start", "x") if c in a.columns), None)  # x-coordinate column
        keep = ["game_id", "action_index"] + [c for c in (type_col, x_col) if c]  # columns to bring across
        j = df.merge(a[keep], on=["game_id", "action_index"], how="left")  # attach them
        print(f"   matched {j[type_col].notna().mean() * 100:.1f}% of rows" if type_col else "   no type column found")  # coverage
        if type_col:  # AUC by action type
            rows = [{"action_type": k, "n": len(d), "positives": int(d.y.sum()),  # group size
                     "auc_vaep": safe_auc(d.y, d.p_vaep), "auc_gat": safe_auc(d.y, d.p_gat)}  # AUCs
                    for k, d in j.groupby(type_col) if d.y.sum() >= MIN_POS_PER_GROUP]  # only groups with enough positives
            t = pd.DataFrame(rows)  # table
            t["gap"] = t.auc_vaep - t.auc_gat  # VAEP minus GAT
            t = t.sort_values("gap", ascending=False)  # biggest VAEP advantage first
            print(t.round(4).to_string(index=False))  # show
            save(t, f"{prefix}G_auc_by_action_type.csv")  # file G (type)
        if x_col:  # AUC by pitch third
            j["third"] = pd.cut(j[x_col], [-1, 35, 70, 106], labels=["defensive", "middle", "attacking"])  # SPADL pitch is 105 m long
            rows = [{"pitch_third": k, "n": len(d), "positives": int(d.y.sum()),  # group size
                     "auc_vaep": safe_auc(d.y, d.p_vaep), "auc_gat": safe_auc(d.y, d.p_gat)}  # AUCs
                    for k, d in j.groupby("third", observed=True) if d.y.sum() >= MIN_POS_PER_GROUP]  # only groups with enough positives
            t = pd.DataFrame(rows)  # table
            t["gap"] = t.auc_vaep - t.auc_gat  # VAEP minus GAT
            print(t.round(4).to_string(index=False))  # show
            save(t, f"{prefix}G_auc_by_pitch_third.csv")  # file G (third)
    print("\nDone.")  # finished
