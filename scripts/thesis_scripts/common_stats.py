# ==============================================================================
# common_stats.py  -  the paired DeLong test (Sun & Xu 2014 fast version)
# ==============================================================================
# Steps 08, 15, 18, 24 and the parity scripts each had their own copy of this code, with a slow
# pure-Python loop for midranks. This file keeps ONE copy. The maths is unchanged; the only
# difference is that midranks come from scipy's rankdata(method="average"), which returns the same
# numbers as the hand-written loop (ties share the average of the ranks they cover) but runs in C.
# ==============================================================================

import numpy as np  # arrays and covariance
from scipy import stats  # normal distribution for the p-value
from scipy.stats import rankdata  # fast midranks


def compute_midrank(x):
    """Midranks of a 1-D array: tied values get the average of the ranks they occupy (1-based)."""
    return rankdata(x, method="average").astype(float)  # same values the old while-loop produced


def fast_delong(predictions_sorted_transposed, label_1_count, auc_form="step18"):
    """DeLong AUCs and their covariance matrix.

    predictions_sorted_transposed : array (n_models, n_examples), positives first then negatives.
    label_1_count                 : how many positives there are.
    auc_form                      : 'step18' or 'step24'. The two original scripts wrote the same
                                    AUC formula with slightly different operator order, which differs
                                    only in the last floating-point digit. Kept so saved numbers match.
    """
    m = label_1_count  # number of positive examples
    n = predictions_sorted_transposed.shape[1] - m  # number of negative examples
    positive_examples = predictions_sorted_transposed[:, :m]  # first m columns are positives
    negative_examples = predictions_sorted_transposed[:, m:]  # the rest are negatives

    tx = rankdata(positive_examples, method="average", axis=1).astype(float)  # midranks inside positives
    ty = rankdata(negative_examples, method="average", axis=1).astype(float)  # midranks inside negatives
    tz = rankdata(predictions_sorted_transposed, method="average", axis=1).astype(float)  # midranks overall

    if auc_form == "step18":  # formula as written in steps 08/15/18
        aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1) * 0.5 / n  # mean positive rank, rescaled
    else:  # formula as written in step 24
        aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)  # algebraically identical

    v01 = (tz[:, :m] - tx) / n  # structural components over the positives
    v10 = 1.0 - (tz[:, m:] - ty) / m  # structural components over the negatives
    sx = np.cov(v01)  # covariance between models on the positives
    sy = np.cov(v10)  # covariance between models on the negatives
    delongcov = sx / m + sy / n  # covariance matrix of the AUC estimates
    return aucs, delongcov  # one AUC per model plus their covariance


def calculate_delong_p_value(y_true, prob_1, prob_2, auc_form="step18"):
    """Paired two-sided DeLong test of AUC(prob_1) - AUC(prob_2).

    Returns (auc_1, auc_2, auc_diff, z, p). z and p are None when the variance of the difference
    is not positive (the original scripts handled that case the same way).
    """
    order = np.argsort(-y_true)  # positives first (stable order is not needed, ranks are tie-averaged)
    y_true_sorted = y_true[order]  # labels in that order
    label_1_count = int(y_true_sorted.sum())  # how many positives
    predictions_sorted = np.vstack([prob_1[order], prob_2[order]])  # 2 x N matrix, same order for both models

    aucs, delongcov = fast_delong(predictions_sorted, label_1_count, auc_form)  # AUCs and covariance
    auc_1, auc_2 = aucs  # unpack the two AUCs
    auc_diff = auc_1 - auc_2  # difference being tested

    variance_diff = delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1]  # Var(A) + Var(B) - 2 Cov(A, B)
    if variance_diff <= 0:  # degenerate sample
        return auc_1, auc_2, auc_diff, None, None  # no valid test statistic

    z_statistic = auc_diff / np.sqrt(variance_diff)  # standardised difference
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_statistic)))  # two-sided p-value, same expression as before
    return auc_1, auc_2, auc_diff, z_statistic, p_value  # everything the callers print or save
