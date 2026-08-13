"""The measures used throughout the review, in one place.

Keeping them here means the champion, the corrected champion and the challenger
are all scored by identical code, which is the only way a comparison between
them means anything.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(y_true, scores))


def gini(y_true: np.ndarray, scores: np.ndarray) -> float:
    return 2.0 * auc(y_true, scores) - 1.0


def ks_statistic(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Largest gap between the churner and non-churner score distributions."""
    order = np.argsort(scores)
    y = np.asarray(y_true)[order]
    positives = y.sum()
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    cum_pos = np.cumsum(y) / positives
    cum_neg = np.cumsum(1 - y) / negatives
    return float(np.max(np.abs(cum_pos - cum_neg)))


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, bins: int = 10
) -> float:
    """PSI between a development distribution and a later one.

    Bin edges come from the development sample only. Re-cutting the edges on the
    later sample would hide exactly the movement the measure exists to detect.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    exp_share = np.histogram(expected, bins=edges)[0] / len(expected)
    act_share = np.histogram(actual, bins=edges)[0] / len(actual)

    # Floor the shares so an empty bin does not send the index to infinity.
    floor = 1e-4
    exp_share = np.clip(exp_share, floor, None)
    act_share = np.clip(act_share, floor, None)
    return float(np.sum((act_share - exp_share) * np.log(act_share / exp_share)))


def categorical_stability_index(expected: pd.Series, actual: pd.Series) -> float:
    """Same measure for a categorical variable, over its levels."""
    levels = sorted(set(expected.dropna().unique()) | set(actual.dropna().unique()))
    exp_share = np.array([(expected == lv).mean() for lv in levels])
    act_share = np.array([(actual == lv).mean() for lv in levels])
    floor = 1e-4
    exp_share = np.clip(exp_share, floor, None)
    act_share = np.clip(act_share, floor, None)
    return float(np.sum((act_share - exp_share) * np.log(act_share / exp_share)))


def calibration_table(
    y_true: np.ndarray, probs: np.ndarray, bins: int = 10
) -> pd.DataFrame:
    """Predicted versus observed churn rate, by band of predicted probability."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(probs)})
    frame["band"] = pd.qcut(frame["p"], q=bins, duplicates="drop")
    table = (
        frame.groupby("band", observed=True)
        .agg(customers=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
        .reset_index()
    )
    table["gap"] = table["predicted"] - table["observed"]
    return table


def expected_calibration_error(
    y_true: np.ndarray, probs: np.ndarray, bins: int = 10
) -> float:
    table = calibration_table(y_true, probs, bins=bins)
    weights = table["customers"] / table["customers"].sum()
    return float(np.sum(weights * table["gap"].abs()))


def lift_at_cut(y_true: np.ndarray, scores: np.ndarray, cut: float = 0.20) -> float:
    """Churn rate in the top-scoring slice, divided by the portfolio rate."""
    y = np.asarray(y_true)
    n_top = max(1, int(round(cut * len(y))))
    top = np.argsort(scores)[::-1][:n_top]
    base = y.mean()
    if base == 0:
        return float("nan")
    return float(y[top].mean() / base)


def score_summary(
    y_true: np.ndarray, probs: np.ndarray, bins: int = 10, cut: float = 0.20
) -> Tuple[float, float, float, float, float]:
    """AUC, Gini, KS, expected calibration error and lift, in that order."""
    return (
        auc(y_true, probs),
        gini(y_true, probs),
        ks_statistic(y_true, probs),
        expected_calibration_error(y_true, probs, bins=bins),
        lift_at_cut(y_true, probs, cut=cut),
    )
