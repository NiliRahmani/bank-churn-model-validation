"""Replication, variable review and performance testing.

Three questions, in the order a review has to answer them:

    1. can the developer's reported result be reproduced at all
    2. is every variable in the model knowable at the moment the model is used
    3. does the model hold up on a period it was not fitted on

Question 2 is the one that changes the answer to question 3 here, which is why
it sits between them.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import metrics
from .champion import CATEGORICAL, FittedModel
from .findings import Finding


def replicate(
    reported_auc: float, achieved_auc: float, cfg: dict
) -> Tuple[Dict[str, float], List[Finding]]:
    """Rebuild the model from the submitted recipe and compare against their number."""
    tolerance = cfg["validation"]["replication_tolerance"]
    difference = abs(achieved_auc - reported_auc)
    outcome = "Replicated" if difference <= tolerance else "Not replicated"

    result = {
        "reported_auc": round(reported_auc, 4),
        "replicated_auc": round(achieved_auc, 4),
        "absolute_difference": round(difference, 4),
        "tolerance": tolerance,
        "outcome": outcome,
    }

    findings: List[Finding] = []
    if outcome == "Not replicated":
        findings.append(
            Finding(
                ref="VF-09",
                severity="High",
                area="Replication",
                title="Reported performance could not be reproduced from the submitted recipe",
                observation=(
                    "Rebuilding the model from sections 3 and 4 of the development "
                    "document gives an AUC of {:.3f} against a reported {:.3f}, a gap of "
                    "{:.3f} against a tolerance of {:.3f}.".format(
                        achieved_auc, reported_auc, difference, tolerance
                    )
                ),
                implication=(
                    "The documentation does not describe the model that produced the "
                    "reported result, so no conclusion in it can be relied on."
                ),
                recommendation=(
                    "Reconcile the build to the documentation and resubmit before any "
                    "further review work is undertaken."
                ),
            )
        )
    return result, findings


def _single_variable_auc(frame: pd.DataFrame, column: str, target: str) -> float:
    """How much of the outcome one variable explains on its own."""
    y = frame[target].values
    if column in CATEGORICAL:
        rates = frame.groupby(column, observed=True)[target].mean()
        score = frame[column].map(rates).values.astype(float)
    else:
        score = frame[column].values.astype(float)

    value = metrics.auc(y, score)
    # Direction does not matter for this test, only strength.
    return max(value, 1.0 - value)


def variable_timing_review(
    development: pd.DataFrame, features: List[str], cfg: dict
) -> Tuple[pd.DataFrame, List[Finding]]:
    """Rank the variables by standalone strength and challenge the strongest ones.

    A variable that predicts the outcome almost perfectly on its own is either
    the single most valuable field the bank owns, or it is not available at the
    time the model runs. The second explanation is far more common, so the test
    raises it for the developer to answer rather than assuming either way.
    """
    target = "churn_next_6m"
    rows = [
        {"variable": column, "standalone_auc": round(_single_variable_auc(development, column, target), 4)}
        for column in features
    ]
    table = pd.DataFrame(rows).sort_values("standalone_auc", ascending=False).reset_index(drop=True)

    threshold = cfg["validation"]["single_variable_auc_flag"]
    table["flagged"] = np.where(table["standalone_auc"] >= threshold, "Yes", "No")

    findings: List[Finding] = []
    flagged = table[table["flagged"] == "Yes"]
    if not flagged.empty:
        names = ", ".join(flagged["variable"])
        strongest = flagged.iloc[0]
        findings.append(
            Finding(
                ref="VF-01",
                severity="High",
                area="Conceptual soundness",
                title="Model uses a variable that is only known after the outcome it predicts",
                observation=(
                    "{} reaches a standalone AUC of {:.3f} against a portfolio in which no "
                    "other variable exceeds {:.3f}. The data dictionary records it as being "
                    "written when a retention call is logged, and retention calls are placed "
                    "after a customer has given notice of leaving. Flagged variables: {}.".format(
                        strongest["variable"],
                        strongest["standalone_auc"],
                        table[table["flagged"] == "No"]["standalone_auc"].max(),
                        names,
                    )
                ),
                implication=(
                    "The variable will be absent, or will be zero for everyone, at the "
                    "moment the model is actually scored. Every performance figure in the "
                    "submission is therefore an overstatement of what the model can do in "
                    "production, and the size of the overstatement is not disclosed."
                ),
                recommendation=(
                    "Remove the variable, refit, and restate all reported performance. Add "
                    "a point-in-time availability check to the development standard so the "
                    "timing of every candidate variable is evidenced before it is used."
                ),
            )
        )
    return table, findings


def performance_row(
    label: str, sample: str, y_true: np.ndarray, probs: np.ndarray, cfg: dict
) -> dict:
    auc_value, gini_value, ks_value, ece, lift = metrics.score_summary(
        y_true,
        probs,
        bins=cfg["validation"]["calibration_bins"],
        cut=cfg["validation"]["selection_rate_cut"],
    )
    return {
        "model": label,
        "sample": sample,
        "customers": len(y_true),
        "churn_rate": round(float(np.mean(y_true)), 4),
        "auc": round(auc_value, 4),
        "gini": round(gini_value, 4),
        "ks": round(ks_value, 4),
        "calibration_error": round(ece, 4),
        "lift_top_20pct": round(lift, 3),
    }


def out_of_time_finding(
    corrected_holdout_auc: float, corrected_oot_auc: float, red_inputs: int
) -> List[Finding]:
    """Raise the choice of test design, measured on the corrected model.

    Measuring this on the corrected model is deliberate. On the model as
    submitted the split design and the post-outcome variable are tangled
    together, and the resulting number would credit this finding with an
    inflation that belongs to VF-01.
    """
    inflation = corrected_holdout_auc - corrected_oot_auc
    return [
        Finding(
            ref="VF-07",
            severity="Medium",
            area="Performance testing",
            title="Only performance evidence is a random split, which cannot detect a population shift",
            observation=(
                "The submission's sole performance evidence is a random 30% holdout drawn "
                "from the same twelve months as the training data. Holding the post-outcome "
                "variable out of both, the random holdout gives AUC {:.3f} against {:.3f} "
                "on the following six months, an inflation of {:.3f}.".format(
                    corrected_holdout_auc, corrected_oot_auc, inflation
                )
            ),
            implication=(
                "The inflation itself is small. The design problem is not: a random split "
                "draws its test customers from the same period as its training customers, "
                "so it cannot detect population movement by construction. {} inputs breach "
                "the red stability threshold over the six months following the build "
                "(section 9), and no test in the submission was capable of showing "
                "that.".format(red_inputs)
            ),
            recommendation=(
                "Make an out-of-time holdout the primary performance evidence for this "
                "model class and keep the random split as a secondary diagnostic. Restate "
                "the business case on the out-of-time figure."
            ),
        )
    ]


def calibration_review(
    y_true: np.ndarray, probs: np.ndarray, cfg: dict
) -> Tuple[pd.DataFrame, List[Finding]]:
    """Compare predicted probabilities against what actually happened."""
    bins = cfg["validation"]["calibration_bins"]
    table = metrics.calibration_table(y_true, probs, bins=bins)
    table = table.assign(band=[str(b) for b in table["band"]]).round(4)

    ece = metrics.expected_calibration_error(y_true, probs, bins=bins)
    mean_predicted = float(np.mean(probs))
    observed = float(np.mean(y_true))

    findings: List[Finding] = []
    if mean_predicted > 1.5 * observed:
        findings.append(
            Finding(
                ref="VF-03",
                severity="Medium",
                area="Calibration",
                title="Predicted probabilities are inflated by the class balancing step",
                observation=(
                    "Measured with the post-outcome variable removed, average predicted "
                    "churn probability is {:.1%} against an observed rate of {:.1%}, a "
                    "factor of {:.1f}. Weighted calibration error across ten bands is "
                    "{:.3f}, and every band over-predicts.".format(
                        mean_predicted, observed, mean_predicted / observed, ece
                    )
                ),
                implication=(
                    "The minority class was oversampled to a 50/50 base rate and the output "
                    "was never mapped back, so the model's probabilities are relative scores "
                    "presented as absolute ones. Any calculation that multiplies them by a "
                    "dollar value -- expected loss of balance, retention campaign business "
                    "case, provisioning input -- is overstated by roughly the same factor."
                ),
                recommendation=(
                    "Either drop the oversampling and use class weights, or keep it and fit "
                    "a calibration mapping on an untouched holdout. Restrict the current "
                    "version to ranking use only until one of the two is in place, and say "
                    "so on the model's approved-use record."
                ),
            )
        )
    return table, findings


def benchmark_finding(
    champion_auc: float, challenger_auc: float, cfg: dict, n_features: int
) -> List[Finding]:
    """Raise the complexity question if the simple model gets close enough."""
    margin = cfg["validation"]["challenger_parity_margin"]
    gap = champion_auc - challenger_auc
    if gap > margin:
        return []

    # Losing to the benchmark and merely tying with it are different problems,
    # and only the first one is a reason to stop using the model.
    outperformed = gap < 0
    if outperformed:
        severity = "High"
        title = "Champion is outperformed out-of-time by a logistic benchmark"
        implication = (
            "The submission justifies the model class on predictive power. Out-of-time it "
            "does not have any: the bank is carrying the monitoring, explainability and "
            "refit burden of a gradient-boosted model in exchange for {:.3f} AUC less "
            "accuracy than a regression it could put in a spreadsheet. Tuning was carried "
            "out against the random split, which is the pattern that produces exactly this "
            "result -- the extra flexibility fitted the development period rather than the "
            "signal.".format(-gap)
        )
        recommendation = (
            "Adopt the benchmark as the production model, or resubmit the gradient-boosted "
            "model with its hyperparameters selected against an out-of-time sample and "
            "with evidence that it beats the benchmark there. Either way the benchmark "
            "comparison belongs in the submission rather than in the review."
        )
    else:
        severity = "Low"
        title = "Gradient boosting is not earning its complexity over a logistic benchmark"
        implication = (
            "The bank is carrying the monitoring and explainability burden of a "
            "gradient-boosted model for a difference of {:.3f} AUC.".format(gap)
        )
        recommendation = (
            "Either document a business reason for the model class that does not rest on "
            "the performance gap, or adopt the logistic benchmark as the production model."
        )

    return [
        Finding(
            ref="VF-02",
            severity=severity,
            area="Model selection",
            title=title,
            observation=(
                "On the out-of-time sample the corrected gradient-boosted model scores AUC "
                "{:.3f} against {:.3f} for a {}-variable logistic regression fitted on the "
                "same data, a difference of {:+.3f} against a parity margin of "
                "{:.3f}.".format(
                    champion_auc, challenger_auc, n_features, gap, margin
                )
            ),
            implication=implication,
            recommendation=recommendation,
        )
    ]
