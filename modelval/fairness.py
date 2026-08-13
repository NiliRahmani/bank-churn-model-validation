"""Who the model selects, and whether the split can be justified.

The model ranks customers for a retention campaign, so the outcome that matters
to a customer is whether they land in the contacted group. That makes selection
rate the right quantity to test, not accuracy.

Two separate questions are kept apart here, because conflating them is the usual
way this analysis goes wrong:

    a. is the selection rate uneven across age bands
    b. if it is, is the model reading age directly through another field

An uneven rate on its own can be legitimate -- groups can genuinely differ in
churn risk. A near-perfect proxy for a protected ground is a different problem,
and it does not stop being one just because the underlying difference is real.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from . import metrics
from .findings import Finding

AGE_BANDS = [(0, 29, "18-29"), (30, 44, "30-44"), (45, 59, "45-59"), (60, 200, "60+")]


def age_band(age: pd.Series) -> pd.Series:
    labels = pd.Series(index=age.index, dtype=object)
    for low, high, name in AGE_BANDS:
        labels[(age >= low) & (age <= high)] = name
    return labels


def selection_table(
    frame: pd.DataFrame, scores: np.ndarray, group_column: str, cfg: dict
) -> pd.DataFrame:
    """Share of each group that lands in the contacted slice, and the impact ratio."""
    cut = cfg["validation"]["selection_rate_cut"]
    threshold = np.quantile(scores, 1.0 - cut)

    work = frame.copy()
    work["selected"] = (scores >= threshold).astype(int)

    table = (
        work.groupby(group_column, observed=True)
        .agg(
            customers=("selected", "size"),
            selected=("selected", "sum"),
            observed_churn=("churn_next_6m", "mean"),
        )
        .reset_index()
    )
    table["selection_rate"] = table["selected"] / table["customers"]
    best = table["selection_rate"].max()
    table["impact_ratio"] = table["selection_rate"] / best
    table["meets_four_fifths"] = np.where(
        table["impact_ratio"] >= cfg["validation"]["four_fifths_threshold"], "Yes", "No"
    )
    return table.round(4)


def proxy_strength(frame: pd.DataFrame, column: str, protected: pd.Series) -> float:
    """How well one categorical field alone recovers membership of the protected group."""
    rates = protected.groupby(frame[column], observed=True).mean()
    score = frame[column].map(rates).values.astype(float)
    value = metrics.auc(protected.values, score)
    return max(value, 1.0 - value)


def review(
    frame: pd.DataFrame, scores: np.ndarray, cfg: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, float, List[Finding]]:
    """Run both questions and return the tables plus any finding they raise."""
    work = frame.copy()
    work["age_band"] = age_band(work["age"])

    by_age = selection_table(work, scores, "age_band", cfg)
    by_province = selection_table(work, scores, "province", cfg)

    protected = (work["age"] >= 60).astype(int)
    bundle_proxy = proxy_strength(work, "product_bundle_code", protected)

    findings: List[Finding] = []
    failing = by_age[by_age["meets_four_fifths"] == "No"]
    if not failing.empty:
        worst = failing.sort_values("impact_ratio").iloc[0]
        findings.append(
            Finding(
                ref="VF-06",
                severity="Medium",
                area="Fairness",
                title="Age band selection rate fails the four-fifths test, through a bundle code that restates age",
                observation=(
                    "At the top {:.0%} campaign cut, customers in the {} band are selected "
                    "at {:.1%} against {:.1%} for the highest band, an impact ratio of "
                    "{:.2f}. Separately, product_bundle_code alone identifies customers "
                    "aged 60 and over with an AUC of {:.3f}, so age band is available to "
                    "the model whether or not age is used directly.".format(
                        cfg["validation"]["selection_rate_cut"],
                        worst["age_band"],
                        worst["selection_rate"],
                        by_age["selection_rate"].max(),
                        worst["impact_ratio"],
                        bundle_proxy,
                    )
                ),
                implication=(
                    "Part of the gap reflects a real difference in churn risk, and the "
                    "observed churn column supports that. The problem is not the gap on its "
                    "own but that the model reaches it through a near-perfect stand-in for "
                    "a protected ground, so removing age from the feature list would not "
                    "have changed the outcome and the submission's statement that no "
                    "protected characteristic is used is not accurate. A retention offer is "
                    "a benefit, so the group being under-selected is being withheld from."
                ),
                recommendation=(
                    "Refit without product_bundle_code and report the performance cost. If "
                    "the field is retained, record the business justification, disclose the "
                    "proxy relationship in the model documentation, and add selection rate "
                    "by age band to the monthly monitoring pack."
                ),
            )
        )
    return by_age, by_province, bundle_proxy, findings
