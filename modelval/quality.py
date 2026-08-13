"""Data quality assessment of the extract the model was built on.

This runs before anything is measured about the model itself. If the inputs
cannot be trusted, the performance numbers further down are describing a
different portfolio than the one the bank actually has.

Rules are grouped by the six dimensions the bank's data governance standard
uses, and each rule is written so it can only pass or fail -- no judgement is
applied at the point of measurement.
"""
from __future__ import annotations

from datetime import date
from typing import List, Tuple

import pandas as pd

from .findings import Finding

VALID_PROVINCES = {"ON", "QC", "BC", "AB", "MB", "NS"}
CREDIT_SCORE_RANGE = (300, 900)


def _rule(dimension: str, rule: str, failed: int, total: int, note: str) -> dict:
    share = failed / total if total else 0.0
    return {
        "dimension": dimension,
        "rule": rule,
        "records_failed": int(failed),
        "share_failed": round(share, 4),
        "outcome": "Fail" if failed > 0 else "Pass",
        "note": note,
    }


def _months_between(start: date, end: date) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month)


def assess(raw: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, List[Finding]]:
    """Score the raw extract against the data quality rules."""
    total = len(raw)
    rules = []

    duplicated = int(raw["customer_id"].duplicated().sum())
    rules.append(
        _rule(
            "Uniqueness",
            "customer_id appears once per observation window",
            duplicated,
            total,
            "Extract was appended twice for part of the load.",
        )
    )

    blank_age = int((raw["age"] == 0).sum())
    rules.append(
        _rule(
            "Completeness",
            "age is populated",
            blank_age,
            total,
            "Age written as 0 where the value was not captured.",
        )
    )

    sentinel_score = int(raw["credit_score"].isin([999, -1]).sum())
    rules.append(
        _rule(
            "Completeness",
            "credit_score is populated",
            sentinel_score,
            total,
            "Two sentinel conventions in use: 999 and -1.",
        )
    )

    out_of_range = int(
        (
            (raw["credit_score"] < CREDIT_SCORE_RANGE[0])
            | (raw["credit_score"] > CREDIT_SCORE_RANGE[1])
        ).sum()
    )
    rules.append(
        _rule(
            "Validity",
            "credit_score falls within 300-900",
            out_of_range,
            total,
            "Same records as the sentinel values above.",
        )
    )

    bad_province = int((~raw["province"].isin(VALID_PROVINCES)).sum())
    rules.append(
        _rule(
            "Validity",
            "province uses the two-letter standard",
            bad_province,
            total,
            "Long-form names and lower case arriving from upstream systems.",
        )
    )

    distinct_province = raw["province"].nunique()
    rules.append(
        _rule(
            "Consistency",
            "province has at most 6 distinct values",
            max(0, distinct_province - len(VALID_PROVINCES)),
            total,
            "{} distinct values observed for 6 real provinces.".format(
                distinct_province
            ),
        )
    )

    zero_balance = int((raw["balance"] == 0).sum())
    rules.append(
        _rule(
            "Accuracy",
            "balance distinguishes a nil balance from a missing one",
            zero_balance,
            total,
            "Zero is being used for both, which an average cannot tell apart.",
        )
    )

    review_date = pd.to_datetime(cfg["review"]["review_date"]).date()
    dev_end = pd.to_datetime(
        cfg["portfolio"]["development"]["window"].split(" to ")[1]
    ).date()
    age_months = _months_between(dev_end, review_date)
    limit = cfg["review"]["max_development_age_months"]
    rules.append(
        _rule(
            "Timeliness",
            "development sample is under {} months old at review".format(limit),
            0 if age_months <= limit else 1,
            1,
            "Development window closed {} months before this review.".format(
                age_months
            ),
        )
    )

    table = pd.DataFrame(rules)
    findings = _raise_findings(table, duplicated, zero_balance, total)
    return table, findings


def _raise_findings(
    table: pd.DataFrame, duplicated: int, zero_balance: int, total: int
) -> List[Finding]:
    findings: List[Finding] = []
    failed = table[table["outcome"] == "Fail"]
    if failed.empty:
        return findings

    dimensions = ", ".join(sorted(failed["dimension"].unique()))
    findings.append(
        Finding(
            ref="VF-04",
            severity="Medium",
            area="Data quality",
            title="Input extract fails data quality rules that the submission does not mention",
            observation=(
                "{} of {} rules fail, across {}. {:,} records ({:.1%}) are duplicated on "
                "customer_id and {:,} records ({:.1%}) carry a balance of zero that stands "
                "in for a value that was never supplied.".format(
                    len(failed),
                    len(table),
                    dimensions,
                    duplicated,
                    duplicated / total,
                    zero_balance,
                    zero_balance / total,
                )
            ),
            implication=(
                "Duplicated customers are counted twice in the training data and are "
                "over-weighted by the fit. Averaging a balance field that encodes "
                "'not supplied' as zero understates average balance across the "
                "portfolio, which affects both the model input and the reporting "
                "built on the same extract."
            ),
            recommendation=(
                "De-duplicate on customer_id at the point of extraction, replace both "
                "credit score sentinels with a null and an explicit missing indicator, "
                "and separate a nil balance from an unsupplied one before the next "
                "refit. Document the treatment in the model development document."
            ),
        )
    )
    return findings
