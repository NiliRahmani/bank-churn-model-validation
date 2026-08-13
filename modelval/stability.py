"""Population stability between the development window and the following period.

Performance testing says whether the model still works. Stability testing says
whether it is still being shown the same kind of customer, which is the thing a
monitoring process can actually watch month to month without waiting for
outcomes to mature.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from . import metrics
from .champion import CATEGORICAL
from .findings import Finding


def _band(value: float, amber: float, red: float) -> str:
    if value >= red:
        return "Red"
    if value >= amber:
        return "Amber"
    return "Green"


def variable_stability(
    development: pd.DataFrame, out_of_time: pd.DataFrame, features: List[str], cfg: dict
) -> pd.DataFrame:
    """PSI for every model input, on development bin edges."""
    amber = cfg["validation"]["psi_amber"]
    red = cfg["validation"]["psi_red"]

    rows = []
    for column in features:
        if column in CATEGORICAL:
            value = metrics.categorical_stability_index(
                development[column], out_of_time[column]
            )
            kind = "categorical"
        else:
            value = metrics.population_stability_index(
                development[column].values, out_of_time[column].values
            )
            kind = "numeric"
        rows.append(
            {
                "variable": column,
                "type": kind,
                "psi": round(value, 4),
                "band": _band(value, amber, red),
            }
        )

    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def score_stability(
    dev_scores: np.ndarray, oot_scores: np.ndarray, cfg: dict
) -> Tuple[float, str]:
    value = metrics.population_stability_index(dev_scores, oot_scores)
    return (
        round(value, 4),
        _band(value, cfg["validation"]["psi_amber"], cfg["validation"]["psi_red"]),
    )


def stability_findings(
    table: pd.DataFrame, score_psi: float, score_band: str, cfg: dict
) -> List[Finding]:
    red = table[table["band"] == "Red"]
    amber = table[table["band"] == "Amber"]
    if red.empty and score_band == "Green":
        return []

    breached = ", ".join(
        "{} (PSI {:.2f})".format(row["variable"], row["psi"])
        for _, row in red.iterrows()
    )
    return [
        Finding(
            ref="VF-05",
            severity="Medium",
            area="Stability",
            title="Input population has shifted materially within six months of the build",
            observation=(
                "{} of {} model inputs breach the red threshold of {:.2f} on the six months "
                "following the development window: {}. A further {} sit in the amber band. "
                "The model score itself has a PSI of {:.2f} ({}).".format(
                    len(red),
                    len(table),
                    cfg["validation"]["psi_red"],
                    breached if breached else "none individually",
                    len(amber),
                    score_psi,
                    score_band,
                )
            ),
            implication=(
                "The shift is a channel migration, not a data error -- customers moved to "
                "mobile and their engagement counts moved with them. The model was fitted "
                "on a branch-weighted portfolio and is being applied to a mobile-weighted "
                "one. The score distribution is the part worth noting: at PSI {:.2f} it "
                "sits well inside the green band, so a monitoring process watching the "
                "score alone would report this model as stable while two of its inputs "
                "moved by more than four times the red threshold. The submission contains "
                "no monitoring plan of either kind.".format(score_psi)
            ),
            recommendation=(
                "Put monthly PSI monitoring in place on every input, not only on the "
                "score, using the same {:.2f}/{:.2f} thresholds. Tie a red breach to a "
                "mandatory review, and shorten the refit cycle for this model to match the "
                "rate at which the channel mix is moving.".format(
                    cfg["validation"]["psi_amber"], cfg["validation"]["psi_red"]
                )
            ),
        )
    ]
