"""The model under review, rebuilt from the developer's own write-up.

Nothing in this file is my design. It is the recipe in sections 3 and 4 of
developer/model_development_document.md, implemented as described so that the
replication test in performance.py is testing their build and not mine. The
two choices that end up mattering -- a random train/test split over a
time-spanning population, and oversampling the minority class without
recalibrating afterwards -- are theirs, and are reproduced deliberately.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CATEGORICAL = ["province", "channel", "product_bundle_code"]

# Everything the developer put in front of the model, in their order.
SUBMITTED_FEATURES = [
    "age",
    "tenure_months",
    "province",
    "num_products",
    "balance",
    "credit_score",
    "is_active_member",
    "has_credit_card",
    "estimated_income",
    "channel",
    "digital_logins_90d",
    "complaints_12m",
    "product_bundle_code",
    "retention_call_flag",
]


@dataclass
class FittedModel:
    """A fitted model plus everything needed to score a new frame the same way."""

    name: str
    estimator: object
    features: List[str]
    design_columns: List[str]
    scaler: Optional[StandardScaler] = None
    reported: Dict[str, float] = field(default_factory=dict)
    # Labels and scores from the developer's own held-out split, kept so the
    # replication test can be run against the number their document quotes.
    holdout: Optional[Tuple[pd.Series, np.ndarray]] = None

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        design = encode(frame, self.features, self.design_columns)
        if self.scaler is not None:
            design = pd.DataFrame(
                self.scaler.transform(design), columns=design.columns
            )
        return self.estimator.predict_proba(design)[:, 1]


def encode(
    frame: pd.DataFrame, features: List[str], columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """One-hot the categoricals and line the columns up with the training design."""
    cats = [c for c in CATEGORICAL if c in features]
    design = pd.get_dummies(frame[features], columns=cats, dtype=float)
    if columns is not None:
        design = design.reindex(columns=columns, fill_value=0.0)
    return design


def _oversample(design: pd.DataFrame, y: pd.Series, seed: int):
    """Repeat minority rows until the two classes are level.

    This is the developer's class balancing step. It changes the base rate the
    model is fitted against, which is why the probabilities it produces are not
    the probabilities the business is told they are.
    """
    rng = np.random.default_rng(seed)
    minority = np.flatnonzero(y.values == 1)
    majority = np.flatnonzero(y.values == 0)
    if len(minority) == 0 or len(minority) >= len(majority):
        return design, y
    extra = rng.choice(minority, size=len(majority) - len(minority), replace=True)
    keep = np.concatenate([majority, minority, extra])
    rng.shuffle(keep)
    return design.iloc[keep].reset_index(drop=True), y.iloc[keep].reset_index(drop=True)


def fit_champion(
    development: pd.DataFrame,
    cfg: dict,
    seed: int,
    features: Optional[List[str]] = None,
    name: str = "champion_as_submitted",
) -> FittedModel:
    """Fit the gradient-boosted churn model exactly as the submission describes it."""
    features = list(features if features is not None else SUBMITTED_FEATURES)
    params = cfg["champion"]

    design = encode(development, features)
    y = development["churn_next_6m"]

    x_train, x_test, y_train, y_test = train_test_split(
        design,
        y,
        test_size=params["test_size"],
        random_state=seed,
        stratify=y,
    )
    if params["balance_classes"]:
        x_train, y_train = _oversample(x_train, y_train, seed)

    estimator = HistGradientBoostingClassifier(
        learning_rate=params["learning_rate"],
        max_iter=params["max_iter"],
        max_leaf_nodes=params["max_leaf_nodes"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=seed,
    )
    estimator.fit(x_train, y_train)

    model = FittedModel(
        name=name,
        estimator=estimator,
        features=features,
        design_columns=list(design.columns),
    )
    holdout_scores = estimator.predict_proba(x_test)[:, 1]
    model.holdout = (y_test.reset_index(drop=True), holdout_scores)
    model.reported = {"holdout_auc": float(roc_auc_score(y_test, holdout_scores))}
    return model


def fit_challenger(
    development: pd.DataFrame, seed: int, features: List[str], name: str = "challenger"
) -> FittedModel:
    """A plain logistic regression on a short variable list.

    Its job is to answer one question: how much of the champion's performance
    actually comes from the gradient boosting, and how much would a model the
    bank can explain to a customer have delivered anyway.
    """
    design = encode(development, features)
    scaler = StandardScaler().fit(design)
    scaled = pd.DataFrame(scaler.transform(design), columns=design.columns)

    estimator = LogisticRegression(max_iter=2000, random_state=seed)
    estimator.fit(scaled, development["churn_next_6m"])

    return FittedModel(
        name=name,
        estimator=estimator,
        features=features,
        design_columns=list(design.columns),
        scaler=scaler,
    )
