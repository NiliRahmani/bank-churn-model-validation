"""The retail-banking portfolio the model under review was built on.

The data is generated rather than downloaded, for one reason: a validation
exercise is only convincing if someone can check the answers. Six defects are
planted here on purpose, and each one has a matching test in the validation
modules that is supposed to catch it:

    1. `retention_call_flag` is written to the record *after* the customer has
       already signalled they are leaving, so it cannot be known at scoring time
    2. the channel mix and digital engagement shift between the development
       window and the out-of-time window
    3. `product_bundle_code` is close to a restatement of age band
    4. a small share of customer records are duplicated
    5. `balance` uses 0 where the value was never supplied
    6. `credit_score` and `province` carry out-of-range and inconsistently
       coded values

docs/planted_defects.md lists all six. I wrote the validation tests first and
checked them against that list afterwards, not the other way round.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

PROVINCES = ["ON", "QC", "BC", "AB", "MB", "NS"]
PROVINCE_WEIGHTS = [0.40, 0.22, 0.15, 0.13, 0.05, 0.05]

CHANNELS = ["branch", "online", "mobile"]
# Logins are driven by channel, so the channel shift in 2024 drags the
# engagement variables with it. That is what makes the drift realistic --
# in practice populations do not move one variable at a time.
LOGINS_BY_CHANNEL = {"branch": 3.0, "online": 12.0, "mobile": 25.0}
CHURN_EFFECT_BY_CHANNEL = {"branch": -0.15, "online": 0.0, "mobile": 0.25}

BUNDLES = ["STD", "PLUS", "SENIOR_PLUS", "STUDENT"]

FEATURES = [
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
TARGET = "churn_next_6m"


@dataclass
class Window:
    """One observation window of the portfolio."""

    name: str
    n_customers: int
    window: str
    channel_weights: Dict[str, float]
    complaint_effect: float
    mobile_effect: float
    intercept: float


DEVELOPMENT = Window(
    name="development",
    n_customers=24000,
    window="2023-01-01 to 2023-12-31",
    channel_weights={"branch": 0.45, "online": 0.25, "mobile": 0.30},
    complaint_effect=0.55,
    mobile_effect=0.25,
    intercept=-2.35,
)

# The 2024 population is not the 2023 population. Customers moved to mobile,
# and complaints started mattering more than they used to.
OUT_OF_TIME = Window(
    name="out_of_time",
    n_customers=8000,
    window="2024-01-01 to 2024-06-30",
    channel_weights={"branch": 0.18, "online": 0.24, "mobile": 0.58},
    complaint_effect=0.85,
    mobile_effect=0.55,
    intercept=-2.20,
)


def _assign_bundle(age: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Product bundle a customer was sold, which tracks their age band closely."""
    draw = rng.random(age.shape[0])
    bundle = np.full(age.shape[0], "STD", dtype=object)
    senior = age >= 60
    student = age <= 24
    middle = ~senior & ~student

    bundle[senior & (draw < 0.85)] = "SENIOR_PLUS"
    bundle[student & (draw < 0.70)] = "STUDENT"
    bundle[middle & (draw < 0.35)] = "PLUS"
    return bundle


def _generate_window(win: Window, rng: np.random.Generator, id_start: int) -> pd.DataFrame:
    n = win.n_customers

    age = np.clip(np.round(18 + rng.gamma(4.0, 8.0, n)), 18, 92).astype(int)
    tenure_months = np.clip(
        np.round(rng.gamma(2.0, 26.0, n) * (age / 50.0)), 0, 300
    ).astype(int)
    province = rng.choice(PROVINCES, size=n, p=PROVINCE_WEIGHTS)

    channel = rng.choice(
        CHANNELS, size=n, p=[win.channel_weights[c] for c in CHANNELS]
    )
    login_rate = np.array([LOGINS_BY_CHANNEL[c] for c in channel])
    digital_logins_90d = rng.poisson(login_rate).astype(int)

    num_products = np.clip(1 + rng.poisson(0.75, n), 1, 4).astype(int)
    has_credit_card = (rng.random(n) < 0.70).astype(int)
    is_active_member = (rng.random(n) < 0.55).astype(int)

    balance = np.round(rng.lognormal(9.1, 1.05, n), 2)
    estimated_income = np.round(
        rng.lognormal(10.9, 0.45, n) * np.clip(age / 45.0, 0.6, 1.35), 2
    )
    credit_score = np.clip(np.round(rng.normal(720, 70, n)), 300, 900).astype(int)

    complaints_12m = rng.poisson(0.18 + 0.22 * (1 - is_active_member)).astype(int)
    product_bundle_code = _assign_bundle(age, rng)

    channel_effect = np.array([CHURN_EFFECT_BY_CHANNEL[c] for c in channel])
    channel_effect = np.where(channel == "mobile", win.mobile_effect, channel_effect)

    logit = (
        win.intercept
        + 0.95 * (1 - is_active_member)
        + win.complaint_effect * (complaints_12m > 0)
        + 0.45 * (num_products == 1)
        - 0.40 * np.log1p(tenure_months / 12.0)
        - 0.30 * has_credit_card
        - 0.012 * (age - 45)
        - 0.0035 * (credit_score - 720)
        + channel_effect
    )
    churn = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)

    # Retention calls are logged against the customer once they have already
    # told the branch they are leaving. It is a consequence of the outcome,
    # not a predictor of it.
    retention_call_flag = np.where(
        churn == 1, rng.random(n) < 0.78, rng.random(n) < 0.04
    ).astype(int)

    frame = pd.DataFrame(
        {
            "customer_id": np.arange(id_start, id_start + n),
            "observation_window": win.window,
            "sample": win.name,
            "age": age,
            "tenure_months": tenure_months,
            "province": province,
            "num_products": num_products,
            "balance": balance,
            "credit_score": credit_score,
            "is_active_member": is_active_member,
            "has_credit_card": has_credit_card,
            "estimated_income": estimated_income,
            "channel": channel,
            "digital_logins_90d": digital_logins_90d,
            "complaints_12m": complaints_12m,
            "product_bundle_code": product_bundle_code,
            "retention_call_flag": retention_call_flag,
            TARGET: churn,
        }
    )
    return frame


def _add_collection_defects(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Apply the wear and tear a real extract picks up on its way to the analyst."""
    n = len(frame)

    # A share of balances were never supplied and were written down as zero
    # rather than left null, which is the version that causes trouble later.
    missing_balance = rng.random(n) < 0.12
    frame.loc[missing_balance, "balance"] = 0.0

    # Two different sentinel conventions for an unavailable bureau score.
    no_score = rng.random(n) < 0.008
    stale_score = rng.random(n) < 0.003
    frame.loc[no_score, "credit_score"] = 999
    frame.loc[stale_score, "credit_score"] = -1

    # Age was blanked to zero on a small number of records.
    frame.loc[rng.random(n) < 0.002, "age"] = 0

    # Province arrives from three upstream systems with three conventions.
    recode = rng.random(n) < 0.03
    long_form = {"ON": "Ontario", "QC": "Quebec", "BC": "British Columbia"}
    idx = frame.index[recode]
    for i in idx:
        code = frame.at[i, "province"]
        style = rng.integers(0, 3)
        if style == 0:
            frame.at[i, "province"] = code.lower()
        elif style == 1:
            frame.at[i, "province"] = long_form.get(code, code)
        else:
            frame.at[i, "province"] = " " + code + " "

    # The extract was appended twice for part of the run.
    dup_idx = rng.choice(frame.index, size=int(0.015 * n), replace=False)
    frame = pd.concat([frame, frame.loc[dup_idx]], ignore_index=True)
    return frame


def build_portfolio(seed: int = 42) -> pd.DataFrame:
    """Return the development and out-of-time samples as one frame."""
    rng = np.random.default_rng(seed)
    dev = _generate_window(DEVELOPMENT, rng, id_start=100000)
    oot = _generate_window(OUT_OF_TIME, rng, id_start=900000)
    combined = pd.concat([dev, oot], ignore_index=True)
    combined = _add_collection_defects(combined, rng)
    return combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def clean_for_modelling(frame: pd.DataFrame) -> pd.DataFrame:
    """Remediate the collection defects so the modelling tests compare like with like.

    This is the treatment the developer should have applied and documented.
    Everything done here is reported in section 3 of the validation report.
    """
    out = frame.drop_duplicates(subset="customer_id", keep="first").copy()

    out["province"] = out["province"].str.strip().str.upper()
    out["province"] = out["province"].replace(
        {"ONTARIO": "ON", "QUEBEC": "QC", "BRITISH COLUMBIA": "BC"}
    )

    # Sentinels become genuine nulls, then a flag so the absence itself stays
    # available to the model instead of being silently imputed away.
    out.loc[out["credit_score"].isin([999, -1]), "credit_score"] = np.nan
    out["credit_score_missing"] = out["credit_score"].isna().astype(int)
    out["credit_score"] = out["credit_score"].fillna(out["credit_score"].median())

    out["balance_not_supplied"] = (out["balance"] == 0).astype(int)

    out.loc[out["age"] == 0, "age"] = np.nan
    out["age"] = out["age"].fillna(out["age"].median()).astype(int)

    return out.reset_index(drop=True)


def split_samples(frame: pd.DataFrame):
    """Development and out-of-time frames, in that order."""
    dev = frame[frame["sample"] == "development"].reset_index(drop=True)
    oot = frame[frame["sample"] == "out_of_time"].reset_index(drop=True)
    return dev, oot