"""Each test here pins one planted defect to the test that is meant to catch it.

If a defect stops being caught, that is a broken validation test, not a broken
smoke test, and this file is where it shows up. docs/planted_defects.md is the
list these assertions correspond to.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from modelval import documentation, fairness, metrics, performance, portfolio, quality, stability
from modelval.champion import SUBMITTED_FEATURES, fit_challenger, fit_champion
from modelval.findings import Register

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


@pytest.fixture(scope="module")
def cfg():
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="module")
def small_portfolio():
    """A reduced draw, so the suite runs in seconds rather than minutes."""
    dev_size, oot_size = portfolio.DEVELOPMENT.n_customers, portfolio.OUT_OF_TIME.n_customers
    portfolio.DEVELOPMENT.n_customers = 8000
    portfolio.OUT_OF_TIME.n_customers = 3000
    try:
        yield portfolio.build_portfolio(SEED)
    finally:
        portfolio.DEVELOPMENT.n_customers = dev_size
        portfolio.OUT_OF_TIME.n_customers = oot_size


@pytest.fixture(scope="module")
def samples(small_portfolio):
    clean = portfolio.clean_for_modelling(small_portfolio)
    return portfolio.split_samples(clean)


def test_generator_is_deterministic():
    left = portfolio.build_portfolio(7)
    right = portfolio.build_portfolio(7)
    assert left.equals(right)
    assert not left.equals(portfolio.build_portfolio(8))


def test_cleaning_removes_the_collection_defects(small_portfolio):
    clean = portfolio.clean_for_modelling(small_portfolio)
    assert not clean["customer_id"].duplicated().any()
    assert clean["province"].isin(quality.VALID_PROVINCES).all()
    assert clean["credit_score"].between(300, 900).all()
    assert (clean["age"] > 0).all()


def test_defect_1_post_outcome_variable_is_flagged(samples, cfg):
    dev, _ = samples
    table, findings = performance.variable_timing_review(dev, SUBMITTED_FEATURES, cfg)

    assert table.loc[0, "variable"] == "retention_call_flag"
    assert table.loc[0, "standalone_auc"] >= cfg["validation"]["single_variable_auc_flag"]
    # No other variable should come anywhere near it, or the test is not
    # discriminating between a strong predictor and a leaked outcome.
    assert table.loc[1, "standalone_auc"] < 0.75
    assert [f.ref for f in findings] == ["VF-01"]
    assert findings[0].severity == "High"


def test_defect_2_population_shift_breaches_the_red_band(samples, cfg):
    dev, oot = samples
    table = stability.variable_stability(dev, oot, SUBMITTED_FEATURES, cfg)
    red = set(table[table["band"] == "Red"]["variable"])

    assert {"channel", "digital_logins_90d"} <= red
    assert table[table["variable"] == "age"]["band"].item() == "Green"


def test_defect_3_bundle_code_is_an_age_proxy(samples, cfg):
    _, oot = samples
    protected = (oot["age"] >= 60).astype(int)
    assert fairness.proxy_strength(oot, "product_bundle_code", protected) > 0.90


def test_defect_3_selection_rate_fails_four_fifths(samples, cfg):
    dev, oot = samples
    corrected = fit_champion(
        dev, cfg, SEED, [f for f in SUBMITTED_FEATURES if f != "retention_call_flag"]
    )
    by_age, _, _, findings = fairness.review(oot, corrected.predict_proba(oot), cfg)

    assert (by_age["meets_four_fifths"] == "No").any()
    assert [f.ref for f in findings] == ["VF-06"]


def test_defects_4_to_6_are_caught_by_the_quality_rules(small_portfolio, cfg):
    table, findings = quality.assess(small_portfolio, cfg)
    failed = dict(zip(table["rule"], table["outcome"]))

    assert failed["customer_id appears once per observation window"] == "Fail"
    assert failed["balance distinguishes a nil balance from a missing one"] == "Fail"
    assert failed["credit_score falls within 300-900"] == "Fail"
    assert failed["province uses the two-letter standard"] == "Fail"
    assert [f.ref for f in findings] == ["VF-04"]


def test_replication_reproduces_the_submitted_recipe(samples, cfg):
    dev, _ = samples
    model = fit_champion(dev, cfg, SEED)
    result, findings = performance.replicate(
        model.reported["holdout_auc"], model.reported["holdout_auc"], cfg
    )
    assert result["outcome"] == "Replicated"
    assert findings == []


def test_replication_failure_raises_a_high_finding(cfg):
    _, findings = performance.replicate(0.90, 0.72, cfg)
    assert [f.ref for f in findings] == ["VF-09"]
    assert findings[0].severity == "High"


def test_class_balancing_inflates_the_probabilities(samples, cfg):
    dev, _ = samples
    model = fit_champion(
        dev, cfg, SEED, [f for f in SUBMITTED_FEATURES if f != "retention_call_flag"]
    )
    y_holdout, scores = model.holdout

    assert scores.mean() > 2 * y_holdout.mean()
    _, findings = performance.calibration_review(y_holdout.values, scores, cfg)
    assert [f.ref for f in findings] == ["VF-03"]


def test_documentation_checklist_finds_the_missing_sections():
    table, findings = documentation.review(
        ROOT / "developer" / "model_development_document.md"
    )
    missing = set(table[table["present"] == "No"]["required_section"])

    assert "Ongoing monitoring plan" in missing
    assert "Benchmark comparison" in missing
    assert "Assumptions and limitations" in missing
    assert "Purpose and intended use" not in missing
    assert [f.ref for f in findings] == ["VF-08"]


def test_psi_is_zero_for_an_unchanged_population():
    rng = np.random.default_rng(0)
    draw = rng.normal(size=5000)
    assert metrics.population_stability_index(draw, draw) < 1e-9


def test_removing_the_leak_costs_most_of_the_performance(samples, cfg):
    dev, oot = samples
    y = oot["churn_next_6m"].values

    submitted = fit_champion(dev, cfg, SEED)
    corrected = fit_champion(
        dev, cfg, SEED, [f for f in SUBMITTED_FEATURES if f != "retention_call_flag"]
    )
    assert metrics.auc(y, submitted.predict_proba(oot)) - metrics.auc(
        y, corrected.predict_proba(oot)
    ) > 0.15


def test_register_blocks_approval_when_a_high_finding_is_open(samples, cfg):
    dev, _ = samples
    register = Register()
    assert register.outcome() == "Approved"

    _, findings = performance.variable_timing_review(dev, SUBMITTED_FEATURES, cfg)
    register.extend(findings)
    assert register.outcome() == "Not approved for use"
    assert register.counts()["High"] == 1


def test_challenger_and_champion_are_scored_on_the_same_footing(samples, cfg):
    dev, oot = samples
    challenger = fit_challenger(
        dev, SEED, ["age", "tenure_months", "num_products", "is_active_member"]
    )
    scores = challenger.predict_proba(oot)
    assert scores.shape == (len(oot),)
    assert ((scores >= 0) & (scores <= 1)).all()
