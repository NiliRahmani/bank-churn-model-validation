"""Run the full validation and write the report.

    python run.py                  full review, writes results/ and assets/
    python run.py --quick          smaller portfolio, for a fast check
    python run.py --seed 7         re-run the whole review on a different draw
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from modelval import documentation, fairness, performance, plots, portfolio, quality, report, stability
from modelval.champion import SUBMITTED_FEATURES, fit_challenger, fit_champion
from modelval.findings import Register

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"
DATA = ROOT / "data"

POST_OUTCOME_VARIABLE = "retention_call_flag"
AGE_PROXY_VARIABLE = "product_bundle_code"

CHALLENGER_FEATURES = [
    "age",
    "tenure_months",
    "num_products",
    "balance",
    "credit_score",
    "is_active_member",
    "has_credit_card",
    "complaints_12m",
]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent validation of RB-CHURN-07.")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Use a smaller portfolio.")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed = cfg["seed"]

    if args.quick:
        portfolio.DEVELOPMENT.n_customers = 6000
        portfolio.OUT_OF_TIME.n_customers = 2000

    for folder in (RESULTS, ASSETS, DATA):
        folder.mkdir(parents=True, exist_ok=True)

    raw = portfolio.build_portfolio(seed)
    raw.to_csv(DATA / "portfolio_extract.csv", index=False)
    register = Register()

    dq_table, dq_findings = quality.assess(raw, cfg)
    register.extend(dq_findings)

    doc_table, doc_findings = documentation.review(
        ROOT / "developer" / "model_development_document.md"
    )
    register.extend(doc_findings)

    clean = portfolio.clean_for_modelling(raw)
    dev, oot = portfolio.split_samples(clean)

    champion = fit_champion(dev, cfg, seed)
    reported = json.loads(
        (ROOT / "developer" / "reported_metrics.json").read_text(encoding="utf-8")
    )
    replication, replication_findings = performance.replicate(
        reported["holdout_auc"], champion.reported["holdout_auc"], cfg
    )
    register.extend(replication_findings)

    variable_table, variable_findings = performance.variable_timing_review(
        dev, SUBMITTED_FEATURES, cfg
    )
    register.extend(variable_findings)

    corrected_features = [f for f in SUBMITTED_FEATURES if f != POST_OUTCOME_VARIABLE]
    corrected = fit_champion(
        dev, cfg, seed, corrected_features, name="champion_corrected"
    )

    no_bundle_features = [f for f in corrected_features if f != AGE_PROXY_VARIABLE]
    no_bundle = fit_champion(
        dev, cfg, seed, no_bundle_features, name="champion_without_bundle_code"
    )

    challenger = fit_challenger(dev, seed, CHALLENGER_FEATURES)

    y_holdout, holdout_scores = champion.holdout
    y_oot = oot["churn_next_6m"].values
    champion_oot = champion.predict_proba(oot)
    corrected_oot = corrected.predict_proba(oot)
    no_bundle_oot = no_bundle.predict_proba(oot)
    challenger_oot = challenger.predict_proba(oot)

    performance_table = pd.DataFrame(
        [
            performance.performance_row(
                "Champion as submitted",
                "Developer random holdout",
                y_holdout.values,
                holdout_scores,
                cfg,
            ),
            performance.performance_row(
                "Champion as submitted", "Out-of-time", y_oot, champion_oot, cfg
            ),
            performance.performance_row(
                "Champion without post-outcome variable",
                "Out-of-time",
                y_oot,
                corrected_oot,
                cfg,
            ),
            performance.performance_row(
                "Logistic benchmark ({} variables)".format(len(CHALLENGER_FEATURES)),
                "Out-of-time",
                y_oot,
                challenger_oot,
                cfg,
            ),
        ]
    )

    # Calibration is assessed on the corrected model. On the model as submitted
    # the post-outcome variable pushes most scores to the two extremes, which
    # masks the oversampling effect rather than removing it -- and the corrected
    # model is the one that could actually be deployed.
    y_corrected_holdout, corrected_holdout_scores = corrected.holdout
    calibration_table, calibration_findings = performance.calibration_review(
        y_corrected_holdout.values, corrected_holdout_scores, cfg
    )
    register.extend(calibration_findings)

    psi_table = stability.variable_stability(dev, oot, SUBMITTED_FEATURES, cfg)
    score_psi, score_band = stability.score_stability(
        corrected.predict_proba(dev), corrected_oot, cfg
    )
    register.extend(stability.stability_findings(psi_table, score_psi, score_band, cfg))

    # Raised after stability so the finding can quote how many inputs moved,
    # which is the reason the split design matters here.
    register.extend(
        performance.out_of_time_finding(
            corrected.reported["holdout_auc"],
            performance_table.loc[2, "auc"],
            int((psi_table["band"] == "Red").sum()),
        )
    )

    age_table, province_table, bundle_proxy, fairness_findings = fairness.review(
        oot, corrected_oot, cfg
    )
    register.extend(fairness_findings)

    register.extend(
        performance.benchmark_finding(
            performance_table.loc[2, "auc"],
            performance_table.loc[3, "auc"],
            cfg,
            len(CHALLENGER_FEATURES),
        )
    )

    figure_curves = [
        ("Champion, developer holdout", y_holdout.values, holdout_scores, performance_table.loc[0, "auc"]),
        ("Champion, out-of-time", y_oot, champion_oot, performance_table.loc[1, "auc"]),
        ("Corrected, out-of-time", y_oot, corrected_oot, performance_table.loc[2, "auc"]),
        ("Logistic benchmark, out-of-time", y_oot, challenger_oot, performance_table.loc[3, "auc"]),
    ]
    plots.write_all(figure_curves, calibration_table, psi_table, age_table, cfg, ASSETS)

    key = {
        "reported_auc": float(reported["holdout_auc"]),
        "oot_auc": float(performance_table.loc[1, "auc"]),
        "corrected_oot_auc": float(performance_table.loc[2, "auc"]),
        "corrected_holdout_auc": float(corrected.reported["holdout_auc"]),
        "challenger_oot_auc": float(performance_table.loc[3, "auc"]),
        "no_bundle_oot_auc": float(
            performance.performance_row(
                "no bundle", "Out-of-time", y_oot, no_bundle_oot, cfg
            )["auc"]
        ),
        "challenger_variables": len(CHALLENGER_FEATURES),
        "top_variable_auc": float(variable_table.loc[0, "standalone_auc"]),
        "next_variable_auc": float(variable_table.loc[1, "standalone_auc"]),
        "mean_predicted": float(np.mean(corrected_holdout_scores)),
        "observed_rate": float(np.mean(y_corrected_holdout.values)),
        "score_psi": score_psi,
        "score_psi_band": score_band,
        "bundle_proxy": float(bundle_proxy),
    }

    register_table = register.to_frame()
    results = {
        "cfg": cfg,
        "dq_table": dq_table,
        "doc_table": doc_table,
        "replication": replication,
        "variable_table": variable_table,
        "performance_table": performance_table,
        "calibration_table": calibration_table,
        "psi_table": psi_table,
        "age_table": age_table,
        "province_table": province_table,
        "register": register,
        "register_table": register_table,
        "key": key,
    }

    for name, frame in (
        ("data_quality", dq_table),
        ("documentation_checklist", doc_table),
        ("variable_review", variable_table),
        ("performance", performance_table),
        ("calibration", calibration_table),
        ("stability", psi_table),
        ("selection_rate_by_age", age_table),
        ("selection_rate_by_province", province_table),
        ("findings_register", register_table),
    ):
        frame.to_csv(RESULTS / "{}.csv".format(name), index=False)

    (RESULTS / "headline_metrics.json").write_text(
        json.dumps(key, indent=2), encoding="utf-8"
    )

    md_path, _ = report.write(
        results, RESULTS, "Model Validation Report - {}".format(cfg["review"]["model_id"])
    )

    counts = register.counts()
    print("Outcome: {}".format(register.outcome()))
    print(
        "Findings: {} High, {} Medium, {} Low".format(
            counts["High"], counts["Medium"], counts["Low"]
        )
    )
    print(
        "Reported AUC {:.3f} -> out-of-time {:.3f} -> corrected out-of-time {:.3f}".format(
            key["reported_auc"], key["oot_auc"], key["corrected_oot_auc"]
        )
    )
    print("Report: {}".format(md_path.relative_to(ROOT)))


if __name__ == "__main__":
    main()
