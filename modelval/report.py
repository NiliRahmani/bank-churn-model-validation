"""Assembles the validation report.

The report is written once as an ordered list of blocks and then rendered to
both Markdown and HTML. Writing it twice by hand is how the two copies end up
disagreeing, and a report that disagrees with itself is worth nothing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd

Block = Tuple[str, Any]

CSS = """
:root { color-scheme: light; }
body {
  font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
  color: #0b0b0b; background: #fcfcfb;
  margin: 0 auto; padding: 32px 40px 56px; max-width: 860px;
  font-size: 10.5pt; line-height: 1.55;
}
h1 { font-size: 21pt; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 13.5pt; margin: 30px 0 10px; padding-bottom: 6px;
     border-bottom: 1px solid #e1e0d9; }
h3 { font-size: 11pt; margin: 20px 0 6px; color: #52514e; }
p { margin: 0 0 10px; }
ul { margin: 0 0 12px; padding-left: 20px; }
li { margin-bottom: 5px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0 18px;
        font-size: 8.8pt; font-variant-numeric: tabular-nums; }
th { text-align: left; background: #f0efec; color: #52514e;
     font-weight: 600; padding: 6px 8px; border-bottom: 1px solid #c3c2b7; }
td { padding: 5px 8px; border-bottom: 1px solid #e1e0d9; vertical-align: top; }
tr:last-child td { border-bottom: 1px solid #c3c2b7; }
figure { margin: 16px 0 20px; }
figure img { width: 100%; border: 1px solid #e1e0d9; border-radius: 4px; }
figcaption { font-size: 8.5pt; color: #898781; margin-top: 6px; }
.meta { color: #52514e; font-size: 9.5pt; margin-bottom: 22px; }
.callout { background: #f0efec; border-left: 3px solid #2a78d6;
           padding: 12px 16px; margin: 16px 0 20px; }
.callout p:last-child { margin-bottom: 0; }
@page { size: A4; margin: 14mm 14mm 16mm; }
h2 { break-after: avoid; }
table, figure { break-inside: avoid; }
"""


def heading(level: int, text: str) -> Block:
    return ("h{}".format(level), text)


def paragraph(text: str) -> Block:
    return ("p", text)


def bullets(items: List[str]) -> Block:
    return ("ul", items)


def table(frame: pd.DataFrame) -> Block:
    return ("table", frame)


def figure(path: str, caption: str) -> Block:
    return ("figure", (path, caption))


def callout(text: str) -> Block:
    return ("callout", text)


def meta(text: str) -> Block:
    return ("meta", text)


def _typeset(text: str) -> str:
    """Source stays ASCII; the rendered report gets a real dash."""
    return str(text).replace(" -- ", " — ")


def _typeset_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if out[column].dtype == object:
            out[column] = out[column].map(_typeset)
    return out


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(c).replace("_", " ") for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for _, row in _typeset_frame(frame).iterrows():
        cells = [str(v).replace("|", "/").replace("\n", " ") for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_markdown(blocks: List[Block]) -> str:
    out: List[str] = []
    for kind, payload in blocks:
        if kind.startswith("h") and len(kind) == 2:
            out.append("{} {}".format("#" * int(kind[1]), payload))
        elif kind == "p":
            out.append(_typeset(payload))
        elif kind == "meta":
            out.append("*{}*".format(_typeset(payload)))
        elif kind == "callout":
            out.append("> {}".format(_typeset(payload)))
        elif kind == "ul":
            out.append("\n".join("- {}".format(_typeset(item)) for item in payload))
        elif kind == "table":
            out.append(_markdown_table(payload))
        elif kind == "figure":
            path, caption = payload
            out.append("![{}]({})\n\n*{}*".format(caption, path, caption))
    return "\n\n".join(out) + "\n"


def _escape(text: str) -> str:
    safe = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return _typeset(safe)


def render_html(blocks: List[Block], title: str) -> str:
    out = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        "<title>{}</title>".format(_escape(title)),
        "<style>{}</style></head><body>".format(CSS),
    ]
    for kind, payload in blocks:
        if kind.startswith("h") and len(kind) == 2:
            out.append("<{0}>{1}</{0}>".format(kind, _escape(payload)))
        elif kind == "p":
            out.append("<p>{}</p>".format(_escape(payload)))
        elif kind == "meta":
            out.append('<p class="meta">{}</p>'.format(_escape(payload)))
        elif kind == "callout":
            out.append('<div class="callout"><p>{}</p></div>'.format(_escape(payload)))
        elif kind == "ul":
            items = "".join("<li>{}</li>".format(_escape(i)) for i in payload)
            out.append("<ul>{}</ul>".format(items))
        elif kind == "table":
            out.append(
                _typeset_frame(payload).to_html(
                    index=False, border=0, escape=True, na_rep=""
                )
            )
        elif kind == "figure":
            path, caption = payload
            out.append(
                '<figure><img src="{}"><figcaption>{}</figcaption></figure>'.format(
                    path, _escape(caption)
                )
            )
    out.append("</body></html>")
    return "\n".join(out)


def build(results: dict) -> List[Block]:
    """The report itself, in reading order."""
    cfg = results["cfg"]
    review = cfg["review"]
    key = results["key"]
    register = results["register"]
    counts = register.counts()

    difference = results["replication"]["absolute_difference"]
    replication_phrase = (
        "reproduces it exactly, seed included"
        if difference == 0
        else "reproduces it to within {:.4f}".format(difference)
    )

    blocks: List[Block] = [
        heading(1, "Model Validation Report"),
        meta(
            "{} ({}), version {}  |  Independent validation  |  "
            "Review date {}  |  Reviewer: {}".format(
                review["model_name"],
                review["model_id"],
                review["version"],
                review["review_date"],
                review["reviewer"],
            )
        ),
        heading(2, "1. Conclusion"),
        callout(
            "Outcome: {}. {} High, {} Medium and {} Low findings are raised. Nothing in "
            "this report turns on a judgement call: each finding is a measurement the "
            "developer could have taken before submitting.".format(
                register.outcome(), counts["High"], counts["Medium"], counts["Low"]
            )
        ),
        paragraph(
            "The submission reports an AUC of {:.3f}. That figure is real: rebuilding the "
            "model from the recipe in the submission {}. It is not, however, a measure of "
            "what the model does in production. One of its fourteen inputs is written to "
            "the customer record after the customer has already given notice of leaving, "
            "so it will be empty at the moment a score is needed. Removing it and testing "
            "on the six months following the build window gives an AUC of {:.3f} -- {:.0f} "
            "Gini points below the figure the business case rests on.".format(
                key["reported_auc"],
                replication_phrase,
                key["corrected_oot_auc"],
                100 * (2 * key["reported_auc"] - 1)
                - 100 * (2 * key["corrected_oot_auc"] - 1),
            )
        ),
        paragraph(
            "At that level the model is beaten by its own benchmark. A logistic regression "
            "on {} variables, fitted on the same data, scores {:.3f} on the same "
            "out-of-time sample against the champion's {:.3f}. The submission justifies "
            "the gradient-boosted model class on predictive power, and out-of-time it does "
            "not have any to justify it with. Those two results are the High findings. The "
            "model is not approved for use in its current form; section 14 sets out what "
            "would change that.".format(
                key["challenger_variables"],
                key["challenger_oot_auc"],
                key["corrected_oot_auc"],
            )
        ),
        heading(2, "2. Scope and approach"),
        paragraph(
            "This review covers the model as submitted for its scheduled periodic "
            "validation: the development sample ({}), the out-of-time sample ({}), the "
            "development document, and the fitted model rebuilt from that document. It "
            "does not cover the production scoring code, the campaign management system "
            "the scores are consumed by, or the vendor data feeds behind two of the "
            "inputs.".format(
                cfg["portfolio"]["development"]["window"],
                cfg["portfolio"]["out_of_time"]["window"],
            )
        ),
        paragraph(
            "Testing was carried out independently rather than by re-reading the "
            "developer's own analysis. Every number in this report was produced by code in "
            "this repository from the raw extract, and every test is one the developer "
            "could have run before submitting."
        ),
        bullets(
            [
                "Data quality assessment of the extract, against the bank's six data quality dimensions",
                "Completeness check of the development document against the required sections",
                "Replication of the reported result from the documented recipe",
                "Variable review, including whether each input is knowable at scoring time",
                "Performance testing on an out-of-time sample the model did not see",
                "Calibration of predicted probabilities against observed outcomes",
                "Population stability of every input, and of the score itself",
                "Comparison against a transparent benchmark model",
                "Selection rate testing across age band and province",
            ]
        ),
        heading(2, "3. Data quality assessment"),
        paragraph(
            "The extract was tested before any modelling work. Rules are applied to the "
            "raw file as received, so the results describe what the developer was working "
            "with rather than what it becomes after treatment."
        ),
        table(results["dq_table"]),
        paragraph(
            "Failures were remediated in this review before the modelling tests, so that "
            "the champion, the benchmark and the corrected refits are all compared on the "
            "same treated data. The treatment applied was: de-duplicate on customer "
            "identifier, translate both credit score sentinels to a null plus an explicit "
            "missing indicator, separate an unsupplied balance from a nil one, and "
            "standardise province codes. None of this treatment is described in the "
            "submission."
        ),
        heading(2, "4. Documentation completeness"),
        table(results["doc_table"]),
        heading(2, "5. Replication"),
        paragraph(
            "The model was rebuilt from sections 3 and 4 of the development document, "
            "using the developer's split, class balancing and hyperparameters, and scored "
            "on their own held-out sample."
        ),
        table(pd.DataFrame([results["replication"]])),
        paragraph(
            "The result replicates. This is a genuine strength of the submission and it "
            "is recorded as such: the recipe in the document is complete enough for a "
            "third party to reproduce the model, which is not always the case. Everything "
            "that follows is about what the reported number measures, not about whether "
            "it is real."
        ),
        heading(2, "6. Variable review"),
        paragraph(
            "Each input was scored on its own against the outcome. The test is deliberately "
            "blunt: a single variable that separates churners from non-churners almost "
            "perfectly is either the most valuable field the bank owns, or it is not "
            "available at the time the model runs."
        ),
        table(results["variable_table"]),
        paragraph(
            "The strongest input reaches {:.3f} on its own, against {:.3f} for the "
            "strongest of the remaining thirteen. The data dictionary records it as "
            "written when a retention call is logged, and retention calls are placed after "
            "a customer has given notice. It is a record of the outcome, not a predictor "
            "of it, and it cannot be populated at the moment a score is needed. This is "
            "finding VF-01.".format(key["top_variable_auc"], key["next_variable_auc"]),
        ),
        heading(2, "7. Performance testing"),
        paragraph(
            "Four states of the model are compared on the same footing: as submitted on "
            "the developer's random holdout, as submitted on the following six months, "
            "refit without the post-outcome variable, and a transparent benchmark."
        ),
        table(results["performance_table"]),
        figure(
            "../assets/roc_comparison.png",
            "The gap between the submitted result and out-of-time performance, before "
            "and after the post-outcome variable is removed.",
        ),
        paragraph(
            "The two causes are worth separating, because they do not carry equal weight. "
            "Moving the model as submitted from the random holdout to the out-of-time "
            "sample costs almost nothing ({:.3f} to {:.3f}), since the post-outcome "
            "variable is present in both. Removing that variable on the same out-of-time "
            "sample costs {:.3f}. Practically all of the gap is one variable, and the "
            "later period accounts for {:.3f} of it once the variable is out of the way.".format(
                key["reported_auc"],
                key["oot_auc"],
                key["oot_auc"] - key["corrected_oot_auc"],
                key["corrected_holdout_auc"] - key["corrected_oot_auc"],
            )
        ),
        heading(2, "8. Calibration"),
        paragraph(
            "The model's output is used as a probability, so it was tested as one: "
            "customers were banded by predicted probability and the predicted rate "
            "compared against what actually happened in each band."
        ),
        paragraph(
            "This test is run on the model with the post-outcome variable removed. On the "
            "model as submitted that variable pushes almost every score to one extreme or "
            "the other, which conceals this problem rather than fixing it, and the "
            "corrected model is in any case the only version that could be deployed."
        ),
        table(results["calibration_table"]),
        figure(
            "../assets/calibration.png",
            "Predicted against observed churn rate by band. Every band sits below the "
            "diagonal, so every band over-predicts.",
        ),
        paragraph(
            "Mean predicted probability is {:.1%} against an observed rate of {:.1%}. The "
            "cause is in the developer's own recipe: the minority class was oversampled "
            "to a level base rate and the output was never mapped back. The ranking is "
            "unaffected, which is why the AUC does not show it, but any calculation that "
            "multiplies these probabilities by a dollar amount is overstated by roughly "
            "the same factor. This is finding VF-03.".format(
                key["mean_predicted"], key["observed_rate"]
            )
        ),
        heading(2, "9. Population stability"),
        paragraph(
            "Stability was measured between the development window and the following six "
            "months, with bin edges taken from the development sample only."
        ),
        table(results["psi_table"]),
        figure(
            "../assets/psi_by_variable.png",
            "Population stability index by input, against the {:.2f} amber and {:.2f} red "
            "thresholds.".format(
                cfg["validation"]["psi_amber"], cfg["validation"]["psi_red"]
            ),
        ),
        paragraph(
            "The movement is a channel migration rather than a data fault: customers moved "
            "to mobile and their engagement counts moved with them. The score distribution "
            "is the part worth dwelling on. At PSI {:.2f} it sits ({}) well inside the "
            "acceptable band, so a monitoring process that watched the score alone would "
            "have reported this model as stable throughout, while two of its inputs moved "
            "by more than four times the red threshold. Score-level monitoring is the "
            "common choice because it is the cheapest, and this is the case it misses. "
            "This is finding VF-05.".format(key["score_psi"], key["score_psi_band"].lower())
        ),
        heading(2, "10. Benchmark comparison"),
        paragraph(
            "A logistic regression on {} variables, fitted on the same treated "
            "development sample and scored on the same out-of-time sample, is used as the "
            "benchmark. It is not proposed as a replacement; it is there to establish how "
            "much of the champion's performance is attributable to its model "
            "class.".format(key["challenger_variables"])
        ),
        paragraph(
            "Corrected champion {:.3f} against benchmark {:.3f}. The benchmark is ahead by "
            "{:.3f} AUC, so the comparison does not end in a question about whether the "
            "extra complexity is worth carrying -- it ends with the simpler model being "
            "more accurate on the period that matters. The champion's hyperparameters were "
            "selected against the random split, which is the ordinary way this outcome "
            "arises: the additional flexibility fitted the development window rather than "
            "the underlying relationship. This is finding VF-02.".format(
                key["corrected_oot_auc"],
                key["challenger_oot_auc"],
                key["challenger_oot_auc"] - key["corrected_oot_auc"],
            )
        ),
        heading(2, "11. Selection rate testing"),
        paragraph(
            "The score decides who a retention campaign contacts, so the outcome that "
            "matters to a customer is whether they fall inside the top {:.0%} the campaign "
            "can afford to reach. Selection rate was tested across age band and "
            "province.".format(cfg["validation"]["selection_rate_cut"])
        ),
        table(results["age_table"]),
        figure(
            "../assets/selection_by_age.png",
            "Share of each age band contacted, against four-fifths of the highest band.",
        ),
        table(results["province_table"]),
        paragraph(
            "Two separate questions are involved and the distinction matters. The first is "
            "whether the selection rate is uneven, and it is. Part of that is legitimate: "
            "the observed churn column shows the bands genuinely differ. The second is "
            "whether the model is reading a protected characteristic through another "
            "field, and it is. Product bundle code alone identifies customers aged 60 and "
            "over with an AUC of {:.3f}, so removing age from the feature list would not "
            "have changed the outcome, and the submission's statement that no protected "
            "characteristic is used is not accurate. Refitting without the bundle code "
            "costs {:.3f} AUC. This is finding VF-06.".format(
                key["bundle_proxy"],
                key["corrected_oot_auc"] - key["no_bundle_oot_auc"],
            )
        ),
        heading(2, "12. Findings register"),
        table(results["register_table"]),
        heading(2, "13. Limitations of this review"),
        paragraph(
            "These bound what the conclusion above can be taken to mean."
        ),
        bullets(
            [
                "The portfolio is generated rather than drawn from the bank's systems, so "
                "the magnitudes are illustrative. The tests, thresholds and the reasoning "
                "are the transferable part; the numbers are not.",
                "Production scoring code was not reviewed, so this report cannot say "
                "whether the fitted model and the deployed model agree.",
                "Only age band and province were available for selection rate testing. "
                "The bank does not collect other protected characteristics, so no "
                "statement is made about grounds that were not testable.",
                "Outcomes were observed over a six-month horizon. A customer who leaves in "
                "month seven is counted as retained throughout.",
                "The benchmark is one logistic regression on one variable list, and the "
                "generated portfolio draws churn from a linear log-odds model. The "
                "benchmark is therefore close to the true functional form by construction "
                "and is advantaged in this comparison. VF-02 stands as a statement about "
                "this model on this period, and about the absence of any benchmark in the "
                "submission. It is not evidence about model classes in general.",
                "Every result comes from a single seed. Sampling variation across seeds "
                "was not quantified and would change the third decimal place.",
            ]
        ),
        heading(2, "14. Conditions for approval"),
        paragraph(
            "The model is not approved for use in its current form. The following would "
            "change that, in the order they should be addressed."
        ),
        bullets(
            [
                "VF-01 must be closed before any use. Remove the post-outcome variable, "
                "refit, and restate every performance figure in the submission. Until that "
                "is done the submission describes a model the bank cannot run.",
                "VF-02 must be closed before any use. Either adopt the benchmark, or "
                "resubmit the gradient-boosted model with its hyperparameters selected "
                "against an out-of-time sample and evidence that it beats the benchmark "
                "there. A model that is less accurate than its own benchmark has no "
                "remaining justification for the burden it carries.",
                "VF-03, VF-05, VF-06 and VF-07 may be carried with named compensating "
                "controls: restrict the model to ranking use until a calibration mapping "
                "is fitted, start monthly stability monitoring on every input rather than "
                "on the score alone, either drop the bundle code or record a justification "
                "and monitor selection rate by age band, and make an out-of-time sample "
                "the primary performance evidence.",
                "VF-04 and VF-08 should be closed before the next scheduled review and do "
                "not block use on their own.",
                "A resubmission should include the out-of-time result as the headline "
                "figure, the benchmark comparison, and the monitoring plan, so that the "
                "next review starts from evidence rather than from reconstruction.",
            ]
        ),
    ]
    return blocks


def write(results: dict, out_dir: Path, title: str) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks = build(results)

    md_path = out_dir / "model_validation_report.md"
    html_path = out_dir / "model_validation_report.html"
    md_path.write_text(render_markdown(blocks), encoding="utf-8")
    html_path.write_text(render_html(blocks, title), encoding="utf-8")
    return md_path, html_path
