"""Figures for the validation report.

Four charts, one per question the report has to settle. Each is built to be read
without its surrounding paragraph, because in practice the committee pack gets
skimmed and the charts are what people stop on.

Series identity is never carried by colour alone -- every series is either
directly labelled or named in a legend that also carries its number.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Validated for all pairs on the light surface, normal vision and CVD.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]

STATUS = {"Green": "#0ca30c", "Amber": "#fab219", "Red": "#d03b3b"}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)


def _style(ax, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    ax.set_title(title, color=INK, loc="left", pad=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1.0)


def roc_comparison(
    curves: Sequence[Tuple[str, np.ndarray, np.ndarray, float]], path: Path
) -> None:
    """One ROC per model state, with the AUC printed in the legend entry."""
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    ax.plot([0, 1], [0, 1], color=BASELINE, linewidth=1.0, linestyle=(0, (4, 4)))
    for i, (label, y_true, scores, auc_value) in enumerate(curves):
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax.plot(
            fpr,
            tpr,
            color=SERIES[i % len(SERIES)],
            linewidth=2.0,
            label="{}  AUC {:.3f}".format(label, auc_value),
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _style(
        ax,
        "False positive rate",
        "True positive rate",
        "Reported performance against performance on the period after the build",
    )
    legend = ax.legend(loc="lower right", frameon=False, labelcolor=INK_SECONDARY)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def calibration(table: pd.DataFrame, path: Path) -> None:
    """Predicted probability against the rate that actually happened."""
    fig, ax = plt.subplots(figsize=(6.4, 5.0))

    ax.plot(
        [0, max(table["predicted"].max(), table["observed"].max()) * 1.05],
        [0, max(table["predicted"].max(), table["observed"].max()) * 1.05],
        color=BASELINE,
        linewidth=1.0,
        linestyle=(0, (4, 4)),
        label="Perfect calibration",
    )
    ax.plot(
        table["predicted"],
        table["observed"],
        color=SERIES[0],
        linewidth=2.0,
        marker="o",
        markersize=8,
        markeredgecolor=SURFACE,
        markeredgewidth=2.0,
        label="Champion, post-outcome variable removed",
    )

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    _style(
        ax,
        "Mean predicted probability",
        "Observed churn rate",
        "Every band predicts more churn than occurred",
    )
    legend = ax.legend(loc="upper left", frameon=False)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def psi_by_variable(table: pd.DataFrame, amber: float, red: float, path: Path) -> None:
    """Stability of each input, coloured by the band it lands in."""
    data = table.sort_values("psi")
    fig, ax = plt.subplots(figsize=(7.6, 5.6))

    colors = [STATUS[band] for band in data["band"]]
    ax.barh(
        data["variable"],
        data["psi"],
        color=colors,
        edgecolor=SURFACE,
        linewidth=1.5,
        height=0.72,
    )

    for threshold, label in ((amber, "amber"), (red, "red")):
        ax.axvline(threshold, color=BASELINE, linewidth=1.0, linestyle=(0, (4, 4)))
        ax.text(
            threshold,
            len(data) - 0.3,
            " {} {:.2f}".format(label, threshold),
            color=INK_MUTED,
            fontsize=8,
            va="top",
        )

    # Only the bars that breached anything get a number. Labelling the twelve
    # variables that did not move adds a column of zeros and nothing else.
    for y, (value, band) in enumerate(zip(data["psi"], data["band"])):
        if band == "Green":
            continue
        ax.text(
            value + 0.012, y, "{:.2f}".format(value), va="center", fontsize=8, color=INK_SECONDARY
        )

    ax.set_xlim(0, max(data["psi"].max() * 1.22, red * 1.3))
    _style(
        ax,
        "Population stability index",
        "",
        "Which inputs moved between the build window and the next six months",
    )
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def selection_by_age(table: pd.DataFrame, threshold: float, path: Path) -> None:
    """Share of each age band contacted by the campaign, against the four-fifths line."""
    data = table.sort_values("age_band")
    best = data["selection_rate"].max()

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    colors = [
        STATUS["Red"] if flag == "No" else SERIES[0] for flag in data["meets_four_fifths"]
    ]
    ax.bar(
        data["age_band"],
        data["selection_rate"],
        color=colors,
        edgecolor=SURFACE,
        linewidth=1.5,
        width=0.62,
    )

    ax.axhline(best * threshold, color=BASELINE, linewidth=1.0, linestyle=(0, (4, 4)))
    ax.text(
        -0.42,
        best * threshold,
        "four-fifths of the highest band",
        color=INK_MUTED,
        fontsize=8,
        va="bottom",
        ha="left",
    )

    for x, (rate, ratio) in enumerate(zip(data["selection_rate"], data["impact_ratio"])):
        ax.text(
            x,
            rate + best * 0.03,
            "{:.1%}\nratio {:.2f}".format(rate, ratio),
            ha="center",
            fontsize=8.5,
            color=INK_SECONDARY,
        )

    ax.set_ylim(0, best * 1.32)
    ax.yaxis.set_major_formatter(lambda v, _: "{:.0%}".format(v))
    _style(
        ax,
        "",
        "Share contacted by the retention campaign",
        "Who the campaign reaches, by age band",
    )
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_all(
    curves: Sequence[Tuple[str, np.ndarray, np.ndarray, float]],
    calibration_table: pd.DataFrame,
    psi_table: pd.DataFrame,
    age_table: pd.DataFrame,
    cfg: dict,
    out_dir: Path,
) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    roc_comparison(curves, out_dir / "roc_comparison.png")
    calibration(calibration_table, out_dir / "calibration.png")
    psi_by_variable(
        psi_table,
        cfg["validation"]["psi_amber"],
        cfg["validation"]["psi_red"],
        out_dir / "psi_by_variable.png",
    )
    selection_by_age(
        age_table, cfg["validation"]["four_fifths_threshold"], out_dir / "selection_by_age.png"
    )
    return [
        "roc_comparison.png",
        "calibration.png",
        "psi_by_variable.png",
        "selection_by_age.png",
    ]
