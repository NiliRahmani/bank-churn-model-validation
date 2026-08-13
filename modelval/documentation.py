"""Check the submission against the sections a development document must contain.

This is the cheapest test in the review and it runs first in practice. If the
monitoring plan is missing, that is knowable before any data is loaded, and it
tends to predict what the rest of the review will find.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .findings import Finding

REQUIRED_SECTIONS = [
    "Purpose and intended use",
    "Data sources and lineage",
    "Variable selection",
    "Model methodology",
    "Performance testing",
    "Assumptions and limitations",
    "Ongoing monitoring plan",
    "Benchmark comparison",
    "Implementation and controls",
]


def _headings(document: str) -> List[str]:
    return [
        re.sub(r"^#+\s*(?:\d+\.\s*)?", "", line).strip()
        for line in document.splitlines()
        if line.startswith("#")
    ]


def review(document_path: Path) -> Tuple[pd.DataFrame, List[Finding]]:
    """Return a section-by-section checklist and any finding it raises."""
    document = document_path.read_text(encoding="utf-8")
    present = [h.lower() for h in _headings(document)]

    rows = []
    for section in REQUIRED_SECTIONS:
        found = any(section.lower() in heading for heading in present)
        rows.append(
            {
                "required_section": section,
                "present": "Yes" if found else "No",
            }
        )
    table = pd.DataFrame(rows)

    missing = table[table["present"] == "No"]["required_section"].tolist()
    findings: List[Finding] = []
    if missing:
        findings.append(
            Finding(
                ref="VF-08",
                severity="Low",
                area="Documentation",
                title="Development document is missing sections the standard requires",
                observation=(
                    "{} of {} required sections are absent: {}.".format(
                        len(missing), len(REQUIRED_SECTIONS), "; ".join(missing)
                    )
                ),
                implication=(
                    "Without a stated limitations section there is no record of where the "
                    "developer already knows the model should not be used, and without a "
                    "monitoring plan there is no agreed trigger for the next review. Both "
                    "gaps put the whole burden of catching deterioration on the annual "
                    "validation cycle, which is too slow for a model whose inputs move "
                    "within six months."
                ),
                recommendation=(
                    "Complete the missing sections before resubmission, and add the "
                    "section checklist to the intake gate so an incomplete submission is "
                    "returned before review effort is spent on it."
                ),
            )
        )
    return table, findings
