"""The findings register.

Every test in this review either raises a finding or it does not, and the
register is the only place findings are recorded. Severity is assigned against
a fixed rule so that two people running the same tests would rate them the
same way:

    High    the model should not be used on the portfolio in its current form
    Medium  the model can be used with a named compensating control while the
            issue is fixed
    Low     the issue does not affect the decision to use the model, but it
            should be closed before the next scheduled review
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

SEVERITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}


@dataclass
class Finding:
    ref: str
    severity: str
    area: str
    title: str
    observation: str
    implication: str
    recommendation: str

    def as_dict(self) -> dict:
        return {
            "ref": self.ref,
            "severity": self.severity,
            "area": self.area,
            "finding": self.title,
            "observation": self.observation,
            "implication": self.implication,
            "recommendation": self.recommendation,
        }


class Register:
    """Collects findings from the individual tests and keeps them ordered."""

    def __init__(self) -> None:
        self._findings: List[Finding] = []

    def add(self, finding: Finding) -> None:
        self._findings.append(finding)

    def extend(self, findings: List[Finding]) -> None:
        self._findings.extend(findings)

    def __len__(self) -> int:
        return len(self._findings)

    @property
    def findings(self) -> List[Finding]:
        return sorted(self._findings, key=lambda f: (SEVERITY_ORDER[f.severity], f.ref))

    def counts(self) -> dict:
        counts = {"High": 0, "Medium": 0, "Low": 0}
        for finding in self._findings:
            counts[finding.severity] += 1
        return counts

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([f.as_dict() for f in self.findings])

    def outcome(self) -> str:
        """The conclusion the register forces, given the approval rule."""
        counts = self.counts()
        if counts["High"] > 0:
            return "Not approved for use"
        if counts["Medium"] > 0:
            return "Approved with conditions"
        return "Approved"
