"""Render the HTML report to a PDF, for the version that gets attached to an email.

Run `python run.py` first -- this reads the HTML that step writes.

    python scripts/make_report_pdf.py

Uses headless Chrome, which is already on most machines and avoids adding a PDF
library to requirements.txt for one file. If Chrome is somewhere unusual, pass
--chrome with the path.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "results" / "model_validation_report.html"
PDF = ROOT / "assets" / "Model_Validation_Report.pdf"

CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium-browser",
]


def find_chrome(override: str = "") -> str:
    if override:
        return override
    for name in ("chrome", "google-chrome", "chromium"):
        found = shutil.which(name)
        if found:
            return found
    for path in CANDIDATES:
        if Path(path).exists():
            return path
    raise SystemExit(
        "Could not find Chrome. Pass the executable path with --chrome."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the validation report to PDF.")
    parser.add_argument("--chrome", default="")
    parser.add_argument("--out", default=str(PDF))
    args = parser.parse_args()

    if not HTML.exists():
        raise SystemExit("{} not found. Run python run.py first.".format(HTML))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as profile:
        command = [
            find_chrome(args.chrome),
            "--headless",
            "--disable-gpu",
            "--no-first-run",
            "--hide-scrollbars",
            "--user-data-dir={}".format(profile),
            "--no-pdf-header-footer",
            "--print-to-pdf={}".format(out),
            HTML.resolve().as_uri(),
        ]
        result = subprocess.run(command, capture_output=True, text=True)

    if not out.exists():
        sys.stderr.write(result.stderr or "Chrome exited without writing a file.\n")
        raise SystemExit(1)

    print("Wrote {} ({:.0f} KB)".format(out, out.stat().st_size / 1024))


if __name__ == "__main__":
    main()
