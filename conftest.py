"""Put the repository root on the import path for the test suite.

Without this, `pytest` finds the tests but cannot import `modelval`, because
pytest adds the test file's own directory to sys.path rather than the project
root. It works under `python -m pytest` purely because that form adds the
working directory, which is an easy way to ship a suite that only runs one way.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
