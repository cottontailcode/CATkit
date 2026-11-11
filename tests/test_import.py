"""Packaging tests: import, version consistency, and Python version."""

import re
import sys
from importlib import metadata

import catkit


def test_version_is_semver_string():
    assert re.fullmatch(r"\d+\.\d+\.\d+", catkit.__version__)


def test_installed_metadata_matches_source_version():
    # pyproject.toml reads the version from catkit.__version__; the installed
    # distribution metadata must therefore report exactly the same string.
    assert metadata.version("catkit-microscopy") == catkit.__version__


def test_running_on_python_311():
    # Milestone 1 is verified on Python 3.11 only (requires-python = ">=3.11,<3.12").
    assert sys.version_info[:2] == (3, 11)
