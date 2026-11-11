"""Skip-or-fail handling for tests that need the scientific stack.

Same policy as ``tests/deps``: a missing package SKIPs with a reason, unless
``CATKIT_REQUIRE_SCIENTIFIC_DEPS=1`` (explicit dependency validation), in which
case it FAILs.
"""

from __future__ import annotations

import importlib
import os

import pytest

ENV_REQUIRE = "CATKIT_REQUIRE_SCIENTIFIC_DEPS"


def require_modules(*names: str) -> None:
    """Call at module level of a test file, before importing ``catkit.localization`` etc."""
    for name in names:
        try:
            importlib.import_module(name)
        except ImportError as exc:
            if os.environ.get(ENV_REQUIRE) == "1":
                pytest.fail(f"{name} is required ({ENV_REQUIRE}=1) but cannot be imported: {exc}", pytrace=False)
            pytest.skip(f"{name} not installed (scientific deps not part of this environment)", allow_module_level=True)
