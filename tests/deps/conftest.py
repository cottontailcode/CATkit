"""Scientific-dependency checks (marker ``deps``).

Default (main dev environment without scientific packages): a missing module
SKIPs the test with a reason. When CATKIT_REQUIRE_SCIENTIFIC_DEPS=1 is set --
i.e. the user explicitly runs dependency validation -- a missing module FAILs.
"""

from __future__ import annotations

import importlib
import os

import pytest

ENV_REQUIRE = "CATKIT_REQUIRE_SCIENTIFIC_DEPS"


def requiring() -> bool:
    return os.environ.get(ENV_REQUIRE) == "1"


@pytest.fixture
def require():
    def _require(module: str):
        try:
            return importlib.import_module(module)
        except ImportError as exc:
            if requiring():
                pytest.fail(f"{module} is required ({ENV_REQUIRE}=1) but cannot be imported: {exc}")
            pytest.skip(f"{module} not installed (scientific deps not part of this environment)")

    return _require


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "tests/deps/" in str(item.fspath).replace(os.sep, "/"):
            item.add_marker(pytest.mark.deps)
