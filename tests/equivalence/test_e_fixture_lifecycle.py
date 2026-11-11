"""The frozen-module fixture must not leak reference modules or harness state past its teardown."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from refharness import isolated_references

pytestmark = pytest.mark.research


def _module_names(research_env):
    return [research_env.manifest[r].path(research_env.root).stem for r in ("REF-LOC", "REF-PHOT")]


def test_fixture_use_inside_body(frozen_modules, research_env):
    names = _module_names(research_env)
    assert all(n in sys.modules for n in names)  # the harness context is active during the test body
    assert frozen_modules["REF-LOC"]._axis(np.arange(3.0), 1.0, 1.0).shape == (3,)  # a real call inside the body


def test_setup_use_teardown_restores_everything(research_env, tmp_path):
    """Self-contained (order independent): same context as the fixture, state checked before and after."""
    names = _module_names(research_env)
    path0, meta0 = list(sys.path), list(sys.meta_path)
    dwb0, prefix0 = sys.dont_write_bytecode, sys.pycache_prefix
    assert not any(n in sys.modules for n in names)
    with isolated_references(research_env.manifest, research_env.root, ["REF-LOC", "REF-PHOT"], tmp_path) as mods:
        assert all(n in sys.modules for n in names)
        assert sys.dont_write_bytecode is True and sys.meta_path != meta0
        mods["REF-LOC"].gaussian_integral_voxels(1.0, 1.0, 1.0, 1.0)
    assert not any(n in sys.modules for n in names)
    assert sys.path == path0 and sys.meta_path == meta0
    assert sys.dont_write_bytecode == dwb0 and sys.pycache_prefix == prefix0
    root = str(research_env.root)
    assert not any(getattr(m, "__file__", None) and root in str(m.__file__) for m in list(sys.modules.values()))
