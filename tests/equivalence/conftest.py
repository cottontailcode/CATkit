"""Fixtures for frozen-reference tests.

Only tests marked ``research`` touch private references; they SKIP (with a
reason) when CATKIT_RESEARCH_SRC is unset. Once it is set, any setup problem
(missing manifest, missing/dataless/changed file, wrong Big-FISH version) is a
FAIL, never a skip. Harness self-tests use fake references and always run.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from catkit.provenance import ManifestError, load_manifest, verify_manifest
from refharness import ENV_MANIFEST, ENV_ROOT, isolated_references

REQUIRED_BIGFISH = "0.6.2"


class ResearchEnv:
    def __init__(self, root: Path, manifest: dict):
        self.root = root
        self.manifest = manifest


@pytest.fixture(scope="session")
def research_env():
    root = os.environ.get(ENV_ROOT)
    if not root:
        pytest.skip(f"{ENV_ROOT} not set: private frozen-reference tests are opt-in")
    manifest_path = os.environ.get(ENV_MANIFEST)
    if not manifest_path:
        pytest.fail(f"{ENV_ROOT} is set but {ENV_MANIFEST} is not")
    if not Path(root).is_dir():
        pytest.fail(f"{ENV_ROOT} does not point to a directory")
    try:
        manifest = load_manifest(manifest_path)
    except (OSError, ManifestError) as exc:
        pytest.fail(f"cannot load reference manifest: {exc}")
    env = ResearchEnv(Path(root), manifest)
    yield env
    # session end: every active reference must still be byte-identical
    active = [r for r, e in manifest.items() if e.import_policy in ("import-isolated", "script-subprocess")]
    bad = [r for r in verify_manifest(manifest, root, active) if not r.ok]
    assert not bad, f"reference files changed during the session: {bad}"


@pytest.fixture(scope="session")
def bigfish_required(research_env):
    try:
        import bigfish
    except ImportError:
        pytest.fail("big-fish is required for frozen-reference tests but is not installed")
    if bigfish.__version__ != REQUIRED_BIGFISH:
        pytest.fail(f"big-fish {bigfish.__version__} installed; frozen references were validated with {REQUIRED_BIGFISH}")
    return bigfish.__version__


@pytest.fixture
def frozen_modules(research_env, bigfish_required, tmp_path):
    """``{"REF-LOC": module, "REF-PHOT": module}``, valid ONLY inside the test body.

    Function-scoped and yielded *inside* the harness context: every call into
    frozen code must happen in the test body, so the import guard is active
    while it runs and the snapshot/hash post-check at teardown happens AFTER
    use. Never keep these module/function objects past the fixture's teardown.
    (Do not turn this into a longer-lived fixture: it would overlap other
    reference contexts and trip the harness's cached-module fail-fast.)
    """
    with isolated_references(research_env.manifest, research_env.root, ["REF-LOC", "REF-PHOT"], tmp_path) as modules:
        yield modules


@pytest.fixture
def frozen_seg_modules(research_env, bigfish_required, tmp_path):
    """``{"REF-SEG", "REF-DET", "REF-DRV"}`` modules for the M2.2-A segmentation/cells E tests.

    Same lifecycle rules as ``frozen_modules`` (valid only inside the test body; never combine the two fixtures
    in one test: reference contexts cannot be nested). REF-DRV is imported only to reach its module-level
    ``seg_params``; its ``process_fov`` is never called.
    """
    with isolated_references(research_env.manifest, research_env.root, ["REF-SEG", "REF-DET", "REF-DRV"], tmp_path) as modules:
        yield modules


@pytest.fixture
def frozen_det_modules(research_env, bigfish_required, tmp_path):
    """``{"REF-DET": module}`` for the M2.2-B detection E tests (same lifecycle rules as ``frozen_modules``)."""
    with isolated_references(research_env.manifest, research_env.root, ["REF-DET"], tmp_path) as modules:
        yield modules


@pytest.fixture
def frozen_io_modules(research_env, bigfish_required, tmp_path):
    """``{"REF-IO", "REF-DET"}`` for the M2.2-D reader E tests (same lifecycle rules as ``frozen_modules``)."""
    with isolated_references(research_env.manifest, research_env.root, ["REF-IO", "REF-DET"], tmp_path) as modules:
        yield modules


@pytest.fixture
def e_stats(record_property):
    """Collects bit-identity statistics and writes them into the JUnit record."""
    from compare import Stats

    stats = Stats()
    yield stats
    record_property("e_float_fields_compared", stats.compared)
    record_property("e_float_fields_bit_identical", stats.bit_identical)
    record_property(
        "e_float_fields_within_tolerance_not_bit_identical",
        "; ".join(f"{k}:{d:.3e}" for k, d in stats.not_bit_identical[:20]) or "none",
    )
