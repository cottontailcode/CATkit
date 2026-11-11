"""Opt-in smoke tests against the PRIVATE frozen references (marker: research).

No scientific comparison happens here. These tests only prove that the
references can be verified and used without modifying them:

* the five frozen port-source modules import under isolation;
* the per-FOV driver imports with its full transitive closure verified and
  nothing outside that closure is loaded from the reference root;
* the two frozen synthetic test scripts run in a guarded subprocess and report
  18/18 and 3/3 passed.
"""

from __future__ import annotations

import sys

import pytest

from refharness import isolated_references, run_reference_script

pytestmark = pytest.mark.research

PORT_SOURCES = ["REF-IO", "REF-SEG", "REF-DET", "REF-LOC", "REF-PHOT"]
EXPECTED_ATTRS = {
    "REF-IO": ["read_header", "load_channel_stacks"],
    "REF-SEG": ["detect_focus_plane", "focus_slice_range", "segment_from_dapi"],
    "REF-DET": ["psf_from_optics", "kernel_params", "log_and_local_max", "noise_sigma_from_positive_log", "assign_cells"],
    "REF-LOC": ["fit_spot", "classify_quality", "gaussian_model", "DEFAULT_CFG"],
    "REF-PHOT": ["measure_spots", "clipped_background", "overlap_flags", "nearest_neighbor_nm"],
}


def test_port_sources_import_isolated(research_env, bigfish_required, tmp_path):
    with isolated_references(research_env.manifest, research_env.root, PORT_SOURCES, tmp_path) as mods:
        for rid, attrs in EXPECTED_ATTRS.items():
            missing = [a for a in attrs if not hasattr(mods[rid], a)]
            assert not missing, f"{rid} lacks {missing}"


def test_driver_imports_with_verified_closure(research_env, bigfish_required, tmp_path):
    path0 = list(sys.path)
    with isolated_references(research_env.manifest, research_env.root, ["REF-DRV"], tmp_path) as mods:
        assert callable(getattr(mods["REF-DRV"], "process_fov", None))
    assert sys.path == path0, "the driver's import-time sys.path edits must be undone"


@pytest.mark.parametrize("ref_id, expected", [("REF-T1", "18/18 passed"), ("REF-T2", "3/3 passed")])
def test_frozen_synthetic_scripts_pass(research_env, bigfish_required, tmp_path, ref_id, expected):
    proc = run_reference_script(research_env.manifest, research_env.root, ref_id, tmp_path)
    tail = "\n".join(proc.stdout.splitlines()[-5:])
    assert proc.returncode == 0, f"{ref_id} exit {proc.returncode}\n{tail}\n{proc.stderr[-2000:]}"
    assert expected in proc.stdout, tail
