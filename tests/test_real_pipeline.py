"""Opt-in end-to-end run of the single-FOV pipeline on ONE user-authorised real ND2 (read-only, local, nothing copied/uploaded).

Skips with a fixed reason unless ``CATKIT_REAL_ND2`` (file) and ``CATKIT_REAL_ND2_ROLES`` (``SEGMENTATION;DET1,DET2``, explicit
metadata channel names) are set. Outputs go to pytest's tmp dir. Only invariants are asserted (no data-specific numbers).
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

from scientific import require_modules

require_modules("numpy", "scipy", "skimage", "bigfish", "nd2")

import catkit.pipeline as PL  # noqa: E402

SKIP = "CATKIT_REAL_ND2 / CATKIT_REAL_ND2_ROLES not set: real-ND2 pipeline test is opt-in"


@pytest.mark.real_nd2
def test_real_nd2_end_to_end_invariants(tmp_path, record_property):
    path, roles_s = os.environ.get("CATKIT_REAL_ND2"), os.environ.get("CATKIT_REAL_ND2_ROLES")
    if not path or not roles_s:
        pytest.skip(SKIP)
    p = Path(path)
    seg, det = roles_s.split(";")
    roles = {"segmentation": seg, "detection": [d for d in det.split(",") if d]}
    st = (p.stat().st_size, p.stat().st_mtime_ns)
    res = PL.run_nd2(p, tmp_path / "out", roles, k=10.0)
    assert st == (p.stat().st_size, p.stat().st_mtime_ns), "the real ND2 file changed"
    meta = json.loads((tmp_path / "out" / "run_metadata.json").read_text())
    assert meta["reconciliation"]["ok"] and str(p.parent) not in json.dumps(meta) and meta["input"]["file_name"] == p.name
    rows = list(csv.DictReader(open(tmp_path / "out" / "spots.csv")))
    assert list(rows[0]) == list(PL.SPOT_COLUMNS) if rows else True
    n_acc = sum(c["n_accepted"] for c in meta["localization_qc"].values())
    assert len(rows) == n_acc == len(res["spots"]["spot_id"])
    for ch, c in meta["localization_qc"].items():
        assert c["n_candidates"] >= c["n_accepted"] >= c["n_qc_passed"]
        record_property(f"pipeline_{ch}", json.dumps({k: c[k] for k in ("n_candidates", "n_accepted", "n_qc_passed")}))
