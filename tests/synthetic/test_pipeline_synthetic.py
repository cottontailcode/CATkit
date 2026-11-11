"""M2.3-A integration tests on synthetic data (ND2-independent). Unit behaviour of segmentation/detection/photometry/fit is
tested elsewhere; this file checks the wiring: schema, units, counts, assignment, empty/no-cell handling, determinism.
"""

from __future__ import annotations

import csv
import json
import warnings

import pytest

from scientific import require_modules

require_modules("numpy", "scipy", "skimage", "bigfish")

import numpy as np  # noqa: E402

import catkit.cells as C  # noqa: E402
import catkit.localization as L  # noqa: E402
import catkit.pipeline as PL  # noqa: E402
import synth  # noqa: E402
from synth_det import CY5, NUC, OPTICS, TRITC, VOX, make_channel, make_image  # noqa: E402

ROLES = {"segmentation": "DAPI", "detection": ["Cy5", "TRITC"]}
@pytest.fixture(scope="module")
def result():
    return PL.analyze_fov(make_image(), ROLES, fov_id="syn", condition="c1")


# ---------------------------------------------------------------- scaling helpers equal the tested M2.1 helpers
@pytest.mark.parametrize("vox", [(150.0, 65.0, 65.0), (150.0, 43.3333333, 43.3333333), (150.0, 43.333, 43.333), (200.0, 100.0, 80.0)])
def test_scaling_helpers_equal_tested_helpers(vox):
    for r in ((3., 5., 5.), (5., 9., 9.), (7., 13., 13.)):
        assert PL.rescale_radii(r, vox) == synth.rescale_radii(r, vox)
    assert PL.rescaled_cfg(L.DEFAULT_CFG, vox) == synth.rescaled_cfg(L.DEFAULT_CFG, vox)
    assert PL.REF_VOX == synth.REF_VOX
    # hand-computed at the baseline: ROI half-width round(7 * 43.3333333 / 65) = 5; aperture xy 5 * 43.3333333 / 65
    assert PL.rescaled_cfg(L.DEFAULT_CFG, (150.0, 65.0, 65.0))["roi_half_xy"] == 5
    assert PL.rescale_radii((3., 5., 5.), (150.0, 65.0, 65.0)) == (3.0, 5 * 43.3333333 / 65.0, 5 * 43.3333333 / 65.0)


# ---------------------------------------------------------------- schema, units, counts, assignment
def test_columns_units_and_dispositions(result):
    s, meta = result["spots"], result["metadata"]
    assert tuple(s) == PL.SPOT_COLUMNS and len({len(v) for v in s.values()}) == 1
    n = len(s["spot_id"])
    assert n == 5 and set(s["channel"]) == {"Cy5", "TRITC"} and set(s["threshold_mode"]) == {"fixed_sigma"}
    assert list(s["channel"]) == ["Cy5"] * 3 + ["TRITC"] * 2                          # channel order of the roles, then candidate order
    # units: init integer voxel index; um columns = index * pixel / z step
    assert np.allclose(s["x_init_um"], s["x_init"] * 0.065, rtol=1e-12, atol=0) and np.allclose(s["z_init_um"], s["z_init"] * 0.15, rtol=1e-12, atol=0)
    assert np.allclose(s["x_fit_um"], s["x_fit"] * 0.065, rtol=1e-12, atol=0, equal_nan=True)
    assert np.issubdtype(s["z_init"].dtype, np.integer) and np.issubdtype(s["z_fit"].dtype, np.floating)
    # (z, y, x) order: fitted positions recover the planted ones
    for ch, truth in (("Cy5", CY5), ("TRITC", TRITC)):
        rows = np.flatnonzero(s["channel"] == ch)
        for i in rows:
            t = min(truth, key=lambda q: (q[0] - s["z_fit"][i]) ** 2 + (q[1] - s["y_fit"][i]) ** 2 + (q[2] - s["x_fit"][i]) ** 2)
            assert abs(s["z_fit"][i] - t[0]) < 1.0 and abs(s["y_fit"][i] - t[1]) < 0.5 and abs(s["x_fit"][i] - t[2]) < 0.5
    # candidate -> accepted -> localized -> qc, per channel
    for ch in ROLES["detection"]:
        c = meta["localization_qc"][ch]
        assert c["n_candidates"] >= c["n_accepted"] == c["n_localized_rows"] >= c["n_qc_passed"]
        assert c["n_localized_rows"] == int((s["channel"] == ch).sum()) == sum(c["quality"].values()) == sum(c["assign_category"].values())
        assert meta["detection_qc"][ch]["n_accepted"] == c["n_accepted"]
    assert meta["reconciliation"]["ok"] and meta["segmentation"]["n_cells"] == 2 and not meta["segmentation"]["segmentation_empty"]
    # candidates table: accepted rows are exactly the spots (key = channel + spot_id, never bare spot_id)
    cand = result["candidates"]
    keys = {(a, int(b)) for a, b, acc in zip(cand["channel"], cand["spot_id"], cand["accepted"]) if acc}
    assert keys == {(a, int(b)) for a, b in zip(s["channel"], s["spot_id"])}
    assert set(cand["mode"]) == {"fixed_sigma"}
    # assignment by use coordinate: planted spots sit in cell 1, cell 2 and outside
    cy = s["channel"] == "Cy5"
    assert sorted(s["cell_base"][cy].tolist()) == [0, 1, 2] and np.array_equal(s["cell_init"], s["cell_base"])
    assert s["assign_category"][cy].tolist().count("outside") >= 1 and (s["tier_frozen"] == "A").all()
    assert (s["exposure_ms_metadata"][cy] == 400.0).all() and (s["exposure_ms_metadata"][~cy] == 300.0).all()


def test_outputs_are_deterministic_and_csv_roundtrips(result, tmp_path):
    again = PL.analyze_fov(make_image(), ROLES, fov_id="syn", condition="c1")
    a, b = tmp_path / "a", tmp_path / "b"
    PL.write_outputs(result, a)
    PL.write_outputs(again, b)
    for f in ("spots.csv", "candidates.csv"):
        assert (a / f).read_text() == (b / f).read_text()
    rows = list(csv.DictReader(open(a / "spots.csv")))
    assert len(rows) == 5 and list(rows[0]) == list(PL.SPOT_COLUMNS)
    s = result["spots"]
    assert [int(r["spot_id"]) for r in rows] == s["spot_id"].tolist()
    assert np.allclose([float(r["x_fit"]) for r in rows], s["x_fit"], rtol=1e-9, atol=0)             # %.10g
    assert rows[0]["optimizer_converged"] in ("True", "False") and rows[0]["fov_id"] == "syn"
    cands = list(csv.DictReader(open(a / "candidates.csv")))
    assert list(cands[0]) == list(__import__("catkit.detection", fromlist=["x"]).CANDIDATE_COLUMNS)
    meta = json.loads((a / "run_metadata.json").read_text())
    assert meta["schema_version"] == PL.SCHEMA_VERSION and meta["channel_roles"] == ROLES and meta["voxel_size_nm_zyx"] == [150.0, 65.0, 65.0]
    assert str(tmp_path) not in (a / "run_metadata.json").read_text()                  # no absolute paths in the metadata


# ---------------------------------------------------------------- empty results and no-cell FOV
def test_no_accepted_spots_gives_empty_tables_not_an_error(tmp_path):
    img = make_image(cy5=make_channel((), seed=21), tritc=make_channel((), seed=22))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = PL.analyze_fov(img, ROLES)
    assert len(r["spots"]["spot_id"]) == 0 and tuple(r["spots"]) == PL.SPOT_COLUMNS
    for ch in ROLES["detection"]:
        c = r["metadata"]["localization_qc"][ch]
        assert c["n_accepted"] == c["n_localized_rows"] == c["n_qc_passed"] == 0 and c["n_candidates"] >= 0
    PL.write_outputs(r, tmp_path)
    assert (tmp_path / "spots.csv").read_text().strip().split(",")[0] == "fov_id" and len((tmp_path / "spots.csv").read_text().strip().splitlines()) == 1


def test_no_cell_fov_keeps_meaning_and_reports_it():
    img = make_image(dapi=np.full((17, 100, 100), 50, np.uint16))
    r = PL.analyze_fov(img, ROLES)
    m = r["metadata"]["segmentation"]
    assert m["n_cells"] == 0 and m["segmentation_empty"] and m["noise_region_fraction"] == 1.0
    assert any("NoCellsWarning" in w and "n_cells=0" in w for w in r["metadata"]["warnings"])
    s = r["spots"]
    assert len(s["spot_id"]) == 5 and (s["cell_init"] == 0).all() and (s["cell_base"] == 0).all() and (s["assign_category"] == "outside").all()


# ---------------------------------------------------------------- inputs are validated, not guessed
def test_role_and_optics_validation():
    img = make_image()
    for roles in ({"segmentation": "DAPI", "detection": []}, {"segmentation": "DAPI", "detection": ["Cy5", "Cy5"]},
                  {"segmentation": "Cy5", "detection": ["Cy5"]}, {"detection": ["Cy5"]}):
        with pytest.raises(PL.PipelineError, match="roles must name"):
            PL.analyze_fov(img, roles)
    with pytest.raises(PL.PipelineError, match="not in the image stack"):
        PL.analyze_fov(img, {"segmentation": "Hoechst", "detection": ["Cy5"]})
    img.header["optics"].pop("TRITC")
    with pytest.raises(PL.PipelineError, match="optics .* missing .*not guessed"):
        PL.analyze_fov(img, ROLES)
    bad = make_image()
    bad.voxel_size_nm = (150.0, 43.3333333, 43.3333333)
    with pytest.raises(ValueError, match="not supported in v0.1"):
        PL.analyze_fov(bad, ROLES)


# ---------------------------------------------------------------- failed localization stays distinguishable
def test_failed_fit_row_is_kept_unreliable_with_initial_coordinates(monkeypatch):
    real = L.fit_spot
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            r = L._nan_result()
            r["fit_error"] = "roi_too_small"
            return r
        return real(*a, **k)

    monkeypatch.setattr(PL.L, "fit_spot", flaky)
    r = PL.analyze_fov(make_image(), ROLES)
    s = r["spots"]
    assert len(s["spot_id"]) == 5                                                  # no row dropped
    assert s["fit_error"][0] == "roi_too_small" and s["localization_quality"][0] == "unreliable" and not s["optimizer_converged"][0]
    assert np.isnan(s["x_fit"][0]) and s["coord_source"][0] == "init" and s["x_use"][0] == s["x_init"][0] and s["tier_frozen"][0] == "B10"
    assert (s["coord_source"][1:] == "fit").all()
    c = r["metadata"]["localization_qc"]["Cy5"]
    assert c["n_fit_error"] == 1 and c["quality"]["unreliable"] == 1 and c["n_qc_passed"] == c["n_accepted"] - 1 and c["coord_source"]["init"] == 1


# ---------------------------------------------------------------- use-coordinate rounding (frozen: numpy half-to-even, clipped)
def test_assign_use_coordinates_rounding_and_clipping_hand_computed():
    lab = np.zeros((10, 12), np.int32)
    lab[2, 4] = 1       # (y=2, x=4)
    lab[4, 6] = 2       # (y=4, x=6)
    lab[0, 11] = 3
    cores, lab_exp = lab.copy(), lab.copy()
    y = np.array([2.5, 3.5, 4.4, -3.0, 9.4, 0.4])
    x = np.array([4.5, 6.5, 6.4, 99.0, -1.0, 10.6])
    a = C.assign_use_coordinates(lab, lab_exp, cores, y, x, np.zeros(6, int), 65.0)
    # rounds: (2.5,4.5)->(2,4) cell 1 ; (3.5,6.5)->(4,6) cell 2 ; (4.4,6.4)->(4,6) cell 2 ; (-3,99)->clip (0,11) cell 3 ; (9.4,-1)->(9,0) bg ; (0.4,10.6)->(0,11) cell 3
    assert a["cell_base"].tolist() == [1, 2, 2, 3, 0, 3]
    assert a["assign_category"].tolist() == ["in_baseline"] * 4 + ["outside", "in_baseline"] and a["cell_fit_vs_init_differs"].tolist() == [True] * 4 + [False, True]
    # EDT of the background at the rounded pixel (9, 0), in nm: distance to the nearest of the three cell pixels
    d = 65.0 * min(np.hypot(9 - 4, 0 - 6), np.hypot(9 - 2, 0 - 4), np.hypot(9 - 0, 0 - 11))
    assert a["dist_to_baseline_mask_nm"][:4].tolist() == [0.0] * 4 and np.isclose(a["dist_to_baseline_mask_nm"][4], d, rtol=1e-12, atol=0)


# ---------------------------------------------------------------- cells.csv (M2.3-B)
def _csv_rows(path):
    return list(csv.DictReader(open(path)))


def test_cells_table_schema_geometry_and_counts(result, tmp_path):
    PL.write_outputs(result, tmp_path)
    rows = _csv_rows(tmp_path / "cells.csv")
    assert list(rows[0]) == list(PL.cells_columns(["Cy5", "TRITC"])) == list(result["cells_columns"])
    assert [int(r["cell_id"]) for r in rows] == [1, 2] and {r["fov_id"] for r in rows} == {"syn"}
    # geometry = baseline expanded labels (area, centroid); core area from the cores; um^2 = px^2 * 0.065^2
    lab = result["labels"]
    for r in rows:
        i = int(r["cell_id"])
        assert int(r["area_px"]) == int((lab == i).sum()) and abs(float(r["area_um2"]) - int(r["area_px"]) * 0.065 ** 2) < 1e-9
        yy, xx = np.nonzero(lab == i)
        assert abs(float(r["centroid_y_px"]) - yy.mean()) < 1e-9 and abs(float(r["centroid_x_px"]) - xx.mean()) < 1e-9
        assert 0 < int(r["core_area_px"]) < int(r["area_px"]) and r["touches_border"] == "False"
    # planted spots: Cy5 -> cell 1, cell 2 (+1 outside); TRITC -> cell 1, cell 2
    by = {int(r["cell_id"]): r for r in rows}
    for cid in (1, 2):
        for key in ("n_candidates_cell_init", "n_accepted_cell_init", "n_accepted_cell_base", "n_qc_passed_cell_base", "n_accepted_cell_exp_diag"):
            assert int(by[cid][f"Cy5_{key}"]) == 1 and int(by[cid][f"TRITC_{key}"]) == 1, (cid, key)
    # reconciliation with the spot populations and the outside tally
    rec = result["metadata"]["cells_reconciliation"]
    assert rec["Cy5"]["n_accepted_cell_base"] == dict(total=3, in_cells=2, outside=1) and rec["TRITC"]["n_accepted_cell_base"]["outside"] == 0
    for ch in ("Cy5", "TRITC"):
        for pop, v in rec[ch].items():
            assert v["in_cells"] == sum(int(r[f"{ch}_{pop}"]) for r in rows) and v["in_cells"] + v["outside"] == v["total"]
    assert [os_ for os_ in result["metadata"]["outputs"]] == ["spots.csv", "candidates.csv", "cells.csv", "run_metadata.json",
                                                              "qc_overlay_01_Cy5.png", "qc_overlay_02_TRITC.png"]


def test_cells_counts_keep_assignment_populations_separate():
    """Hand-built populations: one spot whose initial cell (1) differs from its use-coordinate cell (2), one only in the 8-px variant."""
    labels = np.zeros((20, 20), np.int32)
    labels[2:6, 2:6] = 1
    labels[2:6, 10:14] = 2
    cores = labels.copy()
    cand = dict(cell=np.array([1, 1, 0, 2, 0]))                                        # candidates (initial coordinates): 5 rows
    sp = dict(cell_init=np.array([1, 1, 0]), cell_base=np.array([1, 2, 0]), cell_exp=np.array([1, 2, 2]),   # accepted: 3 of the 5 candidates
              localization_quality=np.array(["good", "suspect", "good"], dtype=object))
    t, rec = PL.cells_table(labels, cores, 0.065, "f", ["Cy5"], {"Cy5": cand}, {"Cy5": sp})
    assert t["cell_id"].tolist() == [1, 2]
    assert t["Cy5_n_candidates_cell_init"].tolist() == [2, 1] and rec["Cy5"]["n_candidates_cell_init"] == dict(total=5, in_cells=3, outside=2)
    assert t["Cy5_n_accepted_cell_init"].tolist() == [2, 0]                           # initial coordinate: both in cell 1
    assert t["Cy5_n_accepted_cell_base"].tolist() == [1, 1]                           # use coordinate: the second moved to cell 2
    assert t["Cy5_n_qc_passed_cell_base"].tolist() == [1, 0]                          # only the 'good' ones, by cell_base
    assert t["Cy5_n_accepted_cell_exp_diag"].tolist() == [1, 2]                       # diagnostic variant also gets the outside spot
    assert rec["Cy5"]["n_accepted_cell_base"]["outside"] == 1 and rec["Cy5"]["n_accepted_cell_exp_diag"]["outside"] == 0
    # a cell id that has no row (here 7) makes the populations irreconcilable -> error, never silent
    with pytest.raises(PL.PipelineError, match="cell rows"):
        PL.cells_table(labels, cores, 0.065, "f", ["Cy5"], {"Cy5": cand}, {"Cy5": dict(sp, cell_base=np.array([1, 7, 0]))})


def test_no_cell_fov_writes_header_only_cells_csv(tmp_path):
    r = PL.analyze_fov(make_image(dapi=np.full((17, 100, 100), 50, np.uint16)), ROLES)
    PL.write_outputs(r, tmp_path, qc_overlays=False)
    text = (tmp_path / "cells.csv").read_text().strip().splitlines()
    assert len(text) == 1 and text[0].startswith("fov_id,cell_id,area_px") and not list(tmp_path.glob("*.png"))
    assert r["metadata"]["cells_reconciliation"]["Cy5"]["n_accepted_cell_base"] == dict(total=3, in_cells=0, outside=3)
