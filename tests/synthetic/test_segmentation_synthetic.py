"""Synthetic contract tests for M2.2-A: known shapes, hand-computed expectations, [N] guards.

No frozen reference is used here. Tests marked ``[F]`` pin frozen semantics with independent oracles;
tests marked ``[N]`` define NEW interface behaviour (guards, wrappers, cell geometry) that has no frozen
counterpart and must not be read as frozen behaviour.
"""

from __future__ import annotations

import warnings

import pytest

from scientific import require_modules

require_modules("numpy", "scipy", "skimage")

import numpy as np  # noqa: E402
from scipy.ndimage import binary_dilation, distance_transform_cdt  # noqa: E402

import catkit.cells as C  # noqa: E402
import catkit.segmentation as S  # noqa: E402
from synth_seg import NUCLEI, make_dapi_stack  # noqa: E402


# ---------------------------------------------------------------- [F] segmentation on known discs
def test_known_discs_label_count_order_and_expansion():
    st = make_dapi_stack()
    res = S.segment_fov(st, 65.0)
    lab, cores = res.cell_labels, res.nucleoid_cores
    assert res.n_cells == len(NUCLEI) == int(lab.max()) == int(cores.max())
    # IDs follow raster-scan order of each object's first pixel (skimage.measure.label): independent oracle
    first = {i: np.flatnonzero(cores.ravel() == i)[0] for i in range(1, len(NUCLEI) + 1)}
    assert [i for i, _ in sorted(first.items(), key=lambda kv: kv[1])] == list(range(1, len(NUCLEI) + 1))
    # expand_labels keeps IDs and only grows; no two cells merge (centres >= 38 px apart, expansion 4 px)
    assert (lab[cores > 0] == cores[cores > 0]).all()
    assert ((lab > 0) | (cores == 0)).all() and (lab > 0).sum() > (cores > 0).sum()
    assert {int(v) for v in np.unique(lab)} == set(range(0, len(NUCLEI) + 1))
    # each core is close to the true disc area (opening on a convex disc removes little)
    areas = sorted(int((cores == i).sum()) for i in range(1, len(NUCLEI) + 1))
    true = sorted(int(np.pi * r * r) for _, _, r in NUCLEI)
    assert all(abs(a - t) <= 0.15 * t for a, t in zip(areas, true))
    assert lab.dtype == cores.dtype == np.int32 and lab.shape == cores.shape == st.shape[1:]


def test_focus_plane_is_the_sharpest_neighbourhood():
    for fz in (0, 3, 6):
        res = S.segment_fov(make_dapi_stack(focus_z=fz, seed=fz), 65.0)
        assert res.focus_best == fz and res.focus_end - res.focus_start == 3 and res.focus_start <= fz < res.focus_end


@pytest.mark.parametrize("a_extra,kept", [(0, True), (1, False)])
def test_small_object_threshold_is_area_lt_min_size(a_extra, kept):
    """[F] semantics: objects with area < min_size are removed. A 3x5 rectangle has area 11 after opening with a
    disk(1) cross (erosion keeps 1x3; dilation gives 5 + 3 + 3 pixels): derived by hand, not by running the code."""
    img = np.zeros((30, 30), np.uint16)
    img[10:13, 10:15] = 200
    _, _, cores = S.segment_from_dapi(img, 11 + a_extra, 1, opening_radius=1)
    assert (int(cores.max()) == 1) is kept
    if kept:
        assert int((cores > 0).sum()) == 11


def test_remove_small_old_skimage_branch_is_selected_by_signature(monkeypatch):
    calls = []

    def fake_old(ar, min_size=64):          # signature without ``max_size``: the pre-0.26 API
        calls.append(("min_size", min_size))
        return ar

    monkeypatch.setattr(S, "remove_small_objects", fake_old)
    S._remove_small(np.ones((3, 3), bool), 15)
    assert calls == [("min_size", 15)]
    seen = []

    def fake_new(ar, max_size=64):
        seen.append(("max_size", max_size))
        return ar

    monkeypatch.setattr(S, "remove_small_objects", fake_new)
    S._remove_small(np.ones((3, 3), bool), 15)
    assert seen == [("max_size", 14)]       # area < 15  <=>  area <= 14


def test_seg_params_python_round_is_half_to_even():
    assert round(2.5) == 2 and round(0.5) == 0 and round(3.5) == 4                 # the primitive the frozen code relies on
    assert S.seg_params(104.0)["expand"] == 2 and S.seg_params(130.0)["opening"] == 0
    assert S.seg_params(65.0) == dict(min_size=15, expand=4, opening=1, dil=10)


# ---------------------------------------------------------------- [F] noise mask: structuring element (SCHEMAS 16.2)
def test_noise_mask_dilation_is_cross_shaped_taxicab():
    """Independent oracle: dilation by the default cross element, ``iterations=n`` == taxicab distance <= n."""
    rng = np.random.default_rng(0)
    for trial in range(5):
        lab = np.zeros((45, 55), np.int32)
        for k in range(1, 4):
            y, x = rng.integers(3, 42), rng.integers(3, 52)
            lab[y:y + rng.integers(1, 4), x:x + rng.integers(1, 4)] = k
        for dil in (1, 3, 10):
            expect = distance_transform_cdt(lab == 0, metric="taxicab") > dil
            assert np.array_equal(S.noise_mask(lab, dil), expect), (trial, dil)


def test_noise_mask_growth_is_diamond_not_square():
    lab = np.zeros((41, 41), np.int32)
    lab[20, 20] = 1
    m = S.noise_mask(lab, 10)
    assert not m[20, 10] and not m[10, 20] and not m[20, 30]            # on-axis reach: 10 px
    assert m[11, 11] and m[29, 29]                                      # (9, 9) off-centre: taxicab 18 > 10 -> outside
    square = ~binary_dilation(lab > 0, structure=np.ones((3, 3), bool), iterations=10)
    assert not square[11, 11] and not np.array_equal(m, square)         # a square element would differ


def test_noise_mask_zero_iterations_frozen_scipy_quirk():
    """[F quirk, documented] iterations < 1 means "repeat until nothing changes": one cell fills the frame."""
    lab = np.zeros((20, 20), np.int32)
    lab[5, 5] = 1
    assert not S.noise_mask(lab, 0).any()
    assert S.noise_mask(np.zeros((20, 20), np.int32), 0).all()          # no cell: still the whole frame is outside


# ---------------------------------------------------------------- [N] no-cell FOV (Q4: keep frozen behaviour + warning)
def test_no_cell_fov_keeps_frozen_noise_region_and_warns():
    flat = np.full((7, 60, 60), 120, np.uint16)
    with pytest.warns(S.NoCellsWarning, match="n_cells=0") as rec:
        res = S.segment_fov(flat, 65.0)
    assert len(rec) >= 1 and res.n_cells == 0 and res.cell_labels.max() == 0 and res.nucleoid_cores.max() == 0
    assert len(res.warnings) == 1 and "n_cells=0" in res.warnings[0]
    assert S.noise_mask(res.cell_labels, res.seg_params["dil"]).all()   # whole frame is the noise region (frozen)
    # this wrapper says nothing about sigma validity: that guard belongs to the detection checkpoints (B/C)


def test_no_cell_external_labels_also_warn_and_cells_present_do_not():
    with pytest.warns(S.NoCellsWarning):
        S.segment_fov(None, 65.0, method="external", external_labels=np.zeros((20, 20), np.int32))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = S.segment_fov(make_dapi_stack(), 65.0)
    assert res.warnings == [] and res.n_cells == 4


def test_no_cell_vs_invalid_input_are_separate():
    """Valid flat image -> n_cells = 0 + warning; empty dimension -> ValueError (no warning, no n_cells)."""
    for shp in ((5, 0, 10), (5, 10, 0), (0, 10, 10)):
        with warnings.catch_warnings():
            warnings.simplefilter("error")                              # also no stray RuntimeWarning
            with pytest.raises(ValueError):
                S.segment_fov(np.zeros(shp, np.uint16), 65.0)
    with pytest.warns(S.NoCellsWarning):
        assert S.segment_fov(np.full((5, 30, 30), 9, np.uint16), 65.0).n_cells == 0


def test_zero_cell_edt_distances_are_numbers_without_nearest_cell_meaning():
    """Frozen low-level result kept raw: finite numbers exist even though no cell exists."""
    cid, dist = C.assign_cells_init(np.array([[0, 5, 5]]), np.zeros((20, 20), np.int32), 0.065)
    assert cid.tolist() == [0] and np.isfinite(dist).all()


# ---------------------------------------------------------------- [N] guards of segment_fov
def test_guard_fewer_than_three_planes():
    for z in (1, 2):
        with pytest.raises(ValueError, match="at least 3"):
            S.segment_fov(make_dapi_stack(shape=(z, 40, 40)), 65.0)
    assert S.segment_fov(make_dapi_stack(shape=(3, 120, 120), focus_z=1), 65.0).n_cells == len(NUCLEI)   # Z == 3 is valid


def test_guard_dimensionality_pixel_size_and_method():
    with pytest.raises(ValueError, match="3D"):
        S.segment_fov(np.zeros((40, 40)), 65.0)
    for bad in (43.3333333, 64.9, 130.0):
        with pytest.raises(ValueError, match="65 nm"):
            S.segment_fov(make_dapi_stack(), bad)
    for m in ("omnipose", "both"):
        with pytest.raises(NotImplementedError, match="M4"):
            S.segment_fov(make_dapi_stack(), 65.0, method=m)
    with pytest.raises(ValueError, match="unknown segmentation method"):
        S.segment_fov(make_dapi_stack(), 65.0, method="cellpose")


def test_external_labels_contract():
    lab = np.zeros((30, 30), np.int64)
    lab[2:8, 2:8] = 1
    lab[15:20, 15:20] = 5                                               # non-consecutive ids are kept as given
    res = S.segment_fov(None, 65.0, method="external", external_labels=lab)
    assert res.method == "external" and res.otsu_threshold is None and res.focus_best is None
    assert res.cell_labels.dtype == np.int32 and res.n_cells == 2 and np.array_equal(res.cell_labels, lab)
    assert np.array_equal(res.nucleoid_cores, res.cell_labels)          # cores default to the labels
    for bad, msg in ((None, "requires external_labels"), (lab.astype(float), "2D non-negative integer"),
                     (-lab, "2D non-negative integer"), (lab[None], "2D non-negative integer")):
        with pytest.raises(ValueError, match=msg):
            S.segment_fov(None, 65.0, method="external", external_labels=bad)
    with pytest.raises(ValueError, match="same shape"):
        S.segment_fov(None, 65.0, method="external", external_labels=lab, external_cores=lab[:10])


# ---------------------------------------------------------------- assign_cells_init (frozen semantics, hand oracle)
def test_assign_cells_init_hand_computed():
    lab = np.zeros((30, 40), np.int32)
    lab[10:14, 10:14] = 2
    co = np.array([[3, 11, 12],      # inside cell 2
                   [0, 11, 17],      # 4 px right of the mask edge (col 13 -> 17)
                   [5, 20, 11],      # 7 px below the mask edge (row 13 -> 20)
                   [1, 0, 0]])       # far corner
    cid, dist = C.assign_cells_init(co, lab, 0.065)
    assert cid.tolist() == [2, 0, 0, 0] and cid.dtype == np.dtype(int)
    assert dist[0] == 0.0
    assert np.isclose(dist[1], 4 * 0.065, rtol=1e-12, atol=0) and np.isclose(dist[2], 7 * 0.065, rtol=1e-12, atol=0)
    assert np.isclose(dist[3], np.hypot(10, 10) * 0.065, rtol=1e-12, atol=0)     # to the nearest mask pixel (10,10)
    assert not (co - co.round()).any() and co.dtype.kind == "i"


def test_assign_cells_init_has_no_shift_parameters_and_uses_initial_coordinates():
    import inspect
    assert list(inspect.signature(C.assign_cells_init).parameters) == ["coords_zyx", "labels", "px_um"]
    lab = np.zeros((10, 10), np.int32)
    lab[4, 4] = 1
    cid, _ = C.assign_cells_init(np.array([[0, 4, 4], [0, 4, 5]]), lab, 0.065)    # integer init lookup, no rounding of fits
    assert cid.tolist() == [1, 0]


# ---------------------------------------------------------------- [N] cell_geometry
def test_cell_geometry_hand_computed():
    lab = np.zeros((10, 12), np.int32)
    lab[0:2, 0:3] = 1                 # touches the top and left borders: 6 px, centroid (0.5, 1.0)
    lab[4:6, 5:7] = 3                 # interior, id 2 absent: 4 px, centroid (4.5, 5.5)
    lab[8:10, 9:12] = 4               # touches bottom/right borders: 6 px, centroid (8.5, 10.0)
    cores = np.zeros_like(lab)
    cores[0, 0:2] = 1                 # core area 2
    cores[4, 5] = 3                   # core area 1
    # id 4 has no core pixels -> 0
    g = C.cell_geometry(lab, cores, 0.065)
    assert g["cell_id"].tolist() == [1, 3, 4]
    assert g["area_px"].tolist() == [6, 4, 6]
    assert np.allclose(g["area_um2"], np.array([6, 4, 6]) * 0.065 ** 2, rtol=1e-12, atol=0)
    assert np.allclose(g["centroid_y_px"], [0.5, 4.5, 8.5]) and np.allclose(g["centroid_x_px"], [1.0, 5.5, 10.0])
    assert g["core_area_px"].tolist() == [2, 1, 0]
    assert g["touches_border"].tolist() == [True, False, True]
    assert set(g) == {"cell_id", "area_px", "area_um2", "centroid_y_px", "centroid_x_px", "core_area_px", "touches_border"}
    assert len({len(v) for v in g.values()}) == 1


def test_cell_geometry_empty_and_guards():
    g = C.cell_geometry(np.zeros((5, 5), np.int32), np.zeros((5, 5), np.int32), 0.065)
    assert all(len(v) == 0 for v in g.values())
    with pytest.raises(ValueError):
        C.cell_geometry(np.zeros((5, 5), np.int32), np.zeros((5, 6), np.int32), 0.065)
    with pytest.raises(ValueError):
        C.cell_geometry(np.zeros((2, 5, 5), np.int32), np.zeros((2, 5, 5), np.int32), 0.065)


def test_cell_geometry_on_segmented_discs_is_consistent_with_segmentation():
    res = S.segment_fov(make_dapi_stack(), 65.0)
    g = C.cell_geometry(res.cell_labels, res.nucleoid_cores, 0.065)
    assert g["cell_id"].tolist() == [1, 2, 3, 4] and not g["touches_border"].any()
    assert (g["core_area_px"] < g["area_px"]).all() and g["area_px"].sum() == int((res.cell_labels > 0).sum())
