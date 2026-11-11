"""Functional checks of the scientific stack the frozen algorithms rely on.

These test real behaviour on small synthetic inputs with known answers, not
just that packages import. They cover exactly the third-party APIs the frozen
reference code calls (Big-FISH detection building blocks, SciPy bounded TRF
least squares, scikit-image segmentation primitives, pandas/tifffile I/O).
"""

from __future__ import annotations

import importlib.metadata as md
import os
import re
import warnings
from pathlib import Path


def requiring() -> bool:
    return os.environ.get("CATKIT_REQUIRE_SCIENTIFIC_DEPS") == "1"


CONSTRAINTS = Path(__file__).resolve().parents[2] / "constraints" / "py311-research.txt"
SHAPE = (17, 64, 64)
VOXEL = (150.0, 65.0, 65.0)
SPOT_RADIUS = (410.0, 97.0, 97.0)  # nm, (z, y, x)


def _synthetic_volume(np, centers, amp=12.0, bg=11.0, noise=1.0, sigma=(2.7, 1.5, 1.5), seed=0):
    rng = np.random.default_rng(seed)
    vol = bg + rng.normal(0.0, noise, SHAPE)
    zz, yy, xx = np.meshgrid(*[np.arange(n, dtype=float) for n in SHAPE], indexing="ij")
    for c in centers:
        vol += amp * np.exp(-0.5 * (((zz - c[0]) / sigma[0]) ** 2 + ((yy - c[1]) / sigma[1]) ** 2 + ((xx - c[2]) / sigma[2]) ** 2))
    return np.rint(vol).astype(np.float32)


CENTERS = [(8, 15, 15), (8, 15, 47), (6, 45, 20), (10, 48, 50)]


def test_installed_versions_match_constraints(require):
    pins = dict(re.findall(r"^([A-Za-z0-9_.-]+)==(\S+)$", CONSTRAINTS.read_text(), flags=re.M))
    np = require("numpy")
    bigfish = require("bigfish")
    assert int(np.__version__.split(".")[0]) == 2
    assert bigfish.__version__ == "0.6.2"
    if requiring():
        key = ["big-fish", "numpy", "scipy", "scikit-image", "pandas", "tifffile", "nd2", "scikit-learn", "matplotlib", "mrc"]
        wrong = {k: (md.version(k), pins[k]) for k in key if md.version(k) != pins[k]}
        assert not wrong, f"installed != pinned: {wrong}"


def test_bigfish_imports_and_radius_conversion(require):
    require("bigfish.stack")
    require("bigfish.detection")
    utils = require("bigfish.detection.utils")
    r = utils.get_object_radius_pixel(voxel_size_nm=VOXEL, object_radius_nm=SPOT_RADIUS, ndim=3)
    assert tuple(round(v, 12) for v in r) == tuple(round(a / b, 12) for a, b in zip(SPOT_RADIUS, VOXEL))


def test_log_filter_float32_keeps_resolution_uint16_collapses(require):
    np = require("numpy")
    bfs = require("bigfish.stack")
    img = _synthetic_volume(np, CENTERS)
    sig = tuple(a / b for a, b in zip(SPOT_RADIUS, VOXEL))
    f32 = bfs.log_filter(img, sig)
    assert f32.dtype == np.float32 and f32.shape == img.shape and np.all(f32 >= 0)
    assert len(np.unique(f32)) > 1000  # continuous response
    u16 = bfs.log_filter(img.astype(np.uint16), sig)
    assert u16.dtype == np.uint16 and len(np.unique(u16)) <= 3  # low-count data collapses to a few integers


def test_local_max_and_threshold_recover_known_spots(require):
    np = require("numpy")
    bfs = require("bigfish.stack")
    bfd = require("bigfish.detection")
    img = _synthetic_volume(np, CENTERS)
    sig = tuple(a / b for a, b in zip(SPOT_RADIUS, VOXEL))
    filt = bfs.log_filter(img, sig)
    mlm = bfd.local_maximum_detection(filt, sig)
    thr = float(np.max(filt)) * 0.3
    spots, _ = bfd.spots_thresholding(filt, mlm, thr, remove_duplicate=True)
    assert spots.dtype == np.int64 and spots.shape[1] == 3
    for c in CENTERS:
        d = np.abs(spots - np.array(c)).max(axis=1)
        assert d.min() <= 1, f"spot near {c} not recovered"


def test_detect_spots_equals_building_blocks_and_auto_single_image(require):
    np = require("numpy")
    bfs = require("bigfish.stack")
    bfd = require("bigfish.detection")
    img = _synthetic_volume(np, CENTERS)
    sig = tuple(a / b for a, b in zip(SPOT_RADIUS, VOXEL))
    filt = bfs.log_filter(img, sig)
    mlm = bfd.local_maximum_detection(filt, sig)
    thr = float(np.max(filt)) * 0.3
    blocks, _ = bfd.spots_thresholding(filt, mlm, thr, remove_duplicate=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = bfd.detect_spots(img, threshold=thr, voxel_size=VOXEL, spot_radius=SPOT_RADIUS, remove_duplicate=True)
        auto_thr = bfd.automated_threshold_setting(filt, mlm)
        ref_auto, ref_auto_thr = bfd.detect_spots(img, threshold=None, return_threshold=True,
                                                  voxel_size=VOXEL, spot_radius=SPOT_RADIUS)
    assert {tuple(r) for r in blocks} == {tuple(r) for r in ref}
    assert auto_thr is not None and auto_thr == ref_auto_thr  # single-image consistency
    direct, _ = bfd.spots_thresholding(filt, mlm, auto_thr, remove_duplicate=True)
    assert {tuple(r) for r in direct} == {tuple(r) for r in ref_auto}


def test_fit_subpixel_returns_float_coords_near_truth(require):
    np = require("numpy")
    bfd = require("bigfish.detection")
    img = _synthetic_volume(np, [(8.3, 30.4, 31.6)], amp=40.0, noise=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = bfd.fit_subpixel(img, np.array([[8, 30, 32]], np.int64), VOXEL, SPOT_RADIUS)
    assert out.dtype.kind == "f"
    assert np.abs(out[0] - np.array([8.3, 30.4, 31.6])).max() < 0.5


def test_scipy_bounded_trf_erf_and_kdtree(require):
    np = require("numpy")
    optimize = require("scipy.optimize")
    special = require("scipy.special")
    spatial = require("scipy.spatial")
    k = np.arange(-6, 7, dtype=float)

    def model(p):
        mu, s, a, b = p
        s2 = np.sqrt(2.0) * s
        return b + a * s * np.sqrt(np.pi / 2) * (special.erf((k + 0.5 - mu) / s2) - special.erf((k - 0.5 - mu) / s2))

    truth = np.array([0.3, 1.7, 20.0, 5.0])
    y = model(truth)
    res = optimize.least_squares(lambda p: model(p) - y, [0.0, 1.0, 10.0, 0.0],
                                 bounds=([-2, 0.5, 0, 0], [2, 4, 100, 50]), method="trf", x_scale="jac", max_nfev=300)
    assert res.success and np.allclose(res.x, truth, atol=1e-6)
    assert hasattr(res, "active_mask") and res.active_mask.shape == (4,) and res.jac.shape == (13, 4)
    d, j = spatial.cKDTree(np.array([[0.0, 0, 0], [3, 4, 0], [10, 0, 0]])).query([[0.0, 0, 0]], k=2)
    assert d[0, 1] == 5.0 and j[0, 1] == 1


def test_skimage_026_primitives(require):
    np = require("numpy")
    require("skimage")
    import inspect

    from skimage import measure
    from skimage.filters import threshold_otsu
    from skimage.morphology import disk, opening, remove_small_objects
    from skimage.segmentation import expand_labels

    # remove_small_objects renamed min_size -> max_size in 0.26; frozen shim uses max_size=min_size-1
    assert "max_size" in inspect.signature(remove_small_objects).parameters
    img = np.zeros((20, 40), bool)
    img[2:5, 2:7] = True  # area 15 -> kept with min_size=15
    img[10:12, 20:27] = True  # area 14 -> removed
    out = remove_small_objects(img, max_size=15 - 1)
    assert out[3, 4] and not out[11, 22]
    blobs = np.zeros((60, 60), float)
    blobs[10:20, 10:20] = 100.0
    blobs[35:50, 30:45] = 120.0
    thr = threshold_otsu(blobs)
    lab = measure.label(opening(blobs > thr, disk(1)))
    exp = expand_labels(lab, distance=4)
    assert lab.max() == 2 and set(np.unique(exp)) == {0, 1, 2}
    assert np.all(exp[lab > 0] == lab[lab > 0])  # IDs preserved by expansion


def test_pandas3_strings_and_csv_roundtrip(require, tmp_path):
    np = require("numpy")
    pd = require("pandas")
    df = pd.DataFrame(dict(reasons=["z_edge;low_snr_peak", "", "large_shift_xy"], x=[1.0 / 3, np.nan, 1952.123456789]))
    assert df.reasons.str.contains("z_edge").tolist() == [True, False, False]
    p = tmp_path / "t.csv"
    df.to_csv(p, index=False, float_format="%.10g")
    back = pd.read_csv(p, keep_default_na=True)
    assert np.isnan(back.x[1]) and abs(back.x[2] - 1952.123456789) < 1e-6
    assert back.reasons.fillna("").tolist() == ["z_edge;low_snr_peak", "", "large_shift_xy"]


def test_tifffile_int32_zlib_roundtrip_and_nd2_api(require, tmp_path):
    np = require("numpy")
    tifffile = require("tifffile")
    nd2 = require("nd2")
    lab = np.arange(64 * 64, dtype=np.int32).reshape(64, 64) % 7
    p = tmp_path / "labels.tif"
    tifffile.imwrite(p, lab, compression="zlib")
    back = tifffile.imread(p)
    assert back.dtype == np.int32 and np.array_equal(back, lab)
    assert hasattr(nd2, "ND2File") and hasattr(nd2, "imread")  # no public ND2 sample: real reading is tested opt-in later
