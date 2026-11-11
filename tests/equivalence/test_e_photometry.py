"""E-layer equivalence: ``catkit.photometry`` vs frozen ``REF-PHOT`` (same process, same inputs).

Every case compares the WHOLE returned structure (all keys, NaN fields, flags,
dtypes) under the explicit per-field specs in ``compare.py`` (unknown key = error).
Each case also asserts independently that the branch it targets was really taken,
so two identical wrong results cannot pass as "coverage".

Equivalence to the frozen code is NOT evidence of real-data photometric accuracy.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from scientific import require_modules

require_modules("numpy", "scipy")

import numpy as np  # noqa: E402

import catkit.photometry as P  # noqa: E402
from compare import (CLIPBG_SPEC, ELLIPSOID_SPEC, EXACT, NEIGHBOR_SPEC, OCCUPANCY_SPEC, OVERLAP_SPEC, PHOT_SPEC,  # noqa: E402
                     call_outcome, compare_dicts, compare_value)
from synth import AP, SHAPE, SHIN, SHOUT, TRUE, VOX, VOX_BASE, make_volume, rescale_radii  # noqa: E402

pytestmark = pytest.mark.research

FUNCS = ["ellipsoid_voxels", "build_occupancy", "clipped_background", "measure_spots", "nearest_neighbor_nm", "overlap_flags"]

# shells whose retained voxel count is exactly 99 / 100 / 101 at centre (20.3, 20.2, 20.1) in a (40,40,40) volume
# (found by search; the counts are re-asserted in the test, not trusted)
MINSHELL_GEOM = {
    99: ((2.0, 2.0, 2.75), (2.5, 3.75, 3.75)),
    100: ((6.25, 2.25, 2.25), (6.75, 3.0, 2.75)),
    101: ((2.25, 2.25, 3.0), (3.0, 2.75, 4.75)),
}
C_FRAC = (20.3, 20.2, 20.1)


def _same_array(a, b):
    """Bitwise-equal comparison that treats NaN == NaN and compares +/-Inf by sign (dtype and shape must match too)."""
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind in "fc")


def _both(fr, po, args, spec, label, stats=None, kwargs=None):
    """Run frozen and port on the same inputs; compare return values (or the exception) under ``spec``.

    Each side gets its own private copy of every array argument; afterwards each copy is compared (NaN/Inf aware)
    with the pristine original, so neither implementation may modify its inputs in place.
    """
    kwargs = kwargs or {}
    outcomes, problems = [], []
    for side, fn in (("frozen", fr), ("port", po)):
        a_args, a_kw = _copy_args(args), {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in kwargs.items()}
        outcomes.append(call_outcome(fn, *a_args, **a_kw))
        for i, (orig, now) in enumerate(zip(args, a_args)):
            if isinstance(orig, np.ndarray) and not _same_array(orig, now):
                problems.append(f"{label}: {side} modified array argument #{i} in place")
        for k, orig in kwargs.items():
            if isinstance(orig, np.ndarray) and not _same_array(orig, a_kw[k]):
                problems.append(f"{label}: {side} modified array keyword {k!r} in place")
    a, b = outcomes
    if a[0] != b[0]:
        return problems + [f"{label}: outcome differs: frozen={a[:2]} port={b[:2]}"], a, b
    if a[0] == "raise":
        return problems + ([] if a == b else [f"{label}: exceptions differ: {a} vs {b}"]), a, b
    if isinstance(a[1], dict):
        return problems + compare_dicts(a[1], b[1], spec, stats, prefix=f"{label}."), a, b
    return problems + compare_value(a[1], b[1], spec, label, stats), a, b


def _copy_args(args):
    return tuple(x.copy() if isinstance(x, np.ndarray) else x for x in args)


# --------------------------------------------------------------------- source identity
def test_port_is_verbatim(frozen_modules):
    L = frozen_modules["REF-PHOT"]
    for name in FUNCS:
        assert inspect.getsource(getattr(P, name)) == inspect.getsource(getattr(L, name)), name
    pa, la = ast.parse(inspect.getsource(P)), ast.parse(inspect.getsource(L))
    assert ast.dump(ast.Module(pa.body[1:], [])) == ast.dump(ast.Module(la.body[1:], []))


# --------------------------------------------------------------------- ellipsoid_voxels / build_occupancy
ELL_CASES = [
    ("int_centre", (8, 25, 35), (3, 5, 5), SHAPE, None),
    ("frac_centre", (8.3, 24.4, 35.7), (3, 5, 5), SHAPE, None),
    ("frac_radii_baseline", (8.3, 24.4, 35.7), rescale_radii((3.0, 5.0, 5.0), VOX_BASE), SHAPE, None),
    ("frac_radii_ref", (8.3, 24.4, 35.7), rescale_radii((3.0, 5.0, 5.0), VOX), SHAPE, None),
    ("annulus", (8.3, 24.4, 35.7), (7, 13, 13), SHAPE, (5, 9, 9)),
    ("annulus_baseline", (8.0, 25.0, 35.0), rescale_radii((7.0, 13.0, 13.0), VOX_BASE), SHAPE,
     rescale_radii((5.0, 9.0, 9.0), VOX_BASE)),
    ("z_low_edge", (0.4, 25, 35), (3, 5, 5), SHAPE, None),
    ("z_high_edge", (16.2, 25, 35), (3, 5, 5), SHAPE, None),
    ("y_edge", (8, 0.0, 35), (3, 5, 5), SHAPE, None),
    ("x_edge", (8, 25, 69.6), (3, 5, 5), SHAPE, None),
    ("corner", (0, 0, 0), (3, 5, 5), SHAPE, None),
    ("far_corner_annulus", (16, 49, 69), (7, 13, 13), SHAPE, (5, 9, 9)),
    ("fully_outside", (100, 100, 100), (3, 5, 5), SHAPE, None),
    ("negative_centre", (-2.5, -3.5, -1.5), (3, 5, 5), SHAPE, None),
    ("tiny_radii", (8, 25, 35), (0.4, 0.4, 0.4), SHAPE, None),
    ("tiny_radii_off_grid", (8.5, 25.5, 35.5), (0.4, 0.4, 0.4), SHAPE, None),
    ("distinct_yx_radii", (8.3, 24.4, 35.7), (2.5, 4.0, 6.5), SHAPE, None),            # ry != rx: catches y/x swaps
    ("distinct_yx_annulus", (8.3, 24.4, 35.7), (4.5, 7.0, 9.5), SHAPE, (2.5, 4.0, 6.5)),
    ("exact_boundary_voxel", (8, 25, 35), (3, 5, 5), SHAPE, None),            # d^2 == 1 exactly on axis voxels
    ("shape_smaller_than_ap", (2, 4, 4), (3, 5, 5), (5, 9, 9), None),
    ("annulus_equal_radii", (8, 25, 35), (3, 5, 5), SHAPE, (3, 5, 5)),         # empty annulus
    ("float32_centre", (np.float32(8.3), np.float32(24.4), np.float32(35.7)), (3, 5, 5), SHAPE, None),
    ("list_inputs", [8.3, 24.4, 35.7], [3, 5, 5], list(SHAPE), None),
]


def test_ellipsoid_voxels_equivalence(frozen_modules, e_stats):
    L = frozen_modules["REF-PHOT"]
    problems, sizes = [], {}
    for name, c, ro, shape, ri in ELL_CASES:
        kw = {} if ri is None else {"r_in": ri}
        pr, a, b = _both(L.ellipsoid_voxels, P.ellipsoid_voxels, (c, ro, shape), ELLIPSOID_SPEC, f"ellipsoid[{name}]", e_stats, kw)
        problems += pr
        sizes[name] = (len(a[1][0][0]), a[1][1])
    assert not problems, "\n".join(problems[:10])
    # branch-trigger assertions (do not rely on frozen == port alone)
    assert sizes["int_centre"][0] == sizes["int_centre"][1] > 0                      # nothing clipped inside the image
    assert sizes["fully_outside"] == (0, sizes["int_centre"][1]) or sizes["fully_outside"][0] == 0
    assert sizes["corner"][0] < sizes["corner"][1]                                    # clipped at the corner
    assert sizes["z_low_edge"][0] < sizes["z_low_edge"][1] and sizes["x_edge"][0] < sizes["x_edge"][1]
    assert sizes["annulus_equal_radii"] == (0, 0)                                     # r_in == r_out -> empty
    assert sizes["negative_centre"][0] < sizes["negative_centre"][1]


def test_build_occupancy_equivalence(frozen_modules, e_stats):
    L = frozen_modules["REF-PHOT"]
    rng = np.random.default_rng(0)
    cases = [
        ("none", SHAPE, [], AP),
        ("single", SHAPE, [(8, 25, 35)], AP),
        ("overlap_pair", SHAPE, [(8.0, 25.0, 30.0), (8.0, 25.0, 36.0)], AP),
        ("edge_and_outside", SHAPE, [(0, 0, 0), (16, 49, 69), (200, 200, 200)], AP),
        ("random_20", SHAPE, [tuple(rng.uniform((0, 0, 0), SHAPE)) for _ in range(20)], AP),
        ("baseline_radii", SHAPE, [(8.0, 25.0, 30.0), (8.0, 25.0, 36.0)], rescale_radii((3.0, 5.0, 5.0), VOX_BASE)),
        ("int_array_centres", SHAPE, np.array([[8, 25, 30], [8, 25, 36]]), AP),
    ]
    problems = []
    for name, shape, centers, ap in cases:
        pr, a, b = _both(L.build_occupancy, P.build_occupancy, (shape, centers, ap), OCCUPANCY_SPEC, f"occ[{name}]", e_stats)
        problems += pr
        assert a[1].dtype == np.uint8
    assert not problems, "\n".join(problems[:10])
    pair = P.build_occupancy(SHAPE, cases[2][2], AP)
    assert pair.max() == 2 and (pair == 2).sum() > 0 and pair.sum() > (pair > 0).sum()   # overlap counted twice


def test_occupancy_uint8_wraps_beyond_255(frozen_modules):
    """Frozen behaviour kept: the uint8 counter wraps at 256 covering apertures (documented limitation, not fixed)."""
    L = frozen_modules["REF-PHOT"]
    for n, expect in ((254, 254), (255, 255), (256, 0), (257, 1)):
        centers = [(8, 25, 35)] * n
        a = L.build_occupancy(SHAPE, centers, (1, 1, 1))
        b = P.build_occupancy(SHAPE, centers, (1, 1, 1))
        assert a.dtype == b.dtype == np.uint8
        assert np.array_equal(a, b)
        assert a[8, 25, 35] == expect, (n, a[8, 25, 35])


# --------------------------------------------------------------------- clipped_background
def _clip_inputs():
    rng = np.random.default_rng(0)
    out = [
        ("empty", np.array([]), {}),
        ("all_nan", np.full(10, np.nan), {}),
        ("all_nonfinite", np.array([np.nan, np.inf, -np.inf]), {}),
        ("n1", np.array([5.0]), {}),
        ("n2", np.array([5.0, 7.0]), {}),
        ("n3", np.array([5.0, 7.0, 6.0]), {}),
        ("n4", np.array([1.0, 2.0, 3.0, 4.0]), {}),
        ("all_same", np.full(50, 10.0), {}),
        ("outliers", np.concatenate([rng.normal(11, 1.1, 3000), [60.0, 70.0]]), {}),
        ("many_outliers", np.concatenate([rng.normal(11, 1.1, 300), rng.normal(300, 5, 100)]), {}),
        ("quantised", np.clip(np.rint(11.6 + rng.normal(0, 1.1, 3000)), 0, None), {}),
        ("mad_zero_std_pos", np.where(rng.random(3000) < 0.03, 11.0, 10.0), {}),
        ("with_inf", np.concatenate([rng.normal(10, 1, 200), [np.inf, -np.inf, np.nan]]), {}),
        ("int_dtype", np.array([1, 2, 3, 4, 5, 6, 7, 8, 100]), {}),
        ("float32", rng.normal(10, 1, 500).astype(np.float32), {}),
        ("2d_input", rng.normal(10, 1, (20, 25)), {}),
        ("list_input", [1.0, 2.0, 2.5, 3.0, 40.0], {}),
        ("tiny_noise", 10.0 + rng.normal(0, 1e-3, 500), {}),
        ("clip_sigma_1", rng.normal(10, 2, 500), {"sigma": 1.0}),
        ("maxiter_0", np.concatenate([rng.normal(10, 1, 200), [90.0]]), {"maxiter": 0}),
        ("maxiter_1", np.concatenate([rng.normal(10, 1, 200), [90.0]]), {"maxiter": 1}),
        ("clip_floor_big", rng.normal(10, 0.01, 200), {"clip_floor": 5.0}),
        ("clip_floor_zero", rng.normal(10, 0.01, 200), {"clip_floor": 0.0}),
    ]
    return out


def test_clipped_background_equivalence(frozen_modules, e_stats):
    L = frozen_modules["REF-PHOT"]
    problems, res = [], {}
    for name, v, kw in _clip_inputs():
        pr, a, b = _both(L.clipped_background, P.clipped_background, (v,), CLIPBG_SPEC, f"clip[{name}]", e_stats, kw)
        problems += pr
        res[name] = a[1]
    assert not problems, "\n".join(problems[:10])
    # branch triggers
    assert res["empty"]["n_kept"] == 0 and np.isnan(res["empty"]["mean"])
    assert res["all_nan"]["n_kept"] == 0 and res["all_nan"]["n_nonfinite"] == 10
    assert res["all_nonfinite"]["n_nonfinite"] == 3
    assert np.isnan(res["n1"]["noise_std"]) and np.isnan(res["n2"]["noise_std"])                 # ddof=1 needs > 2 voxels
    assert np.isfinite(res["n3"]["noise_std"])
    assert res["all_same"]["noise_std"] == 0 and res["all_same"]["n_kept"] == 50 and np.isnan(res["all_same"]["noise_legacy"])
    assert res["outliers"]["n_kept"] < 3002
    assert res["mad_zero_std_pos"]["noise_mad"] == 0 and res["mad_zero_std_pos"]["noise_legacy"] == res["mad_zero_std_pos"]["noise_std"] > 0
    assert res["with_inf"]["n_nonfinite"] == 3
    assert res["maxiter_0"]["n_kept"] == 201
    assert res["maxiter_1"]["n_kept"] == 200


def test_clipped_background_exceptions_and_clip_semantics(frozen_modules):
    L = frozen_modules["REF-PHOT"]
    # arguments that make both implementations raise (message and type must match too)
    for label, args in (("none", (None,)), ("string", ("abc",)), ("ragged", ([[1.0, 2.0], [3.0]],))):
        a, b = call_outcome(L.clipped_background, *args), call_outcome(P.clipped_background, *args)
        assert a[0] == b[0], label
        if a[0] == "raise":
            assert a == b, label
    # clipping keeps ddof=0 std for the clip but ddof=1 for the reported noise
    v = np.array([1.0, 2.0, 3.0, 4.0])
    r = P.clipped_background(v)
    assert r["noise_std"] == pytest.approx(np.std(v, ddof=1), rel=1e-15) and r["noise_std"] != np.std(v)


# --------------------------------------------------------------------- measure_spots
def _noise_vol(seed, bg=10.0, noise=1.0, integer=False):
    rng = np.random.default_rng(seed)
    v = bg + rng.normal(0, noise, SHAPE)
    return np.clip(np.rint(v), 0, 65535) if integer else v


def _meas_cases():
    c1 = [(8, 25, 35)]
    ap, si, so = AP, SHIN, SHOUT
    gv = make_volume([(TRUE["mu"], TRUE["sig"], TRUE["A"])], bg=TRUE["bg"])
    gvn = make_volume([(TRUE["mu"], TRUE["sig"], TRUE["A"])], bg=TRUE["bg"], noise=1.0, seed=3)
    two = [(8.0, 25.0, 30.0), (8.0, 25.0, 36.0)]
    pair_vol = make_volume([(c, TRUE["sig"], 20.0) for c in two], bg=10.0)
    occ_pair = P.build_occupancy(SHAPE, two, ap)
    mad0 = np.full(SHAPE, 10.0)
    mad0[np.random.default_rng(0).random(SHAPE) < 0.03] = 11.0
    tiny = 10.0 + np.random.default_rng(3).normal(0, 1e-3, SHAPE)
    tiny[8, 25, 35] += 5.0
    quant = np.clip(np.rint(11.6 + np.random.default_rng(5).normal(0, 1.1, SHAPE)), 0, 65535)
    shell_nf = np.random.default_rng(5).normal(10, 1, SHAPE)
    shell_nf[8, 25, 45] = np.nan
    shell_nf[8, 25, 22] = np.inf
    ap_nan = _noise_vol(1)
    ap_nan[8, 25, 35] = np.nan
    ap_inf = _noise_vol(1)
    ap_inf[8, 25, 36] = np.inf
    ap_ninf = _noise_vol(1)
    ap_ninf[8, 25, 34] = -np.inf
    small = np.random.default_rng(4).normal(10, 1, (5, 9, 9))
    small_nan = small.copy()
    small_nan[2, 4, 4] = np.nan                            # NaN at the aperture centre, shell entirely outside
    edge_vol = make_volume([((1.2, 24.4, 35.7), TRUE["sig"], 20.0), ((8.3, 24.4, 2.2), TRUE["sig"], 20.0)], bg=10.0, noise=1.0, seed=2)
    rng = np.random.default_rng(1)
    many = [(rng.integers(6, 11), rng.integers(15, 35), rng.integers(15, 55)) for _ in range(40)]
    bap, bsi, bso = (rescale_radii(r, VOX_BASE) for r in ((3.0, 5.0, 5.0), (5.0, 9.0, 9.0), (7.0, 13.0, 13.0)))
    rap, rsi, rso = (rescale_radii(r, VOX) for r in ((3.0, 5.0, 5.0), (5.0, 9.0, 9.0), (7.0, 13.0, 13.0)))
    out_shell = _noise_vol(9)
    out_shell[8, 25, 28] = 4000.0                          # bright voxel only in the shell: must be clipped away
    u16 = np.clip(np.rint(gvn), 0, 65535).astype(np.uint16)
    f32 = gvn.astype(np.float32)
    flatc = np.full(SHAPE, 10.0)
    # (name, image, centres, ap, shin, shout, kwargs, expect(port_result))
    return [
        ("noiseless", gv, c1, ap, si, so, {}, lambda m: abs(m["local_background"][0] - 10) < 0.1 and 0 < m["bg_noise_std"][0] < 0.2),   # Gaussian tails leave a small residual
        ("bg_noise", gvn, c1, ap, si, so, {}, lambda m: m["n_bg_voxels_valid"][0] > 2000 and not m["bg_insufficient"][0]),
        ("noise_only_many", _noise_vol(7), many, ap, si, so, {}, lambda m: (m["net_integrated"] < 0).any() and (m["net_integrated"] > 0).any()),
        ("quantised", quant, c1, ap, si, so, {}, lambda m: m["local_background_median"][0] in (11.0, 12.0)),
        ("mad_zero", mad0, c1, ap, si, so, {}, lambda m: m["mad_zero"][0] and np.isnan(m["snr_mad"][0]) and m["bg_noise_std"][0] > 0),
        ("flat_zero_noise", flatc, c1, ap, si, so, {}, lambda m: m["noise_zero"][0] and m["noise_unreliable"][0] and np.isnan(m["snr_primary"][0])),
        ("tiny_noise", tiny, c1, ap, si, so, {}, lambda m: m["bg_noise_std"][0] < 2e-3 and m["snr_peak_primary"][0] > 1000),
        ("shell_outside_image", small, [(2, 4, 4)], ap, si, so, {}, lambda m: m["bg_insufficient"][0] and np.isnan(m["net_integrated"][0]) and np.isfinite(m["aperture_sum"][0])),
        ("centre_outside_image", _noise_vol(2), [(8, 200, 36)], ap, si, so, {}, lambda m: m["n_aperture_vox"][0] == 0 and m["bg_insufficient"][0] and np.isnan(m["aperture_sum"][0]) and np.isnan(m["peak_raw"][0])),
        ("edges", edge_vol, [(1, 24, 36), (8, 24, 2)], ap, si, so, {}, lambda m: m["aperture_z_truncated"][0] and m["aperture_truncated"][1] and m["shell_truncated"][1]),
        ("no_occ", pair_vol, two, ap, si, so, {"occ": None}, lambda m: (m["n_shell_excluded"] == 0).all()),
        ("with_occ", pair_vol, two, ap, si, so, {"occ": occ_pair}, lambda m: (m["n_shell_excluded"] > 0).all()),
        ("occ_zero_voxels_ok", _noise_vol(2), c1, ap, si, so, {"occ": np.zeros(SHAPE, np.uint8)}, lambda m: m["n_shell_excluded"][0] == 0),
        ("shell_nonfinite", shell_nf, c1, ap, si, so, {}, lambda m: m["bg_nonfinite_voxels"][0] and m["n_shell_nonfinite"][0] == 2 and np.isfinite(m["snr_primary"][0])),
        ("aperture_nan", ap_nan, c1, ap, si, so, {}, lambda m: np.isnan(m["aperture_sum"][0]) and np.isnan(m["peak_raw"][0]) and np.isnan(m["net_integrated"][0]) and not m["bg_insufficient"][0]),
        ("aperture_posinf", ap_inf, c1, ap, si, so, {}, lambda m: np.isposinf(m["aperture_sum"][0]) and np.isposinf(m["peak_raw"][0]) and not m["bg_insufficient"][0]),
        ("aperture_neginf", ap_ninf, c1, ap, si, so, {}, lambda m: np.isneginf(m["aperture_sum"][0]) and not m["bg_insufficient"][0]),
        ("aperture_nan_insufficient_bg", small_nan,
         [(2, 4, 4)], ap, si, so, {}, lambda m: m["bg_insufficient"][0] and np.isnan(m["aperture_sum"][0])),
        ("clip_removes_bright_shell_voxel", out_shell, c1, ap, si, so, {}, lambda m: m["n_bg_voxels_valid"][0] < m["n_shell_vox"][0] and abs(m["local_background"][0] - 10) < 0.2),
        ("uint16_input", u16, c1, ap, si, so, {}, lambda m: m["n_bg_voxels_valid"][0] > 2000),
        ("float32_input", f32, c1, ap, si, so, {}, lambda m: m["n_bg_voxels_valid"][0] > 2000),
        ("baseline_voxel_radii", gvn, c1, bap, bsi, bso, {}, lambda m: m["n_aperture_vox_full"][0] > 0 and m["n_bg_voxels_valid"][0] > 1000),
        ("ref_voxel_radii", gvn, c1, rap, rsi, rso, {}, lambda m: m["n_aperture_vox_full"][0] > 0),
        ("fractional_centres", gvn, [(8.3, 24.4, 35.7), (7.5, 25.5, 30.5)], ap, si, so, {}, lambda m: np.isfinite(m["net_integrated"]).all()),
        ("min_shell_huge", gvn, c1, ap, si, so, {"min_shell": 10 ** 6}, lambda m: m["bg_insufficient"][0] and np.isnan(m["local_background"][0])),
        ("min_shell_zero", flatc, [(2, 4, 4)], ap, si, so, {"min_shell": 0}, lambda m: np.isfinite(m["local_background"][0]) or m["bg_insufficient"][0]),
        ("clip_sigma_1", gvn, c1, ap, si, so, {"clip_sigma": 1.0}, lambda m: m["n_bg_voxels_valid"][0] < m["n_shell_vox"][0]),
        ("negative_values_image", _noise_vol(2, bg=-5.0), c1, ap, si, so, {}, lambda m: m["local_background"][0] < 0),
    ]


def test_measure_spots_equivalence(frozen_modules, e_stats):
    L = frozen_modules["REF-PHOT"]
    problems, failed_expect = [], []
    for name, img, cs, ap, si, so, kw, expect in _meas_cases():
        img_f, img_p = img.copy(), img.copy()
        pr, a, b = _both(lambda *x, **k: L.measure_spots(img_f, *x, **k), lambda *x, **k: P.measure_spots(img_p, *x, **k),
                         (cs, ap, si, so), PHOT_SPEC, f"measure[{name}]", e_stats, kw)
        problems += pr
        assert np.array_equal(img_f, img_p, equal_nan=True)                                      # inputs not modified
        assert a[0] == "ok", (name, a)
        m = b[1]
        assert len(m) == 30 and all(len(v) == len(cs) for v in m.values())
        with np.errstate(all="ignore"):
            if not expect(m):
                failed_expect.append(name)
    assert not problems, "\n".join(problems[:10])
    assert not failed_expect, f"intended branch not triggered: {failed_expect}"


def test_measure_spots_empty_centres(frozen_modules, e_stats):
    L = frozen_modules["REF-PHOT"]
    img = _noise_vol(0)
    for label, cs in (("list", []), ("empty_array_2d", np.empty((0, 3))), ("tuple", ())):
        pr, a, b = _both(L.measure_spots, P.measure_spots, (img, cs, AP, SHIN, SHOUT), PHOT_SPEC, f"empty[{label}]", e_stats)
        assert not pr, pr
        assert a[0] == "ok" and all(len(v) == 0 for v in a[1].values()) and len(a[1]) == 30
    # float arrays are float64, flags are bool arrays (dtype compared by compare_value)
    r = P.measure_spots(img, [], AP, SHIN, SHOUT)
    assert r["aperture_sum"].dtype == np.float64 and r["noise_zero"].dtype == bool


def test_measure_spots_bad_inputs_same_outcome(frozen_modules):
    L = frozen_modules["REF-PHOT"]
    img = _noise_vol(0)
    cases = [
        ("centres_2_cols", (img, [(1, 2)], AP, SHIN, SHOUT), {}),
        ("image_2d", (img[0], [(1, 2, 3)], AP, SHIN, SHOUT), {}),
        ("ap_radii_zero", (img, [(8, 25, 35)], (0, 0, 0), SHIN, SHOUT), {}),
        ("shell_in_zero", (img, [(8, 25, 35)], AP, (0, 0, 0), SHOUT), {}),
        ("occ_wrong_shape", (img, [(8, 25, 35)], AP, SHIN, SHOUT), {"occ": np.zeros((2, 2, 2), np.uint8)}),
    ]
    for label, args, kw in cases:
        with np.errstate(all="ignore"):
            a, b = call_outcome(L.measure_spots, *args, **kw), call_outcome(P.measure_spots, *args, **kw)
        assert a[0] == b[0], (label, a[:2], b[:2])
        if a[0] == "raise":
            assert a == b, label
        else:
            assert not compare_dicts(a[1], b[1], PHOT_SPEC), label


def test_min_shell_boundary_99_100_101(frozen_modules, e_stats):
    """Retained-count semantics: ``min_shell`` is compared with the post-clip count ``n_bg_voxels_valid``."""
    L = frozen_modules["REF-PHOT"]
    img = np.full((40, 40, 40), 10.0)
    problems = []
    flags = {}
    for k, (ri, ro) in MINSHELL_GEOM.items():
        sh, _ = P.ellipsoid_voxels(C_FRAC, ro, img.shape, r_in=ri)
        assert len(sh[0]) == k, (k, len(sh[0]))                                    # geometry really gives k voxels
        pr, a, b = _both(L.measure_spots, P.measure_spots, (img, [C_FRAC], (1.0, 1.0, 1.0), ri, ro), PHOT_SPEC, f"min_shell[{k}]", e_stats)
        problems += pr
        m = b[1]
        assert m["n_shell_vox"][0] == m["n_bg_voxels_valid"][0] == k              # flat image: nothing clipped
        flags[k] = bool(m["bg_insufficient"][0])
    assert not problems, "\n".join(problems[:10])
    assert flags == {99: True, 100: False, 101: False}                              # default min_shell=100: < 100 is insufficient


def test_n_shell_vox_is_pre_clip_and_includes_nonfinite(frozen_modules, e_stats):
    """n_shell_vox counts voxels after occupancy exclusion but BEFORE nonfinite removal and clipping."""
    L = frozen_modules["REF-PHOT"]
    img = np.full((40, 40, 40), 10.0)
    ri, ro = MINSHELL_GEOM[100]
    sh, _ = P.ellipsoid_voxels(C_FRAC, ro, img.shape, r_in=ri)
    img[sh[0][0], sh[1][0], sh[2][0]] = np.nan                                       # one non-finite shell voxel -> 99 retained
    pr, a, b = _both(L.measure_spots, P.measure_spots, (img, [C_FRAC], (1.0, 1.0, 1.0), ri, ro), PHOT_SPEC, "nf99", e_stats)
    assert not pr, pr
    m = b[1]
    assert m["n_shell_vox"][0] == 100 and m["n_shell_nonfinite"][0] == 1 and m["n_bg_voxels_valid"][0] == 99
    assert m["bg_insufficient"][0] and m["bg_nonfinite_voxels"][0]
    # clipped outliers also separate the two counts (and can push the retained count under min_shell)
    img2 = np.full((40, 40, 40), 10.0)
    img2[sh[0][0], sh[1][0], sh[2][0]] = 5000.0
    pr, a, b = _both(L.measure_spots, P.measure_spots, (img2, [C_FRAC], (1.0, 1.0, 1.0), ri, ro), PHOT_SPEC, "clip99", e_stats)
    assert not pr, pr
    m = b[1]
    assert m["n_shell_vox"][0] == 100 and m["n_bg_voxels_valid"][0] == 99 and m["n_shell_nonfinite"][0] == 0 and m["bg_insufficient"][0]


# --------------------------------------------------------------------- nearest_neighbor_nm
def test_nearest_neighbor_equivalence(frozen_modules, e_stats):
    L = frozen_modules["REF-PHOT"]
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 50, (30, 3))
    cases = [
        ("zero_points", np.empty((0, 3)), VOX),
        ("one_point", np.array([[1.0, 2.0, 3.0]]), VOX),
        ("two_points", np.array([[8.0, 25.0, 30.0], [8.0, 25.0, 36.0]]), VOX),
        ("many_points", pts, VOX),
        ("many_baseline_voxel", pts, VOX_BASE),
        ("int_coords", np.array([[8, 25, 30], [8, 25, 36], [9, 25, 31]]), VOX),
        ("list_coords", [[8, 25, 30], [8, 25, 36]], VOX),
        ("duplicate_pair", np.array([[8.0, 25.0, 30.0], [8.0, 25.0, 30.0], [3.0, 3.0, 3.0]]), VOX),
        ("triple_duplicate", np.array([[1.0, 1.0, 1.0]] * 3 + [[9.0, 9.0, 9.0]]), VOX),
        ("equidistant_neighbours", np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, -2.0], [0.0, 2.0, 0.0]]), (1.0, 1.0, 1.0)),
        ("grid_ties", np.array([[z, y, x] for z in range(3) for y in range(3) for x in range(3)], float), (1.0, 1.0, 1.0)),
    ]
    problems, res = [], {}
    for name, c, vox in cases:
        pr, a, b = _both(L.nearest_neighbor_nm, P.nearest_neighbor_nm, (c, vox), NEIGHBOR_SPEC, f"nn[{name}]", e_stats)
        problems += pr
        res[name] = a[1]
    assert not problems, "\n".join(problems[:10])   # distance AND neighbour index j compared; ties are not skipped
    assert len(res["zero_points"][0]) == 0
    assert np.isnan(res["one_point"][0]).all() and list(res["one_point"][1]) == [-1]
    assert np.allclose(res["two_points"][0], 6 * VOX[1]) and list(res["two_points"][1]) == [1, 0]
    assert res["duplicate_pair"][0][0] == 0.0 and res["duplicate_pair"][0][1] == 0.0            # distance-0 tie branch reached
    assert (res["grid_ties"][0] == 1.0).all()                                                       # many equidistant neighbours


def test_nearest_neighbor_exceptions_match(frozen_modules):
    L = frozen_modules["REF-PHOT"]
    for label, args in (("bad_voxel_len", (np.zeros((3, 3)), (1.0, 1.0))), ("bad_cols", (np.zeros((3, 2)), (1.0, 1.0, 1.0)))):
        a, b = call_outcome(L.nearest_neighbor_nm, *args), call_outcome(P.nearest_neighbor_nm, *args)
        assert a[0] == b[0], label
        if a[0] == "raise":
            assert a == b, label


# --------------------------------------------------------------------- overlap_flags
def test_overlap_flags_equivalence(frozen_modules, e_stats):
    L = frozen_modules["REF-PHOT"]
    rng = np.random.default_rng(0)
    roi = (4, 7, 7)
    cases = [
        ("empty", np.empty((0, 3)), AP, roi),
        ("single", np.array([[8.0, 25.0, 35.0]]), AP, roi),
        ("pair_overlap", np.array([[8.0, 25.0, 30.0], [8.0, 25.0, 36.0]]), AP, roi),
        ("exactly_2x_aperture", np.array([[8.0, 25.0, 30.0], [8.0, 25.0, 40.0]]), AP, (4, 10, 10)),
        ("just_beyond_2x", np.array([[8.0, 25.0, 30.0], [8.0, 25.0, 40.0001]]), AP, (4, 10, 10)),
        ("roi_boundary_inclusive", np.array([[8.0, 25.0, 30.0], [12.0, 32.0, 37.0]]), AP, roi),
        ("roi_boundary_exceeded", np.array([[8.0, 25.0, 30.0], [12.0001, 32.0, 37.0]]), AP, roi),
        ("diagonal_2x_ellipsoid", np.array([[0.0, 0.0, 0.0], [6 / np.sqrt(3), 10 / np.sqrt(3), 10 / np.sqrt(3)]]), AP, roi),
        ("duplicate_points", np.array([[8.0, 25.0, 30.0], [8.0, 25.0, 30.0]]), AP, roi),
        ("random_25", rng.uniform((0, 0, 0), (17, 50, 70), (25, 3)), AP, roi),
        ("int_coords", np.array([[8, 25, 30], [8, 25, 36], [8, 40, 50]]), AP, roi),
        ("baseline_radii", np.array([[8.0, 25.0, 30.0], [8.0, 25.0, 40.0]]), rescale_radii((3.0, 5.0, 5.0), VOX_BASE), roi),
        ("list_coords", [[8, 25, 30], [8, 25, 36]], AP, roi),
    ]
    problems, res = [], {}
    for name, c, ap, rh in cases:
        pr, a, b = _both(L.overlap_flags, P.overlap_flags, (c, ap, rh), OVERLAP_SPEC, f"overlap[{name}]", e_stats)
        problems += pr
        res[name] = a[1]
    assert not problems, "\n".join(problems[:10])
    assert len(res["empty"][0]) == 0 and res["single"][0].tolist() == [False] and res["single"][1].tolist() == [0]
    assert res["pair_overlap"][0].all() and (res["pair_overlap"][1] == 1).all()
    assert res["exactly_2x_aperture"][0].all()                     # boundary d/(2 ar) == 1 counts as overlapping (<=)
    assert not res["just_beyond_2x"][0].any()
    assert (res["roi_boundary_inclusive"][1] == 1).all() and (res["roi_boundary_exceeded"][1] == 0).all()
    assert (res["duplicate_points"][1] == 1).all()
