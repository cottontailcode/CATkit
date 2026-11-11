"""Synthetic-data tests for ``catkit.photometry``.

Part 1 re-states the photometry tests of the frozen synthetic script (same data,
same assertions and thresholds; assertion map in docs/ALGORITHMS.md). The frozen
script generates data with the localization model, so these show implementation
consistency on idealised data, NOT real-data photometric accuracy.

Part 2 adds checks whose expected values do NOT come from the functions under
test: voxel sets enumerated by an independent exact-rational brute force, and
hand-computed algebra / SNR.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from scientific import require_modules

require_modules("numpy", "scipy", "bigfish")   # localization (imported by synth) needs Big-FISH

import numpy as np  # noqa: E402

import catkit.localization as L  # noqa: E402
import catkit.photometry as I  # noqa: E402
from synth import SHAPE, TRUE, VOX, VOX_BASE, make_volume, rescale_radii  # noqa: E402


# ------------------------------------------------------------------ part 1: frozen-test port
def test_aperture_net_noiseless_and_units():
    vol = make_volume([(TRUE["mu"], TRUE["sig"], TRUE["A"])], bg=TRUE["bg"])
    ap, sh_in, sh_out = (3, 5, 5), (5, 9, 9), (7, 13, 13)
    c = [TRUE["mu"]]
    m = I.measure_spots(vol, c, ap, sh_in, sh_out)
    idx, _ = I.ellipsoid_voxels(c[0], ap, vol.shape)
    expect = float((vol[idx] - TRUE["bg"]).sum())
    assert abs(m["local_background"][0] - TRUE["bg"]) < 0.1, m["local_background"][0]
    assert abs(m["net_integrated"][0] - expect) / expect < 0.02, (m["net_integrated"][0], expect)
    assert m["n_aperture_vox"][0] == len(idx[0]) == m["n_aperture_vox_full"][0]
    assert not m["aperture_truncated"][0] and not m["aperture_z_truncated"][0]


def test_background_only_not_clipped_and_unbiased():
    vol = make_volume([], bg=10.0, noise=1.0, seed=7)
    rng = np.random.default_rng(1)
    cs = [(rng.integers(6, 11), rng.integers(15, 35), rng.integers(15, 55)) for _ in range(200)]
    m = I.measure_spots(vol, cs, (3, 5, 5), (5, 9, 9), (7, 13, 13))
    net = m["net_integrated"]
    assert np.isfinite(net).all() and (net < 0).any() and (net > 0).any(), "negative net must be kept (no clipping)"
    assert abs(net.mean()) < 3 * net.std() / np.sqrt(len(net)) + 1.0, net.mean()
    assert np.isfinite(m["snr_primary"]).all()


def test_background_mean_not_quantised():
    rng = np.random.default_rng(5)
    vol = np.clip(np.rint(11.6 + rng.normal(0, 1.1, SHAPE)), 0, 65535)      # true mean 11.6 counts, integer-quantised
    m = I.measure_spots(vol, [(8, 25, 35)], (3, 5, 5), (5, 9, 9), (7, 13, 13))
    assert abs(m["local_background"][0] - 11.6) < 0.1 and m["local_background_median"][0] in (11.0, 12.0)


def test_primary_noise_is_clipped_sample_std_ddof1():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    r = I.clipped_background(v)
    assert np.isclose(r["noise_std"], np.std(v, ddof=1)) and r["n_kept"] == 4 and np.isclose(r["mean"], 2.5)
    # clipping: an outlier is removed first and the std is computed on the RETAINED voxels only
    rng = np.random.default_rng(0)
    w = np.concatenate([rng.normal(11, 1.1, 3000), [60.0, 70.0]])
    r = I.clipped_background(w)
    kept = w[np.abs(w - np.median(w)) <= 3 * w.std()]            # first clip pass already removes 60/70; later passes only trim >3 sd tails
    assert r["n_kept"] <= 3000 and 1.0 < r["noise_std"] < 1.25 and abs(r["mean"] - 11) < 0.1
    assert r["noise_std"] < np.std(w, ddof=1)


def test_snr_primary_uses_std_and_mad_is_secondary():
    rng = np.random.default_rng(2)
    vol = np.clip(np.rint(11.6 + rng.normal(0, 1.1, SHAPE)), 0, 65535)    # integer-quantised low-count background
    vol += L.gaussian_model((8.0, 25.0, 35.0, 2.0, 3.0, 2.4, 15.0, 0.0), *(np.arange(n, dtype=float) for n in SHAPE))
    m = I.measure_spots(vol, [(8, 25, 35)], (3, 5, 5), (5, 9, 9), (7, 13, 13))
    nvox, net = m["n_aperture_vox"][0], m["net_integrated"][0]
    assert np.isclose(m["snr_primary"][0], net / (m["bg_noise_std"][0] * np.sqrt(nvox)))
    assert np.isclose(m["snr_mad"][0], net / (m["bg_noise_mad"][0] * np.sqrt(nvox)))
    assert np.isclose(m["snr_peak_primary"][0], m["peak_net"][0] / m["bg_noise_std"][0])
    assert 1.0 < m["bg_noise_std"][0] < 1.3 and m["bg_noise_mad"][0] > m["bg_noise_std"][0]   # MAD of integers overestimates (1.4826)
    assert m["n_bg_voxels_valid"][0] > 2500 and not m["bg_insufficient"][0] and not m["noise_zero"][0]


def test_mad_zero_low_count_background():
    base = np.full(SHAPE, 10.0)
    base[np.random.default_rng(0).random(SHAPE) < 0.03] = 11.0          # >50% identical -> MAD == 0, std > 0
    m = I.measure_spots(base, [(8, 25, 35)], (3, 5, 5), (5, 9, 9), (7, 13, 13))
    assert m["bg_noise_mad"][0] == 0 and m["mad_zero"][0]
    assert np.isnan(m["snr_mad"][0]) and np.isnan(m["snr_peak_mad"][0])           # no fallback for the secondary diagnostic
    assert m["bg_noise_std"][0] > 0 and np.isfinite(m["snr_primary"][0]) and np.isfinite(m["snr_peak_primary"][0]) and not m["noise_unreliable"][0]
    assert np.isclose(m["bg_noise_legacy"][0], m["bg_noise_std"][0])             # Phase-2 rule would have fallen back to std here


def test_degenerate_flat_background_no_floor():
    flat = np.full(SHAPE, 10.0)                                           # zero variance: noise stays 0 / SNR NaN, never a tiny floor
    m = I.measure_spots(flat, [(8, 25, 35)], (3, 5, 5), (5, 9, 9), (7, 13, 13))
    assert m["bg_noise_std"][0] == 0 and m["noise_zero"][0] and m["noise_unreliable"][0]
    assert np.isnan(m["snr_primary"][0]) and np.isnan(m["snr_peak_primary"][0]) and np.isnan(m["snr_mad"][0]) and np.isnan(m["snr_peak_legacy_fitqc"][0])
    assert np.isfinite(m["net_integrated"][0])                            # intensity itself is still reported


def test_tiny_noise_not_floored():
    rng = np.random.default_rng(3)
    vol = 10.0 + rng.normal(0, 1e-3, SHAPE)                               # very low (non-integer) noise: SNR must scale up, not be capped
    vol[8, 25, 35] += 5.0
    m = I.measure_spots(vol, [(8, 25, 35)], (3, 5, 5), (5, 9, 9), (7, 13, 13))
    assert m["bg_noise_std"][0] < 2e-3 and m["snr_peak_primary"][0] > 1000


def test_insufficient_background_samples_flagged():
    small = np.random.default_rng(4).normal(10, 1, (5, 9, 9))             # shell entirely outside the image -> 0 background voxels
    m = I.measure_spots(small, [(2, 4, 4)], (3, 5, 5), (5, 9, 9), (7, 13, 13))
    assert m["bg_insufficient"][0] and m["bg_unreliable"][0] and m["noise_unreliable"][0]
    assert np.isnan(m["snr_primary"][0]) and np.isnan(m["net_integrated"][0]) and np.isfinite(m["aperture_sum"][0])


def test_nonfinite_background_voxels_dropped_and_flagged():
    vol = np.random.default_rng(5).normal(10, 1, SHAPE)
    vol[8, 25, 45] = np.nan; vol[8, 25, 22] = np.inf                     # inside the shell of the spot at (8,25,35)
    m = I.measure_spots(vol, [(8, 25, 35)], (3, 5, 5), (5, 9, 9), (7, 13, 13))
    assert m["bg_nonfinite_voxels"][0] and m["n_shell_nonfinite"][0] == 2
    assert np.isfinite(m["local_background"][0]) and np.isfinite(m["bg_noise_std"][0]) and np.isfinite(m["snr_primary"][0])


def test_edges_flagged_not_dropped_measure_part():
    # measure half of the frozen "edges_flagged_not_dropped"; the fit half lives in test_localization_synthetic.py
    vol = make_volume([((1.2, 24.4, 35.7), TRUE["sig"], 20.0), ((8.3, 24.4, 2.2), TRUE["sig"], 20.0)], bg=10.0, noise=1.0, seed=2)
    m = I.measure_spots(vol, [(1, 24, 36), (8, 24, 2)], (3, 5, 5), (5, 9, 9), (7, 13, 13))
    assert m["aperture_z_truncated"][0] and m["aperture_truncated"][0] and m["aperture_truncated"][1] and m["shell_truncated"][1]
    assert np.isfinite(m["net_integrated"]).all()                     # still measured


def test_neighbors_flagged_and_excluded_from_shell():
    c = [(8.0, 25.0, 30.0), (8.0, 25.0, 36.0)]
    vol = make_volume([(c[0], TRUE["sig"], 20.0), (c[1], TRUE["sig"], 20.0)], bg=10.0)
    ap_ov, nb = I.overlap_flags(c, (3, 5, 5), (4, 7, 7))
    assert ap_ov.all() and (nb == 1).all()
    occ = I.build_occupancy(vol.shape, c, (3, 5, 5))
    a = I.measure_spots(vol, c, (3, 5, 5), (5, 9, 9), (7, 13, 13), occ=None)
    b = I.measure_spots(vol, c, (3, 5, 5), (5, 9, 9), (7, 13, 13), occ=occ)
    assert (b["n_shell_excluded"] > 0).all() and (b["n_shell_vox"] < a["n_shell_vox"]).all()
    d, j = I.nearest_neighbor_nm(c, VOX)
    assert np.allclose(d, 6 * VOX[1]) and list(j) == [1, 0]


# ------------------------------------------------------------------ part 2: independent expectations
def _brute_voxels(center, r_out, shape, r_in=None):
    """Exact-rational enumeration of the same definition (d^2 <= 1, annulus d_in^2 > 1) over the whole image."""
    c = [Fraction(float(v)) for v in center]
    ro = [Fraction(float(v)) for v in r_out]
    ri = None if r_in is None else [Fraction(float(v)) for v in r_in]
    out = set()
    for z in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                p = (z, y, x)
                d2 = sum(((p[k] - c[k]) / ro[k]) ** 2 for k in range(3))
                if d2 > 1:
                    continue
                if ri is not None and sum(((p[k] - c[k]) / ri[k]) ** 2 for k in range(3)) <= 1:
                    continue
                out.add(p)
    return out


SMALL = (9, 14, 14)


@pytest.mark.parametrize("name,center,ro,ri", [
    ("integer_centre", (4, 7, 7), (2, 3, 3), None),
    ("axis_voxels_on_boundary", (4, 7, 7), (3, 5, 5), None),          # d^2 == 1 exactly on the axes
    ("fractional_centre", (4.25, 6.5, 7.75), (2.5, 3.5, 3.5), None),
    ("annulus", (4, 7, 7), (4, 6, 6), (2, 3, 3)),
    ("annulus_fractional", (4.5, 6.25, 7.5), (4, 6, 6), (2.5, 3.5, 3.5)),
    ("clipped_by_image", (0.5, 1.0, 12.5), (3, 5, 5), None),
])
def test_ellipsoid_voxel_set_matches_exact_enumeration(name, center, ro, ri):
    idx, n_full = I.ellipsoid_voxels(center, ro, SMALL, r_in=ri)
    got = set(zip(*(a.tolist() for a in idx)))
    want = _brute_voxels(center, ro, SMALL, ri)
    assert got == want, (name, sorted(got ^ want)[:5])
    assert len(got) == len(idx[0])                                    # no duplicate indices
    assert n_full >= len(got)


def test_n_aperture_vox_full_counts_voxels_outside_the_image():
    # integer centre at the image corner: n_full counts the whole ellipsoid (translation-invariant for an integer centre),
    # the returned indices only the in-image part
    idx, n_full = I.ellipsoid_voxels((0, 0, 0), (2, 3, 3), SMALL)
    whole = _brute_voxels((10, 10, 10), (2, 3, 3), (21, 21, 21))
    inside = _brute_voxels((0, 0, 0), (2, 3, 3), SMALL)
    assert n_full == len(whole) and len(idx[0]) == len(inside) < n_full


@pytest.mark.parametrize("voxel", [VOX, VOX_BASE])
def test_rescaled_aperture_voxel_count_matches_exact_enumeration(voxel):
    ap = rescale_radii((3.0, 5.0, 5.0), voxel)
    shape = (21, 40, 40)
    idx, n_full = I.ellipsoid_voxels((10.0, 20.0, 20.0), ap, shape)
    want = _brute_voxels((10.0, 20.0, 20.0), ap, shape)
    assert set(zip(*(a.tolist() for a in idx))) == want and n_full == len(want)


def test_baseline_voxel_scaled_radii_are_fractional_and_documented():
    ap = rescale_radii((3.0, 5.0, 5.0), VOX_BASE)
    assert ap[0] == pytest.approx(3.0) and ap[1] == pytest.approx(5.0 * 43.3333333 / 65.0)    # 3.333... px, not an integer
    sh_in, sh_out = rescale_radii((5.0, 9.0, 9.0), VOX_BASE), rescale_radii((7.0, 13.0, 13.0), VOX_BASE)
    # shell axis ratios differ between inner and outer ellipsoid: the shell is NOT a single distance threshold
    assert sh_in[0] / sh_in[1] != pytest.approx(sh_out[0] / sh_out[1])


def _brute_idx(center, r_out, shape, r_in=None):
    """Exact-rational voxel set as index arrays (sorted), independent of the tested ellipsoid_voxels."""
    pts = sorted(_brute_voxels(center, r_out, shape, r_in))
    return tuple(np.array([p[k] for p in pts], dtype=np.intp) for k in range(3))


def _boundary_gap(center, r, shape):
    """Smallest exact |d^2 - 1| over off-axis voxels. Voxels exactly ON the ellipsoid surface (e.g. the 5-12-13 triple)
    are decided by float rounding in the frozen code, so hand-computed expectations must avoid them (gap >> 1e-12)."""
    c = [Fraction(float(v)) for v in center]
    rr = [Fraction(float(v)) for v in r]
    g = None
    for p in np.ndindex(*shape):
        if sum(p[k] != c[k] for k in range(3)) < 2:
            continue
        d2 = sum(((p[k] - c[k]) / rr[k]) ** 2 for k in range(3))
        if d2 < 3:
            g = abs(d2 - 1) if g is None else min(g, abs(d2 - 1))
    return float(g)


# radii chosen so that no voxel lies on (or within 1e-4 of) an ellipsoid surface; (3,5,5) / (7,13,13) have exact 3-4-5 / 5-12-13 surface voxels
HAND_AP, HAND_SI, HAND_SO = (3, 4.6, 4.6), (5, 8.6, 8.6), (7, 12.6, 12.6)


def test_net_integrated_algebra_and_snr_by_hand():
    """flat background 10, +5 on exactly one aperture voxel and a tiny deterministic jitter in the shell."""
    img = np.full((17, 40, 40), 10.0)
    ap, si, so = HAND_AP, HAND_SI, HAND_SO
    c = (8, 20, 20)
    assert min(_boundary_gap(c, r, img.shape) for r in (ap, si, so)) > 1e-4   # no surface-ambiguous voxels
    sh = _brute_idx(c, so, img.shape, si)                              # expected shell set: exact-rational oracle, not the port
    n = len(_brute_voxels(c, ap, img.shape))                           # expected aperture size, same oracle
    assert len(sh[0]) % 2 == 0
    jit = np.where(np.arange(len(sh[0])) % 2 == 0, 1.0, -1.0)          # shell values 9/11, mean exactly 10
    img[sh] += jit
    img[8, 20, 20] += 5.0
    m = I.measure_spots(img, [c], ap, si, so)
    assert int(m["n_aperture_vox"][0]) == n
    # by hand: the aperture holds n voxels of value 10 plus 5 extra counts
    assert m["aperture_sum"][0] == 10.0 * n + 5.0
    assert m["local_background"][0] == pytest.approx(10.0, abs=1e-12) and m["n_bg_voxels_valid"][0] == len(sh[0])
    assert m["net_integrated"][0] == pytest.approx(5.0, abs=1e-9)
    assert m["peak_raw"][0] == 15.0 and m["peak_net"][0] == pytest.approx(5.0, abs=1e-12)
    v = img[sh]
    sd = np.sqrt(((v - v.mean()) ** 2).sum() / (len(v) - 1))          # ddof=1, independent of numpy's ddof argument
    assert m["bg_noise_std"][0] == pytest.approx(sd, rel=1e-12)
    assert m["snr_primary"][0] == pytest.approx(5.0 / (sd * np.sqrt(n)), rel=1e-9)
    assert m["snr_peak_primary"][0] == pytest.approx(5.0 / sd, rel=1e-9)


def test_negative_net_and_no_noise_floor_kept():
    img = np.full((17, 40, 40), 10.0)
    c = (8, 20, 20)
    ap, si, so = HAND_AP, HAND_SI, HAND_SO
    assert min(_boundary_gap(c, r, img.shape) for r in (ap, si, so)) > 1e-4
    sh = _brute_idx(c, so, img.shape, si)
    img[sh] += np.where(np.arange(len(sh[0])) % 2 == 0, 1e-6, -1e-6)
    idx = _brute_idx(c, ap, img.shape)
    img[idx] -= 2.0                                                   # aperture darker than the shell
    m = I.measure_spots(img, [c], ap, si, so)
    assert m["net_integrated"][0] < 0 and m["peak_net"][0] < 0       # not clipped at zero
    assert 0 < m["bg_noise_std"][0] < 2e-6 and m["snr_primary"][0] < -1e5   # tiny noise -> huge |SNR|, never floored


def test_aperture_nonfinite_propagates_but_shell_nonfinite_is_dropped():
    rng = np.random.default_rng(1)
    base = 10.0 + rng.normal(0, 1, SHAPE)
    c = [(8, 25, 35)]
    args = (c, (3, 5, 5), (5, 9, 9), (7, 13, 13))
    nan_ap = base.copy(); nan_ap[8, 25, 35] = np.nan
    inf_ap = base.copy(); inf_ap[8, 25, 36] = np.inf
    m_nan, m_inf = I.measure_spots(nan_ap, *args), I.measure_spots(inf_ap, *args)
    assert np.isnan(m_nan["aperture_sum"][0]) and np.isnan(m_nan["net_integrated"][0]) and np.isnan(m_nan["snr_primary"][0])
    assert np.isnan(m_nan["peak_raw"][0])
    assert np.isposinf(m_inf["aperture_sum"][0]) and np.isposinf(m_inf["peak_net"][0]) and np.isposinf(m_inf["snr_primary"][0])
    assert not m_nan["bg_nonfinite_voxels"][0] and not m_inf["bg_nonfinite_voxels"][0]      # aperture voxels are not counted as shell problems
    # -Inf in the aperture: sum-based quantities propagate -Inf, the max-based ones stay finite (peak_raw = max of the rest)
    neg = base.copy(); neg[8, 25, 36] = -np.inf
    m_neg = I.measure_spots(neg, *args)
    assert np.isneginf(m_neg["aperture_sum"][0]) and np.isneginf(m_neg["net_integrated"][0]) and np.isneginf(m_neg["snr_primary"][0])
    assert np.isfinite(m_neg["peak_raw"][0]) and np.isfinite(m_neg["peak_net"][0]) and np.isfinite(m_neg["snr_peak_primary"][0])
    # not "all SNR become Inf": an insufficient-background spot keeps NaN SNR whatever the aperture holds
    small = np.random.default_rng(4).normal(10, 1, (5, 9, 9)); small[2, 4, 5] = np.inf
    m = I.measure_spots(small, [(2, 4, 4)], (3, 5, 5), (5, 9, 9), (7, 13, 13))
    assert m["bg_insufficient"][0] and np.isnan(m["snr_primary"][0]) and np.isposinf(m["aperture_sum"][0])


def test_min_shell_compares_retained_voxels_not_raw_shell_size():
    img = np.full((40, 40, 40), 10.0)
    c = (20.3, 20.2, 20.1)
    ri, ro = (6.25, 2.25, 2.25), (6.75, 3.0, 2.75)                    # 100 shell voxels in a flat image
    sh = _brute_idx(c, ro, img.shape, ri)
    assert len(sh[0]) == 100
    ok = I.measure_spots(img, [c], (1.0, 1.0, 1.0), ri, ro)
    assert ok["n_shell_vox"][0] == ok["n_bg_voxels_valid"][0] == 100 and not ok["bg_insufficient"][0]
    img[sh[0][0], sh[1][0], sh[2][0]] = 5000.0                         # one outlier is clipped away: 99 retained of 100
    low = I.measure_spots(img, [c], (1.0, 1.0, 1.0), ri, ro)
    assert low["n_shell_vox"][0] == 100 and low["n_bg_voxels_valid"][0] == 99 and low["bg_insufficient"][0]
    assert np.isnan(low["local_background"][0]) and np.isnan(low["net_integrated"][0])


def test_occupancy_counts_overlaps_and_uint8_wrap_is_frozen_behaviour():
    two = [(8.0, 25.0, 30.0), (8.0, 25.0, 33.0)]
    occ = I.build_occupancy(SHAPE, two, (3, 5, 5))
    assert occ.dtype == np.uint8 and occ.max() == 2
    a = _brute_voxels(two[0], (3, 5, 5), SHAPE)
    b = _brute_voxels(two[1], (3, 5, 5), SHAPE)
    assert {tuple(p) for p in np.argwhere(occ == 2)} == a & b
    assert {tuple(p) for p in np.argwhere(occ >= 1)} == a | b
    # known frozen limitation (documented, NOT fixed): 256 identical apertures wrap the uint8 counter to 0
    for n, expect in ((255, 255), (256, 0)):
        o = I.build_occupancy(SHAPE, [(8, 25, 35)] * n, (1, 1, 1))
        assert o[8, 25, 35] == expect
