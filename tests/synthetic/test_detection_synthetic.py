"""Synthetic tests for M2.2-B. Marked [T2] = ported assertions of frozen REF-T2; [F] = frozen semantics with an
independent oracle; [N] = new interface/guard (no frozen counterpart); [I] = integration with M2.0/2.1/2.2-A code.
"""

from __future__ import annotations

import warnings

import pytest

from scientific import require_modules

require_modules("numpy", "scipy", "skimage", "bigfish")

import numpy as np  # noqa: E402
import bigfish.detection as bfd  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

import catkit.cells as C  # noqa: E402
import catkit.detection as D  # noqa: E402
import catkit.localization as L  # noqa: E402
import catkit.photometry as P  # noqa: E402
import catkit.segmentation as S  # noqa: E402
from synth import rescale_radii, rescaled_cfg  # noqa: E402
from synth_det import OPTICS, SHAPE, VOX, make_channel  # noqa: E402

# ---------------------------------------------------------------- [T2] REF-T2 tests (fixed-threshold ones), same data/asserts
T2_SHAPE = (17, 160, 160)
KP = D.kernel_params((150.0, 43.3333, 43.3333), (410.0, 97.0, 97.0))


def _t2_synthetic(seed=0, bg=11.0, noise=1.05, n_spots=40):
    rng = np.random.default_rng(seed)
    vol = bg + rng.normal(0, noise, T2_SHAPE)
    zz, yy, xx = np.meshgrid(*[np.arange(n, dtype=float) for n in T2_SHAPE], indexing="ij")
    for _ in range(n_spots):
        c = (rng.uniform(4, 12), rng.uniform(15, 145), rng.uniform(15, 145))
        amp = rng.uniform(0.5, 9.0)
        vol += amp * np.exp(-0.5 * (((zz - c[0]) / 2.0) ** 2 + ((yy - c[1]) / 3.0) ** 2 + ((xx - c[2]) / 3.0) ** 2))
    return np.rint(vol).astype(np.float32)


def test_sigma_estimator_recovers_noise_level_of_clipped_log():
    rng = np.random.default_rng(1)
    img = (11.0 + rng.normal(0, 1.0, T2_SHAPE)).astype(np.float32)
    filt, lm = D.log_and_local_max(img, KP["log_kernel_sigma_px"], KP["minimum_distance_px"])
    sig = D.noise_sigma_from_positive_log(filt, np.ones(T2_SHAPE[1:], bool))
    raw_log = -ndi.gaussian_laplace(img.astype(np.float64), sigma=KP["log_kernel_sigma_px"])
    ref_scale = raw_log.std()
    assert 0.7 < sig / ref_scale < 1.4, (sig, ref_scale)
    assert (filt >= 0).all(), "Big-FISH log_filter clips negative responses to 0"


def test_fixed_thresholds_are_nested_and_in_log_units():
    img = _t2_synthetic()
    filt, lm = D.log_and_local_max(img, KP["log_kernel_sigma_px"], KP["minimum_distance_px"])
    sig = D.noise_sigma_from_positive_log(filt, np.ones(T2_SHAPE[1:], bool))
    spots, _ = bfd.spots_thresholding(filt, lm, 6 * sig, remove_duplicate=True)
    resp = filt[spots[:, 0], spots[:, 1], spots[:, 2]]
    sets = {k: {tuple(r) for r in spots[resp > k * sig]} for k in (6, 10, 12, 14)}
    assert sets[14] <= sets[12] <= sets[10] <= sets[6] and len(sets[6]) > len(sets[14]) > 0, {k: len(v) for k, v in sets.items()}
    raw_sd = float(img.std())
    assert abs(12 * sig - 12 * raw_sd) > 1e-3 and 12 * sig < 12 * raw_sd


# ---------------------------------------------------------------- [F] float32 comparison semantics (independent oracle)
def test_strict_float32_comparison_at_the_boundary():
    """The driver compares float32 ``resp / sigma`` with ``> k``. A float64 evaluation can give a different verdict on
    boundary values; the port must follow float32. Cases are found by search (not assumed) and the trigger is asserted."""
    rng = np.random.default_rng(0)
    differs = []
    for _ in range(20000):
        sigma = float(rng.uniform(0.2, 3.0))
        resp = np.float32(10.0 * sigma)                       # lands next to the boundary in float32
        for r in (resp, np.nextafter(resp, np.float32(0)), np.nextafter(resp, np.float32(np.inf))):
            v32 = np.float32(r) / np.float32(sigma)           # independent float32 oracle
            if (bool(v32 > np.float32(10.0))) != (float(r) / sigma > 10.0):
                differs.append((r, sigma))
                got = D.pass_flags(D.score_candidates(np.array([r], np.float32), sigma))[10][0]
                assert bool(got) == bool(v32 > np.float32(10.0)), (r, sigma)
        if len(differs) >= 5:
            break
    assert differs, "no float32/float64 disagreement found: boundary test would be vacuous"
    s = D.score_candidates(np.array([2.0, np.nextafter(np.float32(2.0), np.float32(9))], np.float32), 0.2)
    assert s.dtype == np.float32
    assert D.pass_flags(np.array([10.0], np.float32))[10].tolist() == [False]           # exactly k is NOT accepted (strict >)
    assert D.pass_flags(np.array([np.nextafter(np.float32(10.0), np.float32(11))], np.float32))[10].tolist() == [True]


def test_empty_pool_and_nonfinite_values():
    filt = np.zeros((9, 20, 20), np.float32)
    mlm = np.zeros(filt.shape, bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spots, resp = D.build_candidate_pool(filt, mlm, 1.0)
    assert spots.shape == (0, 3) and resp.shape == (0,) and resp.dtype == np.float32
    assert D.pass_flags(D.score_candidates(resp, 1.0))[10].shape == (0,)
    assert not D.pass_flags(np.array([np.nan], np.float32))[10][0]                     # NaN never passes


# ---------------------------------------------------------------- [F] pool = detector-returned rows (Revision 2)
def test_pool_uses_returned_coordinates_not_the_returned_mask():
    f = np.zeros((9, 21, 21), np.float32)
    m = np.zeros(f.shape, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if (dy, dx) != (0, 0):
                f[4, 10 + dy, 10 + dx] = 12
                m[4, 10 + dy, 10 + dx] = True            # a connected ring of 8 equal maxima around an empty centre
    spots, resp = D.build_candidate_pool(f, m, 1.0)      # floor 6
    _, det_mask = bfd.spots_thresholding(f, m, 6.0, remove_duplicate=True)
    assert spots.tolist() == [[4, 10, 10]] and resp.tolist() == [0.0]                  # one row; response AT the coordinate is 0
    assert det_mask.sum() > len(spots)                                                 # the detector's mask does not describe the pool
    los = D.score_candidates(resp, 1.0)
    assert not D.pass_flags(los)[6][0]                                                 # not >6 at its own coordinate, yet it IS a candidate


def test_plateau_centroid_is_truncated_toward_zero():
    f = np.zeros((9, 21, 21), np.float32)
    m = np.zeros(f.shape, bool)
    f[4, 5:7, 5:8] = 9.0
    m[4, 5:7, 5:8] = True
    spots, resp = D.build_candidate_pool(f, m, 1.0)
    assert spots.tolist() == [[4, 5, 6]] and spots.dtype == np.int64                   # centroid (4, 5.5, 6.0) -> int64 truncation


# ---------------------------------------------------------------- [F] edge flag, both sides
@pytest.mark.parametrize("Y,X", [(100, 100), (60, 120)])
def test_edge_flag_boundaries_hand_computed(Y, X):
    ys = np.array([9, 10, Y - 11, Y - 10, Y // 2, Y // 2, Y // 2, Y // 2])
    xs = np.array([X // 2, X // 2, X // 2, X // 2, 9, 10, X - 11, X - 10])
    spots = np.stack([np.full(8, 3), ys, xs], 1)
    assert D.edge_flags(spots, Y, X).tolist() == [True, False, False, True, True, False, False, True]


# ---------------------------------------------------------------- [N] guards of detect_fixed_sigma
def _ok_inputs():
    st = make_channel([(8, 30, 30, 200)])
    lab = np.zeros(SHAPE[1:], np.int32)
    lab[20:40, 20:40] = 1
    return st, lab, S.noise_mask(lab, 10)


def test_guards():
    st, lab, mask = _ok_inputs()
    call = lambda s=st, l=lab, m=mask, v=VOX, **kw: D.detect_fixed_sigma(s, l, m, v, OPTICS, **kw)   # noqa: E731
    call()
    for bad, match in ((st[:2], "at least 3"), (st[0], "3D"), (st.astype(np.float32), "uint16"), (st.astype(np.int32), "uint16")):
        with pytest.raises(ValueError, match=match):
            call(s=bad)
    for v in ((150.0, 43.3333333, 43.3333333), (150.0, 65.0, 66.0), (300.0, 65.0, 65.0), (150.0, 65.0)):
        with pytest.raises(ValueError, match="not supported in v0.1"):
            call(v=v)
    for k in (5.99, 0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="candidate floor"):
            call(k=k)
    assert call(k=6.0)[1]["k_sigma"] == 6.0
    with pytest.raises(ValueError, match="matching the stack"):
        call(l=lab[:10])
    with pytest.raises(ValueError, match="matching the stack"):
        call(m=mask.astype(np.uint8))


def test_invalid_noise_sigma_is_an_error_not_a_silent_nan():
    st, lab, mask = _ok_inputs()
    with pytest.raises(D.InvalidNoiseSigma, match="noise region is empty"):
        D.detect_fixed_sigma(st, lab, np.zeros_like(mask), VOX, OPTICS)                 # no pixel left for the noise estimate
    flat = np.zeros(SHAPE, np.uint16)                                                  # all-zero image: no positive LoG response -> sigma NaN
    with pytest.raises(D.InvalidNoiseSigma, match="sigma_LoG"):
        D.detect_fixed_sigma(flat, lab, mask, VOX, OPTICS)
    assert issubclass(D.InvalidNoiseSigma, ValueError)


def test_no_cells_is_allowed_whole_frame_is_noise_region():
    st = make_channel([(8, 30, 30, 200)])
    lab = np.zeros(SHAPE[1:], np.int32)
    t, qc = D.detect_fixed_sigma(st, lab, S.noise_mask(lab, 10), VOX, OPTICS, fov_id="f", channel="c")
    assert qc["background_fraction"] == 1.0 and qc["n_candidates"] >= 1 and (t["cell"] == 0).all()


# ---------------------------------------------------------------- [N] table, qc, validators
def test_detect_table_schema_and_qc():
    st, lab, mask = _ok_inputs()
    st = make_channel([(8, 30, 30, 200), (8, 60, 70, 120), (8, 50, 5, 150), (8, 10, 60, 60)])
    t, qc = D.detect_fixed_sigma(st, lab, mask, VOX, OPTICS, fov_id="fovA", channel="Cy5")
    D.validate_candidates(t)
    assert tuple(t) == D.CANDIDATE_COLUMNS
    assert t["log_response"].dtype == t["log_over_sigma"].dtype == np.float32 and t["accepted"].dtype == bool
    assert set(t["mode"]) == {"fixed_sigma"} and set(t["fov_id"]) == {"fovA"} and set(t["channel"]) == {"Cy5"}
    assert qc["n_candidates"] == len(t["spot_id"]) == 4 and qc["n_accepted"] == int(t["accepted"].sum()) <= qc["n_candidates"]
    assert sum(qc["band_counts"].values()) == qc["n_candidates"]                       # every candidate is in exactly one band (> 6)
    assert qc["threshold_value_log"] == t["candidate_floor_log"][0] == 6 * qc["sigma_log"]
    # coordinates are (z, y, x) integer voxel indices and within the stack; the planted rows are found at their positions
    assert t["z_init"].max() < SHAPE[0] and t["y_init"].max() < SHAPE[1] and t["x_init"].max() < SHAPE[2]
    found = {(int(z), int(y), int(x)) for z, y, x in zip(t["z_init"], t["y_init"], t["x_init"])}
    assert found == {(8, 30, 30), (8, 60, 70), (8, 50, 5), (8, 10, 60)}
    assert sorted(t["cell"].tolist()) == [0, 0, 0, 1]                                  # only the (30, 30) spot is inside cell 1
    assert t["edge_flag"][list(zip(t["y_init"], t["x_init"])).index((50, 5))]


def test_accepted_is_strictly_greater_than_k_for_a_candidate_exactly_at_k():
    st = make_channel([(8, 30, 30, 200), (8, 60, 70, 120), (8, 50, 20, 80)], seed=9)
    lab = np.zeros(SHAPE[1:], np.int32)
    mask = S.noise_mask(lab, 10)
    t, _ = D.detect_fixed_sigma(st, lab, mask, VOX, OPTICS)
    order = np.argsort(t["log_over_sigma"])
    mid = order[len(order) // 2]
    k = float(t["log_over_sigma"][mid])                    # exactly representable: the candidate sits ON the threshold
    t2, qc2 = D.detect_fixed_sigma(st, lab, mask, VOX, OPTICS, k=k)
    assert not t2["accepted"][mid] and (t2["accepted"] == (t["log_over_sigma"] > np.float32(k))).all()
    assert qc2["n_accepted"] == int((t["log_over_sigma"] > k).sum()) < len(order)


def test_thresholds_nested_through_the_table():
    st = make_channel([(8, 30, 30, 200), (8, 60, 70, 40), (8, 50, 20, 14), (8, 10, 60, 9)], noise=5.0, seed=2)
    lab = np.zeros(SHAPE[1:], np.int32)
    t, _ = D.detect_fixed_sigma(st, lab, S.noise_mask(lab, 10), VOX, OPTICS)
    sets = {k: {int(i) for i in t["spot_id"][t[f"pass_fixed_{k}sigma"]]} for k in D.KS}
    assert sets[14] <= sets[12] <= sets[10] <= sets[8] <= sets[6]
    assert set(t["spot_id"][t["accepted"]].tolist()) == sets[10]
    assert len(sets[6]) > len(sets[14])


def test_validators_reject_bad_tables():
    st, lab, mask = _ok_inputs()
    t, _ = D.detect_fixed_sigma(st, lab, mask, VOX, OPTICS, fov_id="f", channel="c")
    bad = dict(t)
    bad["spot_id"] = t["spot_id"][::-1] if len(t["spot_id"]) > 1 else t["spot_id"] + 1
    with pytest.raises(ValueError, match="1..N"):
        D.validate_candidates(bad)
    bad = dict(t, log_over_sigma=t["log_over_sigma"].astype(np.float64))
    with pytest.raises(ValueError, match="float32"):
        D.validate_candidates(bad)
    bad = dict(t, pass_fixed_10sigma=~t["pass_fixed_10sigma"])
    with pytest.raises(ValueError, match="pass_fixed_10sigma"):
        D.validate_candidates(bad)
    t2 = dict(t, mode=np.full(len(t["spot_id"]), "auto", dtype=object))
    with pytest.raises(ValueError, match="appears with modes"):
        D.validate_run_modes([t, t2])
    D.validate_run_modes([t, dict(t, channel=np.full(len(t["spot_id"]), "other", dtype=object), mode=t2["mode"])])   # other channel: fine


# ---------------------------------------------------------------- [I] vertical slice with A (segmentation) and M2.1 (photometry, fit)
def test_integration_segmentation_detection_photometry_localization():
    from synth_seg import make_dapi_stack
    dapi = make_dapi_stack(shape=(7, 100, 100), nuclei=[(30, 30, 9), (62, 72, 9)], seed=1)
    seg = S.segment_fov(dapi, 65.0)
    mask = S.noise_mask(seg.cell_labels, seg.seg_params["dil"])
    truth = [(8.3, 30.4, 29.7, 220.0), (8.0, 60.2, 72.0, 160.0), (8.0, 80.0, 15.0, 180.0)]
    stack = make_channel(truth, seed=7)
    table, qc = D.detect_fixed_sigma(stack, seg.cell_labels, mask, VOX, OPTICS, fov_id="f", channel="Cy5")
    D.validate_candidates(table)
    acc = np.flatnonzero(table["accepted"])
    assert len(acc) == 3
    init = np.stack([table["z_init"], table["y_init"], table["x_init"]], 1)[acc].astype(np.int64)      # (z, y, x) int64: the M2.1 input
    # cell assignment from A: first two planted spots in cells 1 and 2, third outside
    cid, _ = C.assign_cells_init(init, seg.cell_labels, 0.065)
    assert sorted(cid.tolist()) == [0, 1, 2] and np.array_equal(cid, table["cell"][acc])
    # M2.1 photometry and fit, driver call-site rescaling at the baseline voxel (replayed with test helpers)
    ap, shin, shout = (rescale_radii(r, VOX) for r in ((3., 5., 5.), (5., 9., 9.), (7., 13., 13.)))
    cfg = rescaled_cfg(L.DEFAULT_CFG, VOX)
    occ = P.build_occupancy(stack.shape, init, ap)
    prim = P.measure_spots(stack, init.astype(float), ap, shin, shout, occ=occ)
    sig0 = (2.7, 2.2, 2.2)
    for i in range(len(init)):
        fit = L.fit_spot(stack, init[i], sig0, prim["local_background"][i], prim["peak_net"][i], prim["bg_noise_std"][i], cfg, VOX)
        truth_i = min(truth, key=lambda t: sum((a - b) ** 2 for a, b in zip(t[:3], init[i])))
        assert fit["optimizer_converged"] and abs(fit["z_fit"] - truth_i[0]) < 1.0
        assert abs(fit["y_fit"] - truth_i[1]) < 0.5 and abs(fit["x_fit"] - truth_i[2]) < 0.5      # (z, y, x) order survived the chain
    assert (prim["net_integrated"] > 0).all() and qc["n_accepted"] == 3
