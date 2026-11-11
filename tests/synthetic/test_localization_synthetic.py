"""Synthetic-data tests for ``catkit.localization``.

Part 1 re-states the localization tests of the frozen synthetic script (same
data, same assertions and thresholds; see docs/ALGORITHMS.md for the assertion
map). They use the model under test to generate data, so they show
implementation consistency on idealised data, NOT real-data accuracy.

Part 2 adds checks that do NOT depend on the ``erf`` implementation: an
independent midpoint-rule integration with a fixed error budget.
"""

from __future__ import annotations

import pytest

from scientific import require_modules

require_modules("numpy", "scipy", "bigfish")

import numpy as np  # noqa: E402

import catkit.localization as L  # noqa: E402
from synth import SIG0, SHAPE, TRUE, VOX, VOX_BASE, make_volume, rescaled_cfg  # noqa: E402

CFG = dict(L.DEFAULT_CFG)
S0 = SIG0


# ------------------------------------------------------------------ part 1: frozen-test port
def test_axis_order_and_noiseless_recovery():
    vol = make_volume([(TRUE["mu"], TRUE["sig"], TRUE["A"])], bg=TRUE["bg"])
    r = L.fit_spot(vol, (8, 24, 36), S0, 10.0, 15.0, np.nan, CFG, VOX)
    assert r["optimizer_converged"], r
    assert np.allclose([r["z_fit"], r["y_fit"], r["x_fit"]], TRUE["mu"], atol=1e-3), (r["z_fit"], r["y_fit"], r["x_fit"])
    assert np.allclose([r["sigma_z_fit"], r["sigma_y_fit"], r["sigma_x_fit"]], TRUE["sig"], atol=1e-3)
    assert abs(r["A_fit"] - 20) < 1e-3 and abs(r["bg_fit"] - 10) < 1e-3
    # integrated intensity formula
    assert np.isclose(L.gaussian_integral_voxels(20, 2.0, 3.0, 2.4), 20 * (2 * np.pi) ** 1.5 * 2.0 * 3.0 * 2.4)


def test_noisy_recovery_statistics():
    errs, sigs, amps = [], [], []
    for seed in range(40):
        vol = make_volume([(TRUE["mu"], TRUE["sig"], TRUE["A"])], bg=TRUE["bg"], noise=1.0, seed=seed, integer=True)
        r = L.fit_spot(vol, (8, 24, 36), S0, 10.0, 15.0, 1.0, CFG, VOX)
        assert r["optimizer_converged"]
        errs.append([r["z_fit"] - 8.3, r["y_fit"] - 24.4, r["x_fit"] - 35.7])
        sigs.append([r["sigma_z_fit"], r["sigma_y_fit"], r["sigma_x_fit"]])
        amps.append(r["A_fit"])
    errs, sigs = np.array(errs), np.array(sigs)
    assert np.all(np.abs(errs.mean(0)) < [0.15, 0.1, 0.1]), errs.mean(0)       # unbiased
    assert np.all(errs.std(0) < [0.35, 0.2, 0.2]), errs.std(0)                  # precision at SNR~20
    assert np.allclose(np.median(sigs, 0), TRUE["sig"], rtol=0.12), np.median(sigs, 0)
    assert abs(np.median(amps) - 20) < 2.0


def test_uncertainty_reported_when_wellconditioned():
    vol = make_volume([(TRUE["mu"], TRUE["sig"], TRUE["A"])], bg=10.0, noise=1.0, seed=3)
    r = L.fit_spot(vol, (8, 24, 36), S0, 10.0, 15.0, 1.0, CFG, VOX)
    assert r["se_reported"] and 0 < r["se_y_px"] < 0.5 and r["jac_cond"] < CFG["cond_max"], r["jac_cond"]


def test_edges_flagged_not_dropped_fit_part():
    # fit half of the frozen "edges_flagged_not_dropped"; the photometry half is ported in checkpoint B
    vol = make_volume([((1.2, 24.4, 35.7), TRUE["sig"], 20.0), ((8.3, 24.4, 2.2), TRUE["sig"], 20.0)], bg=10.0, noise=1.0, seed=2)
    rz = L.fit_spot(vol, (1, 24, 36), S0, 10.0, 15.0, 1.0, CFG, VOX)
    rx = L.fit_spot(vol, (8, 24, 2), S0, 10.0, 15.0, 1.0, CFG, VOX)
    assert rz["roi_z_truncated"] and rx["roi_xy_truncated"] and not rz["roi_xy_truncated"]


def test_deterministic_failures_preserved():
    vol = make_volume([(TRUE["mu"], TRUE["sig"], TRUE["A"])], bg=10.0, noise=1.0, seed=5)
    # (a) NaN inside ROI
    v2 = vol.copy(); v2[8, 24, 36] = np.nan
    r = L.fit_spot(v2, (8, 24, 36), S0, 10.0, 15.0, 1.0, CFG, VOX)
    assert r["fit_error"] == "nonfinite_roi" and not r["optimizer_converged"] and np.isnan(r["z_fit"])
    # (b) ROI below the minimum voxel count
    r = L.fit_spot(vol, (8, 24, 36), S0, 10.0, 15.0, 1.0, dict(CFG, min_roi_voxels=10 ** 6), VOX)
    assert r["fit_error"] == "roi_too_small"
    # (c) impossible bounds (lower > upper) -> optimiser raises; recorded, never propagated
    r = L.fit_spot(vol, (8, 24, 36), S0, 10.0, 15.0, 1.0, dict(CFG, sigma_xy_bounds_nm=(400.0, 60.0)), VOX)
    assert r["fit_error"] and not r["optimizer_converged"]
    q, why = L.classify_quality(r, np.nan, np.nan, np.nan, 0, CFG)
    assert q == "unreliable" and "fit_error" in why
    # (d) ROI far outside the image content: init beyond the array -> empty ROI recorded as failure
    r = L.fit_spot(vol, (8, 200, 36), S0, 10.0, 15.0, 1.0, CFG, VOX)
    assert r["fit_error"] and not r["optimizer_converged"]


def test_convergence_is_not_quality():
    vol = make_volume([], bg=10.0, noise=1.0, seed=11)               # noise only: optimiser may converge, quality must not be 'good'
    r = L.fit_spot(vol, (8, 24, 36), S0, 10.0, 2.0, 1.0, CFG, VOX)
    q, why = L.classify_quality(r, 0.5, 0.5, 1.0, 0, CFG)
    assert q != "good" and "low_snr_peak" in why, (q, why)


def test_bigfish_fit_subpixel_wrapper_axes():
    vol = make_volume([(TRUE["mu"], TRUE["sig"], TRUE["A"])], bg=TRUE["bg"], noise=0.5, seed=4, integer=True).astype(np.uint16)
    out, unchanged, err = L.run_bigfish_fit_subpixel(vol, [(8, 24, 36)], VOX, (410.0, 97.0, 97.0))
    assert not err[0] and np.allclose(out[0], TRUE["mu"], atol=0.5), out   # (z,y,x) order, voxel units
    assert out.shape == (1, 3)


# ------------------------------------------------------------------ part 2: additional checks
def test_default_cfg_values_and_not_polluted():
    assert L.DEFAULT_CFG == dict(
        roi_half_z=4, roi_half_xy=7, mu_bound_z=2.0, mu_bound_xy=2.0, sigma_xy_bounds_nm=(60.0, 400.0),
        sigma_z_bounds_nm=(150.0, 1000.0), A_max=1.0e5, bg_max=1.0e3, max_nfev=300, min_roi_voxels=60,
        cond_max=1.0e6, large_shift_xy_px=1.5, large_shift_z_slices=1.5, low_snr_peak=3.0)
    assert L.PNAMES == ("mu_z", "mu_y", "mu_x", "sigma_z", "sigma_y", "sigma_x", "A", "bg")


def test_baseline_voxel_recovery_and_sigma_bounds_in_index_units():
    cfg = rescaled_cfg(L.DEFAULT_CFG, VOX_BASE)                 # caller-side replay of the driver rescaling
    assert cfg["roi_half_xy"] == 5 and cfg["mu_bound_xy"] == pytest.approx(2.0 * 43.3333333 / 65.0)
    vol = make_volume([(TRUE["mu"], TRUE["sig"], TRUE["A"])], bg=TRUE["bg"])
    r = L.fit_spot(vol, (8, 24, 36), S0, 10.0, 15.0, np.nan, cfg, VOX_BASE)
    assert r["optimizer_converged"] and r["params_at_bound"] == ""
    assert np.allclose([r["z_fit"], r["y_fit"], r["x_fit"]], TRUE["mu"], atol=1e-3)
    assert np.allclose([r["sigma_z_fit"], r["sigma_y_fit"], r["sigma_x_fit"]], TRUE["sig"], atol=1e-3)
    # same nm limits, different index limits: sigma_xy in [60, 400] nm -> [0.923, 6.154] px at 65 nm, [1.385, 9.231] px at 43.333 nm
    assert 60.0 / 65.0 == pytest.approx(0.9231, abs=1e-4) and 400.0 / 65.0 == pytest.approx(6.1538, abs=1e-4)
    assert 60.0 / VOX[1] == pytest.approx(1.3846, abs=1e-4) and 400.0 / VOX[1] == pytest.approx(9.2308, abs=1e-4)
    # physical width of the fitted spot in nm (index sigma x voxel size)
    assert r["sigma_y_fit"] * VOX_BASE[1] == pytest.approx(3.0 * 65.0, abs=0.1)


def test_partial_failure_keeps_fitted_values():
    vol = make_volume([(TRUE["mu"], TRUE["sig"], TRUE["A"])], bg=10.0, noise=1.0, seed=3)
    r = L.fit_spot(vol, (8, 24, 36), S0, 10.0, 15.0, 1.0, dict(CFG, cond_max=None), VOX)   # TypeError after the optimiser returned
    assert r["fit_error"].startswith("TypeError") and r["optimizer_converged"]
    assert np.isfinite([r[f"{n}_fit"] for n in L.PNAMES]).all() and np.isfinite(r["rss"]) and np.isfinite(r["jac_cond"]) and r["jac_rank"] == 8
    assert r["se_reported"] is False and np.isnan(r["se_y_px"]) and "_roi_lo" not in r
    q, why = L.classify_quality(r, 0.1, 0.1, 10.0, 0, CFG)
    assert q == "unreliable" and "fit_error" in why                 # fall back to the initial coordinates downstream


# ---- independent midpoint-rule integration (no erf) with a fixed error budget
ATOL = RTOL = 1e-6          # budget for |erf-based - midpoint|; convergence check uses budget / 10
M_START, M_MAX = 64, 2 ** 16


def midpoint_axis(k, mu, sigma, m):
    """integral over [k-.5, k+.5] of exp(-(x-mu)^2/(2 sigma^2)) dx, midpoint rule with m sub-intervals per voxel."""
    h = 1.0 / m
    x = (np.asarray(k, float)[:, None] - 0.5) + (np.arange(m) + 0.5) * h
    return h * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2)).sum(axis=1)


def converged_midpoint(k, mu, sigma):
    """Double m until |I_m - I_2m| <= (ATOL + RTOL*|I_2m|)/10 everywhere; return (I_2m, m_final)."""
    m, prev = M_START, midpoint_axis(k, mu, sigma, M_START)
    while m < M_MAX:
        m *= 2
        cur = midpoint_axis(k, mu, sigma, m)
        if np.all(np.abs(prev - cur) <= (ATOL + RTOL * np.abs(cur)) / 10.0):
            return cur, m
        prev = cur
    raise AssertionError(f"midpoint integration did not converge below m={M_MAX} (sigma={sigma}, mu={mu})")


@pytest.mark.parametrize("sigma", [60 / 65, 1.0, 1.385, 2.0, 3.0, 6.154, 9.231])
@pytest.mark.parametrize("mu", [0.0, 0.3, -1.7, 4.5])
def test_axis_matches_independent_midpoint_integration(sigma, mu):
    k = np.arange(-6, 11, dtype=float)
    ref, m_final = converged_midpoint(k, mu, sigma)
    got = L._axis(k, mu, sigma)
    assert np.all(np.abs(got - ref) <= ATOL + RTOL * np.abs(ref)), (sigma, mu, m_final, np.abs(got - ref).max())


def test_midpoint_reference_needs_adaptive_refinement():
    """Documents why the sampling density is not fixed: m=64 misses the budget at the smallest baseline sigma."""
    k, mu, sigma = np.arange(-6, 11, dtype=float), 0.0, 60 / 65
    exact = L._axis(k, mu, sigma)
    err64 = np.abs(midpoint_axis(k, mu, sigma, 64) - exact).max()
    assert err64 > ATOL, err64
    _, m_final = converged_midpoint(k, mu, sigma)
    assert m_final > 64


def test_3d_model_matches_product_of_independent_1d_integrals_non_cubic():
    shape = (9, 11, 13)
    mu, sig, A, bg = (4.3, 5.6, 6.1), (1.1, 1.7, 2.3), 17.0, 3.0      # sz != sy != sx, non-cubic: a swapped axis would fail
    ks = [np.arange(n, dtype=float) for n in shape]
    ref = [converged_midpoint(ks[i], mu[i], sig[i])[0] for i in range(3)]
    expected = bg + A * ref[0][:, None, None] * ref[1][None, :, None] * ref[2][None, None, :]
    got = L.gaussian_model((*mu, *sig, A, bg), *ks)
    assert got.shape == shape
    assert np.all(np.abs(got - expected) <= ATOL + RTOL * np.abs(expected))


def test_3d_axis_mapping_has_power():
    """With equal y/x extents, a y<->x mix-up in the expected array must violate the same budget (the check can fail)."""
    shape = (9, 11, 11)
    mu, sig, A, bg = (4.3, 4.6, 6.1), (1.1, 1.7, 2.3), 17.0, 3.0
    ks = [np.arange(n, dtype=float) for n in shape]
    ref = [converged_midpoint(ks[i], mu[i], sig[i])[0] for i in range(3)]
    good = bg + A * ref[0][:, None, None] * ref[1][None, :, None] * ref[2][None, None, :]
    mixed = bg + A * ref[0][:, None, None] * ref[2][None, :, None] * ref[1][None, None, :]
    got = L.gaussian_model((*mu, *sig, A, bg), *ks)
    assert np.all(np.abs(got - good) <= ATOL + RTOL * np.abs(good))
    assert not np.all(np.abs(got - mixed) <= ATOL + RTOL * np.abs(mixed))


def test_gaussian_integral_equals_model_sum_for_interior_spot():
    shape = (31, 41, 41)
    mu, sig, A = (15.0, 20.3, 19.7), (1.5, 2.0, 1.8), 20.0           # >= 10 slices / 5 sigma from every border: tail loss < 1e-5
    model = L.gaussian_model((*mu, *sig, A, 0.0), *(np.arange(n, dtype=float) for n in shape))
    assert np.isclose(model.sum(), L.gaussian_integral_voxels(A, *sig), rtol=1e-5, atol=0)   # internal consistency only
