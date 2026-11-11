"""E-layer equivalence: ``catkit.localization`` vs frozen ``REF-LOC`` (same process, same inputs).

Every case compares the WHOLE returned structure (all keys, NaN fields, flags,
reasons, partial-failure results) under the explicit per-field tolerances of
``compare.LOC_FIT_SPEC``. A case may also assert independently that the branch
it targets was really taken (equivalence of two identical wrong results would
otherwise prove nothing about coverage).

Equivalence to the frozen code is NOT evidence of real-data localization accuracy.
"""

from __future__ import annotations

import inspect
import ast

import pytest

from scientific import require_modules

require_modules("numpy", "scipy", "bigfish")

import numpy as np  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import catkit.localization as P  # noqa: E402
from compare import EXACT, POS, LOC_FIT_SPEC, call_outcome, compare_dicts, compare_value  # noqa: E402
from synth import SIG0, SHAPE, TRUE, VOX, VOX_BASE, make_volume, rescaled_cfg  # noqa: E402

pytestmark = pytest.mark.research

FUNCS = ["_axis", "gaussian_model", "gaussian_integral_voxels", "roi_bounds", "_nan_result", "fit_spot",
         "run_bigfish_fit_subpixel", "classify_quality"]


# --------------------------------------------------------------------- source identity
def test_port_is_verbatim(frozen_modules):
    L = frozen_modules["REF-LOC"]
    for name in FUNCS:
        assert inspect.getsource(getattr(P, name)) == inspect.getsource(getattr(L, name)), name
    assert P.PNAMES == L.PNAMES
    assert P.DEFAULT_CFG == L.DEFAULT_CFG and P.DEFAULT_CFG is not L.DEFAULT_CFG
    # whole-module AST equality apart from the docstring (comments/whitespace are covered per function above)
    pa, la = ast.parse(inspect.getsource(P)), ast.parse(inspect.getsource(L))
    assert ast.dump(ast.Module(pa.body[1:], [])) == ast.dump(ast.Module(la.body[1:], []))


# --------------------------------------------------------------------- pure functions
def test_axis_model_integral_roi(frozen_modules, e_stats):
    L = frozen_modules["REF-LOC"]
    problems = []
    rng = np.random.default_rng(0)
    for sigma in (60 / 65, 1.0, 1.385, 2.0, 3.0, 9.23, 15.0):
        for mu in (-3.2, 0.0, 0.5, 4.999, 12.0):
            k = np.arange(-8, 20, dtype=float)
            problems += compare_value(L._axis(k, mu, sigma), P._axis(k, mu, sigma), POS, f"_axis s={sigma} mu={mu}", e_stats)
    for _ in range(25):
        p = (*rng.uniform(2, 14, 3), *rng.uniform(0.9, 6, 3), rng.uniform(0, 500), rng.uniform(0, 100))
        kz, ky, kx = (np.arange(a, b, dtype=float) for a, b in ((0, 9), (3, 15), (-2, 12)))
        a, b = L.gaussian_model(p, kz, ky, kx), P.gaussian_model(p, kz, ky, kx)
        assert a.shape == b.shape == (9, 12, 14)
        problems += compare_value(a, b, POS, "gaussian_model", e_stats)
        problems += compare_value(L.gaussian_integral_voxels(p[6], *p[3:6]), P.gaussian_integral_voxels(p[6], *p[3:6]),
                                  EXACT, "gaussian_integral_voxels", e_stats)
    assert not problems, "\n".join(problems[:10])


@pytest.mark.parametrize("init", [(2.5, 2.5, 2.5), (3.5, 3.5, 3.5), (8.5, 24.5, 35.5), (9.5, 25.5, 36.5), (0, 0, 0), (16, 49, 69),
                                  (8.3, 24.4, 35.7), (-0.5, 0.5, 1.5), (100, 100, 100), (np.float32(8.5), np.float64(24.5), 36)])
@pytest.mark.parametrize("shape", [SHAPE, (3, 10, 10), (9, 15, 15), (1, 1, 1)])
def test_roi_bounds(frozen_modules, init, shape):
    L = frozen_modules["REF-LOC"]
    cfg = dict(P.DEFAULT_CFG)
    a, b = call_outcome(L.roi_bounds, init, shape, cfg), call_outcome(P.roi_bounds, init, shape, cfg)
    assert a == b, (a, b)


def test_roi_bounds_half_even_rounding():
    cfg = dict(P.DEFAULT_CFG)
    lo, hi = P.roi_bounds((2.5, 3.5, 4.5), (17, 50, 70), cfg)   # Python round: 2, 4, 4
    assert lo == (0, 0, 0) and hi == (7, 12, 12)                 # z0=2 -> hi 2+4+1; y0=4 -> 4+7+1; x0=4 -> 12


def test_nan_result_template(frozen_modules):
    L = frozen_modules["REF-LOC"]
    assert compare_dicts(L._nan_result(), P._nan_result(), LOC_FIT_SPEC) == []
    assert len(P._nan_result()) == 33 and "_roi_lo" not in P._nan_result()


# --------------------------------------------------------------------- fit_spot cases
def _spot(mu=TRUE["mu"], sig=TRUE["sig"], A=TRUE["A"], bg=TRUE["bg"], **kw):
    return make_volume([(mu, sig, A)], bg=bg, **kw)


def _case(name, image, init=(8, 24, 36), sigma0=SIG0, bg0=10.0, A0=15.0, noise=1.0, cfg=None, voxel=VOX, expect=None):
    return dict(name=name, image=image, init=init, sigma0=sigma0, bg0=bg0, A0=A0, noise=noise,
                cfg=dict(P.DEFAULT_CFG) if cfg is None else cfg, voxel=voxel, expect=expect)


def _pb(r, name):
    return name in r["params_at_bound"].split(",")


def _fit_cases():
    cfg_b = rescaled_cfg(P.DEFAULT_CFG, VOX_BASE)
    clean, noisy = _spot(), [_spot(noise=1.0, seed=s, integer=True) for s in range(8)]
    cases = [
        _case("noiseless_ref_voxel", clean, noise=np.nan, expect=lambda r: r["optimizer_converged"] and r["se_reported"] and r["params_at_bound"] == ""),
        _case("noiseless_baseline_voxel", clean, noise=np.nan, cfg=cfg_b, voxel=VOX_BASE, expect=lambda r: r["optimizer_converged"]),
    ]
    cases += [_case(f"noisy_ref_seed{s}", noisy[s], expect=lambda r: r["optimizer_converged"]) for s in range(8)]
    cases += [_case(f"noisy_baseline_seed{s}", noisy[s], cfg=cfg_b, voxel=VOX_BASE) for s in (0, 1)]
    cases += [
        _case("z_edge", _spot(mu=(1.2, 24.4, 35.7), noise=1.0, seed=2), init=(1, 24, 36), expect=lambda r: r["roi_z_truncated"] and not r["roi_xy_truncated"]),
        _case("xy_edge", _spot(mu=(8.3, 24.4, 2.2), noise=1.0, seed=2), init=(8, 24, 2), expect=lambda r: r["roi_xy_truncated"] and not r["roi_z_truncated"]),
        _case("corner_low", _spot(mu=(1.2, 2.4, 2.2), noise=1.0, seed=2), init=(1, 2, 2), expect=lambda r: r["roi_xy_truncated"] and r["roi_z_truncated"]),
        _case("corner_high", _spot(mu=(15.5, 47.9, 67.8), noise=1.0, seed=2), init=(16, 48, 68), expect=lambda r: r["roi_xy_truncated"] and r["roi_z_truncated"]),
        _case("half_integer_init_a", noisy[1], init=(8.5, 24.5, 35.5), expect=lambda r: r["n_roi_vox"] == 9 * 15 * 15),   # rounds to (8, 24, 36)
        _case("half_integer_init_b", noisy[1], init=(9.5, 25.5, 36.5), expect=lambda r: r["n_roi_vox"] == 9 * 15 * 15),   # rounds to (10, 26, 36)
        _case("fractional_init", noisy[2], init=(8.3, 24.4, 35.7)),
        _case("mu_y_at_upper_bound", _spot(mu=(8.3, 24.4, 35.7), noise=0.2, seed=1), init=(8, 21, 36), expect=lambda r: _pb(r, "mu_y_hi")),
        _case("mu_z_at_lower_bound", _spot(mu=(8.3, 24.4, 35.7), noise=0.2, seed=1), init=(11, 24, 36), expect=lambda r: _pb(r, "mu_z_lo")),
        _case("sigma_xy_at_lower_bound", _spot(sig=(2.0, 0.6, 0.6), A=60.0), expect=lambda r: _pb(r, "sigma_y_lo") or _pb(r, "sigma_x_lo")),
        _case("sigma_at_upper_bound", _spot(sig=(8.0, 14.0, 14.0), A=5.0, noise=0.3, seed=1), expect=lambda r: "sigma_" in r["params_at_bound"] and "_hi" in r["params_at_bound"]),
        _case("A_at_upper_bound", _spot(A=2.0e5), A0=1.0e5, expect=lambda r: _pb(r, "A_hi")),
        _case("bg_at_upper_bound", _spot(bg=1500.0, noise=0.5, seed=1), bg0=900.0, expect=lambda r: _pb(r, "bg_hi")),
        _case("bg_at_lower_bound", _spot(bg=-5.0, noise=0.2, seed=1), bg0=0.5, expect=lambda r: _pb(r, "bg_lo")),
        _case("noise_only", make_volume([], bg=10.0, noise=1.0, seed=11), A0=2.0),
        _case("nan_in_roi", np.where(np.arange(clean.size).reshape(clean.shape) == np.ravel_multi_index((8, 24, 36), clean.shape), np.nan, clean),
              expect=lambda r: r["fit_error"] == "nonfinite_roi" and r["nonfinite"] and np.isnan(r["z_fit"]) and not r["optimizer_converged"]),
        _case("inf_in_roi", np.where(np.arange(clean.size).reshape(clean.shape) == np.ravel_multi_index((8, 24, 36), clean.shape), np.inf, clean),
              expect=lambda r: r["fit_error"] == "nonfinite_roi"),
        _case("nan_outside_roi_is_ignored", np.where(np.arange(clean.size).reshape(clean.shape) == 0, np.nan, clean), expect=lambda r: r["optimizer_converged"]),
        _case("roi_too_small_by_cfg", noisy[0], cfg=dict(P.DEFAULT_CFG, min_roi_voxels=10 ** 6), expect=lambda r: r["fit_error"] == "roi_too_small" and r["n_roi_vox"] == 2025),
        _case("roi_too_small_tiny_image", np.full((2, 5, 5), 10.0), init=(1, 2, 2), expect=lambda r: r["fit_error"] == "roi_too_small" and r["n_roi_vox"] == 50),
        _case("empty_roi_init_outside", noisy[0], init=(8, 200, 36), expect=lambda r: r["fit_error"] == "roi_too_small" and r["n_roi_vox"] == 0),
        _case("negative_init_slice_quirk", noisy[0], init=(-20, 24, 36)),  # frozen quirk: negative upper slice bound wraps; documented, not fixed
        _case("impossible_bounds_exception_in_optimizer", noisy[0], cfg=dict(P.DEFAULT_CFG, sigma_xy_bounds_nm=(400.0, 60.0)),
              expect=lambda r: r["fit_error"] != "" and not r["optimizer_converged"] and np.isnan(r["z_fit"])),
        _case("max_nfev_exhausted", noisy[0], cfg=dict(P.DEFAULT_CFG, max_nfev=2),
              expect=lambda r: r["status"] == 0 and r["optimizer_converged"] is False and np.isfinite([r[f"{n}_fit"] for n in P.PNAMES]).all()),
        _case("exception_after_optimizer_cond_max_none", noisy[0], cfg=dict(P.DEFAULT_CFG, cond_max=None),
              expect=lambda r: (r["fit_error"].startswith("TypeError") and r["optimizer_converged"] and np.isfinite(r["rss"])
                                and r["jac_rank"] == 8 and np.isfinite(r["jac_cond"]) and r["se_reported"] is False
                                and np.isnan(r["se_y_px"]) and "_roi_lo" not in r and "_roi_hi" not in r)),
        _case("noise_nan", noisy[0], noise=np.nan, expect=lambda r: np.isnan(r["reduced_chi2"])),
        _case("noise_zero", noisy[0], noise=0.0, expect=lambda r: np.isnan(r["reduced_chi2"])),
        _case("noise_negative", noisy[0], noise=-1.0, expect=lambda r: np.isnan(r["reduced_chi2"])),
        _case("noise_positive", noisy[0], noise=1.0, expect=lambda r: np.isfinite(r["reduced_chi2"])),
        _case("bg0_nan_fallback", noisy[0], bg0=np.nan),
        _case("A0_nan_fallback", noisy[0], A0=np.nan),
        _case("bg0_A0_nan_fallback", noisy[0], bg0=np.nan, A0=np.nan),
        _case("bg0_above_max_clipped", noisy[0], bg0=1.0e6),
        _case("bg0_negative_clipped", noisy[0], bg0=-5.0),
        _case("A0_negative_floor", noisy[0], A0=-3.0),
        _case("A0_above_max_clipped", noisy[0], A0=1.0e9),
        _case("sigma0_outside_bounds_clipped", noisy[0], sigma0=(0.01, 100.0, 100.0)),
        _case("uint16_image", noisy[0].astype(np.uint16)),
        _case("float32_image", noisy[0].astype(np.float32)),
        _case("other_voxel_nm", noisy[0], voxel=(200.0, 50.0, 50.0)),
        _case("flat_image_sst_zero", np.full(SHAPE, 10.0), expect=lambda r: np.isnan(r["r2"]) or r["fit_error"] != ""),
    ]
    return cases


def _run_fit_cases(L, cases, e_stats):
    problems = []
    for c in cases:
        a = call_outcome(L.fit_spot, c["image"], c["init"], c["sigma0"], c["bg0"], c["A0"], c["noise"], dict(c["cfg"]), c["voxel"])
        b = call_outcome(P.fit_spot, c["image"], c["init"], c["sigma0"], c["bg0"], c["A0"], c["noise"], dict(c["cfg"]), c["voxel"])
        if a[0] != "ok" or b[0] != "ok":
            problems.append(f"[{c['name']}] fit_spot raised (it must never raise): frozen={a[:3]} port={b[:3]}")
            continue
        problems += compare_dicts(a[1], b[1], LOC_FIT_SPEC, e_stats, prefix=f"[{c['name']}] ")
        if c["expect"] is not None and not c["expect"](b[1]):
            r = b[1]
            problems.append(f"[{c['name']}] target branch NOT triggered: status={r['status']} conv={r['optimizer_converged']} "
                            f"fit_error={r['fit_error']!r} bound={r['params_at_bound']!r} n_roi={r['n_roi_vox']}")
    return problems


def test_fit_spot_equivalence_and_branch_triggers(frozen_modules, e_stats):
    problems = _run_fit_cases(frozen_modules["REF-LOC"], _fit_cases(), e_stats)
    assert not problems, f"{len(problems)} problem(s):\n" + "\n".join(problems[:15])


def test_fit_spot_does_not_mutate_inputs(frozen_modules):
    cfg_before = dict(P.DEFAULT_CFG)
    img = _spot(noise=1.0, seed=1)
    img0, init = img.copy(), np.array([8.0, 24.0, 36.0])
    P.fit_spot(img, init, SIG0, 10.0, 15.0, 1.0)
    assert np.array_equal(img, img0) and init.tolist() == [8.0, 24.0, 36.0] and P.DEFAULT_CFG == cfg_before


# --------------------------------------------------------------------- controlled optimiser outputs (stubs)
def _stub(kind, record=None, success=True, status=1):
    def fake(fun, x0, bounds=None, method=None, x_scale=None, max_nfev=None, **kw):
        if record is not None:
            record.append(dict(x0=np.array(x0), lower=np.array(bounds[0]), upper=np.array(bounds[1]), method=method, x_scale=x_scale, max_nfev=max_nfev))
        x = np.array(x0, float)
        f = fun(x)
        n = f.size
        J = np.random.default_rng(0).normal(size=(n, 8))
        mask = np.zeros(8, int)
        if kind == "nonfinite":
            x[3] = np.nan
        elif kind == "rank_deficient":
            J[:, 3] = 0.0
        elif kind == "zero_jacobian":
            J[:] = 0.0
        elif kind == "at_bound":
            mask[0], mask[6], mask[7] = -1, 1, -1
        elif kind == "raise":
            raise RuntimeError("boom")
        return SimpleNamespace(x=x, status=status, message="stub", nfev=1, success=success, fun=f, jac=J, active_mask=mask)
    return fake


STUBS = [
    ("nonfinite_optimizer_output", dict(kind="nonfinite"), lambda r: r["nonfinite"] and np.isnan(r["sigma_z_fit"]) and np.isnan(r["rss"]) and "_roi_lo" not in r),
    ("rank_deficient_jacobian", dict(kind="rank_deficient"), lambda r: r["jac_rank"] == 7 and r["jac_cond"] > 1e6 and not r["se_reported"] and "_roi_lo" in r),
    ("zero_jacobian", dict(kind="zero_jacobian"), lambda r: r["jac_rank"] == 0 and np.isposinf(r["jac_cond"]) and not r["se_reported"]),
    ("params_at_bound_from_active_mask", dict(kind="at_bound"), lambda r: r["params_at_bound"] == "mu_z_lo,A_hi,bg_lo" and not r["se_reported"]),
    ("well_conditioned_stub_reports_se", dict(kind="plain"), lambda r: r["se_reported"] and r["params_at_bound"] == ""),
    ("success_false_not_converged", dict(kind="plain", success=False, status=1), lambda r: r["optimizer_converged"] is False),
    ("status_zero_not_converged", dict(kind="plain", success=True, status=0), lambda r: r["optimizer_converged"] is False),
    ("status_negative_not_converged", dict(kind="plain", success=True, status=-1), lambda r: r["optimizer_converged"] is False),
    ("exception_inside_optimizer", dict(kind="raise"), lambda r: r["fit_error"] == "RuntimeError: boom" and np.isnan(r["mu_z_fit"]) and not r["optimizer_converged"]),
]


@pytest.mark.parametrize("name, kw, expect", STUBS, ids=[s[0] for s in STUBS])
def test_fit_spot_with_controlled_optimizer_output(frozen_modules, monkeypatch, e_stats, name, kw, expect):
    L = frozen_modules["REF-LOC"]
    img = _spot(noise=1.0, seed=3)
    monkeypatch.setattr(L, "least_squares", _stub(**kw))
    monkeypatch.setattr(P, "least_squares", _stub(**kw))
    a = L.fit_spot(img, (8, 24, 36), SIG0, 10.0, 15.0, 1.0, dict(P.DEFAULT_CFG), VOX)
    b = P.fit_spot(img, (8, 24, 36), SIG0, 10.0, 15.0, 1.0, dict(P.DEFAULT_CFG), VOX)
    problems = compare_dicts(a, b, LOC_FIT_SPEC, e_stats)
    assert not problems, "\n".join(problems[:10])
    assert expect(b), {k: v for k, v in b.items() if not k.startswith("_")}


def test_start_values_passed_to_optimizer(frozen_modules, monkeypatch):
    """bg0/A0 fallbacks, clipping into bounds and the optimiser settings, observed via a capturing stub."""
    L = frozen_modules["REF-LOC"]
    img = _spot(noise=1.0, seed=3)
    roi = img[4:13, 17:32, 29:44]                       # ROI of init (8, 24, 36): z 4..12, y 17..31, x 29..43
    lo_hi = {}
    for label, mod in (("frozen", L), ("port", P)):
        rec = []
        monkeypatch.setattr(mod, "least_squares", _stub("plain", rec))
        mod.fit_spot(img, (8, 24, 36), SIG0, np.nan, np.nan, 1.0, dict(P.DEFAULT_CFG), VOX)
        mod.fit_spot(img, (8, 24, 36), (0.01, 100.0, 100.0), -5.0, -3.0, 1.0, dict(P.DEFAULT_CFG), VOX)
        mod.fit_spot(img, (8, 24, 36), SIG0, 1.0e6, 1.0e9, 1.0, dict(P.DEFAULT_CFG), VOX)
        lo_hi[label] = rec
    for fr, po in zip(lo_hi["frozen"], lo_hi["port"]):
        for k in ("x0", "lower", "upper"):
            assert np.array_equal(fr[k], po[k]), k
        assert (fr["method"], fr["x_scale"], fr["max_nfev"]) == (po["method"], po["x_scale"], po["max_nfev"]) == ("trf", "jac", 300)
    r = lo_hi["port"]
    bg_med = float(np.median(roi))
    assert r[0]["x0"][7] == pytest.approx(bg_med, abs=0, rel=0) and r[0]["x0"][6] == pytest.approx(roi.max() - bg_med, abs=0, rel=0)   # fallbacks
    assert r[1]["x0"][7] == 1e-9 and r[1]["x0"][6] == 0.1                                  # bg0 clipped to 0 (+1e-9), A0 floored at 0.1
    szb, sxyb = np.array(P.DEFAULT_CFG["sigma_z_bounds_nm"]) / VOX[0], np.array(P.DEFAULT_CFG["sigma_xy_bounds_nm"]) / VOX[1]
    assert r[1]["x0"][3] == pytest.approx(szb[0] + 1e-9) and r[1]["x0"][4] == pytest.approx(sxyb[1] - 1e-9)  # sigma0 0.01 -> lower, 100 -> upper bound (+-1e-9)
    assert r[2]["x0"][7] == pytest.approx(1e3 - 1e-9) and r[2]["x0"][6] == pytest.approx(1e5 - 1e-9)       # clipped to maxima (-1e-9)
    assert np.allclose(r[0]["lower"], [6.0, 22.0, 34.0, szb[0], sxyb[0], sxyb[0], 0, 0], rtol=0, atol=1e-12)
    assert np.allclose(r[0]["upper"], [10.0, 26.0, 38.0, szb[1], sxyb[1], sxyb[1], 1e5, 1e3], rtol=0, atol=1e-12)


# --------------------------------------------------------------------- classify_quality
def _base_fit():
    r = P.fit_spot(_spot(noise=1.0, seed=3), (8, 24, 36), SIG0, 10.0, 15.0, 1.0, dict(P.DEFAULT_CFG), VOX)
    assert r["optimizer_converged"] and r["se_reported"] and r["params_at_bound"] == ""
    return r


QUALITY_CASES = [  # name, result overrides, (shift_xy, shift_z, snr_peak, neighbors), expected (quality, reasons)
    ("good", {}, (0.1, 0.1, 10.0, 0), ("good", [])),
    ("fit_error", {"fit_error": "x"}, (0.1, 0.1, 10.0, 0), ("unreliable", ["fit_error"])),
    ("not_converged", {"optimizer_converged": False}, (0.1, 0.1, 10.0, 0), ("unreliable", ["not_converged"])),
    ("nonfinite", {"nonfinite": True}, (0.1, 0.1, 10.0, 0), ("unreliable", ["nonfinite"])),
    ("center_out_of_roi", {"center_out_of_roi": True}, (0.1, 0.1, 10.0, 0), ("unreliable", ["center_out_of_roi"])),
    ("amp_nonpositive", {"A_fit": 1e-7}, (0.1, 0.1, 10.0, 0), ("unreliable", ["amp_nonpositive"])),
    ("amp_exactly_1e-6", {"A_fit": 1e-6}, (0.1, 0.1, 10.0, 0), ("unreliable", ["amp_nonpositive"])),
    ("amp_just_above", {"A_fit": 1.0000001e-6}, (0.1, 0.1, 10.0, 0), ("good", [])),
    ("amp_nan_not_flagged", {"A_fit": np.nan}, (0.1, 0.1, 10.0, 0), ("good", [])),
    ("mu_z_lo", {"params_at_bound": "mu_z_lo"}, (0.1, 0.1, 10.0, 0), ("unreliable", ["mu_at_bound"])),
    ("mu_x_hi", {"params_at_bound": "mu_x_hi"}, (0.1, 0.1, 10.0, 0), ("unreliable", ["mu_at_bound"])),
    ("sigma_at_bound", {"params_at_bound": "sigma_y_lo"}, (0.1, 0.1, 10.0, 0), ("suspect", ["sigma_at_bound"])),
    ("A_hi", {"params_at_bound": "A_hi"}, (0.1, 0.1, 10.0, 0), ("suspect", ["A_or_bg_at_bound"])),
    ("A_lo_not_flagged", {"params_at_bound": "A_lo"}, (0.1, 0.1, 10.0, 0), ("good", [])),
    ("bg_lo", {"params_at_bound": "bg_lo"}, (0.1, 0.1, 10.0, 0), ("suspect", ["A_or_bg_at_bound"])),
    ("shift_xy_equal_threshold", {}, (1.5, 0.1, 10.0, 0), ("good", [])),
    ("shift_xy_above", {}, (1.6, 0.1, 10.0, 0), ("suspect", ["large_shift_xy"])),
    ("shift_z_above", {}, (0.1, 1.6, 10.0, 0), ("suspect", ["large_shift_z"])),
    ("shift_nan_ignored", {}, (np.nan, np.nan, 10.0, 0), ("good", [])),
    ("z_edge", {"roi_z_truncated": True}, (0.1, 0.1, 10.0, 0), ("suspect", ["z_edge"])),
    ("xy_edge", {"roi_xy_truncated": True}, (0.1, 0.1, 10.0, 0), ("suspect", ["xy_edge"])),
    ("neighbor", {}, (0.1, 0.1, 10.0, 2), ("suspect", ["neighbor_in_roi"])),
    ("se_not_reported_without_bad", {"se_reported": False}, (0.1, 0.1, 10.0, 0), ("suspect", ["uncertainty_unreliable"])),
    ("se_not_reported_with_bad_suppressed", {"se_reported": False, "nonfinite": True}, (0.1, 0.1, 10.0, 0), ("unreliable", ["nonfinite"])),
    ("snr_nan", {}, (0.1, 0.1, np.nan, 0), ("suspect", ["noise_unreliable"])),
    ("snr_below_threshold", {}, (0.1, 0.1, 2.9, 0), ("suspect", ["low_snr_peak"])),
    ("snr_equal_threshold", {}, (0.1, 0.1, 3.0, 0), ("good", [])),
    ("snr_inf_counts_as_unreliable", {}, (0.1, 0.1, np.inf, 0), ("suspect", ["noise_unreliable"])),
    ("many_reasons_order", {"fit_error": "e", "optimizer_converged": False, "nonfinite": True, "center_out_of_roi": True, "A_fit": 0.0,
                            "params_at_bound": "mu_z_lo,sigma_x_hi,A_hi,bg_lo", "roi_z_truncated": True, "roi_xy_truncated": True},
     (2.0, 2.0, 1.0, 1),
     ("unreliable", ["fit_error", "not_converged", "nonfinite", "center_out_of_roi", "amp_nonpositive", "mu_at_bound", "sigma_at_bound",
                     "A_or_bg_at_bound", "large_shift_xy", "large_shift_z", "z_edge", "xy_edge", "neighbor_in_roi", "low_snr_peak"])),
]


def test_classify_quality_every_branch(frozen_modules):
    L = frozen_modules["REF-LOC"]
    base = _base_fit()
    problems = []
    for name, over, (sxy, sz, snr, nb), expected in QUALITY_CASES:
        r = {**base, **over}
        a, b = call_outcome(L.classify_quality, r, sxy, sz, snr, nb, dict(P.DEFAULT_CFG)), call_outcome(P.classify_quality, r, sxy, sz, snr, nb, dict(P.DEFAULT_CFG))
        problems += compare_value(a, b, EXACT, f"[{name}] classify_quality")
        if b[0] != "ok" or (b[1][0], b[1][1]) != expected:
            problems.append(f"[{name}] expected {expected}, port returned {b}")
    assert not problems, "\n".join(problems)


def test_classify_quality_on_real_fits(frozen_modules):
    L = frozen_modules["REF-LOC"]
    problems = []
    for c in _fit_cases():
        if c["expect"] is None and not c["name"].startswith("noisy"):
            pass
        r = P.fit_spot(c["image"], c["init"], c["sigma0"], c["bg0"], c["A0"], c["noise"], dict(c["cfg"]), c["voxel"])
        for snr, nb in ((10.0, 0), (1.0, 1), (np.nan, 0)):
            a = L.classify_quality(r, 0.5, 0.5, snr, nb, dict(P.DEFAULT_CFG))
            b = P.classify_quality(r, 0.5, 0.5, snr, nb, dict(P.DEFAULT_CFG))
            problems += compare_value(a, b, EXACT, f"[{c['name']} snr={snr}] classify_quality")
    assert not problems, "\n".join(problems[:10])


# --------------------------------------------------------------------- run_bigfish_fit_subpixel
BF_CASES = [
    ("spot", lambda: (_spot(noise=0.5, seed=4, integer=True).astype(np.uint16), [(8, 24, 36)], VOX, (410.0, 97.0, 97.0))),
    ("two_spots_one_far", lambda: (make_volume([(TRUE["mu"], TRUE["sig"], 20.0), ((8.0, 40.0, 10.0), TRUE["sig"], 20.0)], noise=0.5, seed=4, integer=True).astype(np.uint16),
                                   [(8, 24, 36), (8, 40, 10)], VOX, (410.0, 97.0, 97.0))),
    ("empty_coords", lambda: (_spot(noise=0.5, seed=4, integer=True).astype(np.uint16), np.empty((0, 3), int), VOX, (410.0, 97.0, 97.0))),
    ("background_only", lambda: (make_volume([], noise=0.5, seed=4, integer=True).astype(np.uint16), [(8, 24, 36)], VOX, (410.0, 97.0, 97.0))),
    ("coords_outside_image", lambda: (_spot(noise=0.5, seed=4, integer=True).astype(np.uint16), [(8, 24, 500)], VOX, (410.0, 97.0, 97.0))),
    ("image_wrong_ndim", lambda: (np.zeros((50, 70), np.uint16), [(8, 24, 36)], VOX, (410.0, 97.0, 97.0))),
    ("voxel_wrong_length", lambda: (_spot(noise=0.5, seed=4, integer=True).astype(np.uint16), [(8, 24, 36)], (150.0, 43.0), (410.0, 97.0, 97.0))),
    ("float_coords_truncated", lambda: (_spot(noise=0.5, seed=4, integer=True).astype(np.uint16), [(8.7, 24.2, 35.9)], VOX, (410.0, 97.0, 97.0))),
]


@pytest.mark.parametrize("name, build", BF_CASES, ids=[c[0] for c in BF_CASES])
def test_run_bigfish_fit_subpixel_equivalence(frozen_modules, e_stats, name, build):
    L = frozen_modules["REF-LOC"]
    args = build()
    a, b = call_outcome(L.run_bigfish_fit_subpixel, *args), call_outcome(P.run_bigfish_fit_subpixel, *args)
    assert a[0] == b[0], (a[:3], b[:3])
    if a[0] == "raise":
        assert a == b
        return
    (oa, ua, ea), (ob, ub, eb) = a[1], b[1]
    problems = compare_value(oa, ob, POS, "coords", e_stats) + compare_value(ua, ub, EXACT, "unchanged") + compare_value(ea, eb, EXACT, "errors")
    assert not problems, "\n".join(problems)
    n = len(args[1])
    assert ob.shape == (n, 3) and ub.shape == (n,) and len(eb) == n
    if name == "spot":
        assert not eb[0] and np.allclose(ob[0], TRUE["mu"], atol=0.5)
    if name == "image_wrong_ndim":
        assert eb[0] != "" and np.isnan(ob).all()


def test_run_bigfish_fit_subpixel_unchanged_and_error_branches(frozen_modules, monkeypatch, e_stats):
    """Controlled replacement of ``bigfish.detection.fit_subpixel`` (shared module object) to force the branches."""
    import bigfish.detection as bfd

    L = frozen_modules["REF-LOC"]
    img = _spot(noise=0.5, seed=4, integer=True).astype(np.uint16)
    coords = [(8, 24, 36), (5, 10, 10), (7, 30, 40)]

    def returns_input(image, c, voxel, radius):
        return np.asarray(c, float)                      # "silently returns the input coordinates"

    def mixed(image, c, voxel, radius):
        c = np.asarray(c, float)
        if int(c[0, 1]) == 10:
            raise RuntimeError("fit failed")             # -> NaN row + error string
        return c + np.array([[0.25, -0.5, 0.125]])       # -> shifted row, unchanged=False

    for stub, expect_unchanged, expect_err in ((returns_input, [True, True, True], ["", "", ""]),
                                               (mixed, [False, False, False], ["", "RuntimeError: fit failed", ""])):
        monkeypatch.setattr(bfd, "fit_subpixel", stub)
        a = call_outcome(L.run_bigfish_fit_subpixel, img, coords, VOX, (410.0, 97.0, 97.0))
        b = call_outcome(P.run_bigfish_fit_subpixel, img, coords, VOX, (410.0, 97.0, 97.0))
        assert a[0] == b[0] == "ok"
        (oa, ua, ea), (ob, ub, eb) = a[1], b[1]
        problems = compare_value(oa, ob, POS, "coords", e_stats) + compare_value(ua, ub, EXACT, "unchanged") + compare_value(ea, eb, EXACT, "errors")
        assert not problems, "\n".join(problems)
        assert ub.tolist() == expect_unchanged and eb == expect_err
    assert np.isnan(ob[1]).all() and np.allclose(ob[0], np.array(coords[0]) + [0.25, -0.5, 0.125], atol=0, rtol=0)


def test_cond_max_equality_boundary(frozen_modules, monkeypatch, e_stats):
    """SE is reported iff jac_cond < cond_max: exactly equal must NOT report, one ulp above must (mutation `<`->`<=` is caught)."""
    L = frozen_modules["REF-LOC"]
    img = _spot(noise=1.0, seed=3)
    monkeypatch.setattr(L, "least_squares", _stub("plain"))
    monkeypatch.setattr(P, "least_squares", _stub("plain"))
    run = lambda mod, cmax: mod.fit_spot(img, (8, 24, 36), SIG0, 10.0, 15.0, 1.0, dict(P.DEFAULT_CFG, cond_max=cmax), VOX)
    c = run(P, 1.0e6)["jac_cond"]
    assert np.isfinite(c) and c < 1.0e6
    for cmax, reported in ((c, False), (float(np.nextafter(c, np.inf)), True), (float(np.nextafter(c, 0.0)), False)):
        a, b = run(L, cmax), run(P, cmax)
        assert not compare_dicts(a, b, LOC_FIT_SPEC, e_stats)
        assert b["se_reported"] is reported, (cmax, c)
