"""E-layer "call-site replay": frozen chain vs port chain on one synthetic field of view.

The chain mirrors the per-spot call sequence of the frozen per-FOV driver:

    nearest_neighbor_nm + overlap_flags (INIT coordinates)
      -> build_occupancy -> measure_spots(init.astype(float), ..., occ)
      -> fit_spot(init[i], bg0=local_background, A0=peak_net, noise=bg_noise_std)
      -> classify_quality(fit, shift, shift_z, snr_peak_legacy_fitqc, neighbor_in_roi)

This is an E test (function equivalence over a replayed call sequence). It is NOT
P-layer validation: no driver, no I/O, no detection/segmentation/filtering/export.

Because frozen and port sides share the same replay helper, a wiring mistake in the
helper would pass both. ``test_wiring_*`` therefore check the recorded call arguments
against values computed independently in the test body.
"""

from __future__ import annotations

import pytest

from scientific import require_modules

require_modules("numpy", "scipy", "bigfish")

import numpy as np  # noqa: E402

import catkit.localization as PL  # noqa: E402
import catkit.photometry as PP  # noqa: E402
from compare import EXACT, LOC_FIT_SPEC, NEIGHBOR_SPEC, OVERLAP_SPEC, PHOT_SPEC, POS, compare_dicts, compare_value  # noqa: E402
from synth import AP, SHIN, SHOUT, SIG0, SHAPE, TRUE, VOX, VOX_BASE, make_volume, rescale_radii, rescaled_cfg  # noqa: E402

pytestmark = pytest.mark.research

# (z, y, x) true centres; the replay starts from their rounded integer positions
SPOTS = {
    "clean": (8.3, 24.4, 35.7),
    "pair_a": (8.0, 25.0, 12.0),
    "pair_b": (8.0, 25.0, 17.0),
    "edge": (1.2, 10.4, 60.3),
    "nan_spot": (12.0, 40.0, 50.0),
}


def make_fov():
    vol = make_volume([(mu, TRUE["sig"], 20.0) for mu in SPOTS.values()], bg=10.0, noise=1.0, seed=21, integer=True)
    init = np.round(np.array(list(SPOTS.values()))).astype(np.int64)
    vol[tuple(init[4])] = np.nan                       # a NaN at the centre of the last spot (aperture + fit ROI)
    return vol, init


class Recorder:
    """Wraps a module's functions and records every call's arguments (used for the wiring assertions)."""

    def __init__(self, loc, phot):
        self.loc, self.phot = loc, phot
        self.calls = {"measure_spots": [], "fit_spot": [], "classify_quality": []}

    def measure_spots(self, *a, **k):
        self.calls["measure_spots"].append((a, k))
        return self.phot.measure_spots(*a, **k)

    def fit_spot(self, *a, **k):
        self.calls["fit_spot"].append((a, k))
        return self.loc.fit_spot(*a, **k)

    def classify_quality(self, *a, **k):
        self.calls["classify_quality"].append((a, k))
        return self.loc.classify_quality(*a, **k)


def replay(loc, phot, stack, init, voxel, ap, shin, shout, cfg):
    """Replay of the driver's per-spot chain; returns every intermediate and the recorder."""
    rec = Recorder(loc, phot)
    n = len(init)
    nn_nm, nn_j = phot.nearest_neighbor_nm(init, voxel)
    ap_ov, nb_roi = phot.overlap_flags(init, ap, (cfg["roi_half_z"], cfg["roi_half_xy"], cfg["roi_half_xy"]))
    occ = phot.build_occupancy(stack.shape, init, ap)
    prim = rec.measure_spots(stack, init.astype(float), ap, shin, shout, occ=occ)
    fits = [rec.fit_spot(stack, init[i], SIG0, prim["local_background"][i], prim["peak_net"][i], prim["bg_noise_std"][i], cfg, voxel)
            for i in range(n)]
    fc = np.array([[f["z_fit"], f["y_fit"], f["x_fit"]] for f in fits], float)
    dpx = fc - init
    qual, why = zip(*[rec.classify_quality(fits[i], np.hypot(dpx[i, 1], dpx[i, 2]), abs(dpx[i, 0]),
                                           prim["snr_peak_legacy_fitqc"][i], int(nb_roi[i]), cfg) for i in range(n)])
    return dict(nn=(nn_nm, nn_j), overlap=(ap_ov, nb_roi), occ=occ, prim=prim, fits=fits, fc=fc, dpx=dpx,
                qual=list(qual), why=[list(w) for w in why], rec=rec)


def _setup(voxel_name):
    if voxel_name == "ref":
        return VOX, AP, SHIN, SHOUT, dict(PL.DEFAULT_CFG)
    ap, shin, shout = (rescale_radii(r, VOX_BASE) for r in ((3.0, 5.0, 5.0), (5.0, 9.0, 9.0), (7.0, 13.0, 13.0)))
    return VOX_BASE, ap, shin, shout, rescaled_cfg(PL.DEFAULT_CFG, VOX_BASE)


@pytest.mark.parametrize("voxel_name", ["ref", "baseline"])
def test_callsite_replay_frozen_vs_port(frozen_modules, e_stats, voxel_name):
    voxel, ap, shin, shout, cfg = _setup(voxel_name)
    stack, init = make_fov()
    args = (stack, init, voxel, ap, shin, shout, cfg)
    fr = replay(frozen_modules["REF-LOC"], frozen_modules["REF-PHOT"], *args)
    po = replay(PL, PP, *args)

    problems = []
    problems += compare_value(fr["nn"], po["nn"], NEIGHBOR_SPEC, "nn", e_stats)
    problems += compare_value(fr["overlap"], po["overlap"], OVERLAP_SPEC, "overlap", e_stats)
    problems += compare_value(fr["occ"], po["occ"], EXACT, "occ", e_stats)
    problems += compare_dicts(fr["prim"], po["prim"], PHOT_SPEC, e_stats, prefix="prim.")
    for i, (a, b) in enumerate(zip(fr["fits"], po["fits"])):
        problems += compare_dicts(a, b, LOC_FIT_SPEC, e_stats, prefix=f"fit[{i}].")
    problems += compare_value(fr["fc"], po["fc"], POS, "fit_coords", e_stats)
    problems += compare_value(fr["dpx"], po["dpx"], POS, "shifts", e_stats)
    problems += compare_value(fr["qual"], po["qual"], EXACT, "quality")
    problems += compare_value(fr["why"], po["why"], EXACT, "reasons")        # reason lists: content AND order
    assert not problems, "\n".join(problems[:10])

    # the replay must not be vacuous: the field contains successes AND the intended failures
    fits, prim = po["fits"], po["prim"]
    assert fits[0]["optimizer_converged"] and np.isfinite(prim["snr_primary"][0]) and po["qual"][0] in ("good", "suspect")
    assert fits[4]["fit_error"] == "nonfinite_roi" and po["qual"][4] == "unreliable"
    assert po["overlap"][0][1] and po["overlap"][0][2] and po["overlap"][1][1] >= 1
    assert prim["aperture_truncated"][3] or prim["aperture_z_truncated"][3] or prim["shell_truncated"][3]
    assert np.isnan(prim["aperture_sum"][4])                                   # NaN in the aperture propagates (not filtered)
    assert (prim["n_shell_excluded"][[1, 2]] > 0).all()                         # pair members exclude each other from their shells


@pytest.mark.parametrize("voxel_name", ["ref", "baseline"])
def test_wiring_photometry_init_fit_receives_photometry_outputs_quality_uses_legacy_snr(voxel_name):
    """Independent of the shared helper: check recorded call arguments against values computed here."""
    voxel, ap, shin, shout, cfg = _setup(voxel_name)
    stack, init = make_fov()
    out = replay(PL, PP, stack, init, voxel, ap, shin, shout, cfg)
    rec = out["rec"]

    # (1) photometry is measured at the INITIAL integer coordinates, not at fitted ones
    (m_args, m_kw), = rec.calls["measure_spots"]
    assert m_args[0] is stack and np.array_equal(m_args[1], init.astype(float)) and m_args[1].dtype == np.float64
    assert not np.allclose(m_args[1][0], out["fc"][0], atol=1e-6)               # fitted coordinates really differ from init
    ref = PP.measure_spots(stack, init.astype(float), ap, shin, shout, occ=PP.build_occupancy(stack.shape, init, ap))
    assert compare_dicts(ref, out["prim"], PHOT_SPEC) == []                       # same measurement computed directly here
    assert np.array_equal(m_kw["occ"], PP.build_occupancy(stack.shape, init, ap))

    # (2) fit_spot receives init coordinates plus bg0=local_background, A0=peak_net, noise=bg_noise_std (NOT the legacy noise)
    assert len(rec.calls["fit_spot"]) == len(init)
    i = 0
    (f_args, _), = [rec.calls["fit_spot"][i]]
    assert f_args[0] is stack and np.array_equal(f_args[1], init[i]) and f_args[2] == SIG0
    assert f_args[3] == ref["local_background"][i] and f_args[4] == ref["peak_net"][i] and f_args[5] == ref["bg_noise_std"][i]
    assert ref["bg_noise_std"][i] != ref["bg_noise_legacy"][i] and f_args[5] != ref["bg_noise_legacy"][i]   # the two noises differ here
    assert f_args[6] is cfg and f_args[7] == voxel
    for k, (a, _) in enumerate(rec.calls["fit_spot"]):
        assert np.array_equal(a[1], init[k])
        for got, key in ((a[3], "local_background"), (a[4], "peak_net"), (a[5], "bg_noise_std")):
            assert (got == ref[key][k]) or (np.isnan(got) and np.isnan(ref[key][k])), (k, key)

    # (3) classify_quality receives snr_peak_legacy_fitqc (not snr_peak_primary), the init->fit shifts and neighbor_in_roi counts
    assert len(rec.calls["classify_quality"]) == len(init)
    a, _ = rec.calls["classify_quality"][0]
    assert a[3] == ref["snr_peak_legacy_fitqc"][0] and a[3] != ref["snr_peak_primary"][0]
    assert a[2] == pytest.approx(abs(out["fc"][0][0] - init[0][0])) and a[1] == pytest.approx(np.hypot(*(out["fc"][0][1:] - init[0][1:])))
    assert a[4] == int(out["overlap"][1][0]) and a[5] is cfg
    for k, (a, _) in enumerate(rec.calls["classify_quality"]):
        s, l = ref["snr_peak_legacy_fitqc"][k], a[3]
        assert (s == l) or (np.isnan(s) and np.isnan(l)), k
