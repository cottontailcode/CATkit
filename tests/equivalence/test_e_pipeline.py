"""E-layer checks for M2.3-A: (1) the new assignment helpers vs the frozen driver (``core_distances`` is frozen code;
the rest of the block is replayed from the driver source), (2) the pipeline's per-spot wiring vs the frozen call sequence
executed with the FROZEN M2.1 modules on the same accepted batch. Not P-layer: no driver, no I/O.
"""

from __future__ import annotations

import inspect

import pytest

from scientific import require_modules

require_modules("numpy", "scipy", "skimage", "bigfish")

import numpy as np  # noqa: E402
from scipy.ndimage import distance_transform_edt  # noqa: E402
from skimage.segmentation import expand_labels  # noqa: E402

import catkit.cells as C  # noqa: E402
import catkit.detection as D  # noqa: E402
import catkit.pipeline as PL  # noqa: E402
import catkit.segmentation as S  # noqa: E402
from compare import DER, EXACT, LOC_FIT_SPEC, PHOT_SPEC, POS, compare_value  # noqa: E402
from synth_det import CY5, OPTICS, VOX, make_image  # noqa: E402

pytestmark = pytest.mark.research


def _labels():
    from synth_seg import make_dapi_stack
    st = make_dapi_stack(shape=(7, 140, 140), nuclei=[(30, 30, 8), (34, 46, 7), (90, 90, 9), (100, 20, 6), (5, 120, 6)], seed=3)
    seg = S.segment_fov(st, 65.0)
    return seg.cell_labels, seg.nucleoid_cores


def test_core_distances_is_verbatim_and_equal_to_frozen(frozen_seg_modules):
    DRV = frozen_seg_modules["REF-DRV"]
    assert inspect.getsource(C.core_distances) == inspect.getsource(DRV.core_distances)
    assert C.AMBIG_PX == DRV.AMBIG_PX and S.EXP_VARIANT_PX == DRV.EXP_VARIANT_PX
    lab, cores = _labels()
    rng = np.random.default_rng(0)
    yx = np.c_[rng.uniform(-3, 143, 400), rng.uniform(-3, 143, 400)]
    a, b = DRV.core_distances(cores, yx), C.core_distances(cores, yx)
    assert compare_value(a, b, EXACT, "core_distances") == [] and np.isfinite(a[:, 0]).sum() > 20 and np.isinf(a[:, 2]).any()   # branches hit
    assert np.array_equal(expand_labels(cores, distance=DRV.EXP_VARIANT_PX).astype(np.int32), S.expand_cores_variant(cores))


def test_assignment_block_equals_driver_block(frozen_seg_modules):
    DRV = frozen_seg_modules["REF-DRV"]
    lab, cores = _labels()
    lab_exp = S.expand_cores_variant(cores)
    rng = np.random.default_rng(1)
    n = 300
    y, x = rng.uniform(-2, 142, n), rng.uniform(-2, 142, n)
    cell_init = lab[np.clip(np.round(y).astype(int), 0, 139), np.clip(np.round(x).astype(int), 0, 139)]
    got = C.assign_use_coordinates(lab, lab_exp, cores, y, x, cell_init, 65.0)
    # the driver's block, copied
    Y, X = lab.shape
    yx = np.c_[np.clip(np.round(y), 0, Y - 1), np.clip(np.round(x), 0, X - 1)].astype(int)
    cell_base, cell_exp, in_core = lab[yx[:, 0], yx[:, 1]], lab_exp[yx[:, 0], yx[:, 1]], cores[yx[:, 0], yx[:, 1]] > 0
    cd = DRV.core_distances(cores, np.c_[y, x])
    cat = np.select([cell_base > 0, cell_exp > 0], ["in_baseline", "via_expansion"], "outside")
    exp = dict(cell_base=cell_base, cell_exp=cell_exp, in_core=in_core, core_d1_px=cd[:, 0], core_l1=cd[:, 1], core_d2_px=cd[:, 2],
               core_l2=cd[:, 3], assign_category=cat, ambiguous_cell=(cat != "outside") & ((cd[:, 2] - cd[:, 0]) < DRV.AMBIG_PX),
               cell_fit_vs_init_differs=(cell_base != cell_init), dist_to_baseline_mask_nm=distance_transform_edt(lab == 0)[yx[:, 0], yx[:, 1]] * 65.0)
    for key, v in exp.items():
        assert compare_value(v, got[key], EXACT, key) == [], key
    assert set(cat) == {"in_baseline", "via_expansion", "outside"} and exp["ambiguous_cell"].any() and not exp["ambiguous_cell"].all()


def test_per_spot_wiring_equals_frozen_call_sequence(frozen_modules, e_stats):
    """Pipeline columns vs the frozen M2.1 modules driven by the frozen driver's ``localize`` sequence (copied) on one batch."""
    LF, PF = frozen_modules["REF-LOC"], frozen_modules["REF-PHOT"]
    img = make_image()
    res = PL.analyze_fov(img, {"segmentation": "DAPI", "detection": ["Cy5"]}, fov_id="e")
    s = res["spots_by_channel"]["Cy5"]
    stack = img.channels["Cy5"]
    init = np.stack([s["z_init"], s["y_init"], s["x_init"]], 1).astype(np.int64)
    # --- frozen driver sequence (REF-DRV.localize), frozen modules, call-site scaling copied from the driver
    REF = np.array([150.0, 43.3333333, 43.3333333])
    rescale = lambda r: tuple(float(v * ref / w) for v, ref, w in zip(r, REF, VOX))     # noqa: E731
    AP, SHIN, SHOUT = (rescale(r) for r in ((3., 5., 5.), (5., 9., 9.), (7., 13., 13.)))
    CFG = dict(LF.DEFAULT_CFG)
    CFG["roi_half_xy"] = int(round(CFG["roi_half_xy"] * REF[1] / VOX[1]))
    CFG["mu_bound_xy"] *= REF[1] / VOX[1]
    CFG["large_shift_xy_px"] *= REF[1] / VOX[1]
    psf = D.psf_from_optics(OPTICS["NA"], OPTICS["emission_nm"], OPTICS["n_immersion"])
    sr = (psf["sigma_z_nm"], psf["sigma_xy_nm"], psf["sigma_xy_nm"])
    sig0 = tuple(r / v for r, v in zip(sr, VOX))
    nn_nm, _ = PF.nearest_neighbor_nm(init, VOX)
    ap_ov, nb_roi = PF.overlap_flags(init, AP, (CFG["roi_half_z"], CFG["roi_half_xy"], CFG["roi_half_xy"]))
    occ = PF.build_occupancy(stack.shape, init, AP)
    prim = PF.measure_spots(stack, init.astype(float), AP, SHIN, SHOUT, occ=occ)
    fits = [LF.fit_spot(stack, init[i], sig0, prim["local_background"][i], prim["peak_net"][i], prim["bg_noise_std"][i], CFG, VOX) for i in range(len(init))]
    fc = np.array([[f["z_fit"], f["y_fit"], f["x_fit"]] for f in fits])
    dpx = fc - init
    qual = [LF.classify_quality(fits[i], np.hypot(dpx[i, 1], dpx[i, 2]), abs(dpx[i, 0]), prim["snr_peak_legacy_fitqc"][i], int(nb_roi[i]), CFG)
            for i in range(len(init))]
    problems = []
    for j, ax in enumerate("zyx"):
        problems += compare_value(fc[:, j], s[f"{ax}_fit"], POS, f"{ax}_fit", e_stats)
    problems += compare_value(np.array([q for q, _ in qual], dtype=object), s["localization_quality"], EXACT, "quality")
    problems += compare_value(np.array([";".join(w) for _, w in qual], dtype=object), s["localization_flag_reasons"], EXACT, "reasons")
    for key, col in (("snr_primary", "snr_primary"), ("snr_peak_legacy_fitqc", "snr_peak_legacy_fitqc"), ("peak_net", "peak_net"),
                     ("local_background", "local_background"), ("aperture_sum", "aperture_sum"), ("net_integrated", "net_integrated"),
                     ("bg_noise_std", "bg_noise_std"), ("n_aperture_vox", "n_aperture_vox")):
        problems += compare_value(prim[key], s[col], PHOT_SPEC[key], key, e_stats)
    problems += compare_value(nn_nm, s["nearest_spot_distance_nm"], DER, "nn", e_stats) + compare_value(nb_roi, s["neighbor_in_roi_n"], EXACT, "nb")
    problems += compare_value(ap_ov, s["aperture_overlap"], EXACT, "ap_ov")
    f_sigx = np.array([f["sigma_x_fit"] for f in fits]) * VOX[2]
    problems += compare_value(f_sigx, s["gaussian_sigma_x_nm"], LOC_FIT_SPEC["sigma_x_fit"], "sigma_x_nm", e_stats)
    assert not problems, problems[:8]
    assert len(init) == len(CY5) and {q for q, _ in qual} == {"good"}                                        # branch: real fits happened
