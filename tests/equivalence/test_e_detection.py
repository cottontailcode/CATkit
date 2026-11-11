"""E-layer equivalence for M2.2 checkpoint B (fixed_sigma detection): frozen ``REF-DET`` helpers and the
driver's inline pool/score/pass/edge expressions vs ``catkit.detection``.

The driver expressions are replayed in ``frozen_chain`` below (copied from the frozen driver; the driver is not
imported). [N] interface guards and table validators are tested in ``tests/synthetic/test_detection_synthetic.py``.
Equivalence to the frozen code is not evidence of real-data detection accuracy.
"""

from __future__ import annotations

import inspect

import pytest

from scientific import require_modules

require_modules("numpy", "scipy", "skimage", "bigfish")

import numpy as np  # noqa: E402
import bigfish.detection as bfd  # noqa: E402

import catkit.detection as D  # noqa: E402
import catkit.segmentation as S  # noqa: E402
from compare import EXACT, compare_value  # noqa: E402
from synth_det import OPTICS, SHAPE, VOX, make_channel  # noqa: E402

pytestmark = pytest.mark.research

VERBATIM = ["psf_from_optics", "kernel_params", "log_and_local_max", "noise_sigma_from_positive_log"]


def test_port_is_verbatim(frozen_det_modules):
    B = frozen_det_modules["REF-DET"]
    for name in VERBATIM:
        assert inspect.getsource(getattr(D, name)) == inspect.getsource(getattr(B, name)), name
    assert D.SIGMA_XY_COEF == B.SIGMA_XY_COEF and D.FWHM_TO_SIGMA == B.FWHM_TO_SIGMA


OPT_CASES = [(1.4, 580.0, 1.515), (1.45, 670.0, 1.515), (1.2, 461.0, 1.33), (0.95, 520.0, 1.0)]


def test_psf_and_kernel_equivalence(frozen_det_modules):
    B = frozen_det_modules["REF-DET"]
    for na, em, n in OPT_CASES:
        a, b = B.psf_from_optics(na, em, n), D.psf_from_optics(na, em, n)
        assert a == b and list(a) == list(b)
        sr = (a["sigma_z_nm"], a["sigma_xy_nm"], a["sigma_xy_nm"])
        for vox in (VOX, (150.0, 43.3333333, 43.3333333), (200.0, 100.0, 100.0)):
            ka, kb = B.kernel_params(vox, sr), D.kernel_params(vox, sr)
            assert ka == kb
    # independent oracle at 65 nm voxels (hand arithmetic): sigma_xy = 0.21*580/1.4 = 87 nm ; FWHM_z = 2*1.515*580/1.96
    p = D.psf_from_optics(1.4, 580.0, 1.515)
    assert abs(p["sigma_xy_nm"] - 87.0) < 1e-9 and abs(p["fwhm_z_nm"] - 2 * 1.515 * 580.0 / 1.96) < 1e-9
    assert abs(p["sigma_z_nm"] - p["fwhm_z_nm"] / 2.3548) < 0.01


def _fovs():
    spots = [(8, 30, 30, 200), (8, 60, 70, 120), (8, 50, 5, 150), (8, 10, 60, 60), (3, 80, 80, 90), (14, 85, 20, 70), (8, 30, 32, 100)]
    lab = np.zeros(SHAPE[1:], np.int32)
    lab[20:40, 20:40] = 1
    lab[50:70, 60:80] = 2
    return {
        "spots_two_cells": (make_channel(spots), lab),
        "noise_only": (make_channel((), seed=3), lab),
        "no_cells": (make_channel(spots[:3], seed=4), np.zeros(SHAPE[1:], np.int32)),
        "dim_spots_low_noise": (make_channel([(8, 40, 40, 25), (8, 70, 25, 18)], noise=2.0, seed=5), lab),
    }


def frozen_chain(B, stack, lab, vox, opt, px_um):
    """The frozen driver chain for one channel (REF-DET helpers + the driver's inline expressions, copied)."""
    from scipy.ndimage import binary_dilation
    Z, Y, X = stack.shape
    outside = ~binary_dilation(lab > 0, iterations=10)
    psf = B.psf_from_optics(opt["NA"], opt["emission_nm"], opt["n_immersion"])
    sr = (psf["sigma_z_nm"], psf["sigma_xy_nm"], psf["sigma_xy_nm"])
    kp = B.kernel_params(vox, sr)
    filt, mlm = B.log_and_local_max(stack.astype(np.float32), kp["log_kernel_sigma_px"], kp["minimum_distance_px"])
    sigma = B.noise_sigma_from_positive_log(filt, outside)
    spots, _ = bfd.spots_thresholding(filt, mlm, 6 * sigma, remove_duplicate=True)
    resp = filt[spots[:, 0], spots[:, 1], spots[:, 2]]
    cid, dist = B.assign_cells(spots, lab, px_um)
    los = resp / sigma
    out = dict(filt=filt, mlm=mlm, sigma=sigma, spots=spots, resp=resp, los=los, cell=cid, dist=dist,
               raw=stack[spots[:, 0], spots[:, 1], spots[:, 2]].astype(float),
               edge=((spots[:, 1] < 10) | (spots[:, 1] >= Y - 10) | (spots[:, 2] < 10) | (spots[:, 2] >= X - 10)))
    for k in (6, 8, 10, 12, 14):
        out[f"pass{k}"] = los > k
    return out


def test_detection_chain_equivalence(frozen_det_modules, e_stats):
    B = frozen_det_modules["REF-DET"]
    problems, counts = [], {}
    for name, (stack, lab) in _fovs().items():
        fr = frozen_chain(B, stack, lab, VOX, OPTICS, 0.065)
        mask = S.noise_mask(lab, 10)
        table, qc = D.detect_fixed_sigma(stack, lab, mask, VOX, OPTICS, k=10.0, fov_id="f1", channel="c1")
        D.validate_candidates(table)
        # stage-wise: the port's helpers on the frozen LoG output (so a later-stage difference is not hidden by an earlier one)
        filt, mlm = D.log_and_local_max(stack.astype(np.float32), *_kp(D))
        problems += compare_value(fr["filt"], filt, EXACT, f"{name}.filt", e_stats) + compare_value(fr["mlm"], mlm, EXACT, f"{name}.mlm")
        assert fr["sigma"] == qc["sigma_log"] and type(fr["sigma"]) is type(qc["sigma_log"]), name
        for key, col in (("spots", None), ("resp", "log_response"), ("los", "log_over_sigma"), ("cell", "cell"), ("raw", "raw_peak"),
                         ("edge", "edge_flag")):
            got = np.stack([table["z_init"], table["y_init"], table["x_init"]], 1) if key == "spots" else table[col]
            problems += compare_value(fr[key], got, EXACT, f"{name}.{key}", e_stats)
        problems += compare_value(fr["dist"], table["dist_to_mask_um"], EXACT, f"{name}.dist", e_stats)
        for k in (6, 8, 10, 12, 14):
            problems += compare_value(fr[f"pass{k}"], table[f"pass_fixed_{k}sigma"], EXACT, f"{name}.pass{k}")
        problems += compare_value(fr["los"] > 10.0, table["accepted"], EXACT, f"{name}.accepted")
        assert np.array_equal(table["spot_id"], np.arange(1, len(fr["spots"]) + 1)) and (table["candidate_floor_log"] == 6 * fr["sigma"]).all()
        counts[name] = (len(fr["spots"]), int((fr["los"] > 10).sum()))
        assert qc["n_candidates"] == counts[name][0] and qc["n_accepted"] == counts[name][1]
        # port helpers vs the same inline expressions (dtype included)
        sp2, r2 = D.build_candidate_pool(fr["filt"], fr["mlm"], fr["sigma"])
        problems += compare_value(fr["spots"], sp2, EXACT, f"{name}.pool") + compare_value(fr["resp"], r2, EXACT, f"{name}.pool_resp")
    assert not problems, problems[:10]
    # branch triggers: planted spots found, some in cells; pure noise gives few/no candidates; some but not all candidates accepted
    assert counts["spots_two_cells"][1] >= 5 and counts["noise_only"][0] <= counts["spots_two_cells"][0]
    assert counts["no_cells"][1] >= 2
    n_c, n_a = counts["dim_spots_low_noise"]
    assert n_c >= n_a


def _kp(M):
    psf = M.psf_from_optics(**{"NA": OPTICS["NA"], "emission_nm": OPTICS["emission_nm"], "n_immersion": OPTICS["n_immersion"]})
    kp = M.kernel_params(VOX, (psf["sigma_z_nm"], psf["sigma_xy_nm"], psf["sigma_xy_nm"]))
    return kp["log_kernel_sigma_px"], kp["minimum_distance_px"]


def test_sigma_with_z_slices_equivalence(frozen_det_modules):
    B = frozen_det_modules["REF-DET"]
    stack, lab = _fovs()["spots_two_cells"]
    filt, _ = B.log_and_local_max(stack.astype(np.float32), *_kp(D))
    mask = S.noise_mask(lab, 10)
    for zs in (None, slice(0, 5), [2, 5, 8]):
        assert B.noise_sigma_from_positive_log(filt, mask, zs) == D.noise_sigma_from_positive_log(filt, mask, zs)
    assert np.isnan(D.noise_sigma_from_positive_log(np.zeros_like(filt), mask))      # no positive response -> NaN (frozen)
    assert np.isnan(B.noise_sigma_from_positive_log(np.zeros_like(filt), mask))
