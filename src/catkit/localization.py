"""3D Gaussian subpixel localization of already-detected spots on RAW counts, plus a Big-FISH fit_subpixel() cross-check.

Model (voxel-integrated 3D Gaussian, same form as bigfish.detection.spot_modeling.gaussian_3d, but with independent sigma_x, sigma_y):
    I(k_z,k_y,k_x) = bg + A * Fz(k_z) * Fy(k_y) * Fx(k_x),   F_a(k) = sigma_a*sqrt(pi/2) * [erf((k+.5-mu_a)/(sqrt2 sigma_a)) - erf((k-.5-mu_a)/(sqrt2 sigma_a))]
  all in voxel index units (k = voxel/slice index, voxel width 1) => voxel centre convention, identical to the detection table (0-based, (z,y,x)).
  A  = peak height of the continuous Gaussian, in raw counts above bg (the voxel-integrated value at the centre is slightly lower).
  Gaussian integrated intensity = A * (2*pi)^1.5 * sigma_z*sigma_y*sigma_x  (counts*voxel, sigma in voxel/slice units), background excluded.
Parameters: (mu_z, mu_y, mu_x, sigma_z, sigma_y, sigma_x, A, bg); bounded least squares (trf). Convergence of the optimiser is NOT treated as
evidence of a real spot or an accurate position: localization_quality is assigned separately from bounds, shift, edges, neighbours and SNR.

Provenance: ported verbatim from frozen reference REF-LOC
(sha256 b596111de98c3d4b6f077a49c642f7d05a54542835b4e83319b483f985490136).
Only the module docstring and this provenance note differ; see docs/ALGORITHMS.md.
"""
import warnings

import numpy as np
from scipy.optimize import least_squares, OptimizeWarning
from scipy.special import erf

import bigfish.detection as bfd

PNAMES = ("mu_z", "mu_y", "mu_x", "sigma_z", "sigma_y", "sigma_x", "A", "bg")

DEFAULT_CFG = dict(
    roi_half_z=4,              # slices  (+-600 nm)
    roi_half_xy=7,             # pixels  (+-303 nm)
    mu_bound_z=2.0,            # slices: allowed |mu - init|
    mu_bound_xy=2.0,           # pixels
    sigma_xy_bounds_nm=(60.0, 400.0),
    sigma_z_bounds_nm=(150.0, 1000.0),
    A_max=1.0e5, bg_max=1.0e3,
    max_nfev=300,
    min_roi_voxels=60,         # need >> 8 free parameters
    cond_max=1.0e6,            # Jacobian condition number above which uncertainties are not reported
    large_shift_xy_px=1.5, large_shift_z_slices=1.5,
    low_snr_peak=3.0,
)


def _axis(k, mu, sigma):
    s2 = np.sqrt(2.0) * sigma
    return sigma * np.sqrt(np.pi / 2.0) * (erf((k + 0.5 - mu) / s2) - erf((k - 0.5 - mu) / s2))


def gaussian_model(p, kz, ky, kx):
    """p = (mu_z, mu_y, mu_x, sigma_z, sigma_y, sigma_x, A, bg); kz,ky,kx are 1-D index vectors. Returns (nz,ny,nx)."""
    mz, my, mx, sz, sy, sx, A, bg = p
    return bg + A * _axis(kz, mz, sz)[:, None, None] * _axis(ky, my, sy)[None, :, None] * _axis(kx, mx, sx)[None, None, :]


def gaussian_integral_voxels(A, sz, sy, sx):
    return A * (2 * np.pi) ** 1.5 * sz * sy * sx


def roi_bounds(init_zyx, shape, cfg):
    z0, y0, x0 = [int(round(float(v))) for v in init_zyx]
    lo = (max(0, z0 - cfg["roi_half_z"]), max(0, y0 - cfg["roi_half_xy"]), max(0, x0 - cfg["roi_half_xy"]))
    hi = (min(shape[0], z0 + cfg["roi_half_z"] + 1), min(shape[1], y0 + cfg["roi_half_xy"] + 1), min(shape[2], x0 + cfg["roi_half_xy"] + 1))
    return lo, hi


def _nan_result():
    r = {f"{n}_fit": np.nan for n in PNAMES}
    r.update(dict(z_fit=np.nan, y_fit=np.nan, x_fit=np.nan, optimizer_converged=False, status=-99, message="", nfev=0,
                  rss=np.nan, rms=np.nan, r2=np.nan, reduced_chi2=np.nan, jac_cond=np.nan, jac_rank=0, n_roi_vox=0,
                  se_z_px=np.nan, se_y_px=np.nan, se_x_px=np.nan, se_A=np.nan, params_at_bound="", se_reported=False,
                  center_out_of_roi=False, nonfinite=False, roi_z_truncated=False, roi_xy_truncated=False, fit_error=""))
    return r


def fit_spot(image, init_zyx, sigma0_px, bg0, A0, noise, cfg=DEFAULT_CFG, voxel_nm=(150.0, 43.333, 43.333)):
    """Fit one spot. sigma0_px = (sz, sy, sx) start values (voxel/slice units), bg0/A0 start values, noise = bg noise (counts) or NaN.
    Never raises: failures are returned as fit_error / optimizer_converged=False with NaN fields."""
    r = _nan_result()
    try:
        lo, hi = roi_bounds(init_zyx, image.shape, cfg)
        z0, y0, x0 = [int(round(float(v))) for v in init_zyx]
        r["roi_z_truncated"] = (z0 - cfg["roi_half_z"] < 0) or (z0 + cfg["roi_half_z"] > image.shape[0] - 1)
        r["roi_xy_truncated"] = (y0 - cfg["roi_half_xy"] < 0) or (x0 - cfg["roi_half_xy"] < 0) or \
            (y0 + cfg["roi_half_xy"] > image.shape[1] - 1) or (x0 + cfg["roi_half_xy"] > image.shape[2] - 1)
        roi = image[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float64)
        r["n_roi_vox"] = int(roi.size)
        if roi.size < cfg["min_roi_voxels"]:
            r["fit_error"] = "roi_too_small"
            return r
        if not np.all(np.isfinite(roi)):
            r["fit_error"] = "nonfinite_roi"
            r["nonfinite"] = True
            return r
        kz, ky, kx = (np.arange(lo[i], hi[i], dtype=float) for i in range(3))
        init = np.array([float(v) for v in init_zyx])
        szb = np.array(cfg["sigma_z_bounds_nm"]) / voxel_nm[0]
        sxyb = np.array(cfg["sigma_xy_bounds_nm"]) / voxel_nm[1]
        lower = np.array([init[0] - cfg["mu_bound_z"], init[1] - cfg["mu_bound_xy"], init[2] - cfg["mu_bound_xy"],
                          szb[0], sxyb[0], sxyb[0], 0.0, 0.0])
        upper = np.array([init[0] + cfg["mu_bound_z"], init[1] + cfg["mu_bound_xy"], init[2] + cfg["mu_bound_xy"],
                          szb[1], sxyb[1], sxyb[1], cfg["A_max"], cfg["bg_max"]])
        s0 = np.array(sigma0_px, float)
        s0 = np.clip(s0, [szb[0], sxyb[0], sxyb[0]], [szb[1], sxyb[1], sxyb[1]])
        bg0 = float(np.clip(bg0 if np.isfinite(bg0) else np.median(roi), 0.0, cfg["bg_max"]))
        A0 = float(np.clip(A0 if np.isfinite(A0) else roi.max() - bg0, 0.1, cfg["A_max"]))
        p0 = np.array([*init, *s0, A0, bg0])
        p0 = np.clip(p0, lower + 1e-9, upper - 1e-9)

        def resid(p):
            return (gaussian_model(p, kz, ky, kx) - roi).ravel()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            res = least_squares(resid, p0, bounds=(lower, upper), method="trf", x_scale="jac", max_nfev=cfg["max_nfev"])
        p = res.x
        r["status"], r["message"], r["nfev"] = int(res.status), str(res.message), int(res.nfev)
        r["optimizer_converged"] = bool(res.success and res.status > 0)
        for n, v in zip(PNAMES, p):
            r[f"{n}_fit"] = float(v)
        r["z_fit"], r["y_fit"], r["x_fit"] = float(p[0]), float(p[1]), float(p[2])
        if not np.all(np.isfinite(p)):
            r["nonfinite"] = True
            return r
        r["center_out_of_roi"] = not (lo[0] - 0.5 <= p[0] <= hi[0] - 0.5 and lo[1] - 0.5 <= p[1] <= hi[1] - 0.5 and lo[2] - 0.5 <= p[2] <= hi[2] - 0.5)
        n = roi.size
        rss = float(np.sum(res.fun ** 2))
        dof = n - len(p)
        sst = float(np.sum((roi - roi.mean()) ** 2))
        r["rss"], r["rms"] = rss, float(np.sqrt(rss / n))
        r["r2"] = 1.0 - rss / sst if sst > 0 else np.nan
        r["reduced_chi2"] = (rss / dof) / noise ** 2 if (np.isfinite(noise) and noise > 0 and dof > 0) else np.nan
        am = np.asarray(res.active_mask)
        r["params_at_bound"] = ",".join(f"{PNAMES[i]}{'_lo' if am[i] < 0 else '_hi'}" for i in range(len(p)) if am[i] != 0)
        # Jacobian conditioning -> parameter uncertainties only if well conditioned and no parameter sits on a bound
        J = res.jac
        colnorm = np.linalg.norm(J, axis=0)
        colnorm[colnorm == 0] = 1.0
        sv = np.linalg.svd(J / colnorm, compute_uv=False)
        r["jac_rank"] = int((sv > sv.max() * 1e-10).sum())
        r["jac_cond"] = float(sv.max() / sv.min()) if sv.min() > 0 else np.inf
        if r["jac_cond"] < cfg["cond_max"] and r["jac_rank"] == len(p) and not r["params_at_bound"] and dof > 0:
            cov = (rss / dof) * np.linalg.inv(J.T @ J)
            se = np.sqrt(np.clip(np.diag(cov), 0, None))
            r["se_z_px"], r["se_y_px"], r["se_x_px"], r["se_A"] = float(se[0]), float(se[1]), float(se[2]), float(se[6])
            r["se_reported"] = True
        # (kz,ky,kx) kept for caller-side model evaluation
        r["_roi_lo"], r["_roi_hi"] = lo, hi
    except Exception as e:  # noqa: BLE001 - failures are recorded, not raised
        r["fit_error"] = f"{type(e).__name__}: {e}"[:120]
    return r


def run_bigfish_fit_subpixel(image, coords_int, voxel_nm, spot_radius_nm):
    """bigfish.detection.fit_subpixel per spot on the raw image. Returns (N,3) float coords (NaN if the call raised) and error strings.
    NOTE: Big-FISH silently returns the input coordinates when the fit raises RuntimeError or the centre leaves the crop, so
    'unchanged' is reported as bf_unchanged_or_fallback (it cannot distinguish a genuine zero shift from a fallback)."""
    n = len(coords_int)
    out = np.full((n, 3), np.nan)
    err = [""] * n
    c = np.asarray(coords_int, np.int64)
    for i in range(n):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[i] = bfd.fit_subpixel(image, c[i:i + 1], tuple(voxel_nm), tuple(spot_radius_nm))[0]
        except Exception as e:  # noqa: BLE001
            err[i] = f"{type(e).__name__}: {e}"[:120]
    unchanged = np.all(out == c, axis=1)
    return out, unchanged, err


def classify_quality(r, shift_xy_px, shift_z, snr_peak, neighbor_in_roi, cfg=DEFAULT_CFG):
    """-> (localization_quality, reasons). 'unreliable': fit unusable (position falls back to init). 'suspect': fit finite but doubtful.
    'good': none of the checks fired. Convergence alone never yields 'good'."""
    bad, doubt = [], []
    if r["fit_error"]:
        bad.append("fit_error")
    if not r["optimizer_converged"]:
        bad.append("not_converged")
    if r["nonfinite"]:
        bad.append("nonfinite")
    if r["center_out_of_roi"]:
        bad.append("center_out_of_roi")
    if np.isfinite(r["A_fit"]) and r["A_fit"] <= 1e-6:
        bad.append("amp_nonpositive")
    pb = r["params_at_bound"]
    if any(k in pb for k in ("mu_z", "mu_y", "mu_x")):
        bad.append("mu_at_bound")
    if any(k in pb for k in ("sigma_",)):
        doubt.append("sigma_at_bound")
    if "A_hi" in pb or "bg_" in pb:
        doubt.append("A_or_bg_at_bound")
    if np.isfinite(shift_xy_px) and shift_xy_px > cfg["large_shift_xy_px"]:
        doubt.append("large_shift_xy")
    if np.isfinite(shift_z) and shift_z > cfg["large_shift_z_slices"]:
        doubt.append("large_shift_z")
    if r["roi_z_truncated"]:
        doubt.append("z_edge")
    if r["roi_xy_truncated"]:
        doubt.append("xy_edge")
    if neighbor_in_roi > 0:
        doubt.append("neighbor_in_roi")
    if not r["se_reported"] and not bad:
        doubt.append("uncertainty_unreliable")
    if not np.isfinite(snr_peak):
        doubt.append("noise_unreliable")
    elif snr_peak < cfg["low_snr_peak"]:
        doubt.append("low_snr_peak")
    q = "unreliable" if bad else ("suspect" if doubt else "good")
    return q, bad + doubt
