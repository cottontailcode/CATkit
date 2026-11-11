"""Model-independent 3D aperture intensity for detected spots, measured on RAW counts (uint16 (Z,Y,X)).

Definitions (all in voxel index space, coordinates (z,y,x), 0-based voxel centres; centres may be fractional):
  aperture  = voxels with ((z-cz)/rz)^2 + ((y-cy)/ry)^2 + ((x-cx)/rx)^2 <= 1   (ellipsoid, radii in voxels/slices)
  shell     = voxels with r_in < d <= r_out for the same normalised distance d (ellipsoidal annulus), minus the voxels of
              OTHER spots' apertures (same channel) and minus image-outside voxels
  local_background = MEAN of the shell voxels after iterative 3-sigma clipping (std-based clip, half-width >= 1 count). The mean is used instead
              of the median because counts are integers: a median of ~3000 integer voxels is quantised to whole counts (+-0.5 count x ~300
              aperture voxels = +-150 counts of net intensity), whereas the clipped mean has a standard error of ~0.02 count. The median is kept
              as `local_background_median`.
  bg_noise  = 1.4826*MAD of the clipped shell; if MAD == 0 (integer-quantised, low-count data) -> sample std (ddof=1) of the clipped
              shell; if that is also 0/undefined or the shell has < min_shell voxels -> NaN (flag noise_unreliable). NO noise floor.
  net_integrated = sum(aperture raw counts) - local_background * n_aperture_voxels     (counts*voxel; NOT clipped at 0)
  snr_primary    = net_integrated / (noise_std * sqrt(n_aperture_voxels))   (background-limited approximation; NOT calibrated photon SNR)
  snr_peak_primary = (max aperture voxel - local_background) / noise_std
Raw camera counts are not photons. Intensities of different channels must not be compared in absolute terms.

Provenance: ported verbatim from frozen reference REF-PHOT
(sha256 32b0531066c073f35b833b30ce298ab86604ad1a63d78b748b91ce53294bb1ea).
Only the module docstring and this provenance note differ; see docs/ALGORITHMS.md.
"""
import numpy as np
from scipy.spatial import cKDTree


def ellipsoid_voxels(center, r_out, shape, r_in=None):
    """In-bounds voxel index arrays (z,y,x) of the ellipsoid (or annulus if r_in given), plus the voxel count without image clipping."""
    c = np.asarray(center, float)
    ro = np.asarray(r_out, float)
    lo = np.floor(c - ro).astype(int)
    hi = np.ceil(c + ro).astype(int)
    zz, yy, xx = np.meshgrid(*[np.arange(lo[k], hi[k] + 1) for k in range(3)], indexing="ij")
    d2 = ((zz - c[0]) / ro[0]) ** 2 + ((yy - c[1]) / ro[1]) ** 2 + ((xx - c[2]) / ro[2]) ** 2
    sel = d2 <= 1.0
    if r_in is not None:
        ri = np.asarray(r_in, float)
        d2i = ((zz - c[0]) / ri[0]) ** 2 + ((yy - c[1]) / ri[1]) ** 2 + ((xx - c[2]) / ri[2]) ** 2
        sel &= d2i > 1.0
    n_full = int(sel.sum())
    inb = sel & (zz >= 0) & (zz < shape[0]) & (yy >= 0) & (yy < shape[1]) & (xx >= 0) & (xx < shape[2])
    return (zz[inb], yy[inb], xx[inb]), n_full


def build_occupancy(shape, centers, ap_radii):
    """uint8 volume counting, per voxel, how many spots' apertures contain it (used to keep other spots' signal out of a shell)."""
    occ = np.zeros(shape, np.uint8)
    for c in centers:
        idx, _ = ellipsoid_voxels(c, ap_radii, shape)
        occ[idx] += 1  # voxel indices are unique within one spot
    return occ


def clipped_background(v, sigma=3.0, maxiter=5, clip_floor=1.0):
    """-> dict(median, mean, noise_mad, noise_std, noise_legacy, n_kept, n_nonfinite).
    Clipping rule is unchanged from Phase 2: iterative, half-width max(sigma*std, clip_floor=1 count) around the median (counts are integers,
    so deviations < 1 count are not outliers). `noise_std` = sample std (ddof=1) of the RETAINED (post-clip) voxels = PRIMARY noise (Phase 3);
    `noise_mad` = 1.4826*MAD of the retained voxels = secondary diagnostic (0 on quantised low-count backgrounds; no fallback);
    `noise_legacy` = Phase-2 rule (MAD, else std if MAD==0, else NaN), kept only to freeze the Phase-2 fit-quality categories.
    Non-finite voxels are dropped and counted. No numerical floor is ever applied: undefined/zero noise stays NaN/0 and is flagged by the caller."""
    v = np.asarray(v, float)
    fin = np.isfinite(v)
    n_nonfinite = int((~fin).sum())
    v = v[fin]
    nan = float("nan")
    if v.size == 0:
        return dict(median=nan, mean=nan, noise_mad=nan, noise_std=nan, noise_legacy=nan, n_kept=0, n_nonfinite=n_nonfinite)
    for _ in range(maxiter):
        m, s = np.median(v), v.std()
        if s == 0:
            break
        keep = np.abs(v - m) <= max(sigma * s, clip_floor)
        if keep.all():
            break
        v = v[keep]
    med = float(np.median(v))
    mad = float(1.4826 * np.median(np.abs(v - med)))
    sd = float(v.std(ddof=1)) if v.size > 2 else nan
    if mad > 0:
        legacy = mad
    elif np.isfinite(sd) and sd > 0:
        legacy = sd
    else:
        legacy = nan
    return dict(median=med, mean=float(v.mean()), noise_mad=mad, noise_std=sd, noise_legacy=legacy, n_kept=int(v.size), n_nonfinite=n_nonfinite)


def measure_spots(image, centers, ap_radii, shell_in, shell_out, occ=None, min_shell=100, clip_sigma=3.0):
    """Aperture photometry of every centre on the raw image. Returns a dict of arrays (one entry per spot). Nothing is dropped:
    problems become NaN + boolean flags.
    PRIMARY SNR (Phase 3): snr_primary = net/(noise_std*sqrt(n_vox)), snr_peak_primary = (peak - bg)/noise_std, noise_std = sample std (ddof=1) of the
    retained background voxels. SECONDARY: snr_mad / snr_peak_mad with 1.4826*MAD (NaN when MAD == 0). snr_peak_legacy_fitqc uses the Phase-2
    noise rule and exists only because the Phase-2 low_snr_peak fit-quality criterion is frozen."""
    shape = image.shape
    n = len(centers)
    keys = ["aperture_sum", "n_aperture_vox", "n_aperture_vox_full", "local_background", "local_background_median",
            "bg_noise_std", "bg_noise_mad", "bg_noise_legacy", "n_bg_voxels_valid", "n_shell_vox", "n_shell_excluded", "n_shell_nonfinite",
            "net_integrated", "peak_raw", "peak_net", "snr_primary", "snr_peak_primary", "snr_mad", "snr_peak_mad", "snr_peak_legacy_fitqc"]
    out = {k: np.full(n, np.nan) for k in keys}
    for k in ("aperture_truncated", "aperture_z_truncated", "shell_truncated", "noise_unreliable", "bg_unreliable",
              "bg_insufficient", "noise_nonfinite", "noise_zero", "mad_zero", "bg_nonfinite_voxels"):
        out[k] = np.zeros(n, bool)
    for i, c in enumerate(centers):
        ap, n_full = ellipsoid_voxels(c, ap_radii, shape)
        vals = image[ap].astype(np.float64)
        sh, sh_full = ellipsoid_voxels(c, shell_out, shape, r_in=shell_in)
        shv = image[sh].astype(np.float64)
        n_excl = 0
        if occ is not None and shv.size:
            bad = occ[sh] > 0
            n_excl = int(bad.sum())
            shv = shv[~bad]
        bg = clipped_background(shv, clip_sigma)
        out["n_aperture_vox"][i] = vals.size
        out["n_aperture_vox_full"][i] = n_full
        out["n_shell_vox"][i] = shv.size
        out["n_shell_excluded"][i] = n_excl
        out["n_shell_nonfinite"][i] = bg["n_nonfinite"]
        out["n_bg_voxels_valid"][i] = bg["n_kept"]
        out["bg_nonfinite_voxels"][i] = bg["n_nonfinite"] > 0
        out["aperture_truncated"][i] = vals.size < n_full
        out["aperture_z_truncated"][i] = bool(np.ceil(c[0] - ap_radii[0]) < 0 or np.floor(c[0] + ap_radii[0]) > shape[0] - 1)
        out["shell_truncated"][i] = len(sh[0]) < sh_full
        if bg["n_kept"] < min_shell or vals.size == 0:
            out["bg_unreliable"][i] = out["bg_insufficient"][i] = out["noise_unreliable"][i] = True
            out["aperture_sum"][i] = vals.sum() if vals.size else np.nan
            out["peak_raw"][i] = vals.max() if vals.size else np.nan
            continue
        sd, mad = bg["noise_std"], bg["noise_mad"]
        out["local_background"][i] = bg["mean"]
        out["local_background_median"][i] = bg["median"]
        out["bg_noise_std"][i], out["bg_noise_mad"][i], out["bg_noise_legacy"][i] = sd, mad, bg["noise_legacy"]
        out["noise_nonfinite"][i] = not np.isfinite(sd)
        out["noise_zero"][i] = bool(np.isfinite(sd) and sd == 0)
        out["mad_zero"][i] = bool(np.isfinite(mad) and mad == 0)
        out["noise_unreliable"][i] = not (np.isfinite(sd) and sd > 0)
        s = vals.sum()
        out["aperture_sum"][i] = s
        out["net_integrated"][i] = s - bg["mean"] * vals.size
        out["peak_raw"][i] = vals.max()
        out["peak_net"][i] = vals.max() - bg["mean"]
        rt = np.sqrt(vals.size)
        if np.isfinite(sd) and sd > 0:
            out["snr_primary"][i] = out["net_integrated"][i] / (sd * rt)
            out["snr_peak_primary"][i] = out["peak_net"][i] / sd
        if np.isfinite(mad) and mad > 0:
            out["snr_mad"][i] = out["net_integrated"][i] / (mad * rt)
            out["snr_peak_mad"][i] = out["peak_net"][i] / mad
        if np.isfinite(bg["noise_legacy"]):
            out["snr_peak_legacy_fitqc"][i] = out["peak_net"][i] / bg["noise_legacy"]
    return out


def nearest_neighbor_nm(coords_zyx_px, voxel_nm):
    """Nearest OTHER spot (3D, nm) among the given same-channel spots; NaN if only one spot."""
    xyz = np.asarray(coords_zyx_px, float) * np.asarray(voxel_nm, float)
    if len(xyz) < 2:
        return np.full(len(xyz), np.nan), np.full(len(xyz), -1)
    d, j = cKDTree(xyz).query(xyz, k=2)
    return d[:, 1], j[:, 1]


def overlap_flags(coords_zyx_px, ap_radii, roi_half):
    """aperture_overlap: another spot's centre within 2x the aperture ellipsoid (apertures can intersect).
    neighbor_in_roi: another spot's centre inside this spot's fit ROI box (|dz|<=roi_half[0], |dy|,|dx|<=roi_half[1])."""
    c = np.asarray(coords_zyx_px, float)
    ar = np.asarray(ap_radii, float)
    rh = np.asarray(roi_half, float)
    ap_ov = np.zeros(len(c), bool)
    nb_roi = np.zeros(len(c), int)
    for i in range(len(c)):
        d = c - c[i]
        d[i] = np.inf
        ap_ov[i] = bool((((d / (2 * ar)) ** 2).sum(1) <= 1.0).any())
        nb_roi[i] = int((np.abs(d) <= rh).all(1).sum())
    return ap_ov, nb_roi
