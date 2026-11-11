"""Whole-FOV 3D LoG spot detection with noise-normalised thresholds (``fixed_sigma`` mode; ``auto`` is checkpoint C).

Conventions: images are ``(Z, Y, X)``; coordinates ``(z, y, x)`` integer voxel indices (0-based voxel centres);
physical units via the voxel size ``(z, y, x)`` in nm. LoG is computed on a float32 copy of the raw counts; the
LoG response ``filt`` is float32 (Big-FISH negates the LoG and clips negative values to 0).

Provenance. Functions marked *verbatim* are copied unchanged from frozen reference ``REF-DET``
(sha256 8a33a51a20f994060f099f855c5f872949169ad3fc9d48c0870fbb43266da700): ``SIGMA_XY_COEF``, ``FWHM_TO_SIGMA``,
``psf_from_optics``, ``kernel_params``, ``log_and_local_max``, ``noise_sigma_from_positive_log``. The candidate
pool / score / pass / edge expressions are inline in the frozen per-FOV driver ``REF-DRV``; here they are
functions (``build_candidate_pool``, ``score_candidates``, ``pass_flags``, ``edge_flags``) with the same
expressions, labelled [D] (driver expression, unchanged). Everything else is [N] (new interface or guard, no
frozen counterpart): ``InvalidNoiseSigma``, ``detect_fixed_sigma``, ``candidates_table``, ``validate_candidates``,
``validate_run_modes``. Not ported: ``optics_from_nd2`` (belongs to io), ``count_curve``, ``local_shell_stats``
(diagnostics outside the driver chain). See docs/ALGORITHMS.md.

Thresholds are frozen constants: candidate pool ``6 * sigma_LoG`` (>), accepted ``log_over_sigma > k`` with strict
``>`` in float32, default ``k = 10``, ``KS = (6, 8, 10, 12, 14)``, ``edge_flag`` margin 10 px.
"""
import numpy as np

import bigfish
import bigfish.detection as bfd
import bigfish.stack as bfs
from bigfish.detection.utils import get_object_radius_pixel

from catkit.cells import assign_cells_init

SIGMA_XY_COEF = 0.21  # lateral Gaussian approximation of the Airy PSF: sigma = 0.21 * lambda / NA
FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))  # 2.3548

KS = (6, 8, 10, 12, 14)          # [D] REF-DRV: pass_fixed_{k}sigma levels
CANDIDATE_FLOOR_SIGMA = 6.0      # [D] REF-DRV: pool threshold 6 * sigma_LoG
DEFAULT_K = 10.0                 # [D] REF-DRV: accepted = log_over_sigma > 10
EDGE_MARGIN_PX = 10              # [D] REF-DRV: edge_flag margin
BASELINE_VOXEL_NM = (150.0, 65.0, 65.0)   # [N] v0.1 supports only this voxel size


def psf_from_optics(NA, emission_nm, n_immersion):
    """Provisional wide-field PSF estimates (nm). Lateral: sigma=0.21 lambda/NA. Axial: FWHM_z ~ 2 n lambda / NA^2 (paraxial
    approximation; the real wide-field axial PSF is not Gaussian) -> sigma_z = FWHM_z / 2.355."""
    sxy = SIGMA_XY_COEF * emission_nm / NA
    fwhm_z = 2.0 * n_immersion * emission_nm / NA ** 2
    return dict(sigma_xy_nm=sxy, sigma_z_nm=fwhm_z / FWHM_TO_SIGMA, fwhm_xy_nm=sxy * FWHM_TO_SIGMA, fwhm_z_nm=fwhm_z)


def kernel_params(voxel_size_nm, spot_radius_nm, ndim=3):
    """Pixel-space LoG sigma and minimum_distance exactly as detect_spots derives them, plus the effective local-max box."""
    log_sigma = get_object_radius_pixel(voxel_size_nm=voxel_size_nm, object_radius_nm=spot_radius_nm, ndim=ndim)
    min_dist = log_sigma
    box = tuple(int(2 * np.ceil(m) + 1) for m in min_dist)
    return dict(log_kernel_sigma_px=tuple(float(v) for v in log_sigma), minimum_distance_px=tuple(float(v) for v in min_dist),
                local_max_box_voxels=box)


def log_and_local_max(image_f32, log_sigma_px, min_dist_px):
    filt = bfs.log_filter(image_f32, tuple(log_sigma_px))  # float32 in -> float out, no integer rounding
    mask_lm = bfd.local_maximum_detection(filt, tuple(min_dist_px))
    return filt, mask_lm


def noise_sigma_from_positive_log(filt, outside_mask_2d, z_slices=None):
    """Robust sigma of the (unclipped) LoG noise estimated from the clipped response outside cells: for zero-mean Gaussian noise the
    positive half has median 0.6745*sigma. Used ONLY as a noise reference for thresholds; it removes no candidates."""
    vals = filt[:, outside_mask_2d] if z_slices is None else filt[z_slices][:, outside_mask_2d]
    pos = vals[vals > 0]
    return float(np.median(pos) / 0.6745) if pos.size else float("nan")


# ---------------------------------------------------------------- [D] driver expressions as functions
def build_candidate_pool(filt, mlm, sigma, floor_sigma=CANDIDATE_FLOOR_SIGMA):
    """[D] ``spots_thresholding(filt, mlm, floor_sigma * sigma, remove_duplicate=True)``; returns ``(spots, resp)``.

    ``spots`` are the coordinates **returned by the detector, in its order** (int64, ``(N, 3)``; duplicate local
    maxima are merged to their centroid and truncated). ``resp = filt[spots]`` is the response AT the returned
    coordinate (a merged centroid can sit where the response is lower, even 0). The detector's second return value
    (a mask) is deliberately ignored: after 3D de-duplication Big-FISH builds it with 2D indexing, so it does not
    describe the pool.
    """
    spots, _ = bfd.spots_thresholding(filt, mlm, floor_sigma * sigma, remove_duplicate=True)
    resp = filt[spots[:, 0], spots[:, 1], spots[:, 2]]
    return spots, resp


def score_candidates(resp, sigma):
    """[D] ``log_over_sigma = resp / sigma`` (float32 ``resp`` over a Python float: stays float32 under NumPy 2)."""
    return resp / sigma


def pass_flags(log_over_sigma, ks=KS):
    """[D] ``{k: log_over_sigma > k}``: strict ``>``, float32 comparison."""
    return {k: log_over_sigma > k for k in ks}


def edge_flags(spots, Y, X):
    """[D] initial coordinate within ``EDGE_MARGIN_PX`` of an xy border."""
    return (spots[:, 1] < 10) | (spots[:, 1] >= Y - 10) | (spots[:, 2] < 10) | (spots[:, 2] >= X - 10)


# ---------------------------------------------------------------- [N] new interface
class InvalidNoiseSigma(ValueError):
    """[N] sigma_LoG cannot be used (empty/degenerate noise region): the frozen code would silently give NaN/0 thresholds."""


def detect_fixed_sigma(stack, cell_labels, noise_outside, voxel_nm, optics, *, k=DEFAULT_K,
                       floor_sigma=CANDIDATE_FLOOR_SIGMA, fov_id="", channel=""):
    """[N] Frozen driver chain for one FOV x channel, ``fixed_sigma`` mode.

    ``stack``: raw ``(Z, Y, X)`` uint16 of ONE channel (kept as-is; float32 only as LoG input).
    ``cell_labels``: expanded ``(Y, X)`` labels (checkpoint A); ``noise_outside``: ``(Y, X)`` bool from
    ``segmentation.noise_mask`` (applied to every z plane). ``voxel_nm``: ``(z, y, x)`` nm, must be the baseline
    (150, 65, 65). ``optics``: ``dict(NA, emission_nm, n_immersion)`` of this channel.
    Returns ``(table, qc)``; ``table`` is the ``candidates.csv`` columns (``candidates_table``), ``qc`` the
    ``detection_qc`` dict. No candidate is dropped by cells; cell membership is a label.

    Guards (new): 3D uint16 input with ``Z >= 3``; baseline voxel (tolerance 1e-6 nm); ``k >= floor_sigma``;
    shapes of labels/mask match ``(Y, X)``; an empty noise region or a non-finite / non-positive sigma_LoG raises
    ``InvalidNoiseSigma``. Zero cells is NOT an error (whole frame is the noise region, frozen behaviour).
    """
    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError(f"channel stack must be 3D (Z, Y, X); got {stack.ndim}D")
    if stack.dtype != np.uint16:
        raise ValueError(f"channel stack must be uint16 (v0.1); got {stack.dtype}")
    Z, Y, X = stack.shape
    if Z < 3:
        raise ValueError(f"channel stack has Z={Z} planes; at least 3 are required")
    if len(voxel_nm) != 3 or any(abs(float(a) - b) > 1e-6 for a, b in zip(voxel_nm, BASELINE_VOXEL_NM)):
        raise ValueError(f"voxel size {tuple(voxel_nm)} nm (z, y, x) is not supported in v0.1 (only {BASELINE_VOXEL_NM})")
    if not np.isfinite(k) or k < floor_sigma:
        raise ValueError(f"k={k} must be finite and >= candidate floor ({floor_sigma} sigma)")
    cell_labels, noise_outside = np.asarray(cell_labels), np.asarray(noise_outside)
    if cell_labels.shape != (Y, X) or noise_outside.shape != (Y, X) or noise_outside.dtype != bool:
        raise ValueError("cell_labels and noise_outside must be (Y, X) arrays (noise_outside bool) matching the stack")
    if not noise_outside.any():
        raise InvalidNoiseSigma("noise region is empty (cells cover the whole frame after dilation): sigma_LoG is undefined")

    VOX = tuple(float(v) for v in voxel_nm)
    psf = psf_from_optics(optics["NA"], optics["emission_nm"], optics["n_immersion"])
    sr = (psf["sigma_z_nm"], psf["sigma_xy_nm"], psf["sigma_xy_nm"])
    kp = kernel_params(VOX, sr)
    filt, mlm = log_and_local_max(stack.astype(np.float32), kp["log_kernel_sigma_px"], kp["minimum_distance_px"])
    sigma = noise_sigma_from_positive_log(filt, noise_outside)
    if not (np.isfinite(sigma) and sigma > 0):
        raise InvalidNoiseSigma(f"sigma_LoG={sigma!r} is not finite and positive (no positive LoG response outside cells?)")

    spots, resp = build_candidate_pool(filt, mlm, sigma, floor_sigma)
    los = score_candidates(resp, sigma)
    passes = pass_flags(los)
    cid, dist = assign_cells_init(spots, cell_labels, VOX[1] / 1000.0)
    table = candidates_table(
        fov_id, channel, spots, resp, los, stack[spots[:, 0], spots[:, 1], spots[:, 2]].astype(float), cid, dist,
        edge_flags(spots, Y, X), passes, accepted=los > k, mode="fixed_sigma", candidate_floor_log=floor_sigma * sigma)
    bands = (6, 8, 10, 12, 14)
    edges = list(bands) + [np.inf]
    qc = dict(mode="fixed_sigma", k_sigma=float(k), candidate_floor_sigma=float(floor_sigma),
              threshold_value_log=float(floor_sigma * sigma), threshold_over_sigma=float(floor_sigma), sigma_log=float(sigma),
              n_sigma_samples=int((filt[:, noise_outside] > 0).sum()), background_fraction=float(noise_outside.mean()),
              log_kernel_sigma_px=kp["log_kernel_sigma_px"], min_distance_px=kp["minimum_distance_px"],
              local_max_box=kp["local_max_box_voxels"], log_input_dtype="float32", n_local_max_positive=int((mlm & (filt > 0)).sum()),
              n_candidates=int(len(spots)), n_accepted=int(table["accepted"].sum()),
              band_counts={f"{lo}-{hi}": int(((los > lo) & (los <= hi)).sum()) for lo, hi in zip(edges[:-1], edges[1:])},
              bigfish_version=bigfish.__version__, warnings=[])
    return table, qc


CANDIDATE_COLUMNS = (
    "fov_id", "channel", "spot_id", "z_init", "y_init", "x_init", "log_response", "log_over_sigma", "raw_peak", "cell",
    "dist_to_mask_um", "edge_flag", *[f"pass_fixed_{k}sigma" for k in KS], "accepted", "mode", "candidate_floor_log")


def candidates_table(fov_id, channel, spots, resp, log_over_sigma, raw_peak, cell, dist_to_mask_um, edge_flag, passes,
                     accepted, mode, candidate_floor_log):
    """[N] Column dict in ``candidates.csv`` order (docs/SCHEMAS.md section 10); ``spot_id = 1..N`` in detector order.

    ``pass_fixed_*`` columns exist for ``fixed_sigma`` only (other modes pass ``passes={}``).
    ``log_response`` and ``log_over_sigma`` stay float32.
    """
    n = len(spots)
    t = dict(fov_id=np.full(n, fov_id, dtype=object), channel=np.full(n, channel, dtype=object),
             spot_id=np.arange(1, n + 1), z_init=spots[:, 0], y_init=spots[:, 1], x_init=spots[:, 2],
             log_response=resp, log_over_sigma=log_over_sigma, raw_peak=raw_peak, cell=cell, dist_to_mask_um=dist_to_mask_um,
             edge_flag=edge_flag)
    for k in KS:
        if k in passes:
            t[f"pass_fixed_{k}sigma"] = passes[k]
    t.update(accepted=np.asarray(accepted), mode=np.full(n, mode, dtype=object),
             candidate_floor_log=np.full(n, candidate_floor_log, dtype=float))
    return t


def validate_candidates(table):
    """[N] Schema/count invariants of one FOV x channel x mode table; raises ``ValueError`` on the first violation."""
    n = len(table["spot_id"])
    if any(len(v) != n for v in table.values()):
        raise ValueError("candidate table columns have different lengths")
    if not np.array_equal(table["spot_id"], np.arange(1, n + 1)):
        raise ValueError("spot_id must be 1..N in detector order")
    for col in ("fov_id", "channel", "mode"):
        if n and len(set(table[col])) != 1:
            raise ValueError(f"{col} must be uniform within one candidate table")
    if n and table["mode"][0] not in ("fixed_sigma", "auto"):
        raise ValueError(f"unknown mode {table['mode'][0]!r}")
    if table["accepted"].dtype != bool or table["edge_flag"].dtype != bool:
        raise ValueError("accepted/edge_flag must be bool")
    for c in ("log_response", "log_over_sigma"):
        if table[c].dtype != np.float32:
            raise ValueError(f"{c} must be float32")
    if n and table["mode"][0] == "fixed_sigma":
        for k in KS:
            if not np.array_equal(table[f"pass_fixed_{k}sigma"], table["log_over_sigma"] > k):
                raise ValueError(f"pass_fixed_{k}sigma disagrees with log_over_sigma > {k}")
        if (table["accepted"] & ~table["pass_fixed_6sigma"]).any():
            raise ValueError("an accepted candidate must exceed the 6 sigma floor (k >= floor)")


def validate_run_modes(tables):
    """[N] One mode per (fov_id, channel) within a run: raise if two tables of the same FOV/channel have different modes."""
    seen = {}
    for t in tables:
        if not len(t["spot_id"]):
            continue
        key = (t["fov_id"][0], t["channel"][0])
        mode = t["mode"][0]
        if seen.setdefault(key, mode) != mode:
            raise ValueError(f"FOV/channel {key} appears with modes {seen[key]!r} and {mode!r} in one run")
