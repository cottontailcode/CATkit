"""Single-FOV analysis: ND2 -> nuclear-stain Otsu segmentation -> fixed_sigma 3D LoG detection -> Gaussian localization
-> aperture photometry -> cell assignment -> ``spots.csv`` (+ ``candidates.csv``, ``run_metadata.json``).

This module only wires validated pieces together; it contains no new scientific algorithm. The call sequence per
detection channel is the one of the frozen per-FOV driver ``REF-DRV`` (``localize`` and the assignment block):

    accepted rows (initial integer ``(z, y, x)`` of ``detect_fixed_sigma``)
      -> nearest_neighbor_nm / overlap_flags (INITIAL coordinates, within the accepted batch)
      -> build_occupancy -> measure_spots(init.astype(float), AP, SHIN, SHOUT, occ)
      -> fit_spot(init, sigma0 = PSF/voxel, bg0 = local_background, A0 = peak_net, noise = bg_noise_std, CFG, VOX)
      -> classify_quality(fit, shift_xy, shift_z, snr_peak_legacy_fitqc, neighbor_in_roi, CFG)
      -> *_use = fit if quality in {good, suspect} else initial; cell_base = labels[round(use)] (numpy half-to-even, clipped)

Labels: [D] = driver expression/behaviour unchanged; [N] = new (explicit channel roles, scaling helpers moved into
``src``, table assembly, CSV/JSON writing, count reconciliation). Voxel and optics come from the file header; only the
baseline voxel ``(150, 65, 65)`` nm is supported and nothing is resampled or overridden. Channel roles are given by the
caller (``{"segmentation": name, "detection": [names]}``); target identity is never inferred from names.
"""
import csv
import json
import math
import os
import platform
import shutil
import tempfile
import time
import warnings
from pathlib import Path

import nd2
import numpy as np
import scipy
import skimage

import bigfish
import catkit
import catkit.cells as C
import catkit.detection as D
import catkit.localization as L
import catkit.photometry as P
import catkit.qc as QC
import catkit.segmentation as S
from catkit.io import ImageStack, load_nd2_channels
from catkit.provenance import sha256_file

SCHEMA_VERSION = "catkit-0.1-draft"

# ---- [D] REF-DRV constants and call-site scaling (reference scale of aperture/shell/ROI/mu-bound parameters)
REF_VOX = (150.0, 43.3333333, 43.3333333)
AP_REF, SHIN_REF, SHOUT_REF = (3.0, 5.0, 5.0), (5.0, 9.0, 9.0), (7.0, 13.0, 13.0)


def rescale_radii(radii, voxel):
    """[D] ``r * REF_VOX / voxel`` per axis (the driver's ``rescale`` lambda); identical to the tested helper in ``tests/synth.py``."""
    return tuple(float(v * ref / w) for v, ref, w in zip(radii, REF_VOX, voxel))


def rescaled_cfg(cfg, voxel):
    """[D] Fit configuration at ``voxel``: xy ROI half-width, ``mu`` bound and large-shift scaled by ``REF_VOX[1] / voxel[1]``."""
    out = dict(cfg)
    out["roi_half_xy"] = int(round(out["roi_half_xy"] * REF_VOX[1] / voxel[1]))
    out["mu_bound_xy"] *= REF_VOX[1] / voxel[1]
    out["large_shift_xy_px"] *= REF_VOX[1] / voxel[1]
    return out


SPOT_COLUMNS = (
    # identity / provenance
    "fov_id", "condition", "channel", "exposure_ms_metadata", "spot_id", "localized_batch",
    # coordinates
    "z_init", "y_init", "x_init", "z_fit", "y_fit", "x_fit", "x_init_um", "y_init_um", "z_init_um",
    "x_fit_um", "y_fit_um", "z_fit_um", "x_use", "y_use", "z_use", "coord_source",
    # detection
    "log_over_sigma", "threshold_mode",
    # fit
    "optimizer_converged", "gaussian_sigma_x_nm", "gaussian_sigma_y_nm", "gaussian_sigma_z_nm", "gaussian_amplitude",
    "gaussian_background", "gaussian_fit_r2", "jac_cond", "params_at_bound", "fit_error", "se_z_nm", "se_y_nm", "se_x_nm",
    "se_reported", "shift_xy_px", "shift_z_slices",
    # quality
    "localization_quality", "localization_flag_reasons", "flag_z_edge", "flag_large_shift", "flag_neighbor_in_roi",
    "flag_sigma_at_bound", "tier_frozen",
    # photometry
    "snr_primary", "snr_peak_legacy_fitqc", "peak_net", "local_background", "aperture_sum", "net_integrated",
    "n_aperture_vox", "bg_noise_std",
    # flags
    "roi_z_truncated", "roi_xy_truncated", "aperture_z_truncated", "aperture_truncated", "edge_flag",
    "nearest_spot_distance_nm", "neighbor_in_roi_n", "aperture_overlap",
    # assignment
    "cell_init", "cell_base", "cell_exp", "in_core", "core_d1_px", "core_l1", "core_d2_px", "core_l2", "assign_category",
    "ambiguous_cell", "cell_fit_vs_init_differs", "dist_to_baseline_mask_nm",
)


class PipelineError(ValueError):
    """Invalid pipeline input (roles, channels) or a failed internal consistency check."""


class OutputError(PipelineError):
    """The output destination is unsafe, or the outputs could not be written (nothing is published in either case)."""


def _localize(stack, init, VOX, sr, AP, SHIN, SHOUT, CFG):
    """[D] The driver's ``localize`` for one channel's accepted batch. ``init`` is ``(n, 3)`` int64 ``(z, y, x)``.

    Returns ``(fits, prim, nn_nm, ap_ov, nb_roi, quality, reasons)``; ``n == 0`` returns empty arrays (the frozen code never
    runs with zero accepted spots; an empty batch simply has no rows)."""
    n = len(init)
    if n == 0:
        return [], None, np.empty(0), np.empty(0, bool), np.empty(0, int), [], []
    sig0 = tuple(r / v for r, v in zip(sr, VOX))
    nn_nm, _ = P.nearest_neighbor_nm(init, VOX)
    ap_ov, nb_roi = P.overlap_flags(init, AP, (CFG["roi_half_z"], CFG["roi_half_xy"], CFG["roi_half_xy"]))
    occ = P.build_occupancy(stack.shape, init, AP)
    prim = P.measure_spots(stack, init.astype(float), AP, SHIN, SHOUT, occ=occ)
    del occ
    fits = [L.fit_spot(stack, init[i], sig0, prim["local_background"][i], prim["peak_net"][i], prim["bg_noise_std"][i], CFG, VOX)
            for i in range(n)]
    fc = np.array([[f["z_fit"], f["y_fit"], f["x_fit"]] for f in fits], float)
    dpx = fc - init
    qual, why = zip(*[L.classify_quality(fits[i], np.hypot(dpx[i, 1], dpx[i, 2]), abs(dpx[i, 0]), prim["snr_peak_legacy_fitqc"][i],
                                         int(nb_roi[i]), CFG) for i in range(n)])
    return fits, prim, nn_nm, ap_ov, nb_roi, list(qual), list(why)


def _spots_for_channel(stack, cand, VOX, sr, AP, SHIN, SHOUT, CFG, k, fov_id, condition, channel, exposure_ms,
                       labels, labels_exp, cores):
    """[N] Assemble the ``spots.csv`` columns of one channel (accepted candidates only, in candidate order)."""
    acc = np.flatnonzero(cand["accepted"])
    init = np.stack([cand["z_init"][acc], cand["y_init"][acc], cand["x_init"][acc]], 1).astype(np.int64)
    n = len(init)
    px_nm, dz_nm = VOX[1], VOX[0]
    px_um = px_nm / 1000.0
    fits, prim, nn_nm, ap_ov, nb_roi, qual, why = _localize(stack, init, VOX, sr, AP, SHIN, SHOUT, CFG)

    def f(key, dtype=float):
        return np.array([x[key] for x in fits], dtype=dtype) if n else np.empty(0, dtype)

    def p(key, dtype=float):
        return prim[key] if n else np.empty(0, dtype)

    fc = np.stack([f("z_fit"), f("y_fit"), f("x_fit")], 1) if n else np.empty((0, 3))
    dpx = fc - init
    quality = np.array(qual, dtype=object)
    reasons = np.array([";".join(w) for w in why], dtype=object)
    ok = np.isin(quality, ["good", "suspect"])
    use = np.where(ok[:, None], fc, init.astype(float))
    t = dict(
        fov_id=np.full(n, fov_id, dtype=object), condition=np.full(n, condition, dtype=object),
        channel=np.full(n, channel, dtype=object), exposure_ms_metadata=np.full(n, exposure_ms if exposure_ms is not None else np.nan),
        spot_id=cand["spot_id"][acc], localized_batch=np.full(n, f"{k:g}sigma", dtype=object),
        z_init=init[:, 0], y_init=init[:, 1], x_init=init[:, 2], z_fit=fc[:, 0], y_fit=fc[:, 1], x_fit=fc[:, 2],
        x_init_um=init[:, 2] * px_um, y_init_um=init[:, 1] * px_um, z_init_um=init[:, 0] * dz_nm / 1000,
        x_fit_um=fc[:, 2] * px_um, y_fit_um=fc[:, 1] * px_um, z_fit_um=fc[:, 0] * dz_nm / 1000,
        x_use=use[:, 2], y_use=use[:, 1], z_use=use[:, 0], coord_source=np.where(ok, "fit", "init").astype(object),
        log_over_sigma=cand["log_over_sigma"][acc], threshold_mode=np.full(n, "fixed_sigma", dtype=object),
        optimizer_converged=f("optimizer_converged", bool), gaussian_sigma_x_nm=f("sigma_x_fit") * VOX[2],
        gaussian_sigma_y_nm=f("sigma_y_fit") * VOX[1], gaussian_sigma_z_nm=f("sigma_z_fit") * VOX[0],
        gaussian_amplitude=f("A_fit"), gaussian_background=f("bg_fit"), gaussian_fit_r2=f("r2"), jac_cond=f("jac_cond"),
        params_at_bound=f("params_at_bound", object), fit_error=f("fit_error", object),
        se_z_nm=f("se_z_px") * VOX[0], se_y_nm=f("se_y_px") * VOX[1], se_x_nm=f("se_x_px") * VOX[2],
        se_reported=f("se_reported", bool), shift_xy_px=np.hypot(dpx[:, 1], dpx[:, 2]), shift_z_slices=np.abs(dpx[:, 0]),
        localization_quality=quality, localization_flag_reasons=reasons,
        flag_z_edge=np.array(["z_edge" in r for r in reasons], bool), flag_large_shift=np.array(["large_shift" in r for r in reasons], bool),
        flag_neighbor_in_roi=nb_roi > 0, flag_sigma_at_bound=np.array(["sigma_at_bound" in r for r in reasons], bool),
        tier_frozen=np.array([("A" if q == "good" else "B10") if k == 10.0 else "" for q in quality], dtype=object),
        snr_primary=p("snr_primary"), snr_peak_legacy_fitqc=p("snr_peak_legacy_fitqc"), peak_net=p("peak_net"),
        local_background=p("local_background"), aperture_sum=p("aperture_sum"), net_integrated=p("net_integrated"),
        n_aperture_vox=p("n_aperture_vox", int), bg_noise_std=p("bg_noise_std"),
        roi_z_truncated=f("roi_z_truncated", bool), roi_xy_truncated=f("roi_xy_truncated", bool),
        aperture_z_truncated=p("aperture_z_truncated", bool), aperture_truncated=p("aperture_truncated", bool),
        edge_flag=cand["edge_flag"][acc], nearest_spot_distance_nm=nn_nm, neighbor_in_roi_n=nb_roi, aperture_overlap=ap_ov)
    t.update(C.assign_use_coordinates(labels, labels_exp, cores, use[:, 1], use[:, 2], cand["cell"][acc], px_nm))
    return t


def _qc_counts(t, cand_n, acc_n):
    q = t["localization_quality"]
    return dict(
        n_candidates=int(cand_n), n_accepted=int(acc_n), n_localized_rows=int(len(q)),
        n_fit_converged=int(np.sum(t["optimizer_converged"])), n_fit_error=int(np.sum(t["fit_error"] != "")),
        quality={s: int(np.sum(q == s)) for s in ("good", "suspect", "unreliable")}, n_qc_passed=int(np.sum(q == "good")),
        coord_source={s: int(np.sum(t["coord_source"] == s)) for s in ("fit", "init")},
        tier_frozen={s: int(np.sum(t["tier_frozen"] == s)) for s in ("A", "B10")},
        assign_category={s: int(np.sum(t["assign_category"] == s)) for s in ("in_baseline", "via_expansion", "outside")},
        n_cell_init_positive=int(np.sum(t["cell_init"] > 0)), n_cell_base_positive=int(np.sum(t["cell_base"] > 0)),
        n_ambiguous_cell=int(np.sum(t["ambiguous_cell"])), n_cell_fit_vs_init_differs=int(np.sum(t["cell_fit_vs_init_differs"])))


def _check_reconciliation(qc):
    """[N] Raise ``PipelineError`` when the per-channel counts do not reconcile (no spot may be lost or counted twice)."""
    for ch, c in qc.items():
        probs = []
        if not (c["n_candidates"] >= c["n_accepted"] >= c["n_qc_passed"]):
            probs.append("n_candidates >= n_accepted >= n_qc_passed violated")
        if c["n_localized_rows"] != c["n_accepted"]:
            probs.append("spots rows != accepted")
        if sum(c["quality"].values()) != c["n_accepted"] or sum(c["coord_source"].values()) != c["n_accepted"]:
            probs.append("quality/coord_source counts do not sum to n_accepted")
        if sum(c["assign_category"].values()) != c["n_accepted"]:
            probs.append("assign_category counts do not sum to n_accepted")
        if c["n_cell_base_positive"] != c["assign_category"]["in_baseline"]:
            probs.append("cell_base > 0 != in_baseline")
        if probs:
            raise PipelineError(f"channel {ch!r}: " + "; ".join(probs))


CELL_BASE_COLUMNS = ("fov_id", "cell_id", "area_px", "area_um2", "centroid_y_px", "centroid_x_px", "core_area_px", "touches_border")
CELL_COUNT_SUFFIXES = ("n_candidates_cell_init", "n_accepted_cell_init", "n_accepted_cell_base", "n_qc_passed_cell_base", "n_accepted_cell_exp_diag")


def cells_columns(det_names):
    return CELL_BASE_COLUMNS + tuple(f"{ch}_{sfx}" for ch in det_names for sfx in CELL_COUNT_SUFFIXES)


def cells_table(labels, cores, px_um, fov_id, det_names, cands, spots):
    """[N] ``cells.csv`` columns, one row per cell id present in ``labels`` (docs/SCHEMAS.md section 14).

    Geometry (``area_px``, ``area_um2``, centroid) is that of the **baseline expanded labels**; ``core_area_px`` is the nucleoid core
    (before expansion); ``touches_border`` = the expanded cell has a pixel on the image border. Per detection channel, five counts that
    are never mixed: ``n_candidates_cell_init`` (every candidate, cell at the initial coordinate); ``n_accepted_cell_init`` (accepted,
    initial coordinate); ``n_accepted_cell_base`` and ``n_qc_passed_cell_base`` (accepted / accepted-and-good, cell at the *use* coordinate on
    the baseline labels: the frozen analysis assignment); ``n_accepted_cell_exp_diag`` (accepted, 8-px diagnostic expansion, not for analysis).
    Spots outside every cell are not in any row; they are the ``outside`` tallies of the metadata, and ``cells_reconciliation`` checks
    sum(rows) + outside == total for every population."""
    g = C.cell_geometry(labels, cores, px_um)
    ids = g["cell_id"]
    n = len(ids)
    t = dict(fov_id=np.full(n, fov_id, dtype=object), **g)
    rec = {}

    def count(values, mask=None):
        v = np.asarray(values) if mask is None else np.asarray(values)[mask]
        v = v[v > 0]
        return np.bincount(v, minlength=int(ids.max()) + 1)[ids] if n else np.empty(0, int)

    for ch in det_names:
        c, sp = cands[ch], spots[ch]
        pops = {"n_candidates_cell_init": (c["cell"], None), "n_accepted_cell_init": (sp["cell_init"], None),
                "n_accepted_cell_base": (sp["cell_base"], None), "n_qc_passed_cell_base": (sp["cell_base"], sp["localization_quality"] == "good"),
                "n_accepted_cell_exp_diag": (sp["cell_exp"], None)}
        rec[ch] = {}
        for name, (vals, mask) in pops.items():
            t[f"{ch}_{name}"] = count(vals, mask)
            v = np.asarray(vals) if mask is None else np.asarray(vals)[mask]
            rec[ch][name] = dict(total=int(len(v)), in_cells=int(t[f"{ch}_{name}"].sum()), outside=int((v == 0).sum()))
            if rec[ch][name]["in_cells"] + rec[ch][name]["outside"] != rec[ch][name]["total"]:
                raise PipelineError(f"channel {ch!r}: {name}: cell rows ({rec[ch][name]['in_cells']}) + outside ({rec[ch][name]['outside']}) "
                                    f"!= total ({rec[ch][name]['total']})")
    return {key: t[key] for key in cells_columns(det_names)}, rec


def check_roles(roles):
    """[N] Validate the explicit channel roles; returns ``(segmentation_name, [detection names])``."""
    seg_name, det_names = roles.get("segmentation"), list(roles.get("detection", []))
    if not seg_name or not det_names or len(set(det_names)) != len(det_names) or seg_name in det_names:
        raise PipelineError("roles must name one segmentation channel and >= 1 distinct detection channels (not overlapping)")
    return seg_name, det_names


def analyze_fov(image, roles, k=D.DEFAULT_K, fov_id="fov", condition="", input_info=None):
    """[N] Run the vertical slice on an in-memory ``ImageStack`` (ND2-independent). Returns a dict with
    ``spots`` / ``candidates`` (``{channel: column dict}`` / one combined column dict), ``metadata``.

    ``roles = {"segmentation": name, "detection": [names]}`` are explicit; ``image.header`` must hold ``optics`` (per channel:
    ``NA``, ``emission_nm``, ``n_immersion``) and may hold ``exposures_ms`` and ``channel_names``. ``k`` is the accepted
    threshold in units of sigma_LoG (default 10; must be >= 6). Voxel must be the baseline (checked by the detector)."""
    t0 = time.time()
    seg_name, det_names = check_roles(roles)
    missing = [n for n in [seg_name, *det_names] if n not in image.channels]
    if missing:
        raise PipelineError(f"channels {missing} not in the image stack (have {list(image.channels)})")
    hdr = image.header or {}
    optics, names = hdr.get("optics", {}), hdr.get("channel_names", list(image.channels))
    exps = hdr.get("exposures_ms", [None] * len(names))
    VOX = tuple(float(v) for v in image.voxel_size_nm)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        seg = S.segment_fov(image.channels[seg_name], VOX[1])
        labels, cores = seg.cell_labels, seg.nucleoid_cores
        labels_exp = S.expand_cores_variant(cores)
        mask = S.noise_mask(labels, seg.seg_params["dil"])
        AP, SHIN, SHOUT = (rescale_radii(r, VOX) for r in (AP_REF, SHIN_REF, SHOUT_REF))
        CFG = rescaled_cfg(L.DEFAULT_CFG, VOX)
        spots, cands, dqc, lqc, ch_meta, cand_by_ch, proj = {}, [], {}, {}, {}, {}, {}
        for ch in det_names:
            if ch not in optics:
                raise PipelineError(f"optics (NA, emission_nm, n_immersion) missing for channel {ch!r}; not guessed")
            o = optics[ch]
            bad = [f for f in ("NA", "emission_nm", "n_immersion")
                   if not isinstance(o.get(f), (int, float, np.floating)) or isinstance(o.get(f), bool) or not math.isfinite(o[f]) or o[f] <= 0]
            if bad:
                raise PipelineError(f"optics field(s) {bad} missing or invalid for channel {ch!r}; not guessed")
            psf = D.psf_from_optics(o["NA"], o["emission_nm"], o["n_immersion"])
            sr = (psf["sigma_z_nm"], psf["sigma_xy_nm"], psf["sigma_xy_nm"])
            cand, qc = D.detect_fixed_sigma(image.channels[ch], labels, mask, VOX, o, k=k, fov_id=fov_id, channel=ch)
            D.validate_candidates(cand)
            exp = exps[names.index(ch)] if ch in names and len(exps) == len(names) else None
            spots[ch] = _spots_for_channel(image.channels[ch], cand, VOX, sr, AP, SHIN, SHOUT, CFG, float(k), fov_id, condition,
                                           ch, exp, labels, labels_exp, cores)
            cands.append(cand)
            cand_by_ch[ch] = cand
            proj[ch] = image.channels[ch].max(axis=0)
            dqc[ch] = qc
            lqc[ch] = _qc_counts(spots[ch], qc["n_candidates"], qc["n_accepted"])
            ch_meta[ch] = dict(optics=o, exposure_ms=exp, psf_nm=psf, spot_radius_nm=sr)
    cells, cells_rec = cells_table(labels, cores, VOX[1] / 1000.0, fov_id, det_names, cand_by_ch, spots)
    D.validate_run_modes(cands)
    _check_reconciliation(lqc)
    caught = list(dict.fromkeys(f"{w.category.__name__}: {w.message}" for w in rec))
    meta = dict(
        schema_version=SCHEMA_VERSION, catkit_version=catkit.__version__,
        software=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, scikit_image=skimage.__version__,
                      nd2=nd2.__version__, bigfish=bigfish.__version__),
        input=input_info or {}, fov_id=fov_id, condition=condition,
        channel_roles=dict(segmentation=seg_name, detection=det_names), channels=ch_meta,
        voxel_size_nm_zyx=list(VOX), array_axes="(Z, Y, X); coordinates (z, y, x), *_init integer voxel indices, *_fit/*_use voxel units, *_um micrometres",
        detection=dict(mode="fixed_sigma", k_sigma=float(k), candidate_floor_sigma=D.CANDIDATE_FLOOR_SIGMA, ks=list(D.KS),
                       edge_margin_px=D.EDGE_MARGIN_PX),
        segmentation=dict(method=seg.method, n_cells=seg.n_cells, segmentation_empty=seg.n_cells == 0, otsu_threshold=seg.otsu_threshold,
                          focus=dict(start=seg.focus_start, end=seg.focus_end, best=seg.focus_best), seg_params=seg.seg_params,
                          cell_footprint_fraction=float((labels > 0).mean()), noise_region_fraction=float(mask.mean()),
                          expanded_variant_px=S.EXP_VARIANT_PX),
        resolved_config=dict(aperture=AP, shell_in=SHIN, shell_out=SHOUT, fit_cfg=CFG, ref_vox=REF_VOX),
        detection_qc=dqc, localization_qc=lqc, cells_reconciliation=cells_rec,
        reconciliation=dict(ok=True, rule="n_candidates >= n_accepted >= n_qc_passed; spots rows == n_accepted; assignment counts sum to n_accepted"),
        tier_rule=("tier_frozen is written only for fixed_sigma with k = 10: 'A' = accepted and localization_quality == 'good', "
                   "'B10' = accepted and not good; otherwise empty"),
        geometry_basis="cells.csv area/centroid from the baseline expanded labels; core_area_px from the cores before expansion",
        notes=["candidate counts use the initial-coordinate cell lookup (cell_init); assignment counts of accepted spots use cell_base (use coordinate)",
               "dist_to_mask_um (candidates.csv) is measured at the initial coordinate; with zero cells it carries no nearest-cell meaning"],
        warnings=caught, runtime_s=round(time.time() - t0, 2), timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    candidates = {key: np.concatenate([c[key] for c in cands]) for key in D.CANDIDATE_COLUMNS if all(key in c for c in cands)}
    all_spots = {key: np.concatenate([spots[ch][key] for ch in det_names]) for key in SPOT_COLUMNS}
    return dict(spots=all_spots, spots_by_channel=spots, candidates=candidates, metadata=meta, labels=labels, cells=cells,
                cells_columns=cells_columns(det_names), projections=proj, det_names=det_names)


# ---------------------------------------------------------------- output (plain csv / json; no new serialization framework)
def _cell(v):
    if isinstance(v, (bool, np.bool_)):
        return "True" if v else "False"
    if isinstance(v, (float, np.floating)):
        return "" if math.isnan(v) else ("inf" if v == math.inf else "-inf" if v == -math.inf else "%.10g" % v)
    if v is None:
        return ""
    return str(v)


def write_table(path, columns, order):
    """[N] CSV with the given column order: floats ``%.10g`` (NaN -> empty field, +-inf -> ``inf``/``-inf``), bools ``True``/``False``."""
    n = len(columns[order[0]]) if order else 0
    for c in order:
        if len(columns[c]) != n:
            raise PipelineError(f"column {c!r} has {len(columns[c])} rows, expected {n}")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(order)
        for i in range(n):
            w.writerow([_cell(columns[c][i]) for c in order])


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not math.isfinite(float(o)) else float(o)
    return o


def _safe(name):
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)


MAX_NAME_CHARS = 40   # display part of an overlay file name (the index makes it unique; the full channel name is kept in the metadata)


def qc_image_names(det_names):
    """[N] Stable, unique overlay file names: detection order + sanitized channel name truncated to ``MAX_NAME_CHARS`` (names that
    sanitize or truncate alike still differ by their index; file-system name limits cannot be hit)."""
    return {ch: f"qc_overlay_{i:02d}_{_safe(ch)[:MAX_NAME_CHARS]}.png" for i, ch in enumerate(det_names, 1)}


def check_output_dir(out):
    """[N] Refuse an unsafe destination: a symbolic link, an existing non-directory, or a non-empty directory (links inside it
    could point at other files). A missing path or an empty directory is fine. Permission or I/O errors while inspecting the
    destination are reported as ``OutputError``."""
    out = Path(out)
    try:
        if out.is_symlink():
            raise OutputError(f"output path {out} is a symbolic link; refusing to write through it")
        if out.exists():
            if not out.is_dir():
                raise OutputError(f"output path {out} exists and is not a directory")
            if any(out.iterdir()):
                raise OutputError(f"output directory {out} is not empty; choose a new or empty directory")
    except OSError as exc:
        if isinstance(exc, OutputError):
            raise
        raise OutputError(f"cannot inspect output path {out}: {exc}") from exc
    return out


def write_outputs(result, out_dir, qc_overlays=True):
    """[N] Write ``spots.csv``, ``candidates.csv``, ``cells.csv``, ``run_metadata.json`` and (default) one QC overlay per detection
    channel (``qc_overlay_<index>_<channel>.png``; the mapping is ``metadata["qc_images"]``).

    All files are written into a fresh temporary directory next to ``out_dir`` and published with one rename after everything
    succeeded; on any failure the temporary directory is removed and ``out_dir`` is untouched (an existing empty directory stays empty).
    The destination is checked by ``check_output_dir`` before writing and again before publishing.
    The published directory keeps ``mkdtemp``'s owner-only mode (0700); the process umask is never read or changed (that needs a
    process-wide set-and-restore, which races with other threads)."""
    out = check_output_dir(out_dir)
    parent = out.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.catkit-", dir=parent))
    except OSError as exc:
        raise OutputError(f"cannot create a temporary directory next to {out}: {exc}") from exc
    try:
        write_table(tmp / "spots.csv", result["spots"], SPOT_COLUMNS)
        write_table(tmp / "candidates.csv", result["candidates"], D.CANDIDATE_COLUMNS)
        write_table(tmp / "cells.csv", result["cells"], result["cells_columns"])
        files = ["spots.csv", "candidates.csv", "cells.csv", "run_metadata.json"]
        images = {}
        if qc_overlays:
            images = qc_image_names(result["det_names"])
            for ch, f in images.items():
                QC.save_overlay(tmp / f, result["projections"][ch], result["labels"], result["spots"], ch, result["metadata"]["fov_id"])
            files += list(images.values())
        meta = dict(result["metadata"], outputs=files, qc_images=images)
        (tmp / "run_metadata.json").write_text(json.dumps(_jsonable(meta), indent=2))
        check_output_dir(out)                                                   # re-check right before publishing
        os.rename(tmp, out)                                                     # replaces an EMPTY directory in one step; the original survives a failure
    except BaseException as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        if isinstance(exc, OSError) and not isinstance(exc, OutputError):
            raise OutputError(f"could not write outputs to {out}: {exc}") from exc
        raise
    result["metadata"]["outputs"], result["metadata"]["qc_images"] = files, images
    return out


def run_nd2(path, out_dir, roles, k=D.DEFAULT_K, fov_id=None, condition=""):
    """[N] ND2 file -> outputs. Reads only the channels named in ``roles`` (read-only), records the file's name (not its path),
    size and SHA-256 in the metadata. Returns the ``analyze_fov`` result."""
    path = Path(path)
    check_output_dir(out_dir)                                                    # fail before the expensive analysis
    seg_name, det_names = check_roles(roles)
    names = [seg_name, *det_names]
    image = load_nd2_channels(path, names)
    info = dict(file_name=path.name, size_bytes=path.stat().st_size, sha256=sha256_file(path))
    res = analyze_fov(image, roles, k=k, fov_id=fov_id or path.stem, condition=condition, input_info=info)
    res["metadata"]["input"]["nd2_header_warnings"] = image.header.get("warnings", [])
    res["metadata"]["input"]["sizes"] = image.header.get("sizes")
    res["metadata"]["input"]["z_step_um"] = image.header.get("z_step_um")
    res["metadata"]["input"]["axes_calibrated"] = image.header.get("axes_calibrated")
    res["metadata"]["warnings"] += [f"ND2 header: {w}" for w in image.header.get("warnings", [])]
    write_outputs(res, out_dir)
    return res
