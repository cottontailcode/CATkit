"""Spot-to-cell lookup and per-cell geometry.

Provenance. ``assign_cells_init`` is ``assign_cells`` of frozen reference ``REF-DET``
(sha256 8a33a51a20f994060f099f855c5f872949169ad3fc9d48c0870fbb43266da700) with ONE edit: the optional
``shift_dy``/``shift_dx`` parameters and the shifted-mask branch are removed (the frozen driver always uses
(0, 0); v0.1 segments in the FISH frame). With the default shift the two functions are identical in behaviour
(E-tested). The lookup uses the INITIAL (integer) detector coordinates; assignment by fitted coordinates
(``cell_base`` etc.) belongs to M2.3.

``core_distances`` is copied verbatim from ``REF-DRV`` and ``assign_use_coordinates`` is the driver's assignment block ([D], M2.3-A).

``cell_geometry`` is [N]: new in CATkit, no frozen counterpart (synthetic contract tests only).

With zero cells (all-zero ``labels``) ``distance_transform_edt(labels == 0)`` still returns numbers; they are NOT
"distance to the nearest cell" and must not be interpreted as such (an exported missing-value convention is a
separate [N] decision, SCHEMAS section 13).

Frozen behaviour kept on purpose (not guarded): a coordinate outside the image raises ``IndexError`` for values
>= shape and silently wraps for negative values (NumPy indexing); real detector output never contains either.
"""
import numpy as np
from scipy import ndimage as ndi


AMBIG_PX = 2.0   # [D] REF-DRV: ambiguity margin (px), not rescaled


def core_distances(cores, yx, win=14):
    """distance (px) from each (y,x) to the nearest and 2nd-nearest nucleoid core (pre-expansion label), searched in a +-win window."""
    Y, X = cores.shape; out = np.full((len(yx), 4), np.nan)
    for i, (y, x) in enumerate(np.round(yx).astype(int)):
        y0, y1, x0, x1 = max(0, y - win), min(Y, y + win + 1), max(0, x - win), min(X, x + win + 1); sub = cores[y0:y1, x0:x1]; labs = np.unique(sub[sub > 0])
        if not len(labs): continue
        yy, xx = np.nonzero(sub); d = np.hypot(yy + y0 - y, xx + x0 - x); lv = sub[yy, xx]; dm = sorted((d[lv == l].min(), l) for l in labs)
        out[i, 0], out[i, 1] = dm[0]; out[i, 2], out[i, 3] = dm[1] if len(dm) > 1 else (np.inf, 0)
    return out


def assign_use_coordinates(labels, labels_exp, cores, y_use, x_use, cell_init, px_nm):
    """[D] The driver's assignment block: lookup by the USE coordinate (fit if good/suspect, else initial), rounded with
    ``numpy.round`` (half-to-even) and clipped to the image. Returns the ``spots.csv`` assignment columns.

    ``cell_base`` uses the baseline (expanded) labels, ``cell_exp`` the 8-px variant, ``in_core`` the cores;
    ``assign_category`` is ``in_baseline`` / ``via_expansion`` / ``outside``; ``ambiguous_cell`` = assigned and
    ``core_d2 - core_d1 < AMBIG_PX``; ``dist_to_baseline_mask_nm`` = EDT of the baseline background at that pixel.
    """
    Y, X = labels.shape
    yx = np.c_[np.clip(np.round(y_use), 0, Y - 1), np.clip(np.round(x_use), 0, X - 1)].astype(int)
    cell_base = labels[yx[:, 0], yx[:, 1]]
    cell_exp = labels_exp[yx[:, 0], yx[:, 1]]
    cd = core_distances(cores, np.c_[y_use, x_use])
    cat = np.select([cell_base > 0, cell_exp > 0], ["in_baseline", "via_expansion"], "outside")
    return dict(cell_init=np.asarray(cell_init), cell_base=cell_base, cell_exp=cell_exp, in_core=cores[yx[:, 0], yx[:, 1]] > 0,
                core_d1_px=cd[:, 0], core_l1=cd[:, 1], core_d2_px=cd[:, 2], core_l2=cd[:, 3], assign_category=cat,
                ambiguous_cell=(cat != "outside") & ((cd[:, 2] - cd[:, 0]) < AMBIG_PX),
                cell_fit_vs_init_differs=(cell_base != np.asarray(cell_init)),
                dist_to_baseline_mask_nm=ndi.distance_transform_edt(labels == 0)[yx[:, 0], yx[:, 1]] * px_nm)


def assign_cells_init(coords_zyx, labels, px_um):
    """Mask lookup AFTER detection (nothing is dropped). labels: (Y,X) int array in the segmentation frame.
    Returns cell_id (0 = outside) and distance (um) to the nearest mask pixel (0 inside a cell)."""
    lab = labels
    y, x = coords_zyx[:, 1], coords_zyx[:, 2]
    cid = lab[y, x]
    dist = ndi.distance_transform_edt(lab == 0) * px_um
    return cid.astype(int), dist[y, x]


def cell_geometry(labels, cores, px_um):
    """[N] Per-cell geometry for ``cells.csv`` (columns of docs/SCHEMAS.md section 14).

    ``labels`` = expanded cell labels, ``cores`` = nucleoid cores before expansion (both 2D, same shape).
    Returns a dict of equal-length arrays, one entry per cell id present in ``labels`` (ascending):
    ``cell_id``, ``area_px`` (int), ``area_um2`` (= area_px * px_um**2), ``centroid_y_px``/``centroid_x_px``
    (mean pixel index of the expanded cell, 0-based pixel centres), ``core_area_px`` (pixels of the same id in
    ``cores``; 0 if absent), ``touches_border`` (the cell has a pixel in the first/last row or column).
    """
    labels, cores = np.asarray(labels), np.asarray(cores)
    if labels.ndim != 2 or labels.shape != cores.shape:
        raise ValueError("labels and cores must be 2D arrays of the same shape")
    ids = np.unique(labels)
    ids = ids[ids > 0]
    area = np.array([int((labels == i).sum()) for i in ids], dtype=int)
    yy, xx = np.indices(labels.shape)
    cy = np.array([float(yy[labels == i].mean()) for i in ids])
    cx = np.array([float(xx[labels == i].mean()) for i in ids])
    core = np.array([int((cores == i).sum()) for i in ids], dtype=int)
    border = np.zeros(labels.shape, bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    touches = np.array([bool(((labels == i) & border).any()) for i in ids])
    return dict(cell_id=ids.astype(int), area_px=area, area_um2=area * float(px_um) ** 2,
                centroid_y_px=cy, centroid_x_px=cx, core_area_px=core, touches_border=touches)
