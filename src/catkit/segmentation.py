"""Nuclear-stain (Hoechst/DAPI) segmentation: Otsu -> opening -> small-object removal -> label -> expand_labels.

Cell IDs are ``skimage.measure.label`` order (1..N), preserved by ``expand_labels``. The nuclear stain labels
nucleoids; the expanded label image approximates the whole-cell footprint (an approximation, not a membrane
segmentation).

Provenance. Functions marked *verbatim* are copied unchanged from frozen reference ``REF-SEG``
(sha256 48e856f327bd23ae59635c739c49d87f45ff6eb23a66527648ceaea29513aab6) or, for ``seg_params``, from
``REF-DRV`` (sha256 053bd566425d225ecf0e8ef532758125adb44d7f946b4dc597ee84bbda811aac). The frozen code
already contained the skimage >= 0.26 ``max_size`` shim inside ``_remove_small``; it is part of the verbatim body.
Not ported: ``register_mask``, ``mad_display``, ``HS_DX``/``HS_DY`` (experiment-specific shift and display
helpers; v0.1 segments in the FISH frame).

Layers. *Frozen behaviour* (checked by E tests): ``detect_focus_plane``, ``focus_slice_range``, ``_remove_small``,
``segment_from_dapi``, ``seg_params``, and the expression inside ``noise_mask``. *[N] new interface* (checked by
synthetic contract tests only, no frozen counterpart): ``NoCellsWarning``, ``SegmentationResult``,
``segment_fov`` and its guards (``Z >= 3``, 65 nm pixel, ``method``), and ``noise_mask``'s wrapper signature.
See docs/ALGORITHMS.md.
"""
import inspect
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_dilation
from skimage import measure
from skimage.filters import threshold_otsu
from skimage.morphology import disk, opening, remove_small_objects
from skimage.segmentation import expand_labels


def detect_focus_plane(z_stack, num_slices=3):  # verbatim
    scores = [np.std(z_stack[i]) for i in range(z_stack.shape[0])]
    best_center = np.argmax(scores)
    half = num_slices // 2
    start = max(0, best_center - half)
    end = min(z_stack.shape[0], start + num_slices)
    if end == z_stack.shape[0]:
        start = end - num_slices
    return z_stack[start:end]


def focus_slice_range(z_stack, num_slices=3):
    """Same index logic as detect_focus_plane, returning (start, end) for reporting."""
    scores = [np.std(z_stack[i]) for i in range(z_stack.shape[0])]
    best_center = int(np.argmax(scores))
    half = num_slices // 2
    start = max(0, best_center - half)
    end = min(z_stack.shape[0], start + num_slices)
    if end == z_stack.shape[0]:
        start = end - num_slices
    return start, end, best_center, scores


def _remove_small(binary, min_size):
    """Remove objects with area < min_size (HS semantics) on any skimage version."""
    if "max_size" in inspect.signature(remove_small_objects).parameters:
        return remove_small_objects(binary, max_size=min_size - 1)
    return remove_small_objects(binary, min_size=min_size)


def segment_from_dapi(dapi_proj, min_size, expand_px, opening_radius=1):
    """HS pipeline: Otsu -> opening(disk(r)) -> remove_small_objects -> label -> expand_labels.

    Verbatim except `opening_radius` (HS value 1) and the skimage-version shim. Returns (labels, otsu_thresh,
    labels_before_expansion).
    """
    thresh = threshold_otsu(dapi_proj)
    binary = dapi_proj > thresh
    binary = opening(binary, disk(opening_radius))
    binary = _remove_small(binary, min_size)
    labeled = measure.label(binary)
    expanded = expand_labels(labeled, distance=expand_px)
    return expanded, thresh, labeled


def seg_params(px_nm): return dict(min_size=int(round(15 * (65.0 / px_nm) ** 2)), expand=int(round(4 * 65.0 / px_nm)), opening=int(round(65.0 / px_nm)), dil=int(round(10 * 65.0 / px_nm)))  # noqa: E704  (verbatim REF-DRV, Python round = half-to-even)


def noise_mask(labels, dil):
    """Outside-cell region used for the LoG noise estimate: ``~binary_dilation(labels > 0, iterations=dil)``.

    [N] wrapper around an expression that is inline in the frozen driver (``outside = ~binary_dilation(lab > 0,
    iterations=sp["dil"])``). The expression itself is frozen: ``labels`` is the EXPANDED label image, the
    structuring element is SciPy's default (cross, connectivity 1), and a label image without any cell gives an
    all-True mask (the whole frame is "outside"). Returns a 2D bool array; apply it to every z plane.
    """
    return ~binary_dilation(labels > 0, iterations=dil)


EXP_VARIANT_PX = 8   # [D] REF-DRV: diagnostic expansion distance (not rescaled); never used for the analysis baseline


def expand_cores_variant(cores, distance=EXP_VARIANT_PX):
    """[D] ``expand_labels(cores, distance=8).astype(int32)``: the driver's diagnostic variant (``cell_labels_expanded8``)
    used only for ``cell_exp`` / ``assign_category``; the baseline analysis labels are ``segment_fov(...).cell_labels``."""
    return expand_labels(cores, distance=distance).astype(np.int32)


class NoCellsWarning(UserWarning):
    """Segmentation found no cell; the noise region is the whole frame (frozen behaviour, kept on purpose)."""


@dataclass
class SegmentationResult:
    """[N] container for the outputs of ``segment_fov`` (no frozen counterpart)."""

    cell_labels: np.ndarray        # expanded, int32 (Y, X); the assignment baseline
    nucleoid_cores: np.ndarray     # before expansion, int32 (Y, X)
    otsu_threshold: float | None   # None for method="external"
    focus_start: int | None
    focus_end: int | None
    focus_best: int | None
    n_cells: int
    seg_params: dict
    method: str
    warnings: list = field(default_factory=list)


def segment_fov(dapi_stack, px_nm, method="otsu", num_slices=3, external_labels=None, external_cores=None):
    """[N] Frozen driver chain for one FOV: focus planes -> max projection -> ``segment_from_dapi`` -> int32.

    ``method="otsu"`` replays the frozen chain; ``method="external"`` takes user-supplied labels (``cores``
    default to the labels themselves). ``omnipose``/``both`` are not available before M4.

    Guards (new, not frozen behaviour): ``dapi_stack`` must be 3D with ``Z >= num_slices`` (the frozen index
    arithmetic would return a wrong slice for fewer planes); ``px_nm`` must be 65 nm (the only validated pixel
    size). Zero detected cells is **not** an error: it records ``n_cells = 0``, issues ``NoCellsWarning`` and
    appends the same message to ``warnings`` (frozen behaviour: the whole frame becomes the noise region).
    A valid but featureless image (e.g. flat) is a no-cell FOV; an empty dimension or illegal shape is an input
    error (``ValueError``) and never reported as ``n_cells = 0``.
    """
    if method in ("omnipose", "both"):
        raise NotImplementedError("M4")
    if method not in ("otsu", "external"):
        raise ValueError(f"unknown segmentation method {method!r} (expected 'otsu' or 'external')")
    if abs(float(px_nm) - 65.0) > 1e-6:
        raise ValueError(f"pixel size {px_nm} nm is not supported in v0.1 (only 65 nm is validated)")
    sp = seg_params(px_nm)
    msgs: list = []
    if method == "otsu":
        stack = np.asarray(dapi_stack)
        if stack.ndim != 3:
            raise ValueError(f"nuclear-stain stack must be 3D (Z, Y, X); got {stack.ndim}D")
        if 0 in stack.shape[1:]:
            raise ValueError(f"nuclear-stain stack has an empty image plane (shape {stack.shape}); input error, not a no-cell FOV")
        if stack.shape[0] < max(3, num_slices):
            raise ValueError(f"nuclear-stain stack has Z={stack.shape[0]} planes; at least {max(3, num_slices)} are required")
        z_start, z_end, z_best, _ = focus_slice_range(stack, num_slices)
        proj = np.max(detect_focus_plane(stack, num_slices), axis=0)
        lab, thr, cores = segment_from_dapi(proj, sp["min_size"], sp["expand"], opening_radius=sp["opening"])
        lab, cores, thr = lab.astype(np.int32), cores.astype(np.int32), float(thr)
        focus = (int(z_start), int(z_end), int(z_best))
    else:
        if external_labels is None:
            raise ValueError("method='external' requires external_labels")
        lab = np.asarray(external_labels)
        if lab.ndim != 2 or lab.dtype.kind not in "iu" or (lab < 0).any():
            raise ValueError("external_labels must be a 2D non-negative integer array")
        cores = lab if external_cores is None else np.asarray(external_cores)
        if cores.shape != lab.shape:
            raise ValueError("external_cores must have the same shape as external_labels")
        lab, cores, thr = lab.astype(np.int32), cores.astype(np.int32), None
        focus = (None, None, None)
    n_cells = int(np.count_nonzero(np.unique(lab)))
    if n_cells == 0:
        msg = "no cells segmented (n_cells=0): the whole frame is treated as the noise region (frozen behaviour)"
        msgs.append(msg)
        warnings.warn(msg, NoCellsWarning, stacklevel=2)
    return SegmentationResult(lab, cores, thr, *focus, n_cells, sp, method, msgs)
