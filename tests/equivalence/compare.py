"""Field-by-field comparison for E-layer equivalence tests (frozen vs port).

Rules (docs/EQUIVALENCE.md section 2-3):

* every dict key needs an explicit entry in the spec; an **unknown key is an
  error** (no lenient default tolerance);
* ``EXACT``: ints, bools, strings, flags, reasons (order matters), key sets;
  floats under EXACT must be numerically equal (NaN positions equal,
  ``+0.0 == -0.0``);
* ``("float", atol, rtol)``: ``|a-b| <= atol + rtol*|b|`` with BOTH stated;
  NaN positions equal; +-Inf exactly equal (sign included);
* array shape and dtype, tuple/list length and type must match;
* bit-identity of float fields is counted separately and never decides PASS.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EXACT = ("exact",)


def FLOAT(atol: float, rtol: float):
    return ("float", float(atol), float(rtol))


POS = FLOAT(1e-9, 0.0)      # positions in index units
PAR = FLOAT(0.0, 1e-9)      # fit parameters and diagnostics

# Output of ``fit_spot`` (keys of ``_nan_result`` plus the two private ROI tuples).
LOC_FIT_SPEC = {
    **{f"{n}_fit": POS for n in ("mu_z", "mu_y", "mu_x")},
    **{f"{n}_fit": PAR for n in ("sigma_z", "sigma_y", "sigma_x", "A", "bg")},
    "z_fit": POS, "y_fit": POS, "x_fit": POS,
    "rss": PAR, "rms": PAR, "r2": PAR, "reduced_chi2": PAR, "jac_cond": PAR,
    "se_z_px": PAR, "se_y_px": PAR, "se_x_px": PAR, "se_A": PAR,
    **{k: EXACT for k in (
        "optimizer_converged", "status", "message", "nfev", "jac_rank", "n_roi_vox", "params_at_bound",
        "se_reported", "center_out_of_roi", "nonfinite", "roi_z_truncated", "roi_xy_truncated", "fit_error",
        "_roi_lo", "_roi_hi")},
}


# ---- REF-PHOT return structures (every key explicit; raw sums/counts/indices/flags exact, derived 0/1e-12) ----
DER = FLOAT(0.0, 1e-12)     # derived photometry quantities and physical distances

CLIPBG_SPEC = {
    "median": DER, "mean": DER, "noise_mad": DER, "noise_std": DER, "noise_legacy": DER,
    "n_kept": EXACT, "n_nonfinite": EXACT,
}

# measure_spots: 20 float arrays + 10 bool arrays.
# Raw-count quantities (peak_raw, aperture_sum) and voxel counts are EXACT: they are sums/maxima of the same
# voxels, so any difference means a different voxel set or operation order.
PHOT_SPEC = {
    **{k: EXACT for k in (
        "aperture_sum", "peak_raw", "n_aperture_vox", "n_aperture_vox_full", "n_bg_voxels_valid", "n_shell_vox",
        "n_shell_excluded", "n_shell_nonfinite",
        "aperture_truncated", "aperture_z_truncated", "shell_truncated", "noise_unreliable", "bg_unreliable",
        "bg_insufficient", "noise_nonfinite", "noise_zero", "mad_zero", "bg_nonfinite_voxels")},
    **{k: DER for k in (
        "local_background", "local_background_median", "bg_noise_std", "bg_noise_mad", "bg_noise_legacy",
        "net_integrated", "peak_net", "snr_primary", "snr_peak_primary", "snr_mad", "snr_peak_mad",
        "snr_peak_legacy_fitqc")},
}

# ellipsoid_voxels -> ((z, y, x) index arrays, n_full) is compared positionally with EXACT.
ELLIPSOID_SPEC = EXACT
OCCUPANCY_SPEC = EXACT          # uint8 array, compared in its own dtype
NEIGHBOR_SPEC = (DER, EXACT)    # (distance nm, neighbour index j) -- j compared exactly, ties included
OVERLAP_SPEC = (EXACT, EXACT)   # (aperture_overlap flags, neighbor_in_roi counts)


@dataclass
class Stats:
    compared: int = 0
    bit_identical: int = 0
    not_bit_identical: list = field(default_factory=list)  # (key, max abs diff)


def _fmt(x) -> str:
    return repr(x) if not isinstance(x, np.ndarray) else f"array(shape={x.shape}, dtype={x.dtype})"


def _numeric_problem(a: np.ndarray, b: np.ndarray, kind) -> str | None:
    """None if equal under ``kind``; otherwise a description of the first offender."""
    if kind == EXACT and a.dtype.kind in "iub" and b.dtype.kind in "iub":
        bad = a != b          # integers/bools are compared in their own dtype (no lossy float64 round trip)
        if bad.any():
            i = tuple(np.argwhere(bad)[0])
            return f"exact mismatch at {i}: frozen={a[i]!r} port={b[i]!r}"
        return None
    af = a.astype(float) if a.dtype != bool else a
    bf = b.astype(float) if b.dtype != bool else b
    if kind == EXACT:
        bad = ~((af == bf) | (np.isnan(af.astype(float)) & np.isnan(bf.astype(float))))
        if bad.any():
            i = tuple(np.argwhere(bad)[0])
            return f"exact mismatch at {i}: frozen={a[i]!r} port={b[i]!r}"
        return None
    _, atol, rtol = kind
    af, bf = af.astype(float), bf.astype(float)
    nan_a, nan_b = np.isnan(af), np.isnan(bf)
    if (nan_a != nan_b).any():
        i = tuple(np.argwhere(nan_a != nan_b)[0])
        return f"NaN position mismatch at {i}: frozen={af[i]!r} port={bf[i]!r}"
    inf_a, inf_b = np.isinf(af), np.isinf(bf)
    if (inf_a != inf_b).any() or (af[inf_a] != bf[inf_b]).any():
        i = tuple(np.argwhere(inf_a != inf_b)[0]) if (inf_a != inf_b).any() else tuple(np.argwhere(inf_a)[0])
        return f"Inf mismatch at {i}: frozen={af[i]!r} port={bf[i]!r}"
    fin = ~(nan_a | inf_a)
    with np.errstate(invalid="ignore"):  # inf - inf at non-finite positions; those are excluded by ``fin``
        diff = np.abs(af - bf)
        bad = fin & ~(diff <= atol + rtol * np.abs(bf))
    if bad.any():
        i = tuple(np.argwhere(bad)[0])
        return f"|diff|={diff[i]:.3e} > atol({atol}) + rtol({rtol})*|b| at {i}: frozen={af[i]!r} port={bf[i]!r}"
    return None


def compare_value(a, b, kind, label: str, stats: Stats | None = None) -> list[str]:
    """Return a list of problems (empty = equivalent)."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return [f"{label}: array vs non-array ({_fmt(a)} vs {_fmt(b)})"]
        if a.shape != b.shape or a.dtype != b.dtype:
            return [f"{label}: shape/dtype differ: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}"]
        if a.dtype.kind in "OUS":
            ok = np.array_equal(a, b)
            return [] if ok else [f"{label}: object/string arrays differ"]
        prob = _numeric_problem(a, b, kind)
        if stats is not None and a.dtype.kind == "f":
            stats.compared += 1
            if a.tobytes() == b.tobytes():
                stats.bit_identical += 1
            elif prob is None:
                fin = np.isfinite(a) & np.isfinite(b)
                stats.not_bit_identical.append((label, float(np.max(np.abs(a - b)[fin])) if fin.any() else 0.0))
        return [f"{label}: {prob}"] if prob else []
    if isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
        if type(a) is not type(b) or len(a) != len(b):
            return [f"{label}: sequence type/length differ: {type(a).__name__}[{len(a)}] vs {type(b).__name__}[{len(b)}]"]
        out: list[str] = []
        per_element = isinstance(kind[0], tuple)       # a tuple OF specs: one spec per element (e.g. NEIGHBOR_SPEC)
        if per_element and len(kind) != len(a):
            return [f"{label}: spec has {len(kind)} entries for {len(a)} elements"]
        for i, (x, y) in enumerate(zip(a, b)):
            out += compare_value(x, y, kind[i] if per_element else kind, f"{label}[{i}]", stats)
        return out
    if type(a) is not type(b):
        return [f"{label}: type differs: {type(a).__name__} vs {type(b).__name__} ({a!r} vs {b!r})"]
    if isinstance(a, (float, np.floating)):
        prob = _numeric_problem(np.asarray(a), np.asarray(b), kind)
        if stats is not None:
            stats.compared += 1
            if np.asarray(a).tobytes() == np.asarray(b).tobytes():
                stats.bit_identical += 1
            elif prob is None:
                fin = np.isfinite(a) and np.isfinite(b)
                stats.not_bit_identical.append((label, float(abs(a - b)) if fin else 0.0))
        return [f"{label}: {prob}"] if prob else []
    if kind != EXACT:
        return [f"{label}: non-float value {a!r} given float tolerance spec"]
    return [] if a == b else [f"{label}: frozen={a!r} port={b!r}"]


def compare_dicts(frozen: dict, port: dict, spec: dict, stats: Stats | None = None, prefix: str = "") -> list[str]:
    """Compare whole result dicts: key sets first, then every field under its explicit spec."""
    problems: list[str] = []
    if set(frozen) != set(port):
        problems.append(f"{prefix}key sets differ: only-frozen={sorted(set(frozen) - set(port))} only-port={sorted(set(port) - set(frozen))}")
    for key in sorted(set(frozen) & set(port)):
        if key not in spec:
            raise AssertionError(f"{prefix}{key}: no tolerance spec for this key (add it explicitly; there is no default)")
        problems += compare_value(frozen[key], port[key], spec[key], f"{prefix}{key}", stats)
    return problems


def call_outcome(fn, *args, **kwargs):
    """Run ``fn``; return ('ok', value) or ('raise', type name, message) so exceptions can be compared too."""
    try:
        return ("ok", fn(*args, **kwargs))
    except Exception as exc:  # noqa: BLE001 - the point is to compare failures
        return ("raise", type(exc).__name__, str(exc))
