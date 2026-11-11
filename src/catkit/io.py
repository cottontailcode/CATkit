"""Minimal Nikon ND2 reader (v0.1): header inspection and named-channel loading into a standard ``ImageStack``.

Contract (docs/SCHEMAS.md section 5, Revision 2 C3):

* Axes must be exactly ``(Z, C, Y, X)`` as reported by the ``nd2`` library (it omits singleton axes, so a file with
  a position/time axis, or without a Z axis, is an error; there is no selector). Pixel data are ``uint16``.
* Channels are selected by the **metadata channel name**, never by position, and names must be unique. CATkit does
  not guess which channel is the nuclear stain or a FISH channel: the caller names the channels it wants. A missing
  name is an error that lists the names present.
* Voxel size comes from the file (``x == y`` required) and is returned as ``(z, y, x)`` nm. Missing/non-positive/non-finite
  values are errors. v0.1 analysis supports only the baseline voxel ``(150, 65, 65)`` nm: ``load_nd2_channels`` raises
  for anything else unless ``require_baseline_voxel=False`` (the header is always reportable).
* Calibration: ``volume.axesCalibrated`` (x, y, z) of every channel is read. An axis explicitly marked **uncalibrated** (``False``)
  on a requested channel is an error before pixels are read, even if its number equals the baseline; a flag that is absent
  (``None``) is not a rejection but is reported in ``warnings``; a flag that is not a boolean (or a flags value that is not a
  3-sequence) is *malformed* and rejected like an uncalibrated one, never coerced. ``inspect_nd2`` only reports (``axes_calibrated`` and a warning).
* Validation order: axes -> dtype -> channel names -> voxel -> (requested names, calibration, baseline voxel) **before** any pixel
  is read. ``inspect_nd2`` never reads pixel data. The file is opened read-only and closed on every path.
* Nothing is inferred from the file name. Exposures are parsed from the capture text; they are attached to channels
  by order (frozen assumption, flagged in ``warnings`` if the counts disagree) and are informational.

Provenance. ``parse_capturing_exposures`` is copied verbatim from frozen ``REF-IO``
(sha256 305c5d1e535789a45e0a36dd9bd447ae981df657c46428847fbd1ff1224695b2). The header fields mirror ``REF-IO.read_header``
and the optics fields ``REF-DET.optics_from_nd2`` ([D]: same metadata expressions, one file open). Everything else is
[N] (``ImageStack``, validation, error types, voxel reordering to ``(z, y, x)``). Not ported: filename parser, dye map.
"""
import re
from dataclasses import dataclass, field

import numpy as np

import nd2

BASELINE_VOXEL_NM = (150.0, 65.0, 65.0)   # (z, y, x)
AXES = ("Z", "C", "Y", "X")
DZ_STEP_WARN_UM = 1e-3                    # [?] provisional threshold (SCHEMAS section 16 item 1)


class ND2Error(ValueError):
    """The file violates the reader contract (axes, dtype, channel names, voxel, ...)."""


class ChannelNotFound(ND2Error):
    """A requested channel name is not in the file metadata."""


def parse_capturing_exposures(text_info):
    """'Sample N: ... Exposure: X ms' blocks -> list of ms, in acquisition (channel) order."""
    cap = (text_info or {}).get("capturing", "") or ""
    blocks = re.split(r"Sample\s+\d+:", cap)[1:]
    out = []
    for b in blocks:
        m = re.search(r"Exposure:\s*([\d.]+)\s*ms", b)
        out.append(float(m.group(1)) if m else None)
    return out


@dataclass
class ImageStack:
    """[N] Raw channels of one FOV: ``channels[name]`` is ``(Z, Y, X)`` uint16; ``voxel_size_nm = (z, y, x)``."""

    channels: dict
    voxel_size_nm: tuple
    header: dict = field(default_factory=dict)


MALFORMED = "malformed"


def _parse_calibration(raw):
    """Strictly parse ``volume.axesCalibrated``: a 3-sequence (x, y, z) of ``bool`` / ``numpy.bool_`` / ``None`` (absent).
    Anything else (strings, integers, a scalar, the wrong length) is *not* coerced: the axis becomes ``MALFORMED``."""
    def one(v):
        return None if v is None else bool(v) if isinstance(v, (bool, np.bool_)) else MALFORMED

    if raw is None:
        return dict(x=None, y=None, z=None)
    try:
        seq = tuple(raw)
    except TypeError:
        return dict(x=MALFORMED, y=MALFORMED, z=MALFORMED)
    if len(seq) != 3:
        return dict(x=MALFORMED, y=MALFORMED, z=MALFORMED)
    return dict(x=one(seq[0]), y=one(seq[1]), z=one(seq[2]))


def inspect_nd2(path):
    """Header only (no pixel data). Returns a dict; raises ``ND2Error`` for contract violations of the structure.

    Keys: ``sizes`` (dict, ``Z,C,Y,X``), ``dtype``, ``channel_names``, ``voxel_size_nm`` ``(z, y, x)``, ``z_step_um``,
    ``exposures_ms`` (per channel, ``None`` if unavailable), ``optics`` (``{name: {NA, n_immersion, emission_nm,
    objective}}``), ``axes_calibrated`` (``{name: {x, y, z}}``: ``True``/``False``/``None`` = flag absent/``"malformed"``), ``warnings`` (list of str). Voxel *baseline* conformity is not checked here (see
    ``load_nd2_channels``); it is reported as ``is_baseline_voxel``.
    """
    with nd2.ND2File(str(path)) as f:
        axes = tuple(f.sizes)
        if axes != AXES:
            raise ND2Error(f"unsupported ND2 axes {axes}; exactly {AXES} is required (no position/time axis, Z present)")
        sizes = dict(f.sizes)
        dtype = np.dtype(f.dtype)
        if dtype != np.uint16:
            raise ND2Error(f"unsupported ND2 dtype {dtype}; v0.1 requires uint16")
        chans = list(f.metadata.channels)
        names = [c.channel.name for c in chans]
        if len(names) != sizes["C"]:
            raise ND2Error(f"metadata lists {len(names)} channels but the C axis has {sizes['C']}")
        if any(not n for n in names) or len(set(names)) != len(names):
            raise ND2Error(f"channel names must be non-empty and unique; got {names}")
        vox = f.voxel_size()
        vals = (vox.z, vox.y, vox.x)
        if not all(v is not None and np.isfinite(v) and v > 0 for v in vals):
            raise ND2Error(f"voxel size (z, y, x) um = {vals} is missing or not finite and positive")
        voxel_nm = tuple(float(v) * 1000.0 for v in vals)
        if abs(voxel_nm[1] - voxel_nm[2]) > 1e-6:
            raise ND2Error(f"x and y pixel sizes differ: {voxel_nm[2]} vs {voxel_nm[1]} nm")
        step = None
        for e in f.experiment:
            if type(e).__name__ == "ZStackLoop":
                step = e.parameters.stepUm
        exps = parse_capturing_exposures(f.text_info)
        warns = []
        if len(exps) != len(names):
            warns.append(f"{len(exps)} exposure entries for {len(names)} channels: exposures not attached to channels")
            exposures = [None] * len(names)
        else:
            exposures = exps
        if step is not None and abs(step - vox.z) > DZ_STEP_WARN_UM:
            warns.append(f"z voxel {vox.z} um differs from ZStackLoop step {step} um (provisional threshold)")
        axes_cal = {}
        for c in chans:
            axes_cal[c.channel.name] = _parse_calibration(getattr(getattr(c, "volume", None), "axesCalibrated", None))   # nd2 order: (x, y, z)
        for n, fl in axes_cal.items():
            bad = [a for a, v in fl.items() if v is False]
            odd = [a for a, v in fl.items() if v == MALFORMED]
            if odd:
                warns.append(f"channel {n!r}: calibration flag of the {', '.join(odd)} axis is malformed in the ND2 metadata")
            if bad:
                warns.append(f"channel {n!r}: {', '.join(bad)} axis marked uncalibrated in the ND2 metadata")
            elif not odd and any(v is None for v in fl.values()):
                warns.append(f"channel {n!r}: axes calibration flag not present in the ND2 metadata")
        optics = {}
        for c in chans:
            optics[c.channel.name] = dict(
                NA=c.microscope.objectiveNumericalAperture, n_immersion=c.microscope.immersionRefractiveIndex,
                emission_nm=c.channel.emissionLambdaNm, objective=c.microscope.objectiveName)
    return dict(sizes=sizes, dtype=str(dtype), channel_names=names, voxel_size_nm=voxel_nm, z_step_um=step,
                exposures_ms=exposures, optics=optics, warnings=warns, axes_calibrated=axes_cal,
                is_baseline_voxel=all(abs(a - b) <= 1e-6 for a, b in zip(voxel_nm, BASELINE_VOXEL_NM)))


def load_nd2_channels(path, channel_names, require_baseline_voxel=True):
    """Load the named channels as raw ``(Z, Y, X)`` uint16 arrays into an ``ImageStack`` (channels in the requested order).

    All checks run before pixel data are read. ``channel_names`` must be unique and present in the metadata
    (``ChannelNotFound`` lists what is there). With ``require_baseline_voxel`` (default) a voxel other than
    ``(150, 65, 65)`` nm raises ``ND2Error``; pass ``False`` only to inspect/read such a file for diagnostics.
    """
    names = list(channel_names)
    if not names or len(set(names)) != len(names):
        raise ND2Error(f"requested channel names must be non-empty and unique; got {names}")
    hdr = inspect_nd2(path)
    missing = [n for n in names if n not in hdr["channel_names"]]
    if missing:
        raise ChannelNotFound(f"channel(s) {missing} not in the file; available: {hdr['channel_names']}")
    for n in names:                                                            # explicit "uncalibrated" is rejected whatever the number says
        for ax, v in hdr["axes_calibrated"][n].items():
            if v == MALFORMED:
                raise ND2Error(f"channel {n!r}: {ax} axis calibration flag is malformed in the ND2 metadata (not a boolean); "
                               "the pixel size cannot be trusted")
            if v is False:
                raise ND2Error(f"channel {n!r}: {ax} axis is marked uncalibrated in the ND2 metadata; its pixel size "
                               f"({hdr['voxel_size_nm']} nm, z/y/x) cannot be trusted for physical units")
    if require_baseline_voxel and not hdr["is_baseline_voxel"]:
        raise ND2Error(f"voxel (z, y, x) = {hdr['voxel_size_nm']} nm is not supported in v0.1 (only {BASELINE_VOXEL_NM}); "
                       "pass require_baseline_voxel=False to read it anyway")
    stacks = {}
    with nd2.ND2File(str(path)) as f:
        if tuple(f.sizes) != AXES:
            raise ND2Error(f"unexpected axes {tuple(f.sizes)}")
        arr = f.to_dask()  # (Z, C, Y, X), lazy
        for n in names:
            idx = hdr["channel_names"].index(n)
            stacks[n] = np.asarray(arr[:, idx]).copy()
    for n, a in stacks.items():
        if a.dtype != np.uint16 or a.ndim != 3:
            raise ND2Error(f"channel {n!r} loaded as {a.dtype} {a.shape}; expected 3D uint16")
    return ImageStack(stacks, hdr["voxel_size_nm"], hdr)
