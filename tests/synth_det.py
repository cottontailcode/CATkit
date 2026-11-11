"""Synthetic raw-count volumes for the M2.2-B detection tests (pure NumPy; independent of the code under test)."""

from __future__ import annotations

import numpy as np

SHAPE = (17, 100, 100)
VOX = (150.0, 65.0, 65.0)
OPTICS = dict(NA=1.4, emission_nm=580.0, n_immersion=1.515)


def make_channel(spots=(), shape=SHAPE, bg=100.0, noise=6.0, seed=0, sig=(2.4, 1.5, 1.5)):
    """uint16 (Z, Y, X): Gaussian background noise plus planted Gaussian spots ``(z, y, x, amplitude)``."""
    z, y, x = shape
    zz, yy, xx = np.meshgrid(np.arange(z, dtype=float), np.arange(y, dtype=float), np.arange(x, dtype=float), indexing="ij")
    vol = bg + np.random.default_rng(seed).normal(0, noise, shape)
    for cz, cy, cx, amp in spots:
        vol += amp * np.exp(-0.5 * (((zz - cz) / sig[0]) ** 2 + ((yy - cy) / sig[1]) ** 2 + ((xx - cx) / sig[2]) ** 2))
    return np.clip(np.rint(vol), 0, 65535).astype(np.uint16)


# ---- shared two-channel FOV for the M2.3-A pipeline tests: (z, y, x, amplitude); cell 1, cell 2, outside
NUC = [(30, 30, 9), (62, 72, 9)]
CY5 = [(8.3, 30.4, 29.7, 220.0), (8.0, 60.2, 72.0, 160.0), (8.0, 80.0, 15.0, 180.0)]
TRITC = [(8.0, 32.0, 28.0, 200.0), (8.0, 63.0, 70.0, 150.0)]


def make_image(dapi=None, cy5=None, tritc=None):
    """``ImageStack`` with channels DAPI/Cy5/TRITC, baseline voxel, optics and exposures in the header (no file involved)."""
    from catkit.io import ImageStack
    from synth_seg import make_dapi_stack
    hdr = dict(optics={n: OPTICS for n in ("Cy5", "TRITC", "DAPI")}, channel_names=["Cy5", "TRITC", "DAPI"], exposures_ms=[400.0, 300.0, 100.0])
    ch = {"DAPI": make_dapi_stack(shape=(17, 100, 100), nuclei=NUC, seed=1) if dapi is None else dapi,
          "Cy5": make_channel(CY5, seed=7) if cy5 is None else cy5, "TRITC": make_channel(TRITC, seed=8) if tritc is None else tritc}
    return ImageStack(ch, VOX, hdr)
