"""Synthetic nuclear-stain stacks with known nuclei (shared by the M2.2-A tests).

Pure NumPy; nothing here depends on the code under test, so expectations built on it are independent.
"""

from __future__ import annotations

import numpy as np

SHAPE = (7, 120, 120)
# (centre_y, centre_x, radius_px): well separated (min centre distance 38 px >> 2*expand=8)
NUCLEI = [(20, 20, 7), (22, 70, 6), (62, 30, 8), (90, 90, 7)]


def make_dapi_stack(nuclei=NUCLEI, shape=SHAPE, focus_z=3, bg=100.0, amp=1000.0, off_focus=0.25, noise=20.0, seed=0):
    """uint16 (Z, Y, X): bright discs; the plane ``focus_z`` has the highest standard deviation by construction."""
    z, y, x = shape
    yy, xx = np.mgrid[:y, :x]
    disc = np.zeros((y, x))
    for cy, cx, r in nuclei:
        disc[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 1.0
    rng = np.random.default_rng(seed)
    stack = np.empty(shape)
    for k in range(z):
        gain = 1.0 if k == focus_z else off_focus
        stack[k] = bg + gain * amp * disc + (rng.normal(0, noise, (y, x)) if noise else 0.0)
    return np.clip(np.rint(stack), 0, 65535).astype(np.uint16)
