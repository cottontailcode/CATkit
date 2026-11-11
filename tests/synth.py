"""Synthetic volumes and caller-side rescaling helpers shared by M2.1 tests.

``make_volume`` is the generator of the frozen synthetic tests (voxel-integrated
3D Gaussians with known fractional centres). It uses the model under test, so
tests built on it show implementation consistency only; independent checks live
in ``test_localization_synthetic.py`` (midpoint integration).

The rescaling helpers *replay* what the frozen per-FOV driver does at the call
site (M2.3 owns the real helper); they are test code, not part of ``src``.
"""

from __future__ import annotations

import numpy as np

import catkit.localization as L

SHAPE = (17, 50, 70)
VOX = (150.0, 43.333, 43.333)              # voxel of the frozen synthetic tests (also fit_spot's own default)
VOX_BASE = (150.0, 65.0, 65.0)             # v0.1 baseline voxel (z, y, x) nm
REF_VOX = (150.0, 43.3333333, 43.3333333)  # reference scale of aperture/shell/ROI/mu-bound parameters
TRUE = dict(mu=(8.3, 24.4, 35.7), sig=(2.0, 3.0, 2.4), A=20.0, bg=10.0)
SIG0 = (2.7, 2.2, 2.2)
AP, SHIN, SHOUT = (3, 5, 5), (5, 9, 9), (7, 13, 13)


def make_volume(spots, bg=10.0, noise=0.0, seed=0, integer=False, shape=SHAPE):
    kz, ky, kx = (np.arange(n, dtype=float) for n in shape)
    vol = np.full(shape, bg, float)
    for mu, sig, A in spots:
        vol += L.gaussian_model((*mu, *sig, A, 0.0), kz, ky, kx)
    if noise:
        vol += np.random.default_rng(seed).normal(0, noise, shape)
    if integer:
        vol = np.clip(np.rint(vol), 0, 65535)
    return vol


def rescale_radii(radii, voxel):
    """Driver call-site rule: ``r * REF_VOX / voxel`` per axis."""
    return tuple(float(v * ref / w) for v, ref, w in zip(radii, REF_VOX, voxel))


def rescaled_cfg(cfg, voxel):
    """Driver call-site rule for the fit configuration (xy parameters only)."""
    out = dict(cfg)
    out["roi_half_xy"] = int(round(out["roi_half_xy"] * REF_VOX[1] / voxel[1]))
    out["mu_bound_xy"] *= REF_VOX[1] / voxel[1]
    out["large_shift_xy_px"] *= REF_VOX[1] / voxel[1]
    return out
