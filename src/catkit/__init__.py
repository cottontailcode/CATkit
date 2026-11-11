"""CATkit (Cell Atlas Toolkit).

A Python toolkit for cell-resolved 3D fluorescence microscopy analysis.
v0.1: single-field-of-view ND2 workflow (Otsu segmentation, fixed-sigma 3D LoG detection, Gaussian localization,
aperture photometry, cell assignment) behind ``catkit run``. See README.md for scope and limitations.

Copyright (c) 2026 Tiantian Shang. Licensed under GPL-3.0-only (see LICENSE).
"""

# Single source of truth for the package version.
# pyproject.toml reads this value at build time ([tool.setuptools.dynamic]).
__version__ = "0.1.0"

__all__ = ["__version__"]
