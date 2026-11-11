# CATkit — Cell Atlas Toolkit

CATkit analyses cell-resolved 3D fluorescence microscopy. **v0.1.0** implements one validated workflow for a single field of view (FOV) of a
Nikon ND2 file:

`ND2 → nuclear-stain (Hoechst/DAPI) Otsu segmentation → fixed-sigma 3D LoG spot detection → 3D Gaussian localization → aperture photometry → cell assignment → CSV tables + QC overlays`

The numerical core is ported from a frozen research pipeline and checked against it function by function (equivalence tests, see
[`docs/EQUIVALENCE.md`](docs/EQUIVALENCE.md) and [`docs/VALIDATION.md`](docs/VALIDATION.md)). Equivalence with the reference and synthetic tests show that
CATkit computes what the reference computes; they do **not** establish detection accuracy on your data.

> **Status: v0.1.0 (first limited release).** Licensed under GPL-3.0-only (see [License](#license)). The package still carries the
> `Private :: Do Not Upload` classifier so that it cannot be uploaded to PyPI by accident; it is removed only when a PyPI release is decided.

## Install

Requires **Python 3.11** (the only version validated). From a clone of the repository, in a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install . -c constraints/py311-research.txt     # exact versions of the validated dependency stack
catkit --version
```

For development use `python -m pip install -e ".[dev]" -c constraints/py311-research.txt`. Dependencies: NumPy, SciPy, scikit-image, Big-FISH 0.6.2, nd2,
Matplotlib.

## Quick start

```bash
catkit run \
  --input /path/to/image.nd2 \
  --segmentation DAPI \
  --detect Cy5 TRITC \
  --k 10 \
  --output ./results
```

* `--segmentation` and `--detect` take the **channel names stored in the ND2 metadata** (an unknown name is an error that lists the names in the file).
  CATkit never infers which channel is which, and never infers optics or calibration from file names.
* `--k` is the acceptance threshold in units of the LoG noise sigma (strict `>`; default 10; must be ≥ 6).
* The input is only read. `--output` must be a new or empty directory; results are written to a temporary directory and published in one step, so a
  failed run leaves no partial outputs. Errors are reported as one line with exit code 1.

`catkit --help`, `catkit run --help`, `catkit --version`, `catkit env` and `python -m catkit` also work.

## Output files

| File | Content |
|---|---|
| `spots.csv` | one row per **accepted** spot, all detection channels: initial integer and fitted sub-pixel `(z, y, x)`, Gaussian fit, localization quality (`good` / `suspect` / `unreliable`) with reasons, photometry, cell assignment |
| `candidates.csv` | every candidate of the fixed-sigma pool (LoG score `log_over_sigma`, `accepted`, cell at the initial coordinate); join with `spots.csv` on `(fov_id, channel, spot_id)` |
| `cells.csv` | one row per segmented cell: area, centroid, core area, border flag, and per-channel spot counts for separate assignment populations (initial-coordinate candidates/accepted; fitted-coordinate accepted/QC-passed; a diagnostic 8-px expansion) |
| `run_metadata.json` | input file name, size and SHA-256; channel roles; voxel and optics; parameters; segmentation, detection and localization QC; count reconciliation; software versions; warnings; file list |
| `qc_overlay_<n>_<channel>.png` | per detection channel: max projection with cell outlines and spots coloured by localization quality (`n` = position in `--detect`; the mapping to channel names is `qc_images` in `run_metadata.json`) |

Conventions: arrays are `(Z, Y, X)`; coordinates are `(z, y, x)`; `*_init` are integer voxel indices, `*_fit` / `*_use` voxel units, `*_um` micrometres; floats in
CSV use `%.10g` and an empty field means missing. Spots outside every segmented cell belong to no `cells.csv` row; they are tallied in `run_metadata.json`, and
CATkit stops with an error if the counts do not reconcile. Column definitions: [`docs/SCHEMAS.md`](docs/SCHEMAS.md).

## Scientific assumptions and limitations

* **Input:** one FOV per run; ND2 only; axes exactly `(Z, C, Y, X)`; `uint16`. Only the voxel size `(z, y, x) = (150, 65, 65)` nm is supported — other voxel sizes
  and spatial axes marked uncalibrated in the ND2 metadata are rejected, never resampled or overridden.
* **Detection:** fixed-sigma 3D LoG only. Candidates are the pool above 6 σ_LoG; a spot is *accepted* when its score exceeds `k` σ_LoG (default 10, the validated
  workflow's value, not a universally optimal threshold). σ_LoG is estimated from the LoG response in the outside-cell region (positive-half median / 0.6745),
  so it can understate the LoG noise inside cells. Automatic thresholding is **not** implemented.
* **Segmentation:** Otsu on the nuclear-stain channel (3 planes around the sharpest one, max projection). The expanded nucleoid labels approximate the
  cellular region; they are not whole-cell boundaries. Dim cells can be missed and touching cells merged.
* **Intensities** are camera counts (not photons) and are not comparable between channels in absolute terms. `snr_primary` is a background-limited
  approximation.
* **QC classes** (`good` / `suspect` / `unreliable`) classify the quality of the Gaussian fit. They do not establish biological detection accuracy; no ground
  truth is shipped.
* **Not implemented:** automatic threshold, TIFF input, batch processing, masks export, chromatic registration, colocalization, Omnipose, other voxel sizes.

## Tests

```bash
python -m pip install -e ".[dev]" -c constraints/py311-research.txt
CATKIT_REQUIRE_SCIENTIFIC_DEPS=1 python -m pytest -rs
```

The public tests are self-contained (synthetic data and a fake ND2 reader). Tests marked `research` compare CATkit with the private frozen reference code and tests marked
`real_nd2` read a user-authorised ND2 file; both are opt-in and **skip with a fixed reason** when their environment variables are not set (see
[`docs/EQUIVALENCE.md`](docs/EQUIVALENCE.md)). They were run on the maintainer's workstation for this release ([`docs/VALIDATION.md`](docs/VALIDATION.md)). A GitHub Actions
workflow runs the public tests and builds and smoke-tests the distributions on every push.

## Documentation

[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (design), [`docs/SCHEMAS.md`](docs/SCHEMAS.md) (data model and outputs), [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md) (what was ported and every
difference from the reference), [`docs/EQUIVALENCE.md`](docs/EQUIVALENCE.md) and [`docs/VALIDATION.md`](docs/VALIDATION.md) (validation layers and results), [`CHANGELOG.md`](CHANGELOG.md).

## License

Copyright (c) 2026 Tiantian Shang.

CATkit is licensed under **GPL-3.0-only** (SPDX: `GPL-3.0-only`); the full text is in [`LICENSE`](LICENSE). Commercial users may use CATkit under the GPL's terms.
Distributing CATkit, or a modified or combined work that includes CATkit code, may trigger the GPL's obligations (for example providing the corresponding source
under the same licence). Organizations seeking alternative licensing arrangements for proprietary distribution may contact the copyright holder; no
commercial licence terms are currently defined or offered.

CATkit's runtime dependencies (NumPy, SciPy, scikit-image, Big-FISH, nd2, Matplotlib and their dependencies) keep their own permissive licences; they are
installed separately and are not bundled in CATkit's wheel or sdist.
