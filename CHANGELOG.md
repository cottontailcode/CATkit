# Changelog

## 0.1.0 — first limited release (not yet published)

### What it does
`catkit run` analyses one ND2 field of view: Hoechst/DAPI Otsu segmentation → fixed-sigma 3D LoG detection → 3D Gaussian localization → aperture photometry →
cell assignment, and writes `spots.csv`, `candidates.csv`, `cells.csv`, `run_metadata.json` and one QC overlay per detection channel.

### Validation
* Algorithms are ported from a frozen research pipeline and compared with it function by function (E layer): 2,577/2,577 float fields or arrays bit-identical in the
  pinned environment; the reference's own synthetic tests (18/18, 3/3) pass; reference manifest 21/21. See `docs/VALIDATION.md`.
* Synthetic integration tests for every module and the CLI; one real ND2 field of view was analysed read-only as a feasibility check (no ground truth).
* An independent audit of the release candidate found and led to the fixing of: output writer safety (no write through links, non-empty destinations refused, atomic
  publication, no process-wide `umask` manipulation), rejection of ND2 metadata that marks voxel axes as uncalibrated or malformed, and QC overlay file-name
  collisions.

### Known limitations
Single FOV; ND2 only (`ZCYX`, `uint16`); voxel `(150, 65, 65)` nm only; fixed-sigma detection only; expanded nucleoid labels approximate the cellular region; σ_LoG is
estimated outside cells; intensities are camera counts; QC classes are fit-quality classes, not detection accuracy. See the README.

### Deferred
Automatic threshold, TIFF input, batch processing, masks export, chromatic registration, colocalization, Omnipose, additional voxel sizes.

### Release status
Licensed under GPL-3.0-only (`LICENSE`, SPDX `license` metadata, copyright holder Tiantian Shang). The package is still marked `Private :: Do Not Upload`
so it cannot reach PyPI by accident; the classifier is removed only when a PyPI release is decided.
