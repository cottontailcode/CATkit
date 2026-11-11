# Ported Algorithms: what was copied, what was edited

Every algorithm CATkit takes from the frozen research workflow is listed here
with its provenance and with **every** difference from the frozen source.
Frozen files are identified only by `ref_id` (see `docs/EQUIVALENCE.md`); the
private manifest holds paths and hashes. A function may be called *verbatim*
only while it has no entry in the "Edits" column below.

Equivalence to the frozen code (layer E) shows that a port behaves like the
frozen function. It is **not** evidence of real-data accuracy.

## 1. Ported modules

| CATkit module | Frozen source | SHA-256 of the frozen file | Milestone / status |
|---|---|---|---|
| `src/catkit/localization.py` | `REF-LOC` | `b596111de98c3d4b6f077a49c642f7d05a54542835b4e83319b483f985490136` | M2.1 checkpoint A — ported, E-verified |
| `src/catkit/photometry.py` | `REF-PHOT` | `32b0531066c073f35b833b30ce298ab86604ad1a63d78b748b91ce53294bb1ea` | M2.1 checkpoint B — ported, E-verified |
| `src/catkit/segmentation.py` | `REF-SEG` (+ `seg_params` from `REF-DRV`) | `48e856f327bd23ae59635c739c49d87f45ff6eb23a66527648ceaea29513aab6` (`REF-SEG`), `053bd566425d225ecf0e8ef532758125adb44d7f946b4dc597ee84bbda811aac` (`REF-DRV`) | M2.2 checkpoint A — ported, E-verified; plus [N] wrappers |
| `src/catkit/detection.py` | `REF-DET` (+ driver expressions of `REF-DRV`) | `8a33a51a20f994060f099f855c5f872949169ad3fc9d48c0870fbb43266da700` (`REF-DET`) | M2.2 checkpoint B — fixed_sigma; auto is checkpoint C |
| `src/catkit/io.py` | `REF-IO` (+ `optics_from_nd2` of `REF-DET`) | `305c5d1e535789a45e0a36dd9bd447ae981df657c46428847fbd1ff1224695b2` (`REF-IO`) | M2.2 checkpoint D (minimal ND2 reader) — E-verified header/pixels; rest [N] |
| `src/catkit/pipeline.py` | `REF-DRV` (`localize` and assignment block, call-site scaling) | `053bd566425d225ecf0e8ef532758125adb44d7f946b4dc597ee84bbda811aac` | M2.3-A — wiring only; one FOV, fixed_sigma |
| `src/catkit/cells.py` | `REF-DET` (`assign_cells`) | `8a33a51a20f994060f099f855c5f872949169ad3fc9d48c0870fbb43266da700` | M2.2 checkpoint A — one documented edit; `cell_geometry` is [N] |

### `localization.py` (from `REF-LOC`)

| Name | Edits |
|---|---|
| `PNAMES`, `DEFAULT_CFG` | none |
| `_axis`, `gaussian_model`, `gaussian_integral_voxels` | none |
| `roi_bounds`, `_nan_result` | none |
| `fit_spot` | none |
| `classify_quality` | none |
| `run_bigfish_fit_subpixel` | none |

Module-level differences (not algorithmic): the module docstring gains a
provenance paragraph; nothing else. Imports (including the module-level
`import bigfish.detection as bfd`) are unchanged, which is why `big-fish==0.6.2`
is a direct dependency.

How "no edits" is checked (`tests/equivalence/test_e_localization.py::test_port_is_verbatim`,
research mode): `inspect.getsource` of every function is identical to the frozen
source text, the constants compare equal, and the AST of the whole module apart
from the docstring is identical. Additionally, lines 11-203 of the frozen file
were compared with the port's body line by line when the file was created.

### `photometry.py` (from `REF-PHOT`)

| Name | Edits |
|---|---|
| `ellipsoid_voxels`, `build_occupancy` | none |
| `clipped_background` | none |
| `measure_spots` | none |
| `nearest_neighbor_nm`, `overlap_flags` | none |

Module-level differences (not algorithmic): the module docstring gains a
provenance paragraph; nothing else (imports unchanged: NumPy and
`scipy.spatial.cKDTree`; the module does not import `localization`). Checked by
`tests/equivalence/test_e_photometry.py::test_port_is_verbatim` (function source
text and whole-module AST apart from the docstring) and, when the file was
created, by comparing the port's body with lines 18-171 of the frozen file.

### `segmentation.py` (from `REF-SEG`, `REF-DRV`)

| Name | Origin | Edits |
|---|---|---|
| `detect_focus_plane`, `focus_slice_range`, `_remove_small`, `segment_from_dapi` | `REF-SEG` | none (the skimage >= 0.26 `max_size` shim is already inside the frozen `_remove_small`) |
| `seg_params` | `REF-DRV` (module-level one-liner) | none; the port line only appends a lint-marker comment |
| `noise_mask(labels, dil)` | expression inline in `REF-DRV` (`~binary_dilation(lab > 0, iterations=sp["dil"])`) | **[N] wrapper**: same expression, now a function |
| `NoCellsWarning`, `SegmentationResult`, `segment_fov` | none | **[N] new interface**: replays the driver chain (focus -> max projection -> `segment_from_dapi` -> int32) and adds guards: `Z >= 3`, 3D input, pixel size 65 nm only, `method` in `otsu`/`external` (`omnipose`/`both` -> `NotImplementedError("M4")`), `n_cells = 0` -> warning (not an error; frozen noise-region behaviour kept). Reports the Otsu threshold as a Python `float` (frozen returns it in the input dtype). |
| `register_mask`, `mad_display`, `HS_DX`, `HS_DY`, `HS_PX_UM` | `REF-SEG` | **not ported** (experiment-specific shift / display) |
| `expand_cores_variant` (8-px diagnostic) | `REF-DRV` inline | not ported; deferred to M2.3 |

### `detection.py` (from `REF-DET`, `REF-DRV`; fixed_sigma only)

| Name | Origin | Edits |
|---|---|---|
| `SIGMA_XY_COEF`, `FWHM_TO_SIGMA`, `psf_from_optics`, `kernel_params`, `log_and_local_max`, `noise_sigma_from_positive_log` | `REF-DET` | none (source text compared) |
| `build_candidate_pool`, `score_candidates`, `pass_flags`, `edge_flags`, `KS`, `CANDIDATE_FLOOR_SIGMA`, `DEFAULT_K` | expressions inline in `REF-DRV` | **[D]** same expressions, now functions; the detector's second return value (mask) is not used |
| `detect_fixed_sigma`, `candidates_table`, `validate_candidates`, `validate_run_modes`, `InvalidNoiseSigma`, `CANDIDATE_COLUMNS` | none | **[N]** new: orchestrates one FOV x channel; guards (3D uint16, Z >= 3, baseline voxel only, `k >= 6` finite, shapes, empty noise region / non-finite or non-positive sigma_LoG -> `InvalidNoiseSigma`); adds `mode` and `candidate_floor_log` columns; raw_peak is float like the driver |
| `optics_from_nd2` | `REF-DET` | not here (io, checkpoint D) |
| `count_curve`, `local_shell_stats` | `REF-DET` | not ported (diagnostics outside the driver chain) |

Frozen behaviour kept: pool built from the returned coordinates in the returned order, `resp / sigma` in float32 and
strict `> k`, `edge_flag` margin 10 px (not rescaled), centroid truncation of merged maxima (the centroid's response can be
below the threshold, even 0), `6 * sigma` threshold passed as a Python float (float32 comparison inside Big-FISH).
Frozen `REF-T2` tests ported: sigma-estimator recovery and nested thresholds (same data and asserts); the auto-search
collapse test belongs to checkpoint C.

### `io.py` (from `REF-IO`, `REF-DET`; ND2 only)

| Name | Origin | Edits |
|---|---|---|
| `parse_capturing_exposures` | `REF-IO` | none (source text compared) |
| `inspect_nd2` | `REF-IO.read_header` + `REF-DET.optics_from_nd2` | **[D]** same metadata expressions in one file open; voxel returned as `(z, y, x)` nm; adds validation (below) |
| `load_nd2_channels` | `REF-IO.load_channel_stacks` | **[N]** frozen axis assertion becomes an error; channel selection by metadata name (frozen too); adds validation before pixels, baseline-voxel guard, copy per channel |
| `ImageStack`, `ND2Error`, `ChannelNotFound` | none | **[N]** |
| filename parser, dye map | `REF-IO` | not ported (experiment-specific; CATkit never infers channels or roles) |

Validation order (all before pixels are read): axes exactly `(Z, C, Y, X)` -> `uint16` -> channel names present, unique, count equals the C axis
-> voxel finite and positive with `x == y` -> requested names exist -> baseline voxel `(150, 65, 65)` nm (override only with
`require_baseline_voxel=False`). Exposures are attached to channels by order (frozen assumption); a count mismatch yields `None` and a warning.
The `dz` vs `ZStackLoop` step warning threshold (1e-3 um) is provisional. Not implemented in this checkpoint: TIFF reader (explicit axes), channel-role config object,
`candidates.csv`/`cells.csv` writers.

### `pipeline.py` (from `REF-DRV`; M2.3-A, single FOV, fixed_sigma)

No algorithm is added. Per detection channel the call sequence of the frozen driver is reproduced ([D]): accepted rows (initial integer
`(z, y, x)`) -> `nearest_neighbor_nm`, `overlap_flags` (initial coordinates, within the accepted batch) -> `build_occupancy` -> `measure_spots`
-> `fit_spot` with `sigma0 = PSF / voxel`, `bg0 = local_background`, `A0 = peak_net`, `noise = bg_noise_std` -> `classify_quality` ->
`*_use` (fit if good/suspect, else initial) -> cell lookup at `numpy.round(use)` (half-to-even, clipped).
`rescale_radii`/`rescaled_cfg` moved from `tests/synth.py` into `src` ([D]: `r * REF_VOX / voxel`, test-checked equal to the old helpers).
[N]: explicit channel roles (`{"segmentation": name, "detection": [names]}`), `analyze_fov`, `run_nd2`, table assembly, CSV/JSON writers,
count reconciliation (`PipelineError` if `n_candidates >= n_accepted >= n_qc_passed`, rows == accepted, or assignment counts do not sum),
metadata. A channel without optics, a voxel other than the baseline, or an unusable sigma_LoG is an error (not skipped, not guessed).
Zero accepted spots give empty tables; zero cells give `segmentation_empty`, a recorded warning and all spots `outside`.
Also here: `cells.core_distances` (verbatim `REF-DRV`), `cells.assign_use_coordinates` ([D] driver block), `segmentation.expand_cores_variant` ([D]).
Not done: Big-FISH `fit_subpixel` cross-check and jitter columns, tiers B8/D, masks TIFF, `cells.csv`, `pairs.csv`.

### `cells.py` (from `REF-DET`)

| Name | Origin | Edits |
|---|---|---|
| `assign_cells_init` | `REF-DET.assign_cells` | **one edit**: the `shift_dy`/`shift_dx` parameters and the shifted-mask branch are removed (the driver always passes (0, 0)). With the default shift the behaviour is identical (E-tested incl. out-of-range and negative coordinates). Not a verbatim copy; the E test pins the signature and the absence of a shift branch. |
| `cell_geometry` | none | **[N] new**: `area_px`, `area_um2`, centroids, `core_area_px`, `touches_border` (synthetic hand-computed tests only) |

How "no edits" is checked (`tests/equivalence/test_e_segmentation.py::test_port_is_verbatim`, research mode):
`inspect.getsource` of each function equals the frozen text.

## 2. Frozen behaviours preserved on purpose (not bugs fixed here)

Observed while writing the E tests; each is reproduced by the port and compared
against the frozen function.

* `detect_focus_plane`/`focus_slice_range` with fewer than 3 planes: `start` becomes
  negative (Z=1, 2) and the slice is wrong; the port function reproduces it, the guard is in
  `segment_fov` ([N]).
* `~binary_dilation(labels > 0, iterations=n)`: SciPy default cross element (taxicab growth);
  `iterations < 1` means "repeat until stable" (one cell fills the frame). A label image with
  no cell gives an all-True mask.
* `seg_params` uses Python `round` (half-to-even): `round(2.5) = 2`, `round(0.5) = 0`.
* `assign_cells(_init)`: a coordinate >= image size raises `IndexError`; a negative coordinate
  wraps silently.
* `fit_spot` with a **negative** initial coordinate: the ROI upper bound can be
  negative, and the NumPy slice `image[0:-15]` silently wraps, producing a
  non-empty, meaningless ROI instead of `roi_too_small`. Real calls never pass
  negative coordinates (spots come from inside the image).
* `classify_quality`: `A` at its *lower* bound (`A_lo`) raises no reason by
  itself (only `A_hi` and `bg_lo`/`bg_hi` do; a non-positive amplitude is caught
  by `amp_nonpositive`).
* `classify_quality`: a `+Inf` peak SNR is not finite, so it yields
  `noise_unreliable`.
* `fit_spot` writes `jac_rank`/`jac_cond` **before** comparing with
  `cond_max`; a failure at that comparison keeps them (and the parameters and
  `optimizer_converged`) and only adds `fit_error`.
* The number of keys of the all-NaN result template is 33; the private
  `_roi_lo`/`_roi_hi` keys exist only after a fully successful fit.
* `measure_spots` counts: `n_shell_vox` is the shell size after occupancy
  exclusion but **before** non-finite removal and clipping (non-finite voxels are
  included); `n_bg_voxels_valid` is the post-clip retained count and is what
  `min_shell` (default 100) is compared with.
* Non-finite voxels in the **background shell** are dropped and counted; in the
  **aperture** they are not filtered and propagate to `aperture_sum`,
  `peak_raw`, `net_integrated`, `peak_net` (and SNR when the background is
  valid). With insufficient background the SNR stays NaN regardless.
* Clipping uses `std` with `ddof=0`; the reported `noise_std` is `ddof=1`
  (NaN for at most 2 retained voxels). No noise floor is applied anywhere;
  `net_integrated` is not clipped at 0.
* `build_occupancy` is `uint8`: a voxel covered by 256 apertures wraps to 0.
* Ellipsoid membership is evaluated in float (`d² = Σ((k−c)/r)²`, `d² ≤ 1`). A voxel lying *exactly* on the ellipsoid surface in exact arithmetic (e.g. offsets (12, 5) with radii 13: the 5‑12‑13 triple) can therefore be excluded by float rounding (`(12/13)²+(5/13)²` evaluates to slightly above 1). Observed with an exact-rational oracle: the shell `(5,9,9)`/`(7,13,13)` around an integer centre differs from the exact set by 8 voxels, and the port and the frozen code agree with each other (E-layer). Not fixed; hand-computed synthetic expectations use radii with no surface voxels (asserted with a gap > 1e-4).
  Only this behaviour is checked (255 stays 255, 256 gives 0); no claim is made
  about real data.
* `nearest_neighbor_nm` returns the neighbour index `j` from `cKDTree`; for
  duplicate points (distance 0) which index is returned depends on the tree.
  The E tests compare `j` exactly and observed identical values in the pinned
  environment; the driver discards `j`.
* The module docstring of `REF-PHOT` (kept unchanged) describes an earlier noise
  rule (`1.4826*MAD`, std only if MAD is 0). The code's primary noise is the
  clipped sample std (`ddof=1`); the old rule survives as `bg_noise_legacy` /
  `snr_peak_legacy_fitqc`. The code is authoritative.

## 3. Synthetic tests: assertion map

The localization tests of the frozen synthetic script were ported to
`tests/synthetic/test_localization_synthetic.py`. Each ported test keeps the
same data, seeds (40 in the noisy test), thresholds and assertion expressions;
a script comparing the parsed `assert` expressions found them identical
(5 + 5 + 1 + 5 + 1 + 2 = 19 asserts in six tests, plus the fit half of the
edge test). Only the helper names differ (`L`, `CFG`, `VOX`, `make_volume` come
from `catkit.localization` and `tests/synth.py`).

| Frozen test | pytest test | Note |
|---|---|---|
| axis_order_and_noiseless_recovery | `test_axis_order_and_noiseless_recovery` | 5 asserts, identical |
| noisy_recovery_statistics | `test_noisy_recovery_statistics` | 5 asserts, 40 seeds, thresholds unchanged |
| uncertainty_reported_when_wellconditioned | `test_uncertainty_reported_when_wellconditioned` | 1 assert |
| deterministic_failures_preserved | `test_deterministic_failures_preserved` | 5 asserts |
| convergence_is_not_quality | `test_convergence_is_not_quality` | 1 assert |
| bigfish_fit_subpixel_wrapper_axes | `test_bigfish_fit_subpixel_wrapper_axes` | 2 asserts |
| edges_flagged_not_dropped | `test_edges_flagged_not_dropped_fit_part` (localization) + `test_edges_flagged_not_dropped_measure_part` (photometry) | **split**: the only structural edit; assert 3 (fit) is in the localization file, asserts 1-2 (photometry) in the photometry file |
| aperture_net_noiseless_and_units | `test_aperture_net_noiseless_and_units` | 4 asserts, identical |
| background_only_not_clipped_and_unbiased | `test_background_only_not_clipped_and_unbiased` | 3 asserts, 200 centres, identical |
| background_mean_not_quantised | `test_background_mean_not_quantised` | 1 assert |
| primary_noise_is_clipped_sample_std_ddof1 | `test_primary_noise_is_clipped_sample_std_ddof1` | 3 asserts |
| snr_primary_uses_std_and_mad_is_secondary | `test_snr_primary_uses_std_and_mad_is_secondary` | 5 asserts |
| mad_zero_low_count_background | `test_mad_zero_low_count_background` | 4 asserts |
| degenerate_flat_background_no_floor | `test_degenerate_flat_background_no_floor` | 3 asserts |
| tiny_noise_not_floored | `test_tiny_noise_not_floored` | 1 assert |
| insufficient_background_samples_flagged | `test_insufficient_background_samples_flagged` | 2 asserts |
| nonfinite_background_voxels_dropped_and_flagged | `test_nonfinite_background_voxels_dropped_and_flagged` | 2 asserts |
| neighbors_flagged_and_excluded_from_shell | `test_neighbors_flagged_and_excluded_from_shell` | 3 asserts |

Photometry port check (parsed `assert` expressions compared by script): the 11
photometry-only tests contain 31 asserts and the edge test's photometry half 2;
all 33 are identical to the frozen expressions, and the non-assert statements
(data, seeds, radii) of the 11 tests are identical as well. Only the helper
names differ (`I` is `catkit.photometry`; `L` and `make_volume` come from
`catkit.localization` and `tests/synth.py`). Together with the localization
tests this accounts for all 18 frozen tests (the edge test is split in two).

`np.allclose`/`np.isclose` calls keep their original arguments, hence the same
effective (library-default `rtol`) thresholds as the frozen script.

Added tests beyond the frozen script: DEFAULT_CFG literal values; baseline-voxel
(150, 65, 65) nm recovery with the caller-side rescaling replayed in test code;
partial-failure retention; independent midpoint-rule integration of the voxel
model (section 4 of `docs/VALIDATION.md`).

Added photometry tests beyond the frozen script (`test_photometry_synthetic.py`,
part 2): exact-rational brute-force enumeration of aperture/annulus voxel sets
(integer and fractional centres and radii, edge clipping, baseline and reference
scaled radii) and of occupancy overlaps (expectations do not come from
`ellipsoid_voxels`); a hand-computed `net_integrated`/SNR example;
negative net intensity and absence of a noise floor; the `min_shell` 99/100
semantics; aperture-vs-shell non-finite handling; the `uint8` wrap.

## 4. E-layer coverage of `photometry.py`

`tests/equivalence/test_e_photometry.py` compares the whole return structure of
all six functions under explicit specs (`compare.py`: `PHOT_SPEC` 30 keys,
`CLIPBG_SPEC` 7 keys, ellipsoid/occupancy/neighbour/overlap specs; unknown key =
error). Raw sums, maxima, counts, indices and flags are compared exactly
(`peak_raw`, `aperture_sum` explicitly); derived photometry and physical
distances use atol 0, rtol 1e-12. `tests/equivalence/test_e_callsite.py` replays
the driver's call chain; see `docs/VALIDATION.md`.
