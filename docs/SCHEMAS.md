# CATkit Data Schemas — draft v0 (`schema_version: "0.1-draft"`)

Status: **design document (Milestone 2.0)**. No scientific code implements this
yet. Every behaviour attributed to the *frozen* workflow was checked against the
hash-verified reference sources (cited by reference id, e.g. `REF-DET`; the id →
file mapping lives in a private manifest). Items that could not be confirmed are
listed under [To verify](#16-to-verify).

Each column/field carries a **source label**:

| Label | Meaning |
|---|---|
| `frozen` | computed exactly as in the validated reference workflow |
| `frozen-adapted` | same value, generalised input/naming (e.g. user-supplied channel names) |
| `new` | added by CATkit; no frozen counterpart, so no equivalence target |

Units are part of column names where ambiguous (`_px`, `_slices`, `_nm`, `_um`).
Raw camera counts are **not photons**; absolute intensities from different
channels are not comparable.

---

## 1. Images: axes and dtype

* ND2 native array order is `(Z, C, Y, X)`. Channels are selected **by the
  channel name stored in the file metadata**, never by position, and split into
  one `(Z, Y, X)` array per channel (`REF-IO`).
* Arrays keep their **uint16 raw counts**; no normalisation or background
  subtraction is applied to stored data.
* A float32 copy of a channel is used **only** as input to the LoG filter
  (Big-FISH rescales integer input and returns an integer-rounded response,
  which collapses low-count data; float32 keeps the response continuous).
* Biological roles (DNA stain / FISH channel A / FISH channel B) are assigned by
  the user in the configuration. CATkit never guesses them from names.

## 2. Voxel size

* Always stored as `voxel_size_nm = (z, y, x)`. (The ND2 reader returns
  `(x, y, z)`; CATkit reorders explicitly.)
* **v0.1 accepts exactly `(150, 65, 65)` nm** (per-axis tolerance 1e-6 nm, the
  same tolerance the frozen driver asserts). Any other voxel size is an error
  until it is separately validated. `x != y` is always an error.
* Both the voxel-size z (`dz`) and the Z-stack loop step (`z_step`) are recorded
  in `run_metadata.json`; a disagreement produces a warning (threshold: to verify).

## 3. Coordinates

* Index space, 0-based, `(z, y, x)`; **integer positions are voxel centres**;
  voxel `k` spans `[k − 0.5, k + 0.5]`. Fitted positions are fractional indices
  in the same frame (`REF-LOC`).
* Physical units, per axis:
  `x_um = x · px_um`, `y_um = y · px_um`, **`z_um = z · dz_nm / 1000`**.
  Standard errors and Gaussian widths convert the same way (z uses `dz`).
* z is **apparent stage units**; no refractive-index correction is applied in
  M2. (Axial scaling for distances is an M3 concern and is an unvalidated
  assumption there.)

## 4. Reference scales and rescaling (frozen behaviour preserved)

Two different reference scales exist and are both kept unchanged:

| Parameter group | Reference | Rule at pixel size `px` (nm) | Value at 65 nm |
|---|---|---|---|
| Segmentation `min_size` | 65 nm | `round(15 · (65/px)²)` | 15 |
| Segmentation `expand_labels` distance | 65 nm | `round(4 · 65/px)` | 4 |
| Segmentation opening radius | 65 nm | `round(65/px)` | 1 |
| Noise-mask dilation (iterations) | 65 nm | `round(10 · 65/px)` | 10 |
| Aperture semi-axes (z, y, x) | `REF_VOX = (150, 43.3333333, 43.3333333)` | `(3, 5, 5) · REF_VOX / voxel` | (3, 3.33, 3.33) |
| Background shell inner / outer | `REF_VOX` | `(5, 9, 9)`, `(7, 13, 13)` · `REF_VOX / voxel` | (5, 6, 6), (7, 8.67, 8.67) |
| Fit ROI half-width xy | `REF_VOX` | `int(round(7 · 43.3333333/px))` | 5 |
| Fit `mu` bound xy, large-shift xy | `REF_VOX` | `value · 43.3333333/px` | 1.33, 1.0 px |
| Fit sigma bounds | nm | converted with the actual voxel inside the fit | — |

Fixed pixel constants that are **not** rescaled (validated only at 65 nm):
`edge_flag` margin 10 px, core-distance search window 14 px, ambiguity margin
2 px, expanded-label variant 8 px. Note: the fit function's own default voxel
uses 43.333 while the driver uses 43.3333333; ports keep each value as written.

## 5. Input validation (errors, never silent inference)

Errors (`new` guards unless noted):

* ND2 axes other than exactly the reported `(Z, C, Y, X)` order (the nd2 library
  omits singleton axes; `frozen`: assertion); **any extra axis (position, time, …)
  is an error, even if a selector is passed** (multi-FOV/timepoint is out of
  scope); requested channel name absent; duplicate channel names.
* 2D input; dtype other than uint16 (v0.1); `x != y`; voxel not `(150, 65, 65)`.
* ND2 spatial axis explicitly marked **uncalibrated** (`volume.axesCalibrated` is `False`) on a requested channel, even if the number equals
  the baseline: error before pixels are read (a flag that is absent is only a warning; a flag that is not a boolean, or a flags value that is not a 3-sequence, is *malformed* and rejected the same way, never coerced).
* Output destination that is a symbolic link, not a directory, or a non-empty directory: error before any work; outputs are written to a
  temporary directory and published with one rename (all files or none).
* TIFF input without explicit `axes`, channel mapping and `voxel_size_nm`.
  OME-marked TIFFs follow the same contract (OME metadata is never interpreted;
  warning "OME metadata present but not used"); this is a *minimal explicit TIFF
  reader*, not OME-TIFF support (M4).
* Validation order in the reader: axes → shape → dtype (uint16) → unique channel
  names → finite positive voxel (+ baseline, `x == y`), all **before** pixels are
  materialised; header-only inspection reads no pixel data.
* **`Z < 3`** (focus-plane selection takes 3 planes; the frozen index arithmetic
  would produce a wrong slice); `Y` or `X` smaller than `2 · roi_half_xy + 1`.
* `fixed_sigma` mode with an empty background region or a non-finite / zero
  σ_LoG (the frozen code would silently yield a NaN threshold and no spots).

Warnings: `Z < 2 · roi_half_z + 1` (= 9; every fit ROI is z-truncated and
flagged `z_edge`); `dz` vs `z_step` disagreement.

## 6. Cell labels

* 2D `int32 (Y, X)`, `0` = background. IDs are `skimage.measure.label` order
  `1..N`, preserved by `expand_labels` (`REF-SEG`).
* Frozen method (`method = "otsu"`): DAPI/Hoechst plane with maximal standard
  deviation and its neighbours (3 planes) → max projection → Otsu threshold →
  opening (disk) → small-object removal (objects with area `< min_size`) →
  label → `expand_labels`.
* Three label images: `nucleoid_cores` (before expansion), `cell_labels`
  (expanded; **the frozen baseline used for assignment**), and
  `cell_labels_expanded8` (diagnostic variant; not adopted for analysis).
* The segmentation frame equals the FISH frame (no shift applied).
* `method = "external"` (user-supplied labels) is `new`. `omnipose` is optional
  validation only (M4); `both` produces two label sets plus a comparison table
  and **never merges** them.

## 7. Detection

### 7.1 Shared steps (`frozen`)

1. PSF from optics: σ_xy = 0.21 λ_em / NA; σ_z = FWHM_z / 2.3548 with
   FWHM_z = 2 n λ_em / NA² (`spot_radius_nm = (σ_z, σ_xy, σ_xy)`).
2. LoG σ (px) and local-maximum `min_distance` = `spot_radius_nm / voxel`;
   local-max box = `2·ceil(min_distance) + 1` per axis.
3. `log_filter` on **float32 raw counts**; Big-FISH negates the LoG and clips
   negative values to 0 → `log_response` (float32, counts × LoG units).
4. `local_maximum_detection` on `log_response`.

### 7.2 `fixed_sigma` mode (default; frozen)

* **σ_LoG** = `median(log_response[v > 0]) / 0.6745` over voxels outside
  `binary_dilation(cell_labels > 0, iterations = dilation)` (2D mask applied to
  all z planes). Under NumPy 2 this median is computed in float32 and stored as
  a Python float (float32 precision). It assumes zero-mean, symmetric LoG noise
  and is a *noise reference*, not a raw-image noise level.
* **Candidate pool**: `spots_thresholding(log_response, local_max, 6·σ_LoG,
  remove_duplicate=True)`. The comparison is `log_response > threshold`
  evaluated **in float32** (the Python-float threshold is rounded to float32;
  NEP 50). Duplicate local maxima are merged to their centroid, truncated to
  int64. The pool is built from the **returned coordinates and their order**
  of `spots_thresholding`; its second return (the mask after 3D de-duplication)
  must **not** be used to rebuild the pool (it can hold more `True` entries than
  coordinates returned).
* **Score**: `log_over_sigma = log_response_at_candidate / σ_LoG` — **float32**,
  dimensionless. It is a noise-normalised LoG score, **not a raw-image SNR**.
* **Accepted detections**: candidates with `log_over_sigma > k` — **strictly
  greater**, float32 comparison, default `k = 10.0`. The accepted set is a
  subset of the pool (it is *not* re-detected). `k` may be changed by the user;
  `k < candidate_floor_sigma` is an error.
* The default `k = 10` reproduces the frozen workflow for compatibility; it is
  **not** a universally optimal threshold.
* Pool-subset selection and direct thresholding at `k·σ_LoG` agreed on real data
  in historical checks, but are **not guaranteed identical** (centroid truncation
  on plateaus; float32 rounding at the boundary).

### 7.3 `auto` mode (`new`, M2.2)

* Big-FISH 0.6.2 automated elbow threshold, computed **independently per FOV ×
  channel** on that channel's `log_response` (never on a list of images, which
  would yield one shared threshold).
* Accepted detections = **every row returned by the detector**
  `spots_thresholding(log_response, local_max, thr_auto, remove_duplicate=True)`,
  applied directly to all local maxima (not restricted to the fixed-sigma pool,
  because `thr_auto` may lie below 6·σ_LoG). All returned rows have
  `accepted = True`; `log_response` is only the value *at the returned
  coordinate* and the table builder **never re-compares it with `thr_auto`**
  (a centroid of merged maxima can sit at a coordinate whose own response is
  below the threshold, even 0; re-thresholding would wrongly delete an accepted
  row). Regression test: a synthetic ring of equal maxima returns one row whose
  response at the coordinate is not ≥ `thr_auto`, and the row stays accepted and
  present in `candidates.csv`.
* The threshold always comes from the pinned public
  `bigfish.detection.automated_threshold_setting`; no second selector is written.
  `auto_status = "ok_boundary_elbow"`, `breakpoint_index` and
  `breakpoint_at_boundary` are defined on the **actual elbow-search sequence**
  (Big-FISH 0.6.2: candidate thresholds after the 5-point centred moving average
  and truncation `count_spots > 2`, i.e. `thresholds[:count_spots.size]`), not on
  the raw percentile grid. A thin QC-only diagnostic layer (pinned 0.6.2, uses
  Big-FISH's private helpers) recomputes that sequence; if it fails, the QC
  fields become "unavailable" with a warning and the threshold is unchanged.
* In `auto` mode `log_over_sigma` must be finite: σ_LoG ≤ 0 or non-finite is a
  typed **error**, with no change to the threshold's inputs and no fallback.
* `thr_auto` is in LoG units and depends on the candidate grid that Big-FISH
  anchors at `int(percentile_99.9999)`; it is not a physical quantity.
* Failure (`None`, `0`, non-finite, collapsed candidate range, exception) is an
  **error**; there is no fallback to `fixed_sigma`. A valid threshold located at
  the last candidate value is accepted with a **warning**
  (`auto_status = "ok_boundary_elbow"`).

### 7.4 Three levels — never double-counted

| Level | Definition | Where |
|---|---|---|
| **candidate** | row of the pool **returned by the detector** (`fixed_sigma`: `spots_thresholding` at `candidate_floor_sigma · σ_LoG`; see [To verify](#16-to-verify) 7–8) | `candidates.csv` |
| **detection accepted** | `fixed_sigma`: `log_over_sigma > k`; `auto`: every row returned by the detector at `thr_auto` (not re-compared) | `candidates.accepted` = rows of `spots.csv` |
| **QC passed** | accepted **and** `localization_quality == "good"` | `spots.csv` filter |

`n_candidates ≥ n_accepted ≥ n_qc_passed`; `len(spots.csv) == candidates.accepted.sum()`.
These inequalities hold for the global detection set of **one FOV × channel ×
mode** only. `cells.csv` stage columns are separate groupings (see §14).

**IDs and keys.** `spot_id` is local to (run, `fov_id`, `channel`, `mode`);
different modes may reuse the same integers, so **never join across modes on a
bare `spot_id`**. `(fov_id, channel, spot_id)` is a valid key only because one
FOV/channel has exactly one mode per run (enforced by a validator; tested).

### 7.5 Tiers (frozen definitions only)

* `tier_frozen` is written **only** for `mode = fixed_sigma, k = 10`:
  `"A"` = accepted (`log_over_sigma > 10`) and `localization_quality == "good"`;
  `"B10"` = accepted and not `good`. Otherwise it is empty.
* `run_metadata.tier_rule` states the rule in words for every run.
* The frozen sensitivity tiers B8 (8–10, separately re-fitted batch) and D
  (6–8, initial coordinates only) are **deferred to v0.2**.

### 7.6 Batch semantics (`frozen`)

Photometry, the occupancy map that excludes other spots' apertures from each
background shell, nearest-neighbour distances and overlap flags are computed
**within the accepted batch only**. Results for a spot can therefore differ if
the batch composition changes (e.g. a different `k`).

## 8. Spot coordinates per output

| Output | Coordinates used |
|---|---|
| Photometry (aperture, background, SNR, `peak_net`) | **initial** integer coordinates |
| Occupancy, neighbours, overlap flags | initial coordinates, within batch |
| Gaussian fit | starts at initial coordinates; `bg0`, `A0`, `noise` from photometry (`local_background`, `peak_net`, `bg_noise_std`) |
| Quality classification | uses `snr_peak_legacy_fitqc` (frozen) |
| `*_use`, `coord_source` | fit if quality ∈ {good, suspect}, else initial |
| `cell_init`, `dist_to_mask_um` | initial coordinates on `cell_labels` |
| `cell_base`, `cell_exp`, `in_core`, `dist_to_baseline_mask_nm` | `*_use` rounded with `numpy.round` (half-to-even), clipped to the image |

## 9. Fit failure and boundary handling (`frozen`, `REF-LOC`)

* The fit never raises; every result starts as all-NaN with
  `optimizer_converged = False` and fields are filled step by step. The actual
  failure contract (ports must preserve it; do not "clean" partial results):
  * ROI with fewer than 60 voxels → `fit_error = "roi_too_small"`; non-finite
    ROI values → `fit_error = "nonfinite_roi"`, `nonfinite = True`. Both return
    before optimisation: parameters NaN, not converged.
  * After the optimiser returns, `status`, `message`, `nfev`,
    `optimizer_converged` and all fitted parameters are written **before** the
    diagnostics. Non-finite fitted parameters set `nonfinite = True` and return
    early with those values kept.
  * An exception at any later step (residual statistics, Jacobian SVD,
    covariance inversion) only adds `fit_error = "<Type>: <message>"`
    (truncated to 120 characters); **values already written — including
    parameters and `optimizer_converged` — are kept**, later fields stay at
    their initial NaN/False values.
  * An exception inside the optimiser itself leaves parameters NaN.
  * Any `fit_error` makes the quality `unreliable`, so `*_use` falls back to the
    initial coordinates; the partially filled fit columns are still written.
* ROI centred on `int(round(init))`, clipped at image borders (`roi_*_truncated`).
* Standard errors are reported only if the Jacobian is full rank, its condition
  number is < 1e6, and no parameter is at a bound (`se_reported`).
* `unreliable`: fit error, not converged, non-finite, centre outside ROI,
  amplitude ≤ 1e-6, or a position parameter at its bound. `suspect`: sigma at
  bound, amplitude/background at bound, large shift, z/xy edge, neighbour in
  ROI, unreliable uncertainty, unreliable noise or low peak SNR. `good`: none.
  **Convergence alone never implies `good`.**
* No row is ever dropped because of a flag.

## 10. `candidates.csv` (one row per candidate, per channel)

| Column | Unit | Source |
|---|---|---|
| `fov_id`, `channel` | — | frozen-adapted |
| `spot_id` | 1..N in pool order | frozen |
| `z_init`, `y_init`, `x_init` | index | frozen |
| `log_response` | LoG units (float32) | new (frozen intermediate) |
| `log_over_sigma` | — (float32) | frozen |
| `raw_peak` | counts; raw voxel at the initial integer coordinate | frozen |
| `cell` | label id on `cell_labels` at the initial coordinate | frozen |
| `dist_to_mask_um` | µm | frozen |
| `edge_flag` | init within 10 px of an xy border | frozen |
| `pass_fixed_{6,8,10,12,14}sigma` | bool (`log_over_sigma > k`) | frozen |
| `accepted` | bool | new |
| `mode` | `fixed_sigma` \| `auto` | new |
| `candidate_floor_log` | LoG units | new |

`candidate_floor_log` is the LoG threshold **used to generate that mode's
candidate set** (`fixed_sigma`: `6·σ_LoG`; `auto`: `thr_auto`). It is **not** a
per-row minimum response (merged centroids can have a lower response at their
coordinate). `pass_fixed_*` are filled for `fixed_sigma` only.

## 11. `spots.csv` (accepted detections only)

Frozen primary-table columns are kept verbatim (superset):

* **Identity / provenance**: `fov_id`, `condition` (user-supplied, optional),
  `channel`, `exposure_ms_metadata`, `spot_id`, `localized_batch`.
* **Coordinates**: `z_init, y_init, x_init`; `z_fit, y_fit, x_fit`;
  `x_init_um, y_init_um, z_init_um`; `x_fit_um, y_fit_um, z_fit_um`;
  `x_use, y_use, z_use`, `coord_source`.
* **Detection**: `log_over_sigma`, `threshold_mode` (new).
* **Fit**: `optimizer_converged`, `gaussian_sigma_{x,y,z}_nm`,
  `gaussian_amplitude`, `gaussian_background`, `gaussian_fit_r2`, `jac_cond`,
  `params_at_bound`, `fit_error`, `se_{z,y,x}_nm`, `se_reported`,
  `shift_xy_px`, `shift_z_slices`.
* **Quality**: `localization_quality`, `localization_flag_reasons`,
  `flag_z_edge`, `flag_large_shift`, `flag_neighbor_in_roi`,
  `flag_sigma_at_bound`, `tier_frozen`.
* **Photometry**: `snr_primary`, `snr_peak_legacy_fitqc`, `peak_net`,
  `local_background`; added from the same frozen function output (`new` export
  columns, not new algorithms): `aperture_sum`, `net_integrated`,
  `n_aperture_vox`, `bg_noise_std`, with the frozen semantics in §12. They are
  validated against the frozen function's return values; historical CSV files
  do not contain them, so they have no R-layer baseline.
* **Flags**: `roi_z_truncated`, `roi_xy_truncated`, `aperture_z_truncated`,
  `aperture_truncated`, `edge_flag`, `nearest_spot_distance_nm`,
  `neighbor_in_roi_n`, `aperture_overlap`.
* **Assignment**: `cell_init`, `cell_base`, `cell_exp`, `in_core`,
  `core_d1_px`, `core_l1`, `core_d2_px`, `core_l2`, `assign_category`
  (`in_baseline` / `via_expansion` / `outside`), `ambiguous_cell`,
  `cell_fit_vs_init_differs`, `dist_to_baseline_mask_nm`.

The frozen Big-FISH `fit_subpixel` cross-check columns and the jitter re-fit
table are deferred (v0.2).

## 12. Intensity definitions — not interchangeable

| Quantity | Definition | Unit |
|---|---|---|
| `log_response` / `log_over_sigma` | LoG detection response / noise-normalised score | LoG units / — |
| `raw_peak` | raw voxel value at the initial integer coordinate (candidates) | counts |
| `peak_net` | max raw voxel in the aperture − `local_background` | counts |
| `aperture_sum` | sum of raw voxels in the ellipsoidal aperture (in-image voxels only); still reported when the background is insufficient | counts·voxel |
| `n_aperture_vox` | number of aperture voxels **actually sampled** (inside the image); `< n_aperture_vox_full` ⇔ `aperture_truncated` | voxels |
| `local_background` | **mean** of the background shell after iterative 3σ clipping (other spots' apertures and out-of-image voxels excluded) | counts |
| `bg_noise_std` | sample std (ddof = 1) of the retained shell voxels — primary noise; NaN if ≤ 2 voxels retained or background insufficient; 0 is kept (flag `noise_zero`), no floor | counts |
| `net_integrated` | `aperture_sum − local_background · n_aperture_vox`; negative values kept (not clipped at 0); NaN when the background is insufficient (`bg_insufficient`) | counts·voxel |
| `snr_primary` | `net_integrated / (bg_noise_std · √n_aperture_vox)` — background-limited approximation, not photon SNR | — |
| `snr_peak_legacy_fitqc` | peak SNR with the legacy noise rule; used **only** by the quality classifier | — |
| `gaussian_amplitude` | peak height of the continuous fitted Gaussian above its background (not a voxel value) | counts |
| `gaussian_background` | fitted constant background | counts |

## 13. `detection_qc` (per channel, in `run_metadata.json`; implemented in M2.2)

`mode`, `k_sigma`, `candidate_floor_sigma`, `threshold_value_log`,
`threshold_over_sigma`, `sigma_log`, `n_sigma_samples`, `background_fraction`,
`log_kernel_sigma_px`, `min_distance_px`, `local_max_box`, `log_input_dtype`,
`n_local_max_positive`, `n_candidates`, `n_accepted`, `n_qc_passed`, counts per
score band (6–8, 8–10, 10–12, 12–14, ≥14), and for `auto`: `auto_status`,
`p99_9999`, `candidate_grid {min, max, step, n}`, `range_collapsed`,
`n_curve_points`, `breakpoint_index`, `breakpoint_at_boundary`, path of the
saved elbow curve; plus `bigfish_version` and `warnings[]`.

Segmentation QC adds `n_cells` and, when no cell was segmented, the flag
`segmentation_empty` (valid input, all-zero labels; an input error is a different
thing). With zero cells every `cell` is 0, and the distances from
`distance_transform_edt(labels == 0)` carry no "nearest cell" meaning: any
exported missing-value convention for `dist_to_mask_um` is a separate [N]
decision. `candidate_floor_log` is defined in §10; the `auto` elbow fields are
defined on the retained search sequence (§7.3).

Note: σ_LoG is sampled outside cells and may underestimate the LoG noise inside
cells; the size of that difference depends on dye and sample and must be
assessed on each dataset.

## 14. Other outputs

* **`cells.csv`** (`new`, implemented M2.3-B): `fov_id`, `cell_id`, `area_px`, `area_um2`,
  `centroid_y_px`, `centroid_x_px`, `core_area_px`, `touches_border` (geometry of the
  **baseline expanded labels**; `core_area_px` from the cores before expansion), and per
  detection channel `<channel>_` + five counts that are never mixed:
  `n_candidates_cell_init` (all candidates, initial coordinate), `n_accepted_cell_init`
  (accepted, initial coordinate), `n_accepted_cell_base` and `n_qc_passed_cell_base`
  (accepted / accepted-and-`good`, **use** coordinate on the baseline labels = the frozen
  analysis assignment), `n_accepted_cell_exp_diag` (accepted, diagnostic 8-px expansion; not
  for analysis). One row per cell id present in the labels; spots outside every cell are in
  no row and are tallied in `run_metadata.json` (`cells_reconciliation`: for each population
  rows + outside = total, otherwise the run fails). ([To verify](#16-to-verify) 9 is closed.)
  Candidate counts use the **initial-coordinate** `cell`; accepted/QC counts (M2.3)
  use `cell_base` from the use-coordinate, so one spot may fall in different cells
  in the two groupings (counter-example test). Real cell rows list positive IDs
  only; candidates with `cell = 0` are reconciled by an explicit
  `n_candidates_outside`, so Σ(cell rows) + outside = `n_candidates`.
* **Masks**: `masks/{cell_labels,nucleoid_cores,cell_labels_expanded8}.tif`,
  int32, zlib-compressed.
* **`pairs.csv`** (M3 sketch): raw and registered coordinates side by side;
  per-spot match status `matched | not_mutual | ambiguous_second_nearest |
  beyond_gate`.
* **`run_metadata.json`**: `qc_images` (channel -> overlay file name, `qc_overlay_<index>_<channel>.png` with the channel part sanitized and cut to 40 characters; the index makes it unique, the full channel name is the metadata key), `outputs`, `schema_version`; input file names and SHA-256;
  user channel map and roles; voxel (`dz`, `z_step`); optics per channel;
  resolved configuration including every rescaled parameter; `tier_rule`;
  `detection_qc` per channel; catkit and dependency versions; warnings;
  `timestamp` and `runtime_s` (excluded from determinism checks).

## 15. Output precision

CSV conventions (implemented in M2.3-A): floats are written with `%.10g`; a missing value (NaN) is an **empty field**, `+inf`/`-inf` are
`inf`/`-inf`; booleans are `True`/`False`; columns follow the order of §10/§11. `spots.csv` uses `fov_id` (the frozen driver's
`image_id`), `localized_batch = "<k>sigma"` (`"10sigma"` for the default) and `threshold_mode = "fixed_sigma"`; `cell_exp` /
`assign_category` use the 8-px diagnostic expansion of the cores (never the analysis baseline), `ambiguous_cell` uses a 2 px margin.
`candidates.csv` is written next to `spots.csv`; join candidates and spots on `(fov_id, channel, spot_id)` only within one mode.

CATkit writes CSV floats with `%.10g`. The frozen workflow wrote `%.6g`, which
limits the precision of any comparison against historical files (see
`docs/EQUIVALENCE.md`).

## 16. To verify

1. Warning threshold for `dz` vs `z_step` disagreement.
2. ~~Structuring element of the noise-mask dilation~~ — **closed (M2.2-A)**: SciPy
   default cross element, `iterations=n` equals taxicab distance <= n (diamond
   growth); fixed by `test_noise_mask_dilation_is_cross_shaped_taxicab` and
   E-compared with the frozen expression. Not shown: effect on σ_LoG of other
   structuring elements.
3. ~~Behaviour when no cells are segmented~~ — **closed (M2.2-A, user decision Q4)**:
   frozen behaviour kept (whole frame = noise region) with `NoCellsWarning`,
   `n_cells = 0` recorded. The σ_LoG guard (empty background / non-finite / zero)
   is applied by detection (M2.2-B/C), not here.
4. General equivalence of pool-subset selection and direct thresholding
   (empirical only so far).
5. Byte-level determinism of TIFF writing (metadata tags).
6. Segmentation parameter convention at other pixel sizes (two historical
   conventions exist at 43.33 nm); out of scope for v0.1.

Deferred to M2.2–M2.3 (from the M2.0 audit):

7. *(implemented in M2.2-B for `fixed_sigma`)* Candidate membership is defined by the pool **actually returned by the
   detector** (`spots_thresholding` output, with centroid merging), not by
   re-evaluating `log_over_sigma > floor`; the two can differ at plateaus and
   at the float32 boundary.
8. *(decided, Revision 2 — to be implemented in C)* `auto` mode candidate
   universe and IDs: every accepted spot must have a
   candidate row; the fixed 6σ pool must not be reused.
9. *(closed in M2.3-B: candidate counts use the initial-coordinate `cell`; accepted/QC counts use `cell_base`)* `cells.csv` candidate counts: unaccepted candidates have no fitted/use
   coordinate, so candidate counts must use the initial-coordinate assignment
   (`cell`), or a documented fallback; to be fixed before implementation.
10. P-layer metadata: restrict the comparison to fields common to both outputs.
11. Fingerprint: if outputs ever differ, add difference statistics (hash and the
    first values alone cannot quantify the error).
