# Equivalence and Validation Policy (Milestone 2.0)

How CATkit proves that ported code behaves like the validated frozen workflow,
and how other correctness evidence is kept separate. Results are recorded only
in `docs/VALIDATION.md`, and only for checks that were actually run.

## 1. Frozen references

* Reference files are identified by `ref_id` (e.g. `REF-LOC`). A **private**
  manifest (TSV, outside the repository) maps each id to its relative path, full
  SHA-256, size, role, dependencies and `import_policy`
  (`import-isolated | script-subprocess | read | never`).
* Reference files are never copied into the repository. Ported code carries a
  provenance note: `ref_id` + full SHA-256.
* `tools/verify_reference_manifest.py` verifies the whole manifest
  (exit 0 = all ok, 1 = problem, 2 = setup error); paths are hidden by default.

### Safety order for every active reference

1. file exists, its **fully resolved path (symlinks followed) lies inside the
   resolved reference root**, and it is **not a cloud placeholder** (checked via
   `st_flags` before any read, because reading a placeholder would trigger a
   download);
2. size + **SHA-256** match — for the **whole dependency closure** declared in
   the manifest, before anything is imported;
3. **controlled use**: in-process import or guarded subprocess.

### Import isolation (`tests/equivalence/refharness.py`)

* **Fail fast on cached modules**: if any module located under the reference
  root, or any module with a reference's name, is already in `sys.modules`,
  loading is refused (a cached module would run without re-reading the
  verified file and would bypass the import guard).
* `sys.dont_write_bytecode = True` and `sys.pycache_prefix` → a temp dir: no
  `.pyc` is written next to (or read from) reference sources.
* A meta-path **import guard** refuses any module under the reference root that
  is not in the verified closure (transitive imports included); after loading,
  `sys.modules` is diffed to confirm nothing unverified was loaded.
* `finally` restores `sys.path`, `sys.meta_path` and both bytecode settings,
  and removes from `sys.modules` **only** modules located under the reference
  root (third-party packages first imported during loading are kept).
* Subprocesses: `python -I -B -X pycache_prefix=<tmp>` with the same guard
  installed by a bootstrap, `MPLCONFIGDIR=<tmp>`, working directory `<tmp>`.
  The whole dependency closure is *verified*, but only the entry script runs
  and only `import-isolated` dependencies may be *imported*; `read`/`never`
  dependencies are refused by the guard. (The guard covers the import system,
  not a script that reads and `exec`s a file itself.)
  (`-I` ignores all `PYTHON*` environment variables, so cache control uses
  command-line options.)
* Before/after **snapshot** of the directories actually used (non-recursive,
  plus `__pycache__`): name, size, mtime, flags; hashes re-verified afterwards.
  A snapshot **detects** changes after the fact — it is **not** write protection.
  These post-use checks also run when the test body raises or a subprocess
  times out; a post-check failure is raised as a `ReferenceCheckError` chained
  to (and quoting) the original error, which is otherwise re-raised unchanged.
* References whose import has side effects beyond `sys.path` edits and
  read-only hash checks (scripts that run analyses or write files on import) are
  `never` imported.

### PASS / FAIL / SKIP

| Situation | Result |
|---|---|
| Harness self-tests (fake references) | always run in plain `pytest` |
| `research` tests, `CATKIT_RESEARCH_SRC` unset | SKIP with reason |
| `CATKIT_RESEARCH_SRC` set but manifest missing/malformed, file missing or outside root, dataless, hash mismatch, cached reference module, import error, Big-FISH ≠ 0.6.2, snapshot change | **FAIL** (setup errors appear as pytest ERROR) |
| `deps` tests, package absent, `CATKIT_REQUIRE_SCIENTIFIC_DEPS` unset | SKIP with reason |
| `deps` tests, package absent, `CATKIT_REQUIRE_SCIENTIFIC_DEPS=1` | **FAIL** |

Evidence: `pytest -rA --junitxml=<file>`; summaries in `docs/VALIDATION.md`.

## 2. Validation layers

General rules: integers, booleans, categories, strings, masks and IDs compare
**exactly**; NaN positions must match; ±Inf compares exactly; floats use
`|a − b| ≤ atol + rtol·|b|` with a tolerance chosen **per output**. **Both
`atol` and `rtol` are always stated; a value of 0 is written explicitly.** Never
rely on library defaults (e.g. `numpy.isclose` defaults rtol 1e-5, atol 1e-8).
"exact" means exact numerical equality for floats (`atol = rtol = 0`, NaN positions
equal; e.g. `+0.0 == -0.0`). Bit-identity is a stricter, separate property; it is
reported for E-layer outputs but is not what "exact" means here. Units follow
the axis (x, y in px or nm; z in slices or nm, converted with `dz`). A change in
any discrete classification is a FAIL even when floats are within tolerance.

| Layer | Compares | Pass rule |
|---|---|---|
| **E** function equivalence | frozen function vs CATkit port, same process, same inputs, same dependencies | table §3 (E column); bit-identity expected and reported |
| **P** pipeline equivalence | frozen per-FOV driver vs CATkit pipeline on identical synthetic arrays (loader/optics reader of the driver monkeypatched; outputs in a temp dir; no jitter): labels, cores, focus planes, Otsu threshold, σ_LoG per channel, candidates, accepted table, metadata | table §3 (P column); identical row counts and `spot_id` sets |
| **R** private regression (M2.3, user-authorised FOV) | CATkit rerun vs historical frozen outputs on disk (hashes recorded in the private manifest) | §4 |
| **D** determinism | two runs, same environment/config/input/seed | §5 |
| **V** synthetic ground truth (M2.3) | CATkit vs known truth, independent of frozen code | thresholds fixed in `VALIDATION.md` **before** the run |
| fingerprint (diagnostic) | frozen functions in two environments | reported only; **not a gate** |

## 3. Per-output tolerances (E and P)

P compares against frozen CSV files written with `%.6g` (≈ 6 significant
digits), hence rtol 1e-5; values read from JSON keep full precision.

| Output group | dtype | E: atol / rtol | P: atol / rtol | Notes |
|---|---|---|---|---|
| labels, cores, focus-plane indices, integer spot coordinates, `spot_id`, cell ids | int | exact | exact | |
| flags, quality, reasons, `params_at_bound`, tiers, `pass_fixed_*`, `accepted` | bool / str | exact | exact | any flip = FAIL |
| `log_response` array, local-max mask | float32 / bool | exact | — (not in CSV) | |
| **Thresholds / noise**: Otsu threshold, σ_LoG | float | 0 / 0 | 0 / 1e-12 (JSON) | σ_LoG has float32 precision |
| **Detection score** `log_over_sigma` | float32 | 0 / 0 | 0 / 1e-5 | |
| **Positions (index units)** `z_fit` [slices], `y_fit`, `x_fit` [px]; `*_use` | float64 | 1e-9 / 0 | 0 / 1e-5 | absolute in index units |
| **Positions (physical)** `*_um`, `dist_*_nm`, `nearest_spot_distance_nm` | float64 | 0 / 1e-12 | 0 / 1e-5 | z via `dz` |
| **Fit parameters** widths (`gaussian_sigma_*_nm`), amplitude, background | float64 | 0 / 1e-9 | 0 / 1e-5 | |
| **Fit diagnostics** `rss`, `rms`, `r2`, `reduced_chi2`, `jac_cond`, `se_*` | float64 | 0 / 1e-9 | 0 / 1e-5 | ±Inf exact |
| **Derived shifts** `shift_xy_px`, `shift_z_slices` | float64 | 1e-9 / 0 | 0 / 1e-5 | |
| **Photometry sums/counts** `aperture_sum`, `n_aperture_vox`, shell counts | float64 / int | exact | exact (where a reference value exists) | sums of integers |
| **Photometry derived** background, noise, `net_integrated`, `peak_net`, SNRs | float64 | 0 / 1e-12 | 0 / 1e-5 (where a reference value exists) | |

P compares only fields present in **both** outputs, keyed by the unique composite
key `(fov_id, channel, spot_id)`; missing rows in either direction are a FAIL.
Columns that CATkit exports but the frozen driver never wrote
(`aperture_sum`, `net_integrated`, `n_aperture_vox`, `bg_noise_std`) are compared
against the frozen *function* return values (E, or P via the reference
photometry call), never claimed against historical CSV files.

A float within tolerance but not bit-identical in E is a PASS with a recorded
note explaining the cause.

## 4. R — private-data regression

* Exact: row counts, `spot_id`, initial coordinates, cell assignments, quality,
  tiers, flags, label images.
* Fitted coordinates (index units): x/y atol 0.01 px, rtol 0; z atol 0.01
  slice, rtol 0.
* µm coordinates: x/y atol 0.01 · px_um, rtol 0; z atol 0.01 · dz_um, rtol 0.
* `log_over_sigma`, fit parameters, fit diagnostics, photometry present in the
  historical files: atol 0, rtol 1e-5 (CSV `%.6g` precision).
* Columns absent from the historical files (e.g. the four new photometry
  exports) have **no R baseline**; this is reported, not treated as a pass.
* **Any discrete change (quality, tier, cell assignment) is a FAIL.** A detailed
  per-spot diff is produced locally (never committed); the user adjudicates each
  case. Tolerances are never relaxed automatically.
* If the fingerprint shows numerical drift between environments, R results are
  interpreted together with it.

## 5. D — determinism

* CSV outputs: byte-identical.
* TIFF: decoded arrays identical, then file bytes compared; any byte difference
  is traced to a specific metadata tag and documented.
* `run_metadata.json`: identical after removing `timestamp`, `runtime_s`,
  `hostname` and absolute paths.

## 6. Running

```bash
# main environment: self-tests run, research/deps tests skip with reasons
python -m pytest -rA

# dependency validation in a fresh venv (scientific stack installed from constraints)
CATKIT_REQUIRE_SCIENTIFIC_DEPS=1 \
CATKIT_RESEARCH_SRC=<reference root> CATKIT_RESEARCH_MANIFEST=<private manifest> \
.venv-depcheck/bin/python -m pytest -rA -p no:cacheprovider --junitxml=<file>

# diagnostic fingerprint (run in two environments, then compare)
<python> -I tools/reference_fingerprint.py run --out A.json
<python> -I tools/reference_fingerprint.py compare A.json B.json
```
