#!/usr/bin/env python
"""DIAGNOSTIC numeric fingerprint of frozen reference functions in the current environment.

Runs a few frozen functions (loaded by ref_id through the isolated, hash-verified
harness) on fixed synthetic inputs and prints a JSON fingerprint: per output its
dtype, shape, SHA-256 of the bytes and a few summary values, plus package
versions. Comparing fingerprints from two environments shows WHERE numerical
drift occurs; it is evidence for investigation, not a pass/fail gate.

Usage:
    <python> -I tools/reference_fingerprint.py run --out FILE.json      (needs CATKIT_RESEARCH_SRC/_MANIFEST
                                                                         or --root/--manifest)
    <python> -I tools/reference_fingerprint.py compare A.json B.json

Only the standard library, numpy, scipy, scikit-image and big-fish are needed
(catkit need not be installed: src/ is added to sys.path explicitly).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests" / "equivalence"))

from catkit.provenance import load_manifest  # noqa: E402
from refharness import isolated_references  # noqa: E402

VOXEL = (150.0, 65.0, 65.0)
REF_VOX = (150.0, 43.3333333, 43.3333333)


def _desc(np, value):
    a = np.asarray(value)
    flat = a.ravel()
    summary = [float(x) if a.dtype.kind in "fiu" else str(x) for x in flat[:4]] if flat.size else []
    if a.dtype.kind == "O":
        data = json.dumps(flat.tolist(), default=str).encode()
    else:
        data = np.ascontiguousarray(a).tobytes()
    return dict(dtype=str(a.dtype), shape=list(a.shape), sha256=hashlib.sha256(data).hexdigest(), head=summary)


def run(root: str, manifest_path: str) -> dict:
    import numpy as np

    manifest = load_manifest(manifest_path)
    out: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as tmp, isolated_references(
        manifest, root, ["REF-DET", "REF-LOC", "REF-PHOT", "REF-SEG"], Path(tmp)
    ) as m:
        det, loc, phot, seg = m["REF-DET"], m["REF-LOC"], m["REF-PHOT"], m["REF-SEG"]
        rng = np.random.default_rng(20261108)
        shape = (17, 64, 64)
        vol = 11.0 + rng.normal(0.0, 1.0, shape)
        kz, ky, kx = (np.arange(n, dtype=float) for n in shape)
        truths = [(8.3, 20.4, 21.7), (7.6, 44.2, 40.9), (9.1, 30.5, 50.2)]
        for mu in truths:
            vol += loc.gaussian_model((*mu, 2.1, 1.9, 1.7, 25.0, 0.0), kz, ky, kx)
        img = np.clip(np.rint(vol), 0, 65535).astype(np.uint16)

        psf = det.psf_from_optics(1.45, 670.0, 1.515)
        sr = (psf["sigma_z_nm"], psf["sigma_xy_nm"], psf["sigma_xy_nm"])
        kp = det.kernel_params(VOXEL, sr)
        filt, mlm = det.log_and_local_max(img.astype(np.float32), kp["log_kernel_sigma_px"], kp["minimum_distance_px"])
        sigma = det.noise_sigma_from_positive_log(filt, np.ones(shape[1:], bool))
        out["psf"] = _desc(np, [psf[k] for k in sorted(psf)])
        out["log_filter"] = _desc(np, filt)
        out["local_max_mask"] = _desc(np, mlm)
        out["sigma_log"] = _desc(np, sigma)

        rescale = lambda r: tuple(float(v * ref / w) for v, ref, w in zip(r, REF_VOX, VOXEL))
        ap, shin, shout = rescale((3.0, 5.0, 5.0)), rescale((5.0, 9.0, 9.0)), rescale((7.0, 13.0, 13.0))
        init = np.array([[round(c) for c in t] for t in truths], np.int64)
        occ = phot.build_occupancy(shape, init, ap)
        prim = phot.measure_spots(img, init.astype(float), ap, shin, shout, occ=occ, min_shell=50)
        for key in ("aperture_sum", "local_background", "bg_noise_std", "net_integrated", "snr_primary", "snr_peak_legacy_fitqc"):
            out[f"photometry.{key}"] = _desc(np, prim[key])

        cfg = dict(loc.DEFAULT_CFG)
        cfg["roi_half_xy"] = int(round(cfg["roi_half_xy"] * REF_VOX[1] / VOXEL[1]))
        cfg["mu_bound_xy"] *= REF_VOX[1] / VOXEL[1]
        cfg["large_shift_xy_px"] *= REF_VOX[1] / VOXEL[1]
        sig0 = tuple(r / v for r, v in zip(sr, VOXEL))
        fits = [loc.fit_spot(img, init[i], sig0, prim["local_background"][i], prim["peak_net"][i], prim["bg_noise_std"][i], cfg, VOXEL)
                for i in range(len(init))]
        out["fit.params"] = _desc(np, [[f[f"{n}_fit"] for n in loc.PNAMES] for f in fits])
        out["fit.diag"] = _desc(np, [[f["rss"], f["r2"], f["jac_cond"], f["se_z_px"], f["se_y_px"], f["se_x_px"]] for f in fits])
        out["fit.quality"] = _desc(np, [loc.classify_quality(f, 0.0, 0.0, prim["snr_peak_legacy_fitqc"][i], 0, cfg)[0]
                                       for i, f in enumerate(fits)])

        yy, xx = np.mgrid[0:96, 0:96]
        dapi = np.zeros((5, 96, 96))
        for cy, cx in ((25, 25), (25, 70), (70, 45)):
            dapi += 80.0 * np.exp(-(((yy - cy) / 8.0) ** 2 + ((xx - cx) / 5.0) ** 2))[None]
        dapi += np.random.default_rng(1).normal(0, 2.0, dapi.shape) + 20.0
        proj = np.max(seg.detect_focus_plane(dapi, 3), axis=0)
        labels, thr, cores = seg.segment_from_dapi(proj, 15, 4, opening_radius=1)
        out["seg.otsu_thr"] = _desc(np, thr)
        out["seg.labels"] = _desc(np, labels.astype(np.int32))
        out["seg.cores"] = _desc(np, cores.astype(np.int32))

    import importlib.metadata as md

    versions = {}
    for p in ("big-fish", "numpy", "scipy", "scikit-image", "pandas"):
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            versions[p] = None
    return dict(python=platform.python_version(), platform=platform.platform(), versions=versions, outputs=out)


def compare(a: dict, b: dict) -> int:
    print("versions A:", a["versions"])
    print("versions B:", b["versions"])
    same = diff = 0
    for key in sorted(set(a["outputs"]) | set(b["outputs"])):
        oa, ob = a["outputs"].get(key), b["outputs"].get(key)
        if oa is None or ob is None:
            print(f"MISSING   {key}")
            diff += 1
        elif oa["sha256"] == ob["sha256"]:
            same += 1
            print(f"identical {key}")
        else:
            diff += 1
            print(f"DIFFERENT {key}: dtype {oa['dtype']}/{ob['dtype']} shape {oa['shape']}/{ob['shape']} head {oa['head']} vs {ob['head']}")
    print(f"{same} identical, {diff} different (diagnostic only; not a gate)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="diagnostic numeric fingerprint of frozen reference functions")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--root", default=os.environ.get("CATKIT_RESEARCH_SRC"))
    r.add_argument("--manifest", default=os.environ.get("CATKIT_RESEARCH_MANIFEST"))
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        if not args.root or not args.manifest:
            print("error: --root/--manifest (or CATKIT_RESEARCH_SRC/_MANIFEST) required", file=sys.stderr)
            return 2
        Path(args.out).write_text(json.dumps(run(args.root, args.manifest), indent=1))
        print("wrote", args.out)
        return 0
    return compare(json.loads(Path(args.a).read_text()), json.loads(Path(args.b).read_text()))


if __name__ == "__main__":
    raise SystemExit(main())
