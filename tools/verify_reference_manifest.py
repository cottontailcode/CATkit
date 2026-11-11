#!/usr/bin/env python
"""Verify every file of a private reference manifest (exists, not dataless, size, SHA-256).

Usage:
    python tools/verify_reference_manifest.py --manifest PATH --root DIR [--show-paths]
    (or set CATKIT_RESEARCH_MANIFEST and CATKIT_RESEARCH_SRC)

Prints one line per ref_id. Paths are hidden unless --show-paths is given,
so the output can be pasted into notes without exposing private locations.

Exit codes: 0 = all ok, 1 = at least one problem, 2 = setup error.
"""

from __future__ import annotations

import argparse
import os
import sys

from catkit.provenance import ManifestError, load_manifest, verify_manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=os.environ.get("CATKIT_RESEARCH_MANIFEST"))
    ap.add_argument("--root", default=os.environ.get("CATKIT_RESEARCH_SRC"))
    ap.add_argument("--show-paths", action="store_true")
    args = ap.parse_args(argv)
    if not args.manifest or not args.root:
        print("error: --manifest/--root (or CATKIT_RESEARCH_MANIFEST/CATKIT_RESEARCH_SRC) required", file=sys.stderr)
        return 2
    if not os.path.isdir(args.root):
        print("error: reference root is not a directory", file=sys.stderr)
        return 2
    try:
        manifest = load_manifest(args.manifest)
    except (OSError, ManifestError) as exc:
        print(f"error: cannot load manifest: {exc}", file=sys.stderr)
        return 2
    results = verify_manifest(manifest, args.root)
    for res in results:
        where = f"  {manifest[res.ref_id].relpath}" if args.show_paths else ""
        detail = f" ({res.detail})" if res.detail else ""
        print(f"{res.status:14s} {res.ref_id}{detail}{where}")
    n_bad = sum(not r.ok for r in results)
    print(f"{len(results)} entries, {n_bad} problem(s)")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
