"""Command-line interface for CATkit.

Entry points (both call :func:`main`):

* ``catkit ...``            console script declared in pyproject.toml
* ``python -m catkit ...``  module execution via ``catkit/__main__.py``

Exit codes:

* 0 -- success (including ``--help`` and ``--version``)
* 1 -- the command ran but the input or analysis was rejected (message on stderr, no traceback)
* 2 -- usage error: invalid arguments, or no command given
"""

from __future__ import annotations

import argparse
import platform
import sys
from importlib import metadata

from catkit import __version__

DIST_NAME = "catkit-microscopy"

# Packages whose versions `catkit env` reports.
# Scientific dependencies (numpy, scipy, ...) will be added here in Milestone 2.
REPORTED_PACKAGES = ("pip", "setuptools", "pytest")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2  # same code argparse uses for invalid arguments


def package_version(name: str) -> str:
    """Return the installed version of ``name``, or ``"not installed"``."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def cmd_env(args: argparse.Namespace) -> int:
    """Print the software environment (basis for run provenance)."""
    print(f"catkit {__version__} (distribution: {DIST_NAME} {package_version(DIST_NAME)})")
    print(f"python {platform.python_version()}")
    print(f"python executable {sys.executable}")
    print(f"platform {platform.platform()}")
    for name in REPORTED_PACKAGES:
        print(f"{name} {package_version(name)}")
    return EXIT_OK


def cmd_run(args: argparse.Namespace) -> int:
    """Run the single-FOV pipeline (``catkit.pipeline.run_nd2``) and print a short summary."""
    from pathlib import Path

    # scientific imports are deferred so that `catkit --version` / `catkit env` stay light
    from catkit import pipeline
    from catkit.detection import InvalidNoiseSigma
    from catkit.io import ND2Error
    from catkit.provenance import is_dataless

    src, out = Path(args.input), Path(args.output)
    if not src.is_file():
        print(f"catkit: error: input file not found: {src}", file=sys.stderr)
        return EXIT_ERROR
    if is_dataless(src):
        print(f"catkit: error: {src.name} is only a cloud placeholder (not downloaded to this machine); synchronise it first", file=sys.stderr)
        return EXIT_ERROR
    try:
        pipeline.check_output_dir(out)                                           # unsafe/non-empty destination: fail before any work
    except (pipeline.OutputError, OSError) as exc:
        print(f"catkit: error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    roles = {"segmentation": args.segmentation, "detection": list(args.detect)}
    try:
        res = pipeline.run_nd2(src, out, roles, k=args.k)
    except (ND2Error, pipeline.PipelineError, InvalidNoiseSigma, ValueError, OSError) as exc:
        print(f"catkit: error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    m = res["metadata"]
    print(f"catkit run: {m['fov_id']}: {m['segmentation']['n_cells']} cells (segmentation channel {args.segmentation}), "
          f"k = {args.k:g}, voxel (z,y,x) nm = {tuple(m['voxel_size_nm_zyx'])}")
    for ch, c in m["localization_qc"].items():
        print(f"  {ch}: candidates {c['n_candidates']}, accepted {c['n_accepted']}, QC-passed (good) {c['n_qc_passed']}, "
              f"accepted in cells {c['n_cell_base_positive']}")
    for w in m["warnings"]:
        print(f"  warning: {w}", file=sys.stderr)
    print(f"outputs written to {out}: {', '.join(m['outputs'])}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser and its subcommands."""
    parser = argparse.ArgumentParser(
        prog="catkit",
        description="CATkit (Cell Atlas Toolkit): cell-resolved 3D fluorescence microscopy analysis.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    env_parser = subparsers.add_parser(
        "env", help="print Python and package versions (for provenance and debugging)"
    )
    env_parser.set_defaults(func=cmd_env)

    run_parser = subparsers.add_parser(
        "run", help="analyse one ND2 field of view (Otsu segmentation + fixed-sigma 3D detection + localization + photometry)",
        description="Analyse one ND2 file (one FOV). Channel names are the metadata names of the ND2 file (see `catkit run` errors for the "
                    "available names); CATkit never guesses which channel is which. Supported in v0.1: voxel (150, 65, 65) nm, uint16, "
                    "axes (Z, C, Y, X), fixed-sigma detection.")
    run_parser.add_argument("--input", required=True, help="path to the .nd2 file (read-only)")
    run_parser.add_argument("--segmentation", required=True, metavar="CHANNEL", help="nuclear-stain channel used for Otsu segmentation")
    run_parser.add_argument("--detect", required=True, nargs="+", metavar="CHANNEL", help="one or more spot-detection channels")
    run_parser.add_argument("--k", type=float, default=10.0, help="accepted threshold in units of sigma_LoG, strict > (default 10; must be >= 6)")
    run_parser.add_argument("--output", required=True, help="output directory: new or empty; written atomically (all files or none)")
    run_parser.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` (default: ``sys.argv[1:]``), run the command, return an exit code.

    ``--help``, ``--version``, and invalid arguments are handled by argparse,
    which raises ``SystemExit`` (code 0 or 2) instead of returning.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help(sys.stderr)
        return EXIT_USAGE

    return args.func(args)
