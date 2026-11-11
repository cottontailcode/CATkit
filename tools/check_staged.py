#!/usr/bin/env python3
"""Pre-commit safety check for the CATkit repository.

Scans files that are staged for commit (default) or all tracked files (--all)
and reports:

* forbidden file types (raw microscopy images, arrays, git bundles, ...)
* files larger than a size limit
* lines containing any term from a private deny-list

The deny-list itself is deliberately kept OUTSIDE the repository, because it
contains the very terms that must not be published. Pass it with --denylist or
the CATKIT_DENYLIST environment variable: one term per line, blank lines and
lines starting with '#' ignored. Matching is case-insensitive and whole-token:
a term must not be directly preceded or followed by a letter or digit, so a
short term such as "abc" does not match inside "abcde".

This is a quick screen, not a full privacy review: it cannot detect sensitive
content that is not on the deny-list, and it does not inspect Git history.

Exit codes: 0 = no problems, 1 = problems found, 2 = usage/setup error.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

FORBIDDEN_SUFFIXES = {
    ".nd2", ".czi", ".lif", ".tif", ".tiff", ".npy", ".npz",
    ".h5", ".hdf5", ".bundle", ".ipynb",
}
MAX_BYTES = 1_000_000  # 1 MB


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def list_files(all_tracked: bool) -> list[str]:
    if all_tracked:
        out = git("ls-files")
    else:
        # Added, copied, modified, or renamed files in the index (deleted files are skipped).
        out = git("diff", "--cached", "--name-only", "--diff-filter=ACMR")
    return [line for line in out.splitlines() if line]


def read_staged_text(path: str, all_tracked: bool) -> str:
    """Return the content that will actually be committed (the index version)."""
    if all_tracked:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    return git("show", f":{path}")


def load_denylist(path: Path) -> list[re.Pattern]:
    patterns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        term = line.strip()
        if term and not term.startswith("#"):
            patterns.append(re.compile(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", re.IGNORECASE))
    return patterns


def find_problems(files: list[str], patterns: list[re.Pattern], all_tracked: bool) -> list[str]:
    problems = []
    for path in files:
        suffix = Path(path).suffix.lower()
        if suffix in FORBIDDEN_SUFFIXES:
            problems.append(f"{path}: forbidden file type '{suffix}'")
            continue
        if Path(path).exists() and Path(path).stat().st_size > MAX_BYTES:
            problems.append(f"{path}: larger than {MAX_BYTES} bytes")
            continue
        text = read_staged_text(path, all_tracked)
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern in patterns:
                if pattern.search(line):
                    # Do not echo the matched term: the report itself may end up in logs.
                    problems.append(f"{path}:{lineno}: matches a deny-list term")
                    break  # one report per line is enough
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true", help="scan all tracked files instead of staged files")
    parser.add_argument("--denylist", type=Path, default=os.environ.get("CATKIT_DENYLIST"),
                        help="path to the private deny-list (default: $CATKIT_DENYLIST)")
    args = parser.parse_args(argv)

    if args.denylist is None or not Path(args.denylist).is_file():
        print("check_staged: deny-list not found; set --denylist or CATKIT_DENYLIST", file=sys.stderr)
        return 2

    # Always operate from the repository root so paths from git resolve correctly.
    os.chdir(git("rev-parse", "--show-toplevel").strip())

    files = list_files(args.all)
    problems = find_problems(files, load_denylist(Path(args.denylist)), args.all)
    scope = "tracked" if args.all else "staged"
    for problem in problems:
        print(problem)
    print(f"check_staged: scanned {len(files)} {scope} file(s), {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
