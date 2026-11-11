"""File provenance helpers: SHA-256 hashing, placeholder detection, reference manifests.

This module is generic and contains no scientific code. It is used to

* record input-file hashes (later: ``run_metadata.json``), and
* verify a *reference manifest* -- a tab-separated list of files whose exact
  content must be confirmed before they are read, imported or executed.

Verification order for every file (the order matters):

1. the file exists and its fully resolved path (symlinks followed) lies
   inside the resolved reference root;
2. the file is not a cloud "dataless" placeholder (reading a placeholder would
   trigger a download, so it must be detected *before* hashing);
3. the size and the SHA-256 match the manifest.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "SF_DATALESS",
    "MANIFEST_COLUMNS",
    "IMPORT_POLICIES",
    "ManifestEntry",
    "ManifestError",
    "VerifyResult",
    "sha256_file",
    "is_dataless",
    "load_manifest",
    "verify_entry",
    "verify_manifest",
    "dependency_closure",
    "snapshot_dirs",
    "diff_snapshots",
]

#: macOS ``st_flags`` bit set on file-provider placeholders whose content is
#: not stored locally (``ls -lO`` shows ``dataless``).
SF_DATALESS = 0x40000000

MANIFEST_COLUMNS = (
    "ref_id",
    "relpath",
    "sha256",
    "bytes",
    "lines",
    "tier",
    "role",
    "import_policy",
    "depends_on",
    "milestone",
    "notes",
)

#: How a reference may be used. ``never`` entries are hashed for provenance
#: only and must not be imported or executed.
IMPORT_POLICIES = frozenset({"import-isolated", "script-subprocess", "read", "never"})

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    """The manifest file is malformed."""


@dataclass(frozen=True)
class ManifestEntry:
    ref_id: str
    relpath: str
    sha256: str
    bytes: int
    lines: int
    tier: str
    role: str
    import_policy: str
    depends_on: tuple[str, ...]
    milestone: str
    notes: str

    def path(self, root: str | os.PathLike[str]) -> Path:
        return Path(root) / self.relpath


@dataclass(frozen=True)
class VerifyResult:
    ref_id: str
    status: str  # "ok" | "missing" | "outside_root" | "dataless" | "size_mismatch" | "hash_mismatch"
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1 << 20) -> str:
    """Hex SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def is_dataless(path: str | os.PathLike[str]) -> bool:
    """True if the file is a cloud placeholder without local content.

    Uses ``os.stat`` only (metadata), so it never triggers a download.
    Always False on platforms without ``st_flags``.
    """
    return bool(getattr(os.stat(path), "st_flags", 0) & SF_DATALESS)


def load_manifest(path: str | os.PathLike[str]) -> dict[str, ManifestEntry]:
    """Parse and validate a TSV manifest; returns entries keyed by ``ref_id``."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise ManifestError(
                f"manifest header must be exactly {MANIFEST_COLUMNS}, got {reader.fieldnames}"
            )
        entries: dict[str, ManifestEntry] = {}
        for lineno, row in enumerate(reader, start=2):
            rid = (row["ref_id"] or "").strip()
            if not rid:
                raise ManifestError(f"line {lineno}: empty ref_id")
            if rid in entries:
                raise ManifestError(f"line {lineno}: duplicate ref_id {rid}")
            rel = row["relpath"] or ""
            if not rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
                raise ManifestError(f"line {lineno}: relpath must be relative without '..'")
            digest = (row["sha256"] or "").strip()
            if not _SHA256_RE.match(digest):
                raise ManifestError(f"line {lineno}: sha256 must be 64 lowercase hex characters")
            policy = (row["import_policy"] or "").strip()
            if policy not in IMPORT_POLICIES:
                raise ManifestError(f"line {lineno}: unknown import_policy {policy!r}")
            try:
                n_bytes, n_lines = int(row["bytes"]), int(row["lines"])
            except (TypeError, ValueError) as exc:
                raise ManifestError(f"line {lineno}: bytes/lines must be integers") from exc
            deps = tuple(d.strip() for d in (row["depends_on"] or "").split(";") if d.strip())
            entries[rid] = ManifestEntry(
                ref_id=rid,
                relpath=rel,
                sha256=digest,
                bytes=n_bytes,
                lines=n_lines,
                tier=row["tier"] or "",
                role=row["role"] or "",
                import_policy=policy,
                depends_on=deps,
                milestone=row["milestone"] or "",
                notes=row["notes"] or "",
            )
    for entry in entries.values():
        for dep in entry.depends_on:
            if dep not in entries:
                raise ManifestError(f"{entry.ref_id}: depends_on unknown ref_id {dep}")
    for rid in entries:  # reject dependency cycles early
        dependency_closure(entries, rid)
    return entries


def verify_entry(entry: ManifestEntry, root: str | os.PathLike[str]) -> VerifyResult:
    """Existence -> inside root -> dataless -> size -> SHA-256, stopping at the first failure.

    The containment check uses fully resolved paths (symlinks followed), so a
    relative path or symlink that points outside ``root`` is rejected before
    any content is read.
    """
    path = entry.path(root)
    if not path.is_file():
        return VerifyResult(entry.ref_id, "missing", "file not found")
    resolved_root = Path(root).resolve()
    resolved = path.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        return VerifyResult(entry.ref_id, "outside_root", "resolved path is outside the reference root")
    if is_dataless(path):
        return VerifyResult(entry.ref_id, "dataless", "cloud placeholder; sync it locally first")
    size = path.stat().st_size
    if size != entry.bytes:
        return VerifyResult(entry.ref_id, "size_mismatch", f"{size} bytes, expected {entry.bytes}")
    digest = sha256_file(path)
    if digest != entry.sha256:
        return VerifyResult(entry.ref_id, "hash_mismatch", f"sha256 {digest[:12]}..., expected {entry.sha256[:12]}...")
    return VerifyResult(entry.ref_id, "ok")


def verify_manifest(
    manifest: dict[str, ManifestEntry],
    root: str | os.PathLike[str],
    ref_ids: list[str] | None = None,
) -> list[VerifyResult]:
    """Verify the given entries (default: all) in manifest order."""
    ids = list(manifest) if ref_ids is None else list(ref_ids)
    unknown = [r for r in ids if r not in manifest]
    if unknown:
        raise KeyError(f"unknown ref_id(s): {unknown}")
    return [verify_entry(manifest[r], root) for r in ids]


def dependency_closure(manifest: dict[str, ManifestEntry], ref_id: str) -> list[str]:
    """``ref_id`` plus all transitive ``depends_on`` entries, dependencies first."""
    order: list[str] = []
    state: dict[str, str] = {}

    def visit(rid: str, stack: tuple[str, ...]) -> None:
        if state.get(rid) == "done":
            return
        if state.get(rid) == "active":
            raise ManifestError(f"dependency cycle: {' -> '.join(stack + (rid,))}")
        state[rid] = "active"
        for dep in manifest[rid].depends_on:
            visit(dep, stack + (rid,))
        state[rid] = "done"
        order.append(rid)

    visit(ref_id, ())
    return order


def snapshot_dirs(dirs: list[str | os.PathLike[str]]) -> dict[str, tuple]:
    """Non-recursive listing of each directory (and its ``__pycache__``).

    Records name, type, size, mtime_ns and st_flags. This only *detects*
    changes after the fact; it is not write protection.
    """
    snap: dict[str, tuple] = {}
    targets: list[Path] = []
    for d in dirs:
        d = Path(d)
        targets.append(d)
        if (d / "__pycache__").is_dir():
            targets.append(d / "__pycache__")
    for d in targets:
        if not d.is_dir():
            snap[str(d)] = ("<missing>",)
            continue
        for entry in os.scandir(d):
            st = entry.stat(follow_symlinks=False)
            snap[os.path.join(str(d), entry.name)] = (
                entry.is_dir(follow_symlinks=False),
                st.st_size,
                st.st_mtime_ns,
                getattr(st, "st_flags", 0),
            )
    return snap


def diff_snapshots(before: dict[str, tuple], after: dict[str, tuple]) -> list[str]:
    """Human-readable differences between two snapshots (empty list = identical)."""
    changes = []
    for key in sorted(set(before) | set(after)):
        if key not in after:
            changes.append(f"removed: {key}")
        elif key not in before:
            changes.append(f"added: {key}")
        elif before[key] != after[key]:
            changes.append(f"changed: {key}")
    return changes
