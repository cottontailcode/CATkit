"""Tests for catkit.provenance and tools/verify_reference_manifest.py (fake files only)."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from catkit import provenance as P

TOOL = Path(__file__).resolve().parents[1] / "tools" / "verify_reference_manifest.py"


def write_manifest(path: Path, rows: list[dict]) -> Path:
    lines = ["\t".join(P.MANIFEST_COLUMNS)]
    for row in rows:
        lines.append("\t".join(str(row.get(c, "")) for c in P.MANIFEST_COLUMNS))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def entry_for(root: Path, rid: str, rel: str, content: bytes, **extra) -> dict:
    f = root / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(content)
    row = dict(
        ref_id=rid, relpath=rel, sha256=hashlib.sha256(content).hexdigest(), bytes=len(content),
        lines=content.count(b"\n"), tier="t", role="r", import_policy="import-isolated",
        depends_on="", milestone="M", notes="",
    )
    row.update(extra)
    return row


@pytest.fixture
def ref_tree(tmp_path):
    root = tmp_path / "refs"
    rows = [
        entry_for(root, "REF-A", "pkg/a.py", b"x = 1\n"),
        entry_for(root, "REF-B", "pkg/b.py", b"y = 2\n", depends_on="REF-A"),
    ]
    manifest = write_manifest(tmp_path / "manifest.tsv", rows)
    return root, manifest


def test_sha256_file_matches_hashlib(tmp_path):
    f = tmp_path / "f.bin"
    data = os.urandom(3 * (1 << 20) + 17)  # spans several chunks
    f.write_bytes(data)
    assert P.sha256_file(f) == hashlib.sha256(data).hexdigest()


def test_is_dataless_false_for_regular_file(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("hello")
    assert P.is_dataless(f) is False


def test_verify_ok_and_closure(ref_tree):
    root, manifest_path = ref_tree
    manifest = P.load_manifest(manifest_path)
    assert [r.status for r in P.verify_manifest(manifest, root)] == ["ok", "ok"]
    assert P.dependency_closure(manifest, "REF-B") == ["REF-A", "REF-B"]


def test_verify_detects_hash_size_and_missing(ref_tree):
    root, manifest_path = ref_tree
    manifest = P.load_manifest(manifest_path)
    (root / "pkg/a.py").write_bytes(b"x = 9\n")  # same size, different content
    (root / "pkg/b.py").unlink()
    status = {r.ref_id: r.status for r in P.verify_manifest(manifest, root)}
    assert status == {"REF-A": "hash_mismatch", "REF-B": "missing"}
    (root / "pkg/a.py").write_bytes(b"x = 10\n")
    assert P.verify_entry(manifest["REF-A"], root).status == "size_mismatch"


def test_dataless_is_checked_before_hashing(ref_tree, monkeypatch):
    root, manifest_path = ref_tree
    manifest = P.load_manifest(manifest_path)
    monkeypatch.setattr(P, "is_dataless", lambda path: True)

    def must_not_hash(*a, **k):
        raise AssertionError("sha256 must not be computed for a dataless file")

    monkeypatch.setattr(P, "sha256_file", must_not_hash)
    assert P.verify_entry(manifest["REF-A"], root).status == "dataless"


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r: r[0].update(sha256="ABC"), "sha256"),
        (lambda r: r[0].update(import_policy="maybe"), "import_policy"),
        (lambda r: r[0].update(relpath="../escape.py"), "relpath"),
        (lambda r: r[1].update(ref_id="REF-A"), "duplicate"),
        (lambda r: r[1].update(depends_on="REF-NOPE"), "unknown"),
        (lambda r: r[0].update(depends_on="REF-B"), "cycle"),
    ],
)
def test_malformed_manifest_rejected(tmp_path, mutate, message):
    root = tmp_path / "refs"
    rows = [
        entry_for(root, "REF-A", "a.py", b"a\n"),
        entry_for(root, "REF-B", "b.py", b"b\n", depends_on="REF-A"),
    ]
    mutate(rows)
    with pytest.raises(P.ManifestError, match=message):
        P.load_manifest(write_manifest(tmp_path / "m.tsv", rows))


def test_bad_header_rejected(tmp_path):
    (tmp_path / "m.tsv").write_text("ref_id\tsha256\nX\tY\n")
    with pytest.raises(P.ManifestError, match="header"):
        P.load_manifest(tmp_path / "m.tsv")


def test_snapshot_detects_added_changed_removed(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "keep.py").write_text("a")
    (d / "gone.py").write_text("b")
    before = P.snapshot_dirs([d])
    (d / "gone.py").unlink()
    (d / "__pycache__").mkdir()
    (d / "__pycache__" / "keep.cpython-311.pyc").write_bytes(b"\0")
    os.utime(d / "keep.py", ns=(1, 1))
    diff = P.diff_snapshots(before, P.snapshot_dirs([d]))
    joined = "\n".join(diff)
    assert "removed:" in joined and "added:" in joined and "changed:" in joined
    assert P.diff_snapshots(before, before) == []


def run_tool(*args, env=None):
    return subprocess.run([sys.executable, str(TOOL), *args], capture_output=True, text=True, env=env)


def test_tool_exit_codes(ref_tree, tmp_path):
    root, manifest_path = ref_tree
    ok = run_tool("--manifest", str(manifest_path), "--root", str(root))
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert str(root) not in ok.stdout  # paths hidden by default
    (root / "pkg/a.py").write_bytes(b"x = 9\n")
    bad = run_tool("--manifest", str(manifest_path), "--root", str(root))
    assert bad.returncode == 1 and "hash_mismatch" in bad.stdout
    env = {k: v for k, v in os.environ.items() if not k.startswith("CATKIT_")}
    missing = run_tool(env=env)
    assert missing.returncode == 2
    broken = tmp_path / "broken.tsv"
    broken.write_text("nonsense\n")
    assert run_tool("--manifest", str(broken), "--root", str(root)).returncode == 2
