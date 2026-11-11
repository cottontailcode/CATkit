"""Negative regression tests for harness defects found in the M2.0 audit (B1-B3 + exception paths).

Each test reproduces a scenario that the first M2.0 harness accepted silently.
All use fake references in tmp_path; no private data.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from catkit import provenance as P
from catkit.provenance import MANIFEST_COLUMNS, load_manifest
from refharness import ReferenceCheckError, isolated_references, run_reference_script


def make_refs(base: Path, files: dict[str, tuple[str, str, str, str]]):
    root = base / "research"
    lines = ["\t".join(MANIFEST_COLUMNS)]
    for rid, (rel, src, policy, deps) in files.items():
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        data = src.encode()
        f.write_bytes(data)
        lines.append("\t".join([rid, rel, hashlib.sha256(data).hexdigest(), str(len(data)), str(data.count(b"\n")),
                                "t", "r", policy, deps, "M", ""]))
    (base / "manifest.tsv").write_text("\n".join(lines) + "\n")
    return root, load_manifest(base / "manifest.tsv")


def uniq(tmp_path, stem):
    return f"{stem}_{abs(hash(str(tmp_path))) % 10**8}"


@pytest.fixture
def scratch(tmp_path):
    d = tmp_path / "scratch"
    d.mkdir()
    return d


# --- B1: cached modules must not bypass verification -------------------------------------------

def test_b1_stale_cached_module_with_same_path_is_refused(tmp_path, scratch):
    name = uniq(tmp_path, "fake_cached")
    root, man = make_refs(tmp_path, {"REF-A": (f"{name}.py", "VALUE = 1\n", "import-isolated", "")})
    stale = types.ModuleType(name)
    stale.__file__ = str(root / f"{name}.py")
    stale.VALUE = 999  # what a stale/modified in-memory copy would return
    sys.modules[name] = stale
    try:
        with pytest.raises(ReferenceCheckError, match="already loaded"):
            with isolated_references(man, root, ["REF-A"], scratch) as mods:
                pytest.fail(f"harness yielded a cached module (VALUE={mods['REF-A'].VALUE})")
    finally:
        sys.modules.pop(name, None)


def test_b1_cached_unverified_research_dependency_is_refused(tmp_path, scratch):
    top, dep = uniq(tmp_path, "fake_top"), uniq(tmp_path, "fake_cached_dep")
    root, man = make_refs(tmp_path, {"REF-A": (f"{top}.py", f"import {dep}\nVALUE = {dep}.X\n", "import-isolated", "")})
    (root / f"{dep}.py").write_text("X = 5\n")  # exists under the root but is NOT in the manifest
    cached = types.ModuleType(dep)
    cached.__file__ = str(root / f"{dep}.py")
    cached.X = 5
    sys.modules[dep] = cached
    try:
        with pytest.raises(ReferenceCheckError, match="already loaded"):
            with isolated_references(man, root, ["REF-A"], scratch):
                pytest.fail("harness accepted a cached, unverified research dependency")
    finally:
        sys.modules.pop(dep, None)
        sys.modules.pop(top, None)


# --- B2: subprocess must enforce dependency import_policy --------------------------------------

def test_b2_subprocess_refuses_dependency_with_policy_never(tmp_path, scratch):
    lib = uniq(tmp_path, "fake_never_dep")
    root, man = make_refs(tmp_path, {
        "REF-N": (f"scripts/{lib}.py", "print('NEVER-DEP EXECUTED')\n", "never", ""),
        "REF-S": ("scripts/tests/t.py",
                  "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n"
                  f"import {lib}\n", "script-subprocess", "REF-N"),
    })
    proc = run_reference_script(man, root, "REF-S", scratch)
    assert "NEVER-DEP EXECUTED" not in proc.stdout
    assert proc.returncode != 0 and "refused by guard" in proc.stderr


def test_b2_subprocess_still_allows_import_isolated_dependency(tmp_path, scratch):
    lib = uniq(tmp_path, "fake_ok_dep")
    root, man = make_refs(tmp_path, {
        "REF-L": (f"scripts/{lib}.py", "Y = 7\n", "import-isolated", ""),
        "REF-S": ("scripts/tests/t.py",
                  "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n"
                  f"import {lib}\nprint('Y', {lib}.Y)\n", "script-subprocess", "REF-L"),
    })
    proc = run_reference_script(man, root, "REF-S", scratch)
    assert proc.returncode == 0 and "Y 7" in proc.stdout, proc.stderr


# --- B3: resolved path must stay inside the reference root --------------------------------------

def test_b3_symlink_escaping_root_is_rejected_before_reading(tmp_path, monkeypatch):
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"SECRET = 1\n")
    root = tmp_path / "research"
    root.mkdir()
    (root / "link.py").symlink_to(outside)
    data = outside.read_bytes()
    lines = ["\t".join(MANIFEST_COLUMNS),
             "\t".join(["REF-L", "link.py", hashlib.sha256(data).hexdigest(), str(len(data)), "1", "t", "r",
                        "import-isolated", "", "M", ""])]
    (tmp_path / "m.tsv").write_text("\n".join(lines) + "\n")
    man = load_manifest(tmp_path / "m.tsv")

    def must_not_hash(*a, **k):
        raise AssertionError("content of an out-of-root target must not be read")

    monkeypatch.setattr(P, "sha256_file", must_not_hash)
    assert P.verify_entry(man["REF-L"], root).status == "outside_root"


def test_b3_symlink_inside_root_is_allowed(tmp_path):
    root = tmp_path / "research"
    (root / "real").mkdir(parents=True)
    (root / "real" / "a.py").write_bytes(b"A = 1\n")
    (root / "alias.py").symlink_to(root / "real" / "a.py")
    data = b"A = 1\n"
    lines = ["\t".join(MANIFEST_COLUMNS),
             "\t".join(["REF-A", "alias.py", hashlib.sha256(data).hexdigest(), str(len(data)), "1", "t", "r",
                        "import-isolated", "", "M", ""])]
    (tmp_path / "m.tsv").write_text("\n".join(lines) + "\n")
    assert P.verify_entry(load_manifest(tmp_path / "m.tsv")["REF-A"], root).status == "ok"


# --- post-use checks must also run on exception / timeout paths ----------------------------------

def _tamper_keep_size_and_mtime(path: Path, new: bytes):
    st = path.stat()
    assert len(new) == st.st_size
    path.write_bytes(new)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))  # snapshot cannot see it; only the hash can


def test_post_check_runs_when_body_raises_and_keeps_original_error(tmp_path, scratch):
    name = uniq(tmp_path, "fake_tamper")
    root, man = make_refs(tmp_path, {"REF-A": (f"{name}.py", "X = 1\n", "import-isolated", "")})
    with pytest.raises(ReferenceCheckError, match="REF-A=hash_mismatch") as info:
        with isolated_references(man, root, ["REF-A"], scratch):
            _tamper_keep_size_and_mtime(root / f"{name}.py", b"X = 2\n")
            raise RuntimeError("original test failure")
    assert isinstance(info.value.__cause__, RuntimeError)
    assert "original test failure" in str(info.value)


def test_body_exception_without_tampering_is_propagated_unchanged(tmp_path, scratch):
    name = uniq(tmp_path, "fake_plain")
    root, man = make_refs(tmp_path, {"REF-A": (f"{name}.py", "X = 1\n", "import-isolated", "")})
    with pytest.raises(RuntimeError, match="plain failure"):
        with isolated_references(man, root, ["REF-A"], scratch):
            raise RuntimeError("plain failure")


def test_post_check_runs_on_subprocess_timeout(tmp_path, scratch):
    root, man = make_refs(tmp_path, {
        "REF-S": ("scripts/t.py",
                  "import time\nfrom pathlib import Path\n(Path(__file__).parent / 'leak.txt').write_text('x')\n"
                  "print('wrote', flush=True)\ntime.sleep(60)\n", "script-subprocess", ""),
    })
    with pytest.raises(ReferenceCheckError, match="changed") as info:
        run_reference_script(man, root, "REF-S", scratch, timeout=5)
    assert isinstance(info.value.__cause__, subprocess.TimeoutExpired)


def test_subprocess_timeout_without_changes_reraises_timeout(tmp_path, scratch):
    root, man = make_refs(tmp_path, {"REF-S": ("scripts/t.py", "import time\ntime.sleep(60)\n", "script-subprocess", "")})
    with pytest.raises(subprocess.TimeoutExpired):
        run_reference_script(man, root, "REF-S", scratch, timeout=3)


def test_post_check_still_rehashes_when_snapshot_itself_fails(tmp_path, scratch, monkeypatch):
    import refharness

    name = uniq(tmp_path, "fake_snapfail")
    root, man = make_refs(tmp_path, {"REF-A": (f"{name}.py", "X = 1\n", "import-isolated", "")})
    real_snapshot = refharness.snapshot_dirs
    calls = {"n": 0}

    def flaky_snapshot(dirs):
        calls["n"] += 1
        if calls["n"] > 1:  # the "before" snapshot works, the post-use one fails
            raise OSError("simulated snapshot failure")
        return real_snapshot(dirs)

    monkeypatch.setattr(refharness, "snapshot_dirs", flaky_snapshot)
    with pytest.raises(ReferenceCheckError) as info:
        with isolated_references(man, root, ["REF-A"], scratch):
            _tamper_keep_size_and_mtime(root / f"{name}.py", b"X = 2\n")
    msg = str(info.value)
    assert "snapshot error: OSError" in msg, msg
    assert "REF-A=hash_mismatch" in msg, "hash re-verification must still run when the snapshot fails"
