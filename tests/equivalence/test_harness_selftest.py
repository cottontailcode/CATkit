"""Self-tests of the reference harness using FAKE references (always run; no private data)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from catkit import provenance as P
from catkit.provenance import MANIFEST_COLUMNS, load_manifest
from refharness import (
    ReferenceCheckError,
    isolated_references,
    run_reference_script,
)


def make_refs(tmp_path: Path, files: dict[str, tuple[str, str, str, str]]):
    """files: ref_id -> (relpath, source, import_policy, depends_on). Returns (root, manifest)."""
    root = tmp_path / "research"
    lines = ["\t".join(MANIFEST_COLUMNS)]
    for rid, (rel, src, policy, deps) in files.items():
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        data = src.encode()
        f.write_bytes(data)
        lines.append("\t".join([rid, rel, hashlib.sha256(data).hexdigest(), str(len(data)), str(data.count(b"\n")),
                                "t", "r", policy, deps, "M", ""]))
    (tmp_path / "manifest.tsv").write_text("\n".join(lines) + "\n")
    return root, load_manifest(tmp_path / "manifest.tsv")


@pytest.fixture
def scratch(tmp_path):
    d = tmp_path / "scratch"
    d.mkdir()
    return d


def _uniq(tmp_path, stem):
    return f"{stem}_{abs(hash(str(tmp_path))) % 10**8}"


def test_import_ok_and_state_restored(tmp_path, scratch):
    a, b = _uniq(tmp_path, "fake_a"), _uniq(tmp_path, "fake_b")
    root, man = make_refs(tmp_path, {
        "REF-B": (f"lib/{b}.py", "VALUE = 2\n", "import-isolated", ""),
        "REF-A": (f"lib/{a}.py", f"import {b}\nVALUE = {b}.VALUE + 1\n", "import-isolated", "REF-B"),
    })
    path0, meta0, dwb0, pre0 = list(sys.path), list(sys.meta_path), sys.dont_write_bytecode, sys.pycache_prefix
    with isolated_references(man, root, ["REF-A"], scratch) as mods:
        assert mods["REF-A"].VALUE == 3
        assert sys.dont_write_bytecode is True and sys.pycache_prefix == str(scratch / "pycache")
    assert sys.path == path0 and sys.meta_path == meta0
    assert sys.dont_write_bytecode == dwb0 and sys.pycache_prefix == pre0
    assert a not in sys.modules and b not in sys.modules
    assert not list(root.rglob("*.pyc")) and not list(root.rglob("__pycache__"))


def test_hash_mismatch_refused_before_import(tmp_path, scratch):
    a = _uniq(tmp_path, "fake_sentinel")
    sentinel = tmp_path / "imported.flag"
    root, man = make_refs(tmp_path, {
        "REF-A": (f"{a}.py", f"open({str(sentinel)!r}, 'w').write('x')\n", "import-isolated", ""),
    })
    (root / f"{a}.py").write_text(f"open({str(sentinel)!r}, 'w').write('y')\n")  # same length, different bytes
    with pytest.raises(ReferenceCheckError, match="hash_mismatch"):
        with isolated_references(man, root, ["REF-A"], scratch):
            pass
    assert not sentinel.exists(), "module must not be imported when its hash does not match"


def test_dependency_hash_mismatch_refused(tmp_path, scratch):
    a, b = _uniq(tmp_path, "fake_top"), _uniq(tmp_path, "fake_dep")
    root, man = make_refs(tmp_path, {
        "REF-B": (f"{b}.py", "X = 1\n", "import-isolated", ""),
        "REF-A": (f"{a}.py", f"import {b}\n", "import-isolated", "REF-B"),
    })
    (root / f"{b}.py").write_text("X = 2\n")
    with pytest.raises(ReferenceCheckError, match="REF-B=hash_mismatch"):
        with isolated_references(man, root, ["REF-A"], scratch):
            pass


def test_dataless_refused(tmp_path, scratch, monkeypatch):
    a = _uniq(tmp_path, "fake_dl")
    root, man = make_refs(tmp_path, {"REF-A": (f"{a}.py", "X = 1\n", "import-isolated", "")})
    monkeypatch.setattr(P, "is_dataless", lambda p: True)
    with pytest.raises(ReferenceCheckError, match="REF-A=dataless"):
        with isolated_references(man, root, ["REF-A"], scratch):
            pass


def test_unregistered_transitive_import_refused(tmp_path, scratch):
    a, c = _uniq(tmp_path, "fake_reg"), _uniq(tmp_path, "fake_unreg")
    root, man = make_refs(tmp_path, {"REF-A": (f"lib/{a}.py", f"import {c}\n", "import-isolated", "")})
    (root / "lib" / f"{c}.py").write_text("Y = 1\n")  # same directory, not in the manifest
    with pytest.raises(ImportError, match="refused by guard"):
        with isolated_references(man, root, ["REF-A"], scratch):
            pass
    assert a not in sys.modules and c not in sys.modules


def test_import_exception_restores_state_and_keeps_third_party(tmp_path, scratch):
    a, tp = _uniq(tmp_path, "fake_boom"), _uniq(tmp_path, "fake_thirdparty")
    site = tmp_path / "site"
    site.mkdir()
    (site / f"{tp}.py").write_text("Z = 1\n")
    root, man = make_refs(tmp_path, {"REF-A": (f"{a}.py", f"import {tp}\nraise RuntimeError('boom')\n", "import-isolated", "")})
    sys.path.insert(0, str(site))
    try:
        path0, dwb0, pre0 = list(sys.path), sys.dont_write_bytecode, sys.pycache_prefix
        with pytest.raises(RuntimeError, match="boom"):
            with isolated_references(man, root, ["REF-A"], scratch):
                pass
        assert sys.path == path0 and sys.dont_write_bytecode == dwb0 and sys.pycache_prefix == pre0
        assert a not in sys.modules
        assert tp in sys.modules, "third-party modules outside the reference root must not be purged"
    finally:
        sys.path.remove(str(site))
        sys.modules.pop(tp, None)


def test_name_collision_refused(tmp_path, scratch):
    root, man = make_refs(tmp_path, {"REF-A": ("json.py", "X = 1\n", "import-isolated", "")})
    with pytest.raises(ReferenceCheckError, match="already loaded"):
        with isolated_references(man, root, ["REF-A"], scratch):
            pass


def test_policy_never_cannot_be_imported(tmp_path, scratch):
    a = _uniq(tmp_path, "fake_never")
    root, man = make_refs(tmp_path, {"REF-A": (f"{a}.py", "X = 1\n", "never", "")})
    with pytest.raises(ReferenceCheckError, match="import_policy=never"):
        with isolated_references(man, root, ["REF-A"], scratch):
            pass


def test_change_during_use_detected(tmp_path, scratch):
    a = _uniq(tmp_path, "fake_writer")
    root, man = make_refs(tmp_path, {"REF-A": (f"{a}.py", "X = 1\n", "import-isolated", "")})
    with pytest.raises(ReferenceCheckError, match="integrity check failed after use: changed"):
        with isolated_references(man, root, ["REF-A"], scratch):
            (root / "new_output.csv").write_text("x")


def test_script_subprocess_flags_guard_and_no_pyc(tmp_path, scratch):
    lib, b = _uniq(tmp_path, "fake_lib"), _uniq(tmp_path, "fake_unreg2")
    root, man = make_refs(tmp_path, {
        "REF-L": (f"scripts/{lib}.py", "def f():\n    return 41\n", "import-isolated", ""),
        "REF-S": ("scripts/tests/t.py",
                  "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n"
                  f"import {lib}\nprint('RESULT', {lib}.f() + 1)\n", "script-subprocess", "REF-L"),
    })
    proc = run_reference_script(man, root, "REF-S", scratch)
    assert proc.returncode == 0, proc.stderr
    assert "RESULT 42" in proc.stdout
    assert "dont_write_bytecode=1" in proc.stdout and f"pycache_prefix={scratch / 'pycache'}" in proc.stdout
    assert "isolated=1" in proc.stdout
    assert not list(root.rglob("*.pyc")) and not list(root.rglob("__pycache__"))
    # a script importing an unregistered sibling module is refused inside the subprocess too
    root3, man3 = make_refs(tmp_path / "second", {
        "REF-S": ("scripts/tests/t.py",
                  "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n"
                  f"import {b}\n", "script-subprocess", ""),
    })
    (root3 / "scripts" / f"{b}.py").write_text("Q = 1\n")
    proc3 = run_reference_script(man3, root3, "REF-S", scratch)
    assert proc3.returncode != 0 and "refused by guard" in proc3.stderr
