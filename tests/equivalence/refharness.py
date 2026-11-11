"""Isolated access to hash-verified frozen reference code (test infrastructure).

Nothing here is scientific. It enforces, for every *active* reference:

1. availability: the file exists and is not a cloud placeholder;
2. integrity: size + SHA-256 match the private manifest (whole dependency
   closure, before anything is imported);
3. controlled use: in-process import with an import guard, or a subprocess
   run with the same guard.

Import isolation:

* ``sys.dont_write_bytecode = True`` and ``sys.pycache_prefix`` -> a temp dir,
  so no ``.pyc`` is written next to (or read from) reference sources;
* a meta-path guard refuses any module located under the reference root that
  is not part of the verified closure (transitive imports included);
* ``sys.path``, ``sys.meta_path``, the bytecode settings, and the reference
  modules in ``sys.modules`` are restored in ``finally`` -- third-party
  modules first imported during loading (numpy, ...) are deliberately kept;
* a before/after snapshot of the directories actually used detects changes.
  A snapshot only *detects*; it is not write protection.

Subprocesses run as ``python -I -B -X pycache_prefix=<tmp>``. ``-I`` ignores
all ``PYTHON*`` environment variables, so cache control must use ``-B``/``-X``.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

from catkit.provenance import (
    ManifestEntry,
    dependency_closure,
    diff_snapshots,
    snapshot_dirs,
    verify_manifest,
)

ENV_ROOT = "CATKIT_RESEARCH_SRC"
ENV_MANIFEST = "CATKIT_RESEARCH_MANIFEST"


class ReferenceCheckError(RuntimeError):
    """A reference failed availability/integrity checks or isolation rules."""


def _resolve(p: str | os.PathLike[str]) -> Path:
    return Path(p).resolve()


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class ResearchImportGuard(importlib.abc.MetaPathFinder):
    """Refuse imports that resolve under ``root`` unless the file is allowed."""

    def __init__(self, root: Path, allowed_files: set[Path]):
        self.root = _resolve(root)
        self.allowed = {_resolve(f) for f in allowed_files}
        self.refused: list[str] = []

    def find_spec(self, fullname, path=None, target=None):  # noqa: D401 - importlib API
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None:
            return None
        locations = [spec.origin] if spec.origin else list(spec.submodule_search_locations or [])
        for loc in locations:
            if loc and loc not in ("built-in", "frozen"):
                p = _resolve(loc)
                if _under(p, self.root) and p not in self.allowed:
                    self.refused.append(fullname)
                    raise ImportError(f"unverified reference module refused by guard: {fullname}")
        return None  # defer to the normal import machinery


def verify_closure(manifest: dict[str, ManifestEntry], root, ref_ids: list[str]) -> list[str]:
    """Verify every entry in the dependency closure of ``ref_ids``; return the closure."""
    closure: list[str] = []
    for rid in ref_ids:
        for dep in dependency_closure(manifest, rid):
            if dep not in closure:
                closure.append(dep)
    bad = [r for r in verify_manifest(manifest, root, closure) if not r.ok]
    if bad:
        raise ReferenceCheckError("reference verification failed: " + "; ".join(f"{r.ref_id}={r.status}" for r in bad))
    return closure


def _refuse_cached_modules(root: Path, names: dict[str, str]) -> None:
    """Fail fast if any module from the reference root, or any reference module name, is already loaded.

    A cached module would be returned by ``import`` without re-reading the
    verified file (and the import guard is never consulted for it), so its
    presence breaks the guarantee that verified bytes are what runs.
    """
    cached = []
    for mod_name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if mod_name in names.values() or (f and _under(_resolve(f), root)):
            cached.append(mod_name)
    if cached:
        raise ReferenceCheckError(
            "reference-root or reference-named modules already loaded in sys.modules "
            f"(cannot guarantee verified code runs): {sorted(cached)}"
        )


def _post_check_problems(manifest, root: Path, closure: list[str], dirs, before) -> list[str]:
    """Snapshot diff + full re-verification; never raises (returns problems).

    The two checks are independent: a failure of one is reported and the other
    still runs.
    """
    problems: list[str] = []
    try:
        problems += [f"changed: {c}" for c in diff_snapshots(before, snapshot_dirs(dirs))][:10]
    except Exception as exc:  # noqa: BLE001 - report, do not mask
        problems.append(f"snapshot error: {type(exc).__name__}: {exc}")
    try:
        problems += [f"{r.ref_id}={r.status}" for r in verify_manifest(manifest, root, closure) if not r.ok]
    except Exception as exc:  # noqa: BLE001 - report, do not mask
        problems.append(f"re-verification error: {type(exc).__name__}: {exc}")
    return problems


def _raise_post_check(manifest, root, closure, dirs, before, primary, phase: str) -> None:
    problems = _post_check_problems(manifest, root, closure, dirs, before)
    if problems:
        msg = f"reference integrity check failed after {phase}: " + "; ".join(problems)
        if primary is not None:
            msg += f" | original error: {type(primary).__name__}: {primary}"
        raise ReferenceCheckError(msg) from primary


@contextmanager
def isolated_references(manifest: dict[str, ManifestEntry], root, ref_ids: list[str], scratch: Path):
    """Verify, then import ``ref_ids`` (policy ``import-isolated``) and yield {ref_id: module}."""
    root = _resolve(root)
    closure = verify_closure(manifest, root, ref_ids)
    for rid in closure:
        if manifest[rid].import_policy != "import-isolated":
            raise ReferenceCheckError(f"{rid} has import_policy={manifest[rid].import_policy}; in-process import not allowed")
    files = {rid: _resolve(manifest[rid].path(root)) for rid in closure}
    dirs = sorted({f.parent for f in files.values()})
    names = {rid: files[rid].stem for rid in closure}
    _refuse_cached_modules(root, names)

    before = snapshot_dirs(dirs)
    saved_path = list(sys.path)
    saved_meta = list(sys.meta_path)
    saved_modules = set(sys.modules)
    saved_dwb = sys.dont_write_bytecode
    saved_prefix = sys.pycache_prefix
    guard = ResearchImportGuard(root, set(files.values()))
    pycache = Path(scratch) / "pycache"
    pycache.mkdir(parents=True, exist_ok=True)
    primary: BaseException | None = None
    try:
        sys.dont_write_bytecode = True
        sys.pycache_prefix = str(pycache)
        sys.meta_path.insert(0, guard)
        for d in reversed(dirs):
            sys.path.insert(0, str(d))
        importlib.invalidate_caches()
        modules = {}
        for rid in ref_ids:
            mod = importlib.import_module(names[rid])
            if _resolve(mod.__file__) != files[rid]:
                raise ReferenceCheckError(f"{rid} resolved to an unexpected file")
            modules[rid] = mod
        stray = [
            n for n in set(sys.modules) - saved_modules
            if getattr(sys.modules[n], "__file__", None)
            and _under(_resolve(sys.modules[n].__file__), root)
            and _resolve(sys.modules[n].__file__) not in guard.allowed
        ]
        if stray:
            raise ReferenceCheckError(f"unverified reference modules were loaded: {sorted(stray)}")
        yield modules
    except BaseException as exc:  # recorded so post-use checks can report alongside it
        primary = exc
        raise
    finally:
        sys.meta_path[:] = saved_meta
        sys.path[:] = saved_path
        sys.dont_write_bytecode = saved_dwb
        sys.pycache_prefix = saved_prefix
        for n in set(sys.modules) - saved_modules:
            f = getattr(sys.modules.get(n), "__file__", None)
            if n in names.values() or (f and _under(_resolve(f), root)):
                del sys.modules[n]
        _raise_post_check(manifest, root, closure, dirs, before, primary, "use")


# Executed inside the subprocess (``python -I -B -X pycache_prefix=... -c BOOTSTRAP``).
BOOTSTRAP = r"""
import importlib.abc, importlib.machinery, json, os, runpy, sys
cfg = json.loads(sys.argv[1])
root = os.path.realpath(cfg["root"]); allowed = {os.path.realpath(p) for p in cfg["allowed"]}
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None:
            return None
        for loc in ([spec.origin] if spec.origin else list(spec.submodule_search_locations or [])):
            if loc and loc not in ("built-in", "frozen"):
                p = os.path.realpath(loc)
                if (p == root or p.startswith(root + os.sep)) and p not in allowed:
                    raise ImportError("unverified reference module refused by guard: " + fullname)
        return None
sys.meta_path.insert(0, Guard())
print("BOOTSTRAP dont_write_bytecode=%s pycache_prefix=%s isolated=%s" % (
    sys.flags.dont_write_bytecode, sys.pycache_prefix, sys.flags.isolated), flush=True)
sys.argv = [cfg["script"]]
runpy.run_path(cfg["script"], run_name="__main__")
"""


def run_reference_script(
    manifest: dict[str, ManifestEntry],
    root,
    ref_id: str,
    scratch: Path,
    python: str = sys.executable,
    timeout: float = 600,
) -> subprocess.CompletedProcess:
    """Verify the closure, then run a ``script-subprocess`` reference in a guarded subprocess.

    Verification covers the whole dependency closure, but *import permission*
    is narrower: only the entry script itself runs, and only dependencies with
    ``import_policy == "import-isolated"`` may be imported. ``read``/``never``
    dependencies are hash-checked yet refused by the guard if imported.
    (The guard covers the import system; it does not intercept a script that
    reads and ``exec``s a file by itself.)
    """
    root = _resolve(root)
    if manifest[ref_id].import_policy != "script-subprocess":
        raise ReferenceCheckError(f"{ref_id} is not a script-subprocess reference")
    closure = verify_closure(manifest, root, [ref_id])
    verified = {r: _resolve(manifest[r].path(root)) for r in closure}
    entry = verified[ref_id]
    importable = [str(verified[r]) for r in closure if r != ref_id and manifest[r].import_policy == "import-isolated"]
    dirs = sorted({f.parent for f in verified.values()})
    scratch = Path(scratch)
    for sub in ("pycache", "mpl", "cwd"):
        (scratch / sub).mkdir(parents=True, exist_ok=True)
    cfg = json.dumps(dict(root=str(root), allowed=importable + [str(entry)], script=str(entry)))
    env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", ""), "MPLCONFIGDIR": str(scratch / "mpl")}
    cmd = [python, "-I", "-B", "-X", f"pycache_prefix={scratch / 'pycache'}", "-c", BOOTSTRAP, cfg]
    before = snapshot_dirs(dirs)
    primary: BaseException | None = None
    proc = None
    try:
        proc = subprocess.run(cmd, cwd=scratch / "cwd", env=env, capture_output=True, text=True, timeout=timeout)
    except BaseException as exc:  # e.g. TimeoutExpired; checks below still run
        primary = exc
    _raise_post_check(manifest, root, closure, dirs, before, primary, "script run")
    if primary is not None:
        raise primary
    return proc
