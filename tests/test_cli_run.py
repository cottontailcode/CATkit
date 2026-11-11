"""``catkit run``: argument handling, error reporting and a synthetic end-to-end run through the real CLI code path.

ND2 files cannot be synthesised, so the end-to-end test patches ``nd2.ND2File`` with the fake reader of ``test_io_fake.py``
(its data are the synthetic FOV of the pipeline tests). Real-file runs are opt-in (``tests/test_real_pipeline.py``).
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from scientific import require_modules

require_modules("numpy", "scipy", "skimage", "bigfish", "nd2", "matplotlib")

import numpy as np  # noqa: E402

import catkit.io as IO  # noqa: E402
import catkit.pipeline as PL  # noqa: E402
from catkit.cli import EXIT_ERROR, EXIT_OK, EXIT_USAGE, main  # noqa: E402
from synth_det import make_image  # noqa: E402
from test_io_fake import make_fake  # noqa: E402

NAMES = ("Cy5", "TRITC", "DAPI")


def patch_nd2(monkeypatch, vox=(0.15, 0.065, 0.065), names=NAMES, img=None):
    img = img or make_image()
    data = np.stack([img.channels[n] for n in names], axis=1)                      # (Z, C, Y, X)
    Fake, state = make_fake(names=names, vox=vox, data=data, text={"capturing": "Sample 1:\n Exposure: 400 ms\nSample 2:\n Exposure: 300 ms\nSample 3:\n Exposure: 100 ms\n"})
    monkeypatch.setattr(IO.nd2, "ND2File", Fake)
    return state


@pytest.fixture
def nd2_file(tmp_path):
    f = tmp_path / "fov_a.nd2"
    f.write_bytes(b"not a real nd2; the reader is patched")
    return f


def run(nd2_file, out, *extra, seg="DAPI", det=("Cy5", "TRITC")):
    return main(["run", "--input", str(nd2_file), "--segmentation", seg, "--detect", *det, "--output", str(out), *extra])


# ---------------------------------------------------------------- argument handling
def test_run_help_and_required_arguments(capsys):
    with pytest.raises(SystemExit) as e:
        main(["run", "--help"])
    out = capsys.readouterr().out
    assert e.value.code == 0 and "--segmentation" in out and "--detect" in out and "default 10" in out and "--k" in out
    for argv in (["run"], ["run", "--input", "x.nd2"], ["run", "--input", "x.nd2", "--segmentation", "DAPI", "--output", "o"],
                 ["run", "--input", "x.nd2", "--segmentation", "DAPI", "--detect", "--output", "o"],
                 ["run", "--input", "x.nd2", "--segmentation", "DAPI", "--detect", "Cy5", "--output", "o", "--k", "ten"]):
        with pytest.raises(SystemExit) as e:
            main(argv)
        assert e.value.code == EXIT_USAGE


def test_subprocess_exit_codes(tmp_path):
    p = subprocess.run([sys.executable, "-m", "catkit", "run"], capture_output=True, text=True)
    assert p.returncode == EXIT_USAGE and "required" in p.stderr
    p = subprocess.run([sys.executable, "-m", "catkit", "run", "--input", str(tmp_path / "none.nd2"), "--segmentation", "DAPI", "--detect", "Cy5",
                        "--output", str(tmp_path / "o")], capture_output=True, text=True)
    assert p.returncode == EXIT_ERROR and "input file not found" in p.stderr and "Traceback" not in p.stderr


# ---------------------------------------------------------------- errors are reported, not raised; nothing is written on rejection
def test_error_messages(monkeypatch, nd2_file, tmp_path, capsys):
    patch_nd2(monkeypatch)
    cases = [
        (dict(seg="Hoechst"), "not in the file; available: ['Cy5', 'TRITC', 'DAPI']"),
        (dict(det=("Cy5", "Nope")), "not in the file; available"),
        (dict(det=("Cy5", "Cy5")), "roles must name"),
        (dict(seg="Cy5", det=("Cy5",)), "roles must name"),
    ]
    for kw, msg in cases:
        assert run(nd2_file, tmp_path / "bad", **kw) == EXIT_ERROR
        assert msg in capsys.readouterr().err
    assert run(nd2_file, tmp_path / "bad", "--k", "5") == EXIT_ERROR
    assert "candidate floor" in capsys.readouterr().err
    assert not (tmp_path / "bad").exists()


def test_incompatible_voxel_is_reported(monkeypatch, nd2_file, tmp_path, capsys):
    patch_nd2(monkeypatch, vox=(0.15, 0.043333333333333335, 0.043333333333333335))
    assert run(nd2_file, tmp_path / "o") == EXIT_ERROR
    err = capsys.readouterr().err
    assert "not supported in v0.1" in err and "(150.0, 65.0, 65.0)" in err and not (tmp_path / "o").exists()


def test_missing_optics_is_reported(monkeypatch, nd2_file, tmp_path, capsys):
    patch_nd2(monkeypatch)
    real = IO.inspect_nd2

    def no_tritc_optics(path):
        h = real(path)
        h["optics"].pop("TRITC")
        return h

    monkeypatch.setattr(IO, "inspect_nd2", no_tritc_optics)
    assert run(nd2_file, tmp_path / "o") == EXIT_ERROR
    err = capsys.readouterr().err
    assert "optics" in err and "'TRITC'" in err and "not guessed" in err and not (tmp_path / "o").exists()


def test_existing_results_are_not_overwritten(monkeypatch, nd2_file, tmp_path, capsys):
    patch_nd2(monkeypatch)
    out = tmp_path / "o"
    out.mkdir()
    (out / "spots.csv").write_text("keep me")
    assert run(nd2_file, out) == EXIT_ERROR and "not empty" in capsys.readouterr().err
    assert (out / "spots.csv").read_text() == "keep me"


# ---------------------------------------------------------------- end to end through the CLI; same code path as the Python API
def test_end_to_end_cli_equals_python_api(monkeypatch, nd2_file, tmp_path, capsys):
    patch_nd2(monkeypatch)
    out = tmp_path / "cli_out"
    assert run(nd2_file, out) == EXIT_OK
    stdout = capsys.readouterr().out
    assert "2 cells" in stdout and "Cy5: candidates 3, accepted 3" in stdout and "outputs written to" in stdout
    assert sorted(p.name for p in out.iterdir()) == ["candidates.csv", "cells.csv", "qc_overlay_01_Cy5.png", "qc_overlay_02_TRITC.png",
                                                     "run_metadata.json", "spots.csv"]
    # the Python API on the same in-memory stack gives byte-identical tables
    img = make_image()
    hdr = IO.inspect_nd2(nd2_file)                                                 # the same (patched) header the CLI run sees
    img.header, img.voxel_size_nm = hdr, hdr["voxel_size_nm"]
    ref = PL.analyze_fov(img, {"segmentation": "DAPI", "detection": ["Cy5", "TRITC"]}, fov_id="fov_a")
    PL.write_outputs(ref, tmp_path / "api_out")
    for f in ("spots.csv", "candidates.csv", "cells.csv"):
        a, b = (out / f).read_text(), (tmp_path / "api_out" / f).read_text()
        assert a == b, f
    import json
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["input"]["file_name"] == "fov_a.nd2" and str(tmp_path) not in json.dumps(meta) and meta["channel_roles"]["detection"] == ["Cy5", "TRITC"]
    assert set(meta["outputs"]) == {"spots.csv", "candidates.csv", "cells.csv", "run_metadata.json", "qc_overlay_01_Cy5.png", "qc_overlay_02_TRITC.png"}
