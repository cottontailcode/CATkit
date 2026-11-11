"""Regression tests for the v0.1 audit blockers (synthetic data and throw-away files only; never a real ND2):

P0  output writer: no overwrite through links, no partial results, clear errors
P1  uncalibrated ND2 voxel metadata is rejected before pixels are read
P1  QC overlay file names cannot collide
P2  an individual None optics field is a clean error
"""

from __future__ import annotations

import hashlib
import os

import pytest

from scientific import require_modules

require_modules("numpy", "scipy", "skimage", "bigfish", "nd2", "matplotlib")

import numpy as np  # noqa: E402

import catkit.io as IO  # noqa: E402
import catkit.pipeline as PL  # noqa: E402
import catkit.qc as QC  # noqa: E402
from catkit.cli import EXIT_ERROR, EXIT_OK, main  # noqa: E402
from synth_det import make_channel, make_image  # noqa: E402
from synth_seg import make_dapi_stack  # noqa: E402
from test_io_fake import make_fake  # noqa: E402

NAMES = ("Cy5", "TRITC", "DAPI")


def patch_nd2(monkeypatch, names=NAMES, img=None, **kw):
    img = img or make_image()
    data = np.stack([img.channels[n] for n in names], axis=1)
    text = {"capturing": "".join(f"Sample {i + 1}:\n Exposure: 100 ms\n" for i in range(len(names)))}
    Fake, state = make_fake(names=names, data=data, text=text, **kw)
    monkeypatch.setattr(IO.nd2, "ND2File", Fake)
    return state


@pytest.fixture
def nd2_file(tmp_path):
    f = tmp_path / "in" / "fov_a.nd2"
    f.parent.mkdir()
    f.write_bytes(b"precious input bytes " * 50)
    return f


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def cli(nd2_file, out, seg="DAPI", det=("Cy5", "TRITC")):
    return main(["run", "--input", str(nd2_file), "--segmentation", seg, "--detect", *det, "--output", str(out)])


# ---------------------------------------------------------------- P0: output safety
def test_overlay_symlink_to_input_cannot_overwrite_it(monkeypatch, nd2_file, tmp_path, capsys):
    patch_nd2(monkeypatch)
    before = sha(nd2_file)
    out = tmp_path / "res"
    out.mkdir()
    (out / "qc_overlay_01_Cy5.png").symlink_to(nd2_file)               # the audit's attack, with the old and the new file name
    (out / "qc_overlay_Cy5.png").symlink_to(nd2_file)
    assert cli(nd2_file, out) == EXIT_ERROR
    assert "not empty" in capsys.readouterr().err
    assert sha(nd2_file) == before and sorted(p.name for p in out.iterdir()) == ["qc_overlay_01_Cy5.png", "qc_overlay_Cy5.png"]


def test_output_path_that_is_a_symlink_or_file_is_rejected(monkeypatch, nd2_file, tmp_path, capsys):
    patch_nd2(monkeypatch)
    before = sha(nd2_file)
    link = tmp_path / "link_to_input"
    link.symlink_to(nd2_file)
    assert cli(nd2_file, link) == EXIT_ERROR and "symbolic link" in capsys.readouterr().err
    dirlink = tmp_path / "link_to_dir"
    (tmp_path / "somedir").mkdir()
    dirlink.symlink_to(tmp_path / "somedir")
    assert cli(nd2_file, dirlink) == EXIT_ERROR and "symbolic link" in capsys.readouterr().err
    assert cli(nd2_file, nd2_file) == EXIT_ERROR and "not a directory" in capsys.readouterr().err        # the input itself as destination
    assert sha(nd2_file) == before and list((tmp_path / "somedir").iterdir()) == []


def test_non_empty_destination_is_rejected_and_untouched(monkeypatch, nd2_file, tmp_path, capsys):
    patch_nd2(monkeypatch)
    out = tmp_path / "res"
    out.mkdir()
    (out / "notes.txt").write_text("mine")
    (out / "sub").mkdir()
    assert cli(nd2_file, out) == EXIT_ERROR and "not empty" in capsys.readouterr().err
    assert (out / "notes.txt").read_text() == "mine" and sorted(p.name for p in out.iterdir()) == ["notes.txt", "sub"]


def test_write_failure_publishes_nothing_and_reports_cleanly(monkeypatch, nd2_file, tmp_path, capsys):
    patch_nd2(monkeypatch)
    real = QC.save_overlay

    def failing(path, projection, labels, spots, channel, fov_id=""):
        if channel == "TRITC":
            raise OSError("No space left on device")
        return real(path, projection, labels, spots, channel, fov_id)

    monkeypatch.setattr(QC, "save_overlay", failing)
    out = tmp_path / "res"
    assert cli(nd2_file, out) == EXIT_ERROR
    err = capsys.readouterr().err
    assert "No space left on device" in err and "Traceback" not in err
    assert not out.exists() and [p.name for p in tmp_path.iterdir() if p.name != "in"] == []          # no half set, no temp leftovers
    # an existing EMPTY destination stays empty after a failure, and a retry then succeeds
    out.mkdir()
    assert cli(nd2_file, out) == EXIT_ERROR and list(out.iterdir()) == []
    monkeypatch.setattr(QC, "save_overlay", real)
    assert cli(nd2_file, out) == EXIT_OK
    assert sorted(p.name for p in out.iterdir()) == ["candidates.csv", "cells.csv", "qc_overlay_01_Cy5.png", "qc_overlay_02_TRITC.png",
                                                     "run_metadata.json", "spots.csv"]
    assert [p.name for p in tmp_path.iterdir() if p.name not in ("in", "res")] == []


def test_empty_existing_destination_and_new_destination_succeed(monkeypatch, nd2_file, tmp_path):
    patch_nd2(monkeypatch)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert cli(nd2_file, empty) == EXIT_OK and (empty / "spots.csv").exists()
    assert cli(nd2_file, tmp_path / "a" / "b" / "new") == EXIT_OK and (tmp_path / "a" / "b" / "new" / "cells.csv").exists()


def test_api_write_outputs_has_the_same_protection(tmp_path):
    r = PL.analyze_fov(make_image(), {"segmentation": "DAPI", "detection": ["Cy5", "TRITC"]})
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"x")
    out = tmp_path / "res"
    out.mkdir()
    (out / "spots.csv").symlink_to(victim)
    with pytest.raises(PL.OutputError, match="not empty"):
        PL.write_outputs(r, out)
    assert victim.read_bytes() == b"x"


# ---------------------------------------------------------------- P1: uncalibrated voxel metadata
@pytest.mark.parametrize("flags,axis", [((False, True, True), "x"), ((True, False, True), "y"), ((True, True, False), "z"), ((False, False, False), "x")])
def test_uncalibrated_axis_is_rejected_before_pixels(monkeypatch, nd2_file, tmp_path, capsys, flags, axis):
    state = patch_nd2(monkeypatch, calibrated=flags)                     # the numbers still equal the baseline (150, 65, 65) nm
    assert IO.inspect_nd2(nd2_file)["is_baseline_voxel"]
    with pytest.raises(IO.ND2Error, match=rf"channel 'Cy5': {axis} axis is marked uncalibrated"):
        IO.load_nd2_channels(nd2_file, ["Cy5"])
    assert state.dask_calls == 0
    assert cli(nd2_file, tmp_path / "res") == EXIT_ERROR
    assert "uncalibrated" in capsys.readouterr().err and state.dask_calls == 0 and not (tmp_path / "res").exists()


def test_calibration_of_unrequested_channels_and_missing_flags(monkeypatch, nd2_file):
    patch_nd2(monkeypatch, calibrated={"TRITC": (False, False, False)})           # only TRITC is uncalibrated
    assert IO.load_nd2_channels(nd2_file, ["Cy5", "DAPI"]).voxel_size_nm == pytest.approx((150.0, 65.0, 65.0))
    with pytest.raises(IO.ND2Error, match="channel 'TRITC'"):
        IO.load_nd2_channels(nd2_file, ["Cy5", "TRITC"])
    h = IO.inspect_nd2(nd2_file)                                                  # inspection reports it without rejecting
    assert h["axes_calibrated"]["TRITC"] == dict(x=False, y=False, z=False) and any("uncalibrated" in w and "TRITC" in w for w in h["warnings"])
    patch_nd2(monkeypatch, calibrated=(None, None, None))                         # flag not present: not "explicitly uncalibrated"
    st = IO.load_nd2_channels(nd2_file, ["Cy5"])
    assert any("calibration flag" in w for w in st.header["warnings"])


# ---------------------------------------------------------------- P1: QC overlay file names
def test_qc_overlay_names_cannot_collide(monkeypatch, nd2_file, tmp_path):
    names = ("Cy5 A", "Cy5/A", "DAPI")
    img = make_image()
    img.channels = {"Cy5 A": make_channel([(8.0, 30.0, 30.0, 220.0)], seed=1), "Cy5/A": make_channel([(8.0, 60.0, 70.0, 220.0), (8.0, 80.0, 15.0, 200.0)], seed=2),
                    "DAPI": make_dapi_stack(shape=(17, 100, 100), nuclei=[(30, 30, 9), (62, 72, 9)], seed=1)}
    patch_nd2(monkeypatch, names=names, img=img)
    out = tmp_path / "res"
    assert cli(nd2_file, out, det=("Cy5 A", "Cy5/A")) == EXIT_OK
    import json
    meta = json.loads((out / "run_metadata.json").read_text())
    files = meta["qc_images"]
    assert set(files) == {"Cy5 A", "Cy5/A"} and len(set(files.values())) == 2 and all("/" not in f and (out / f).is_file() for f in files.values())
    assert (out / files["Cy5 A"]).read_bytes() != (out / files["Cy5/A"]).read_bytes()          # two different images were kept
    assert sorted(f for f in meta["outputs"] if f.endswith(".png")) == sorted(files.values())
    assert files == {"Cy5 A": "qc_overlay_01_Cy5_A.png", "Cy5/A": "qc_overlay_02_Cy5_A.png"}   # stable: detection order + sanitized name


# ---------------------------------------------------------------- P2: a single None optics field
@pytest.mark.parametrize("field", ["NA", "emission_nm", "n_immersion"])
def test_none_optics_field_is_a_clean_error(monkeypatch, nd2_file, tmp_path, capsys, field):
    patch_nd2(monkeypatch)
    real = IO.inspect_nd2

    def broken(path):
        h = real(path)
        h["optics"]["Cy5"][field] = None
        return h

    monkeypatch.setattr(IO, "inspect_nd2", broken)
    assert cli(nd2_file, tmp_path / "res") == EXIT_ERROR
    err = capsys.readouterr().err
    assert field in err and "'Cy5'" in err and "Traceback" not in err and not (tmp_path / "res").exists()


# ---------------------------------------------------------------- second audit round
def test_publish_failure_keeps_an_existing_empty_destination(monkeypatch, nd2_file, tmp_path, capsys):
    import errno
    patch_nd2(monkeypatch)
    out = tmp_path / "res"
    out.mkdir()
    real = PL.os.rename

    def failing(src, dst):
        if str(dst) == str(out):
            raise OSError(errno.EIO, "Input/output error")
        return real(src, dst)

    monkeypatch.setattr(PL.os, "rename", failing)
    assert cli(nd2_file, out) == EXIT_ERROR
    assert "Input/output error" in capsys.readouterr().err
    assert out.is_dir() and list(out.iterdir()) == []                          # the pre-existing empty directory survives
    assert [p.name for p in tmp_path.iterdir() if p.name not in ("in", "res")] == []
    monkeypatch.setattr(PL.os, "rename", real)
    assert cli(nd2_file, out) == EXIT_OK and (out / "spots.csv").exists()


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="running as root: permission bits are not enforced")
def test_unreadable_destination_is_a_clean_error(monkeypatch, nd2_file, tmp_path, capsys):
    patch_nd2(monkeypatch)
    out = tmp_path / "locked"
    out.mkdir()
    out.chmod(0o300)                                                           # writable but not listable
    try:
        assert cli(nd2_file, out) == EXIT_ERROR
        err = capsys.readouterr().err
        assert "Traceback" not in err and "locked" in err
        with pytest.raises(PL.OutputError):
            PL.check_output_dir(out)
    finally:
        out.chmod(0o700)


@pytest.mark.parametrize("flags,what", [(("False", True, True), "x axis"), ((True, 0, True), "y axis"), ((True, True, "no"), "z axis"),
                                        (False, "x axis"), ((True, True), "x axis"), (True, "x axis")])               # structural: first axis named
def test_malformed_calibration_flags_are_rejected_before_pixels(monkeypatch, nd2_file, tmp_path, capsys, flags, what):
    state = patch_nd2(monkeypatch, calibrated=flags)
    h = IO.inspect_nd2(nd2_file)                                               # inspection reports, it does not raise or crash
    assert any("malformed" in w for w in h["warnings"])
    with pytest.raises(IO.ND2Error, match=rf"channel 'Cy5'.*{what}.*malformed|channel 'Cy5'.*malformed.*{what}"):
        IO.load_nd2_channels(nd2_file, ["Cy5"])
    assert state.dask_calls == 0
    assert cli(nd2_file, tmp_path / "res") == EXIT_ERROR and "malformed" in capsys.readouterr().err and not (tmp_path / "res").exists()


def test_numpy_bool_flags_are_accepted(monkeypatch, nd2_file):
    patch_nd2(monkeypatch, calibrated=(np.True_, np.True_, np.True_))
    assert IO.load_nd2_channels(nd2_file, ["Cy5"]).header["axes_calibrated"]["Cy5"] == dict(x=True, y=True, z=True)


def test_very_long_channel_name_does_not_break_the_overlay_file_name(monkeypatch, nd2_file, tmp_path):
    long = "C" * 239
    img = make_image()
    img.channels = {long: make_channel([(8.0, 30.0, 30.0, 220.0)], seed=1), "DAPI": img.channels["DAPI"]}
    patch_nd2(monkeypatch, names=(long, "DAPI"), img=img)
    out = tmp_path / "res"
    assert cli(nd2_file, out, det=(long,)) == EXIT_OK
    import json
    meta = json.loads((out / "run_metadata.json").read_text())
    f = meta["qc_images"][long]                                                # the full channel name is kept as the metadata key
    assert (out / f).is_file() and len(f.encode()) < 100 and f.startswith("qc_overlay_01_")


def test_write_outputs_never_touches_the_process_umask(monkeypatch, tmp_path):
    """Reading the umask requires setting it (a process-wide race with other threads), so the writer must not call os.umask at all."""
    import os
    import stat
    r = PL.analyze_fov(make_image(), {"segmentation": "DAPI", "detection": ["Cy5", "TRITC"]})
    calls = []
    real = os.umask
    monkeypatch.setattr(PL.os, "umask", lambda m: calls.append(m) or real(m))      # os is the shared module: record any call
    before = real(0o027)
    try:
        PL.write_outputs(r, tmp_path / "res", qc_overlays=False)
        assert calls == [] and real(0o027) == 0o027                                  # no call, and the mask is unchanged
    finally:
        real(before)
    mode = stat.S_IMODE((tmp_path / "res").stat().st_mode)
    assert mode & 0o700 == 0o700 and not mode & 0o022                              # owner full access, never group/world-writable
