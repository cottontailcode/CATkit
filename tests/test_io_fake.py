"""Reader contract tests with a fake ``nd2.ND2File`` (they prove the reader's logic, not the real nd2 library or real files).

Real-file behaviour is covered by the opt-in tests in ``tests/equivalence/test_e_io.py``.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from scientific import require_modules

require_modules("numpy", "nd2")

import numpy as np  # noqa: E402

import catkit.io as IO  # noqa: E402

TEXT = {"capturing": "Cam\nSample 1:\n  Exposure: 100 ms\nSample 2:\n  Exposure: 50 ms\nSample 3:\n  Exposure: 25.5 ms\n"}


class ZStackLoop:                         # the reader recognises the experiment loop by its class name
    def __init__(self, step):
        self.parameters = NS(stepUm=step)


def make_fake(sizes=None, dtype="uint16", names=("Cy5", "TRITC", "DAPI"), vox=(0.15, 0.065, 0.065), text=TEXT, step=0.15, shape=(5, 12, 14),
              data=None, calibrated=(True, True, True), optics=None):
    if data is not None:
        shape = (data.shape[0], data.shape[2], data.shape[3])
    sizes = sizes or {"Z": shape[0], "C": len(names), "Y": shape[1], "X": shape[2]}
    state = NS(opened=0, closed=0, dask_calls=0)
    if data is None:
        data = np.stack([np.full(shape, 100 * (i + 1), np.uint16) for i in range(len(names))], axis=1)   # (Z, C, Y, X): channel i == 100*(i+1)

    class Fake:
        def __init__(self, path):
            self.path = path
            self.sizes = sizes
            self.dtype = np.dtype(dtype)
            def cal(n):                       # (x, y, z) flags like nd2: one tuple for all channels, or {channel: tuple}
                c = calibrated.get(n, (True, True, True)) if isinstance(calibrated, dict) else calibrated
                return NS(axesCalibrated=c)

            kw = dict(optics or {})
            self.metadata = NS(channels=[NS(channel=NS(name=n, emissionLambdaNm=kw.get(("emission", n), 500.0 + i)), volume=cal(n), microscope=NS(
                objectiveNumericalAperture=kw.get(("NA", n), 1.45), immersionRefractiveIndex=1.515, objectiveName="obj")) for i, n in enumerate(names)])
            self.experiment = [ZStackLoop(step)] if step is not None else []
            self.text_info = text

        def voxel_size(self):
            return NS(x=vox[2], y=vox[1], z=vox[0])

        def to_dask(self):
            state.dask_calls += 1
            return data

        def __enter__(self):
            state.opened += 1
            return self

        def __exit__(self, *a):
            state.closed += 1
            return False

    return Fake, state


def patch(monkeypatch, **kw):
    Fake, state = make_fake(**kw)
    monkeypatch.setattr(IO.nd2, "ND2File", Fake)
    return state


def test_inspect_header_fields_and_units(monkeypatch):
    state = patch(monkeypatch, vox=(0.15, 0.065, 0.065))
    h = IO.inspect_nd2("x.nd2")
    assert h["sizes"] == {"Z": 5, "C": 3, "Y": 12, "X": 14} and h["dtype"] == "uint16"
    assert h["channel_names"] == ["Cy5", "TRITC", "DAPI"]
    assert h["voxel_size_nm"] == pytest.approx((150.0, 65.0, 65.0)) and h["is_baseline_voxel"]
    assert h["exposures_ms"] == [100.0, 50.0, 25.5] and h["z_step_um"] == 0.15 and h["warnings"] == []
    assert h["optics"]["TRITC"] == dict(NA=1.45, n_immersion=1.515, emission_nm=501.0, objective="obj")
    assert state.dask_calls == 0 and state.opened == state.closed == 1          # header-only never touches pixels


def test_voxel_is_reordered_to_zyx(monkeypatch):
    patch(monkeypatch, vox=(0.15, 0.05, 0.05))                                   # fake returns (z, y, x) = given order
    assert IO.inspect_nd2("x")["voxel_size_nm"] == pytest.approx((150.0, 50.0, 50.0))


def test_channels_selected_by_name_not_position(monkeypatch):
    patch(monkeypatch, vox=(0.15, 0.065, 0.065))
    st = IO.load_nd2_channels("x", ["DAPI", "Cy5"])
    assert list(st.channels) == ["DAPI", "Cy5"]
    assert (st.channels["DAPI"] == 300).all() and (st.channels["Cy5"] == 100).all()
    assert st.channels["DAPI"].shape == (5, 12, 14) and st.channels["DAPI"].dtype == np.uint16
    assert st.voxel_size_nm == pytest.approx((150.0, 65.0, 65.0)) and st.header["channel_names"] == ["Cy5", "TRITC", "DAPI"]


def test_loaded_arrays_are_copies(monkeypatch):
    patch(monkeypatch, vox=(0.15, 0.065, 0.065))
    a = IO.load_nd2_channels("x", ["Cy5"]).channels["Cy5"]
    a[:] = 7
    assert (IO.load_nd2_channels("x", ["Cy5"]).channels["Cy5"] == 100).all()


@pytest.mark.parametrize("kw,match", [
    (dict(sizes={"Z": 5, "C": 3, "Y": 12, "X": 14, "P": 2}), "unsupported ND2 axes"),            # extra position axis
    (dict(sizes={"T": 2, "Z": 5, "C": 3, "Y": 12, "X": 14}), "unsupported ND2 axes"),
    (dict(sizes={"C": 3, "Y": 12, "X": 14}), "unsupported ND2 axes"),                            # no Z axis
    (dict(sizes={"C": 3, "Z": 5, "Y": 12, "X": 14}), "unsupported ND2 axes"),                    # wrong order
    (dict(dtype="uint8"), "uint16"), (dict(dtype="float32"), "uint16"),
    (dict(names=("A", "A", "B")), "unique"), (dict(names=("A", "", "B")), "unique"),
    (dict(sizes={"Z": 5, "C": 4, "Y": 12, "X": 14}), "3 channels but the C axis has 4"),
    (dict(vox=(0.15, 0.065, 0.0)), "missing or not finite"), (dict(vox=(0.15, float("nan"), 0.065)), "missing or not finite"),
    (dict(vox=(-0.15, 0.065, 0.065)), "missing or not finite"), (dict(vox=(None, 0.065, 0.065)), "missing or not finite"),
    (dict(vox=(0.15, 0.065, 0.066)), "x and y pixel sizes differ"),
])
def test_structure_violations_are_errors_and_close_the_file(monkeypatch, kw, match):
    state = patch(monkeypatch, **kw)
    with pytest.raises(IO.ND2Error, match=match):
        IO.inspect_nd2("x")
    assert state.dask_calls == 0 and state.opened == state.closed == 1            # closed on error; no pixels read
    with pytest.raises(IO.ND2Error):
        IO.load_nd2_channels("x", ["A"])
    assert state.dask_calls == 0


def test_missing_channel_lists_available_and_reads_no_pixels(monkeypatch):
    state = patch(monkeypatch, vox=(0.15, 0.065, 0.065))
    with pytest.raises(IO.ChannelNotFound, match=r"\['Hoechst'\].*available: \['Cy5', 'TRITC', 'DAPI'\]"):
        IO.load_nd2_channels("x", ["Cy5", "Hoechst"])
    assert state.dask_calls == 0 and state.opened == state.closed
    for bad in ([], ["Cy5", "Cy5"]):
        with pytest.raises(IO.ND2Error, match="unique"):
            IO.load_nd2_channels("x", bad)


def test_baseline_voxel_checked_before_pixels_with_explicit_override(monkeypatch):
    state = patch(monkeypatch, vox=(0.15, 0.043333333333333335, 0.043333333333333335))
    h = IO.inspect_nd2("x")
    assert not h["is_baseline_voxel"] and h["voxel_size_nm"][1] == pytest.approx(43.3333333333)
    with pytest.raises(IO.ND2Error, match="not supported in v0.1"):
        IO.load_nd2_channels("x", ["Cy5"])
    assert state.dask_calls == 0
    st = IO.load_nd2_channels("x", ["Cy5"], require_baseline_voxel=False)
    assert state.dask_calls == 1 and st.voxel_size_nm[1] == pytest.approx(43.3333333333)


def test_exposure_and_zstep_mismatch_are_warnings_not_guesses(monkeypatch):
    patch(monkeypatch, vox=(0.15, 0.065, 0.065), text={"capturing": "Sample 1:\n Exposure: 10 ms\n"}, step=0.3)
    h = IO.inspect_nd2("x")
    assert h["exposures_ms"] == [None, None, None]
    assert any("exposure entries" in w for w in h["warnings"]) and any("ZStackLoop step" in w for w in h["warnings"])
    patch(monkeypatch, vox=(0.15, 0.065, 0.065), text=None, step=None)
    h = IO.inspect_nd2("x")
    assert h["z_step_um"] is None and h["exposures_ms"] == [None, None, None]


def test_parse_capturing_exposures_cases():
    p = IO.parse_capturing_exposures
    assert p(TEXT) == [100.0, 50.0, 25.5] and p(None) == [] and p({}) == [] and p({"capturing": None}) == []
    assert p({"capturing": "Sample 1:\n no exposure here\nSample 2:\n Exposure: 7 ms"}) == [None, 7.0]
