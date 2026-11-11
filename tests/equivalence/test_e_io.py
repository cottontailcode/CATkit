"""ND2 reader: source identity with frozen ``REF-IO`` (research env) and the opt-in real-file smoke tests.

Real-file tests read ONE user-authorised file, read-only, locally; nothing is copied or uploaded. They are opt-in and
SKIP with a fixed reason unless both ``CATKIT_REAL_ND2`` (the file) and ``CATKIT_REAL_ND2_CHANNELS`` (comma-separated
metadata channel names to load; CATkit never infers roles or names) are set. The header comparison against the frozen
reader additionally needs the research environment.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest

from scientific import require_modules

require_modules("numpy", "nd2")

import numpy as np  # noqa: E402

import catkit.io as IO  # noqa: E402

REAL_SKIP = "CATKIT_REAL_ND2 / CATKIT_REAL_ND2_CHANNELS not set: real-ND2 smoke test is opt-in"


@pytest.fixture
def real_nd2():
    path, chans = os.environ.get("CATKIT_REAL_ND2"), os.environ.get("CATKIT_REAL_ND2_CHANNELS")
    if not path or not chans:
        pytest.skip(REAL_SKIP)
    p = Path(path)
    if not p.is_file():
        pytest.fail("CATKIT_REAL_ND2 does not point to a file")
    st = p.stat()
    yield p, [c for c in chans.split(",") if c]
    st2 = p.stat()
    assert (st.st_size, st.st_mtime_ns) == (st2.st_size, st2.st_mtime_ns), "the real ND2 file changed during the test"


@pytest.mark.research
def test_parse_capturing_exposures_is_verbatim(frozen_io_modules):
    G = frozen_io_modules["REF-IO"]
    assert inspect.getsource(IO.parse_capturing_exposures) == inspect.getsource(G.parse_capturing_exposures)


@pytest.mark.real_nd2
def test_real_header_and_pixels_smoke(real_nd2, record_property):
    path, names = real_nd2
    h = IO.inspect_nd2(path)
    record_property("real_nd2_sizes", str(h["sizes"]))
    record_property("real_nd2_voxel_nm_zyx", str(h["voxel_size_nm"]))
    record_property("real_nd2_channels", str(h["channel_names"]))
    assert tuple(h["sizes"]) == ("Z", "C", "Y", "X") and h["dtype"] == "uint16"
    assert set(names) <= set(h["channel_names"])
    if not h["is_baseline_voxel"]:                                   # the default load must refuse a non-baseline voxel
        with pytest.raises(IO.ND2Error, match="not supported in v0.1"):
            IO.load_nd2_channels(path, names)
    st = IO.load_nd2_channels(path, names, require_baseline_voxel=False)
    Z, Y, X = h["sizes"]["Z"], h["sizes"]["Y"], h["sizes"]["X"]
    assert list(st.channels) == names and st.voxel_size_nm == h["voxel_size_nm"]
    for n, a in st.channels.items():
        assert a.shape == (Z, Y, X) and a.dtype == np.uint16 and a.max() > a.min()
        record_property(f"real_nd2_{n}_min_median_max", f"{int(a.min())}/{int(np.median(a))}/{int(a.max())}")


@pytest.mark.real_nd2
@pytest.mark.research
def test_real_header_and_first_channel_equal_frozen_reader(real_nd2, frozen_io_modules):
    path, names = real_nd2
    G, B = frozen_io_modules["REF-IO"], frozen_io_modules["REF-DET"]
    fh, fo = G.read_header(path), B.optics_from_nd2(path)
    h = IO.inspect_nd2(path)
    assert fh["sizes"] == h["sizes"] and fh["dtype"] == h["dtype"] and fh["channel_names"] == h["channel_names"]
    assert (fh["dz_um"] * 1000.0, fh["px_y_um"] * 1000.0, fh["px_x_um"] * 1000.0) == h["voxel_size_nm"]
    assert fh["z_step_um"] == h["z_step_um"] and fh["exposures_ms"] == h["exposures_ms"]
    assert fo == h["optics"]
    fstacks, _ = G.load_channel_stacks(path, wanted=(names[0],))
    port = IO.load_nd2_channels(path, [names[0]], require_baseline_voxel=False)
    assert fstacks[names[0]].dtype == port.channels[names[0]].dtype
    assert np.array_equal(fstacks[names[0]], port.channels[names[0]])
