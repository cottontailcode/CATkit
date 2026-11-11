"""QC overlay: coordinate conventions and content (the figure is a diagnostic; positions must be exactly right)."""

from __future__ import annotations

import pytest

from scientific import require_modules

require_modules("numpy", "skimage", "matplotlib")

import matplotlib.image as mimg  # noqa: E402
import numpy as np  # noqa: E402

import catkit.qc as QC  # noqa: E402


def make_case():
    Y, X = 60, 90
    proj = np.full((Y, X), 100.0)
    proj[40, 70] = 4000.0                                    # a single bright pixel at (y=40, x=70)
    labels = np.zeros((Y, X), np.int32)
    labels[10:20, 10:25] = 1
    spots = dict(channel=np.array(["Cy5", "Cy5", "Cy5", "TRITC"], dtype=object),
                 x_use=np.array([70.0, 12.3, 50.0, 5.0]), y_use=np.array([40.0, 15.7, 30.0, 5.0]),
                 localization_quality=np.array(["good", "suspect", "unreliable", "good"], dtype=object))
    return proj, labels, spots


def test_positions_extent_and_legend():
    proj, labels, spots = make_case()
    fig = QC.draw_overlay(proj, labels, spots, "Cy5", fov_id="f")
    ax = fig.axes[0]
    img = ax.get_images()[0]
    assert list(img.get_extent()) == [-0.5, 89.5, 59.5, -0.5]                          # pixel centres at integers, row 0 on top
    assert ax.get_xlim() == (-0.5, 89.5) and ax.get_ylim() == (59.5, -0.5)
    offs = {c.get_label().split(" ")[0]: c.get_offsets() for c in ax.collections if c.get_label().split(" ")[0] in QC.QUALITY_STYLE}
    assert offs["good"].tolist() == [[70.0, 40.0]] and offs["suspect"].tolist() == [[12.3, 15.7]] and offs["unreliable"].tolist() == [[50.0, 30.0]]
    labels_txt = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "good (n=1)" in labels_txt and "suspect (n=1)" in labels_txt and "unreliable (n=1)" in labels_txt            # TRITC row ignored
    # outline lies on the inner boundary of the labelled rectangle
    out = next(c for c in ax.collections if c.get_label().startswith("cell outline")).get_offsets()
    xs, ys = out[:, 0], out[:, 1]
    assert xs.min() == 10 and xs.max() == 24 and ys.min() == 10 and ys.max() == 19 and len(out) == 2 * 15 + 2 * 8


def test_marker_is_drawn_over_the_bright_pixel():
    """Rendered check: the display position of data (x=70, y=40) is the brightest image pixel (hollow marker leaves it visible)."""
    proj, labels, spots = make_case()
    spots["x_use"], spots["y_use"] = spots["x_use"].copy(), spots["y_use"].copy()
    spots["localization_quality"][0] = "unreliable"       # marker colour is irrelevant; hollow circles do not hide the centre
    fig = QC.draw_overlay(proj, labels, spots, "Cy5")
    fig.canvas.draw()
    ax = fig.axes[0]
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(float)
    H = buf.shape[0]
    px, py = ax.transData.transform((70.0, 40.0))
    r, c = int(round(H - py)), int(round(px))
    neigh = buf[r - 1:r + 2, c - 1:c + 2].sum(axis=2)
    assert neigh.max() > 3 * 200                                             # a white pixel under the marker centre
    far = buf[r - 1:r + 2, c - 12:c - 8].sum(axis=2)
    assert far.max() < 3 * 200
    assert buf.shape[1] == 90                                                # one figure pixel per image pixel (full resolution)
    # the cell outline is visible at full resolution: a boundary pixel of the labelled rectangle is clearly blue (not grey background)
    ox, oy = ax.transData.transform((17.0, 10.0))
    orow, ocol = int(round(H - oy)), int(round(ox))
    px = buf[orow - 1:orow + 2, ocol - 1:ocol + 2].reshape(-1, 3)
    assert (px[:, 2] - px[:, 0]).max() > 80


def test_save_and_validation(tmp_path):
    proj, labels, spots = make_case()
    f = QC.save_overlay(tmp_path / "o.png", proj, labels, spots, "Cy5")
    im = mimg.imread(f)
    assert im.shape[1] == 90 and im.shape[0] > 60
    with pytest.raises(ValueError, match="same .Y, X. shape"):
        QC.draw_overlay(proj, labels[:10], spots, "Cy5")
