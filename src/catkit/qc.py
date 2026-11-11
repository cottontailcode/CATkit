"""Minimal QC overlay for one detection channel (diagnostic figure, not a publication figure). [N]

The image is the maximum projection of the channel over Z, shown at full resolution (one figure pixel per image pixel,
rounded). Conventions: image row = ``y``, column = ``x``; pixel centres sit at integer coordinates, so a spot at
``(y, x)`` is drawn at data coordinates ``(x, y)`` and the image extent is ``(-0.5, X - 0.5, Y - 0.5, -0.5)`` (origin at the top left).

Plotted positions are the **use coordinates** of ``spots.csv`` (``x_use``, ``y_use``): the *fitted* position for
``good`` and ``suspect`` localizations and the *initial* integer position for ``unreliable`` ones (whose fit is not
trusted). Marker colour = ``localization_quality``. Cell outlines are the baseline (expanded) ``cell_labels``.
"""
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

QUALITY_STYLE = {"good": "lime", "suspect": "yellow", "unreliable": "red"}


def draw_overlay(projection, labels, spots, channel, fov_id="", dpi=100):
    """Return a ``Figure``. ``projection``: ``(Y, X)`` array; ``labels``: ``(Y, X)`` int cell labels;
    ``spots``: column dict with ``channel``, ``x_use``, ``y_use``, ``localization_quality`` (rows of other channels are ignored)."""
    from skimage.segmentation import find_boundaries

    proj = np.asarray(projection, float)
    Y, X = proj.shape
    if np.shape(labels) != (Y, X):
        raise ValueError("labels and projection must have the same (Y, X) shape")
    lo, hi = np.percentile(proj, [1, 99.95])
    img = np.clip((proj - lo) / (hi - lo if hi > lo else 1.0), 0, 1)
    fig = Figure(figsize=(X / dpi, Y / dpi + 0.45), dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, Y / dpi / (Y / dpi + 0.45)])
    ax.imshow(img, cmap="gray", interpolation="nearest", extent=(-0.5, X - 0.5, Y - 0.5, -0.5))
    yy, xx = np.nonzero(find_boundaries(np.asarray(labels), mode="inner"))
    ax.scatter(xx, yy, s=2.5, c="deepskyblue", linewidths=0, marker="s", label="cell outline (baseline labels)")   # ~1.5 px squares: visible at full resolution
    sel = np.asarray(spots["channel"]) == channel
    counts = {}
    for q, color in QUALITY_STYLE.items():
        m = sel & (np.asarray(spots["localization_quality"]) == q)
        counts[q] = int(m.sum())
        ax.scatter(np.asarray(spots["x_use"], float)[m], np.asarray(spots["y_use"], float)[m], s=28, facecolors="none",
                   edgecolors=color, linewidths=0.8, label=f"{q} (n={counts[q]})")
    ax.set_xlim(-0.5, X - 0.5)
    ax.set_ylim(Y - 0.5, -0.5)
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.6, markerscale=3)
    fig.text(0.005, 0.995, f"{fov_id} | {channel}: max projection over Z | accepted spots at USE coordinates "
             "(fit for good/suspect, initial for unreliable) | x right, y down", fontsize=8, va="top", ha="left")
    return fig


def save_overlay(path, projection, labels, spots, channel, fov_id=""):
    fig = draw_overlay(projection, labels, spots, channel, fov_id)
    fig.savefig(path)
    return path
