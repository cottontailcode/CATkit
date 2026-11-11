"""E-layer equivalence (frozen vs port, same process, same inputs) for M2.2 checkpoint A.

Frozen sources: ``REF-SEG`` (focus planes, small-object shim, ``segment_from_dapi``), ``REF-DRV`` (``seg_params``
and the inline dilation expression), ``REF-DET`` (``assign_cells``). Equivalence to the frozen code is NOT evidence
of real-data segmentation accuracy. New [N] interfaces (``segment_fov`` guards, ``cell_geometry``, ...) are NOT
tested here; see ``tests/synthetic/test_segmentation_synthetic.py``.

Every case compares the whole returned structure (shape, dtype, values) and asserts independently that the branch
it targets was taken. The driver replay is classified E (function equivalence over a replayed call chain), not P.
"""

from __future__ import annotations

import inspect

import pytest

from scientific import require_modules

require_modules("numpy", "scipy", "skimage", "bigfish")

import numpy as np  # noqa: E402
from scipy.ndimage import binary_dilation  # noqa: E402

import catkit.cells as C  # noqa: E402
import catkit.segmentation as S  # noqa: E402
from compare import EXACT, FLOAT, call_outcome, compare_value  # noqa: E402
from synth_seg import NUCLEI, SHAPE, make_dapi_stack  # noqa: E402

pytestmark = pytest.mark.research

DIST = FLOAT(0.0, 1e-12)  # physical distances (docs/EQUIVALENCE.md)


def _both(fr, po, args, spec, label, stats=None):
    """Run both on private copies of the arrays; compare the outcome (value or exception) and input immutability."""
    outs, problems = [], []
    for side, fn in (("frozen", fr), ("port", po)):
        a = tuple(x.copy() if isinstance(x, np.ndarray) else x for x in args)
        outs.append(call_outcome(fn, *a))
        for i, (o, n) in enumerate(zip(args, a)):
            if isinstance(o, np.ndarray) and not (o.dtype == n.dtype and np.array_equal(o, n)):
                problems.append(f"{label}: {side} modified array argument #{i} in place")
    x, y = outs
    if x[0] != y[0]:
        return problems + [f"{label}: outcome differs: frozen={x[:2]} port={y[:2]}"], x, y
    if x[0] == "raise":
        return problems + ([] if x == y else [f"{label}: exceptions differ: {x} vs {y}"]), x, y
    return problems + compare_value(x[1], y[1], spec, label, stats), x, y


# ------------------------------------------------------------------ source identity
def test_port_is_verbatim(frozen_seg_modules):
    SEG, DRV = frozen_seg_modules["REF-SEG"], frozen_seg_modules["REF-DRV"]
    for name in ("detect_focus_plane", "focus_slice_range", "_remove_small", "segment_from_dapi"):
        assert inspect.getsource(getattr(S, name)).split("  # noqa")[0] == inspect.getsource(getattr(SEG, name)), name
    # seg_params is a one-line frozen def; the port line only appends a flake8 marker/comment
    assert inspect.getsource(S.seg_params).split("  # noqa")[0] == inspect.getsource(DRV.seg_params).rstrip("\n")


def test_assign_cells_init_differs_only_by_removed_shift(frozen_seg_modules):
    """The one documented edit: shift parameters/branch removed. Pin the facts so a wider divergence is noticed."""
    DET = frozen_seg_modules["REF-DET"]
    assert list(inspect.signature(DET.assign_cells).parameters) == ["coords_zyx", "labels", "px_um", "shift_dy", "shift_dx"]
    assert list(inspect.signature(C.assign_cells_init).parameters) == ["coords_zyx", "labels", "px_um"]
    import ast
    fn = ast.parse(inspect.getsource(C.assign_cells_init)).body[0]
    assert "shift" not in ast.dump(ast.Module(fn.body[1:], []))          # no shift branch left in the body


# ------------------------------------------------------------------ focus planes
def _stack_with_peak(z, peak, n=40, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.normal(100, 1, (z, n, n))
    s[peak] += rng.normal(0, 50, (n, n))   # highest std at ``peak``
    return s


FOCUS_CASES = [("first", 9, 0), ("second", 9, 1), ("middle", 9, 4), ("penultimate", 9, 7), ("last", 9, 8),
               ("z3_mid", 3, 1), ("z3_first", 3, 0), ("z3_last", 3, 2), ("z4_a", 4, 1), ("z4_last", 4, 3), ("z4_first", 4, 0),
               ("z5_mid", 5, 2)]


@pytest.mark.parametrize("name,z,peak", FOCUS_CASES)
def test_focus_equivalence(frozen_seg_modules, name, z, peak):
    SEG = frozen_seg_modules["REF-SEG"]
    st = _stack_with_peak(z, peak)
    p1, a, b = _both(SEG.detect_focus_plane, S.detect_focus_plane, (st,), EXACT, f"{name}.plane")
    p2, c, d = _both(SEG.focus_slice_range, S.focus_slice_range, (st,), EXACT, f"{name}.range")
    assert not (p1 + p2), p1 + p2
    start, end, best, scores = d[1]
    assert best == peak and end - start == 3 and start <= best < end          # branch / independent oracle
    assert (start, end) == (min(max(0, peak - 1), z - 3), min(max(0, peak - 1), z - 3) + 3)


def test_focus_ties_and_even_num_slices(frozen_seg_modules):
    SEG = frozen_seg_modules["REF-SEG"]
    flat = np.full((6, 8, 8), 5.0)                       # all std equal -> argmax picks plane 0
    p, a, _ = _both(SEG.focus_slice_range, S.focus_slice_range, (flat,), EXACT, "ties")
    assert not p and a[1][2] == 0
    st = _stack_with_peak(9, 4)
    for ns in (1, 2, 4, 5):
        p, _, _ = _both(lambda s, n=ns: SEG.focus_slice_range(s, n), lambda s, n=ns: S.focus_slice_range(s, n), (st,), EXACT, f"ns{ns}")
        assert not p


def test_focus_fewer_than_three_planes_frozen_quirk_preserved(frozen_seg_modules):
    """Z<3: the frozen index arithmetic gives a wrong slice; the port function reproduces it (the guard lives in segment_fov)."""
    SEG = frozen_seg_modules["REF-SEG"]
    for z in (1, 2):
        st = _stack_with_peak(z, 0)
        p, a, b = _both(SEG.focus_slice_range, S.focus_slice_range, (st,), EXACT, f"z{z}")
        assert not p
        assert a[1][0] < 0                                   # start went negative: the quirk the guard exists for


# ------------------------------------------------------------------ _remove_small and segment_from_dapi
def test_remove_small_boundary_equivalence(frozen_seg_modules):
    SEG = frozen_seg_modules["REF-SEG"]
    b = np.zeros((6, 50), bool)
    for row, n in ((0, 14), (2, 15), (4, 16)):
        b[row, 1:1 + n] = True
    for ms in (1, 14, 15, 16, 17, 40):
        p, a, c = _both(SEG._remove_small, S._remove_small, (b, ms), EXACT, f"min_size{ms}")
        assert not p, p
    kept = S._remove_small(b, 15)
    assert not kept[0].any() and kept[2].sum() == 15 and kept[4].sum() == 16      # area < min_size removed
    assert "max_size" in inspect.signature(S.remove_small_objects).parameters        # skimage 0.26 branch is the one taken


def _images():
    st = make_dapi_stack()
    proj = st[2:5].max(axis=0)
    touching = make_dapi_stack(nuclei=[(0, 0, 8), (60, 119, 9), (60, 60, 6)], seed=3)[2:5].max(axis=0)   # cells on the border
    merged = make_dapi_stack(nuclei=[(40, 40, 8), (40, 52, 8)], seed=4)[2:5].max(axis=0)              # touching nuclei
    return {
        "uint16": proj, "float64": proj.astype(float), "float32": proj.astype(np.float32), "int32": proj.astype(np.int32),
        "border": touching, "touching": merged, "constant": np.full((60, 60), 7, np.uint16),
        "noise_only": np.random.default_rng(5).normal(100, 20, (80, 80)).astype(np.float64),
        "two_values": np.where(np.arange(900).reshape(30, 30) % 7 == 0, 200, 10).astype(np.uint16),
    }


PARAMS = [(15, 4, 1), (34, 6, 2), (33, 6, 2), (15, 8, 1), (1, 1, 1), (200, 4, 1), (15, 0, 1), (15, 4, 3)]


def test_segment_from_dapi_equivalence(frozen_seg_modules, e_stats):
    SEG = frozen_seg_modules["REF-SEG"]
    problems, summary = [], {}
    for iname, img in _images().items():
        for ms, ex, op in PARAMS:
            fr = lambda x, ms=ms, ex=ex, op=op: SEG.segment_from_dapi(x, ms, ex, opening_radius=op)      # noqa: E731
            po = lambda x, ms=ms, ex=ex, op=op: S.segment_from_dapi(x, ms, ex, opening_radius=op)        # noqa: E731
            p, a, b = _both(fr, po, (img,), (EXACT, EXACT, EXACT), f"{iname}/{ms},{ex},{op}", e_stats)
            problems += p
            if a[0] == "ok":
                lab, thr, cores = a[1]
                summary[(iname, ms, ex, op)] = int(cores.max())
                assert lab.dtype == b[1][0].dtype and cores.dtype == b[1][2].dtype                      # dtype as returned by skimage
                assert np.array_equal(lab, b[1][0]) and np.array_equal(cores, b[1][2])                  # label arrays exact
                assert ((lab == cores) | (cores == 0)).all()                                             # expand_labels keeps core ids
    assert not problems, problems[:10]
    # branch-trigger: the images really exercise zero cells, one merged blob, many cells
    assert summary[("constant", 15, 4, 1)] == 0 and summary[("uint16", 15, 4, 1)] == len(NUCLEI)
    assert summary[("touching", 15, 4, 1)] == 1 and summary[("noise_only", 15, 4, 1)] >= 0
    assert summary[("uint16", 200, 4, 1)] < len(NUCLEI) + 1


# ------------------------------------------------------------------ seg_params
SEG_PX = [65.0, 43.3333333, 43.333, 65 * 2 / 3, 32.5, 104.0, 130.0, 50.0, 97.5, 65.0001, 300.0]


def test_seg_params_equivalence_and_oracle(frozen_seg_modules):
    DRV = frozen_seg_modules["REF-DRV"]
    for px in SEG_PX:
        assert S.seg_params(px) == DRV.seg_params(px), px
        assert all(type(v) is int for v in S.seg_params(px).values())
    # hand-computed literals (independent of both implementations)
    assert S.seg_params(65.0) == dict(min_size=15, expand=4, opening=1, dil=10)
    assert S.seg_params(104.0) == dict(min_size=6, expand=2, opening=1, dil=6)    # 4*65/104 = 2.5 -> 2 (half-to-even, not 3); 6.25 -> 6
    assert S.seg_params(130.0) == dict(min_size=4, expand=2, opening=0, dil=5)    # 65/130 = 0.5 -> 0 (half-to-even)
    assert S.seg_params(43.3333333) == dict(min_size=34, expand=6, opening=2, dil=15)
    assert S.seg_params(32.5) == dict(min_size=60, expand=8, opening=2, dil=20)


# ------------------------------------------------------------------ noise mask (REF-DRV inline expression)
def _frozen_outside(lab, dil):
    """The driver expression, copied: ``outside = ~binary_dilation(lab > 0, iterations=sp["dil"])``."""
    return ~binary_dilation(lab > 0, iterations=dil)


def _labels_for_mask():
    st = make_dapi_stack()
    lab = S.segment_from_dapi(st[2:5].max(axis=0), 15, 4, opening_radius=1)[0].astype(np.int32)
    corner = np.zeros((40, 50), np.int32)
    corner[0:3, 0:3] = 1
    edge = np.zeros((40, 50), np.int32)
    edge[39, 20:24] = 2
    single = np.zeros((41, 41), np.int32)
    single[20, 20] = 1
    return {"segmented": lab, "empty": np.zeros((30, 30), np.int32), "full": np.ones((30, 30), np.int32),
            "corner": corner, "edge": edge, "single": single, "int64": lab.astype(np.int64)}


@pytest.mark.parametrize("dil", [1, 2, 10, 15, 0])
def test_noise_mask_equivalence(dil):
    """No frozen module needed (the expression is inline in the driver), but the replay is the frozen reference."""
    for name, lab in _labels_for_mask().items():
        a, b = _frozen_outside(lab.copy(), dil), S.noise_mask(lab.copy(), dil)
        assert a.dtype == b.dtype == bool and np.array_equal(a, b), (name, dil)
    # branch triggers
    assert S.noise_mask(np.zeros((30, 30), np.int32), 10).all()                     # empty -> whole frame outside
    assert not S.noise_mask(np.ones((30, 30), np.int32), 10).any()                  # full -> nothing outside
    assert not np.array_equal(S.noise_mask(_labels_for_mask()["single"], 10), S.noise_mask(_labels_for_mask()["single"], 9))


def test_noise_mask_uses_expanded_labels_not_cores():
    labs = _labels_for_mask()["segmented"]
    st = make_dapi_stack()
    _, _, cores = S.segment_from_dapi(st[2:5].max(axis=0), 15, 4, opening_radius=1)
    assert (labs > 0).sum() > (cores > 0).sum()                                      # expansion grew the footprint
    assert not np.array_equal(S.noise_mask(labs, 10), S.noise_mask(cores.astype(np.int32), 10))


# ------------------------------------------------------------------ assign_cells
def _coords(lab, n=60, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, lab.shape[0], n)
    x = rng.integers(0, lab.shape[1], n)
    z = rng.integers(0, 9, n)
    return np.stack([z, y, x], axis=1).astype(np.int64)


@pytest.mark.parametrize("px_um", [0.065, 0.0433333, 1.0])
def test_assign_cells_equivalence(frozen_seg_modules, e_stats, px_um):
    DET = frozen_seg_modules["REF-DET"]
    problems, n_in, n_out = [], 0, 0
    for name, lab in _labels_for_mask().items():
        co = _coords(lab)
        p, a, b = _both(lambda c, l, f=DET.assign_cells: f(c, l, px_um), lambda c, l: C.assign_cells_init(c, l, px_um), (co, lab),
                        (EXACT, DIST), name, e_stats)
        problems += p
        if a[0] == "ok":
            n_in += int((a[1][0] > 0).sum())
            n_out += int((a[1][0] == 0).sum())
            assert a[1][0].dtype == b[1][0].dtype == np.dtype(int)
    assert not problems, problems[:10]
    assert n_in > 0 and n_out > 0                                                    # both inside-cell and outside lookups occurred


def test_assign_cells_out_of_range_and_negative_frozen_behaviour(frozen_seg_modules):
    """Not guarded (frozen behaviour): >= shape raises IndexError, negative wraps. Both implementations agree."""
    DET = frozen_seg_modules["REF-DET"]
    lab = _labels_for_mask()["segmented"]
    for co in (np.array([[0, lab.shape[0], 3]]), np.array([[0, 3, lab.shape[1]]]), np.array([[0, -1, -2]])):
        p, a, b = _both(lambda c, l: DET.assign_cells(c, l, 0.065), lambda c, l: C.assign_cells_init(c, l, 0.065), (co, lab), (EXACT, DIST), "oob")
        assert not p and a[0] == b[0]
    assert call_outcome(C.assign_cells_init, np.array([[0, lab.shape[0], 3]]), lab, 0.065)[0] == "raise"
    assert call_outcome(C.assign_cells_init, np.array([[0, -1, -2]]), lab, 0.065)[0] == "ok"


# ------------------------------------------------------------------ driver front-half replay (E, not P)
def _frozen_front_half(SEG, DRV, stack, px_nm):
    sp = DRV.seg_params(px_nm)
    z_start, z_end, z_best, _ = SEG.focus_slice_range(stack, 3)
    proj = np.max(SEG.detect_focus_plane(stack, 3), axis=0)
    lab, thr, cores = SEG.segment_from_dapi(proj, sp["min_size"], sp["expand"], opening_radius=sp["opening"])
    lab = lab.astype(np.int32)
    cores = cores.astype(np.int32)
    outside = ~binary_dilation(lab > 0, iterations=sp["dil"])
    return dict(sp=sp, focus=(z_start, z_end, z_best), proj=proj, lab=lab, thr=thr, cores=cores, outside=outside, n_cells=int(lab.max()))


def test_front_half_replay_equivalence(frozen_seg_modules, e_stats):
    SEG, DRV, DET = (frozen_seg_modules[k] for k in ("REF-SEG", "REF-DRV", "REF-DET"))
    problems = []
    stacks = {
        "four_nuclei_focus3": make_dapi_stack(),
        "focus_at_first": make_dapi_stack(focus_z=0, seed=1),
        "focus_at_last": make_dapi_stack(focus_z=6, seed=2),
        "z3_minimum": make_dapi_stack(shape=(3, 120, 120), focus_z=1, seed=6),
        "border_cells": make_dapi_stack(nuclei=[(0, 0, 8), (60, 119, 9), (60, 60, 6)], seed=3),
        "no_cells_constant": np.full(SHAPE, 50, np.uint16),
    }
    ncells = {}
    for name, stack in stacks.items():
        fr = _frozen_front_half(SEG, DRV, stack, 65.0)
        with pytest.warns(S.NoCellsWarning) if fr["n_cells"] == 0 else _nullcontext():
            res = S.segment_fov(stack, 65.0)
        mask = S.noise_mask(res.cell_labels, res.seg_params["dil"])
        for key, got in (("lab", res.cell_labels), ("cores", res.nucleoid_cores), ("outside", mask)):
            problems += compare_value(fr[key], got, EXACT, f"{name}.{key}", e_stats)
        # frozen returns the Otsu threshold in the input dtype; segment_fov [N] reports float(thr): values compared
        problems += compare_value(np.float64(fr["thr"]), np.float64(res.otsu_threshold), EXACT, f"{name}.thr", e_stats)
        assert (res.focus_start, res.focus_end, res.focus_best) == tuple(int(v) for v in fr["focus"]), name
        assert res.seg_params == fr["sp"] and res.n_cells == fr["n_cells"], name
        # downstream lookup on the replayed labels
        co = _coords(fr["lab"], seed=9)
        pa = DET.assign_cells(co, fr["lab"], 0.065)
        pb = C.assign_cells_init(co, res.cell_labels, 0.065)
        problems += compare_value(pa[0], pb[0], EXACT, f"{name}.cid") + compare_value(pa[1], pb[1], DIST, f"{name}.dist", e_stats)
        ncells[name] = fr["n_cells"]
    assert not problems, problems[:10]
    assert ncells["four_nuclei_focus3"] == len(NUCLEI) and ncells["no_cells_constant"] == 0     # branch triggers
    assert ncells["border_cells"] == 3 and ncells["z3_minimum"] == len(NUCLEI)


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
