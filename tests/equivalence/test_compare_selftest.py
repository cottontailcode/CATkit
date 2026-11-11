"""Self-tests of the E-layer comparison helper (no frozen code involved)."""

from __future__ import annotations

import pytest

from scientific import require_modules

require_modules("numpy")

import numpy as np  # noqa: E402

from compare import EXACT, FLOAT, POS, Stats, call_outcome, compare_dicts, compare_value  # noqa: E402


def test_unknown_key_is_an_error():
    with pytest.raises(AssertionError, match="no tolerance spec"):
        compare_dicts({"mystery": 1.0}, {"mystery": 1.0}, {})


def test_key_set_difference_reported():
    probs = compare_dicts({"a": 1}, {"a": 1, "_roi_lo": (0, 0, 0)}, {"a": EXACT, "_roi_lo": EXACT})
    assert any("key sets differ" in p for p in probs)


def test_float_tolerance_boundary_uses_atol_and_rtol():
    assert compare_value(1.0, 1.0 + 5e-10, FLOAT(1e-9, 0.0), "x") == []
    assert compare_value(1.0, 1.0 + 2e-9, FLOAT(1e-9, 0.0), "x") != []
    assert compare_value(100.0, 100.0 + 5e-7, FLOAT(0.0, 1e-9), "x") != []
    assert compare_value(100.0, 100.0 + 5e-7, FLOAT(0.0, 1e-6), "x") == []


def test_nan_inf_handling():
    nan = float("nan")
    assert compare_value(nan, nan, POS, "x") == []
    assert compare_value(nan, 1.0, POS, "x") != []
    assert compare_value(float("inf"), float("inf"), POS, "x") == []
    assert compare_value(float("inf"), float("-inf"), POS, "x") != []
    assert compare_value(float("inf"), 1e300, POS, "x") != []
    a, b = np.array([1.0, np.nan, np.inf]), np.array([1.0, np.nan, np.inf])
    assert compare_value(a, b, EXACT, "arr") == []
    assert compare_value(a, np.array([1.0, 2.0, np.inf]), EXACT, "arr") != []


def test_signed_zero_equal_but_not_bit_identical():
    stats = Stats()
    assert compare_value(0.0, -0.0, EXACT, "z", stats) == []
    assert stats.compared == 1 and stats.bit_identical == 0 and stats.not_bit_identical


def test_exact_float_differing_by_one_ulp_fails():
    assert compare_value(1.0, float(np.nextafter(1.0, 2.0)), EXACT, "x") != []


def test_type_shape_dtype_and_sequence_checks():
    assert compare_value(True, 1, EXACT, "x") != []
    assert compare_value(np.zeros(3), np.zeros(4), EXACT, "x") != []
    assert compare_value(np.zeros(3), np.zeros(3, dtype=np.float32), EXACT, "x") != []
    assert compare_value((1, 2), [1, 2], EXACT, "x") != []
    assert compare_value(["a", "b"], ["b", "a"], EXACT, "reasons") != []  # order matters
    assert compare_value("", "", EXACT, "s") == []


def test_non_float_with_float_spec_is_rejected():
    assert compare_value("a", "a", POS, "x") != []


def test_call_outcome_captures_exceptions():
    def boom():
        raise ValueError("x")

    assert call_outcome(boom) == ("raise", "ValueError", "x")
    assert call_outcome(lambda: 3) == ("ok", 3)


def test_large_integer_arrays_are_not_compared_through_float64():
    a = np.array([2 ** 53, 5], dtype=np.int64)
    b = np.array([2 ** 53 + 1, 5], dtype=np.int64)       # equal after a float64 cast, different as integers
    assert a.astype(float).tolist() == b.astype(float).tolist()
    assert compare_value(a, b, EXACT, "ints") != []
    assert compare_value(a, a.copy(), EXACT, "ints") == []
    assert compare_value(np.array([True, False]), np.array([True, True]), EXACT, "bools") != []


def test_per_element_spec_for_tuples():
    from compare import NEIGHBOR_SPEC

    d, j = np.array([1.0, 2.0]), np.array([1, 0])
    assert compare_value((d, j), (d.copy(), j.copy()), NEIGHBOR_SPEC, "nn") == []
    # derived distance may differ by 1e-12 relative, the neighbour index never
    assert compare_value((d, j), (d * (1 + 5e-13), j), NEIGHBOR_SPEC, "nn") == []
    assert compare_value((d, j), (d, np.array([1, 1])), NEIGHBOR_SPEC, "nn") != []
    assert compare_value((d, j), (d,), NEIGHBOR_SPEC, "nn") != []
    assert compare_value((d, j), (d, j, j), NEIGHBOR_SPEC, "nn") != []


def test_photometry_spec_covers_all_measure_spots_keys_explicitly():
    from compare import CLIPBG_SPEC, EXACT, PHOT_SPEC

    assert len(PHOT_SPEC) == 30 and len(CLIPBG_SPEC) == 7
    for raw in ("peak_raw", "aperture_sum", "n_aperture_vox", "n_bg_voxels_valid", "bg_insufficient"):
        assert PHOT_SPEC[raw] == EXACT
    assert PHOT_SPEC["peak_net"] == FLOAT(0.0, 1e-12) and PHOT_SPEC["snr_primary"] == FLOAT(0.0, 1e-12)
    # raw peak must really be compared exactly: a 1-ulp difference is reported
    a = {k: np.zeros(1) if PHOT_SPEC[k] != EXACT else np.zeros(1) for k in PHOT_SPEC}
    b = {k: v.copy() for k, v in a.items()}
    b["peak_raw"] = np.array([np.nextafter(0.0, 1.0)])
    assert any("peak_raw" in p for p in compare_dicts(a, b, PHOT_SPEC))
