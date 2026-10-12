from types import SimpleNamespace

import numpy as np

from furrow_perceiver.furrow_tracker import FurrowTracker


def _make_strip(x_center, y_center, is_valid=True):
    return SimpleNamespace(x_center=x_center, y_center=y_center, is_valid=is_valid)


def test_init_sets_dimensions_and_strips():
    tracker = FurrowTracker(num_strips=4)
    tracker.init(np.zeros((40, 60)))

    assert tracker.width == 60
    assert tracker.height == 40
    assert tracker.pin_y == 20  # 2 * height // 4
    assert len(tracker.strips) == 4


def test_init_is_noop_once_dimensions_are_set():
    tracker = FurrowTracker(num_strips=4)
    tracker.init(np.zeros((40, 60)))
    tracker.init(np.zeros((80, 100)))

    assert (tracker.width, tracker.height) == (60, 40)
    assert len(tracker.strips) == 4


def test_regress_strips_fits_line_through_valid_centerpoints():
    tracker = FurrowTracker()
    tracker._width, tracker._height = 100, 100
    # Collinear points on x = 1*y + 10
    tracker._strips = [
        _make_strip(x_center=10, y_center=0),
        _make_strip(x_center=20, y_center=10),
        _make_strip(x_center=30, y_center=20),
    ]

    tracker.regress_strips()

    assert tracker.reg_slope == 1.0
    assert tracker.reg_intercept == 10.0


def test_regress_strips_ignores_invalid_strips():
    tracker = FurrowTracker()
    tracker._width, tracker._height = 100, 100
    tracker._strips = [
        _make_strip(x_center=10, y_center=0),
        _make_strip(x_center=20, y_center=10),
        _make_strip(x_center=30, y_center=20),
        _make_strip(x_center=9999, y_center=9999, is_valid=False),
    ]

    tracker.regress_strips()

    assert tracker.reg_slope == 1.0
    assert tracker.reg_intercept == 10.0


def test_regress_strips_returns_none_with_two_or_fewer_valid_strips():
    tracker = FurrowTracker()
    tracker._width, tracker._height = 100, 100
    tracker._strips = [
        _make_strip(x_center=10, y_center=0),
        _make_strip(x_center=20, y_center=10),
    ]

    tracker.regress_strips()

    assert tracker.reg_slope is None
    assert tracker.reg_intercept is None


def test_regress_strips_returns_none_for_vertical_line():
    tracker = FurrowTracker()
    tracker._width, tracker._height = 100, 100
    tracker._strips = [
        _make_strip(x_center=10, y_center=0),
        _make_strip(x_center=10, y_center=10),
        _make_strip(x_center=10, y_center=20),
    ]

    tracker.regress_strips()

    assert tracker.reg_slope is None
    assert tracker.reg_intercept is None


def test_get_reg_x_uses_fitted_line():
    tracker = FurrowTracker()
    tracker._reg_slope, tracker._reg_intercept = 1.0, 10.0

    assert tracker.get_reg_x(5) == 15


def test_get_reg_x_returns_none_without_a_fit():
    tracker = FurrowTracker()
    tracker._reg_slope, tracker._reg_intercept = None, None

    assert tracker.get_reg_x(5) is None


def test_get_error_returns_offset_from_fitted_centerline():
    tracker = FurrowTracker()
    tracker._width = 100
    tracker.guidance_offset_x = -40  # default
    tracker._reg_slope, tracker._reg_intercept = 1.0, 10.0
    tracker._pin_y = 50

    # guidance_x = 100 // 2 + (-40) = 10; pin_x = get_reg_x(50) = 60
    assert tracker.get_error() == 10 - 60


def test_get_error_returns_none_without_a_fit():
    tracker = FurrowTracker()
    tracker._width = 100
    tracker._pin_y = 50
    tracker._reg_slope, tracker._reg_intercept = None, None

    assert tracker.get_error() is None


def test_get_error_handles_pin_x_equal_to_zero():
    tracker = FurrowTracker()
    tracker._width = 100
    tracker._pin_y = 50
    tracker._reg_slope, tracker._reg_intercept = 1.0, -50.0

    assert tracker.get_reg_x(tracker.pin_y) == 0
    # guidance_x = 100 // 2 + (-40) = 10; error = guidance_x - pin_x = 10 - 0
    assert tracker.get_error() == 10
