import math

import numpy as np

from furrow_perceiver.furrow_strip_tracker import (
    DEPTH_HFOV,
    FURROW_MAX_WIDTH,
    FURROW_MIN_WIDTH,
    FurrowStripTracker,
)


def _make_plateau(width: int, left_bound: int, right_bound: int, wall_val, floor_val):
    # Builds a 1D depth row with a far floor and wall values.
    arr = np.full(width, wall_val, dtype=np.float64)
    arr[left_bound:right_bound] = floor_val
    return arr


def test_find_bounds_default_search_start():
    tracker = FurrowStripTracker(strip_height=4, strip_idx=0, overall_dim=(40, 50))
    convolution = _make_plateau(50, 15, 35, wall_val=600, floor_val=1000)

    l_bound, r_bound = tracker.find_bounds(convolution)

    assert (l_bound, r_bound) == (15, 35)
    assert tracker._reference_distance == 1000


def test_find_bounds_explicit_search_start():
    tracker = FurrowStripTracker(strip_height=4, strip_idx=0, overall_dim=(40, 50))
    # Shift the plateau off-center so the default (len // 2) search_start would miss it,
    # and confirm an explicit search_start is used.
    convolution = _make_plateau(50, 5, 20, wall_val=600, floor_val=1000)

    l_bound, r_bound = tracker.find_bounds(convolution, search_start=10)

    assert (l_bound, r_bound) == (5, 20)


def test_process_results_computes_furrow_width_from_bounds():
    tracker = FurrowStripTracker(strip_height=10, strip_idx=0, overall_dim=(100, 200))
    tracker._left_bound, tracker._right_bound = 80, 120
    tracker._reference_distance = 1000

    tracker.process_results(previous_tracker=None)

    rad_per_px = DEPTH_HFOV / 200
    expected_width = abs((120 - 80) * rad_per_px * 1000)
    assert tracker._furrow_width == expected_width
    assert tracker._delta_x_from_previous == 0


def test_process_results_delta_x_from_previous():
    tracker = FurrowStripTracker(strip_height=10, strip_idx=1, overall_dim=(100, 200))
    tracker._left_bound, tracker._right_bound = 80, 120
    tracker._x_center = 100
    tracker._reference_distance = 1000

    previous = FurrowStripTracker(strip_height=10, strip_idx=0, overall_dim=(100, 200))
    previous._x_center = 90

    tracker.process_results(previous_tracker=previous)

    assert tracker._delta_x_from_previous == 10


def test_check_validity_true_for_typical_furrow():
    tracker = FurrowStripTracker(strip_height=10, strip_idx=0, overall_dim=(100, 200))
    tracker._left_bound, tracker._right_bound = 80, 120
    tracker._reference_distance = 1000
    tracker.process_results(previous_tracker=None)

    assert FURROW_MIN_WIDTH <= tracker._furrow_width <= FURROW_MAX_WIDTH
    assert tracker.check_validity() is True


def test_check_validity_false_when_too_narrow():
    tracker = FurrowStripTracker(strip_height=10, strip_idx=0, overall_dim=(100, 200))
    tracker._left_bound, tracker._right_bound = 99, 101
    tracker._reference_distance = 1000
    tracker.process_results(previous_tracker=None)

    assert tracker._furrow_width < FURROW_MIN_WIDTH
    assert tracker.check_validity() is False


def test_check_validity_false_when_too_wide():
    tracker = FurrowStripTracker(strip_height=10, strip_idx=0, overall_dim=(100, 200))
    tracker._left_bound, tracker._right_bound = 10, 190
    tracker._reference_distance = 1000
    tracker.process_results(previous_tracker=None)

    assert tracker._furrow_width > FURROW_MAX_WIDTH
    assert tracker.check_validity() is False


def test_check_validity_false_when_left_bound_in_deadband():
    # Otherwise-valid width (span of 40px, same as test_check_validity_true_for_typical_furrow),
    # but the left bound falls inside the deadband margin near search_min, as if the left wall
    # was never actually found.
    tracker = FurrowStripTracker(strip_height=10, strip_idx=0, overall_dim=(100, 200))
    assert tracker._bound_deadband == 6
    tracker._left_bound, tracker._right_bound = 3, 43
    tracker._reference_distance = 1000
    tracker.process_results(previous_tracker=None)

    assert FURROW_MIN_WIDTH <= tracker._furrow_width <= FURROW_MAX_WIDTH
    assert tracker.check_validity() is False


def test_check_validity_false_when_right_bound_in_deadband():
    tracker = FurrowStripTracker(strip_height=10, strip_idx=0, overall_dim=(100, 200))
    assert tracker._bound_deadband == 6
    tracker._left_bound, tracker._right_bound = 160, 197
    tracker._reference_distance = 1000
    tracker.process_results(previous_tracker=None)

    assert FURROW_MIN_WIDTH <= tracker._furrow_width <= FURROW_MAX_WIDTH
    assert tracker.check_validity() is False
