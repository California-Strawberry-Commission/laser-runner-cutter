import numpy as np
import pytest

from furrow_perceiver.furrow_strip_tracker import FurrowStripTracker

IMAGE_HEIGHT = 40
IMAGE_WIDTH = 200
STRIP_HEIGHT = 10


def _furrow_image(left, right, wall=600, floor=1000):
    # A depth image of a furrow, walls are near and floor is far
    image = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), wall, dtype=np.uint16)
    image[:, left:right] = floor
    return image


def _tracker(strip_idx=0):
    return FurrowStripTracker(
        strip_height=STRIP_HEIGHT,
        strip_idx=strip_idx,
        overall_dim=(IMAGE_HEIGHT, IMAGE_WIDTH),
    )


def test_process_locates_a_centered_furrow():
    tracker = _tracker()

    tracker.process(_furrow_image(80, 120), None)

    assert tracker.is_valid
    assert tracker.left_bound == pytest.approx(80, abs=2)
    assert tracker.right_bound == pytest.approx(120, abs=2)
    assert tracker.x_center == pytest.approx(100, abs=2)


def test_process_follows_the_furrow_off_center():
    tracker = _tracker()

    tracker.process(_furrow_image(70, 110), None)

    assert tracker.is_valid
    assert tracker.x_center == pytest.approx(90, abs=2)


def test_process_invalid_when_furrow_misses_the_image_center():
    tracker = _tracker()

    tracker.process(_furrow_image(40, 80), None)

    assert not tracker.is_valid


def test_process_invalid_when_no_walls_are_visible():
    tracker = _tracker()

    tracker.process(np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 1000, dtype=np.uint16), None)

    assert not tracker.is_valid


def test_process_invalid_when_furrow_is_too_narrow():
    tracker = _tracker()

    tracker.process(_furrow_image(99, 103), None)

    assert not tracker.is_valid


def test_process_invalid_when_furrow_is_too_wide():
    tracker = _tracker()

    tracker.process(_furrow_image(20, 180), None)

    assert not tracker.is_valid


def test_strip_index_counts_up_from_the_bottom_of_the_image():
    image = _furrow_image(80, 120)
    bottom = _tracker(strip_idx=0)
    above = _tracker(strip_idx=1)

    bottom.process(image, None)
    above.process(image, None)

    assert bottom.y_center - above.y_center == STRIP_HEIGHT
