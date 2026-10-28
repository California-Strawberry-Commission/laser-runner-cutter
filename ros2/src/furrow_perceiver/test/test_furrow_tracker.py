import numpy as np
import pytest

from furrow_perceiver.furrow_tracker import FurrowTracker

IMAGE_SIZE = 200


def _furrow_image(top_center, bottom_center, half_width=20, wall=600, floor=1000):
    image = np.full((IMAGE_SIZE, IMAGE_SIZE), wall, dtype=np.uint16)
    for row in range(IMAGE_SIZE):
        center = int(top_center + (bottom_center - top_center) * row / (IMAGE_SIZE - 1))
        image[row, center - half_width : center + half_width] = floor
    return image


def test_process_sizes_itself_from_the_first_image():
    tracker = FurrowTracker(num_strips=10)

    tracker.process(_furrow_image(90, 130))

    assert (tracker.width, tracker.height) == (IMAGE_SIZE, IMAGE_SIZE)
    assert tracker.pin_y == IMAGE_SIZE // 2
    assert len(tracker.strips) == 10


def test_process_reuses_its_strips_across_frames():
    tracker = FurrowTracker(num_strips=10)
    image = _furrow_image(90, 130)

    tracker.process(image)
    strips = tracker.strips

    tracker.process(image)

    assert tracker.strips is strips
    assert len(tracker.strips) == 10


def test_process_tracks_a_furrow_veering_right():
    tracker = FurrowTracker()

    tracker.process(_furrow_image(90, 130))

    assert tracker.reg_slope == pytest.approx(0.2, abs=0.05)
    assert tracker.get_error() is not None


def test_process_tracks_a_furrow_veering_left():
    tracker = FurrowTracker()

    tracker.process(_furrow_image(130, 90))

    assert tracker.reg_slope == pytest.approx(-0.2, abs=0.05)
    assert tracker.get_error() is not None


def test_error_moves_with_the_furrow():
    centered, shifted_right = FurrowTracker(), FurrowTracker()

    centered.process(_furrow_image(90, 130))
    shifted_right.process(_furrow_image(110, 150))

    assert shifted_right.get_error() < centered.get_error()
    assert shifted_right.get_error() == pytest.approx(centered.get_error() - 20, abs=3)


def test_guidance_offset_shifts_the_error_one_for_one():
    image = _furrow_image(90, 130)
    no_offset, offset = FurrowTracker(), FurrowTracker()
    no_offset.guidance_offset_x = 0
    offset.guidance_offset_x = -40

    no_offset.process(image)
    offset.process(image)

    assert no_offset.get_error() - offset.get_error() == pytest.approx(40)


def test_process_reports_no_error_without_a_furrow():
    tracker = FurrowTracker()

    tracker.process(np.full((IMAGE_SIZE, IMAGE_SIZE), 1000, dtype=np.uint16))

    assert not any(strip.is_valid for strip in tracker.strips)
    assert tracker.get_error() is None


def test_perfectly_straight_furrow_has_zero_error():
    tracker = FurrowTracker()
    tracker.guidance_offset_x = 0

    tracker.process(_furrow_image(100, 100))

    assert all(strip.is_valid for strip in tracker.strips)
    assert tracker.reg_slope == pytest.approx(0, abs=0.01)
    assert tracker.get_error() == pytest.approx(0, abs=2)


def test_straight_furrow_off_center_reports_its_offset():
    tracker = FurrowTracker()
    tracker.guidance_offset_x = 0

    tracker.process(_furrow_image(90, 90))

    assert tracker.reg_slope == pytest.approx(0, abs=0.01)
    assert tracker.get_error() == pytest.approx(10, abs=2)
