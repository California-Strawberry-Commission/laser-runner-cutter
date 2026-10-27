import asyncio
import logging

import numpy as np
import pytest
from std_msgs.msg import Header

import furrow_perceiver.furrow_perceiver_node as furrow_perceiver_node
from furrow_perceiver.furrow_tracker import FurrowTracker


class _FurrowTrackerStub:

    def __init__(self, error):
        self._error = error
        self.guidance_offset_x = 0

    def process(self, depth_img):
        pass

    def get_error(self):
        return self._error


class _FurrowAnnotatorStub:
    def annotate(self, img, draw_timings=False):
        pass


def _depth_image_kwargs(height=40, width=60, fill=1000):
    depth = np.full((height, width), fill, dtype=np.uint16)
    return {
        "header": Header(),
        "height": height,
        "width": width,
        "encoding": "16UC1",
        "is_bigendian": 0,
        "step": width * 2,
        "data": depth.tobytes(),
    }


@pytest.fixture(autouse=True)
def reset_shared_state(monkeypatch):
    fresh_state = furrow_perceiver_node.SharedState()
    fresh_state.logger = logging.getLogger("test_furrow_perceiver_node")
    monkeypatch.setattr(furrow_perceiver_node, "shared_state", fresh_state)
    monkeypatch.setattr(
        furrow_perceiver_node.state_topic, "publish", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        furrow_perceiver_node.debug_img_topic, "publish", lambda *a, **kw: None
    )
    monkeypatch.setattr(furrow_perceiver_node.perceiver_params, "guidance_offset", 0)
    yield fresh_state


@pytest.fixture
def published_results(monkeypatch):
    results = []
    monkeypatch.setattr(
        furrow_perceiver_node.tracker_result_topic,
        "publish",
        lambda **kwargs: results.append(kwargs),
    )
    return results


def test_set_guidance_offset_without_tracker(reset_shared_state):
    assert reset_shared_state.tracker is None

    result = asyncio.run(furrow_perceiver_node.set_guidance_offset(None, data=-12))

    assert result == {"success": True}
    assert furrow_perceiver_node.perceiver_params.guidance_offset == -12


def test_set_guidance_offset_applies_to_existing_tracker(reset_shared_state):
    reset_shared_state.tracker = _FurrowTrackerStub(error=0.0)

    asyncio.run(furrow_perceiver_node.set_guidance_offset(None, data=-12))

    assert reset_shared_state.tracker.guidance_offset_x == -12
    assert furrow_perceiver_node.perceiver_params.guidance_offset == -12


def test_on_depth_image_publishes_invalid_when_no_error(
    reset_shared_state, published_results
):
    reset_shared_state.tracker = _FurrowTrackerStub(error=None)
    reset_shared_state.annotator = _FurrowAnnotatorStub()

    asyncio.run(furrow_perceiver_node.on_depth_image(None, **_depth_image_kwargs()))

    assert published_results == [
        {"linear_deviation": 0.0, "heading": 0.0, "is_valid": False}
    ]


def test_on_depth_image_publishes_deviation_when_error_available(
    reset_shared_state, published_results
):
    reset_shared_state.tracker = _FurrowTrackerStub(error=-37)
    reset_shared_state.annotator = _FurrowAnnotatorStub()

    asyncio.run(furrow_perceiver_node.on_depth_image(None, **_depth_image_kwargs()))

    assert published_results == [
        {"linear_deviation": -37.0, "heading": 0.0, "is_valid": True}
    ]


def test_on_depth_image_initializes_tracker_once(reset_shared_state, published_results):
    kwargs = _depth_image_kwargs()

    asyncio.run(furrow_perceiver_node.on_depth_image(None, **kwargs))
    tracker = reset_shared_state.tracker

    assert isinstance(tracker, FurrowTracker)
    assert reset_shared_state.annotator is not None

    asyncio.run(furrow_perceiver_node.on_depth_image(None, **kwargs))

    assert reset_shared_state.tracker is tracker
