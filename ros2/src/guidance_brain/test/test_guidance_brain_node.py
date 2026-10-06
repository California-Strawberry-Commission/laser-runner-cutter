import asyncio
import logging
import time
from types import SimpleNamespace

import pytest

import guidance_brain.guidance_brain_node as guidance_brain_node
from common_interfaces.msg import PID


async def _noop_set_twist(twist):
    pass


async def _noop_restart_node(node_name):
    return SimpleNamespace(success=True, message="")


@pytest.fixture(autouse=True)
def reset_shared_state(monkeypatch):
    fresh_state = guidance_brain_node.SharedState()
    fresh_state.follower_pid = PID(p=50.0)
    fresh_state.logger = logging.getLogger("test_guidance_brain_node")
    monkeypatch.setattr(guidance_brain_node, "shared_state", fresh_state)
    monkeypatch.setattr(
        guidance_brain_node.state_topic, "publish", lambda *a, **kw: None
    )
    monkeypatch.setattr(guidance_brain_node.amiga_node, "set_twist", _noop_set_twist)
    monkeypatch.setattr(
        guidance_brain_node.lifecycle_manager, "restart_node", _noop_restart_node
    )
    yield fresh_state


@pytest.fixture
def restart_calls(monkeypatch):
    # Record which node was requested to be restarted.
    calls = []

    async def fake_restart_node(node_name):
        calls.append(node_name)
        return SimpleNamespace(success=True, message="")

    monkeypatch.setattr(
        guidance_brain_node.lifecycle_manager, "restart_node", fake_restart_node
    )
    return calls


async def _run_briefly(coro, duration: float = 0.05):
    task = asyncio.ensure_future(coro)
    await asyncio.sleep(duration)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def test_set_p_updates_shared_state(reset_shared_state):
    result = asyncio.run(guidance_brain_node.set_p(None, data=75.0))

    assert reset_shared_state.follower_pid.p == 75.0
    assert result == {"success": True}


def test_set_i_updates_shared_state(reset_shared_state):
    result = asyncio.run(guidance_brain_node.set_i(None, data=1.5))

    assert reset_shared_state.follower_pid.i == 1.5
    assert result == {"success": True}


def test_set_d_updates_shared_state(reset_shared_state):
    result = asyncio.run(guidance_brain_node.set_d(None, data=0.5))

    assert reset_shared_state.follower_pid.d == 0.5
    assert result == {"success": True}


def test_set_speed_updates_shared_state(reset_shared_state):
    result = asyncio.run(guidance_brain_node.set_speed(None, data=30.0))

    assert reset_shared_state.speed == 30.0
    assert result == {"success": True}


def test_on_fp_forw_result_updates_state_when_going_forward(reset_shared_state):
    reset_shared_state.go_direction = guidance_brain_node.GoDirection.FORWARD

    asyncio.run(
        guidance_brain_node.on_fp_forw_result(
            None, linear_deviation=12.3, heading=0.0, is_valid=True
        )
    )

    assert reset_shared_state.error == 12.3
    assert reset_shared_state.perceiver_valid is True


def test_on_fp_forw_result_ignored_when_going_backward(reset_shared_state):
    reset_shared_state.go_direction = guidance_brain_node.GoDirection.BACKWARD
    reset_shared_state.error = 0.0
    reset_shared_state.perceiver_valid = False

    asyncio.run(
        guidance_brain_node.on_fp_forw_result(
            None, linear_deviation=12.3, heading=0.0, is_valid=True
        )
    )

    assert reset_shared_state.error == 0.0
    assert reset_shared_state.perceiver_valid is False


def test_on_fp_back_result_updates_state_when_going_backward(reset_shared_state):
    reset_shared_state.go_direction = guidance_brain_node.GoDirection.BACKWARD

    asyncio.run(
        guidance_brain_node.on_fp_back_result(
            None, linear_deviation=-7.0, heading=0.0, is_valid=True
        )
    )

    assert reset_shared_state.error == -7.0
    assert reset_shared_state.perceiver_valid is True


def test_guidance_task_sends_p_scaled_twist_while_perceiver_valid(
    reset_shared_state, monkeypatch
):
    calls = []

    async def fake_set_twist(twist):
        calls.append(twist)

    monkeypatch.setattr(guidance_brain_node.amiga_node, "set_twist", fake_set_twist)
    reset_shared_state.perceiver_valid = True
    reset_shared_state.error = 10.0
    reset_shared_state.follower_pid.p = 50.0
    reset_shared_state.speed = 20.0

    asyncio.run(
        _run_briefly(
            guidance_brain_node._guidance_task(guidance_brain_node.GoDirection.FORWARD)
        )
    )

    assert len(calls) > 0
    expected_x = 50.0 * 10.0 * guidance_brain_node.P_SCALING
    expected_y = 20.0 * guidance_brain_node.FEET_PER_MIN_TO_METERS_PER_SEC
    assert calls[-1].x == pytest.approx(expected_x)
    assert calls[-1].y == pytest.approx(expected_y)


def test_guidance_task_flips_speed_sign_when_going_backward(
    reset_shared_state, monkeypatch
):
    calls = []

    async def fake_set_twist(twist):
        calls.append(twist)

    monkeypatch.setattr(guidance_brain_node.amiga_node, "set_twist", fake_set_twist)
    reset_shared_state.perceiver_valid = True
    reset_shared_state.error = 10.0
    reset_shared_state.follower_pid.p = 50.0
    reset_shared_state.speed = 20.0

    asyncio.run(
        _run_briefly(
            guidance_brain_node._guidance_task(guidance_brain_node.GoDirection.BACKWARD)
        )
    )

    expected_y = -20.0 * guidance_brain_node.FEET_PER_MIN_TO_METERS_PER_SEC
    assert calls[-1].y == pytest.approx(expected_y)


def test_guidance_task_zeroes_command_when_perceiver_invalid(
    reset_shared_state, monkeypatch
):
    calls = []

    async def fake_set_twist(twist):
        calls.append(twist)

    monkeypatch.setattr(guidance_brain_node.amiga_node, "set_twist", fake_set_twist)
    reset_shared_state.perceiver_valid = False

    asyncio.run(
        _run_briefly(
            guidance_brain_node._guidance_task(guidance_brain_node.GoDirection.FORWARD),
            duration=0.05,
        )
    )

    assert len(calls) > 0
    assert calls[-1].x == 0.0
    assert calls[-1].y == 0.0
    assert reset_shared_state.command == 0.0


def test_guidance_task_exits_after_one_second_without_valid_perceiver(
    reset_shared_state, restart_calls
):
    reset_shared_state.perceiver_valid = False

    async def run():
        await asyncio.wait_for(
            guidance_brain_node._guidance_task(guidance_brain_node.GoDirection.FORWARD),
            timeout=2.0,
        )

    asyncio.run(run())  # should return before time out

    assert reset_shared_state.guidance_active is False
    assert restart_calls == ["furrow_perceiver_forward"]


def test_start_task_rejects_second_task_while_one_is_active(reset_shared_state):
    async def run():
        first = guidance_brain_node._start_task(asyncio.sleep(10), name="first")
        # Let the first task actually start running
        await asyncio.sleep(0)

        second_coro = asyncio.sleep(10)
        second = guidance_brain_node._start_task(second_coro, name="second")
        if not second:
            second_coro.close()  # _start_task rejected it without awaiting

        await guidance_brain_node._stop_current_task()
        return first, second

    first, second = asyncio.run(run())

    assert first is True
    assert second is False


def test_stop_current_task_cancels_running_task_then_reports_nothing_to_stop():
    async def run():
        guidance_brain_node._start_task(asyncio.sleep(10))
        await asyncio.sleep(
            0
        )  # let the task actually start running before cancelling it
        stopped = await guidance_brain_node._stop_current_task()
        stopped_again = await guidance_brain_node._stop_current_task()
        return stopped, stopped_again

    stopped, stopped_again = asyncio.run(run())

    assert stopped is True
    assert stopped_again is False


def test_go_forward_starts_guidance_task(reset_shared_state):
    reset_shared_state.perceiver_valid = False

    async def run():
        result = await guidance_brain_node.go_forward(None)
        # let the started task run and set state
        await asyncio.sleep(0.02)
        direction = reset_shared_state.go_direction
        active = reset_shared_state.guidance_active
        await guidance_brain_node._stop_current_task()
        return result, direction, active

    result, direction, active = asyncio.run(run())

    assert result == {"success": True}
    assert direction == guidance_brain_node.GoDirection.FORWARD
    assert active is True


def test_go_backward_starts_guidance_task(reset_shared_state):
    reset_shared_state.perceiver_valid = False

    async def run():
        result = await guidance_brain_node.go_backward(None)
        await asyncio.sleep(0.02)
        direction = reset_shared_state.go_direction
        active = reset_shared_state.guidance_active
        await guidance_brain_node._stop_current_task()
        return result, direction, active

    result, direction, active = asyncio.run(run())

    assert result == {"success": True}
    assert direction == guidance_brain_node.GoDirection.BACKWARD
    assert active is True


def test_on_fp_forw_result_stamps_message_time(reset_shared_state):
    reset_shared_state.go_direction = guidance_brain_node.GoDirection.FORWARD
    assert reset_shared_state.last_perceiver_msg_time == 0.0

    asyncio.run(
        guidance_brain_node.on_fp_forw_result(
            None, linear_deviation=1.0, heading=0.0, is_valid=True
        )
    )

    assert reset_shared_state.last_perceiver_msg_time > 0.0


def test_guidance_task_restarts_perceiver_when_messages_stop(
    reset_shared_state, restart_calls
):

    reset_shared_state.perceiver_valid = True
    reset_shared_state.error = 0.0

    async def run():
        await asyncio.wait_for(
            guidance_brain_node._guidance_task(guidance_brain_node.GoDirection.FORWARD),
            timeout=2.0,
        )

    asyncio.run(run())

    assert reset_shared_state.guidance_active is False
    assert restart_calls == ["furrow_perceiver_forward"]


def test_stop_with_no_active_task_reports_failure(reset_shared_state):
    result = asyncio.run(guidance_brain_node.stop(None))

    assert result == {"success": False}
