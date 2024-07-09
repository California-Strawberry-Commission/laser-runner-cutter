from contextlib import asynccontextmanager


class CameraContext:
    def __init__(self, camera_node):
        self._camera_node = camera_node

    @asynccontextmanager
    async def laser_detection_settings(self):
        get_state_res = await self._camera_node.get_state()
        prev_exposure_us = get_state_res.state.exposure_us
        prev_gain_db = get_state_res.state.gain_db
        await self._camera_node.set_exposure(exposure_us=1.0)
        await self._camera_node.set_gain(gain_db=0.0)
        try:
            yield
        finally:
            await self._camera_node.set_gain(gain_db=prev_gain_db)
            await self._camera_node.set_exposure(exposure_us=prev_exposure_us)
