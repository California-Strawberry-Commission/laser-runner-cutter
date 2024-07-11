from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Callable, Optional, Tuple

from .rgbd_frame import RgbdFrame


class State(Enum):
    DISCONNECTED = auto()
    CONNECTING = auto()
    STREAMING = auto()


class RgbdCamera(ABC):
    @property
    @abstractmethod
    def state(self) -> State:
        """
        Returns:
            State: Current state of the camera device.
        """
        pass

    @abstractmethod
    def start(
        self,
        exposure_us: float,
        gain_db: float,
        frame_callback: Optional[Callable[[RgbdFrame], None]],
    ):
        """
        Connects device and starts streaming.

        Args:
            exposure_us (float): Exposure time in microseconds.
            gain_db (float): Gain level in dB.
            frame_callback (Callable[[RgbdFrame], None]): Callback that gets called when a new frame is available.
        """
        pass

    @abstractmethod
    def stop(self):
        """
        Stops streaming and disconnects device.
        """
        pass

    @property
    @abstractmethod
    def exposure_us(self) -> float:
        """
        Returns:
            float: Exposure time in microseconds.
        """
        pass

    @exposure_us.setter
    @abstractmethod
    def exposure_us(self, exposure_us: float):
        """
        Set the exposure time of the camera.

        Args:
            exposure_us (float): Exposure time in microseconds.
        """
        pass

    @abstractmethod
    def get_exposure_us_range(self) -> Tuple[float, float]:
        """
        Returns:
            Tuple[float, float]: (min, max) exposure times in microseconds.
        """
        pass

    @property
    @abstractmethod
    def gain_db(self) -> float:
        """
        Returns:
            float: Gain level in dB.
        """
        pass

    @gain_db.setter
    @abstractmethod
    def gain_db(self, gain_db: float):
        """
        Set the gain level of the camera.

        Args:
            gain_db (float): Gain level in dB.
        """
        pass

    @abstractmethod
    def get_gain_db_range(self) -> Tuple[float, float]:
        """
        Returns:
            Tuple[float, float]: (min, max) gain levels in dB.
        """
        pass
