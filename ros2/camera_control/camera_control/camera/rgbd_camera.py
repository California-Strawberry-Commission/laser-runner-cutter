from abc import ABC, abstractmethod
from typing import Optional, Tuple
from .rgbd_frame import RgbdFrame


class RgbdCamera(ABC):
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Returns:
            bool: Whether the camera is connected.
        """
        pass

    @abstractmethod
    def initialize(self):
        """
        Set up the camera.
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

    @abstractmethod
    def get_frame(self) -> Optional[RgbdFrame]:
        """
        Get the latest available color and depth frames from the camera.

        Returns:
            Optional[RgbdFrame]: The color and depth frames, or None if not available.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Close connection to the camera.
        """
        pass
