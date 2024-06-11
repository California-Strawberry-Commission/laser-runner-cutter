from abc import ABC, abstractmethod


class LaserDAC(ABC):
    @abstractmethod
    def initialize(self):
        """
        Set up the laser DAC and search for online DACs.
        """
        pass

    @abstractmethod
    def connect(self, dac_idx: int):
        """
        Connect to the specified DAC.

        Args:
            dac_idx (int): Index of the DAC to connect to.
        """
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Returns:
            bool: Whether the DAC is connected.
        """
        pass

    @abstractmethod
    def set_color(self, r: float, g: float, b: float, i: float):
        """
        Set the color of the laser.

        Args:
            r (float): Red channel, with value normalized to [0, 1]
            g (float): Green channel, with value normalized to [0, 1]
            b (float): Blue channel, with value normalized to [0, 1]
            i (float): Intensity, with value normalized to [0, 1]
        """
        pass

    @abstractmethod
    def add_point(self, x: float, y: float):
        """
        Add a point to be rendered by the DAC. (0, 0) corresponds to bottom left.

        Args:
            x (float): x coordinate normalized to [0, 1]
            y (float): y coordinate normalized to [0, 1]
        """
        pass

    @abstractmethod
    def remove_point(self):
        """
        Remove the last added point.
        """
        pass

    @abstractmethod
    def clear_points(self):
        """
        Remove all points.
        """
        pass

    @abstractmethod
    def play(self, fps: int, pps: int, transition_duration_ms: float):
        """
        Start playback of points.

        Args:
            fps (int): Target frames per second.
            pps (int): Target points per second. This should not exceed the capability of the DAC and laser projector.
            transition_duration_ms (float): Duration in ms to turn the laser off between subsequent points in the same frame.
        """
        pass

    @abstractmethod
    def stop(self):
        """
        Stop playback of points.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Close connection to laser DAC.
        """
        pass
